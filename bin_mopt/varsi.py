"""Variance-ordered insertion and refinement using existing covariance data.

Adapted from fast_VarSI.py: cache group variances, covariance sums, and
compatibility masks. Context matrices and covariance dictionaries are read-only.
"""

import math

import numpy as np

from .utils import clean_variance


def _group_state(context, groups):
    variances, covariance_sums, masks = [], [], []
    for group in groups:
        indices = np.asarray(group, dtype=int)
        variances.append(clean_variance(
            context.scaled_covariance[np.ix_(indices, indices)].sum(), tiny=1.e-8,
        ))
        covariance_sums.append(context.scaled_covariance[indices].sum(axis=0))
        masks.append(context.compatible[indices].all(axis=0))
    return variances, covariance_sums, masks


def _position_groups(context, assignment):
    assignment = np.asarray(assignment, dtype=bool)
    if assignment.ndim != 2 or assignment.shape[0] != context.n_terms:
        raise ValueError("Assignment must have one row per context term.")
    if np.any(assignment.sum(axis=1) > 1):
        raise ValueError("VarSI requires non-overlapping groups.")
    groups = [np.flatnonzero(column).tolist() for column in assignment.T if column.any()]
    if any(not context.compatible[np.ix_(group, group)].all() for group in groups):
        raise ValueError("VarSI groups must contain compatible terms.")
    return groups


def _assignment(context, groups):
    assignment = np.zeros((context.n_terms, len(groups)), dtype=bool)
    for column, group in enumerate(groups):
        assignment[group, column] = True
    return assignment


def varsi_ordered(context, assignment, new_positions):
    """Insert new terms by descending c_i**2 Var(P_i), fixing initial groups.

    Each term minimizes (sum_g sqrt(Var(H_g)))**2 over compatible groups.
    As in the fast VarSI-O defaults, open a singleton only if none is compatible.
    """
    groups = _position_groups(context, assignment)
    positions = list(new_positions)
    covered = {position for group in groups for position in group}
    if (len(positions) != len(set(positions)) or covered.intersection(positions)
            or covered.union(positions) != set(range(context.n_terms))):
        raise ValueError("New positions must cover exactly the unassigned terms.")
    variances, covariance_sums, masks = _group_state(context, groups)
    single_variances = context.scaled_covariance.diagonal()
    for position in sorted(positions, key=lambda i: -single_variances[i]):
        term_variance = float(single_variances[position])
        score = sum(math.sqrt(variance) for variance in variances)
        best_metric, best_group, best_variance = None, None, term_variance
        for group_index, mask in enumerate(masks):
            if not mask[position]:
                continue
            variance = clean_variance(
                variances[group_index] + term_variance
                + 2 * covariance_sums[group_index][position], tiny=1.e-8,
            )
            metric = (score - math.sqrt(variances[group_index]) + math.sqrt(variance)) ** 2
            if best_metric is None or metric < best_metric:
                best_metric, best_group, best_variance = metric, group_index, variance
        if best_group is None:
            groups.append([position])
            variances.append(term_variance)
            covariance_sums.append(context.scaled_covariance[position].copy())
            masks.append(context.compatible[position].copy())
        else:
            groups[best_group].append(position)
            variances[best_group] = best_variance
            covariance_sums[best_group] = (
                covariance_sums[best_group] + context.scaled_covariance[position]
            )
            masks[best_group] = masks[best_group] & context.compatible[position]
    return _assignment(context, groups)


def varsi_refine(context, assignment, *, movable_positions=None, max_sweeps=100,
                 tiny=1.e-9):
    """Accept at most one best improving move per sweep, as in fast VarSI-OR.

    Restrict moves to movable_positions when supplied. No new singleton groups
    are opened during refinement, matching the fast variant's default.
    Return the refined partition and the number of accepted moves.
    """
    if max_sweeps < 1:
        raise ValueError("max_sweeps must be positive.")
    groups = _position_groups(context, assignment)
    if sum(map(len, groups)) != context.n_terms:
        raise ValueError("Refinement requires a complete partition.")
    movable = set(range(context.n_terms) if movable_positions is None else movable_positions)
    if not movable.issubset(range(context.n_terms)):
        raise ValueError("Movable positions must be context term indices.")
    variances, covariance_sums, masks = _group_state(context, groups)
    single_variances = context.scaled_covariance.diagonal()
    accepted_moves = 0
    for _ in range(max_sweeps):
        score = sum(math.sqrt(variance) for variance in variances)
        best_metric, best_move = score * score, None
        for source, group in enumerate(groups):
            for term_index, position in enumerate(group):
                if position not in movable:
                    continue
                term_variance = float(single_variances[position])
                source_variance = 0.0 if len(group) == 1 else clean_variance(
                    variances[source] + term_variance
                    - 2 * covariance_sums[source][position], tiny=1.e-8,
                )
                for destination, mask in enumerate(masks):
                    if destination == source or not mask[position]:
                        continue
                    destination_variance = clean_variance(
                        variances[destination] + term_variance
                        + 2 * covariance_sums[destination][position], tiny=1.e-8,
                    )
                    new_score = (score - math.sqrt(variances[source])
                                 + math.sqrt(source_variance)
                                 - math.sqrt(variances[destination])
                                 + math.sqrt(destination_variance))
                    metric = new_score * new_score
                    if metric < best_metric - tiny:
                        best_metric = metric
                        best_move = source, term_index, destination
        if best_move is None:
            break
        source, term_index, destination = best_move
        position = groups[source].pop(term_index)
        groups[destination].append(position)
        if not groups[source]:
            del groups[source]
        # Rebuild after moving/deleting a group to avoid stale aggregates.
        variances, covariance_sums, masks = _group_state(context, groups)
        accepted_moves += 1
    return _assignment(context, groups), accepted_moves
