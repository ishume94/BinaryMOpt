"""Classical binary optimization of Pauli measurement groupings.

The measurement objective contains square roots of group variances. The
SciPy/HiGHS backend solves an exact-cover MILP over a deterministic pool of
commuting clique candidates with precomputed ``sqrt(variance)`` costs.
Large-neighborhood refinement ranks feasible partitions by measurement cost;
standard ICS subsequently refines their coefficient splitting.

Thus the backend returns the lowest exact measurement cost it finds, but does
not claim a global optimum over every possible clique. The MILPs enforce
coverage, compatibility, and membership limits as hard constraints.
"""

import math
import time
from dataclasses import dataclass
from functools import partial

import networkx as nx
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, OptimizeResult, milp as scipy_milp
from scipy.sparse import csc_matrix, lil_matrix, vstack

from .ics import (
    optimize_fixed_binary_support,
    optimize_from_nonoverlapping_groups,
)
from .utils import PauliTerm, clean_real, clean_variance, get_covariance
from .utils import terms_fully_commute
from .weighting import WEIGHT_HEURISTICS


@dataclass(frozen=True)
class OptimizationContext:
    """Dense real covariance and compatibility data in stable term order."""

    terms: tuple[PauliTerm, ...]
    coefficients: np.ndarray
    covariance: np.ndarray
    compatible: np.ndarray
    scaled_covariance: np.ndarray

    @property
    def n_terms(self) -> int:
        return len(self.terms)


@dataclass
class OptimizationResult:
    """A feasible grouping and its exact measurement cost."""

    mode: str
    groups: tuple[tuple[int, ...], ...]
    assignment: np.ndarray
    omega: np.ndarray
    variances: tuple[float, ...]
    score: float
    eps_sq_m: float
    cap_groups: int
    max_split: int
    solver_status: str
    solver_objective: float
    iterations: int
    n_candidates: int
    sample_ratios: tuple[float, ...] | None = None
    history: tuple[tuple[int, float], ...] = ()
    observation_history: tuple[tuple[int, float], ...] = ()

    def __post_init__(self):
        cleaned = []
        previous = math.inf
        for iteration, objective in self.history:
            point = (int(iteration), float(objective))
            if not np.isfinite(point[1]):
                raise ValueError("history objectives must be finite")
            if point[1] > previous + 1.0e-12:
                raise ValueError("history must be monotone nonincreasing")
            cleaned.append(point)
            previous = min(previous, point[1])
        self.history = tuple(cleaned)
        observations = []
        for iteration, objective in self.observation_history:
            point = (int(iteration), float(objective))
            if not np.isfinite(point[1]):
                raise ValueError("observation history objectives must be finite")
            observations.append(point)
        self.observation_history = tuple(observations)

    def term_groups(self, context: OptimizationContext):
        """Return each group as ``(PauliTerm, split coefficient)`` records."""

        output = []
        for group_index, group in enumerate(self.groups):
            output.append(
                tuple(
                    (context.terms[position], complex(self.omega[position, group_index]))
                    for position in group
                )
            )
        return tuple(output)


@dataclass(frozen=True)
class CandidatePool:
    """Validated candidate cliques and a guaranteed feasible cover."""

    candidates: tuple[tuple[int, ...], ...]
    feasible_cover: tuple[tuple[int, ...], ...]
    metadata: dict


def make_optimization_context(
    measurable_terms: list[PauliTerm], covariances: dict
) -> OptimizationContext:
    """Build matrices for fully commuting measurement optimization."""

    terms = tuple(measurable_terms)
    if not terms:
        raise ValueError("At least one non-identity Pauli term is required.")
    coefficients = np.asarray(
        [clean_real(term.coefficient, tiny=1.0e-9) for term in terms],
        dtype=float,
    )
    n_terms = len(terms)
    covariance = np.zeros((n_terms, n_terms), dtype=float)
    compatible = np.eye(n_terms, dtype=bool)
    for left in range(n_terms):
        for right in range(left, n_terms):
            is_compatible = terms_fully_commute(terms[left], terms[right])
            compatible[left, right] = is_compatible
            compatible[right, left] = is_compatible
            if not is_compatible:
                continue
            value = clean_real(
                get_covariance(terms[left], terms[right], covariances),
                tiny=1.0e-7,
            )
            covariance[left, right] = value
            covariance[right, left] = value
    diagonal = np.diag(covariance).copy()
    if np.any(diagonal < -1.0e-8):
        raise ValueError("The covariance matrix contains a negative variance.")
    diagonal[diagonal < 0.0] = 0.0
    np.fill_diagonal(covariance, diagonal)
    scaled_covariance = (
        coefficients[:, np.newaxis]
        * covariance
        * coefficients[np.newaxis, :]
    )
    return OptimizationContext(
        terms=terms,
        coefficients=coefficients,
        covariance=covariance,
        compatible=compatible,
        scaled_covariance=scaled_covariance,
    )


def _incompatibility_graph(context: OptimizationContext) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(context.n_terms))
    for left in range(context.n_terms):
        for right in range(left + 1, context.n_terms):
            if not bool(context.compatible[left, right]):
                graph.add_edge(left, right)
    return graph


def _groups_from_coloring(coloring: dict[int, int]) -> list[list[int]]:
    grouped = {}
    for node, color in coloring.items():
        grouped.setdefault(int(color), []).append(int(node))
    return [sorted(grouped[color]) for color in sorted(grouped)]


def coloring_group_bound(
    context: OptimizationContext,
    *,
    random_seed: int = 7,
    augmentation: int = 4,
):
    """Return a seeded random-sequential color count plus spare group slots.

    Coloring the complement/incompatibility graph is equivalent to covering
    the commutativity graph with cliques.  Following the requested standard,
    ``cap_groups`` is the NetworkX random-sequential color count augmented by
    four (or the explicitly supplied ``augmentation``).  The seed makes this
    otherwise-random upper bound reproducible.
    """

    if augmentation < 0:
        raise ValueError("augmentation must be non-negative.")
    strategy = partial(
        nx.coloring.strategy_random_sequential,
        seed=int(random_seed),
    )
    coloring = nx.coloring.greedy_color(
        _incompatibility_graph(context), strategy=strategy
    )
    groups = _groups_from_coloring(coloring)
    return len(groups) + int(augmentation), groups


def _group_variance(
    context: OptimizationContext, group: list[int] | tuple[int, ...]
) -> float:
    positions = np.asarray(group, dtype=int)
    value = context.scaled_covariance[np.ix_(positions, positions)].sum()
    return clean_variance(value, tiny=1.0e-8)


def singleton_measurement_bound(context: OptimizationContext) -> float:
    """Eq. (6) for the always-feasible singleton partition."""

    score = sum(
        math.sqrt(clean_variance(value, tiny=1.0e-8))
        for value in np.diag(context.scaled_covariance)
    )
    return score * score


def _incompatible_pair_count(
    context: OptimizationContext, assignment: np.ndarray
) -> float:
    incompatibility = np.logical_not(context.compatible).astype(float)
    np.fill_diagonal(incompatibility, 0.0)
    total = 0.0
    for group_index in range(assignment.shape[1]):
        column = assignment[:, group_index].astype(float)
        total += 0.5 * float(column @ incompatibility @ column)
    return total


def measurement_objective(
    context: OptimizationContext, assignment, omega
):
    """Return group variances, score, and ``epsilon^2 M`` (Eqs. 6/14)."""

    assignment = np.asarray(assignment, dtype=bool)
    omega = np.asarray(omega, dtype=float) * assignment
    variances = []
    for group_index in range(assignment.shape[1]):
        weights = omega[:, group_index]
        variance = clean_variance(
            weights @ context.covariance @ weights, tiny=1.0e-8
        )
        variances.append(variance)
    score = sum(math.sqrt(variance) for variance in variances)
    return tuple(variances), score, score * score


def _validate_assignment(
    context: OptimizationContext,
    assignment,
    *,
    max_split: int,
    exact_once: bool,
) -> None:
    assignment = np.asarray(assignment, dtype=bool)
    if assignment.ndim != 2 or assignment.shape[0] != context.n_terms:
        raise ValueError("Assignment must have shape (n_terms, n_groups).")
    coverage = assignment.sum(axis=1)
    if exact_once and not np.all(coverage == 1):
        raise ValueError("Non-overlapping assignments must cover every term once.")
    if not exact_once and (np.any(coverage < 1) or np.any(coverage > max_split)):
        raise ValueError("Overlapping assignment violates its coverage bounds.")
    if _incompatible_pair_count(context, assignment) != 0.0:
        raise ValueError("At least one group contains incompatible Pauli words.")


def _assignment_from_groups(n_terms: int, groups) -> np.ndarray:
    assignment = np.zeros((n_terms, len(groups)), dtype=bool)
    for group_index, group in enumerate(groups):
        assignment[list(group), group_index] = True
    return assignment


def _groups_from_assignment(assignment: np.ndarray):
    return tuple(
        tuple(np.flatnonzero(assignment[:, group_index]).tolist())
        for group_index in range(assignment.shape[1])
        if np.any(assignment[:, group_index])
    )


def relative_variance_weights(
    context: OptimizationContext,
    assignment,
    *,
    max_iterations: int = 100,
    tolerance: float = 1.0e-10,
    damping: float = 0.5,
) -> np.ndarray:
    """Assign overlapping coefficients without continuous optimization.

    For every shared term, its coefficient is distributed in proportion to
    ``sqrt(V(group without that term))``.  This is the explicit heuristic
    inferred from the paper's "relative variances, excluding the split term"
    description.  Equal shares are used when all leave-one-out variances are
    zero.  Row normalization incorporates the number of groups sharing a term
    and enforces Eq. (3) exactly.
    """

    assignment = np.asarray(assignment, dtype=bool)
    coverage = assignment.sum(axis=1)
    if np.any(coverage < 1):
        raise ValueError("Every term needs at least one selected group.")
    omega = (
        context.coefficients[:, np.newaxis]
        * assignment
        / coverage[:, np.newaxis]
    )
    for _ in range(max_iterations):
        covariance_action = context.covariance @ omega
        group_variances = np.sum(omega * covariance_action, axis=0)
        target = np.zeros_like(omega)
        for term_index in range(context.n_terms):
            memberships = np.flatnonzero(assignment[term_index])
            if memberships.size == 1:
                target[term_index, memberships[0]] = context.coefficients[term_index]
                continue
            old_weights = omega[term_index, memberships]
            leave_one_out = (
                group_variances[memberships]
                - 2.0
                * old_weights
                * covariance_action[term_index, memberships]
                + np.square(old_weights) * context.covariance[term_index, term_index]
            )
            leave_one_out[np.abs(leave_one_out) < 1.0e-10] = 0.0
            if np.any(leave_one_out < 0.0):
                raise ValueError("A leave-one-out group variance is negative.")
            relative_std = np.sqrt(leave_one_out)
            denominator = sum(relative_std)
            if denominator <= tolerance:
                shares = np.full(memberships.size, 1.0 / memberships.size)
            else:
                shares = np.asarray(relative_std, dtype=float) / denominator
            target[term_index, memberships] = (
                context.coefficients[term_index] * shares
            )
        updated = damping * target + (1.0 - damping) * omega
        updated *= assignment
        if np.max(np.abs(updated - omega)) <= tolerance:
            omega = updated
            break
        omega = updated

    # Enforce coefficient reconstruction to floating-point precision after
    # damping.  This leaves the relative shares unchanged.
    for term_index in range(context.n_terms):
        memberships = np.flatnonzero(assignment[term_index])
        total = float(omega[term_index, memberships].sum())
        coefficient = context.coefficients[term_index]
        if abs(total) <= tolerance:
            omega[term_index, memberships] = coefficient / memberships.size
        else:
            omega[term_index, memberships] *= coefficient / total
        residual = coefficient - float(omega[term_index, memberships].sum())
        omega[term_index, memberships[-1]] += residual
    return omega


def fixed_support_ics(
    context: OptimizationContext,
    assignment,
    *,
    initial_omega=None,
    n_iterations: int = 5,
    deadline=None,
):
    """Run the copied GFlow-VQE ICS routine without changing memberships."""

    assignment = np.asarray(assignment, dtype=bool)
    if np.any(assignment.sum(axis=0) == 0):
        raise ValueError("ICS requires compact support without empty groups.")
    _validate_assignment(
        context,
        assignment,
        max_split=assignment.shape[1],
        exact_once=False,
    )
    omega, sample_ratios = optimize_fixed_binary_support(
        context,
        assignment,
        initial_omega=initial_omega,
        n_iter=n_iterations,
        deadline=deadline,
    )
    sample_ratios = np.asarray(sample_ratios, dtype=float).reshape(-1)
    if (
        sample_ratios.size != assignment.shape[1]
        or not np.all(np.isfinite(sample_ratios))
        or np.any(sample_ratios < 0.0)
        or not np.isclose(sample_ratios.sum(), 1.0)
    ):
        raise ValueError("ICS returned invalid shot ratios.")
    variances, score, eps_sq_m = measurement_objective(
        context, assignment, omega
    )
    return omega, variances, score, eps_sq_m, sample_ratios


def _greedy_partition(context: OptimizationContext, order) -> list[list[int]]:
    groups: list[list[int]] = []
    variances: list[float] = []
    for position in order:
        best = None
        for group_index, group in enumerate(groups):
            if not all(context.compatible[position, other] for other in group):
                continue
            new_group = [*group, int(position)]
            new_variance = _group_variance(context, new_group)
            delta = math.sqrt(new_variance) - math.sqrt(variances[group_index])
            candidate = (delta, group_index, new_variance)
            if best is None or candidate < best:
                best = candidate
        if best is None:
            groups.append([int(position)])
            variances.append(_group_variance(context, [int(position)]))
        else:
            _, group_index, new_variance = best
            groups[group_index].append(int(position))
            variances[group_index] = new_variance
    return [sorted(group) for group in groups]


def _first_fit_partition(context: OptimizationContext, order) -> list[list[int]]:
    """Build a deterministic clique partition in the supplied term order."""

    groups: list[list[int]] = []
    for position in order:
        position = int(position)
        for group in groups:
            if all(context.compatible[position, other] for other in group):
                group.append(position)
                break
        else:
            groups.append([position])
    return [sorted(group) for group in groups]


def _partition_score(
    context: OptimizationContext,
    groups: list[list[int]] | tuple[tuple[int, ...], ...],
) -> float:
    return sum(math.sqrt(_group_variance(context, group)) for group in groups)


def _refine_partition_by_relocation(
    context: OptimizationContext,
    groups,
    *,
    max_moves: int = 1000,
    tolerance: float = 1.0e-12,
    deadline=None,
):
    """Apply globally best feasible one-term relocations.

    Every candidate move preserves exact cover and commuting-clique
    constraints. Its change in the paper's score is evaluated analytically
    from the scaled covariance matrix; the lowest improving move is accepted
    and the search repeats to a one-move local optimum.
    """

    groups = [sorted(int(position) for position in group) for group in groups]
    groups = [group for group in groups if group]
    if not groups:
        raise ValueError("A partition must contain at least one group.")

    assignment = _assignment_from_groups(context.n_terms, groups)
    _validate_assignment(context, assignment, max_split=1, exact_once=True)
    scaled_covariance = context.scaled_covariance
    accepted = 0

    while accepted < max_moves:
        if deadline is not None and time.monotonic() >= float(deadline):
            break
        variances = np.asarray(
            [_group_variance(context, group) for group in groups], dtype=float
        )
        standard_deviations = np.sqrt(variances)
        score = float(standard_deviations.sum())
        covariance_sums = [
            scaled_covariance[np.asarray(group, dtype=int), :].sum(axis=0)
            for group in groups
        ]
        locations = np.empty(context.n_terms, dtype=int)
        for group_index, group in enumerate(groups):
            locations[np.asarray(group, dtype=int)] = group_index

        best = None
        for term_index in range(context.n_terms):
            source_index = int(locations[term_index])
            term_variance = float(scaled_covariance[term_index, term_index])
            source_variance = clean_variance(
                variances[source_index]
                + term_variance
                - 2.0 * covariance_sums[source_index][term_index],
                tiny=1.0e-8,
            )
            for target_index, target_group in enumerate(groups):
                if target_index == source_index:
                    continue
                if not all(
                    context.compatible[term_index, other]
                    for other in target_group
                ):
                    continue
                target_variance = clean_variance(
                    variances[target_index]
                    + term_variance
                    + 2.0 * covariance_sums[target_index][term_index],
                    tiny=1.0e-8,
                )
                proposed_score = (
                    score
                    - standard_deviations[source_index]
                    - standard_deviations[target_index]
                    + math.sqrt(source_variance)
                    + math.sqrt(target_variance)
                )
                delta = proposed_score - score
                candidate = (
                    float(delta),
                    int(term_index),
                    int(source_index),
                    int(target_index),
                )
                if delta < -tolerance and (best is None or candidate < best):
                    best = candidate

        if best is None:
            break
        _, term_index, source_index, target_index = best
        groups[source_index].remove(term_index)
        groups[target_index].append(term_index)
        groups[target_index].sort()
        if not groups[source_index]:
            groups.pop(source_index)
        accepted += 1

    return [sorted(group) for group in groups], accepted


def _seed_partitions(
    context: OptimizationContext,
    *,
    random_partitions: int,
    random_seed: int,
    deadline=None,
):
    """Return reproducible full partitions used by the MILP and refinement."""

    graph = _incompatibility_graph(context)
    partitions = []
    for strategy in (
        "saturation_largest_first",
        "largest_first",
        "smallest_last",
    ):
        if deadline is not None and time.monotonic() >= float(deadline):
            break
        coloring = nx.coloring.greedy_color(graph, strategy=strategy)
        partitions.append(_groups_from_coloring(coloring))
    if deadline is None or time.monotonic() < float(deadline):
        random_coloring = nx.coloring.greedy_color(
            graph,
            strategy=partial(
                nx.coloring.strategy_random_sequential,
                seed=int(random_seed),
            ),
        )
        partitions.append(_groups_from_coloring(random_coloring))

    positions = np.arange(context.n_terms)
    source_order = sorted(
        positions,
        key=lambda item: (context.terms[int(item)].source_order, int(item)),
    )
    coefficient_order = sorted(
        positions,
        key=lambda item: (-abs(context.coefficients[int(item)]), int(item)),
    )
    single_variances = np.diag(context.scaled_covariance)
    variance_order = sorted(
        positions,
        key=lambda item: (-single_variances[int(item)], int(item)),
    )
    deterministic_orders = [
        positions.tolist(),
        source_order,
        coefficient_order,
        variance_order,
    ]
    for order in deterministic_orders:
        if deadline is not None and time.monotonic() >= float(deadline):
            break
        partitions.append(_first_fit_partition(context, order))
        partitions.append(_greedy_partition(context, order))

    rng = np.random.default_rng(random_seed)
    for _ in range(max(0, random_partitions)):
        if deadline is not None and time.monotonic() >= float(deadline):
            break
        order = rng.permutation(positions).tolist()
        partitions.append(_first_fit_partition(context, order))
        partitions.append(_greedy_partition(context, order))

    unique = {}
    for partition in partitions:
        canonical = tuple(sorted(tuple(sorted(group)) for group in partition))
        unique.setdefault(canonical, [list(group) for group in partition])
    growth_orders = [coefficient_order, variance_order, source_order]
    return list(unique.values()), growth_orders


def _grow_clique(context: OptimizationContext, seed, order):
    group = list(dict.fromkeys(int(position) for position in seed))
    for position in order:
        position = int(position)
        if position in group:
            continue
        if all(context.compatible[position, other] for other in group):
            group.append(position)
    return tuple(sorted(group))


def _candidate_cliques(
    context: OptimizationContext,
    *,
    random_partitions: int,
    random_seed: int,
    max_candidates: int,
    deadline=None,
):
    if int(max_candidates) < 1:
        raise ValueError("max_candidates must be positive")
    partitions, growth_orders = _seed_partitions(
        context,
        random_partitions=random_partitions,
        random_seed=random_seed,
        deadline=deadline,
    )
    positions = np.arange(context.n_terms)
    effective_limit = max(context.n_terms, int(max_candidates))

    candidates: dict[tuple[int, ...], None] = {}
    for position in positions:
        candidates[(int(position),)] = None
    for partition in partitions:
        if deadline is not None and time.monotonic() >= float(deadline):
            break
        for group in partition:
            candidate = tuple(sorted(int(item) for item in group))
            if candidate in candidates:
                continue
            if len(candidates) >= effective_limit:
                break
            candidates[candidate] = None
        if len(candidates) >= effective_limit:
            break

    growth_orders = [*growth_orders, positions.tolist()]
    seeds = list(candidates)
    for seed in seeds:
        if deadline is not None and time.monotonic() >= float(deadline):
            break
        for order in growth_orders:
            candidate = _grow_clique(context, seed, order)
            if candidate not in candidates and len(candidates) < effective_limit:
                candidates[candidate] = None
            if len(candidates) >= effective_limit:
                break
        if len(candidates) >= effective_limit:
            break
    cliques = list(candidates)
    covered = set()
    for clique in cliques:
        if not clique:
            raise RuntimeError("candidate clique pool contains an empty clique")
        for offset, left in enumerate(clique):
            covered.add(left)
            for right in clique[offset + 1 :]:
                if not context.compatible[left, right]:
                    raise RuntimeError("candidate clique pool contains an incompatible group")
    for position in range(context.n_terms):
        if position not in covered:
            singleton = (position,)
            if singleton not in candidates:
                candidates[singleton] = None
                cliques.append(singleton)
    if len(cliques) > effective_limit:
        raise RuntimeError("candidate clique pool exceeds its effective limit")
    return cliques, partitions


def _solve_exact_cover_milp(
    context: OptimizationContext,
    candidates,
    cap_groups: int,
    time_limit: float,
    fallback_groups=None,
):
    n_candidates = len(candidates)
    incidence = lil_matrix((context.n_terms, n_candidates), dtype=float)
    costs = np.empty(n_candidates, dtype=float)
    for candidate_index, group in enumerate(candidates):
        incidence[list(group), candidate_index] = 1.0
        costs[candidate_index] = math.sqrt(_group_variance(context, group))
    constraint_matrix = vstack(
        [incidence.tocsc(), csc_matrix(np.ones((1, n_candidates)))],
        format="csc",
    )
    lower = np.concatenate([np.ones(context.n_terms), [-np.inf]])
    upper = np.concatenate([np.ones(context.n_terms), [float(cap_groups)]])
    if float(time_limit) <= 0.0:
        result = OptimizeResult(
            x=None,
            message="Optimization deadline reached before candidate-cover MILP.",
            status=1,
        )
    else:
        result = scipy_milp(
            c=costs,
            integrality=np.ones(n_candidates, dtype=int),
            bounds=Bounds(np.zeros(n_candidates), np.ones(n_candidates)),
            constraints=LinearConstraint(constraint_matrix, lower, upper),
            options={
                "presolve": True,
                "time_limit": max(1.0e-6, float(time_limit)),
                "mip_rel_gap": 0.0,
            },
        )
    if result.x is None:
        if fallback_groups is None:
            if context.n_terms <= cap_groups:
                fallback_groups = [(position,) for position in range(context.n_terms)]
            else:
                raise RuntimeError(
                    "HiGHS found no feasible clique cover and no fallback was supplied: "
                    "{}".format(result.message)
                )
        fallback_assignment = _assignment_from_groups(
            context.n_terms, fallback_groups
        )
        _validate_assignment(
            context, fallback_assignment, max_split=1, exact_once=True
        )
        if len(fallback_groups) > cap_groups:
            raise RuntimeError("fallback clique cover exceeds cap_groups")
        result.message = "{} Using feasible fallback cover.".format(result.message)
        return [tuple(group) for group in fallback_groups], result
    selected = [
        candidates[index]
        for index, value in enumerate(result.x)
        if value > 0.5
    ]
    return selected, result


class BinaryMeasurementOptimizer:
    """SciPy/HiGHS standard binary optimizer for the paper objectives."""

    def __init__(
        self,
        context: OptimizationContext,
        *,
        cap_groups: int,
        time_limit: float = 300.0,
        random_partitions: int = 12,
        random_seed: int = 7,
        max_candidates: int = 4000,
        deadline=None,
    ):
        if cap_groups < 1:
            raise ValueError("cap_groups must be positive.")
        self.context = context
        self.cap_groups = int(cap_groups)
        self.time_limit = float(time_limit)
        self.random_partitions = int(random_partitions)
        self.random_seed = int(random_seed)
        self.max_candidates = int(max_candidates)
        self.deadline = (
            float(deadline)
            if deadline is not None
            else time.monotonic() + max(0.0, self.time_limit)
        )

    def _remaining(self):
        return max(0.0, self.deadline - time.monotonic())

    def optimize_non_overlapping(self) -> OptimizationResult:
        candidates, seed_partitions = _candidate_cliques(
            self.context,
            random_partitions=self.random_partitions,
            random_seed=self.random_seed,
            max_candidates=self.max_candidates,
            deadline=self.deadline,
        )
        fallback_groups = next(
            (
                partition
                for partition in seed_partitions
                if len(partition) <= self.cap_groups
            ),
            None,
        )
        if fallback_groups is None and self.context.n_terms <= self.cap_groups:
            fallback_groups = [
                [position] for position in range(self.context.n_terms)
            ]
        groups, solver_result = _solve_exact_cover_milp(
            self.context,
            candidates,
            self.cap_groups,
            self._remaining(),
            fallback_groups=fallback_groups,
        )
        feasible_starts = [groups]
        feasible_starts.extend(
            partition
            for partition in seed_partitions
            if len(partition) <= self.cap_groups
        )
        best_groups = None
        best_score = math.inf
        best_moves = 0
        starts_refined = 0
        for starting_groups in feasible_starts:
            if self._remaining() <= 0.0 and best_groups is not None:
                break
            refined_groups, accepted_moves = _refine_partition_by_relocation(
                self.context, starting_groups, deadline=self.deadline
            )
            starts_refined += 1
            refined_score = _partition_score(self.context, refined_groups)
            candidate_key = tuple(
                sorted(tuple(sorted(group)) for group in refined_groups)
            )
            best_key = (
                tuple(sorted(tuple(sorted(group)) for group in best_groups))
                if best_groups is not None
                else None
            )
            if (
                refined_score < best_score - 1.0e-12
                or (
                    abs(refined_score - best_score) <= 1.0e-12
                    and (best_key is None or candidate_key < best_key)
                )
            ):
                best_groups = refined_groups
                best_score = refined_score
                best_moves = accepted_moves

        groups = tuple(tuple(group) for group in best_groups)
        assignment = _assignment_from_groups(self.context.n_terms, groups)
        _validate_assignment(
            self.context,
            assignment,
            max_split=1,
            exact_once=True,
        )
        omega = self.context.coefficients[:, np.newaxis] * assignment
        variances, score, eps_sq_m = measurement_objective(
            self.context, assignment, omega
        )
        return OptimizationResult(
            mode="non-overlapping",
            groups=groups,
            assignment=assignment,
            omega=omega,
            variances=variances,
            score=score,
            eps_sq_m=eps_sq_m,
            cap_groups=self.cap_groups,
            max_split=1,
            solver_status=(
                "Candidate-pool MILP: {}; exact relocation starts={}, "
                "accepted_moves={}".format(
                    solver_result.message, starts_refined, best_moves
                )
            ),
            solver_objective=float(best_score),
            iterations=best_moves,
            n_candidates=len(candidates),
            history=((0, eps_sq_m),),
        )

    def optimize_nonoverlap_ics(
        self,
        nonoverlap: OptimizationResult | None = None,
        *,
        n_iterations: int = 5,
    ) -> OptimizationResult:
        """Run standard GFlow-VQE ICS from a binary non-overlap grouping."""

        if nonoverlap is None:
            nonoverlap = self.optimize_non_overlapping()
        if nonoverlap.mode != "non-overlapping":
            raise ValueError("Standard ICS requires a non-overlapping result.")
        if len(nonoverlap.groups) != nonoverlap.assignment.shape[1]:
            raise ValueError(
                "The non-overlap group and assignment columns differ."
            )
        assignment_groups = _groups_from_assignment(nonoverlap.assignment)
        if tuple(tuple(group) for group in nonoverlap.groups) != assignment_groups:
            raise ValueError(
                "The non-overlap group labels do not match assignment."
            )
        _validate_assignment(
            self.context,
            nonoverlap.assignment,
            max_split=1,
            exact_once=True,
        )
        assignment, omega, sample_ratios = (
            optimize_from_nonoverlapping_groups(
                self.context,
                nonoverlap.assignment,
                n_iter=n_iterations,
                condition="fc",
                deadline=self.deadline,
            )
        )
        deadline_reached = self._remaining() <= 0.0
        observed_max_split = int(assignment.sum(axis=1).max())
        _validate_assignment(
            self.context,
            assignment,
            max_split=observed_max_split,
            exact_once=False,
        )
        sample_ratios = np.asarray(sample_ratios, dtype=float).reshape(-1)
        if (
            sample_ratios.size != assignment.shape[1]
            or not np.all(np.isfinite(sample_ratios))
            or np.any(sample_ratios < 0.0)
            or not np.isclose(sample_ratios.sum(), 1.0)
        ):
            raise ValueError("Standard ICS returned invalid shot ratios.")
        variances, score, eps_sq_m = measurement_objective(
            self.context, assignment, omega
        )
        return OptimizationResult(
            mode="Bin(non-overlap)+ICS",
            groups=_groups_from_assignment(assignment),
            assignment=assignment,
            omega=omega,
            variances=variances,
            score=score,
            eps_sq_m=eps_sq_m,
            cap_groups=self.cap_groups,
            max_split=observed_max_split,
            solver_status=(
                "{} | Standard GFlow-VQE ICS initialized from binary "
                "non-overlap; iterations={}, observed_max_split={}{}".format(
                    nonoverlap.solver_status,
                    n_iterations,
                    observed_max_split,
                    "; TIME_LIMIT retained feasible coefficients"
                    if deadline_reached
                    else "",
                )
            ),
            solver_objective=eps_sq_m,
            iterations=n_iterations,
            n_candidates=nonoverlap.n_candidates,
            sample_ratios=tuple(float(value) for value in sample_ratios),
            history=((0, eps_sq_m),),
        )


__all__ = [
    "BinaryMeasurementOptimizer",
    "OptimizationContext",
    "OptimizationResult",
    "coloring_group_bound",
    "fixed_support_ics",
    "make_optimization_context",
    "measurement_objective",
    "relative_variance_weights",
    "singleton_measurement_bound",
]

# ---------------------------------------------------------------------------
# Measurement-aware candidate generation and refinement
# ---------------------------------------------------------------------------

import itertools
import time
from collections.abc import Mapping, Sequence
from typing import Any, Optional

from .measurement import (
    GroupingProblem,
    GroupingSolution,
    canonicalize_assignment,
    coerce_problem,
    evaluate_assignment,
    fragment_variances,
    greedy_feasible_assignment,
    groups_from_assignment,
    is_feasible,
    measurement_aware_edge_reward,
    proxy_objective,
    repair_assignment,
)

class RegisteredMethod:
    """Callable exact-name method registration with immutable routing data."""

    def __init__(self, name, family, clique_initializer):
        self.name = name
        self.family = family
        self.clique_initializer = bool(clique_initializer)
        self.__name__ = name

    def __call__(self, problem=None, **kwargs):
        return optimize_grouping(problem, method=self.name, **kwargs)

    def __getitem__(self, key):
        if key == "family":
            return self.family
        if key == "clique_initializer":
            return self.clique_initializer
        raise KeyError(key)


METHOD_REGISTRY = {
    "milp": RegisteredMethod("milp", "milp", True),
    "milp_lns": RegisteredMethod("milp_lns", "milp_lns", True),
}

_ITERATIVE_HISTORY_FAMILIES = frozenset({"milp_lns"})

AVAILABLE_METHODS = tuple(METHOD_REGISTRY)

def _solver_options(options: Optional[Mapping[str, Any]]) -> dict[str, Any]:
    out = {
        "time_limit": None,
        "threads": None,
        "seed": 0,
        "quiet": True,
        "mip_gap": None,
    }
    if options:
        out.update(dict(options))
    return out


def _finish_solution(
    problem: GroupingProblem,
    x: np.ndarray,
    *,
    method: str,
    status: str,
    runtime: float,
    weight_heuristic: str,
    heuristic_options: Optional[Mapping[str, Any]],
    overlap_penalty: float,
    node_costs: Optional[np.ndarray],
    rewards: np.ndarray,
    metadata: Optional[dict[str, Any]] = None,
    omega: Optional[np.ndarray] = None,
    history=None,
    observation_history=None,
) -> GroupingSolution:
    raw_x = np.asarray(x, dtype=np.int8).copy()
    x = canonicalize_assignment(raw_x)
    if not is_feasible(problem, x):
        raise RuntimeError(f"{method} returned an infeasible assignment")
    if omega is None:
        evaluated = evaluate_assignment(
            problem,
            x,
            weight_heuristic=weight_heuristic,
            heuristic_options=heuristic_options,
            overlap_penalty=overlap_penalty,
            node_costs=node_costs,
            rewards=rewards,
        )
        omega = evaluated["split_coefficients"]
        variances = evaluated["variances"]
        score = evaluated["score"]
        epsm = evaluated["epsilon2M"]
        proxy = evaluated["proxy_value"]
    else:
        # Reorder jointly optimized coefficients consistently with x.
        # canonicalize_assignment may permute columns, so identify the same order.
        original = raw_x
        omega = np.asarray(omega, dtype=float)
        # If no reliable column map is available, reconstruct by matching supports.
        reordered = np.zeros_like(omega)
        used = set()
        for a in range(x.shape[1]):
            target = tuple(x[:, a].tolist())
            found = None
            for b in range(original.shape[1]):
                if b not in used and tuple(original[:, b].tolist()) == target:
                    found = b
                    break
            if found is None:
                found = a
            used.add(found)
            reordered[:, a] = omega[:, found]
        omega = reordered
        variances = fragment_variances(problem, omega)
        score = float(np.sqrt(variances).sum())
        epsm = score * score
        proxy = proxy_objective(
            problem, x, overlap_penalty=overlap_penalty, node_costs=node_costs, rewards=rewards
        )
    return GroupingSolution(
        assignment=x,
        groups=groups_from_assignment(problem, x),
        split_coefficients=np.asarray(omega, dtype=float),
        variances=np.asarray(variances, dtype=float),
        score=float(score),
        epsilon2M=float(epsm),
        proxy_value=float(proxy),
        method=method,
        status=str(status),
        runtime=float(runtime),
        metadata=dict(metadata or {}),
        history=(
            [(0, float(epsm))]
            if history is None
            else list(history)
        ),
        observation_history=(
            [(0, float(epsm))]
            if observation_history is None
            else list(observation_history)
        ),
    )


def _lns_reconstruct(
    problem: GroupingProblem,
    incumbent: np.ndarray,
    destroyed: np.ndarray,
    rewards: np.ndarray,
    node_costs: np.ndarray,
    overlap_penalty: float,
    rng: np.random.Generator,
) -> np.ndarray:
    x = incumbent.copy()
    x[destroyed, :] = 0
    order = sorted(
        destroyed.tolist(),
        key=lambda i: int(np.sum(~problem.compatibility[i])),
        reverse=True,
    )
    for i in order:
        choices = []
        for a in range(problem.n_groups):
            members = np.flatnonzero(x[:, a])
            if members.size == 0 or np.all(problem.compatibility[i, members]):
                incremental = -float(rewards[i, members].sum())
                choices.append((incremental + 1.0e-8 * rng.random(), a))
        if not choices:
            return greedy_feasible_assignment(
                problem,
                rewards=rewards,
                overlap_penalty=overlap_penalty,
                random_state=int(rng.integers(2**31 - 1)),
                randomize=True,
            )
        _, a = min(choices)
        x[i, a] = 1
        # Add an overlap only when its pairwise reward pays for the regularizer.
        if problem.max_overlap[i] > 1 and rng.random() < 0.65:
            extras = []
            for b in range(problem.n_groups):
                if b == a:
                    continue
                members = np.flatnonzero(x[:, b])
                if (members.size == 0 or np.all(problem.compatibility[i, members])):
                    delta = float(
                        overlap_penalty
                        + node_costs[i]
                        - rewards[i, members].sum()
                    )
                    extras.append((delta, b))
            for delta, b in sorted(extras):
                if delta < 0.0 and x[i].sum() < problem.max_overlap[i] and rng.random() < 0.75:
                    x[i, b] = 1
    return canonicalize_assignment(x)


def _refine_feasible_lns(
    problem: GroupingProblem,
    *,
    overlap_penalty: float = 0.0,
    node_costs: Optional[np.ndarray] = None,
    weight_heuristic: str = "iterative_relative_std",
    heuristic_options: Optional[Mapping[str, Any]] = None,
    solver_options: Optional[Mapping[str, Any]] = None,
    initial_assignment: Optional[np.ndarray] = None,
) -> GroupingSolution:
    """Feasible large-neighborhood search ranked by the true postprocessed cost."""
    options = dict(solver_options or {})
    opts = _solver_options(options)
    seed = int(opts.get("seed") or 0)
    node_scores, _, R = measurement_aware_edge_reward(problem)
    node = node_scores if node_costs is None else np.asarray(node_costs, dtype=float)
    start = time.monotonic()
    if options.get("deadline") is not None:
        deadline = float(options["deadline"])
    elif opts.get("time_limit") is not None:
        deadline = start + max(0.0, float(opts["time_limit"]))
    else:
        deadline = math.inf
    # Keep a useful search budget on small instances without making the default
    # grow linearly to several thousand full objective evaluations on large ones.
    default_iterations = max(
        100,
        min(1000, int(math.ceil(200000.0 / max(problem.n_terms, 1)))),
    )
    iterations = max(0, int(options.get("iterations", default_iterations)))
    restarts = max(1, int(options.get("restarts", 1)))
    destroy_fraction = float(options.get("destroy_fraction", 0.2))
    min_destroy = int(options.get("min_destroy", 1))
    accepted = 0
    evaluated = 0
    total_iterations = 0
    restarts_completed = 0
    restart_records = []
    best = None
    best_eval = None
    seen = set()
    history = []
    observation_history = []

    for restart in range(restarts):
        if restart > 0 and time.monotonic() >= deadline:
            break
        restart_seed = seed + restart
        rng = np.random.default_rng(restart_seed)
        if restart == 0 and initial_assignment is not None:
            incumbent = repair_assignment(
                problem,
                initial_assignment,
                rewards=R,
                overlap_penalty=overlap_penalty,
                random_state=restart_seed,
            )
        else:
            incumbent = greedy_feasible_assignment(
                problem,
                rewards=R,
                overlap_penalty=overlap_penalty,
                random_state=restart_seed,
                randomize=restart > 0,
            )
        incumbent_eval = evaluate_assignment(
            problem,
            incumbent,
            weight_heuristic=weight_heuristic,
            heuristic_options=heuristic_options,
            overlap_penalty=overlap_penalty,
            node_costs=node,
            rewards=R,
        )
        evaluated += 1
        observation_history.append(
            (len(observation_history), float(incumbent_eval["epsilon2M"]))
        )
        incumbent_key = np.packbits(
            incumbent.ravel().astype(np.uint8)
        ).tobytes()
        seen.add(incumbent_key)
        if (
            best_eval is None
            or incumbent_eval["epsilon2M"]
            < best_eval["epsilon2M"] - 1.0e-14
        ):
            best = incumbent.copy()
            best_eval = incumbent_eval
        history.append((len(history), float(best_eval["epsilon2M"])))
        restart_iterations = 0
        restart_accepted = 0
        for iteration in range(iterations):
            if time.monotonic() >= deadline:
                break
            restart_iterations += 1
            total_iterations += 1
            k = min(
                problem.n_terms,
                max(
                    min_destroy,
                    int(math.ceil(destroy_fraction * problem.n_terms)),
                ),
            )
            if rng.random() < 0.5:
                row_impact = np.sum(
                    np.abs(incumbent_eval["split_coefficients"]), axis=1
                )
                probs = (
                    row_impact
                    + (float(row_impact.mean()) if row_impact.size else 0.0)
                    * 0.05
                    + 1.0e-12
                )
                probs = probs / probs.sum()
                destroyed = rng.choice(
                    problem.n_terms, size=k, replace=False, p=probs
                )
            else:
                destroyed = rng.choice(
                    problem.n_terms, size=k, replace=False
                )
            candidate = _lns_reconstruct(
                problem,
                incumbent,
                np.asarray(destroyed),
                R,
                node,
                overlap_penalty,
                rng,
            )
            if not is_feasible(problem, candidate):
                continue
            candidate_key = np.packbits(
                candidate.ravel().astype(np.uint8)
            ).tobytes()
            if candidate_key in seen:
                continue
            seen.add(candidate_key)
            cand_eval = evaluate_assignment(
                problem,
                candidate,
                weight_heuristic=weight_heuristic,
                heuristic_options=heuristic_options,
                overlap_penalty=overlap_penalty,
                node_costs=node,
                rewards=R,
            )
            evaluated += 1
            observation_history.append(
                (len(observation_history), float(cand_eval["epsilon2M"]))
            )
            delta = float(
                cand_eval["epsilon2M"] - incumbent_eval["epsilon2M"]
            )
            if delta < -1.0e-14:
                incumbent, incumbent_eval = candidate, cand_eval
                accepted += 1
                restart_accepted += 1
            if (
                best_eval is None
                or cand_eval["epsilon2M"]
                < best_eval["epsilon2M"] - 1.0e-14
            ):
                best, best_eval = candidate.copy(), cand_eval
            history.append((len(history), float(best_eval["epsilon2M"])))
        restarts_completed += 1
        restart_records.append(
            {
                "restart": restart,
                "seed": restart_seed,
                "iterations": restart_iterations,
                "accepted_moves": restart_accepted,
                "incumbent_epsilon2M": float(incumbent_eval["epsilon2M"]),
            }
        )
    if best is None or best_eval is None:
        raise RuntimeError("LNS failed to construct a feasible incumbent")
    elapsed = time.monotonic() - start
    timed_out = time.monotonic() >= deadline
    return _finish_solution(
        problem,
        best,
        method="milp_lns",
        status="TIME_LIMIT_FEASIBLE" if timed_out else "FEASIBLE_LNS",
        runtime=elapsed,
        weight_heuristic=weight_heuristic,
        heuristic_options=heuristic_options,
        overlap_penalty=overlap_penalty,
        node_costs=node,
        rewards=R,
        metadata={
            "iterations": total_iterations,
            "iterations_per_restart": iterations,
            "restarts_requested": restarts,
            "restarts_completed": restarts_completed,
            "restart_records": tuple(restart_records),
            "evaluations": evaluated,
            "unique_supports": len(seen),
            "accepted_moves": accepted,
            "deadline_reached": timed_out,
            "solver_threads_requested": options.get("threads"),
            "solver_threads_used": 1,
        },
        history=history,
        observation_history=observation_history,
    )


def coerce_problem_from_legacy_call(args: Sequence[Any], kwargs: Mapping[str, Any]) -> GroupingProblem:
    """Best-effort adapter used by the minimally patched legacy driver."""
    names = {
        "coefficients": ("coefficients", "coeffs", "c", "hamiltonian_coefficients"),
        "covariance": ("covariance", "cov", "cov_matrix", "C"),
        "compatibility": ("compatibility", "adjacency", "commutation_matrix"),
        "n_groups": ("n_groups", "num_groups", "N_f", "n_fragments"),
        "max_overlap": ("max_overlap", "max_overlaps", "overlap_limit"),
        "labels": ("labels", "paulis", "terms"),
    }
    found: dict[str, Any] = {}
    for target, aliases in names.items():
        for alias in aliases:
            if alias in kwargs:
                found[target] = kwargs[alias]
                break
    for value in args:
        if isinstance(value, GroupingProblem):
            return value
        if isinstance(value, (nx.Graph, Mapping)):
            try:
                return coerce_problem(
                    value,
                    n_groups=found.get("n_groups"),
                    max_overlap=found.get("max_overlap", 1),
                )
            except Exception:
                pass
    arrays = []
    for value in list(args) + list(kwargs.values()):
        try:
            arr = np.asarray(value)
        except Exception:
            continue
        if arr.dtype == object and arr.ndim == 0:
            continue
        arrays.append(arr)
    if "coefficients" not in found:
        for arr in arrays:
            if arr.ndim == 1 and np.issubdtype(arr.dtype, np.number):
                found["coefficients"] = arr
                break
    n = len(np.asarray(found.get("coefficients", [])).reshape(-1))
    for arr in arrays:
        if arr.shape != (n, n):
            continue
        if "compatibility" not in found and (arr.dtype == bool or np.array_equal(arr, arr.astype(bool))):
            found["compatibility"] = arr.astype(bool)
        elif "covariance" not in found and np.issubdtype(arr.dtype, np.number):
            found["covariance"] = arr.astype(float)
    if "n_groups" not in found:
        integers = [v for v in args if isinstance(v, (int, np.integer)) and int(v) > 0]
        found["n_groups"] = int(integers[0]) if integers else max(n, 1)
    return coerce_problem(
        coefficients=found.get("coefficients"),
        covariance=found.get("covariance"),
        compatibility=found.get("compatibility"),
        n_groups=found.get("n_groups"),
        max_overlap=found.get("max_overlap", 1),
        labels=found.get("labels"),
    )


def _absolute_deadline(options):
    options = dict(options or {})
    if options.get("deadline") is not None:
        return float(options["deadline"])
    if options.get("time_limit") is None:
        return math.inf
    return time.monotonic() + max(0.0, float(options["time_limit"]))


def _remaining(deadline):
    if math.isinf(deadline):
        return None
    return max(0.0, float(deadline) - time.monotonic())


def _options_with_remaining(options, deadline):
    updated = dict(options or {})
    remaining = _remaining(deadline)
    if remaining is not None:
        updated["time_limit"] = remaining
    updated["deadline"] = deadline
    return updated


def _clone_problem(problem, *, max_overlap=None, n_groups=None):
    return GroupingProblem(
        coefficients=np.asarray(problem.coefficients, dtype=float).copy(),
        covariance=np.asarray(problem.covariance, dtype=float).copy(),
        compatibility=np.asarray(problem.compatibility, dtype=bool).copy(),
        n_groups=problem.n_groups if n_groups is None else int(n_groups),
        max_overlap=problem.max_overlap if max_overlap is None else max_overlap,
        labels=list(problem.labels),
    )


def _canonical_partition(groups):
    return tuple(sorted(tuple(sorted(int(item) for item in group)) for group in groups))


def _bounded_clique_cover(compatible, bound, *, seed, deadline):
    """Construct a bounded compatible partition with deadline-aware search."""

    compatible = np.asarray(compatible, dtype=bool)
    n_terms = int(compatible.shape[0])
    bound = int(bound)
    if bound < 1:
        raise ValueError("group_bound must be positive")
    if n_terms == 0:
        return []
    if bound >= n_terms:
        return [[index] for index in range(n_terms)]
    if bool(np.all(compatible)):
        return [list(range(n_terms))]

    incompatibility_degree = np.sum(~compatible, axis=1)
    rng = np.random.default_rng(int(seed))
    orders = [
        sorted(
            range(n_terms),
            key=lambda index: (-int(incompatibility_degree[index]), index),
        ),
        list(range(n_terms)),
    ]
    if deadline is None or time.monotonic() < float(deadline):
        orders.append(rng.permutation(n_terms).tolist())
    for order in orders:
        groups = []
        for position in order:
            choices = [
                group_index
                for group_index, group in enumerate(groups)
                if all(compatible[position, other] for other in group)
            ]
            if choices:
                group_index = max(choices, key=lambda index: len(groups[index]))
                groups[group_index].append(int(position))
            elif len(groups) < bound:
                groups.append([int(position)])
            else:
                break
        else:
            return [sorted(group) for group in groups]

    incompatibility = nx.Graph()
    incompatibility.add_nodes_from(range(n_terms))
    edge_rows, edge_columns = np.nonzero(
        np.triu(~compatible, k=1)
    )
    incompatibility.add_edges_from(
        (int(left), int(right))
        for left, right in zip(edge_rows, edge_columns)
    )
    dsatur = nx.coloring.greedy_color(
        incompatibility,
        strategy="saturation_largest_first",
        interchange=False,
    )
    used_colors = 1 + max(dsatur.values(), default=-1)
    if used_colors <= bound:
        return [
            sorted(
                node for node, color in dsatur.items() if color == group_index
            )
            for group_index in range(used_colors)
        ]

    colors = np.full(n_terms, -1, dtype=int)
    expired = False
    deadline_expired_on_entry = bool(
        deadline is not None and time.monotonic() >= float(deadline)
    )
    seed_search_allowed = deadline_expired_on_entry and n_terms <= 24
    search_nodes = 0
    seed_node_limit = 100_000

    def search(colored_count):
        nonlocal expired, search_nodes
        search_nodes += 1
        if seed_search_allowed and search_nodes > seed_node_limit:
            expired = True
            return False
        if (
            not seed_search_allowed
            and deadline is not None
            and time.monotonic() >= float(deadline)
        ):
            expired = True
            return False
        if colored_count == n_terms:
            return True
        uncolored = np.flatnonzero(colors < 0)
        node = max(
            uncolored,
            key=lambda index: (
                len(
                    {
                        int(colors[other])
                        for other in range(n_terms)
                        if not compatible[index, other] and colors[other] >= 0
                    }
                ),
                int(incompatibility_degree[index]),
                -int(index),
            ),
        )
        forbidden = {
            int(colors[other])
            for other in range(n_terms)
            if not compatible[node, other] and colors[other] >= 0
        }
        used = sorted(set(int(value) for value in colors if value >= 0))
        color_order = [color for color in used if color not in forbidden]
        if len(used) < bound:
            color_order.append(len(used))
        for color in color_order:
            colors[node] = color
            if search(colored_count + 1):
                return True
            colors[node] = -1
        return False

    if search(0):
        return [
            sorted(np.flatnonzero(colors == color).tolist())
            for color in sorted(set(colors.tolist()))
        ]
    if expired:
        raise RuntimeError(
            "deadline expired before a group-bounded feasible cover was found"
        )
    raise ValueError("group_bound is too small for the compatibility graph")


def generate_candidate_pool(
    problem_or_context,
    *,
    random_partitions=12,
    max_candidate_groups=4000,
    seed=7,
    group_bound=None,
    deadline=None,
):
    """Build one reproducible validated clique pool for all clique methods.

    The return value contains canonical candidates, a feasible exact cover, and
    metadata recording the requested/effective limits. Singleton columns are
    always reserved before any larger clique is admitted.
    """

    requested = int(max_candidate_groups)
    if requested < 1:
        raise ValueError("max_candidate_groups must be positive")
    random_partitions = max(0, int(random_partitions))
    seed = int(seed)

    if isinstance(problem_or_context, OptimizationContext):
        context = problem_or_context
        n_terms = context.n_terms
        compatible = context.compatible
        bound = n_terms if group_bound is None else int(group_bound)
        if deadline is not None and time.monotonic() >= float(deadline):
            partitions = []
            growth_orders = []
            feasible = [[position] for position in range(n_terms)]
            deterministic = []
            growth = []
        else:
            partitions, growth_orders = _seed_partitions(
                context,
                random_partitions=random_partitions,
                random_seed=seed,
                deadline=deadline,
            )
            feasible = next(
                (partition for partition in partitions if len(partition) <= bound),
                None,
            )
            if feasible is None:
                feasible = [[position] for position in range(n_terms)]
            deterministic = [
                group for partition in partitions for group in partition
            ]
            growth = []
            for group in deterministic:
                if deadline is not None and time.monotonic() >= float(deadline):
                    break
                for order in [*growth_orders, list(range(n_terms))]:
                    if deadline is not None and time.monotonic() >= float(deadline):
                        break
                    growth.append(_grow_clique(context, group, order))
    elif isinstance(problem_or_context, GroupingProblem):
        problem = problem_or_context
        n_terms = problem.n_terms
        compatible = problem.compatibility
        bound = problem.n_groups if group_bound is None else int(group_bound)
        if deadline is not None and time.monotonic() >= float(deadline):
            partitions = []
            feasible = [[position] for position in range(n_terms)]
            deterministic = []
            growth = []
        else:
            graph = nx.Graph()
            graph.add_nodes_from(range(n_terms))
            for left in range(n_terms):
                if deadline is not None and time.monotonic() >= float(deadline):
                    break
                for right in range(left + 1, n_terms):
                    if not compatible[left, right]:
                        graph.add_edge(left, right)
            partitions = []
            for strategy in (
                "saturation_largest_first",
                "largest_first",
                "smallest_last",
            ):
                if deadline is not None and time.monotonic() >= float(deadline):
                    break
                partitions.append(
                    _groups_from_coloring(
                        nx.coloring.greedy_color(graph, strategy=strategy)
                    )
                )
            rng = np.random.default_rng(seed)
            for _ in range(random_partitions):
                if deadline is not None and time.monotonic() >= float(deadline):
                    break
                order = rng.permutation(n_terms).tolist()
                groups = []
                for position in order:
                    for group in groups:
                        if all(compatible[position, other] for other in group):
                            group.append(int(position))
                            break
                    else:
                        groups.append([int(position)])
                partitions.append([sorted(group) for group in groups])
            feasible = next(
                (partition for partition in partitions if len(partition) <= bound),
                None,
            )
            if feasible is None:
                feasible = [[position] for position in range(n_terms)]
            deterministic = [
                group for partition in partitions for group in partition
            ]
            growth = []
            orders = [
                list(range(n_terms)),
                sorted(
                    range(n_terms),
                    key=lambda item: (-abs(problem.coefficients[item]), item),
                ),
            ]
            for seed_group in deterministic:
                if deadline is not None and time.monotonic() >= float(deadline):
                    break
                for order in orders:
                    group = list(
                        dict.fromkeys(int(item) for item in seed_group)
                    )
                    for position in order:
                        if position not in group and all(
                            compatible[position, other] for other in group
                        ):
                            group.append(position)
                    growth.append(tuple(sorted(group)))
    else:
        raise TypeError(
            "problem_or_context must be GroupingProblem or OptimizationContext"
        )

    if len(feasible) > bound:
        feasible = _bounded_clique_cover(
            compatible,
            bound,
            seed=seed,
            deadline=deadline,
        )
    proposed_cover = _canonical_partition(feasible)
    if len(proposed_cover) > bound:
        raise RuntimeError("candidate pool failed to retain a group-bounded cover")
    base_effective_limit = max(n_terms, requested)
    effective_limit = base_effective_limit
    candidates = {}
    for position in range(n_terms):
        candidates[(position,)] = None
    additional_cover_groups = [
        group for group in proposed_cover if group not in candidates
    ]
    if len(candidates) + len(additional_cover_groups) <= effective_limit:
        feasible_cover = proposed_cover
        for group in additional_cover_groups:
            candidates[group] = None
    elif n_terms <= bound:
        feasible_cover = tuple((position,) for position in range(n_terms))
    else:
        effective_limit = len(candidates) + len(additional_cover_groups)
        feasible_cover = proposed_cover
        for group in additional_cover_groups:
            candidates[group] = None
    for raw_group in [*deterministic, *growth]:
        if deadline is not None and time.monotonic() >= float(deadline):
            break
        group = tuple(sorted(int(item) for item in raw_group))
        if not group or group in candidates:
            continue
        if len(candidates) >= effective_limit:
            break
        candidates[group] = None

    covered = set()
    for group in candidates:
        for offset, left in enumerate(group):
            if left < 0 or left >= n_terms:
                raise RuntimeError("candidate clique contains an invalid term index")
            covered.add(left)
            for right in group[offset + 1 :]:
                if not compatible[left, right]:
                    raise RuntimeError("candidate pool contains an incompatible clique")
    for position in range(n_terms):
        if position not in covered or (position,) not in candidates:
            raise RuntimeError("candidate pool lost singleton coverage")
    if any(group not in candidates for group in feasible_cover):
        raise RuntimeError("candidate pool lost its feasible cover")
    metadata = {
        "requested_limit": requested,
        "base_effective_limit": base_effective_limit,
        "effective_limit": effective_limit,
        "effective_limit_expanded_for_feasible_cover": bool(
            effective_limit > base_effective_limit
        ),
        "actual_candidate_count": len(candidates),
        "random_partitions": random_partitions,
        "seed": seed,
        "group_bound": bound,
        "singleton_count": n_terms,
        "fallback_is_all_singletons": all(len(group) == 1 for group in feasible_cover),
        "deadline_reached": bool(
            deadline is not None and time.monotonic() >= float(deadline)
        ),
    }
    return CandidatePool(
        candidates=tuple(candidates),
        feasible_cover=feasible_cover,
        metadata=metadata,
    )


def generate_candidate_clique_pool(
    problem_or_context,
    *,
    random_partitions=12,
    seed=7,
    max_candidate_groups=4000,
    group_bound=None,
    deadline=None,
):
    """Compatibility spelling for the public shared-pool constructor."""

    return generate_candidate_pool(
        problem_or_context,
        random_partitions=random_partitions,
        max_candidate_groups=max_candidate_groups,
        seed=seed,
        group_bound=group_bound,
        deadline=deadline,
    )


def _exact_clique_cost(problem, clique):
    members = np.asarray(clique, dtype=int)
    coefficients = np.asarray(problem.coefficients, dtype=float)[members]
    covariance = np.asarray(problem.covariance, dtype=float)[
        np.ix_(members, members)
    ]
    variance = float(coefficients @ covariance @ coefficients)
    if variance < -1.0e-8:
        raise ValueError("candidate clique has a negative fragment variance")
    return math.sqrt(max(0.0, variance))


def _assignment_from_candidate_cover(problem, selected):
    if len(selected) > problem.n_groups:
        raise RuntimeError("candidate cover exceeds the available group bound")
    assignment = np.zeros(
        (problem.n_terms, problem.n_groups), dtype=np.int8
    )
    for alpha, clique in enumerate(selected):
        assignment[list(clique), alpha] = 1
    if not is_feasible(problem, assignment):
        raise RuntimeError("candidate-cover backend returned an infeasible partition")
    if not np.all(assignment.sum(axis=1) == 1):
        raise RuntimeError("candidate-cover backend did not return exact coverage")
    return assignment


def _solve_candidate_pool(problem, pool, *, family, solver_options, deadline):
    candidates = list(pool.candidates)
    fallback = list(pool.feasible_cover)
    if len(fallback) > problem.n_groups:
        fallback_assignment = greedy_feasible_assignment(
            _clone_problem(problem, max_overlap=1),
            random_state=int(dict(solver_options or {}).get("seed", 0)),
        )
        fallback = [
            tuple(np.flatnonzero(fallback_assignment[:, alpha]).tolist())
            for alpha in range(problem.n_groups)
            if np.any(fallback_assignment[:, alpha])
        ]
    augmented = 0
    known = set(candidates)
    for clique in fallback:
        clique = tuple(clique)
        if clique not in known:
            candidates.append(clique)
            known.add(clique)
            augmented += 1
    costs = np.asarray(
        [_exact_clique_cost(problem, clique) for clique in candidates],
        dtype=float,
    )
    remaining = _remaining(deadline)
    if remaining is not None and remaining <= 0.0:
        return _assignment_from_candidate_cover(problem, fallback), {
            "candidate_solver": "fallback",
            "candidate_status": "TIME_LIMIT",
            "candidate_selector_variable_kind": "z_C",
            "candidate_pool_local_augmentation": augmented,
            **pool.metadata,
            "candidate_pool_deadline_reached": bool(
                pool.metadata.get("deadline_reached", False)
            ),
            "candidate_selector_deadline_reached": True,
        }

    if family not in AVAILABLE_METHODS:
        raise ValueError("unsupported candidate-cover method {!r}".format(family))
    selected = None
    incidence = lil_matrix(
        (problem.n_terms, len(candidates)), dtype=float
    )
    for index, clique in enumerate(candidates):
        incidence[list(clique), index] = 1.0
    matrix = vstack(
        [incidence.tocsc(), csc_matrix(np.ones((1, len(candidates))))],
        format="csc",
    )
    lower = np.concatenate([np.ones(problem.n_terms), [-np.inf]])
    upper = np.concatenate(
        [np.ones(problem.n_terms), [float(problem.n_groups)]]
    )
    options = {"presolve": True, "mip_rel_gap": 0.0}
    remaining = _remaining(deadline)
    if remaining is not None:
        options["time_limit"] = max(1.0e-6, remaining)
    result = scipy_milp(
        c=costs,
        integrality=np.ones(len(candidates), dtype=int),
        bounds=Bounds(
            np.zeros(len(candidates)), np.ones(len(candidates))
        ),
        constraints=LinearConstraint(matrix, lower, upper),
        options=options,
    )
    if result.x is not None:
        selected = [
            candidates[index]
            for index, value in enumerate(result.x)
            if value > 0.5
        ]
    status = str(result.message)
    backend = "SciPy/HiGHS candidate-cover MILP"
    selector_metadata = {
        "selector_backend": "scipy_highs_milp",
        "selector_kind": "candidate_cover",
        "selector_variable_kind": "z_C",
        "solver_threads_requested": dict(solver_options or {}).get(
            "threads"
        ),
        "solver_threads_used": None,
    }

    if not selected:
        selected = fallback
        status = "{}; feasible fallback cover".format(status)
    selector_metadata.setdefault(
        "selector_backend",
        selector_metadata.get("backend_family", family),
    )
    selector_metadata.setdefault("selector_kind", "candidate_cover")
    selector_metadata.setdefault("selector_variable_kind", "z_C")
    selector_history = tuple(selector_metadata.pop("history", ()) or ())
    selector_observation_history = tuple(
        selector_metadata.pop("observation_history", ()) or ()
    )
    assignment = _assignment_from_candidate_cover(problem, selected)
    return assignment, {
        "candidate_solver": backend,
        "candidate_status": status,
        "candidate_pool_local_augmentation": augmented,
        **pool.metadata,
        **selector_metadata,
        "candidate_selector_history": selector_history,
        "candidate_selector_observation_history": (
            selector_observation_history
        ),
        "candidate_pool_deadline_reached": bool(
            pool.metadata.get("deadline_reached", False)
        ),
        "candidate_selector_deadline_reached": bool(
            selector_metadata.get("deadline_reached", False)
        ),
    }


def _solution_from_exact_evaluation(
    problem,
    assignment,
    *,
    method,
    status,
    runtime,
    weight_heuristic,
    heuristic_options,
    overlap_penalty,
    node_costs,
    metadata,
    history,
    observation_history=None,
):
    evaluated = evaluate_assignment(
        problem,
        assignment,
        weight_heuristic=weight_heuristic,
        heuristic_options=heuristic_options,
        overlap_penalty=overlap_penalty,
        node_costs=node_costs,
    )
    exact_history = list(history or [(0, float(evaluated["epsilon2M"]))])
    if not exact_history:
        exact_history = [(0, float(evaluated["epsilon2M"]))]
    exact_observation_history = list(
        observation_history or [(0, float(evaluated["epsilon2M"]))]
    )
    if not exact_observation_history:
        exact_observation_history = [(0, float(evaluated["epsilon2M"]))]
    return GroupingSolution(
        assignment=np.asarray(evaluated["assignment"], dtype=np.int8),
        groups=groups_from_assignment(problem, evaluated["assignment"]),
        split_coefficients=np.asarray(
            evaluated["split_coefficients"], dtype=float
        ),
        variances=np.asarray(evaluated["variances"], dtype=float),
        score=float(evaluated["score"]),
        epsilon2M=float(evaluated["epsilon2M"]),
        proxy_value=float(evaluated["proxy_value"]),
        method=method,
        status=status,
        runtime=float(runtime),
        metadata=dict(metadata or {}),
        history=exact_history,
        observation_history=exact_observation_history,
    )


def _row_supports(n_groups, maximum):
    supports = []
    for count in range(1, min(int(maximum), n_groups) + 1):
        supports.extend(itertools.combinations(range(n_groups), count))
    return supports


def _exact_rank_candidates(
    problem,
    initial_assignments,
    *,
    method,
    weight_heuristic,
    heuristic_options,
    overlap_penalty,
    node_costs,
    deadline,
    exhaustive_limit,
    metadata,
    start_time,
):
    best_assignment = None
    best_evaluation = None
    seen = set()
    history = []
    ranking_observation_history = []
    evaluations = 0

    def consider(raw_assignment):
        nonlocal best_assignment, best_evaluation, evaluations
        assignment = canonicalize_assignment(raw_assignment)
        key = np.packbits(assignment.ravel().astype(np.uint8)).tobytes()
        if key in seen or not is_feasible(problem, assignment):
            return
        seen.add(key)
        evaluated = evaluate_assignment(
            problem,
            assignment,
            weight_heuristic=weight_heuristic,
            heuristic_options=heuristic_options,
            overlap_penalty=overlap_penalty,
            node_costs=node_costs,
        )
        evaluations += 1
        ranking_observation_history.append(
            (len(ranking_observation_history), float(evaluated["epsilon2M"]))
        )
        if (
            best_evaluation is None
            or evaluated["epsilon2M"]
            < best_evaluation["epsilon2M"] - 1.0e-12
        ):
            best_assignment = np.asarray(
                evaluated["assignment"], dtype=np.int8
            ).copy()
            best_evaluation = evaluated
            history.append((len(history), float(evaluated["epsilon2M"])))

    for assignment in initial_assignments:
        if assignment is not None:
            consider(assignment)
    if best_assignment is None:
        consider(
            greedy_feasible_assignment(
                problem,
                random_state=int(metadata.get("seed", 0)),
            )
        )

    row_options = [
        _row_supports(problem.n_groups, problem.max_overlap[index])
        for index in range(problem.n_terms)
    ]
    total_candidates = 1
    for options in row_options:
        total_candidates *= len(options)
        if total_candidates > int(exhaustive_limit):
            break
    exhaustive = total_candidates <= int(exhaustive_limit)
    completed = False
    if exhaustive:
        completed = True
        for rows in itertools.product(*row_options):
            remaining = _remaining(deadline)
            if remaining is not None and remaining <= 0.0:
                completed = False
                break
            assignment = np.zeros(
                (problem.n_terms, problem.n_groups), dtype=np.int8
            )
            for term_index, columns in enumerate(rows):
                assignment[term_index, list(columns)] = 1
            consider(assignment)

    if best_assignment is None:
        raise RuntimeError("no feasible grouping incumbent was constructed")
    deadline_reached = (
        _remaining(deadline) is not None and _remaining(deadline) <= 0.0
    )
    if exhaustive and completed:
        status = "OPTIMAL_EXHAUSTIVE_EXACT"
    elif deadline_reached:
        status = "TIME_LIMIT_FEASIBLE"
    else:
        status = "EXACT_RANKED_CANDIDATES"
    metadata = {
        **metadata,
        "exact_candidate_ranking": True,
        "exhaustive_exact": bool(exhaustive and completed),
        "enumeration_candidate_bound": int(total_candidates),
        "exact_evaluations": evaluations,
        "unique_supports": len(seen),
        "deadline_reached": deadline_reached,
    }
    backend_histories = []
    for key in ("backend_exact_history", "milp_lns_exact_history"):
        raw = metadata.get(key, ())
        if raw:
            backend_histories.extend(raw)
    if backend_histories:
        combined_history = []
        running_best = math.inf
        final_objective = float(best_evaluation["epsilon2M"])
        for _, raw_value in [*backend_histories, *history]:
            value = float(raw_value)
            if value < final_objective - 1.0e-9:
                continue
            if value <= running_best + 1.0e-12:
                running_best = min(running_best, value)
                combined_history.append((len(combined_history), running_best))
        if (
            not combined_history
            or abs(combined_history[-1][1] - final_objective) > 1.0e-12
        ):
            combined_history.append((len(combined_history), final_objective))
        history = combined_history
    optimizer_observations = []
    iterative_history = (
        METHOD_REGISTRY[method]["family"] in _ITERATIVE_HISTORY_FAMILIES
    )
    if iterative_history:
        for key in (
            "candidate_selector_observation_history",
            "backend_observation_history",
            "milp_lns_observation_history",
        ):
            raw = metadata.get(key, ())
            if raw:
                optimizer_observations.extend(raw)
    if iterative_history:
        final_objective = float(best_evaluation["epsilon2M"])
        consistency_tolerance = 1.0e-9 * max(1.0, abs(final_objective))
        observation_history = []
        history = []
        running_best = math.inf
        raw_observations = (
            optimizer_observations
            if optimizer_observations
            else ranking_observation_history
        )
        for _, raw_value in raw_observations:
            value = float(raw_value)
            if value < final_objective - consistency_tolerance:
                raise RuntimeError(
                    "optimizer observation is better than the selected exact "
                    "solution"
                )
            # The optimizer callback and final exact ranking evaluate the same
            # objective. Clamp only roundoff-scale undershoots so that the
            # cumulative trace cannot finish below the returned objective.
            value = max(value, final_objective)
            observation_history.append((len(observation_history), value))
            running_best = min(running_best, value)
            history.append((len(history), running_best))
        if not history or history[-1][1] > final_objective + 1.0e-12:
            observation_history.append(
                (len(observation_history), final_objective)
            )
            history.append((len(history), final_objective))
        else:
            history[-1] = (history[-1][0], final_objective)
    else:
        observation_history = ranking_observation_history
    return _solution_from_exact_evaluation(
        problem,
        best_assignment,
        method=method,
        status=status,
        runtime=time.monotonic() - start_time,
        weight_heuristic=weight_heuristic,
        heuristic_options=heuristic_options,
        overlap_penalty=overlap_penalty,
        node_costs=node_costs,
        metadata=metadata,
        history=history,
        observation_history=observation_history,
    )


def _candidate_configuration(candidate_options, solver_options, problem):
    candidate_options = dict(candidate_options or {})
    seed = int(
        candidate_options.get(
            "seed", dict(solver_options or {}).get("seed", 0)
        )
    )
    return {
        "random_partitions": int(
            candidate_options.get("random_partitions", 12)
        ),
        "max_candidate_groups": int(
            candidate_options.get("max_candidate_groups", 4000)
        ),
        "seed": seed,
        "group_bound": int(
            candidate_options.get("group_bound", problem.n_groups)
        ),
    }


def _pad_assignment(assignment, n_groups):
    assignment = np.asarray(assignment, dtype=np.int8)
    if assignment.shape[1] > n_groups:
        raise RuntimeError("initializer uses more than the available groups")
    padded = np.zeros((assignment.shape[0], n_groups), dtype=np.int8)
    padded[:, : assignment.shape[1]] = assignment
    return padded


def _nonoverlap_initial_candidates(
    problem,
    *,
    method,
    weight_heuristic,
    heuristic_options,
    overlap_penalty,
    node_costs,
    solver_options,
    candidate_options,
    deadline,
    legacy_context,
):
    descriptor = METHOD_REGISTRY[method]
    family = descriptor["family"]
    nonoverlap_problem = _clone_problem(problem, max_overlap=1)
    metadata = {
        "seed": int(dict(solver_options or {}).get("seed", 0)),
        "initialization_method": method,
        "clique_initializer": bool(descriptor["clique_initializer"]),
    }
    candidates = [
        greedy_feasible_assignment(
            nonoverlap_problem,
            random_state=int(metadata["seed"]),
        )
    ]
    pool = None
    if descriptor["clique_initializer"]:
        configuration = _candidate_configuration(
            candidate_options, solver_options, nonoverlap_problem
        )
        pool = generate_candidate_pool(
            nonoverlap_problem, deadline=deadline, **configuration
        )
        pool_family = "milp" if family == "milp_lns" else family
        pool_assignment, pool_metadata = _solve_candidate_pool(
            nonoverlap_problem,
            pool,
            family=pool_family,
            solver_options=solver_options,
            deadline=deadline,
        )
        candidates.append(pool_assignment)
        metadata.update(pool_metadata)

    if (
        legacy_context is not None
        and (_remaining(deadline) is None or _remaining(deadline) > 0.0)
    ):
        configuration = _candidate_configuration(
            candidate_options, solver_options, nonoverlap_problem
        )
        legacy_optimizer = BinaryMeasurementOptimizer(
            legacy_context,
            cap_groups=nonoverlap_problem.n_groups,
            time_limit=(
                300.0
                if _remaining(deadline) is None
                else _remaining(deadline)
            ),
            random_partitions=configuration["random_partitions"],
            random_seed=configuration["seed"],
            max_candidates=configuration["max_candidate_groups"],
            deadline=deadline,
        )
        legacy_result = legacy_optimizer.optimize_non_overlapping()
        candidates.append(
            _pad_assignment(
                legacy_result.assignment, nonoverlap_problem.n_groups
            )
        )
        metadata.update(
            {
                "legacy_candidate_clique_milp": True,
                "legacy_solver_status": legacy_result.solver_status,
                "actual_candidate_count": legacy_result.n_candidates,
                "relocation_moves": legacy_result.iterations,
            }
        )

    if (
        family == "milp_lns"
        and candidates
        and (_remaining(deadline) is None or _remaining(deadline) > 0.0)
    ):
        refined = _refine_feasible_lns(
            nonoverlap_problem,
            overlap_penalty=overlap_penalty,
            node_costs=node_costs,
            weight_heuristic=weight_heuristic,
            heuristic_options=heuristic_options,
            solver_options=_options_with_remaining(solver_options, deadline),
            initial_assignment=candidates[-1],
        )
        candidates.append(np.asarray(refined.assignment, dtype=np.int8))
        metadata["milp_lns_refinement_status"] = refined.status
        metadata["milp_lns_exact_history"] = tuple(refined.history)
        metadata["milp_lns_observation_history"] = tuple(
            refined.observation_history
        )

    return nonoverlap_problem, candidates, metadata


def _optimize_nonoverlap(
    problem,
    *,
    method,
    weight_heuristic,
    heuristic_options,
    overlap_penalty,
    node_costs,
    solver_options,
    candidate_options,
    deadline,
    legacy_context,
    start_time,
):
    nonoverlap_problem, candidates, metadata = _nonoverlap_initial_candidates(
        problem,
        method=method,
        weight_heuristic=weight_heuristic,
        heuristic_options=heuristic_options,
        overlap_penalty=overlap_penalty,
        node_costs=node_costs,
        solver_options=solver_options,
        candidate_options=candidate_options,
        deadline=deadline,
        legacy_context=legacy_context,
    )
    exhaustive_limit = int(
        dict(solver_options or {}).get("exact_enumeration_limit", 65_536)
    )
    return _exact_rank_candidates(
        nonoverlap_problem,
        candidates,
        method=method,
        weight_heuristic=weight_heuristic,
        heuristic_options=heuristic_options,
        overlap_penalty=overlap_penalty,
        node_costs=node_costs,
        deadline=deadline,
        exhaustive_limit=exhaustive_limit,
        metadata={**metadata, "grouping_mode": "non_overlapping"},
        start_time=start_time,
    )


def optimize_grouping(
    problem=None,
    *,
    method="milp",
    grouping_mode="non_overlapping",
    coefficients=None,
    covariance=None,
    compatibility=None,
    n_groups=None,
    max_overlap=1,
    labels=None,
    overlap_penalty=0.0,
    node_costs=None,
    weight_heuristic="iterative_relative_std",
    heuristic_options=None,
    solver_options=None,
    candidate_options=None,
    legacy_context=None,
):
    """Optimize a non-overlapping partition with the selected binary method.

    Standard ICS and the separate O-clique implementation provide coefficient
    splitting after partition optimization.
    """

    if method not in METHOD_REGISTRY:
        raise ValueError(
            "unknown method {!r}; choose from {}".format(
                method, AVAILABLE_METHODS
            )
        )
    if grouping_mode != "non_overlapping":
        raise ValueError("Only grouping_mode='non_overlapping' is supported.")
    if weight_heuristic not in WEIGHT_HEURISTICS:
        raise ValueError(
            "unknown weight heuristic {!r}; choose from {}".format(
                weight_heuristic, WEIGHT_HEURISTICS
            )
        )
    instance = coerce_problem(
        problem,
        coefficients=coefficients,
        covariance=covariance,
        compatibility=compatibility,
        n_groups=n_groups,
        max_overlap=max_overlap,
        labels=labels,
    )
    solver_options = dict(solver_options or {})
    candidate_options = dict(candidate_options or {})
    heuristic_options = dict(heuristic_options or {})
    deadline = _absolute_deadline(solver_options)
    solver_options["deadline"] = deadline
    start_time = time.monotonic()

    return _optimize_nonoverlap(
        instance,
        method=method,
        weight_heuristic=weight_heuristic,
        heuristic_options=heuristic_options,
        overlap_penalty=overlap_penalty,
        node_costs=node_costs,
        solver_options=solver_options,
        candidate_options=candidate_options,
        deadline=deadline,
        legacy_context=legacy_context,
        start_time=start_time,
    )


for _method_name, _registered_method in METHOD_REGISTRY.items():
    globals()[_method_name] = _registered_method


__all__ = [
    "AVAILABLE_METHODS",
    "METHOD_REGISTRY",
    "BinaryMeasurementOptimizer",
    "CandidatePool",
    "GroupingProblem",
    "GroupingSolution",
    "OptimizationContext",
    "OptimizationResult",
    "coloring_group_bound",
    "fixed_support_ics",
    "generate_candidate_clique_pool",
    "generate_candidate_pool",
    "make_optimization_context",
    "measurement_objective",
    "optimize_grouping",
    "relative_variance_weights",
    "singleton_measurement_bound",
    *AVAILABLE_METHODS,
]
