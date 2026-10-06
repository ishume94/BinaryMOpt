"""Overlapping commuting-clique selection with a finite-profile MILP and ICS.

Each binary variable activates one candidate clique. Continuous nonnegative
profile multipliers reconstruct the Hamiltonian under the seed's group cap.
Their weighted standard deviations give a linear upper bound on the sum of
fragment standard deviations. The final coefficients are refined by array-based ICS.
Solver bounds describe this finite-profile model, not global overlap optimality.
"""

from collections import Counter
import time

import networkx as nx
import numpy as np


DEFAULT_MAX_CANDIDATES = 1800


def _checked_variances(context, omega):
    """Preserve positive variances; clip only tiny negative roundoff."""
    omega = np.asarray(omega, dtype=float)
    variances = np.sum(omega * (context.covariance @ omega), axis=0)
    if not np.all(np.isfinite(variances)):
        raise ValueError("Nonfinite group variance.")
    if np.any(variances < -1.0e-10):
        raise ValueError("ICS encountered a negative group variance.")
    return np.maximum(variances, 0.0)


def covariance_measurement_cost(context, omega):
    """Return (sum_g sqrt(omega_g.T C omega_g)) squared."""
    return float(np.sqrt(_checked_variances(context, omega)).sum() ** 2)


def _ics_shot_fractions(context, omega):
    std = np.sqrt(_checked_variances(context, omega))
    if not np.any(std > 0.0):
        return np.full(omega.shape[1], 1.0 / omega.shape[1])
    floor = max(float(std.max()) * 1.0e-12, 1.0e-15)
    std = np.maximum(std, floor)
    return std / std.sum()


def expand_partition_for_ics(context, partition):
    """Match standard ICS expansion, including stable group-major ties."""
    partition = np.asarray(partition, dtype=bool)
    if partition.ndim != 2 or partition.shape[0] != context.n_terms:
        raise ValueError("partition has the wrong shape.")
    if np.any(partition.sum(axis=0) == 0):
        raise ValueError("ICS requires nonempty groups.")
    if not np.all(partition.sum(axis=1) == 1):
        raise ValueError("ICS initialization must be non-overlapping.")
    for group in range(partition.shape[1]):
        members = np.flatnonzero(partition[:, group])
        if not np.all(context.compatible[np.ix_(members, members)]):
            raise ValueError("A partition group contains incompatible terms.")
    support = partition.copy()
    group_major = [int(i) for group in range(support.shape[1])
                   for i in np.flatnonzero(partition[:, group])]
    order = sorted(group_major, key=lambda i: -abs(context.coefficients[i]))
    for i in order:
        for group in range(support.shape[1]):
            if not support[i, group] and np.all(context.compatible[i, support[:, group]]):
                support[i, group] = True
    return support


def dense_ics(context, support, *, initial_omega=None, n_iter=5, deadline=None, linear_solver="svd"):
    """Optimize coefficients on fixed overlapping support by standard ICS.

    A free variable transfers one term's coefficient from its last supported
    group into another supported group. These variables enforce reconstruction
    exactly. With shot fractions fixed, the stationarity equations are linear;
    np.linalg.lstsq handles rank-deficient covariance blocks by default, as in
    repository ICS. The optional "qr" solver uses SciPy pivoted QR (gelsy) with
    the same dimension-scaled machine-epsilon rank cutoff. Shot fractions and
    coefficients alternate for n_iter iterations.

    The initial coefficient array remains the affine origin in every solve,
    matching standard ICS's correction formulation. The returned point is the
    best actual measurement objective encountered, including the initializer.
    No sign constraints are imposed on the optimized coefficients.
    """
    total_start = time.perf_counter()
    if linear_solver not in ("svd", "qr"):
        raise ValueError("linear_solver must be svd or qr.")
    if linear_solver == "qr":
        from scipy.linalg import lstsq as scipy_lstsq
    support = np.asarray(support, dtype=bool)
    if support.ndim != 2 or support.shape[0] != context.n_terms:
        raise ValueError("support has the wrong shape.")
    if support.shape[1] == 0 or np.any(support.sum(axis=0) == 0):
        raise ValueError("ICS requires nonempty groups.")
    coverage = support.sum(axis=1)
    if np.any(coverage == 0):
        raise ValueError("Every term must be supported.")
    if n_iter < 1:
        raise ValueError("n_iter must be at least one.")
    members_by_group = [np.flatnonzero(support[:, g]) for g in range(support.shape[1])]
    for members in members_by_group:
        if not np.all(context.compatible[np.ix_(members, members)]):
            raise ValueError("A support group contains incompatible terms.")
    if initial_omega is None:
        initial_omega = context.coefficients[:, None] * support / coverage[:, None]
    else:
        initial_omega = np.asarray(initial_omega, dtype=float).copy()
    if initial_omega.shape != support.shape:
        raise ValueError("initial_omega has the wrong shape.")
    if not np.all(np.isfinite(initial_omega)):
        raise ValueError("initial_omega is nonfinite.")
    if np.any(np.abs(initial_omega[~support]) > 1.0e-14):
        raise ValueError("initial_omega is nonzero outside support.")
    if not np.allclose(initial_omega.sum(axis=1), context.coefficients,
                       atol=1.0e-9, rtol=1.0e-9):
        raise ValueError("initial_omega does not reconstruct the Hamiltonian.")
    initial_omega[~support] = 0.0
    best = initial_omega.copy()
    current = initial_omega.copy()
    best_cost = covariance_measurement_cost(context, best)
    history = [best_cost]
    info = dict(initial_epsilon2M=best_cost, completed_iterations=0,
                n_free_coefficients=int(np.sum(coverage - 1)),
                assembly_time_s=0.0, linear_solve_time_s=0.0,
                coefficient_update_time_s=0.0, deadline_reached=False,
                minimum_linear_rank=None, linear_solver=linear_solver)

    def expired():
        return deadline is not None and time.monotonic() >= float(deadline)

    if info['n_free_coefficients'] and not expired():
        assembly_start = time.perf_counter()
        fixed = np.array([np.flatnonzero(row)[-1] for row in support], dtype=int)
        # Standard fixed-support ICS orders free variables by group then term.
        free_terms = []
        free_groups = []
        for group, members in enumerate(members_by_group):
            for term in members:
                if group != fixed[term]:
                    free_terms.append(int(term))
                    free_groups.append(group)
        free_terms = np.asarray(free_terms, dtype=int)
        free_groups = np.asarray(free_groups, dtype=int)
        fixed_groups = fixed[free_terms]
        group_blocks = []
        for group, members in enumerate(members_by_group):
            variable_indices = np.flatnonzero((free_groups == group) | (fixed_groups == group))
            variable_terms = free_terms[variable_indices]
            signs = np.where(free_groups[variable_indices] == group, 1.0, -1.0)
            covariance = context.covariance[np.ix_(variable_terms, variable_terms)]
            block = covariance * signs[:, None] * signs[None, :]
            baseline_gradient = signs * (context.covariance[np.ix_(variable_terms, members)]
                                         @ initial_omega[members, group])
            group_blocks.append((variable_indices, block, baseline_gradient))
        info['assembly_time_s'] += time.perf_counter() - assembly_start
        n_free = len(free_terms)
        for _ in range(n_iter):
            if expired():
                break
            fractions = _ics_shot_fractions(context, current)
            assembly_start = time.perf_counter()
            matrix = np.zeros((n_free, n_free), dtype=float)
            vector = np.zeros(n_free, dtype=float)
            for group, (indices, block, gradient) in enumerate(group_blocks):
                matrix[np.ix_(indices, indices)] += block / fractions[group]
                vector[indices] -= gradient / fractions[group]
            info['assembly_time_s'] += time.perf_counter() - assembly_start
            solve_start = time.perf_counter()
            if linear_solver == "svd":
                solution, _, rank, _ = np.linalg.lstsq(matrix, vector, rcond=None)
            else:
                solution, _, rank, _ = scipy_lstsq(
                    matrix, vector,
                    cond=np.finfo(matrix.dtype).eps * max(matrix.shape),
                    lapack_driver="gelsy", check_finite=False,
                )
            info['linear_solve_time_s'] += time.perf_counter() - solve_start
            info['minimum_linear_rank'] = (int(rank) if info['minimum_linear_rank'] is None
                                          else min(info['minimum_linear_rank'], int(rank)))
            update_start = time.perf_counter()
            trial = initial_omega.copy()
            np.add.at(trial, (free_terms, free_groups), solution)
            np.add.at(trial, (free_terms, fixed_groups), -solution)
            if not np.all(np.isfinite(trial)):
                raise ValueError("ICS produced nonfinite coefficients.")
            cost = covariance_measurement_cost(context, trial)
            history.append(cost)
            current = trial
            if cost < best_cost:
                best, best_cost = trial.copy(), cost
            info['coefficient_update_time_s'] += time.perf_counter() - update_start
            info['completed_iterations'] += 1
    if not np.allclose(best.sum(axis=1), context.coefficients, atol=1.0e-9, rtol=1.0e-9):
        raise ValueError("ICS coefficients do not reconstruct the Hamiltonian.")
    info.update(epsilon2M=best_cost, epsilon2M_history=history,
                deadline_reached=expired(), total_time_s=time.perf_counter() - total_start)
    return best, _ics_shot_fractions(context, best), info


def _o_partition_cliques(partition, n_terms):
    support = np.asarray(partition, dtype=bool)
    if (support.ndim != 2 or support.shape[0] != n_terms
            or not np.all(support.sum(axis=1) == 1)
            or np.any(~support.any(axis=0))):
        raise ValueError("Expected a nonempty, non-overlapping partition")
    return [tuple(np.flatnonzero(support[:, g]).tolist())
            for g in range(support.shape[1])]


def _o_grow_clique(core, order, compatibility):
    """Maximally extend a clique in one specified vertex priority order."""
    members = list(core)
    available = np.ones(compatibility.shape[0], dtype=bool)
    for vertex in members:
        available &= compatibility[vertex]
    available[members] = False
    for vertex in order:
        if available[vertex]:
            members.append(int(vertex))
            available &= compatibility[vertex]
            available[vertex] = False
    return tuple(sorted(members))


def _o_first_fit_cliques(order, compatibility):
    groups = []
    allowed = []
    for vertex in order:
        for group, available in zip(groups, allowed):
            if available[vertex]:
                group.append(int(vertex))
                available &= compatibility[vertex]
                break
        else:
            groups.append([int(vertex)])
            allowed.append(compatibility[vertex].copy())
    return [tuple(sorted(group)) for group in groups]


def build_o_clique_candidates(coefficients, compatibility, milp_partition,
                             si_partition, *, random_seed=7,
                             max_candidates=DEFAULT_MAX_CANDIDATES, random_partitions=8,
                             move_budget=1000, max_exchange_covers=48,
                             random_cover_trials=6000, kempe_trials=6000,
                             max_random_covers=160):
    """Return ``(cliques, covers, metadata)`` for bounded overlapping selection.

    ``cliques[j]`` is a sorted tuple of term indices. ``covers[name]`` lists
    candidate indices belonging to a complete cover. The four mandatory cover
    names are ``milp``, ``milp_expanded``, ``si``, and ``si_expanded``. Expansion
    uses the same descending-coefficient, group-major tie order as standard ICS.
    All original MILP and SI cliques are retained, including nonmaximal cliques.

    Additional complete covers come from complement graph coloring and first-fit
    partitions, with maximal extensions. Finally, insertions remove *all*
    incompatible members from an existing clique and regrow it in two orders;
    this permits memberships blocked by the original partition. The reduced
    clique is retained as well when capacity permits. No maximum-clique
    enumeration is used. A seed too large for ``max_candidates`` raises an error
    instead of silently dropping its mandatory covers.
    """
    started = time.perf_counter()
    coefficients = np.asarray(coefficients, dtype=float)
    compatibility = np.asarray(compatibility, dtype=bool)
    n_terms = coefficients.size
    if coefficients.ndim != 1 or n_terms == 0 or not np.all(np.isfinite(coefficients)):
        raise ValueError("Expected a finite, nonempty coefficient vector")
    if (compatibility.shape != (n_terms, n_terms)
            or not np.array_equal(compatibility, compatibility.T)):
        raise ValueError("Expected a symmetric compatibility matrix")
    if (max_candidates < 1 or random_partitions < 0 or move_budget < 0
            or max_exchange_covers < 0 or random_cover_trials < 0
            or kempe_trials < 0 or max_random_covers < 0):
        raise ValueError("Invalid candidate-pool size or generation budget")
    compatibility = compatibility.copy()
    np.fill_diagonal(compatibility, True)
    milp = _o_partition_cliques(milp_partition, n_terms)
    si = _o_partition_cliques(si_partition, n_terms)
    magnitude = np.abs(coefficients)
    degree = compatibility.sum(axis=1)
    coefficient_order = np.argsort(-magnitude, kind="stable")
    degree_order = np.lexsort((-magnitude, -degree))
    rng = np.random.default_rng(random_seed)
    cliques, origins, indices, covers = [], [], {}, {}
    accepted_moves = 0
    examined_moves = 0

    def add_clique(clique, source):
        clique = tuple(sorted(set(int(i) for i in clique)))
        if not clique:
            return None
        if clique in indices:
            index = indices[clique]
            origins[index].add(source)
            return index
        if len(cliques) >= max_candidates:
            return None
        vertices = np.asarray(clique, dtype=int)
        if (vertices[0] < 0 or vertices[-1] >= n_terms
                or not np.all(compatibility[np.ix_(vertices, vertices)])):
            raise ValueError(f"Invalid commuting clique generated by {source}")
        index = len(cliques)
        indices[clique] = index
        cliques.append(clique)
        origins.append({source})
        return index

    def add_cover(groups, name, mandatory=False):
        missing = set(groups).difference(indices)
        if len(cliques) + len(missing) > max_candidates:
            if mandatory:
                raise ValueError("max_candidates is too small to retain MILP/SI covers")
            return False
        covers[name] = list(dict.fromkeys(add_clique(group, name) for group in groups))
        return True

    def expanded(groups, order=None):
        # Group-major stable ties are required for equality with the ICS seed.
        if order is None:
            group_major = [i for group in groups for i in group]
            order = sorted(group_major, key=lambda i: -magnitude[i])
        return [_o_grow_clique(group, order, compatibility) for group in groups]

    for name, groups in (("milp", milp), ("si", si)):
        add_cover(groups, name, mandatory=True)
        add_cover(expanded(groups), name + "_expanded", mandatory=True)
    mandatory_count = len(cliques)
    # Full feasible K-group alternatives are essential for coefficient templates:
    # unrelated clique templates generally cannot reconstruct every coefficient
    # together under the K-group limit. Expanded seed covers always retain the
    # original MILP assignment and therefore obey that limit.
    alternative_orders = [("degree", degree_order),
                          ("reverse_coefficient", coefficient_order[::-1])]
    for trial in range(3):
        alternative_orders.append((f"random_{trial}", rng.permutation(n_terms)))
    exchange_bases = [("milp_expanded", expanded(milp))]
    for label, order in alternative_orders:
        name = "milp_expanded_" + label
        groups = expanded(milp, order)
        if add_cover(groups, name):
            exchange_bases.append((name, groups))

    exchange_proposals = []
    for base_name, groups in exchange_bases:
        memberships = np.zeros((n_terms, len(groups)), dtype=bool)
        for column, group in enumerate(groups):
            memberships[list(group), column] = True
        multiplicity = memberships.sum(axis=1)
        for column, group in enumerate(groups):
            vertices = np.asarray(group, dtype=int)
            outsiders = np.flatnonzero(~memberships[:, column])
            options = []
            for vertex in outsiders:
                blockers = vertices[~compatibility[vertex, vertices]]
                if blockers.size and np.all(multiplicity[blockers] >= 2):
                    penalty = float(magnitude[blockers].sum())
                    score = float(magnitude[vertex]) / (penalty + 1.e-15)
                    options.append((-score, int(vertex)))
            options.sort()
            for rank, (_, vertex) in enumerate(options[:8]):
                exchange_proposals.append((rank, float(rng.random()), base_name,
                                           groups, column, vertex))
    exchange_proposals.sort(key=lambda item: item[:2])
    seen_exchange_covers = {tuple(sorted(set(cover))) for cover in covers.values()}
    n_exchange_covers = 0
    n_swap_covers = 0
    # A paired move also handles singly covered blockers: transfer the entering
    # term from its source to the target and put every target blocker in the
    # vacated source when commutation permits. This is a complete K-partition
    # before it is expanded, so no coefficient can become uncovered.
    swap_proposals = []
    for source_name, partition_groups in (("milp", milp), ("si", si)):
        if len(partition_groups) > len(milp):
            continue
        owner = np.empty(n_terms, dtype=int)
        for column, group in enumerate(partition_groups):
            owner[list(group)] = column
        for vertex in coefficient_order:
            source_column = int(owner[vertex])
            source_remaining = [i for i in partition_groups[source_column] if i != vertex]
            for target_column, group in enumerate(partition_groups):
                if target_column == source_column:
                    continue
                blockers = [i for i in group if not compatibility[vertex, i]]
                if not blockers:
                    continue
                if source_remaining and not np.all(compatibility[np.ix_(blockers, source_remaining)]):
                    continue
                penalty = sum(magnitude[i] for i in blockers)
                score = float(magnitude[vertex]) / (penalty + 1.e-15)
                swap_proposals.append((-score, float(rng.random()), source_name,
                                       partition_groups, source_column, target_column,
                                       int(vertex), blockers, source_remaining))
    swap_proposals.sort(key=lambda item: item[:2])
    for _, _, source_name, groups, source_column, target_column, vertex, blockers, remaining in swap_proposals:
        if n_swap_covers >= max_exchange_covers // 2:
            break
        partition_trial = list(groups)
        partition_trial[source_column] = tuple(sorted(remaining + blockers))
        partition_trial[target_column] = tuple(sorted([i for i in groups[target_column]
                                                       if i not in blockers] + [vertex]))
        for label, order in (("coefficient", coefficient_order), ("degree", degree_order)):
            trial_groups = expanded(partition_trial, order)
            # Compare clique tuples directly: new candidates need not yet exist.
            signature_groups = tuple(sorted(set(trial_groups)))
            if any(signature_groups == tuple(sorted(cliques[index] for index in cover))
                   for cover in covers.values()):
                continue
            name = f"exchange_swap_{n_swap_covers}_{source_name}_{label}"
            if add_cover(trial_groups, name):
                n_swap_covers += 1
                n_exchange_covers += 1
                seen_exchange_covers.add(tuple(sorted(set(covers[name]))))
                add_clique(partition_trial[source_column], "move_swap_reduced:" + source_name)
                add_clique(partition_trial[target_column], "move_swap_reduced:" + source_name)
            if n_swap_covers >= max_exchange_covers // 2:
                break
    for _, _, base_name, groups, column, vertex in exchange_proposals:
        if n_exchange_covers >= max_exchange_covers:
            break
        core = tuple(sorted([i for i in groups[column] if compatibility[vertex, i]]
                            + [vertex]))
        for label, order in (("coefficient", coefficient_order), ("degree", degree_order)):
            replacement = _o_grow_clique(core, order, compatibility)
            index = add_clique(replacement, "exchange:" + base_name)
            if index is None:
                break
            trial_groups = list(groups)
            trial_groups[column] = replacement
            signature = tuple(sorted(set(indices[group] for group in trial_groups)))
            if signature in seen_exchange_covers:
                continue
            name = f"exchange_{n_exchange_covers}_{label}"
            if add_cover(trial_groups, name):
                seen_exchange_covers.add(signature)
                n_exchange_covers += 1
            if n_exchange_covers >= max_exchange_covers:
                break
    cover_limit = max(mandatory_count, int(0.50 * max_candidates))
    base_groups = [("milp", group) for group in milp]
    base_groups += [("si", group) for group in si]

    # Each graph coloring is a non-overlapping partition into commuting groups.
    conflict = ~compatibility
    np.fill_diagonal(conflict, False)
    graph = nx.from_numpy_array(conflict)
    coloring_strategies = ("largest_first", "smallest_last",
                           "saturation_largest_first", "connected_sequential_bfs")
    for strategy in coloring_strategies:
        colors = nx.coloring.greedy_color(graph, strategy=strategy)
        groups = [tuple(sorted(i for i, color in colors.items() if color == label))
                  for label in sorted(set(colors.values()))]
        name = "color_" + strategy
        if len(cliques) < cover_limit and add_cover(groups, name):
            base_groups += [(name, group) for group in groups]
            add_cover(expanded(groups), name + "_expanded")

    orders = [("coefficient", coefficient_order), ("degree", degree_order),
              ("weighted_degree", np.argsort(-(magnitude + 1.e-15) * degree,
                                               kind="stable"))]
    # Random multiplicative priorities retain coefficient bias but can displace
    # blockers; a random permutation also explores beyond coefficient ordering.
    for trial in range(random_partitions):
        if trial % 3 == 2:
            order = rng.permutation(n_terms)
        else:
            noise = rng.uniform(0.1, 2.0, n_terms)
            power = 0.5 if trial % 3 == 0 else 1.0
            order = np.argsort(-(magnitude + 1.e-15) ** power * noise,
                               kind="stable")
        orders.append((f"random_{trial}", order))
    for label, order in orders:
        if len(cliques) >= cover_limit:
            break
        name = "firstfit_" + label
        groups = _o_first_fit_cliques(order, compatibility)
        if add_cover(groups, name):
            base_groups += [(name, group) for group in groups]
            add_cover(expanded(groups, order), name + "_expanded")

    # Complete randomized covers receive priority over arbitrary clique fillers.
    # Use a separate stream: increasing other candidate budgets must not change
    # these sampled covers. These are cheap graph operations; coefficient scoring
    # happens later and only for retained full covers.
    sample_start = time.perf_counter()
    cover_rng = np.random.default_rng(random_seed + 10)
    random_pool = []
    kempe_pool = []
    known_supports = {tuple(sorted(cliques[index] for index in cover))
                      for cover in covers.values()}
    sampled_supports = set(known_supports)
    n_feasible_random = 0
    n_kempe_changes = 0

    def collect_support(name, partition_groups, order, destination):
        if len(partition_groups) > len(milp):
            return
        trial_groups = list(dict.fromkeys(
            _o_grow_clique(group, order, compatibility) for group in partition_groups))
        signature = tuple(sorted(trial_groups))
        if signature in sampled_supports:
            return
        sampled_supports.add(signature)
        destination.append((name, trial_groups))

    # Probe alternative expansions before sampling new complete partitions.
    for trial in range(30):
        order = cover_rng.permutation(n_terms)
        collect_support(f"random_seed_expansion_{trial}", milp, order, random_pool)
    for trial in range(random_cover_trials):
        power = (0., .25, .5, 1., 2.)[trial % 5]
        noise = cover_rng.uniform(.01, 3., n_terms)
        order = np.argsort(-(magnitude + 1.e-15) ** power * noise, kind="stable")
        partition_groups = _o_first_fit_cliques(order, compatibility)
        if len(partition_groups) <= len(milp):
            n_feasible_random += 1
            collect_support(f"random_complete_firstfit_{trial}", partition_groups,
                            order, random_pool)

    # Kempe-chain exchanges swap one connected component between two color
    # classes of the conflict graph. Unlike a one-step blocker exchange, a chain
    # may relocate multiple mutually dependent blockers while preserving K.
    kempe_groups = [set(group) for group in milp]
    for trial in range(kempe_trials):
        if len(kempe_groups) < 2:
            break
        left, right = cover_rng.choice(len(kempe_groups), 2, replace=False)
        vertices = list(kempe_groups[left] | kempe_groups[right])
        vertex = int(cover_rng.choice(vertices))
        component, frontier = {vertex}, [vertex]
        while frontier:
            current = frontier.pop()
            opposite = (kempe_groups[right] if current in kempe_groups[left]
                        else kempe_groups[left])
            additions = {i for i in opposite if not compatibility[current, i]} - component
            component.update(additions)
            frontier.extend(additions)
        left_component = kempe_groups[left] & component
        right_component = kempe_groups[right] & component
        if (left_component == kempe_groups[left]
                and right_component == kempe_groups[right]):
            continue
        kempe_groups[left] = (kempe_groups[left] - left_component) | right_component
        kempe_groups[right] = (kempe_groups[right] - right_component) | left_component
        kempe_groups = [group for group in kempe_groups if group]
        n_kempe_changes += 1
        collect_support(f"random_complete_kempe_{trial}",
                        [tuple(sorted(group)) for group in kempe_groups],
                        coefficient_order, kempe_pool)
        if trial % 12 == 0:
            kempe_groups = [set(group) for group in milp]

    # Keep room for both families when feasible first-fit covers are plentiful.
    # Independent uniform subsampling avoids favoring only early random draws.
    # Smaller families are retained in full and lend unused capacity to the other.
    random_quota = min(len(random_pool), int(.70 * max_random_covers))
    kempe_quota = min(len(kempe_pool), max_random_covers - random_quota)
    random_quota = min(len(random_pool), max_random_covers - kempe_quota)
    retain_rng = np.random.default_rng(random_seed + 20)
    if len(random_pool) > random_quota:
        chosen = sorted(retain_rng.choice(len(random_pool), random_quota, replace=False).tolist())
        random_selected = [random_pool[index] for index in chosen]
    else:
        random_selected = random_pool
    if len(kempe_pool) > kempe_quota:
        chosen = sorted(retain_rng.choice(len(kempe_pool), kempe_quota, replace=False).tolist())
        kempe_selected = [kempe_pool[index] for index in chosen]
    else:
        kempe_selected = kempe_pool
    # Interleave to share any remaining global clique capacity between families.
    retained_random_covers = 0
    skipped_random_covers = 0
    for position in range(max(len(random_selected), len(kempe_selected))):
        for family in (random_selected, kempe_selected):
            if position < len(family):
                name, trial_groups = family[position]
                if add_cover(trial_groups, name):
                    retained_random_covers += 1
                else:
                    skipped_random_covers += 1
    random_cover_info = {
        "random_cover_trials": int(random_cover_trials), "kempe_trials": int(kempe_trials),
        "max_random_covers": int(max_random_covers),
        "feasible_firstfit_trials": n_feasible_random,
        "unique_random_covers": len(random_pool), "unique_kempe_covers": len(kempe_pool),
        "kempe_changes": n_kempe_changes, "retained_covers": retained_random_covers,
        "skipped_for_clique_capacity": skipped_random_covers,
        "runtime_s": time.perf_counter() - sample_start,
    }

    # Prioritize cheap blocker removals, but rotate sources and terms to prevent
    # the first partition from consuming the entire move budget.
    proposals = []
    seen_bases = set()
    for source, group in base_groups:
        if group in seen_bases:
            continue
        seen_bases.add(group)
        vertices = np.asarray(group, dtype=int)
        inside = np.zeros(n_terms, dtype=bool)
        inside[vertices] = True
        blockers = ~compatibility[:, vertices]
        blocker_count = blockers.sum(axis=1)
        blocker_weight = blockers @ magnitude[vertices]
        outside = np.flatnonzero(~inside)
        if not outside.size:
            continue
        # Include zero-blocker additions and blocker-removing moves alike.
        priority = (magnitude[outside] + np.median(magnitude)) / (
            blocker_weight[outside] + np.median(magnitude) + 1.e-15)
        ranked = outside[np.lexsort((-magnitude[outside], blocker_count[outside],
                                    -priority))]
        # Equal opportunity across base groups at each rank; jitter only breaks
        # ties, deterministically for a given seed.
        proposals.extend((rank, float(rng.random()), source, group, int(vertex))
                         for rank, vertex in enumerate(ranked[:min(n_terms, 24)]))
    proposals.sort(key=lambda item: item[:2])
    for _, _, source, group, vertex in proposals:
        if examined_moves >= move_budget or len(cliques) >= max_candidates:
            break
        examined_moves += 1
        core = tuple(sorted([i for i in group if compatibility[vertex, i]] + [vertex]))
        before = len(cliques)
        add_clique(core, "move_reduced:" + source)
        add_clique(_o_grow_clique(core, coefficient_order, compatibility),
                   "move_coefficient:" + source)
        add_clique(_o_grow_clique(core, degree_order, compatibility),
                   "move_degree:" + source)
        accepted_moves += len(cliques) > before

    if not np.all(np.bincount([i for clique in cliques for i in clique],
                             minlength=n_terms) > 0):
        raise AssertionError("Candidate pool lost term coverage")
    for name, cover in covers.items():
        if set(i for index in cover for i in cliques[index]) != set(range(n_terms)):
            raise AssertionError(f"Candidate cover {name} is incomplete")
    source_counts = Counter(source for source_set in origins for source in source_set)
    metadata = {
        "n_candidates": len(cliques), "max_candidates": int(max_candidates),
        "candidate_limit_reached": len(cliques) >= max_candidates,
        "moves_deferred_for_candidate_limit": (
            max(0, min(move_budget, len(proposals)) - examined_moves)
            if len(cliques) >= max_candidates else 0),
        "mandatory_candidates": mandatory_count, "n_covers": len(covers),
        "n_exchange_covers": n_exchange_covers, "n_swap_covers": n_swap_covers,
        "max_exchange_covers": int(max_exchange_covers),
        "seed_group_cap": len(milp),
        "covers_within_seed_cap": [name for name, cover in covers.items()
                                   if len(cover) <= len(milp)],
        "random_seed": int(random_seed), "random_partitions": int(random_partitions),
        "examined_moves": examined_moves, "moves_adding_candidates": accepted_moves,
        "move_budget": int(move_budget), "source_counts": dict(sorted(source_counts.items())),
        "candidate_sources": [sorted(source_set) for source_set in origins],
        "random_complete_covers": random_cover_info,
        "runtime_s": time.perf_counter() - started,
    }
    return cliques, covers, metadata


def support_from_cliques(n_terms, cliques, selected):
    support = np.zeros((n_terms, len(selected)), dtype=bool)
    for g, q in enumerate(selected):
        support[list(cliques[q]), g] = True
    return support


def validate_overlap(context, support, omega, group_cap):
    support = np.asarray(support, dtype=bool)
    omega = np.asarray(omega, dtype=float)
    if support.shape != omega.shape or support.shape[0] != context.n_terms:
        raise ValueError("Inconsistent coefficient/support shapes")
    if support.shape[1] > group_cap or not np.all(support.any(axis=1)):
        raise ValueError("Invalid group cap or incomplete coverage")
    if not np.all(support.any(axis=0)) or not np.all(np.isfinite(omega)):
        raise ValueError("Empty group or nonfinite coefficient")
    error = float(np.max(np.abs(omega.sum(axis=1) - context.coefficients)))
    if error > 1.e-9 or np.any(np.abs(omega[~support]) > 1.e-12):
        raise ValueError(f"Hamiltonian reconstruction failed: {error}")
    for group in support.T:
        indices = np.flatnonzero(group)
        if not np.all(context.compatible[np.ix_(indices, indices)]):
            raise ValueError("Noncommuting candidate group")
    return {"max_reconstruction_error": error, "compatible": True,
            "within_group_cap": True}


def transport_coefficients(context, source_support, source_omega, support):
    """Retain signed current weights, routing them to the most similar clique."""
    intersection = source_support.T.astype(float) @ support.astype(float)
    union = source_support.sum(axis=0)[:, None] + support.sum(axis=0)[None, :] - intersection
    similarity = intersection / np.maximum(union, 1)
    omega = np.zeros(support.shape, dtype=float)
    for old in range(source_support.shape[1]):
        terms = np.flatnonzero(source_omega[:, old])
        destinations = np.argmax(np.where(support[terms], similarity[old], -np.inf), axis=1)
        omega[terms, destinations] += source_omega[terms, old]
    return omega


def choose_guidance(context, support, initial_omega, requested="auto"):
    """Compare one coefficient solve with each allowed three-sweep guide."""
    started = time.perf_counter()
    from bin_mopt.measurement import GroupingProblem
    from bin_mopt.weighting import assign_overlapping_weights
    problem = GroupingProblem(context.coefficients, context.covariance, context.compatible,
                              n_groups=support.shape[1], max_overlap=support.shape[1])
    timings = {}
    choices = ("ics", "relative_std", "relative_variance", "score_softmax")
    for name in choices if requested == "auto" else (requested,):
        samples = []
        for _ in range(3 if requested == "auto" else 1):
            start = time.perf_counter()
            if name == "ics":
                omega, _, _ = dense_ics(context, support, initial_omega=initial_omega,
                                        n_iter=1, linear_solver="qr")
            else:
                omega = assign_overlapping_weights(problem, support, heuristic=name,
                                                    initial_omega=initial_omega, sweeps=3)
            samples.append(time.perf_counter() - start)
        timings[name] = {"median_s": float(np.median(samples)), "samples_s": samples,
                         "epsilon2M": covariance_measurement_cost(context, omega)}
    chosen = min(timings, key=lambda name: timings[name]["median_s"])
    return chosen, {"selected": chosen, "timings": timings,
                    "runtime_s": time.perf_counter() - started,
                    "comparison": "one QR ICS step versus each individual three-sweep guide"}


def guide_coefficients(context, support, omega, method):
    if method == "ics":
        return dense_ics(context, support, initial_omega=omega, n_iter=1,
                         linear_solver="qr")[0]
    from bin_mopt.measurement import GroupingProblem
    from bin_mopt.weighting import assign_overlapping_weights
    problem = GroupingProblem(context.coefficients, context.covariance, context.compatible,
                              n_groups=support.shape[1], max_overlap=support.shape[1])
    proposed = assign_overlapping_weights(problem, support, heuristic=method,
                                          initial_omega=omega, sweeps=3)
    return min((omega, proposed), key=lambda value: covariance_measurement_cost(context, value))


def build_coefficient_profiles(context, cliques, covers, baseline_support, baseline_omega,
                               *, guidance="auto", profile_scale=2.0):
    """Make a finite linear model from complete feasible coefficient decompositions.

    Each profile a has cost ||a||_C. A selected clique can take a convex mixture
    of its scaled profiles and zero. The MILP's sum of profile norms is an upper
    bound on the physical sum of group standard deviations. Full decompositions
    supply feasible warm starts, including the unchanged MILP-ICS incumbent.
    """
    start = time.perf_counter()
    if not np.isfinite(profile_scale) or profile_scale < 1.:
        raise ValueError("profile_scale must be finite and at least one")
    cap = baseline_support.shape[1]
    method, guidance_info = choose_guidance(context, baseline_support, baseline_omega, guidance)
    print(f"  Guidance: {method}, selection_runtime_s={guidance_info['runtime_s']:.6f}",
          flush=True)
    profiles, by_clique, signatures, decompositions = [], [[] for _ in cliques], {}, []

    def add_profile(q, values):
        values = np.asarray(values, dtype=float)
        key = (q, tuple(np.round(values, 13)))
        if key in signatures:
            return signatures[key]
        members = np.asarray(cliques[q], dtype=int)
        variance = float(values @ context.covariance[np.ix_(members, members)] @ values)
        if variance < -1.e-10:
            raise ValueError("Negative profile variance")
        index = len(profiles)
        profiles.append({"clique": q, "values": values * profile_scale,
                         "std": float(np.sqrt(max(variance, 0))) * profile_scale})
        by_clique[q].append(index)
        signatures[key] = index
        return index

    def add_decomposition(name, selected, omega):
        # Identical supports represent one physical measurement setting.
        unique = list(dict.fromkeys(selected))
        if len(unique) != len(selected):
            merged = np.zeros((context.n_terms, len(unique)))
            positions = {q: g for g, q in enumerate(unique)}
            for g, q in enumerate(selected):
                merged[:, positions[q]] += omega[:, g]
            selected, omega = unique, merged
        pids = [add_profile(q, omega[list(cliques[q]), g]) for g, q in enumerate(selected)]
        decompositions.append({"name": name, "selected": selected, "profiles": pids,
                               "epsilon2M": covariance_measurement_cost(context, omega)})

    # Match expanded MILP cliques to the actual baseline columns, including order.
    clique_index = {clique: q for q, clique in enumerate(cliques)}
    baseline_selected = [clique_index[tuple(np.flatnonzero(group))] for group in baseline_support.T]
    add_decomposition("MILP-ICS", baseline_selected, baseline_omega)
    seen = {tuple(sorted(baseline_selected))}
    for name, selected in covers.items():
        signature = tuple(sorted(selected))
        if len(selected) > cap or signature in seen:
            continue
        seen.add(signature)
        support = support_from_cliques(context.n_terms, cliques, selected)
        initial = transport_coefficients(context, baseline_support, baseline_omega, support)
        omega = guide_coefficients(context, support, initial, method)
        validate_overlap(context, support, omega, cap)
        add_decomposition(name, selected, omega)
    # Raw profiles also let the binary solver recombine cliques outside known covers.
    for q, members in enumerate(cliques):
        add_profile(q, context.coefficients[list(members)])
    info = {"n_profiles": len(profiles), "n_guided_covers": len(decompositions) - 1,
            "profile_scale": profile_scale, "guidance": guidance_info,
            "runtime_s": time.perf_counter() - start,
            "cover_costs": [{"name": d["name"], "epsilon2M": d["epsilon2M"],
                             "groups": len(d["selected"])} for d in decompositions]}
    print(f"  Coefficient profiles: count={len(profiles)}, "
          f"guided_covers={info['n_guided_covers']}, runtime_s={info['runtime_s']:.6f}",
          flush=True)
    return profiles, by_clique, decompositions, info


def _solve_profile_model(model, threads):
    """Use concurrent SCIP workers without multiplying numerical thread pools."""
    import warnings
    from threadpoolctl import threadpool_limits

    method = "solveConcurrent" if threads > 1 else "optimize"
    if threads > 1 and not hasattr(model, method):
        raise RuntimeError(
            "This PySCIPOpt version does not support concurrent solving; "
            "install a concurrent-capable SCIP build or use --threads 1."
        )
    print(f"  SCIP solve: method={method}, thread_limit={threads}", flush=True)
    started = time.perf_counter()
    if threads > 1:
        # PySCIPOpt otherwise warns and silently substitutes optimize() when
        # SCIP lacks its task-processing interface. Do not misreport that as
        # a parallel solve. SCIP's own C threads remain independent of BLAS.
        try:
            with warnings.catch_warnings(), threadpool_limits(limits=1):
                warnings.filterwarnings(
                    "error", message="SCIP was compiled without task processing interface.*"
                )
                model.solveConcurrent()
        except UserWarning as exc:
            raise RuntimeError(
                "This SCIP build cannot run concurrent workers; "
                "use a concurrent-capable build or --threads 1."
            ) from exc
    else:
        model.optimize()
    return {
        "solver_backend": "SCIP", "solver_method": method,
        "solver_threads_requested": int(threads),
        "parallel_thread_limit": int(threads),
        "solver_s": time.perf_counter() - started,
        "scip_solving_time_s": float(model.getSolvingTime()),
    }


def solve_profile_milp(context, cliques, profiles, by_clique, decompositions, group_cap,
                       *, time_limit_s=30.0, threads=1, random_seed=7,
                       profile_scale=2.0, excluded=()):
    """Select overlapping cliques and reconstruct every coefficient linearly."""
    from pyscipopt import Model, quicksum
    started = time.perf_counter()
    model = Model("overlapping_clique_profiles")
    model.hideOutput()
    # SCIP concurrent workers share this elapsed-time budget. CPU time summed
    # over workers must not become the stopping criterion.
    model.setIntParam("timing/clocktype", 2)
    model.setRealParam("limits/time", float(time_limit_s))
    model.setIntParam("parallel/maxnthreads", int(threads))
    model.setIntParam("parallel/minnthreads", int(threads))
    model.setIntParam("randomization/randomseedshift", int(random_seed))
    model.setRealParam("numerics/feastol", 1.e-9)
    objective_scale = max(np.sqrt(decompositions[0]["epsilon2M"]), 1.e-12)
    y = [model.addVar(vtype="B", name=f"clique_{q}") for q in range(len(cliques))]
    weights = [model.addVar(lb=0., ub=1., obj=p["std"] / objective_scale,
                            name=f"profile_{j}") for j, p in enumerate(profiles)]
    for q, indices in enumerate(by_clique):
        model.addCons(quicksum(weights[j] for j in indices) <= y[q])
    model.addCons(quicksum(y) <= group_cap)
    rows = [[] for _ in range(context.n_terms)]
    for j, profile in enumerate(profiles):
        for term, value in zip(cliques[profile["clique"]], profile["values"]):
            if value != 0.:
                rows[term].append((j, float(value)))
    for i, entries in enumerate(rows):
        if not entries:
            if context.coefficients[i] != 0.:
                raise ValueError("Coefficient profiles leave a term uncovered")
            continue
        scale = max(abs(context.coefficients[i]), max(abs(value) for _, value in entries), 1.e-12)
        model.addCons(quicksum((value / scale) * weights[j] for j, value in entries)
                      == float(context.coefficients[i]) / scale)
    excluded = set(tuple(sorted(s)) for s in excluded)
    for selected in excluded:
        if len(selected) == group_cap:
            model.addCons(quicksum(y[q] for q in selected) <= group_cap - 1)
        else:
            inside = set(selected)
            model.addCons(quicksum(1 - y[q] for q in selected)
                          + quicksum(y[q] for q in range(len(y)) if q not in inside) >= 1)
    warm_starts = 0
    for decomposition in sorted(decompositions, key=lambda d: d["epsilon2M"])[:20]:
        if tuple(sorted(decomposition["selected"])) in excluded:
            continue
        sol = model.createSol()
        for q in decomposition["selected"]:
            model.setSolVal(sol, y[q], 1.)
        for j in decomposition["profiles"]:
            model.setSolVal(sol, weights[j], 1. / profile_scale)
        warm_starts += int(model.addSol(sol, free=True))
    construction_s = time.perf_counter() - started
    execution_info = _solve_profile_model(model, int(threads))
    info = {"status": str(model.getStatus()), "gap": float(model.getGap()),
            "dual_bound_sum_std": float(model.getDualbound()) * objective_scale,
            "nodes": int(model.getNNodes()), "model_build_s": construction_s,
            "accepted_warm_starts": warm_starts,
            "runtime_s": time.perf_counter() - started, "objective_is_upper_bound": True,
            **execution_info}
    sol = model.getBestSol()
    if sol is None:
        return None, info
    selected = [q for q in range(len(cliques)) if model.getSolVal(sol, y[q]) > .5]
    positions = {q: g for g, q in enumerate(selected)}
    omega = np.zeros((context.n_terms, len(selected)))
    for j, profile in enumerate(profiles):
        value = model.getSolVal(sol, weights[j])
        if profile["clique"] in positions and value != 0.:
            omega[list(cliques[profile["clique"]]), positions[profile["clique"]]] += value * profile["values"]
    support = support_from_cliques(context.n_terms, cliques, selected)
    residual = context.coefficients - omega.sum(axis=1)
    error = float(np.max(np.abs(residual)))
    if error > 1.e-7 or not np.all(support.any(axis=1)):
        info.update(rejected_reconstruction_error=error)
        return None, info
    # Only repair numerical equality-constraint residuals; never missing coverage.
    omega[np.arange(context.n_terms), np.argmax(support, axis=1)] += residual
    validate_overlap(context, support, omega, group_cap)
    info.update(surrogate_epsilon2M=(float(model.getSolObjVal(sol)) * objective_scale) ** 2,
                decoded_epsilon2M=covariance_measurement_cost(context, omega),
                repaired_reconstruction_error=error, selected_cliques=selected)
    if np.sqrt(info["decoded_epsilon2M"]) > np.sqrt(info["surrogate_epsilon2M"]) + 1.e-7:
        raise ValueError("Decoded physical cost exceeds the profile upper bound")
    return (support, omega, selected), info


def optimize_overlapping_cliques(context, seed, si, *, time_limit_s=30., rounds=1,
                                 threads=1, random_seed=7, max_candidates=DEFAULT_MAX_CANDIDATES,
                                 random_partitions=8, move_budget=1000,
                                 max_exchange_covers=48, ics_iterations=5,
                                 guidance="auto", shortlist=6,
                                 random_cover_trials=6000, kempe_trials=6000,
                                 max_random_covers=160):
    """Optimize a finite overlapping clique pool, preserving MILP-ICS fallback."""
    if (not np.isfinite(time_limit_s) or time_limit_s <= 0 or rounds < 1
            or threads < 1 or ics_iterations < 1 or shortlist < 0):
        raise ValueError("Invalid solve, thread, or coefficient-refinement budget")
    seed = np.asarray(seed, dtype=bool)
    si = np.asarray(si, dtype=bool)
    started = time.perf_counter()
    baseline_support = expand_partition_for_ics(context, seed)
    baseline_omega, _, baseline_info = dense_ics(
        context, baseline_support, initial_omega=context.coefficients[:, None] * seed,
        n_iter=ics_iterations, linear_solver="qr")
    best_support, best_omega = baseline_support, baseline_omega
    best_cost, winner = baseline_info["epsilon2M"], "MILP-ICS"
    baseline_s = time.perf_counter() - started
    print(f"  MILP-ICS: groups={baseline_support.shape[1]}, "
          f"epsilon2M={best_cost:.12g}, runtime_s={baseline_s:.6f}", flush=True)
    cliques, covers, candidate_info = build_o_clique_candidates(
        context.coefficients, context.compatible, seed, si, random_seed=random_seed,
        max_candidates=max_candidates, random_partitions=random_partitions,
        move_budget=move_budget, max_exchange_covers=max_exchange_covers,
        random_cover_trials=random_cover_trials, kempe_trials=kempe_trials,
        max_random_covers=max_random_covers)
    print(f"  Candidate generation: cliques={len(cliques)}/{max_candidates}, covers={len(covers)}, "
          f"moves={candidate_info['examined_moves']}/{move_budget}, "
          f"random_covers_skipped_for_capacity="
          f"{candidate_info['random_complete_covers']['skipped_for_clique_capacity']}, "
          f"runtime_s={candidate_info['runtime_s']:.6f}", flush=True)
    profiles, by_clique, decompositions, profile_info = build_coefficient_profiles(
        context, cliques, covers, baseline_support, baseline_omega, guidance=guidance)
    finalists = []
    # Refine promising feasible decompositions before solving and expose those
    # improved coefficients to the MILP, so it can recombine their fragments.
    refinement_start = time.perf_counter()
    for decomposition in sorted(decompositions[1:], key=lambda d: d["epsilon2M"])[:shortlist]:
        selected = decomposition["selected"]
        support = support_from_cliques(context.n_terms, cliques, selected)
        omega = np.zeros(support.shape)
        for g, j in enumerate(decomposition["profiles"]):
            omega[list(cliques[selected[g]]), g] = profiles[j]["values"] / 2.
        omega, _, info = dense_ics(context, support, initial_omega=omega,
                                   n_iter=ics_iterations, linear_solver="qr")
        std = np.sqrt(_checked_variances(context, omega))
        for g, j in enumerate(decomposition["profiles"]):
            values = omega[list(cliques[selected[g]]), g] * 2.
            q = selected[g]
            # Append improved profiles; keep older profiles available to the model.
            new_id = len(profiles)
            profiles.append({"clique": q, "values": values,
                             "std": float(std[g]) * 2.})
            by_clique[q].append(new_id)
            decomposition["profiles"][g] = new_id
        decomposition["epsilon2M"] = info["epsilon2M"]
        finalists.append({"source": decomposition["name"], **info})
        if info["epsilon2M"] < best_cost:
            best_support, best_omega, best_cost = support, omega, info["epsilon2M"]
            winner = decomposition["name"]
    shortlist_s = time.perf_counter() - refinement_start
    print(f"  Shortlist ICS: refined_covers={len(finalists)}, "
          f"best_groups={best_support.shape[1]}, best_epsilon2M={best_cost:.12g}, "
          f"runtime_s={shortlist_s:.6f}", flush=True)
    excluded, milp_rounds = [], []
    selection_start = time.perf_counter()
    for iteration in range(rounds):
        milp_start = time.perf_counter()
        result, info = solve_profile_milp(
            context, cliques, profiles, by_clique, decompositions, seed.shape[1],
            time_limit_s=time_limit_s / rounds, threads=threads,
            random_seed=random_seed + iteration, excluded=excluded)
        info.update(milp_time_s=time.perf_counter() - milp_start, ics_time_s=0.0)
        milp_rounds.append(info)
        timing_summary = (f"model_build_s={info['model_build_s']:.6f}, "
                          f"solver_s={info['solver_s']:.6f}, "
                          f"runtime_s={info['milp_time_s']:.6f}")
        if result is None:
            print(f"  Binary round {iteration + 1}: status={info['status']}, "
                  f"no usable incumbent, {timing_summary}", flush=True)
            continue
        support, omega, selected = result
        excluded.append(selected)
        print(f"  Binary round {iteration + 1}: status={info['status']}, "
              f"groups={support.shape[1]}, epsilon2M={info['decoded_epsilon2M']:.12g}, "
              f"{timing_summary}", flush=True)
        ics_start = time.perf_counter()
        omega, _, ics_info = dense_ics(context, support, initial_omega=omega,
                                       n_iter=ics_iterations, linear_solver="qr")
        info.update(ics_time_s=time.perf_counter() - ics_start,
                    post_ics_epsilon2M=ics_info["epsilon2M"],
                    post_ics_n_groups=support.shape[1])
        finalists.append({"source": f"binary_round_{iteration + 1}", **ics_info})
        print(f"  Binary round {iteration + 1} ICS: groups={support.shape[1]}, "
              f"epsilon2M={ics_info['epsilon2M']:.12g}, "
              f"runtime_s={info['ics_time_s']:.6f}", flush=True)
        if ics_info["epsilon2M"] < best_cost:
            best_support, best_omega, best_cost = support, omega, ics_info["epsilon2M"]
            winner = f"binary_round_{iteration + 1}"
    selection_s = time.perf_counter() - selection_start
    validation_start = time.perf_counter()
    validation = validate_overlap(context, best_support, best_omega, seed.shape[1])
    if best_cost > baseline_info["epsilon2M"] + 1.e-10:
        raise AssertionError("Final solution lost the MILP-ICS incumbent")
    validation_s = time.perf_counter() - validation_start
    info = {"epsilon2M": best_cost, "winner": winner, "groups": best_support.shape[1],
            "group_cap": seed.shape[1], "baseline": baseline_info,
            "baseline_s": baseline_s, "candidate_generation": candidate_info,
            "profiles": profile_info, "n_final_profiles": len(profiles),
            "shortlist_refinement_s": shortlist_s, "finalists": finalists,
            "milp_rounds": milp_rounds, "selection_and_final_ics_s": selection_s,
            "final_validation_s": validation_s,
            "validation": validation, "runtime_s": time.perf_counter() - started}
    return best_support, best_omega, info
