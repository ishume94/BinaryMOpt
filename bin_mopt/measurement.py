"""Measurement-aware data model and objective utilities for BinMOpt.

The routines in this module are backend-independent.  A grouping is represented
by a binary matrix ``x[i, alpha]``.  Each row is therefore a binary vector that
can contain more than one active entry, which directly represents overlapping
measurement groups.
"""

from dataclasses import dataclass, field
from math import erf, exp, pi, sqrt
from typing import Any, Iterable, Mapping, Optional, Sequence

import networkx as nx
import numpy as np

from .weighting import WEIGHT_HEURISTICS, assign_overlapping_weights

ArrayLike = Any


@dataclass
class GroupingProblem:
    """Validated input for overlapping Pauli grouping.

    Parameters
    ----------
    coefficients
        Hamiltonian coefficients ``c_i``.
    covariance
        Real symmetrized covariance matrix ``C_ij``.  The implementation uses
        ``Re Cov(P_i, P_j)`` because fragment variances are real.
    compatibility
        Boolean matrix; ``compatibility[i, j]`` is true when terms ``i`` and
        ``j`` may be measured in the same group.
    n_groups
        Maximum number of labeled groups available to assignment-based methods.
    max_overlap
        Scalar or length-``n_terms`` vector specifying the maximum number of
        groups to which each term may be assigned.
    """

    coefficients: ArrayLike
    covariance: ArrayLike
    compatibility: ArrayLike
    n_groups: int
    max_overlap: ArrayLike = 1
    labels: Optional[Sequence[Any]] = None

    def __post_init__(self) -> None:
        c = np.asarray(self.coefficients, dtype=float).reshape(-1)
        C = np.asarray(self.covariance, dtype=float)
        A = np.asarray(self.compatibility, dtype=bool)
        n = c.size
        if n < 1:
            raise ValueError("at least one Pauli term is required")
        if C.shape != (n, n):
            raise ValueError(f"covariance must have shape {(n, n)}, got {C.shape}")
        if A.shape != (n, n):
            raise ValueError(f"compatibility must have shape {(n, n)}, got {A.shape}")
        try:
            n_groups_value = int(self.n_groups)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("n_groups must be a positive integer") from exc
        if float(self.n_groups) != float(n_groups_value) or n_groups_value < 1:
            raise ValueError("n_groups must be a positive integer")
        C = 0.5 * (C + C.T)
        if not np.all(np.isfinite(C)) or not np.all(np.isfinite(c)):
            raise ValueError("coefficients and covariance must be finite")
        A = np.logical_and(A, A.T)
        np.fill_diagonal(A, True)
        raw_mo = np.asarray(self.max_overlap)
        try:
            mo = raw_mo.astype(int)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("max_overlap must contain integers") from exc
        if not np.all(np.asarray(raw_mo, dtype=float) == np.asarray(mo, dtype=float)):
            raise ValueError("max_overlap must contain integers")
        if mo.ndim == 0:
            mo = np.full(n, int(mo), dtype=int)
        else:
            mo = mo.reshape(-1)
        if mo.size != n:
            raise ValueError("max_overlap must be scalar or have one entry per Pauli term")
        if np.any(mo < 1):
            raise ValueError("all max_overlap entries must be at least one")
        mo = np.minimum(mo, n_groups_value)
        labels = list(range(n)) if self.labels is None else list(self.labels)
        if len(labels) != n:
            raise ValueError("labels must contain one entry per Pauli term")
        self.coefficients = c
        self.covariance = C
        self.compatibility = A
        self.n_groups = n_groups_value
        self.max_overlap = mo
        self.labels = labels

    @property
    def n_terms(self) -> int:
        return int(self.coefficients.size)


@dataclass
class GroupingSolution:
    assignment: np.ndarray
    groups: list[list[Any]]
    split_coefficients: np.ndarray
    variances: np.ndarray
    score: float
    epsilon2M: float
    proxy_value: float
    method: str
    status: str
    runtime: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    history: list[tuple[int, float]] = field(default_factory=list)
    observation_history: list[tuple[int, float]] = field(default_factory=list)

    def __post_init__(self):
        cleaned = []
        previous = float("inf")
        for iteration, objective in self.history:
            point = (int(iteration), float(objective))
            if not np.isfinite(point[1]):
                raise ValueError("history objectives must be finite")
            if point[1] > previous + 1.0e-12:
                raise ValueError("history must be monotone nonincreasing")
            cleaned.append(point)
            previous = min(previous, point[1])
        self.history = cleaned
        observations = []
        for iteration, objective in self.observation_history:
            point = (int(iteration), float(objective))
            if not np.isfinite(point[1]):
                raise ValueError("observation history objectives must be finite")
            observations.append(point)
        self.observation_history = observations

    def as_dict(self) -> dict[str, Any]:
        return {
            "assignment": self.assignment.copy(),
            "groups": [list(g) for g in self.groups],
            "split_coefficients": self.split_coefficients.copy(),
            "variances": self.variances.copy(),
            "score": float(self.score),
            "epsilon2M": float(self.epsilon2M),
            "proxy_value": float(self.proxy_value),
            "method": self.method,
            "status": self.status,
            "runtime": float(self.runtime),
            "metadata": dict(self.metadata),
            "history": list(self.history),
            "observation_history": list(self.observation_history),
        }


def coerce_problem(
    problem: Any = None,
    *,
    coefficients: Any = None,
    covariance: Any = None,
    compatibility: Any = None,
    n_groups: Optional[int] = None,
    max_overlap: Any = 1,
    labels: Optional[Sequence[Any]] = None,
) -> GroupingProblem:
    """Create :class:`GroupingProblem` from arrays, a mapping, or a NetworkX graph.

    For graph inputs, an edge denotes compatibility.  Coefficients are read from
    the first available node attribute among ``coefficient``, ``coeff``, ``c``,
    and ``weight``.  Variances are read from ``variance`` or ``var``; edge
    covariances are read from ``covariance``, ``cov``, or ``weight``.
    Explicit array arguments always take precedence over graph attributes.
    """
    if isinstance(problem, GroupingProblem):
        return problem
    if isinstance(problem, Mapping):
        data = dict(problem)
        coefficients = data.get("coefficients", data.get("c", coefficients))
        covariance = data.get("covariance", data.get("C", covariance))
        compatibility = data.get("compatibility", data.get("adjacency", compatibility))
        n_groups = data.get("n_groups", data.get("N_f", n_groups))
        max_overlap = data.get("max_overlap", max_overlap)
        labels = data.get("labels", labels)
    elif isinstance(problem, nx.Graph):
        nodes = list(problem.nodes)
        index = {node: i for i, node in enumerate(nodes)}
        n = len(nodes)
        if labels is None:
            labels = nodes
        if coefficients is None:
            coefficients = []
            for node in nodes:
                attrs = problem.nodes[node]
                for key in ("coefficient", "coeff", "c", "weight"):
                    if key in attrs:
                        coefficients.append(float(attrs[key]))
                        break
                else:
                    coefficients.append(1.0)
        if covariance is None:
            covariance = np.zeros((n, n), dtype=float)
            for node in nodes:
                i = index[node]
                attrs = problem.nodes[node]
                covariance[i, i] = float(attrs.get("variance", attrs.get("var", 1.0)))
            for u, v, attrs in problem.edges(data=True):
                i, j = index[u], index[v]
                val = float(attrs.get("covariance", attrs.get("cov", attrs.get("weight", 0.0))))
                covariance[i, j] = covariance[j, i] = val
        if compatibility is None:
            compatibility = np.eye(n, dtype=bool)
            for u, v in problem.edges:
                i, j = index[u], index[v]
                compatibility[i, j] = compatibility[j, i] = True
        if n_groups is None:
            n_groups = n
    elif problem is not None and coefficients is None:
        # Treat a one-dimensional positional argument as the coefficient vector.
        arr = np.asarray(problem)
        if arr.ndim == 1:
            coefficients = arr
    if coefficients is None or covariance is None or compatibility is None:
        raise TypeError(
            "coefficients, covariance, and compatibility are required.  Alternatively, "
            "pass a GroupingProblem, a mapping containing these arrays, or a NetworkX graph."
        )
    if n_groups is None:
        n_groups = len(np.asarray(coefficients).reshape(-1))
    return GroupingProblem(
        coefficients=coefficients,
        covariance=covariance,
        compatibility=compatibility,
        n_groups=n_groups,
        max_overlap=max_overlap,
        labels=labels,
    )


def measurement_aware_edge_reward(problem: GroupingProblem) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return node scores, two-term scores, and pairwise measurement savings.

    ``R[i,j] = s_i + s_j - s_ij`` is nonnegative up to numerical tolerance when
    the covariance matrix is positive semidefinite.  All square roots are
    evaluated before optimization, so ``R`` supplies fixed coefficients for
    the linearized pair-reward objective.
    """
    c = problem.coefficients
    C = problem.covariance
    v = np.maximum(c * c * np.diag(C), 0.0)
    s = np.sqrt(v)
    pair_var = v[:, None] + v[None, :] + 2.0 * (c[:, None] * c[None, :] * C)
    relevant_negative = np.logical_and(problem.compatibility, pair_var < -1.0e-8)
    if np.any(relevant_negative):
        i, j = np.argwhere(relevant_negative)[0]
        raise ValueError(
            "a compatible two-term variance is negative for terms {} and {}: {}".format(
                int(i), int(j), float(pair_var[i, j])
            )
        )
    pair_var[pair_var < 0.0] = 0.0
    s_pair = np.sqrt(pair_var)
    R = s[:, None] + s[None, :] - s_pair
    R = np.maximum(0.5 * (R + R.T), 0.0)
    R[~problem.compatibility] = 0.0
    np.fill_diagonal(R, 0.0)
    return s, s_pair, R


def canonicalize_assignment(x: ArrayLike) -> np.ndarray:
    x = np.asarray(x, dtype=np.int8)
    if x.ndim != 2:
        raise ValueError("assignment must be a two-dimensional binary array")
    x = (x > 0).astype(np.int8)
    # Put nonempty groups first, then use a deterministic lexicographic ordering.
    keys = []
    for a in range(x.shape[1]):
        column = tuple(int(v) for v in x[:, a])
        keys.append((0 if any(column) else 1, -sum(column), tuple(-v for v in column), a))
    order = [item[-1] for item in sorted(keys)]
    return x[:, order]


def groups_from_assignment(problem: GroupingProblem, x: ArrayLike, *, include_empty: bool = False) -> list[list[Any]]:
    x = np.asarray(x, dtype=bool)
    groups: list[list[Any]] = []
    for a in range(x.shape[1]):
        group = [problem.labels[i] for i in np.flatnonzero(x[:, a])]
        if group or include_empty:
            groups.append(group)
    return groups


def is_feasible(problem: GroupingProblem, x: ArrayLike, *, explain: bool = False):
    x = np.asarray(x, dtype=int)
    reasons: list[str] = []
    if x.shape != (problem.n_terms, problem.n_groups):
        reasons.append(f"assignment shape must be {(problem.n_terms, problem.n_groups)}")
    elif np.any((x != 0) & (x != 1)):
        reasons.append("assignment is not binary")
    else:
        counts = x.sum(axis=1)
        if np.any(counts < 1):
            reasons.append("at least one Pauli term is uncovered")
        if np.any(counts > problem.max_overlap):
            reasons.append("at least one Pauli term exceeds max_overlap")
        incompatible = np.argwhere(~problem.compatibility)
        for i, j in incompatible:
            if i < j and np.any(x[i] & x[j]):
                reasons.append(f"incompatible terms {i} and {j} share a group")
                break
    return (not reasons, reasons) if explain else not reasons


def proxy_objective(
    problem: GroupingProblem,
    x: ArrayLike,
    *,
    overlap_penalty: float = 0.0,
    node_costs: Optional[ArrayLike] = None,
    rewards: Optional[ArrayLike] = None,
) -> float:
    """Evaluate an auxiliary binary score used only for candidate generation."""
    x = np.asarray(x, dtype=float)
    if rewards is None:
        _, _, rewards = measurement_aware_edge_reward(problem)
    R = np.asarray(rewards, dtype=float)
    value = float(overlap_penalty * np.sum(x.sum(axis=1) - 1.0))
    if node_costs is None:
        node, _, _ = measurement_aware_edge_reward(problem)
    else:
        node = np.asarray(node_costs, dtype=float).reshape(-1)
    value += float(np.sum(node[:, None] * x))
    for a in range(problem.n_groups):
        idx = np.flatnonzero(x[:, a] > 0.5)
        if idx.size > 1:
            value -= float(np.triu(R[np.ix_(idx, idx)], 1).sum())
    return value


def fragment_variances(problem: GroupingProblem, omega: ArrayLike) -> np.ndarray:
    """Evaluate fragment variances using only active group submatrices.

    A dense ``einsum`` over every term and every labeled group scales as
    ``O(n_terms**2 * n_groups)`` even though the coefficient matrix is sparse.
    Evaluating each active clique separately instead costs
    ``sum_alpha O(|G_alpha|**2)``, which is substantially smaller for the
    many-group instances targeted by the explicit formulation.
    """

    omega = np.asarray(omega, dtype=float)
    if omega.shape != (problem.n_terms, problem.n_groups):
        raise ValueError("split coefficient array has an invalid shape")
    vals = np.zeros(problem.n_groups, dtype=float)
    for alpha in range(problem.n_groups):
        members = np.flatnonzero(np.abs(omega[:, alpha]) > 0.0)
        if members.size == 0:
            continue
        weights = omega[members, alpha]
        covariance = problem.covariance[np.ix_(members, members)]
        vals[alpha] = float(weights @ covariance @ weights)
    if np.any(vals < -1.0e-8):
        index = int(np.argmin(vals))
        raise ValueError(
            "fragment {} has a negative variance {}".format(index, float(vals[index]))
        )
    vals[vals < 0.0] = 0.0
    return vals


def evaluate_assignment(
    problem: GroupingProblem,
    x: ArrayLike,
    *,
    weight_heuristic: str = "iterative_relative_std",
    heuristic_options: Optional[Mapping[str, Any]] = None,
    overlap_penalty: float = 0.0,
    node_costs: Optional[ArrayLike] = None,
    rewards: Optional[ArrayLike] = None,
) -> dict[str, Any]:
    x = canonicalize_assignment(x)
    options = dict(heuristic_options or {})
    omega = assign_overlapping_weights(problem, x, heuristic=weight_heuristic, **options)
    variances = fragment_variances(problem, omega)
    score = float(np.sqrt(variances).sum())
    return {
        "assignment": x,
        "split_coefficients": omega,
        "variances": variances,
        "score": score,
        "epsilon2M": score * score,
        "proxy_value": proxy_objective(
            problem, x, overlap_penalty=overlap_penalty, node_costs=node_costs, rewards=rewards
        ),
    }


def _compatible_with_group(problem: GroupingProblem, i: int, members: Iterable[int]) -> bool:
    members = list(members)
    return not members or bool(np.all(problem.compatibility[i, members]))


def greedy_feasible_assignment(
    problem: GroupingProblem,
    *,
    rewards: Optional[ArrayLike] = None,
    overlap_penalty: float = 0.0,
    random_state: Optional[int] = None,
    randomize: bool = False,
) -> np.ndarray:
    """Construct a feasible clique cover and then add profitable overlaps."""
    rng = np.random.default_rng(random_state)
    incompat = nx.Graph()
    incompat.add_nodes_from(range(problem.n_terms))
    for i in range(problem.n_terms):
        for j in range(i + 1, problem.n_terms):
            if not problem.compatibility[i, j]:
                incompat.add_edge(i, j)
    strategy = "random_sequential" if randomize else "saturation_largest_first"
    colors = nx.coloring.greedy_color(incompat, strategy=strategy, interchange=False)
    used = 1 + max(colors.values(), default=-1)
    if used > problem.n_groups:
        # A bounded backtracking coloring is exact for the small cases where greedy fails.
        order = sorted(range(problem.n_terms), key=lambda i: incompat.degree(i), reverse=True)
        assignment = [-1] * problem.n_terms
        def dfs(pos: int) -> bool:
            if pos == len(order):
                return True
            node = order[pos]
            color_order = list(range(problem.n_groups))
            if randomize:
                rng.shuffle(color_order)
            for color in color_order:
                if all(assignment[nbr] != color for nbr in incompat.neighbors(node)):
                    assignment[node] = color
                    if dfs(pos + 1):
                        return True
                    assignment[node] = -1
            return False
        if not dfs(0):
            raise ValueError("n_groups is too small to cover the incompatibility graph")
        colors = {i: assignment[i] for i in range(problem.n_terms)}
    x = np.zeros((problem.n_terms, problem.n_groups), dtype=np.int8)
    for i, color in colors.items():
        x[i, int(color)] = 1
    node_scores, _, default_rewards = measurement_aware_edge_reward(problem)
    if rewards is None:
        rewards = default_rewards
    R = np.asarray(rewards, dtype=float)
    candidates: list[tuple[float, int, int]] = []
    for i in range(problem.n_terms):
        for a in range(problem.n_groups):
            if x[i, a] or x[i].sum() >= problem.max_overlap[i]:
                continue
            members = np.flatnonzero(x[:, a])
            if _compatible_with_group(problem, i, members):
                delta = float(overlap_penalty + node_scores[i] - R[i, members].sum())
                candidates.append((delta, i, a))
    if randomize:
        rng.shuffle(candidates)
    candidates.sort(key=lambda item: item[0])
    for delta, i, a in candidates:
        if delta >= 0.0 or x[i, a] or x[i].sum() >= problem.max_overlap[i]:
            continue
        members = np.flatnonzero(x[:, a])
        if _compatible_with_group(problem, i, members):
            x[i, a] = 1
    return canonicalize_assignment(x)


def repair_assignment(
    problem: GroupingProblem,
    x: ArrayLike,
    *,
    rewards: Optional[ArrayLike] = None,
    overlap_penalty: float = 0.0,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """Project an arbitrary binary matrix onto a feasible overlapping support."""
    rng = np.random.default_rng(random_state)
    x = np.asarray(x, dtype=np.int8)
    if x.shape != (problem.n_terms, problem.n_groups):
        x = np.resize(x, (problem.n_terms, problem.n_groups))
    x = (x > 0).astype(np.int8)
    node_scores, _, default_rewards = measurement_aware_edge_reward(problem)
    if rewards is None:
        rewards = default_rewards
    R = np.asarray(rewards, dtype=float)
    # Resolve conflicts by removing the endpoint whose removal loses less reward.
    for a in range(problem.n_groups):
        changed = True
        while changed:
            changed = False
            members = list(np.flatnonzero(x[:, a]))
            for pos, i in enumerate(members):
                for j in members[pos + 1:]:
                    if not problem.compatibility[i, j]:
                        gi = float(R[i, np.flatnonzero(x[:, a])].sum())
                        gj = float(R[j, np.flatnonzero(x[:, a])].sum())
                        remove = i if gi <= gj else j
                        x[remove, a] = 0
                        changed = True
                        break
                if changed:
                    break
    # Enforce upper overlap limits.
    for i in range(problem.n_terms):
        active = list(np.flatnonzero(x[i]))
        while len(active) > problem.max_overlap[i]:
            losses = []
            for a in active:
                members = np.flatnonzero(x[:, a])
                losses.append((float(R[i, members].sum()) - overlap_penalty - node_scores[i], a))
            _, remove_a = min(losses)
            x[i, remove_a] = 0
            active.remove(remove_a)
    # Cover uncovered nodes using the lowest incremental auxiliary score.
    for i in range(problem.n_terms):
        if x[i].sum() > 0:
            continue
        choices = []
        for a in range(problem.n_groups):
            members = np.flatnonzero(x[:, a])
            if _compatible_with_group(problem, i, members):
                choices.append((-float(R[i, members].sum()), rng.random(), a))
        if not choices:
            # Restart from a guaranteed feasible construction rather than silently returning an invalid sample.
            return greedy_feasible_assignment(
                problem, rewards=R, overlap_penalty=overlap_penalty, random_state=random_state, randomize=True
            )
        _, _, a = min(choices)
        x[i, a] = 1
    return canonicalize_assignment(x)
