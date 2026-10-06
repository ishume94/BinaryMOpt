"""Exact-name coefficient reconstruction heuristics for overlapping groups."""

import numpy as np


WEIGHT_HEURISTICS = (
    "iterative_relative_std",  # Iterative leave-one-out rule.
    "equal",
    "relative_std",
    "relative_variance",
    "score_softmax",
)


def _validated_support(problem, assignment):
    assignment = np.asarray(assignment, dtype=bool)
    expected = (int(problem.n_terms), int(problem.n_groups))
    if assignment.shape != expected:
        raise ValueError(
            "assignment must have shape {}, got {}".format(
                expected, assignment.shape
            )
        )
    coverage = assignment.sum(axis=1)
    if np.any(coverage < 1):
        raise ValueError("every term needs at least one selected group")
    maximum = np.asarray(problem.max_overlap, dtype=int).reshape(-1)
    if np.any(coverage > maximum):
        raise ValueError("assignment exceeds max_overlap")
    for alpha in range(problem.n_groups):
        members = np.flatnonzero(assignment[:, alpha])
        if members.size and not np.all(
            np.asarray(problem.compatibility, dtype=bool)[np.ix_(members, members)]
        ):
            raise ValueError("a group contains incompatible terms")
    return assignment, coverage


def _equal_weights(problem, assignment, coverage):
    return (
        np.asarray(problem.coefficients, dtype=float)[:, np.newaxis]
        * assignment
        / coverage[:, np.newaxis]
    ).astype(float)


def _initial_weights(problem, assignment, coverage, initial_omega):
    if initial_omega is None:
        return _equal_weights(problem, assignment, coverage)
    omega = np.array(initial_omega, dtype=float, copy=True)
    if omega.shape != assignment.shape or not np.all(np.isfinite(omega)):
        raise ValueError("initial_omega must be finite and have the assignment shape")
    if np.any(np.abs(omega[~assignment]) > 1.e-12):
        raise ValueError("initial_omega is nonzero outside the selected support")
    if not np.allclose(omega.sum(axis=1), problem.coefficients, atol=1.e-9, rtol=1.e-9):
        raise ValueError("initial_omega does not reconstruct the Hamiltonian")
    omega *= assignment
    return omega


def _renormalize(problem, assignment, omega, tolerance):
    coefficients = np.asarray(problem.coefficients, dtype=float)
    omega = np.asarray(omega, dtype=float)
    omega *= assignment
    for term_index in range(problem.n_terms):
        memberships = np.flatnonzero(assignment[term_index])
        total = float(omega[term_index, memberships].sum())
        coefficient = float(coefficients[term_index])
        if abs(total) <= tolerance:
            omega[term_index, memberships] = coefficient / memberships.size
        else:
            omega[term_index, memberships] *= coefficient / total
        residual = coefficient - float(omega[term_index, memberships].sum())
        omega[term_index, memberships[-1]] += residual
    omega *= assignment
    return omega


def _group_state(problem, assignment, omega):
    covariance = np.asarray(problem.covariance, dtype=float)
    action = np.zeros_like(omega)
    variances = np.zeros(problem.n_groups, dtype=float)
    for alpha in range(problem.n_groups):
        members = np.flatnonzero(assignment[:, alpha])
        if members.size == 0:
            continue
        weights = omega[members, alpha]
        local_action = covariance[np.ix_(members, members)] @ weights
        action[members, alpha] = local_action
        variances[alpha] = float(weights @ local_action)
    return action, variances


def iterative_relative_std_weights(
    problem,
    assignment,
    *,
    max_iterations=100,
    tolerance=1.0e-10,
    damping=0.5,
    initial_omega=None,
):
    """Reproduce the stable leave-one-out standard-deviation iteration."""

    assignment, coverage = _validated_support(problem, assignment)
    if int(max_iterations) < 1:
        raise ValueError("max_iterations must be at least one")
    if not 0.0 < float(damping) <= 1.0:
        raise ValueError("damping must lie in (0, 1]")
    omega = _initial_weights(problem, assignment, coverage, initial_omega)
    covariance = np.asarray(problem.covariance, dtype=float)
    coefficients = np.asarray(problem.coefficients, dtype=float)
    for _ in range(int(max_iterations)):
        covariance_action = covariance @ omega
        group_variances = np.sum(omega * covariance_action, axis=0)
        target = np.zeros_like(omega)
        for term_index in range(problem.n_terms):
            memberships = np.flatnonzero(assignment[term_index])
            if memberships.size == 1:
                target[term_index, memberships[0]] = coefficients[term_index]
                continue
            old_weights = omega[term_index, memberships]
            leave_one_out = (
                group_variances[memberships]
                - 2.0
                * old_weights
                * covariance_action[term_index, memberships]
                + np.square(old_weights) * covariance[term_index, term_index]
            )
            leave_one_out[np.abs(leave_one_out) < 1.0e-10] = 0.0
            if np.any(leave_one_out < 0.0):
                raise ValueError("a leave-one-out group variance is negative")
            relative_std = np.sqrt(leave_one_out)
            denominator = float(relative_std.sum())
            if denominator <= float(tolerance):
                shares = np.full(memberships.size, 1.0 / memberships.size)
            else:
                shares = relative_std / denominator
            target[term_index, memberships] = coefficients[term_index] * shares
        updated = float(damping) * target + (1.0 - float(damping)) * omega
        updated *= assignment
        if np.max(np.abs(updated - omega)) <= float(tolerance):
            omega = updated
            break
        omega = updated
    return _renormalize(problem, assignment, omega, float(tolerance))


def assign_overlapping_weights(
    problem,
    assignment,
    *,
    heuristic="iterative_relative_std",
    gamma=None,
    beta=5.0,
    delta=1.0e-12,
    sweeps=3,
    damping=0.5,
    max_iterations=100,
    tolerance=1.0e-10,
    initial_omega=None,
):
    """Reconstruct coefficients, optionally iterating from current weights.

    ``initial_omega`` must reconstruct the Hamiltonian on the selected support;
    new memberships may start at zero. Iterative rules retain their usual update
    formulas and stopping criteria. The noniterative ``equal`` rule always
    returns equal shares.
    """

    if heuristic not in WEIGHT_HEURISTICS:
        raise ValueError(
            "unknown weight heuristic {!r}; choose from {}".format(
                heuristic, WEIGHT_HEURISTICS
            )
        )
    if heuristic == "iterative_relative_std":
        return iterative_relative_std_weights(
            problem,
            assignment,
            max_iterations=max_iterations,
            tolerance=tolerance,
            damping=damping,
            initial_omega=initial_omega,
        )

    assignment, coverage = _validated_support(problem, assignment)
    omega = _initial_weights(problem, assignment, coverage, initial_omega)
    if heuristic == "equal":
        return _equal_weights(problem, assignment, coverage)
    if int(sweeps) < 1:
        raise ValueError("sweeps must be at least one")
    if not 0.0 < float(damping) <= 1.0:
        raise ValueError("damping must lie in (0, 1]")
    if float(delta) < 0.0:
        raise ValueError("delta must be non-negative")

    covariance = np.asarray(problem.covariance, dtype=float)
    coefficients = np.asarray(problem.coefficients, dtype=float)
    diagonal = np.diag(covariance)
    if heuristic == "relative_std":
        exponent = 0.5 if gamma is None else float(gamma)
    elif heuristic == "relative_variance":
        exponent = 1.0 if gamma is None else float(gamma)
    else:
        exponent = None

    for _ in range(int(sweeps)):
        covariance_action, group_variances = _group_state(
            problem, assignment, omega
        )
        target = np.zeros_like(omega)
        for term_index in range(problem.n_terms):
            memberships = np.flatnonzero(assignment[term_index])
            if memberships.size == 1:
                target[term_index, memberships[0]] = coefficients[term_index]
                continue
            old_weights = omega[term_index, memberships]
            background = (
                group_variances[memberships]
                - 2.0
                * old_weights
                * covariance_action[term_index, memberships]
                + old_weights * old_weights * diagonal[term_index]
            )
            background[np.abs(background) < 1.0e-10] = 0.0
            if np.any(background < 0.0):
                raise ValueError(
                    "a leave-one-out fragment variance is negative for term {}".format(
                        term_index
                    )
                )
            if heuristic in ("relative_std", "relative_variance"):
                raw = np.power(background + float(delta), exponent)
            else: #Score-Softmax
                covariance_without = (
                    covariance_action[term_index, memberships]
                    - old_weights * diagonal[term_index]
                )
                trial_share = coefficients[term_index] / memberships.size
                trial_variance = (
                    background
                    + 2.0 * trial_share * covariance_without
                    + trial_share * trial_share * diagonal[term_index]
                )
                trial_variance[np.abs(trial_variance) < 1.0e-10] = 0.0
                if np.any(trial_variance < 0.0):
                    raise ValueError(
                        "a trial fragment variance is negative for term {}".format(
                            term_index
                        )
                    )
                score_change = np.sqrt(trial_variance) - np.sqrt(background) #Score-Softmax
                logits = -float(beta) * score_change
                logits -= np.max(logits)
                raw = np.exp(np.clip(logits, -700.0, 700.0))
            denominator = float(raw.sum())
            if not np.all(np.isfinite(raw)) or denominator <= 0.0:
                shares = np.full(memberships.size, 1.0 / memberships.size)
            else:
                shares = raw / denominator
            target[term_index, memberships] = coefficients[term_index] * shares
        omega = float(damping) * target + (1.0 - float(damping)) * omega
        omega = _renormalize(
            problem, assignment, omega, max(float(tolerance), 1.0e-15)
        )
    return omega


__all__ = [
    "WEIGHT_HEURISTICS",
    "assign_overlapping_weights",
    "iterative_relative_std_weights",
]
