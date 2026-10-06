"""GFlow-VQE iterative coefficient splitting for binary groupings.

The coefficient optimizer is adapted from
``GFlow-VQE/gflow_vqe/overlapping_helpers.py``.  The bridge at
the bottom is specific to :mod:`bin_mopt`.  It supports both standard ICS,
which expands a non-overlapping partition before optimizing coefficients, and
fixed-support ICS for memberships selected by the binary overlapping solver.
"""

from copy import deepcopy
import time

import numpy as np
from tequila.grouping.binary_rep import BinaryPauliString
from tequila.grouping.binary_utils import term_commutes_with_group


class OverlappingAuxiliary:
    """Pass a covariance dictionary and iteration count to ICS."""

    def __init__(self, cov_dict, n_iter=5, deadline=None):
        self.cov_dict = cov_dict
        self.n_iter = n_iter
        self.deadline = None if deadline is None else float(deadline)


class OverlappingGroupsWoFixed:
    """Eliminate one coefficient per shared term using reconstruction."""

    def __init__(self, o_groups, o_terms, term_exists_in):
        def exclude_fixed_coeffs(o_groups, o_terms, term_exists_in):
            fixed_grp = []
            o_groups_wo_fixed = deepcopy(o_groups)
            term_exists_in_wo_fixed = deepcopy(term_exists_in)
            for term_idx, term in enumerate(o_terms):
                fixed_grp.append(term_exists_in[term_idx][-1])
                o_groups_wo_fixed[fixed_grp[term_idx]].remove(term)
                term_exists_in_wo_fixed[term_idx].remove(fixed_grp[term_idx])
            n_coeff_grp = np.array(
                [len(group) for group in o_groups_wo_fixed]
            )
            init_idx = [
                sum(n_coeff_grp[:index])
                for index in range(len(o_groups_wo_fixed))
            ]
            return (
                fixed_grp,
                o_groups_wo_fixed,
                term_exists_in_wo_fixed,
                n_coeff_grp,
                init_idx,
            )

        def get_term_idxs(
            o_terms, o_groups_wo_fixed, term_exists_in_wo_fixed, init_idx
        ):
            term_idxs = {}
            for term_idx, term in enumerate(o_terms):
                cur_idxs = {}
                for group_idx in term_exists_in_wo_fixed[term_idx]:
                    cur_idxs[group_idx] = (
                        init_idx[group_idx]
                        + o_groups_wo_fixed[group_idx].index(term)
                    )
                term_idxs[term.binary_tuple()] = cur_idxs
            return term_idxs

        (
            self.fixed_grp,
            self.o_groups,
            self.term_exists_in,
            self.n_coeff_grp,
            self.init_idx,
        ) = exclude_fixed_coeffs(o_groups, o_terms, term_exists_in)
        self.term_idxs = get_term_idxs(
            o_terms,
            self.o_groups,
            self.term_exists_in,
            self.init_idx,
        )


def get_cov(term1, term2, cov_dict):
    """Return a commuting covariance in either stored orientation."""

    key = (term1.binary_tuple(), term2.binary_tuple())
    if key in cov_dict:
        return cov_dict[key]
    reverse_key = (key[1], key[0])
    if reverse_key in cov_dict:
        return cov_dict[reverse_key]
    raise KeyError(
        "Covariance not found for terms {} and {}.".format(*key)
    )


def cov_term_w_group(term, group, cov_dict):
    covariance = 0.0
    for group_term in group:
        covariance += (
            group_term.get_coeff() * get_cov(term, group_term, cov_dict)
        )
    return covariance


def get_opt_sample_size(groups, cov_dict):
    """Allocate shots in proportion to the group standard deviations."""

    weights = np.zeros(len(groups))
    for group_index, group in enumerate(groups):
        variance = 0.0
        for term1 in group:
            for term2 in group:
                variance += (
                    term1.coeff
                    * term2.coeff
                    * get_cov(term1, term2, cov_dict)
                )
        real_variance = float(np.real_if_close(variance))
        if real_variance < 0.0 and abs(real_variance) <= 1.0e-10:
            real_variance = 0.0
        if real_variance < 0.0:
            raise ValueError("ICS encountered a negative group variance.")
        weights[group_index] = np.sqrt(real_variance)
    if not np.any(weights > 0.0):
        return np.full(len(groups), 1.0 / len(groups))
    # The GFlow equations divide by every shot fraction. A tiny floor makes
    # valid zero-variance groups numerically safe without changing support.
    floor = max(float(weights.max()) * 1.0e-12, 1.0e-15)
    weights = np.maximum(weights, floor)
    return weights / np.sum(weights)


class OverlappingGroups:
    """The GFlow-VQE/Tequila ICS coefficient optimizer."""

    def __init__(self, no_groups, o_terms, term_exists_in):
        self.no_groups = no_groups
        self.o_terms = o_terms
        self.term_exists_in = term_exists_in
        self.o_groups = [[] for _ in range(len(no_groups))]
        for term_index, term in enumerate(o_terms):
            for group_index in term_exists_in[term_index]:
                self.o_groups[group_index].append(term)
        self.wo_fixed = OverlappingGroupsWoFixed(
            self.o_groups, self.o_terms, self.term_exists_in
        )

    @classmethod
    def init_from_groups(cls, groups, terms, condition="fc"):
        """GFlow-VQE standard ICS initialization from a partition."""

        groups = [list(group) for group in groups]
        terms = list(terms)
        grouped_keys = [
            term.binary_tuple() for group in groups for term in group
        ]
        reference_keys = [term.binary_tuple() for term in terms]
        if len(grouped_keys) != len(set(grouped_keys)):
            raise ValueError("ICS initialization groups must not overlap.")
        if set(grouped_keys) != set(reference_keys):
            raise ValueError(
                "ICS initialization groups do not cover the reference terms."
            )

        newly_added = [[] for _ in groups]
        overlapping_terms = []
        term_exists_in = []
        for term in sorted(
            terms, key=lambda value: np.abs(value.coeff), reverse=True
        ):
            group_indices = []
            for group_index, group in enumerate(groups):
                commutes = term_commutes_with_group(
                    term, group, condition
                ) and term_commutes_with_group(
                    term, newly_added[group_index], condition
                )
                if commutes:
                    group_indices.append(group_index)
                    newly_added[group_index].append(term)
            if len(group_indices) > 1:
                overlapping_terms.append(term.term_w_coeff(0.0))
                term_exists_in.append(group_indices)
        return cls(groups, overlapping_terms, term_exists_in)

    def optimize_pauli_coefficients(self, cov_dict, sample_size):
        def prep_mat_single_row(term, group_index):
            mat_single = np.zeros(np.sum(self.wo_fixed.n_coeff_grp))
            for term2 in self.o_groups[group_index]:
                term2_idx_dict = self.wo_fixed.term_idxs[
                    term2.binary_tuple()
                ]
                covariance = np.real_if_close(
                    get_cov(term, term2, cov_dict)
                )
                if term2 in self.wo_fixed.o_groups[group_index]:
                    mat_single[term2_idx_dict[group_index]] -= (
                        covariance / sample_size[group_index]
                    )
                else:
                    for coefficient_index in term2_idx_dict.values():
                        mat_single[coefficient_index] += (
                            covariance / sample_size[group_index]
                        )
            return mat_single

        def prep_b_single_row(term, group_index):
            return np.real_if_close(
                cov_term_w_group(
                    term, self.no_groups[group_index], cov_dict
                )
                / sample_size[group_index]
            )

        matrix_size = int(np.sum(self.wo_fixed.n_coeff_grp))
        matrix = np.zeros((matrix_size, matrix_size))
        vector = np.zeros((1, matrix_size))
        row_index = 0
        for group_index, group in enumerate(self.wo_fixed.o_groups):
            for term1 in group:
                matrix[row_index] += prep_mat_single_row(
                    term1, group_index
                )
                vector[0, row_index] += prep_b_single_row(
                    term1, group_index
                )
                fixed_group_index = self.wo_fixed.fixed_grp[
                    self.o_terms.index(term1)
                ]
                matrix[row_index] -= prep_mat_single_row(
                    term1, fixed_group_index
                )
                vector[0, row_index] -= prep_b_single_row(
                    term1, fixed_group_index
                )
                row_index += 1
        solution = np.linalg.lstsq(matrix, vector.T, rcond=None)[0]
        return solution.T[0]

    def overlapping_groups_from_coeff(self, coeff):
        def add_coeff_times_term(cur_coeff, term, group_index):
            for term_index, group_term in enumerate(
                final_overlapping_groups[group_index]
            ):
                if group_term.binary_tuple() == term.binary_tuple():
                    final_overlapping_groups[group_index][term_index].set_coeff(
                        cur_coeff + group_term.get_coeff()
                    )
                    return
            final_overlapping_groups[group_index].append(
                term.term_w_coeff(cur_coeff)
            )

        final_overlapping_groups = deepcopy(self.no_groups)
        for term_index, term in enumerate(self.o_terms):
            fixed_group_coefficient = 0.0
            for group_index in self.wo_fixed.term_exists_in[term_index]:
                coefficient = coeff[
                    self.wo_fixed.init_idx[group_index]
                    + self.wo_fixed.o_groups[group_index].index(term)
                ]
                fixed_group_coefficient -= coefficient
                add_coeff_times_term(coefficient, term, group_index)
            fixed_group_index = self.wo_fixed.fixed_grp[term_index]
            add_coeff_times_term(
                fixed_group_coefficient, term, fixed_group_index
            )
        return final_overlapping_groups

    def optimal_overlapping_groups(self, overlap_aux):
        current_groups = self.no_groups
        self.completed_iterations = 0
        self.deadline_reached = False
        for _ in range(overlap_aux.n_iter):
            if (
                overlap_aux.deadline is not None
                and time.monotonic() >= overlap_aux.deadline
            ):
                self.deadline_reached = True
                break
            sample_size = get_opt_sample_size(
                current_groups, overlap_aux.cov_dict
            )
            coefficients = self.optimize_pauli_coefficients(
                overlap_aux.cov_dict, sample_size
            )
            current_groups = self.overlapping_groups_from_coeff(coefficients)
            self.completed_iterations += 1
        if (
            overlap_aux.deadline is not None
            and time.monotonic() >= overlap_aux.deadline
        ):
            self.deadline_reached = True
        return current_groups


def _binary_vector(term):
    n_qubits = len(term.ops)
    x_bits = [0.0] * n_qubits
    z_bits = [0.0] * n_qubits
    for qubit, pauli in term.pauli_tuple:
        if pauli in ("X", "Y"):
            x_bits[qubit] = 1.0
        if pauli in ("Z", "Y"):
            z_bits[qubit] = 1.0
    return np.asarray(x_bits + z_bits)


def _context_binary_data(context):
    terms = [
        BinaryPauliString(_binary_vector(term), context.coefficients[index])
        for index, term in enumerate(context.terms)
    ]
    keys = [term.binary_tuple() for term in terms]
    index_by_key = {key: index for index, key in enumerate(keys)}
    cov_dict = {}
    for left in range(context.n_terms):
        for right in range(left, context.n_terms):
            if context.compatible[left, right]:
                cov_dict[(keys[left], keys[right])] = context.covariance[
                    left, right
                ]
    return terms, index_by_key, cov_dict


def _groups_to_arrays(context, binary_groups, index_by_key):
    assignment = np.zeros(
        (context.n_terms, len(binary_groups)), dtype=bool
    )
    omega = np.zeros(assignment.shape, dtype=float)
    for group_index, group in enumerate(binary_groups):
        seen = set()
        for term in group:
            term_index = index_by_key[term.binary_tuple()]
            if term_index in seen:
                raise ValueError("ICS returned a duplicate term in one group.")
            seen.add(term_index)
            assignment[term_index, group_index] = True
            omega[term_index, group_index] += float(
                np.real_if_close(term.get_coeff())
            )
    if not np.allclose(
        omega.sum(axis=1), context.coefficients, atol=1.0e-9, rtol=1.0e-9
    ):
        raise ValueError("ICS coefficients do not reconstruct the Hamiltonian.")
    return assignment, omega


def _direct_sample_ratios(context, omega):
    """Return a feasible allocation without entering another ICS iteration."""

    variances = np.sum(omega * (context.covariance @ omega), axis=0)
    variances[np.abs(variances) < 1.0e-10] = 0.0
    if np.any(variances < 0.0):
        raise ValueError("ICS encountered a negative group variance.")
    weights = np.sqrt(variances)
    if not np.any(weights > 0.0):
        return np.full(omega.shape[1], 1.0 / omega.shape[1])
    floor = max(float(weights.max()) * 1.0e-12, 1.0e-15)
    weights = np.maximum(weights, floor)
    return weights / weights.sum()


def optimize_from_nonoverlapping_groups(
    context, assignment, *, n_iter=5, condition="fc", deadline=None
):
    """Run standard GFlow-VQE ICS from a binary non-overlap partition."""

    assignment = np.asarray(assignment, dtype=bool)
    if assignment.ndim != 2 or assignment.shape[0] != context.n_terms:
        raise ValueError("assignment has the wrong shape.")
    if np.any(assignment.sum(axis=0) == 0):
        raise ValueError("ICS requires compact groups without empty columns.")
    if not np.all(assignment.sum(axis=1) == 1):
        raise ValueError("Standard ICS initialization must be non-overlapping.")
    if n_iter < 1:
        raise ValueError("n_iter must be at least one.")

    if deadline is not None and time.monotonic() >= float(deadline):
        omega = context.coefficients[:, np.newaxis] * assignment
        return (
            assignment.copy(),
            omega,
            _direct_sample_ratios(context, omega),
        )

    terms, index_by_key, cov_dict = _context_binary_data(context)
    initial_groups = [
        [terms[index] for index in np.flatnonzero(assignment[:, group])]
        for group in range(assignment.shape[1])
    ]
    # GFlow-VQE derives the stable tie order from the supplied groups when its
    # optional ``terms`` argument is omitted.  Preserve that group-major order
    # here so equal-magnitude coefficients produce the same support expansion.
    group_major_terms = [term for group in initial_groups for term in group]
    optimizer = OverlappingGroups.init_from_groups(
        initial_groups, group_major_terms, condition=condition
    )
    auxiliary = OverlappingAuxiliary(
        cov_dict, n_iter=n_iter, deadline=deadline
    )
    binary_groups = optimizer.optimal_overlapping_groups(auxiliary)
    sample_ratios = get_opt_sample_size(binary_groups, cov_dict)
    final_assignment, omega = _groups_to_arrays(
        context, binary_groups, index_by_key
    )
    return final_assignment, omega, np.asarray(sample_ratios, dtype=float)


def optimize_fixed_binary_support(
    context, assignment, *, initial_omega=None, n_iter=5, deadline=None
):
    """Bridge a bin_mopt support into the copied GFlow-VQE ICS classes."""

    assignment = np.asarray(assignment, dtype=bool)
    if assignment.ndim != 2 or assignment.shape[0] != context.n_terms:
        raise ValueError("assignment has the wrong shape.")
    if np.any(assignment.sum(axis=0) == 0):
        raise ValueError("ICS requires compact support without empty groups.")
    coverage = assignment.sum(axis=1)
    if np.any(coverage < 1):
        raise ValueError("Every term needs at least one selected group.")
    if n_iter < 1:
        raise ValueError("n_iter must be at least one.")

    if initial_omega is None:
        initial_omega = (
            context.coefficients[:, np.newaxis]
            * assignment
            / coverage[:, np.newaxis]
        )
    else:
        initial_omega = np.asarray(initial_omega, dtype=float).copy()
        if initial_omega.shape != assignment.shape:
            raise ValueError("initial_omega has the wrong shape.")
        if np.any(np.abs(initial_omega[~assignment]) > 1.0e-14):
            raise ValueError("initial_omega is nonzero outside the support.")
        if not np.allclose(
            initial_omega.sum(axis=1),
            context.coefficients,
            atol=1.0e-9,
            rtol=1.0e-9,
        ):
            raise ValueError(
                "initial_omega does not reconstruct the Hamiltonian."
            )

    if deadline is not None and time.monotonic() >= float(deadline):
        return initial_omega, _direct_sample_ratios(context, initial_omega)

    terms, index_by_key, cov_dict = _context_binary_data(context)

    no_groups = [[] for _ in range(assignment.shape[1])]
    overlapping_terms = []
    term_exists_in = []
    for term_index, term in enumerate(terms):
        memberships = np.flatnonzero(assignment[term_index]).tolist()
        for group_index in memberships:
            no_groups[group_index].append(
                term.term_w_coeff(initial_omega[term_index, group_index])
            )
        if len(memberships) > 1:
            overlapping_terms.append(term.term_w_coeff(0.0))
            term_exists_in.append(memberships)

    optimizer = OverlappingGroups(
        no_groups, overlapping_terms, term_exists_in
    )
    auxiliary = OverlappingAuxiliary(
        cov_dict, n_iter=n_iter, deadline=deadline
    )
    binary_groups = optimizer.optimal_overlapping_groups(auxiliary)
    sample_ratios = get_opt_sample_size(binary_groups, cov_dict)

    omega = np.zeros(assignment.shape, dtype=float)
    for group_index, group in enumerate(binary_groups):
        for term in group:
            term_index = index_by_key[term.binary_tuple()]
            if not assignment[term_index, group_index]:
                raise ValueError("ICS changed the fixed binary support.")
            omega[term_index, group_index] += float(
                np.real_if_close(term.get_coeff())
            )
    if not np.allclose(
        omega.sum(axis=1), context.coefficients, atol=1.0e-9, rtol=1.0e-9
    ):
        raise ValueError("ICS coefficients do not reconstruct the Hamiltonian.")

    def eps_sq_m(weights):
        variances = np.sum(weights * (context.covariance @ weights), axis=0)
        variances[np.abs(variances) < 1.0e-10] = 0.0
        if np.any(variances < 0.0):
            raise ValueError("ICS produced a negative group variance.")
        return float(np.square(np.sqrt(variances).sum()))

    if eps_sq_m(omega) > eps_sq_m(initial_omega) + 1.0e-12:
        omega = initial_omega
        initial_groups = []
        for group_index in range(assignment.shape[1]):
            initial_groups.append(
                [
                    terms[term_index].term_w_coeff(
                        omega[term_index, group_index]
                    )
                    for term_index in np.flatnonzero(
                        assignment[:, group_index]
                    )
                ]
            )
        sample_ratios = get_opt_sample_size(initial_groups, cov_dict)
    return omega, np.asarray(sample_ratios, dtype=float)


__all__ = [
    "OverlappingAuxiliary",
    "OverlappingGroups",
    "OverlappingGroupsWoFixed",
    "get_opt_sample_size",
    "optimize_fixed_binary_support",
    "optimize_from_nonoverlapping_groups",
]
