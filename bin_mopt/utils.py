"""Hamiltonian, wavefunction, and fast covariance utilities.

The Pauli-action covariance path is adapted from a fast VarSI version
It constructs all rows
``P_i |psi>`` 
once and obtains the covariance matrix from a Gram product, 
instead of repeatedly applying every Pauli pair.
"""

import math
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np
import tequila as tq
from openfermion import QubitOperator
from openfermion.linalg import get_ground_state, get_sparse_operator
from tequila.hamiltonian import QubitHamiltonian


DEFAULT_COVARIANCE_CHUNKSIZE = 128


@dataclass(frozen=True)
class PauliTerm:
    """One indexed Pauli word and its Hamiltonian coefficient."""

    index: int
    pauli_tuple: tuple[tuple[int, str], ...]
    ops: tuple[str, ...]
    coefficient: complex
    word: str
    source_order: int


_ACTION_STATE = None
_ACTION_N_QUBITS = None
_ACTION_TERMS = None


def default_cov_workers() -> int:
    return max(1, min(8, os.cpu_count() or 1))


def clean_complex(value, tiny: float = 1.0e-12) -> complex:
    value = complex(value)
    imag = 0.0 if abs(value.imag) < tiny else value.imag
    return complex(value.real, imag)


def clean_real(value, tiny: float = 1.0e-9) -> float:
    value = complex(value)
    if abs(value.imag) > tiny:
        raise ValueError("Expected a real value, got {}.".format(value))
    return float(value.real)


def clean_variance(value, tiny: float = 1.0e-9) -> float:
    value = complex(value)
    if abs(value.imag) > tiny:
        raise ValueError("Expected a real variance, got {}.".format(value))
    real_value = float(value.real)
    if real_value < 0.0 and abs(real_value) <= tiny:
        return 0.0
    if real_value < 0.0:
        raise ValueError("Computed a negative variance: {}.".format(real_value))
    return real_value


def pauli_word(pauli_tuple) -> str:
    if not pauli_tuple:
        return "I"
    return " ".join("{}{}".format(pauli, qubit) for qubit, pauli in pauli_tuple)


def make_terms(qubit_operator, n_qubits: int) -> list[PauliTerm]:
    """Create a stable indexed representation, with identity first."""

    source_items = list(qubit_operator.terms.items())
    source_order = {
        pauli_tuple: position
        for position, (pauli_tuple, _) in enumerate(source_items)
    }
    items = sorted(source_items, key=lambda item: (bool(item[0]), item[0]))
    terms = []
    for index, (pauli_tuple_value, coefficient) in enumerate(items):
        pauli_tuple_value = tuple(
            (int(qubit), str(pauli)) for qubit, pauli in pauli_tuple_value
        )
        pauli_by_qubit = dict(pauli_tuple_value)
        terms.append(
            PauliTerm(
                index=index,
                pauli_tuple=pauli_tuple_value,
                ops=tuple(
                    pauli_by_qubit.get(qubit, "I") for qubit in range(n_qubits)
                ),
                coefficient=clean_complex(coefficient),
                word=pauli_word(pauli_tuple_value),
                source_order=source_order[pauli_tuple_value],
            )
        )
    return terms


def terms_fully_commute(term1: PauliTerm, term2: PauliTerm) -> bool:
    anticommutes = sum(
        op1 != "I" and op2 != "I" and op1 != op2
        for op1, op2 in zip(term1.ops, term2.ops)
    )
    return anticommutes % 2 == 0


def tequila_wavefunction_from_array(state_vector):
    return tq.QubitWaveFunction.from_array(np.asarray(state_vector, dtype=complex))


def pauli_hamiltonian_for_term(term: PauliTerm):
    return QubitHamiltonian.from_openfermion(
        QubitOperator(term.pauli_tuple, 1.0)
    )


def wavefunction_array(wfn, dimension: int) -> np.ndarray:
    array = np.asarray(wfn.to_array(), dtype=complex).reshape(-1)
    if array.size != dimension:
        raise ValueError(
            "Expected wavefunction array of size {}, got {}.".format(
                dimension, array.size
            )
        )
    return array


def action_row_for_term(
    term: PauliTerm, reference_wfn, dimension: int
) -> np.ndarray:
    return wavefunction_array(
        pauli_hamiltonian_for_term(term)(reference_wfn), dimension
    )


def _init_action_worker(state_vector, n_qubits, terms):
    global _ACTION_STATE, _ACTION_N_QUBITS, _ACTION_TERMS
    _ACTION_STATE = tequila_wavefunction_from_array(state_vector)
    _ACTION_N_QUBITS = int(n_qubits)
    _ACTION_TERMS = list(terms)


def _action_rows_chunk(term_positions):
    dimension = 2**_ACTION_N_QUBITS
    return [
        (
            position,
            action_row_for_term(
                _ACTION_TERMS[position], _ACTION_STATE, dimension
            ),
        )
        for position in term_positions
    ]


def iter_index_chunks(n_items: int, chunksize: int):
    for start in range(0, n_items, chunksize):
        yield list(range(start, min(start + chunksize, n_items)))


def action_matrix_gib(n_terms: int, n_qubits: int) -> float:
    """Memory occupied by the complex128 Pauli-action matrix alone."""

    return n_terms * (2**n_qubits) * np.dtype(complex).itemsize / (1024**3)


def covariance_workspace_gib(n_terms: int, n_qubits: int) -> float:
    """Conservative core workspace for actions, conjugate copy, and Gram."""

    action_gib = action_matrix_gib(n_terms, n_qubits)
    gram_gib = n_terms * n_terms * np.dtype(complex).itemsize / (1024**3)
    return 2.0 * action_gib + gram_gib


def build_action_matrix(
    terms: list[PauliTerm],
    state_vector,
    n_qubits: int,
    max_workers: int = 1,
    chunksize: int = DEFAULT_COVARIANCE_CHUNKSIZE,
) -> np.ndarray:
    dimension = 2**n_qubits
    state_vector = np.asarray(state_vector, dtype=complex).reshape(-1)
    if state_vector.size != dimension:
        raise ValueError(
            "Expected wavefunction size {}, got {}.".format(
                dimension, state_vector.size
            )
        )
    if max_workers < 1:
        raise ValueError("max_workers must be at least one.")
    if chunksize < 1:
        raise ValueError("chunksize must be at least one.")

    actions = np.empty((len(terms), dimension), dtype=complex)
    if max_workers == 1:
        reference_wfn = tequila_wavefunction_from_array(state_vector)
        for position, term in enumerate(terms):
            actions[position] = action_row_for_term(
                term, reference_wfn, dimension
            )
        return actions

    automatic_chunksize = max(1, math.ceil(len(terms) / (4 * max_workers)))
    task_chunksize = min(chunksize, automatic_chunksize)
    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_init_action_worker,
        initargs=(state_vector, n_qubits, terms),
    ) as executor:
        chunks = iter_index_chunks(len(terms), task_chunksize)
        for chunk_rows in executor.map(_action_rows_chunk, chunks):
            for position, row in chunk_rows:
                actions[position] = row
    return actions


def build_covariance_dictionary(
    terms: list[PauliTerm],
    state_vector,
    n_qubits: int,
    max_workers: int = 1,
    chunksize: int = DEFAULT_COVARIANCE_CHUNKSIZE,
):
    """Build all fully commuting covariances from one action matrix.

    Pass the complete term list, including identity, when the result may later
    be adapted to Tequila grouping routines.  The binary optimizer itself
    filters identity from its assignment variables.
    """

    state_vector = np.asarray(state_vector, dtype=complex).reshape(-1)
    actions = build_action_matrix(
        terms, state_vector, n_qubits, max_workers, chunksize
    )
    single_values = actions.dot(state_vector.conjugate())
    gram = actions.conjugate().dot(actions.T)
    single_expectations = {
        term.index: clean_complex(single_values[position])
        for position, term in enumerate(terms)
    }

    covariances = {}
    for left_position, left in enumerate(terms):
        for right_position in range(left_position, len(terms)):
            right = terms[right_position]
            if not terms_fully_commute(left, right):
                continue
            covariance = clean_complex(
                gram[left_position, right_position]
                - single_expectations[left.index]
                * single_expectations[right.index]
            )
            covariances[(left.index, right.index)] = covariance
    return covariances, single_expectations


def get_covariance(
    term1: PauliTerm, term2: PauliTerm, covariances: dict
) -> complex:
    key = tuple(sorted((term1.index, term2.index)))
    return covariances[key]


def hamiltonian_expectation(
    terms: list[PauliTerm], single_expectations: dict
) -> float:
    value = 0.0 + 0.0j
    for term in terms:
        if term.pauli_tuple:
            value += term.coefficient * single_expectations[term.index]
        else:
            value += term.coefficient
    return clean_real(value, tiny=1.0e-7)


def normalize_wfn_method(method: str) -> str:
    normalized = str(method).upper()
    aliases = {"FULLCI": "FCI", "FULL-CI": "FCI", "CI-SD": "CISD"}
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"FCI", "HF", "CISD"}:
        raise ValueError("Use FCI, HF, or CISD, not {!r}.".format(method))
    return normalized


def _bitstate_to_list(bitstate):
    if hasattr(bitstate, "array"):
        bitstate = bitstate.array
    elif hasattr(bitstate, "to_array"):
        bitstate = bitstate.to_array()
    elif hasattr(bitstate, "binary"):
        return [int(character) for character in str(bitstate.binary)]
    if hasattr(bitstate, "tolist"):
        bitstate = bitstate.tolist()
    return [int(value) for value in bitstate]


def _occupations_to_sparse_basis_index(occupations) -> int:
    """Index in OpenFermion's big-endian sparse-operator convention."""

    index = 0
    n_qubits = len(occupations)
    for qubit, occupation in enumerate(occupations):
        if int(occupation):
            index |= 1 << (n_qubits - 1 - qubit)
    return index


def _alpha_beta_reordering_phase(alpha_occ, beta_occ, ordering: str) -> float:
    if ordering == "blocked":
        return 1.0
    if ordering != "interleaved":
        raise ValueError("Unknown spin-orbital ordering {!r}.".format(ordering))
    inversions = sum(
        1
        for beta_orbital in beta_occ
        for alpha_orbital in alpha_occ
        if alpha_orbital > beta_orbital
    )
    return -1.0 if inversions % 2 else 1.0


def _map_spin_state_to_qubits(molecule, spin_occupations):
    if hasattr(molecule, "transformation") and hasattr(
        molecule.transformation, "map_state"
    ):
        mapped = molecule.transformation.map_state(state=spin_occupations)
        return _bitstate_to_list(mapped)
    return _bitstate_to_list(spin_occupations)


def _build_spin_occupation(
    alpha_occ, beta_occ, n_spatial_orbitals: int, ordering: str
):
    occupations = [0] * (2 * n_spatial_orbitals)
    if ordering == "interleaved":
        for orbital in alpha_occ:
            occupations[2 * orbital] = 1
        for orbital in beta_occ:
            occupations[2 * orbital + 1] = 1
    elif ordering == "blocked":
        for orbital in alpha_occ:
            occupations[orbital] = 1
        for orbital in beta_occ:
            occupations[n_spatial_orbitals + orbital] = 1
    else:
        raise ValueError("Unknown spin-orbital ordering {!r}.".format(ordering))
    return occupations


def _infer_spin_orbital_ordering(
    reference_spin_state, n_spatial_orbitals: int, n_alpha: int, n_beta: int
) -> str:
    alpha_occ = list(range(n_alpha))
    beta_occ = list(range(n_beta))
    for ordering in ("interleaved", "blocked"):
        expected = _build_spin_occupation(
            alpha_occ, beta_occ, n_spatial_orbitals, ordering
        )
        if expected == reference_spin_state:
            return ordering
    raise RuntimeError(
        "Could not infer interleaved or blocked spin-orbital ordering."
    )


def get_hf_qubit_statevector(molecule) -> np.ndarray:
    if not hasattr(molecule, "_reference_state"):
        raise RuntimeError("The Tequila molecule has no _reference_state().")
    reference_spin_state = _bitstate_to_list(molecule._reference_state())
    reference_qubit_state = _map_spin_state_to_qubits(
        molecule, reference_spin_state
    )
    state = np.zeros(2 ** len(reference_qubit_state), dtype=complex)
    # OpenFermion's sparse matrices use the reversed bit significance.  Using
    # a little-endian index here gives an incorrect HF energy for H2/H4.
    state[_occupations_to_sparse_basis_index(reference_qubit_state)] = 1.0
    return state


def get_cisd_qubit_statevector(molecule, tiny: float = 1.0e-12) -> np.ndarray:
    try:
        from pyscf.fci import cistring
        from tequila.quantumchemistry.pyscf_interface import QuantumChemistryPySCF
    except Exception as exc:  # pragma: no cover - dependency error path
        raise RuntimeError(
            "CISD construction requires Tequila's PySCF backend and PySCF."
        ) from exc

    if not hasattr(molecule, "_reference_state"):
        raise RuntimeError("The Tequila molecule has no _reference_state().")
    quantum_chemistry = QuantumChemistryPySCF.from_tequila(molecule)
    mean_field = quantum_chemistry._get_hf()
    cisd_solver = quantum_chemistry._run_cisd()
    if not hasattr(cisd_solver, "ci"):
        raise RuntimeError("The PySCF CISD solver exposes no CI coefficients.")
    try:
        fci_vector = cisd_solver.to_fcivec()
    except TypeError:
        fci_vector = cisd_solver.to_fcivec(cisd_solver.ci)

    reference_spin_state = _bitstate_to_list(molecule._reference_state())
    n_spatial_orbitals = mean_field.mo_coeff.shape[1]
    electrons = mean_field.mol.nelec
    if isinstance(electrons, (tuple, list)):
        n_alpha, n_beta = int(electrons[0]), int(electrons[1])
    else:
        n_alpha = int(electrons) // 2
        n_beta = int(electrons) - n_alpha
    if len(reference_spin_state) != 2 * n_spatial_orbitals:
        raise NotImplementedError(
            "CISD state construction assumes no qubit tapering."
        )
    ordering = _infer_spin_orbital_ordering(
        reference_spin_state, n_spatial_orbitals, n_alpha, n_beta
    )
    alpha_strings = cistring.make_strings(range(n_spatial_orbitals), n_alpha)
    beta_strings = cistring.make_strings(range(n_spatial_orbitals), n_beta)
    fci_vector = np.asarray(fci_vector).reshape(
        len(alpha_strings), len(beta_strings)
    )
    reference_qubit_state = _map_spin_state_to_qubits(
        molecule, reference_spin_state
    )
    state = np.zeros(2 ** len(reference_qubit_state), dtype=complex)
    for alpha_index, alpha_string in enumerate(alpha_strings):
        alpha_occ = [
            orbital
            for orbital in range(n_spatial_orbitals)
            if (int(alpha_string) >> orbital) & 1
        ]
        for beta_index, beta_string in enumerate(beta_strings):
            coefficient = fci_vector[alpha_index, beta_index]
            if abs(coefficient) < tiny:
                continue
            beta_occ = [
                orbital
                for orbital in range(n_spatial_orbitals)
                if (int(beta_string) >> orbital) & 1
            ]
            spin_occ = _build_spin_occupation(
                alpha_occ, beta_occ, n_spatial_orbitals, ordering
            )
            qubit_occ = _map_spin_state_to_qubits(molecule, spin_occ)
            phase = _alpha_beta_reordering_phase(
                alpha_occ, beta_occ, ordering
            )
            state[_occupations_to_sparse_basis_index(qubit_occ)] += (
                phase * coefficient
            )
    norm = np.linalg.norm(state)
    if norm < tiny:
        raise RuntimeError("Constructed CISD statevector has near-zero norm.")
    return state / norm


def statevector_expectation_value(sparse_operator, statevector):
    return np.vdot(statevector, sparse_operator.dot(statevector))


def get_variance_wavefunction(
    molecule,
    qubit_operator,
    method: str = "FCI",
    sparse_hamiltonian=None,
):
    """Return ``(energy, statevector)`` for FCI, CISD, or HF covariance."""

    method = normalize_wfn_method(method)
    if sparse_hamiltonian is None:
        sparse_hamiltonian = get_sparse_operator(qubit_operator)
    if method == "FCI":
        return get_ground_state(sparse_hamiltonian)
    if method == "HF":
        state = get_hf_qubit_statevector(molecule)
    else:
        state = get_cisd_qubit_statevector(molecule)
    energy = statevector_expectation_value(sparse_hamiltonian, state)
    return energy, state


__all__ = [
    "DEFAULT_COVARIANCE_CHUNKSIZE",
    "PauliTerm",
    "action_matrix_gib",
    "build_action_matrix",
    "build_covariance_dictionary",
    "clean_complex",
    "clean_real",
    "clean_variance",
    "default_cov_workers",
    "get_covariance",
    "get_variance_wavefunction",
    "hamiltonian_expectation",
    "make_terms",
    "normalize_wfn_method",
    "terms_fully_commute",
]
