"""STO-3G Jordan--Wigner molecular Hamiltonians used in GFlow-VQE. Extend later.

Each helper returns the same five-tuple as we do in GFlow-VQE module:
``(molecule, tequila_hamiltonian, fermion_hamiltonian, n_paulis,
qubit_operator)``.
"""

import math

import tequila as tq
from openfermion.transforms import reverse_jordan_wigner


def _molecule(geometry: str, *, frozen_core: bool = True):
    molecule = tq.chemistry.Molecule(
        geometry=geometry,
        basis_set="sto3g",
        transformation="JordanWigner",
        backend="pyscf",
        frozen_core=frozen_core,
    )
    hamiltonian = molecule.make_hamiltonian()
    qubit_operator = hamiltonian.to_openfermion()
    fermion_hamiltonian = reverse_jordan_wigner(qubit_operator)
    n_measurable = sum(bool(pauli_word) for pauli_word in qubit_operator.terms)
    return (
        molecule,
        hamiltonian,
        fermion_hamiltonian,
        n_measurable,
        qubit_operator,
    )


def H2(*, frozen_core: bool = True):
    return _molecule("H 0.0 0.0 0.0\nH 0.0 0.0 1.0", frozen_core=frozen_core)


def LiH(*, frozen_core: bool = True):
    return _molecule("Li 0.0 0.0 0.0\nH 0.0 0.0 1.0", frozen_core=frozen_core)


def MgO(*, frozen_core: bool = True):
    return _molecule("Mg 0.0 0.0 0.0\nO 0.0 0.0 1.75", frozen_core=frozen_core)


def N2(*, frozen_core: bool = True):
    return _molecule("N 0.0 0.0 0.0\nN 0.0 0.0 1.0", frozen_core=frozen_core)


def H4(*, frozen_core: bool = True):
    return _molecule(
        "H 0.0 0.0 0.0\n"
        "H 0.0 0.0 1.0\n"
        "H 0.0 0.0 2.0\n"
        "H 0.0 0.0 3.0",
        frozen_core=frozen_core,
    )


def H6(*, frozen_core: bool = True):
    return _molecule(
        "H 0.0 0.0 0.0\n"
        "H 0.0 0.0 1.0\n"
        "H 0.0 0.0 2.0\n"
        "H 0.0 0.0 3.0\n"
        "H 0.0 0.0 4.0\n"
        "H 0.0 0.0 5.0",
        frozen_core=frozen_core,
    )


def BeH2(*, frozen_core: bool = True):
    return _molecule("Be 0.0 0.0 0.0\nH 0.0 0.0 1.0\nH 0.0 0.0 -1.0", frozen_core=frozen_core)


def H2O(*, frozen_core: bool = True):
    radius = 1.0
    theta = math.radians(107.6)
    x_coord = radius * math.sin(theta / 2.0)
    z_coord = radius * math.cos(theta / 2.0)
    return _molecule(
        (
            "O 0.0 0.0 0.0\n"
            "H {:.10f} 0.0 {:.10f}\n"
            "H {:.10f} 0.0 {:.10f}"
        ).format(x_coord, z_coord, -x_coord, z_coord),
        frozen_core=frozen_core,
    )


def H2Os(*, frozen_core: bool = True):
    radius = 1.5
    theta = math.radians(107.6)
    x_coord = radius * math.sin(theta / 2.0)
    z_coord = radius * math.cos(theta / 2.0)
    return _molecule(
        (
            "O 0.0 0.0 0.0\n"
            "H {:.10f} 0.0 {:.10f}\n"
            "H {:.10f} 0.0 {:.10f}"
        ).format(x_coord, z_coord, -x_coord, z_coord),
        frozen_core=frozen_core,
    )


def NH3(*, frozen_core: bool = True):
    """Geometry-scan ammonia with 1 angstrom N-H bonds and 107-degree angles."""
    cosine = math.cos(math.radians(107.0))
    radius = math.sqrt(2 * (1 - cosine) / 3)
    z = math.sqrt((1 + 2 * cosine) / 3)
    atoms = [("N", 0.0, 0.0, 0.0)] + [
        ("H", radius * math.cos(phi), radius * math.sin(phi), z)
        for phi in (math.radians(angle) for angle in (0.0, 120.0, 240.0))
    ]
    geometry = "\n".join(
        f"{atom} {x:.12f} {y:.12f} {z:.12f}" for atom, x, y, z in atoms
    )
    return _molecule(geometry, frozen_core=frozen_core)


MOLECULES = {
    function.__name__: function
    for function in (H2, LiH, MgO, N2, H4, H6, BeH2, H2O, H2Os, NH3)
}


def get_molecule(name: str, *, frozen_core: bool = True):
    """Build a supported molecule; set frozen_core=False to include all electrons."""

    normalized = str(name).casefold()
    for canonical_name, function in MOLECULES.items():
        if canonical_name.casefold() == normalized:
            return canonical_name, function(frozen_core=frozen_core)
    choices = ", ".join(MOLECULES)
    raise ValueError("Unknown molecule {!r}. Choose one of: {}.".format(name, choices))


__all__ = [*MOLECULES, "MOLECULES", "get_molecule"]
