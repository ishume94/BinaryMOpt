"""Open-boundary lattice Hamiltonians with at least four qubits for BinMOpt.

Spin models use Pauli matrices (S = sigma/2 is not used). Fermionic models
are spinful, with orbitals (site 0 up, site 0 down, site 1 up, ...) and a
Jordan--Wigner mapping. Haldane and Emery follow the equations in
https://pennylane.ai/demos/tutorial_how_to_build_spin_hamiltonians : Haldane
uses +phi on c_i^dagger c_j for each next-neighbor pair i < j; Emery is the
uniform t-U-V model in that tutorial, without additional orbital parameters.
"""

from dataclasses import dataclass, field
from math import ceil, isqrt, sqrt
from pathlib import Path

import numpy as np
from openfermion import FermionOperator, QubitOperator, jordan_wigner
from tequila import QubitHamiltonian


MODELS = ("tfim", "heisenberg", "xxz", "xyz", "kitaev", "fermi-hubbard", "haldane", "emery")
FERMION_MODELS = frozenset(("fermi-hubbard", "haldane", "emery"))
MODEL_NAMES = {
    "tfim": "Transverse-field Ising", "heisenberg": "Heisenberg",
    "xxz": "XXZ", "xyz": "XYZ", "kitaev": "Kitaev honeycomb",
    "fermi-hubbard": "Fermi-Hubbard", "haldane": "Haldane (tutorial convention)",
    "emery": "Emery (tutorial t-U-V model)",
}
DEFAULT_PARAMETERS = {
    "tfim": {"j": 1.0, "h": 1.0},
    "heisenberg": {"j": 1.0, "j2": 0.0, "hx": 0.0, "hy": 0.0, "hz": 0.0},
    "xxz": {"j": 1.0, "delta": 1.5, "j2": 0.0, "hx": 0.0, "hy": 0.0, "hz": 0.0},
    "xyz": {"jx": 1.0, "jy": 0.8, "jz": 1.2, "j2": 0.0, "hx": 0.0, "hy": 0.0, "hz": 0.0},
    "kitaev": {"kx": 1.0, "ky": 1.0, "kz": 1.0, "hx": 0.0, "hy": 0.0, "hz": 0.0,
               "j": 0.0, "gamma": 0.0},
    "fermi-hubbard": {"t": 1.0, "u": 4.0},
    "haldane": {"t": 1.0, "t2": 0.2, "phi": float(np.pi / 2)},
    "emery": {"t": 1.0, "u": 4.0, "v": 1.0},
}
EQUATIONS = {
    "tfim": r"H=-J\sum_{(i,j)\in E}Z_iZ_j-h\sum_i X_i",
    "heisenberg": r"H=J\sum_{(i,j)\in E}(X_iX_j+Y_iY_j+Z_iZ_j)",
    "xxz": r"H=J\sum_{(i,j)\in E}(X_iX_j+Y_iY_j+\Delta Z_iZ_j)",
    "xyz": r"H=\sum_{(i,j)\in E}(J_xX_iX_j+J_yY_iY_j+J_zZ_iZ_j)",
    "kitaev": r"H=\sum_{a\in\{x,y,z\}}K_a\sum_{(i,j)\in E_a}\sigma_i^a\sigma_j^a",
    "fermi-hubbard": r"H=-t\sum_{(i,j)\in E,s}(c_{is}^{\dagger}c_{js}+\mathrm{h.c.})"
                      r"+U\sum_i n_{i\uparrow}n_{i\downarrow}",
    "haldane": r"H=-t\sum_{(i,j)\in E,s}(c_{is}^{\dagger}c_{js}+\mathrm{h.c.})"
               r"-t_2\sum_{(i,j)\in E_2,s}(e^{i\phi}c_{is}^{\dagger}c_{js}+\mathrm{h.c.}),\quad i<j",
    "emery": r"H=-t\sum_{(i,j)\in E,s}(c_{is}^{\dagger}c_{js}+\mathrm{h.c.})"
             r"+U\sum_i n_{i\uparrow}n_{i\downarrow}+V\sum_{(i,j)\in E}n_i n_j",
}


def model_name(value):
    name = "-".join(str(value).strip().lower().replace("_", "-").split())
    aliases = {"ising": "tfim", "transverse-ising": "tfim",
               "transverse-field-ising": "tfim", "kitaev-honeycomb": "kitaev",
               "hubbard": "fermi-hubbard"}
    name = aliases.get(name, name)
    if name not in MODELS:
        raise ValueError("Model must be one of: " + ", ".join(MODELS))
    return name


@dataclass
class LatticeHamiltonian:
    model: str
    n_qubits: int
    lattice: str
    shape: tuple[int, int]
    positions: np.ndarray
    bonds: tuple[tuple[int, int, str], ...]
    parameters: dict[str, float]
    qubit_operator: QubitOperator
    fermion_operator: FermionOperator | None
    notes: tuple[str, ...]
    disorder: dict = field(default_factory=dict)

    @property
    def n_sites(self):
        return len(self.positions)

    @property
    def tequila_hamiltonian(self):
        return QubitHamiltonian.from_openfermion(self.qubit_operator)

    @property
    def equation_lines(self):
        lines = [EQUATIONS[self.model]]
        if self.model == "kitaev":
            if self.parameters["j"]:
                lines.append(r"+J\sum_{(i,j)\in E}(X_iX_j+Y_iY_j+Z_iZ_j)")
            if self.parameters["gamma"]:
                lines.append(r"+\Gamma\sum_a\sum_{(i,j)\in E_a}"
                             r"(\sigma_i^b\sigma_j^c+\sigma_i^c\sigma_j^b),\quad \{a,b,c\}=\{x,y,z\}")
        if self.parameters.get("j2", 0):
            lines.append(r"+J_2\sum_{(i,j)\in E_2}(X_iX_j+Y_iY_j+Z_iZ_j)")
        if self.disorder:
            for shell in ("E", "E_a", "E_2"):
                lines = [line.replace(r"\sum_{(i,j)\in " + shell + "}",
                                      r"\sum_{(i,j)\in " + shell + r"}g_{ij}") for line in lines]
            if self.model == "tfim":
                lines[0] = lines[0].replace(r"-h\sum_i X_i", r"-\sum_i(h+\eta_i^x)X_i")
            else:
                lines.append(r"+\sum_i\sum_{a\in\{x,y,z\}}(h_a+\eta_i^a)\sigma_i^a")
            lines.append(r"g_{ij}=1+W_b u_{ij},\quad \eta_i^a=W_f v_i^a,\quad u_{ij},v_i^a\sim U[-1,1]")
            return lines
        fields = "".join(rf"+h_{a}\sum_i {a.upper()}_i" for a in "xyz" if self.parameters.get("h" + a, 0))
        if fields:
            lines.append(fields)
        return lines

    @property
    def equation(self):
        return "".join(self.equation_lines)

    def metadata(self):
        return {"model": self.model, "n_qubits": self.n_qubits,
                "n_sites": self.n_sites, "lattice": self.lattice,
                "shape": list(self.shape), "boundary": "open",
                "positions": self.positions.tolist(), "bonds": list(self.bonds),
                "parameters": dict(self.parameters), "equation": self.equation,
                "disorder": dict(self.disorder),
                "mapping": "Jordan-Wigner" if self.model in FERMION_MODELS else "direct Pauli",
                "notes": list(self.notes)}


def _geometry(lattice, n_sites, shape):
    """Make full cells plus, when needed, a connected last-cell fragment.

    Index cells with x outermost and y innermost. Honeycomb basis/vectors
    match the tutorial; its intracell bonds are X, +y bonds Y, +x bonds Z.
    Neighbor shells use bulk distances, never the shortest distance remaining
    in a small cluster (which could misidentify a missing shell).
    """
    basis_size = {"chain": 1, "rectangle": 1, "square": 1, "honeycomb": 2, "lieb": 3}[lattice]
    n_cells = ceil(n_sites / basis_size)
    if shape is None:
        ny = 1 if lattice == "chain" else isqrt(n_cells)
        while n_cells % ny:
            ny -= 1
        shape = (n_cells // ny, ny)
    if (len(shape) != 2 or any(isinstance(v, bool) or int(v) != v or v < 1 for v in shape)
            or int(shape[0]) * int(shape[1]) != n_cells):
        raise ValueError(f"--shape NX NY must contain exactly {n_cells} cells for this model/qubit count.")
    nx, ny = (int(v) for v in shape)
    if lattice == "square" and nx != ny:
        raise ValueError("A square lattice requires a square site count and NX = NY (e.g. 4, 9, 16 spin qubits).")
    if lattice == "chain" and ny != 1:
        raise ValueError("A chain requires --shape N_SITES 1.")
    vectors = np.array([[1., 0.], [0., 1.]])
    basis = np.array([[0., 0.]])
    distance_sq = 1.0
    if lattice == "honeycomb":
        vectors = np.array([[1., 0.], [.5, sqrt(3) / 2]])
        basis = np.array([[0., 0.], [.5, 1 / (2 * sqrt(3))]])
        distance_sq = 1 / 3
    elif lattice == "lieb":
        basis = np.array([[0., 0.], [.5, 0.], [0., .5]])
        distance_sq = .25
    positions = np.array([x * vectors[0] + y * vectors[1] + point
                          for x in range(nx) for y in range(ny) for point in basis])[:n_sites]
    bonds = []
    for i in range(n_sites):
        for j in range(i + 1, n_sites):
            displacement = positions[j] - positions[i]
            if not np.isclose(displacement @ displacement, distance_sq, atol=1.e-10, rtol=0):
                continue
            kind = "NN"
            if lattice == "honeycomb":
                cell_i, cell_j = i // 2, j // 2
                kind = "X" if cell_i == cell_j else "Y" if cell_j - cell_i == 1 and cell_i // ny == cell_j // ny else "Z"
            bonds.append((i, j, kind))
    notes = []
    if n_sites % basis_size:
        notes.append(f"Open {lattice} fragment: {n_sites} sites from {nx} x {ny} parent cells; last cell incomplete.")
    return (nx, ny), positions, bonds, notes


def build_lattice(model, n_qubits, *, lattice=None, shape=None,
                  bond_disorder=0.0, field_disorder=0.0, disorder_seed=0, **parameters):
    """Build one model with the exact requested register size, without padding.

    Defaults: chains for ordinary spin models, compact rectangles for Hubbard,
    honeycomb for Kitaev/Haldane, Lieb for Emery. All boundaries are open.
    Spin models accept integers >= 4; spinful fermion models require an even
    count. Shape counts unit cells, including an incomplete final cell.
    Spin-only disorder multiplies each bond's entire exchange tensor by
    1 + bond_disorder * U[-1,1] and adds field_disorder * U[-1,1] to each
    local field component (only X for TFIM). Independent PCG64 streams for
    bonds and fields are initialized from disorder_seed, not the state seed.
    Kitaev also accepts nearest-neighbor Heisenberg J and symmetric Gamma
    exchange between the other two axes on each bond. All use Pauli units.
    Other Hamiltonian parameters must be in DEFAULT_PARAMETERS[model].
    """
    model = model_name(model)
    if isinstance(n_qubits, bool) or int(n_qubits) != n_qubits or n_qubits < 4:
        raise ValueError("The number of qubits must be an integer of at least four.")
    n_qubits = int(n_qubits)
    fermionic = model in FERMION_MODELS
    bond_disorder, field_disorder = float(bond_disorder), float(field_disorder)
    if any(not np.isfinite(w) or w < 0 for w in (bond_disorder, field_disorder)):
        raise ValueError("Disorder strengths must be finite and nonnegative.")
    if isinstance(disorder_seed, bool) or int(disorder_seed) != disorder_seed or disorder_seed < 0:
        raise ValueError("The disorder seed must be a nonnegative integer.")
    disorder_seed = int(disorder_seed)
    if fermionic and (bond_disorder or field_disorder or disorder_seed):
        raise ValueError("Bond and field disorder are implemented for direct spin models only.")
    if fermionic and n_qubits % 2:
        raise ValueError("Spinful fermionic models require an even number of qubits (two per site).")
    n_sites = n_qubits // 2 if fermionic else n_qubits
    defaults = {"kitaev": "honeycomb", "haldane": "honeycomb", "emery": "lieb", "fermi-hubbard": "rectangle"}
    lattice = defaults.get(model, "chain") if lattice is None else str(lattice).lower()
    allowed = {"kitaev": {"honeycomb"}, "fermi-hubbard": {"rectangle"},
               "emery": {"lieb", "rectangle", "chain"}}.get(model, {"rectangle", "square", "chain", "honeycomb"})
    if lattice not in allowed:
        raise ValueError(f"{model} supports lattices: {', '.join(sorted(allowed))}.")
    params = DEFAULT_PARAMETERS[model].copy()
    if parameters.keys() - params.keys():
        raise ValueError(f"Parameters for {model} are: {', '.join(params)}.")
    params.update({key: float(value) for key, value in parameters.items()})
    if not all(np.isfinite(value) for value in params.values()):
        raise ValueError("All model parameters must be finite real numbers.")
    shape, positions, bonds, notes = _geometry(lattice, n_sites, shape)
    if model == "haldane" or params.get("j2", 0):
        next_distance_sq = {"honeycomb": 1., "rectangle": 2., "square": 2., "chain": 4.}[lattice]
        for i in range(n_sites):
            for j in range(i + 1, n_sites):
                delta = positions[j] - positions[i]
                if np.isclose(delta @ delta, next_distance_sq, atol=1.e-10, rtol=0):
                    bonds.append((i, j, "NNN"))
    if model == "haldane":
        notes.append("Next-neighbor convention: i < j, coefficient -t2 exp(+i phi) on c_i^dagger c_j.")
        if not any(kind == "NNN" for _, _, kind in bonds):
            notes.append("This cluster has no next-neighbor bonds; t2 and phi do not contribute.")
    if model == "kitaev":
        missing = sorted(set("XYZ") - {kind for _, _, kind in bonds})
        if missing:
            notes.append("This open cluster has no " + "/".join(missing) + " bonds.")

    disorder = {}
    bond_factors = np.ones(len(bonds))
    field_axes = "x" if model == "tfim" else "xyz"
    field_offsets = np.zeros((n_sites, len(field_axes)))
    if bond_disorder or field_disorder:
        bond_seed, field_seed = np.random.SeedSequence(disorder_seed).spawn(2)
        bond_rng = np.random.Generator(np.random.PCG64(bond_seed))
        field_rng = np.random.Generator(np.random.PCG64(field_seed))
        bond_factors += bond_disorder * bond_rng.uniform(-1., 1., len(bonds))
        field_offsets += field_disorder * field_rng.uniform(-1., 1., field_offsets.shape)
        disorder = {"bond_strength": bond_disorder, "field_strength": field_disorder,
                    "seed": disorder_seed, "generator": "PCG64 / SeedSequence.spawn(2)",
                    "bond_factors": bond_factors.tolist(), "field_axes": list(field_axes),
                    "field_offsets": field_offsets.tolist()}
        notes.append("Bond width scales with |g_ij|; realized bond factors and field offsets are saved in JSON.")

    fermion_operator = None
    operator = QubitOperator()
    if not fermionic:
        if model == "tfim":
            for (i, j, _), factor in zip(bonds, bond_factors):
                operator += QubitOperator(((i, "Z"), (j, "Z")), -params["j"] * factor)
            for i in range(n_sites):
                operator += QubitOperator(((i, "X"),), -(params["h"] + field_offsets[i, 0]))
        else:
            if model == "heisenberg":
                couplings = {a: params["j"] for a in "XYZ"}
            elif model == "xxz":
                couplings = dict(X=params["j"], Y=params["j"], Z=params["j"] * params["delta"])
            else:
                prefix = "k" if model == "kitaev" else "j"
                couplings = {a: params[prefix + a.lower()] for a in "XYZ"}
            for (i, j, kind), factor in zip(bonds, bond_factors):
                for a in (kind if model == "kitaev" else "XYZ"):
                    coefficient = params["j2"] if kind == "NNN" else couplings[a]
                    operator += QubitOperator(((i, a), (j, a)), coefficient * factor)
                if model == "kitaev":
                    if params["j"]:
                        for a in "XYZ":
                            operator += QubitOperator(((i, a), (j, a)), params["j"] * factor)
                    if params["gamma"]:
                        b, c = (a for a in "XYZ" if a != kind)
                        operator += QubitOperator(((i, b), (j, c)), params["gamma"] * factor)
                        operator += QubitOperator(((i, c), (j, b)), params["gamma"] * factor)
            for axis, a in enumerate("xyz"):
                for i in range(n_sites):
                    operator += QubitOperator(((i, a.upper()),), params["h" + a] + field_offsets[i, axis])
    else:
        fermion_operator = FermionOperator()
        numbers = [FermionOperator(((q, 1), (q, 0))) for q in range(n_qubits)]
        for i, j, kind in bonds:
            hopping = params["t2"] * np.exp(1j * params["phi"]) if kind == "NNN" else params["t"]
            for spin in range(2):
                p, q = 2 * i + spin, 2 * j + spin
                fermion_operator += FermionOperator(((p, 1), (q, 0)), -hopping)
                fermion_operator += FermionOperator(((q, 1), (p, 0)), -np.conjugate(hopping))
            if model == "emery":
                fermion_operator += params["v"] * (numbers[2*i] + numbers[2*i+1]) * (numbers[2*j] + numbers[2*j+1])
        if "u" in params:
            for i in range(n_sites):
                fermion_operator += params["u"] * numbers[2*i] * numbers[2*i+1]
        operator = jordan_wigner(fermion_operator)
    operator.compress(abs_tol=1.e-12)
    if any(abs(complex(c).imag) > 1.e-10 for c in operator.terms.values()):
        raise ValueError("The constructed Hamiltonian has non-real Pauli coefficients.")
    if not any(term for term in operator.terms):
        raise ValueError("The chosen parameters give no nonidentity Hamiltonian terms.")
    return LatticeHamiltonian(model, n_qubits, lattice, shape, positions, tuple(bonds),
                              params, operator, fermion_operator, tuple(notes), disorder)


def plot_lattice(model, output_stem):
    """Save PNG and SVG with site indices, actual bonds, equation and parameters."""
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.lines import Line2D
    from matplotlib.patches import FancyArrowPatch

    equations = model.equation_lines
    height = max(8., 6.8 + .5 * len(equations))
    fig = Figure(figsize=(12, height))
    FigureCanvasAgg(fig)
    equation_top = (2.0 + .5 * (len(equations) - 1)) / height
    graph_bottom = (2.0 + .5 * len(equations) + .25) / height
    ax = fig.add_axes((.07, graph_bottom, .86, 1 - 1.25 / height - graph_bottom))
    colors = {"NN": "#52677D", "X": "#D55E00", "Y": "#009E73", "Z": "#0072B2", "NNN": "#9467BD"}
    kinds = []
    factors = model.disorder.get("bond_factors", [1.] * len(model.bonds))
    for (i, j, kind), factor in zip(model.bonds, factors):
        display_kind = kind if model.model == "kitaev" or kind == "NNN" else "NN"
        if display_kind not in kinds:
            kinds.append(display_kind)
        p, q = model.positions[[i, j]]
        if display_kind == "NNN" and model.model == "haldane":
            ax.add_patch(FancyArrowPatch(q, p, arrowstyle="-|>", mutation_scale=12,
                                        shrinkA=12, shrinkB=12, linewidth=1.5,
                                        linestyle="--", color=colors[display_kind], zorder=1))
        else:
            ax.plot([p[0], q[0]], [p[1], q[1]], color=colors[display_kind], linewidth=2.5 * abs(factor),
                    linestyle="--" if display_kind == "NNN" else "-", zorder=1)
    ax.scatter(*model.positions.T, s=420, color="#F3E5C8", edgecolor="#263442", linewidth=1.4, zorder=2)
    for i, (x, y) in enumerate(model.positions):
        ax.text(x, y, str(i), ha="center", va="center", fontsize=11, zorder=3)
    ax.set_aspect("equal")
    ax.margins(.20)
    ax.axis("off")
    labels = {"NN": "nearest neighbors", "NNN": "next neighbors: arrow j to i" if model.model == "haldane" else "next neighbors (J2)",
              "X": "x-type bonds", "Y": "y-type bonds", "Z": "z-type bonds"}
    handles = [Line2D([0], [0], color=colors[k], lw=2, linestyle="--" if k == "NNN" else "-", label=labels[k]) for k in kinds]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, 1.12), ncol=len(handles), frameon=False)
    title = MODEL_NAMES[model.model]
    if model.model == "kitaev" and (model.parameters["j"] or model.parameters["gamma"]):
        title = "Kitaev-Heisenberg-Gamma honeycomb"
    fig.text(.5, 1 - .28 / height, f"{title} | {model.n_qubits} qubits, {model.n_sites} sites",
             ha="center", va="top", fontsize=18)
    fig.text(.5, 1 - .7 / height, f"Open {model.lattice} | {model.shape[0]} x {model.shape[1]} parent unit cells",
             ha="center", va="top", fontsize=12)
    for row, equation in enumerate(equations):
        fig.text(.5, equation_top - .5 / height * row, "$" + equation + "$", ha="center", va="top", fontsize=12)
    symbols = {"j": "J", "h": "h", "delta": r"\Delta", "jx": "J_x", "jy": "J_y", "jz": "J_z",
               "kx": "K_x", "ky": "K_y", "kz": "K_z", "u": "U", "v": "V", "phi": r"\phi", "t2": "t_2", "t": "t",
               "j2": "J_2", "hx": "h_x", "hy": "h_y", "hz": "h_z", "gamma": r"\Gamma"}
    values = [f"{symbols[k]}={v:.6g}" for k, v in model.parameters.items()]
    if model.disorder:
        values += [f"W_b={model.disorder['bond_strength']:.6g}",
                   f"W_f={model.disorder['field_strength']:.6g}",
                   rf"s_{{\mathrm{{disorder}}}}={model.disorder['seed']}"]
    parameter_lines = ["$" + r",\quad ".join(values[i:i+7]) + "$" for i in range(0, len(values), 7)]
    fig.text(.5, 1.48 / height, "\n".join(parameter_lines), ha="center", va="top", fontsize=10, linespacing=1.5)
    convention = ("Site i has qubits 2i (up), 2i+1 (down); n_i = n_i,up + n_i,down; Jordan-Wigner mapping."
                  if model.model in FERMION_MODELS else "Site labels are qubit indices. X, Y, Z are Pauli matrices; S = sigma/2 is not used.")
    fig.text(.5, .85 / height, convention + "\n" + "\n".join(model.notes), ha="center", va="top", fontsize=9, linespacing=1.6)
    stem = Path(output_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix in (".png", ".svg"):
        path = Path(str(stem) + suffix)
        fig.savefig(path, dpi=180, bbox_inches="tight")
        paths.append(path)
    return paths
