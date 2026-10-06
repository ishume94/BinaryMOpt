"""Compare SI, SI-ICS, MILP and MILP-ICS for one exact lattice reference state.

Use the same fully commuting SI, non-overlapping MILP, and standard ICS
workflow as driver_geometry_scan.py. Save a comparison bar plot (PNG/SVG),
a lattice diagram with its equation, and the numerical results (JSON/CSV).

Examples: python -u driver_lattice.py xxz 12 --delta 1.5 --j2 0.3
          python -u driver_lattice.py heisenberg 16 --lattice square --j2 0.5
          python -u driver_lattice.py kitaev 8 --kx 1 --ky 0.8 --kz 1.2
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import json
from pathlib import Path
import time

import numpy as np
from openfermion import get_sparse_operator
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh
from tequila.grouping.binary_rep import BinaryHamiltonian
from tequila.grouping.binary_utils import sorted_insertion_grouping

from bin_mopt.lattice import DEFAULT_PARAMETERS, FERMION_MODELS, MODELS, MODEL_NAMES, build_lattice, model_name, plot_lattice
from bin_mopt.ics import optimize_from_nonoverlapping_groups
from bin_mopt.measurement import GroupingProblem, is_feasible
from bin_mopt.opt import (
    coloring_group_bound, make_optimization_context, measurement_objective,
    optimize_grouping,
)
from bin_mopt.utils import (
    clean_real, covariance_workspace_gib, hamiltonian_expectation, make_terms,
    terms_fully_commute,
)


METHODS = ("SI", "SI-ICS", "MILP", "MILP-ICS")


def _finite_float(value):
    number = float(value)
    if not np.isfinite(number):
        raise argparse.ArgumentTypeError("must be finite")
    return number


def _nonnegative_int(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return number


def _positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def _positive_float(value):
    number = _finite_float(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def _nonnegative_float(value):
    number = _finite_float(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return number


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=model_name, choices=MODELS)
    parser.add_argument("n_qubits", type=int, help="Exact qubit count, at least 4; fermionic models require an even count.")
    parser.add_argument("--lattice", choices=("chain", "square", "rectangle", "honeycomb", "lieb"),
                        help="Default: spin chains, Hubbard rectangle, Kitaev/Haldane honeycomb, Emery Lieb.")
    parser.add_argument("--shape", nargs=2, type=_positive_int, metavar=("NX", "NY"),
                        help="Parent unit cells; default is a chain or a compact factorization.")
    parameter_help = {
        "j": "Nearest-neighbor J for TFIM/Heisenberg/XXZ (default 1), or Kitaev Heisenberg exchange (default 0).",
        "h": "TFIM transverse field in -h sum X (default 1).",
        "delta": "XXZ nearest-neighbor anisotropy (default 1.5).",
        "j2": "Isotropic next-neighbor exchange for Heisenberg/XXZ/XYZ (default 0).",
        "jx": "XYZ x coupling (default 1).", "jy": "XYZ y coupling (default 0.8).",
        "jz": "XYZ z coupling (default 1.2).",
        "kx": "Kitaev XX coupling (default 1).", "ky": "Kitaev YY coupling (default 1).",
        "kz": "Kitaev ZZ coupling (default 1).",
        "gamma": "Kitaev symmetric off-diagonal exchange on each bond (default 0).",
        "hx": "Uniform +hx sum X for exchange/Kitaev models (default 0).",
        "hy": "Uniform +hy sum Y for exchange/Kitaev models (default 0).",
        "hz": "Uniform +hz sum Z for exchange/Kitaev models (default 0).",
        "t": "Fermionic nearest-neighbor hopping (default 1).",
        "u": "Hubbard/Emery on-site repulsion (default 4).",
        "v": "Emery nearest-neighbor density interaction (default 1).",
        "t2": "Haldane next-neighbor hopping (default 0.2).",
        "phi": "Haldane phase in radians (default pi/2).",
    }
    for name, help_text in parameter_help.items():
        parser.add_argument("--" + name, type=_finite_float, default=None, help=help_text)
    parser.add_argument("--bond-disorder", type=_nonnegative_float, default=0., metavar="WB",
                        help="Spin models: multiply each bond's exchange tensor by 1 + WB*Uniform[-1,1].")
    parser.add_argument("--field-disorder", type=_nonnegative_float, default=0., metavar="WF",
                        help="Spin models: add WF*Uniform[-1,1] to each local field; TFIM uses only X.")
    parser.add_argument("--disorder-seed", type=_nonnegative_int, default=0,
                        help="Bond/field realization seed, independent of --seed (default 0).")
    parser.add_argument("--particles", type=_nonnegative_int,
                        help="Fermionic particle count (default half filling, one per site).")
    parser.add_argument("--spin-up", type=_nonnegative_int,
                        help="Fermionic up-spin count (default ceil(particles/2)); down = particles - up.")
    parser.add_argument("--seed", type=_nonnegative_int, default=7,
                        help="Seed for the ground-state start and optimization (default 7).")
    parser.add_argument("--cov-workers", type=_positive_int, default=1)
    parser.add_argument("--solver-threads", type=_positive_int, default=None,
                        help="MILP solver threads (default: --cov-workers, as in the geometry scan).")
    parser.add_argument("--solver-time-limit-s", type=_positive_float, default=36000.,
                        help="Time limit for the single MILP optimization; ICS runs independently.")
    parser.add_argument("--ics-iterations", type=_positive_int, default=5,
                        help="Standard ICS iterations for each of SI and MILP (default 5).")
    parser.add_argument("--verbose", action="store_true", help="Print all groups and coefficients.")
    parser.add_argument("--output-directory", type=Path, default=Path("."))
    parser.add_argument("--build-only", action="store_true",
                        help="Save the Hamiltonian description and lattice plots, then stop before diagonalization.")
    args = parser.parse_args(argv)
    args.solver_threads = args.cov_workers if args.solver_threads is None else args.solver_threads
    args.parameters = {k: getattr(args, k) for k in parameter_help if getattr(args, k) is not None}
    if args.parameters.keys() - DEFAULT_PARAMETERS[args.model].keys():
        parser.error(f"Parameters for {args.model}: " + ", ".join(DEFAULT_PARAMETERS[args.model]))
    if args.model not in FERMION_MODELS and (args.particles is not None or args.spin_up is not None):
        parser.error("--particles and --spin-up apply only to spinful fermionic models.")
    if args.model in FERMION_MODELS and (args.bond_disorder or args.field_disorder or args.disorder_seed):
        parser.error("Bond and field disorder are implemented for direct spin models only.")
    if args.n_qubits < 4 or (args.model in FERMION_MODELS and args.n_qubits % 2):
        parser.error("Use at least 4 qubits; spinful fermionic models require an even count.")
    if args.model in FERMION_MODELS:
        sites = args.n_qubits // 2
        args.particles = sites if args.particles is None else args.particles
        args.spin_up = (args.particles + 1) // 2 if args.spin_up is None else args.spin_up
        if not (0 <= args.spin_up <= sites and 0 <= args.particles - args.spin_up <= sites):
            parser.error("Particle/spin counts must satisfy 0 <= N_up,N_down <= number of sites.")
    return args


def exact_ground_state(model, *, particles=None, spin_up=None, seed=7):
    """Diagonalize the full spin space or a specified fermionic (N_up,N_down) sector.

    Keep the complete 2**n register for the covariance routines. In a degenerate
    ground space, the seeded eigensolver/projection selects a pure state; the
    reported gap flags degeneracy and no unique ground state is claimed.
    """
    sparse = get_sparse_operator(model.qubit_operator, n_qubits=model.n_qubits).tocsc()
    if np.all(sparse.data.imag == 0):
        sparse = sparse.real
    indices = np.arange(2 ** model.n_qubits)
    sector = {"kind": "full spin Hilbert space"}
    if model.model in FERMION_MODELS:
        particles = model.n_sites if particles is None else particles
        spin_up = (particles + 1) // 2 if spin_up is None else spin_up
        if (int(particles) != particles or int(spin_up) != spin_up or
                not (0 <= spin_up <= model.n_sites and 0 <= particles - spin_up <= model.n_sites)):
            raise ValueError("Invalid fermionic particle/spin sector.")
        counts = [sum((indices >> (model.n_qubits - 1 - q)) & 1
                      for q in range(spin, model.n_qubits, 2)) for spin in range(2)]
        indices = indices[(counts[0] == spin_up) & (counts[1] == particles - spin_up)]
        sector = {"kind": "fixed fermionic occupations", "particles": int(particles),
                  "spin_up": int(spin_up), "spin_down": int(particles - spin_up)}
    elif particles is not None or spin_up is not None:
        raise ValueError("Fermionic particle/spin sectors do not apply to direct spin models.")
    block = sparse[indices, :][:, indices]
    rng = np.random.default_rng(seed)
    initial = rng.standard_normal(len(indices))
    if np.iscomplexobj(block.data):
        initial = initial + 1j * rng.standard_normal(len(indices))
    initial /= np.linalg.norm(initial)
    if len(indices) <= 256:
        energies, vectors = eigh(block.toarray())
        ground = np.abs(energies - energies[0]) <= 1.e-10 * max(1., abs(energies[0]))
        basis = vectors[:, ground]
        vector = basis @ (basis.conj().T @ initial)
    else:
        energies, vectors = eigsh(block, k=2, which="SA", v0=initial, tol=1.e-11, maxiter=100000)
        order = np.argsort(energies)
        energies, vectors = energies[order], vectors[:, order]
        vector = vectors[:, 0]
    vector /= np.linalg.norm(vector)
    state = np.zeros(2 ** model.n_qubits, dtype=complex)
    state[indices] = vector
    pivot = int(np.argmax(np.abs(state)))
    state *= np.exp(-1j * np.angle(state[pivot]))
    energy = float(np.vdot(state, sparse @ state).real)
    residual = float(np.linalg.norm(sparse @ state - energy * state))
    if residual > 1.e-8 * max(1., abs(energy)):
        raise ValueError(f"Ground-state eigenvector residual too large: {residual:.6g}")
    gap = float(energies[1] - energies[0]) if len(energies) > 1 else None
    reference = {"energy": energy, "residual_norm": residual, "sector": sector,
                 "sector_dimension": len(indices), "gap_in_sector": gap,
                 "degenerate_within_tolerance": gap is not None and gap <= 1.e-9 * max(1., abs(energy)),
                 "state_selection_seed": seed}
    return state, reference


def record_result(context, assignment, omega, *, partition=False):
    """Validate and score fragments with the same contract as the geometry scan."""
    assignment = np.asarray(assignment, dtype=bool)
    problem = GroupingProblem(context.coefficients, context.covariance, context.compatible,
                              n_groups=assignment.shape[1],
                              max_overlap=1 if partition else assignment.shape[1])
    if not is_feasible(problem, assignment) or np.any(~assignment.any(axis=0)):
        raise RuntimeError("Invalid coverage, commutation, or empty grouping fragment.")
    if (not np.all(np.isfinite(omega)) or np.any(np.abs(omega[~assignment]) > 1.e-12)
            or not np.allclose(omega.sum(axis=1), context.coefficients, rtol=1.e-9, atol=1.e-9)):
        raise RuntimeError("Group coefficients do not reconstruct the Hamiltonian.")
    variances, _, cost = measurement_objective(context, assignment, omega)
    return {"epsilon2M": float(cost), "variances": list(variances),
            "n_groups": assignment.shape[1], "groups": [
                [{"word": context.terms[i].word, "coefficient": float(omega[i, g])}
                 for i in np.flatnonzero(assignment[:, g])] for g in range(assignment.shape[1])]}


def si_partition(hamiltonian, context):
    """Tequila SI with native binary-term ordering for equal-coefficient ties."""
    binary = BinaryHamiltonian.init_from_qubit_hamiltonian(hamiltonian)
    terms = [term for term in binary.binary_terms if any(term.binary_tuple())]
    groups = sorted_insertion_grouping(terms, condition="fc")
    index_by_word = {term.pauli_tuple: i for i, term in enumerate(context.terms)}
    assignment = np.zeros((context.n_terms, len(groups)), dtype=bool)
    for column, group in enumerate(groups):
        for term in group:
            bits = term.binary_tuple()
            n_qubits = len(bits) // 2
            word = tuple((q, "Y" if bits[q] and bits[q + n_qubits] else
                          "X" if bits[q] else "Z") for q in range(n_qubits)
                         if bits[q] or bits[q + n_qubits])
            assignment[index_by_word[word], column] = True
    return assignment


def milp_partition(context, args):
    """One non-overlapping MILP call, with the geometry-scan candidate settings."""
    group_bound, _ = coloring_group_bound(context, random_seed=args.seed, augmentation=4)
    problem = GroupingProblem(context.coefficients, context.covariance, context.compatible,
                              n_groups=group_bound, max_overlap=1, labels=list(range(context.n_terms)))
    start = time.perf_counter()
    solution = optimize_grouping(
        problem, method="milp", grouping_mode="non_overlapping",
        weight_heuristic="iterative_relative_std",
        heuristic_options={"sweeps": 3, "beta": 5.0, "delta": 1.e-12},
        solver_options={"time_limit": args.solver_time_limit_s,
                        "deadline": time.monotonic() + args.solver_time_limit_s,
                        "threads": args.solver_threads, "seed": args.seed,
                        "quiet": True},
        candidate_options={"random_partitions": 12, "max_candidate_groups": 4000,
                           "seed": args.seed, "group_bound": int(group_bound)},
        legacy_context=context,
    )
    assignment = np.asarray(solution.assignment, dtype=bool)
    assignment = assignment[:, assignment.any(axis=0)]
    record_result(context, assignment, context.coefficients[:, None] * assignment, partition=True)
    return assignment, {"status": solution.status, "runtime_s": time.perf_counter() - start,
                        "group_bound": int(group_bound), "metadata": solution.metadata}


def plot_comparison(model, results, output_stem):
    """Use the same first four color-cycle entries as driver_geometry_scan.py."""
    from itertools import cycle, islice
    import matplotlib as mpl
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    fig = Figure(figsize=(7, 4.6), layout="constrained")
    FigureCanvasAgg(fig)
    ax = fig.subplots()
    colors = list(islice(cycle(mpl.rcParams["axes.prop_cycle"].by_key()["color"]), len(METHODS)))
    values = [results[method]["epsilon2M"] for method in METHODS]
    bars = ax.bar(METHODS, values, color=colors, width=.65, zorder=3)
    ax.bar_label(bars, labels=[f"{value:.5g}" for value in values], padding=4)
    ax.set_ylim(0, max(values) * 1.18 if max(values) > 0 else 1.)
    ax.set_ylabel(r"$\varepsilon^2 M$", fontsize=18)
    ax.set_title(f"{MODEL_NAMES[model.model]}: {model.n_qubits} qubits\n"
                 f"Open {model.lattice}, {model.shape[0]} x {model.shape[1]} parent cells", fontsize=12)
    ax.grid(axis="y", alpha=.2, zorder=0)
    for extension in (".png", ".svg"):
        path = Path(str(output_stem) + extension)
        fig.savefig(path, dpi=200)
        print(f"Saved {path}", flush=True)
    return fig


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def output_stem(model, *, seed, particles=None, spin_up=None):
    """Readable Hamiltonian parameters, geometry, and reference-state settings."""
    parameters = "_".join(
        f"{key}={str(float(value)).removesuffix('.0')}"
        for key, value in model.parameters.items()
    )
    stem = (f"{model.model}_{model.n_qubits}q_{model.lattice}_"
            f"{model.shape[0]}x{model.shape[1]}_{parameters}")
    if model.disorder:
        stem += (f"_Wb={str(float(model.disorder['bond_strength'])).removesuffix('.0')}"
                 f"_Wf={str(float(model.disorder['field_strength'])).removesuffix('.0')}"
                 f"_dseed{model.disorder['seed']}")
    if particles is not None:
        stem += f"_ne{particles}_nup{spin_up}"
    return f"{stem}_seed{seed}"


def exact_covariances(terms, state, n_qubits, *, max_workers=1):
    """Full-amplitude Pauli actions with the package's commuting-pair contract.

    Tequila 1.9.11's wavefunction.items() discards amplitudes <= 1e-6, so its
    sparse Pauli-action route is unsuitable for these exact covariances.
    Apply each Pauli as a signed bit permutation in OpenFermion's MSB order.
    The action rows contain unit Paulis; coefficients enter only downstream.
    """
    state = np.asarray(state, dtype=complex).reshape(-1)
    if state.size != 2 ** n_qubits or not np.isclose(np.vdot(state, state).real, 1., rtol=0, atol=1.e-10):
        raise ValueError("Covariances require a normalized state in the declared qubit register.")
    if max_workers < 1:
        raise ValueError("Covariance workers must be positive.")
    indices = np.arange(state.size, dtype=np.int64)
    actions = np.empty((len(terms), state.size), dtype=complex)

    def apply(position):
        term = terms[position]
        flip = 0
        phase = np.ones(state.size, dtype=complex)
        for qubit, pauli in term.pauli_tuple:
            bit = 1 << (n_qubits - 1 - qubit)
            if pauli in "XY":
                flip ^= bit
            if pauli in "YZ":
                phase *= 1 - 2 * ((indices & bit) != 0)
            if pauli == "Y":
                phase *= 1j
        actions[position, indices ^ flip] = phase * state

    if max_workers == 1:
        for position in range(len(terms)):
            apply(position)
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for _ in executor.map(apply, range(len(terms))):
                pass
    single = actions @ state.conjugate()
    gram = actions.conjugate() @ actions.T
    means = {term.index: clean_real(single[i], tiny=1.e-10) for i, term in enumerate(terms)}
    covariance = {}
    for i, left in enumerate(terms):
        for j in range(i, len(terms)):
            right = terms[j]
            if terms_fully_commute(left, right):
                covariance[left.index, right.index] = clean_real(
                    gram[i, j] - means[left.index] * means[right.index], tiny=1.e-10)
    return covariance, means


def main(argv=None):
    args = parse_args(argv)
    np.random.seed(args.seed)
    model = build_lattice(args.model, args.n_qubits, lattice=args.lattice, shape=args.shape,
                          bond_disorder=args.bond_disorder, field_disorder=args.field_disorder,
                          disorder_seed=args.disorder_seed, **args.parameters)
    metadata = model.metadata()
    config = {"lattice": metadata, "particles": args.particles, "spin_up": args.spin_up,
              "seed": args.seed, "methods": list(METHODS), "condition": "fc",
              "ics_iterations": args.ics_iterations,
              "solver_time_limit_s": args.solver_time_limit_s, "solver_threads": args.solver_threads,
              "cov_workers": args.cov_workers}
    stem = output_stem(model, seed=args.seed, particles=args.particles, spin_up=args.spin_up)
    output = args.output_directory.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = {"configuration": config, "pauli_terms": [
        {"pauli": term.word, "coefficient": float(term.coefficient.real)}
        for term in make_terms(model.qubit_operator, model.n_qubits)]}
    report_path = output / f"{stem}_results.json"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Model={model.model} qubits={model.n_qubits} sites={model.n_sites} lattice={model.lattice} shape={model.shape}", flush=True)
    print("Parameters=" + json.dumps(model.parameters), flush=True)
    if model.disorder:
        print(f"Disorder: Wb={args.bond_disorder}, Wf={args.field_disorder}, seed={args.disorder_seed}", flush=True)
    for note in model.notes:
        print(note, flush=True)
    for path in plot_lattice(model, output / f"{stem}_lattice"):
        print(f"Saved {path}", flush=True)
    if args.build_only:
        print(f"Saved {report_path}", flush=True)
        return model

    all_terms = make_terms(model.qubit_operator, model.n_qubits)
    workspace = covariance_workspace_gib(len(all_terms), model.n_qubits)
    print(f"Exact diagonalization; Pauli terms={len(all_terms)}, covariance core workspace={workspace:.3f} GiB", flush=True)
    start = time.perf_counter()
    state, reference = exact_ground_state(model, particles=args.particles, spin_up=args.spin_up, seed=args.seed)
    print("Reference=" + json.dumps(reference), flush=True)
    covariances, expectations = exact_covariances(all_terms, state, model.n_qubits, max_workers=args.cov_workers)
    action_energy = hamiltonian_expectation(all_terms, expectations)
    if abs(action_energy - reference["energy"]) > 1.e-8 * max(1., abs(reference["energy"])):
        raise ValueError("Pauli-action energy does not match the exact reference energy.")
    context = make_optimization_context([term for term in all_terms if term.pauli_tuple], covariances)
    report.update(reference=reference, covariance_entries=len(covariances),
                  covariance_energy=action_energy, preprocessing_runtime_s=time.perf_counter() - start)
    del state, covariances
    print(f"Exact covariance entries={report['covariance_entries']} preprocessing_s={report['preprocessing_runtime_s']:.3f}", flush=True)
    print("Running the single non-overlapping MILP optimization...", flush=True)
    milp, milp_info = milp_partition(context, args)
    report["milp_optimization_calls"] = 1
    report["milp"] = milp_info
    print(f"MILP status={milp_info['status']}; group bound={milp_info['group_bound']}; "
          f"runtime_s={milp_info['runtime_s']:.6f}", flush=True)
    start = time.perf_counter()
    si = si_partition(model.tequila_hamiltonian, context)
    si_runtime = time.perf_counter() - start
    results = {}
    for base, initial, refined, runtime in [("SI", si, "SI-ICS", si_runtime),
                                            ("MILP", milp, "MILP-ICS", milp_info["runtime_s"])]:
        results[base] = record_result(context, initial, context.coefficients[:, None] * initial, partition=True)
        results[base]["runtime_s"] = runtime
        start = time.perf_counter()
        # Each baseline receives the same standard ICS call and iteration count;
        # the MILP deadline does not truncate either refinement, as in the scan.
        support, omega, ratios = optimize_from_nonoverlapping_groups(
            context, initial, n_iter=args.ics_iterations, condition="fc",
        )
        results[refined] = record_result(context, support, omega)
        results[refined]["runtime_s"] = time.perf_counter() - start
        results[refined]["sample_ratios"] = np.asarray(ratios, dtype=float).tolist()
    report["methods"] = results
    print("epsilon^2 M: " + "  ".join(f"{method}={results[method]['epsilon2M']:.12g}" for method in METHODS), flush=True)
    for method in METHODS:
        result = results[method]
        print(f"  {method}: groups={result['n_groups']} runtime_s={result['runtime_s']:.6f}", flush=True)
        if args.verbose:
            for group_index, group in enumerate(result["groups"], 1):
                print(f"    Group {group_index}: " + ", ".join(
                    f"{term['coefficient']:.12g} [{term['word']}]" for term in group), flush=True)
    report_path.write_text(json.dumps(report, indent=2, default=_json_default, allow_nan=False) + "\n")
    csv_path = output / f"{stem}_results.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("method", "epsilon2M", "n_groups", "runtime_s"))
        writer.writeheader()
        for method in METHODS:
            writer.writerow({"method": method, **{key: results[method][key]
                             for key in ("epsilon2M", "n_groups", "runtime_s")}})
    plot_comparison(model, results, output / f"{stem}_comparison")
    print(f"Saved {report_path}", flush=True)
    print(f"Saved {csv_path}", flush=True)
    return report


def cli():
    main()


if __name__ == "__main__":
    cli()
