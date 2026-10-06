"""Compare a capped overlapping-clique MILP with a single non-overlapping seed.

Run ``python -u driver_o_clique.py H4`` (or H6, LiH, BeH2, N2, H2O, NH3, MgO). Chemistry uses
STO-3G/Jordan-Wigner at R=1 angstrom with frozen core and FCI by default.
The overlapping solver is implemented in bin_mopt.overlapping. Its finite
profile objective is an upper bound; MILP-ICS remains a feasible incumbent.
--report-wfn evaluates final measurement costs with another wavefunction while
all grouping and coefficient optimization use --wfn (also the reporting default).
"""

import argparse
import json
from pathlib import Path
import tempfile
import time

import numpy as np

from bin_mopt.overlapping import (
    DEFAULT_MAX_CANDIDATES, covariance_measurement_cost, dense_ics, expand_partition_for_ics,
    optimize_overlapping_cliques, validate_overlap,
)

import tequila as tq
from openfermion.linalg import get_sparse_operator
from openfermion.utils import count_qubits
from tequila.grouping.binary_rep import BinaryHamiltonian
from tequila.grouping.binary_utils import sorted_insertion_grouping

from bin_mopt.measurement import GroupingProblem
from bin_mopt.opt import (
    coloring_group_bound,
    make_optimization_context,
    optimize_grouping,
)
from bin_mopt.utils import (
    build_covariance_dictionary,
    clean_real,
    covariance_workspace_gib,
    get_variance_wavefunction,
    hamiltonian_expectation,
    make_terms,
)


MOLECULES = ("H4", "H6", "LiH", "BeH2", "N2", "H2O", "NH3", "MgO")
DEFAULT_SOLVER_TIME_LIMIT_S = 20 * 60 * 60


def molecule_name(value):
    for name in MOLECULES:
        if name.casefold() == str(value).casefold():
            return name
    raise ValueError("Choose one of: " + ", ".join(MOLECULES))


def molecular_geometry(name, distance=1.0):
    """Canonical geometry-scan geometries in angstrom for the supported molecules."""
    distance = float(distance)
    if not np.isfinite(distance) or distance <= 0:
        raise ValueError("The bond distance must be positive and finite.")
    name = molecule_name(name)
    if name in ("H4", "H6"):
        atoms = [("H", 0.0, 0.0, i * distance) for i in range(int(name[1:]))]
    elif name == "LiH":
        atoms = [("Li", 0.0, 0.0, 0.0), ("H", 0.0, 0.0, distance)]
    elif name == "BeH2":
        atoms = [("Be", 0.0, 0.0, 0.0), ("H", 0.0, 0.0, distance),
                 ("H", 0.0, 0.0, -distance)]
    elif name == "N2":
        atoms = [("N", 0.0, 0.0, 0.0), ("N", 0.0, 0.0, distance)]
    elif name == "MgO":
        atoms = [("Mg", 0.0, 0.0, 0.0), ("O", 0.0, 0.0, distance)]
    elif name == "H2O":
        half_angle = np.deg2rad(107.6) / 2.0
        x, z = distance * np.sin(half_angle), distance * np.cos(half_angle)
        atoms = [("O", 0.0, 0.0, 0.0), ("H", x, 0.0, z), ("H", -x, 0.0, z)]
    elif name == "NH3":
        # Match the geometry scan: equal N-H bonds and 107-degree H-N-H angles.
        cosine = np.cos(np.deg2rad(107.0))
        radius = distance * np.sqrt(2 * (1 - cosine) / 3)
        z = distance * np.sqrt((1 + 2 * cosine) / 3)
        atoms = [("N", 0.0, 0.0, 0.0)] + [
            ("H", radius * np.cos(phi), radius * np.sin(phi), z)
            for phi in np.deg2rad([0.0, 120.0, 240.0])
        ]
    return "\n".join(f"{atom} {x:.12f} {y:.12f} {z:.12f}"
                     for atom, x, y, z in atoms)


def molecular_si_partition(hamiltonian, context):
    """Preserve Tequila's stable term ordering for sorted-insertion ties."""
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
    if not np.all(assignment.sum(axis=1) == 1):
        raise ValueError("SI failed to partition the nonidentity Pauli terms.")
    return assignment


def build_molecular_data(name, distance=1.0, threads=1, frozen_core=True, wfn="FCI", *,
                         cov_workers=None):
    """Return optimization data and shared molecular inputs for final reporting."""
    total_start = time.perf_counter()
    distance = float(distance)
    if int(threads) < 1:
        raise ValueError("threads must be positive.")
    cov_workers = int(threads) if cov_workers is None else int(cov_workers)
    if cov_workers < 1:
        raise ValueError("cov_workers must be positive.")
    wfn = str(wfn).upper()
    if wfn not in ("FCI", "CISD"):
        raise ValueError("Wavefunction must be FCI or CISD.")
    geometry = molecular_geometry(name, distance)
    name = molecule_name(name)
    print(f"Building {name}: R={distance:g} angstrom, wavefunction={wfn}, "
          f"frozen_core={bool(frozen_core)}...", flush=True)
    molecule = tq.chemistry.Molecule(
        geometry=geometry, basis_set="sto3g", transformation="JordanWigner",
        backend="pyscf", frozen_core=bool(frozen_core),
    )
    hamiltonian = molecule.make_hamiltonian()
    operator = hamiltonian.to_openfermion()
    n_qubits = int(count_qubits(operator))
    all_terms = make_terms(operator, n_qubits)
    estimated_workspace = covariance_workspace_gib(len(all_terms), n_qubits)
    if estimated_workspace > 16.0:
        raise MemoryError(f"Estimated covariance workspace {estimated_workspace:.3f} GiB "
                          "exceeds the 16 GiB driver limit.")
    hamiltonian_s = time.perf_counter() - total_start
    print(f"  Hamiltonian: qubits={n_qubits}, terms={sum(bool(term.pauli_tuple) for term in all_terms)}, "
          f"runtime_s={hamiltonian_s:.6f}", flush=True)
    wavefunction_inputs = (molecule, operator, all_terms, n_qubits, cov_workers)
    context, state_info = build_wavefunction_context(*wavefunction_inputs, wfn)
    si_start = time.perf_counter()
    si = molecular_si_partition(hamiltonian, context)
    si_cost = covariance_measurement_cost(context, context.coefficients[:, None] * si)
    si_time = time.perf_counter() - si_start
    print(f"  SI: groups={si.shape[1]}, epsilon2M={si_cost:.12g}, "
          f"runtime_s={si_time:.6f}", flush=True)
    metadata = dict(
        molecule=name, distance_angstrom=float(distance), geometry_angstrom=geometry,
        basis="sto3g", mapping="JordanWigner", condition="fc",
        frozen_core=bool(frozen_core), n_qubits=n_qubits, n_terms=int(context.n_terms),
        covariance_workers=cov_workers, hamiltonian_time_s=hamiltonian_s,
        si_time_s=si_time, molecular_setup_time_s=time.perf_counter() - total_start,
        **state_info,
    )
    return context, si, metadata, wavefunction_inputs


def build_wavefunction_context(molecule, operator, all_terms, n_qubits, cov_workers, wfn):
    """Evaluate a state on the shared Hamiltonian without choosing any groups."""
    wavefunction_start = time.perf_counter()
    print(f"  Preparing {wfn} reference state...", flush=True)
    sparse = get_sparse_operator(operator, n_qubits=n_qubits)
    energy, state = get_variance_wavefunction(
        molecule, operator, method=wfn, sparse_hamiltonian=sparse,
    )
    energy = clean_real(energy)
    state = np.asarray(state, dtype=complex).reshape(-1)
    norm_error = abs(float(np.vdot(state, state).real) - 1.0)
    if norm_error > 1.e-8:
        raise ValueError(f"Wavefunction normalization error is too large: {norm_error}")
    residual = float(np.linalg.norm(sparse @ state - energy * state))
    if wfn == "FCI" and residual > 1.e-8:
        raise ValueError(f"Ground-state residual is too large: {residual}")
    n_electrons = int(molecule.n_electrons)
    electron_counts = np.fromiter((i.bit_count() for i in range(state.size)), dtype=int)
    sector_leakage = float(np.sum(np.abs(state[electron_counts != n_electrons]) ** 2))
    if sector_leakage > 1.e-8:
        raise ValueError(f"State is outside the {n_electrons}-active-electron sector.")
    wavefunction_s = time.perf_counter() - wavefunction_start
    print(f"  {wfn} reference state: energy={energy:.12g}, "
          f"runtime_s={wavefunction_s:.6f}", flush=True)
    print(f"  Computing {wfn} covariances: qubits={n_qubits}, "
          f"terms={sum(bool(term.pauli_tuple) for term in all_terms)}, "
          f"cov_workers={cov_workers}...", flush=True)
    covariance_start = time.perf_counter()
    covariances, expectations = build_covariance_dictionary(
        all_terms, state, n_qubits, max_workers=cov_workers, chunksize=128,
    )
    energy_error = abs(float(hamiltonian_expectation(all_terms, expectations)) - energy)
    if energy_error > 1.e-8:
        raise ValueError("Pauli-action energy does not match the wavefunction energy.")
    context = make_optimization_context(
        [term for term in all_terms if term.pauli_tuple], covariances,
    )
    covariance_time = time.perf_counter() - covariance_start
    print(f"  Covariances and grouping context: runtime_s={covariance_time:.6f}", flush=True)
    metadata = dict(
        wavefunction=wfn, n_active_electrons=n_electrons, energy=float(energy), state_residual=residual,
        state_norm_error=norm_error, electron_sector_leakage=sector_leakage,
        pauli_energy_error=energy_error, wavefunction_time_s=wavefunction_s,
        covariance_time_s=covariance_time,
    )
    return context, metadata


def solve_milp_seed(context, time_limit_s=DEFAULT_SOLVER_TIME_LIMIT_S, threads=1, random_seed=7):
    """Run the canonical non-overlap pipeline once and return a compact seed."""
    if not np.isfinite(time_limit_s) or float(time_limit_s) <= 0 or int(threads) < 1:
        raise ValueError("The MILP time limit and thread count must be positive.")
    total_start = time.perf_counter()
    group_bound, _ = coloring_group_bound(context, random_seed=int(random_seed), augmentation=4)
    problem = GroupingProblem(
        context.coefficients, context.covariance, context.compatible,
        n_groups=group_bound, max_overlap=1, labels=list(range(context.n_terms)),
    )
    solution = optimize_grouping(
        problem, method="milp", grouping_mode="non_overlapping",
        weight_heuristic="relative_std",
        heuristic_options={"sweeps": 3, "beta": 5.0, "delta": 1.e-12},
        solver_options={"time_limit": float(time_limit_s),
                        "deadline": time.monotonic() + float(time_limit_s),
                        "threads": int(threads), "seed": int(random_seed),
                        "quiet": True},
        candidate_options={"random_partitions": 12, "max_candidate_groups": 40000,
                           "seed": int(random_seed), "group_bound": int(group_bound)},
        legacy_context=context,
    )
    assignment = np.asarray(solution.assignment, dtype=bool)
    assignment = assignment[:, assignment.any(axis=0)]
    if assignment.shape[0] != context.n_terms or not np.all(assignment.sum(axis=1) == 1):
        raise ValueError("MILP seed does not partition every term exactly once.")
    for column in range(assignment.shape[1]):
        members = np.flatnonzero(assignment[:, column])
        if not np.all(context.compatible[np.ix_(members, members)]):
            raise ValueError("MILP seed includes incompatible terms.")
    # Solver metadata may contain NumPy scalars/arrays; keep the public result
    # serializable without retaining Hamiltonians, state vectors, or actions.
    solver_metadata = json.loads(json.dumps(
        solution.metadata,
        default=lambda value: value.tolist() if isinstance(value, np.ndarray)
        else value.item() if isinstance(value, np.generic) else str(value),
    ))
    metadata = dict(
        status=str(solution.status), runtime_s=time.perf_counter() - total_start,
        group_bound=int(group_bound), n_groups=int(assignment.shape[1]),
        epsilon2M=float(solution.epsilon2M), solver_time_limit_s=float(time_limit_s),
        random_seed=int(random_seed), threads=int(threads), milp_optimization_calls=1,
        weight_heuristic="relative_std", heuristic_sweeps=3, metadata=solver_metadata,
    )
    return assignment, metadata


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def positive_float(value):
    number = float(value)
    if not np.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be positive and finite")
    return number


def nonnegative_int(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return number


def serialize_grouping(context, support, omega):
    return {"n_groups": int(support.shape[1]),
            "epsilon2M": covariance_measurement_cost(context, omega),
            "shared_terms": int(np.count_nonzero(support.sum(axis=1) > 1)),
            "memberships": int(support.sum()),
            "groups": [[{"word": context.terms[i].word, "coefficient": float(omega[i, g])}
                        for i in np.flatnonzero(support[:, g])]
                       for g in range(support.shape[1])]}


def load_seed_file(path, context, configuration):
    """Reuse one saved seed, checking the physical problem and Pauli coefficients."""
    saved = json.loads(Path(path).read_text())
    old = saved["configuration"]
    for key in ("molecule", "distance_angstrom", "wavefunction", "basis", "mapping", "frozen_core"):
        if old[key] != configuration[key]:
            raise ValueError(f"Saved seed has a different {key}")
    groups = saved["results"]["MILP"]["groups"]
    index = {term.word: i for i, term in enumerate(context.terms)}
    seed = np.zeros((context.n_terms, len(groups)), dtype=bool)
    for g, group in enumerate(groups):
        for term in group:
            i = index[term["word"]]
            if not np.isclose(term["coefficient"], context.coefficients[i], atol=1.e-10, rtol=1.e-9):
                raise ValueError("Saved seed has different Hamiltonian coefficients")
            seed[i, g] = True
    if not np.all(seed.sum(axis=1) == 1):
        raise ValueError("Saved seed is not a partition of this Hamiltonian")
    validate_overlap(context, seed, context.coefficients[:, None] * seed, seed.shape[1])
    return seed


def write_json_report(path, report):
    """Atomically save a seed or complete result, without a partially written cache."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix="." + path.name + ".", suffix=".tmp",
                                         delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(report, handle, indent=2, default=lambda value: value.item()
                      if isinstance(value, np.generic) else str(value))
            handle.write("\n")
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("molecule", type=molecule_name, choices=MOLECULES)
    parser.add_argument("--r", type=positive_float, default=1.0)
    parser.add_argument("--wfn", type=str.upper, choices=("FCI", "CISD"), default="FCI")
    parser.add_argument("--report-wfn", type=str.upper, choices=("FCI", "CISD"), default=None,
                        help="Wavefunction for final measurement costs only (default: --wfn)")
    parser.add_argument("--all-electron", action="store_true")
    parser.add_argument("--seed-time-limit-s", type=positive_float, default=DEFAULT_SOLVER_TIME_LIMIT_S,
                        help="Initial MILP optimization budget in seconds (default: 72000 = 20 hours)")
    parser.add_argument("--overlap-time-limit-s", type=positive_float, default=DEFAULT_SOLVER_TIME_LIMIT_S,
                        help="Total SCIP solve budget across rounds (default: 72000 = 20 hours); "
                        "excludes construction and ICS")
    parser.add_argument("--rounds", type=positive_int, default=1,
                        help="Distinct binary selections; more rounds exclude earlier supports")
    parser.add_argument("--threads", type=positive_int, default=1,
                        help="Numerical-library threads and concurrent SCIP workers (default: 1); "
                        "values above 1 enable parallel overlap solving")
    parser.add_argument("--cov-workers", type=positive_int, default=1,
                        help="Parallel covariance workers, independent of --threads (default: 1)")
    parser.add_argument("--seed", type=nonnegative_int, default=7)
    parser.add_argument("--max-candidates", type=positive_int, default=DEFAULT_MAX_CANDIDATES,
                        help=f"Maximum candidate cliques (default: {DEFAULT_MAX_CANDIDATES})")
    parser.add_argument("--random-partitions", type=positive_int, default=8)
    parser.add_argument("--move-budget", type=positive_int, default=1000)
    parser.add_argument("--exchange-covers", type=positive_int, default=48)
    parser.add_argument("--random-cover-trials", type=nonnegative_int, default=6000)
    parser.add_argument("--kempe-trials", type=nonnegative_int, default=6000)
    parser.add_argument("--max-random-covers", type=nonnegative_int, default=160)
    parser.add_argument("--ics-iterations", type=positive_int, default=5)
    parser.add_argument("--control-ics-iterations", type=positive_int, default=100,
                        help="Extra comparison on unchanged supports, excluded from algorithm timing")
    parser.add_argument("--shortlist", type=positive_int, default=6)
    parser.add_argument("--guidance", choices=("auto", "ics", "relative_std", "relative_variance", "score_softmax"),
                        default="auto", help="Candidate guide: heuristics use three sweeps; ics uses one QR "
                        "solve; auto chooses the fastest timed option. All finish with ICS.")
    parser.add_argument("--seed-file", type=Path,
                        help="Reuse a saved MILP partition; if missing, solve and save it here before "
                        "the overlap search")
    parser.add_argument("--seed-only", action="store_true",
                        help="Prepare and cache only the initial MILP partition; requires --seed-file")
    parser.add_argument("--output", type=Path, help="Default: o_clique_results/<molecule>.json")
    args = parser.parse_args(argv)
    args.report_wfn = args.report_wfn or args.wfn
    if args.seed_only and args.seed_file is None:
        parser.error("--seed-only requires --seed-file")
    return args


def print_comparison(report):
    """Print measurement costs and distinguish stage, cumulative, and control times."""
    configuration = report["configuration"]
    timings = report["timings"]
    optimization = report["optimization"]
    print(f"\n{configuration['molecule']} comparison "
          f"(optimization={configuration['wavefunction']}, "
          f"reporting={configuration['report_wavefunction']}, "
          f"R={configuration['distance_angstrom']:g} angstrom)",
          flush=True)
    print(f"{'Method':<12} {'Groups':>7} {'epsilon^2 M':>18} {'Stage (s)':>12} {'Total (s)':>12}")
    for method in ("SI", "SI-ICS", "MILP", "MILP-ICS", "O-clique"):
        result = report["results"][method]
        print(f"{method:<12} {result['n_groups']:>7d} {result['epsilon2M']:>18.12g} "
              f"{result['runtime_s']:>12.6f} {result['whole_procedure_s']:>12.6f}")
    print("Measurement counts are epsilon^2 M; for target error epsilon, M=(epsilon^2 M)/epsilon^2.")
    print("Stage: SI grouping, MILP seed, each ICS refinement, or O-clique search after MILP-ICS.")
    print("Total: shared preparation (including SI) plus each method's prerequisite stages.")
    print("Method totals exclude the additional SI-ICS comparison and long-ICS controls, "
          "except SI-ICS's own refinement, and exclude alternate-wavefunction reporting.")
    if timings["cached_seed"]:
        print("MILP seed reused: its stage time measures loading and validation; "
              "these totals exclude the original seed optimization.")
    print("\nProcess timings (s; indented entries are included in their parent):")
    stages = [
        ("Shared preparation", timings["preparation_s"]),
        ("  Hamiltonian", configuration["hamiltonian_time_s"]),
        ("  Reference state and validation", configuration["wavefunction_time_s"]),
        ("  Covariances and grouping context", configuration["covariance_time_s"]),
        ("  SI grouping and evaluation", configuration["si_time_s"]),
        ("MILP seed " + ("loading and validation" if timings["cached_seed"] else "optimization"),
         timings["seed_s"]),
        ("MILP-ICS expansion and refinement", optimization["baseline_s"]),
        ("O-clique search", timings["overlapping_search_s"]),
        ("  Candidate generation", optimization["candidate_generation"]["runtime_s"]),
        ("  Coefficient profiles", optimization["profiles"]["runtime_s"]),
        ("    Guidance selection/calibration", optimization["profiles"]["guidance"]["runtime_s"]),
        ("  Shortlist ICS refinement", optimization["shortlist_refinement_s"]),
    ]
    for number, round_info in enumerate(optimization["milp_rounds"], start=1):
        stages.extend([
            (f"  Round {number} MILP ({round_info['status']})", round_info["milp_time_s"]),
            ("    Model construction", round_info["model_build_s"]),
            ("    Solver", round_info["solver_s"]),
            (f"  Round {number} ICS refinement", round_info["ics_time_s"]),
        ])
    stages.extend([
        ("  Final validation", optimization["final_validation_s"]),
        ("SI-ICS comparison", timings["si_ics_comparison_s"]),
        ("MILP-ICS long control", timings["milp_ics_control_s"]),
        ("O-clique long control", timings["o_clique_control_s"]),
        ("Alternate-wavefunction reporting", timings["reporting_s"]),
        ("O-clique whole procedure", timings["whole_procedure_s"]),
        ("Driver computation including comparisons", timings["driver_total_s"]),
    ])
    for label, elapsed in stages:
        print(f"{label:<48} {elapsed:12.6f}")
    print("Timings exclude startup imports and JSON output.", flush=True)


def main(argv=None):
    from threadpoolctl import threadpool_limits, threadpool_info
    args = parse_args(argv)
    started = time.perf_counter()
    print(f"Resources: cov_workers={args.cov_workers}, threads={args.threads}; "
          f"seed_time_limit_s={args.seed_time_limit_s:g}, "
          f"overlap_time_limit_s={args.overlap_time_limit_s:g}", flush=True)
    with threadpool_limits(limits=args.threads):
        context, si, configuration, wavefunction_inputs = build_molecular_data(
            args.molecule, args.r, args.threads, not args.all_electron, args.wfn,
            cov_workers=args.cov_workers)
        configuration.update(random_seed=args.seed, threads=args.threads,
                             ics_iterations=args.ics_iterations, max_candidates=args.max_candidates,
                             random_partitions=args.random_partitions, move_budget=args.move_budget,
                             max_exchange_covers=args.exchange_covers, shortlist=args.shortlist,
                             random_cover_trials=args.random_cover_trials,
                             kempe_trials=args.kempe_trials, max_random_covers=args.max_random_covers,
                             seed_time_limit_s=args.seed_time_limit_s,
                             overlap_time_limit_s=args.overlap_time_limit_s, rounds=args.rounds,
                             guidance=args.guidance)
        preparation_s = time.perf_counter() - started
        seed_start = time.perf_counter()
        seed_reused = args.seed_file is not None and args.seed_file.exists()
        if seed_reused:
            seed = load_seed_file(args.seed_file, context, configuration)
            seed_info = {"status": "reused", "path": str(args.seed_file), "milp_optimization_calls": 0}
        else:
            if args.seed_file is not None:
                print("non-overlapping seed not found, producing MILP seed", flush=True)
            seed, seed_info = solve_milp_seed(context, args.seed_time_limit_s, args.threads, args.seed)
            if args.seed_file is not None:
                seed_info.update(path=str(args.seed_file), cache_created=True)
        if args.seed_file is not None:
            seed_report = {
                "configuration": configuration, "seed": seed_info,
                "results": {"MILP": serialize_grouping(
                    context, seed, context.coefficients[:, None] * seed)},
                "timings": {"preparation_s": preparation_s,
                            "seed_s": time.perf_counter() - seed_start,
                            "whole_procedure_s": time.perf_counter() - started,
                            "cached_seed": seed_reused,
                            "definition": "Molecular preparation and MILP seed solving or validation; "
                                          "excludes startup imports and seed JSON output"},
            }
            if not seed_reused:
                write_json_report(args.seed_file, seed_report)
                print(f"Saved non-overlapping seed to {args.seed_file}", flush=True)
        seed_s = time.perf_counter() - seed_start
        print(f"  MILP seed: {seed.shape[1]} groups, "
              f"epsilon2M={covariance_measurement_cost(context, context.coefficients[:, None] * seed):.12g}, "
              f"runtime_s={seed_s:.6f}, status={seed_info['status']}", flush=True)
        if args.seed_only:
            print(f"{args.molecule}: seed preparation complete; "
                  f"total_s={seed_report['timings']['whole_procedure_s']:.6f}; "
                  f"seed_file={args.seed_file}", flush=True)
            return seed_report
        support, omega, optimization = optimize_overlapping_cliques(
            context, seed, si, time_limit_s=args.overlap_time_limit_s, rounds=args.rounds,
            threads=args.threads, random_seed=args.seed, max_candidates=args.max_candidates,
            random_partitions=args.random_partitions, move_budget=args.move_budget,
            max_exchange_covers=args.exchange_covers, ics_iterations=args.ics_iterations,
            guidance=args.guidance, shortlist=args.shortlist,
            random_cover_trials=args.random_cover_trials, kempe_trials=args.kempe_trials,
            max_random_covers=args.max_random_covers)
        procedure_s = time.perf_counter() - started
        # SI-ICS is an additional comparison only; exclude it from the algorithm's
        # whole-procedure timing and report its time separately.
        comparison_start = time.perf_counter()
        si_support = expand_partition_for_ics(context, si)
        si_omega, _, si_info = dense_ics(context, si_support,
                                        initial_omega=context.coefficients[:, None] * si,
                                        n_iter=args.ics_iterations, linear_solver="qr")
        comparison_s = time.perf_counter() - comparison_start
        print(f"  SI-ICS: groups={si_support.shape[1]}, epsilon2M={si_info['epsilon2M']:.12g}, "
              f"runtime_s={comparison_s:.6f}", flush=True)
        control_start = time.perf_counter()
        baseline_support = expand_partition_for_ics(context, seed)
        baseline_control_omega, _, baseline_control = dense_ics(
            context, baseline_support, initial_omega=context.coefficients[:, None] * seed,
            n_iter=args.control_ics_iterations, linear_solver="qr")
        baseline_control_s = time.perf_counter() - control_start
        print(f"  MILP-ICS long control: groups={baseline_support.shape[1]}, "
              f"epsilon2M={baseline_control['epsilon2M']:.12g}, "
              f"runtime_s={baseline_control_s:.6f}", flush=True)
        selected_control_start = time.perf_counter()
        selected_control_omega, _, selected_control = dense_ics(
            context, support, initial_omega=omega,
            n_iter=args.control_ics_iterations, linear_solver="qr")
        selected_control_s = time.perf_counter() - selected_control_start
        control_s = time.perf_counter() - control_start
        print(f"  O-clique long control: groups={support.shape[1]}, "
              f"epsilon2M={selected_control['epsilon2M']:.12g}, "
              f"runtime_s={selected_control_s:.6f}", flush=True)
        original_cliques = {tuple(np.flatnonzero(g)) for g in baseline_support.T}
        selected_cliques = {tuple(np.flatnonzero(g)) for g in support.T}
        optimization["cliques_added_vs_milp_ics"] = len(selected_cliques - original_cliques)
        optimization["cliques_removed_vs_milp_ics"] = len(original_cliques - selected_cliques)
        results = {
            "MILP": serialize_grouping(context, seed, context.coefficients[:, None] * seed),
            "MILP-ICS": {"n_groups": seed.shape[1], "epsilon2M": optimization["baseline"]["epsilon2M"],
                         "runtime_s": optimization["baseline_s"]},
            "SI": serialize_grouping(context, si, context.coefficients[:, None] * si),
            "SI-ICS": serialize_grouping(context, si_support, si_omega),
            "O-clique": serialize_grouping(context, support, omega),
        }
        # Final reporting starts only after every optimization and control run.
        # Reuse the original Hamiltonian and term ordering for both wavefunctions.
        configuration["report_wavefunction"] = args.report_wfn
        configuration["report_energy"] = configuration["energy"]
        reporting_s = 0.0
        if args.report_wfn != args.wfn:
            reporting_start = time.perf_counter()
            # The optimizer returns the baseline cost but not its coefficients.
            # Reproduce that baseline with the same optimization covariances.
            baseline_omega, _, _ = dense_ics(
                context, baseline_support, initial_omega=context.coefficients[:, None] * seed,
                n_iter=args.ics_iterations, linear_solver="qr")
            print(f"Final measurement evaluation: {args.report_wfn} "
                  f"(groups and coefficients optimized with {args.wfn})", flush=True)
            report_context, report_state = build_wavefunction_context(
                *wavefunction_inputs, args.report_wfn)
            configuration["report_energy"] = report_state["energy"]
            configuration["report_state"] = report_state
            final_coefficients = {
                "MILP": context.coefficients[:, None] * seed,
                "MILP-ICS": baseline_omega,
                "SI": context.coefficients[:, None] * si,
                "SI-ICS": si_omega,
                "O-clique": omega,
            }
            for method, coefficients in final_coefficients.items():
                results[method]["optimization_epsilon2M"] = results[method]["epsilon2M"]
                results[method]["epsilon2M"] = covariance_measurement_cost(report_context, coefficients)
            for control, coefficients in ((baseline_control, baseline_control_omega),
                                          (selected_control, selected_control_omega)):
                control.update(optimization_wavefunction=args.wfn, report_wavefunction=args.report_wfn)
                control["optimization_epsilon2M"] = control["epsilon2M"]
                control["epsilon2M"] = covariance_measurement_cost(report_context, coefficients)
            reporting_s = time.perf_counter() - reporting_start
        timings = {"preparation_s": preparation_s, "seed_s": seed_s,
                   "si_s": configuration["si_time_s"],
                   "si_whole_procedure_s": preparation_s,
                   "si_ics_whole_procedure_s": preparation_s + comparison_s,
                   "milp_whole_procedure_s": preparation_s + seed_s,
                   "milp_ics_only_s": optimization["baseline_s"],
                   "milp_ics_whole_procedure_s": preparation_s + seed_s + optimization["baseline_s"],
                   "overlapping_search_s": optimization["runtime_s"] - optimization["baseline_s"],
                   "whole_procedure_s": procedure_s, "si_ics_comparison_s": comparison_s,
                   "long_ics_controls_s": control_s,
                   "milp_ics_control_s": baseline_control_s,
                   "o_clique_control_s": selected_control_s,
                   "reporting_s": reporting_s,
                   "cached_seed": seed_reused,
                   "stage_definition": "SI grouping/evaluation, seed solving/loading, each ICS refinement, "
                                       "or overlapping search after MILP-ICS; total times include shared "
                                       "preparation (including SI) and prerequisite stages",
                   "definition": "Elapsed from main before molecular preparation through final validation; "
                                 "includes seed, calibration, candidates, profiles, MILP and ICS; "
                                 "excludes startup imports, SI-ICS and long-ICS controls, "
                                 "alternate-wavefunction reporting, and JSON output"}
        method_timings = {
            "SI": (configuration["si_time_s"], timings["si_whole_procedure_s"]),
            "SI-ICS": (comparison_s, timings["si_ics_whole_procedure_s"]),
            "MILP": (seed_s, timings["milp_whole_procedure_s"]),
            "MILP-ICS": (optimization["baseline_s"], timings["milp_ics_whole_procedure_s"]),
            "O-clique": (timings["overlapping_search_s"], procedure_s),
        }
        for method, (stage_s, total_s) in method_timings.items():
            results[method].update(runtime_s=stage_s, whole_procedure_s=total_s)
        timings["driver_total_s"] = time.perf_counter() - started
        report = {"configuration": configuration, "seed": seed_info, "results": results,
                  "optimization": optimization, "timings": timings,
                  "threadpools": threadpool_info(), "si_ics": si_info,
                  "long_ics_controls": {"iterations": args.control_ics_iterations,
                                        "MILP-ICS": baseline_control,
                                        "O-clique": selected_control}}
    print_comparison(report)
    output = args.output or Path("o_clique_results") / (args.molecule.lower() + ".json")
    write_json_report(output, report)
    baseline_cost = results["MILP-ICS"]["epsilon2M"]
    gain = 100 * (1 - results["O-clique"]["epsilon2M"] / baseline_cost) if baseline_cost else 0.
    print(f"{args.molecule}: O-clique groups={results['O-clique']['n_groups']}, "
          f"epsilon2M({args.report_wfn})={results['O-clique']['epsilon2M']:.12g}, "
          f"improvement={gain:.3f}%, total={procedure_s:.3f}s; saved {output}", flush=True)
    return report


if __name__ == "__main__":
    main()
