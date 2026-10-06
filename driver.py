import argparse
import time

import numpy as np
from openfermion.linalg import get_sparse_operator
from openfermion.utils import count_qubits

from bin_mopt.hamiltonians import MOLECULES, get_molecule
from bin_mopt.measurement import GroupingProblem
from bin_mopt.opt import (
    AVAILABLE_METHODS,
    BinaryMeasurementOptimizer,
    OptimizationResult,
    coloring_group_bound,
    make_optimization_context,
    measurement_objective,
    optimize_grouping,
)
from bin_mopt.utils import (
    DEFAULT_COVARIANCE_CHUNKSIZE,
    action_matrix_gib,
    build_covariance_dictionary,
    clean_real,
    covariance_workspace_gib,
    default_cov_workers,
    get_variance_wavefunction,
    hamiltonian_expectation,
    make_terms,
)
from bin_mopt.weighting import WEIGHT_HEURISTICS


# Ordinary constants keep one copied driver self-contained and record the
# precise scientific and optimizer configuration used for a calculation.
cov_workers = default_cov_workers()
cov_chunksize = DEFAULT_COVARIANCE_CHUNKSIZE
solver_time_limit_s = 300.0
random_partitions = 12
random_seed = 7
color_bound_augmentation = 4
max_candidate_groups = 4000
max_covariance_workspace_gib = 16.0
ics_iterations = 5

lns_iterations = 200
lns_destroy_fraction = 0.15
lns_restarts = 1

weight_sweeps = 3
weight_beta = 5.0
weight_delta = 1.0e-12


def _positive_int(value):
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if parsed < 1 or str(parsed) != str(value).strip():
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _positive_float(value):
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("must be a positive number") from exc
    if not np.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be a positive number")
    return parsed


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Build a molecular Jordan-Wigner Hamiltonian, construct the stable "
            "sparse covariance dictionary, optimize a fully commuting grouping, "
            "and apply the established ICS refinement."
        )
    )
    parser.add_argument(
        "molecule",
        help="Molecule helper: {}.".format(", ".join(MOLECULES)),
    )
    parser.add_argument(
        "--method",
        choices=AVAILABLE_METHODS,
        default="milp",
        help="Exact optimizer name (default: milp).",
    )
    parser.add_argument(
        "--wfn",
        type=lambda value: str(value).upper(),
        choices=("FCI", "CISD", "HF"),
        default="FCI",
        help="Reference wavefunction for covariance construction (default: FCI).",
    )
    parser.add_argument(
        "--cov-workers",
        type=_positive_int,
        default=None,
        metavar="N",
        help="Covariance workers (default from the driver constant).",
    )
    parser.add_argument(
        "--weight-heuristic",
        choices=WEIGHT_HEURISTICS,
        default="iterative_relative_std",
        help="Exact split-coefficient heuristic name.",
    )
    parser.add_argument(
        "--solver-time-limit-s",
        type=_positive_float,
        default=None,
        metavar="T",
        help="One wall-clock limit shared by all binary-optimizer phases.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print every Pauli word and split coefficient in every group.",
    )
    parser.add_argument(
        "--non-overlapping",
        action="store_true",
        help="Assign every term to exactly one binary group (default).",
    )
    return parser.parse_args(argv)


def _drop_empty_columns(assignment, omega=None):
    assignment = np.asarray(assignment, dtype=bool)
    keep = np.any(assignment, axis=0)
    compact_assignment = assignment[:, keep]
    if omega is None:
        return compact_assignment, None
    return compact_assignment, np.asarray(omega, dtype=float)[:, keep]


def _groups_from_assignment(assignment):
    return tuple(
        tuple(np.flatnonzero(assignment[:, group_index]).tolist())
        for group_index in range(assignment.shape[1])
    )


def _solution_to_result(
    solution,
    context,
    cap_groups,
):
    assignment, omega = _drop_empty_columns(
        solution.assignment, solution.split_coefficients
    )
    if assignment.shape[1] < 1:
        raise RuntimeError("The selected optimizer returned no nonempty group.")
    groups = _groups_from_assignment(assignment)
    if any(not np.all(context.compatible[np.ix_(group, group)]) for group in groups):
        raise RuntimeError("A group contains incompatible Pauli words.")
    if not np.all(assignment.sum(axis=1) == 1):
        raise RuntimeError("A non-overlapping result must cover every term once.")
    omega = context.coefficients[:, np.newaxis] * assignment
    variances, score, eps_sq_m = measurement_objective(context, assignment, omega)
    raw_history = list(getattr(solution, "history", ()) or ())
    if not raw_history:
        raw_history = [(0, float(eps_sq_m))]
    history = tuple((int(step), float(value)) for step, value in raw_history)
    if any(
        history[index][1] > history[index - 1][1] + 1.0e-10
        for index in range(1, len(history))
    ):
        raise RuntimeError("Optimizer history is not monotone nonincreasing.")
    observation_history = tuple(
        (int(step), float(value))
        for step, value in (getattr(solution, "observation_history", ()) or ())
    )
    metadata = dict(getattr(solution, "metadata", {}) or {})
    n_candidates = int(
        metadata.get(
            "actual_candidate_count",
            metadata.get("candidate_count", metadata.get("columns", 0)),
        )
        or 0
    )
    return OptimizationResult(
        mode="non-overlapping",
        groups=groups,
        assignment=assignment,
        omega=np.asarray(omega, dtype=float),
        variances=tuple(float(value) for value in variances),
        score=float(score),
        eps_sq_m=float(eps_sq_m),
        cap_groups=int(cap_groups),
        max_split=1,
        solver_status="method={} mode={} status={}".format(
            solution.method, "non_overlapping", solution.status
        ),
        solver_objective=float(eps_sq_m),
        iterations=max(0, len(history) - 1),
        n_candidates=n_candidates,
        history=history,
        observation_history=observation_history,
    )


def _print_group_summary(result, context, verbose):
    split_counts = result.assignment.sum(axis=1)
    print("  Group sizes={}".format([len(group) for group in result.groups]))
    print(
        "  Pauli sharing counts={}".format(
            {
                int(count): int(np.count_nonzero(split_counts == count))
                for count in np.unique(split_counts)
            }
        )
    )
    if not verbose:
        return
    for group_index, group in enumerate(result.groups):
        print(
            "  Group {} ({} terms), fragment variance={:.12g}".format(
                group_index + 1,
                len(group),
                float(result.variances[group_index]),
            )
        )
        for position in group:
            term = context.terms[position]
            print(
                "    Pauli={} original_coefficient={:.12g} "
                "split_coefficient={:.12g}".format(
                    term.word,
                    float(context.coefficients[position]),
                    float(result.omega[position, group_index]),
                )
            )


def _print_result(result, context, runtime_s, wfn_label, verbose):
    print("")
    print("Lowest grouping found:")
    print("  Mode={}".format(result.mode))
    print("  Number of groups={}".format(len(result.groups)))
    print("  Effective maximum Pauli split={}".format(result.max_split))
    print("  Compatible groups=True")
    print("  epsilon^2 M(wfn={})={:.12g}".format(wfn_label, result.eps_sq_m))
    print("  Measurement score S={:.12g}".format(result.score))
    print("  Candidate/local solver status={}".format(result.solver_status))
    print("  Candidate cliques={}".format(result.n_candidates))
    print("  Optimization runtime_s={:.6f}".format(runtime_s))
    if result.sample_ratios is not None:
        print("  ICS shot ratios={}".format([float(v) for v in result.sample_ratios]))
    _print_group_summary(result, context, verbose)


def _solver_options(method, deadline, selected_limit, threads):
    options = {
        "deadline": float(deadline),
        "time_limit": float(selected_limit),
        "threads": int(threads),
        "seed": int(random_seed),
        "quiet": True,
    }
    if method == "milp_lns":
        options.update(
            {
                "iterations": int(lns_iterations),
                "destroy_fraction": float(lns_destroy_fraction),
                "restarts": int(lns_restarts),
            }
        )
    return options


def _print_method_parameters(
    method,
    weight_heuristic,
    selected_cov_workers,
    threads,
    selected_limit,
):
    print("Optimization parameters:")
    print("  method={}".format(method))
    print("  grouping_mode=non_overlapping")
    print("  weight_heuristic={}".format(weight_heuristic))
    print("  seed={}".format(random_seed))
    print("  cov_workers={}".format(selected_cov_workers))
    print("  solver_threads={}".format(threads))
    print("  solver_time_limit_s={}".format(selected_limit))
    print("  random_partitions={}".format(random_partitions))
    print("  max_candidate_groups={}".format(max_candidate_groups))
    if method == "milp_lns":
        print("  iterations={}".format(lns_iterations))
        print("  destroy_fraction={}".format(lns_destroy_fraction))
        print("  restarts={}".format(lns_restarts))


def _print_solver_thread_use():
    print("Solver thread use=SciPy MILP does not expose direct control")


def main(argv=None):
    args = parse_args(argv)
    selected_cov_workers = cov_workers if args.cov_workers is None else args.cov_workers
    solver_threads = selected_cov_workers
    selected_limit = (
        solver_time_limit_s
        if args.solver_time_limit_s is None
        else args.solver_time_limit_s
    )

    print("Building {} Jordan-Wigner Hamiltonian...".format(args.molecule), flush=True)
    canonical_name, molecule_data = get_molecule(args.molecule)
    molecule, _, _, reported_n_paulis, qubit_operator = molecule_data
    n_qubits = int(count_qubits(qubit_operator))
    all_terms = make_terms(qubit_operator, n_qubits)
    measurable_terms = [term for term in all_terms if term.pauli_tuple]
    if len(measurable_terms) != reported_n_paulis:
        raise ValueError(
            "Hamiltonian helper reported {} measurable terms, found {}.".format(
                reported_n_paulis, len(measurable_terms)
            )
        )

    estimated_action_gib = action_matrix_gib(len(all_terms), n_qubits)
    estimated_workspace_gib = covariance_workspace_gib(len(all_terms), n_qubits)
    if estimated_workspace_gib > max_covariance_workspace_gib:
        raise MemoryError(
            "Estimated covariance workspace is {:.3f} GiB, above the driver "
            "safety limit {:.3f} GiB.".format(
                estimated_workspace_gib, max_covariance_workspace_gib
            )
        )

    sparse_hamiltonian = get_sparse_operator(qubit_operator, n_qubits=n_qubits)
    energy, state_vector = get_variance_wavefunction(
        molecule,
        qubit_operator,
        method=args.wfn,
        sparse_hamiltonian=sparse_hamiltonian,
    )
    state_vector = np.asarray(state_vector, dtype=complex).reshape(-1)
    reference_energy = clean_real(energy, tiny=1.0e-7)

    print("Molecule={}".format(canonical_name), flush=True)
    print("Method={}".format(args.method), flush=True)
    print("Mode=non_overlapping", flush=True)
    print("Mapping=Jordan-Wigner", flush=True)
    print("Compatibility graph=fully commuting", flush=True)
    print("Covariance wavefunction={}".format(args.wfn), flush=True)
    print("Covariance workers={}".format(selected_cov_workers), flush=True)
    print("Solver threads={}".format(solver_threads), flush=True)
    print("{} Energy={:.16g}".format(args.wfn, reference_energy), flush=True)
    print("Number of qubits={}".format(n_qubits), flush=True)
    print(
        "Number of Pauli products to measure={}".format(len(measurable_terms)),
        flush=True,
    )
    print(
        "Building covariance dictionary with {} worker(s); Pauli-action "
        "matrix={:.3f} GiB; core workspace={:.3f} GiB...".format(
            selected_cov_workers, estimated_action_gib, estimated_workspace_gib
        ),
        flush=True,
    )
    covariance_start = time.perf_counter()
    covariances, single_expectations = build_covariance_dictionary(
        all_terms,
        state_vector,
        n_qubits,
        max_workers=selected_cov_workers,
        chunksize=cov_chunksize,
    )
    covariance_runtime = time.perf_counter() - covariance_start
    action_energy = hamiltonian_expectation(all_terms, single_expectations)
    if abs(action_energy - reference_energy) > 1.0e-7:
        raise ValueError(
            "Pauli-action energy {} does not match reference energy {}.".format(
                action_energy, reference_energy
            )
        )
    print(
        "Covariance entries={} runtime_s={:.6f}".format(
            len(covariances), covariance_runtime
        ),
        flush=True,
    )
    print("Pauli-action energy check={:.16g}".format(action_energy), flush=True)

    context = make_optimization_context(measurable_terms, covariances)
    group_bound, coloring_groups = coloring_group_bound(
        context,
        random_seed=random_seed,
        augmentation=color_bound_augmentation,
    )
    cap_groups = int(group_bound)
    print("Complement colors={}".format(len(coloring_groups)), flush=True)
    print("Group bound={}".format(cap_groups), flush=True)
    print(
        "Coloring group sizes={}".format([len(g) for g in coloring_groups]), flush=True
    )
    _print_method_parameters(
        args.method,
        args.weight_heuristic,
        selected_cov_workers,
        solver_threads,
        selected_limit,
    )
    _print_solver_thread_use()

    problem = GroupingProblem(
        coefficients=context.coefficients,
        covariance=context.covariance,
        compatibility=context.compatible,
        n_groups=cap_groups,
        max_overlap=1,
        labels=list(range(context.n_terms)),
    )
    heuristic_options = {
        "sweeps": int(weight_sweeps),
        "beta": float(weight_beta),
        "delta": float(weight_delta),
    }
    candidate_options = {
        "random_partitions": int(random_partitions),
        "max_candidate_groups": int(max_candidate_groups),
        "seed": int(random_seed),
        "group_bound": int(cap_groups),
    }

    optimization_start = time.perf_counter()
    deadline = time.monotonic() + float(selected_limit)
    solution = optimize_grouping(
        problem,
        method=args.method,
        grouping_mode="non_overlapping",
        weight_heuristic=args.weight_heuristic,
        heuristic_options=heuristic_options,
        solver_options=_solver_options(
            args.method, deadline, selected_limit, solver_threads
        ),
        candidate_options=candidate_options,
        legacy_context=context,
    )
    binary_runtime = time.perf_counter() - optimization_start
    binary_result = _solution_to_result(solution, context, cap_groups)
    metadata = dict(getattr(solution, "metadata", {}) or {})
    print(
        "Actual candidate count={}".format(
            metadata.get(
                "actual_candidate_count",
                metadata.get("candidate_count", binary_result.n_candidates),
            )
        )
    )
    print("Candidate group bound={}".format(cap_groups))
    ics_optimizer = BinaryMeasurementOptimizer(
        context,
        cap_groups=cap_groups,
        time_limit=max(float(selected_limit), 1.0e-6),
        random_partitions=random_partitions,
        random_seed=random_seed,
        max_candidates=max_candidate_groups,
        deadline=deadline,
    )
    ics_start = time.perf_counter()
    ics_result = ics_optimizer.optimize_nonoverlap_ics(
        binary_result, n_iterations=ics_iterations
    )
    ics_runtime = time.perf_counter() - ics_start

    print("Binary optimization runtime_s={:.6f}".format(binary_runtime))
    print("Additional ICS runtime_s={:.6f}".format(ics_runtime))
    _print_result(binary_result, context, binary_runtime, args.wfn, args.verbose)
    _print_result(ics_result, context, ics_runtime, args.wfn, args.verbose)
    print("Pipeline runtime_s={:.6f}".format(binary_runtime + ics_runtime))

    return binary_result, ics_result


def cli():
    """Console-script wrapper with a conventional zero exit status."""

    main()


if __name__ == "__main__":
    cli()
