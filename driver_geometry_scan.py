"""Transfer one molecular MILP partition across a bond-length scan.

Run ``python -u driver_geometry_scan.py LiH`` (or H2, MgO, SiO, H4, H6, BeH2, N2, H2O, NH3).
R is the bond distance in angstrom; H chains have uniform spacing and linear
BeH2 stretches both Be--H bonds equally. Use STO-3G/Jordan--Wigner. Only the
--ropt reference (default: 1.0 A) invokes MILP; each geometry supplies its own coefficients and
FCI (default) or CISD covariances. SI uses Tequila, as in VarSI. All ICS curves
use the same standard non-overlap ICS bridge as driver.py. --report-wfn selects
the wavefunction for final measurement costs without changing optimization.
--all-electron disables the frozen-core approximation at every geometry.
"""

import argparse
import csv
from decimal import Decimal
import json
from pathlib import Path
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tequila as tq
from openfermion.linalg import get_sparse_operator
from openfermion.utils import count_qubits
from tequila.grouping.binary_rep import BinaryHamiltonian
from tequila.grouping.binary_utils import sorted_insertion_grouping

from bin_mopt.varsi import varsi_ordered, varsi_refine
from bin_mopt.ics import optimize_from_nonoverlapping_groups
from bin_mopt.measurement import GroupingProblem, is_feasible
from bin_mopt.opt import (
    coloring_group_bound,
    make_optimization_context,
    measurement_objective,
    optimize_grouping,
)
from bin_mopt.utils import (
    build_covariance_dictionary,
    clean_real,
    clean_variance,
    covariance_workspace_gib,
    get_variance_wavefunction,
    hamiltonian_expectation,
    make_terms,
)


REFERENCE_DISTANCE = 1.0
METHODS = ("SI", "SI-ICS", "MILP", "MILP+ICS")
REFINED_METHODS = ("MILP-R", "MILP-R-ICS")
MOLECULES = ("H2", "LiH", "MgO", "SiO", "H4", "H6", "BeH2", "N2", "H2O", "NH3")
WAVEFUNCTIONS = ("FCI", "CISD")


def molecule_name(value):
    for name in MOLECULES:
        if name.casefold() == value.casefold():
            return name
    raise argparse.ArgumentTypeError("choose one of: " + ", ".join(MOLECULES))


def geometry_string(name, distance):
    if name in ("H2", "H4", "H6"):
        atoms = [("H", 0.0, 0.0, i * distance) for i in range(int(name[1:]))]
    elif name == "LiH":
        atoms = [("Li", 0.0, 0.0, 0.0), ("H", 0.0, 0.0, distance)]
    elif name in ("MgO", "SiO"):
        atoms = [(name[:-1], 0.0, 0.0, 0.0), ("O", 0.0, 0.0, distance)]
    elif name == "BeH2":
        atoms = [("Be", 0.0, 0.0, 0.0), ("H", 0.0, 0.0, distance),
                 ("H", 0.0, 0.0, -distance)]
    elif name == "N2":
        atoms = [("N", 0.0, 0.0, 0.0), ("N", 0.0, 0.0, distance)]
    elif name == "H2O":
        # Match the 107.6-degree H-O-H angle in bin_mopt.hamiltonians.H2O.
        half_angle = np.deg2rad(107.6) / 2
        x = distance * np.sin(half_angle)
        z = distance * np.cos(half_angle)
        atoms = [("O", 0.0, 0.0, 0.0), ("H", x, 0.0, z), ("H", -x, 0.0, z)]
    elif name == "NH3":
        # Three H atoms at azimuths 0, 120, 240 degrees around N.
        # cos(H-N-H) = (z^2 - radius^2/2) / distance^2.
        cosine = np.cos(np.deg2rad(107.0))
        radius = distance * np.sqrt(2 * (1 - cosine) / 3)
        z = distance * np.sqrt((1 + 2 * cosine) / 3)
        atoms = [("N", 0.0, 0.0, 0.0)] + [
            ("H", radius * np.cos(phi), radius * np.sin(phi), z)
            for phi in np.deg2rad([0.0, 120.0, 240.0])
        ]
    else:
        raise ValueError(f"Unsupported molecule: {name}")
    return "\n".join(f"{atom} {x:.12f} {y:.12f} {z:.12f}" for atom, x, y, z in atoms)


def positive_float(value):
    number = float(value)
    if not np.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be positive and finite")
    return number


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("molecule", type=molecule_name, choices=MOLECULES,
                        help="Molecule to scan; R is its bond distance in angstrom.")
    parser.add_argument("--wfn", type=str.upper, choices=WAVEFUNCTIONS,
                        default=None, help="Wavefunction for optimization and ICS (default: FCI).")
    parser.add_argument("--report-wfn", type=str.upper, choices=WAVEFUNCTIONS,
                        default=None, help="Wavefunction for reported measurement costs (default: --wfn).")
    parser.add_argument("--all-electron", action="store_true",
                        help="Include all electrons by building every molecule with frozen_core=False.")
    parser.add_argument("--rmin", type=positive_float, default=0.5,
                        help="First scan distance in angstrom (default: 0.5).")
    parser.add_argument("--rmax", type=positive_float, default=1.5,
                        help="Maximum scan distance in angstrom (default: 1.5); spacing is 0.1.")
    parser.add_argument("--ropt", type=positive_float, default=REFERENCE_DISTANCE,
                        help="Distance for the single reference MILP optimization (default: 1.0).")
    parser.add_argument("--solver-time-limit-s", type=positive_float, default=72000.0,
                        help="Time limit for the single reference MILP pipeline.")
    parser.add_argument("--cov-workers", type=positive_int, default=1)
    parser.add_argument("--ics-iterations", type=positive_int, default=5,
                        help="ICS iterations (default: 5, matching Tequila).")
    parser.add_argument("--new-terms", choices=("greedy", "varsi-o", "varsi-or"),
                        default="greedy", help="Insertion method for new Pauli terms (default: greedy).")
    parser.add_argument("--new-refine-all", action="store_true",
                        help="Refine all terms whenever new terms appear, after any insertion method.")
    parser.add_argument("--refine-milp", action="store_true",
                        help="Refine all terms at every non-reference geometry; implies --new-refine-all.")
    parser.add_argument("--max-sweeps", type=positive_int, default=100,
                        help="Maximum VarSI refinement sweeps per geometry (default: 100).")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--verbose", action="store_true",
                        help="Print WARNING lines for individual new Pauli terms.")
    parser.add_argument("--save-groups", action="store_true",
                        help="Save reusable groupings for every geometry and method to JSON.")
    args = parser.parse_args(argv)
    args.new_refine_all = args.new_refine_all or args.refine_milp
    args.wfn_provided = args.wfn is not None
    args.wfn = args.wfn or "FCI"
    args.report_wfn = args.report_wfn or args.wfn
    if args.rmax < args.rmin:
        parser.error("--rmax must be greater than or equal to --rmin")
    # Decimal arithmetic includes aligned endpoints without changing spacing
    # or accidentally adding a point beyond rmax due to float roundoff.
    start, stop = Decimal(str(args.rmin)), Decimal(str(args.rmax))
    step = Decimal("0.1")
    args.distances = [float(start + i * step) for i in range(int((stop - start) // step) + 1)]
    return args


def build_geometry(name, distance, cov_workers, wfn="FCI", report_wfn=None, *,
                   frozen_core=True):
    wfn = str(wfn).upper()
    report_wfn = wfn if report_wfn is None else str(report_wfn).upper()
    if wfn not in WAVEFUNCTIONS or report_wfn not in WAVEFUNCTIONS:
        raise ValueError("Wavefunction must be FCI or CISD.")
    geometry = geometry_string(name, distance)
    molecule = tq.chemistry.Molecule(
        geometry=geometry, basis_set="sto3g", transformation="JordanWigner",
        backend="pyscf", frozen_core=frozen_core,
    )
    hamiltonian = molecule.make_hamiltonian()
    operator = hamiltonian.to_openfermion()
    n_qubits = int(count_qubits(operator))
    all_terms = make_terms(operator, n_qubits)
    estimated_workspace = covariance_workspace_gib(len(all_terms), n_qubits)
    if estimated_workspace > 16.0:
        raise MemoryError(f"Estimated covariance workspace {estimated_workspace:.3f} GiB "
                          "exceeds the 16 GiB driver limit.")
    sparse = get_sparse_operator(operator, n_qubits=n_qubits)
    optimization = build_wavefunction_context(
        molecule, operator, sparse, all_terms, n_qubits, cov_workers, wfn,
    )
    reporting = optimization if report_wfn == wfn else build_wavefunction_context(
        molecule, operator, sparse, all_terms, n_qubits, cov_workers, report_wfn,
    )
    return hamiltonian, optimization, reporting


def build_wavefunction_context(molecule, operator, sparse, all_terms, n_qubits,
                               cov_workers, wfn):
    energy, state = get_variance_wavefunction(
        molecule, operator, method=wfn, sparse_hamiltonian=sparse,
    )
    energy = clean_real(energy)
    state = np.asarray(state, dtype=complex).reshape(-1)
    residual = float(np.linalg.norm(sparse @ state - energy * state))
    if wfn == "FCI" and residual > 1.e-8:
        raise ValueError(f"Ground-state residual is too large: {residual}")
    electron_counts = np.array([i.bit_count() for i in range(state.size)])
    n_electrons = int(molecule.n_electrons)
    if np.sum(np.abs(state[electron_counts != n_electrons]) ** 2) > 1.e-8:
        raise ValueError(f"State is outside the {n_electrons}-active-electron sector.")
    print(f"  Recomputing {wfn} covariances: "
          f"qubits={n_qubits}, terms={len(all_terms) - 1}...", flush=True)
    covariances, expectations = build_covariance_dictionary(
        all_terms, state, n_qubits, max_workers=cov_workers, chunksize=128,
    )
    if abs(hamiltonian_expectation(all_terms, expectations) - energy) > 1.e-8:
        raise ValueError("Pauli-action energy does not match the wavefunction energy.")
    context = make_optimization_context(
        [term for term in all_terms if term.pauli_tuple], covariances,
    )
    return context, energy, residual


def validate_grouping(context, assignment, omega, *, partition=False):
    assignment = np.asarray(assignment, dtype=bool)
    problem = GroupingProblem(
        context.coefficients, context.covariance, context.compatible,
        n_groups=assignment.shape[1],
        max_overlap=1 if partition else assignment.shape[1],
    )
    if not is_feasible(problem, assignment) or np.any(~assignment.any(axis=0)):
        raise ValueError("Invalid coverage, commutation, or empty grouping fragment.")
    if (not np.all(np.isfinite(omega)) or np.any(np.abs(omega[~assignment]) > 1.e-12)
            or not np.allclose(omega.sum(axis=1), context.coefficients,
                               atol=1.e-9, rtol=1.e-9)):
        raise ValueError("Fragment coefficients do not reconstruct the Hamiltonian.")


def reference_partition(context, args):
    start = time.perf_counter()
    group_bound, _ = coloring_group_bound(context, random_seed=args.seed, augmentation=4)
    problem = GroupingProblem(
        context.coefficients, context.covariance, context.compatible,
        n_groups=group_bound, max_overlap=1, labels=list(range(context.n_terms)),
    )
    # This is the only MILP invocation. These settings match driver.py.
    solution = optimize_grouping(
        problem, method="milp", grouping_mode="non_overlapping",
        weight_heuristic="iterative_relative_std",
        heuristic_options={"sweeps": 3, "beta": 5.0, "delta": 1.e-12},
        solver_options={"time_limit": args.solver_time_limit_s,
                        "deadline": time.monotonic() + args.solver_time_limit_s,
                        "threads": args.cov_workers, "seed": args.seed,
                        "quiet": True},
        candidate_options={"random_partitions": 12, "max_candidate_groups": 40000,
                           "seed": args.seed, "group_bound": int(group_bound)},
        legacy_context=context,
    )
    assignment = np.asarray(solution.assignment, dtype=bool)
    assignment = assignment[:, assignment.any(axis=0)]
    validate_grouping(context, assignment, context.coefficients[:, None] * assignment,
                      partition=True)
    info = {"distance_angstrom": args.ropt, "status": solution.status,
            "runtime_s": time.perf_counter() - start, "group_bound": int(group_bound),
            "epsilon2M": float(solution.epsilon2M), "metadata": solution.metadata}
    return assignment, info


def transfer_partition(reference_context, reference_assignment, context, *,
                       new_terms="greedy", max_sweeps=100, verbose=False,
                       new_refine_all=False, refine_milp=False, timings=None):
    """Transfer memberships, insert new words, then apply requested refinement.

    The caller enables refine_milp only at non-reference geometries. It refines
    all terms, including when new words appear, using the target context.
    Return the final partition, new words, and partition before refinement.
    If supplied, timings records MILP-R refinement seconds, excluding transfer.
    """
    if timings is not None:
        timings["MILP-R"] = 0.0
    if new_terms not in ("greedy", "varsi-o", "varsi-or"):
        raise ValueError(f"Unknown new-term method: {new_terms}")
    reference_index = {term.pauli_tuple: i for i, term in enumerate(reference_context.terms)}
    assignment = np.zeros((context.n_terms, reference_assignment.shape[1]), dtype=bool)
    missing = []
    for i, term in enumerate(context.terms):
        if term.pauli_tuple in reference_index:
            assignment[i] = reference_assignment[reference_index[term.pauli_tuple]]
        else:
            missing.append(i)
    new_words = [context.terms[i].word for i in missing]
    print(f"  New Pauli operators found: {len(new_words)}", flush=True)
    # Absent (zero-coefficient) reference words need no measurements.
    assignment = assignment[:, assignment.any(axis=0)]
    if new_terms == "greedy":
        # Insert larger coefficients first; each decision uses this geometry's
        # covariance and minimizes the exact incremental epsilon^2 M.
        for i in sorted(missing, key=lambda i: -abs(context.coefficients[i])):
            omega = context.coefficients[:, None] * assignment
            variances, score, cost = measurement_objective(context, assignment, omega)
            candidates = []
            coefficient = context.coefficients[i]
            for group in range(assignment.shape[1]):
                if np.all(context.compatible[i, assignment[:, group]]):
                    variance = clean_variance(
                        variances[group]
                        + 2 * coefficient * (context.covariance[i] @ omega[:, group])
                        + coefficient**2 * context.covariance[i, i], tiny=1.e-8,
                    )
                    new_score = score - np.sqrt(variances[group]) + np.sqrt(variance)
                    candidates.append((float(new_score**2 - cost), group))
            if candidates:
                increase, group = min(candidates)
                assignment[i, group] = True
                destination = f"compatible group {group + 1}; delta epsilon^2 M={increase:.10g}"
            else:
                assignment = np.column_stack((assignment, np.zeros(context.n_terms, dtype=bool)))
                assignment[i, -1] = True
                destination = "new singleton group (no compatible group exists)"
            if verbose:
                print(f"  WARNING: new Pauli operator found: {context.terms[i].word}; "
                      f"added to {destination}", flush=True)
    elif missing:
        assignment = varsi_ordered(context, assignment, missing)
    initial_assignment = assignment.copy()
    refine_all = refine_milp or (new_refine_all and bool(missing))
    if refine_all or (missing and new_terms == "varsi-or"):
        if refine_milp:
            # Compare at this geometry, after insertion and before ICS.
            _, _, cost_before = measurement_objective(
                context, assignment, context.coefficients[:, None] * assignment,
            )
        refinement_start = time.perf_counter()
        assignment, accepted_moves = varsi_refine(
            context, assignment, movable_positions=None if refine_all else missing,
            max_sweeps=max_sweeps,
        )
        if timings is not None:
            timings["MILP-R"] = time.perf_counter() - refinement_start
        scope = "all terms" if refine_all else "new terms"
        print(f"  VarSI-OR: {accepted_moves} improving moves of {scope} "
              f"(maximum sweeps: {max_sweeps})", flush=True)
        if refine_milp:
            _, _, cost_after = measurement_objective(
                context, assignment, context.coefficients[:, None] * assignment,
            )
            gain = cost_before - cost_after
            savings = (f"{100 * gain / cost_before:.2f}%" if cost_before > 0
                       else "n/a (zero initial cost)")
            print(f"  MILP refinement gain (optimization wavefunction): epsilon^2 M {cost_before:.10g} -> "
                  f"{cost_after:.10g}; reduction={gain:.10g}; "
                  f"measurement savings={savings}", flush=True)
    if verbose and new_terms != "greedy":
        for i in missing:
            group = int(np.flatnonzero(assignment[i])[0])
            print(f"  WARNING: new Pauli operator found: {context.terms[i].word}; "
                  f"{new_terms} final group {group + 1}", flush=True)
    validate_grouping(context, assignment, context.coefficients[:, None] * assignment,
                      partition=True)
    return assignment, new_words, initial_assignment


def si_partition(hamiltonian, context):
    # Preserve BinaryHamiltonian's term order for equal-coefficient SI ties,
    # matching /home/ishume/VarSI/VarSI.py.
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


def evaluate(context, assignment):
    omega = context.coefficients[:, None] * assignment
    validate_grouping(context, assignment, omega, partition=True)
    return record_result(context, assignment, omega)


def record_result(context, assignment, omega):
    validate_grouping(context, assignment, omega)
    variances, _, cost = measurement_objective(context, assignment, omega)
    return {"epsilon2M": float(cost), "variances": list(variances),
            "groups": [[{"word": context.terms[i].word,
                         "coefficient": float(omega[i, g])}
                        for i in np.flatnonzero(assignment[:, g])]
                       for g in range(assignment.shape[1])]}


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def output_stem(molecule, ropt):
    reference_tag = str(float(ropt)).replace(".", "_")
    return f"{molecule.lower()}_geometry_scan_ropt_{reference_tag}"


def save_outputs(output, report, save_groups=False):
    methods = METHODS + (REFINED_METHODS if report["configuration"].get("refine_milp") else ())
    stem = output_stem(report["configuration"]["molecule"], report["reference"]["distance_angstrom"])
    (output / f"{stem}_results.json").write_text(
        json.dumps(report, indent=2, default=json_default, allow_nan=False) + "\n"
    )
    with (output / f"{stem}_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["distance_angstrom", "wavefunction", "energy",
                                                   "report_wavefunction", "report_energy", "new_pauli_count",
                                                   *methods])
        writer.writeheader()
        for row in report["geometries"]:
            writer.writerow({"distance_angstrom": row["distance_angstrom"],
                             "wavefunction": row["wavefunction"], "energy": row["energy"],
                             "report_wavefunction": row["report_wavefunction"],
                             "report_energy": row["report_energy"],
                             "new_pauli_count": row["new_pauli_count"],
                             **{m: row["methods"][m]["epsilon2M"] for m in methods}})
    if save_groups:
        wavefunction = report["configuration"]["wavefunction"]
        groupings = {
            "format_version": 1,
            "pauli_word_format": "OpenFermion strings, e.g. X0 Y2 Z3; zero-based qubit indices",
            "identity_excluded": True,
            "coefficient_convention": "Fragment coefficients; sum across groups reconstructs each nonidentity Hamiltonian coefficient",
            "configuration": report["configuration"],
            "reference": report["reference"],
            "geometries": report["geometries"],
        }
        (output / f"{stem}_{wavefunction}_groupings.json").write_text(
            json.dumps(groupings, indent=2, default=json_default, allow_nan=False) + "\n"
        )


def plot_results(output, rows, molecule, wfn=None, ropt=REFERENCE_DISTANCE,
                 refine_milp=False, report_wfn=None):
    fig, ax = plt.subplots(figsize=(7, 4.6), constrained_layout=True)
    labels = {"MILP+ICS": "MILP-ICS"}
    methods = METHODS + (REFINED_METHODS if refine_milp else ())
    for method, marker, style in zip(methods, ("o", "s", "^", "D", "v", "P"),
                                     ("-", "--", "-", "--", "-.", ":")):
        line, = ax.plot([row["distance_angstrom"] for row in rows],
                [row["methods"][method]["epsilon2M"] for row in rows],
                marker=marker, linestyle=style, markersize=4, label=labels.get(method, method))
        if method == "MILP":
            changed = [row for row in rows if row.get("new_pauli_count", 0) > 0]
            if changed:
                ax.scatter([row["distance_angstrom"] for row in changed],
                           [row["methods"]["MILP"]["epsilon2M"] for row in changed],
                           marker="o", s=150, facecolors="none", edgecolors=line.get_color(),
                           linewidths=1.5, zorder=4)
    if rows and min(row["distance_angstrom"] for row in rows) <= ropt <= max(
        row["distance_angstrom"] for row in rows
    ):
        ax.axvline(ropt, color="0.55", linestyle=":", linewidth=1)
    ax.set_xlabel(r"$R$ ($\mathrm{\AA}$)", fontsize=18)
    ax.set_ylabel(r"$\varepsilon^2 M$", fontsize=18)
    if report_wfn is not None:
        ax.set_title(f"Optimization: {wfn}; reporting: {report_wfn}")
    ax.grid(alpha=0.2)
    ax.legend(loc="best")
    stem = output_stem(molecule, ropt)
    if wfn is not None:
        stem += f"_{wfn}"
    if report_wfn is not None:
        stem += f"_report_{report_wfn}"
    for extension in ("png", "svg"):
        fig.savefig(output / f"{stem}.{extension}", dpi=200)
    plt.close(fig)


def print_scan_gains(rows, refine_milp=False):
    """Compare total measurement costs for equal precision at each geometry."""
    methods = METHODS + (REFINED_METHODS if refine_milp else ())
    totals = {method: sum(row["methods"][method]["epsilon2M"] for row in rows)
              for method in methods}
    print("Overall gains across the scan (summed epsilon^2 M; equal precision at every geometry):",
          flush=True)
    print("  Totals: " + "  ".join(f"{method}={totals[method]:.10g}" for method in methods),
          flush=True)
    for method in methods[2:]:
        for baseline in ("SI", "SI-ICS"):
            cost, baseline_cost = totals[method], totals[baseline]
            savings = (f"{100 * (1 - cost / baseline_cost):.2f}%"
                       if baseline_cost > 0 else "n/a (zero baseline cost)")
            ratio = (f"{baseline_cost / cost:.6g}x" if cost > 0 else
                     "infinite" if baseline_cost > 0 else "n/a (both costs zero)")
            print(f"  {method} vs {baseline}: measurement savings={savings}; "
                  f"baseline/method cost={ratio}", flush=True)


def main(argv=None):
    args = parse_args(argv)
    frozen_core = not args.all_electron
    output = Path.cwd()
    np.random.seed(args.seed)
    print(f"Building reference {args.molecule} at R={args.ropt:g} A ({args.wfn})...", flush=True)
    print(f"Optimization wavefunction={args.wfn}; reporting wavefunction={args.report_wfn}; "
          f"frozen_core={frozen_core}",
          flush=True)
    reference = build_geometry(args.molecule, args.ropt, args.cov_workers,
                               wfn=args.wfn, report_wfn=args.report_wfn, frozen_core=frozen_core)
    reference_context = reference[1][0]
    reference_report_context = reference[2][0]
    print("Running the single non-overlapping MILP optimization...", flush=True)
    assignment, info = reference_partition(reference_context, args)
    info["optimization_epsilon2M"] = info["epsilon2M"]
    info["epsilon2M"] = evaluate(reference_report_context, assignment)["epsilon2M"]
    info["wavefunction"] = args.wfn
    info["report_wavefunction"] = args.report_wfn
    print(f"Reference status={info['status']}; groups={assignment.shape[1]}; "
          f"epsilon2M(wfn={args.report_wfn})={info['epsilon2M']:.12g}; "
          f"MILP time at R={args.ropt:g} A: {info['runtime_s']:.6f} s", flush=True)
    print("Stage timings exclude shared chemistry/covariances and final reporting; "
          "ICS times exclude initial grouping, and MILP-R times refinement only.", flush=True)
    report = {"configuration": {"molecule": args.molecule,
              "basis": "sto3g", "mapping": "JordanWigner",
              "frozen_core": frozen_core,
              "orbitals": ("Tequila default active space and native orbital ordering at each geometry"
                           if frozen_core else "All orbitals and native orbital ordering at each geometry"),
              "reference_geometry_angstrom": geometry_string(args.molecule, args.ropt),
              "rmin_angstrom": args.rmin, "rmax_angstrom": args.rmax,
              "ropt_angstrom": args.ropt, "spacing_angstrom": 0.1,
              "wavefunction": args.wfn, "condition": "fc",
              "report_wavefunction": args.report_wfn,
              "new_terms": args.new_terms, "max_sweeps": args.max_sweeps,
              "new_refine_all": args.new_refine_all, "refine_milp": args.refine_milp,
              "ics_iterations": args.ics_iterations, "seed": args.seed,
              "solver_time_limit_s": args.solver_time_limit_s,
              "cov_workers": args.cov_workers, "milp_optimization_calls": 1},
              "reference": {**info, "partition": evaluate(reference_report_context, assignment)},
              "geometries": []}
    save_outputs(output, report, save_groups=args.save_groups)
    for distance in sorted(set(args.distances)):
        print(f"Evaluating d={distance:.3f} A...", flush=True)
        hamiltonian, optimization, reporting = (
            reference if distance == args.ropt else
            build_geometry(args.molecule, distance, args.cov_workers,
                           wfn=args.wfn, report_wfn=args.report_wfn, frozen_core=frozen_core)
        )
        context, energy, residual = optimization
        report_context, report_energy, report_residual = reporting
        timings = {}
        if distance == args.ropt:
            timings["MILP"] = info["runtime_s"]
        transferred, new_words, before_refinement = transfer_partition(
            reference_context, assignment, context,
            new_terms=args.new_terms, max_sweeps=args.max_sweeps,
            verbose=args.verbose,
            new_refine_all=args.new_refine_all,
            refine_milp=args.refine_milp and distance != args.ropt,
            timings=timings if args.refine_milp else None,
        )
        si_start = time.perf_counter()
        si = si_partition(hamiltonian, context)
        timings["SI"] = time.perf_counter() - si_start
        results = {}
        milp = before_refinement if args.refine_milp else transferred
        initializations = [("SI", si, "SI-ICS"), ("MILP", milp, "MILP+ICS")]
        if args.refine_milp:
            initializations.append(("MILP-R", transferred, "MILP-R-ICS"))
        for base, initial, refined in initializations:
            results[base] = evaluate(report_context, initial)
            ics_start = time.perf_counter()
            support, omega, _ = optimize_from_nonoverlapping_groups(
                context, initial, n_iter=args.ics_iterations, condition="fc",
            )
            timings[refined] = time.perf_counter() - ics_start
            results[refined] = record_result(report_context, support, omega)
        report["geometries"].append({"distance_angstrom": distance,
                                    "geometry_angstrom": geometry_string(args.molecule, distance),
                                    "energy": energy, "wavefunction": args.wfn,
                                    "report_wavefunction": args.report_wfn,
                                    "report_energy": report_energy,
                                    "report_state_residual": report_residual,
                                    "n_qubits": len(context.terms[0].ops),
                                    "new_pauli_count": len(new_words), "new_pauli_words": new_words,
                                    "state_residual": residual, "methods": results,
                                    "timings_s": timings})
        print(f"  epsilon^2 M(wfn={args.report_wfn}): " + "  ".join(f"{m}={result['epsilon2M']:.10g}"
                                            for m, result in results.items()),
              flush=True)
        print("  Stage times (s): " + "  ".join(
            f"{method.replace('MILP+ICS', 'MILP-ICS')}={timings[method]:.6f}"
            for method in results if method in timings), flush=True)
        save_outputs(output, report, save_groups=args.save_groups)
    print_scan_gains(report["geometries"], refine_milp=args.refine_milp)
    plot_results(output, report["geometries"], args.molecule,
                 wfn=args.wfn if args.wfn_provided or args.report_wfn != args.wfn else None,
                 ropt=args.ropt, refine_milp=args.refine_milp,
                 report_wfn=args.report_wfn if args.report_wfn != args.wfn else None)
    print(f"Saved results and plot to {output}", flush=True)
    if args.save_groups:
        print(f"Saved reusable groupings to "
              f"{output / (output_stem(args.molecule, args.ropt) + '_' + args.wfn + '_groupings.json')}",
              flush=True)
    return report


if __name__ == "__main__":
    main()
