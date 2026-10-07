"""Run SI-ICS long controls at R=1 angstrom, optimizing CISD and reporting FCI.

Run ``python -u long_si_ics.py --cov-workers 12`` for all eight molecules, or
pass molecule names (for example, ``H4 LiH``) to run a subset. Each SI partition
receives 100 ICS iterations from its original coefficients.
"""

import argparse
import csv
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

from driver_o_clique import (
    build_molecular_data,
    build_wavefunction_context,
    molecule_name,
    positive_int,
    serialize_grouping,
    write_json_report,
)
from bin_mopt.overlapping import (
    covariance_measurement_cost,
    dense_ics,
    expand_partition_for_ics,
    validate_overlap,
)


MOLECULES = ("H4", "LiH", "BeH2", "H2O", "N2", "H6", "NH3", "MgO")
DISTANCE_ANGSTROM = 1.0
OPTIMIZATION_WAVEFUNCTION = "CISD"
REPORT_WAVEFUNCTION = "FCI"
ICS_ITERATIONS = 100


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("molecules", nargs="*", type=molecule_name,
                        metavar="MOLECULE", help="Default: " + ", ".join(MOLECULES))
    parser.add_argument("--cov-workers", type=positive_int, default=12,
                        help="Covariance workers for both states (default: 12)")
    parser.add_argument("--threads", type=positive_int, default=1,
                        help="Numerical-library threads, as in O-clique (default: 1)")
    parser.add_argument("--output-dir", type=Path, default=Path("long_si_ics_results"))
    args = parser.parse_args(argv)
    args.molecules = list(dict.fromkeys(args.molecules or MOLECULES))
    args.output_dir = args.output_dir.resolve()
    return args


def run_molecule(name, *, cov_workers=12, threads=1):
    """Optimize on CISD before constructing the FCI reporting context."""
    started = time.perf_counter()
    context, si, configuration, wavefunction_inputs = build_molecular_data(
        name, distance=DISTANCE_ANGSTROM, threads=threads, frozen_core=True,
        wfn=OPTIMIZATION_WAVEFUNCTION, cov_workers=cov_workers,
    )
    preparation_s = time.perf_counter() - started
    initial_omega = context.coefficients[:, None] * si
    ics_start = time.perf_counter()
    support = expand_partition_for_ics(context, si)
    print(f"  SI-ICS long control: {ICS_ITERATIONS} QR iterations on "
          f"{support.shape[1]} fixed groups...", flush=True)
    omega, fractions, ics_info = dense_ics(
        context, support, initial_omega=initial_omega,
        n_iter=ICS_ITERATIONS, linear_solver="qr",
    )
    ics_s = time.perf_counter() - ics_start
    if ics_info["n_free_coefficients"] and ics_info["completed_iterations"] != ICS_ITERATIONS:
        raise RuntimeError("SI-ICS did not complete all 100 requested iterations.")
    validation = validate_overlap(context, support, omega, si.shape[1])
    print(f"  SI-ICS(CISD): epsilon2M={ics_info['epsilon2M']:.12g}, "
          f"completed_iterations={ics_info['completed_iterations']}, "
          f"runtime_s={ics_s:.6f}", flush=True)

    # Reporting is deliberately delayed until optimization is complete. Only
    # the covariance matrix changes; the support and coefficients stay fixed.
    reporting_start = time.perf_counter()
    report_context, report_state = build_wavefunction_context(
        *wavefunction_inputs, REPORT_WAVEFUNCTION,
    )
    if ([term.word for term in context.terms] !=
            [term.word for term in report_context.terms] or
            not np.array_equal(context.coefficients, report_context.coefficients)):
        raise ValueError("Optimization and reporting Hamiltonians differ.")
    results = {}
    for method, selected_support, coefficients in (
        ("SI", si, initial_omega), ("SI-ICS", support, omega),
    ):
        result = serialize_grouping(report_context, selected_support, coefficients)
        result.update(
            optimization_epsilon2M=covariance_measurement_cost(context, coefficients),
            optimization_wavefunction=OPTIMIZATION_WAVEFUNCTION,
            report_wavefunction=REPORT_WAVEFUNCTION,
        )
        results[method] = result
    reporting_s = time.perf_counter() - reporting_start
    configuration.update(
        optimization_wavefunction=OPTIMIZATION_WAVEFUNCTION,
        report_wavefunction=REPORT_WAVEFUNCTION,
        report_energy=report_state["energy"], report_state=report_state,
        ics_iterations=ICS_ITERATIONS, linear_solver="qr", threads=threads,
        initialization="SI partition coefficients",
        support="standard ICS expansion, fixed for all iterations",
    )
    report = {
        "configuration": configuration,
        "results": results,
        "optimization": {
            **ics_info,
            "wavefunction": OPTIMIZATION_WAVEFUNCTION,
            "history_wavefunction": OPTIMIZATION_WAVEFUNCTION,
            "shot_fractions": fractions.tolist(),
        },
        "validation": validation,
        "timings": {
            "preparation_s": preparation_s,
            "si_ics_s": ics_s,
            "optimization_whole_procedure_s": preparation_s + ics_s,
            "reporting_s": reporting_s,
            "total_s": time.perf_counter() - started,
            "scope": "Total includes FCI reporting; excludes imports and result writing.",
        },
        "threadpools": threadpool_info(),
    }
    print(f"  {name}: SI-ICS epsilon2M(FCI)={results['SI-ICS']['epsilon2M']:.12g}, "
          f"groups={support.shape[1]}, total_s={report['timings']['total_s']:.3f}", flush=True)
    return report


def summary_row(report):
    configuration = report["configuration"]
    si, ics = report["results"]["SI"], report["results"]["SI-ICS"]
    return {
        "molecule": configuration["molecule"],
        "distance_angstrom": configuration["distance_angstrom"],
        "optimization_wavefunction": configuration["optimization_wavefunction"],
        "report_wavefunction": configuration["report_wavefunction"],
        "covariance_workers": configuration["covariance_workers"],
        "threads": configuration["threads"],
        "ics_iterations": configuration["ics_iterations"],
        "completed_iterations": report["optimization"]["completed_iterations"],
        "n_qubits": configuration["n_qubits"],
        "n_terms": configuration["n_terms"],
        "n_groups": ics["n_groups"],
        "si_epsilon2M_CISD": si["optimization_epsilon2M"],
        "si_ics_epsilon2M_CISD": ics["optimization_epsilon2M"],
        "si_epsilon2M_FCI": si["epsilon2M"],
        "si_ics_epsilon2M_FCI": ics["epsilon2M"],
        "si_ics_s": report["timings"]["si_ics_s"],
        "total_s": report["timings"]["total_s"],
    }


def save_summary(output_dir):
    """Include completed molecules already in this directory when adding a subset."""
    rows = []
    for name in MOLECULES:
        path = output_dir / f"{name.lower()}.json"
        if path.exists():
            rows.append(summary_row(json.loads(path.read_text())))
    write_json_report(output_dir / "summary.json", rows)
    if rows:
        with (output_dir / "summary.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return rows


def main(argv=None):
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"SI-ICS: R=1 angstrom, frozen-core STO-3G/Jordan-Wigner, "
          f"CISD optimization, FCI reporting, {ICS_ITERATIONS} iterations; "
          f"cov_workers={args.cov_workers}, threads={args.threads}", flush=True)
    rows = []
    with threadpool_limits(limits=args.threads):
        for name in args.molecules:
            # Tequila chemistry files are scratch; retain only explicit reports.
            original_directory = Path.cwd()
            with tempfile.TemporaryDirectory(prefix=f"long-si-ics-{name.lower()}-") as scratch:
                try:
                    os.chdir(scratch)
                    report = run_molecule(name, cov_workers=args.cov_workers, threads=args.threads)
                finally:
                    os.chdir(original_directory)
            output = args.output_dir / f"{name.lower()}.json"
            write_json_report(output, report)
            rows = save_summary(args.output_dir)
            print(f"  Saved {output}", flush=True)
    print("\nMolecule  Groups       SI-ICS epsilon2M(FCI)  Iterations", flush=True)
    for row in rows:
        print(f"{row['molecule']:<8} {row['n_groups']:>6} "
              f"{row['si_ics_epsilon2M_FCI']:>27.12g} {row['completed_iterations']:>11}")
    return rows


if __name__ == "__main__":
    main()
