"""Compare Pauli variance and covariance distributions for four fixed systems.

Run on a machine with sufficient memory for exact lattice ground states:
  python covariance_comparison_paper.py --max-memory-gib 0
  python covariance_comparison_paper.py --max-memory-gib 0 --break-y 0.4 0.9

The memory limit is the same core-workspace guard as covariance_plot.py;
0 disables it, and the estimate excludes eigensolver and worker temporaries.
Use --dry-run to build Hamiltonians and report estimates without solving.

Each system's density is independently peak-normalized.
Saves separate covariance and variance comparisons as PNG and SVG; --break-y
also saves broken-axis copies. H2O uses the default geometry, STO-3G, frozen core,
and FCI reference from covariance_plot.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import covariance_plot as base


# Keep the supplied model arguments together so the original parser determines
# all unspecified Hamiltonian and reference-state defaults.
SYSTEMS = (
    (r"K-H-$\Gamma$", "#0072B2", (
        "kitaev", "23", "--shape", "4", "3", "--kx", "-1", "--ky", "-0.8",
        "--kz", "-1.2", "--j", "0.2", "--gamma", "0.4", "--hx", "0.2",
        "--hy", "0.1", "--hz", "0.3", "--bond-disorder", "0.5",
        "--field-disorder", "0.2", "--disorder-seed", "0", "--seed", "7")),
    ("XYZ", "#D55E00", (
        "xyz", "25", "--lattice", "square", "--j2", "0.5", "--hx", "0.2",
        "--hy", "0.1", "--hz", "0.3", "--bond-disorder", "0.5",
        "--field-disorder", "0.3", "--disorder-seed", "0")),
    ("FH", "#009E73", ("fermi-hubbard", "24", "--t", "1", "--u", "4")),
    (r"H$_2$O", "#CC79A7", ("H2O",)),
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-directory", "--output-dir", type=Path,
                        default=Path("covariance_comparison_plots"))
    parser.add_argument("--condition", choices=("fc", "qwc"), default="fc")
    parser.add_argument("--cov-workers", type=int, default=1)
    parser.add_argument("--bw-adjust", type=base.finite_float, default=0.8)
    parser.add_argument("--variance-bw-adjust", type=base.finite_float, default=0.2)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--no-grid", action="store_true")
    parser.add_argument("--break-y", nargs=2, type=base.finite_float,
                        metavar=("LOW", "HIGH"),
                        help="Also save copies omitting LOW to HIGH (0 < LOW < HIGH < 1).")
    parser.add_argument("--max-memory-gib", type=base.finite_float, default=16.,
                        help="Core covariance workspace limit per system; 0 disables (default: 16).")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if (args.cov_workers < 1 or args.dpi < 1 or args.bw_adjust <= 0
            or args.variance_bw_adjust <= 0 or args.max_memory_gib < 0):
        parser.error("Workers, DPI and bandwidth must be positive; memory limit must be nonnegative.")
    if args.break_y is not None and not (0 < args.break_y[0] < args.break_y[1] < 1):
        parser.error("--break-y requires 0 < LOW < HIGH < 1.")
    return args


def collect_samples(system_argv, args):
    """Compute one system at a time, retaining only its small sample arrays."""
    system = base.parse_args(list(system_argv))
    if system.molecule:
        _, (molecule, _, _, _, operator) = base.get_molecule(
            system.molecule, frozen_core=not system.all_electron)
        n_qubits = int(base.count_qubits(operator))
    else:
        model = base.build_lattice(
            system.system, system.n_qubits, lattice=system.lattice, shape=system.shape,
            bond_disorder=system.bond_disorder, field_disorder=system.field_disorder,
            disorder_seed=system.disorder_seed, **system.parameters)
        operator, n_qubits = model.qubit_operator, model.n_qubits
    terms = [term for term in base.make_terms(operator, n_qubits) if term.pauli_tuple]
    workspace = base.covariance_workspace_gib(len(terms), n_qubits)
    print(f"Qubits={n_qubits}; nonidentity terms={len(terms)}; "
          f"core covariance workspace={workspace:.3f} GiB "
          "(excludes diagonalization and worker temporaries)", flush=True)
    if args.dry_run:
        return None
    if args.max_memory_gib and workspace > args.max_memory_gib:
        raise MemoryError(
            f"Core covariance workspace {workspace:.3f} GiB exceeds --max-memory-gib "
            f"{args.max_memory_gib:g}. Raise the limit only with sufficient resources.")
    if system.molecule:
        energy, state = base.get_variance_wavefunction(molecule, operator, method=system.wfn)
        print(f"Reference={system.wfn}; energy={energy}", flush=True)
    else:
        state, reference = base.exact_ground_state(
            model, particles=system.particles, spin_up=system.spin_up, seed=system.seed)
        print(f"Reference={reference}", flush=True)
    covariances, _ = base.exact_covariances(
        terms, state, n_qubits, max_workers=args.cov_workers)
    return {
        "covariance": base.covariance_samples(terms, covariances, args.condition),
        "variance": base.variance_samples(terms, covariances),
    }


def plot_comparison(datasets, args, *, kind, broken_y=False):
    if kind not in ("variance", "covariance"):
        raise ValueError(f"Unknown distribution: {kind}")
    x_limits = (0, 1) if kind == "variance" else (-1, 1)
    np, plt = base.np, base.plt
    if broken_y:
        low, high = args.break_y
        figure, axes = plt.subplots(
            2, 1, sharex=True, figsize=(7, 4.5),
            gridspec_kw={"height_ratios": (1 - high, low)})
    else:
        figure, axis = plt.subplots(figsize=(7, 4.5))
        axes = (axis,)
    smoothing = args.variance_bw_adjust if kind == "variance" else args.bw_adjust
    for label, color, samples_by_kind in datasets:
        samples = samples_by_kind[kind]
        if not samples.size:
            axes[0].plot([], [], color=color, label=f"{label} (no samples)")
        elif samples.size < 2 or np.ptp(samples) <= 1.e-12:
            for axis in axes:
                axis.axvline(float(samples.mean()), color=color, label=label)
        else:
            kde = base.gaussian_kde(
                samples, bw_method=lambda estimator: estimator.scotts_factor() * smoothing)
            x = np.linspace(*x_limits, 2049)
            density = kde(x)
            density /= density.max()
            for axis in axes:
                axis.plot(x, density, color=color, label=label, zorder=3)
                axis.fill_between(x, density, color=color, alpha=0.35)
    axes[0].legend()
    axes[-1].set_xlabel(r"Variance $C_{ii}$" if kind == "variance" else r"Covariance $C_{ij}$")
    for axis in axes:
        axis.set_yticks(np.arange(11) / 10)
        axis.set(xlim=x_limits, ylim=(0, 1))
        axis.tick_params(direction="out")
        if not args.no_grid:
            axis.grid(alpha=0.2)
    if broken_y:
        upper, lower = axes
        low, high = args.break_y
        ticks = np.arange(11) / 10
        upper.set_yticks(ticks[ticks >= high])
        lower.set_yticks(ticks[ticks <= low])
        upper.set_ylim(high, 1)
        lower.set_ylim(0, low)
        upper.spines.bottom.set_visible(False)
        lower.spines.top.set_visible(False)
        upper.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
        if not args.no_grid:
            upper.xaxis.tick_top()
        upper.tick_params(axis="x", which="both", top=not args.no_grid, labeltop=False)
        lower.xaxis.tick_bottom()
        for axis, y in ((upper, 0), (lower, 1)):
            axis.plot([0, 1], [y, y], transform=axis.transAxes,
                      marker=[(-1, -0.5), (1, 0.5)], markersize=12,
                      linestyle="none", color="black", markeredgecolor="black",
                      markeredgewidth=1, clip_on=False)
        figure.supylabel("Relative density")
    else:
        axes[0].set_ylabel("Relative density")
    figure.tight_layout()
    if broken_y:
        figure.subplots_adjust(hspace=0.15)
    args.output_directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for extension in ("png", "svg"):
        suffix = "_break_y" if broken_y else ""
        path = args.output_directory / f"covariance_comparison_paper_{args.condition}_{kind}{suffix}.{extension}"
        figure.savefig(path, dpi=args.dpi, bbox_inches="tight")
        paths.append(path)
        print(f"Saved {path}", flush=True)
    plt.close(figure)
    if args.break_y is not None and not broken_y:
        paths.extend(plot_comparison(datasets, args, kind=kind, broken_y=True))
    return paths


def main(argv=None):
    args = parse_args(argv)
    datasets = []
    for label, color, system_argv in SYSTEMS:
        print(f"Preparing {label}: {' '.join(system_argv)}", flush=True)
        samples = collect_samples(system_argv, args)
        if samples is not None:
            datasets.append((label, color, samples))
    if args.dry_run:
        return []
    paths = []
    for kind in ("variance", "covariance"):
        paths.extend(plot_comparison(datasets, args, kind=kind))
    return paths


if __name__ == "__main__":
    main()
