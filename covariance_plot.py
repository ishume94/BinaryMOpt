"""Plot off-diagonal covariances and diagonal variances of Hamiltonian Paulis.

Use the lattice driver's MODEL N_QUBITS syntax and Hamiltonian/state flags:
  python covariance_plot.py kitaev 23 --shape 4 3 --kx -1 --ky -0.8 --kz -1.2 \
      --j 0.2 --gamma 0.4 --hx 0.2 --hy 0.1 --hz 0.3 \
      --bond-disorder 0.5 --field-disorder 0.2 --disorder-seed 0 --seed 7
  python covariance_plot.py xyz 25 --lattice square --j2 0.5 \
      --hx 0.2 --hy 0.1 --hz 0.3 --bond-disorder 0.5 \
      --field-disorder 0.3 --disorder-seed 0
  python covariance_plot.py fermi-hubbard 24 --t 1 --u 4 
  python covariance_plot.py LiH --wfn CISD
  python covariance_plot.py H2O --distance 1.2

Without --distance, molecules use bin_mopt.hamiltonians geometries. With it,
use driver_geometry_scan geometries (angstrom). Frozen core remains the default.
Lattices use the exact seeded ground state from driver_lattice, including its
fermionic occupation-sector defaults. No grouping or optimization is performed.

Each distinct nonidentity commuting pair appears once, including zero covariance.
C_ij = <P_i P_j> - <P_i><P_j> uses unit Pauli words, without coefficient weights
or variance normalization. Save separate covariance and variance KDEs as PNG
and SVG, scaling each plotted maximum to one (relative density, not unit-area
probability density), with fixed x limits [-1, 1]. The variance plot includes
one diagonal entry C_ii = 1 - <P_i>**2 per nonidentity term, including zeros.
Gaussian KDE uses Scott's bandwidth multiplied by 0.8 for covariances and
0.2 for variances (adjustable separately with --bw-adjust and
--variance-bw-adjust).
Use --break-y LOW HIGH (for example, --break-y 0.4 0.9) to also save
broken-axis copies omitting that y interval, while retaining the standard plots.
The default is full commutation; --condition qwc selects its QWC subset.

The existing exact routines allocate full-register arrays even for fermionic
sectors. --dry-run reports the core covariance workspace before diagonalization;
it excludes eigensolver/sparse-Hamiltonian and worker temporary memory. The
16 GiB default guard may be raised explicitly; 0 disables it. A 24-qubit exact
calculation requires very large resources with this backend.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/bin_mopt_covariance_plot_mpl")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from openfermion import count_qubits
from scipy.stats import gaussian_kde

from bin_mopt.hamiltonians import MOLECULES, get_molecule
from bin_mopt.lattice import DEFAULT_PARAMETERS, FERMION_MODELS, build_lattice, model_name
from bin_mopt.utils import (
    covariance_workspace_gib, get_variance_wavefunction, make_terms,
    terms_fully_commute,
)
from driver_lattice import exact_covariances, exact_ground_state, output_stem


def finite_float(value):
    result = float(value)
    if not np.isfinite(result):
        raise argparse.ArgumentTypeError("must be finite")
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("system", help="Molecule name or lattice model (same names as the drivers).")
    parser.add_argument("n_qubits", nargs="?", type=int, help="Required for lattice models only.")
    parser.add_argument("--lattice", choices=("chain", "square", "rectangle", "honeycomb", "lieb"))
    parser.add_argument("--shape", nargs=2, type=int, metavar=("NX", "NY"))
    parameter_names = sorted({key for values in DEFAULT_PARAMETERS.values() for key in values})
    for name in parameter_names:
        parser.add_argument("--" + name, type=finite_float)
    parser.add_argument("--bond-disorder", type=finite_float, default=0.)
    parser.add_argument("--field-disorder", type=finite_float, default=0.)
    parser.add_argument("--disorder-seed", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7, help="Lattice ground-state selection seed (default: 7).")
    parser.add_argument("--particles", type=int)
    parser.add_argument("--spin-up", type=int)
    parser.add_argument("--distance", type=finite_float, help="Molecular bond distance in angstrom.")
    parser.add_argument("--wfn", type=str.upper, choices=("FCI", "CISD", "HF"), default=None)
    parser.add_argument("--all-electron", action="store_true")
    parser.add_argument("--condition", type=str.lower, choices=("fc", "qwc"), default="fc")
    parser.add_argument("--cov-workers", type=int, default=1)
    parser.add_argument("--output-directory", "--output-dir", type=Path, default=Path("covariance_plots"))
    parser.add_argument("--bw-adjust", type=finite_float, default=0.8,
                        help="Gaussian covariance KDE: Scott bandwidth multiplier (default: 0.8).")
    parser.add_argument("--variance-bw-adjust", type=finite_float, default=0.2,
                        help="Gaussian variance KDE: Scott bandwidth multiplier (default: 0.2).")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--no-grid", action="store_true")
    parser.add_argument("--break-y", nargs=2, type=finite_float, metavar=("LOW", "HIGH"),
                        help="Also save broken-y plots, omitting LOW to HIGH (0 < LOW < HIGH < 1).")
    parser.add_argument("--max-memory-gib", type=finite_float, default=16., help="Core covariance limit; 0 disables.")
    parser.add_argument("--dry-run", action="store_true", help="Build Hamiltonian and estimate memory only.")
    args = parser.parse_args(argv)
    if args.break_y is not None and not (0 < args.break_y[0] < args.break_y[1] < 1):
        parser.error("--break-y requires 0 < LOW < HIGH < 1.")
    args.parameters = {name: getattr(args, name) for name in parameter_names if getattr(args, name) is not None}
    molecules = {name.casefold(): name for name in MOLECULES}
    args.molecule = molecules.get(args.system.casefold())
    if (args.cov_workers < 1 or args.dpi < 1 or args.bw_adjust <= 0
            or args.variance_bw_adjust <= 0 or args.max_memory_gib < 0):
        parser.error("Workers, DPI and bandwidth must be positive; memory limit must be nonnegative.")
    if args.seed < 0 or args.disorder_seed < 0 or min(args.bond_disorder, args.field_disorder) < 0:
        parser.error("Seeds and disorder strengths must be nonnegative.")
    if args.molecule:
        if (args.n_qubits is not None or args.lattice or args.shape or args.parameters
                or args.bond_disorder or args.field_disorder or args.disorder_seed
                or args.particles is not None or args.spin_up is not None):
            parser.error("Lattice options cannot be used with a molecule.")
        if args.distance is not None and (args.distance <= 0 or args.molecule == "H2Os"):
            parser.error("--distance must be positive and requires a geometry-scan molecule (not H2Os).")
        args.wfn = args.wfn or "FCI"
    else:
        try:
            args.system = model_name(args.system)
        except ValueError as error:
            parser.error(str(error) + "; molecules: " + ", ".join(MOLECULES))
        if args.n_qubits is None or args.n_qubits < 4:
            parser.error("Lattices require N_QUBITS >= 4.")
        if args.distance is not None or args.wfn is not None or args.all_electron:
            parser.error("--distance, --wfn and --all-electron apply only to molecules.")
        if args.parameters.keys() - DEFAULT_PARAMETERS[args.system].keys():
            parser.error("Parameters for this model: " + ", ".join(DEFAULT_PARAMETERS[args.system]))
        if args.shape and min(args.shape) < 1:
            parser.error("Shape dimensions must be positive.")
        if args.system in FERMION_MODELS:
            sites = args.n_qubits // 2
            args.particles = sites if args.particles is None else args.particles
            args.spin_up = (args.particles + 1) // 2 if args.spin_up is None else args.spin_up
            if args.n_qubits % 2 or not (0 <= args.spin_up <= sites and 0 <= args.particles - args.spin_up <= sites):
                parser.error("Fermionic models require even qubit counts and valid up/down occupations.")
        elif args.particles is not None or args.spin_up is not None:
            parser.error("Particle/spin counts apply only to fermionic models.")
    return args


def covariance_samples(terms, covariances, condition="fc"):
    """Select unique off-diagonal commuting pairs, retaining exact zeros."""
    values = []
    for i, left in enumerate(terms):
        if not left.pauli_tuple:
            continue
        for right in terms[i + 1:]:
            if not right.pauli_tuple or not terms_fully_commute(left, right):
                continue
            if condition == "qwc" and any(a != "I" and b != "I" and a != b
                                          for a, b in zip(left.ops, right.ops)):
                continue
            value = complex(covariances[left.index, right.index])
            if not np.isfinite(value) or abs(value.imag) > 1.e-8:
                raise ValueError(f"Non-real or non-finite covariance: {value}")
            values.append(value.real)
    return np.asarray(values, dtype=float)


def variance_samples(terms, covariances):
    """Select one diagonal variance per nonidentity Pauli, including zeros."""
    values = []
    for term in terms:
        if not term.pauli_tuple:
            continue
        value = complex(covariances[term.index, term.index])
        if not np.isfinite(value) or abs(value.imag) > 1.e-8:
            raise ValueError(f"Non-real or non-finite variance: {value}")
        if value.real < -1.e-8 or value.real > 1 + 1.e-8:
            raise ValueError(f"Pauli variance outside [0, 1]: {value}")
        values.append(float(np.clip(value.real, 0, 1)))
    return np.asarray(values, dtype=float)


def plot_distribution(samples, title, stem, args, *, kind="covariance", broken_y=False):
    if kind not in ("covariance", "variance"):
        raise ValueError(f"Unknown distribution: {kind}")
    variance = kind == "variance"
    count_label = "terms" if variance else "pairs"
    if broken_y:
        low, high = args.break_y
        figure, axes = plt.subplots(
            2, 1, sharex=True, figsize=(7, 4.5),
            gridspec_kw={"height_ratios": (1 - high, low)})
    else:
        figure, axis = plt.subplots(figsize=(7, 4.5))
        axes = (axis,)
    if not samples.size:
        message = "No nonidentity terms" if variance else "No distinct commuting pairs"
        axes[0].text(0.5, 0.5, message, ha="center", transform=axes[0].transAxes)
    elif samples.size < 2 or np.ptp(samples) <= 1.e-12:
        for axis in axes:
            axis.axvline(float(samples.mean()), label=f"Constant {kind} (KDE undefined)")
        axes[0].legend()
    else:
        smoothing = args.variance_bw_adjust if variance else args.bw_adjust
        kde = gaussian_kde(samples, bw_method=lambda estimator: estimator.scotts_factor() * smoothing)
        x = np.linspace(-1, 1, 2049)
        density = kde(x)
        density /= density.max()
        for axis in axes:
            axis.plot(x, density, color="#3D64B6")
            axis.fill_between(x, density, color="#3D64B6", alpha=0.35)
    xlabel = r"Variance $C_{ii}$" if variance else r"Covariance $C_{ij}$"
    selection = "Diagonal" if variance else args.condition.upper()
    axes[0].set_title(f"{title} | {selection} | {samples.size:,} {count_label}")
    axes[-1].set_xlabel(xlabel)
    for axis in axes:
        axis.set_yticks(np.arange(11) / 10)
        axis.set(xlim=(-1, 1), ylim=(0, 1))
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
        path = args.output_directory / f"{stem}_{args.condition}_{kind}{suffix}.{extension}"
        figure.savefig(path, dpi=args.dpi, bbox_inches="tight")
        paths.append(path)
        print(f"Saved {path}", flush=True)
    plt.close(figure)
    if args.break_y is not None and not broken_y:
        paths.extend(plot_distribution(samples, title, stem, args, kind=kind, broken_y=True))
    return paths


def main(argv=None):
    args = parse_args(argv)
    if args.molecule:
        if args.distance is None:
            _, (molecule, _, _, _, operator) = get_molecule(args.molecule, frozen_core=not args.all_electron)
        else:
            import tequila as tq
            from driver_geometry_scan import geometry_string
            molecule = tq.chemistry.Molecule(
                geometry=geometry_string(args.molecule, args.distance), basis_set="sto3g",
                transformation="JordanWigner", backend="pyscf", frozen_core=not args.all_electron)
            operator = molecule.make_hamiltonian().to_openfermion()
        n_qubits = int(count_qubits(operator))
        stem = args.molecule + (f"_R={args.distance:g}" if args.distance is not None else "")
        stem += f"_{args.wfn}" + ("_all-electron" if args.all_electron else "_frozen-core")
        title = stem
    else:
        model = build_lattice(args.system, args.n_qubits, lattice=args.lattice, shape=args.shape,
                              bond_disorder=args.bond_disorder, field_disorder=args.field_disorder,
                              disorder_seed=args.disorder_seed, **args.parameters)
        operator, n_qubits = model.qubit_operator, model.n_qubits
        stem = output_stem(model, seed=args.seed, particles=args.particles, spin_up=args.spin_up)
        title = f"{model.model}, {n_qubits} qubits, {model.shape[0]}×{model.shape[1]}, seed {args.seed}"
        print(stem, flush=True)
    terms = [term for term in make_terms(operator, n_qubits) if term.pauli_tuple]
    workspace = covariance_workspace_gib(len(terms), n_qubits)
    print(f"Qubits={n_qubits}; nonidentity terms={len(terms)}; core covariance workspace={workspace:.3f} GiB "
          "(excludes diagonalization and worker temporaries)", flush=True)
    if args.dry_run:
        return []
    if args.max_memory_gib and workspace > args.max_memory_gib:
        raise MemoryError(f"Core covariance workspace {workspace:.3f} GiB exceeds --max-memory-gib "
                          f"{args.max_memory_gib:g}. Raise the limit only with sufficient resources.")
    if args.molecule:
        energy, state = get_variance_wavefunction(molecule, operator, method=args.wfn)
        print(f"Reference={args.wfn}; energy={energy}", flush=True)
    else:
        state, reference = exact_ground_state(model, particles=args.particles, spin_up=args.spin_up, seed=args.seed)
        print(f"Reference={reference}", flush=True)
    covariances, _ = exact_covariances(terms, state, n_qubits, max_workers=args.cov_workers)
    samples = covariance_samples(terms, covariances, args.condition)
    variances = variance_samples(terms, covariances)
    paths = plot_distribution(samples, title, stem, args)
    paths.extend(plot_distribution(variances, title, stem, args, kind="variance"))
    return paths


if __name__ == "__main__":
    main()
