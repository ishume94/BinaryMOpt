# Binary Measurement Optimization

[![arXiv](https://img.shields.io/badge/arXiv-2610.10339-b31b1b.svg)](https://arxiv.org/abs/2610.10339)

bin_mopt uses classical binary optimization to reduce measurement costs for quantum energy estimation in molecular and lattice Hamiltonians. It constructs fully commuting Pauli groups through MILP-based clique selection, uses these groups to initialize iterative coefficient splitting (MILP-ICS), and directly optimizes overlapping supports with O-clique. The methods are described in [Binary Optimization of Measurement Groupings for Quantum Energy Estimation](https://arxiv.org/abs/2610.10339).

## Objective and methods

For fragments $H_g$ satisfying $H=\sum_\alpha H_\alpha$, the reported measurement cost is

$$
\varepsilon^2 M = \left(\sum_\alpha \sqrt{\mathrm{Var}_\psi(H_\alpha)}\right)^2.
$$

Covariances depend on the chosen reference state. Overlapping fragments may share Pauli terms, but their coefficients must reconstruct the Hamiltonian. Identity terms do not require measurement and are excluded from grouping.

The binary optimizer supports two methods:

- **`milp`**: mixed-integer linear optimization of a non-overlapping exact cover from candidate cliques.
- **`milp_lns`**: a MILP seed followed by large-neighborhood refinement, retaining improvements in the measured covariance objective.

Returned groups are checked for commutation, coverage, and coefficient reconstruction. Candidate-pool restrictions, proxy objectives, and solver time limits constrain optimality claims: a solver certificate applies to the model solved. Iterative coefficient splitting (ICS) refines coefficients producing an overlapping grouping; sorted insertion (SI) provides a reference grouping. Both SI and the MILP non-overlapping groupings, can be used as initializations of ICS, yielding the SI-ICS and MILP-ICS results.

## Installation

Use Python 3.11 or newer. From the repository checkout:

```bash
python -m pip install -e .
```

The standard optimizers use SciPy/HiGHS; the overlapping-clique  driver (O-Clique) also requires PySCIPOpt. Molecular Hamiltonians use Tequila, PySCF, STO-3G, and Jordan–Wigner mapping, with frozen core enabled by default.

## Drivers

| Driver | Calculation |
|---|---|
| `driver.py` | Molecular grouping with `milp` or `milp_lns`, followed by ICS. |
| `driver_lattice.py` | Lattice Hamiltonians with SI, SI-ICS, MILP, and MILP-ICS comparisons. |
| `driver_geometry_scan.py` | Molecular MILP at a specified bond length, or transfer one partition across a geometry scan. |
| `driver_o_clique.py` | Select overlapping candidate cliques under the MILP seed's group cap, then refine coefficients with ICS. |

Examples:

```bash
python -u driver.py H4 --method milp --cov-workers 1
python -u driver.py H4 --method milp_lns --solver-time-limit-s 300
python -u driver_lattice.py tfim 8 --solver-time-limit-s 300
python -u driver_geometry_scan.py LiH --rmin 0.8 --rmax 1.4 --ropt 1.0 --solver-time-limit-s 300
python -u driver_o_clique.py H4 --guidance relative_variance
```

`driver.py` optimizes a non-overlapping partition, then applies standard ICS, which can expand compatible supports and split coefficients across groups. Use `driver_o_clique.py` for overlapping candidate-clique optimization. The reference state is selected with `--wfn FCI`, `CISD`, or `HF` (default: FCI).

`driver_geometry_scan.py` is the recommended route for obtaining molecular MILP results at a specific bond length. For a single geometry, set `--rmin`, `--rmax`, and `--ropt` to the same distance in angstrom. For example, LiH at 1.6 Å:

```bash
python -u driver_geometry_scan.py LiH --rmin 1.6 --rmax 1.6 --ropt 1.6 --solver-time-limit-s 300 --cov-workers 1
```

This compares SI, SI-ICS, MILP, and MILP-ICS at that single geometry. For a scan, the driver solves the initial MILP only at `--ropt` and transfers memberships by Pauli word. `--new-terms` and `--refine-milp` control subsequent heuristic insertion/refinement. `--report-wfn` changes final cost evaluation without changing the state used for grouping or ICS. Geometry scans and the overlapping-clique driver support `--all-electron` to disable frozen core.

### O-Clique

`driver_o_clique.py` supports H4, H6, LiH, BeH2, H2O, NH3, and N2. NH3 has equal N–H bonds and 107° H–N–H angles. `--r` sets the bond length in angstrom (default: 1). The `--guidance` choices are `relative_std`, `relative_variance`, and `score_softmax` (three sweeps). `ics` (one coefficient solve) is also available as a weight guidance choice for development option, however it may increase substantially optimization time. Every choice finishes with ICS refinement over the coefficients of Pauli operators with multiplicities higher than one. The driver prints an SI, SI-ICS, MILP, MILP-ICS, and O-clique comparison with group counts, $\varepsilon^2M$, stage and total times, plus a process timing breakdown.

`--max-candidates` defaults to 1,800 and caps the overlapping candidate-clique pool.

`--seed-time-limit-s` and `--overlap-time-limit-s` each default to **72,000 seconds (20 hours)**. The overlap solver budget is shared across binary rounds; molecular preparation, candidate/profile and model construction, and ICS add to elapsed time. These limits do not cap the whole process. `--cov-workers` controls covariance workers independently of `--threads`, which controls numerical library threads and concurrent SCIP workers; both default to 1.

Use one seed file when comparing guides. If it is missing, the first run produces one MILP seed and saves it immediately before the overlap search; later runs load it. Existing invalid or mismatched seed files remain errors.

```bash
for guide in relative_std relative_variance score_softmax; do
  python -u driver_o_clique.py H4 \
    --seed-file o_clique_results/h4.json \
    --cov-workers 1 --threads 1 \
    --guidance "$guide" \
    --output "o_clique_results/h4_${guide}.json"
done
```

Without `--seed-file`, each run computes a new starting MILP partition. For shorter runs, set both limits explicitly, for example `--seed-time-limit-s 300 --overlap-time-limit-s 300`.

The clique MILP uses a finite set of coefficient profiles and a linear upper bound on the sum of fragment standard deviations. Squaring this bound gives an upper bound on $\varepsilon^2M$.

## Citation

```bibtex
@misc{huidobromeezs2026binaryoptimizationmeasurementgroupings,
  title={Binary Optimization of Measurement Groupings for Quantum Energy Estimation},
  author={Isaac L. Huidobro-Meezs and Rodrigo A. Vargas-Hernández},
  year={2026},
  eprint={2610.10339},
  archivePrefix={arXiv},
  primaryClass={quant-ph},
  url={https://arxiv.org/abs/2610.10339},
}
```
