# Preference-conditioned UAV routing: revised source

This repository contains research code for energy-constrained UAV routing with soft time windows. The `revision_v3/` package is the **frozen compact implementation** used for the revised experiments: preference input, an action-level physical feature head, dual-speed actions, safe-return masking, and POMO multi-start decoding. The formal model does not use FiLM or the energy-gating module (EGA). Historical PC-EGA-POMO prototype code remains in the original top-level `model/`, `train/`, and `scripts/` directories; see [its archived README](docs/README_LEGACY_PC_EGA_POMO.md). Do not treat that prototype as the revised paper's final method.

## Source map

| Path | Purpose |
| --- | --- |
| `revision_v3/` | Frozen compact policy, physical environment, instance generator, training and evaluation source. |
| `env/revision_v3_gu_physics.py` | Gu-v3 UAV physics used by the revised source. |
| `formal_runs/20261022/` | Original first-dispatch, Pareto continuation, and paired-evaluation runner snapshots. |
| `formal_runs/20261024/` | Original POMO main-panel evaluator, timing gate, and load-cleanup source snapshots. |
| `diagnostics/cleanup_selection_20261027/` | Later *post hoc* development diagnostic for selecting candidates after load cleanup; not part of the original blind main panel. |
| `docs/FROZEN_SOURCE_MANIFEST.md` | Source hashes and the relationship between frozen and later diagnostic code. |

The original formal runner snapshots retain absolute paths to the private experiment environment and its frozen artifacts. They document the executed implementation; they are **not** turnkey commands for reproducing the numerical tables from this repository alone. Model checkpoints, instance files, result rows, blind-test material, cloud backups, logs, and manuscript files are not included.

## Environment

The revised code was developed with Python 3.11 and PyTorch. Install the dependencies appropriate to your platform; `requirements-vrp311.txt` records the historical environment, including optional baseline solvers. Gurobi runs require a separate installation and license. To inspect the available revised commands from the repository root:

```bash
python -m revision_v3.train --help
python -m revision_v3.evaluate --help
```

The formal POMO identity used the `physical_pref_v3_wait40_slack5` base variant, followed by the separately frozen 40-update Pareto continuation (`D`) and protected candidate decoder (`B`). Source hashes and the frozen decision record identify the exact snapshot. Numerical claims require the corresponding protocols and completion records; the source tree alone does not supply those records.

## Reporting scope

- A shared policy responds to preference inputs. The revised evidence does not establish a general quality advantage over separately trained fixed-preference policies.
- Paired-evaluation and main-panel scripts distinguish training arm, decoding arm, candidate tier, physical feasibility, and wall-clock timing.
- The cleanup-selection diagnostic compares old and corrected selection on **previously seen W05-I development instances**. It does not replace the original main panel or constitute a new blind test.
- `OR-Tools` and `PyVRP` proxy solutions require replay in the common UAV physics before comparison. Feasibility, unserved customers, tardiness, energy, and time should be reported alongside cost.

The repository currently has no explicit software license. Public visibility alone does not grant a reuse license; the owner must choose and add one before describing this release as licensed open source.
