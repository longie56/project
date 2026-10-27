# PC-EGA-POMO for Energy-Constrained UAV Routing

This repository contains research code for a preference-conditioned deep reinforcement learning approach to energy-constrained unmanned aerial vehicle (UAV) routing with soft time windows. The core method, PC-EGA-POMO, extends policy optimization with multiple optima (POMO) by adding preference conditioning, energy-aware action scoring, dual-speed actions, and safe-return feasibility masking.

The project was developed for the paper:

> A preference-conditioned deep reinforcement learning algorithm for energy-constrained UAV routing with soft time windows

## What This Project Does

The code studies a UAV routing setting where each route must jointly consider:

- customer visiting order,
- economical-speed and fast-speed flight modes,
- payload-aware energy consumption,
- battery and payload constraints,
- soft time-window penalties,
- repeated depot return and replenishment,
- time-energy preference trade-offs.

The main idea is to generate routes with a neural policy while explicitly injecting physical feasibility signals into decoding. The learned policy can also be combined with adaptive large neighborhood search (ALNS) for local repair and quality improvement.

## Main Features

- Preference-conditioned route generation with time-energy preference vectors.
- Dual-speed action space: each customer can be selected with economical speed or fast speed.
- Action-level physical features, including flight time, energy use, waiting time, tardiness risk, and energy margin.
- Energy-gating bias and safe-return feasibility mask for energy-safe decoding.
- POMO-style multi-start rollout training.
- Optional ALNS repair using the neural solution as a warm start.
- Baseline evaluation utilities for OR-Tools, PyVRP, and Gurobi.
- Synthetic Solomon-style instance generation for R, C, RC, and mixed spatial distributions.

## Repository Structure

```text
.
|-- alns/          # ALNS repair solver
|-- data/          # Solomon-style instance generator and Solomon loaders
|-- env/           # UAV routing environment and transition logic
|-- eval/          # OR-Tools, PyVRP, Gurobi, and route replay utilities
|-- model/         # PC-EGA-POMO model, decoder, encoder, and variants
|-- scripts/       # Training, evaluation, auditing, and figure/table utilities
|-- train/         # Training loop and checkpoint handling
|-- utils/         # Visualization utilities
|-- config.py      # Main physical constants and hyperparameters
|-- main.py        # Minimal training entry point
`-- requirements-vrp311.txt
```

Large experimental artifacts, checkpoints, figures, tables, logs, and manuscript-processing files are intentionally excluded from the public repository.

## Installation

The code was developed with Python 3.11. A conda environment is recommended.

```bash
conda create -n vrp311 python=3.11
conda activate vrp311
pip install -r requirements-vrp311.txt
```

Notes:

- Gurobi is optional for most neural training and baseline experiments, but scripts using `eval/gurobi_baseline.py` require a working Gurobi installation and license.
- CUDA is recommended for training larger models, especially N50 and N100 instances.
- CPU execution is sufficient for small smoke tests and some baseline utilities.

## Quick Smoke Test

Run the physics-consistency smoke test from the repository root:

```bash
python scripts/smoke_physics_consistency.py
```

This checks that feasible actions produced by the environment mask preserve the route horizon and the required return-to-depot energy reserve.

## Training

A minimal training run can be launched with:

```bash
python main.py
```

For configurable experiments, use `scripts/train_variants.py`.

Example N20 training:

```bash
python scripts/train_variants.py ^
  --variant full ^
  --epochs 100 ^
  --steps-per-epoch 200 ^
  --batch-size 64 ^
  --n-nodes 20 ^
  --n-pomo 20 ^
  --problem-type MIXED ^
  --pref-mode random ^
  --output-dir checkpoints/n20_full
```

On Linux or macOS, replace the Windows line-continuation character `^` with `\`.

The main model variants are defined in `model/variants.py`:

- `full`: FiLM preference modulation + energy gating + speed-mode bias.
- `film`: preference modulation only.
- `ega`: FiLM + energy gating.
- `no_film`: ablation without FiLM.
- `no_action_features`: ablation without action-level physical feature scoring.
- `vanilla`: basic model without the proposed conditioning and energy-aware modules.

## Evaluation

The main static evaluation script is:

```bash
python scripts/compare_baselines_n20.py
```

Example evaluation with a trained checkpoint:

```bash
python scripts/compare_baselines_n20.py ^
  --instances 20 ^
  --n-nodes 20 ^
  --problem-type R ^
  --pref-mode random ^
  --variant full ^
  --checkpoint checkpoints/n20_full/full_best.pt ^
  --run-drl ^
  --run-alns ^
  --run-ortools-physical ^
  --run-pyvrp ^
  --output-json results_n20_r.json
```

The repository does not include pretrained checkpoints by default. To evaluate a pretrained model, place the checkpoint file under a local `checkpoints/` directory and pass its path through `--checkpoint`.

## Dynamic Order Insertion

Dynamic order insertion experiments are supported by:

```bash
python scripts/experiment_dynamic_insertion.py
```

This evaluates replanning from an executed route snapshot after new customer orders are inserted.

## Experiment Protocol

Additional notes for reproducibility and reporting are provided in:

- `docs/experiment_protocol.md`
- `docs/reproducibility_environment.md`

These documents describe dataset naming, method families, metrics, and protocol distinctions such as real-time evaluation, quality-budget evaluation, and long-budget Gurobi diagnostics.

## Important Reporting Notes

When using this code for experiments, please distinguish:

- `PC-EGA-POMO`: neural policy inference only.
- `PC-EGA-POMO + ALNS`: neural policy followed by ALNS repair.
- `OR-Tools physical` and `PyVRP`: classical proxy baselines replayed in the unified physical environment.
- `Gurobi`: exact or incumbent MILP diagnostics depending on solver status, time limit, and certificate quality.

For constrained UAV routing, feasibility, unvisited customers, and wall-clock time should be reported together with cost. A lower cost with worse feasibility should not be interpreted as a clean improvement.

## Data

The project includes a Solomon-style generator and Solomon benchmark loader utilities. In the associated experiments, test instances are generated by integrating spatial and time-window characteristics inspired by the Solomon benchmark with UAV-specific operational settings such as payload, battery, speed, and safe-return constraints.

## Citation

If you use this repository, please cite the associated manuscript once it becomes available. A BibTeX entry will be added after publication or public preprint release.

## License

No license file is currently included. Before reusing or redistributing this code, please contact the authors or add an explicit open-source license to the repository.
