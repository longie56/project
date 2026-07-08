# Reproducibility Environment

Current verified environment on this machine:

- Python: `D:\anaconda\envs\vrp311\python.exe`
- Python version: 3.11.15
- Torch: 2.12.0+cpu
- NumPy: 2.4.6
- OR-Tools: 9.15.6755
- PyVRP: 0.13.4
- Gurobi Python API: 13.0.2
- Matplotlib: 3.10.9
- tqdm: 4.67.3

Known bad or incomplete environments:

- `.venv312` is broken because `.venv312\pyvenv.cfg` points to a mojibake path
  under `C:\Users\...Python312`.
- `D:\anaconda\python.exe` starts, but currently misses core packages such as
  `torch`, `numpy`, `ortools`, `pyvrp`, `gurobipy`, and `matplotlib`.
- `C:\ProgramData\Miniconda3\python.exe` has a broken NumPy import and misses
  most experiment dependencies.
- `D:\anaconda\envs\gln_ppo\python.exe` has Torch 1.11.0, but misses OR-Tools,
  PyVRP, Gurobi, and tqdm.

## Smoke Test

Run from the project root:

```powershell
D:\anaconda\envs\vrp311\python.exe scripts\smoke_physics_consistency.py
```

Expected output:

```text
physics consistency smoke passed: instances=8, n_nodes=50, checked_actions=800, scenario=full
```

This smoke test checks that currently feasible actions in `VRPTWEnv.get_mask()`
do not violate the route horizon before completion and preserve enough energy
for the required return-to-depot reserve.

## Notes

The verified `vrp311` environment uses CPU Torch. It is suitable for smoke
tests, table generation, Gurobi/OR-Tools/PyVRP baselines, and small evaluations.
Large neural training or large-batch inference should be run in a separate CUDA
environment and recorded here before the final experiments.
