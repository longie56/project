# Experiment Protocol

This document is the reporting contract for the UAV-VRPTW paper experiments.
Tables and claims should follow it unless a later protocol file explicitly
supersedes it.

## Dataset Naming

- Default data source: synthetic Solomon-style generator in
  `data/solomon_generator.py`.
- Use the wording `synthetic Solomon-style R/C/RC instances`.
- Do not call these rows the published Solomon benchmark unless the experiment
  uses `data/solomon_loader.py` with named Solomon files and reports that fact.
- Record `problem_type`, `n_nodes`, `tightness`, `seed`, preference mode, and
  checkpoint path for every result JSON.

## Method Families

- `PC-EGA-POMO greedy`: neural policy only, no repair.
- `PC-EGA-POMO + ALNS`: neural policy warm start plus ALNS repair. Report total
  wall time as the DRL time plus ALNS time when both are used.
- `OR-Tools physical`: proxy route/order search followed by `VRPTWEnv` replay.
  This is a strong heuristic baseline, not the exact UAV action-space model.
- `PyVRP`: capacitated VRPTW proxy followed by `VRPTWEnv` replay. This is a
  strong heuristic baseline, not the exact UAV action-space model.
- `Gurobi-MILP`: exact project MILP formulation only for rows with certified
  status. If Gurobi stops by time limit, report it as an incumbent and bound.

## Metrics

Every main table must include:

- cost,
- physical cost component,
- time-window penalty,
- average unvisited customers,
- feasibility rate,
- observed wall time.

For quality diagnostics, also report median, standard deviation, and p90 cost or
gap when the sample size is large enough.

Rank methods feasibility-first, then unvisited count, then cost. A lower cost
with worse feasibility is not a clean win.

## Time Budgets

Do not mix these protocol families:

- `realtime`: strict observed wall-clock budget, for example 5 seconds. Report
  both configured solver time limit and observed wall time. A baseline whose
  observed wall time exceeds the budget is not a strict realtime baseline.
- `quality`: larger stated budget, for example 15s, 30s, or 300s. These numbers
  can support quality claims, not realtime claims.
- `MILP proof/incumbent`: long Gurobi runs. Report status, incumbent, objective
  bound, MIP gap, and runtime. Only rows with `OPTIMAL` status or a clearly
  stated small certified MIP gap can support optimality claims.

## N20 Required Evidence

- Run R, C, and RC separately with matched instance counts.
- Prefer at least 100 instances per type for final tables.
- Include fixed preference `(0.5, 0.5)` and random preference
  `alpha ~ U(0.1, 0.9), beta = 1 - alpha` if making preference-general claims.
- Gurobi can be used as exact reference only for certified rows. Otherwise, use
  `incumbent`, `bound`, and `MIP gap` language.

## N50 Required Evidence

- Run R, C, RC, and MIXED type evaluations. A single R-only table is preliminary.
- Include DRL, DRL+ALNS, OR-Tools physical, and PyVRP for final quality tables
  unless a dependency is unavailable and clearly disclosed.
- Separate 5s realtime tables from 15s/30s/300s quality tables.
- Gurobi N50 results are long-run incumbent/proof diagnostics unless certified.

## Dynamic Insertion

- Report dynamic insertion separately from static routing.
- Include emergency count, executed ratio, urgency window, and replan time.
- Keep feasibility and unvisited count next to cost in the main dynamic table.

## Claim Wording

Safe claim shape:

`The learned policy provides fast feasible solutions, and ALNS repair improves
solution quality under the stated wall-clock budget.`

Unsafe unless certified by the current results:

- `Gurobi proves the optimum`.
- `The method outperforms the exact optimum`.
- `The 5s protocol is strict` when observed wall time is above 5s.
