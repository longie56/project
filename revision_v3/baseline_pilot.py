"""Pre-freeze same-physics baseline smoke with end-to-end time accounting."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from revision_v3.evaluate import run_method
from revision_v3.instances import generate
from revision_v3.spec import ROOT, SPEC_PATH, TASK


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--methods", nargs="+", choices=("greedy", "alns", "nsga2"),
                    default=("greedy", "alns", "nsga2"))
    ap.add_argument("--n-nodes", nargs="+", type=int, choices=(20, 50, 100),
                    default=(20, 50, 100))
    ap.add_argument("--large-budget-s", type=float, default=180.0)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = []
    for n in args.n_nodes:
        generator = torch.Generator(device="cpu").manual_seed(920000+n)
        nodes, depot, _ = generate(1, n, "RC", TASK["baseline_tightness"],
                                   generator=generator)
        pref = torch.tensor([0.5, 0.5])
        budget = 60.0 if n < 100 else args.large_budget_s
        for method in args.methods:
            result = run_method(method, nodes[0], depot[0], pref,
                                time_limit_s=budget, seed=920000+n)
            metrics = result.get("metrics", {})
            row = {"n_nodes": n, "kind": "RC", "seed": 920000+n,
                   "method": method, "budget_s": budget,
                   "elapsed_s": result.get("total_elapsed_s"),
                   "within_total_budget": result.get("within_total_budget"),
                   "comparison_eligible_under_budget": result.get(
                       "comparison_eligible_under_budget"),
                   "feasible": result.get("feasible"),
                   "failure_reason": result.get("failure_reason"),
                   "raw_cost": metrics.get("cost"),
                   "unserved": metrics.get("unserved"),
                   "depot_returns": metrics.get("depot_returns"),
                   "construction_action_limit_hit": result.get("construction_action_limit_hit"),
                   "search_iterations": result.get("search_iterations"),
                   "speed_neighborhood_proposals": result.get(
                       "speed_neighborhood_proposals"),
                   "evaluations": result.get("evaluations")}
            rows.append(row)
            print(json.dumps(row), flush=True)
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    payload = {"purpose": "pre-freeze single-instance baseline interface smoke",
               "scope": "one disposable RC instance per scale, not a quality comparison",
               "task_spec_sha256": digest(SPEC_PATH),
               "code_sha256": {name: digest(ROOT / name) for name in (
                   "env/revision_v3_gu_physics.py", "revision_v3/environment.py",
                   "revision_v3/instances.py", "revision_v3/baselines.py",
                   "revision_v3/evaluate.py", "revision_v3/baseline_pilot.py")},
               "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2)+"\n", encoding="utf-8")
    print("Saved", args.output, flush=True)


if __name__ == "__main__":
    main()
