"""Disposable interface smoke for the disclosed route-order solver proxies.

This uses one RC instance to verify package APIs and complete v3 replay. It is
not a method comparison or a formal evaluation on the frozen test split.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import torch

from revision_v3.baselines import solve_proxy
from revision_v3.instances import generate
from revision_v3.spec import ROOT, SPEC_PATH, TASK


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budget-s", type=float, default=20.0)
    parser.add_argument("--n-nodes", type=int, choices=(20, 50, 100), default=20)
    parser.add_argument("--kind", choices=("R", "C", "RC"), default="RC")
    parser.add_argument("--instance-seed", type=int, default=920020)
    parser.add_argument("--methods", nargs="+", choices=("ortools", "pyvrp", "gurobi"),
                        default=("ortools", "pyvrp", "gurobi"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.budget_s < 5:
        parser.error("The full search-and-replay budget must be at least 5 seconds")
    generator = torch.Generator(device="cpu").manual_seed(args.instance_seed)
    nodes, depot, _ = generate(1, args.n_nodes, args.kind,
                               TASK["baseline_tightness"], generator=generator)
    preference = torch.tensor([[0.5, 0.5]])
    rows = []
    packages = {"ortools": "ortools", "pyvrp": "pyvrp", "gurobi": "gurobipy"}
    for method in args.methods:
        result = solve_proxy(method, nodes, depot, preference,
                             time_limit_s=args.budget_s)
        native = result["native"]
        row = {
            "method": method,
            "package_version": importlib.metadata.version(packages[method]),
            "native_status": native["status"],
            "native_objective": native.get("native_objective"),
            "native_bound": native.get("native_bound"),
            "native_mip_gap": native.get("native_mip_gap"),
            "full_task_feasible": result["feasible"],
            "full_task_cost": result.get("metrics", {}).get("cost"),
            "within_total_budget": result.get("within_total_budget"),
            "total_elapsed_s": result["total_elapsed_s"],
            "exact_physical_model": result.get("exact_physical_model", False),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
    record = {
        "purpose": "pre-freeze proxy package interface and physical-replay smoke",
        "instance": (f"disposable {args.kind} n{args.n_nodes} generator seed "
                     f"{args.instance_seed}; time preference 0.5"),
        "requested_budget_s_per_method": args.budget_s,
        "task_spec_sha256": sha(SPEC_PATH),
        "source_sha256": {
            name: sha(ROOT / name) for name in (
                "revision_v3/baselines.py", "revision_v3/environment.py",
                "revision_v3/instances.py", "revision_v3/proxy_interface_pilot.py",
            )
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
