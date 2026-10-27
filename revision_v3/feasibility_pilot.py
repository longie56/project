"""Pilot-only v3 instance feasibility and full-physics greedy runtime probe.

Seeds here are disjoint from the frozen validation/test seeds. This script
does not modify task_spec.json and cannot be used as a final result table.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import torch

from env.revision_v3_gu_physics import SCENARIOS
from revision_v3.baselines import greedy
from revision_v3.instances import generate
from revision_v3.spec import BUDGET_J, DEPOT_TURN_S, SERVICE_S, SIDE_M, SPEC_PATH, TASK


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-nodes", type=int, nargs="+", choices=(20, 50, 100), default=(20, 50, 100))
    ap.add_argument("--per-type", type=int, default=2)
    ap.add_argument("--max-candidates", type=int, default=4)
    ap.add_argument("--energy-scenario", choices=SCENARIOS, default="gu_nominal")
    ap.add_argument("--reserve-fraction", type=float, default=0.0)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.per_type < 1 or args.max_candidates < 1:
        ap.error("--per-type and --max-candidates must be positive")
    if not 0 <= args.reserve_fraction < 1:
        ap.error("--reserve-fraction must be in [0,1)")
    if args.output.exists():
        raise FileExistsError(args.output)
    physics = SCENARIOS[args.energy_scenario]
    rows = []
    for n in args.n_nodes:
        for kind_idx, kind in enumerate(("R", "C", "RC"), start=1):
            for sample_idx in range(args.per_type):
                seed = 510000 + n*100 + kind_idx*10 + sample_idx
                gen = torch.Generator(device="cpu").manual_seed(seed)
                nodes, depot, _ = generate(1, n, kind, TASK["baseline_tightness"],
                                           generator=gen, device="cpu")
                pref = torch.tensor([[0.5, 0.5]], dtype=torch.float32)
                d = torch.linalg.vector_norm(nodes[0, :, :2] - depot[0], dim=1)*SIDE_M
                cargo = nodes[0, :, 2]
                lower_energy = (physics.cruise_j(d, 12.0, cargo)
                                + physics.hover_j(SERVICE_S, cargo)
                                + physics.cruise_j(d, 12.0, 0.0))
                lower_time = DEPOT_TURN_S + 2*d/15.0 + SERVICE_S
                start = time.perf_counter()
                report = greedy(nodes, depot, pref,
                                max_candidates=args.max_candidates,
                                reserve_fraction=args.reserve_fraction,
                                energy_scenario=physics)
                elapsed = time.perf_counter()-start
                metrics = report["metrics"]
                row = {
                    "n_nodes": n, "kind": kind, "sample_index": sample_idx,
                    "seed": seed, "feasible": bool(report["feasible"]),
                    "energy_scenario": args.energy_scenario,
                    "reserve_fraction": args.reserve_fraction,
                    "failure_reason": report["failure_reason"],
                    "elapsed_seconds": elapsed,
                    "greedy_candidate_orders": report["candidate_orders"],
                    "unserved": int(metrics["unserved"]),
                    "completion_clock_s": float(metrics["completion_clock_s"]),
                    "horizon_s": float(TASK["mission_horizon_seconds_by_n"][str(n)]),
                    "flight_time_s": float(metrics["flight_time_s"]),
                    "total_energy_j": float(metrics["total_energy_j"]),
                    "depot_returns": int(metrics["depot_returns"]),
                    "min_return_battery_j": float(metrics["min_return_battery_j"]),
                    "solo_energy_unreachable_count": int((lower_energy >
                        physics.battery_budget_j*(1-args.reserve_fraction)).sum()),
                    "solo_time_unreachable_count": int((lower_time > row_horizon(n)).sum()),
                }
                rows.append(row)
                print(json.dumps(row, ensure_ascii=False), flush=True)
    summary = {}
    for n in args.n_nodes:
        sample = [r for r in rows if r["n_nodes"] == n]
        summary[str(n)] = {
            "count": len(sample),
            "greedy_feasible_rate": sum(r["feasible"] for r in sample)/len(sample),
            "total_solo_energy_unreachable": sum(r["solo_energy_unreachable_count"] for r in sample),
            "total_solo_time_unreachable": sum(r["solo_time_unreachable_count"] for r in sample),
            "mean_greedy_seconds": sum(r["elapsed_seconds"] for r in sample)/len(sample),
            "worst_greedy_seconds": max(r["elapsed_seconds"] for r in sample),
        }
    payload = {
        "purpose": "pilot only, no blind test", "gpu_required": False,
        "host": platform.node(), "python": platform.python_version(),
        "torch": torch.__version__,
        "energy_scenario": args.energy_scenario,
        "reserve_fraction": args.reserve_fraction,
        "task_spec_sha256": hashlib.sha256(SPEC_PATH.read_bytes()).hexdigest(),
        "rows": rows, "summary": summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    print("Saved", args.output)


def row_horizon(n: int) -> float:
    return float(TASK["mission_horizon_seconds_by_n"][str(n)])


if __name__ == "__main__":
    main()
