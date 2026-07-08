import argparse
import json
import math
import os
import random
import sys
import time

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alns.alns_solver import HybridALNSSolver
from config import DEVICE
from data.solomon_generator import SolomonGenerator
from eval.gurobi_baseline import GurobiN20Solver
from model.variants import VARIANTS
from scripts.compare_drl_gurobi_n20 import (
    _extract_nested_actions,
    load_model,
    solve_alns,
    solve_drl,
)


def _finite(value):
    return value is not None and math.isfinite(float(value))


def _gap_to_bound(cost, bound):
    if not _finite(cost) or not _finite(bound) or float(bound) <= 0:
        return None
    return (float(cost) - float(bound)) / float(bound) * 100.0


def _load_warm_start_rows(path):
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return {int(row["instance"]): row for row in payload.get("rows", [])}


def _write_result(path, payload):
    if not path:
        return
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def _warm_start_stats(actions, n_nodes):
    customer_action_count = n_nodes * 2
    clean_actions = [int(action) for action in actions]
    return_count = sum(1 for action in clean_actions if action >= customer_action_count)
    needed_returns = max(return_count, max(0, len(clean_actions) - int(n_nodes)))
    return {
        "length": len(clean_actions),
        "returns": return_count,
        "needed_returns": needed_returns,
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Step-indexed MILP run for N50 UAV-VRPTW. "
            "Defaults favor high-quality incumbents; use --gurobi-proof-mode "
            "when you explicitly want stronger dual-bound effort. Time limit 0 "
            "means no Gurobi TimeLimit parameter is set."
        )
    )
    parser.add_argument("--instances", type=int, default=1)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--n-nodes", type=int, default=50)
    parser.add_argument("--problem-type", default="R")
    parser.add_argument("--scenario", choices=["full", "simple"], default="full")
    parser.add_argument("--tightness", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pref-alpha", type=float, default=0.5)
    parser.add_argument("--pref-beta", type=float, default=0.5)
    parser.add_argument("--variant", choices=list(VARIANTS.keys()), default="full")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--n-pomo", type=int, default=50)
    parser.add_argument("--decode-type", choices=["greedy", "sample"], default="greedy")
    parser.add_argument("--sample-rollouts", type=int, default=0)
    parser.add_argument("--sample-temperature", type=float, default=1.0)
    parser.add_argument("--multi-start-rollouts", type=int, default=0)
    parser.add_argument("--augment-8-fold", action="store_true")
    parser.add_argument("--run-alns-warm-start", action="store_true")
    parser.add_argument("--alns-time-limit", type=float, default=15.0)
    parser.add_argument("--alns-max-iter", type=int, default=2500)
    parser.add_argument("--alns-destroy-min", type=int, default=1)
    parser.add_argument("--alns-destroy-max", type=int, default=12)
    parser.add_argument("--alns-max-insert-positions", type=int, default=18)
    parser.add_argument("--alns-swap-neighbors", type=int, default=28)
    parser.add_argument("--alns-restart-rounds", type=int, default=1)
    parser.add_argument("--warm-start-json", default="")
    parser.add_argument("--warm-start-methods", default="ortools_physical,alns,drl")
    parser.add_argument("--disable-drl-warm-start", action="store_true")
    parser.add_argument("--gurobi-time-limit", type=float, default=3600.0)
    parser.add_argument("--gurobi-mip-gap", type=float, default=0.20)
    parser.add_argument("--gurobi-max-returns", type=int, default=0)
    parser.add_argument(
        "--no-auto-fit-warm-start-returns",
        action="store_true",
        help=(
            "Keep the configured --gurobi-max-returns even if a warm start needs "
            "more depot returns. By default, implicit max_returns is raised only "
            "as far as needed to fit the best available warm starts."
        ),
    )
    parser.add_argument("--gurobi-threads", type=int, default=0)
    parser.add_argument("--gurobi-nodefile-start-gb", type=float, default=0.5)
    parser.add_argument("--gurobi-nodefile-dir", default="")
    parser.add_argument("--gurobi-log-file", default="")
    parser.add_argument(
        "--gurobi-proof-mode",
        action="store_true",
        help="Use bound/proof-oriented Gurobi settings instead of incumbent-quality settings.",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--output-json", default="results_n50_gurobi_incumbent.json")
    args = parser.parse_args()

    user_max_returns = int(args.gurobi_max_returns)
    if args.gurobi_max_returns <= 0:
        if args.gurobi_proof_mode:
            args.gurobi_max_returns = args.n_nodes
        else:
            args.gurobi_max_returns = min(args.n_nodes, 20)
    auto_fit_warm_start_returns = (
        user_max_returns <= 0
        and not args.gurobi_proof_mode
        and not args.no_auto_fit_warm_start_returns
    )

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # Generate the full prefix so start-index uses exactly the same instances as
    # the comparison scripts.
    total_instances = args.start_index + args.instances
    generator = SolomonGenerator()
    nodes_batch, depot_batch, visited_batch = generator.generate(
        batch_size=total_instances,
        n_nodes=args.n_nodes,
        problem_type=args.problem_type,
        tightness=args.tightness,
        device=DEVICE,
    )

    pref = torch.tensor([[args.pref_alpha, args.pref_beta]], dtype=torch.float32, device=DEVICE)
    pref = pref / pref.sum(dim=-1, keepdim=True)

    model = load_model(args.variant, args.checkpoint)
    alns_solver = HybridALNSSolver(model, scenario=args.scenario) if args.run_alns_warm_start else None
    warm_rows = _load_warm_start_rows(args.warm_start_json)
    warm_methods = [item.strip() for item in args.warm_start_methods.split(",") if item.strip()]

    solver = GurobiN20Solver(
        time_limit_s=args.gurobi_time_limit,
        mip_gap=args.gurobi_mip_gap,
        max_returns=args.gurobi_max_returns,
        verbose=args.verbose,
        scenario=args.scenario,
        proof_mode=args.gurobi_proof_mode,
        threads=args.gurobi_threads,
        seed=args.seed,
        log_file=args.gurobi_log_file,
        nodefile_start_gb=args.gurobi_nodefile_start_gb,
        nodefile_dir=args.gurobi_nodefile_dir,
    )
    print(
        "Gurobi mode="
        + ("proof" if args.gurobi_proof_mode else "incumbent_quality")
        + f" | time_limit={args.gurobi_time_limit}s"
        + f" | mip_gap={args.gurobi_mip_gap}"
        + f" | max_returns={args.gurobi_max_returns}"
    )

    result = {
        "config": vars(args),
        "milp_model_notes": {
            "formulation": "step-indexed action MILP",
            "run_mode": "proof" if args.gurobi_proof_mode else "incumbent_quality",
            "nodes": "0 is depot, 1..N are customers",
            "speed_modes": "normal and fast binary arc modes",
            "physics": "flight time, load-dependent energy, hover waiting energy, battery reset on depot return",
            "energy_gate": "customer service reserves economical return-to-depot energy plus safety margin",
            "objective": "PHYS_WEIGHT * physical_cost + PEN_WEIGHT * soft_tardiness",
            "validation": "decoded Gurobi actions are replayed by VRPTWEnv for final reported cost",
            "time_limit_zero": "no Gurobi TimeLimit parameter is set",
        },
        "rows": [],
    }
    _write_result(args.output_json, result)

    for local_idx in range(args.instances):
        idx = args.start_index + local_idx
        nodes = nodes_batch[idx : idx + 1]
        depot = depot_batch[idx : idx + 1]
        visited = visited_batch[idx : idx + 1]
        pref_i = pref.clone()

        drl = solve_drl(
            model=model,
            nodes=nodes,
            pref=pref_i,
            depot=depot,
            visited_mask=visited,
            n_pomo=args.n_pomo,
            decode_type=args.decode_type,
            sample_rollouts=args.sample_rollouts,
            sample_temperature=args.sample_temperature,
            multi_start_rollouts=args.multi_start_rollouts,
            augment_8_fold=args.augment_8_fold,
            scenario=args.scenario,
            repair_mode="none",
        )

        alns = None
        if args.run_alns_warm_start:
            alns = solve_alns(
                alns_solver=alns_solver,
                nodes=nodes,
                pref=pref_i,
                depot=depot,
                visited_mask=visited,
                n_pomo=args.n_pomo,
                decode_type=args.decode_type,
                time_limit_s=args.alns_time_limit,
                max_iter=args.alns_max_iter,
                destroy_min=args.alns_destroy_min,
                destroy_max=args.alns_destroy_max,
                max_insert_positions=args.alns_max_insert_positions,
                swap_neighbors=args.alns_swap_neighbors,
                sample_rollouts=args.sample_rollouts,
                sample_temperature=args.sample_temperature,
                multi_start_rollouts=args.multi_start_rollouts,
                augment_8_fold=args.augment_8_fold,
                initial_actions=drl["route_actions"],
                restart_rounds=args.alns_restart_rounds,
            )

        warm_start_actions = []
        if idx in warm_rows:
            for method in warm_methods:
                actions = _extract_nested_actions(warm_rows[idx], method)
                if actions:
                    warm_start_actions.append(actions)
        if alns is not None and alns.get("actions"):
            warm_start_actions.append(alns["actions"])
        if not args.disable_drl_warm_start:
            warm_start_actions.append(drl["route_actions"])

        deduped = []
        seen = set()
        for actions in warm_start_actions:
            key = tuple(int(x) for x in actions)
            if key and key not in seen:
                seen.add(key)
                deduped.append([int(x) for x in actions])

        warm_start_stats = [
            _warm_start_stats(actions, args.n_nodes)
            for actions in deduped
        ]
        solve_max_returns = int(args.gurobi_max_returns)
        if auto_fit_warm_start_returns and warm_start_stats:
            needed_returns = max(stat["needed_returns"] for stat in warm_start_stats)
            solve_max_returns = min(args.n_nodes, max(solve_max_returns, needed_returns))
        solver.max_returns = solve_max_returns

        print(
            f"[{idx:03d}] warm starts={len(deduped)} | "
            f"DRL={drl['cost']:.4f} feas={int(bool(drl.get('feasible', False)))}"
            + (
                f" | ALNS={alns['cost']:.4f} feas={int(bool(alns.get('feasible', False)))}"
                if alns is not None
                else ""
            )
        )
        if warm_start_stats:
            stats_text = ", ".join(
                f"#{start_idx}:len={stat['length']},returns={stat['returns']}"
                for start_idx, stat in enumerate(warm_start_stats)
            )
            print(f"[{idx:03d}] warm start stats: {stats_text} | max_returns_used={solve_max_returns}")
            if solve_max_returns > args.gurobi_max_returns:
                print(
                    f"[{idx:03d}] raised max_returns from {args.gurobi_max_returns} "
                    f"to {solve_max_returns} to fit warm starts"
                )
        t0 = time.perf_counter()
        gurobi = solver.solve_instance(
            nodes=nodes,
            pref=pref_i,
            depot=depot,
            warm_start_actions=deduped if deduped else None,
            visited_mask=visited,
        )
        elapsed = time.perf_counter() - t0
        env_eval = gurobi["env_eval"]
        row = {
            "instance": idx,
            "drl": drl,
            "alns": alns,
            "warm_start_count": len(deduped),
            "warm_start_stats": warm_start_stats,
            "gurobi_max_returns_used": solve_max_returns,
            "gurobi_status": gurobi["status"],
            "gurobi_sol_count": gurobi["sol_count"],
            "gurobi_time_s": gurobi["elapsed_s"],
            "gurobi_outer_time_s": elapsed,
            "gurobi_mip_obj": gurobi["mip_obj"],
            "gurobi_obj_bound": gurobi["obj_bound"],
            "gurobi_mip_gap": gurobi["mip_gap"],
            "gurobi_env": env_eval,
            "gurobi_actions": [int(x) for x in env_eval.get("actions", [])],
            "gap_drl_to_incumbent_pct": _gap_to_bound(drl["cost"], env_eval.get("cost")),
            "gap_alns_to_incumbent_pct": None if alns is None else _gap_to_bound(alns["cost"], env_eval.get("cost")),
            "gap_gurobi_incumbent_to_bound_pct": _gap_to_bound(env_eval.get("cost"), gurobi["obj_bound"]),
            "gap_drl_to_bound_pct": _gap_to_bound(drl["cost"], gurobi["obj_bound"]),
            "gap_alns_to_bound_pct": None if alns is None else _gap_to_bound(alns["cost"], gurobi["obj_bound"]),
            "mip_obj_env_delta_pct": _gap_to_bound(env_eval.get("cost"), gurobi["mip_obj"]),
        }
        result["rows"].append(row)
        _write_result(args.output_json, result)
        print(
            f"[{idx:03d}] Gurobi(env)={env_eval['cost']:.4f} feasible={int(env_eval['feasible'])} "
            f"status={gurobi['status']} sol={gurobi['sol_count']} "
            f"bound={gurobi['obj_bound']:.4f} mip_gap={gurobi['mip_gap']:.6f} "
            f"incumbent_bound_gap={row['gap_gurobi_incumbent_to_bound_pct']} "
            f"time={gurobi['elapsed_s']:.1f}s"
        )

    _write_result(args.output_json, result)
    print(f"Saved Gurobi result json: {args.output_json}")


if __name__ == "__main__":
    main()
