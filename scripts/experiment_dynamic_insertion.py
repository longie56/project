import argparse
import json
import os
import random
import sys
import time

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alns.alns_solver import HybridALNSSolver
from config import DEVICE
from data.solomon_generator import SolomonGenerator
from data.solomon_loader import load_solomon_dataset
from eval.ortools_physical_baseline import ORToolsPhysicalBaselineSolver
from eval.pyvrp_baseline import PyVRPBaselineSolver
from eval.route_baseline_utils import priority_to_actions
from model.pomo import PC_EGA_POMO
from model.variants import VARIANTS


def load_model_weights(model, checkpoint_path):
    state = torch.load(checkpoint_path, map_location=DEVICE)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=False)
    model.eval()


def build_instances(args):
    if args.dataset == "solomon":
        instances = load_solomon_dataset(args.instance_names.split(","), device=DEVICE)
        for instance in instances:
            instance["pref"] = _build_pref(args)
        return instances

    generator = SolomonGenerator()
    nodes, depot, visited = generator.generate(
        batch_size=args.instances,
        n_nodes=args.n_nodes,
        problem_type=args.problem_type,
        tightness=args.tightness,
        device=DEVICE,
    )
    instances = []
    for idx in range(args.instances):
        instances.append(
            {
                "name": f"synthetic_{idx:03d}",
                "nodes": nodes[idx : idx + 1],
                "depot": depot[idx : idx + 1],
                "visited_mask": visited[idx : idx + 1],
                "pref": _build_pref(args),
            }
        )
    return instances


def _build_pref(args):
    if args.pref_mode == "random":
        if not (0.0 <= args.pref_random_low < args.pref_random_high <= 1.0):
            raise ValueError("--pref-random-low/high must satisfy 0 <= low < high <= 1.")
        alpha_pref = (
            torch.rand(1, 1, device=DEVICE)
            * (args.pref_random_high - args.pref_random_low)
            + args.pref_random_low
        )
        return torch.cat([alpha_pref, 1.0 - alpha_pref], dim=1)
    pref = torch.tensor([[args.pref_time, args.pref_energy]], dtype=torch.float32, device=DEVICE)
    return pref / pref.sum(dim=-1, keepdim=True)


def create_emergency_nodes(snapshot, base_nodes, emergency_count, urgency_window, seed_jitter=0.08):
    curr_pos = snapshot["curr_pos"]
    curr_time = snapshot["curr_time"]
    service_time = base_nodes[:, :, 5].mean().item()
    base_demand = base_nodes[:, :, 2].mean().item()

    emergency_xy = curr_pos.unsqueeze(1).repeat(1, emergency_count, 1)
    emergency_xy = emergency_xy + torch.randn_like(emergency_xy) * float(seed_jitter)
    emergency_xy = emergency_xy.clamp(0.0, 1.0)

    demand = torch.full((1, emergency_count), max(0.02, base_demand * 0.6), device=base_nodes.device)
    ready = curr_time.unsqueeze(1).repeat(1, emergency_count)
    due = ready + float(urgency_window)
    service = torch.full((1, emergency_count), service_time, device=base_nodes.device)

    return torch.stack([emergency_xy[:, :, 0], emergency_xy[:, :, 1], demand, ready, due, service], dim=2)


def _remaining_due_priority(nodes, visited_mask, emergency_count):
    n_nodes = nodes.size(1)
    emergency_start = n_nodes - emergency_count
    remaining = [idx for idx in range(n_nodes) if not bool(visited_mask[0, idx].item())]
    emergency = [idx for idx in remaining if idx >= emergency_start]
    regular = [idx for idx in remaining if idx < emergency_start]
    regular.sort(key=lambda idx: (float(nodes[0, idx, 4].item()), float(nodes[0, idx, 3].item())))
    # Emergency orders are intentionally first: this baseline is a transparent
    # "quick insertion" heuristic rather than a long global search.
    return emergency + regular


def solve_greedy_insert(nodes, pref, depot, visited_mask, snapshot, emergency_count, scenario):
    start = time.perf_counter()
    priority = _remaining_due_priority(nodes, visited_mask, emergency_count)
    result = priority_to_actions(
        nodes=nodes,
        pref=pref,
        depot=depot,
        priority=priority,
        visited_mask=visited_mask,
        curr_pos=snapshot["curr_pos"],
        curr_time=snapshot["curr_time"],
        curr_load=snapshot["curr_load"],
        curr_energy=snapshot["curr_energy"],
        scenario=scenario,
    )
    result["elapsed_s"] = time.perf_counter() - start
    result["priority"] = priority
    return result


def solve_model_after_insertion(model_solver, nodes, pref, depot, visited_mask, snapshot, args):
    start = time.perf_counter()
    result = model_solver.get_initial_solution(
        nodes,
        pref,
        depot,
        visited_mask,
        snapshot["curr_pos"],
        snapshot["curr_time"],
        snapshot["curr_load"],
        snapshot["curr_energy"],
        n_pomo=args.n_pomo,
        decode_type=args.decode_type,
        sample_rollouts=args.sample_rollouts,
        sample_temperature=args.sample_temperature,
        multi_start_rollouts=args.multi_start_rollouts,
        augment_8_fold=args.augment_8_fold,
    )
    result["elapsed_s"] = time.perf_counter() - start
    result["actions"] = [int(x) for x in (result.get("actions") or result.get("route_actions") or [])]
    return result


def solve_alns_after_insertion(model_solver, nodes, pref, depot, visited_mask, snapshot, args, initial_actions=None):
    start = time.perf_counter()
    result = model_solver.optimize(
        nodes,
        pref,
        depot,
        visited_mask,
        snapshot["curr_pos"],
        snapshot["curr_time"],
        snapshot["curr_load"],
        snapshot["curr_energy"],
        max_alns_iter=args.alns_max_iter,
        time_limit_s=args.alns_time_limit,
        destroy_min=args.alns_destroy_min,
        destroy_max=args.alns_destroy_max,
        max_insert_positions=args.alns_max_insert_positions,
        swap_neighbors=args.alns_swap_neighbors,
        n_pomo=args.n_pomo,
        decode_type=args.decode_type,
        sample_rollouts=args.sample_rollouts,
        sample_temperature=args.sample_temperature,
        multi_start_rollouts=args.multi_start_rollouts,
        augment_8_fold=args.augment_8_fold,
        initial_actions=initial_actions,
        init_source=args.alns_init_source,
        restart_rounds=args.alns_restart_rounds,
    )
    best = dict(result["best_solution"])
    best["elapsed_s"] = time.perf_counter() - start
    best["solver_elapsed_s"] = result["elapsed_s"]
    best["iterations"] = result["iterations"]
    best["actions"] = [int(x) for x in (best.get("actions") or best.get("route_actions") or [])]
    return best


def solve_ortools_after_insertion(solver, nodes, pref, depot, visited_mask, snapshot):
    return solver.solve_instance(
        nodes,
        pref,
        depot,
        visited_mask=visited_mask,
        curr_pos=snapshot["curr_pos"],
        curr_time=snapshot["curr_time"],
        curr_load=snapshot["curr_load"],
        curr_energy=snapshot["curr_energy"],
    )


def solve_pyvrp_after_insertion(solver, nodes, pref, depot, visited_mask, snapshot):
    return solver.solve_instance(
        nodes,
        pref,
        depot,
        visited_mask=visited_mask,
        curr_pos=snapshot["curr_pos"],
        curr_time=snapshot["curr_time"],
        curr_load=snapshot["curr_load"],
        curr_energy=snapshot["curr_energy"],
    )


def summarize_rows(rows, methods):
    summary = {"instances": len(rows)}
    for method in methods:
        payloads = [row[method] for row in rows if method in row and row[method] is not None]
        if not payloads:
            continue
        method_summary = {}
        for key in ("cost", "phys", "pen", "unvisit", "elapsed_s", "total_elapsed_s", "emergency_penalty_proxy"):
            values = [float(item[key]) for item in payloads if item.get(key) is not None]
            if values:
                out_key = "avg_time_s" if key == "elapsed_s" else f"avg_{key}"
                method_summary[out_key] = sum(values) / len(values)
        feasible_values = [bool(item.get("feasible", False)) for item in payloads]
        if feasible_values:
            method_summary["feasible_rate"] = sum(float(x) for x in feasible_values) / len(feasible_values)
        summary[method] = method_summary
    return summary


def compact_solution(result):
    keep = {
        "cost",
        "phys",
        "pen",
        "unvisit",
        "feasible",
        "elapsed_s",
        "total_elapsed_s",
        "iterations",
        "depot_return_count",
        "num_vehicles",
        "emergency_penalty_proxy",
    }
    out = {key: result[key] for key in keep if key in result}
    if "actions" in result:
        out["actions"] = [int(x) for x in result["actions"]]
    elif "route_actions" in result:
        out["actions"] = [int(x) for x in result["route_actions"]]
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["synthetic", "solomon"], default="synthetic")
    parser.add_argument("--instance-names", default="r101")
    parser.add_argument("--instances", type=int, default=10)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--variant", choices=list(VARIANTS.keys()), default="full")
    parser.add_argument("--decode-type", choices=["greedy", "sample"], default="greedy")
    parser.add_argument("--n-pomo", type=int, default=None)
    parser.add_argument("--sample-rollouts", type=int, default=0)
    parser.add_argument("--sample-temperature", type=float, default=1.0)
    parser.add_argument("--multi-start-rollouts", type=int, default=0)
    parser.add_argument("--augment-8-fold", action="store_true")
    parser.add_argument("--n-nodes", type=int, default=50)
    parser.add_argument("--problem-type", default="R")
    parser.add_argument("--scenario", choices=["full", "simple"], default="full")
    parser.add_argument("--tightness", type=float, default=0.7)
    parser.add_argument("--pref-mode", choices=["fixed", "random"], default="fixed")
    parser.add_argument("--pref-time", type=float, default=0.5)
    parser.add_argument("--pref-energy", type=float, default=0.5)
    parser.add_argument("--pref-random-low", type=float, default=0.1)
    parser.add_argument("--pref-random-high", type=float, default=0.9)
    parser.add_argument("--executed-ratio", type=float, default=0.3)
    parser.add_argument("--emergency-count", type=int, default=2)
    parser.add_argument("--urgency-window", type=float, default=0.15)
    parser.add_argument("--emergency-jitter", type=float, default=0.08)
    parser.add_argument("--run-greedy-insert", action="store_true")
    parser.add_argument("--run-drl", action="store_true")
    parser.add_argument("--run-alns", action="store_true")
    parser.add_argument("--run-standalone-alns", action="store_true")
    parser.add_argument("--run-ortools-physical", action="store_true")
    parser.add_argument("--run-pyvrp", action="store_true")
    parser.add_argument("--alns-time-limit", type=float, default=5.0)
    parser.add_argument("--alns-max-iter", type=int, default=1200)
    parser.add_argument("--alns-destroy-min", type=int, default=1)
    parser.add_argument("--alns-destroy-max", type=int, default=8)
    parser.add_argument("--alns-max-insert-positions", type=int, default=12)
    parser.add_argument("--alns-swap-neighbors", type=int, default=18)
    parser.add_argument("--alns-init-source", choices=["auto", "model", "heuristic"], default="auto")
    parser.add_argument("--alns-restart-rounds", type=int, default=1)
    parser.add_argument("--ortools-time-limit", type=float, default=5.0)
    parser.add_argument("--ortools-first-solution", default="PATH_CHEAPEST_ARC")
    parser.add_argument("--ortools-local-search", default="GUIDED_LOCAL_SEARCH")
    parser.add_argument("--ortools-route-time-limit", type=float, default=3.0)
    parser.add_argument("--pyvrp-time-limit", type=float, default=5.0)
    parser.add_argument("--pyvrp-route-time-limit", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()
    if args.n_pomo is None:
        args.n_pomo = args.n_nodes
    if not any([args.run_greedy_insert, args.run_drl, args.run_alns, args.run_standalone_alns, args.run_ortools_physical, args.run_pyvrp]):
        raise ValueError("Enable at least one method with --run-greedy-insert/--run-drl/--run-alns/--run-standalone-alns/--run-ortools-physical/--run-pyvrp.")

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    model = PC_EGA_POMO(**VARIANTS[args.variant]).to(DEVICE)
    if args.checkpoint:
        load_model_weights(model, args.checkpoint)
    model_solver = HybridALNSSolver(model, device=DEVICE, scenario=args.scenario)
    heuristic_solver = HybridALNSSolver(None, device=DEVICE, scenario=args.scenario)

    ortools_solver = None
    if args.run_ortools_physical:
        ortools_solver = ORToolsPhysicalBaselineSolver(
            time_limit_s=args.ortools_time_limit,
            first_solution_strategy=args.ortools_first_solution,
            local_search_metaheuristic=args.ortools_local_search,
            route_time_limit=args.ortools_route_time_limit,
            scenario=args.scenario,
            routing_mode="serial_order",
        )
    pyvrp_solver = None
    if args.run_pyvrp:
        pyvrp_solver = PyVRPBaselineSolver(
            time_limit_s=args.pyvrp_time_limit,
            route_time_limit=args.pyvrp_route_time_limit,
            scenario=args.scenario,
        )

    instances = build_instances(args)
    method_names = []
    if args.run_greedy_insert:
        method_names.append("greedy_insert")
    if args.run_drl:
        method_names.append("drl_replan")
    if args.run_alns:
        method_names.append("drl_alns_repair")
    if args.run_standalone_alns:
        method_names.append("standalone_alns")
    if args.run_ortools_physical:
        method_names.append("ortools_physical")
    if args.run_pyvrp:
        method_names.append("pyvrp")

    rows = []
    for idx, instance in enumerate(instances):
        nodes = instance["nodes"]
        depot = instance["depot"]
        visited_mask = instance["visited_mask"]
        pref = instance["pref"]

        with torch.no_grad():
            base_result = model.inference_rollout(
                nodes,
                pref,
                depot,
                visited_mask=visited_mask,
                N_pomo=args.n_pomo,
                decode_type=args.decode_type,
                sample_rollouts=args.sample_rollouts,
                sample_temperature=args.sample_temperature,
                multi_start_rollouts=args.multi_start_rollouts,
                augment_8_fold=args.augment_8_fold,
                scenario=args.scenario,
            )

        route_len = int(base_result["best_route_lengths"][0].item()) if base_result.get("best_route_lengths") is not None else base_result["best_routes"].size(1)
        base_actions = base_result["best_routes"][0, :route_len]
        served_target = max(1, int(nodes.size(1) * args.executed_ratio))
        snapshot = model_solver.simulate_prefix_state(
            nodes,
            pref,
            depot,
            base_actions,
            served_customers_target=served_target,
            visited_mask=visited_mask,
        )

        emergency_nodes = create_emergency_nodes(
            snapshot=snapshot,
            base_nodes=nodes,
            emergency_count=args.emergency_count,
            urgency_window=args.urgency_window,
            seed_jitter=args.emergency_jitter,
        )
        augmented_nodes = torch.cat([nodes, emergency_nodes], dim=1)
        augmented_visited = torch.cat(
            [
                snapshot["visited_mask"],
                torch.zeros(1, args.emergency_count, dtype=torch.bool, device=DEVICE),
            ],
            dim=1,
        )

        row = {
            "instance": instance["name"],
            "pref": [float(pref[0, 0].item()), float(pref[0, 1].item())],
            "served_before_insertion": int(snapshot["served_customers"]),
            "executed_prefix_actions": [int(x) for x in snapshot["executed_actions"]],
        }

        drl_result = None
        if args.run_greedy_insert:
            row["greedy_insert"] = compact_solution(
                solve_greedy_insert(
                    augmented_nodes,
                    pref,
                    depot,
                    augmented_visited,
                    snapshot,
                    args.emergency_count,
                    args.scenario,
                )
            )
        if args.run_drl or args.run_alns:
            drl_result = solve_model_after_insertion(
                model_solver,
                augmented_nodes,
                pref,
                depot,
                augmented_visited,
                snapshot,
                args,
            )
            if args.run_drl:
                row["drl_replan"] = compact_solution(drl_result)
        if args.run_alns:
            start = time.perf_counter()
            alns_result = solve_alns_after_insertion(
                model_solver,
                augmented_nodes,
                pref,
                depot,
                augmented_visited,
                snapshot,
                args,
                initial_actions=(None if drl_result is None else drl_result.get("actions")),
            )
            alns_result["total_elapsed_s"] = (0.0 if drl_result is None else drl_result.get("elapsed_s", 0.0)) + alns_result["elapsed_s"]
            alns_result["wall_elapsed_s"] = time.perf_counter() - start
            row["drl_alns_repair"] = compact_solution(alns_result)
        if args.run_standalone_alns:
            standalone_args = argparse.Namespace(**vars(args))
            standalone_args.alns_init_source = "heuristic"
            standalone_result = solve_alns_after_insertion(
                heuristic_solver,
                augmented_nodes,
                pref,
                depot,
                augmented_visited,
                snapshot,
                standalone_args,
                initial_actions=None,
            )
            row["standalone_alns"] = compact_solution(standalone_result)
        if args.run_ortools_physical:
            row["ortools_physical"] = compact_solution(
                solve_ortools_after_insertion(
                    ortools_solver,
                    augmented_nodes,
                    pref,
                    depot,
                    augmented_visited,
                    snapshot,
                )
            )
        if args.run_pyvrp:
            row["pyvrp"] = compact_solution(
                solve_pyvrp_after_insertion(
                    pyvrp_solver,
                    augmented_nodes,
                    pref,
                    depot,
                    augmented_visited,
                    snapshot,
                )
            )

        rows.append(row)
        printable = [f"[{idx:03d}]", f"served={row['served_before_insertion']}"]
        for method in method_names:
            result = row[method]
            time_value = result.get("total_elapsed_s", result.get("elapsed_s", 0.0))
            printable.append(
                f"{method}={result['cost']:.4f} pen={result['pen']:.3f} "
                f"feas={int(bool(result['feasible']))} time={time_value:.3f}s"
            )
        print(" | ".join(printable))

    output = {
        "config": {
            "dataset": args.dataset,
            "instances": len(instances),
            "n_nodes": args.n_nodes,
            "problem_type": args.problem_type,
            "scenario": args.scenario,
            "tightness": args.tightness,
            "pref_mode": args.pref_mode,
            "pref": [float(instances[0]["pref"][0, 0].item()), float(instances[0]["pref"][0, 1].item())],
            "pref_random_low": args.pref_random_low,
            "pref_random_high": args.pref_random_high,
            "executed_ratio": args.executed_ratio,
            "emergency_count": args.emergency_count,
            "urgency_window": args.urgency_window,
            "checkpoint": args.checkpoint,
            "alns_time_limit": args.alns_time_limit,
            "ortools_time_limit": args.ortools_time_limit,
            "pyvrp_time_limit": args.pyvrp_time_limit,
            "seed": args.seed,
        },
        "summary": summarize_rows(rows, method_names),
        "rows": rows,
    }

    if args.output_json:
        output_dir = os.path.dirname(os.path.abspath(args.output_json))
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"Saved result json: {args.output_json}")
    print(json.dumps(output["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
