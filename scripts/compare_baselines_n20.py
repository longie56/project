import argparse
import json
import os
import random
import sys
import time

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alns.alns_solver import HybridALNSSolver
from config import DEVICE, PEN_WEIGHT, PHYS_WEIGHT
from data.solomon_generator import SolomonGenerator
from eval.ortools_baseline import ORToolsBaselineSolver
from eval.ortools_physical_baseline import ORToolsPhysicalBaselineSolver
from eval.pyvrp_baseline import PyVRPBaselineSolver
from eval.route_baseline_utils import choose_best_action, evaluate_actions_with_env, priority_to_actions
from env.vrp_env import VRPTWEnv
from model.pomo import PC_EGA_POMO
from model.variants import VARIANTS

try:
    from eval.lkh3_baseline import LKH3BaselineSolver
except ModuleNotFoundError:
    LKH3BaselineSolver = None


def load_model(variant, checkpoint_path):
    state = None
    encoder_ff_hidden = None
    if checkpoint_path:
        try:
            ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
        except TypeError:
            ckpt = torch.load(checkpoint_path, map_location=DEVICE)
        state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        ff_weight = state.get("encoder.encoder.layers.0.linear1.weight") if isinstance(state, dict) else None
        if ff_weight is not None and ff_weight.dim() == 2:
            encoder_ff_hidden = int(ff_weight.shape[0])
    model = PC_EGA_POMO(**VARIANTS[variant], encoder_ff_hidden=encoder_ff_hidden).to(DEVICE)
    if state is not None:
        missing, unexpected = model.load_state_dict(state, strict=False)
        if encoder_ff_hidden is not None:
            print(
                f"Loaded {checkpoint_path} with encoder_ff_hidden={encoder_ff_hidden} "
                f"(missing={len(missing)}, unexpected={len(unexpected)})"
            )
    model.eval()
    return model


def _extract_customer_priority(actions, n_nodes, visited_mask):
    seen = set()
    priority = []
    initially_visited = visited_mask[0].detach().cpu().bool().tolist()
    for action in actions:
        action = int(action)
        if action < 0 or action >= n_nodes * 2:
            continue
        node = action // 2
        if initially_visited[node] or node in seen:
            continue
        seen.add(node)
        priority.append(node)

    missing = [
        idx
        for idx in range(n_nodes)
        if not initially_visited[idx] and idx not in seen
    ]
    return priority + missing


def _repair_by_appending_missing_actions(nodes, pref, depot, actions, visited_mask, scenario):
    device = nodes.device
    batch_size, n_nodes, _ = nodes.shape
    if batch_size != 1:
        raise ValueError("_repair_by_appending_missing_actions only supports batch size 1.")

    env = VRPTWEnv(nodes.clone(), pref.clone(), depot.clone(), scenario=scenario)
    env.load_state(
        visited_mask.clone(),
        depot.clone(),
        torch.zeros(batch_size, device=device),
        torch.ones(batch_size, device=device),
        torch.ones(batch_size, device=device),
    )

    repaired_actions = []
    for action in actions:
        action = int(action)
        mask = env.get_mask()
        if 0 <= action < mask.size(1) and mask[0, action]:
            env.step(torch.tensor([action], device=device))
            repaired_actions.append(action)
        if env._compute_done().all():
            break

    return_actions = [n_nodes * 2, n_nodes * 2 + 1]
    for node_idx in range(n_nodes):
        if env.visited[0, node_idx]:
            continue

        mask = env.get_mask()
        customer_actions = []
        normal_action = node_idx * 2
        fast_action = node_idx * 2 + 1
        if mask[0, normal_action]:
            customer_actions.append(normal_action)
        if mask[0, fast_action]:
            customer_actions.append(fast_action)

        selected_action = choose_best_action(env, customer_actions)
        if selected_action is None and not env._at_depot()[0]:
            return_action = choose_best_action(env, [a for a in return_actions if mask[0, a]])
            if return_action is not None:
                env.step(torch.tensor([return_action], device=device))
                repaired_actions.append(return_action)
                mask = env.get_mask()
                customer_actions = []
                if mask[0, normal_action]:
                    customer_actions.append(normal_action)
                if mask[0, fast_action]:
                    customer_actions.append(fast_action)
                selected_action = choose_best_action(env, customer_actions)

        if selected_action is None:
            continue

        env.step(torch.tensor([selected_action], device=device))
        repaired_actions.append(selected_action)

    if not env._at_depot()[0]:
        mask = env.get_mask()
        return_action = choose_best_action(env, [a for a in return_actions if mask[0, a]])
        if return_action is not None:
            env.step(torch.tensor([return_action], device=device))
            repaired_actions.append(return_action)

    unvisit = (~env.visited).sum(-1).float()
    total_cost = env.phys * PHYS_WEIGHT + env.pen * PEN_WEIGHT + unvisit * 20.0
    feasible = bool((unvisit == 0).all().item() and env._at_depot().all().item() and env._compute_done().all().item())
    return {
        "actions": repaired_actions,
        "cost": float(total_cost.item()),
        "phys": float(env.phys.item()),
        "pen": float(env.pen.item()),
        "unvisit": float(unvisit.item()),
        "feasible": feasible,
    }


def solve_drl(
    model,
    nodes,
    pref,
    depot,
    visited_mask,
    n_pomo,
    decode_type,
    sample_rollouts,
    sample_temperature,
    multi_start_rollouts,
    augment_8_fold,
    scenario,
    repair_mode="none",
):
    t0 = time.perf_counter()
    with torch.no_grad():
        result = model.inference_rollout(
            nodes,
            pref,
            depot,
            visited_mask=visited_mask,
            N_pomo=n_pomo,
            decode_type=decode_type,
            sample_rollouts=sample_rollouts,
            sample_temperature=sample_temperature,
            multi_start_rollouts=multi_start_rollouts,
            augment_8_fold=augment_8_fold,
            scenario=scenario,
        )
    route_len = int(result["best_route_lengths"][0].item()) if result.get("best_route_lengths") is not None else result["best_routes"].size(1)
    metrics = {
        "cost": float(result["best_cost"][0].item()),
        "phys": float(result["best_phys"][0].item()),
        "pen": float(result["best_pen"][0].item()),
        "unvisit": float(result["best_unvisit"][0].item()),
        "feasible": bool(result["feasible"][0].item()),
        "route_actions": [int(x) for x in result["best_routes"][0, :route_len].detach().cpu().tolist()],
        "repair_applied": False,
    }

    if repair_mode != "none" and not metrics["feasible"]:
        repair_candidates = []
        if repair_mode == "priority":
            priority = _extract_customer_priority(metrics["route_actions"], nodes.size(1), visited_mask)
            repair_candidates.append(
                priority_to_actions(
                    nodes=nodes,
                    pref=pref,
                    depot=depot,
                    priority=priority,
                    visited_mask=visited_mask,
                    scenario=scenario,
                )
            )
        repair_candidates.append(
            _repair_by_appending_missing_actions(
                nodes=nodes,
                pref=pref,
                depot=depot,
                actions=metrics["route_actions"],
                visited_mask=visited_mask,
                scenario=scenario,
            )
        )
        repaired = min(repair_candidates, key=lambda x: (not x["feasible"], x["unvisit"], x["cost"]))
        original_key = (not metrics["feasible"], metrics["unvisit"], metrics["cost"])
        repaired_key = (not repaired["feasible"], repaired["unvisit"], repaired["cost"])
        if repaired_key < original_key:
            metrics.update(
                {
                    "cost": float(repaired["cost"]),
                    "phys": float(repaired["phys"]),
                    "pen": float(repaired["pen"]),
                    "unvisit": float(repaired["unvisit"]),
                    "feasible": bool(repaired["feasible"]),
                    "route_actions": [int(x) for x in repaired["actions"]],
                    "repair_applied": True,
                }
            )

    metrics["elapsed_s"] = time.perf_counter() - t0
    return metrics


def solve_alns(
    solver,
    nodes,
    pref,
    depot,
    visited_mask,
    n_pomo,
    decode_type,
    sample_rollouts,
    sample_temperature,
    multi_start_rollouts,
    augment_8_fold,
    args,
    initial_actions,
    time_limit_s=None,
):
    batch_size = nodes.size(0)
    effective_time_limit = args.alns_time_limit if time_limit_s is None else float(time_limit_s)
    effective_time_limit = max(0.05, effective_time_limit)
    t0 = time.perf_counter()
    result = solver.optimize(
        nodes=nodes,
        pref=pref,
        depot=depot,
        visited_mask=visited_mask,
        curr_pos=depot.clone(),
        curr_time=torch.zeros(batch_size, device=nodes.device),
        curr_load=torch.ones(batch_size, device=nodes.device),
        curr_energy=torch.ones(batch_size, device=nodes.device),
        max_alns_iter=args.alns_max_iter,
        time_limit_s=effective_time_limit,
        destroy_min=args.alns_destroy_min,
        destroy_max=args.alns_destroy_max,
        max_insert_positions=args.alns_max_insert_positions,
        swap_neighbors=args.alns_swap_neighbors,
        n_pomo=n_pomo,
        decode_type=decode_type,
        sample_rollouts=sample_rollouts,
        sample_temperature=sample_temperature,
        multi_start_rollouts=multi_start_rollouts,
        augment_8_fold=augment_8_fold,
        initial_actions=initial_actions,
        repair_mode=args.alns_repair_mode,
        guided_attempts=args.alns_guided_attempts,
        guided_min_keep_ratio=args.alns_guided_min_keep_ratio,
        guided_decode_type=args.alns_guided_decode_type,
        guided_sample_rollouts=args.alns_guided_sample_rollouts,
        guided_pomo_size=args.alns_guided_pomo,
        guided_restarts=args.alns_guided_restarts,
        init_source=args.alns_init_source,
        restart_rounds=args.alns_restart_rounds,
    )
    elapsed = time.perf_counter() - t0
    best = result["best_solution"]
    if initial_actions:
        initial_eval = evaluate_actions_with_env(
            nodes=nodes,
            pref=pref,
            depot=depot,
            actions=initial_actions,
            visited_mask=visited_mask,
            scenario=args.scenario,
        )
        best_key = (not best["feasible"], best["unvisit"], best["cost"])
        initial_key = (not initial_eval["feasible"], initial_eval["unvisit"], initial_eval["cost"])
        if initial_key < best_key:
            best = initial_eval
            best["priority"] = []
    return {
        "cost": float(best["cost"]),
        "phys": float(best["phys"]),
        "pen": float(best["pen"]),
        "unvisit": float(best["unvisit"]),
        "feasible": bool(best["feasible"]),
        "actions": [int(x) for x in best.get("actions", [])],
        "priority": [int(x) for x in best.get("priority", [])],
        "initial_cost": float(result.get("initial_solution", {}).get("cost", float("inf"))),
        "iterations": int(result.get("iterations", 0)),
        "elapsed_s": elapsed,
        "time_limit_s": effective_time_limit,
    }


def _finite_values(values):
    out = []
    for value in values:
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if torch.isfinite(torch.tensor(value, dtype=torch.float32)):
            out.append(value)
    return out


def _mean(values):
    values = _finite_values(values)
    return sum(values) / len(values) if values else None


def _std(values):
    values = _finite_values(values)
    if not values:
        return None
    if len(values) == 1:
        return 0.0
    return float(torch.tensor(values, dtype=torch.float32).std(unbiased=True).item())


def _percentile(values, q):
    values = _finite_values(values)
    if not values:
        return None
    return float(torch.quantile(torch.tensor(values, dtype=torch.float32), q).item())


def summarize_rows(rows, method_names):
    summary = {"instances": len(rows)}
    for method in method_names:
        method_rows = [r[method] for r in rows]
        costs = [x.get("cost") for x in method_rows]
        phys = [x.get("phys") for x in method_rows]
        pen = [x.get("pen") for x in method_rows]
        unvisit = [x.get("unvisit") for x in method_rows]
        times = [x.get("elapsed_s") for x in method_rows]
        method_summary = {
            "avg_cost": _mean(costs),
            "std_cost": _std(costs),
            "median_cost": _percentile(costs, 0.5),
            "p90_cost": _percentile(costs, 0.9),
            "avg_phys": _mean(phys),
            "std_phys": _std(phys),
            "avg_pen": _mean(pen),
            "std_pen": _std(pen),
            "avg_unvisit": _mean(unvisit),
            "feasible_rate": sum(float(x["feasible"]) for x in method_rows) / len(method_rows),
            "avg_time_s": _mean(times),
            "std_time_s": _std(times),
            "median_time_s": _percentile(times, 0.5),
            "p90_time_s": _percentile(times, 0.9),
        }
        total_times = [x.get("total_elapsed_s") for x in method_rows if x.get("total_elapsed_s") is not None]
        if total_times:
            method_summary["avg_total_time_s"] = sum(total_times) / len(total_times)
        vehicle_values = [
            int(x["num_vehicles"])
            for x in method_rows
            if x.get("num_vehicles") is not None
        ]
        if vehicle_values:
            counts = {}
            for value in vehicle_values:
                counts[str(value)] = counts.get(str(value), 0) + 1
            method_summary["num_vehicles_counts"] = counts
            method_summary["avg_num_vehicles"] = sum(vehicle_values) / len(vehicle_values)
        summary[method] = method_summary
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instances", type=int, default=10)
    parser.add_argument("--n-nodes", type=int, default=20)
    parser.add_argument("--problem-type", default="R")
    parser.add_argument("--scenario", choices=["full", "simple"], default="full")
    parser.add_argument("--tightness", type=float, default=0.7)
    parser.add_argument("--pref-mode", choices=["fixed", "random"], default="fixed")
    parser.add_argument("--pref-alpha", type=float, default=0.5)
    parser.add_argument("--pref-beta", type=float, default=0.5)
    parser.add_argument("--pref-random-low", type=float, default=0.1)
    parser.add_argument("--pref-random-high", type=float, default=0.9)
    parser.add_argument("--variant", choices=list(VARIANTS.keys()), default="full")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--decode-type", choices=["greedy", "sample"], default="greedy")
    parser.add_argument("--n-pomo", type=int, default=None)
    parser.add_argument("--sample-rollouts", type=int, default=0)
    parser.add_argument("--sample-temperature", type=float, default=1.0)
    parser.add_argument("--multi-start-rollouts", type=int, default=0)
    parser.add_argument("--augment-8-fold", action="store_true")
    parser.add_argument("--drl-repair", choices=["none", "append", "priority"], default="none")
    parser.add_argument("--run-alns", action="store_true")
    parser.add_argument("--run-drl", action="store_true")
    parser.add_argument("--run-ortools", action="store_true")
    parser.add_argument("--run-ortools-physical", action="store_true")
    parser.add_argument("--run-pyvrp", action="store_true")
    parser.add_argument("--run-lkh3", action="store_true")
    parser.add_argument("--lkh3-path", default="LKH")
    parser.add_argument("--lkh3-time-limit", type=float, default=10.0)
    parser.add_argument("--lkh3-max-trials", type=int, default=1000)
    parser.add_argument("--lkh3-runs", type=int, default=1)
    parser.add_argument("--ortools-time-limit", type=float, default=5.0)
    parser.add_argument("--ortools-num-vehicles", type=int, default=0)
    parser.add_argument("--ortools-vehicle-candidates", default="")
    parser.add_argument("--ortools-first-solution-candidates", default="")
    parser.add_argument("--ortools-local-search-candidates", default="")
    parser.add_argument("--ortools-first-solution", default="PATH_CHEAPEST_ARC")
    parser.add_argument("--ortools-local-search", default="GUIDED_LOCAL_SEARCH")
    parser.add_argument("--ortools-waiting-slack", type=float, default=2.0)
    parser.add_argument("--ortools-route-time-limit", type=float, default=3.0)
    parser.add_argument("--ortools-share-candidate-time", action="store_true")
    parser.add_argument(
        "--ortools-physical-routing-mode",
        choices=["serial_order", "parallel_routes"],
        default="serial_order",
    )
    parser.add_argument("--pyvrp-time-limit", type=float, default=10.0)
    parser.add_argument("--pyvrp-num-vehicles", type=int, default=0)
    parser.add_argument("--pyvrp-vehicle-candidates", default="")
    parser.add_argument("--pyvrp-route-time-limit", type=float, default=3.0)
    parser.add_argument("--pyvrp-share-candidate-time", action="store_true")
    parser.add_argument("--alns-time-limit", type=float, default=2.0)
    parser.add_argument(
        "--total-time-budget",
        type=float,
        default=0.0,
        help="If > 0 and DRL+ALNS are both enabled, ALNS receives the remaining per-instance time budget.",
    )
    parser.add_argument(
        "--time-budget-safety",
        type=float,
        default=0.25,
        help="Seconds subtracted from the remaining total budget to avoid small wall-clock overruns.",
    )
    parser.add_argument("--alns-max-iter", type=int, default=300)
    parser.add_argument("--alns-destroy-min", type=int, default=1)
    parser.add_argument("--alns-destroy-max", type=int, default=3)
    parser.add_argument("--alns-max-insert-positions", type=int, default=5)
    parser.add_argument("--alns-swap-neighbors", type=int, default=16)
    parser.add_argument("--alns-repair-mode", choices=["classic", "model_guided"], default="classic")
    parser.add_argument("--alns-guided-attempts", type=int, default=3)
    parser.add_argument("--alns-guided-min-keep-ratio", type=float, default=0.35)
    parser.add_argument("--alns-guided-decode-type", choices=["greedy", "sample"], default="sample")
    parser.add_argument("--alns-guided-sample-rollouts", type=int, default=1)
    parser.add_argument("--alns-guided-pomo", type=int, default=4)
    parser.add_argument("--alns-guided-restarts", type=int, default=2)
    parser.add_argument("--alns-init-source", choices=["auto", "model", "heuristic"], default="auto")
    parser.add_argument("--alns-restart-rounds", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()
    if args.n_pomo is None:
        args.n_pomo = args.n_nodes

    if args.run_lkh3 and LKH3BaselineSolver is None:
        raise RuntimeError("LKH-3 baseline module is unavailable in this workspace.")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    generator = SolomonGenerator()
    nodes_batch, depot_batch, visited_batch = generator.generate(
        batch_size=args.instances,
        n_nodes=args.n_nodes,
        problem_type=args.problem_type,
        tightness=args.tightness,
        device=DEVICE,
    )
    if args.pref_mode == "random":
        if not (0.0 <= args.pref_random_low < args.pref_random_high <= 1.0):
            raise ValueError("--pref-random-low/high must satisfy 0 <= low < high <= 1.")
        alpha_pref = (
            torch.rand(args.instances, 1, device=DEVICE)
            * (args.pref_random_high - args.pref_random_low)
            + args.pref_random_low
        )
        pref_batch = torch.cat([alpha_pref, 1.0 - alpha_pref], dim=1)
    else:
        pref_batch = torch.tensor(
            [[args.pref_alpha, args.pref_beta]], dtype=torch.float32, device=DEVICE
        ).expand(args.instances, -1)
        pref_batch = pref_batch / pref_batch.sum(dim=-1, keepdim=True).clamp_min(1e-9)

    need_model = args.run_drl or (args.run_alns and args.alns_init_source != "heuristic")
    model = None
    alns_solver = None
    if need_model:
        model = load_model(args.variant, args.checkpoint)
    if args.run_alns:
        alns_solver = HybridALNSSolver(model, scenario=args.scenario)

    ortools_solver = ORToolsBaselineSolver(
        time_limit_s=args.ortools_time_limit,
        num_vehicles=(args.ortools_num_vehicles if args.ortools_num_vehicles > 0 else None),
        vehicle_candidates=(
            [int(tok.strip()) for tok in args.ortools_vehicle_candidates.split(",") if tok.strip()]
            if args.ortools_vehicle_candidates
            else None
        ),
        first_solution_candidates=(
            [tok.strip() for tok in args.ortools_first_solution_candidates.split(",") if tok.strip()]
            if args.ortools_first_solution_candidates
            else None
        ),
        local_search_candidates=(
            [tok.strip() for tok in args.ortools_local_search_candidates.split(",") if tok.strip()]
            if args.ortools_local_search_candidates
            else None
        ),
        first_solution_strategy=args.ortools_first_solution,
        local_search_metaheuristic=args.ortools_local_search,
        waiting_slack=args.ortools_waiting_slack,
        route_time_limit=args.ortools_route_time_limit,
        scenario=args.scenario,
        share_candidate_time=args.ortools_share_candidate_time,
    ) if args.run_ortools else None
    ortools_physical_solver = ORToolsPhysicalBaselineSolver(
        time_limit_s=args.ortools_time_limit,
        num_vehicles=(args.ortools_num_vehicles if args.ortools_num_vehicles > 0 else None),
        vehicle_candidates=(
            [int(tok.strip()) for tok in args.ortools_vehicle_candidates.split(",") if tok.strip()]
            if args.ortools_vehicle_candidates
            else None
        ),
        first_solution_candidates=(
            [tok.strip() for tok in args.ortools_first_solution_candidates.split(",") if tok.strip()]
            if args.ortools_first_solution_candidates
            else None
        ),
        local_search_candidates=(
            [tok.strip() for tok in args.ortools_local_search_candidates.split(",") if tok.strip()]
            if args.ortools_local_search_candidates
            else None
        ),
        first_solution_strategy=args.ortools_first_solution,
        local_search_metaheuristic=args.ortools_local_search,
        waiting_slack=args.ortools_waiting_slack,
        route_time_limit=args.ortools_route_time_limit,
        scenario=args.scenario,
        routing_mode=args.ortools_physical_routing_mode,
        share_candidate_time=args.ortools_share_candidate_time,
    ) if args.run_ortools_physical else None
    pyvrp_solver = PyVRPBaselineSolver(
        time_limit_s=args.pyvrp_time_limit,
        num_vehicles=(args.pyvrp_num_vehicles if args.pyvrp_num_vehicles > 0 else None),
        vehicle_candidates=(
            [int(tok.strip()) for tok in args.pyvrp_vehicle_candidates.split(",") if tok.strip()]
            if args.pyvrp_vehicle_candidates
            else None
        ),
        route_time_limit=args.pyvrp_route_time_limit,
        scenario=args.scenario,
        share_candidate_time=args.pyvrp_share_candidate_time,
    ) if args.run_pyvrp else None
    lkh3_solver = LKH3BaselineSolver(
        executable_path=args.lkh3_path,
        max_trials=args.lkh3_max_trials,
        runs=args.lkh3_runs,
        time_limit_s=args.lkh3_time_limit,
    ) if args.run_lkh3 else None

    method_names = []
    if args.run_drl:
        method_names.append("drl")
    if args.run_alns:
        method_names.append("alns")
    if args.run_ortools:
        method_names.append("ortools")
    if args.run_ortools_physical:
        method_names.append("ortools_physical")
    if args.run_pyvrp:
        method_names.append("pyvrp")
    if args.run_lkh3:
        method_names.append("lkh3")
    if not method_names:
        raise ValueError(
            "Please enable at least one method: "
            "--run-drl/--run-alns/--run-ortools/--run-ortools-physical/--run-lkh3."
        )

    rows = []
    for idx in range(args.instances):
        nodes = nodes_batch[idx : idx + 1]
        depot = depot_batch[idx : idx + 1]
        visited = visited_batch[idx : idx + 1]
        pref_i = pref_batch[idx : idx + 1].clone()

        row = {
            "instance": idx,
            "pref": [float(pref_i[0, 0].item()), float(pref_i[0, 1].item())],
        }
        if args.run_drl:
            row["drl"] = solve_drl(
                model,
                nodes,
                pref_i,
                depot,
                visited,
                args.n_pomo,
                args.decode_type,
                args.sample_rollouts,
                args.sample_temperature,
                args.multi_start_rollouts,
                args.augment_8_fold,
                args.scenario,
                args.drl_repair,
            )
        if args.run_alns:
            if args.run_drl:
                initial_actions = row["drl"]["route_actions"]
            else:
                initial_actions = None
            alns_time_limit = args.alns_time_limit
            if args.total_time_budget > 0 and args.run_drl:
                remaining = args.total_time_budget - row["drl"]["elapsed_s"] - args.time_budget_safety
                alns_time_limit = min(args.alns_time_limit, max(0.05, remaining))
            row["alns"] = solve_alns(
                alns_solver,
                nodes,
                pref_i,
                depot,
                visited,
                args.n_pomo,
                args.decode_type,
                args.sample_rollouts,
                args.sample_temperature,
                args.multi_start_rollouts,
                args.augment_8_fold,
                args,
                initial_actions,
                time_limit_s=alns_time_limit,
            )
            if args.run_drl:
                row["alns"]["total_elapsed_s"] = row["drl"]["elapsed_s"] + row["alns"]["elapsed_s"]
        if args.run_ortools:
            row["ortools"] = ortools_solver.solve_instance(nodes, pref_i, depot, visited_mask=visited)
        if args.run_ortools_physical:
            row["ortools_physical"] = ortools_physical_solver.solve_instance(
                nodes,
                pref_i,
                depot,
                visited_mask=visited,
            )
        if args.run_pyvrp:
            row["pyvrp"] = pyvrp_solver.solve_instance(nodes, pref_i, depot, visited_mask=visited)
        if args.run_lkh3:
            row["lkh3"] = lkh3_solver.solve_instance(nodes, pref_i, depot, visited_mask=visited)
        rows.append(row)

        printable = [f"[{idx:03d}]"]
        for method in method_names:
            result = row[method]
            vehicle_suffix = (
                f" veh={int(result['num_vehicles'])}"
                if result.get("num_vehicles") is not None
                else ""
            )
            printable.append(
                f"{method.upper()}={result['cost']:.4f} feas={int(result['feasible'])}"
                f"{vehicle_suffix} time={result['elapsed_s']:.3f}s"
            )
        print(" | ".join(printable))

    output = {
        "config": {
            "instances": args.instances,
            "n_nodes": args.n_nodes,
            "problem_type": args.problem_type,
            "scenario": args.scenario,
            "tightness": args.tightness,
            "pref_mode": args.pref_mode,
            "pref": [float(pref_batch[0, 0].item()), float(pref_batch[0, 1].item())],
            "pref_random_low": args.pref_random_low,
            "pref_random_high": args.pref_random_high,
            "variant": args.variant,
            "checkpoint": args.checkpoint,
            "decode_type": args.decode_type,
            "n_pomo": args.n_pomo,
            "sample_rollouts": args.sample_rollouts,
            "sample_temperature": args.sample_temperature,
            "multi_start_rollouts": args.multi_start_rollouts,
            "augment_8_fold": args.augment_8_fold,
            "drl_repair": args.drl_repair,
            "ortools_time_limit": args.ortools_time_limit,
            "run_ortools_physical": args.run_ortools_physical,
            "ortools_physical_routing_mode": args.ortools_physical_routing_mode,
            "ortools_num_vehicles": args.ortools_num_vehicles,
            "ortools_vehicle_candidates": args.ortools_vehicle_candidates,
            "ortools_first_solution_candidates": args.ortools_first_solution_candidates,
            "ortools_local_search_candidates": args.ortools_local_search_candidates,
            "ortools_first_solution": args.ortools_first_solution,
            "ortools_local_search": args.ortools_local_search,
            "ortools_waiting_slack": args.ortools_waiting_slack,
            "ortools_route_time_limit": args.ortools_route_time_limit,
            "ortools_share_candidate_time": args.ortools_share_candidate_time,
            "pyvrp_time_limit": args.pyvrp_time_limit,
            "pyvrp_num_vehicles": args.pyvrp_num_vehicles,
            "pyvrp_vehicle_candidates": args.pyvrp_vehicle_candidates,
            "pyvrp_route_time_limit": args.pyvrp_route_time_limit,
            "pyvrp_share_candidate_time": args.pyvrp_share_candidate_time,
            "alns_time_limit": args.alns_time_limit,
            "total_time_budget": args.total_time_budget,
            "time_budget_safety": args.time_budget_safety,
            "alns_init_source": args.alns_init_source,
            "alns_restart_rounds": args.alns_restart_rounds,
            "lkh3_time_limit": args.lkh3_time_limit,
            "lkh3_max_trials": args.lkh3_max_trials,
            "lkh3_runs": args.lkh3_runs,
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
