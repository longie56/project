import argparse
import json
import os
import random
import sys
import time
from statistics import median

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DEVICE, PEN_WEIGHT, PHYS_WEIGHT
from alns.alns_solver import HybridALNSSolver
from data.solomon_generator import SolomonGenerator
from eval.gurobi_baseline import GurobiN20Solver
from eval.route_baseline_utils import choose_best_action
from env.vrp_env import VRPTWEnv
from model.pomo import PC_EGA_POMO
from model.variants import VARIANTS


def _safe_mean(values):
    return sum(values) / len(values) if values else None


def _safe_std(values):
    if len(values) <= 1:
        return 0.0 if values else None
    tensor = torch.tensor(values, dtype=torch.float32)
    return float(tensor.std(unbiased=True).item())


def _safe_percentile(values, q):
    if not values:
        return None
    tensor = torch.tensor(values, dtype=torch.float32)
    return float(torch.quantile(tensor, q).item())


def _finite_or_none(value):
    if value is None:
        return None
    tensor = torch.tensor(value, dtype=torch.float32)
    return float(value) if torch.isfinite(tensor) else None


def _is_finite(value):
    if value is None:
        return False
    return bool(torch.isfinite(torch.tensor(value, dtype=torch.float32)).item())


def _fmt_metric(value, digits=4):
    if value is None:
        return "NA"
    tensor = torch.tensor(value, dtype=torch.float32)
    if not torch.isfinite(tensor):
        return "NA"
    return f"{float(value):.{digits}f}"


def _extract_nested_actions(row, method_name):
    payload = row.get(method_name)
    if isinstance(payload, dict):
        actions = payload.get("actions") or payload.get("route_actions")
        if actions:
            return [int(x) for x in actions]

    flat_keys = [
        f"{method_name}_actions",
        f"{method_name}_route_actions",
        "actions" if method_name == "gurobi" else "",
    ]
    for key in flat_keys:
        if key and row.get(key):
            return [int(x) for x in row[key]]
    return None


def load_model(variant, checkpoint_path):
    model = PC_EGA_POMO(**VARIANTS[variant]).to(DEVICE)
    if checkpoint_path:
        try:
            ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
        except TypeError:
            ckpt = torch.load(checkpoint_path, map_location=DEVICE)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            state = ckpt["model_state_dict"]
        else:
            state = ckpt
        model.load_state_dict(state, strict=False)
    model.eval()
    return model


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

    if repair_mode == "append" and not metrics["feasible"]:
        repaired = _repair_by_appending_missing_actions(
            nodes=nodes,
            pref=pref,
            depot=depot,
            actions=metrics["route_actions"],
            visited_mask=visited_mask,
            scenario=scenario,
        )
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
    alns_solver,
    nodes,
    pref,
    depot,
    visited_mask,
    n_pomo,
    decode_type,
    time_limit_s,
    max_iter,
    destroy_min,
    destroy_max,
    max_insert_positions,
    swap_neighbors,
    sample_rollouts,
    sample_temperature,
    multi_start_rollouts,
    augment_8_fold,
    initial_actions=None,
    restart_rounds=1,
):
    batch_size = nodes.size(0)
    curr_pos = depot.clone()
    curr_time = torch.zeros(batch_size, device=nodes.device)
    curr_load = torch.ones(batch_size, device=nodes.device)
    curr_energy = torch.ones(batch_size, device=nodes.device)

    t0 = time.perf_counter()
    result = alns_solver.optimize(
        nodes=nodes,
        pref=pref,
        depot=depot,
        visited_mask=visited_mask,
        curr_pos=curr_pos,
        curr_time=curr_time,
        curr_load=curr_load,
        curr_energy=curr_energy,
        max_alns_iter=max_iter,
        time_limit_s=time_limit_s,
        destroy_min=destroy_min,
        destroy_max=destroy_max,
        max_insert_positions=max_insert_positions,
        swap_neighbors=swap_neighbors,
        n_pomo=n_pomo,
        decode_type=decode_type,
        sample_rollouts=sample_rollouts,
        sample_temperature=sample_temperature,
        multi_start_rollouts=multi_start_rollouts,
        augment_8_fold=augment_8_fold,
        initial_actions=initial_actions,
        restart_rounds=restart_rounds,
    )
    elapsed = time.perf_counter() - t0
    initial = result["initial_solution"]
    best = result["best_solution"]
    return {
        "initial_cost": float(initial["cost"]),
        "cost": float(best["cost"]),
        "phys": float(best["phys"]),
        "pen": float(best["pen"]),
        "unvisit": float(best["unvisit"]),
        "feasible": bool(best["feasible"]),
        "actions": [int(x) for x in best.get("actions", [])],
        "priority": [int(x) for x in best.get("priority", [])],
        "elapsed_s": elapsed,
        "iterations": int(result["iterations"]),
        "improve_pct_vs_initial": (
            (float(initial["cost"]) - float(best["cost"])) / max(float(initial["cost"]), 1e-9) * 100.0
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instances", type=int, default=10)
    parser.add_argument("--n-nodes", type=int, default=20)
    parser.add_argument("--problem-type", default="R")
    parser.add_argument("--scenario", choices=["full", "simple"], default="full")
    parser.add_argument("--tightness", type=float, default=0.7)
    parser.add_argument("--variant", choices=list(VARIANTS.keys()), default="full")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--n-pomo", type=int, default=None)
    parser.add_argument("--decode-type", choices=["greedy", "sample"], default="greedy")
    parser.add_argument("--sample-rollouts", type=int, default=0)
    parser.add_argument("--sample-temperature", type=float, default=1.0)
    parser.add_argument("--multi-start-rollouts", type=int, default=0)
    parser.add_argument("--augment-8-fold", action="store_true")
    parser.add_argument("--drl-repair", choices=["none", "append"], default="none")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pref-mode", choices=["fixed", "random"], default="fixed")
    parser.add_argument("--pref-alpha", type=float, default=0.5)
    parser.add_argument("--pref-beta", type=float, default=0.5)
    parser.add_argument("--pref-random-low", type=float, default=0.1)
    parser.add_argument("--pref-random-high", type=float, default=0.9)
    parser.add_argument("--gurobi-time-limit", type=float, default=60.0)
    parser.add_argument("--gurobi-time-limit-alt", type=float, default=0.0)
    parser.add_argument("--gurobi-mip-gap", type=float, default=0.01)
    parser.add_argument(
        "--gurobi-max-returns",
        type=int,
        default=0,
        help=(
            "Maximum depot-return actions allowed in the MILP. Use 0 to auto-tighten "
            "from warm starts when available; this greatly improves proof strength."
        ),
    )
    parser.add_argument("--gurobi-proof-mode", action="store_true")
    parser.add_argument("--gurobi-threads", type=int, default=0)
    parser.add_argument("--gurobi-log-file", default="")
    parser.add_argument("--gurobi-nodefile-start-gb", type=float, default=0.5)
    parser.add_argument("--gurobi-nodefile-dir", default="")
    parser.add_argument("--skip-gurobi", action="store_true")
    parser.add_argument("--reference-json", default="")
    parser.add_argument("--gurobi-warm-start-json", default="")
    parser.add_argument("--gurobi-warm-start-methods", default="ortools_physical,alns,drl")
    parser.add_argument("--disable-drl-warm-start", action="store_true")
    parser.add_argument("--run-alns", action="store_true")
    parser.add_argument("--alns-time-limit", type=float, default=2.0)
    parser.add_argument("--alns-max-iter", type=int, default=100)
    parser.add_argument("--alns-destroy-min", type=int, default=1)
    parser.add_argument("--alns-destroy-max", type=int, default=3)
    parser.add_argument("--alns-max-insert-positions", type=int, default=6)
    parser.add_argument("--alns-swap-neighbors", type=int, default=12)
    parser.add_argument("--alns-restart-rounds", type=int, default=1)
    parser.add_argument("--allow-nonoptimal", action="store_true")
    parser.add_argument("--gate-method", choices=["drl", "alns"], default="drl")
    parser.add_argument("--gate-policy", choices=["strict_optimal", "budget_feasible"], default="strict_optimal")
    parser.add_argument("--gate-gap-threshold", type=float, default=5.0)
    parser.add_argument("--gate-min-compared", type=int, default=1)
    parser.add_argument("--realtime-threshold-s", type=float, default=3.0)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()
    if args.n_pomo is None:
        args.n_pomo = args.n_nodes
    reference_rows = {}
    if args.reference_json:
        with open(args.reference_json, "r", encoding="utf-8") as f:
            reference_data = json.load(f)
        reference_rows = {
            int(row["instance"]): row
            for row in reference_data.get("rows", [])
            if row.get("gurobi_cost_env") not in [None, float("inf")]
        }
    warm_start_rows = {}
    warm_start_arg = args.gurobi_warm_start_methods.strip().lower()
    if warm_start_arg in {"", "none", "off", "false", "no"}:
        warm_start_methods = []
    else:
        warm_start_methods = [
            item.strip()
            for item in args.gurobi_warm_start_methods.split(",")
            if item.strip()
        ]
    if args.gurobi_warm_start_json:
        with open(args.gurobi_warm_start_json, "r", encoding="utf-8") as f:
            warm_start_data = json.load(f)
        warm_start_rows = {
            int(row["instance"]): row
            for row in warm_start_data.get("rows", [])
        }

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    # Keep the test set identical to compare_baselines_n20.py: generate data
    # before model construction, since model initialization consumes torch RNG.
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

    model = load_model(args.variant, args.checkpoint)
    alns_solver = HybridALNSSolver(model, scenario=args.scenario) if args.run_alns else None
    gurobi_solver = None if args.skip_gurobi else GurobiN20Solver(
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
    gurobi_solver_alt = None
    if (not args.skip_gurobi) and args.gurobi_time_limit_alt and args.gurobi_time_limit_alt > 0:
        gurobi_solver_alt = GurobiN20Solver(
            time_limit_s=args.gurobi_time_limit_alt,
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

    rows = []
    for idx in range(args.instances):
        nodes = nodes_batch[idx : idx + 1]
        depot = depot_batch[idx : idx + 1]
        visited = visited_batch[idx : idx + 1]
        pref_i = pref_batch[idx : idx + 1].clone()

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
            repair_mode=args.drl_repair,
        )

        alns = None
        if args.run_alns:
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
        if idx in warm_start_rows:
            for method_name in warm_start_methods:
                actions = _extract_nested_actions(warm_start_rows[idx], method_name)
                if actions:
                    warm_start_actions.append(actions)
        if alns is not None and alns.get("actions"):
            warm_start_actions.append(alns["actions"])
        if not args.disable_drl_warm_start:
            warm_start_actions.append(drl["route_actions"])

        deduped_warm_starts = []
        seen_warm_starts = set()
        for actions in warm_start_actions:
            key = tuple(int(x) for x in actions)
            if key and key not in seen_warm_starts:
                deduped_warm_starts.append([int(x) for x in actions])
                seen_warm_starts.add(key)

        gurobi = None
        gurobi_alt_eval = None
        gurobi_eval = None
        if not args.skip_gurobi:
            gurobi_t0 = time.perf_counter()
            try:
                gurobi = gurobi_solver.solve_instance(
                    nodes=nodes,
                    pref=pref_i,
                    depot=depot,
                    warm_start_actions=deduped_warm_starts if deduped_warm_starts else None,
                    visited_mask=visited,
                )
            except Exception as exc:
                gurobi = {
                    "elapsed_s": time.perf_counter() - gurobi_t0,
                    "mip_obj": float("inf"),
                    "obj_bound": float("inf"),
                    "mip_gap": float("inf"),
                    "status": -1,
                    "sol_count": 0,
                    "max_returns_used": None,
                    "env_eval": {
                        "actions": [],
                        "cost": float("inf"),
                        "phys": float("inf"),
                        "pen": float("inf"),
                        "unvisit": float("inf"),
                        "feasible": False,
                        "invalid_action": None,
                        "invalid_reason": f"gurobi_exception: {type(exc).__name__}: {exc}",
                    },
                }
            if gurobi_solver_alt is not None:
                try:
                    gurobi_alt = gurobi_solver_alt.solve_instance(
                        nodes=nodes,
                        pref=pref_i,
                        depot=depot,
                        warm_start_actions=deduped_warm_starts if deduped_warm_starts else None,
                        visited_mask=visited,
                    )
                    gurobi_alt_eval = gurobi_alt["env_eval"]
                except Exception:
                    gurobi_alt_eval = None
            gurobi_eval = gurobi["env_eval"]
        elif idx in reference_rows:
            ref = reference_rows[idx]
            gurobi_eval = {
                "cost": ref.get("gurobi_cost_env", float("inf")),
                "phys": ref.get("gurobi_phys_env", float("inf")),
                "pen": ref.get("gurobi_pen_env", float("inf")),
                "unvisit": ref.get("gurobi_unvisit_env", float("inf")),
                "feasible": ref.get("gurobi_feasible_env", False),
                "invalid_action": ref.get("gurobi_invalid_action"),
                "invalid_reason": ref.get("gurobi_invalid_reason", "loaded from reference json"),
            }
            gurobi = {
                "elapsed_s": ref.get("gurobi_time_s", 0.0),
                "mip_obj": ref.get("gurobi_mip_obj", ref.get("gurobi_cost_env", float("inf"))),
                "obj_bound": ref.get("gurobi_obj_bound", float("inf")),
                "mip_gap": ref.get("gurobi_mip_gap", float("inf")),
                "status": ref.get("gurobi_status"),
                "sol_count": ref.get("gurobi_sol_count", 0),
            }

        if gurobi_eval is None:
            gap_pct = None
        elif gurobi_eval["cost"] == float("inf") or gurobi_eval["cost"] <= 0:
            gap_pct = float("inf")
        else:
            gap_pct = (drl["cost"] - gurobi_eval["cost"]) / gurobi_eval["cost"] * 100.0

        gap_alns_pct = None if gurobi_eval is None else float("inf")
        if alns is not None and gurobi_eval is not None and gurobi_eval["cost"] != float("inf") and gurobi_eval["cost"] > 0:
            gap_alns_pct = (alns["cost"] - gurobi_eval["cost"]) / gurobi_eval["cost"] * 100.0

        if gurobi is None or gurobi_eval is None:
            obj_env_delta_pct = None
        elif gurobi["mip_obj"] == float("inf") or gurobi_eval["cost"] == float("inf"):
            obj_env_delta_pct = float("inf")
        else:
            obj_env_delta_pct = (gurobi_eval["cost"] - gurobi["mip_obj"]) / max(abs(gurobi["mip_obj"]), 1e-9) * 100.0

        gurobi_bound_finite = False
        drl_gap_to_bound_pct = None
        alns_gap_to_bound_pct = None
        if gurobi is not None:
            gurobi_bound_finite = _is_finite(gurobi.get("obj_bound")) and gurobi.get("obj_bound", 0.0) > 0
            if gurobi_bound_finite:
                drl_gap_to_bound_pct = (drl["cost"] - gurobi["obj_bound"]) / gurobi["obj_bound"] * 100.0
                if alns is not None:
                    alns_gap_to_bound_pct = (alns["cost"] - gurobi["obj_bound"]) / gurobi["obj_bound"] * 100.0

        row = {
            "instance": idx,
            "pref": [float(pref_i[0, 0].item()), float(pref_i[0, 1].item())],
            "drl_cost": drl["cost"],
            "drl_phys": drl["phys"],
            "drl_pen": drl["pen"],
            "drl_unvisit": drl["unvisit"],
            "drl_feasible": drl["feasible"],
            "drl_time_s": drl["elapsed_s"],
            "drl_repair_applied": bool(drl.get("repair_applied", False)),
            "drl_actions": [int(x) for x in drl.get("route_actions", [])],
            "alns_cost": None if alns is None else alns["cost"],
            "alns_phys": None if alns is None else alns["phys"],
            "alns_pen": None if alns is None else alns["pen"],
            "alns_unvisit": None if alns is None else alns["unvisit"],
            "alns_feasible": None if alns is None else alns["feasible"],
            "alns_time_s": None if alns is None else alns["elapsed_s"],
            "alns_actions": None if alns is None else [int(x) for x in alns.get("actions", [])],
            "total_time_s": drl["elapsed_s"] + (0.0 if alns is None else alns["elapsed_s"]),
            "alns_iterations": None if alns is None else alns["iterations"],
            "alns_improve_pct_vs_initial": None if alns is None else alns["improve_pct_vs_initial"],
            "alns_improve_pct_vs_drl": (
                None if alns is None else (drl["cost"] - alns["cost"]) / max(drl["cost"], 1e-9) * 100.0
            ),
            "gurobi_cost_env": None if gurobi_eval is None else gurobi_eval["cost"],
            "gurobi_phys_env": None if gurobi_eval is None else gurobi_eval["phys"],
            "gurobi_pen_env": None if gurobi_eval is None else gurobi_eval["pen"],
            "gurobi_unvisit_env": None if gurobi_eval is None else gurobi_eval["unvisit"],
            "gurobi_feasible_env": None if gurobi_eval is None else gurobi_eval["feasible"],
            "gurobi_invalid_action": None if gurobi_eval is None else gurobi_eval.get("invalid_action"),
            "gurobi_invalid_reason": None if gurobi_eval is None else gurobi_eval.get("invalid_reason", ""),
            "gurobi_actions": None if gurobi_eval is None else [int(x) for x in gurobi_eval.get("actions", [])],
            "gurobi_time_s": None if gurobi is None else gurobi["elapsed_s"],
            "gurobi_mip_obj": None if gurobi is None else gurobi["mip_obj"],
            "gurobi_obj_bound": None if gurobi is None else gurobi.get("obj_bound", float("inf")),
            "gurobi_mip_gap": None if gurobi is None else gurobi.get("mip_gap", float("inf")),
            "gurobi_bound_finite": None if gurobi is None else gurobi_bound_finite,
            "gurobi_status": None if gurobi is None else gurobi["status"],
            "gurobi_sol_count": 0 if gurobi is None else gurobi.get("sol_count", 0),
            "gurobi_max_returns_used": None if gurobi is None else gurobi.get("max_returns_used"),
            "gurobi_obj_env_delta_pct": obj_env_delta_pct,
            "gurobi_alt_time_limit_s": None if gurobi_alt_eval is None else args.gurobi_time_limit_alt,
            "gurobi_alt_cost_env": None if gurobi_alt_eval is None else gurobi_alt_eval["cost"],
            "gurobi_warm_start_count": len(deduped_warm_starts),
            "gap_drl_pct": gap_pct,
            "gap_alns_pct": gap_alns_pct,
            "drl_gap_to_gurobi_bound_pct": drl_gap_to_bound_pct,
            "alns_gap_to_gurobi_bound_pct": alns_gap_to_bound_pct,
        }
        rows.append(row)
        if args.skip_gurobi and args.run_alns and gurobi_eval is not None:
            print(
                f"[{idx:03d}] DRL={row['drl_cost']:.4f} | "
                f"ALNS={row['alns_cost']:.4f} | "
                f"REF={row['gurobi_cost_env']:.4f} | "
                f"gap(drl/alns)={row['gap_drl_pct']:.2f}%/{row['gap_alns_pct']:.2f}% | "
                f"improve={row['alns_improve_pct_vs_drl']:.2f}% | "
                f"time(drl/alns)={row['drl_time_s']:.3f}/{row['alns_time_s']:.3f}s"
            )
        elif args.skip_gurobi and args.run_alns:
            print(
                f"[{idx:03d}] DRL={row['drl_cost']:.4f} | "
                f"ALNS={row['alns_cost']:.4f} | "
                f"improve={row['alns_improve_pct_vs_drl']:.2f}% | "
                f"time(drl/alns)={row['drl_time_s']:.3f}/{row['alns_time_s']:.3f}s"
            )
        elif args.skip_gurobi and gurobi_eval is not None:
            print(
                f"[{idx:03d}] DRL={row['drl_cost']:.4f} | "
                f"REF={row['gurobi_cost_env']:.4f} | "
                f"gap={row['gap_drl_pct']:.2f}% | "
                f"feasible={row['drl_feasible']} | "
                f"time={row['drl_time_s']:.3f}s"
            )
        elif args.skip_gurobi:
            print(
                f"[{idx:03d}] DRL={row['drl_cost']:.4f} | "
                f"feasible={row['drl_feasible']} | "
                f"time={row['drl_time_s']:.3f}s"
            )
        elif args.run_alns:
            print(
                f"[{idx:03d}] DRL={row['drl_cost']:.4f} | "
                f"ALNS={row['alns_cost']:.4f} | "
                f"Gurobi(env)={row['gurobi_cost_env']:.4f} | "
                f"gap(drl/alns)={row['gap_drl_pct']:.2f}%/{row['gap_alns_pct']:.2f}% | "
                f"feas(drl/gb)={int(row['drl_feasible'])}/{int(row['gurobi_feasible_env'])} | "
                f"status={row['gurobi_status']} | sol={row['gurobi_sol_count']} | "
                f"bound={_fmt_metric(row['gurobi_obj_bound'])} | mip_gap={_fmt_metric(row['gurobi_mip_gap'])} | "
                f"cert_gap={_fmt_metric(row['drl_gap_to_gurobi_bound_pct'], 2)}% | "
                f"time(drl/alns/gb)={row['drl_time_s']:.3f}/{row['alns_time_s']:.3f}/{row['gurobi_time_s']:.3f}s"
            )
        else:
            print(
                f"[{idx:03d}] DRL={row['drl_cost']:.4f} | "
                f"Gurobi(env)={row['gurobi_cost_env']:.4f} | "
                f"gap={row['gap_drl_pct']:.2f}% | "
                f"feas(drl/gb)={int(row['drl_feasible'])}/{int(row['gurobi_feasible_env'])} | "
                f"status={row['gurobi_status']} | sol={row['gurobi_sol_count']} | "
                f"bound={_fmt_metric(row['gurobi_obj_bound'])} | mip_gap={_fmt_metric(row['gurobi_mip_gap'])} | "
                f"cert_gap={_fmt_metric(row['drl_gap_to_gurobi_bound_pct'], 2)}% | "
                f"time(drl/gb)={row['drl_time_s']:.3f}/{row['gurobi_time_s']:.3f}s"
            )

    gap_key = "gap_alns_pct" if (args.gate_method == "alns" and args.run_alns) else "gap_drl_pct"
    if args.skip_gurobi and not reference_rows:
        valid_rows = []
        diagnostic_rows = []
        avg_gap_drl = None
        avg_gap_alns = None
        diagnostic_avg_gap_drl = None
        diagnostic_avg_gap_alns = None
        gate_avg_gap = None
        drl_win = None
        alns_win = None
        avg_obj_env_delta = None
        gap_drl_values = []
        gap_alns_values = []
        gurobi_alt_gain_values = []
        optimal_count = None
        time_limit_count = None
        gate_pass = None
        gate_reason = "SKIPPED: Gurobi comparison disabled."
    else:
        if args.allow_nonoptimal:
            valid_rows = [
                r
                for r in rows
                if torch.isfinite(torch.tensor(r[gap_key]))
                and r["gurobi_feasible_env"]
                and r["gurobi_bound_finite"]
                and _is_finite(r["gurobi_mip_gap"])
            ]
        else:
            valid_rows = [
                r
                for r in rows
                if torch.isfinite(torch.tensor(r[gap_key]))
                and r["gurobi_feasible_env"]
                and r["gurobi_status"] == 2
            ]
        diagnostic_rows = [
            r
            for r in rows
            if torch.isfinite(torch.tensor(r[gap_key])) and r["gurobi_feasible_env"]
        ]
        avg_gap_drl = (
            sum(r["gap_drl_pct"] for r in valid_rows) / len(valid_rows)
            if valid_rows
            else None
        )
        avg_gap_alns = (
            sum(r["gap_alns_pct"] for r in valid_rows if torch.isfinite(torch.tensor(r["gap_alns_pct"])))
            / sum(1 for r in valid_rows if torch.isfinite(torch.tensor(r["gap_alns_pct"])))
            if sum(1 for r in valid_rows if torch.isfinite(torch.tensor(r["gap_alns_pct"]))) > 0
            else None
        ) if args.run_alns else None
        diagnostic_avg_gap_drl = sum(r["gap_drl_pct"] for r in diagnostic_rows) / max(len(diagnostic_rows), 1)
        diagnostic_avg_gap_alns = (
            sum(r["gap_alns_pct"] for r in diagnostic_rows if torch.isfinite(torch.tensor(r["gap_alns_pct"])))
            / max(sum(1 for r in diagnostic_rows if torch.isfinite(torch.tensor(r["gap_alns_pct"]))), 1)
        ) if args.run_alns else None
        gate_avg_gap = avg_gap_alns if (gap_key == "gap_alns_pct") else avg_gap_drl
        drl_win = sum(1 for r in valid_rows if r["gap_drl_pct"] <= 0.0)
        alns_win = sum(1 for r in valid_rows if r["gap_alns_pct"] <= 0.0) if args.run_alns else None
        avg_obj_env_delta = (
            sum(r["gurobi_obj_env_delta_pct"] for r in valid_rows) / len(valid_rows)
            if valid_rows
            else None
        )
        gap_drl_values = [r["gap_drl_pct"] for r in diagnostic_rows]
        gap_alns_values = [
            r["gap_alns_pct"]
            for r in diagnostic_rows
            if torch.isfinite(torch.tensor(r["gap_alns_pct"]))
        ] if args.run_alns else []
        gurobi_alt_gain_values = []
        if gurobi_solver_alt is not None:
            for r in rows:
                c_base = r["gurobi_cost_env"]
                c_alt = r["gurobi_alt_cost_env"]
                if c_alt is not None and c_base not in [None, float("inf")] and c_alt not in [None, float("inf")] and c_base > 0:
                    gurobi_alt_gain_values.append((c_base - c_alt) / c_base * 100.0)
        optimal_count = sum(1 for r in rows if r["gurobi_status"] == 2)
        time_limit_count = sum(1 for r in rows if r["gurobi_status"] == 9)
        finite_bound_count = sum(1 for r in rows if r.get("gurobi_bound_finite"))
        incumbent_count = sum(
            1
            for r in rows
            if r.get("gurobi_feasible_env")
            and r.get("gurobi_cost_env") is not None
            and torch.isfinite(torch.tensor(r.get("gurobi_cost_env")))
        )
        all_optimal = optimal_count == len(rows)
        require_optimal = args.gate_policy == "strict_optimal"
        enough_compared = len(valid_rows) >= args.gate_min_compared
        gap_within_threshold = gate_avg_gap <= args.gate_gap_threshold if gate_avg_gap is not None else False
        gate_pass = (all_optimal or not require_optimal) and enough_compared and gap_within_threshold
        if require_optimal and not all_optimal:
            gate_reason = "FAILED: not all instances reached Gurobi OPTIMAL status."
        elif incumbent_count == 0:
            gate_reason = (
                "FAILED: Gurobi returned no feasible incumbent within the budget "
                "or terminated before producing one."
            )
        elif args.allow_nonoptimal and finite_bound_count < args.gate_min_compared:
            gate_reason = (
                "FAILED: Gurobi returned feasible incumbent(s) but no finite objective bound/MIP gap; "
                "reported gap is incumbent-only, not certified."
            )
        elif not enough_compared:
            gate_reason = (
                f"FAILED: compared_count({len(valid_rows)}) < gate_min_compared({args.gate_min_compared})."
            )
        elif not gap_within_threshold:
            gate_reason = (
                f"FAILED: {gap_key} mean({gate_avg_gap:.4f}) > gate_gap_threshold({args.gate_gap_threshold:.4f})."
            )
        else:
            if require_optimal:
                gate_reason = (
                    "PASSED: strict optimality gate satisfied for the compared rows."
                )
            else:
                gate_reason = (
                    "PASSED: budget-feasible diagnostic gate satisfied using "
                    "Gurobi incumbent/bound data; this is not an optimality certificate."
                )
    if (not args.skip_gurobi) and (not args.allow_nonoptimal) and len(valid_rows) < len(rows):
        dropped = len(rows) - len(valid_rows)
        print(
            f"[warning] strict mode dropped {dropped} instance(s) "
            "because Gurobi was not OPTIMAL or env-feasible."
        )

    has_gurobi_metrics = (not args.skip_gurobi) or bool(reference_rows)
    exact_optimality_claim_allowed = bool(
        has_gurobi_metrics
        and optimal_count is not None
        and optimal_count == len(rows)
        and len(rows) > 0
    )
    gate_interpretation = None
    if gate_pass is not None:
        if exact_optimality_claim_allowed and args.gate_policy == "strict_optimal":
            gate_interpretation = "strict_optimality_certificate"
        elif args.allow_nonoptimal or args.gate_policy == "budget_feasible":
            gate_interpretation = "budget_feasible_diagnostic_not_optimality_certificate"
        else:
            gate_interpretation = "not_certified_for_optimality_claims"
    summary = {
        "config": {
            "instances": args.instances,
            "n_nodes": args.n_nodes,
            "problem_type": args.problem_type,
            "scenario": args.scenario,
            "tightness": args.tightness,
            "variant": args.variant,
            "checkpoint": args.checkpoint,
            "pref_mode": args.pref_mode,
            "pref": [float(pref_batch[0, 0].item()), float(pref_batch[0, 1].item())],
            "pref_random_low": args.pref_random_low,
            "pref_random_high": args.pref_random_high,
            "n_pomo": args.n_pomo,
            "decode_type": args.decode_type,
            "sample_rollouts": args.sample_rollouts,
            "sample_temperature": args.sample_temperature,
            "multi_start_rollouts": args.multi_start_rollouts,
            "augment_8_fold": args.augment_8_fold,
            "drl_repair": args.drl_repair,
            "skip_gurobi": args.skip_gurobi,
            "reference_json": args.reference_json,
            "gurobi_warm_start_json": args.gurobi_warm_start_json,
            "gurobi_warm_start_methods": warm_start_methods,
            "gurobi_time_limit": args.gurobi_time_limit,
            "gurobi_time_limit_alt": args.gurobi_time_limit_alt,
            "gurobi_mip_gap": args.gurobi_mip_gap,
            "gurobi_max_returns": args.gurobi_max_returns,
            "gurobi_proof_mode": args.gurobi_proof_mode,
            "gurobi_threads": args.gurobi_threads,
            "gurobi_log_file": args.gurobi_log_file,
            "gurobi_nodefile_start_gb": args.gurobi_nodefile_start_gb,
            "gurobi_nodefile_dir": args.gurobi_nodefile_dir,
            "drl_warm_start": not args.disable_drl_warm_start,
            "run_alns": args.run_alns,
            "alns_time_limit": args.alns_time_limit,
            "alns_max_iter": args.alns_max_iter,
            "alns_destroy_min": args.alns_destroy_min,
            "alns_destroy_max": args.alns_destroy_max,
            "alns_max_insert_positions": args.alns_max_insert_positions,
            "alns_swap_neighbors": args.alns_swap_neighbors,
            "alns_restart_rounds": args.alns_restart_rounds,
            "allow_nonoptimal": args.allow_nonoptimal,
            "gate_method": args.gate_method,
            "gate_policy": args.gate_policy,
            "gate_gap_threshold": args.gate_gap_threshold,
            "gate_min_compared": args.gate_min_compared,
            "realtime_threshold_s": args.realtime_threshold_s,
        },
        "summary": {
            "avg_gap_drl_pct": avg_gap_drl,
            "avg_gap_alns_pct": avg_gap_alns,
            "median_gap_drl_pct": median(gap_drl_values) if gap_drl_values else None,
            "std_gap_drl_pct": _safe_std(gap_drl_values),
            "p90_gap_drl_pct": _safe_percentile(gap_drl_values, 0.9),
            "median_gap_alns_pct": median(gap_alns_values) if gap_alns_values else None,
            "std_gap_alns_pct": _safe_std(gap_alns_values),
            "p90_gap_alns_pct": _safe_percentile(gap_alns_values, 0.9),
            "diagnostic_avg_gap_drl_pct_all_feasible": diagnostic_avg_gap_drl,
            "diagnostic_avg_gap_alns_pct_all_feasible": diagnostic_avg_gap_alns,
            "diagnostic_count_all_feasible": len(diagnostic_rows),
            "drl_win_count": drl_win,
            "alns_win_count": alns_win,
            "compared_count": len(valid_rows),
            "optimal_count": optimal_count,
            "time_limit_count": time_limit_count,
            "finite_bound_count": None if not has_gurobi_metrics else sum(1 for r in rows if r.get("gurobi_bound_finite")),
            "interrupted_count": None if not has_gurobi_metrics else sum(1 for r in rows if r["gurobi_status"] == 11),
            "drl_feasible_rate": sum(float(r["drl_feasible"]) for r in rows) / len(rows),
            "alns_feasible_rate": (
                sum(float(r["alns_feasible"]) for r in rows if r["alns_feasible"] is not None)
                / max(sum(1 for r in rows if r["alns_feasible"] is not None), 1)
            ) if args.run_alns else None,
            "gurobi_feasible_rate_env": (
                None if not has_gurobi_metrics else sum(float(r["gurobi_feasible_env"]) for r in rows) / len(rows)
            ),
            "avg_drl_time_s": sum(r["drl_time_s"] for r in rows) / len(rows),
            "drl_realtime_rate": sum(1 for r in rows if r["drl_time_s"] <= args.realtime_threshold_s) / len(rows),
            "avg_total_time_s": sum(r["total_time_s"] for r in rows) / len(rows),
            "total_realtime_rate": sum(1 for r in rows if r["total_time_s"] <= args.realtime_threshold_s) / len(rows),
            "avg_alns_time_s": (
                sum(r["alns_time_s"] for r in rows if r["alns_time_s"] is not None)
                / max(sum(1 for r in rows if r["alns_time_s"] is not None), 1)
            ) if args.run_alns else None,
            "alns_realtime_rate": (
                sum(1 for r in rows if r["alns_time_s"] is not None and r["alns_time_s"] <= args.realtime_threshold_s)
                / max(sum(1 for r in rows if r["alns_time_s"] is not None), 1)
            ) if args.run_alns else None,
            "avg_gurobi_time_s": None if not has_gurobi_metrics else sum(r["gurobi_time_s"] for r in rows) / len(rows),
            "gurobi_realtime_rate": (
                None
                if not has_gurobi_metrics
                else sum(1 for r in rows if r["gurobi_time_s"] <= args.realtime_threshold_s) / len(rows)
            ),
            "avg_gurobi_alt_gain_pct": _safe_mean(gurobi_alt_gain_values),
            "avg_gurobi_obj_env_delta_pct": avg_obj_env_delta,
            "gate_metric": gap_key,
            "gate_metric_mean": gate_avg_gap,
            "gate_pass": gate_pass,
            "gate_reason": gate_reason,
            "gate_interpretation": gate_interpretation,
            "exact_optimality_claim_allowed": exact_optimality_claim_allowed,
            "avg_alns_improve_pct_vs_drl": _finite_or_none(
                sum(r["alns_improve_pct_vs_drl"] for r in rows if r["alns_improve_pct_vs_drl"] is not None)
                / max(sum(1 for r in rows if r["alns_improve_pct_vs_drl"] is not None), 1)
            ),
        },
        "rows": rows,
    }

    if args.output_json:
        output_dir = os.path.dirname(os.path.abspath(args.output_json))
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"Saved result json: {args.output_json}")

    print(json.dumps(summary["summary"], ensure_ascii=False, indent=2))
    if gate_pass:
        print(f"[decision] PASS -> {gate_reason}")
        if not exact_optimality_claim_allowed:
            print(
                "[decision] NOTE -> Treat Gurobi rows as incumbent/bound "
                "diagnostics, not exact-optimal references."
            )
    elif gate_pass is None:
        print(f"[decision] INFO -> {gate_reason}")
    else:
        print(f"[decision] HOLD -> {gate_reason}")


if __name__ == "__main__":
    main()
