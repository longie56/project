"""Frozen v3 evaluation; one instance, preference and method per JSONL row."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from env.revision_v3_gu_physics import SCENARIOS
from revision_v3.baselines import alns, greedy, nsga2, solve_proxy
from revision_v3.common_eval import replay_actions
from revision_v3.policy import AnalyticLocalSpeed, V3Policy, VARIANTS
from revision_v3.selection import select_best_complete
from revision_v3.spec import ROOT, SPEC_PATH, TASK, validate_formal_experiment
from revision_v3.train import SOURCE_FILES
from revision_v3.speed_intervention import assert_fixed_flights
from revision_v3.evaluation_panels import validate_test_panel


FIXED_SPEED_METHODS = {"neural_fixed_slow": 0, "neural_fixed_fast": 1}
METHODS = ("neural", "neural_analytic_speed", "neural_alns", "neural_fixed_slow", "neural_fixed_fast", "greedy",
           "alns", "nsga2", "alns_waitaware", "nsga2_waitaware",
           "ortools", "pyvrp", "gurobi")
NEURAL_METHODS = ("neural", "neural_analytic_speed", "neural_alns", *FIXED_SPEED_METHODS)
PROXY_METHODS = ("ortools", "pyvrp", "gurobi")


def quality(report):
    metrics = report["metrics"]
    return (not report["feasible"], int(metrics["unserved"]), float(metrics["cost"]))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def neural(model, nodes, depot, pref, *, pomo, device, objective_variant,
           reserve_fraction, energy_scenario, fixed_speed_mode=None):
    started = time.perf_counter()
    with torch.inference_mode():
        output = model.rollout(nodes[None].to(device), depot[None].to(device),
                               pref[None].to(device), pomo=pomo, greedy=True,
                               greedy_strategy=TASK["formal_greedy_strategy"],
                               return_actions=True,
                               objective_variant=objective_variant,
                               reserve_fraction=reserve_fraction,
                               energy_scenario=energy_scenario,
                               fixed_speed_mode=fixed_speed_mode)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    decoded_s = time.perf_counter()-started
    best_idx = int(select_best_complete(output)[0])
    chosen = output["actions"][0, best_idx]
    valid = output["action_valid"][0, best_idx]
    actions = chosen[valid].detach().cpu().tolist()
    if fixed_speed_mode is not None:
        assert_fixed_flights(actions, nodes.size(0), fixed_speed_mode)
    report = replay_actions(nodes, depot, pref, actions,
                            provenance="v3_neural_greedy_pomo",
                            objective_variant=objective_variant,
                            reserve_fraction=reserve_fraction,
                            energy_scenario=energy_scenario)
    raw_cost = float(output["cost"][0, best_idx])
    replay_cost = float(report["metrics"]["cost"])
    if abs(raw_cost-replay_cost) > max(0.02, abs(raw_cost)*1e-3):
        raise RuntimeError("Neural environment and common replay disagree on cost")
    report["method"] = "neural"
    report["decode_elapsed_s"] = decoded_s
    report["total_elapsed_s"] = time.perf_counter()-started
    report["native_pomo_selection_cost"] = float(output["selection_cost"][0, best_idx])
    report["exact_physical_model"] = True
    return report


def run_method(method, nodes, depot, pref, *, time_limit_s, seed,
               model=None, pomo=None, device=None, objective_variant="C0",
               reserve_fraction=0.0, energy_scenario=None):
    started = time.perf_counter()
    if method in NEURAL_METHODS:
        if model is None:
            raise ValueError("Neural evaluation requires a v3 checkpoint")
        if (method == "neural_analytic_speed"
                and not isinstance(model.speed_head, AnalyticLocalSpeed)):
            raise ValueError(
                "Analytic-speed evaluation requires the analytic speed head; "
                "a result label alone cannot select the speed rule")
        neural_result = neural(model, nodes, depot, pref, pomo=pomo, device=device,
                               objective_variant=objective_variant,
                               reserve_fraction=reserve_fraction,
                               energy_scenario=energy_scenario,
                               fixed_speed_mode=FIXED_SPEED_METHODS.get(method))
        if method in ("neural", "neural_analytic_speed", *FIXED_SPEED_METHODS):
            result = neural_result
            result["method"] = method
            result["speed_rule"] = ("analytic_local" if method == "neural_analytic_speed"
                                    else method if method in FIXED_SPEED_METHODS else "learned")
            if method in FIXED_SPEED_METHODS:
                result["speed_intervention"] = {
                    "fixed_mode": FIXED_SPEED_METHODS[method],
                    "all_delivery_and_return_flights_checked": True,
                    "opposite_speed_fallback": False,
                    "interpretation": "same_checkpoint_action_restriction_with_distribution_shift",
                    "loading_and_waiting_masks": "original_dual_speed_environment",
                    "incomplete_and_over_budget_runs_remain_in_denominator": True,
                }
        else:
            order = []
            for event in neural_result["events"]:
                if event["kind"] == "deliver" and event["order"] not in order:
                    order.append(event["order"])
            order.extend(i for i in range(nodes.size(0)) if i not in order)
            remaining_s = time_limit_s-(time.perf_counter()-started)
            if remaining_s > 0:
                result = alns(nodes, depot, pref, time_limit_s=remaining_s,
                              seed=seed, initial_order=order,
                              objective_variant=objective_variant,
                              reserve_fraction=reserve_fraction,
                              energy_scenario=energy_scenario,
                              search_loaded_wait=True)
            else:
                result = neural_result
                result["alns_skipped_reason"] = "neural_decode_exhausted_budget"
            if quality(result) > quality(neural_result):
                result = neural_result
            result["method"] = "neural_alns"
            result["neural_decode_elapsed_s"] = neural_result["total_elapsed_s"]
    elif method == "greedy":
        result = greedy(nodes, depot, pref, objective_variant=objective_variant,
                        time_limit_s=time_limit_s,
                        reserve_fraction=reserve_fraction,
                        energy_scenario=energy_scenario)
        result["method"] = "greedy"
    elif method in ("alns", "alns_waitaware"):
        result = alns(nodes, depot, pref, time_limit_s=time_limit_s, seed=seed,
                      objective_variant=objective_variant,
                      reserve_fraction=reserve_fraction,
                      energy_scenario=energy_scenario,
                      search_loaded_wait=(method == "alns_waitaware"))
        result["method"] = method
    elif method in ("nsga2", "nsga2_waitaware"):
        result = nsga2(nodes, depot, pref, time_limit_s=time_limit_s, seed=seed,
                       objective_variant=objective_variant,
                       reserve_fraction=reserve_fraction,
                       energy_scenario=energy_scenario,
                       search_loaded_wait=(method == "nsga2_waitaware"))
        result["method"] = method
    else:
        result = solve_proxy(method, nodes, depot, pref,
                             time_limit_s=time_limit_s, seed=seed,
                             objective_variant=objective_variant,
                             reserve_fraction=reserve_fraction,
                             energy_scenario=energy_scenario)
    result.setdefault("physics_id", (energy_scenario or SCENARIOS["gu_nominal"]).id)
    result.setdefault("failure_reason", None)
    if not isinstance(result.get("metrics"), dict):
        result["metrics"] = {
            "unserved": int(nodes.size(0)),
            "flight_time_s": None,
            "total_energy_j": None,
            "tardiness_s": None,
            "cost": None,
            "depot_returns": None,
            "min_return_battery_j": None,
            "delivery_leg_count": None,
            "return_leg_count": None,
            "fast_delivery_fraction": None,
            "fast_return_fraction": None,
            "fast_all_flight_fraction": None,
        }
    result["total_elapsed_s"] = time.perf_counter()-started
    result["time_limit_s"] = time_limit_s
    result["within_total_budget"] = result["total_elapsed_s"] <= time_limit_s
    result["comparison_eligible_under_budget"] = bool(
        result["within_total_budget"] and result.get("feasible", False)
    )
    if method in PROXY_METHODS:
        result["comparison_scope"] = "route_order_proxy_with_v3_physical_repair"
        result["native_objective_scope"] = "route_order_surrogate_only"
        result["native_bound_gap_scope"] = "route_order_surrogate_only"
        result["eligible_for_primary_same_task_ranking"] = False
    elif method in FIXED_SPEED_METHODS:
        result["comparison_scope"] = "fixed_speed_inference_intervention"
        result["native_objective_scope"] = "v3_physical_replay"
        result["native_bound_gap_scope"] = None
        result["eligible_for_primary_same_task_ranking"] = False
    else:
        result["comparison_scope"] = "same_v3_physical_problem"
        result["native_objective_scope"] = "v3_physical_replay"
        result["native_bound_gap_scope"] = None
        result["eligible_for_primary_same_task_ranking"] = bool(
            result["comparison_eligible_under_budget"]
        )
    return result


def main():
    parser = argparse.ArgumentParser(description="TRE Gu-v3 common evaluation")
    sensitivity = json.loads((ROOT / "revision_v3/sensitivity_spec.json").read_text(
        encoding="utf-8"))
    baseline_search_seed_base = int(sensitivity["baseline_search_seed_base"])
    parser.add_argument("--instances", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--n-nodes", type=int, choices=(20, 50, 100), required=True)
    parser.add_argument("--kind", choices=("R", "C", "RC"), required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--objective-variant", choices=("C0", "C1"), default="C0")
    parser.add_argument("--energy-scenario", choices=SCENARIOS, default="gu_nominal")
    parser.add_argument("--reserve-fraction", type=float, choices=(0.0, 0.05, 0.10), default=0.0)
    parser.add_argument("--time-window-scale", type=float, choices=(0.75, 1.0, 1.25), default=1.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--allow-route-only-diagnostic", action="store_true",
                        help="Explicitly evaluate an uncalibrated route checkpoint on test")
    parser.add_argument("--budget-tier", choices=("online", "long"), default="long")
    parser.add_argument("--time-limit-s", type=float,
                        help="Explicit budget override; recorded in the output header")
    parser.add_argument("--preference-set", choices=("baseline", "full"),
                        default="baseline",
                        help="Three paired comparison weights or all eleven response weights")
    parser.add_argument("--search-seed", type=int,
                        default=baseline_search_seed_base)
    parser.add_argument("--pomo", type=int)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int)
    parser.add_argument("--evaluation-panel", default="main",
                        choices=("main", "long_subset", "learned_analytic", "response_curve", "fixed_speed", "sensitivity"))
    parser.add_argument("--resume", action="store_true",
                        help="Continue only a hash- and prefix-verified JSONL output")
    args = parser.parse_args()
    comparison_sections = sensitivity.get("comparison_sections", {})
    primary_methods = tuple(comparison_sections.get("primary_same_physical_task", ()))
    proxy_methods = tuple(comparison_sections.get("route_order_proxy_diagnostic", ()))
    intervention_methods = tuple(comparison_sections.get("speed_intervention_diagnostic", ()))
    if (set(primary_methods) & set(proxy_methods)
            or set(proxy_methods) != set(PROXY_METHODS)
            or not set(primary_methods).issubset(METHODS)
            or set(intervention_methods) != set(FIXED_SPEED_METHODS)
            or set(intervention_methods) & (set(primary_methods) | set(proxy_methods))):
        raise ValueError("Sensitivity specification has invalid formal comparison sections")
    if args.split == "test" and args.method not in set(primary_methods) | set(proxy_methods) | set(intervention_methods):
        parser.error("Method is outside the frozen formal comparison sections")
    energy_scenario = SCENARIOS[args.energy_scenario]
    long_budget_key = ("solver_proxy_long_budget_seconds_by_n"
                       if args.method in PROXY_METHODS
                       else "same_task_long_budget_seconds_by_n")
    frozen_time_limit_s = (float(TASK["online_end_to_end_budget_seconds"])
                           if args.budget_tier == "online" else
                           float(TASK[long_budget_key][str(args.n_nodes)]))
    if args.time_limit_s is None:
        args.time_limit_s = frozen_time_limit_s
    preference_grid = (TASK["baseline_preference_values"]
                       if args.preference_set == "baseline" else
                       TASK["preference_test_grid"])
    if args.time_limit_s <= 0:
        parser.error("Time limit must be positive")
    if args.method in NEURAL_METHODS and not args.checkpoint:
        parser.error("Neural evaluation requires --checkpoint")
    if args.split == "test" and TASK["status"] != "frozen_formal":
        parser.error("Blind test evaluation requires the frozen formal task")
    if args.split == "test" and sensitivity.get("status") != "frozen_formal":
        parser.error("Blind test evaluation requires the frozen sensitivity matrix")
    if args.split == "test" and args.allow_route_only_diagnostic:
        parser.error("Route-only diagnostic override is restricted to validation")
    if args.split == "test" and abs(args.time_limit_s-frozen_time_limit_s) > 1e-9:
        parser.error("Blind-test time limit differs from the frozen budget tier")
    panel_count = None
    if args.split == "test":
        try:
            panel_count = validate_test_panel(args, sensitivity)
        except ValueError as exc:
            parser.error(str(exc))
    if args.output.resolve().is_relative_to(args.instances.resolve()):
        parser.error("Evaluation output must stay outside the frozen dataset directory")
    manifest_path = args.instances / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (TASK["status"] == "frozen_formal"
            and manifest.get("status") != "frozen_formal_not_yet_tested"):
        raise ValueError("Formal evaluation requires the untouched frozen dataset")
    if digest(SPEC_PATH) != manifest["task_spec_sha256"]:
        raise ValueError("Frozen data/task specification mismatch")
    if manifest.get("formal_instance_seed_bases") != TASK.get("formal_instance_seed_bases"):
        raise ValueError("Frozen validation/test seed namespace mismatch")
    if digest(ROOT / TASK["physics_spec"]) != manifest["physics_spec_sha256"]:
        raise ValueError("Frozen data/physics specification mismatch")
    sensitivity_path = ROOT / "revision_v3/sensitivity_spec.json"
    if (TASK["status"] == "frozen_formal"
            and manifest.get("sensitivity_spec_sha256") != digest(sensitivity_path)):
        raise ValueError("Frozen data/comparison specification mismatch")
    if digest(ROOT / "revision_v3/instances.py") != manifest["generator_sha256"]:
        raise ValueError("Frozen data/generator mismatch")
    records = [row for row in manifest["sets"]
               if row["split"] == args.split and row["n_nodes"] == args.n_nodes
               and row["kind"] == args.kind]
    if len(records) != 1:
        raise ValueError("Requested frozen subset not uniquely present")
    row = records[0]
    expected_count = (TASK["validation_per_n_and_type"] if args.split == "val"
                      else TASK["test_per_n_and_type"][str(args.n_nodes)])
    expected_seed = (TASK["formal_instance_seed_bases"][args.split]
                     + args.n_nodes*10 + ("R", "C", "RC").index(args.kind) + 1)
    if (TASK["status"] == "frozen_formal"
            and (row.get("count") != expected_count or row.get("seed") != expected_seed
                 or row.get("file")
                 != f"{args.split}_n{args.n_nodes}_{args.kind}.pt")):
        raise ValueError("Frozen subset metadata differs from the formal task")
    if digest(args.instances / row["file"]) != row["sha256"]:
        raise ValueError("Frozen instance file hash mismatch")
    data = torch.load(args.instances / row["file"], map_location="cpu", weights_only=True)
    end = row["count"] if args.count is None else min(row["count"], args.start+args.count)
    if panel_count is not None:
        if panel_count > row["count"]:
            raise ValueError("Frozen panel requires more instances than its manifest subset")
        end = panel_count
    if args.start < 0 or end <= args.start:
        parser.error("Empty or invalid instance slice")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_hash = None
    if args.method in NEURAL_METHODS:
        checkpoint_hash = digest(args.checkpoint)
        state = torch.load(args.checkpoint, map_location=device, weights_only=True)
        if state["task_spec_sha256"] != digest(SPEC_PATH):
            raise ValueError("Checkpoint trained on a different v3 task specification")
        if args.split == "test":
            lock_path = Path(state["config"].get("hardware_profile") or "")
            if (not lock_path.is_file()
                    or state.get("hardware_profile_sha256") != digest(lock_path)):
                raise ValueError("Test checkpoint differs from its formal scale hardware lock")
        current_source = {name: digest(ROOT / name) for name in SOURCE_FILES}
        if state.get("source_sha256") != current_source:
            raise ValueError("Checkpoint source differs from current v3 physics or policy")
        variant = state["config"]["variant"]
        if args.split == "test":
            validate_formal_experiment(
                variant=variant, n_nodes=args.n_nodes,
                pref_mode=state["config"].get("pref_mode"),
                fixed_alpha=state["config"].get("fixed_alpha"),
                objective_variant=state["config"].get("objective_variant", "C0"),
                energy_scenario=state["config"].get("energy_scenario", "gu_nominal"),
            )
        if (args.split == "test" and VARIANTS[variant].get("use_speed_aux", False)
                and VARIANTS[variant].get("use_decoupled_route_speed", False)
                and not args.allow_route_only_diagnostic):
            selection = state.get("speed_stage_selection")
            if not isinstance(selection, dict) or selection.get("rule") != (
                    "validation_pareto_overall_high_time_feasibility_v1"):
                raise ValueError("Test evaluation requires a frozen validation-selected speed checkpoint")
            stage = selection.get("selected_stage")
            if stage not in ("route_only", "calibrated"):
                raise ValueError("Selected speed stage is invalid")
            if (stage == "calibrated") != (state.get("calibration") is not None):
                raise ValueError("Selected stage and checkpoint calibration metadata disagree")
            if selection.get("dataset_manifest_sha256") != digest(manifest_path):
                raise ValueError("Selected speed stage used a different frozen dataset")
            if selection.get("selector_source_sha256") != digest(
                    ROOT / "scripts/select_v3_speed_stage.py"):
                raise ValueError("Selected speed stage used a different selector implementation")
        if state["config"]["n_nodes"] != args.n_nodes:
            raise ValueError("Checkpoint scale differs from frozen evaluation scale")
        if state["config"].get("objective_variant", "C0") != args.objective_variant:
            raise ValueError("Checkpoint objective variant differs from evaluation")
        trained_energy_scenario = state["config"].get("energy_scenario", "gu_nominal")
        transfer_evaluation = trained_energy_scenario != args.energy_scenario
        model = V3Policy(**VARIANTS[variant]).to(device)
        model.load_state_dict(state["model"])
        if args.method == "neural_analytic_speed":
            if not model.use_factorized_speed:
                raise ValueError("Analytic speed control requires a factorized policy")
            model.speed_head = AnalyticLocalSpeed().to(device)
        model.eval()
    frozen_online_pomo = TASK.get("online_pomo_by_n", {}).get(str(args.n_nodes))
    expected_pomo = (frozen_online_pomo
                     if args.budget_tier == "online" and frozen_online_pomo is not None
                     else args.n_nodes)
    pomo = expected_pomo if args.pomo is None else args.pomo
    if pomo < 1 or pomo > args.n_nodes:
        parser.error("POMO starts outside 1..N")
    if args.split == "test" and args.method in NEURAL_METHODS and pomo != expected_pomo:
        parser.error("Blind-test POMO starts differ from the frozen budget tier")
    header = {"record_type": "header", "task_spec_sha256": digest(SPEC_PATH),
              "physics_spec_sha256": manifest["physics_spec_sha256"],
              "sensitivity_spec_sha256": digest(
                  ROOT / "revision_v3/sensitivity_spec.json"),
              "instance_manifest_sha256": digest(manifest_path),
              "subset_file_sha256": row["sha256"],
              "evaluation_source_sha256": {
                  name: digest(ROOT / name) for name in (
                      "revision_v3/evaluate.py", "revision_v3/baselines.py",
                      "revision_v3/common_eval.py", "revision_v3/environment.py",
                      "revision_v3/speed_intervention.py", "revision_v3/evaluation_panels.py",
                      "env/revision_v3_gu_physics.py")},
              "checkpoint_sha256": checkpoint_hash,
              "checkpoint_hardware_profile_sha256": (
                  state.get("hardware_profile_sha256")
                  if args.method in NEURAL_METHODS else None),
              "checkpoint_training_seed": (
                  state["config"].get("seed") if args.method in NEURAL_METHODS else None),
              "checkpoint_variant": (
                  state["config"].get("variant") if args.method in NEURAL_METHODS else None),
              "checkpoint_epoch": (
                  state.get("epoch") if args.method in NEURAL_METHODS else None),
              "checkpoint_calibration": (
                  state.get("calibration") if args.method in NEURAL_METHODS else None),
              "checkpoint_speed_stage_selection": (
                  state.get("speed_stage_selection") if args.method in NEURAL_METHODS else None),
              "route_only_diagnostic_override": args.allow_route_only_diagnostic,
              "method": args.method, "split": args.split, "n_nodes": args.n_nodes,
               "comparison_section": (
                   "route_order_proxy_diagnostic" if args.method in proxy_methods
                   else "speed_intervention_diagnostic" if args.method in intervention_methods
                   else "primary_same_physical_task" if args.method in primary_methods
                   else "development_only"),
              "speed_rule": ("analytic_local" if args.method == "neural_analytic_speed"
                             else args.method if args.method in FIXED_SPEED_METHODS
                             else "learned" if args.method in ("neural", "neural_alns")
                             else None),
              "greedy_strategy": (TASK["formal_greedy_strategy"]
                                  if args.method in NEURAL_METHODS else None),
              "objective_variant": args.objective_variant,
              "energy_scenario": args.energy_scenario,
              "reserve_fraction": args.reserve_fraction,
              "time_window_scale": args.time_window_scale,
              "checkpoint_energy_scenario": (
                  trained_energy_scenario if args.method in NEURAL_METHODS else None),
              "checkpoint_preference_mode": (
                  state["config"].get("pref_mode") if args.method in NEURAL_METHODS else None),
              "checkpoint_fixed_alpha": (
                  state["config"].get("fixed_alpha") if args.method in NEURAL_METHODS else None),
              "evaluation_type": (
                  "frozen_policy_transfer" if args.method in NEURAL_METHODS
                  and transfer_evaluation else "matched_scenario_or_non_neural"),
               "kind": args.kind, "preference_set": args.preference_set,
               "evaluation_panel": args.evaluation_panel,
               "instance_selection_rule": "manifest_order_prefix_before_outcomes",
               "preference_grid": preference_grid, "budget_tier": args.budget_tier,
               "search_seed": args.search_seed, "time_limit_s": args.time_limit_s,
               "pomo": pomo if args.method in NEURAL_METHODS else None,
               "slice_start": args.start, "slice_end": end,
               "subset_instance_count": row["count"],
               "planned_row_count": (end-args.start)*len(preference_grid)}
    expected_keys = [
        (f"{args.split}_n{args.n_nodes}_{args.kind}_{index:04d}", float(alpha))
        for index in range(args.start, end) for alpha in preference_grid
    ]
    completed_keys = []
    mode = "x"
    if args.output.exists():
        if not args.resume:
            raise FileExistsError("Evaluation output exists; pass --resume after inspection")
        with args.output.open(encoding="utf-8") as existing:
            existing_rows = [json.loads(line) for line in existing if line.strip()]
        if not existing_rows or existing_rows[0] != header:
            raise ValueError("Resume header differs from the current frozen protocol")
        if any(row.get("record_type") != "instance" for row in existing_rows[1:]):
            raise ValueError("Resume output contains a non-instance row after its header")
        completed_keys = [
            (row.get("instance_id"), float(row.get("alpha_time")))
            for row in existing_rows[1:]
        ]
        if completed_keys != expected_keys[:len(completed_keys)]:
            raise ValueError("Resume rows are not a unique ordered prefix of the frozen plan")
        mode = "a"
    elif args.resume:
        raise FileNotFoundError("Resume requested but evaluation output is absent")
    completed = set(completed_keys)
    with args.output.open(mode, encoding="utf-8") as handle:
        if mode == "x":
            handle.write(json.dumps(header) + "\n")
        for index in range(args.start, end):
            nodes, depot = data["nodes"][index].clone(), data["depot"][index]
            if args.time_window_scale != 1.0:
                nodes[:, 4] = torch.minimum(
                    nodes[:, 3] + args.time_window_scale*(nodes[:, 4]-nodes[:, 3]),
                    torch.full_like(nodes[:, 4], float(TASK["mission_horizon_seconds_by_n"][str(args.n_nodes)])),
                )
            for alpha in preference_grid:
                instance_id = f"{args.split}_n{args.n_nodes}_{args.kind}_{index:04d}"
                if (instance_id, float(alpha)) in completed:
                    continue
                pref = torch.tensor([alpha, 1-alpha], dtype=nodes.dtype)
                result = run_method(args.method, nodes, depot, pref,
                    time_limit_s=args.time_limit_s, seed=args.search_seed+index,
                    model=model, pomo=pomo, device=device,
                    objective_variant=args.objective_variant,
                    reserve_fraction=args.reserve_fraction,
                    energy_scenario=energy_scenario)
                record = {"record_type": "instance", "instance_id": instance_id,
                          "instance_index": index, "alpha_time": alpha,
                          "alpha_energy": 1-alpha, "result": result}
                handle.write(json.dumps(record, default=str) + "\n")
                handle.flush()
                print(json.dumps({"instance": record["instance_id"],
                                  "alpha": alpha, "method": args.method,
                                  "feasible": result.get("feasible"),
                                  "reason": result.get("failure_reason")}), flush=True)


if __name__ == "__main__":
    main()
