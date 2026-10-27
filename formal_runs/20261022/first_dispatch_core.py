"""Frozen first-dispatch POMO confirmation on new physical instances."""
from __future__ import annotations

import hashlib
import json
import math
import os
import socket
import statistics
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
N = int(os.environ.get("N_NODES", "0"))
if N not in (20, 50, 100):
    raise RuntimeError("N_NODES must be 20, 50 or 100")
SOURCE = Path("/root/sj-tmp/tre_revision_v3_compact_formal_20261018")
BACKUP = (Path("/root/sj-tmp/tre_first_backup_20261019/tre_revision_v3_compact_formal_20261018/formal_hardware_first5090")
          if N in (20, 50) else Path("/root/sj-tmp/tre_revision_v3_compact_formal_20261018/formal_runs_gpu0"))
P1 = "physical_pref_v3_wait40_slack5"
WEIGHTS = {
    20: "5aa5ffd169096d74ee87dd3a0522d1ca7a435ad9a3672237524e925c308c655a",
    50: "e3c5b49717e17a3e93f044d51bf7e6ad58006effe630132d83784a9c422a086f",
    100: "1ae349a5d843dd625f6bfdf60ecf8f3f06827bd03c253cfa1e4d8ae9778a0fa2",
}
SOURCES = {
    "revision_v3/policy.py": "eb6d583feacf4243b314d7a022b3aa985c20e00fe2b86c5a036deaa84ae7ef69",
    "revision_v3/instances.py": "086184a7229c4025e27a116509cf7570ea0685e3956428446b67bd183635a513",
    "revision_v3/environment.py": "446d379838b4a5600b76ae3590436d298daa2f53887eac120e74cb0429a310ed",
    "revision_v3/common_eval.py": "eac0bcd8b5f659393f908cba3c677c2ecc1e345e0d19dcd2295a9884175e08ff",
    "revision_v3/selection.py": "52927fdef5e7419363e8426359c90fc7451d031ec1cbd48c2b5fddf7326518bb",
    "revision_v3/task_spec.json": "4e1f1a8071e30eb8cd46ebd9a9be86fdbdd794406d976ddc8523a234b700d8a4",
    "env/revision_v3_gu_physics.py": "b214cf6390a64f0b33b8f342ec4e0639b80ba6e2972083fcb5e90c10832a4b79",
}
sys.path.insert(0, str(SOURCE))

import torch
from revision_v3.common_eval import replay_actions
from revision_v3.environment import RouteEnvironment
from revision_v3.instances import generate
from revision_v3.policy import V3Policy, VARIANTS
from revision_v3.selection import select_best_complete
import revision_v3.selection as selection
from revision_v3.spec import TASK


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_once(name: str, payload: dict) -> None:
    with (HERE / name).open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")


def model_path(arm: str) -> Path:
    return BACKUP / f"{arm}_n{N}_s11_random_C0_gu_nominal/speed_calibration/speed_stage_selection.pt"


def gpu0_anchor() -> tuple[str, int, float] | None:
    root = Path("/root/sj-tmp/neural_baselines_formal_20261022/runs")
    for folder in sorted(root.glob("symnco_n50_*"), key=lambda p: p.stat().st_mtime, reverse=True):
        path = folder / "updates.jsonl"
        if not path.exists() or (folder / "completion.json").exists():
            continue
        rows = [json.loads(s) for s in path.read_text().splitlines() if s.strip()]
        deltas = [b["elapsed_s"] - a["elapsed_s"] for a, b in zip(rows, rows[1:])
                  if b["update"] == a["update"] + 1 and b["update"] % 100 != 0]
        if len(deltas) >= 20:
            return str(path), rows[-1]["update"], statistics.median(deltas[-20:])
    return None


def check_gpu0(anchor: tuple[str, int, float] | None) -> None:
    if anchor is None:
        return
    path, last, baseline = anchor
    rows = [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]
    delta = [b["elapsed_s"] - a["elapsed_s"] for a, b in zip(rows, rows[1:])
             if b["update"] == a["update"] + 1 and b["update"] % 100 != 0
             and b["update"] > last]
    if (len(delta) >= 20 and statistics.median(delta[-20:-10]) > baseline * 1.05
            and statistics.median(delta[-10:]) > baseline * 1.05):
        raise RuntimeError("GPU0 sustained same-task update slowdown above 5%")


def admit() -> tuple[dict, tuple[str, int, float] | None]:
    if socket.gethostname() != "c02f1a677fe6" or os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
        raise RuntimeError("Wrong host/GPU1 pin")
    owners = subprocess.check_output(["nvidia-smi", "-i", "1", "--query-compute-apps=pid",
                                      "--format=csv,noheader"], text=True).strip()
    if owners:
        raise RuntimeError(f"GPU1 occupied: {owners}")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("GPU1 CUDA/BF16 unavailable")
    if os.statvfs(HERE).f_bavail * os.statvfs(HERE).f_frsize < 10 * 1024**3:
        raise RuntimeError("Disk below 10GiB")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    for path, digest in SOURCES.items():
        if sha(SOURCE / path) != digest:
            raise RuntimeError(f"Frozen source changed: {path}")
    manifests = {}
    for arm, expected in ((P1, WEIGHTS[N]),):
        path = model_path(arm)
        if sha(path) != expected:
            raise RuntimeError(f"Checkpoint changed: {arm}")
        manifest_path = path.parent.parent / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        args, plan = manifest["args"], manifest["plan"]
        if (manifest["status"] != "completed_speed_stage_selected"
                or manifest["selected_stage"] != "route_only"
                or manifest["selected_checkpoint_sha256"] != expected
                or args["variant"] != arm or args["seed"] != 11 or args["n_nodes"] != N
                or args["pref_mode"] != "random" or args["objective_variant"] != "C0"
                or args["energy_scenario"] != "gu_nominal" or plan["epochs"] != 100
                or plan["updates_per_epoch"] != {20: 50, 50: 40, 100: 30}[N]
                or plan["pomo"] != N
                or plan["effective_batch"] != 16
                or manifest["source_sha256"]["revision_v3/policy.py"]
                != SOURCES["revision_v3/policy.py"]):
            raise RuntimeError(f"Frozen model identity differs: {arm}")
        manifests[arm] = sha(manifest_path)
    anchor = gpu0_anchor()
    return {"protocol_sha256": sha(HERE / "PROTOCOL.md"), "script_sha256": sha(Path(__file__)),
            "source_sha256": SOURCES, "weights_sha256": {P1: WEIGHTS[N]},
            "manifest_sha256": manifests, "gpu": 1, "torch": torch.__version__,
            "gpu0_same_task_anchor": anchor, "blind_test_read": False}, anchor


def load_model(arm: str) -> V3Policy:
    state = torch.load(model_path(arm), map_location="cpu", weights_only=True)
    if (state["config"]["variant"] != arm or state["config"]["seed"] != 11
            or state["config"]["n_nodes"] != N
            or state["task_spec_sha256"] != SOURCES["revision_v3/task_spec.json"]
            or state["speed_stage_selection"]["selected_stage"] != "route_only"):
        raise RuntimeError(f"Checkpoint content mismatch: {arm}")
    model = V3Policy(**VARIANTS[arm]).cuda().eval()
    model.load_state_dict(state["model"], strict=True)
    return model


def replay_selected(out: dict, idx: int, nodes: torch.Tensor, depot: torch.Tensor,
                    pref: torch.Tensor) -> dict:
    actions = out["actions"][0, idx][out["action_valid"][0, idx]].detach().cpu().tolist()
    report = replay_actions(nodes[0].cpu(), depot[0].cpu(), pref[0].cpu(), actions,
                            provenance="first_dispatch_cross_scale_probe")
    raw_cost = float(out["cost"][0, idx])
    raw_late = float(out["telemetry"]["tardiness_s"].reshape(1, N)[0, idx])
    cost = float(report["metrics"]["cost"])
    late = float(report["metrics"]["tardiness_s"])
    if abs(raw_cost-cost) > max(.02, .001*abs(raw_cost)) or abs(raw_late-late) > .05:
        raise RuntimeError("Frozen rollout and Gu replay differ")
    events = report["events"]
    per_order = [max(0., max(float(e["arrival_s"]), float(nodes[0, e["order"], 3]))
                     - float(nodes[0, e["order"], 4]))
                 for e in events if e["kind"] == "deliver"]
    if len(per_order) != N or abs(sum(per_order)-late) > .05:
        raise RuntimeError("Per-order lateness/replay mismatch")
    return {"candidate_index": idx, "raw_c0": raw_cost, "replay_c0": cost,
            "raw_tardiness_s": raw_late, "tardiness_s": late,
            "late_order_fraction": sum(x > 0 for x in per_order)/N,
            "late_order_p95_s": sorted(per_order)[math.ceil(.95 * N) - 1],
            "flight_time_s": float(report["metrics"]["flight_time_s"]),
            "energy_j": float(report["metrics"]["total_energy_j"]),
            "waiting_s": float(report["metrics"]["wait_time_s"]),
            "hover_energy_j": float(report["metrics"]["hover_energy_j"]),
            "minimum_return_battery_j": float(report["metrics"]["min_return_battery_j"]),
            "unserved": int(report["metrics"]["unserved"]),
            "feasible": bool(report["feasible"]),
            "complete": bool(report["feasible"]) and int(report["metrics"]["unserved"]) == 0,
            "actions": actions,
            "actions_sha256": hashlib.sha256(json.dumps(actions).encode()).hexdigest()}


def initial_groups(nodes: torch.Tensor, depot: torch.Tensor,
                   pref: torch.Tensor) -> tuple[dict[int, list[int]], int]:
    env = RouteEnvironment(nodes, depot, pref, reserve_fraction=0,
                           objective_variant="C0")
    legal = env.get_mask()[0, 2*N:3*N]
    count = int(legal.sum())
    if count == 0:
        return {N: list(range(N))}, 0
    available = torch.where(legal)[0]
    starts = available[(torch.arange(N, device=nodes.device) * count // N)]
    groups = defaultdict(list)
    for lane, first_id in enumerate(starts.tolist()):
        groups[first_id].append(lane)
    return dict(groups), count


def evaluate(model: V3Policy, nodes: torch.Tensor, depot: torch.Tensor,
             alpha: float, check_original: bool) -> dict:
    pref = torch.tensor([[alpha, 1-alpha]], dtype=nodes.dtype, device="cuda:0")
    groups, legal_first = initial_groups(nodes, depot, pref)
    representatives = [lanes[0] for lanes in groups.values()]
    original_greedy_action = selection.greedy_action
    calls = 0
    forced = 0
    changed = 0
    branched_lanes = set()

    def first_dispatch_action(logits, legal, n, strategy="joint"):
        nonlocal calls, forced, changed
        action = original_greedy_action(logits, legal, n, strategy)
        action = action.clone()
        for lanes in groups.values():
            for ordinal, lane in enumerate(lanes[1:], 1):
                if lane in branched_lanes or not bool(action[lane] < 2*n):
                    continue
                legal_flights = torch.where(legal[lane, :2*n])[0]
                if len(legal_flights) == 0:
                    raise RuntimeError("Greedy flight action absent from legal mask")
                rank = min(len(legal_flights)-1,
                           (ordinal-1) * len(legal_flights) // (len(lanes)-1))
                chosen = legal_flights[rank]
                changed += int(bool(chosen != action[lane]))
                action[lane] = chosen
                branched_lanes.add(lane)
                forced += 1
        calls += 1
        return action

    started = time.perf_counter()
    selection.greedy_action = first_dispatch_action
    try:
        with torch.inference_mode():
            out = model.rollout(nodes, depot, pref, pomo=N, greedy=True,
                                greedy_strategy=TASK["formal_greedy_strategy"],
                                return_actions=True, objective_variant="C0",
                                reserve_fraction=0, energy_scenario=None)
    finally:
        selection.greedy_action = original_greedy_action
    torch.cuda.synchronize()
    if calls == 0 or out["cost"].shape != (1, N):
        raise RuntimeError("First-dispatch branch was not invoked on POMO")
    for first_id, lanes in groups.items():
        if first_id < N and int(out["actions"][0, lanes[0], 0]) != 2*N+first_id:
            raise RuntimeError("Preserved baseline first load changed")
    complete = (out["feasible"][0].bool() & (out["unserved"][0] == 0)
                & ~out["rollout_limit_hit"][0].bool())
    costs = out["cost"][0].detach().cpu().tolist()
    late = out["telemetry"]["tardiness_s"].reshape(1, N)[0].detach().cpu().tolist()
    preserved = [i for i in representatives if bool(complete[i])]
    if not preserved:
        raise RuntimeError("No complete preserved baseline lane")
    baseline_idx = min(preserved, key=lambda i: (costs[i], i))
    eligible = [i for i in range(N) if bool(complete[i])
                and costs[i] <= costs[baseline_idx] + 1e-6
                and late[i] <= late[baseline_idx] + .05]
    branch_idx = min(eligible, key=lambda i: (late[i], costs[i], i))
    base = replay_selected(out, baseline_idx, nodes, depot, pref)
    branch = replay_selected(out, branch_idx, nodes, depot, pref)
    fallback = False
    if (branch["replay_c0"] > base["replay_c0"] + 1e-5
            or branch["tardiness_s"] > base["tardiness_s"] + .05):
        branch, branch_idx, fallback = base, baseline_idx, True

    original_qualification = None
    if check_original:
        with torch.inference_mode():
            original = model.rollout(nodes, depot, pref, pomo=N, greedy=True,
                                     greedy_strategy=TASK["formal_greedy_strategy"],
                                     return_actions=True, objective_variant="C0",
                                     reserve_fraction=0, energy_scenario=None)
        original_idx = int(select_best_complete(original)[0])
        original_plan = replay_selected(original, original_idx, nodes, depot, pref)
        if (original_plan["actions_sha256"] != base["actions_sha256"]
                or abs(original_plan["replay_c0"]-base["replay_c0"]) > 1e-5
                or abs(original_plan["tardiness_s"]-base["tardiness_s"]) > .05):
            raise RuntimeError("Preserved lanes do not reproduce frozen baseline")
        original_qualification = {"matched": True, "original_candidate_index": original_idx}
    elapsed = time.perf_counter() - started
    return {"S": base, "B": branch, "candidate_count": N,
            "complete_candidate_count": int(complete.sum()),
            "initial_legal_first_load_count": legal_first,
            "preserved_baseline_lane_count": len(representatives),
            "forced_first_dispatch_lane_count": forced,
            "changed_first_dispatch_lane_count": changed,
            "pareto_eligible_count": len(eligible),
            "selection_changed": branch_idx != baseline_idx,
            "gu_fallback": fallback,
            "original_qualification": original_qualification,
            "elapsed_s": elapsed, "within_5s": elapsed <= 5.}


def main() -> None:
    started = time.monotonic()
    write_once("started.json", {"at_utc": datetime.now(timezone.utc).isoformat(),
                                "panel": f"N{N} POMO{N} first-dispatch independent confirmation"})
    try:
        admission, anchor = admit()
        write_once("admission.json", admission)
        model = load_model(P1)
        rows = []
        with (HERE / "results.jsonl").open("x", encoding="utf-8") as stream:
            for ki, kind in enumerate(("R", "C", "RC"), 1):
                for replicate in range(5):
                    seed = 27200000 + 500 + ki + replicate*10000
                    nodes, depot, _ = generate(1, N, kind, TASK["baseline_tightness"],
                                               generator=torch.Generator(device="cpu").manual_seed(seed),
                                               device="cuda")
                    instance_sha = hashlib.sha256(nodes.cpu().numpy().tobytes()
                                                   + depot.cpu().numpy().tobytes()).hexdigest()
                    for alpha in (.1, .5, .9):
                        if time.monotonic() - started > 1800:
                            raise TimeoutError("Frozen 30-minute cap")
                        check_gpu0(anchor)
                        row = {"n": N, "kind": kind, "replicate": replicate, "seed": seed,
                               "instance_sha256": instance_sha, "alpha": alpha,
                               "P1": evaluate(model, nodes, depot, alpha,
                                              check_original=(len(rows) == 0))}
                        rows.append(row)
                        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                        stream.flush()
                        print(f"{len(rows)}/45 {kind} {replicate} {alpha}", flush=True)
        if len(rows) != 45 or len({(x["kind"], x["replicate"], x["alpha"]) for x in rows}) != 45:
            raise RuntimeError("Incomplete/duplicate panel")
        summary = {selector: {
            "mean_c0": statistics.mean(row["P1"][selector]["replay_c0"] for row in rows),
            "mean_tardiness_s": statistics.mean(row["P1"][selector]["tardiness_s"] for row in rows),
            "mean_late_order_fraction": statistics.mean(row["P1"][selector]["late_order_fraction"] for row in rows),
            "mean_flight_time_s": statistics.mean(row["P1"][selector]["flight_time_s"] for row in rows),
            "mean_energy_j": statistics.mean(row["P1"][selector]["energy_j"] for row in rows),
            "complete": sum(row["P1"][selector]["complete"] for row in rows),
            "feasible": sum(row["P1"][selector]["feasible"] for row in rows),
        } for selector in ("S", "B")}
        groups = defaultdict(list)
        for row in rows:
            groups[(row["kind"], row["replicate"])].append(row)
        group_delta = []
        for group in groups.values():
            cost_s = statistics.mean(row["P1"]["S"]["replay_c0"] for row in group)
            cost_b = statistics.mean(row["P1"]["B"]["replay_c0"] for row in group)
            late_s = statistics.mean(row["P1"]["S"]["tardiness_s"] for row in group)
            late_b = statistics.mean(row["P1"]["B"]["tardiness_s"] for row in group)
            group_delta.append((cost_s-cost_b, late_s-late_b))
        any_group_wins = sum(c > 1e-5 or l > .05 for c, l in group_delta)
        joint_group_wins = sum(c > 1e-5 and l > .05 for c, l in group_delta)
        s, b = summary["S"], summary["B"]
        no_worse = all(row["P1"]["B"]["replay_c0"]
                       <= row["P1"]["S"]["replay_c0"] + 1e-5
                       and row["P1"]["B"]["tardiness_s"]
                       <= row["P1"]["S"]["tardiness_s"] + .05 for row in rows)
        gate = (all(summary[key]["complete"] == 45 and summary[key]["feasible"] == 45
                     for key in summary) and no_worse
                and s["mean_c0"]-b["mean_c0"] >= .001*s["mean_c0"]
                and s["mean_tardiness_s"]-b["mean_tardiness_s"]
                >= .01*s["mean_tardiness_s"]
                 and any_group_wins >= 10 and joint_group_wins >= 5)
        write_once("summary.json", {"rows": 45, "physical_instances": 15,
                                    "arms": summary, "physical_any_wins": any_group_wins,
                                    "physical_joint_wins": joint_group_wins,
                                    "no_worse_by_replay": no_worse,
                                    "selection_changed": sum(row["P1"]["selection_changed"] for row in rows),
                                    "gu_fallbacks": sum(row["P1"]["gu_fallback"] for row in rows),
                                    "frozen_gate": gate, "blind_test_read": False})
        write_once("completion.json", {"rows": 45, "results_sha256": sha(HERE / "results.jsonl"),
                                       "summary_sha256": sha(HERE / "summary.json"),
                                       "elapsed_s": time.monotonic() - started})
    except BaseException as error:
        write_once("failure.json", {"error": repr(error), "traceback": traceback.format_exc(),
                                    "elapsed_s": time.monotonic() - started})
        raise


if __name__ == "__main__":
    os.nice(10)
    main()
