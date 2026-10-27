"""Locked POMO main-panel evaluator shared by the two isolated candidate tiers.

Importing this module never reads a panel instance or starts an evaluation.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import platform
import signal
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
PANEL = Path("/root/sj-tmp/common_main_panel_90_cpu_replacement_20261023")
EVAL = Path("/root/sj-tmp/pomo_pareto_paired_eval_20261022")
SOURCE = Path("/root/sj-tmp/tre_revision_v3_compact_formal_20261018")
SYM = Path("/root/sj-tmp/neural_baselines_formal_20261022")
PREFS = (.1, .5, .9)
SEEDS = (11, 23, 37)
SCALES = (20, 50, 100)
TIERS = ("eight_candidate", "same_candidate_full")
sys.path[:0] = [str(SOURCE), str(EVAL), str(SYM / "paired_n20_n50_20261022")]

import torch
from manifest_cleanup import remove_undelivered_loads
from revision_v3.common_eval import replay_actions
from revision_v3.environment import RouteEnvironment
import revision_v3.selection as selection
import paired_eval as paired
from pomo_timing_gate import TimingGateRejected, classify, decide


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_once(path: Path, payload: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")


def timeout_handler(*_) -> None:
    raise TimeoutError("POMO end-to-end wall budget exceeded")


def gpu_state(device_index: int) -> dict:
    uuids = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], text=True)
    mapping = {int(x.split(",")[0].strip()): x.split(",")[1].strip()
               for x in uuids.splitlines()}
    if device_index not in mapping:
        raise RuntimeError("Requested GPU absent")
    raw = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name",
         "--format=csv,noheader"], text=True)
    owners = [x for x in raw.splitlines() if x.split(",")[0].strip() == mapping[device_index]]
    return {"utc": utc(), "gpu_index": device_index, "gpu_uuid": mapping[device_index],
            "compute_owners": owners}


def core_for(n: int):
    """Load the unchanged frozen core with the correct node count for this scale."""
    os.environ["N_NODES"] = str(n)
    spec = importlib.util.spec_from_file_location(f"frozen_first_dispatch_n{n}",
                                                  EVAL / "first_dispatch_core.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("Frozen POMO first-dispatch core unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def preflight(tier: str, output_dir: Path, wrapper: Path, device_index: int):
    if tier not in TIERS or platform.node() != "c02f1a677fe6":
        raise RuntimeError("Wrong host or candidate tier")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(device_index):
        raise RuntimeError("GPU remapping must be explicit")
    lock_path = PANEL / "FINAL_NEURAL_MODEL_LOCK.json"
    if not lock_path.exists():
        raise RuntimeError("Final POMO model lock absent; no replacement-panel neural run")
    lock = json.loads(lock_path.read_text())
    if (not isinstance(lock, dict) or
            lock.get("neural_candidate_tiers") != list(TIERS)):
        raise RuntimeError("Final lock does not explicitly authorize both frozen tiers")
    by_n = lock.get("pomo", {}).get("by_n", {})
    if set(by_n) != {str(n) for n in SCALES}:
        raise RuntimeError("Final lock lacks three explicit POMO scale decisions")
    gpu = gpu_state(device_index)
    if gpu["compute_owners"]:
        raise RuntimeError("GPU occupied; POMO cannot preempt")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA/BF16 unavailable")
    if os.statvfs(output_dir.parent).f_bavail * os.statvfs(output_dir.parent).f_frsize < 10 * 1024**3:
        raise RuntimeError("Disk below 10 GiB")
    frozen = json.loads((output_dir / "SHA_MANIFEST.json").read_text())
    for key, path in (("runner_sha256", wrapper),
                      ("common_sha256", Path(__file__)),
                      ("timing_gate_sha256", HERE / "pomo_timing_gate.py"),
                      ("protocol_sha256", output_dir / "PROTOCOL.md"),
                      ("paired_eval_sha256", EVAL / "paired_eval.py"),
                      ("first_dispatch_core_sha256", EVAL / "first_dispatch_core.py"),
                      ("cleanup_sha256", SYM / "paired_n20_n50_20261022/manifest_cleanup.py"),
                      ("common_replay_sha256", SOURCE / "revision_v3/common_eval.py"),
                      ("policy_sha256", SOURCE / "revision_v3/policy.py"),
                      ("environment_sha256", SOURCE / "revision_v3/environment.py"),
                      ("task_spec_sha256", SOURCE / "revision_v3/task_spec.json"),
                      ("physics_sha256", SOURCE / "env/revision_v3_gu_physics.py"),
                      ("neural_join_lock_sha256", PANEL / "NEURAL_JOIN_LOCK.md"),
                      ("clarification_sha256", HERE / "CLARIFICATION.md"),
                      ("instance_manifest_sha256", PANEL / "instance_manifest_90.json")):
        if sha(path) != frozen[key]:
            raise RuntimeError(f"Frozen SHA mismatch: {key}")
    if (frozen["candidate_tier"] != tier or frozen["planned_rows"] != 810 or
            lock.get("pomo_evaluation_sha256", {}).get(tier) != frozen["runner_sha256"]):
        raise RuntimeError("Final method lock does not pin this POMO tier runner")
    source_keys = ("common_sha256", "timing_gate_sha256", "paired_eval_sha256",
                   "first_dispatch_core_sha256", "cleanup_sha256",
                   "common_replay_sha256", "policy_sha256", "environment_sha256",
                   "task_spec_sha256", "physics_sha256")
    if lock.get("pomo_source_sha256") != {key: frozen[key] for key in source_keys}:
        raise RuntimeError("Final method lock does not pin the frozen POMO implementation")
    manifest = json.loads((PANEL / "instance_manifest_90.json").read_text())
    if len(manifest["items"]) != 90 or manifest["preferences"] != list(PREFS):
        raise RuntimeError("Replacement instance manifest is incomplete")
    models = {}
    for n in SCALES:
        entry = by_n[str(n)]
        arm, decoder = entry.get("training_arm"), entry.get("decode_arm")
        if arm not in ("P1", "F", "D") or decoder not in ("S", "B"):
            raise RuntimeError("Final method arm/decoder is missing or invalid")
        if set(entry.get("weights", {})) != {str(s) for s in SEEDS}:
            raise RuntimeError("Final lock lacks all three POMO training seeds")
        for seed in SEEDS:
            base, base_sha = paired.base_checkpoint(n, seed)
            path, digest = ((base, base_sha) if arm == "P1" else
                            paired.continuation(n, seed, arm, base_sha))
            pinned = entry["weights"][str(seed)]
            if pinned != {"path": str(path), "sha256": digest}:
                raise RuntimeError(f"Final POMO checkpoint identity differs: N{n} s{seed}")
            models[f"n{n}_s{seed}"] = {"path": str(path), "sha256": digest,
                                      "training_arm": arm, "decode_arm": decoder}
    admission = {"utc": utc(), "host": platform.node(), "device": device_index,
                 "gpu_state": gpu, "candidate_tier": tier,
                 "candidate_counts": {str(n): 8 if tier == "eight_candidate" else n for n in SCALES},
                 "planned_rows": 810, "pilot_seed": 11, "pilot_alpha": .5,
                 "final_neural_model_lock_sha256": sha(lock_path),
                 "instance_manifest_sha256": frozen["instance_manifest_sha256"],
                 "sha_manifest_sha256": sha(output_dir / "SHA_MANIFEST.json"),
                 "runner_sha256": frozen["runner_sha256"],
                 "common_sha256": frozen["common_sha256"],
                 "timing_gate_sha256": frozen["timing_gate_sha256"],
                 "weights": models, "quality_fields_read": False}
    return manifest, admission, lock


def load_item(item: dict):
    path = PANEL / item["file"]
    if sha(path) != item["file_sha256"]:
        raise RuntimeError("Frozen instance file SHA mismatch")
    data = torch.load(path, map_location="cpu", weights_only=True)
    if (data["n"], data["kind"], data["replicate"], data["seed"]) != (
            item["n"], item["kind"], item["replicate"], item["seed"]):
        raise RuntimeError("Frozen instance identity mismatch")
    nodes, depot = data["nodes"], data["depot"]
    digest = hashlib.sha256(nodes.numpy().tobytes() + depot.numpy().tobytes()).hexdigest()
    if digest != item["tensor_sha256"]:
        raise RuntimeError("Frozen instance tensor SHA mismatch")
    return nodes, depot


def plan_from_out(core, out: dict, index: int, nodes, depot, pref, n: int) -> dict:
    original_actions = [int(x) for x in out["actions"][0, index][out["action_valid"][0, index]].cpu().tolist()]
    raw = replay_actions(nodes[0].cpu(), depot[0].cpu(), pref[0].cpu(), original_actions)
    cost = float(raw["metrics"]["cost"])
    late = float(raw["metrics"]["tardiness_s"])
    native_cost = float(out["cost"][0, index])
    native_late = float(out["telemetry"]["tardiness_s"].reshape(1, -1)[0, index])
    if abs(cost - native_cost) > max(.02, .001 * abs(native_cost)) or abs(late - native_late) > .05:
        raise RuntimeError("Frozen POMO rollout and Gu replay differ")
    actions, cleanup = remove_undelivered_loads(original_actions, n)
    report = replay_actions(nodes[0].cpu(), depot[0].cpu(), pref[0].cpu(), actions)
    m = report["metrics"]
    complete = bool(report["feasible"]) and int(m["unserved"]) == 0
    per_order = [None] * n
    for event in report["events"]:
        if event["kind"] == "deliver":
            order = int(event["order"])
            arrival = float(event["arrival_s"])
            ready = float(nodes[0, order, 3])
            due = float(nodes[0, order, 4])
            per_order[order] = max(0., max(arrival, ready)-due)
    if complete and (any(x is None for x in per_order) or
                     abs(sum(per_order)-float(m["tardiness_s"])) > .05):
        raise RuntimeError("Per-order lateness and Gu replay disagree")
    valid_late = [x for x in per_order if x is not None]
    return {"candidate_index": index, "original_actions": original_actions,
            "original_actions_sha256": hashlib.sha256(json.dumps(original_actions).encode()).hexdigest(),
            "actions": actions, "actions_sha256": hashlib.sha256(json.dumps(actions).encode()).hexdigest(),
            "cleanup_removed_loads": cleanup["removed_load_count"],
            "events": report["events"], "complete": complete, "feasible": bool(report["feasible"]),
            "unserved": int(m["unserved"]), "cost_c0": float(m["cost"]),
            "tardiness_s": float(m["tardiness_s"]), "flight_time_s": float(m["flight_time_s"]),
            "per_order_tardiness_s": per_order,
            "late_order_fraction": sum(x > 0 for x in valid_late)/n,
            "late_order_p95_s": (sorted(valid_late)[math.ceil(.95*n)-1]
                                   if len(valid_late) == n else None),
            "total_energy_j": float(m["total_energy_j"]),
            "hover_energy_j": float(m["hover_energy_j"]),
            "waiting_s": float(m["wait_time_s"]),
            "min_return_battery_j": m["min_return_battery_j"],
            "failure_reason": report["failure_reason"],
            "failure_state": None,
            "failure_state_status": "not_applicable" if complete else "unavailable",
            "native_cost": native_cost, "native_tardiness_s": native_late,
            "replay_error": abs(cost - native_cost)}


def rank(out, indices):
    candidates = list(indices)
    complete = [i for i in candidates if bool(out["feasible"][0, i])
                and int(out["unserved"][0, i]) == 0
                and not bool(out["rollout_limit_hit"][0, i])]
    if complete:
        return min(complete, key=lambda i: (float(out["cost"][0, i]), i))
    fewest = min(int(out["unserved"][0, i]) for i in candidates)
    eligible = [i for i in candidates if int(out["unserved"][0, i]) == fewest]
    no_limit = [i for i in eligible if not bool(out["rollout_limit_hit"][0, i])]
    if no_limit:
        eligible = no_limit
    return min(eligible, key=lambda i: (float(out["cost"][0, i]), i))


def independent_s(model, core, nodes, depot, pref, n: int, k: int):
    began = time.perf_counter()
    with torch.inference_mode():
        out = model.rollout(nodes, depot, pref, pomo=k, greedy=True,
                            greedy_strategy=core.TASK["formal_greedy_strategy"],
                            return_actions=True, objective_variant="C0",
                            reserve_fraction=0, energy_scenario=None)
    torch.cuda.synchronize()
    decode_s = time.perf_counter() - began
    idx = rank(out, range(k))
    plan = plan_from_out(core, out, idx, nodes, depot, pref, n)
    return plan, decode_s, time.perf_counter()-began-decode_s


def branch_pair(model, core, nodes, depot, pref, n: int, k: int):
    env = RouteEnvironment(nodes, depot, pref, reserve_fraction=0, objective_variant="C0")
    legal = env.get_mask()[0, 2*n:3*n]
    count = int(legal.sum())
    groups = defaultdict(list)
    if count:
        available = torch.where(legal)[0]
        starts = available[(torch.arange(k, device=nodes.device) * count // k)]
        for lane, first_id in enumerate(starts.tolist()):
            groups[first_id].append(lane)
    else:
        groups[n] = list(range(k))
    representatives = [lanes[0] for lanes in groups.values()]
    original = selection.greedy_action
    branched, calls = set(), 0

    def first_dispatch(logits, mask, n_nodes, strategy="joint"):
        nonlocal calls
        action = original(logits, mask, n_nodes, strategy).clone()
        for lanes in groups.values():
            for ordinal, lane in enumerate(lanes[1:], 1):
                if lane in branched or not bool(action[lane] < 2*n_nodes):
                    continue
                choices = torch.where(mask[lane, :2*n_nodes])[0]
                if len(choices) == 0:
                    raise RuntimeError("Greedy flight action absent from legal mask")
                pos = min(len(choices)-1, (ordinal-1)*len(choices)//(len(lanes)-1))
                action[lane] = choices[pos]
                branched.add(lane)
        calls += 1
        return action

    selection.greedy_action = first_dispatch
    began = time.perf_counter()
    try:
        with torch.inference_mode():
            out = model.rollout(nodes, depot, pref, pomo=k, greedy=True,
                                greedy_strategy=core.TASK["formal_greedy_strategy"],
                                return_actions=True, objective_variant="C0",
                                reserve_fraction=0, energy_scenario=None)
    finally:
        selection.greedy_action = original
    torch.cuda.synchronize()
    decode_s = time.perf_counter() - began
    if calls == 0 or out["cost"].shape != (1, k):
        raise RuntimeError("First-dispatch branch was not invoked")
    for first_id, lanes in groups.items():
        if first_id < n and int(out["actions"][0, lanes[0], 0]) != 2*n+first_id:
            raise RuntimeError("Preserved baseline first load changed")
    base_idx = rank(out, representatives)
    complete = (out["feasible"][0].bool() & (out["unserved"][0] == 0)
                & ~out["rollout_limit_hit"][0].bool())
    costs = out["cost"][0].detach().cpu().tolist()
    late = out["telemetry"]["tardiness_s"].reshape(1, k)[0].detach().cpu().tolist()
    eligible = [i for i in range(k) if bool(complete[i]) and
                costs[i] <= costs[base_idx] + 1e-6 and late[i] <= late[base_idx] + .05]
    branch_idx = min(eligible, key=lambda i: (late[i], costs[i], i)) if eligible else base_idx
    baseline = plan_from_out(core, out, base_idx, nodes, depot, pref, n)
    branch = plan_from_out(core, out, branch_idx, nodes, depot, pref, n)
    fallback = False
    if (branch["cost_c0"] > baseline["cost_c0"] + 1e-5 or
            branch["tardiness_s"] > baseline["tardiness_s"] + .05):
        branch, branch_idx, fallback = baseline, base_idx, True
    return baseline, branch, {"branch_changed": branch_idx != base_idx,
                               "branch_gu_fallback": fallback,
                               "preserved_baseline_lane_count": len(representatives),
                               "branch_candidate_count": k,
                               "branch_complete_candidate_count": int(complete.sum())}, \
           decode_s, time.perf_counter()-began-decode_s


def eval_one(model, core, item, alpha, seed, identity, admission_sha, tier, device):
    started = time.perf_counter()
    n = item["n"]
    k = 8 if tier == "eight_candidate" else n
    budget = 60 if n in (20, 50) else 300
    row = {"n": n, "kind": item["kind"], "replicate": item["replicate"],
           "instance_seed": item["seed"], "instance_file_sha256": item["file_sha256"],
           "instance_sha256": item["tensor_sha256"], "alpha": alpha,
           "method": "POMO", "training_seed": seed,
           "training_arm": identity["training_arm"], "decode_arm": identity["decode_arm"],
           "weight_sha256": identity["sha256"], "candidate_tier": tier,
           "candidate_count": k, "budget_s": budget, "admission_sha256": admission_sha}
    signal.setitimer(signal.ITIMER_REAL, budget)
    try:
        raw_nodes, raw_depot = load_item(item)
        nodes, depot = raw_nodes.to(device), raw_depot.to(device)
        pref = torch.tensor([[alpha, 1-alpha]], dtype=nodes.dtype, device=device)
        s_independent, s_decode, s_replay = independent_s(model, core, nodes, depot, pref, n, k)
        s_preserved, b, detail, b_decode, b_replay = branch_pair(
            model, core, nodes, depot, pref, n, k)
        if (s_independent["actions_sha256"] != s_preserved["actions_sha256"] or
                abs(s_independent["cost_c0"]-s_preserved["cost_c0"]) > 1e-5 or
                abs(s_independent["tardiness_s"]-s_preserved["tardiness_s"]) > .05):
            raise RuntimeError("Preserved branch lane differs from independent S")
        chosen = s_independent if identity["decode_arm"] == "S" else b
        row.update({"status": "evaluated", "S": s_independent, "B": b,
                    "selected": chosen, "native_s": s_decode+b_decode,
                    "replay_s": s_replay+b_replay, **detail})
    except BaseException as exc:
        empty_sha = hashlib.sha256(b"[]").hexdigest()
        row.update({"status": "timeout" if isinstance(exc, TimeoutError) else "error",
                    "error_type": type(exc).__name__, "error_message": str(exc),
                    "traceback": traceback.format_exc(), "native_s": None,
                    "replay_s": None,
                    "selected": {"actions": [], "actions_sha256": empty_sha, "events": [],
                                 "complete": False, "feasible": False,
                                 "per_order_tardiness_s": [None]*n,
                                 "late_order_fraction": None,
                                 "late_order_p95_s": None,
                                 "failure_reason": str(exc), "failure_state": None,
                                 "failure_state_status": "unavailable"}})
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    row["elapsed_s"] = time.perf_counter() - started
    row["within_budget"] = row["elapsed_s"] <= budget
    return row


def sidecar(row, admission):
    chosen = row["selected"]
    return {"panel": "main90", "unique_key": [row["candidate_tier"], row["n"],
            row["kind"], row["replicate"], row["alpha"], row["training_seed"], "POMO"],
            "method": "POMO", "instance_sha256": row["instance_sha256"],
            "actions": chosen["actions"], "actions_sha256": chosen["actions_sha256"],
            "events": chosen["events"], "complete": chosen["complete"],
            "feasible": chosen["feasible"], "failure_reason": chosen["failure_reason"],
            "failure_state": chosen["failure_state"],
            "failure_state_status": chosen["failure_state_status"],
            "source_sha256": admission["runner_sha256"],
            "task_spec_sha256": admission["source_sha256"]["task_spec_sha256"],
            "physics_sha256": admission["source_sha256"]["physics_sha256"],
            "environment_sha256": admission["source_sha256"]["environment_sha256"],
            "policy_sha256": admission["source_sha256"]["policy_sha256"],
            "checkpoint_sha256": row["weight_sha256"],
            "panel_manifest_sha256": admission["instance_manifest_sha256"]}


def run(tier: str, output_dir: Path, wrapper: Path, device_index: int):
    protected = [output_dir / name for name in (
        "admission.json", "started.json", "results.jsonl", "sidecar.jsonl",
        "timing_only.jsonl", "timing_gate.json", "completion.json", "failure.json")]
    if any(path.exists() for path in protected):
        raise FileExistsError("POMO tier already attempted; preserve original evidence")
    manifest, admission, _ = preflight(tier, output_dir, wrapper, device_index)
    items = {n: [x for x in manifest["items"] if x["n"] == n] for n in SCALES}
    if any(len(items[n]) != 30 for n in SCALES):
        raise RuntimeError("Replacement manifest lacks 30 physical items per scale")
    admission["source_sha256"] = {
        key: json.loads((output_dir / "SHA_MANIFEST.json").read_text())[key]
        for key in ("task_spec_sha256", "physics_sha256", "environment_sha256", "policy_sha256")}
    write_once(output_dir / "admission.json", admission)
    admission_sha = sha(output_dir / "admission.json")
    write_once(output_dir / "started.json", {"utc": utc(), "pid": os.getpid(),
        "gpu_index": device_index, "candidate_tier": tier,
        "admission_sha256": admission_sha})
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    signal.signal(signal.SIGALRM, timeout_handler)
    device = torch.device("cuda:0")
    count = errors = 0
    began = time.perf_counter()
    pilot = []
    pilot_keys = {(x["n"], x["kind"], x["replicate"], .5, 11)
                  for n in SCALES for x in items[n][:3]}
    try:
        with (output_dir / "results.jsonl").open("x") as result, \
             (output_dir / "sidecar.jsonl").open("x") as evidence, \
             (output_dir / "timing_only.jsonl").open("x") as timings:
            def emit(item, alpha, seed, model, core, prefix_index=None):
                nonlocal count, errors
                state = gpu_state(device_index)
                if state["compute_owners"] and not all(
                        str(os.getpid()) in owner for owner in state["compute_owners"]):
                    raise RuntimeError("GPU owner appeared during POMO arm")
                identity = admission["weights"][f"n{item['n']}_s{seed}"]
                torch.cuda.reset_peak_memory_stats(device)
                row_started = time.perf_counter()
                row = eval_one(model, core, item, alpha, seed, identity,
                               admission_sha, tier, device)
                result.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+"\n")
                result.flush()
                evidence.write(json.dumps(sidecar(row, admission), ensure_ascii=False,
                                          allow_nan=False)+"\n")
                evidence.flush()
                elapsed = time.perf_counter() - row_started
                timing = {"n": item["n"], "alpha": alpha, "training_seed": seed,
                          "manifest_prefix_index": prefix_index,
                          "candidate_tier": tier, "candidate_count": row["candidate_count"],
                          "budget_s": row["budget_s"], "elapsed_s": elapsed,
                          "budget_qualified": elapsed <= row["budget_s"],
                          "native_s": row["native_s"],
                          "replay_s": row["replay_s"],
                          "gpu_peak_memory_bytes": torch.cuda.max_memory_allocated(device),
                          "status": row["status"]}
                timings.write(json.dumps(timing, allow_nan=False)+"\n")
                timings.flush()
                count += 1
                errors += row["status"] != "evaluated"
                if prefix_index is not None:
                    pilot.append(timing)
                    if classify(timing, tier) == "stop":
                        write_once(output_dir / "timing_gate.json", {
                            "decision": "stop", "observed_pilot_rows": len(pilot),
                            "reason": "first-batch error or end-to-end budget overrun",
                            "timing_only_sha256": sha(output_dir / "timing_only.jsonl"),
                            "quality_fields_read": False})
                        raise TimingGateRejected("POMO first-batch timing gate stopped")
                elif row["status"] != "evaluated" or not timing["budget_qualified"]:
                    raise RuntimeError("POMO row failed or exceeded the frozen wall budget; preserve prefix")

            for n in SCALES:
                core = core_for(n)
                seed = 11
                identity = admission["weights"][f"n{n}_s{seed}"]
                model = paired.load_model(core, Path(identity["path"]), n, seed,
                                          identity["training_arm"])
                for prefix_index, item in enumerate(items[n][:3]):
                    emit(item, .5, seed, model, core, prefix_index)
                del model
                torch.cuda.empty_cache()
            decision = decide(pilot, tier)
            write_once(output_dir / "timing_gate.json", {
                **decision, "timing_only_sha256": sha(output_dir / "timing_only.jsonl"),
                "quality_fields_read": False})
            if decision["decision"] == "stop":
                raise TimingGateRejected("POMO first-batch timing gate stopped")
            for n in SCALES:
                core = core_for(n)
                models = {}
                for seed in SEEDS:
                    identity = admission["weights"][f"n{n}_s{seed}"]
                    models[seed] = paired.load_model(core, Path(identity["path"]), n,
                                                     seed, identity["training_arm"])
                for item in items[n]:
                    for alpha in PREFS:
                        for seed in SEEDS:
                            if (n, item["kind"], item["replicate"], alpha, seed) in pilot_keys:
                                continue
                            emit(item, alpha, seed, models[seed], core)
                    print(json.dumps({"n": n, "kind": item["kind"],
                        "replicate": item["replicate"], "rows": count,
                        "error_status_rows": errors}), flush=True)
                del models
                torch.cuda.empty_cache()
        if count != 810:
            raise RuntimeError(f"Incomplete POMO tier denominator: {count}/810")
        write_once(output_dir / "completion.json", {"utc": utc(), "rows": count,
            "error_status_rows": errors, "elapsed_total_s": time.perf_counter()-began,
            "results_sha256": sha(output_dir / "results.jsonl"),
            "sidecar_sha256": sha(output_dir / "sidecar.jsonl"),
            "timing_only_sha256": sha(output_dir / "timing_only.jsonl"),
            "timing_gate_sha256": sha(output_dir / "timing_gate.json"),
            "admission_sha256": admission_sha})
    except BaseException:
        write_once(output_dir / "failure.json", {"utc": utc(), "rows": count,
            "traceback": traceback.format_exc(),
            "partial_results_sha256": sha(output_dir / "results.jsonl") if
                (output_dir / "results.jsonl").exists() else None,
            "partial_sidecar_sha256": sha(output_dir / "sidecar.jsonl") if
                (output_dir / "sidecar.jsonl").exists() else None,
            "partial_timing_sha256": sha(output_dir / "timing_only.jsonl") if
                (output_dir / "timing_only.jsonl").exists() else None})
        raise
