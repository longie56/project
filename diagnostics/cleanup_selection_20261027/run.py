"""Read-only W05-I candidate replay under a corrected selection interface."""
from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
import struct
import sys
import time
import traceback
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
W05 = ROOT / "experiments/w05_independent_threshold_20261026"
FORMAL = W05 / "formal_attempt_001"
SOURCE = ROOT / "cloud_backup_c02f1a677fe6_20261025/sj-tmp/tre_revision_v3_compact_formal_20261018"
CLEANUP = ROOT / "experiments/pomo_pmoco_new_screen_20261024/manifest_cleanup.py"
FINAL_LOCK = ROOT / "experiments/common_main_panel_90_cpu_replacement_20261023/FINAL_NEURAL_MODEL_LOCK.json"
PREFS = (.5, .6, .625, .65, .675, .7, .9)
SEEDS = (11, 23, 37)
KINDS = ("R", "C", "RC")
SCALES = (20, 50, 100)
EXPECTED_SOURCE = {
    "revision_v3/common_eval.py": "eac0bcd8b5f659393f908cba3c677c2ecc1e345e0d19dcd2295a9884175e08ff",
    "revision_v3/task_spec.json": "4e1f1a8071e30eb8cd46ebd9a9be86fdbdd794406d976ddc8523a234b700d8a4",
    "revision_v3/environment.py": "446d379838b4a5600b76ae3590436d298daa2f53887eac120e74cb0429a310ed",
    "env/revision_v3_gu_physics.py": "b214cf6390a64f0b33b8f342ec4e0639b80ba6e2972083fcb5e90c10832a4b79",
}
EXPECTED_CLEANUP = "8273e15363a1228d5ad4d5538b7a16da7b64ab22f8fbfa2d0752aaac56a97477"
EXPECTED_LOCK = "20d017703bcd8268acf6721e6fee7849dc575630472261871163ba04382de6d1"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def action_sha(actions: list[int]) -> str:
    return hashlib.sha256(json.dumps(actions).encode()).hexdigest()


def f32(value: float) -> float:
    return struct.unpack("f", struct.pack("f", value))[0]


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def close(actual, expected, *, atol: float = .05) -> bool:
    if actual is None or expected is None:
        return actual is None and expected is None
    return math.isclose(float(actual), float(expected), rel_tol=1e-5, abs_tol=atol)


def validate_sources() -> dict:
    completion = json.loads((FORMAL / "completion.json").read_text())
    require(not (FORMAL / "failure.json").exists(), "W05-I formal failure marker")
    require((completion["rows"], completion["unique_keys"], completion["candidate_lanes"]) ==
            (945, 945, 53550), "W05-I completion denominator")
    for name in ("results", "candidates", "sidecar"):
        require(sha(FORMAL / f"{name}.jsonl") == completion[f"{name}_sha256"],
                f"W05-I {name} SHA")
    require(sha(FINAL_LOCK) == EXPECTED_LOCK, "final model lock SHA")
    require(sha(CLEANUP) == EXPECTED_CLEANUP, "cleanup source SHA")
    for name, digest in EXPECTED_SOURCE.items():
        require(sha(SOURCE / name) == digest, f"Gu/source SHA {name}")
    run_lock = json.loads((W05 / "RUN_LOCK.json").read_text())
    for key, filename in (("runner_sha256", "run_w05.py"),
                          ("protocol_sha256", "PROTOCOL.md"),
                          ("panel_manifest_sha256", "panel_manifest.json"),
                          ("candidate_reselect_sha256", "candidate_reselect.py")):
        require(sha(W05 / filename) == run_lock[key], f"W05-I run lock {key}")
    require(run_lock["common_replay_sha256"] == EXPECTED_SOURCE["revision_v3/common_eval.py"],
            "W05-I replay source lock")
    manifest = json.loads((W05 / "panel_manifest.json").read_text())
    require(len(manifest["items"]) == 45, "W05-I physical instance count")
    expected_items = {(n, kind, rep) for n in SCALES for kind in KINDS for rep in range(5)}
    items = {(x["n"], x["kind"], x["replicate"]): x for x in manifest["items"]}
    require(set(items) == expected_items, "W05-I instance matrix")
    for item in items.values():
        require(sha(W05 / item["file"]) == item["file_sha256"],
                "W05-I instance file SHA")
    return {"completion": completion, "manifest": manifest, "items": items,
            "model_lock": json.loads(FINAL_LOCK.read_text()), "run_lock": run_lock}


def raw_s_index(candidates: list[dict], reps: list[int]) -> int:
    complete = [i for i in reps if candidates[i]["rollout_feasible"]
                and candidates[i]["rollout_unserved"] == 0
                and not candidates[i]["rollout_limit_hit"]]
    if complete:
        return min(complete, key=lambda i: (candidates[i]["rollout_cost_c0"], i))
    fewest = min(candidates[i]["rollout_unserved"] for i in reps)
    eligible = [i for i in reps if candidates[i]["rollout_unserved"] == fewest]
    no_limit = [i for i in eligible if not candidates[i]["rollout_limit_hit"]]
    if no_limit:
        eligible = no_limit
    return min(eligible, key=lambda i: (candidates[i]["rollout_cost_c0"], i))


def worker(payload: tuple[dict, dict, dict, dict]) -> dict:
    row, pool, side, item = payload
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    sys.path[:0] = [str(SOURCE), str(CLEANUP.parent), str(HERE)]
    import torch
    from manifest_cleanup import remove_undelivered_loads
    from revision_v3.common_eval import replay_actions
    from selection_v2 import representatives, select_post_cleanup

    torch.set_num_threads(1)
    n, kind, rep, alpha, seed = row["unique_key"]
    require((n, kind, rep, seed) == (item["n"], item["kind"], item["replicate"],
                                     row["training_seed"]), "row/instance identity")
    data = torch.load(W05 / item["file"], map_location="cpu", weights_only=True)
    require((data["n"], data["kind"], data["replicate"], data["seed"]) ==
            (n, kind, rep, item["seed"]), "instance metadata")
    nodes, depot = data["nodes"][0], data["depot"][0]
    require(hashlib.sha256(data["nodes"].numpy().tobytes() +
                            data["depot"].numpy().tobytes()).hexdigest() ==
            item["tensor_sha256"], "instance tensor SHA")
    pref = torch.tensor([alpha, 1-alpha], dtype=nodes.dtype)
    require(row["instance_sha256"] == pool["instance_sha256"] ==
            side["instance_sha256"] == item["tensor_sha256"], "row/pool/side input SHA")
    require(row["checkpoint_sha256"] == pool["checkpoint_sha256"] ==
            side["checkpoint_sha256"], "row/pool/side weight SHA")
    require(pool["instance_id"] == f"n{n}_{kind}_{rep}" and
            close(pool["alpha"], f32(alpha), atol=1e-7), "candidate pool key")
    require(pool["replay_error_count"] == 0 and len(pool["candidates"]) ==
            pool["candidate_count"] == n, "original candidate pool denominator")
    require([c["lane"] for c in pool["candidates"]] == list(range(n)),
            "candidate lane order")
    t0 = time.perf_counter()
    cleaned = []
    for source in pool["candidates"]:
        original = source["actions"]
        require(action_sha(original) == source["actions_sha256"],
                "original candidate action SHA")
        actions, detail = remove_undelivered_loads(original, n)
        try:
            replay = replay_actions(nodes, depot, pref, actions)
            m = replay["metrics"]
            entry = {"lane": source["lane"], "original_actions": original,
                     "actions_sha256": action_sha(actions),
                     "cleanup_removed_loads": detail["removed_load_count"],
                     "rollout_limit_hit": source["rollout_limit_hit"],
                     "complete": bool(replay["feasible"]) and int(m["unserved"]) == 0,
                     "feasible": bool(replay["feasible"]), "unserved": int(m["unserved"]),
                     "cost_c0": float(m["cost"]), "tardiness_s": float(m["tardiness_s"]),
                     "flight_time_s": float(m["flight_time_s"]),
                     "total_energy_j": float(m["total_energy_j"]), "error": None,
                     "actions": actions}
        except Exception as exc:
            entry = {"lane": source["lane"], "original_actions": original,
                     "actions_sha256": action_sha(actions),
                     "cleanup_removed_loads": detail["removed_load_count"],
                     "rollout_limit_hit": source["rollout_limit_hit"],
                     "complete": False, "feasible": False, "unserved": n,
                     "cost_c0": float("inf"), "tardiness_s": float("inf"),
                     "flight_time_s": None, "total_energy_j": None,
                     "error": f"{type(exc).__name__}: {exc}", "actions": actions}
        cleaned.append(entry)
    reps = representatives(cleaned)
    require(len(reps) == row["preserved_baseline_lane_count"],
            "preserved S representative count")
    old_s = row["S"]["candidate_index"]
    old_b = row["selected_lane"]
    require(old_s == raw_s_index(pool["candidates"], reps),
            "archived S differs from raw selection rule")
    require(row["selected"]["candidate_index"] == old_b and
            row["selected"]["original_actions_sha256"] ==
            pool["candidates"][old_b]["actions_sha256"], "archived B candidate identity")
    for name, lane in (("S", old_s), ("selected", old_b)):
        source = row[name]
        target = cleaned[lane]
        require(source["actions"] == target["actions"] and
                source["actions_sha256"] == target["actions_sha256"],
                f"archived {name} cleaned actions mismatch")
        require(close(source["cost_c0"], target["cost_c0"], atol=.02) and
                close(source["tardiness_s"], target["tardiness_s"], atol=.05) and
                close(source["total_energy_j"], target["total_energy_j"], atol=5.),
                f"archived {name} Gu metrics mismatch")
    require(side["actions_sha256"] == row["selected"]["actions_sha256"] and
            action_sha(side["actions"]) == side["actions_sha256"],
            "W05-I sidecar selected action mismatch")
    new_s, new_b, _ = select_post_cleanup(cleaned)
    def view(lane: int) -> dict:
        x = cleaned[lane]
        return {k: x[k] for k in ("lane", "actions_sha256", "cleanup_removed_loads",
                                  "complete", "feasible", "unserved", "cost_c0",
                                  "tardiness_s", "flight_time_s", "total_energy_j", "error")}
    return {"unique_key": row["unique_key"], "instance_sha256": item["tensor_sha256"],
            "checkpoint_sha256": row["checkpoint_sha256"], "candidate_count": n,
            "cleanup_changed_candidates": sum(x["cleanup_removed_loads"] > 0 for x in cleaned),
            "cleaned_replay_error_candidates": sum(x["error"] is not None for x in cleaned),
            "old_S": view(old_s), "new_S": view(new_s),
            "old_B": view(old_b), "new_B": view(new_b),
            "S_reselected": new_s != old_s, "B_reselected": new_b != old_b,
            "replay_elapsed_s": time.perf_counter() - t0}


def summarize(path: Path, rows: int) -> dict:
    groups = defaultdict(list)
    all_rows = []
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            all_rows.append(row)
            groups[row["unique_key"][0]].append(row)
    require(len(all_rows) == rows and rows == 945, "summary denominator")
    def mean_delta(rows: list[dict], method: str, field: str):
        values = [r[f"new_{method}"][field]-r[f"old_{method}"][field]
                  for r in rows if r[f"new_{method}"][field] is not None
                  and r[f"old_{method}"][field] is not None
                  and math.isfinite(r[f"new_{method}"][field])
                  and math.isfinite(r[f"old_{method}"][field])]
        return statistics.fmean(values) if values else None
    def cell(rs: list[dict]):
        return {"planned_settings": len(rs),
                "S_reselected": sum(r["S_reselected"] for r in rs),
                "B_reselected": sum(r["B_reselected"] for r in rs),
                "old_B_complete": sum(r["old_B"]["complete"] for r in rs),
                "new_B_complete": sum(r["new_B"]["complete"] for r in rs),
                "candidate_replay_errors": sum(r["cleaned_replay_error_candidates"] for r in rs),
                "mean_replay_elapsed_s": statistics.fmean(r["replay_elapsed_s"] for r in rs),
                "B_mean_delta_C0": mean_delta(rs, "B", "cost_c0"),
                "B_mean_delta_tardiness_s": mean_delta(rs, "B", "tardiness_s"),
                "B_mean_delta_flight_s": mean_delta(rs, "B", "flight_time_s"),
                "B_mean_delta_energy_j": mean_delta(rs, "B", "total_energy_j"),
                "S_mean_delta_C0": mean_delta(rs, "S", "cost_c0")}
    return {"scope": "posthoc_independent_W05_I_development_not_blind_main90",
            "original_54_group_reselection": 0,
            "overall": cell(all_rows),
            "by_scale": {str(n): cell(groups[n]) for n in SCALES},
            "selection_rows_sha256": sha(path)}


def main() -> None:
    started = time.perf_counter()
    evidence = validate_sources()
    out = HERE / "attempt_001"
    out.mkdir(exist_ok=False)
    def write_once(name: str, payload: dict):
        with (out / name).open("x") as stream:
            json.dump(payload, stream, indent=2, allow_nan=False)
            stream.write("\n")
    write_once("admission.json", {"protocol_sha256": sha(HERE / "PROTOCOL.md"),
                               "runner_sha256": sha(Path(__file__)),
                               "selection_v2_sha256": sha(HERE / "selection_v2.py"),
                               "W05_completion_sha256": sha(FORMAL / "completion.json"),
                               "W05_run_lock_sha256": sha(W05 / "RUN_LOCK.json"),
                               "final_model_lock_sha256": sha(FINAL_LOCK),
                               "Gu_source_sha256": EXPECTED_SOURCE["revision_v3/common_eval.py"],
                               "cleanup_source_sha256": EXPECTED_CLEANUP,
                               "planned_settings": 945, "planned_candidate_lanes": 53550})
    result_path = out / "selection_rows.jsonl"
    count = 0
    try:
        expected = {(n, kind, rep, alpha, seed) for n in SCALES for kind in KINDS
                    for rep in range(5) for alpha in PREFS for seed in SEEDS}
        seen = set()
        def payloads():
            with (FORMAL / "results.jsonl").open() as results, \
                 (FORMAL / "candidates.jsonl").open() as candidates, \
                 (FORMAL / "sidecar.jsonl").open() as sides:
                for rl, cl, sl in zip(results, candidates, sides, strict=True):
                    row, pool, side = json.loads(rl), json.loads(cl), json.loads(sl)
                    key = tuple(row["unique_key"])
                    require(key in expected and key not in seen, "W05-I unique matrix key")
                    seen.add(key)
                    n, kind, rep, _, seed = key
                    item = evidence["items"][n, kind, rep]
                    require(row["instance_seed"] == item["seed"] and
                            row["instance_sha256"] == item["tensor_sha256"],
                            "W05-I instance pairing")
                    weight = evidence["model_lock"]["pomo"]["by_n"][str(n)]["weights"][str(seed)]["sha256"]
                    require(row["checkpoint_sha256"] == weight and
                            row["training_arm"] == "D" and row["decode_arm"] == "B",
                            "W05-I model identity")
                    require(side["unique_key"] == row["unique_key"] and
                            side["panel"] == "W05-I", "W05-I sidecar pairing")
                    yield row, pool, side, item
            require(seen == expected, "W05-I expected 945 setting matrix")
        iterator = iter(payloads())
        with result_path.open("x") as output, ProcessPoolExecutor(max_workers=6) as executor:
            pending = set()
            exhausted = False
            while pending or not exhausted:
                while len(pending) < 12 and not exhausted:
                    try:
                        pending.add(executor.submit(worker, next(iterator)))
                    except StopIteration:
                        exhausted = True
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    row = future.result()
                    output.write(json.dumps(row, allow_nan=False, separators=(",", ":"))+"\n")
                    output.flush()
                    count += 1
                    if count % 25 == 0:
                        print(json.dumps({"completed": count, "planned": 945,
                                          "elapsed_s": round(time.perf_counter()-started, 1)}),
                              flush=True)
        summary = summarize(result_path, count)
        write_once("SUMMARY.json", summary)
        write_once("completion.json", {"rows": count,
                  "selection_rows_sha256": sha(result_path),
                  "summary_sha256": sha(out / "SUMMARY.json"),
                  "admission_sha256": sha(out / "admission.json"),
                  "elapsed_s": time.perf_counter()-started})
    except BaseException as exc:
        write_once("failure.json", {"rows": count, "error_type": type(exc).__name__,
                   "error": str(exc), "traceback": traceback.format_exc(),
                   "partial_result_sha256": sha(result_path) if result_path.exists() else None})
        raise


if __name__ == "__main__":
    main()
