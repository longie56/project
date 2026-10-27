"""Prospective physical-instance panel for matched POMO F/D and dispatch arms."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

FORMAL = Path("/root/sj-tmp/pomo_pareto_first_dispatch_formal_20261022")
CONTROL = Path("/root/sj-tmp/pomo_c0_matched_ablation_20261022")
EVAL = Path("/root/sj-tmp/pomo_pareto_paired_eval_20261022")
P1 = "physical_pref_v3_wait40_slack5"
SEED11 = {
    20: (Path("/root/sj-tmp/tre_first_backup_20261019/tre_revision_v3_compact_formal_20261018/formal_hardware_first5090"),
         "5aa5ffd169096d74ee87dd3a0522d1ca7a435ad9a3672237524e925c308c655a"),
    50: (Path("/root/sj-tmp/tre_first_backup_20261019/tre_revision_v3_compact_formal_20261018/formal_hardware_first5090"),
         "e3c5b49717e17a3e93f044d51bf7e6ad58006effe630132d83784a9c422a086f"),
    100: (Path("/root/sj-tmp/tre_revision_v3_compact_formal_20261018/formal_runs_gpu0"),
          "1ae349a5d843dd625f6bfdf60ecf8f3f06827bd03c253cfa1e4d8ae9778a0fa2"),
}
CAP = {20: 7200, 50: 10800, 100: 28800}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_once(path: Path, data: object) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def original_plan(core, model, nodes, depot, pref, n: int) -> dict:
    import torch

    began = time.perf_counter()
    with torch.inference_mode():
        out = model.rollout(nodes, depot, pref, pomo=n, greedy=True,
                            greedy_strategy=core.TASK["formal_greedy_strategy"],
                            return_actions=True, objective_variant="C0",
                            reserve_fraction=0, energy_scenario=None)
    torch.cuda.synchronize()
    index = int(core.select_best_complete(out)[0])
    selected = core.replay_selected(out, index, nodes, depot, pref)
    selected["candidate_count"] = n
    selected["complete_candidate_count"] = int((out["feasible"][0].bool()
        & (out["unserved"][0] == 0) & ~out["rollout_limit_hit"][0].bool()).sum())
    selected["elapsed_s"] = time.perf_counter() - began
    selected["within_5s"] = selected["elapsed_s"] <= 5
    return selected


def base_checkpoint(n: int, seed: int) -> tuple[Path, str]:
    if seed == 11:
        directory, expected = SEED11[n]
    else:
        gpu = 0 if seed == 23 else 1
        directory = FORMAL / f"runs_gpu{gpu}"
        report = json.loads((FORMAL / "supervisor" / f"base_n{n}_s{seed}/completion.json").read_text())
        if report.get("epochs") != 100:
            raise RuntimeError("Incomplete base training")
        expected = report["model_sha256"]
    path = (directory / f"{P1}_n{n}_s{seed}_random_C0_gu_nominal"
            / "speed_calibration/speed_stage_selection.pt")
    if not path.is_file() or sha(path) != expected:
        raise RuntimeError(f"Base model SHA mismatch: N{n} seed{seed}")
    return path, expected


def continuation(n: int, seed: int, arm: str, base_sha: str) -> tuple[Path, str]:
    directory = ((FORMAL / "pareto") if arm == "D" else (CONTROL / "runs"))
    directory = directory / f"n{n}_s{seed}"
    name = "model_D_u40.pt" if arm == "D" else "model_F_u40.pt"
    report = json.loads((directory / "completion.json").read_text())
    path = directory / name
    if report.get("updates") != 40 or sha(path) != report["checkpoint_sha256"]:
        raise RuntimeError(f"Incomplete {arm} continuation: N{n} seed{seed}")
    import torch
    state = torch.load(path, map_location="cpu", weights_only=True)
    if (state.get("n_nodes") != n or state.get("training_seed") != seed
            or state.get("updates") != 40 or state.get("base_checkpoint_sha256") != base_sha):
        raise RuntimeError(f"{arm} checkpoint identity mismatch")
    return path, report["checkpoint_sha256"]


def load_model(core, path: Path, n: int, seed: int, arm: str):
    import torch
    data = torch.load(path, map_location="cpu", weights_only=True)
    if arm == "P1":
        config = data.get("config", {})
        if (config.get("n_nodes") != n or config.get("seed") != seed
                or config.get("variant") != P1 or config.get("objective_variant") != "C0"
                or data.get("purpose") != "formal_validation_selected_speed_stage"):
            raise RuntimeError("P1 base checkpoint identity mismatch")
    model = core.V3Policy(**core.VARIANTS[P1]).cuda().eval()
    model.load_state_dict(data["model"], strict=True)
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-nodes", type=int, choices=(20, 50, 100), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    n = args.n_nodes
    began = time.monotonic()
    if socket.gethostname() != "c02f1a677fe6":
        raise RuntimeError("Wrong host")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    try:
        import torch

        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("CUDA BF16 unavailable")
        torch.set_num_threads(1)
        os.environ["N_NODES"] = str(n)
        import first_dispatch_core as core

        for name, expected in core.SOURCES.items():
            if sha(core.SOURCE / name) != expected or sha(FORMAL / "code" / name) != expected:
                raise RuntimeError(f"Frozen source changed: {name}")
        if sha(EVAL / "first_dispatch_core.py") != sha(Path(core.__file__)):
            raise RuntimeError("Evaluation core changed")
        identities = {}
        models = {}
        for seed in (11, 23, 37):
            base, base_sha = base_checkpoint(n, seed)
            identities[f"P1_s{seed}"] = {"path": str(base), "sha256": base_sha}
            for arm in ("F", "D"):
                path, digest = continuation(n, seed, arm, base_sha)
                identities[f"{arm}_s{seed}"] = {"path": str(path), "sha256": digest}
        for seed in (11, 23, 37):
            for arm in ("P1", "F", "D"):
                identity = identities[f"{arm}_s{seed}"]
                models[(seed, arm)] = load_model(core, Path(identity["path"]), n, seed, arm)
        write_once(args.output_dir / "started.json", {
            "at_utc": datetime.now(timezone.utc).isoformat(), "n_nodes": n,
            "models": identities, "script_sha256": sha(Path(__file__)),
            "core_sha256": sha(Path(core.__file__)),
            "protocol_sha256": sha(EVAL / "COMPARISON_ABLATION_PROTOCOL.md"),
            "blind_test_read": False,
        })
        rows = 0
        instance_hashes = {}
        with (args.output_dir / "results.jsonl").open("x", encoding="utf-8") as stream:
            for kind_index, kind in enumerate(("R", "C", "RC"), 1):
                for replicate in range(15):
                    instance_seed = 29300000 + 100000 * n + 1000 * kind_index + replicate
                    nodes, depot, _ = core.generate(
                        1, n, kind, core.TASK["baseline_tightness"],
                        generator=torch.Generator(device="cpu").manual_seed(instance_seed),
                        device="cuda")
                    digest = hashlib.sha256(nodes.cpu().numpy().tobytes()
                                             + depot.cpu().numpy().tobytes()).hexdigest()
                    instance_hashes[f"{kind}_{replicate}"] = digest
                    for alpha in (.1, .5, .9):
                        pref = torch.tensor([[alpha, 1 - alpha]], dtype=nodes.dtype, device="cuda")
                        for seed in (11, 23, 37):
                            for arm in ("P1", "F", "D"):
                                if time.monotonic() - began > CAP[n]:
                                    raise TimeoutError("Frozen paired evaluation scale cap")
                                identity = identities[f"{arm}_s{seed}"]
                                model = models[(seed, arm)]
                                original = original_plan(core, model, nodes, depot, pref, n)
                                branch = core.evaluate(model, nodes, depot, alpha, check_original=False)
                                if (original["actions_sha256"] != branch["S"]["actions_sha256"]
                                        or abs(original["replay_c0"] - branch["S"]["replay_c0"]) > 1e-5
                                        or abs(original["tardiness_s"] - branch["S"]["tardiness_s"]) > .05):
                                    raise RuntimeError("Preserved branch lane differs from original POMO")
                                row = {"n": n, "kind": kind, "replicate": replicate,
                                       "instance_seed": instance_seed, "instance_sha256": digest,
                                       "alpha": alpha, "training_seed": seed, "arm": arm,
                                       "model_sha256": identity["sha256"],
                                       "S": original, "B": branch["B"],
                                       "branch_elapsed_s": branch["elapsed_s"],
                                       "branch_within_5s": branch["within_5s"],
                                       "branch_candidate_count": branch["candidate_count"],
                                       "branch_complete_candidate_count": branch["complete_candidate_count"],
                                       "branch_changed": branch["selection_changed"],
                                       "branch_gu_fallback": branch["gu_fallback"]}
                                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                                stream.flush()
                                rows += 1
                                print(f"N{n} {rows}/1215 {kind} {replicate} alpha{alpha} s{seed} {arm}",
                                      flush=True)
        if rows != 1215 or len(instance_hashes) != 45:
            raise RuntimeError("Incomplete paired evaluation matrix")
        write_once(args.output_dir / "completion.json", {
            "rows": rows, "physical_instances": len(instance_hashes),
            "instance_sha256": instance_hashes,
            "results_sha256": sha(args.output_dir / "results.jsonl"),
            "elapsed_s": time.monotonic() - began,
            "blind_test_read": False,
        })
    except BaseException as exc:
        write_once(args.output_dir / "failure.json", {
            "error": repr(exc), "traceback": traceback.format_exc(),
            "elapsed_s": time.monotonic() - began,
        })
        raise


if __name__ == "__main__":
    main()
