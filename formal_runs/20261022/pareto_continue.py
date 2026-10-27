"""Frozen 40-update Pareto continuation for one completed POMO base model."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import torch

from revision_v3.instances import generate
from revision_v3.policy import V3Policy, VARIANTS
from revision_v3.spec import TASK


VARIANT = "physical_pref_v3_wait40_slack5"
UPDATES = 40


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_once(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def dominance_score(out: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Same signed rank and tolerances as the completed N50 pilot."""
    cost = out["cost"].detach().float()
    b, p = cost.shape
    late = out["telemetry"]["tardiness_s"].reshape(b, p).detach().float()
    complete = (out["feasible"].bool() & (out["unserved"] == 0)
                & ~out["rollout_limit_hit"].bool())
    if not bool(torch.isfinite(cost).all() and torch.isfinite(late).all()):
        raise FloatingPointError("Nonfinite cost or tardiness")
    cheaper = cost[:, :, None] <= cost[:, None, :] + 1e-6
    earlier = late[:, :, None] <= late[:, None, :] + .05
    strictly = ((cost[:, :, None] < cost[:, None, :] - 1e-6)
                | (late[:, :, None] < late[:, None, :] - .05))
    dominates = (cheaper & earlier & strictly
                 & complete[:, :, None] & complete[:, None, :])
    score = dominates.sum(dim=1).float() - dominates.sum(dim=2).float()
    score = torch.where(complete, score, p + 1 + out["unserved"].float())
    comparable = dominates.sum(dim=(1, 2)).float() / (p * (p - 1))
    return score, comparable


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-nodes", type=int, choices=(20, 50, 100), required=True)
    parser.add_argument("--seed", type=int, choices=(11, 23, 37), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--microbatch", type=int, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    cap_s = {20: 1800, 50: 3600, 100: 7200}[args.n_nodes]
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    try:
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("BF16 CUDA is required")
        if 16 % args.microbatch or args.microbatch > 16:
            raise ValueError("Microbatch must divide frozen effective batch 16")
        source = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        config = source.get("config", {})
        if (source.get("purpose") != "formal_validation_selected_speed_stage"
                or int(config.get("seed", -1)) != args.seed
                or int(config.get("n_nodes", -1)) != args.n_nodes
                or config.get("variant") != VARIANT
                or config.get("objective_variant") != "C0"
                or config.get("pref_mode") != "random"
                or config.get("energy_scenario") != "gu_nominal"):
            raise RuntimeError("Base checkpoint is not the frozen POMO identity")
        model = V3Policy(**VARIANTS[VARIANT]).cuda().train()
        model.load_state_dict(source["model"], strict=True)
        write_once(args.output_dir / "started.json", {
            "at_utc": datetime.now(timezone.utc).isoformat(), "n_nodes": args.n_nodes,
            "seed": args.seed, "base_checkpoint": str(args.checkpoint.resolve()),
            "base_checkpoint_sha256": sha(args.checkpoint),
            "effective_batch": 16, "microbatch": args.microbatch,
            "updates": UPDATES, "cap_s": cap_s, "blind_test_read": False,
            "script_sha256": sha(Path(__file__)),
        })
        torch.manual_seed(271828 + 10000 * args.n_nodes + args.seed)
        torch.cuda.manual_seed_all(271828 + 10000 * args.n_nodes + args.seed)
        generator = torch.Generator(device="cpu").manual_seed(
            26500000 + 10000 * args.n_nodes + args.seed)
        optimizer = torch.optim.Adam(model.optimizer_parameter_groups(1e-5, 10.0))
        with (args.output_dir / "updates.jsonl").open("x", encoding="utf-8") as stream:
            for update in range(UPDATES):
                if time.monotonic() - started > cap_s:
                    raise TimeoutError("Frozen Pareto continuation time cap")
                optimizer.zero_grad(set_to_none=True)
                components = []
                for micro in range(16 // args.microbatch):
                    kind = ("R", "C", "RC")[(update * (16 // args.microbatch) + micro) % 3]
                    nodes, depot, _ = generate(
                        args.microbatch, args.n_nodes, kind, TASK["baseline_tightness"],
                        generator=generator, device="cuda")
                    lo, hi = TASK["preference_train_interval"]
                    alpha = lo + (hi - lo) * torch.rand(args.microbatch, generator=generator)
                    pref = torch.stack([alpha, 1 - alpha], dim=1).cuda()
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        out = model.rollout(nodes, depot, pref, pomo=args.n_nodes,
                                            objective_variant="C0")
                        score, comparable = dominance_score(out)
                        advantage = (score - score.mean(dim=1, keepdim=True)).detach()
                        scale = advantage.std(unbiased=False).detach().clamp_min(1.0)
                        loss = ((advantage / scale) * out["logprob_sum"]).mean()
                        loss = loss - .002 * out["entropy"].mean()
                    if not bool(torch.isfinite(loss)):
                        raise FloatingPointError(f"Nonfinite loss at update {update + 1}")
                    (loss / (16 // args.microbatch)).backward()
                    late = out["telemetry"]["tardiness_s"].reshape(args.microbatch,
                                                                    args.n_nodes)
                    components.append((float(loss.detach()), float(out["cost"].mean()),
                                       float(late.mean()), float(comparable.mean()),
                                       float((out["feasible"].bool() & (out["unserved"] == 0)
                                              & ~out["rollout_limit_hit"].bool()).float().mean())))
                norms = model.clip_gradients_(1.0)
                if not all(math.isfinite(float(x)) for x in norms.values()):
                    raise FloatingPointError(f"Nonfinite gradient at update {update + 1}")
                optimizer.step()
                row = {"update": update + 1, "n_nodes": args.n_nodes, "seed": args.seed,
                       "mean_loss": sum(x[0] for x in components) / len(components),
                       "mean_c0": sum(x[1] for x in components) / len(components),
                       "mean_tardiness_s": sum(x[2] for x in components) / len(components),
                       "mean_comparable_pair_fraction": sum(x[3] for x in components) / len(components),
                       "complete_rate": sum(x[4] for x in components) / len(components),
                       "elapsed_s": time.monotonic() - started}
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                stream.flush()
                if (update + 1) % 10 == 0:
                    print(f"N{args.n_nodes} seed{args.seed} D {update + 1}/40", flush=True)
        checkpoint = args.output_dir / "model_D_u40.pt"
        torch.save({"model": model.state_dict(), "n_nodes": args.n_nodes,
                    "training_seed": args.seed, "updates": UPDATES,
                    "base_checkpoint_sha256": sha(args.checkpoint),
                    "script_sha256": sha(Path(__file__)), "blind_test_read": False},
                   checkpoint)
        write_once(args.output_dir / "completion.json", {
            "updates": UPDATES, "checkpoint_sha256": sha(checkpoint),
            "updates_sha256": sha(args.output_dir / "updates.jsonl"),
            "elapsed_s": time.monotonic() - started, "blind_test_read": False,
        })
    except BaseException as exc:
        write_once(args.output_dir / "failure.json", {
            "error": repr(exc), "traceback": traceback.format_exc(),
            "elapsed_s": time.monotonic() - started,
        })
        raise


if __name__ == "__main__":
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    main()
