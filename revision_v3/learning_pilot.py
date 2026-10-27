"""Short, pre-freeze learning smoke on disposable online instances.

This checks numerical stability and whether greedy decoding improves at all.
It is not a formal training run or a model-selection result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
from pathlib import Path

import torch

from revision_v3.instances import generate
from revision_v3.policy import V3Policy, VARIANTS
from revision_v3.selection import select_best_complete
from revision_v3.spec import ROOT, SPEC_PATH, TASK
from revision_v3.train import preferences


@torch.no_grad()
def small_validation(model, n_nodes: int, device: torch.device):
    model.eval()
    rows = []
    for kind_idx, kind in enumerate(("R", "C", "RC"), 1):
        gen = torch.Generator(device="cpu").manual_seed(610000+n_nodes*10+kind_idx)
        nodes, depot, _ = generate(1, n_nodes, kind, TASK["baseline_tightness"],
                                   generator=gen, device=device)
        for alpha in (0.1, 0.5, 0.9):
            pref = torch.tensor([[alpha, 1-alpha]], dtype=nodes.dtype, device=device)
            out = model.rollout(nodes, depot, pref, pomo=n_nodes, greedy=True,
                                return_actions=True)
            chosen = int(select_best_complete(out)[0])
            actions = out["actions"][0, chosen]
            active = out["action_valid"][0, chosen]
            flight = actions[active & (actions < 2*n_nodes)]
            fast = float((flight % 2 == 1).float().mean()) if len(flight) else None
            telemetry = out["telemetry"]
            idx = chosen
            rows.append({"kind": kind, "alpha_time": alpha,
                         "raw_cost": float(out["cost"][0, chosen]),
                         "selection_cost": float(out["selection_cost"][0, chosen]),
                         "feasible": bool(out["feasible"][0, chosen]),
                         "rollout_limit_hit": bool(out["rollout_limit_hit"][0, chosen]),
                         "unserved": int(out["unserved"][0, chosen]),
                         "flight_time_s": float(telemetry["flight_time_s"][idx]),
                         "total_energy_j": float(telemetry["total_energy_j"][idx]),
                         "tardiness_s": float(telemetry["tardiness_s"][idx]),
                         "fast_flight_fraction": fast})
    model.train()
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-nodes", type=int, choices=(20, 50, 100), required=True)
    ap.add_argument("--variant", choices=VARIANTS, required=True)
    ap.add_argument("--microbatch", type=int, required=True)
    ap.add_argument("--grad-accum-steps", type=int, required=True)
    ap.add_argument("--updates", type=int, default=20)
    ap.add_argument("--entropy-coefficient", type=float, default=0.02,
                    help="Diagnostic only; formal runs use the frozen epoch schedule")
    ap.add_argument("--speed-aux-weight", type=float,
                    default=float(TASK["training_speed_aux_weight"]),
                    help="Candidate-only auxiliary weight; record before formal freeze")
    ap.add_argument("--init-checkpoint", type=Path,
                    help="Disposable pilot checkpoint for a speed-head fine-tuning pilot")
    ap.add_argument("--freeze-route", action="store_true",
                    help="Update only the factorized speed head")
    ap.add_argument("--speed-route-grad-scale", type=float,
                    help="Override route-cost policy-gradient scale into the speed head")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.microbatch < 1 or args.grad_accum_steps < 1 or args.updates < 1:
        ap.error("Batch, accumulation and updates must be positive")
    if args.entropy_coefficient < 0:
        ap.error("Entropy coefficient must be nonnegative")
    if args.speed_aux_weight < 0:
        ap.error("Speed auxiliary weight must be nonnegative")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 CUDA required")
    torch.manual_seed(711000+args.n_nodes)
    torch.cuda.manual_seed_all(711000+args.n_nodes)
    gen = torch.Generator(device="cpu").manual_seed(811000+args.n_nodes)
    device = torch.device("cuda")
    model = V3Policy(**VARIANTS[args.variant]).to(device)
    if args.init_checkpoint is not None:
        source = torch.load(args.init_checkpoint, map_location="cpu", weights_only=True)
        if source["n_nodes"] != args.n_nodes or source["variant"] != args.variant:
            raise ValueError("Pilot checkpoint does not match scale and variant")
        model.load_state_dict(source["model"])
    if args.speed_route_grad_scale is not None:
        if not 0.0 <= args.speed_route_grad_scale <= 1.0:
            ap.error("Speed route gradient scale must be in [0,1]")
        model.speed_route_grad_scale = args.speed_route_grad_scale
    if args.freeze_route:
        if not model.use_factorized_speed or args.init_checkpoint is None:
            ap.error("Speed-only fine-tuning needs a factorized speed checkpoint")
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith("speed_head."))
    head_lr_multiplier = float(TASK["training_speed_head_lr_multiplier"])
    optimizer = torch.optim.Adam(model.optimizer_parameter_groups(
        1e-4, head_lr_multiplier))
    start = time.perf_counter()
    checkpoints = {0, max(1, args.updates//2), args.updates}
    validation = {}
    history = []
    for update in range(args.updates+1):
        if update in checkpoints:
            validation[str(update)] = small_validation(model, args.n_nodes, device)
            print(json.dumps({"validated_after_updates": update,
                              "feasible": sum(r["feasible"] for r in validation[str(update)]),
                              "mean_selection_cost": statistics.mean(
                                  r["selection_cost"] for r in validation[str(update)])}),
                  flush=True)
        if update == args.updates:
            break
        optimizer.zero_grad(set_to_none=True)
        losses, feasible, unserved, limit_hits, speed_aux_losses = [], [], [], [], []
        for micro in range(args.grad_accum_steps):
            kind = ("R", "C", "RC")[(update*args.grad_accum_steps+micro) % 3]
            nodes, depot, _ = generate(args.microbatch, args.n_nodes, kind,
                                        TASK["baseline_tightness"], generator=gen,
                                        device=device)
            pref = preferences(args.microbatch, gen, mode="random").to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model.rollout(nodes, depot, pref, pomo=args.n_nodes)
                costs = out["selection_cost"]
                advantage = (costs-costs.mean(1, keepdim=True)).detach()
                scale = advantage.std(unbiased=False).detach().clamp_min(1.0)
                loss = ((advantage/scale)*out["logprob_sum"]).mean()
                loss -= args.entropy_coefficient*out["entropy"].mean()
                if model.use_speed_aux:
                    loss += args.speed_aux_weight*out["speed_aux_loss"]
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"Nonfinite loss at update {update+1}")
            (loss/args.grad_accum_steps).backward()
            losses.append(float(loss.detach()))
            feasible.append(float(out["feasible"].float().mean()))
            unserved.append(float(out["unserved"].float().mean()))
            limit_hits.append(float(out["rollout_limit_hit"].float().mean()))
            speed_aux_losses.append(float(out["speed_aux_loss"].detach()))
        grad_norms = model.clip_gradients_(1.0)
        if not all(math.isfinite(value) for value in grad_norms.values()):
            raise FloatingPointError(f"Nonfinite gradient at update {update+1}")
        optimizer.step()
        row = {"update": update+1, "loss": statistics.mean(losses),
               "rollout_feasible_rate": statistics.mean(feasible),
               "rollout_mean_unserved": statistics.mean(unserved),
               "rollout_limit_hit_rate": statistics.mean(limit_hits),
               "speed_aux_loss": statistics.mean(speed_aux_losses),
               "preclip_grad_norm": grad_norms["main"],
               "preclip_speed_head_grad_norm": grad_norms["speed_head"],
               "elapsed_s": time.perf_counter()-start}
        history.append(row)
        print(json.dumps(row), flush=True)
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    payload = {"purpose": "pre-freeze disposable short learning diagnostic",
               "n_nodes": args.n_nodes, "variant": args.variant,
               "microbatch": args.microbatch,
               "grad_accum_steps": args.grad_accum_steps,
               "effective_batch": args.microbatch*args.grad_accum_steps,
               "updates": args.updates,
               "entropy_coefficient": args.entropy_coefficient,
               "speed_aux_weight": args.speed_aux_weight,
               "init_checkpoint_sha256": (digest(args.init_checkpoint)
                                          if args.init_checkpoint else None),
               "freeze_route": args.freeze_route,
               "speed_route_grad_scale": model.speed_route_grad_scale,
               "speed_head_lr_multiplier": head_lr_multiplier,
               "gpu": torch.cuda.get_device_name(0),
               "torch": torch.__version__, "task_spec_sha256": digest(SPEC_PATH),
               "code_sha256": {name: digest(ROOT / name) for name in (
                   "env/revision_v3_gu_physics.py", "revision_v3/environment.py",
                   "revision_v3/policy.py", "revision_v3/selection.py",
                   "revision_v3/instances.py",
                   "revision_v3/train.py", "revision_v3/learning_pilot.py")},
               "validation": validation, "history": history,
               "runtime_s": time.perf_counter()-start}
    checkpoint = args.output.with_suffix(".pt")
    if checkpoint.exists():
        raise FileExistsError(checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"purpose": "pilot_only_never_formal_checkpoint",
                "n_nodes": args.n_nodes, "variant": args.variant,
                "model": model.state_dict(),
                "task_spec_sha256": digest(SPEC_PATH)}, checkpoint)
    payload["pilot_checkpoint_file"] = checkpoint.name
    payload["pilot_checkpoint_sha256"] = digest(checkpoint)
    args.output.write_text(json.dumps(payload, indent=2)+"\n", encoding="utf-8")
    print("Saved", args.output, flush=True)


if __name__ == "__main__":
    main()
