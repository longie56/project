"""Gu-v3 training only. Formal runs are exactly 100 epochs after pilot freeze.

This runner does not load legacy checkpoints. Every validation instance is
read from the frozen v3 manifest; the test split is never consulted here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from env.revision_v3_gu_physics import SCENARIOS
from revision_v3.hardware_profile import choices as microbatch_choices, validate_lock
from revision_v3.instances import generate
from revision_v3.policy import V3Policy, VARIANTS
from revision_v3.selection import select_best_complete
from revision_v3.spec import ROOT, SPEC_PATH, TASK


SOURCE_FILES = (
    "env/revision_v3_gu_physics.py", "revision_v3/spec.py",
    "revision_v3/environment.py", "revision_v3/instances.py",
    "revision_v3/policy.py", "revision_v3/selection.py",
    "revision_v3/hardware_profile.py", "revision_v3/train.py",
    "revision_v3/sensitivity_spec.json",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_frozen(directory: Path, n_nodes: int):
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["task_spec_sha256"] != digest(SPEC_PATH):
        raise ValueError("Task specification differs from frozen instance manifest")
    if manifest.get("formal_instance_seed_bases") != TASK.get("formal_instance_seed_bases"):
        raise ValueError("Frozen validation/test seed namespace differs from task")
    if manifest["physics_spec_sha256"] != digest(ROOT / TASK["physics_spec"]):
        raise ValueError("Physics specification differs from frozen instance manifest")
    if manifest["generator_sha256"] != digest(ROOT / "revision_v3/instances.py"):
        raise ValueError("Instance generator differs from frozen manifest")
    sensitivity_path = ROOT / "revision_v3/sensitivity_spec.json"
    if manifest.get("sensitivity_spec_sha256") != digest(sensitivity_path):
        raise ValueError("Comparison specification differs from frozen manifest")
    if (manifest.get("dataset_id") != TASK["task_id"]
            or manifest.get("status") != "frozen_formal_not_yet_tested"
            or manifest.get("tightness") != TASK["baseline_tightness"]
            or manifest.get("train_seeds_reserved") != TASK["training_seeds"]):
        raise ValueError("Frozen dataset identity differs from formal task")
    expected = {}
    for split in ("val", "test"):
        for scale in (20, 50, 100):
            count = (TASK["validation_per_n_and_type"] if split == "val"
                     else TASK["test_per_n_and_type"][str(scale)])
            for type_idx, kind in enumerate(("R", "C", "RC"), 1):
                expected[(split, scale, kind)] = {
                    "count": count,
                    "seed": (TASK["formal_instance_seed_bases"][split]
                             + scale*10 + type_idx),
                    "file": f"{split}_n{scale}_{kind}.pt",
                }
    actual = {}
    for row in manifest.get("sets", ()):
        key = (row.get("split"), row.get("n_nodes"), row.get("kind"))
        if key in actual:
            raise ValueError(f"Duplicate frozen instance subset: {key}")
        actual[key] = row
    if actual.keys() != expected.keys():
        raise ValueError("Frozen dataset must contain the exact 18 subsets")
    for key, fields in expected.items():
        row = actual[key]
        if any(row.get(name) != value for name, value in fields.items()):
            raise ValueError(f"Frozen subset metadata differs: {key}")
        if digest(directory / row["file"]) != row.get("sha256"):
            raise ValueError(f"Frozen instance hash mismatch: {row['file']}")
    entries = [x for x in manifest["sets"] if x["split"] == "val" and x["n_nodes"] == n_nodes]
    if len(entries) != 3 or {x["kind"] for x in entries} != {"R", "C", "RC"}:
        raise ValueError("Frozen validation must contain R, C and RC")
    return manifest, entries


def preferences(batch: int, generator: torch.Generator, *, mode: str = "random",
                fixed_alpha: float = 0.5):
    if mode == "random":
        low, high = TASK["preference_train_interval"]
        alpha = low + (high-low) * torch.rand(batch, generator=generator)
    elif mode == "heldout":
        (left_low, left_high), (right_low, right_high) = TASK[
            "preference_heldout_training_intervals"
        ]
        branch = torch.rand(batch, generator=generator) < 0.5
        left = left_low + (left_high-left_low)*torch.rand(batch, generator=generator)
        right = right_low + (right_high-right_low)*torch.rand(batch, generator=generator)
        alpha = torch.where(branch, left, right)
    else:
        alpha = torch.full((batch,), fixed_alpha)
    return torch.stack([alpha, 1-alpha], dim=-1)


def checkpoint_validation_preferences(mode: str, fixed_alpha: float):
    """Keep held-out preference values out of checkpoint selection."""
    if mode == "heldout":
        return (0.1, 0.3, 0.7, 0.9)
    if mode == "fixed":
        return (float(fixed_alpha),)
    return tuple(TASK["validation_preference_values"])


@torch.no_grad()
def validate(model, entries, directory: Path, *, n_pomo: int, chunk: int,
             device: torch.device, objective_variant: str, energy_scenario,
             validation_preferences):
    model.eval()
    rows = []
    for entry in entries:
        payload = torch.load(directory / entry["file"], map_location="cpu", weights_only=True)
        for alpha in validation_preferences:
            costs, feas, unserved, limit_hits = [], [], [], []
            for start in range(0, entry["count"], chunk):
                stop = min(start+chunk, entry["count"])
                nodes = payload["nodes"][start:stop].to(device)
                depot = payload["depot"][start:stop].to(device)
                pref = torch.tensor([alpha, 1-alpha], device=device).expand(stop-start, 2)
                out = model.rollout(nodes, depot, pref, pomo=n_pomo, greedy=True,
                                    greedy_strategy=TASK["formal_greedy_strategy"],
                                    objective_variant=objective_variant,
                                    energy_scenario=energy_scenario)
                chosen = select_best_complete(out)
                index = torch.arange(stop-start, device=device)
                costs.extend(out["cost"][index, chosen].cpu().tolist())
                feas.extend(out["feasible"][index, chosen].cpu().tolist())
                unserved.extend(out["unserved"][index, chosen].cpu().tolist())
                limit_hits.extend(out["rollout_limit_hit"][index, chosen].cpu().tolist())
            rows.append({"kind": entry["kind"], "alpha": alpha,
                         "mean_raw_cost": sum(costs)/len(costs),
                         "feasible_rate": sum(feas)/len(feas),
                         "mean_unserved": sum(unserved)/len(unserved),
                         "rollout_limit_hit_rate": sum(limit_hits)/len(limit_hits)})
    model.train()
    # Feasibility dominates selection; invalid routes cannot win by a low raw cost.
    penalties = TASK["training_selection_penalties"]
    selection = sum(r["mean_raw_cost"]
                    + float(penalties["per_unserved_order"])*r["mean_unserved"]
                    + float(penalties["incomplete_or_invalid_route"])
                    *(1-r["feasible_rate"]) for r in rows) / len(rows)
    return selection, rows


def save_checkpoint(path: Path, epoch: int, model, optimizer, history, args,
                    generator: torch.Generator):
    payload = {
        "epoch": epoch, "model": model.state_dict(),
        "optimizer": optimizer.state_dict(), "history": history,
        "config": {key: str(value) if isinstance(value, Path) else value
                   for key, value in vars(args).items()
                   if key not in ("resume", "dry_run")},
        "task_spec_sha256": digest(SPEC_PATH),
        "hardware_profile_sha256": (digest(args.hardware_profile)
                                    if args.hardware_profile is not None else None),
        "source_sha256": {name: digest(ROOT / name) for name in SOURCE_FILES},
        "cpu_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all(),
        "data_rng": generator.get_state(),
    }
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temp)
    os.replace(temp, path)


def reconcile_best_checkpoint(latest_path: Path, best_path: Path,
                              latest: dict) -> None:
    """Finish an epoch commit interrupted after latest but before best.

    Formal runs save latest first. If the latest epoch has the best validation
    score but the process stopped before saving best, its exact weights can be
    copied from latest. Any other mismatch is a hard error, not a silent
    change of the selected checkpoint.
    """
    history = latest["history"]
    epoch = int(latest["epoch"])
    if len(history) != epoch or history[-1]["epoch"] != epoch:
        raise ValueError("Latest checkpoint history is not an epoch commit")
    validated = [row for row in history
                 if row["validation_selection"] is not None]
    if not validated:
        if best_path.exists():
            raise ValueError("Best checkpoint exists before first validation")
        return
    best_row = min(validated, key=lambda row: row["validation_selection"])
    expected_epoch = int(best_row["epoch"])
    if best_path.exists():
        best = torch.load(best_path, map_location="cpu", weights_only=True)
        if (best.get("config") != latest["config"]
                or best.get("task_spec_sha256") != latest["task_spec_sha256"]
                or best.get("hardware_profile_sha256")
                != latest.get("hardware_profile_sha256")
                or best.get("source_sha256") != latest["source_sha256"]):
            raise ValueError("Best and latest checkpoints belong to different runs")
        best_epoch = int(best["epoch"])
        best_history = best.get("history", [])
        if (len(best_history) != best_epoch
                or best_history[-1]["epoch"] != best_epoch
                or best_history[-1]["validation_selection"] is None
                or best_history != history[:best_epoch]):
            raise ValueError("Best checkpoint has inconsistent validation history")
        if best_epoch == expected_epoch:
            return
    if expected_epoch != epoch:
        raise ValueError("Best checkpoint does not match committed validation")
    temp = best_path.with_suffix(best_path.suffix + ".tmp")
    shutil.copy2(latest_path, temp)
    os.replace(temp, best_path)


def main():
    parser = argparse.ArgumentParser(description="Physically consistent TRE v3 training")
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--n-nodes", type=int, choices=(20, 50, 100), required=True)
    parser.add_argument("--seed", type=int, choices=TASK["training_seeds"], required=True)
    parser.add_argument("--instances", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--microbatch", type=int, required=True)
    parser.add_argument("--grad-accum-steps", type=int, required=True)
    parser.add_argument("--updates-per-epoch", type=int)
    parser.add_argument("--pomo", type=int)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--entropy-start", type=float, default=0.02)
    parser.add_argument("--entropy-end", type=float, default=0.002)
    parser.add_argument("--pref-mode", choices=("random", "heldout", "fixed"), default="random")
    parser.add_argument("--objective-variant", choices=("C0", "C1"), default="C0")
    parser.add_argument("--energy-scenario", choices=SCENARIOS, default="gu_nominal")
    parser.add_argument("--fixed-alpha", type=float, default=0.5)
    parser.add_argument("--val-chunk", type=int, default=64)
    parser.add_argument("--save-every", type=int,
                        default=int(TASK["training_checkpoint_every_epochs"]))
    parser.add_argument("--hardware-profile", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    from revision_v3.spec import validate_formal_experiment
    try:
        validate_formal_experiment(
            variant=args.variant, n_nodes=args.n_nodes, pref_mode=args.pref_mode,
            fixed_alpha=args.fixed_alpha, objective_variant=args.objective_variant,
            energy_scenario=args.energy_scenario)
    except ValueError as exc:
        parser.error(str(exc))
    energy_scenario = SCENARIOS[args.energy_scenario]
    if args.microbatch < 1 or args.grad_accum_steps < 1 or args.val_chunk < 1:
        parser.error("Microbatch, accumulation and validation chunk must be positive")
    if args.resume and args.dry_run:
        parser.error("Resume and dry-run cannot be combined")
    frozen_updates = TASK["optimizer_updates_per_epoch_by_n"][str(args.n_nodes)]
    if args.updates_per_epoch is None:
        args.updates_per_epoch = frozen_updates
    if args.updates_per_epoch is None:
        parser.error("v3 cloud pilot has not frozen optimizer updates per epoch for this scale")
    if frozen_updates is not None and args.updates_per_epoch != frozen_updates:
        parser.error("Requested updates differ from frozen v3 scale protocol")
    if frozen_updates is None and not args.dry_run:
        parser.error("Formal training blocked: v3 cloud pilot has not frozen per-epoch updates")
    if TASK["status"] != "frozen_formal" and not args.dry_run:
        parser.error("Formal training blocked: task_spec.json is still a pilot candidate")
    if (not args.dry_run and json.loads(
            (ROOT / "revision_v3/sensitivity_spec.json").read_text(encoding="utf-8")
            ).get("status") != "frozen_formal"):
        parser.error("Formal training blocked: sensitivity matrix is not frozen")
    if not args.dry_run and args.variant not in TASK["formal_variant_allowlist"]:
        parser.error("Variant is not part of the frozen v3 formal comparison protocol")
    if not args.dry_run:
        nonlocal_weight = TASK.get("speed_calibration_nonlocal_weight")
        if (isinstance(nonlocal_weight, bool)
                or not isinstance(nonlocal_weight, (int, float))
                or not math.isfinite(float(nonlocal_weight))
                or not 1.0 <= float(nonlocal_weight) <= 20.0):
            parser.error("Formal speed calibration weight must be frozen before any 100-epoch run")
    if not args.dry_run:
        from scripts.audit_v3_formal_speed_protocol import audit as audit_speed_protocol
        if audit_speed_protocol()["status"] != "passed":
            parser.error("Formal speed architecture/supervision parity gate failed")
    if not args.dry_run and (
        args.lr != float(TASK["training_learning_rate"])
        or args.entropy_start != float(TASK["training_entropy_start"])
        or args.entropy_end != float(TASK["training_entropy_end"])
        or args.save_every != int(TASK["training_checkpoint_every_epochs"])
    ):
        parser.error("Formal optimizer/checkpoint settings differ from frozen v3 protocol")
    if not args.dry_run:
        n_key = str(args.n_nodes)
        expected_effective = TASK["training_effective_batch"]
        expected_val_chunk = TASK["validation_chunk_by_n"][n_key]
        if (expected_effective is None
                or args.microbatch not in microbatch_choices(TASK, args.n_nodes)
                or args.microbatch*args.grad_accum_steps != expected_effective):
            parser.error("Formal microbatch/accumulation differs from the predeclared v3 choices")
        if expected_val_chunk is None or args.val_chunk != expected_val_chunk:
            parser.error("Formal validation chunk differs from frozen v3 protocol")
        if args.hardware_profile is None or not args.hardware_profile.is_file():
            parser.error("Formal training requires a locked scale hardware profile")
        current_source = {name: digest(ROOT / name) for name in SOURCE_FILES}
        validate_lock(
            args.hardware_profile, task=TASK, task_hash=digest(SPEC_PATH),
            dataset_manifest_hash=digest(args.instances / "manifest.json"),
            source_hashes=current_source, n_nodes=args.n_nodes)
    if args.updates_per_epoch < 1:
        parser.error("Updates per epoch must be positive")
    if args.pref_mode == "fixed" and args.fixed_alpha not in TASK["preference_fixed_training_values"]:
        parser.error("Fixed preference is pre-registered at 0.1, 0.5 or 0.9")
    validation_preferences = checkpoint_validation_preferences(args.pref_mode, args.fixed_alpha)
    manifest, entries = verify_frozen(args.instances, args.n_nodes)
    validation_every = int(TASK["validation_every_epochs"])
    if validation_every < 1:
        parser.error("Validation cadence must be positive")
    args.pomo = args.n_nodes if args.pomo is None else args.pomo
    if args.pomo < 1 or args.pomo > args.n_nodes:
        parser.error("POMO starts must be between 1 and n_nodes")
    if not args.dry_run and args.pomo != args.n_nodes:
        parser.error("Formal v3 protocol uses N POMO slots balanced across legal initial loads")
    plan = {"epochs": TASK["training_epochs"], "updates_per_epoch": args.updates_per_epoch,
            "microbatch": args.microbatch, "accumulation": args.grad_accum_steps,
            "effective_batch": args.microbatch*args.grad_accum_steps,
            "pomo": args.pomo, "objective_variant": args.objective_variant,
            "pomo_start_rule": TASK["training_pomo_starts"],
            "energy_scenario": args.energy_scenario,
            "speed_aux_weight": (float(TASK["training_speed_aux_weight"])
                                 if VARIANTS[args.variant].get("use_speed_aux", False) else 0.0),
            "speed_head_lr_multiplier": float(TASK["training_speed_head_lr_multiplier"]),
            "validation_every_epochs": validation_every,
            "validation_pref": validation_preferences,
            "training_selection_penalties": TASK["training_selection_penalties"],
            "task_spec_sha256": digest(SPEC_PATH),
            "dataset_manifest_sha256": digest(args.instances / "manifest.json"),
            "hardware_profile_sha256": (digest(args.hardware_profile)
                                        if args.hardware_profile is not None
                                        and args.hardware_profile.is_file() else None)}
    print(json.dumps(plan, indent=2))
    if args.dry_run:
        return
    if (not torch.cuda.is_available()
            or torch.cuda.get_device_capability()[0] < 8
            or not torch.cuda.is_bf16_supported()):
        raise RuntimeError(
            "Formal v3 training requires native BF16 Tensor Core support "
            "(NVIDIA compute capability >= 8.0)")
    if TASK["training_epochs"] != 100:
        raise RuntimeError("Formal protocol requires 100 epochs")
    device = torch.device("cuda")
    if args.resume:
        if not args.output_dir.is_dir() or not args.resume.is_file():
            raise FileNotFoundError("Resume directory/checkpoint missing")
    else:
        args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    generator = torch.Generator(device="cpu").manual_seed(args.seed + 900000)
    model = V3Policy(**VARIANTS[args.variant]).to(device)
    optimizer = torch.optim.Adam(model.optimizer_parameter_groups(
        args.lr, float(TASK["training_speed_head_lr_multiplier"])))
    history = []
    first_epoch = 1
    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=True)
        if state["task_spec_sha256"] != digest(SPEC_PATH):
            raise ValueError("Cannot resume under a different task specification")
        if state.get("hardware_profile_sha256") != digest(args.hardware_profile):
            raise ValueError("Cannot resume after changing the scale hardware lock")
        current_source = {name: digest(ROOT / name) for name in SOURCE_FILES}
        if state.get("source_sha256") != current_source:
            raise ValueError("Cannot resume after changing v3 training or physics source")
        old = state["config"]
        for key in ("variant", "n_nodes", "seed", "microbatch", "grad_accum_steps",
                    "updates_per_epoch", "pomo", "pref_mode", "fixed_alpha",
                    "objective_variant", "energy_scenario", "lr",
                    "entropy_start", "entropy_end", "val_chunk", "save_every",
                    "hardware_profile"):
            current = getattr(args, key)
            if old[key] != (str(current) if isinstance(current, Path) else current):
                raise ValueError(f"Resume configuration changed: {key}")
        reconcile_best_checkpoint(args.resume, args.output_dir / "v3_best.pt",
                                  state)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        history = state["history"]
        torch.set_rng_state(state["cpu_rng"].cpu())
        torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda_rng"]])
        generator.set_state(state["data_rng"].cpu())
        first_epoch = state["epoch"] + 1
        if first_epoch > 101:
            raise ValueError("Checkpoint is beyond the 100-epoch protocol")
        metrics_path = args.output_dir / "metrics.jsonl"
        if metrics_path.exists():
            archived = args.output_dir / (
                f"metrics.before_resume_epoch{state['epoch']}_"
                f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}.jsonl"
            )
            shutil.copy2(metrics_path, archived)
        metrics_path.write_text(
            "".join(json.dumps(row) + "\n" for row in history), encoding="utf-8"
        )
    else:
        record = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
                  "plan": plan, "args": vars(args), "dataset_manifest": manifest,
                  "source_sha256": {x: digest(ROOT / x) for x in SOURCE_FILES},
                  "environment": {"python": platform.python_version(),
                                  "torch": torch.__version__, "cuda": torch.version.cuda,
                                  "gpu": torch.cuda.get_device_name(0)}}
        (args.output_dir / "run_manifest.json").write_text(
            json.dumps(record, indent=2, default=str) + "\n", encoding="utf-8")
    prior_elapsed_s = float(history[-1]["elapsed_s"]) if history else 0.0
    start = time.perf_counter()
    best = min((r["validation_selection"] for r in history
                if r["validation_selection"] is not None), default=float("inf"))
    for epoch in range(first_epoch, 101):
        model.train()
        losses, feasibilities, limit_hits, speed_aux_losses = [], [], [], []
        coefficient = args.entropy_start + (args.entropy_end-args.entropy_start)*(epoch-1)/99
        for update in range(args.updates_per_epoch):
            optimizer.zero_grad(set_to_none=True)
            for micro in range(args.grad_accum_steps):
                kind = ("R", "C", "RC")[(update*args.grad_accum_steps+micro) % 3]
                nodes, depot, _ = generate(args.microbatch, args.n_nodes, kind,
                    TASK["baseline_tightness"], generator=generator, device=device)
                pref = preferences(args.microbatch, generator, mode=args.pref_mode,
                                   fixed_alpha=args.fixed_alpha).to(device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    output = model.rollout(nodes, depot, pref, pomo=args.pomo,
                                           objective_variant=args.objective_variant,
                                           energy_scenario=energy_scenario)
                    costs = output["selection_cost"]
                    baseline = costs.mean(dim=1, keepdim=True)
                    advantage = (costs-baseline).detach()
                    scale = advantage.std(unbiased=False).detach().clamp_min(1.0)
                    loss = ((advantage/scale)*output["logprob_sum"]).mean()
                    loss = loss - coefficient*output["entropy"].mean()
                    if model.use_speed_aux:
                        loss = loss + float(TASK["training_speed_aux_weight"])*output["speed_aux_loss"]
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError(
                        f"Nonfinite training loss at epoch {epoch}, update {update+1}, "
                        f"microbatch {micro+1}")
                (loss/args.grad_accum_steps).backward()
                losses.append(float(loss.detach()))
                feasibilities.append(float(output["feasible"].float().mean().detach()))
                limit_hits.append(float(output["rollout_limit_hit"].float().mean().detach()))
                speed_aux_losses.append(float(output["speed_aux_loss"].detach()))
            grad_norms = model.clip_gradients_(1.0)
            if not all(math.isfinite(value) for value in grad_norms.values()):
                raise FloatingPointError(
                    f"Nonfinite gradient at epoch {epoch}, update {update+1}")
            optimizer.step()
        if epoch % validation_every == 0 or epoch == 100:
            selection, validation = validate(model, entries, args.instances,
                                             n_pomo=args.pomo, chunk=args.val_chunk,
                                             device=device,
                                             objective_variant=args.objective_variant,
                                             energy_scenario=energy_scenario,
                                             validation_preferences=validation_preferences)
        else:
            selection, validation = None, []
        if selection is not None and not math.isfinite(selection):
            raise FloatingPointError(f"Nonfinite validation score at epoch {epoch}")
        row = {"epoch": epoch, "mean_train_loss": sum(losses)/len(losses),
               "train_feasible_rate": sum(feasibilities)/len(feasibilities),
               "train_rollout_limit_hit_rate": sum(limit_hits)/len(limit_hits),
               "train_speed_aux_loss": sum(speed_aux_losses)/len(speed_aux_losses),
               "validation_selection": selection, "validation": validation,
               "elapsed_s": prior_elapsed_s + time.perf_counter()-start}
        history.append(row)
        with (args.output_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        if epoch % args.save_every == 0 or epoch == 100:
            save_checkpoint(args.output_dir / "v3_latest.pt", epoch, model, optimizer,
                            history, args, generator)
        if selection is not None and selection < best:
            best = selection
            save_checkpoint(args.output_dir / "v3_best.pt", epoch, model, optimizer,
                            history, args, generator)
        print(json.dumps({"epoch": epoch, "validation_selection": selection,
                          "train_feasible_rate": row["train_feasible_rate"],
                          "elapsed_s": row["elapsed_s"]}), flush=True)
    needs_calibration = bool(model.use_factorized_speed and model.use_speed_aux
                             and model.use_decoupled_route_speed)
    summary = {"status": "completed", "epochs": 100,
               "next_required_stage": ("training_only_speed_calibration"
                                       if needs_calibration else None),
               "best_validation_selection": best,
               "duration_s": prior_elapsed_s + time.perf_counter()-start,
               "best_checkpoint_sha256": digest(args.output_dir / "v3_best.pt"),
               "last_checkpoint_sha256": digest(args.output_dir / "v3_latest.pt")}
    (args.output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    run_manifest_path = args.output_dir / "run_manifest.json"
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    run_manifest["status"] = ("route_training_completed_calibration_pending"
                              if needs_calibration else "completed")
    run_manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
    run_manifest["training_summary_sha256"] = digest(args.output_dir / "training_summary.json")
    temp_manifest = run_manifest_path.with_suffix(".json.tmp")
    temp_manifest.write_text(json.dumps(run_manifest, indent=2, default=str) + "\n",
                             encoding="utf-8")
    os.replace(temp_manifest, run_manifest_path)


if __name__ == "__main__":
    main()
