"""Time the exact formal v3 optimizer-update body without freezing a dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import torch

from env.revision_v3_gu_physics import SCENARIOS
from revision_v3.instances import generate
from revision_v3.policy import V3Policy, VARIANTS
from revision_v3.spec import ROOT, SPEC_PATH, TASK
from revision_v3.train import preferences


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-nodes", type=int, choices=(20, 50, 100), required=True)
    ap.add_argument("--variant", choices=VARIANTS, required=True)
    ap.add_argument("--microbatch", type=int, required=True)
    ap.add_argument("--grad-accum-steps", type=int, required=True)
    ap.add_argument("--updates", type=int, default=3)
    ap.add_argument("--objective-variant", choices=("C0", "C1"), default="C0")
    ap.add_argument("--energy-scenario", choices=SCENARIOS, default="gu_nominal")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.microbatch < 1 or args.grad_accum_steps < 1 or args.updates < 2:
        ap.error("Microbatch/accumulation must be positive and at least 2 updates are required")
    if (not torch.cuda.is_available()
            or torch.cuda.get_device_capability()[0] < 8
            or not torch.cuda.is_bf16_supported()):
        raise RuntimeError("Native BF16 CUDA (compute capability >= 8.0) is required")
    torch.manual_seed(96123+args.n_nodes)
    torch.cuda.manual_seed_all(96123+args.n_nodes)
    generator = torch.Generator(device="cpu").manual_seed(996123+args.n_nodes)
    device = torch.device("cuda")
    model = V3Policy(**VARIANTS[args.variant]).to(device)
    optimizer = torch.optim.Adam(model.optimizer_parameter_groups(
        1e-4, float(TASK["training_speed_head_lr_multiplier"])))
    rows = []
    torch.cuda.reset_peak_memory_stats()
    for update in range(args.updates):
        torch.cuda.synchronize()
        begun = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        feasibilities = []
        for micro in range(args.grad_accum_steps):
            kind = ("R", "C", "RC")[(update*args.grad_accum_steps+micro) % 3]
            nodes, depot, _ = generate(args.microbatch, args.n_nodes, kind,
                                        TASK["baseline_tightness"],
                                        generator=generator, device=device)
            pref = preferences(args.microbatch, generator, mode="random").to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model.rollout(
                    nodes, depot, pref, pomo=args.n_nodes,
                    objective_variant=args.objective_variant,
                    energy_scenario=SCENARIOS[args.energy_scenario])
                costs = output["selection_cost"]
                baseline = costs.mean(dim=1, keepdim=True)
                advantage = (costs-baseline).detach()
                scale = advantage.std(unbiased=False).detach().clamp_min(1.0)
                loss = ((advantage/scale)*output["logprob_sum"]).mean()
                loss = loss - 0.02*output["entropy"].mean()
                if model.use_speed_aux:
                    loss = loss + float(TASK["training_speed_aux_weight"])*output["speed_aux_loss"]
            (loss/args.grad_accum_steps).backward()
            feasibilities.append(float(output["feasible"].float().mean().detach()))
        model.clip_gradients_(1.0)
        optimizer.step()
        torch.cuda.synchronize()
        row = {"update": update+1, "seconds": time.perf_counter()-begun,
               "training_rollout_feasible_rate": statistics.mean(feasibilities)}
        rows.append(row)
        print(json.dumps(row), flush=True)
    payload = {
        "purpose": "pre-freeze exact-formal-update timing, not training result",
        "n_nodes": args.n_nodes, "variant": args.variant,
        "objective_variant": args.objective_variant,
        "energy_scenario": args.energy_scenario,
        "microbatch": args.microbatch, "grad_accum_steps": args.grad_accum_steps,
        "effective_batch": args.microbatch*args.grad_accum_steps,
        "seconds_per_update_excluding_first": statistics.mean(x["seconds"] for x in rows[1:]),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "task_spec_sha256": hashlib.sha256(SPEC_PATH.read_bytes()).hexdigest(),
        "code_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                        for name in ("env/revision_v3_gu_physics.py",
                                     "revision_v3/environment.py", "revision_v3/instances.py",
                                     "revision_v3/policy.py",
                                     "revision_v3/train.py", "revision_v3/update_pilot.py")},
        "gpu": torch.cuda.get_device_name(0),
        "gpu_total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        "gpu_compute_capability": torch.cuda.get_device_capability(0),
        "torch": torch.__version__,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2)+"\n", encoding="utf-8")
    print("Saved", args.output)


if __name__ == "__main__":
    main()
