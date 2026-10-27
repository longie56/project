"""Cloud resource pilot for the Gu-v3 graph, before fixing 100-epoch steps.

This writes measured memory, time and feasibility only. It cannot edit the
formal task spec or launch full training, so the author can review the pilot.
"""

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


def measure(n_nodes, microbatch, *, pomo, steps, generator, variant,
            energy_scenario):
    device = torch.device("cuda")
    model = V3Policy(**VARIANTS[variant]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    step_seconds = []
    feasible = []
    unserved = []
    for step in range(steps):
        torch.cuda.synchronize()
        step_start = time.perf_counter()
        nodes, depot, _ = generate(microbatch, n_nodes, ("R", "C", "RC")[step % 3],
                                   TASK["baseline_tightness"],
                                   generator=generator, device=device)
        alpha = 0.1 + 0.8*torch.rand(microbatch, generator=generator)
        pref = torch.stack([alpha, 1-alpha], dim=-1).to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            result = model.rollout(nodes, depot, pref, pomo=pomo,
                                   energy_scenario=energy_scenario)
            cost = result["selection_cost"]
            advantage = (cost-cost.mean(1, keepdim=True)).detach()
            loss = (advantage*result["logprob_sum"]).mean()
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        step_seconds.append(time.perf_counter()-step_start)
        feasible.append(float(result["feasible"].float().mean()))
        unserved.append(float(result["unserved"].float().mean()))
    with torch.no_grad():
        nodes, depot, _ = generate(1, n_nodes, "RC", TASK["baseline_tightness"],
                                   generator=generator, device=device)
        pref = torch.tensor([[0.5, 0.5]], device=device)
        val_start = time.perf_counter()
        validation = model.rollout(nodes, depot, pref, pomo=pomo, greedy=True,
                                   energy_scenario=energy_scenario)
        torch.cuda.synchronize()
        val_seconds = time.perf_counter()-val_start
    return {"n_nodes": n_nodes, "microbatch": microbatch, "pomo": pomo,
            "variant": variant, "energy_scenario": energy_scenario.id,
            "steps": steps,
            "seconds_per_update": statistics.mean(step_seconds[1:] if steps > 1 else step_seconds),
            "seconds_per_update_with_first_step": statistics.mean(step_seconds),
            "per_step_seconds": step_seconds,
            "seconds_per_validation_instance": val_seconds,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "pilot_feasible_rate": sum(feasible)/len(feasible),
            "pilot_mean_unserved": sum(unserved)/len(unserved),
            "validation_candidate_feasible": bool(validation["feasible"].any()),
            "status": "passed"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-nodes", type=int, choices=(20, 50, 100), required=True)
    parser.add_argument("--variant", choices=VARIANTS, default="full")
    parser.add_argument("--energy-scenario", choices=SCENARIOS, default="gu_nominal")
    parser.add_argument("--microbatches", type=int, nargs="+", default=(1, 2, 4, 8, 16, 32))
    parser.add_argument("--pomo", type=int)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("A BF16 CUDA GPU is required for the v3 resource pilot")
    if args.steps < 1 or any(x < 1 for x in args.microbatches):
        parser.error("Steps and microbatches must be positive")
    if args.output.exists():
        raise FileExistsError("Pilot output exists; use a fresh path")
    rows = []
    for microbatch in args.microbatches:
        torch.manual_seed(987123+args.n_nodes)
        torch.cuda.manual_seed_all(987123+args.n_nodes)
        generator = torch.Generator(device="cpu").manual_seed(987123+args.n_nodes)
        try:
            row = measure(args.n_nodes, microbatch,
                          pomo=args.n_nodes if args.pomo is None else args.pomo,
                          steps=args.steps, generator=generator,
                          variant=args.variant,
                          energy_scenario=SCENARIOS[args.energy_scenario])
        except torch.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            row = {"n_nodes": args.n_nodes, "microbatch": microbatch,
                   "status": "cuda_oom", "error": str(exc).split("\n")[0]}
        rows.append(row)
        print(json.dumps(row), flush=True)
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    payload = {"task_spec_sha256": digest(SPEC_PATH),
               "code_sha256": {name: digest(ROOT / name) for name in (
                   "env/revision_v3_gu_physics.py", "revision_v3/environment.py",
                   "revision_v3/policy.py", "revision_v3/pilot.py")},
               "variant": args.variant,
               "energy_scenario": args.energy_scenario,
               "gpu": torch.cuda.get_device_name(0),
               "compute_capability": torch.cuda.get_device_capability(0),
               "total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
               "torch": torch.__version__, "cuda": torch.version.cuda,
               "formal_training_epochs": 100,
               "formal_updates_per_epoch": TASK["optimizer_updates_per_epoch_by_n"][str(args.n_nodes)],
               "measurements": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("Saved", args.output)


if __name__ == "__main__":
    main()
