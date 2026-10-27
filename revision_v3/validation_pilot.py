"""Measure batched v3 validation throughput on pilot-only synthetic inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import time
from pathlib import Path

import torch

from env.revision_v3_gu_physics import SCENARIOS
from revision_v3.instances import generate
from revision_v3.policy import V3Policy, VARIANTS
from revision_v3.spec import ROOT, SPEC_PATH, TASK
from revision_v3.train import checkpoint_validation_preferences, validate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-nodes", type=int, choices=(20, 50, 100), required=True)
    ap.add_argument("--variant", choices=VARIANTS, default="full")
    ap.add_argument("--chunks", type=int, nargs="+", default=(1, 2, 4, 8, 16, 32))
    ap.add_argument("--full-sweep", action="store_true",
                    help="Time train.validate over synthetic R/C/RC sets and all checkpoint-selection preferences")
    ap.add_argument("--objective-variant", choices=("C0", "C1"), default="C0")
    ap.add_argument("--energy-scenario", choices=SCENARIOS, default="gu_nominal")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.full_sweep and len(args.chunks) != 1:
        ap.error("A full validation sweep requires exactly one selected chunk")
    if args.output.exists():
        raise FileExistsError(args.output)
    if (not torch.cuda.is_available()
            or torch.cuda.get_device_capability()[0] < 8
            or not torch.cuda.is_bf16_supported()):
        raise RuntimeError("Native BF16 CUDA (compute capability >= 8.0) is required")
    torch.manual_seed(78123+args.n_nodes)
    torch.cuda.manual_seed_all(78123+args.n_nodes)
    device = torch.device("cuda")
    model = V3Policy(**VARIANTS[args.variant]).to(device).eval()
    rows = []
    if args.full_sweep:
        chunk = args.chunks[0]
        if chunk < 1:
            ap.error("Validation chunk must be positive")
        count = int(TASK["validation_per_n_and_type"])
        preferences = checkpoint_validation_preferences("random", 0.5)
        with tempfile.TemporaryDirectory(prefix="tre_v3_validation_pilot_") as tmp:
            directory = Path(tmp)
            entries = []
            generator = torch.Generator(device="cpu").manual_seed(520000+args.n_nodes)
            for kind in ("R", "C", "RC"):
                nodes, depot, _ = generate(count, args.n_nodes, kind,
                                           TASK["baseline_tightness"],
                                           generator=generator, device=torch.device("cpu"))
                filename = f"pilot_{kind}.pt"
                torch.save({"nodes": nodes, "depot": depot}, directory / filename)
                entries.append({"kind": kind, "count": count, "file": filename})
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            begun = time.perf_counter()
            try:
                selection, validation_rows = validate(
                    model, entries, directory, n_pomo=args.n_nodes, chunk=chunk,
                    device=device, objective_variant=args.objective_variant,
                    energy_scenario=SCENARIOS[args.energy_scenario],
                    validation_preferences=preferences)
                torch.cuda.synchronize()
                rows.append({
                    "chunk": chunk, "status": "passed", "seconds_full_sweep": time.perf_counter()-begun,
                    "instance_preference_evaluations": len(entries)*count*len(preferences),
                    "validation_row_count": len(validation_rows),
                    "selection_finite": bool(torch.isfinite(torch.tensor(selection))),
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                })
            except torch.OutOfMemoryError as exc:
                rows.append({"chunk": chunk, "status": "cuda_oom",
                             "error": str(exc).split("\n")[0]})
                torch.cuda.empty_cache()
    for chunk in ([] if args.full_sweep else args.chunks):
        if chunk < 1:
            ap.error("All chunks must be positive")
        generator = torch.Generator(device="cpu").manual_seed(520000+args.n_nodes)
        nodes, depot, _ = generate(chunk, args.n_nodes, "RC",
                                   TASK["baseline_tightness"], generator=generator,
                                   device=device)
        pref = torch.tensor([[0.5, 0.5]], device=device).expand(chunk, 2)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        begun = time.perf_counter()
        try:
            # train.validate() uses no_grad() without autocast; time that exact
            # path so the 100-epoch validation budget is not understated.
            with torch.no_grad():
                result = model.rollout(nodes, depot, pref, pomo=args.n_nodes,
                                       greedy=True,
                                       objective_variant=args.objective_variant,
                                       energy_scenario=SCENARIOS[args.energy_scenario])
            torch.cuda.synchronize()
            elapsed = time.perf_counter()-begun
            row = {"chunk": chunk, "status": "passed",
                   "seconds_per_chunk": elapsed,
                   "seconds_per_instance": elapsed/chunk,
                   "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                   "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                   "any_feasible_rate": float(result["feasible"].any(dim=1).float().mean())}
            del result
        except torch.OutOfMemoryError as exc:
            row = {"chunk": chunk, "status": "cuda_oom",
                   "error": str(exc).split("\n")[0]}
            torch.cuda.empty_cache()
        rows.append(row)
        print(json.dumps(row), flush=True)
    payload = {"purpose": ("pilot-only complete checkpoint-validation timing"
                           if args.full_sweep else "pilot-only validation batching"),
               "n_nodes": args.n_nodes,
               "variant": args.variant,
               "objective_variant": args.objective_variant,
               "energy_scenario": args.energy_scenario,
               "task_spec_sha256": hashlib.sha256(SPEC_PATH.read_bytes()).hexdigest(),
               "code_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                               for name in ("env/revision_v3_gu_physics.py",
                                            "revision_v3/environment.py",
                                            "revision_v3/instances.py",
                                            "revision_v3/policy.py",
                                            "revision_v3/train.py",
                                            "revision_v3/validation_pilot.py")},
               "gpu": torch.cuda.get_device_name(0),
               "gpu_total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
               "gpu_compute_capability": torch.cuda.get_device_capability(0),
               "torch": torch.__version__,
               "measurements": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2)+"\n", encoding="utf-8")
    print("Saved", args.output)


if __name__ == "__main__":
    main()
