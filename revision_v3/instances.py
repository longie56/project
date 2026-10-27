"""Synthetic Solomon-style v3 instances, with physical parcel masses and seconds."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from revision_v3.hardware_profile import choices as microbatch_choices
from revision_v3.spec import ROOT, SERVICE_S, SPEC_PATH, TASK, horizon_s


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate(batch: int, n_nodes: int, kind: str, tightness: float,
             *, generator: torch.Generator, device: torch.device | str = "cpu"):
    """Return nodes[B,N,6], depot[B,2] and an all-unserved mask.

    Generation is CPU seeded and transferred afterwards so CPU/CUDA instances
    and all baseline inputs share the same bytes. No Solomon files are parsed.
    """
    kind = kind.upper()
    if kind not in {"R", "C", "RC"}:
        raise ValueError("Choose R, C or RC")
    if not (0 <= tightness <= 1):
        raise ValueError("tightness must be in [0,1]")
    if batch <= 0:
        raise ValueError("batch must be positive")
    horizon = horizon_s(n_nodes)
    depot = torch.tensor(TASK["depot_xy_unit"], dtype=torch.float32).expand(batch, 2).clone()
    uniform = torch.rand(batch, n_nodes, 2, generator=generator)
    if kind == "R":
        xy = uniform
    else:
        n_clusters = 5 if n_nodes <= 20 else 8
        centers = torch.rand(batch, n_clusters, 2, generator=generator) * 0.8 + 0.1
        cluster_ids = torch.randint(n_clusters, (batch, n_nodes), generator=generator)
        selected = centers.gather(1, cluster_ids[..., None].expand(-1, -1, 2))
        clustered = (selected + 0.06 * torch.randn(batch, n_nodes, 2,
                                                   generator=generator)).clamp(0, 1)
        if kind == "C":
            xy = clustered
        else:
            split = n_nodes // 2
            xy = torch.cat([clustered[:, :split], uniform[:, split:]], dim=1)
    lo, hi = TASK["customer_mass_uniform_kg"]
    demand = lo + (hi - lo) * torch.rand(batch, n_nodes, generator=generator)
    rlo, rhi = TASK["ready_time_uniform_fraction_of_horizon"]
    ready = horizon * (rlo + (rhi-rlo) * torch.rand(batch, n_nodes, generator=generator))
    window = horizon * (0.08 + 0.20 * float(tightness))
    due = torch.minimum(ready + window, torch.full_like(ready, horizon))
    service = torch.full((batch, n_nodes), SERVICE_S)
    nodes = torch.stack([xy[..., 0], xy[..., 1], demand, ready, due, service], dim=-1)
    unserved = torch.ones(batch, n_nodes, dtype=torch.bool)
    return nodes.to(device), depot.to(device), unserved.to(device)


def freeze(output_dir: Path, *, tightness: float = 0.7):
    if TASK["status"] != "frozen_formal":
        raise RuntimeError("Blind test and validation may be frozen only after v3 pilot approval")
    if TASK.get("formal_greedy_strategy") != "joint":
        raise RuntimeError("Formal greedy strategy must be explicitly frozen as joint")
    sensitivity_path = ROOT / "revision_v3/sensitivity_spec.json"
    sensitivity = json.loads(sensitivity_path.read_text(encoding="utf-8"))
    if sensitivity.get("status") != "frozen_formal":
        raise RuntimeError("Sensitivity and comparison protocols must be frozen first")
    frozen_weight = TASK.get("speed_calibration_nonlocal_weight")
    if (isinstance(frozen_weight, bool)
            or not isinstance(frozen_weight, (int, float))
            or not 1.0 <= float(frozen_weight) <= 20.0):
        raise RuntimeError("Formal speed calibration weight must be explicitly frozen")
    from revision_v3.policy import VARIANTS
    shared_prior = TASK.get("formal_shared_wait_prior")
    if (not isinstance(shared_prior, dict)
            or set(shared_prior) != {"wait_urgency_coefficient",
                                     "wait_slack_coefficient",
                                     "wait_slack_buffer_fraction"}):
        raise RuntimeError("The common formal wait prior must be declared")
    for variant in TASK["formal_variant_allowlist"]:
        config = VARIANTS.get(variant)
        if config is None or any(float(config.get(key, 0.10 if key == "wait_slack_buffer_fraction" else 0.0))
                                 != float(value) for key, value in shared_prior.items()):
            raise RuntimeError(f"Formal variant {variant} lacks the shared wait prior")
    if any(TASK["optimizer_updates_per_epoch_by_n"][str(n)] is None
           for n in (20, 50, 100)):
        raise RuntimeError("All three scale-specific update counts must be frozen first")
    if TASK["training_effective_batch"] is None or any(
        TASK["training_microbatch_by_n"][str(n)] is None
        or TASK["training_grad_accum_steps_by_n"][str(n)] is None
        or TASK["training_microbatch_by_n"][str(n)]
        * TASK["training_grad_accum_steps_by_n"][str(n)] != TASK["training_effective_batch"]
        for n in (20, 50, 100)
    ):
        raise RuntimeError("All three scales must share a frozen effective batch")
    if not isinstance(TASK.get("training_microbatch_candidates_by_n"), dict):
        raise RuntimeError("Formal microbatch choices must be declared before freeze")
    for n in (20, 50, 100):
        if TASK["training_microbatch_candidates_by_n"].get(str(n)) is None:
            raise RuntimeError(f"Formal N{n} microbatch choices are not frozen")
        microbatch_choices(TASK, n)
    if any(not isinstance(TASK.get("online_pomo_by_n", {}).get(str(n)), int)
           or not 1 <= TASK["online_pomo_by_n"][str(n)] <= n
           for n in (20, 50, 100)):
        raise RuntimeError("Online POMO starts must be frozen for every scale")
    if any(TASK["validation_chunk_by_n"][str(n)] is None
           for n in (20, 50, 100)):
        raise RuntimeError("Validation chunks must be fixed before dataset freeze")
    if tightness != float(TASK["baseline_tightness"]):
        raise ValueError("Frozen validation/test tightness must match the task specification")
    seed_bases = TASK.get("formal_instance_seed_bases")
    if (not isinstance(seed_bases, dict)
            or set(seed_bases) != {"val", "test"}
            or any(isinstance(value, bool) or not isinstance(value, int) or value < 0
                   for value in seed_bases.values())
            or seed_bases["val"] == seed_bases["test"]
            or any(value in (310000, 410000) for value in seed_bases.values())):
        raise RuntimeError("Formal validation/test must have distinct new seed bases")
    output_dir = output_dir.resolve()
    staging_dir = output_dir.with_name(output_dir.name + ".staging")
    if output_dir.exists():
        raise FileExistsError(f"Frozen dataset destination already exists: {output_dir}")
    if staging_dir.exists():
        raise FileExistsError(
            f"Preserved interrupted dataset staging directory requires inspection: {staging_dir}")
    staging_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "dataset_id": TASK["task_id"],
        "status": "frozen_formal_not_yet_tested",
        "task_spec_sha256": file_hash(SPEC_PATH),
        "sensitivity_spec_sha256": file_hash(sensitivity_path),
        "physics_spec_sha256": file_hash(ROOT / TASK["physics_spec"]),
        "generator_sha256": file_hash(Path(__file__)),
        "torch_version": torch.__version__,
        "description": "synthetic Solomon-style, not published Solomon instances",
        "tightness": tightness,
        "train_seeds_reserved": TASK["training_seeds"],
        "formal_instance_seed_bases": seed_bases,
        "sets": [],
    }
    for split in ("val", "test"):
        for n_nodes in (20, 50, 100):
            count = (TASK["validation_per_n_and_type"] if split == "val"
                     else TASK["test_per_n_and_type"][str(n_nodes)])
            for type_idx, kind in enumerate(("R", "C", "RC"), 1):
                seed = seed_bases[split] + n_nodes*10 + type_idx
                gen = torch.Generator(device="cpu").manual_seed(seed)
                nodes, depot, unserved = generate(count, n_nodes, kind, tightness,
                                                   generator=gen)
                path = staging_dir / f"{split}_n{n_nodes}_{kind}.pt"
                torch.save({"nodes": nodes, "depot": depot, "unserved": unserved}, path)
                manifest["sets"].append({
                    "split": split, "n_nodes": n_nodes, "kind": kind,
                    "count": count, "seed": seed, "file": path.name,
                    "sha256": file_hash(path),
                })
    (staging_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    staging_dir.replace(output_dir)
    return manifest


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--tightness", type=float, default=0.7)
    args = ap.parse_args()
    freeze(args.output_dir, tightness=args.tightness)
