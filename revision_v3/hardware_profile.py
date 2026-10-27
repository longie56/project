"""Predeclared microbatch choices and one immutable choice per formal scale."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def choices(task: dict, n_nodes: int) -> tuple[int, ...]:
    key = str(n_nodes)
    effective = task.get("training_effective_batch")
    default = task["training_microbatch_by_n"][key]
    accumulation = task["training_grad_accum_steps_by_n"][key]
    if effective is None or default is None or accumulation is None:
        raise ValueError("Formal effective batch and fallback microbatch are not frozen")
    effective, default, accumulation = map(int, (effective, default, accumulation))
    if effective < 1 or default < 1 or default*accumulation != effective:
        raise ValueError("Frozen fallback microbatch/accumulation is inconsistent")
    declared = (task.get("training_microbatch_candidates_by_n") or {}).get(key)
    options = [default] if declared is None else declared
    if (not isinstance(options, list) or not options
            or any(not isinstance(value, int) or value < 1
                   or effective % value for value in options)
            or len(set(options)) != len(options) or default not in options):
        raise ValueError("Formal microbatch candidates must divide effective batch")
    return tuple(sorted(options))


def validate_lock(path: Path, *, task: dict, task_hash: str,
                  dataset_manifest_hash: str, source_hashes: dict,
                  n_nodes: int) -> dict:
    profile = json.loads(path.read_text(encoding="utf-8"))
    microbatch = profile.get("microbatch")
    effective = task["training_effective_batch"]
    allowed = choices(task, n_nodes)
    if (profile.get("n_nodes") != n_nodes
            or profile.get("task_spec_sha256") != task_hash
            or profile.get("dataset_manifest_sha256") != dataset_manifest_hash
            or profile.get("source_sha256") != source_hashes
            or profile.get("effective_batch") != effective
            or microbatch not in allowed
            or profile.get("grad_accum_steps") != effective // microbatch):
        raise ValueError("Formal scale hardware lock differs from frozen protocol")
    # A profile selected on one GPU model must not silently authorize a run on
    # another card, even when both cards can execute the chosen microbatch.
    import torch
    if torch.cuda.is_available() and (
            profile.get("gpu") != torch.cuda.get_device_name(0)
            or tuple(profile.get("gpu_compute_capability", ()))
            != tuple(torch.cuda.get_device_capability(0))
            or profile.get("gpu_total_memory_bytes")
            != torch.cuda.get_device_properties(0).total_memory):
        raise ValueError("Current GPU differs from the locked formal scale hardware")
    if len(allowed) > 1:
        admission_path = path.with_name(path.name.replace(
            "_hardware_profile.json", "_hardware_admission.json"))
        if (not admission_path.is_file()
                or profile.get("admission_sha256") != digest(admission_path)):
            raise ValueError("Multi-option scale lock lacks its immutable admission evidence")
        admission = json.loads(admission_path.read_text(encoding="utf-8"))
        if (admission.get("status") != "admitted"
                or admission.get("selected_microbatch") != microbatch
                or admission.get("task_spec_sha256") != task_hash
                or admission.get("dataset_manifest_sha256") != dataset_manifest_hash
                or admission.get("source_sha256") != source_hashes):
            raise ValueError("Hardware admission differs from the locked formal protocol")
    return profile
