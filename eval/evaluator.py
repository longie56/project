import time

import torch

from config import DEVICE
from data.augmentation import augment_instance_by_8_fold
from eval.metrics import summarize_instances


def _select_augmented_best(result, aug_factor, base_batch_size):
    batch_idx = torch.arange(base_batch_size, device=result["best_cost"].device)
    best_cost = result["best_cost"].reshape(aug_factor, base_batch_size)
    best_aug_idx = best_cost.argmin(dim=0)

    selected = {}
    for key, value in result.items():
        if not torch.is_tensor(value):
            selected[key] = value
            continue
        if value.dim() == 1 and value.size(0) == aug_factor * base_batch_size:
            reshaped = value.reshape(aug_factor, base_batch_size)
            selected[key] = reshaped[best_aug_idx, batch_idx]
        elif value.dim() >= 2 and value.size(0) == aug_factor * base_batch_size:
            new_shape = (aug_factor, base_batch_size) + tuple(value.shape[1:])
            reshaped = value.reshape(*new_shape)
            selected[key] = reshaped[best_aug_idx, batch_idx]
        else:
            selected[key] = value
    return selected


def evaluate(model, dataset, decode_type="greedy", n_pomo=None, device=DEVICE, augmentation_enable=False):
    model.eval()
    instance_metrics = []
    with torch.no_grad():
        for instance in dataset:
            eval_instance = augment_instance_by_8_fold(instance) if augmentation_enable else instance

            nodes = eval_instance["nodes"].to(device)
            instance_n_pomo = nodes.size(1) if n_pomo is None else n_pomo
            depot = eval_instance["depot"].to(device)
            visited_mask = eval_instance.get("visited_mask")
            if visited_mask is not None:
                visited_mask = visited_mask.to(device)
            pref = eval_instance.get("pref")
            if pref is None:
                pref = torch.tensor([[0.5, 0.5]], dtype=torch.float32, device=device).expand(nodes.size(0), -1)
            else:
                pref = pref.to(device)

            t0 = time.perf_counter()
            result = model.inference_rollout(
                nodes,
                pref,
                depot,
                visited_mask=visited_mask,
                N_pomo=instance_n_pomo,
                decode_type=decode_type,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            if augmentation_enable:
                result = _select_augmented_best(
                    result,
                    aug_factor=eval_instance["aug_factor"],
                    base_batch_size=eval_instance["base_batch_size"],
                )

            instance_metrics.append(
                {
                    "name": instance.get("name", "unknown"),
                    "cost": float(result["best_cost"].mean().item()),
                    "phys": float(result["best_phys"].mean().item()),
                    "pen": float(result["best_pen"].mean().item()),
                    "unvisit": float(result["best_unvisit"].mean().item()),
                    "feasible": float(result["feasible"].float().mean().item()),
                    "elapsed_ms": elapsed_ms,
                }
            )
    return summarize_instances(instance_metrics), instance_metrics
