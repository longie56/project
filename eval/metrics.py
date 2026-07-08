import torch


def summarize_batch(result):
    return {
        "avg_cost": float(result["best_cost"].mean().item()),
        "avg_phys": float(result["best_phys"].mean().item()),
        "avg_pen": float(result["best_pen"].mean().item()),
        "avg_unvisit": float(result["best_unvisit"].mean().item()),
        "feasible_rate": float(result["feasible"].float().mean().item()),
    }


def summarize_instances(instance_metrics):
    if not instance_metrics:
        return {}
    keys = ["cost", "phys", "pen", "unvisit", "feasible", "elapsed_ms"]
    summary = {}
    for key in keys:
        values = torch.tensor([m[key] for m in instance_metrics], dtype=torch.float32)
        summary[f"avg_{key}"] = float(values.mean().item())
    return summary
