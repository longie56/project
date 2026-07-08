from pathlib import Path

import torch

from config import DEVICE


def load_solomon_instance(path, device=DEVICE):
    path = Path(path)
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 7 and parts[0].isdigit():
                rows.append([float(x) for x in parts])

    if not rows:
        raise ValueError(f"Failed to parse Solomon instance: {path}")

    tensor = torch.tensor(rows, dtype=torch.float32, device=device)
    coords = tensor[:, 1:3]
    demands = tensor[:, 3]
    ready = tensor[:, 4]
    due = tensor[:, 5]
    service = tensor[:, 6]

    depot = coords[0:1]
    customers = coords[1:]
    capacity = _read_capacity(path)

    coord_scale = coords.max().clamp_min(1.0)
    time_scale = due.max().clamp_min(1.0)
    demand_scale = max(float(capacity), 1.0)

    nodes = torch.stack(
        [
            customers[:, 0] / coord_scale,
            customers[:, 1] / coord_scale,
            demands[1:] / demand_scale,
            ready[1:] / time_scale,
            due[1:] / time_scale,
            service[1:] / time_scale,
        ],
        dim=1,
    ).unsqueeze(0)

    return {
        "name": path.stem,
        "nodes": nodes,
        "depot": (depot / coord_scale).squeeze(0).unsqueeze(0),
        "visited_mask": torch.zeros(1, nodes.size(1), dtype=torch.bool, device=device),
        "capacity": capacity,
        "coord_scale": float(coord_scale.item()),
        "time_scale": float(time_scale.item()),
    }


def load_solomon_dataset(instance_names, root=None, device=DEVICE):
    root = Path(root) if root is not None else Path(__file__).resolve().parent / "solomon"
    dataset = []
    for name in instance_names:
        filename = name if str(name).endswith(".txt") else f"{name}.txt"
        dataset.append(load_solomon_instance(root / filename, device=device))
    return dataset


def _read_capacity(path):
    with Path(path).open("r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    for idx, line in enumerate(lines):
        if line.startswith("NUMBER") and idx + 1 < len(lines):
            parts = lines[idx + 1].split()
            if len(parts) >= 2:
                return float(parts[1])
    raise ValueError(f"Failed to read vehicle capacity from {path}")