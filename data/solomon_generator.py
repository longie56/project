import random

import torch

from config import *


class SolomonGenerator:
    """
    Synthetic Solomon-style instance generator.

    This is not a parser for the published Solomon benchmark files. It creates
    randomized R/C/RC-like coordinates, demands, and time windows under the same
    physical scale used by the UAV environment, so papers/results should describe
    it as synthetic Solomon-style data unless real benchmark files are added.
    """

    @staticmethod
    def generate(batch_size, n_nodes=100, problem_type='R', tightness=0.7, device=DEVICE):
        problem_type = str(problem_type).upper()
        if problem_type in {"MIX", "MIXED", "R+C+RC", "R_C_RC", "R,C,RC", "R_C_RC_MIXED"}:
            types = ["R", "C", "RC"]
            base = batch_size // len(types)
            remainder = batch_size % len(types)
            counts = [base + (1 if idx < remainder else 0) for idx in range(len(types))]
            node_chunks, depot_chunks, visited_chunks = [], [], []
            for count, ptype in zip(counts, types):
                if count <= 0:
                    continue
                nodes, depot, visited = SolomonGenerator.generate(
                    count, n_nodes, ptype, tightness, device
                )
                node_chunks.append(nodes)
                depot_chunks.append(depot)
                visited_chunks.append(visited)
            nodes = torch.cat(node_chunks, dim=0)
            depot = torch.cat(depot_chunks, dim=0)
            visited_mask = torch.cat(visited_chunks, dim=0)
            perm = torch.randperm(nodes.size(0), device=device)
            return nodes[perm], depot[perm], visited_mask[perm]

        total_pts = n_nodes + 1
        if problem_type == 'C':
            coords = _gen_clustered(batch_size, total_pts, device)
        elif problem_type == 'RC':
            coords = _gen_mixed(batch_size, total_pts, device)
        else:
            coords = torch.rand(batch_size, total_pts, 2, device=device)

        coords[:, 0] = torch.rand(batch_size, 2, device=device) * 0.2 + 0.4
        depot = coords[:, 0]
        customers = coords[:, 1:]

        demands = torch.rand(batch_size, n_nodes, device=device) * 0.45 + 0.05
        dist_to_depot = torch.norm(customers - depot.unsqueeze(1), dim=2) * GRID_SCALE
        min_arrival = dist_to_depot / V_NORM / MAX_TIME

        service_time = BASE_SERVICE_TIME_NORM
        if n_nodes >= LARGE_SCALE_TIME_WINDOW_START_N:
            service_time = min(BASE_SERVICE_TIME_NORM, SERVICE_TOTAL_TIME_NORM / max(1, n_nodes))

        ready = torch.zeros(batch_size, n_nodes, device=device)
        due = torch.ones(batch_size, n_nodes, device=device) * 2.0
        n_tight_min = max(1, int(n_nodes * 0.4))
        for b in range(batch_size):
            n_tight = random.randint(n_tight_min, n_nodes)
            tight_idx = torch.randperm(n_nodes, device=device)[:n_tight]
            if n_nodes >= LARGE_SCALE_TIME_WINDOW_START_N:
                window = LARGE_SCALE_MIN_WINDOW_NORM + tightness * LARGE_SCALE_WINDOW_SLACK_NORM
                latest_ready = max(0.1, LARGE_SCALE_TIME_HORIZON_NORM - window - service_time)
                sampled_ready = torch.rand(n_tight, device=device) * latest_ready
                travel_floor = torch.clamp(min_arrival[b, tight_idx] * 0.8, min=0.05)
                ready[b, tight_idx] = torch.maximum(sampled_ready, travel_floor)
                due[b, tight_idx] = torch.clamp(
                    ready[b, tight_idx] + window,
                    max=LARGE_SCALE_TIME_HORIZON_NORM,
                )
            else:
                ready[b, tight_idx] = torch.clamp(min_arrival[b, tight_idx] * 0.8, min=0.1)
                due[b, tight_idx] = min_arrival[b, tight_idx] * 1.5 + 0.5 + tightness * 0.8

        service = torch.ones(batch_size, n_nodes, device=device) * service_time
        nodes = torch.stack(
            [customers[..., 0], customers[..., 1], demands, ready, due, service], dim=2
        )
        visited_mask = torch.zeros(batch_size, n_nodes, dtype=torch.bool, device=device)
        return nodes, depot, visited_mask


def _gen_clustered(batch, n, device):
    coords = torch.zeros(batch, n, 2, device=device)
    coords[:, 0] = torch.rand(batch, 2, device=device) * 0.2 + 0.4
    n_clusters = random.randint(5, 8)
    centers = torch.rand(batch, n_clusters, 2, device=device)
    for i in range(1, n):
        c = torch.randint(0, n_clusters, (batch,), device=device)
        coords[:, i] = centers[range(batch), c] + torch.randn(batch, 2, device=device) * 0.05
        coords[:, i] = coords[:, i].clamp(0, 1)
    return coords


def _gen_mixed(batch, n, device):
    coords = torch.zeros(batch, n, 2, device=device)
    coords[:, 0] = torch.rand(batch, 2, device=device) * 0.2 + 0.4
    half = (n - 1) // 2
    n_clusters = random.randint(3, 6)
    centers = torch.rand(batch, n_clusters, 2, device=device)
    for i in range(1, 1 + half):
        c = torch.randint(0, n_clusters, (batch,), device=device)
        coords[:, i] = centers[range(batch), c] + torch.randn(batch, 2, device=device) * 0.05
        coords[:, i] = coords[:, i].clamp(0, 1)
    coords[:, 1 + half :] = torch.rand(batch, n - 1 - half, 2, device=device)
    return coords
