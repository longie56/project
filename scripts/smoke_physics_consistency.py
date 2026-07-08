import argparse
import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DEVICE, E_MAX, E_SAFE_MARGIN, MAX_ROUTE_TIME_NORM, V_NORM
from data.solomon_generator import SolomonGenerator
from env.energy_model import compute_energy
from env.vrp_env import VRPTWEnv
from seed import set_global_seed


def check_instance(nodes, depot, pref, visited, scenario):
    env = VRPTWEnv(nodes, pref, depot, scenario=scenario)
    env.load_state(
        visited,
        depot.clone(),
        torch.zeros(1, device=nodes.device),
        torch.ones(1, device=nodes.device),
        torch.ones(1, device=nodes.device),
    )

    mask = env.get_mask()[0]
    n_nodes = nodes.size(1)
    checked = 0
    for action in torch.nonzero(mask, as_tuple=False).view(-1).tolist():
        trial = VRPTWEnv(nodes.clone(), pref.clone(), depot.clone(), scenario=scenario)
        trial.load_state(
            visited.clone(),
            depot.clone(),
            torch.zeros(1, device=nodes.device),
            torch.ones(1, device=nodes.device),
            torch.ones(1, device=nodes.device),
        )
        trial.step(torch.tensor([int(action)], device=nodes.device))
        checked += 1

        if not trial._all_visited()[0] and trial.time[0] > MAX_ROUTE_TIME_NORM + 1e-6:
            raise AssertionError(
                f"Mask allowed action {action} that exceeds route horizon before completion: "
                f"time={float(trial.time[0].item()):.6f}"
            )

        if scenario != "simple" and action < n_nodes * 2:
            e_back = compute_energy(trial.pos, trial.depot, V_NORM, trial.load)
            remaining_after_return = trial.energy * E_MAX - e_back
            if remaining_after_return[0] < E_SAFE_MARGIN * E_MAX - 1e-3:
                raise AssertionError(
                    f"Mask allowed unsafe customer action {action}: "
                    f"post_return_margin={float((remaining_after_return[0] / E_MAX).item()):.6f}"
                )
    return checked


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-nodes", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--problem-type", default="R")
    parser.add_argument("--tightness", type=float, default=0.7)
    parser.add_argument("--scenario", choices=["full", "simple"], default="full")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_global_seed(args.seed)
    nodes, depot, visited = SolomonGenerator.generate(
        args.batch_size, args.n_nodes, args.problem_type, args.tightness, DEVICE
    )
    pref = torch.rand(args.batch_size, 2, device=DEVICE)
    pref = pref / pref.sum(dim=-1, keepdim=True).clamp_min(1e-6)

    total_checked = 0
    for idx in range(args.batch_size):
        total_checked += check_instance(
            nodes[idx : idx + 1],
            depot[idx : idx + 1],
            pref[idx : idx + 1],
            visited[idx : idx + 1],
            args.scenario,
        )
    print(
        f"physics consistency smoke passed: instances={args.batch_size}, "
        f"n_nodes={args.n_nodes}, checked_actions={total_checked}, scenario={args.scenario}"
    )


if __name__ == "__main__":
    main()
