import math
from typing import Iterable, List, Optional

import torch

from config import PEN_WEIGHT, PHYS_WEIGHT
from env.vrp_env import VRPTWEnv


def _step_incremental_cost(env: VRPTWEnv, action: int) -> float:
    return env.estimate_action_objective(int(action))


def choose_best_action(env: VRPTWEnv, candidates: Iterable[int], mode_policy: str = "greedy") -> Optional[int]:
    candidates = [int(action) for action in candidates]
    if not candidates:
        return None
    if mode_policy == "fast":
        fast_candidates = [action for action in candidates if action % 2 == 1]
        if fast_candidates:
            return fast_candidates[0]
    elif mode_policy == "normal":
        normal_candidates = [action for action in candidates if action % 2 == 0]
        if normal_candidates:
            return normal_candidates[0]

    best_action = None
    best_cost = None
    for action in candidates:
        step_cost = _step_incremental_cost(env, int(action))
        if best_cost is None or step_cost < best_cost:
            best_cost = step_cost
            best_action = int(action)
    return best_action


def evaluate_actions_with_env(
    nodes,
    pref,
    depot,
    actions,
    visited_mask=None,
    curr_pos=None,
    curr_time=None,
    curr_load=None,
    curr_energy=None,
    scenario="full",
    mode_policy="greedy",
):
    device = nodes.device
    batch_size, n_nodes, _ = nodes.shape
    if batch_size != 1:
        raise ValueError("evaluate_actions_with_env only supports batch size 1.")

    if visited_mask is None:
        visited_mask = torch.zeros(batch_size, n_nodes, dtype=torch.bool, device=device)
    if curr_pos is None:
        curr_pos = depot.clone()
    if curr_time is None:
        curr_time = torch.zeros(batch_size, device=device)
    if curr_load is None:
        curr_load = torch.ones(batch_size, device=device)
    if curr_energy is None:
        curr_energy = torch.ones(batch_size, device=device)

    env = VRPTWEnv(nodes.clone(), pref.clone(), depot.clone(), scenario=scenario)
    env.load_state(
        visited_mask.clone(),
        curr_pos.clone(),
        curr_time.clone(),
        curr_load.clone(),
        curr_energy.clone(),
    )

    executed = []
    invalid_action = None
    invalid_reason = ""
    for act in actions:
        if env._compute_done().all():
            break
        action = int(act)
        mask = env.get_mask()
        if action < 0 or action >= mask.size(1):
            invalid_action = action
            invalid_reason = "out_of_range"
            break
        if not bool(mask[0, action].item()):
            invalid_action = action
            invalid_reason = "masked_action"
            break
        env.step(torch.tensor([action], device=device))
        executed.append(action)
        if env._compute_done().all():
            break

    unvisit = (~env.visited).sum(-1).float()
    total_cost = env.phys * PHYS_WEIGHT + env.pen * PEN_WEIGHT + unvisit * 20.0
    feasible = bool(
        invalid_action is None
        and (unvisit == 0).all().item()
        and env._at_depot().all().item()
        and env._compute_done().all().item()
    )
    cost = float(total_cost.item()) if invalid_action is None else 1.0e12
    return {
        "actions": executed,
        "cost": cost,
        "phys": float(env.phys.item()),
        "pen": float(env.pen.item()),
        "unvisit": float(unvisit.item()),
        "feasible": feasible,
        "invalid_action": invalid_action,
        "invalid_reason": invalid_reason,
        "depot_return_count": sum(1 for action in executed if int(action) >= n_nodes * 2),
    }


def append_missing_actions_to_eval(
    nodes,
    pref,
    depot,
    actions,
    visited_mask=None,
    curr_pos=None,
    curr_time=None,
    curr_load=None,
    curr_energy=None,
    scenario="full",
    mode_policy="greedy",
):
    device = nodes.device
    batch_size, n_nodes, _ = nodes.shape
    if batch_size != 1:
        raise ValueError("append_missing_actions_to_eval only supports batch size 1.")

    if visited_mask is None:
        visited_mask = torch.zeros(batch_size, n_nodes, dtype=torch.bool, device=device)
    if curr_pos is None:
        curr_pos = depot.clone()
    if curr_time is None:
        curr_time = torch.zeros(batch_size, device=device)
    if curr_load is None:
        curr_load = torch.ones(batch_size, device=device)
    if curr_energy is None:
        curr_energy = torch.ones(batch_size, device=device)

    env = VRPTWEnv(nodes.clone(), pref.clone(), depot.clone(), scenario=scenario)
    env.load_state(
        visited_mask.clone(),
        curr_pos.clone(),
        curr_time.clone(),
        curr_load.clone(),
        curr_energy.clone(),
    )

    repaired_actions = []
    for action in actions:
        action = int(action)
        mask = env.get_mask()
        if 0 <= action < mask.size(1) and mask[0, action]:
            env.step(torch.tensor([action], device=device))
            repaired_actions.append(action)

    return_actions = [n_nodes * 2, n_nodes * 2 + 1]
    for node_idx in range(n_nodes):
        if env.visited[0, node_idx]:
            continue

        mask = env.get_mask()
        customer_actions = []
        normal_action = node_idx * 2
        fast_action = node_idx * 2 + 1
        if mask[0, normal_action]:
            customer_actions.append(normal_action)
        if mask[0, fast_action]:
            customer_actions.append(fast_action)

        selected_action = choose_best_action(env, customer_actions, mode_policy=mode_policy)
        if selected_action is None and not env._at_depot()[0]:
            return_action = choose_best_action(env, [a for a in return_actions if mask[0, a]], mode_policy=mode_policy)
            if return_action is not None:
                env.step(torch.tensor([return_action], device=device))
                repaired_actions.append(return_action)
                mask = env.get_mask()
                customer_actions = []
                if mask[0, normal_action]:
                    customer_actions.append(normal_action)
                if mask[0, fast_action]:
                    customer_actions.append(fast_action)
                selected_action = choose_best_action(env, customer_actions, mode_policy=mode_policy)

        if selected_action is None:
            continue

        env.step(torch.tensor([selected_action], device=device))
        repaired_actions.append(selected_action)

    if not env._at_depot()[0]:
        mask = env.get_mask()
        return_action = choose_best_action(env, [a for a in return_actions if mask[0, a]], mode_policy=mode_policy)
        if return_action is not None:
            env.step(torch.tensor([return_action], device=device))
            repaired_actions.append(return_action)

    return evaluate_actions_with_env(
        nodes=nodes,
        pref=pref,
        depot=depot,
        actions=repaired_actions,
        visited_mask=visited_mask,
        curr_pos=curr_pos,
        curr_time=curr_time,
        curr_load=curr_load,
        curr_energy=curr_energy,
        scenario=scenario,
    )


def explicit_actions_to_eval(
    nodes,
    pref,
    depot,
    action_routes: List[List[int]],
    visited_mask=None,
    curr_pos=None,
    curr_time=None,
    curr_load=None,
    curr_energy=None,
    scenario="full",
    mode_policy="greedy",
):
    device = nodes.device
    batch_size, n_nodes, _ = nodes.shape
    if batch_size != 1:
        raise ValueError("explicit_actions_to_eval only supports batch size 1.")

    if visited_mask is None:
        visited_mask = torch.zeros(batch_size, n_nodes, dtype=torch.bool, device=device)
    if curr_pos is None:
        curr_pos = depot.clone()
    if curr_time is None:
        curr_time = torch.zeros(batch_size, device=device)
    if curr_load is None:
        curr_load = torch.ones(batch_size, device=device)
    if curr_energy is None:
        curr_energy = torch.ones(batch_size, device=device)

    env = VRPTWEnv(nodes.clone(), pref.clone(), depot.clone(), scenario=scenario)
    env.load_state(
        visited_mask.clone(),
        curr_pos.clone(),
        curr_time.clone(),
        curr_load.clone(),
        curr_energy.clone(),
    )

    actions: List[int] = []
    return_actions = [n_nodes * 2, n_nodes * 2 + 1]

    for route in action_routes:
        for action in route:
            mask = env.get_mask()
            if not mask[0, int(action)]:
                continue
            env.step(torch.tensor([int(action)], device=device))
            actions.append(int(action))
            if env._compute_done().all():
                break
        if env._compute_done().all():
            break
        if not env._at_depot()[0]:
            mask = env.get_mask()
            return_action = choose_best_action(env, [a for a in return_actions if mask[0, a]], mode_policy=mode_policy)
            if return_action is not None:
                env.step(torch.tensor([return_action], device=device))
                actions.append(return_action)
            if env._compute_done().all():
                break

    if not env._at_depot()[0]:
        mask = env.get_mask()
        return_action = choose_best_action(env, [a for a in return_actions if mask[0, a]], mode_policy=mode_policy)
        if return_action is not None:
            env.step(torch.tensor([return_action], device=device))
            actions.append(return_action)

    result = evaluate_actions_with_env(
        nodes=nodes,
        pref=pref,
        depot=depot,
        actions=actions,
        visited_mask=visited_mask,
        curr_pos=curr_pos,
        curr_time=curr_time,
        curr_load=curr_load,
        curr_energy=curr_energy,
        scenario=scenario,
    )
    result["actions"] = actions
    return result


def customer_routes_to_eval(
    nodes,
    pref,
    depot,
    route_buckets: List[List[int]],
    visited_mask=None,
    curr_pos=None,
    curr_time=None,
    curr_load=None,
    curr_energy=None,
    scenario="full",
    mode_policy="greedy",
):
    device = nodes.device
    batch_size, n_nodes, _ = nodes.shape
    if batch_size != 1:
        raise ValueError("customer_routes_to_eval only supports batch size 1.")

    if visited_mask is None:
        visited_mask = torch.zeros(batch_size, n_nodes, dtype=torch.bool, device=device)
    if curr_pos is None:
        curr_pos = depot.clone()
    if curr_time is None:
        curr_time = torch.zeros(batch_size, device=device)
    if curr_load is None:
        curr_load = torch.ones(batch_size, device=device)
    if curr_energy is None:
        curr_energy = torch.ones(batch_size, device=device)

    env = VRPTWEnv(nodes.clone(), pref.clone(), depot.clone(), scenario=scenario)
    env.load_state(
        visited_mask.clone(),
        curr_pos.clone(),
        curr_time.clone(),
        curr_load.clone(),
        curr_energy.clone(),
    )

    actions: List[int] = []
    return_actions = [n_nodes * 2, n_nodes * 2 + 1]

    for route in route_buckets:
        for node_idx in route:
            if env.visited[0, int(node_idx)]:
                continue
            mask = env.get_mask()
            customer_actions = []
            normal_action = int(node_idx) * 2
            fast_action = int(node_idx) * 2 + 1
            if mask[0, normal_action]:
                customer_actions.append(normal_action)
            if mask[0, fast_action]:
                customer_actions.append(fast_action)

            selected_action = choose_best_action(env, customer_actions, mode_policy=mode_policy)
            if selected_action is None:
                return_action = choose_best_action(env, [a for a in return_actions if mask[0, a]], mode_policy=mode_policy)
                if return_action is not None:
                    env.step(torch.tensor([return_action], device=device))
                    actions.append(return_action)
                    mask = env.get_mask()
                    customer_actions = []
                    if mask[0, normal_action]:
                        customer_actions.append(normal_action)
                    if mask[0, fast_action]:
                        customer_actions.append(fast_action)
                    selected_action = choose_best_action(env, customer_actions, mode_policy=mode_policy)

            if selected_action is None:
                continue

            env.step(torch.tensor([selected_action], device=device))
            actions.append(selected_action)
            if env._compute_done().all():
                break

        if env._compute_done().all():
            break

        if not env._at_depot()[0]:
            mask = env.get_mask()
            return_action = choose_best_action(env, [a for a in return_actions if mask[0, a]], mode_policy=mode_policy)
            if return_action is not None:
                env.step(torch.tensor([return_action], device=device))
                actions.append(return_action)
            if env._compute_done().all():
                break

    if not env._at_depot()[0]:
        mask = env.get_mask()
        return_action = choose_best_action(env, [a for a in return_actions if mask[0, a]], mode_policy=mode_policy)
        if return_action is not None:
            env.step(torch.tensor([return_action], device=device))
            actions.append(return_action)

    result = evaluate_actions_with_env(
        nodes=nodes,
        pref=pref,
        depot=depot,
        actions=actions,
        visited_mask=visited_mask,
        curr_pos=curr_pos,
        curr_time=curr_time,
        curr_load=curr_load,
        curr_energy=curr_energy,
        scenario=scenario,
    )
    result["actions"] = actions
    return result


def priority_to_actions(
    nodes,
    pref,
    depot,
    priority: List[int],
    visited_mask=None,
    curr_pos=None,
    curr_time=None,
    curr_load=None,
    curr_energy=None,
    scenario="full",
    mode_policy="greedy",
):
    device = nodes.device
    batch_size, n_nodes, _ = nodes.shape
    if batch_size != 1:
        raise ValueError("priority_to_actions only supports batch size 1.")

    if visited_mask is None:
        visited_mask = torch.zeros(batch_size, n_nodes, dtype=torch.bool, device=device)
    if curr_pos is None:
        curr_pos = depot.clone()
    if curr_time is None:
        curr_time = torch.zeros(batch_size, device=device)
    if curr_load is None:
        curr_load = torch.ones(batch_size, device=device)
    if curr_energy is None:
        curr_energy = torch.ones(batch_size, device=device)

    env = VRPTWEnv(nodes.clone(), pref.clone(), depot.clone(), scenario=scenario)
    env.load_state(
        visited_mask.clone(),
        curr_pos.clone(),
        curr_time.clone(),
        curr_load.clone(),
        curr_energy.clone(),
    )

    actions: List[int] = []
    return_actions = [n_nodes * 2, n_nodes * 2 + 1]
    remaining = [int(x) for x in priority if 0 <= int(x) < n_nodes]

    for node_idx in remaining:
        if env.visited[0, node_idx]:
            continue
        mask = env.get_mask()
        customer_actions = []
        normal_action = node_idx * 2
        fast_action = node_idx * 2 + 1
        if mask[0, normal_action]:
            customer_actions.append(normal_action)
        if mask[0, fast_action]:
            customer_actions.append(fast_action)

        selected_action = choose_best_action(env, customer_actions, mode_policy=mode_policy)
        if selected_action is None:
            return_action = choose_best_action(env, [a for a in return_actions if mask[0, a]], mode_policy=mode_policy)
            if return_action is not None:
                env.step(torch.tensor([return_action], device=device))
                actions.append(return_action)
                mask = env.get_mask()
                customer_actions = []
                if mask[0, normal_action]:
                    customer_actions.append(normal_action)
                if mask[0, fast_action]:
                    customer_actions.append(fast_action)
                selected_action = choose_best_action(env, customer_actions, mode_policy=mode_policy)

        if selected_action is None:
            continue

        env.step(torch.tensor([selected_action], device=device))
        actions.append(selected_action)

    if not env._at_depot()[0]:
        mask = env.get_mask()
        return_action = choose_best_action(env, [a for a in return_actions if mask[0, a]], mode_policy=mode_policy)
        if return_action is not None:
            env.step(torch.tensor([return_action], device=device))
            actions.append(return_action)

    return evaluate_actions_with_env(
        nodes=nodes,
        pref=pref,
        depot=depot,
        actions=actions,
        visited_mask=visited_mask,
        curr_pos=curr_pos,
        curr_time=curr_time,
        curr_load=curr_load,
        curr_energy=curr_energy,
        scenario=scenario,
    )


def sort_route_buckets_by_time(nodes, route_buckets: List[List[int]]) -> List[List[int]]:
    def route_key(route: List[int]):
        if not route:
            return (math.inf, math.inf)
        due_values = [float(nodes[0, idx, 4].item()) for idx in route]
        ready_values = [float(nodes[0, idx, 3].item()) for idx in route]
        return (min(due_values), min(ready_values))

    return [list(route) for route in sorted(route_buckets, key=route_key)]


def route_buckets_to_priority(nodes, route_buckets: List[List[int]]) -> List[int]:
    ordered_routes = sort_route_buckets_by_time(nodes, route_buckets)
    priority: List[int] = []
    for route in ordered_routes:
        priority.extend(route)
    return priority
