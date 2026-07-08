import math
import random
import time

import torch

from config import (
    DEVICE,
    E_MAX,
    GRID_SCALE,
    MAX_PAYLOAD_KG,
    MAX_TIME,
    N_POMO,
    PEN_WEIGHT,
    PHYS_WEIGHT,
    P_EMPTY_FAST,
    P_EMPTY_NORM,
    P_GAMMA,
    P_HOVER,
    V_FAST,
    V_NORM,
)
from env.vrp_env import VRPTWEnv
from eval.route_baseline_utils import customer_routes_to_eval, sort_route_buckets_by_time


class HybridALNSSolver:
    def __init__(self, pomo_model, device=DEVICE, scenario="full"):
        self.model = pomo_model
        self.device = device
        self.scenario = str(scenario).lower()
        if self.model is not None:
            self.model.eval()

    def _build_heuristic_priority(
        self,
        nodes,
        pref,
        depot,
        visited_mask,
        curr_pos,
        curr_time,
        curr_load,
        curr_energy,
    ):
        node_count = nodes.size(1)
        visited_set = set(torch.nonzero(visited_mask[0], as_tuple=False).view(-1).tolist())
        remaining = [idx for idx in range(node_count) if idx not in visited_set]
        if not remaining:
            return []

        pref_time = float(pref[0, 0].item())
        pref_energy = float(pref[0, 1].item())
        curr_xy = curr_pos[0]
        rows = []
        for idx in remaining:
            node = nodes[0, idx]
            dist = torch.norm(node[:2] - curr_xy).item() * GRID_SCALE
            due = float(node[4].item())
            ready = float(node[3].item())
            demand = float(node[2].item())
            t_norm = dist / V_NORM / MAX_TIME
            t_fast = dist / V_FAST / MAX_TIME
            arrival_norm = float(curr_time[0].item()) + t_norm
            arrival_fast = float(curr_time[0].item()) + t_fast
            wait_norm = max(0.0, ready - arrival_norm)
            wait_fast = max(0.0, ready - arrival_fast)
            energy_norm = (
                (P_EMPTY_NORM + P_GAMMA * float(curr_load[0].item()) * MAX_PAYLOAD_KG) * (dist / V_NORM)
            ) / E_MAX
            energy_fast = (
                (P_EMPTY_FAST + P_GAMMA * float(curr_load[0].item()) * MAX_PAYLOAD_KG) * (dist / V_FAST)
            ) / E_MAX
            best_mode_score = min(
                energy_norm * 10.0 * pref_energy + t_norm * 5.0 * pref_time + wait_norm * P_HOVER * pref_energy,
                energy_fast * 10.0 * pref_energy + t_fast * 5.0 * pref_time + wait_fast * P_HOVER * pref_energy,
            )
            tardiness_risk = min(max(0.0, arrival_norm - due), max(0.0, arrival_fast - due))
            score = 4.0 * due + 2.0 * ready + 0.2 * demand + best_mode_score + 10.0 * tardiness_risk
            rows.append((score, idx))
        rows.sort(key=lambda x: x[0])
        return [idx for _, idx in rows]

    def _remaining_nodes(self, nodes, visited_mask):
        visited_set = set(torch.nonzero(visited_mask[0], as_tuple=False).view(-1).tolist())
        return [idx for idx in range(nodes.size(1)) if idx not in visited_set]

    def _build_seed_priorities(
        self,
        nodes,
        pref,
        depot,
        visited_mask,
        curr_pos,
        curr_time,
        curr_load,
        curr_energy,
    ):
        remaining = self._remaining_nodes(nodes, visited_mask)
        if not remaining:
            return []

        node_data = nodes[0]
        depot_xy = depot[0]
        seed_priorities = []

        heuristic = self._build_heuristic_priority(
            nodes,
            pref,
            depot,
            visited_mask,
            curr_pos,
            curr_time,
            curr_load,
            curr_energy,
        )
        seed_priorities.append(heuristic)

        by_due = sorted(
            remaining,
            key=lambda idx: (
                float(node_data[idx, 4].item()),
                float(node_data[idx, 3].item()),
                float(torch.norm(node_data[idx, :2] - depot_xy).item()),
            ),
        )
        seed_priorities.append(by_due)

        by_ready = sorted(
            remaining,
            key=lambda idx: (
                float(node_data[idx, 3].item()),
                float(node_data[idx, 4].item()),
                float(torch.norm(node_data[idx, :2] - depot_xy).item()),
            ),
        )
        seed_priorities.append(by_ready)

        angles = []
        for idx in remaining:
            rel = node_data[idx, :2] - depot_xy
            angle = math.atan2(float(rel[1].item()), float(rel[0].item()))
            radius = float(torch.norm(rel).item())
            angles.append((angle, radius, idx))
        sweep = [idx for _, _, idx in sorted(angles)]
        seed_priorities.append(sweep)
        seed_priorities.append(list(reversed(sweep)))
        if len(sweep) >= 8:
            for ratio in (0.25, 0.5, 0.75):
                cut = int(round(len(sweep) * ratio))
                seed_priorities.append(sweep[cut:] + sweep[:cut])

        nearest = []
        unused = set(remaining)
        pos = curr_pos[0].clone()
        while unused:
            best_idx = min(
                unused,
                key=lambda idx: (
                    float(torch.norm(node_data[idx, :2] - pos).item())
                    + 0.15 * float(node_data[idx, 4].item())
                    + 0.05 * float(node_data[idx, 3].item())
                ),
            )
            nearest.append(best_idx)
            unused.remove(best_idx)
            pos = node_data[best_idx, :2]
        seed_priorities.append(nearest)

        # N50 is dominated by route geometry once feasibility is handled. Add
        # several TSP-like skeletons early, then let exact env replay decide
        # whether their time/energy behavior is acceptable.
        farthest_from_depot = max(
            remaining,
            key=lambda idx: float(torch.norm(node_data[idx, :2] - depot_xy).item()),
        )
        earliest_due = min(remaining, key=lambda idx: float(node_data[idx, 4].item()))
        latest_due = max(remaining, key=lambda idx: float(node_data[idx, 4].item()))
        for start_node in (None, farthest_from_depot, earliest_due, latest_due):
            seed_priorities.append(
                self._nearest_priority_from_start(
                    nodes,
                    depot,
                    remaining,
                    start_node=start_node,
                    due_weight=0.00,
                )
            )
            seed_priorities.append(
                self._nearest_priority_from_start(
                    nodes,
                    depot,
                    remaining,
                    start_node=start_node,
                    due_weight=0.03,
                )
            )
        farthest_insert = self._farthest_insertion_priority(nodes, depot, remaining)
        seed_priorities.append(farthest_insert)

        for insertion_order in (by_due, by_ready, nearest, sweep, list(reversed(sweep))):
            seed_priorities.append(self._cheapest_insertion_priority(nodes, depot, insertion_order))

        # These proxy-optimized seeds are cheap to build and give ALNS several
        # globally different starting orders before expensive env replay starts.
        proxy_optimized = []
        for priority in seed_priorities:
            if len(priority) >= 4:
                proxy_optimized.append(self._proxy_2opt_priority(nodes, depot, priority, max_passes=2))
                proxy_optimized.append(self._euclidean_2opt_priority(nodes, depot, priority, max_passes=3))
        seed_priorities.extend(proxy_optimized)

        seed_priorities.sort(
            key=lambda priority: (
                0.65 * self._proxy_priority_cost(nodes, depot, priority)
                + 0.35 * self._euclidean_priority_cost(nodes, depot, priority)
            )
        )

        unique = []
        seen = set()
        for priority in seed_priorities:
            completed = self._complete_priority(priority, nodes.size(1), visited_mask)
            key = tuple(completed)
            if key not in seen:
                unique.append(completed)
                seen.add(key)
        return unique

    def _extract_customer_priority(self, action_sequence, node_count, visited_mask=None):
        seen = set()
        priority = []
        visited_set = set()
        if visited_mask is not None:
            visited_set = set(torch.nonzero(visited_mask[0], as_tuple=False).view(-1).tolist())

        for action in action_sequence.tolist():
            action = int(action)
            if action < node_count * 2:
                node_idx = action // 2
                if node_idx not in seen and node_idx not in visited_set:
                    priority.append(node_idx)
                    seen.add(node_idx)
        return priority

    def _complete_priority(self, priority, node_count, visited_mask):
        visited_set = set(torch.nonzero(visited_mask[0], as_tuple=False).view(-1).tolist())
        completed = []
        used = set()
        for node in priority:
            if node not in visited_set and node not in used:
                completed.append(node)
                used.add(node)
        for node in range(node_count):
            if node not in visited_set and node not in used:
                completed.append(node)
        return completed

    def _flatten_route_buckets(self, route_buckets):
        priority = []
        for route in route_buckets:
            priority.extend(int(node) for node in route)
        return priority

    def _bucket_key(self, route_buckets):
        return tuple(tuple(int(node) for node in route) for route in route_buckets if route)

    def _solution_key(self, solution):
        if solution is None:
            return (True, float("inf"), float("inf"))
        return (
            not bool(solution.get("feasible", False)),
            float(solution.get("unvisit", float("inf"))),
            float(solution.get("cost", float("inf"))),
        )

    def _is_better_solution(self, candidate, incumbent):
        return self._solution_key(candidate) < self._solution_key(incumbent)

    def _actions_to_route_buckets(self, actions, node_count, visited_mask=None):
        visited_set = set()
        if visited_mask is not None:
            visited_set = set(torch.nonzero(visited_mask[0], as_tuple=False).view(-1).tolist())

        seen = set(visited_set)
        buckets = []
        current = []
        for action in actions:
            action = int(action)
            if action < node_count * 2:
                node_idx = action // 2
                if node_idx not in seen:
                    current.append(node_idx)
                    seen.add(node_idx)
            elif current:
                buckets.append(current)
                current = []
        if current:
            buckets.append(current)
        return buckets

    def _normalize_route_buckets(self, route_buckets, node_count, visited_mask):
        visited_set = set(torch.nonzero(visited_mask[0], as_tuple=False).view(-1).tolist())
        seen = set()
        normalized = []
        for route in route_buckets:
            clean = []
            for node in route:
                node = int(node)
                if 0 <= node < node_count and node not in visited_set and node not in seen:
                    clean.append(node)
                    seen.add(node)
            if clean:
                normalized.append(clean)
        for node in range(node_count):
            if node not in visited_set and node not in seen:
                normalized.append([node])
        return normalized

    def _proxy_priority_cost(self, nodes, depot, priority):
        if not priority:
            return 0.0
        cost = self._proxy_priority_arc_cost(nodes, depot, None, priority[0])
        for prev_node, next_node in zip(priority[:-1], priority[1:]):
            cost += self._proxy_priority_arc_cost(nodes, depot, prev_node, next_node)
        cost += self._proxy_priority_arc_cost(nodes, depot, priority[-1], None)
        return cost

    def _serial_physical_proxy_cost(self, nodes, pref, depot, priority):
        if not priority:
            return 0.0
        pref_time = float(pref[0, 0].item())
        pref_energy = float(pref[0, 1].item())
        load_proxy = 1.0

        def xy(node):
            return depot[0] if node is None else nodes[0, int(node), :2]

        def dist(from_node, to_node):
            return float(torch.norm(xy(from_node) - xy(to_node)).item() * GRID_SCALE)

        def arc_cost(from_node, to_node):
            d = dist(from_node, to_node)
            t_norm = d / V_NORM / MAX_TIME
            t_fast = d / V_FAST / MAX_TIME
            e_norm = ((P_EMPTY_NORM + P_GAMMA * load_proxy * MAX_PAYLOAD_KG) * (d / V_NORM)) / E_MAX
            e_fast = ((P_EMPTY_FAST + P_GAMMA * load_proxy * MAX_PAYLOAD_KG) * (d / V_FAST)) / E_MAX
            phys_norm = e_norm * 10.0 * pref_energy + t_norm * 5.0 * pref_time
            phys_fast = e_fast * 10.0 * pref_energy + t_fast * 5.0 * pref_time
            return PHYS_WEIGHT * min(phys_norm, phys_fast)

        cost = 0.0
        clock = 0.0
        prev = None
        for node in priority:
            node = int(node)
            d = dist(prev, node)
            cost += arc_cost(prev, node)
            arrival = clock + d / V_FAST / MAX_TIME
            ready = float(nodes[0, node, 3].item())
            due = float(nodes[0, node, 4].item())
            service = float(nodes[0, node, 5].item())
            start_service = max(arrival, ready)
            cost += PEN_WEIGHT * max(0.0, start_service - due)
            clock = start_service + service
            prev = node
        cost += arc_cost(prev, None)
        return cost

    def _serial_proxy_search(self, nodes, pref, depot, priority, deadline=None, max_trials=1000):
        current = [int(node) for node in priority]
        if len(current) <= 3:
            return current
        current_cost = self._serial_physical_proxy_cost(nodes, pref, depot, current)
        best = list(current)
        best_cost = current_cost
        temperature = max(1.0, 0.03 * current_cost)
        trials = 0
        while trials < max_trials:
            if deadline is not None and time.perf_counter() >= deadline:
                break
            trials += 1
            cand = list(current)
            move = random.choice(["relocate", "swap", "two_opt"])
            if move == "relocate":
                src = random.randrange(len(cand))
                dst = random.randrange(len(cand))
                if src == dst:
                    continue
                node = cand.pop(src)
                cand.insert(dst, node)
            elif move == "swap":
                i, j = random.sample(range(len(cand)), 2)
                cand[i], cand[j] = cand[j], cand[i]
            else:
                i, j = sorted(random.sample(range(len(cand)), 2))
                if j - i <= 1:
                    continue
                cand = cand[:i] + list(reversed(cand[i:j])) + cand[j:]
            cand_cost = self._serial_physical_proxy_cost(nodes, pref, depot, cand)
            delta = cand_cost - current_cost
            if delta < 0 or random.random() < math.exp(-delta / max(temperature, 1e-6)):
                current = cand
                current_cost = cand_cost
            if cand_cost < best_cost:
                best = cand
                best_cost = cand_cost
            temperature *= 0.997
        return best

    def _cheapest_insertion_priority(self, nodes, depot, insertion_order):
        route = []
        for node in insertion_order:
            node = int(node)
            if node in route:
                continue
            if not route:
                route.append(node)
                continue
            best_pos = 0
            best_delta = None
            for pos in range(len(route) + 1):
                prev_node = route[pos - 1] if pos > 0 else None
                next_node = route[pos] if pos < len(route) else None
                before = self._proxy_priority_arc_cost(nodes, depot, prev_node, next_node)
                after = (
                    self._proxy_priority_arc_cost(nodes, depot, prev_node, node)
                    + self._proxy_priority_arc_cost(nodes, depot, node, next_node)
                )
                due = float(nodes[0, node, 4].item())
                delta = after - before + 0.01 * due
                if best_delta is None or delta < best_delta:
                    best_delta = delta
                    best_pos = pos
            route.insert(best_pos, node)
        return route

    def _proxy_2opt_priority(self, nodes, depot, priority, max_passes=2):
        route = [int(node) for node in priority]
        if len(route) <= 3:
            return route
        best_cost = self._proxy_priority_cost(nodes, depot, route)
        for _ in range(max(1, int(max_passes))):
            improved = False
            for i in range(len(route) - 2):
                for j in range(i + 2, len(route) + 1):
                    candidate = route[:i] + list(reversed(route[i:j])) + route[j:]
                    candidate_cost = self._proxy_priority_cost(nodes, depot, candidate)
                    if candidate_cost + 1e-12 < best_cost:
                        route = candidate
                        best_cost = candidate_cost
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
        return route

    def _euclidean_arc_cost(self, nodes, depot, from_node, to_node):
        from_xy = depot[0] if from_node is None else nodes[0, int(from_node), :2]
        to_xy = depot[0] if to_node is None else nodes[0, int(to_node), :2]
        return float(torch.norm(to_xy - from_xy).item())

    def _euclidean_priority_cost(self, nodes, depot, priority):
        route = [int(node) for node in priority]
        if not route:
            return 0.0
        cost = self._euclidean_arc_cost(nodes, depot, None, route[0])
        for a, b in zip(route[:-1], route[1:]):
            cost += self._euclidean_arc_cost(nodes, depot, a, b)
        cost += self._euclidean_arc_cost(nodes, depot, route[-1], None)
        return cost

    def _euclidean_2opt_priority(self, nodes, depot, priority, max_passes=4):
        route = [int(node) for node in priority]
        if len(route) <= 3:
            return route
        best_cost = self._euclidean_priority_cost(nodes, depot, route)
        for _ in range(max(1, int(max_passes))):
            improved = False
            for i in range(len(route) - 2):
                for j in range(i + 2, len(route) + 1):
                    candidate = route[:i] + list(reversed(route[i:j])) + route[j:]
                    candidate_cost = self._euclidean_priority_cost(nodes, depot, candidate)
                    if candidate_cost + 1e-12 < best_cost:
                        route = candidate
                        best_cost = candidate_cost
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
        return route

    def _nearest_priority_from_start(self, nodes, depot, remaining, start_node=None, due_weight=0.0):
        node_data = nodes[0]
        unused = set(int(node) for node in remaining)
        priority = []
        pos = depot[0].clone() if start_node is None else node_data[int(start_node), :2].clone()
        if start_node is not None and int(start_node) in unused:
            priority.append(int(start_node))
            unused.remove(int(start_node))
        while unused:
            picked = min(
                unused,
                key=lambda idx: (
                    float(torch.norm(node_data[idx, :2] - pos).item())
                    + due_weight * float(node_data[idx, 4].item())
                ),
            )
            priority.append(picked)
            unused.remove(picked)
            pos = node_data[picked, :2]
        return priority

    def _farthest_insertion_priority(self, nodes, depot, remaining):
        remaining = [int(node) for node in remaining]
        if len(remaining) <= 2:
            return list(remaining)
        node_data = nodes[0]
        depot_xy = depot[0]
        first = max(remaining, key=lambda idx: float(torch.norm(node_data[idx, :2] - depot_xy).item()))
        second = max(
            [idx for idx in remaining if idx != first],
            key=lambda idx: float(torch.norm(node_data[idx, :2] - node_data[first, :2]).item()),
        )
        route = [first, second]
        unused = set(remaining)
        unused.remove(first)
        unused.remove(second)
        while unused:
            node = max(
                unused,
                key=lambda idx: min(
                    float(torch.norm(node_data[idx, :2] - node_data[r, :2]).item())
                    for r in route
                ),
            )
            best_pos = 0
            best_delta = None
            for pos in range(len(route) + 1):
                prev_node = route[pos - 1] if pos > 0 else None
                next_node = route[pos] if pos < len(route) else None
                before = self._euclidean_arc_cost(nodes, depot, prev_node, next_node)
                after = (
                    self._euclidean_arc_cost(nodes, depot, prev_node, node)
                    + self._euclidean_arc_cost(nodes, depot, node, next_node)
                )
                delta = after - before
                if best_delta is None or delta < best_delta:
                    best_delta = delta
                    best_pos = pos
            route.insert(best_pos, node)
            unused.remove(node)
        return route

    def _proxy_priority_arc_cost(self, nodes, depot, from_node, to_node):
        """Cheap physics-aware proxy used only to rank ALNS move candidates."""
        if from_node is None:
            from_xy = depot[0]
            from_due = 0.0
        else:
            from_xy = nodes[0, int(from_node), :2]
            from_due = float(nodes[0, int(from_node), 4].item())
        if to_node is None:
            to_xy = depot[0]
            to_ready = 0.0
            to_due = 0.0
        else:
            to_xy = nodes[0, int(to_node), :2]
            to_ready = float(nodes[0, int(to_node), 3].item())
            to_due = float(nodes[0, int(to_node), 4].item())

        dist = float(torch.norm(to_xy - from_xy).item() * GRID_SCALE)
        # Greedy replay chooses the better mode later; this proxy only ranks
        # candidate insertions, so use optimistic flight time plus due pressure.
        travel_fast = dist / V_FAST / MAX_TIME
        travel_norm = dist / V_NORM / MAX_TIME
        time_pressure = 0.15 * max(0.0, from_due + travel_fast - to_due)
        ready_gap = 0.02 * max(0.0, to_ready - from_due - travel_fast)
        return min(travel_fast * 5.0, travel_norm * 5.0) + time_pressure + ready_gap

    def _rank_insert_positions(self, route, node, nodes, depot, max_positions):
        all_positions = list(range(len(route) + 1))
        if max_positions is None or max_positions <= 0 or len(all_positions) <= max_positions:
            return all_positions

        scored = []
        for pos in all_positions:
            prev_node = route[pos - 1] if pos > 0 else None
            next_node = route[pos] if pos < len(route) else None
            before = self._proxy_priority_arc_cost(nodes, depot, prev_node, next_node)
            after = (
                self._proxy_priority_arc_cost(nodes, depot, prev_node, node)
                + self._proxy_priority_arc_cost(nodes, depot, node, next_node)
            )
            due = float(nodes[0, int(node), 4].item())
            scored.append((after - before + 0.01 * due, pos))

        keep = {0, len(route)}
        for _, pos in sorted(scored, key=lambda item: item[0]):
            keep.add(pos)
            if len(keep) >= max_positions:
                break
        return sorted(keep)

    def _worst_proxy_positions(self, priority, nodes, depot, count):
        if not priority:
            return []
        scored = []
        for idx, node in enumerate(priority):
            prev_node = priority[idx - 1] if idx > 0 else None
            next_node = priority[idx + 1] if idx + 1 < len(priority) else None
            with_node = (
                self._proxy_priority_arc_cost(nodes, depot, prev_node, node)
                + self._proxy_priority_arc_cost(nodes, depot, node, next_node)
            )
            without_node = self._proxy_priority_arc_cost(nodes, depot, prev_node, next_node)
            # High positive saving means this customer is currently awkwardly placed.
            scored.append((with_node - without_node, idx))
        return [idx for _, idx in sorted(scored, reverse=True)[:count]]

    def _step_incremental_cost(self, env, action):
        idx = 0
        n_nodes = env.nodes.size(1)
        is_customer = action < n_nodes * 2
        node_idx = action // 2
        is_fast = action % 2 == 1
        speed = V_FAST if is_fast else V_NORM
        p_empty = P_EMPTY_FAST if is_fast else P_EMPTY_NORM

        if is_customer:
            target = env.nodes[idx, node_idx, :2]
        else:
            target = env.depot[idx]

        dist = torch.norm(target - env.pos[idx]) * GRID_SCALE
        t_flight = dist / speed
        t_norm = t_flight / MAX_TIME
        arrival = env.time[idx] + t_norm
        e_flight = (p_empty + P_GAMMA * env.load[idx] * MAX_PAYLOAD_KG) * t_flight

        pen_inc = torch.tensor(0.0, device=env.dev)
        wait_e = torch.tensor(0.0, device=env.dev)
        if is_customer:
            ready = env.nodes[idx, node_idx, 3]
            due = env.nodes[idx, node_idx, 4]
            start = torch.maximum(arrival, ready)
            wait_e = P_HOVER * torch.relu(start - arrival) * MAX_TIME
            pen_inc = torch.relu(start - due)

        phys_inc = ((e_flight + wait_e) / E_MAX) * 10.0 * env.pref[idx, 1] + t_norm * 5.0 * env.pref[idx, 0]
        return float((PHYS_WEIGHT * phys_inc + PEN_WEIGHT * pen_inc).item())

    def _choose_best_action(self, env, candidates):
        if not candidates:
            return None
        best_action = None
        best_cost = None
        for action in candidates:
            step_cost = self._step_incremental_cost(env, action)
            if best_cost is None or step_cost < best_cost:
                best_cost = step_cost
                best_action = action
        return best_action

    def _simulate_priority(
        self,
        nodes,
        pref,
        depot,
        visited_mask,
        curr_pos,
        curr_time,
        curr_load,
        curr_energy,
        priority,
    ):
        env = VRPTWEnv(nodes.clone(), pref.clone(), depot.clone(), scenario=self.scenario)
        env.load_state(
            visited_mask.clone(),
            curr_pos.clone(),
            curr_time.clone(),
            curr_load.clone(),
            curr_energy.clone(),
        )

        return_actions = [nodes.size(1) * 2, nodes.size(1) * 2 + 1]
        actions = []

        def pick_customer_action(node_idx):
            mask = env.get_mask()
            feasible = []
            action_norm = node_idx * 2
            action_fast = node_idx * 2 + 1
            if mask[0, action_norm]:
                feasible.append(action_norm)
            if mask[0, action_fast]:
                feasible.append(action_fast)
            return self._choose_best_action(env, feasible)

        def pick_return_action():
            mask = env.get_mask()
            feasible = [action for action in return_actions if mask[0, action]]
            return self._choose_best_action(env, feasible)

        for node_idx in priority:
            if env.visited[0, node_idx]:
                continue
            selected_action = pick_customer_action(node_idx)
            if selected_action is None:
                return_action = pick_return_action()
                if return_action is not None:
                    _, _ = env.step(torch.tensor([return_action], device=self.device))
                    actions.append(return_action)
                    selected_action = pick_customer_action(node_idx)

            if selected_action is None:
                continue

            _, _ = env.step(torch.tensor([selected_action], device=self.device))
            actions.append(selected_action)

        if not env._at_depot()[0]:
            return_action = pick_return_action()
            if return_action is not None:
                _, _ = env.step(torch.tensor([return_action], device=self.device))
                actions.append(return_action)

        unvisit = (~env.visited).sum(-1).float()
        total_cost = env.phys * PHYS_WEIGHT + env.pen * PEN_WEIGHT + unvisit * 20.0
        feasible = bool((unvisit == 0).all().item() and env._at_depot().all().item() and env._compute_done().all().item())
        route_buckets = self._actions_to_route_buckets(actions, nodes.size(1), visited_mask=visited_mask)
        return {
            "priority": priority,
            "actions": actions,
            "route_actions": list(actions),
            "route_buckets": route_buckets,
            "cost": float(total_cost.item()),
            "phys": float(env.phys.item()),
            "pen": float(env.pen.item()),
            "unvisit": float(unvisit.item()),
            "feasible": feasible,
            "visited_mask": env.visited.clone(),
            "curr_pos": env.pos.clone(),
            "curr_time": env.time.clone(),
            "curr_load": env.load.clone(),
            "curr_energy": env.energy.clone(),
        }

    def _clone_env_state(self, env):
        cloned = VRPTWEnv(env.nodes.clone(), env.pref.clone(), env.depot.clone(), scenario=self.scenario)
        cloned.load_state(
            env.visited.clone(),
            env.pos.clone(),
            env.time.clone(),
            env.load.clone(),
            env.energy.clone(),
        )
        cloned.phys = env.phys.clone()
        cloned.pen = env.pen.clone()
        return cloned

    def _env_total_cost(self, env):
        unvisit = (~env.visited).sum(-1).float()
        total_cost = env.phys * PHYS_WEIGHT + env.pen * PEN_WEIGHT + unvisit * 20.0
        feasible = bool((unvisit == 0).all().item() and env._at_depot().all().item() and env._compute_done().all().item())
        return float(total_cost.item()), float(env.phys.item()), float(env.pen.item()), float(unvisit.item()), feasible

    def _beam_simulate_priority(
        self,
        nodes,
        pref,
        depot,
        visited_mask,
        curr_pos,
        curr_time,
        curr_load,
        curr_energy,
        priority,
        beam_width=8,
        deadline=None,
    ):
        """Optimize trip splits and speed modes for a fixed customer order."""
        env0 = VRPTWEnv(nodes.clone(), pref.clone(), depot.clone(), scenario=self.scenario)
        env0.load_state(
            visited_mask.clone(),
            curr_pos.clone(),
            curr_time.clone(),
            curr_load.clone(),
            curr_energy.clone(),
        )

        node_count = nodes.size(1)
        return_actions = [node_count * 2, node_count * 2 + 1]
        labels = [(env0, [])]

        for node_idx in priority:
            node_idx = int(node_idx)
            if deadline is not None and time.perf_counter() >= deadline:
                break
            if labels and labels[0][0].visited[0, node_idx]:
                continue

            candidates = []
            for env, actions in labels:
                if deadline is not None and time.perf_counter() >= deadline:
                    break
                prep = [(env, actions)]
                if not bool(env._at_depot()[0].item()):
                    mask = env.get_mask()
                    for ret_action in return_actions:
                        if not bool(mask[0, ret_action].item()):
                            continue
                        ret_env = self._clone_env_state(env)
                        ret_env.step(torch.tensor([ret_action], device=self.device))
                        prep.append((ret_env, actions + [ret_action]))

                for base_env, base_actions in prep:
                    mask = base_env.get_mask()
                    for customer_action in (node_idx * 2, node_idx * 2 + 1):
                        if not bool(mask[0, customer_action].item()):
                            continue
                        next_env = self._clone_env_state(base_env)
                        next_env.step(torch.tensor([customer_action], device=self.device))
                        candidates.append((next_env, base_actions + [customer_action]))

            if not candidates:
                # Keep the best partial labels; the final unvisit penalty will
                # reject incomplete orders unless no feasible continuation exists.
                break
            candidates.sort(key=lambda item: self._env_total_cost(item[0])[0])
            labels = candidates[: max(1, int(beam_width))]

        finals = []
        for env, actions in labels:
            if bool(env._at_depot()[0].item()):
                finals.append((env, actions))
                continue
            mask = env.get_mask()
            for ret_action in return_actions:
                if not bool(mask[0, ret_action].item()):
                    continue
                ret_env = self._clone_env_state(env)
                ret_env.step(torch.tensor([ret_action], device=self.device))
                finals.append((ret_env, actions + [ret_action]))
        if not finals:
            finals = labels

        finals.sort(key=lambda item: self._env_total_cost(item[0])[0])
        best_env, best_actions = finals[0]
        cost, phys, pen, unvisit, feasible = self._env_total_cost(best_env)
        route_buckets = self._actions_to_route_buckets(best_actions, node_count, visited_mask=visited_mask)
        return {
            "priority": [int(node) for node in priority],
            "actions": [int(action) for action in best_actions],
            "route_actions": [int(action) for action in best_actions],
            "route_buckets": route_buckets,
            "cost": cost,
            "phys": phys,
            "pen": pen,
            "unvisit": unvisit,
            "feasible": feasible,
            "visited_mask": best_env.visited.clone(),
            "curr_pos": best_env.pos.clone(),
            "curr_time": best_env.time.clone(),
            "curr_load": best_env.load.clone(),
            "curr_energy": best_env.energy.clone(),
            "split_policy": "beam",
        }

    def _evaluate_action_sequence(
        self,
        nodes,
        pref,
        depot,
        visited_mask,
        curr_pos,
        curr_time,
        curr_load,
        curr_energy,
        actions,
    ):
        env = VRPTWEnv(nodes.clone(), pref.clone(), depot.clone(), scenario=self.scenario)
        env.load_state(
            visited_mask.clone(),
            curr_pos.clone(),
            curr_time.clone(),
            curr_load.clone(),
            curr_energy.clone(),
        )

        executed = []
        for action in actions:
            env.step(torch.tensor([int(action)], device=self.device))
            executed.append(int(action))
            if env._compute_done().all():
                break

        unvisit = (~env.visited).sum(-1).float()
        total_cost = env.phys * PHYS_WEIGHT + env.pen * PEN_WEIGHT + unvisit * 20.0
        feasible = bool((unvisit == 0).all().item() and env._at_depot().all().item() and env._compute_done().all().item())
        route_buckets = self._actions_to_route_buckets(executed, nodes.size(1), visited_mask=visited_mask)
        return {
            "actions": executed,
            "route_actions": executed,
            "route_buckets": route_buckets,
            "priority": self._flatten_route_buckets(route_buckets),
            "cost": float(total_cost.item()),
            "phys": float(env.phys.item()),
            "pen": float(env.pen.item()),
            "unvisit": float(unvisit.item()),
            "feasible": feasible,
            "visited_mask": env.visited.clone(),
            "curr_pos": env.pos.clone(),
            "curr_time": env.time.clone(),
            "curr_load": env.load.clone(),
            "curr_energy": env.energy.clone(),
        }

    def _polish_action_sequence(
        self,
        solution,
        nodes,
        pref,
        depot,
        visited_mask,
        curr_pos,
        curr_time,
        curr_load,
        curr_energy,
        deadline=None,
        max_trials=120,
        max_passes=2,
    ):
        """Cheaply improve a fixed route by changing modes and pruning returns.

        This preserves the DRL route order, so it is safe for short real-time
        repair: every accepted move is replayed by the exact environment before
        it can replace the incumbent.
        """
        actions = [int(x) for x in (solution.get("actions") or solution.get("route_actions") or [])]
        if not actions:
            return solution

        best = dict(solution)
        best["actions"] = actions
        best["route_actions"] = actions
        n_customer_actions = nodes.size(1) * 2
        return_actions = {n_customer_actions, n_customer_actions + 1}
        trials = 0

        def evaluate(candidate_actions):
            return self._evaluate_action_sequence(
                nodes,
                pref,
                depot,
                visited_mask,
                curr_pos,
                curr_time,
                curr_load,
                curr_energy,
                candidate_actions,
            )

        for _ in range(max(1, int(max_passes))):
            improved = False
            idx = 0
            while idx < len(actions):
                if trials >= max_trials:
                    return best
                if deadline is not None and time.perf_counter() >= deadline:
                    return best

                action = int(actions[idx])
                candidates = []
                if action < n_customer_actions:
                    candidates.append(action ^ 1)
                elif action in return_actions:
                    candidates.append(n_customer_actions + (1 - (action - n_customer_actions)))
                    # Consecutive or premature returns sometimes survive model
                    # rollout. Removing one is cheap to test and often useful.
                    if idx < len(actions) - 1:
                        candidates.append(None)

                accepted = False
                for replacement in candidates:
                    if trials >= max_trials:
                        return best
                    if deadline is not None and time.perf_counter() >= deadline:
                        return best
                    candidate_actions = list(actions)
                    if replacement is None:
                        del candidate_actions[idx]
                    else:
                        candidate_actions[idx] = int(replacement)
                    candidate = evaluate(candidate_actions)
                    trials += 1
                    if self._is_better_solution(candidate, best):
                        actions = [int(x) for x in candidate.get("actions", candidate_actions)]
                        candidate["actions"] = actions
                        candidate["route_actions"] = actions
                        best = candidate
                        improved = True
                        accepted = True
                        break
                if not accepted:
                    idx += 1
            if not improved:
                break
        return best

    def _greedy_insert(
        self,
        base_priority,
        removed_nodes,
        evaluate_priority_fn,
        max_insert_positions=6,
        deadline=None,
        position_rank_fn=None,
    ):
        route = list(base_priority)
        for node in removed_nodes:
            if deadline is not None and time.perf_counter() >= deadline:
                break
            best_priority = None
            best_eval = None
            if position_rank_fn is not None:
                candidate_positions = position_rank_fn(route, node, max_insert_positions)
            else:
                all_positions = list(range(len(route) + 1))
                if (
                    max_insert_positions is not None
                    and max_insert_positions > 0
                    and len(all_positions) > max_insert_positions
                ):
                    keep = {0, len(route)}
                    middle = [p for p in all_positions if p not in keep]
                    random.shuffle(middle)
                    for p in middle[: max(0, max_insert_positions - len(keep))]:
                        keep.add(p)
                    candidate_positions = sorted(keep)
                else:
                    candidate_positions = all_positions

            for insert_idx in candidate_positions:
                if deadline is not None and time.perf_counter() >= deadline:
                    break
                candidate = route[:insert_idx] + [node] + route[insert_idx:]
                candidate_eval = evaluate_priority_fn(candidate)
                if best_eval is None or candidate_eval["cost"] < best_eval["cost"]:
                    best_priority = candidate
                    best_eval = candidate_eval
            if best_priority is None:
                break
            route = best_priority
        return route

    def _regret_insert(
        self,
        base_priority,
        removed_nodes,
        evaluate_priority_fn,
        max_insert_positions=8,
        deadline=None,
        position_rank_fn=None,
        regret_weight=0.35,
    ):
        route = list(base_priority)
        remaining = [int(node) for node in removed_nodes]
        while remaining:
            if deadline is not None and time.perf_counter() >= deadline:
                break
            best_choice = None
            random.shuffle(remaining)
            for node in remaining:
                if deadline is not None and time.perf_counter() >= deadline:
                    break
                if position_rank_fn is not None:
                    candidate_positions = position_rank_fn(route, node, max_insert_positions)
                else:
                    candidate_positions = list(range(len(route) + 1))
                    if max_insert_positions is not None and max_insert_positions > 0:
                        candidate_positions = candidate_positions[:max_insert_positions]

                scored = []
                for insert_idx in candidate_positions:
                    if deadline is not None and time.perf_counter() >= deadline:
                        break
                    candidate = route[:insert_idx] + [node] + route[insert_idx:]
                    candidate_eval = evaluate_priority_fn(candidate)
                    scored.append((candidate_eval["cost"], insert_idx, candidate))
                if not scored:
                    continue
                scored.sort(key=lambda item: item[0])
                best_cost, best_pos, best_priority = scored[0]
                second_cost = scored[1][0] if len(scored) > 1 else best_cost
                regret = max(0.0, second_cost - best_cost)
                # Lower score is better: good immediate cost, but prioritize
                # nodes that are painful to postpone.
                score = best_cost - regret_weight * regret
                if best_choice is None or score < best_choice[0]:
                    best_choice = (score, best_cost, regret, node, best_pos, best_priority)
            if best_choice is None:
                break
            _, _, _, chosen_node, _, chosen_priority = best_choice
            route = chosen_priority
            remaining.remove(chosen_node)
        if remaining:
            return self._greedy_insert(
                route,
                remaining,
                evaluate_priority_fn=evaluate_priority_fn,
                max_insert_positions=max_insert_positions,
                deadline=deadline,
                position_rank_fn=position_rank_fn,
            )
        return route

    def _proxy_greedy_insert(self, base_priority, removed_nodes, nodes, depot, max_insert_positions=4):
        route = list(base_priority)
        for node in removed_nodes:
            node = int(node)
            positions = self._rank_insert_positions(route, node, nodes, depot, max_insert_positions)
            best_pos = 0
            best_delta = None
            for pos in positions:
                prev_node = route[pos - 1] if pos > 0 else None
                next_node = route[pos] if pos < len(route) else None
                before = self._proxy_priority_arc_cost(nodes, depot, prev_node, next_node)
                after = (
                    self._proxy_priority_arc_cost(nodes, depot, prev_node, node)
                    + self._proxy_priority_arc_cost(nodes, depot, node, next_node)
                )
                due = float(nodes[0, node, 4].item())
                delta = after - before + 0.01 * due
                if best_delta is None or delta < best_delta:
                    best_delta = delta
                    best_pos = pos
            route.insert(best_pos, node)
        return route

    def _extract_route_actions_from_rollout(self, rollout):
        routes = rollout.get("best_routes")
        lengths = rollout.get("best_route_lengths")
        if routes is None:
            return None
        if lengths is not None:
            route_len = int(lengths[0].item())
        else:
            route_len = routes.size(1)
        return [int(x) for x in routes[0, :route_len].detach().cpu().tolist()]

    def _model_guided_tail_repair(
        self,
        nodes,
        pref,
        depot,
        visited_mask,
        curr_pos,
        curr_time,
        curr_load,
        curr_energy,
        base_actions,
        deadline,
        n_pomo,
        decode_type,
        sample_rollouts,
        sample_temperature=1.0,
        guided_attempts=3,
        min_keep_ratio=0.35,
        guided_decode_type="sample",
        guided_sample_rollouts=1,
        guided_pomo_size=None,
        guided_restarts=2,
    ):
        served_total = sum(1 for act in base_actions if int(act) < nodes.size(1) * 2)
        if served_total <= 2:
            return None

        candidate_keep = []
        for ratio in torch.linspace(min_keep_ratio, 0.85, steps=max(2, guided_attempts)).tolist():
            keep = max(1, min(served_total - 1, int(round(served_total * float(ratio)))))
            if keep not in candidate_keep:
                candidate_keep.append(keep)

        best = None
        if guided_pomo_size is None:
            guided_n_pomo = max(2, min(int(n_pomo), 4))
        else:
            guided_n_pomo = max(1, int(guided_pomo_size))
        guided_rollouts = max(0, int(guided_sample_rollouts))

        for served_keep in candidate_keep:
            remaining_s = deadline - time.perf_counter()
            if remaining_s <= 0.15:
                break
            prefix_state = self.simulate_prefix_state(
                nodes,
                pref,
                depot,
                torch.tensor(base_actions, dtype=torch.long, device=self.device),
                served_keep,
                visited_mask=visited_mask,
                curr_pos=curr_pos,
                curr_time=curr_time,
                curr_load=curr_load,
                curr_energy=curr_energy,
            )
            if prefix_state["served_customers"] <= 0:
                continue

            seen_suffixes = set()
            for restart_idx in range(max(1, int(guided_restarts))):
                remaining_s = deadline - time.perf_counter()
                if remaining_s <= 0.12:
                    break
                decode_mode = "greedy" if restart_idx == 0 else guided_decode_type
                restart_rollouts = 0 if decode_mode == "greedy" else guided_rollouts
                with torch.no_grad():
                    suffix = self.model.inference_rollout(
                        nodes,
                        pref,
                        depot,
                        visited_mask=prefix_state["visited_mask"],
                        curr_pos=prefix_state["curr_pos"],
                        curr_time=prefix_state["curr_time"],
                        curr_load=prefix_state["curr_load"],
                        curr_energy=prefix_state["curr_energy"],
                        N_pomo=guided_n_pomo,
                        decode_type=decode_mode,
                        sample_rollouts=restart_rollouts,
                        sample_temperature=sample_temperature,
                    )
                suffix_actions = self._extract_route_actions_from_rollout(suffix)
                if not suffix_actions:
                    continue
                suffix_key = tuple(suffix_actions)
                if suffix_key in seen_suffixes:
                    continue
                seen_suffixes.add(suffix_key)

                candidate_actions = prefix_state["executed_actions"] + suffix_actions
                candidate_eval = self._evaluate_action_sequence(
                    nodes,
                    pref,
                    depot,
                    visited_mask,
                    curr_pos,
                    curr_time,
                    curr_load,
                    curr_energy,
                    candidate_actions,
                )
                candidate_eval["priority"] = self._complete_priority(
                    self._extract_customer_priority(
                        torch.tensor(candidate_actions, dtype=torch.long, device=self.device),
                        nodes.size(1),
                        visited_mask=visited_mask,
                    ),
                    nodes.size(1),
                    visited_mask,
                )
                if best is None or candidate_eval["cost"] < best["cost"]:
                    best = candidate_eval
        return best

    def _evaluate_route_buckets(
        self,
        nodes,
        pref,
        depot,
        visited_mask,
        curr_pos,
        curr_time,
        curr_load,
        curr_energy,
        route_buckets,
    ):
        normalized = self._normalize_route_buckets(route_buckets, nodes.size(1), visited_mask)
        candidates = []
        seen = set()
        for buckets in (normalized, sort_route_buckets_by_time(nodes, normalized)):
            key = self._bucket_key(buckets)
            if key in seen:
                continue
            seen.add(key)
            for mode_policy in ("greedy", "fast", "normal"):
                env_eval = customer_routes_to_eval(
                    nodes=nodes,
                    pref=pref,
                    depot=depot,
                    route_buckets=buckets,
                    visited_mask=visited_mask,
                    curr_pos=curr_pos,
                    curr_time=curr_time,
                    curr_load=curr_load,
                    curr_energy=curr_energy,
                    scenario=self.scenario,
                    mode_policy=mode_policy,
                )
                env_eval["route_buckets"] = [list(route) for route in buckets]
                env_eval["priority"] = self._flatten_route_buckets(buckets)
                env_eval["route_actions"] = list(env_eval.get("actions", []))
                env_eval["mode_policy"] = mode_policy
                candidates.append(env_eval)

        best = None
        for env_eval in candidates:
            if best is None:
                best = env_eval
                continue
            best_key = (not best["feasible"], best["unvisit"], best["cost"])
            eval_key = (not env_eval["feasible"], env_eval["unvisit"], env_eval["cost"])
            if eval_key < best_key:
                best = env_eval
        return best

    def _greedy_env_rebuild(
        self,
        nodes,
        pref,
        depot,
        visited_mask,
        curr_pos,
        curr_time,
        curr_load,
        curr_energy,
        deadline=None,
    ):
        """Build complete action sequences by repeatedly choosing in-env actions.

        This is a cheap construction portfolio used before stochastic ALNS. It
        differs from priority replay: every choice is made from the current
        state and only among currently feasible masked actions, then the final
        sequence is scored by the same environment as the learned policy.
        """

        device = nodes.device
        batch_size, node_count, _ = nodes.shape
        if batch_size != 1:
            return None

        variants = [
            {"step": 1.00, "due": 0.00, "dist": 0.00, "slack": 0.00, "fast_bias": 0.00},
            {"step": 1.00, "due": 0.30, "dist": 0.05, "slack": 0.40, "fast_bias": 0.00},
            {"step": 1.00, "due": 0.80, "dist": 0.05, "slack": 0.80, "fast_bias": -0.02},
            {"step": 1.00, "due": 0.05, "dist": 0.70, "slack": 0.20, "fast_bias": 0.00},
            {"step": 1.00, "due": 0.20, "dist": 0.35, "slack": 0.60, "fast_bias": -0.04},
            {"step": 0.80, "due": 1.10, "dist": 0.05, "slack": 1.20, "fast_bias": -0.06},
            {"step": 1.15, "due": 0.10, "dist": 0.95, "slack": 0.25, "fast_bias": 0.02},
            {"step": 0.95, "due": 0.45, "dist": 0.30, "slack": 0.90, "fast_bias": -0.03},
        ]

        best = None
        for params in variants:
            if deadline is not None and time.perf_counter() >= deadline:
                break

            env = VRPTWEnv(nodes.clone(), pref.clone(), depot.clone(), scenario=self.scenario)
            env.load_state(
                visited_mask.clone(),
                curr_pos.clone(),
                curr_time.clone(),
                curr_load.clone(),
                curr_energy.clone(),
            )
            actions = []
            guard = 0
            while not env._compute_done().all().item() and guard < node_count * 4 + 8:
                guard += 1
                if deadline is not None and time.perf_counter() >= deadline:
                    break

                mask = env.get_mask()
                candidate_actions = []
                for node_idx in range(node_count):
                    if env.visited[0, node_idx]:
                        continue
                    normal_action = node_idx * 2
                    fast_action = normal_action + 1
                    if mask[0, normal_action]:
                        candidate_actions.append(normal_action)
                    if mask[0, fast_action]:
                        candidate_actions.append(fast_action)

                if candidate_actions:
                    best_action = None
                    best_score = None
                    for action in candidate_actions:
                        node_idx = int(action) // 2
                        node = nodes[0, node_idx]
                        speed = V_FAST if int(action) % 2 else V_NORM
                        dist = torch.norm(node[:2] - env.pos[0]).item() * GRID_SCALE
                        arrival = float(env.time[0].item()) + dist / speed / MAX_TIME
                        start = max(arrival, float(node[3].item()))
                        slack = float(node[4].item()) - start
                        score = (
                            params["step"] * env.estimate_action_objective(int(action))
                            + params["due"] * float(node[4].item())
                            + params["dist"] * (dist / GRID_SCALE)
                            + params["slack"] * max(0.0, -slack)
                            + params["fast_bias"] * (1.0 if int(action) % 2 else 0.0)
                        )
                        if best_score is None or score < best_score:
                            best_score = score
                            best_action = int(action)
                    selected = best_action
                else:
                    return_actions = [node_count * 2, node_count * 2 + 1]
                    feasible_returns = [action for action in return_actions if mask[0, action]]
                    if not feasible_returns:
                        break
                    selected = min(
                        feasible_returns,
                        key=lambda action: env.estimate_action_objective(int(action)),
                    )

                env.step(torch.tensor([int(selected)], device=device))
                actions.append(int(selected))

            unvisit = (~env.visited).sum(-1).float()
            total_cost = env.phys * PHYS_WEIGHT + env.pen * PEN_WEIGHT + unvisit * 20.0
            feasible = bool(
                unvisit.eq(0).all().item()
                and env._at_depot().all().item()
                and env._compute_done().all().item()
            )
            candidate = {
                "actions": [int(action) for action in actions],
                "route_actions": [int(action) for action in actions],
                "route_buckets": self._actions_to_route_buckets(actions, node_count, visited_mask=visited_mask),
                "priority": self._extract_customer_priority(
                    torch.tensor(actions, dtype=torch.long, device=device),
                    node_count,
                    visited_mask=visited_mask,
                ),
                "cost": float(total_cost.item()),
                "phys": float(env.phys.item()),
                "pen": float(env.pen.item()),
                "unvisit": float(unvisit.item()),
                "feasible": feasible,
                "init_source": "greedy_rebuild",
            }
            if self._is_better_solution(candidate, best):
                best = candidate

        return best

    def _solution_to_route_buckets(self, solution, nodes, visited_mask):
        if solution.get("route_buckets"):
            return self._normalize_route_buckets(solution["route_buckets"], nodes.size(1), visited_mask)
        actions = solution.get("actions") or solution.get("route_actions") or []
        buckets = self._actions_to_route_buckets(actions, nodes.size(1), visited_mask=visited_mask)
        if not buckets:
            buckets = [list(solution.get("priority", []))]
        return self._normalize_route_buckets(buckets, nodes.size(1), visited_mask)

    def _route_level_improve(self, solution, evaluate_buckets_fn, nodes, visited_mask, max_moves=80, deadline=None):
        buckets = self._solution_to_route_buckets(solution, nodes, visited_mask)
        best = evaluate_buckets_fn(buckets)
        if best is None:
            return solution

        def consider(candidate_buckets):
            nonlocal best
            if deadline is not None and time.perf_counter() >= deadline:
                return False
            candidate = evaluate_buckets_fn(candidate_buckets)
            if candidate is not None and candidate["cost"] + 1e-9 < best["cost"]:
                best = candidate
                return True
            return False

        move_count = 0
        improved = True
        while improved and move_count < max_moves:
            improved = False
            current = self._solution_to_route_buckets(best, nodes, visited_mask)
            if len(current) <= 0:
                break

            sorted_current = sort_route_buckets_by_time(nodes, current)
            if self._bucket_key(sorted_current) != self._bucket_key(current):
                move_count += 1
                if consider(sorted_current):
                    improved = True
                    continue

            moves = []
            route_count = len(current)
            for src_r, route in enumerate(current):
                for src_i in range(len(route)):
                    for dst_r in range(route_count + 1):
                        dst_len = 0 if dst_r == route_count else len(current[dst_r])
                        for dst_i in range(dst_len + 1):
                            if src_r == dst_r and (dst_i == src_i or dst_i == src_i + 1):
                                continue
                            moves.append(("relocate", src_r, src_i, dst_r, dst_i))
            for r1 in range(route_count):
                for r2 in range(r1 + 1, route_count):
                    for i in range(len(current[r1])):
                        for j in range(len(current[r2])):
                            moves.append(("swap", r1, i, r2, j))
                    if len(current[r1]) >= 2 and len(current[r2]) >= 2:
                        for i in range(1, len(current[r1])):
                            for j in range(1, len(current[r2])):
                                moves.append(("cross", r1, i, r2, j))
            for r, route in enumerate(current):
                if len(route) >= 4:
                    for i in range(len(route) - 2):
                        for j in range(i + 2, len(route) + 1):
                            moves.append(("reverse", r, i, j, -1))
                if len(route) >= 4:
                    for cut in range(1, len(route)):
                        moves.append(("split", r, cut, -1, -1))
            for r1 in range(route_count):
                for r2 in range(route_count):
                    if r1 != r2:
                        moves.append(("merge", r1, r2, -1, -1))

            random.shuffle(moves)
            for move in moves:
                if move_count >= max_moves:
                    break
                if deadline is not None and time.perf_counter() >= deadline:
                    return best
                move_count += 1
                kind, a, b, c, d = move
                candidate = [list(route) for route in current]
                if kind == "relocate":
                    src_r, src_i, dst_r, dst_i = a, b, c, d
                    if src_r >= len(candidate) or src_i >= len(candidate[src_r]):
                        continue
                    node = candidate[src_r].pop(src_i)
                    if not candidate[src_r]:
                        del candidate[src_r]
                        if dst_r > src_r:
                            dst_r -= 1
                    if dst_r >= len(candidate):
                        candidate.append([node])
                    else:
                        if src_r == dst_r and dst_i > src_i:
                            dst_i -= 1
                        dst_i = max(0, min(dst_i, len(candidate[dst_r])))
                        candidate[dst_r].insert(dst_i, node)
                elif kind == "swap":
                    r1, i, r2, j = a, b, c, d
                    if r1 >= len(candidate) or r2 >= len(candidate):
                        continue
                    if i >= len(candidate[r1]) or j >= len(candidate[r2]):
                        continue
                    candidate[r1][i], candidate[r2][j] = candidate[r2][j], candidate[r1][i]
                elif kind == "cross":
                    r1, i, r2, j = a, b, c, d
                    if r1 >= len(candidate) or r2 >= len(candidate):
                        continue
                    left = candidate[r1][:i] + candidate[r2][j:]
                    right = candidate[r2][:j] + candidate[r1][i:]
                    candidate[r1] = left
                    candidate[r2] = right
                elif kind == "reverse":
                    r, i, j = a, b, c
                    if r >= len(candidate) or j > len(candidate[r]):
                        continue
                    candidate[r] = candidate[r][:i] + list(reversed(candidate[r][i:j])) + candidate[r][j:]
                elif kind == "split":
                    r, cut = a, b
                    if r >= len(candidate) or cut <= 0 or cut >= len(candidate[r]):
                        continue
                    left = candidate[r][:cut]
                    right = candidate[r][cut:]
                    candidate = candidate[:r] + [left, right] + candidate[r + 1 :]
                elif kind == "merge":
                    r1, r2 = a, b
                    if r1 >= len(candidate) or r2 >= len(candidate) or r1 == r2:
                        continue
                    merged = candidate[r1] + candidate[r2]
                    candidate[r1] = merged
                    del candidate[r2]
                if consider(candidate):
                    improved = True
                    break
        return best

    def _route_compact_improve(self, solution, evaluate_buckets_fn, nodes, visited_mask, depot=None, max_moves=80, deadline=None):
        """Deterministically remove unnecessary depot returns.

        DRL tends to produce safe but over-segmented UAV trips on N50. Random
        ALNS can eventually merge those trips, but a targeted compaction pass is
        much cheaper and more reliable under a 5-15 second repair budget.
        """
        buckets = self._solution_to_route_buckets(solution, nodes, visited_mask)
        best = evaluate_buckets_fn(buckets)
        if best is None:
            return solution

        def consider(candidate_buckets):
            nonlocal best
            if deadline is not None and time.perf_counter() >= deadline:
                return False
            candidate = evaluate_buckets_fn(candidate_buckets)
            if candidate is not None and self._is_better_solution(candidate, best):
                best = candidate
                return True
            return False

        move_count = 0
        improved = True
        while improved and move_count < max_moves:
            improved = False
            current = self._solution_to_route_buckets(best, nodes, visited_mask)
            route_count = len(current)
            if route_count <= 1:
                break

            # Cheap deterministic 2-opt candidates inside each route. Random
            # local search can miss obvious geometric crossings under a tight
            # 15s budget, so rank a small set by Euclidean saving and let the
            # exact environment evaluator decide whether the change is valid.
            two_opt_moves = []
            depot_xy = (
                depot[0].to(device=nodes.device, dtype=nodes.dtype)
                if depot is not None
                else torch.zeros(2, device=nodes.device, dtype=nodes.dtype)
            )
            for r, route in enumerate(current):
                if len(route) < 4:
                    continue
                for i in range(len(route) - 2):
                    for j in range(i + 2, len(route) + 1):
                        if i == 0:
                            before_i = depot_xy
                        else:
                            before_i = nodes[0, route[i - 1], :2]
                        first = nodes[0, route[i], :2]
                        last = nodes[0, route[j - 1], :2]
                        if j == len(route):
                            after_j = depot_xy
                        else:
                            after_j = nodes[0, route[j], :2]
                        old_dist = torch.norm(before_i - first) + torch.norm(last - after_j)
                        new_dist = torch.norm(before_i - last) + torch.norm(first - after_j)
                        saving = float((old_dist - new_dist).item())
                        if saving > 1e-6:
                            two_opt_moves.append((-saving, r, i, j))
            two_opt_moves.sort(key=lambda item: item[0])
            for _, r, i, j in two_opt_moves[: max(4, min(24, route_count * 2))]:
                if move_count >= max_moves:
                    break
                candidate = [list(route) for route in current]
                candidate[r] = candidate[r][:i] + list(reversed(candidate[r][i:j])) + candidate[r][j:]
                move_count += 1
                if consider(candidate):
                    improved = True
                    break
            if improved:
                continue

            # First try adjacent route merges. These correspond exactly to
            # deleting one return-to-depot boundary in the action sequence.
            for idx in range(route_count - 1):
                if move_count >= max_moves:
                    break
                candidate = [list(route) for route in current]
                candidate[idx] = candidate[idx] + candidate[idx + 1]
                del candidate[idx + 1]
                move_count += 1
                if consider(candidate):
                    improved = True
                    break
            if improved:
                continue

            # Then try moving short routes into neighboring routes. Many elite
            # rollouts create 1-2 customer trips that can be absorbed safely.
            short_routes = sorted(range(route_count), key=lambda r: len(current[r]))
            for src_r in short_routes:
                if move_count >= max_moves:
                    break
                if len(current[src_r]) > 2:
                    continue
                for dst_r in (src_r - 1, src_r + 1):
                    if move_count >= max_moves:
                        break
                    if dst_r < 0 or dst_r >= route_count or dst_r == src_r:
                        continue
                    for at_front in (False, True):
                        candidate = [list(route) for route in current]
                        moving = candidate[src_r]
                        if at_front:
                            candidate[dst_r] = moving + candidate[dst_r]
                        else:
                            candidate[dst_r] = candidate[dst_r] + moving
                        del candidate[src_r]
                        move_count += 1
                        if consider(candidate):
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                continue

            # Finally test a few non-adjacent merges ranked by route endpoints.
            ranked_pairs = []
            for i in range(route_count):
                for j in range(i + 1, route_count):
                    tail = current[i][-1]
                    head = current[j][0]
                    rev_tail = current[j][-1]
                    rev_head = current[i][0]
                    d_forward = float(torch.norm(nodes[0, tail, :2] - nodes[0, head, :2]).item())
                    d_reverse = float(torch.norm(nodes[0, rev_tail, :2] - nodes[0, rev_head, :2]).item())
                    ranked_pairs.append((d_forward, i, j, False))
                    ranked_pairs.append((d_reverse, i, j, True))
            ranked_pairs.sort(key=lambda item: item[0])
            for _, i, j, reverse_order in ranked_pairs[: max(4, min(24, route_count * 2))]:
                if move_count >= max_moves:
                    break
                candidate = [list(route) for route in current]
                if reverse_order:
                    candidate[i] = candidate[j] + candidate[i]
                else:
                    candidate[i] = candidate[i] + candidate[j]
                del candidate[j]
                move_count += 1
                if consider(candidate):
                    improved = True
                    break
        return best

    def _route_exchange_improve(
        self,
        solution,
        evaluate_buckets_fn,
        nodes,
        depot,
        visited_mask,
        max_moves=160,
        candidate_pool=320,
        deadline=None,
    ):
        """Intensify across route boundaries with ranked exact-eval moves.

        The stochastic ALNS loop is good at escaping a basin, but under a 15s
        budget it can waste many evaluations on weak cross-route moves. This
        pass builds a large route-exchange candidate pool using a cheap proxy,
        then spends exact environment replays only on the most promising moves.
        """
        buckets = self._solution_to_route_buckets(solution, nodes, visited_mask)
        best = evaluate_buckets_fn(buckets)
        if best is None:
            return solution

        def arc_cost(from_node, to_node):
            return self._proxy_priority_arc_cost(nodes, depot, from_node, to_node)

        def node_at(route, pos):
            if pos < 0 or pos >= len(route):
                return None
            return route[pos]

        def route_remove_delta(route, start, length):
            segment = route[start : start + length]
            prev_node = node_at(route, start - 1)
            next_node = node_at(route, start + length)
            old_cost = arc_cost(prev_node, segment[0]) + arc_cost(segment[-1], next_node)
            new_cost = arc_cost(prev_node, next_node)
            return new_cost - old_cost

        def route_insert_delta(route, pos, segment):
            prev_node = node_at(route, pos - 1)
            next_node = node_at(route, pos)
            old_cost = arc_cost(prev_node, next_node)
            new_cost = arc_cost(prev_node, segment[0]) + arc_cost(segment[-1], next_node)
            return new_cost - old_cost

        def consider(candidate_buckets):
            nonlocal best
            if deadline is not None and time.perf_counter() >= deadline:
                return False
            candidate = evaluate_buckets_fn(candidate_buckets)
            if candidate is not None and self._is_better_solution(candidate, best):
                best = candidate
                return True
            return False

        move_count = 0
        improved = True
        while improved and move_count < max_moves:
            improved = False
            current = self._solution_to_route_buckets(best, nodes, visited_mask)
            route_count = len(current)
            if route_count <= 1:
                break

            scored_moves = []

            # Single-customer relocate across routes. This is the workhorse for
            # repairing over-conservative trip boundaries.
            for src_r, route in enumerate(current):
                for src_i, node in enumerate(route):
                    remove_delta = route_remove_delta(route, src_i, 1)
                    for dst_r, dst_route in enumerate(current):
                        if dst_r == src_r:
                            continue
                        positions = self._rank_insert_positions(dst_route, node, nodes, depot, max_positions=5)
                        for dst_i in positions:
                            insert_delta = route_insert_delta(dst_route, dst_i, [node])
                            due_bias = 0.002 * float(nodes[0, int(node), 4].item())
                            scored_moves.append((remove_delta + insert_delta + due_bias, "relocate", src_r, src_i, 1, dst_r, dst_i, False))

            # Short segment relocate, with forward and reversed segment order.
            for src_r, route in enumerate(current):
                for seg_len in (2, 3):
                    if len(route) < seg_len:
                        continue
                    for src_i in range(len(route) - seg_len + 1):
                        segment = route[src_i : src_i + seg_len]
                        remove_delta = route_remove_delta(route, src_i, seg_len)
                        for dst_r, dst_route in enumerate(current):
                            if dst_r == src_r:
                                continue
                            # Try a few proxy-good positions using the segment head as anchor.
                            anchor = segment[0]
                            positions = self._rank_insert_positions(dst_route, anchor, nodes, depot, max_positions=4)
                            for dst_i in positions:
                                fwd_delta = remove_delta + route_insert_delta(dst_route, dst_i, segment)
                                scored_moves.append((fwd_delta, "relocate", src_r, src_i, seg_len, dst_r, dst_i, False))
                                rev_segment = list(reversed(segment))
                                rev_delta = remove_delta + route_insert_delta(dst_route, dst_i, rev_segment)
                                scored_moves.append((rev_delta, "relocate", src_r, src_i, seg_len, dst_r, dst_i, True))

            # Swap awkward customers between different routes.
            for r1 in range(route_count):
                for r2 in range(r1 + 1, route_count):
                    route1 = current[r1]
                    route2 = current[r2]
                    for i, a in enumerate(route1):
                        prev_a = node_at(route1, i - 1)
                        next_a = node_at(route1, i + 1)
                        old_a = arc_cost(prev_a, a) + arc_cost(a, next_a)
                        for j, b in enumerate(route2):
                            prev_b = node_at(route2, j - 1)
                            next_b = node_at(route2, j + 1)
                            old_b = arc_cost(prev_b, b) + arc_cost(b, next_b)
                            new_a = arc_cost(prev_a, b) + arc_cost(b, next_a)
                            new_b = arc_cost(prev_b, a) + arc_cost(a, next_b)
                            due_bias = 0.001 * abs(float(nodes[0, int(a), 4].item()) - float(nodes[0, int(b), 4].item()))
                            scored_moves.append((new_a + new_b - old_a - old_b + due_bias, "swap", r1, i, 1, r2, j, False))

            scored_moves.sort(key=lambda item: item[0])
            for _, kind, src_r, src_i, seg_len, dst_r, dst_i, reverse_segment in scored_moves[: max(8, int(candidate_pool))]:
                if move_count >= max_moves:
                    break
                if deadline is not None and time.perf_counter() >= deadline:
                    return best
                candidate = [list(route) for route in current]
                if kind == "relocate":
                    if src_r >= len(candidate) or dst_r >= len(candidate) or src_r == dst_r:
                        continue
                    if src_i < 0 or src_i + seg_len > len(candidate[src_r]):
                        continue
                    segment = candidate[src_r][src_i : src_i + seg_len]
                    del candidate[src_r][src_i : src_i + seg_len]
                    if not candidate[src_r]:
                        del candidate[src_r]
                        if dst_r > src_r:
                            dst_r -= 1
                    if dst_r < 0 or dst_r >= len(candidate):
                        continue
                    if reverse_segment:
                        segment = list(reversed(segment))
                    dst_i = max(0, min(dst_i, len(candidate[dst_r])))
                    candidate[dst_r][dst_i:dst_i] = segment
                elif kind == "swap":
                    if src_r >= len(candidate) or dst_r >= len(candidate):
                        continue
                    if src_i >= len(candidate[src_r]) or dst_i >= len(candidate[dst_r]):
                        continue
                    candidate[src_r][src_i], candidate[dst_r][dst_i] = candidate[dst_r][dst_i], candidate[src_r][src_i]
                else:
                    continue
                move_count += 1
                if consider(candidate):
                    improved = True
                    break
        return best

    def _local_swap_improve(self, priority, evaluate_priority_fn, max_neighbors=12, deadline=None):
        route = list(priority)
        if len(route) <= 2:
            return route
        best_eval = evaluate_priority_fn(route)
        improved = True
        while improved:
            improved = False
            pairs = []
            for i in range(len(route) - 1):
                for j in range(i + 1, len(route)):
                    pairs.append((i, j))
            random.shuffle(pairs)
            pairs = pairs[:max_neighbors] if max_neighbors > 0 else pairs
            for i, j in pairs:
                if deadline is not None and time.perf_counter() >= deadline:
                    return route
                cand = list(route)
                cand[i], cand[j] = cand[j], cand[i]
                cand_eval = evaluate_priority_fn(cand)
                if cand_eval["cost"] + 1e-9 < best_eval["cost"]:
                    route = cand
                    best_eval = cand_eval
                    improved = True
                    break
        return route

    def _local_relocate_improve(self, priority, evaluate_priority_fn, max_moves=24, deadline=None):
        route = list(priority)
        if len(route) <= 2:
            return route
        best_eval = evaluate_priority_fn(route)
        improved = True
        while improved:
            improved = False
            moves = []
            for src in range(len(route)):
                for dst in range(len(route) + 1):
                    if dst == src or dst == src + 1:
                        continue
                    moves.append((src, dst))
            random.shuffle(moves)
            moves = moves[:max_moves] if max_moves > 0 else moves
            for src, dst in moves:
                if deadline is not None and time.perf_counter() >= deadline:
                    return route
                cand = list(route)
                node = cand.pop(src)
                if dst > src:
                    dst -= 1
                cand.insert(dst, node)
                cand_eval = evaluate_priority_fn(cand)
                if cand_eval["cost"] + 1e-9 < best_eval["cost"]:
                    route = cand
                    best_eval = cand_eval
                    improved = True
                    break
        return route

    def _local_2opt_improve(self, priority, evaluate_priority_fn, max_moves=24, deadline=None):
        route = list(priority)
        if len(route) <= 3:
            return route
        best_eval = evaluate_priority_fn(route)
        improved = True
        while improved:
            improved = False
            moves = []
            for i in range(len(route) - 2):
                for j in range(i + 2, len(route)):
                    moves.append((i, j))
            random.shuffle(moves)
            moves = moves[:max_moves] if max_moves > 0 else moves
            for i, j in moves:
                if deadline is not None and time.perf_counter() >= deadline:
                    return route
                cand = route[:i] + list(reversed(route[i:j])) + route[j:]
                cand_eval = evaluate_priority_fn(cand)
                if cand_eval["cost"] + 1e-9 < best_eval["cost"]:
                    route = cand
                    best_eval = cand_eval
                    improved = True
                    break
        return route

    def _double_bridge_shake(self, priority):
        route = list(priority)
        if len(route) < 8:
            random.shuffle(route)
            return route
        cuts = sorted(random.sample(range(1, len(route)), 4))
        a, b, c, d = cuts
        p1 = route[:a]
        p2 = route[a:b]
        p3 = route[b:c]
        p4 = route[c:d]
        p5 = route[d:]
        variants = [
            p1 + p3 + p2 + p4 + p5,
            p1 + p4 + p3 + p2 + p5,
            p1 + list(reversed(p3)) + p2 + p4 + p5,
            p1 + p2 + p4 + p3 + p5,
        ]
        return random.choice(variants)

    def _randomized_seed_priority(self, nodes, depot, visited_mask):
        remaining = self._remaining_nodes(nodes, visited_mask)
        if not remaining:
            return []
        node_data = nodes[0]
        depot_xy = depot[0]
        mode = random.choice(["due_noise", "sweep_noise", "nearest_noise"])
        if mode == "due_noise":
            return sorted(
                remaining,
                key=lambda idx: (
                    float(node_data[idx, 4].item())
                    + random.uniform(-0.15, 0.15)
                    + 0.05 * float(torch.norm(node_data[idx, :2] - depot_xy).item())
                ),
            )
        if mode == "sweep_noise":
            angle_rows = []
            offset = random.uniform(-math.pi, math.pi)
            for idx in remaining:
                rel = node_data[idx, :2] - depot_xy
                angle = math.atan2(float(rel[1].item()), float(rel[0].item())) + offset
                angle_rows.append((angle % (2.0 * math.pi), random.random(), idx))
            return [idx for _, _, idx in sorted(angle_rows)]

        priority = []
        unused = set(remaining)
        pos = depot_xy.clone()
        while unused:
            ranked = sorted(
                unused,
                key=lambda idx: (
                    float(torch.norm(node_data[idx, :2] - pos).item())
                    + 0.10 * float(node_data[idx, 4].item())
                    + random.uniform(0.0, 0.20)
                ),
            )
            pick = ranked[min(len(ranked) - 1, random.randint(0, min(3, len(ranked) - 1)))]
            priority.append(pick)
            unused.remove(pick)
            pos = node_data[pick, :2]
        return priority

    def get_initial_solution(
        self,
        nodes,
        pref,
        depot,
        visited_mask,
        curr_pos,
        curr_time,
        curr_load,
        curr_energy,
        n_pomo=N_POMO,
        decode_type="greedy",
        sample_rollouts=0,
        sample_temperature=1.0,
        multi_start_rollouts=0,
        augment_8_fold=False,
        initial_actions=None,
        init_source="auto",
    ):
        rollout = None
        use_model = self.model is not None and init_source in {"auto", "model"}
        if initial_actions is not None:
            action_tensor = torch.tensor(initial_actions, dtype=torch.long, device=self.device)
            priority = self._extract_customer_priority(
                action_tensor, nodes.size(1), visited_mask=visited_mask
            )
        elif use_model:
            with torch.no_grad():
                rollout = self.model.inference_rollout(
                    nodes,
                    pref,
                    depot,
                    visited_mask=visited_mask,
                    curr_pos=curr_pos,
                    curr_time=curr_time,
                    curr_load=curr_load,
                    curr_energy=curr_energy,
                    N_pomo=n_pomo,
                    decode_type=decode_type,
                    sample_rollouts=sample_rollouts,
                    sample_temperature=sample_temperature,
                    multi_start_rollouts=multi_start_rollouts,
                    augment_8_fold=augment_8_fold,
                )

            priority = self._extract_customer_priority(
                rollout["best_routes"][0], nodes.size(1), visited_mask=visited_mask
            )
        else:
            priority = self._build_heuristic_priority(
                nodes,
                pref,
                depot,
                visited_mask,
                curr_pos,
                curr_time,
                curr_load,
                curr_energy,
            )
        priority = self._complete_priority(priority, nodes.size(1), visited_mask)
        if initial_actions is not None:
            action_eval = self._evaluate_action_sequence(
                nodes,
                pref,
                depot,
                visited_mask,
                curr_pos,
                curr_time,
                curr_load,
                curr_energy,
                initial_actions,
            )
            priority_eval = self._simulate_priority(
                nodes,
                pref,
                depot,
                visited_mask,
                curr_pos,
                curr_time,
                curr_load,
                curr_energy,
                priority,
            )
            action_eval["original_action_cost"] = action_eval["cost"]
            action_eval["original_action_feasible"] = action_eval["feasible"]
            action_eval["priority_replay_cost"] = priority_eval["cost"]
            action_eval["priority_replay_feasible"] = priority_eval["feasible"]
            model_eval = action_eval
            if self._is_better_solution(priority_eval, model_eval):
                model_eval = priority_eval
                model_eval["original_action_cost"] = action_eval["cost"]
                model_eval["original_action_feasible"] = action_eval["feasible"]
                model_eval["route_actions"] = [int(x) for x in model_eval.get("actions", [])]
        else:
            model_eval = self._simulate_priority(
                nodes,
                pref,
                depot,
                visited_mask,
                curr_pos,
                curr_time,
                curr_load,
                curr_energy,
                priority,
            )
            model_eval["route_actions"] = (
                self._extract_route_actions_from_rollout(rollout)
                if rollout is not None
                else list(model_eval["actions"])
            )
        model_eval["model_rollout"] = rollout
        model_eval["init_source"] = "model" if rollout is not None else ("actions" if initial_actions is not None else "heuristic")
        return model_eval

    def optimize(
        self,
        nodes,
        pref,
        depot,
        visited_mask,
        curr_pos,
        curr_time,
        curr_load,
        curr_energy,
        max_alns_iter=100,
        time_limit_s=2.0,
        destroy_min=1,
        destroy_max=3,
        max_insert_positions=6,
        swap_neighbors=12,
        n_pomo=N_POMO,
        decode_type="greedy",
        sample_rollouts=0,
        sample_temperature=1.0,
        multi_start_rollouts=0,
        augment_8_fold=False,
        initial_actions=None,
        repair_mode="classic",
        guided_attempts=3,
        guided_min_keep_ratio=0.35,
        guided_decode_type="sample",
        guided_sample_rollouts=1,
        guided_pomo_size=None,
        guided_restarts=2,
        init_source="auto",
        restart_rounds=1,
    ):
        if nodes.size(0) != 1:
            raise ValueError("HybridALNSSolver currently supports batch_size=1 for ALNS experiments.")

        start_time = time.perf_counter()
        deadline = start_time + time_limit_s
        short_budget = time_limit_s <= 4.0
        preprocess_fraction = (0.22 if time_limit_s >= 3.0 else 0.10) if short_budget else 0.30
        preprocess_cap = (1.0 if time_limit_s >= 3.0 else 0.6) if short_budget else 3.0
        preprocess_deadline = start_time + min(max(0.15, preprocess_fraction * float(time_limit_s)), preprocess_cap)
        rounds = max(1, int(restart_rounds))
        if rounds > 1:
            global_initial = None
            global_best = None
            total_iterations = 0
            for round_idx in range(rounds):
                if time.perf_counter() >= deadline:
                    break
                remaining = deadline - time.perf_counter()
                round_budget = max(0.1, remaining / max(1, rounds - round_idx))
                # Portfolio restarts deliberately enter different basins:
                # 0) DRL action seed, 1) heuristic construction seed,
                # 2) sampled model seed. The solver still returns the best
                # exact-env replay under the same global wall-clock deadline.
                round_initial_actions = initial_actions
                round_init_source = init_source
                round_decode_type = decode_type
                round_sample_rollouts = sample_rollouts
                round_multi_start_rollouts = multi_start_rollouts
                if round_idx % 3 == 1:
                    round_initial_actions = None
                    round_init_source = "heuristic"
                    round_sample_rollouts = 0
                    round_multi_start_rollouts = 0
                elif round_idx % 3 == 2:
                    round_initial_actions = None
                    round_init_source = "auto"
                    round_decode_type = "sample" if self.model is not None else decode_type
                    round_sample_rollouts = max(1, int(sample_rollouts))
                    round_multi_start_rollouts = max(1, int(multi_start_rollouts))
                round_result = self.optimize(
                    nodes=nodes,
                    pref=pref,
                    depot=depot,
                    visited_mask=visited_mask,
                    curr_pos=curr_pos,
                    curr_time=curr_time,
                    curr_load=curr_load,
                    curr_energy=curr_energy,
                    max_alns_iter=max(1, int(max_alns_iter / rounds)),
                    time_limit_s=round_budget,
                    destroy_min=destroy_min,
                    destroy_max=destroy_max,
                    max_insert_positions=max_insert_positions,
                    swap_neighbors=swap_neighbors,
                    n_pomo=n_pomo,
                    decode_type=round_decode_type,
                    sample_rollouts=round_sample_rollouts,
                    sample_temperature=sample_temperature,
                    multi_start_rollouts=round_multi_start_rollouts,
                    augment_8_fold=augment_8_fold,
                    initial_actions=round_initial_actions,
                    repair_mode=repair_mode,
                    guided_attempts=guided_attempts,
                    guided_min_keep_ratio=guided_min_keep_ratio,
                    guided_decode_type=guided_decode_type,
                    guided_sample_rollouts=guided_sample_rollouts,
                    guided_pomo_size=guided_pomo_size,
                    guided_restarts=guided_restarts,
                    init_source=round_init_source,
                    restart_rounds=1,
                )
                total_iterations += int(round_result.get("iterations", 0))
                if global_initial is None:
                    global_initial = round_result["initial_solution"]
                candidate = round_result["best_solution"]
                if global_best is None or candidate["cost"] < global_best["cost"]:
                    global_best = candidate
            return {
                "initial_solution": global_initial if global_initial is not None else {},
                "best_solution": global_best if global_best is not None else {},
                "iterations": total_iterations,
                "elapsed_s": time.perf_counter() - start_time,
            }

        initial = self.get_initial_solution(
            nodes,
            pref,
            depot,
            visited_mask,
            curr_pos,
            curr_time,
            curr_load,
            curr_energy,
            n_pomo=n_pomo,
            decode_type=decode_type,
            sample_rollouts=sample_rollouts,
            sample_temperature=sample_temperature,
            multi_start_rollouts=multi_start_rollouts,
            augment_8_fold=augment_8_fold,
            initial_actions=initial_actions,
            init_source=init_source,
        )
        current = dict(initial)
        best = dict(initial)
        if initial_actions is not None and time.perf_counter() < preprocess_deadline:
            polished = self._polish_action_sequence(
                best,
                nodes,
                pref,
                depot,
                visited_mask,
                curr_pos,
                curr_time,
                curr_load,
                curr_energy,
                deadline=preprocess_deadline,
                max_trials=(80 if short_budget else 240),
                max_passes=(1 if short_budget else 2),
            )
            if self._is_better_solution(polished, best):
                best = polished
                current = polished
        sim_cache = {tuple(initial["priority"]): initial}
        if tuple(best.get("priority", [])) not in sim_cache:
            sim_cache[tuple(best.get("priority", []))] = best
        bucket_cache = {}
        effective_max_insert_positions = max_insert_positions
        if effective_max_insert_positions is None or effective_max_insert_positions <= 0:
            if time_limit_s <= 6:
                effective_max_insert_positions = 4
            elif time_limit_s <= 20:
                effective_max_insert_positions = 10
            else:
                effective_max_insert_positions = 22
        if time_limit_s <= 6:
            # Elite DRL routes are already locally polished. Very small
            # destroy sizes mostly rediscover the same basin, so short-budget
            # repair needs a slightly bolder shake while still respecting the
            # wall-clock deadline.
            effective_destroy_max = min(destroy_max, 8)
            effective_swap_neighbors = min(swap_neighbors, 12)
        elif time_limit_s <= 20:
            effective_destroy_max = min(destroy_max, 16)
            effective_swap_neighbors = min(swap_neighbors, 36)
        else:
            effective_destroy_max = destroy_max
            effective_swap_neighbors = swap_neighbors
        use_exact_short_repair = short_budget and time_limit_s >= 3.0
        use_beam_split = False
        # Route-exchange is useful for long quality runs, but the exact replay
        # evaluations are too expensive for the 15s real-time protocol. Keep it
        # out of short runs so stochastic LNS gets the budget.
        use_route_exchange = time_limit_s > 20.0

        def evaluate_priority_fn(priority_list):
            key = tuple(priority_list)
            if key in sim_cache:
                return sim_cache[key]
            value = self._simulate_priority(
                nodes,
                pref,
                depot,
                visited_mask,
                curr_pos,
                curr_time,
                curr_load,
                curr_energy,
                priority_list,
            )
            sim_cache[key] = value
            return value

        def evaluate_buckets_fn(route_buckets):
            normalized = self._normalize_route_buckets(route_buckets, nodes.size(1), visited_mask)
            key = self._bucket_key(normalized)
            if key in bucket_cache:
                return bucket_cache[key]
            value = self._evaluate_route_buckets(
                nodes,
                pref,
                depot,
                visited_mask,
                curr_pos,
                curr_time,
                curr_load,
                curr_energy,
                normalized,
            )
            bucket_cache[key] = value
            if value is not None:
                sim_cache[tuple(value["priority"])] = value
            return value

        def position_rank_fn(route, node, max_positions):
            return self._rank_insert_positions(route, node, nodes, depot, max_positions)

        # Short-budget quality hinges on starting from a good priority order.
        # Evaluate several deterministic seeds once, then let ALNS intensify the
        # best one. This is cheap compared with random destroy/repair misses.
        if time.perf_counter() < preprocess_deadline:
            for seed_priority in self._build_seed_priorities(
                nodes,
                pref,
                depot,
                visited_mask,
                curr_pos,
                curr_time,
                curr_load,
                curr_energy,
            ):
                if time.perf_counter() >= preprocess_deadline:
                    break
                candidate = evaluate_priority_fn(seed_priority)
                if candidate["cost"] < best["cost"]:
                    best = candidate
                    current = candidate

        if time.perf_counter() < preprocess_deadline:
            greedy_rebuilt = self._greedy_env_rebuild(
                nodes,
                pref,
                depot,
                visited_mask,
                curr_pos,
                curr_time,
                curr_load,
                curr_energy,
                deadline=preprocess_deadline,
            )
            if greedy_rebuilt is not None and self._is_better_solution(greedy_rebuilt, best):
                best = greedy_rebuilt
                current = greedy_rebuilt
                sim_cache[tuple(best.get("priority", []))] = best
                if best.get("route_buckets"):
                    bucket_cache[self._bucket_key(best["route_buckets"])] = best

        if (not short_budget) and time.perf_counter() < preprocess_deadline and len(best.get("priority", [])) > 3:
            proxy_priority = self._serial_proxy_search(
                nodes,
                pref,
                depot,
                best["priority"],
                deadline=preprocess_deadline,
                max_trials=(500 if short_budget else 3000),
            )
            candidate = evaluate_priority_fn(proxy_priority)
            if candidate["cost"] < best["cost"]:
                best = candidate
                current = candidate

        # Spend a small deterministic burst on the best seed before stochastic
        # ALNS. This often finds the same improvements that random search finds
        # later, but within a few seconds.
        if time.perf_counter() < preprocess_deadline and len(best["priority"]) > 2:
            intensify_moves = 120 if time_limit_s <= 6 else (220 if time_limit_s <= 12 else 420)
            intensified = self._local_relocate_improve(
                best["priority"],
                evaluate_priority_fn=evaluate_priority_fn,
                max_moves=max(3 * effective_swap_neighbors, intensify_moves),
                deadline=preprocess_deadline,
            )
            intensified = self._local_2opt_improve(
                intensified,
                evaluate_priority_fn=evaluate_priority_fn,
                max_moves=max(3 * effective_swap_neighbors, intensify_moves),
                deadline=preprocess_deadline,
            )
            candidate = evaluate_priority_fn(intensified)
            if candidate["cost"] < best["cost"]:
                best = candidate
                current = candidate
        if (
            ((not short_budget) or (initial_actions is not None and use_exact_short_repair))
            and time.perf_counter() < preprocess_deadline
            and len(best.get("priority", [])) > 2
        ):
            compact_candidate = self._route_compact_improve(
                best,
                evaluate_buckets_fn=evaluate_buckets_fn,
                nodes=nodes,
                visited_mask=visited_mask,
                depot=depot,
                max_moves=(28 if time_limit_s <= 6 else (56 if time_limit_s <= 15 else 120)),
                deadline=preprocess_deadline,
            )
            if compact_candidate is not None and compact_candidate["cost"] < best["cost"]:
                best = compact_candidate
                current = compact_candidate
            route_candidate = self._route_level_improve(
                best,
                evaluate_buckets_fn=evaluate_buckets_fn,
                nodes=nodes,
                visited_mask=visited_mask,
                max_moves=(18 if short_budget else max(4 * effective_swap_neighbors, 48)),
                deadline=preprocess_deadline,
            )
            if route_candidate is not None and route_candidate["cost"] < best["cost"]:
                best = route_candidate
                current = route_candidate
            if use_route_exchange:
                exchange_candidate = self._route_exchange_improve(
                    best,
                    evaluate_buckets_fn=evaluate_buckets_fn,
                    nodes=nodes,
                    depot=depot,
                    visited_mask=visited_mask,
                    max_moves=160,
                    candidate_pool=420,
                    deadline=preprocess_deadline,
                )
                if exchange_candidate is not None and exchange_candidate["cost"] < best["cost"]:
                    best = exchange_candidate
                    current = exchange_candidate

        if use_beam_split and time.perf_counter() < preprocess_deadline and len(best.get("priority", [])) > 2:
            beam_candidate = self._beam_simulate_priority(
                nodes,
                pref,
                depot,
                visited_mask,
                curr_pos,
                curr_time,
                curr_load,
                curr_energy,
                best["priority"],
                beam_width=(6 if time_limit_s <= 6 else 10),
                deadline=preprocess_deadline,
            )
            if self._is_better_solution(beam_candidate, best):
                best = beam_candidate
                current = beam_candidate

        temperature = max(1.0, 0.05 * float(best["cost"]))
        iter_count = 0
        no_improve_count = 0

        if repair_mode == "model_guided":
            guided = self._model_guided_tail_repair(
                nodes,
                pref,
                depot,
                visited_mask,
                curr_pos,
                curr_time,
                curr_load,
                curr_energy,
                base_actions=initial.get("route_actions") or [],
                deadline=deadline,
                n_pomo=n_pomo,
                decode_type=decode_type,
                sample_rollouts=sample_rollouts,
                sample_temperature=sample_temperature,
                guided_attempts=guided_attempts,
                min_keep_ratio=guided_min_keep_ratio,
                guided_decode_type=guided_decode_type,
                guided_sample_rollouts=guided_sample_rollouts,
                guided_pomo_size=guided_pomo_size,
                guided_restarts=guided_restarts,
            )
            iter_count = 1 if guided is not None else 0
            if guided is not None and guided["cost"] < best["cost"]:
                best = guided
                current = guided
            return {
                "initial_solution": initial,
                "best_solution": best,
                "iterations": iter_count,
                "elapsed_s": time.perf_counter() - start_time,
            }

        while iter_count < max_alns_iter and time.perf_counter() < deadline:
            iter_count += 1
            if len(current["priority"]) <= 1:
                break

            base_solution = best if random.random() < 0.35 else current
            base_priority = list(base_solution["priority"])
            if random.random() < 0.12 and len(base_priority) >= 8:
                base_priority = self._double_bridge_shake(base_priority)

            base_len = len(base_priority)
            if base_len <= 1:
                break
            large_destroy_fraction = 0.25 if time_limit_s <= 10 else 0.60
            if random.random() < 0.35 and base_len > effective_destroy_max:
                upper = min(base_len, max(effective_destroy_max + 1, int(math.ceil(base_len * large_destroy_fraction))))
                lower = min(upper, max(destroy_min, effective_destroy_max))
                destroy_count = random.randint(lower, upper)
            else:
                destroy_count = min(base_len, random.randint(destroy_min, effective_destroy_max))

            destroy_mode = random.choice([
                "random",
                "random",
                "contiguous",
                "tail",
                "worst_proxy",
                "worst_proxy",
                "route",
                "route",
            ])
            removed_nodes = []
            destroyed = list(base_priority)
            if destroy_mode == "route" and base_solution.get("route_buckets"):
                route_buckets = [list(route) for route in base_solution.get("route_buckets", []) if route]
                if route_buckets:
                    random.shuffle(route_buckets)
                    route_destroy_count = 1 if len(route_buckets) <= 1 else random.randint(1, min(2, len(route_buckets)))
                    removed_set = set()
                    for route in route_buckets[:route_destroy_count]:
                        removed_set.update(int(node) for node in route)
                    if removed_set:
                        removed_nodes = [node for node in base_priority if node in removed_set]
                        if len(removed_nodes) > effective_destroy_max:
                            removed_keep = set(random.sample(removed_nodes, effective_destroy_max))
                            removed_nodes = [node for node in removed_nodes if node in removed_keep]
                            removed_set = removed_keep
                        destroyed = [node for node in base_priority if node not in removed_set]
            if not removed_nodes and destroy_mode == "worst_proxy":
                remove_indices = self._worst_proxy_positions(
                    base_priority,
                    nodes,
                    depot,
                    destroy_count,
                )
            elif not removed_nodes and destroy_mode == "contiguous" and len(base_priority) > destroy_count:
                start_idx = random.randint(0, len(base_priority) - destroy_count)
                remove_indices = list(range(start_idx, start_idx + destroy_count))
            elif not removed_nodes and destroy_mode == "tail" and len(base_priority) > destroy_count:
                remove_indices = list(range(len(base_priority) - destroy_count, len(base_priority)))
            elif not removed_nodes:
                remove_indices = random.sample(range(len(base_priority)), destroy_count)
                remove_indices = sorted(remove_indices, reverse=True)
                destroyed = list(base_priority)
                removed_nodes = []
                for idx in remove_indices:
                    removed_nodes.append(destroyed.pop(idx))
                removed_nodes.reverse()

            use_regret = (
                (not short_budget and (random.random() < 0.55 or len(removed_nodes) >= max(4, destroy_max)))
                or (use_exact_short_repair and len(removed_nodes) >= 3 and random.random() < 0.35)
            )
            if short_budget and not use_exact_short_repair:
                candidate_priority = self._proxy_greedy_insert(
                    destroyed,
                    removed_nodes,
                    nodes=nodes,
                    depot=depot,
                    max_insert_positions=effective_max_insert_positions,
                )
            elif use_regret:
                candidate_priority = self._regret_insert(
                    destroyed,
                    removed_nodes,
                    evaluate_priority_fn=evaluate_priority_fn,
                    max_insert_positions=effective_max_insert_positions,
                    deadline=deadline,
                    position_rank_fn=position_rank_fn,
                    regret_weight=0.35,
                )
            else:
                candidate_priority = self._greedy_insert(
                    destroyed,
                    removed_nodes,
                    evaluate_priority_fn=evaluate_priority_fn,
                    max_insert_positions=effective_max_insert_positions,
                    deadline=deadline,
                    position_rank_fn=position_rank_fn,
                )
            if candidate_priority is None:
                continue
            if (not short_budget) or use_exact_short_repair:
                local_budget = effective_swap_neighbors if not short_budget else max(4, effective_swap_neighbors)
                candidate_priority = self._local_swap_improve(
                    candidate_priority,
                    evaluate_priority_fn=evaluate_priority_fn,
                    max_neighbors=local_budget,
                    deadline=deadline,
                )
                candidate_priority = self._local_relocate_improve(
                    candidate_priority,
                    evaluate_priority_fn=evaluate_priority_fn,
                    max_moves=(max(2 * local_budget, 6) if not short_budget else 8),
                    deadline=deadline,
                )
                candidate_priority = self._local_2opt_improve(
                    candidate_priority,
                    evaluate_priority_fn=evaluate_priority_fn,
                    max_moves=(max(2 * local_budget, 6) if not short_budget else 8),
                    deadline=deadline,
                )
            candidate = evaluate_priority_fn(candidate_priority)
            route_refine_prob = 0.10 if use_exact_short_repair else (0.0 if short_budget else 0.45)
            if time.perf_counter() < deadline and random.random() < route_refine_prob:
                if use_route_exchange and random.random() < 0.55:
                    route_candidate = self._route_exchange_improve(
                        candidate,
                        evaluate_buckets_fn=evaluate_buckets_fn,
                        nodes=nodes,
                        depot=depot,
                        visited_mask=visited_mask,
                        max_moves=(6 if short_budget else max(effective_swap_neighbors, 18)),
                        candidate_pool=(40 if short_budget else 120),
                        deadline=deadline,
                    )
                else:
                    route_candidate = self._route_level_improve(
                        candidate,
                        evaluate_buckets_fn=evaluate_buckets_fn,
                        nodes=nodes,
                        visited_mask=visited_mask,
                        max_moves=(6 if short_budget else max(effective_swap_neighbors, 16)),
                        deadline=deadline,
                    )
                if route_candidate is not None and route_candidate["cost"] < candidate["cost"]:
                    candidate = route_candidate
                    candidate_priority = list(candidate["priority"])

            delta = candidate["cost"] - current["cost"]
            accept = delta < 0 or random.random() < math.exp(-delta / max(temperature, 1e-6))
            if accept:
                current = candidate
            if candidate["cost"] < best["cost"]:
                best = candidate
                current = candidate
                no_improve_count = 0
                temperature = max(temperature, 0.03 * float(best["cost"]))
            else:
                no_improve_count += 1
                if no_improve_count >= max(30, len(best["priority"])):
                    shake_priority = self._complete_priority(
                        self._randomized_seed_priority(nodes, depot, visited_mask),
                        nodes.size(1),
                        visited_mask,
                    )
                    if random.random() < 0.5:
                        shake_priority = self._double_bridge_shake(best["priority"])
                    current = evaluate_priority_fn(shake_priority)
                    no_improve_count = 0
                    temperature = max(temperature, 0.04 * float(best["cost"]))
            temperature *= 0.995

        if time.perf_counter() < deadline and len(best.get("priority", [])) > 2:
            compact_candidate = self._route_compact_improve(
                best,
                evaluate_buckets_fn=evaluate_buckets_fn,
                nodes=nodes,
                visited_mask=visited_mask,
                depot=depot,
                max_moves=(24 if time_limit_s <= 6 else (48 if time_limit_s <= 15 else 96)),
                deadline=deadline,
            )
            if compact_candidate is not None and compact_candidate["cost"] < best["cost"]:
                best = compact_candidate
            if use_route_exchange:
                exchange_candidate = self._route_exchange_improve(
                    best,
                    evaluate_buckets_fn=evaluate_buckets_fn,
                    nodes=nodes,
                    depot=depot,
                    visited_mask=visited_mask,
                    max_moves=120,
                    candidate_pool=420,
                    deadline=deadline,
                )
                if exchange_candidate is not None and exchange_candidate["cost"] < best["cost"]:
                    best = exchange_candidate

        if use_beam_split and time.perf_counter() < deadline and len(best.get("priority", [])) > 2:
            beam_candidate = self._beam_simulate_priority(
                nodes,
                pref,
                depot,
                visited_mask,
                curr_pos,
                curr_time,
                curr_load,
                curr_energy,
                best["priority"],
                beam_width=(6 if time_limit_s <= 6 else 10),
                deadline=deadline,
            )
            if self._is_better_solution(beam_candidate, best):
                best = beam_candidate

        return {
            "initial_solution": initial,
            "best_solution": best,
            "iterations": iter_count,
            "elapsed_s": time.perf_counter() - start_time,
        }

    def simulate_prefix_state(
        self,
        nodes,
        pref,
        depot,
        route_actions,
        served_customers_target,
        visited_mask=None,
        curr_pos=None,
        curr_time=None,
        curr_load=None,
        curr_energy=None,
    ):
        batch_size = nodes.size(0)
        device = nodes.device
        if batch_size != 1:
            raise ValueError("simulate_prefix_state currently supports batch_size=1.")
        if visited_mask is None:
            visited_mask = torch.zeros(batch_size, nodes.size(1), dtype=torch.bool, device=device)
        if curr_pos is None:
            curr_pos = depot.clone()
        if curr_time is None:
            curr_time = torch.zeros(batch_size, device=device)
        if curr_load is None:
            curr_load = torch.ones(batch_size, device=device)
        if curr_energy is None:
            curr_energy = torch.ones(batch_size, device=device)

        env = VRPTWEnv(nodes.clone(), pref.clone(), depot.clone(), scenario=self.scenario)
        env.load_state(
            visited_mask.clone(),
            curr_pos.clone(),
            curr_time.clone(),
            curr_load.clone(),
            curr_energy.clone(),
        )

        served = 0
        executed_actions = []
        for action in route_actions.tolist():
            action = int(action)
            _, done = env.step(torch.tensor([action], device=device))
            executed_actions.append(action)
            if action < nodes.size(1) * 2:
                served += 1
            if served >= served_customers_target or done.all():
                break

        return {
            "visited_mask": env.visited.clone(),
            "curr_pos": env.pos.clone(),
            "curr_time": env.time.clone(),
            "curr_load": env.load.clone(),
            "curr_energy": env.energy.clone(),
            "executed_actions": executed_actions,
            "served_customers": served,
        }
