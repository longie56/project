import torch

from config import *
from env.energy_model import check_direct_return_feasible, compute_energy

AT_DEPOT_EPS = 1e-5


class VRPTWEnv:
    def __init__(self, nodes, pref, depot, scenario="full"):
        self.B, self.N, _ = nodes.shape
        self.dev = nodes.device
        self.nodes = nodes
        self.pref = pref
        self.depot = depot
        self.scenario = str(scenario).lower()
        if self.scenario not in {"full", "simple"}:
            raise ValueError(f"Unsupported scenario: {scenario}")

    def load_state(self, visited_mask, curr_pos, curr_time, curr_load, curr_energy):
        self.visited = visited_mask.clone().bool()
        self.pos = curr_pos.clone()
        self.time = curr_time.clone()
        self.load = curr_load.clone()
        self.energy = curr_energy.clone()
        self.phys = torch.zeros(self.B, device=self.dev)
        self.pen = torch.zeros(self.B, device=self.dev)

    def is_simple(self):
        return self.scenario == "simple"

    def _at_depot(self):
        return torch.norm(self.pos - self.depot, dim=-1) <= AT_DEPOT_EPS

    def _all_visited(self):
        return self.visited.all(dim=-1)

    def _customer_mask(self):
        mask = torch.zeros(self.B, self.N * 2, dtype=torch.bool, device=self.dev)
        demand = self.nodes[:, :, 2]
        can_visit = (~self.visited) & (self.load.unsqueeze(1) >= demand)

        # Base feasibility: unvisited and load enough.
        mask[:, 0::2] = can_visit
        if not self.is_simple():
            mask[:, 1::2] = can_visit

        node_xy = self.nodes[:, :, :2]
        dist = torch.norm(node_xy - self.pos.unsqueeze(1), dim=-1) * GRID_SCALE
        ready = self.nodes[:, :, 3]
        service = self.nodes[:, :, 5]
        unvisited_count = (~self.visited).sum(dim=-1, keepdim=True)
        is_last_customer = unvisited_count <= 1
        completion_norm = torch.maximum(self.time.unsqueeze(1) + dist / V_NORM / MAX_TIME, ready) + service
        time_ok_norm = is_last_customer | (completion_norm <= MAX_ROUTE_TIME_NORM)
        mask[:, 0::2] &= time_ok_norm
        if not self.is_simple():
            completion_fast = torch.maximum(self.time.unsqueeze(1) + dist / V_FAST / MAX_TIME, ready) + service
            time_ok_fast = is_last_customer | (completion_fast <= MAX_ROUTE_TIME_NORM)
            mask[:, 1::2] &= time_ok_fast

        if self.is_simple():
            return mask

        # Vectorized energy gate for all customers at once.
        pos_expand = self.pos.unsqueeze(1)  # [B,1,2]
        depot_expand = self.depot.unsqueeze(1)  # [B,1,2]
        load_expand = self.load.unsqueeze(1)  # [B,1]
        energy_expand = self.energy.unsqueeze(1)  # [B,1]
        return_load = torch.clamp(load_expand - demand, min=0.0)

        e_to_norm = compute_energy(pos_expand, node_xy, V_NORM, load_expand)
        e_to_fast = compute_energy(pos_expand, node_xy, V_FAST, load_expand)
        # Keep return-leg speed consistent with env energy gate semantics: always economical mode.
        e_back = compute_energy(node_xy, depot_expand, V_NORM, return_load)

        # A customer action consumes waiting hover energy too if it arrives before
        # the time window opens. The mask must be at least as strict as step().
        wait_norm = torch.relu(ready - (self.time.unsqueeze(1) + dist / V_NORM / MAX_TIME))
        wait_fast = torch.relu(ready - (self.time.unsqueeze(1) + dist / V_FAST / MAX_TIME))
        wait_e_norm = P_HOVER * wait_norm * MAX_TIME
        wait_e_fast = P_HOVER * wait_fast * MAX_TIME

        feasible_norm = (energy_expand * E_MAX - (e_to_norm + wait_e_norm + e_back)) / E_MAX >= E_SAFE_MARGIN
        feasible_fast = (energy_expand * E_MAX - (e_to_fast + wait_e_fast + e_back)) / E_MAX >= E_SAFE_MARGIN

        mask[:, 0::2] &= feasible_norm
        mask[:, 1::2] &= feasible_fast
        return mask

    def _return_mask(self):
        mask = torch.zeros(self.B, 2, dtype=torch.bool, device=self.dev)
        at_depot = self._at_depot()
        all_visited = self._all_visited()
        dist = torch.norm(self.depot - self.pos, dim=-1) * GRID_SCALE
        arrival_norm = self.time + dist / V_NORM / MAX_TIME
        time_ok_norm = all_visited | (arrival_norm <= MAX_ROUTE_TIME_NORM)
        if self.is_simple():
            mask[:, 0] = (~at_depot) & time_ok_norm
        else:
            mask[:, 0] = (~at_depot) & check_direct_return_feasible(
                self.energy, self.pos, self.depot, V_NORM, self.load
            ) & time_ok_norm
            arrival_fast = self.time + dist / V_FAST / MAX_TIME
            time_ok_fast = all_visited | (arrival_fast <= MAX_ROUTE_TIME_NORM)
            mask[:, 1] = (~at_depot) & check_direct_return_feasible(
                self.energy, self.pos, self.depot, V_FAST, self.load
            ) & time_ok_fast
        return mask

    def estimate_action_objective(self, action):
        batch_idx = 0
        n_actions = self.N * 2
        is_customer = int(action) < n_actions
        node_idx = min(int(action) // 2, self.N - 1)
        if self.is_simple():
            speed = V_NORM
            p_empty = P_EMPTY_NORM
        else:
            is_fast = int(action) % 2 == 1
            speed = V_FAST if is_fast else V_NORM
            p_empty = P_EMPTY_FAST if is_fast else P_EMPTY_NORM

        target = self.nodes[batch_idx, node_idx, :2] if is_customer else self.depot[batch_idx]
        dist = torch.norm(target - self.pos[batch_idx]) * GRID_SCALE
        t_flight = dist / speed
        t_norm = t_flight / MAX_TIME
        arrival = self.time[batch_idx] + t_norm

        pen_inc = torch.tensor(0.0, device=self.dev)
        if self.is_simple():
            phys_inc = t_norm * 5.0
            if is_customer:
                ready = self.nodes[batch_idx, node_idx, 3]
                due = self.nodes[batch_idx, node_idx, 4]
                start = torch.maximum(arrival, ready)
                pen_inc = torch.relu(start - due)
                phys_inc = phys_inc + torch.relu(start - arrival)
        else:
            e_flight = (p_empty + P_GAMMA * self.load[batch_idx] * MAX_PAYLOAD_KG) * t_flight
            wait_e = torch.tensor(0.0, device=self.dev)
            phys_inc = t_norm * 5.0 * self.pref[batch_idx, 0]
            if is_customer:
                ready = self.nodes[batch_idx, node_idx, 3]
                due = self.nodes[batch_idx, node_idx, 4]
                start = torch.maximum(arrival, ready)
                wait_e = P_HOVER * torch.relu(start - arrival) * MAX_TIME
                pen_inc = torch.relu(start - due)
            phys_inc = phys_inc + ((e_flight + wait_e) / E_MAX) * 10.0 * self.pref[batch_idx, 1]
        return float((PHYS_WEIGHT * phys_inc + PEN_WEIGHT * pen_inc).item())

    def step(self, action, inactive_mask=None):
        if inactive_mask is None:
            inactive_mask = torch.zeros(self.B, dtype=torch.bool, device=self.dev)

        batch_idx = torch.arange(self.B, device=self.dev)
        active_mask = ~inactive_mask
        is_node = (action < self.N * 2) & active_mask
        is_return = (~is_node) & active_mask
        node = torch.clamp(torch.div(action, 2, rounding_mode="trunc"), max=self.N - 1)
        if self.is_simple():
            v = torch.full_like(self.time, V_NORM)
            p_e = torch.full_like(self.time, P_EMPTY_NORM)
        else:
            v = torch.where(action % 2 == 1, V_FAST, V_NORM)
            p_e = torch.where(action % 2 == 1, P_EMPTY_FAST, P_EMPTY_NORM)

        tgt = self.pos.clone()
        tgt = torch.where(is_node.unsqueeze(-1), self.nodes[batch_idx, node, :2], tgt)
        tgt = torch.where(is_return.unsqueeze(-1), self.depot, tgt)

        dist = torch.zeros(self.B, device=self.dev)
        dist[active_mask] = torch.norm(tgt[active_mask] - self.pos[active_mask], dim=-1) * GRID_SCALE
        t_flight = dist / v
        t_norm = t_flight / MAX_TIME
        arr = self.time + t_norm
        e_flight = (p_e + P_GAMMA * self.load * MAX_PAYLOAD_KG) * t_flight

        if is_node.any():
            i_t = node[is_node]
            b_t = batch_idx[is_node]
            ready = self.nodes[b_t, i_t, 3]
            due = self.nodes[b_t, i_t, 4]
            serv = self.nodes[b_t, i_t, 5]
            start = torch.max(arr[is_node], ready)
            wait_delta = torch.relu(start - arr[is_node])
            wait_e = P_HOVER * wait_delta * MAX_TIME

            self.time[is_node] = start + serv
            if not self.is_simple():
                self.energy[is_node] -= (e_flight[is_node] + wait_e) / E_MAX
            self.load[is_node] = torch.clamp(self.load[is_node] - self.nodes[b_t, i_t, 2], min=0.0)
            self.visited[is_node, i_t] = True
            self.pen[is_node] += torch.relu(start - due)
            if self.is_simple():
                self.phys[is_node] += t_norm[is_node] * 5.0 + wait_delta
            else:
                self.phys[is_node] += (
                    (e_flight[is_node] + wait_e) / E_MAX * 10.0 * self.pref[is_node, 1]
                    + t_norm[is_node] * 5.0 * self.pref[is_node, 0]
                )

        if is_return.any():
            if self.is_simple():
                self.phys[is_return] += t_norm[is_return] * 5.0
            else:
                self.phys[is_return] += (
                    e_flight[is_return] / E_MAX * 10.0 * self.pref[is_return, 1]
                    + t_norm[is_return] * 5.0 * self.pref[is_return, 0]
                )
            self.time[is_return] = arr[is_return]
            self.energy[is_return] = 1.0
            self.load[is_return] = 1.0

        self.pos[active_mask] = tgt[active_mask]
        done = self._compute_done()
        return self._get_reward(), done

    def _compute_done(self):
        all_vis = self._all_visited()
        at_depot = self._at_depot()
        customer_mask = self._customer_mask()
        return_mask = self._return_mask()
        no_customer = ~customer_mask.any(dim=-1)
        no_return = ~return_mask.any(dim=-1)
        completed = all_vis & at_depot
        stuck_at_depot = at_depot & ~all_vis & no_customer
        stranded = ~at_depot & no_customer & no_return
        timeout_before_completion = (self.time > MAX_ROUTE_TIME_NORM) & ~all_vis
        return (
            completed
            | stuck_at_depot
            | stranded
            | ((self.energy < E_SAFE_MARGIN) if not self.is_simple() else torch.zeros_like(completed))
            | timeout_before_completion
        )

    def _get_reward(self):
        r = -self.phys * PHYS_WEIGHT - self.pen * PEN_WEIGHT
        r[self._all_visited() & self._at_depot()] += COMPLETION_BONUS
        return r - STEP_PENALTY

    def get_mask(self, inactive_mask=None):
        mask = torch.cat([self._customer_mask(), self._return_mask()], dim=-1)
        if inactive_mask is not None and inactive_mask.any():
            mask[inactive_mask] = False
            mask[inactive_mask, self.N * 2] = True

        at_depot = self._at_depot()
        all_masked = ~mask.any(dim=-1)
        mask[all_masked & at_depot, self.N * 2] = True
        return mask
