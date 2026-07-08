import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *
from env.energy_model import compute_energy

class Decoder(nn.Module):
    def __init__(
        self,
        d_model=D_MODEL,
        use_film=True,
        use_ega=True,
        use_mode_mask=True,
        use_action_features=True,
    ):
        super().__init__()
        self.Wq_ctx = nn.Linear(d_model, d_model, bias=False)
        self.Wq_state = nn.Linear(3, d_model, bias=False)
        self.Wq_pref = nn.Linear(2, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.temp = d_model ** 0.5
        self.use_film = use_film
        self.use_ega = use_ega
        self.use_mode_mask = use_mode_mask
        self.use_action_features = use_action_features
        self.action_condition = nn.Sequential(
            nn.Linear(d_model + 5, d_model),
            nn.ReLU(),
            nn.Linear(d_model, ACTION_SCORE_HIDDEN),
        )
        self.action_feature_mlp = nn.Sequential(
            nn.Linear(8, ACTION_SCORE_HIDDEN),
            nn.ReLU(),
            nn.Linear(ACTION_SCORE_HIDDEN, ACTION_SCORE_HIDDEN),
        )
        self.action_score_out = nn.Linear(ACTION_SCORE_HIDDEN, 1)

    def _mode_preference_bias(self, pref, n_nodes):
        device = pref.device
        dtype = pref.dtype
        batch = pref.size(0)
        time_minus_energy = (pref[:, 0] - pref[:, 1]).unsqueeze(1) * MODE_BIAS_SCALE
        bias = torch.zeros(batch, n_nodes * 2 + 2, device=device, dtype=dtype)
        bias[:, 0 : n_nodes * 2 : 2] = -time_minus_energy
        bias[:, 1 : n_nodes * 2 : 2] = time_minus_energy
        bias[:, n_nodes * 2] = -time_minus_energy.squeeze(1)
        bias[:, n_nodes * 2 + 1] = time_minus_energy.squeeze(1)
        return bias

    def _energy_gate_bias(self, curr_pos, node_pos, depot_pos, curr_load, curr_time, curr_energy):
        batch, n_nodes, _ = node_pos.shape
        device = node_pos.device
        dtype = node_pos.dtype

        expanded_depot = depot_pos.unsqueeze(1).expand(-1, 2, -1)
        target_pos = torch.cat([node_pos.repeat_interleave(2, dim=1), expanded_depot], dim=1)

        node_speeds = torch.tensor([V_NORM, V_FAST], device=device, dtype=dtype).repeat(n_nodes)
        action_speeds = torch.cat(
            [node_speeds, torch.tensor([V_NORM, V_FAST], device=device, dtype=dtype)], dim=0
        ).unsqueeze(0).expand(batch, -1)

        curr_pos_exp = curr_pos.unsqueeze(1).expand(-1, target_pos.size(1), -1)
        curr_load_exp = curr_load.unsqueeze(1).expand(-1, target_pos.size(1))
        flight_energy = compute_energy(
            curr_pos_exp.reshape(-1, 2),
            target_pos.reshape(-1, 2),
            action_speeds.reshape(-1),
            curr_load_exp.reshape(-1),
        ).reshape(batch, -1)

        customer_load_after = torch.clamp(curr_load.unsqueeze(1) - self.current_demands, min=0.0)
        depot_exp = depot_pos.unsqueeze(1).expand(-1, n_nodes, -1)
        return_energy_customer = compute_energy(
            node_pos.reshape(-1, 2),
            depot_exp.reshape(-1, 2),
            torch.full((batch * n_nodes,), V_NORM, device=device, dtype=dtype),
            customer_load_after.reshape(-1),
        ).reshape(batch, n_nodes)
        return_energy = torch.cat(
            [return_energy_customer.repeat_interleave(2, dim=1), torch.zeros(batch, 2, device=device, dtype=dtype)],
            dim=1,
        )

        travel_dist = torch.norm(target_pos - curr_pos.unsqueeze(1), dim=-1) * GRID_SCALE
        travel_time = (travel_dist / action_speeds) / MAX_TIME
        ready = torch.cat(
            [self.current_ready.repeat_interleave(2, dim=1), torch.zeros(batch, 2, device=device, dtype=dtype)],
            dim=1,
        )
        is_depot = torch.zeros(batch, n_nodes * 2 + 2, device=device, dtype=dtype)
        is_depot[:, -2:] = 1.0
        wait_time = torch.relu(ready - (curr_time.unsqueeze(1) + travel_time))
        wait_time = torch.where(is_depot > 0.5, torch.zeros_like(wait_time), wait_time)
        wait_energy = P_HOVER * wait_time * MAX_TIME

        projected_margin = curr_energy.unsqueeze(1) - (flight_energy + wait_energy + return_energy) / E_MAX
        gate = torch.sigmoid(EGA_GATE_SCALE * (projected_margin - E_SAFE_MARGIN) / E_SAFE_MARGIN)
        gate = gate.clamp_min(EGA_MIN_GATE)
        gate[:, -2:] = 1.0
        return torch.log(gate)

    def _action_feature_logits(self, current_ctx, state, curr_pos, node_pos, depot_pos, pref):
        batch, n_nodes, _ = node_pos.shape
        device = node_pos.device
        dtype = node_pos.dtype

        repeated_nodes = node_pos.repeat_interleave(2, dim=1)
        repeated_demands = self.current_demands.repeat_interleave(2, dim=1)
        expanded_depot = depot_pos.unsqueeze(1).expand(-1, 2, -1)
        target_pos = torch.cat([repeated_nodes, expanded_depot], dim=1)

        mode_fast = torch.zeros(batch, n_nodes * 2 + 2, device=device, dtype=dtype)
        mode_fast[:, 1 : n_nodes * 2 : 2] = 1.0
        mode_fast[:, -1] = 1.0
        action_speeds = torch.where(mode_fast > 0.5, V_FAST, V_NORM)

        curr_pos_exp = curr_pos.unsqueeze(1).expand(-1, target_pos.size(1), -1)
        curr_load_exp = state[:, 0].unsqueeze(1).expand(-1, target_pos.size(1))
        flight_energy = compute_energy(
            curr_pos_exp.reshape(-1, 2),
            target_pos.reshape(-1, 2),
            action_speeds.reshape(-1),
            curr_load_exp.reshape(-1),
        ).reshape(batch, -1) / E_MAX

        travel_dist = torch.norm(target_pos - curr_pos.unsqueeze(1), dim=-1) * GRID_SCALE
        travel_time = (travel_dist / action_speeds) / MAX_TIME

        ready = torch.cat(
            [self.current_ready.repeat_interleave(2, dim=1), torch.zeros(batch, 2, device=device, dtype=dtype)],
            dim=1,
        )
        due = torch.cat(
            [
                self.current_due.repeat_interleave(2, dim=1),
                torch.full((batch, 2), MAX_ROUTE_TIME_NORM, device=device, dtype=dtype),
            ],
            dim=1,
        )
        arrival = state[:, 1].unsqueeze(1) + travel_time
        wait_time = torch.relu(ready - arrival)
        service_start = torch.maximum(arrival, ready)
        lateness = torch.relu(service_start - due)
        demand_ratio = torch.cat(
            [repeated_demands, torch.zeros(batch, 2, device=device, dtype=dtype)],
            dim=1,
        )
        is_depot = torch.zeros(batch, n_nodes * 2 + 2, device=device, dtype=dtype)
        is_depot[:, -2:] = 1.0
        wait_time = torch.where(is_depot > 0.5, torch.zeros_like(wait_time), wait_time)
        lateness = torch.where(is_depot > 0.5, torch.zeros_like(lateness), lateness)
        wait_energy = (P_HOVER * wait_time * MAX_TIME) / E_MAX
        projected_energy = state[:, 2].unsqueeze(1) - flight_energy - wait_energy
        energy_margin = projected_energy - E_SAFE_MARGIN

        action_features = torch.stack(
            [
                travel_time,
                flight_energy,
                wait_time,
                lateness,
                energy_margin,
                demand_ratio,
                mode_fast,
                is_depot,
            ],
            dim=-1,
        )
        condition = torch.cat([current_ctx, state, pref], dim=-1)
        cond_hidden = self.action_condition(condition).unsqueeze(1)
        feat_hidden = self.action_feature_mlp(action_features)
        interaction = torch.tanh(feat_hidden + cond_hidden)
        return ACTION_SCORE_SCALE * self.action_score_out(interaction).squeeze(-1)

    def forward(
        self,
        current_ctx,
        action_ctx,
        state,
        mask,
        curr_pos,
        node_pos,
        depot_pos,
        pref,
        gamma,
        beta,
        demands,
        ready_times,
        due_times,
    ):
        self.current_demands = demands
        self.current_ready = ready_times
        self.current_due = due_times
        q = self.Wq_ctx(current_ctx) + self.Wq_state(state)
        if self.use_film or self.use_mode_mask:
            q = q + self.Wq_pref(pref)
        if self.use_film:
            q = q * (gamma + 1.0) + beta
        q = q.unsqueeze(1)
        k = self.Wk(action_ctx)
        logits = (q @ k.transpose(1, 2)).squeeze(1) / self.temp
        logits = LOGIT_CLIPPING * torch.tanh(logits)
        if self.use_action_features:
            logits = logits + self._action_feature_logits(current_ctx, state, curr_pos, node_pos, depot_pos, pref)
        if self.use_ega:
            logits = logits + self._energy_gate_bias(curr_pos, node_pos, depot_pos, state[:, 0], state[:, 1], state[:, 2])
        if self.use_mode_mask:
            logits = logits + self._mode_preference_bias(pref, node_pos.size(1))
        logits = logits.masked_fill(~mask, -1e9)
        return F.softmax(logits, dim=-1)
