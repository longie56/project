# model/pomo.py
import torch
import torch.nn as nn

from config import *
from env.vrp_env import VRPTWEnv
from model.decoder import Decoder
from model.encoder import TransformerEncoder
from model.pc_ega_module import FiLM


class PC_EGA_POMO(nn.Module):
    def __init__(
        self,
        use_film=True,
        use_ega=True,
        use_mode_mask=True,
        use_action_features=True,
        encoder_ff_hidden=None,
    ):
        super().__init__()
        self.use_film = use_film
        self.use_ega = use_ega
        self.use_mode_mask = use_mode_mask
        self.use_action_features = use_action_features
        self.encoder = TransformerEncoder(
            dim_feedforward=ENCODER_FF_HIDDEN if encoder_ff_hidden is None else int(encoder_ff_hidden)
        )
        self.film = FiLM()
        self.decoder = Decoder(
            use_film=use_film,
            use_ega=use_ega,
            use_mode_mask=use_mode_mask,
            use_action_features=use_action_features,
        )
        self.critic = nn.Sequential(nn.Linear(D_MODEL, D_MODEL), nn.ReLU(), nn.Linear(D_MODEL, 1))

    def _expand_for_pomo(self, tensor, pomo_size):
        shape = tensor.shape[1:]
        return tensor.unsqueeze(1).expand(-1, pomo_size, *shape).reshape(-1, *shape)

    def _augment_xy8(self, xy):
        x = xy[..., 0]
        y = xy[..., 1]
        aug = [
            torch.stack([x, y], dim=-1),
            torch.stack([1.0 - x, y], dim=-1),
            torch.stack([x, 1.0 - y], dim=-1),
            torch.stack([1.0 - x, 1.0 - y], dim=-1),
            torch.stack([y, x], dim=-1),
            torch.stack([1.0 - y, x], dim=-1),
            torch.stack([y, 1.0 - x], dim=-1),
            torch.stack([1.0 - y, 1.0 - x], dim=-1),
        ]
        return aug

    def _build_augmented_inputs(self, nodes, depot, curr_pos, augment_8_fold):
        if not augment_8_fold:
            return [(nodes, depot, curr_pos)]

        node_xy_aug = self._augment_xy8(nodes[:, :, :2])
        depot_aug = self._augment_xy8(depot)
        curr_pos_aug = self._augment_xy8(curr_pos)
        augmented = []
        for aug_nodes_xy, aug_depot, aug_curr_pos in zip(node_xy_aug, depot_aug, curr_pos_aug):
            aug_nodes = nodes.clone()
            aug_nodes[:, :, :2] = aug_nodes_xy
            augmented.append((aug_nodes, aug_depot, aug_curr_pos))
        return augmented

    def _build_pomo_start_actions(self, env, pref_exp, pomo_size, start_offset=0):
        total_batch = env.B
        device = env.dev
        base_batch = total_batch // pomo_size
        start_nodes = torch.arange(pomo_size, device=device).unsqueeze(0).expand(base_batch, -1)
        start_nodes = (start_nodes + int(start_offset)) % env.N
        start_nodes = start_nodes.reshape(-1)
        start_mask = env.get_mask()
        row_idx = torch.arange(total_batch, device=device)
        start_norm = start_nodes * 2
        start_fast = start_nodes * 2 + 1
        norm_ok = start_mask[row_idx, start_norm]
        fast_ok = start_mask[row_idx, start_fast]

        node_xy = env.nodes[row_idx, start_nodes, :2]
        ready = env.nodes[row_idx, start_nodes, 3]
        due = env.nodes[row_idx, start_nodes, 4]
        dist = torch.norm(node_xy - env.depot, dim=-1) * GRID_SCALE

        def immediate_cost(speed, empty_power):
            t_flight = dist / speed
            t_norm = t_flight / MAX_TIME
            arrival = t_norm
            start = torch.maximum(arrival, ready)
            wait_e = P_HOVER * torch.relu(start - arrival) * MAX_TIME
            e_flight = (empty_power + P_GAMMA * env.load[row_idx] * MAX_PAYLOAD_KG) * t_flight
            pen = torch.relu(start - due)
            return (
                PHYS_WEIGHT
                * (
                    (e_flight + wait_e) / E_MAX * 10.0 * pref_exp[:, 1]
                    + t_norm * 5.0 * pref_exp[:, 0]
                )
                + PEN_WEIGHT * pen
            )

        norm_cost = immediate_cost(V_NORM, P_EMPTY_NORM)
        fast_cost = immediate_cost(V_FAST, P_EMPTY_FAST)
        prefer_fast = fast_cost < norm_cost
        start_actions = torch.where(prefer_fast, start_fast, start_norm)
        alt_actions = torch.where(prefer_fast, start_norm, start_fast)
        start_ok = torch.where(prefer_fast, fast_ok, norm_ok)
        alt_ok = torch.where(prefer_fast, norm_ok, fast_ok)
        fallback_actions = start_mask.float().argmax(dim=-1)
        return torch.where(start_ok, start_actions, torch.where(alt_ok, alt_actions, fallback_actions))

    def _rollout(
        self,
        nodes,
        pref,
        depot,
        visited_mask,
        curr_pos,
        curr_time,
        curr_load,
        curr_energy,
        N_pomo=None,
        decode_type="sample",
        return_routes=False,
        start_offset=0,
        sample_temperature=1.0,
        scenario="full",
    ):
        B, N, _ = nodes.shape
        device = nodes.device
        if N_pomo is None:
            N_pomo = N
        pomo_size = min(N_pomo, N)

        depot_feat = torch.zeros(B, 1, 6, device=device)
        depot_feat[:, :, :2] = depot.unsqueeze(1)
        all_nodes = torch.cat([nodes, depot_feat], 1)
        h = self.encoder(all_nodes)
        node_emb = h[:, :N]
        depot_emb = h[:, N:]

        gamma, beta = self.film(pref)
        if not self.use_film:
            gamma = torch.zeros_like(gamma)
            beta = torch.zeros_like(beta)

        depot_exp = self._expand_for_pomo(depot, pomo_size)
        pref_exp = self._expand_for_pomo(pref, pomo_size)
        gamma_exp = self._expand_for_pomo(gamma, pomo_size)
        beta_exp = self._expand_for_pomo(beta, pomo_size)
        visited_exp = self._expand_for_pomo(visited_mask, pomo_size)
        curr_pos_exp = self._expand_for_pomo(curr_pos, pomo_size)
        curr_time_exp = self._expand_for_pomo(curr_time, pomo_size)
        curr_load_exp = self._expand_for_pomo(curr_load, pomo_size)
        curr_energy_exp = self._expand_for_pomo(curr_energy, pomo_size)
        nodes_exp = nodes.unsqueeze(1).expand(-1, pomo_size, -1, -1).reshape(B * pomo_size, N, 6)

        env = VRPTWEnv(
            nodes_exp,
            pref_exp,
            depot_exp,
            scenario=scenario,
        )
        env.load_state(visited_exp, curr_pos_exp, curr_time_exp, curr_load_exp, curr_energy_exp)

        node_ctx = node_emb.unsqueeze(1).expand(-1, pomo_size, -1, -1).reshape(B * pomo_size, N, D_MODEL)
        depot_ctx = depot_emb.unsqueeze(1).expand(-1, pomo_size, -1, -1).reshape(B * pomo_size, 1, D_MODEL)
        action_ctx = torch.cat([node_ctx.repeat_interleave(2, dim=1), depot_ctx.repeat(1, 2, 1)], dim=1)
        current_ctx = depot_ctx.squeeze(1)

        start_actions = self._build_pomo_start_actions(env, pref_exp, pomo_size, start_offset=start_offset)
        _, done = env.step(start_actions)
        batch_idx = torch.arange(B * pomo_size, device=device)
        start_nodes = torch.clamp(torch.div(start_actions, 2, rounding_mode="trunc"), max=N - 1)
        start_is_customer = start_actions < N * 2
        current_ctx = torch.where(
            start_is_customer.unsqueeze(-1),
            node_ctx[batch_idx, start_nodes],
            depot_ctx.squeeze(1),
        )

        log_probs, phys, pen, unvisit, ent = [], [], [], [], []
        action_history = [start_actions.clone()] if return_routes else None
        step = 0
        while not done.all() and step < MAX_STEPS:
            prev_done = done.clone()
            mask = env.get_mask(inactive_mask=prev_done)
            state = torch.stack([env.load, env.time, env.energy], -1)
            probs = self.decoder(
                current_ctx,
                action_ctx,
                state,
                mask,
                env.pos,
                env.nodes[:, :, :2],
                env.depot,
                pref_exp,
                gamma_exp,
                beta_exp,
                env.nodes[:, :, 2],
                env.nodes[:, :, 3],
                env.nodes[:, :, 4],
            )
            if decode_type != "greedy" and abs(float(sample_temperature) - 1.0) > 1e-9:
                temperature = max(float(sample_temperature), 1e-3)
                probs = probs.clamp_min(1e-12).pow(1.0 / temperature)
                probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            dist = torch.distributions.Categorical(probs)
            if decode_type == "greedy":
                action = probs.argmax(dim=-1)
            else:
                action = dist.sample()
            _, step_done = env.step(action, inactive_mask=prev_done)

            step_log_prob = dist.log_prob(action)
            step_log_prob[prev_done] = 0.0
            step_entropy = dist.entropy()
            step_entropy[prev_done] = 0.0

            done = prev_done | step_done
            next_nodes = torch.clamp(torch.div(action, 2, rounding_mode="trunc"), max=N - 1)
            next_is_customer = action < N * 2
            current_ctx = torch.where(
                next_is_customer.unsqueeze(-1),
                node_ctx[batch_idx, next_nodes],
                depot_ctx.squeeze(1),
            )
            log_probs.append(step_log_prob)
            phys.append(env.phys.clone())
            pen.append(env.pen.clone())
            unvisit.append((~env.visited).sum(-1).float())
            ent.append(step_entropy.mean())
            if return_routes:
                action_history.append(action.clone())
            step += 1

        if log_probs:
            T = len(log_probs)
            log_probs = torch.stack(log_probs, 1).reshape(B, pomo_size, T)
            phys = torch.stack(phys, 1).reshape(B, pomo_size, T)[:, :, -1]
            pen = torch.stack(pen, 1).reshape(B, pomo_size, T)[:, :, -1]
            unvisit = torch.stack(unvisit, 1).reshape(B, pomo_size, T)[:, :, -1]
            entropy = torch.stack(ent).mean().reshape(1).expand(B, pomo_size)
        else:
            log_probs = torch.zeros(B, pomo_size, 0, device=device)
            phys = env.phys.reshape(B, pomo_size)
            pen = env.pen.reshape(B, pomo_size)
            unvisit = (~env.visited).sum(-1).float().reshape(B, pomo_size)
            entropy = torch.zeros(B, pomo_size, device=device)
        value = self.critic(node_emb.mean(1)).squeeze(-1)
        value = value.unsqueeze(1).expand(-1, pomo_size)
        routes = None
        finished = done.reshape(B, pomo_size)
        at_depot = env._at_depot().reshape(B, pomo_size)
        if return_routes:
            routes = torch.stack(action_history, dim=1).reshape(B, pomo_size, -1)
        return {
            "log_probs": log_probs,
            "phys": phys,
            "pen": pen,
            "unvisit": unvisit,
            "entropy": entropy,
            "value": value,
            "routes": routes,
            "finished": finished,
            "at_depot": at_depot,
        }

    def _select_best_from_rollouts(self, rollout_results):
        if not rollout_results:
            raise ValueError("rollout_results must contain at least one candidate.")

        per_rollout = []
        for rollout in rollout_results:
            total_cost = rollout["phys"] * PHYS_WEIGHT + rollout["pen"] * PEN_WEIGHT + rollout["unvisit"] * 20.0
            candidate_success = (rollout["unvisit"] == 0) & rollout["finished"] & rollout["at_depot"]
            selection_cost = total_cost + (~candidate_success).float() * INFERENCE_INFEASIBLE_PENALTY
            best_idx = selection_cost.argmin(dim=1)
            batch_idx = torch.arange(total_cost.size(0), device=total_cost.device)
            reduced = {
                "best_cost": total_cost[batch_idx, best_idx],
                "best_phys": rollout["phys"][batch_idx, best_idx],
                "best_pen": rollout["pen"][batch_idx, best_idx],
                "best_unvisit": rollout["unvisit"][batch_idx, best_idx],
                "finished": rollout["finished"][batch_idx, best_idx],
                "at_depot": rollout["at_depot"][batch_idx, best_idx],
                "all_costs": total_cost,
                "all_selection_costs": selection_cost,
            }
            routes = rollout["routes"]
            reduced["best_routes"] = routes[batch_idx, best_idx] if routes is not None else None
            per_rollout.append(reduced)

        stacked_costs = torch.stack([item["best_cost"] for item in per_rollout], dim=0)
        stacked_selection_costs = torch.stack([item["all_selection_costs"].min(dim=1).values for item in per_rollout], dim=0)
        best_candidate_idx = stacked_selection_costs.argmin(dim=0)
        batch_idx = torch.arange(stacked_selection_costs.size(1), device=stacked_selection_costs.device)

        selected = {
            "best_cost": stacked_costs[best_candidate_idx, batch_idx],
        }
        for key in ["best_phys", "best_pen", "best_unvisit", "finished", "at_depot"]:
            stacked_values = torch.stack([item[key] for item in per_rollout], dim=0)
            selected[key] = stacked_values[best_candidate_idx, batch_idx]

        routes = per_rollout[0]["best_routes"]
        if routes is not None:
            route_lengths = torch.tensor(
                [item["best_routes"].size(1) for item in per_rollout],
                dtype=torch.long,
                device=stacked_costs.device,
            )
            selected_lengths = route_lengths[best_candidate_idx]
            max_route_len = int(route_lengths.max().item())
            padded_routes = []
            for item in per_rollout:
                route_tensor = item["best_routes"]
                pad_len = max_route_len - route_tensor.size(1)
                if pad_len > 0:
                    pad_value = route_tensor[:, -1:].expand(-1, pad_len)
                    route_tensor = torch.cat([route_tensor, pad_value], dim=1)
                padded_routes.append(route_tensor)
            stacked_routes = torch.stack(padded_routes, dim=0)
            selected["best_routes"] = stacked_routes[best_candidate_idx, batch_idx]
            selected["best_route_lengths"] = selected_lengths
        else:
            selected["best_routes"] = None
            selected["best_route_lengths"] = None

        selected["feasible"] = (
            (selected["best_unvisit"] == 0) & selected["finished"] & selected["at_depot"]
        )
        selected["all_costs"] = torch.cat([item["all_costs"] for item in per_rollout], dim=1)
        return selected

    def forward(
        self,
        nodes,
        pref,
        depot,
        visited_mask,
        curr_pos,
        curr_time,
        curr_load,
        curr_energy,
        N_pomo=None,
        scenario="full",
    ):
        rollout = self._rollout(
            nodes,
            pref,
            depot,
            visited_mask,
            curr_pos,
            curr_time,
            curr_load,
            curr_energy,
            N_pomo=N_pomo,
            decode_type="sample",
            return_routes=False,
            scenario=scenario,
        )
        return (
            rollout["log_probs"],
            rollout["phys"],
            rollout["pen"],
            rollout["unvisit"],
            rollout["entropy"],
            rollout["value"],
            rollout["finished"],
            rollout["at_depot"],
        )

    def teacher_forcing_loss(
        self,
        nodes,
        pref,
        depot,
        teacher_actions,
        visited_mask=None,
        curr_pos=None,
        curr_time=None,
        curr_load=None,
        curr_energy=None,
        scenario="full",
        ignore_index=-1,
    ):
        B, N, _ = nodes.shape
        device = nodes.device
        if visited_mask is None:
            visited_mask = torch.zeros(B, N, dtype=torch.bool, device=device)
        if curr_pos is None:
            curr_pos = depot.clone()
        if curr_time is None:
            curr_time = torch.zeros(B, device=device)
        if curr_load is None:
            curr_load = torch.ones(B, device=device)
        if curr_energy is None:
            curr_energy = torch.ones(B, device=device)

        depot_feat = torch.zeros(B, 1, 6, device=device)
        depot_feat[:, :, :2] = depot.unsqueeze(1)
        all_nodes = torch.cat([nodes, depot_feat], 1)
        h = self.encoder(all_nodes)
        node_ctx = h[:, :N]
        depot_ctx = h[:, N:]

        gamma, beta = self.film(pref)
        if not self.use_film:
            gamma = torch.zeros_like(gamma)
            beta = torch.zeros_like(beta)

        action_ctx = torch.cat([node_ctx.repeat_interleave(2, dim=1), depot_ctx.repeat(1, 2, 1)], dim=1)
        current_ctx = depot_ctx.squeeze(1)

        env = VRPTWEnv(nodes.clone(), pref.clone(), depot.clone(), scenario=scenario)
        env.load_state(
            visited_mask.clone(),
            curr_pos.clone(),
            curr_time.clone(),
            curr_load.clone(),
            curr_energy.clone(),
        )

        total_nll = torch.zeros((), device=device)
        valid_steps = torch.zeros((), device=device)
        invalid_steps = torch.zeros((), device=device)
        teacher_actions = teacher_actions.to(device=device, dtype=torch.long)
        batch_idx = torch.arange(B, device=device)

        for step in range(teacher_actions.size(1)):
            action = teacher_actions[:, step]
            active = action.ne(ignore_index) & (~env._compute_done())
            if not active.any():
                continue

            mask = env.get_mask()
            valid = active & (action >= 0) & (action < mask.size(1))
            valid = valid & mask[batch_idx, action.clamp(0, mask.size(1) - 1)]
            invalid_steps = invalid_steps + (active & ~valid).float().sum()
            if not valid.any():
                continue

            state = torch.stack([env.load, env.time, env.energy], -1)
            probs = self.decoder(
                current_ctx,
                action_ctx,
                state,
                mask,
                env.pos,
                env.nodes[:, :, :2],
                env.depot,
                pref,
                gamma,
                beta,
                env.nodes[:, :, 2],
                env.nodes[:, :, 3],
                env.nodes[:, :, 4],
            )
            selected_probs = probs[batch_idx[valid], action[valid]].clamp_min(1e-12)
            total_nll = total_nll - selected_probs.log().sum()
            valid_steps = valid_steps + valid.float().sum()

            step_action = torch.full((B,), self.N * 2 if hasattr(self, "N") else N * 2, dtype=torch.long, device=device)
            step_action[valid] = action[valid]
            env.step(step_action, inactive_mask=~valid)

            next_nodes = torch.clamp(torch.div(step_action, 2, rounding_mode="trunc"), max=N - 1)
            next_is_customer = (step_action < N * 2) & valid
            current_ctx = torch.where(
                next_is_customer.unsqueeze(-1),
                node_ctx[batch_idx, next_nodes],
                depot_ctx.squeeze(1),
            )

        loss = total_nll / valid_steps.clamp_min(1.0)
        return {
            "loss": loss,
            "nll": total_nll.detach(),
            "valid_steps": valid_steps.detach(),
            "invalid_steps": invalid_steps.detach(),
        }

    def inference_rollout(
        self,
        nodes,
        pref,
        depot,
        visited_mask=None,
        curr_pos=None,
        curr_time=None,
        curr_load=None,
        curr_energy=None,
        N_pomo=None,
        decode_type="greedy",
        sample_rollouts=0,
        sample_temperature=1.0,
        multi_start_rollouts=0,
        augment_8_fold=False,
        scenario="full",
    ):
        B = nodes.size(0)
        device = nodes.device
        N = nodes.size(1)
        if N_pomo is None:
            N_pomo = N
        if visited_mask is None:
            visited_mask = torch.zeros(B, N, dtype=torch.bool, device=device)
        if curr_pos is None:
            curr_pos = depot.clone()
        if curr_time is None:
            curr_time = torch.zeros(B, device=device)
        if curr_load is None:
            curr_load = torch.ones(B, device=device)
        if curr_energy is None:
            curr_energy = torch.ones(B, device=device)
        augmented_inputs = self._build_augmented_inputs(nodes, depot, curr_pos, augment_8_fold)

        def add_rollout(candidate_decode_type, start_offset):
            for aug_nodes, aug_depot, aug_curr_pos in augmented_inputs:
                rollout_results.append(
                    self._rollout(
                        aug_nodes,
                        pref,
                        aug_depot,
                        visited_mask,
                        aug_curr_pos,
                        curr_time,
                        curr_load,
                        curr_energy,
                        N_pomo=N_pomo,
                        decode_type=candidate_decode_type,
                        return_routes=True,
                        start_offset=start_offset,
                        sample_temperature=sample_temperature,
                        scenario=scenario,
                    )
                )

        rollout_results = []
        used_offsets = set()

        add_rollout(decode_type, 0)
        used_offsets.add(0)

        offset_step = max(1, N // max(1, min(N, int(N_pomo))))
        extra_multi_start = max(0, int(multi_start_rollouts))
        for idx in range(extra_multi_start):
            offset = ((idx + 1) * offset_step) % max(1, N)
            while offset in used_offsets and len(used_offsets) < N:
                offset = (offset + 1) % N
            used_offsets.add(offset)
            add_rollout(decode_type, offset)

        for sample_idx in range(max(0, int(sample_rollouts))):
            offset = ((extra_multi_start + sample_idx + 1) * offset_step) % max(1, N)
            while offset in used_offsets and len(used_offsets) < N:
                offset = (offset + 1) % N
            used_offsets.add(offset)
            rollout_results.append(
                self._rollout(
                    nodes,
                    pref,
                    depot,
                    visited_mask,
                    curr_pos,
                    curr_time,
                    curr_load,
                    curr_energy,
                    N_pomo=N_pomo,
                    decode_type="sample",
                    return_routes=True,
                    start_offset=offset,
                    sample_temperature=sample_temperature,
                    scenario=scenario,
                )
            )
        return self._select_best_from_rollouts(rollout_results)
