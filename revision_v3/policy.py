"""Preference-conditioned POMO construction on explicit v3 load/flight actions."""

from __future__ import annotations

import torch
import torch.nn as nn

from env.revision_v3_gu_physics import EnergyScenario
from revision_v3.environment import RouteEnvironment
from revision_v3.spec import (BUDGET_J, DEPOT_TURN_S, Q_MAX_KG, SIDE_M,
                              SPEEDS_M_S, TASK, horizon_s)


DIM = 128


class AnalyticLocalSpeed(nn.Module):
    """Exact immediate C0/C1 speed rule for an inference-only control.

    The same route network and physical features are retained.  Replacing a
    trained speed head with this rule does not constitute an independently
    trained solver baseline.
    """

    def forward(self, speed_inputs: torch.Tensor) -> torch.Tensor:
        return (-(speed_inputs[..., 5]+speed_inputs[..., 6]
                  +speed_inputs[..., 7])).clamp(-4.0, 4.0).unsqueeze(-1)


class ResidualGlobalSpeed(nn.Module):
    """Pilot candidate: learn a downstream correction to exact local C0/C1 cost.

    The zero-initialized residual starts at the physical local speed rule.
    Only full-route counterfactuals can teach a nonlocal correction.
    """

    def __init__(self, input_dim: int, positive_only: bool = False):
        super().__init__()
        self.positive_only = bool(positive_only)
        self.residual = nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(),
                                      nn.Linear(64, 1))
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(self, speed_inputs: torch.Tensor) -> torch.Tensor:
        local = (-(speed_inputs[..., 5]+speed_inputs[..., 6]
                   +speed_inputs[..., 7])).clamp(-4.0, 4.0).unsqueeze(-1)
        correction = self.residual(speed_inputs)
        if self.positive_only:
            # A learned nonlocal signal may add the benefit of preventing
            # downstream tardiness, but cannot overturn a local fast choice.
            correction = correction.clamp_min(0.0)
        return (local + correction).clamp(-4.0, 4.0)


class V3Encoder(nn.Module):
    """Explicit 128-wide attention encoder; no legacy v1 config import."""

    def __init__(self):
        super().__init__()
        self.input_proj = nn.Linear(6, DIM)
        layer = nn.TransformerEncoderLayer(
            d_model=DIM, nhead=4, dim_feedforward=512,
            dropout=0.0, activation="relu", batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=3)

    def forward(self, nodes):
        return self.transformer(self.input_proj(nodes))


class PMOCOStyleHyper(nn.Module):
    """Preference-generated decoder projections adapted from Lin et al. 2022.

    The published MOCVRP decoder generates Wq/Wk/Wv/Wcombine from a
    two-objective preference. Here the same four-matrix design is mapped to
    v3 load, flight, return and wait actions. The state and hard physics masks
    necessarily differ from the published CVRP task.
    """

    def __init__(self):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(2, 256), nn.ReLU(),
                                   nn.Linear(256, 256), nn.ReLU(),
                                   nn.Linear(256, 8))
        self.projections = nn.ModuleList([
            nn.Linear(2, DIM*DIM, bias=False) for _ in range(4)
        ])

    def forward(self, pref):
        latent = self.trunk(pref).reshape(pref.size(0), 4, 2)
        return tuple(layer(latent[:, idx]).reshape(-1, DIM, DIM) / DIM**0.5
                     for idx, layer in enumerate(self.projections))

VARIANTS = {
    "full": {},
    "speed_aux_candidate": {"use_mode_bias": False, "use_speed_aux": True},
    "factorized_speed_aux_candidate": {
        "use_mode_bias": False, "use_speed_aux": True,
        "use_factorized_speed": True,
    },
    "factorized_global_speed_candidate": {
        "use_mode_bias": False, "use_speed_aux": True,
        "use_factorized_speed": True, "speed_route_grad_scale": 0.05,
    },
    "factorized_lookahead_speed_candidate": {
        "use_mode_bias": False, "use_speed_aux": True,
        "use_factorized_speed": True, "use_speed_lookahead_inputs": True,
    },
    "factorized_decoupled_lookahead_candidate": {
        "use_mode_bias": False, "use_speed_aux": True,
        "use_factorized_speed": True, "use_speed_lookahead_inputs": True,
        "use_decoupled_route_speed": True,
    },
    # Formal-candidate variants keep historical v3 "full" checkpoints
    # readable while clearly naming the zero-manual-bias speed protocol.
    "full_factorized_v3": {
        "use_mode_bias": False, "use_speed_aux": True,
        "use_factorized_speed": True, "use_speed_lookahead_inputs": True,
        "use_decoupled_route_speed": True,
    },
    "factorized_wo_ega": {
        "use_mode_bias": False, "use_speed_aux": True,
        "use_factorized_speed": True, "use_ega": False,
        "use_speed_lookahead_inputs": True, "use_decoupled_route_speed": True,
    },
    "factorized_wo_film": {
        "use_mode_bias": False, "use_speed_aux": True,
        "use_factorized_speed": True, "use_film": False,
        "use_speed_lookahead_inputs": True, "use_decoupled_route_speed": True,
    },
    "factorized_wo_route_physical_head": {
        "use_mode_bias": False, "use_speed_aux": True,
        "use_factorized_speed": True, "use_physical_features": False,
        "use_speed_lookahead_inputs": True, "use_decoupled_route_speed": True,
    },
    "factorized_wo_speed_physical_inputs": {
        "use_mode_bias": False, "use_speed_aux": True,
        "use_factorized_speed": True, "use_speed_physical_inputs": False,
        "use_speed_lookahead_inputs": True, "use_decoupled_route_speed": True,
    },
    "factorized_wo_speed_aux": {
        "use_mode_bias": False, "use_speed_aux": False,
        "use_factorized_speed": True, "speed_route_grad_scale": 1.0,
        "use_speed_lookahead_inputs": True, "use_decoupled_route_speed": True,
    },
    "factorized_wo_speed_head": {
        "use_mode_bias": False, "use_speed_aux": True,
        "use_factorized_speed": False,
    },
    "pmoco_adapted_v3_with_aux": {
        "use_ega": False, "use_mode_bias": False,
        "use_film": False, "use_physical_features": False,
        "use_pref_query": False, "use_pmo_hyper": True,
        "use_speed_aux": True, "use_factorized_speed": True,
        "use_speed_lookahead_inputs": True, "use_decoupled_route_speed": True,
    },
    "wo_ega": {"use_ega": False},
    "wo_mode_bias": {"use_mode_bias": False},
    "wo_film": {"use_film": False},
    "wo_physical_features": {"use_physical_features": False},
    "wo_film_wo_mode_bias": {"use_film": False, "use_mode_bias": False},
    "wo_physical_features_wo_mode_bias": {
        "use_physical_features": False, "use_mode_bias": False,
    },
    "wo_pref_query": {"use_pref_query": False},
    "attention_speed_baseline": {
        "use_ega": False, "use_mode_bias": False,
        "use_film": False, "use_physical_features": False,
    },
    "pmoco_adapted_v3": {
        "use_ega": False, "use_mode_bias": False,
        "use_film": False, "use_physical_features": False,
        "use_pref_query": False, "use_pmo_hyper": True,
    },
}

# Pre-freeze candidate: the same deadline-aware depot-wait prior is applied to
# every neural comparison. The original names retain their exact old behavior
# so that disposable checkpoint diagnostics remain reproducible.
_WAIT40_BASE_VARIANTS = (
    "full_factorized_v3", "pmoco_adapted_v3_with_aux",
    "factorized_wo_ega", "factorized_wo_film",
    "factorized_wo_route_physical_head",
    "factorized_wo_speed_physical_inputs",
    "factorized_wo_speed_aux", "factorized_wo_speed_head",
)
VARIANTS.update({
    f"{name}_wait40": {**VARIANTS[name], "wait_urgency_coefficient": 40.0}
    for name in _WAIT40_BASE_VARIANTS
})
VARIANTS.update({
    f"{name}_wait40_slack5": {
        **VARIANTS[name], "wait_urgency_coefficient": 40.0,
        "wait_slack_coefficient": 5.0,
        "wait_slack_buffer_fraction": 0.10,
    }
    for name in _WAIT40_BASE_VARIANTS
})
VARIANTS["full_factorized_v3_wait40_slack5_residual_candidate"] = {
    **VARIANTS["full_factorized_v3_wait40_slack5"],
    "use_speed_residual": True,
}
VARIANTS["full_factorized_v3_wait40_slack5_positive_residual_candidate"] = {
    **VARIANTS["full_factorized_v3_wait40_slack5"],
    "use_speed_residual": True,
    "use_speed_positive_residual": True,
}
VARIANTS["full_factorized_v3_wait40_slack5_physical_film_candidate"] = {
    **VARIANTS["full_factorized_v3_wait40_slack5"],
    "film_target": "physical",
}
VARIANTS["factorized_wo_ega_wait40_slack5_physical_film_candidate"] = {
    **VARIANTS["factorized_wo_ega_wait40_slack5"],
    "film_target": "physical",
}

# The compact revision removes both optional components. Historical variant
# names remain available only to reproduce archived pilot checkpoints.
VARIANTS["physical_pref_v3_wait40_slack5"] = {
    **VARIANTS["full_factorized_v3_wait40_slack5"],
    "use_film": False, "use_ega": False,
    "matched_physical_head_init": True,
}
VARIANTS["physical_pref_wo_route_features_v3_wait40_slack5"] = {
    **VARIANTS["physical_pref_v3_wait40_slack5"],
    "use_physical_features": False,
}


class V3Policy(nn.Module):
    def __init__(self, *, use_ega=True, use_mode_bias=True, use_film=True,
                 matched_physical_head_init=False,
                 use_physical_features=True, use_pref_query=True,
                 use_pmo_hyper=False, use_speed_aux=False,
                 use_factorized_speed=False, use_speed_physical_inputs=True,
                 speed_route_grad_scale=0.0,
                 use_speed_lookahead_inputs=False,
                 use_decoupled_route_speed=False,
                 use_speed_residual=False,
                 use_speed_positive_residual=False,
                 wait_urgency_coefficient=0.0,
                 wait_slack_coefficient=0.0,
                 wait_slack_buffer_fraction=0.10,
                 film_target="query"):
        super().__init__()
        self.use_ega = use_ega
        self.use_mode_bias = use_mode_bias
        self.use_film = use_film
        if film_target not in ("query", "physical"):
            raise ValueError("FiLM target must be query or physical")
        if use_film and film_target == "physical" and not use_physical_features:
            raise ValueError("Physical FiLM requires the physical action head")
        self.film_target = film_target
        self.use_physical_features = use_physical_features
        self.use_pref_query = use_pref_query
        self.use_pmo_hyper = use_pmo_hyper
        self.use_speed_aux = use_speed_aux
        self.use_factorized_speed = use_factorized_speed
        self.use_speed_physical_inputs = use_speed_physical_inputs
        self.use_speed_lookahead_inputs = use_speed_lookahead_inputs
        self.use_decoupled_route_speed = use_decoupled_route_speed
        self.use_speed_residual = use_speed_residual
        self.use_speed_positive_residual = use_speed_positive_residual
        if wait_urgency_coefficient < 0:
            raise ValueError("Depot wait urgency coefficient must be nonnegative")
        self.wait_urgency_coefficient = float(wait_urgency_coefficient)
        if wait_slack_coefficient < 0 or not 0 <= wait_slack_buffer_fraction <= 1:
            raise ValueError("Invalid depot wait slack prior")
        self.wait_slack_coefficient = float(wait_slack_coefficient)
        self.wait_slack_buffer_fraction = float(wait_slack_buffer_fraction)
        if use_decoupled_route_speed and not use_factorized_speed:
            raise ValueError("Decoupled route/speed sampling requires a factorized speed head")
        if use_speed_residual and not use_factorized_speed:
            raise ValueError("Residual speed head requires factorized speed actions")
        if use_speed_positive_residual and not use_speed_residual:
            raise ValueError("Positive residual requires the residual speed head")
        if not 0.0 <= speed_route_grad_scale <= 1.0:
            raise ValueError("speed_route_grad_scale must be in [0,1]")
        self.speed_route_grad_scale = float(speed_route_grad_scale)
        self.encoder = V3Encoder()
        self.state_proj = nn.Linear(8, DIM)
        self.pref_proj = nn.Linear(2, DIM) if use_pref_query else None
        self.key_proj = nn.Linear(DIM, DIM, bias=False) if not use_pmo_hyper else None
        self.query_proj = nn.Linear(DIM, DIM, bias=False) if not use_pmo_hyper else None
        self.speed_embedding = nn.Parameter(torch.randn(2, DIM) * 0.02)
        self.load_embedding = nn.Parameter(torch.randn(DIM) * 0.02)
        self.return_embedding = nn.Parameter(torch.randn(DIM) * 0.02)
        self.wait_embedding = nn.Parameter(torch.randn(DIM) * 0.02)
        physical_head = (nn.Sequential(nn.Linear(10, 64), nn.ReLU(),
                                        nn.Linear(64, 1))
                         if use_physical_features or matched_physical_head_init else None)
        # Consume the same initialization draws in the matched route-feature
        # ablation, without retaining an unused module or optimizer parameters.
        self.physical_head = physical_head if use_physical_features else None
        self.pmo_hyper = PMOCOStyleHyper() if use_pmo_hyper else None
        speed_input_dim = 18 if use_speed_lookahead_inputs else 10
        self.speed_head = (ResidualGlobalSpeed(speed_input_dim,
                                              positive_only=use_speed_positive_residual)
                           if use_speed_residual else
                           nn.Sequential(nn.Linear(speed_input_dim, 64), nn.ReLU(),
                                         nn.Linear(64, 1))
                           if use_factorized_speed else None)
        if self.speed_head is not None and not use_speed_residual:
            nn.init.zeros_(self.speed_head[-1].weight)
            nn.init.zeros_(self.speed_head[-1].bias)
        # Keep shared initialization identical in Full and no-FiLM ablations.
        # The residual modulation starts as the identity map.
        film_width = DIM if film_target == "query" else 64
        self.film = (nn.Sequential(nn.Linear(2, film_width), nn.ReLU(),
                                   nn.Linear(film_width, 2*film_width))
                     if use_film else None)
        if self.film is not None:
            nn.init.zeros_(self.film[-1].weight)
            nn.init.zeros_(self.film[-1].bias)
        self.logit_clip = 10.0
        self.mode_bias_scale = 0.7

    @staticmethod
    def _normalized_nodes(nodes, n):
        h = horizon_s(n)
        x = nodes.clone()
        x[..., 2] /= Q_MAX_KG
        x[..., 3] /= h
        x[..., 4] /= h
        x[..., 5] /= h
        return x

    def encode(self, nodes, depot):
        b, n, _ = nodes.shape
        depot_features = torch.zeros(b, 1, 6, device=nodes.device, dtype=nodes.dtype)
        depot_features[:, 0, :2] = depot
        encoding = self.encoder(torch.cat([self._normalized_nodes(nodes, n), depot_features], 1))
        return encoding[:, :n], encoding[:, n]

    def _action_keys(self, node_emb, depot_emb):
        b, n, d = node_emb.shape
        flight = node_emb[:, :, None, :] + self.speed_embedding[None, None, :, :]
        flight = flight.reshape(b, 2*n, d)
        loading = node_emb + self.load_embedding
        returning = depot_emb[:, None, :] + self.return_embedding + self.speed_embedding[None, :, :]
        waiting = depot_emb[:, None, :] + self.wait_embedding
        return torch.cat([flight, loading, returning, waiting], dim=1)

    def optimizer_parameter_groups(self, base_lr: float, head_multiplier: float = 1.0):
        if self.speed_head is None:
            return [{"params": list(self.parameters()), "lr": base_lr}]
        head = list(self.speed_head.parameters())
        head_ids = {id(parameter) for parameter in head}
        main = [parameter for parameter in self.parameters()
                if id(parameter) not in head_ids]
        return [{"params": main, "lr": base_lr},
                {"params": head, "lr": base_lr*head_multiplier}]

    def clip_gradients_(self, max_norm: float = 1.0):
        if self.speed_head is None or not self.use_speed_aux:
            return {"main": float(torch.nn.utils.clip_grad_norm_(
                self.parameters(), max_norm)), "speed_head": 0.0}
        head = list(self.speed_head.parameters())
        head_ids = {id(parameter) for parameter in head}
        main = [parameter for parameter in self.parameters()
                if id(parameter) not in head_ids]
        return {"main": float(torch.nn.utils.clip_grad_norm_(main, max_norm)),
                "speed_head": float(torch.nn.utils.clip_grad_norm_(head, max_norm))}

    def _physical_features(self, env: RouteEnvironment):
        b, n = env.B, env.N
        device, dtype = env.dev, env.dtype
        h = env.horizon_s
        z = torch.zeros(b, n, device=device, dtype=dtype)
        customer_features = []
        for mode, speed in enumerate(SPEEDS_M_S):
            safe, travel, wait, fly_e, hover_e, finish = env._customer_options(speed)
            late = torch.clamp_min(finish - env.nodes[:, :, 5] - env.nodes[:, :, 4], 0)
            back_dist = env.depot_dist_m
            # The gate's margin must use the same deadline-admissible return
            # modes as the hard customer mask. A slower, cheaper return that
            # misses H is not an available reserve for this action.
            return_energies = env.customer_return_energies()
            return_times = torch.stack([
                finish + back_dist/back_speed for back_speed in SPEEDS_M_S
            ], dim=-1)
            admissible = return_times <= h + 1e-4
            back_e = torch.where(
                admissible.any(dim=-1),
                return_energies.masked_fill(~admissible, float("inf")).min(dim=-1).values,
                return_energies.min(dim=-1).values,
            )
            margin = (env.battery_j[:, None] - fly_e - hover_e - back_e - env.reserve_j) / BUDGET_J
            features = torch.stack([
                travel/h, fly_e/BUDGET_J, wait/h, hover_e/BUDGET_J,
                late/h, margin, env.nodes[:, :, 2]/Q_MAX_KG,
                torch.full_like(z, float(mode)), z, z,
            ], dim=-1)
            customer_features.append(features)
        flight_features = torch.stack(customer_features, dim=2).reshape(b, 2*n, 10)
        loading_features = torch.zeros(b, n, 10, device=device, dtype=dtype)
        loading_features[:, :, 6] = env.nodes[:, :, 2]/Q_MAX_KG
        loading_features[:, :, 8] = 1
        return_features = torch.zeros(b, 2, 10, device=device, dtype=dtype)
        back_dist = torch.linalg.vector_norm(env.pos-env.depot, dim=-1)*SIDE_M
        for mode, speed in enumerate(SPEEDS_M_S):
            back_e = env.cruise_j(back_dist, speed, env.load_kg).to(dtype)
            return_features[:, mode, 0] = back_dist/speed/h
            return_features[:, mode, 1] = back_e/BUDGET_J
            return_features[:, mode, 5] = (env.battery_j-back_e-env.reserve_j)/BUDGET_J
            return_features[:, mode, 7] = mode
            return_features[:, mode, 9] = 1
        wait_features = torch.zeros(b, 1, 10, device=device, dtype=dtype)
        wait_features[:, 0, 0] = (
            env.next_depot_release_s() - env.time_s
        ).nan_to_num(posinf=h).clamp(min=0) / h
        wait_features[:, 0, 9] = 1
        return torch.cat([flight_features, loading_features, return_features, wait_features], dim=1)

    def _speed_lookahead_features(self, env: RouteEnvironment):
        """Preference-independent downstream deadline context for each speed pair."""
        b, n, h = env.B, env.N, env.horizon_s
        device, dtype = env.dev, env.dtype
        unserved = ~env.served
        other = unserved[:, None, :] & ~torch.eye(
            n, dtype=torch.bool, device=device)[None]
        due = env.nodes[:, None, :, 4].expand(b, n, n)
        ready = env.nodes[:, None, :, 3].expand(b, n, n)
        due_min = due.masked_fill(~other, float("inf")).min(dim=-1).values
        ready_min = ready.masked_fill(~other, float("inf")).min(dim=-1).values
        due_min = torch.where(torch.isfinite(due_min), due_min, torch.full_like(due_min, h))
        ready_min = torch.where(torch.isfinite(ready_min), ready_min, torch.full_like(ready_min, h))
        finish_slow = env._customer_options(SPEEDS_M_S[0])[-1]
        finish_fast = env._customer_options(SPEEDS_M_S[1])[-1]
        clock = (env.time_s/h)[:, None].expand(b, n)
        remaining = (unserved.sum(1).to(dtype)[:, None]-1).clamp_min(0).expand(b, n)/n
        loaded_remaining = ((env.loaded & unserved).sum(1).to(dtype)[:, None]-1)
        loaded_remaining = loaded_remaining.clamp_min(0).expand(b, n)/n
        mass_after = (env.load_kg[:, None]-env.nodes[:, :, 2]).clamp_min(0)/Q_MAX_KG
        flight = torch.stack([
            clock, remaining, loaded_remaining, mass_after,
            ((due_min-finish_slow)/h).clamp(-1, 1),
            ((due_min-finish_fast)/h).clamp(-1, 1),
            ((ready_min-finish_slow)/h).clamp(-1, 1),
            ((ready_min-finish_fast)/h).clamp(-1, 1),
        ], dim=-1)
        return_dist = torch.linalg.vector_norm(env.pos-env.depot, dim=-1)*SIDE_M
        return_slow = env.time_s + return_dist/SPEEDS_M_S[0]
        return_fast = env.time_s + return_dist/SPEEDS_M_S[1]
        due_return = env.nodes[:, :, 4].masked_fill(~unserved, float("inf")).min(1).values
        ready_return = env.nodes[:, :, 3].masked_fill(~unserved, float("inf")).min(1).values
        due_return = torch.where(torch.isfinite(due_return), due_return,
                                 torch.full_like(due_return, h))
        ready_return = torch.where(torch.isfinite(ready_return), ready_return,
                                   torch.full_like(ready_return, h))
        zero = torch.zeros(b, device=device, dtype=dtype)
        returning = torch.stack([
            env.time_s/h, unserved.sum(1).to(dtype)/n, zero, zero,
            ((due_return-return_slow)/h).clamp(-1, 1),
            ((due_return-return_fast)/h).clamp(-1, 1),
            ((ready_return-return_slow)/h).clamp(-1, 1),
            ((ready_return-return_fast)/h).clamp(-1, 1),
        ], dim=-1)
        return torch.cat([flight, returning[:, None]], dim=1)

    @staticmethod
    def _depot_wait_urgency_risk(env: RouteEnvironment) -> torch.Tensor:
        """Optimistic added tardiness of loaded orders from one depot wait.

        Orders are hypothetically served directly at the faster cruise mode.
        This is a route-priority signal; the shared hard physics mask and raw
        objective remain unchanged. It uses no outcome from a held-out route.
        """
        release = env.next_depot_release_s()
        release = torch.where(torch.isfinite(release), release, env.time_s)
        leg_s = env.depot_dist_m / SPEEDS_M_S[-1]
        now = env.time_s[:, None] + DEPOT_TURN_S + leg_s
        after = release[:, None] + DEPOT_TURN_S + leg_s
        ready, due = env.nodes[:, :, 3], env.nodes[:, :, 4]
        current_late = (torch.maximum(now, ready)-due).clamp_min(0)
        after_late = (torch.maximum(after, ready)-due).clamp_min(0)
        loaded = env.loaded & ~env.served
        return ((after_late-current_late).clamp_min(0)
                * loaded.to(env.dtype)).sum(dim=1) / 3600.0

    @staticmethod
    def _depot_wait_slack_consumption(
        env: RouteEnvironment, buffer_fraction: float
    ) -> torch.Tensor:
        """Added near-deadline pressure across all unserved orders from waiting.

        The direct fastest-flight arrival is a proxy, not an exact future route
        bound.  It changes only the policy prior, never the physical mask or C0.
        """
        release = env.next_depot_release_s()
        release = torch.where(torch.isfinite(release), release, env.time_s)
        leg_s = env.depot_dist_m / SPEEDS_M_S[-1]
        ready, due = env.nodes[:, :, 3], env.nodes[:, :, 4]
        buffer_s = buffer_fraction * env.horizon_s
        now = torch.maximum(env.time_s[:, None] + DEPOT_TURN_S + leg_s, ready)
        after = torch.maximum(release[:, None] + DEPOT_TURN_S + leg_s, ready)
        before_pressure = (now - due + buffer_s).clamp_min(0)
        after_pressure = (after - due + buffer_s).clamp_min(0)
        return ((after_pressure-before_pressure).clamp_min(0)
                * (~env.served).to(env.dtype)).sum(dim=1) / 3600.0

    def logits(self, env: RouteEnvironment, node_emb, depot_emb, current_emb,
               *, pmo_params=None, mask=None, return_components=False,
               static_action_emb=None, static_keys=None, static_values=None):
        b, n = env.B, env.N
        state = torch.stack([
            env.pos[:, 0], env.pos[:, 1], env.load_kg/Q_MAX_KG,
            env.time_s/env.horizon_s, env.battery_j/BUDGET_J,
            env.served.float().mean(1), env.loaded.float().mean(1),
            env.at_depot().to(env.dtype),
        ], dim=1)
        query = current_emb + self.state_proj(state)
        if self.use_pref_query:
            query = query + self.pref_proj(env.pref)
        if self.use_film and self.film_target == "query":
            gamma, beta = self.film(env.pref).chunk(2, dim=1)
            query = query * (1 + gamma) + beta
        action_emb = (static_action_emb if static_action_emb is not None
                      else self._action_keys(node_emb, depot_emb))
        if self.use_pmo_hyper:
            if pmo_params is None or mask is None:
                raise ValueError("PMOCO-adapted decoder requires generated parameters and mask")
            wq, wk, wv, wcombine = pmo_params
            q = torch.bmm(query[:, None, :], wq)
            k = static_keys if static_keys is not None else torch.bmm(action_emb, wk)
            v = static_values if static_values is not None else torch.bmm(action_emb, wv)
            attention = torch.bmm(q, k.transpose(1, 2)).squeeze(1) / DIM**0.5
            attention = attention.masked_fill(~mask, -1e9)
            context = torch.bmm(torch.softmax(attention, dim=1)[:, None, :], v)
            context = torch.bmm(context, wcombine).squeeze(1)
            raw = (context[:, None, :] * action_emb).sum(-1) / DIM**0.5
        else:
            keys = static_keys if static_keys is not None else self.key_proj(action_emb)
            raw = (self.query_proj(query)[:, None, :] * keys).sum(-1) / DIM**0.5
        attention_logits = self.logit_clip * torch.tanh(raw)
        scores = attention_logits
        zero = torch.zeros_like(scores) if return_components else None
        physical_logits = zero
        ega_logits = zero
        mode_bias_logits = zero
        features = None
        if self.use_physical_features or self.use_ega or self.use_factorized_speed:
            features = self._physical_features(env)
        if self.use_physical_features:
            if self.use_film and self.film_target == "physical":
                gamma, beta = self.film(env.pref).chunk(2, dim=1)
                physical_hidden = self.physical_head[0](features)
                physical_hidden = (physical_hidden * (1 + gamma[:, None, :])
                                   + beta[:, None, :])
                physical_logits = self.physical_head[2](
                    self.physical_head[1](physical_hidden)).squeeze(-1)
            else:
                physical_logits = self.physical_head(features).squeeze(-1)
            scores = scores + physical_logits
        if self.use_ega:
            # The hard mask remains authoritative. Penalise choices whose
            # best deadline-admissible return leaves little energy beyond the
            # required reserve. The fixed 5%-budget centre and softness give
            # a visible bias across roughly 0-20% spare energy, not merely
            # within a fraction of one percent of the hard boundary.
            margin = features[:, :2*n, 5]
            gate = torch.sigmoid((margin - 0.05) / 0.05).clamp_min(1e-4)
            ega_logits = torch.cat([
                gate.log(), torch.zeros(b, n+3, device=env.dev, dtype=env.dtype)
            ], dim=1)
            scores = scores + ega_logits
        if self.use_mode_bias:
            delta = (env.pref[:, 0] - env.pref[:, 1])[:, None] * self.mode_bias_scale
            mode = torch.tensor([-1.0, 1.0], device=env.dev, dtype=env.dtype)
            fly_bias = (delta[:, :, None] * mode).expand(-1, n, -1).reshape(b, 2*n)
            load_bias = torch.zeros(b, n, device=env.dev, dtype=env.dtype)
            return_bias = delta * mode
            wait_bias = torch.zeros(b, 1, device=env.dev, dtype=env.dtype)
            mode_bias_logits = torch.cat([fly_bias, load_bias, return_bias, wait_bias], dim=1)
            scores = scores + mode_bias_logits
        factorized_speed_logits = None
        speed_inputs = None
        if self.use_factorized_speed:
            if mask is None:
                raise ValueError("Factorized speed head requires the legal action mask")
            slow = torch.cat([features[:, 0:2*n:2], features[:, 3*n:3*n+1]], dim=1)
            fast = torch.cat([features[:, 1:2*n:2], features[:, 3*n+1:3*n+2]], dim=1)
            scale = TASK["objective_scale"]
            time_delta = ((fast[..., 0]-slow[..., 0])*env.horizon_s
                          / float(scale["flight_time_seconds_divisor"])/0.01)
            energy_delta = (((fast[..., 1]+fast[..., 3])
                             -(slow[..., 1]+slow[..., 3]))*BUDGET_J
                            / float(scale["energy_joules_divisor"])/0.01)
            late_delta = ((fast[..., 4]-slow[..., 4])*env.horizon_s
                          / float(scale["tardiness_seconds_divisor"])/0.01)
            alpha = env.pref[:, 0, None].expand_as(time_delta)
            beta = env.pref[:, 1, None].expand_as(time_delta)
            weighted_late = (late_delta if env.objective_variant == "C0"
                             else alpha*late_delta)
            speed_inputs = torch.stack([
                alpha, beta,
                time_delta, energy_delta, late_delta,
                alpha*time_delta, beta*energy_delta, weighted_late,
                slow[..., 5], fast[..., 5],
            ], dim=-1)
            if self.use_speed_lookahead_inputs:
                speed_inputs = torch.cat([
                    speed_inputs, self._speed_lookahead_features(env),
                ], dim=-1)
            if not self.use_speed_physical_inputs:
                # Remove every non-preference speed input, including the
                # downstream lookahead features, in the physical-input ablation.
                speed_inputs = torch.cat([
                    speed_inputs[..., :2],
                    torch.zeros_like(speed_inputs[..., 2:]),
                ], dim=-1)
            factorized_speed_logits = self.speed_head(speed_inputs).squeeze(-1)
            flight_legal = mask[:, :2*n].reshape(b, n, 2).to(scores.dtype)
            flight_base = ((scores[:, :2*n].reshape(b, n, 2)*flight_legal).sum(dim=-1)
                           / flight_legal.sum(dim=-1).clamp_min(1))
            # The route REINFORCE gradient trains customer/return selection;
            # local physical counterfactuals train the separate speed head.
            if self.training and self.use_speed_aux:
                detached_speed_logits = factorized_speed_logits.detach()
                used_speed_logits = (detached_speed_logits
                                     + self.speed_route_grad_scale
                                     * (factorized_speed_logits-detached_speed_logits))
            else:
                used_speed_logits = factorized_speed_logits
            flight_diff = used_speed_logits[:, :n]
            flight_modes = torch.stack([-0.5*flight_diff, 0.5*flight_diff], dim=-1)
            if self.use_decoupled_route_speed:
                # The sum of the legal speed probabilities is one, so the
                # customer's marginal route logit is exactly flight_base.
                # This prevents |speed_diff| from attracting route choices.
                flight_modes = torch.log_softmax(
                    flight_modes.masked_fill(~mask[:, :2*n].reshape(b, n, 2), -1e9),
                    dim=-1)
            flight_scores = (flight_base[..., None]+flight_modes).reshape(b, 2*n)
            return_legal = mask[:, 3*n:3*n+2].to(scores.dtype)
            return_base = ((scores[:, 3*n:3*n+2]*return_legal).sum(dim=-1, keepdim=True)
                           / return_legal.sum(dim=-1, keepdim=True).clamp_min(1))
            return_diff = used_speed_logits[:, n:n+1]
            return_modes = torch.cat([-0.5*return_diff, 0.5*return_diff], dim=1)
            if self.use_decoupled_route_speed:
                return_modes = torch.log_softmax(
                    return_modes.masked_fill(~mask[:, 3*n:3*n+2], -1e9), dim=1)
            return_scores = return_base+return_modes
            scores = torch.cat([flight_scores, scores[:, 2*n:3*n],
                                return_scores, scores[:, 3*n+2:]], dim=1)
        wait_urgency_logit = None
        if self.wait_urgency_coefficient or self.wait_slack_coefficient:
            wait_urgency_logit = (
                -self.wait_urgency_coefficient
                * self._depot_wait_urgency_risk(env)
            )
            if self.wait_slack_coefficient:
                wait_urgency_logit = wait_urgency_logit - (
                    self.wait_slack_coefficient
                    * self._depot_wait_slack_consumption(
                        env, self.wait_slack_buffer_fraction)
                )
            scores = torch.cat([
                scores[:, :-1], scores[:, -1:]+wait_urgency_logit[:, None]
            ], dim=1)
        if return_components:
            return {
                "attention_logits": attention_logits,
                "physical_logits": physical_logits,
                "ega_logits": ega_logits,
                "mode_bias_logits": mode_bias_logits,
                "physical_features": features,
                "factorized_speed_logits": factorized_speed_logits,
                "speed_inputs": speed_inputs,
                "wait_urgency_logit": wait_urgency_logit,
                "total_logits": scores,
            }
        return scores

    @staticmethod
    def _expand(tensor, pomo):
        return tensor[:, None].expand(-1, pomo, *tensor.shape[1:]).reshape(
            tensor.size(0)*pomo, *tensor.shape[1:]
        )

    @torch.no_grad()
    def _speed_aux_target(self, env: RouteEnvironment, customer_idx: torch.Tensor,
                          flight: torch.Tensor) -> torch.Tensor:
        """Exact immediate C0/C1 fast-minus-slow cost at the current state.

        This omits future route effects, so it is only an auxiliary local
        physics target. Hard feasibility remains in the shared action mask.
        """
        scale = TASK["objective_scale"]
        alpha = env.pref[:, 0, None]
        beta = env.pref[:, 1, None]
        due = env.nodes[:, :, 4]
        service = env.nodes[:, :, 5]
        costs = []
        for speed in SPEEDS_M_S:
            _, travel, _, fly_e, hover_e, finish = env._customer_options(speed)
            late = torch.clamp_min(finish-service-due, 0)
            flight_term = travel / float(scale["flight_time_seconds_divisor"])
            energy_term = (fly_e+hover_e) / float(scale["energy_joules_divisor"])
            late_term = late / float(scale["tardiness_seconds_divisor"])
            if env.objective_variant == "C0":
                costs.append(alpha*flight_term + beta*energy_term + late_term)
            else:
                costs.append(alpha*(flight_term+late_term) + beta*energy_term)
        rows = torch.arange(env.B, device=env.dev)
        flight_delta = (costs[1]-costs[0])[rows, customer_idx]
        back_dist = torch.linalg.vector_norm(env.pos-env.depot, dim=-1)*SIDE_M
        return_costs = []
        for speed in SPEEDS_M_S:
            return_costs.append(
                env.pref[:, 0]*back_dist/speed/float(scale["flight_time_seconds_divisor"])
                + env.pref[:, 1]*env.cruise_j(back_dist, speed, env.load_kg).to(env.dtype)
                / float(scale["energy_joules_divisor"])
            )
        delta = torch.where(flight, flight_delta, return_costs[1]-return_costs[0])
        return (-delta/0.01).clamp(-4.0, 4.0)

    @torch.no_grad()
    def _speed_aux_target_from_features(self, env: RouteEnvironment,
                                        features: torch.Tensor,
                                        customer_idx: torch.Tensor,
                                        flight: torch.Tensor) -> torch.Tensor:
        """Same target using already computed action physics, with no replay."""
        rows = torch.arange(env.B, device=env.dev)
        slow_flight = features[rows, 2*customer_idx]
        fast_flight = features[rows, 2*customer_idx+1]
        slow_return = features[:, 3*env.N]
        fast_return = features[:, 3*env.N+1]
        slow = torch.where(flight[:, None], slow_flight, slow_return)
        fast = torch.where(flight[:, None], fast_flight, fast_return)
        scale = TASK["objective_scale"]
        alpha, beta = env.pref[:, 0], env.pref[:, 1]
        def local_cost(x):
            time = x[:, 0]*env.horizon_s/float(scale["flight_time_seconds_divisor"])
            energy = (x[:, 1]+x[:, 3])*BUDGET_J/float(scale["energy_joules_divisor"])
            late = x[:, 4]*env.horizon_s/float(scale["tardiness_seconds_divisor"])
            if env.objective_variant == "C0":
                return alpha*time + beta*energy + late
            return alpha*(time+late) + beta*energy
        return (-(local_cost(fast)-local_cost(slow))/0.01).clamp(-4.0, 4.0)

    def rollout(self, nodes, depot, pref, *, pomo=None, greedy=False,
                greedy_strategy="joint",
                return_actions=False, reserve_fraction=0.0,
                objective_variant="C0", energy_scenario: EnergyScenario | None = None,
                fixed_speed_mode=None):
        environment_type = RouteEnvironment
        environment_kwargs = {}
        if fixed_speed_mode is not None:
            if self.training or not greedy:
                raise ValueError("Fixed-speed intervention is greedy evaluation only")
            from revision_v3.speed_intervention import FixedSpeedEnvironment
            environment_type = FixedSpeedEnvironment
            environment_kwargs["fixed_speed_mode"] = fixed_speed_mode
        b, n, _ = nodes.shape
        pomo = n if pomo is None else min(int(pomo), n)
        node_emb, depot_emb = self.encode(nodes, depot)
        pmo_params = None
        if self.use_pmo_hyper:
            pmo_params = tuple(self._expand(parameter, pomo)
                               for parameter in self.pmo_hyper(pref))
        env = environment_type(self._expand(nodes, pomo), self._expand(depot, pomo),
                               self._expand(pref, pomo), reserve_fraction=reserve_fraction,
                               objective_variant=objective_variant,
                               energy_scenario=energy_scenario, **environment_kwargs)
        node_emb = self._expand(node_emb, pomo)
        depot_emb = self._expand(depot_emb, pomo)
        # Action identities and preference-generated key/value transforms stay
        # fixed for the entire route. Reusing their autograd graph across
        # decisions avoids storing hundreds of duplicate projections.
        static_action_emb = self._action_keys(node_emb, depot_emb)
        if self.use_pmo_hyper:
            static_keys = torch.bmm(static_action_emb, pmo_params[1])
            static_values = torch.bmm(static_action_emb, pmo_params[2])
        else:
            static_keys = self.key_proj(static_action_emb)
            static_values = None
        current = depot_emb
        logprob_sum = torch.zeros(b*pomo, device=nodes.device, dtype=nodes.dtype)
        entropy_sum = torch.zeros_like(logprob_sum)
        speed_aux_sum = torch.zeros((), device=nodes.device, dtype=nodes.dtype)
        speed_aux_count = torch.zeros((), device=nodes.device, dtype=nodes.dtype)
        history = [] if return_actions else None
        history_active = [] if return_actions else None

        initial_mask = env.get_mask()
        # POMO starts are first *loading* actions from the depot. Typically
        # only a small subset of customers can be loaded at time zero. An
        # arange(N) start then repeats the first legal load for every illegal
        # index, strongly biasing the training baseline and wasting greedy
        # POMO slots. Spread starts evenly over all currently legal loads for
        # both full training POMO and reduced online POMO. When no load is
        # legal, the existing action-mask fallback selects depot waiting.
        legal_load = initial_mask.reshape(b, pomo, -1)[:, 0,
                                          env.load_offset:env.load_offset+n]
        customer_ids = torch.arange(n, device=nodes.device)[None, :].expand(b, -1)
        available = torch.where(legal_load, customer_ids, n).sort(dim=1).values
        counts = legal_load.sum(dim=1, keepdim=True)
        ranks = torch.div(torch.arange(pomo, device=nodes.device)[None, :] * counts,
                          pomo, rounding_mode="floor").clamp(max=n-1)
        starts = available.gather(1, ranks)
        first = env.load_offset + starts.reshape(-1)
        rows = torch.arange(b*pomo, device=nodes.device)
        initially_active = ~env.done()
        first = torch.where(initial_mask[rows, first], first,
                            initial_mask.to(torch.float32).argmax(dim=1))
        env.step(first)
        if return_actions:
            history.append(first)
            history_active.append(initially_active)

        # Legal routes can reload parcels after a depot return. A physically
        # feasible N100 pilot construction used 436 actions, and a harder
        # efficiency/reserve sensitivity pilot used 662. A 4*N+8 guard would
        # silently cut off both. Keep a finite guard against no-progress
        # cycles and explicitly report any trajectory that still reaches it.
        max_decisions = 10*n + 32
        for _ in range(max_decisions):
            done = env.done()
            if bool(done.all()):
                break
            mask = env.get_mask(done)
            components = self.logits(env, node_emb, depot_emb, current,
                                     pmo_params=pmo_params, mask=mask,
                                     return_components=self.training and self.use_speed_aux,
                                     static_action_emb=static_action_emb,
                                     static_keys=static_keys,
                                     static_values=static_values)
            logits = components["total_logits"] if isinstance(components, dict) else components
            logits = logits.masked_fill(~mask, -1e9)
            distribution = torch.distributions.Categorical(logits=logits)
            if greedy:
                from revision_v3.selection import greedy_action
                action = greedy_action(logits, mask, n, greedy_strategy)
            else:
                action = distribution.sample()
            if self.training and self.use_speed_aux:
                rows = torch.arange(b*pomo, device=nodes.device)
                chosen_flight = action < 2*n
                customer_idx = (action//2).clamp(max=n-1)
                flight_pair = (mask[rows, 2*customer_idx]
                               & mask[rows, 2*customer_idx+1])
                chosen_return = ((action >= 3*n) & (action < 3*n+2))
                return_pair = mask[:, 3*n] & mask[:, 3*n+1]
                valid_aux = (~done) & ((chosen_flight & flight_pair)
                                       | (chosen_return & return_pair))
                if self.use_factorized_speed:
                    flight_diff = components["factorized_speed_logits"][rows, customer_idx]
                    return_diff = components["factorized_speed_logits"][:, n]
                else:
                    flight_diff = logits[rows, 2*customer_idx+1] - logits[rows, 2*customer_idx]
                    return_diff = logits[:, 3*n+1] - logits[:, 3*n]
                predicted = torch.where(chosen_flight, flight_diff, return_diff)
                target = self._speed_aux_target_from_features(
                    env, components["physical_features"], customer_idx, chosen_flight)
                speed_aux_sum = speed_aux_sum + (
                    (predicted-target).square() * valid_aux.to(nodes.dtype)
                ).sum()
                speed_aux_count = speed_aux_count + valid_aux.sum().to(nodes.dtype)
            logprob_sum = logprob_sum + torch.where(done, 0.0, distribution.log_prob(action))
            entropy_sum = entropy_sum + torch.where(done, 0.0, distribution.entropy())
            env.step(action, inactive=done)
            fly = action < 2*n
            idx = (action//2).clamp(max=n-1)
            current = torch.where(fly[:, None], node_emb[rows, idx],
                                  torch.where((action >= 3*n)[:, None], depot_emb, current))
            if return_actions:
                history.append(action)
                history_active.append(~done)
        rollout_limit_hit = (~env.done()).reshape(b, pomo)
        score = env.score().reshape(b, pomo)
        unserved = (~env.served).sum(1).reshape(b, pomo)
        feasible = env.complete().reshape(b, pomo)
        rules = TASK["training_selection_penalties"]
        penalty = (score + float(rules["per_unserved_order"])*unserved
                   + float(rules["incomplete_or_invalid_route"])
                   *(~feasible).to(score.dtype))
        result = {
            "cost": score, "selection_cost": penalty,
            "unserved": unserved, "feasible": feasible,
            "rollout_limit_hit": rollout_limit_hit,
            "logprob_sum": logprob_sum.reshape(b, pomo),
            "entropy": entropy_sum.reshape(b, pomo),
            "speed_aux_loss": speed_aux_sum/speed_aux_count.clamp_min(1.0),
            "speed_aux_count": speed_aux_count.detach(),
            "telemetry": env.telemetry(),
        }
        if return_actions:
            result["actions"] = torch.stack(history, dim=1).reshape(b, pomo, -1)
            result["action_valid"] = torch.stack(history_active, dim=1).reshape(b, pomo, -1)
        return result
