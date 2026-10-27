"""Vectorised, physically consistent warehouse-loading and flight state.

Node columns: x, y (unit-square coordinates), parcel mass (kg), ready (s),
desired due (s), hovering handoff time (s). An order is deliverable only when
its parcel is explicitly in ``loaded``. The aircraft physics is imported from
the single Gu kernel, never reimplemented in the decoder or baselines.

Action IDs: 0..2N-1 = fly to customer i at speed i%2; 2N..3N-1 = load
parcel i at depot; 3N/3N+1 = return at economical/fast speed; 3N+2 =
wait at depot until the next useful order-release departure time.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from env.revision_v3_gu_physics import EnergyScenario, SCENARIOS
from revision_v3.spec import (
    BUDGET_J, DEPOT_TURN_S, Q_MAX_KG, RESERVE_FRACTION, SIDE_M,
    SPEEDS_M_S, horizon_s, objective,
)


@dataclass
class Snapshot:
    """Complete restart state; all tensors have a batch leading dimension."""

    position: torch.Tensor
    clock_s: torch.Tensor
    battery_j: torch.Tensor
    onboard_kg: torch.Tensor
    loaded: torch.Tensor
    served: torch.Tensor
    flight_time_s: torch.Tensor
    wait_time_s: torch.Tensor
    service_time_s: torch.Tensor
    depot_time_s: torch.Tensor
    depot_wait_time_s: torch.Tensor
    cruise_energy_j: torch.Tensor
    hover_energy_j: torch.Tensor
    tardiness_s: torch.Tensor
    return_count: torch.Tensor
    min_return_battery_j: torch.Tensor


class RouteEnvironment:
    def __init__(self, nodes: torch.Tensor, depot: torch.Tensor, pref: torch.Tensor,
                 *, reserve_fraction: float = RESERVE_FRACTION, record: bool = False,
                 objective_variant: str = "C0",
                 energy_scenario: EnergyScenario | None = None):
        if nodes.ndim != 3 or nodes.size(-1) != 6:
            raise ValueError("v3 nodes must have shape [batch, customers, 6]")
        if depot.shape != (nodes.size(0), 2) or pref.shape != (nodes.size(0), 2):
            raise ValueError("v3 depot and preference must have shapes [batch,2]")
        if not bool(torch.isfinite(nodes).all() & torch.isfinite(depot).all()
                    & torch.isfinite(pref).all()):
            raise ValueError("v3 instance and preference must be finite")
        if bool((nodes[:, :, 2] < 0).any() or (nodes[:, :, 2] > Q_MAX_KG).any()
                or (nodes[:, :, 3] < 0).any() or (nodes[:, :, 5] < 0).any()):
            raise ValueError("v3 cargo, ready time and service must be nonnegative and cargo within capacity")
        self.nodes = nodes
        self.depot = depot
        self.pref = pref
        self.B, self.N, _ = nodes.shape
        self.dev = nodes.device
        self.dtype = nodes.dtype
        self.horizon_s = horizon_s(self.N)
        self.energy_scenario = energy_scenario or SCENARIOS["gu_nominal"]
        self.budget_j = self.energy_scenario.battery_budget_j
        if not 0 <= reserve_fraction < 1:
            raise ValueError("Additional reserve fraction must be in [0,1)")
        self.reserve_j = self.budget_j * float(reserve_fraction)
        if objective_variant not in ("C0", "C1"):
            raise ValueError("Objective variant must be C0 or C1")
        self.objective_variant = objective_variant
        self.record = bool(record)
        self.events = [[] for _ in range(self.B)] if record else None
        self.depot_dist_m = torch.linalg.vector_norm(
            self.nodes[:, :, :2] - self.depot[:, None, :], dim=-1
        ) * SIDE_M
        self.reset()

    def cruise_j(self, distance_m, speed_m_s, payload_kg):
        return self.energy_scenario.cruise_j(
            distance_m, speed_m_s, payload_kg, _assume_valid=True
        )

    def hover_j(self, seconds, payload_kg):
        return self.energy_scenario.hover_j(seconds, payload_kg, _assume_valid=True)

    @property
    def load_offset(self) -> int:
        return 2 * self.N

    @property
    def return_offset(self) -> int:
        return 3 * self.N

    @property
    def action_count(self) -> int:
        return 3 * self.N + 3

    @property
    def wait_action(self) -> int:
        return 3 * self.N + 2

    def reset(self):
        b, n = self.B, self.N
        z = lambda: torch.zeros(b, dtype=self.dtype, device=self.dev)
        self.pos = self.depot.clone()
        self.time_s = z()
        self.battery_j = torch.full((b,), self.budget_j, dtype=self.dtype, device=self.dev)
        self.load_kg = z()
        self.loaded = torch.zeros(b, n, dtype=torch.bool, device=self.dev)
        self.served = torch.zeros(b, n, dtype=torch.bool, device=self.dev)
        self.flight_time_s = z()
        self.wait_time_s = z()
        self.service_time_s = z()
        self.depot_time_s = z()
        self.depot_wait_time_s = z()
        self.cruise_energy_j = z()
        self.hover_energy_j = z()
        self.tardiness_s = z()
        self.return_count = torch.zeros(b, dtype=torch.long, device=self.dev)
        self.min_return_battery_j = torch.full((b,), self.budget_j, dtype=self.dtype, device=self.dev)
        self._invalidate_action_cache()

    def _invalidate_action_cache(self):
        # A mask/feature query is repeated by done(), get_mask(), policy
        # logits and step() at each unchanged state. Keep one physical replay
        # per state; invalidate only after the state actually advances.
        self._mask_cache = None
        self._customer_options_cache = {}
        self._customer_return_energies_cache = None

    def snapshot(self) -> Snapshot:
        return Snapshot(
            position=self.pos.clone(), clock_s=self.time_s.clone(),
            battery_j=self.battery_j.clone(), onboard_kg=self.load_kg.clone(),
            loaded=self.loaded.clone(), served=self.served.clone(),
            flight_time_s=self.flight_time_s.clone(),
            wait_time_s=self.wait_time_s.clone(),
            service_time_s=self.service_time_s.clone(),
            depot_time_s=self.depot_time_s.clone(),
            depot_wait_time_s=self.depot_wait_time_s.clone(),
            cruise_energy_j=self.cruise_energy_j.clone(),
            hover_energy_j=self.hover_energy_j.clone(),
            tardiness_s=self.tardiness_s.clone(),
            return_count=self.return_count.clone(),
            min_return_battery_j=self.min_return_battery_j.clone(),
        )

    def load_snapshot(self, snapshot: Snapshot):
        for name in Snapshot.__dataclass_fields__:
            value = getattr(snapshot, name)
            if value.shape[0] != self.B:
                raise ValueError(f"Snapshot batch mismatch: {name}")
        self.pos = snapshot.position.clone()
        self.time_s = snapshot.clock_s.clone()
        self.battery_j = snapshot.battery_j.clone()
        self.load_kg = snapshot.onboard_kg.clone()
        self.loaded = snapshot.loaded.clone()
        self.served = snapshot.served.clone()
        for name in (
            "flight_time_s", "wait_time_s", "service_time_s", "depot_time_s",
            "depot_wait_time_s",
            "cruise_energy_j", "hover_energy_j", "tardiness_s", "return_count",
            "min_return_battery_j",
        ):
            setattr(self, name, getattr(snapshot, name).clone())
        self._assert_inventory()
        self._invalidate_action_cache()

    def _assert_inventory(self):
        if bool(torch.any(self.loaded & self.served)):
            raise ValueError("A delivered parcel cannot remain onboard")
        mass = (self.loaded.to(self.dtype) * self.nodes[:, :, 2]).sum(dim=1)
        if bool(torch.any(torch.abs(mass - self.load_kg) > 1e-4)):
            raise ValueError("Onboard mass differs from the parcel manifest")
        if bool(torch.any(self.load_kg < -1e-6) or torch.any(self.load_kg > Q_MAX_KG + 1e-6)):
            raise ValueError("Onboard mass violates the payload capacity")

    def at_depot(self) -> torch.Tensor:
        return torch.linalg.vector_norm(self.pos - self.depot, dim=-1) < 1e-6

    def complete(self) -> torch.Tensor:
        return self.served.all(dim=1) & self.at_depot() & (self.time_s <= self.horizon_s)

    def _customer_options(self, speed: float):
        cached = self._customer_options_cache.get(speed)
        if cached is not None:
            return cached
        xy = self.nodes[:, :, :2]
        ready = self.nodes[:, :, 3]
        service_s = self.nodes[:, :, 5]
        dist_m = torch.linalg.vector_norm(xy - self.pos[:, None, :], dim=-1) * SIDE_M
        turn_s = self.at_depot().to(self.dtype) * DEPOT_TURN_S
        depart_s = self.time_s[:, None] + turn_s[:, None]
        travel_s = dist_m / speed
        arrival_s = depart_s + travel_s
        wait_s = torch.clamp_min(ready - arrival_s, 0)
        finish_s = arrival_s + wait_s + service_s
        load = self.load_kg[:, None].expand(-1, self.N)
        fly_e = self.cruise_j(dist_m, speed, load).to(self.dtype)
        hover_e = self.hover_j(wait_s + service_s, load).to(self.dtype)
        after_e = self.battery_j[:, None] - fly_e - hover_e
        safe_return = torch.zeros_like(after_e, dtype=torch.bool)
        return_energies = self.customer_return_energies()
        for mode, back_speed in enumerate(SPEEDS_M_S):
            back_e = return_energies[:, :, mode]
            back_s = self.depot_dist_m / back_speed
            safe_return |= ((after_e - back_e >= self.reserve_j - 1e-3)
                            & (finish_s + back_s <= self.horizon_s + 1e-4))
        result = (safe_return, travel_s, wait_s, fly_e, hover_e, finish_s)
        self._customer_options_cache[speed] = result
        return result

    def customer_return_energies(self) -> torch.Tensor:
        """Energy to depot after delivering each candidate customer."""
        if self._customer_return_energies_cache is None:
            after_load = torch.clamp_min(
                self.load_kg[:, None] - self.nodes[:, :, 2], 0
            )
            self._customer_return_energies_cache = torch.stack([
                self.cruise_j(self.depot_dist_m, speed, after_load).to(self.dtype)
                for speed in SPEEDS_M_S
            ], dim=-1)
        return self._customer_return_energies_cache

    def next_depot_release_s(self) -> torch.Tensor:
        """Earliest useful depot departure time for an unreleased order."""
        candidate = self.nodes[:, :, 3] - DEPOT_TURN_S - self.depot_dist_m/SPEEDS_M_S[1]
        possible = (~self.served) & (candidate > self.time_s[:, None] + 1.0)
        candidate = torch.where(possible, candidate, torch.full_like(candidate, float("inf")))
        return candidate.min(dim=1).values

    def raw_mask(self) -> torch.Tensor:
        if self._mask_cache is not None:
            return self._mask_cache
        mask = torch.zeros(self.B, self.action_count, dtype=torch.bool, device=self.dev)
        at_depot = self.at_depot()
        remaining = ~self.served
        can_fly_order = self.loaded & remaining
        for mode, speed in enumerate(SPEEDS_M_S):
            safe, *_ = self._customer_options(speed)
            mask[:, mode:2*self.N:2] = can_fly_order & safe
        # Loading is a separate warehouse operation. It neither moves the UAV
        # nor consumes/creates cargo; each parcel ID can be loaded only once.
        capacity_ok = self.load_kg[:, None] + self.nodes[:, :, 2] <= Q_MAX_KG + 1e-6
        departure_possible = self.time_s[:, None] + DEPOT_TURN_S < self.horizon_s
        # A new manifest cannot include an order that is already impossible
        # to fly and bring home as a one-customer sortie from the depot.
        candidate_load = (self.load_kg[:, None] + self.nodes[:, :, 2]).clamp(max=Q_MAX_KG)
        depot_dist = self.depot_dist_m
        candidate_possible = torch.zeros_like(capacity_ok)
        candidate_return_energy = {
            speed: self.cruise_j(depot_dist, speed, self.load_kg[:, None]).to(self.dtype)
            for speed in SPEEDS_M_S
        }
        for outbound_speed in SPEEDS_M_S:
            arrival = self.time_s[:, None] + DEPOT_TURN_S + depot_dist/outbound_speed
            wait = torch.clamp_min(self.nodes[:, :, 3] - arrival, 0)
            finish = arrival + wait + self.nodes[:, :, 5]
            e_out = self.cruise_j(depot_dist, outbound_speed, candidate_load).to(self.dtype)
            e_hover = self.hover_j(wait + self.nodes[:, :, 5], candidate_load).to(self.dtype)
            for inbound_speed in SPEEDS_M_S:
                e_back = candidate_return_energy[inbound_speed]
                candidate_possible |= (
                    (self.budget_j - e_out - e_hover - e_back >= self.reserve_j - 1e-3)
                    & (finish + depot_dist/inbound_speed <= self.horizon_s + 1e-4)
                )
        mask[:, self.load_offset:self.return_offset] = (
            at_depot[:, None] & remaining & ~self.loaded & capacity_ok
            & departure_possible & candidate_possible
        )
        back_dist = torch.linalg.vector_norm(self.pos - self.depot, dim=-1) * SIDE_M
        for mode, speed in enumerate(SPEEDS_M_S):
            back_e = self.cruise_j(back_dist, speed, self.load_kg).to(self.dtype)
            mask[:, self.return_offset + mode] = (
                ~at_depot
                & (self.battery_j - back_e >= self.reserve_j - 1e-3)
                & (self.time_s + back_dist/speed <= self.horizon_s + 1e-4)
            )
        release = self.next_depot_release_s()
        mask[:, self.wait_action] = (
            at_depot & (~self.served).any(dim=1)
            & torch.isfinite(release) & (release <= self.horizon_s)
        )
        self._mask_cache = mask
        return mask

    def done(self) -> torch.Tensor:
        mask = self.raw_mask()
        return self.complete() | ~mask.any(dim=1) | (self.time_s > self.horizon_s)

    def get_mask(self, inactive: torch.Tensor | None = None) -> torch.Tensor:
        mask = self.raw_mask().clone()
        terminal = self.complete() | ~mask.any(dim=1)
        if inactive is not None:
            terminal |= inactive
        # The caller ignores the dummy action when a row is terminal; it only
        # keeps categorical sampling numerically defined.
        mask[terminal] = False
        mask[terminal, 0] = True
        return mask

    def score(self) -> torch.Tensor:
        total_e = self.cruise_energy_j + self.hover_energy_j
        return objective(self.flight_time_s, total_e, self.tardiness_s,
                         self.pref, variant=self.objective_variant)

    def telemetry(self) -> dict[str, torch.Tensor]:
        return {
            "flight_time_s": self.flight_time_s.clone(),
            "wait_time_s": self.wait_time_s.clone(),
            "service_time_s": self.service_time_s.clone(),
            "depot_time_s": self.depot_time_s.clone(),
            "depot_wait_time_s": self.depot_wait_time_s.clone(),
            "completion_clock_s": self.time_s.clone(),
            "cruise_energy_j": self.cruise_energy_j.clone(),
            "hover_energy_j": self.hover_energy_j.clone(),
            "total_energy_j": (self.cruise_energy_j + self.hover_energy_j).clone(),
            "tardiness_s": self.tardiness_s.clone(),
            "unserved": (~self.served).sum(dim=1),
            "depot_returns": self.return_count.clone(),
            "remaining_battery_j": self.battery_j.clone(),
            "min_return_battery_j": self.min_return_battery_j.clone(),
            "onboard_kg": self.load_kg.clone(),
            "loaded": self.loaded.clone(),
            "served": self.served.clone(),
            "warehouse": (~self.loaded & ~self.served).clone(),
            "feasible": self.complete().clone(),
            "cost": self.score().clone(),
        }

    def step(self, actions: torch.Tensor, *, inactive: torch.Tensor | None = None):
        actions = actions.long().reshape(self.B)
        if inactive is None:
            inactive = torch.zeros(self.B, dtype=torch.bool, device=self.dev)
        active = ~inactive & ~self.done()
        allowed = self.raw_mask()
        rows = torch.arange(self.B, device=self.dev)
        if bool(torch.any(active & ~allowed[rows, actions])):
            raise ValueError("Invalid v3 action: parcel, capacity, battery or deadline rule")
        load_act = active & (actions >= self.load_offset) & (actions < self.return_offset)
        fly_act = active & (actions < self.load_offset)
        return_act = active & (actions >= self.return_offset)
        return_act &= actions < self.wait_action
        wait_act = active & (actions == self.wait_action)

        if bool(wait_act.any()):
            release = self.next_depot_release_s()
            elapsed = release[wait_act] - self.time_s[wait_act]
            self.time_s[wait_act] = release[wait_act]
            self.depot_wait_time_s[wait_act] += elapsed
            if self.record:
                for r in rows[wait_act].tolist():
                    self.events[r].append({"kind": "wait_at_depot",
                                           "until_s": float(self.time_s[r]),
                                           "depot_wait_total_s": float(self.depot_wait_time_s[r])})

        if bool(load_act.any()):
            rr = rows[load_act]
            idx = actions[load_act] - self.load_offset
            self.loaded[rr, idx] = True
            self.load_kg[load_act] = (
                self.load_kg[load_act] + self.nodes[rr, idx, 2]
            ).clamp(max=Q_MAX_KG)
            if self.record:
                for r, j in zip(rr.tolist(), idx.tolist()):
                    self.events[r].append({"kind": "load", "order": j,
                                           "sortie_index": int(self.return_count[r])+1,
                                           "clock_s": float(self.time_s[r]),
                                           "onboard_kg": float(self.load_kg[r]),
                                           "manifest_order_ids": torch.where(self.loaded[r])[0].tolist()})

        if bool(fly_act.any()):
            rr = rows[fly_act]
            idx = actions[fly_act] // 2
            modes = actions[fly_act] % 2
            speed = SPEEDS_M_S[0] + modes.to(self.dtype) * (SPEEDS_M_S[1] - SPEEDS_M_S[0])
            was_depot = self.at_depot()[fly_act]
            turn = was_depot.to(self.dtype) * DEPOT_TURN_S
            before_load = self.load_kg[fly_act].clone()
            target = self.nodes[rr, idx, :2]
            dist = torch.linalg.vector_norm(target - self.pos[fly_act], dim=-1) * SIDE_M
            travel = dist / speed
            arrival = self.time_s[fly_act] + turn + travel
            ready = self.nodes[rr, idx, 3]
            due = self.nodes[rr, idx, 4]
            service = self.nodes[rr, idx, 5]
            wait = torch.clamp_min(ready - arrival, 0)
            fly_e = self.cruise_j(dist, speed, before_load).to(self.dtype)
            hover_e = self.hover_j(wait + service, before_load).to(self.dtype)
            self.time_s[fly_act] = arrival + wait + service
            self.flight_time_s[fly_act] += travel
            self.wait_time_s[fly_act] += wait
            self.service_time_s[fly_act] += service
            self.depot_time_s[fly_act] += turn
            self.cruise_energy_j[fly_act] += fly_e
            self.hover_energy_j[fly_act] += hover_e
            self.battery_j[fly_act] -= fly_e + hover_e
            self.tardiness_s[fly_act] += torch.clamp_min(arrival + wait - due, 0)
            self.load_kg[fly_act] = (
                self.load_kg[fly_act] - self.nodes[rr, idx, 2]
            ).clamp_min(0)
            self.loaded[rr, idx] = False
            self.served[rr, idx] = True
            self.pos[fly_act] = target
            if self.record:
                for k, r in enumerate(rr.tolist()):
                    if bool(was_depot[k]):
                        self.events[r].append({
                            "kind": "depart_depot",
                            "sortie_index": int(self.return_count[r])+1,
                            "departure_s": float(self.time_s[r]-travel[k]-wait[k]-service[k]),
                            "manifest_order_ids": torch.where(self.loaded[r])[0].tolist()
                            + [int(idx[k])],
                            "manifest_mass_kg": float(before_load[k]),
                        })
                    self.events[r].append({"kind": "deliver", "order": int(idx[k]),
                                           "speed_m_s": float(speed[k]),
                                           "arrival_s": float(arrival[k]),
                                           "completion_s": float(self.time_s[r]),
                                           "before_load_kg": float(before_load[k]),
                                           "after_load_kg": float(self.load_kg[r]),
                                           "flight_energy_j": float(fly_e[k]),
                                           "hover_energy_j": float(hover_e[k])})

        if bool(return_act.any()):
            rr = rows[return_act]
            modes = actions[return_act] - self.return_offset
            speed = SPEEDS_M_S[0] + modes.to(self.dtype) * (SPEEDS_M_S[1] - SPEEDS_M_S[0])
            dist = torch.linalg.vector_norm(self.pos[return_act] - self.depot[return_act], dim=-1) * SIDE_M
            travel = dist / speed
            before_load = self.load_kg[return_act].clone()
            returned_ids = [torch.where(self.loaded[r])[0].tolist() for r in rr.tolist()]
            fly_e = self.cruise_j(dist, speed, before_load).to(self.dtype)
            self.time_s[return_act] += travel
            self.flight_time_s[return_act] += travel
            self.cruise_energy_j[return_act] += fly_e
            self.battery_j[return_act] -= fly_e
            arrival_battery = self.battery_j[return_act].clone()
            self.min_return_battery_j[return_act] = torch.minimum(
                self.min_return_battery_j[return_act], arrival_battery
            )
            self.pos[return_act] = self.depot[return_act]
            self.return_count[return_act] += 1
            # Undelivered parcels remain unserved; on return they go back to
            # depot inventory. The next sortie needs explicit reloading.
            self.loaded[return_act] = False
            self.load_kg[return_act] = 0
            self.battery_j[return_act] = self.budget_j
            if self.record:
                for k, r in enumerate(rr.tolist()):
                    self.events[r].append({"kind": "return", "speed_m_s": float(speed[k]),
                                           "arrival_s": float(self.time_s[r]),
                                           "returned_cargo_kg": float(before_load[k]),
                                           "returned_order_ids": returned_ids[k],
                                           "sortie_index": int(self.return_count[r]),
                                           "flight_energy_j": float(fly_e[k]),
                                           "battery_on_arrival_j": float(arrival_battery[k])})

        self._assert_inventory()
        self._invalidate_action_cache()
        return self.score(), self.done()
