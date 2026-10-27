"""The sole v3 plan adapter used by neural, heuristic and proxy baselines."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import torch

from env.revision_v3_gu_physics import EnergyScenario
from revision_v3.environment import RouteEnvironment
from revision_v3.spec import Q_MAX_KG, SPEEDS_M_S, TASK


@dataclass(frozen=True)
class Sortie:
    # All parcels in visits are loaded at depot before departure. A visit is
    # (zero-based customer ID, mode 0=12 m/s or 1=15 m/s).
    visits: tuple[tuple[int, int], ...]
    return_mode: int = 0

    @property
    def manifest(self):
        return tuple(idx for idx, _ in self.visits)


@dataclass(frozen=True)
class RoutePlan:
    sorties: tuple[Sortie, ...]
    provenance: str = "direct_physical_plan"
    metadata: dict = field(default_factory=dict)


def replay(nodes: torch.Tensor, depot: torch.Tensor, pref: torch.Tensor,
           plan: RoutePlan, *, reserve_fraction: float = 0.0,
           objective_variant: str = "C0",
           energy_scenario: EnergyScenario | None = None):
    """Replay one instance with exact manifest, physics, clock and hard masks.

    Returns an explicit failure reason. No solver's own objective is trusted as
    the common v3 cost or feasibility value.
    """
    if nodes.ndim == 2:
        nodes = nodes[None]
    if depot.ndim == 1:
        depot = depot[None]
    if pref.ndim == 1:
        pref = pref[None]
    if nodes.size(0) != 1 or depot.size(0) != 1 or pref.size(0) != 1:
        raise ValueError("Common replay evaluates one fixed instance at a time")
    env = RouteEnvironment(nodes, depot, pref, reserve_fraction=reserve_fraction,
                           record=True, objective_variant=objective_variant,
                           energy_scenario=energy_scenario)
    used = set()
    for sortie_idx, sortie in enumerate(plan.sorties):
        if not sortie.visits:
            return _report(env, plan, False, "empty_sortie", sortie_idx)
        if len(set(sortie.manifest)) != len(sortie.manifest):
            return _report(env, plan, False, "duplicate_in_manifest", sortie_idx)
        if any(i < 0 or i >= env.N or i in used for i in sortie.manifest):
            return _report(env, plan, False, "unknown_or_duplicate_order", sortie_idx)
        mass = sum(float(nodes[0, i, 2]) for i in sortie.manifest)
        if mass > Q_MAX_KG + 1e-6:
            return _report(env, plan, False, "payload_capacity", sortie_idx)
        for i in sortie.manifest:
            action = env.load_offset + i
            if not bool(env.raw_mask()[0, action]):
                return _report(env, plan, False, f"load_not_feasible:{i}", sortie_idx)
            env.step(torch.tensor([action], device=nodes.device))
        for i, mode in sortie.visits:
            if mode not in (0, 1):
                return _report(env, plan, False, "invalid_speed_mode", sortie_idx)
            action = 2*i + mode
            if not bool(env.raw_mask()[0, action]):
                return _report(env, plan, False, f"delivery_not_feasible:{i}:{mode}", sortie_idx)
            env.step(torch.tensor([action], device=nodes.device))
            used.add(i)
        if sortie.return_mode not in (0, 1):
            return _report(env, plan, False, "invalid_return_mode", sortie_idx)
        back_action = env.return_offset + sortie.return_mode
        if not bool(env.raw_mask()[0, back_action]):
            return _report(env, plan, False, "depot_return_not_feasible", sortie_idx)
        env.step(torch.tensor([back_action], device=nodes.device))
    return _report(env, plan, bool(env.complete()[0]),
                   None if bool(env.complete()[0]) else "unserved_or_horizon", len(plan.sorties))


def _report(env, plan, feasible, reason, failed_sortie):
    telemetry = env.telemetry()
    scalar = {}
    for key, value in telemetry.items():
        if key in {"loaded", "served", "warehouse"}:
            scalar[key] = [bool(x) for x in value[0].tolist()]
        elif value[0].numel() == 1:
            scalar[key] = value[0].item()
    deliveries = [event for event in env.events[0] if event["kind"] == "deliver"]
    returns = [event for event in env.events[0] if event["kind"] == "return"]
    scalar["delivery_leg_count"] = len(deliveries)
    scalar["return_leg_count"] = len(returns)
    scalar["fast_delivery_fraction"] = (
        sum(event["speed_m_s"] == SPEEDS_M_S[1] for event in deliveries)
        / len(deliveries) if deliveries else None)
    scalar["fast_return_fraction"] = (
        sum(event["speed_m_s"] == SPEEDS_M_S[1] for event in returns)
        / len(returns) if returns else None)
    all_flights = deliveries + returns
    scalar["fast_all_flight_fraction"] = (
        sum(event["speed_m_s"] == SPEEDS_M_S[1] for event in all_flights)
        / len(all_flights) if all_flights else None)
    if plan.sorties:
        sortie_rows = [
            {"manifest": list(s.manifest), "visits": [list(x) for x in s.visits],
             "return_mode": s.return_mode} for s in plan.sorties
        ]
    else:
        sortie_rows = []
        active_sortie = None
        for event in env.events[0]:
            if event["kind"] == "depart_depot":
                active_sortie = {
                    "sortie_index": event["sortie_index"],
                    "manifest": event["manifest_order_ids"],
                    "manifest_mass_kg": event["manifest_mass_kg"],
                    "visits": [], "return_mode": None,
                    "returned_order_ids": [],
                }
            elif event["kind"] == "deliver" and active_sortie is not None:
                mode = 0 if event["speed_m_s"] == SPEEDS_M_S[0] else 1
                active_sortie["visits"].append([event["order"], mode])
            elif event["kind"] == "return" and active_sortie is not None:
                active_sortie["return_mode"] = (
                    0 if event["speed_m_s"] == SPEEDS_M_S[0] else 1
                )
                active_sortie["returned_order_ids"] = event["returned_order_ids"]
                sortie_rows.append(active_sortie)
                active_sortie = None
        if active_sortie is not None:
            sortie_rows.append(active_sortie)
    return {
        "physics_id": env.energy_scenario.id,
        "battery_budget_j": env.budget_j,
        "additional_reserve_j": env.reserve_j,
        "task_id": TASK["task_id"],
        "objective_variant": env.objective_variant,
        "plan_provenance": plan.provenance,
        "plan_metadata": plan.metadata,
        "feasible": feasible,
        "failure_reason": reason,
        "failed_sortie_index": failed_sortie,
        "metrics": scalar,
        "events": env.events[0],
        "sorties": sortie_rows,
    }


def plan_from_explicit_actions(actions: Iterable[int], n_nodes: int) -> RoutePlan:
    """Convert a neural action stream to sorties, preserving loading IDs.

    A neural policy can load an order and return without delivering it. That
    trajectory cannot be silently turned into a complete RoutePlan; it raises.
    """
    sorties = []
    loaded = []
    visits = []
    for a in actions:
        a = int(a)
        if 2*n_nodes <= a < 3*n_nodes:
            if visits:
                raise ValueError("Loading outside a depot loading phase")
            loaded.append(a-2*n_nodes)
        elif 0 <= a < 2*n_nodes:
            i, mode = divmod(a, 2)
            if i not in loaded:
                raise ValueError("Neural trajectory serves an unloaded order")
            loaded.remove(i)
            visits.append((i, mode))
        elif 3*n_nodes <= a < 3*n_nodes+2:
            if loaded:
                raise ValueError("Undelivered loaded parcels remain at depot return")
            sorties.append(Sortie(tuple(visits), a-3*n_nodes))
            visits = []
        elif a == 3*n_nodes+2:
            raise ValueError("Depot waiting actions require replay_actions, not RoutePlan conversion")
        else:
            raise ValueError(f"Unknown action ID: {a}")
    if visits or loaded:
        raise ValueError("Action stream does not finish with an empty-cargo depot return")
    return RoutePlan(tuple(sorties), provenance="explicit_neural_actions")


def replay_actions(nodes: torch.Tensor, depot: torch.Tensor, pref: torch.Tensor,
                   actions: Iterable[int], *, reserve_fraction: float = 0.0,
                   provenance: str = "explicit_action_stream",
                   objective_variant: str = "C0",
                   energy_scenario: EnergyScenario | None = None):
    """Evaluate a neural action stream, including returned undelivered cargo."""
    if nodes.ndim == 2:
        nodes = nodes[None]
    if depot.ndim == 1:
        depot = depot[None]
    if pref.ndim == 1:
        pref = pref[None]
    if nodes.size(0) != 1 or depot.size(0) != 1 or pref.size(0) != 1:
        raise ValueError("Common replay evaluates one fixed instance at a time")
    env = RouteEnvironment(nodes, depot, pref, reserve_fraction=reserve_fraction,
                           record=True, objective_variant=objective_variant,
                           energy_scenario=energy_scenario)
    stream = [int(a) for a in actions]
    for k, action in enumerate(stream):
        if action < 0 or action >= env.action_count:
            return _report(env, RoutePlan((), provenance), False,
                           f"unknown_action:{action}", k)
        if bool(env.complete()[0]):
            return _report(env, RoutePlan((), provenance), False,
                           "action_after_complete", k)
        if not bool(env.raw_mask()[0, action]):
            return _report(env, RoutePlan((), provenance), False,
                           f"infeasible_action:{action}", k)
        env.step(torch.tensor([action], device=nodes.device))
    feasible = bool(env.complete()[0])
    result = _report(env, RoutePlan((), provenance), feasible,
                     None if feasible else "unfinished_action_stream", len(stream))
    result["actions"] = stream
    return result
