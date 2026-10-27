"""Change speed decisions while keeping every non-speed action in order.

This is a causal diagnostic, not a feasible-solution repair heuristic.  If a
changed speed makes a later fixed action illegal, the counterfactual is marked
infeasible rather than silently changing the route, load or wait sequence.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch

from env.revision_v3_gu_physics import EnergyScenario
from revision_v3.common_eval import replay_actions
from revision_v3.environment import RouteEnvironment


def base_action(action: int, n_nodes: int) -> tuple[str, int]:
    if 0 <= action < 2*n_nodes:
        return "deliver", action//2
    if 2*n_nodes <= action < 3*n_nodes:
        return "load", action-2*n_nodes
    if 3*n_nodes <= action < 3*n_nodes+2:
        return "return", 0
    if action == 3*n_nodes+2:
        return "wait", 0
    raise ValueError(f"Unknown action ID: {action}")


def analytic_on_fixed_actions(
    nodes: torch.Tensor, depot: torch.Tensor, pref: torch.Tensor,
    reference_actions: Iterable[int], *, objective_variant: str = "C0",
    reserve_fraction: float = 0.0,
    energy_scenario: EnergyScenario | None = None,
    forced_reference_action_indices: frozenset[int] | None = None,
) -> dict:
    """Select exact local C0/C1 speed on a fixed load/deliver/wait/return order."""
    stream = [int(action) for action in reference_actions]
    if nodes.ndim == 3:
        nodes = nodes[0]
    if depot.ndim == 2:
        depot = depot[0]
    if pref.ndim == 2:
        pref = pref[0]
    nodes, depot, pref = (tensor.detach().cpu() for tensor in (nodes, depot, pref))
    n_nodes = nodes.size(0)
    reference = replay_actions(
        nodes, depot, pref, stream, objective_variant=objective_variant,
        reserve_fraction=reserve_fraction, energy_scenario=energy_scenario,
        provenance="fixed_sequence_reference")
    if not reference["feasible"]:
        raise ValueError("Reference trajectory must be complete and physically feasible")
    env = RouteEnvironment(nodes[None], depot[None], pref[None], record=False,
                           objective_variant=objective_variant,
                           reserve_fraction=reserve_fraction,
                           energy_scenario=energy_scenario)
    chosen: list[int] = []
    changed_action_indices: list[int] = []
    both_speeds_legal = 0
    failure: dict | None = None
    forced_reference_action_indices = forced_reference_action_indices or frozenset()
    for index, reference_action in enumerate(stream):
        kind, item = base_action(reference_action, n_nodes)
        allowed = env.raw_mask()[0]
        if kind == "deliver":
            candidates = (2*item, 2*item+1)
        elif kind == "return":
            candidates = (env.return_offset, env.return_offset+1)
        else:
            candidates = (reference_action,)
        legal = [action for action in candidates if bool(allowed[action])]
        if not legal:
            failure = {"at_action_index": index, "base_kind": kind,
                       "base_item": item, "reason": "fixed_base_action_became_illegal"}
            break
        if kind in ("deliver", "return"):
            if len(legal) == 2:
                both_speeds_legal += 1
            snapshot = env.snapshot()
            before = float(env.score()[0])
            costs = []
            for action in legal:
                env.step(torch.tensor([action]))
                costs.append((float(env.score()[0])-before, action))
                env.load_snapshot(snapshot)
            action = (reference_action if index in forced_reference_action_indices
                      else min(costs)[1])
            if action not in legal:
                failure = {"at_action_index": index, "base_kind": kind,
                           "base_item": item,
                           "reason": "forced_reference_speed_became_illegal"}
                break
            if action != reference_action:
                changed_action_indices.append(index)
        else:
            action = reference_action
        env.step(torch.tensor([action]))
        chosen.append(action)
    if failure is None and not bool(env.complete()[0]):
        failure = {"at_action_index": len(stream),
                   "reason": "fixed_sequence_finished_incomplete"}
    counterfactual = (
        replay_actions(nodes, depot, pref, chosen,
                       objective_variant=objective_variant,
                       reserve_fraction=reserve_fraction,
                       energy_scenario=energy_scenario,
                       provenance="fixed_sequence_exact_local_speed")
        if failure is None else None)
    if counterfactual is not None:
        if not counterfactual["feasible"]:
            raise RuntimeError("Environment and common replay disagree on fixed sequence")
        if ([base_action(a, n_nodes) for a in chosen]
                != [base_action(a, n_nodes) for a in stream]):
            raise RuntimeError("Counterfactual changed a non-speed action")
    return {
        "reference_feasible": True,
        "counterfactual_feasible": counterfactual is not None,
        "fixed_base_action_sequence_equal": counterfactual is not None,
        "failure": failure,
        "flight_decisions_with_both_speeds_legal": both_speeds_legal,
        "changed_speed_actions": len(changed_action_indices),
        "changed_action_indices": changed_action_indices,
        "reference_cost": float(reference["metrics"]["cost"]),
        "counterfactual_cost": (
            float(counterfactual["metrics"]["cost"])
            if counterfactual is not None else None),
        "counterfactual_minus_reference_cost": (
            float(counterfactual["metrics"]["cost"])
            - float(reference["metrics"]["cost"])
            if counterfactual is not None else None),
        "reference_metrics": reference["metrics"],
        "counterfactual_metrics": (
            counterfactual["metrics"] if counterfactual is not None else None),
        "reference_actions": stream,
        "counterfactual_actions": chosen if counterfactual is not None else None,
    }


def feasibility_repaired_analytic_on_fixed_actions(
    nodes: torch.Tensor, depot: torch.Tensor, pref: torch.Tensor,
    reference_actions: Iterable[int], *, objective_variant: str = "C0",
    reserve_fraction: float = 0.0,
    energy_scenario: EnergyScenario | None = None,
) -> dict:
    """Backtrack changed speeds until the unchanged base-action stream completes.

    This keeps the original route, load and wait sequence; the repair only
    restores reference speeds at previously changed flight decisions.  It is
    a feasibility-preserving control, not a globally optimal speed solver.
    """
    stream = [int(action) for action in reference_actions]
    forced: set[int] = set()
    original = analytic_on_fixed_actions(
        nodes, depot, pref, stream, objective_variant=objective_variant,
        reserve_fraction=reserve_fraction, energy_scenario=energy_scenario)
    current = original
    for _ in range(len(stream)+1):
        if current["counterfactual_feasible"]:
            current["unrepaired_counterfactual_feasible"] = original[
                "counterfactual_feasible"]
            current["unrepaired_failure"] = original["failure"]
            current["fallback_to_reference_speed_count"] = len(forced)
            current["forced_reference_action_indices"] = sorted(forced)
            return current
        remaining = [index for index in current["changed_action_indices"]
                     if index not in forced]
        if not remaining:
            raise RuntimeError("Cannot restore a feasible fixed action sequence")
        forced.add(remaining[-1])
        current = analytic_on_fixed_actions(
            nodes, depot, pref, stream, objective_variant=objective_variant,
            reserve_fraction=reserve_fraction, energy_scenario=energy_scenario,
            forced_reference_action_indices=frozenset(forced))
    raise RuntimeError("Fixed-sequence repair exceeded its finite action bound")
