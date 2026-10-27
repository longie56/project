"""Single source for declared task choices and the published aircraft kernel."""

from __future__ import annotations

import json
from pathlib import Path

from env import revision_v3_gu_physics as aircraft

ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = Path(__file__).with_name("task_spec.json")
TASK = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
SIDE_M = float(TASK["coordinate_side_m"])
SERVICE_S = float(TASK["service_seconds"])
DEPOT_TURN_S = float(TASK["depot_turnaround_seconds_before_each_sortie"])
Q_MAX_KG = float(aircraft.MAX_PAYLOAD_KG)
BUDGET_J = float(aircraft.BATTERY_MISSION_BUDGET_J)
SPEEDS_M_S = tuple(float(v) for v in aircraft.SPEEDS_M_S)
RESERVE_FRACTION = float(TASK["main_extra_battery_reserve_fraction"])


def validate_formal_experiment(*, variant, n_nodes, pref_mode, fixed_alpha,
                               objective_variant, energy_scenario):
    """Prevent the compact revision from silently expanding its run matrix."""
    if TASK.get("revision_scope") != "compact_no_film_no_ega_20261017":
        return
    identity = dict(variant=variant, n_nodes=n_nodes, pref_mode=pref_mode,
                    fixed_alpha=fixed_alpha if pref_mode == "fixed" else None,
                    objective_variant=objective_variant, energy_scenario=energy_scenario)
    if identity not in TASK.get("formal_experiment_matrix", []):
        raise ValueError("Experiment is outside the approved compact 36-run matrix")


def horizon_s(n_nodes: int) -> float:
    try:
        return float(TASK["mission_horizon_seconds_by_n"][str(n_nodes)])
    except KeyError as exc:
        raise ValueError(f"Unregistered v3 customer count: {n_nodes}") from exc


def objective(time_s, energy_j, tardiness_s, pref, *, variant: str = "C0"):
    scale = TASK["objective_scale"]
    flight = time_s / scale["flight_time_seconds_divisor"]
    energy = energy_j / scale["energy_joules_divisor"]
    lateness = tardiness_s / scale["tardiness_seconds_divisor"]
    if variant == "C0":
        return pref[..., 0]*flight + pref[..., 1]*energy + lateness
    if variant == "C1":
        return pref[..., 0]*(flight+lateness) + pref[..., 1]*energy
    raise ValueError("Objective variant must be C0 or C1")
