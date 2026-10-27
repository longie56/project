"""Gu et al. six-rotor physical kernel for the next TRE revision scenario.

All public aircraft values live in the v3 JSON specification. This module
implements its no-wind, steady, level-flight branch. The separate revision_v3
package uses this kernel with actual onboard cargo. The legacy route
environment, generator and trainer remain a historical v1 proxy.

Source: https://doi.org/10.1016/j.cja.2025.103605, Eqs. (12)-(17), Table 5.
The source curve is modeled rather than measured flight telemetry.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch


SPEC_PATH = (
    Path(__file__).resolve().parents[1]
    / "docs"
    / "pretrain_20261012"
    / "revision_v3_gu_hexacopter_physics.json"
)
SPEC = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
SOURCE = SPEC["source_parameters"]
CHOICES = SPEC["declared_study_choices"]

RHO = SOURCE["air_density_kg_m3"]
G = SOURCE["gravity_m_s2"]
EMPTY_MASS_KG = SOURCE["airframe_mass_kg"] + SOURCE["battery_mass_kg"]
MAX_PAYLOAD_KG = SOURCE["maximum_payload_kg"]
ROTOR_DISK_AREA_M2 = SOURCE["rotor_count"] * SOURCE["single_rotor_disk_area_m2"]
BASE_CDA_M2 = (
    SOURCE["airframe_drag_coefficient"] * SOURCE["airframe_projected_area_m2"]
    + SOURCE["battery_drag_coefficient"] * SOURCE["battery_projected_area_m2"]
)
FULL_CARGO_CDA_M2 = (
    SOURCE["payload_drag_coefficient"] * SOURCE["full_payload_projected_area_m2"]
)
SPEEDS_M_S = tuple(CHOICES["cruise_speed_modes_m_s"])
EFFICIENCY = CHOICES["nominal_battery_to_propeller_efficiency"]
BATTERY_NOMINAL_J = SOURCE["battery_mass_kg"] * SOURCE["battery_specific_energy_j_kg"]
BATTERY_MISSION_BUDGET_J = (
    BATTERY_NOMINAL_J * SOURCE["depth_of_discharge"] / SOURCE["safety_factor"]
)


def _tensor(value, reference=None):
    if reference is None:
        return torch.as_tensor(value, dtype=torch.float64)
    return torch.as_tensor(value, dtype=reference.dtype, device=reference.device)


def _validate_payload(payload):
    if bool(torch.any(~torch.isfinite(payload))) or bool(torch.any(payload < -1e-6)) or bool(
        torch.any(payload > MAX_PAYLOAD_KG + 1e-6)
    ):
        raise ValueError(f"Onboard cargo must be in [0, {MAX_PAYLOAD_KG}] kg")


def effective_drag_area_m2(payload_kg, *, payload_area_mode="linear",
                           drag_multiplier=1.0, _assume_valid=False):
    """Declared intermediate-load geometry; the source gives endpoints only."""
    payload = _tensor(payload_kg)
    if not _assume_valid:
        _validate_payload(payload)
    payload = payload.clamp(0, MAX_PAYLOAD_KG)
    if payload_area_mode == "linear":
        cargo_fraction = payload / MAX_PAYLOAD_KG
    elif payload_area_mode == "fixed_container":
        cargo_fraction = torch.ones_like(payload)
    else:
        raise ValueError("Unknown cargo frontal-area rule")
    if drag_multiplier <= 0:
        raise ValueError("Drag multiplier must be positive")
    return drag_multiplier * (BASE_CDA_M2 + FULL_CARGO_CDA_M2 * cargo_fraction)


def steady_power_w(speed_m_s, payload_kg, *, efficiency=EFFICIENCY,
                   payload_area_mode="linear", drag_multiplier=1.0,
                   rotor_area_multiplier=1.0, model="gu", _assume_valid=False):
    """Battery-side power from Gu et al. Eqs. (12)-(17).

    The implicit induced-inflow numerator is v_h² = mg/(2 rho A), using
    hover thrust T_h=mg. In forward flight the outer power uses total
    thrust T=sqrt((mg)²+D²). A bounded Newton solve selects the unique
    nonnegative root; hover inflow is its upper bound for v>=0.
    """
    payload = _tensor(payload_kg)
    speed = _tensor(speed_m_s, payload)
    payload, speed = torch.broadcast_tensors(payload, speed)
    if not _assume_valid:
        _validate_payload(payload)
    payload = payload.clamp(0, MAX_PAYLOAD_KG)
    if not _assume_valid:
        if bool(torch.any(~torch.isfinite(speed))) or bool(torch.any(speed < 0)) or bool(
            torch.any(speed > SOURCE["maximum_airspeed_m_s"])
        ):
            raise ValueError("Airspeed outside the source model envelope")
    if efficiency not in SOURCE["battery_to_propeller_efficiency_options"]:
        raise ValueError("Use a source-reported conversion efficiency")
    if rotor_area_multiplier <= 0:
        raise ValueError("Rotor-area multiplier must be positive")
    if model == "hover_increment_drag":
        # A structurally different, source-matched alternative: empty-aircraft
        # Gu cruise, nonlinear mass^(3/2) hover increment, and cargo parasite
        # power. This is a sensitivity model, not another measured platform.
        empty = torch.zeros_like(payload)
        kwargs = dict(efficiency=efficiency, payload_area_mode="linear",
                      drag_multiplier=drag_multiplier,
                      rotor_area_multiplier=rotor_area_multiplier, model="gu",
                      _assume_valid=_assume_valid)
        empty_cruise = steady_power_w(speed, empty, **kwargs)
        hover_delta = (steady_power_w(0.0, payload, **kwargs)
                       - steady_power_w(0.0, empty, **kwargs))
        cargo_cda = FULL_CARGO_CDA_M2 * payload / MAX_PAYLOAD_KG
        cargo_parasite = 0.5 * RHO * drag_multiplier * cargo_cda * speed.pow(3) / efficiency
        return empty_cruise + hover_delta + cargo_parasite
    if model != "gu":
        raise ValueError("Unknown aerodynamic model")

    weight = (EMPTY_MASS_KG + payload) * G
    drag = 0.5 * RHO * effective_drag_area_m2(
        payload, payload_area_mode=payload_area_mode,
        drag_multiplier=drag_multiplier, _assume_valid=_assume_valid) * speed.square()
    thrust = torch.hypot(weight, drag)
    pitch = torch.atan2(drag, weight)
    sin_pitch, cos_pitch = torch.sin(pitch), torch.cos(pitch)
    hover_vi_squared = weight / (2 * RHO * ROTOR_DISK_AREA_M2 * rotor_area_multiplier)
    induced_velocity = torch.sqrt(hover_vi_squared)
    horizontal = speed * cos_pitch
    vertical = speed * sin_pitch
    # Five Newton steps reach double-precision accuracy throughout the
    # registered 0-20 m/s, 0-2.4 kg envelope (including sensitivity bounds).
    # The former 12 steps repeated the already-converged root in every action
    # mask and dominated short GPU rollouts.
    for _ in range(5):
        norm = torch.hypot(horizontal, vertical + induced_velocity)
        residual = induced_velocity * norm - hover_vi_squared
        derivative = norm + induced_velocity * (vertical + induced_velocity) / norm
        induced_velocity = torch.clamp_min(induced_velocity - residual/derivative, 0)
    return thrust * (speed * sin_pitch + induced_velocity) / efficiency


def cruise_energy_j(distance_m, speed_m_s, payload_kg, *, efficiency=EFFICIENCY,
                    payload_area_mode="linear", drag_multiplier=1.0,
                    rotor_area_multiplier=1.0, model="gu", _assume_valid=False):
    distance = _tensor(distance_m)
    speed = _tensor(speed_m_s, distance)
    payload = _tensor(payload_kg, distance)
    if not _assume_valid:
        if bool(torch.any(~torch.isfinite(distance))) or bool(torch.any(distance < 0)):
            raise ValueError("Flight distance must be finite and nonnegative")
        if bool(torch.any(speed <= 0)):
            raise ValueError("Cruise speed must be positive")
    return steady_power_w(speed, payload, efficiency=efficiency,
                          payload_area_mode=payload_area_mode,
                          drag_multiplier=drag_multiplier,
                          rotor_area_multiplier=rotor_area_multiplier,
                          model=model, _assume_valid=_assume_valid) * distance / speed


def hover_energy_j(seconds, payload_kg, *, efficiency=EFFICIENCY,
                   payload_area_mode="linear", drag_multiplier=1.0,
                   rotor_area_multiplier=1.0, model="gu", _assume_valid=False):
    duration = _tensor(seconds)
    payload = _tensor(payload_kg, duration)
    if not _assume_valid:
        if bool(torch.any(~torch.isfinite(duration))) or bool(torch.any(duration < 0)):
            raise ValueError("Hover duration must be finite and nonnegative")
    return steady_power_w(0.0, payload, efficiency=efficiency,
                          payload_area_mode=payload_area_mode,
                          drag_multiplier=drag_multiplier,
                          rotor_area_multiplier=rotor_area_multiplier,
                          model=model, _assume_valid=_assume_valid) * duration


@dataclass(frozen=True)
class EnergyScenario:
    """One physical replay law shared by masks, policy, and search baselines."""

    id: str = "gu_nominal"
    model: str = "gu"
    efficiency: float = EFFICIENCY
    payload_area_mode: str = "linear"
    drag_multiplier: float = 1.0
    rotor_area_multiplier: float = 1.0
    battery_budget_multiplier: float = 1.0

    @property
    def battery_budget_j(self):
        if self.battery_budget_multiplier <= 0:
            raise ValueError("Battery-budget multiplier must be positive")
        return BATTERY_MISSION_BUDGET_J * self.battery_budget_multiplier

    def power_w(self, speed_m_s, payload_kg):
        return steady_power_w(speed_m_s, payload_kg, efficiency=self.efficiency,
                              payload_area_mode=self.payload_area_mode,
                              drag_multiplier=self.drag_multiplier,
                              rotor_area_multiplier=self.rotor_area_multiplier,
                              model=self.model)

    def cruise_j(self, distance_m, speed_m_s, payload_kg, *, _assume_valid=False):
        return cruise_energy_j(distance_m, speed_m_s, payload_kg,
                               efficiency=self.efficiency,
                               payload_area_mode=self.payload_area_mode,
                               drag_multiplier=self.drag_multiplier,
                               rotor_area_multiplier=self.rotor_area_multiplier,
                               model=self.model, _assume_valid=_assume_valid)

    def hover_j(self, seconds, payload_kg, *, _assume_valid=False):
        return hover_energy_j(seconds, payload_kg, efficiency=self.efficiency,
                              payload_area_mode=self.payload_area_mode,
                              drag_multiplier=self.drag_multiplier,
                              rotor_area_multiplier=self.rotor_area_multiplier,
                               model=self.model, _assume_valid=_assume_valid)


SCENARIOS = {
    "gu_nominal": EnergyScenario(),
    "gu_eta050": EnergyScenario("gu_eta050", efficiency=0.5),
    "gu_fixed_container": EnergyScenario("gu_fixed_container",
                                         payload_area_mode="fixed_container"),
    "gu_drag090": EnergyScenario("gu_drag090", drag_multiplier=0.9),
    "gu_drag110": EnergyScenario("gu_drag110", drag_multiplier=1.1),
    "gu_rotor090": EnergyScenario("gu_rotor090", rotor_area_multiplier=0.9),
    "gu_rotor110": EnergyScenario("gu_rotor110", rotor_area_multiplier=1.1),
    "gu_battery080": EnergyScenario("gu_battery080", battery_budget_multiplier=0.8),
    "gu_battery120": EnergyScenario("gu_battery120", battery_budget_multiplier=1.2),
    "hover_increment_drag": EnergyScenario("hover_increment_drag",
                                            model="hover_increment_drag"),
}


def allowed_speed_m_s(speed_m_s):
    speed = _tensor(speed_m_s)
    return (speed == SPEEDS_M_S[0]) | (speed == SPEEDS_M_S[1])
