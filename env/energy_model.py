# env/energy_model.py
import torch

from config import *


def _to_tensor(value, ref_tensor):
    if torch.is_tensor(value):
        return value.to(device=ref_tensor.device, dtype=ref_tensor.dtype)
    return torch.full_like(ref_tensor, float(value))


def compute_energy(pos1, pos2, speed, load):
    dist = torch.norm(pos2 - pos1, dim=-1) * GRID_SCALE
    speed_t = _to_tensor(speed, dist)
    load_t = _to_tensor(load, dist)
    p_empty = torch.where(
        torch.isclose(speed_t, torch.full_like(speed_t, V_FAST)),
        torch.full_like(dist, P_EMPTY_FAST),
        torch.full_like(dist, P_EMPTY_NORM),
    )
    t_flight = dist / speed_t
    power = p_empty + P_GAMMA * load_t * MAX_PAYLOAD_KG
    return power * t_flight


def compute_flight_time_norm(pos1, pos2, speed):
    dist = torch.norm(pos2 - pos1, dim=-1) * GRID_SCALE
    speed_t = _to_tensor(speed, dist)
    return (dist / speed_t) / MAX_TIME


def compute_hover_energy(wait_time):
    return P_HOVER * wait_time * MAX_TIME


def check_direct_return_feasible(energy_curr, pos, depot, speed, load):
    e_back = compute_energy(pos, depot, speed, load)
    return (energy_curr * E_MAX - e_back) / E_MAX >= E_SAFE_MARGIN


def check_energy_feasible(
    energy_curr,
    pos,
    target,
    depot,
    speed,
    load,
    return_load=None,
    curr_time=None,
    ready_time=None,
):
    if return_load is None:
        return_load = load
    e_to = compute_energy(pos, target, speed, load)
    wait_e = torch.zeros_like(e_to)
    if curr_time is not None and ready_time is not None:
        curr_time_t = _to_tensor(curr_time, e_to)
        ready_time_t = _to_tensor(ready_time, e_to)
        while curr_time_t.dim() < e_to.dim():
            curr_time_t = curr_time_t.unsqueeze(-1)
        while ready_time_t.dim() < e_to.dim():
            ready_time_t = ready_time_t.unsqueeze(-1)
        arrival = curr_time_t + compute_flight_time_norm(pos, target, speed)
        wait_e = compute_hover_energy(torch.relu(ready_time_t - arrival))
    e_back = compute_energy(target, depot, V_NORM, return_load)
    return (energy_curr * E_MAX - (e_to + wait_e + e_back)) / E_MAX >= E_SAFE_MARGIN
