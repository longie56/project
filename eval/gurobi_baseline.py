import math
import time

import torch

from config import (
    DEVICE,
    E_MAX,
    E_SAFE_MARGIN,
    GRID_SCALE,
    MAX_ROUTE_TIME_NORM,
    MAX_PAYLOAD_KG,
    MAX_TIME,
    PEN_WEIGHT,
    PHYS_WEIGHT,
    P_EMPTY_FAST,
    P_EMPTY_NORM,
    P_GAMMA,
    P_HOVER,
    V_FAST,
    V_NORM,
)
from env.vrp_env import VRPTWEnv

try:
    import gurobipy as gp
    from gurobipy import GRB
except Exception:  # pragma: no cover - runtime dependency
    gp = None
    GRB = None
def evaluate_actions_with_env(
    nodes,
    pref,
    depot,
    actions,
    visited_mask=None,
    curr_pos=None,
    curr_time=None,
    curr_load=None,
    curr_energy=None,
    scenario="full",
):
    device = nodes.device
    B, N, _ = nodes.shape
    if B != 1:
        raise ValueError("evaluate_actions_with_env only supports batch size 1.")

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

    env = VRPTWEnv(nodes.clone(), pref.clone(), depot.clone(), scenario=scenario)
    env.load_state(
        visited_mask.clone(),
        curr_pos.clone(),
        curr_time.clone(),
        curr_load.clone(),
        curr_energy.clone(),
    )

    executed = []
    invalid_action = None
    invalid_reason = ""
    for act in actions:
        act = int(act)
        if env._compute_done().all():
            break

        mask = env.get_mask()
        if act < 0 or act >= mask.size(1) or not bool(mask[0, act].item()):
            invalid_action = act
            invalid_reason = "action is infeasible under VRPTWEnv.get_mask()"
            break

        env.step(torch.tensor([act], device=device))
        executed.append(act)
        if env._compute_done().all():
            break

    unvisit = (~env.visited).sum(-1).float()
    total_cost = env.phys * PHYS_WEIGHT + env.pen * PEN_WEIGHT + unvisit * 20.0
    feasible = invalid_action is None and bool(
        (unvisit == 0).all().item() and env._at_depot().all().item() and env._compute_done().all().item()
    )
    if invalid_action is not None:
        total_cost = torch.full_like(total_cost, math.inf)
    return {
        "actions": executed,
        "cost": float(total_cost.item()),
        "phys": float(env.phys.item()),
        "pen": float(env.pen.item()),
        "unvisit": float(unvisit.item()),
        "feasible": feasible,
        "invalid_action": invalid_action,
        "invalid_reason": invalid_reason,
    }


class GurobiN20Solver:
    """
    Step-indexed MILP baseline for small/medium instances.
    Objective, physical dynamics, and action feasibility are replayed in VRPTWEnv
    after optimization for strict fairness.
    """

    def __init__(
        self,
        time_limit_s=60.0,
        mip_gap=0.01,
        max_returns=None,
        verbose=False,
        scenario="full",
        proof_mode=False,
        threads=0,
        seed=42,
        log_file="",
        nodefile_start_gb=0.5,
        nodefile_dir="",
    ):
        self.time_limit_s = float(time_limit_s)
        self.mip_gap = float(mip_gap)
        self.max_returns = max_returns
        self.verbose = verbose
        self.scenario = str(scenario).lower()
        self.proof_mode = bool(proof_mode)
        self.threads = int(threads)
        self.seed = int(seed)
        self.log_file = str(log_file or "")
        self.nodefile_start_gb = float(nodefile_start_gb)
        self.nodefile_dir = str(nodefile_dir or "")

    def _check_dependency(self):
        if gp is None:
            raise RuntimeError(
                "gurobipy is not available. Please install Gurobi and activate license first."
            )

    def solve_instance(
        self,
        nodes,
        pref,
        depot,
        warm_start_actions=None,
        visited_mask=None,
        curr_pos=None,
        curr_time=None,
        curr_load=None,
        curr_energy=None,
    ):
        self._check_dependency()
        t0 = time.perf_counter()

        # Force single-instance solve for reproducibility.
        if nodes.size(0) != 1:
            raise ValueError("GurobiN20Solver currently supports batch size = 1.")

        device = nodes.device
        N = nodes.size(1)
        if self.max_returns is None:
            max_returns = N
        else:
            max_returns = int(self.max_returns)
        if max_returns <= 0 and warm_start_actions:
            starts = warm_start_actions if isinstance(warm_start_actions[0], (list, tuple)) else [warm_start_actions]
            depot_actions = {N * 2, N * 2 + 1}
            # Use the largest return count among provided warm starts. Using the
            # minimum can make stronger starts impossible to embed in the
            # step-indexed sequence, leaving Gurobi without an incumbent.
            max_returns = max(
                1,
                min(
                    N,
                    max(sum(1 for a in actions if int(a) in depot_actions) for actions in starts if actions),
                ),
            )
        elif max_returns <= 0:
            max_returns = N
        # Positions are 0..L, so the model has exactly L transitions/actions.
        # A feasible tour with N customer visits and R depot returns needs N + R actions.
        L = N + max_returns

        if visited_mask is None:
            visited_mask = torch.zeros(1, N, dtype=torch.bool, device=device)
        if curr_pos is None:
            curr_pos = depot.clone()
        if curr_time is None:
            curr_time = torch.zeros(1, device=device)
        if curr_load is None:
            curr_load = torch.ones(1, device=device)
        if curr_energy is None:
            curr_energy = torch.ones(1, device=device)

        if torch.norm(curr_pos[0] - depot[0]).item() > 1e-5:
            raise ValueError(
                "GurobiN20Solver is a depot-rooted static MILP baseline: "
                "curr_pos must equal depot. Use the environment/ALNS repair "
                "interface for en-route emergency insertion states."
            )

        # Customers already visited are removed from the MILP service requirement.
        visited_np = visited_mask[0].detach().cpu().numpy()

        # Build augmented node list: 0=depot, 1..N=customers
        all_xy = torch.cat([depot.unsqueeze(1), nodes[:, :, :2]], dim=1)[0].detach().cpu()
        demand = torch.cat([torch.zeros(1), nodes[0, :, 2].detach().cpu()], dim=0)
        ready = torch.cat([torch.zeros(1), nodes[0, :, 3].detach().cpu()], dim=0)
        due = torch.cat([torch.zeros(1), nodes[0, :, 4].detach().cpu()], dim=0)
        service = torch.cat([torch.zeros(1), nodes[0, :, 5].detach().cpu()], dim=0)

        pref0 = float(pref[0, 0].detach().cpu().item())
        pref1 = float(pref[0, 1].detach().cpu().item())
        start_time = float(curr_time[0].detach().cpu().item())
        start_load = float(curr_load[0].detach().cpu().item())
        start_energy = float(curr_energy[0].detach().cpu().item())
        is_simple = self.scenario == "simple"

        V = list(range(N + 1))
        C = list(range(1, N + 1))
        P = list(range(L + 1))  # positions
        S = [0] if is_simple else [0, 1]  # 0 norm, 1 fast
        steps = list(range(L))  # transitions p -> p+1

        allowed_arcs = [(i, j) for i in V for j in V]

        t_norm = {}
        e0 = {}
        e1 = {}
        for i in V:
            for j in V:
                d = float(torch.norm(all_xy[j] - all_xy[i]).item() * GRID_SCALE)
                t_norm[(i, j, 0)] = d / V_NORM / MAX_TIME
                t_norm[(i, j, 1)] = d / V_FAST / MAX_TIME
                e0[(i, j, 0)] = (P_EMPTY_NORM * d / V_NORM) / E_MAX
                e0[(i, j, 1)] = (P_EMPTY_FAST * d / V_FAST) / E_MAX
                e1[(i, j, 0)] = (P_GAMMA * MAX_PAYLOAD_KG * d / V_NORM) / E_MAX
                e1[(i, j, 1)] = (P_GAMMA * MAX_PAYLOAD_KG * d / V_FAST) / E_MAX

        model = gp.Model("uav_vrptw_milp_baseline")
        model.Params.OutputFlag = 1 if (self.verbose or self.log_file) else 0
        if math.isfinite(self.time_limit_s) and self.time_limit_s > 0:
            model.Params.TimeLimit = self.time_limit_s
        model.Params.MIPGap = max(0.0, self.mip_gap)
        model.Params.Presolve = 2 if self.proof_mode else 1
        model.Params.Symmetry = 2
        model.Params.Seed = self.seed
        if self.threads > 0:
            model.Params.Threads = self.threads
        if self.log_file:
            model.Params.LogFile = self.log_file
        if self.nodefile_start_gb > 0:
            model.Params.NodefileStart = self.nodefile_start_gb
        if self.nodefile_dir:
            model.Params.NodefileDir = self.nodefile_dir
        if self.proof_mode:
            # Proof mode spends more effort improving the dual bound.
            model.Params.StartNodeLimit = max(500, N * 50)
            model.Params.MIPFocus = 3
            model.Params.Heuristics = 0.1
            model.Params.Cuts = 3
        else:
            # Incumbent mode is better for large n when optimality proof is out of reach.
            model.Params.StartNodeLimit = max(20000, N * 400)
            model.Params.MIPFocus = 1
            model.Params.Heuristics = 0.8
            model.Params.Cuts = 1
            if self.time_limit_s >= 30:
                model.Params.NoRelHeurTime = min(300.0, max(5.0, self.time_limit_s * 0.2))
                model.Params.ImproveStartTime = min(600.0, max(60.0, self.time_limit_s * 0.5))
            elif self.time_limit_s <= 0:
                model.Params.NoRelHeurTime = 300.0
                model.Params.ImproveStartTime = 600.0

        y = model.addVars(P, V, vtype=GRB.BINARY, name="y")
        z = model.addVars(steps, allowed_arcs, S, vtype=GRB.BINARY, name="z")
        time_ub = MAX_ROUTE_TIME_NORM + 2.0
        t = model.addVars(P, lb=0.0, ub=time_ub, vtype=GRB.CONTINUOUS, name="t")
        arr = model.addVars(steps, lb=0.0, ub=time_ub, vtype=GRB.CONTINUOUS, name="arr")
        s_start = model.addVars(P, lb=0.0, ub=time_ub, vtype=GRB.CONTINUOUS, name="s")
        wait = model.addVars(P, lb=0.0, ub=time_ub, vtype=GRB.CONTINUOUS, name="wait")
        tardy = model.addVars(P, lb=0.0, ub=time_ub, vtype=GRB.CONTINUOUS, name="tardy")
        load = model.addVars(P, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="load")
        f_time = model.addVars(steps, lb=0.0, ub=time_ub, vtype=GRB.CONTINUOUS, name="f_time")
        q = None if is_simple else model.addVars(steps, allowed_arcs, S, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="q")
        energy = None if is_simple else model.addVars(P, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="energy")
        f_energy = None if is_simple else model.addVars(steps, lb=0.0, ub=2.0, vtype=GRB.CONTINUOUS, name="f_energy")
        required_customers = [j for j in C if not visited_np[j - 1]]
        required_count = len(required_customers)
        all_served = model.addVars(P, vtype=GRB.BINARY, name="all_served")
        hover_energy_norm = (P_HOVER * MAX_TIME) / E_MAX

        # Tight Big-M values are important for proving optimality. A single
        # loose M=10 is feasible but makes the LP relaxation very weak.
        M_time = time_ub
        # Return-to-depot resets load to 1.0. Inactive customer load-transfer
        # constraints must therefore allow load[p + 1] = 1 even when load[p]
        # is low and the inactive customer demand is high.
        M_load = 2.0
        # Energy deactivation needs to cover inactive customer arcs that include
        # a worst-case early-arrival hover reserve. M=1 can wrongly reject valid
        # MIP starts while still inactive in those constraints.
        M_energy = max(2.0, 1.0 + hover_energy_norm * time_ub)

        # Initial state
        model.addConstr(y[0, 0] == 1)
        for j in C:
            model.addConstr(y[0, j] == 0)
        model.addConstr(t[0] == start_time)
        model.addConstr(load[0] == start_load)
        if not is_simple:
            model.addConstr(energy[0] == start_energy)
        model.addConstr(y[L, 0] == 1)  # must end at depot

        # One node per position
        for p in P:
            model.addConstr(gp.quicksum(y[p, i] for i in V) == 1)

        # Visit each unvisited customer exactly once; visited customers never visited.
        for j in C:
            if visited_np[j - 1]:
                model.addConstr(gp.quicksum(y[p, j] for p in P) == 0)
            else:
                model.addConstr(gp.quicksum(y[p, j] for p in P) == 1)

        # Match VRPTWEnv timeout semantics: while there are still unserved
        # customers, the route time may not exceed MAX_ROUTE_TIME_NORM. Once all
        # required customers are served, the final return-to-depot can finish
        # slightly after that horizon, just as env._compute_done() allows.
        for p in P:
            if required_count == 0:
                model.addConstr(all_served[p] == 1)
            else:
                served_expr = gp.quicksum(y[q_idx, j] for q_idx in range(p + 1) for j in required_customers)
                model.addConstr(served_expr >= required_count * all_served[p])
                model.addConstr(served_expr <= (required_count - 1) + all_served[p])
            model.addConstr(t[p] <= MAX_ROUTE_TIME_NORM + (time_ub - MAX_ROUTE_TIME_NORM) * all_served[p])

        # Flow linking between y and z
        for p in steps:
            for i in V:
                model.addConstr(gp.quicksum(z[p, i, j, s] for j in V if (i, j) in allowed_arcs for s in S) == y[p, i])
            for j in V:
                model.addConstr(gp.quicksum(z[p, i, j, s] for i in V if (i, j) in allowed_arcs for s in S) == y[p + 1, j])

        # Remove useless symmetry. Customer self-loops cannot be part of a valid visit
        # sequence, and fast depot self-loops are equivalent to normal depot self-loops.
        for p in steps:
            for i in C:
                for s in S:
                    if (i, i) in allowed_arcs:
                        model.addConstr(z[p, i, i, s] == 0)
            if not is_simple and (0, 0) in allowed_arcs:
                model.addConstr(z[p, 0, 0, 1] == 0)

        # Consecutive depot self-loops are only useful as tail padding after all work is done.
        # If the sequence idles at depot for one step, force all remaining positions to stay depot.
        for p in range(max(0, L - 1)):
            model.addConstr(y[p, 0] + y[p + 1, 0] - 1 <= y[p + 2, 0])

        if not is_simple:
            # Linearization q = load[p] * z[p,i,j,s]
            for p in steps:
                for i, j in allowed_arcs:
                    for s in S:
                        model.addConstr(q[p, i, j, s] <= load[p])
                        model.addConstr(q[p, i, j, s] <= z[p, i, j, s])
                        model.addConstr(q[p, i, j, s] >= load[p] - (1 - z[p, i, j, s]))
                        model.addConstr(q[p, i, j, s] >= 0.0)

        # Flight time / flight energy definitions
        for p in steps:
            model.addConstr(
                f_time[p]
                == gp.quicksum(t_norm[(i, j, s)] * z[p, i, j, s] for (i, j) in allowed_arcs for s in S)
            )
            if not is_simple:
                model.addConstr(
                    f_energy[p]
                    == gp.quicksum(
                        e0[(i, j, s)] * z[p, i, j, s] + e1[(i, j, s)] * q[p, i, j, s]
                        for (i, j) in allowed_arcs
                        for s in S
                    )
                )
            model.addConstr(arr[p] == t[p] + f_time[p])

        # Time, load, energy dynamics
        for p in steps:
            customer_flag = gp.quicksum(y[p + 1, j] for j in C)

            # service start and waiting
            model.addConstr(s_start[p + 1] >= arr[p])
            model.addConstr(wait[p + 1] >= s_start[p + 1] - arr[p])
            model.addConstr(wait[p + 1] <= M_time * customer_flag)
            model.addConstr(tardy[p + 1] <= M_time * customer_flag)

            # Depot arrival dynamics
            model.addConstr(t[p + 1] <= arr[p] + M_time * (1 - y[p + 1, 0]))
            model.addConstr(t[p + 1] >= arr[p] - M_time * (1 - y[p + 1, 0]))
            model.addConstr(load[p + 1] <= 1.0 + M_load * (1 - y[p + 1, 0]))
            model.addConstr(load[p + 1] >= 1.0 - M_load * (1 - y[p + 1, 0]))
            if not is_simple:
                model.addConstr(energy[p + 1] <= 1.0 + M_energy * (1 - y[p + 1, 0]))
                model.addConstr(energy[p + 1] >= 1.0 - M_energy * (1 - y[p + 1, 0]))

            if not is_simple:
                # Customer arrival dynamics
                model.addConstr(
                    energy[p + 1] <= energy[p] - f_energy[p] - wait[p + 1] * hover_energy_norm + M_energy * y[p + 1, 0]
                )
                model.addConstr(
                    energy[p + 1] >= energy[p] - f_energy[p] - wait[p + 1] * hover_energy_norm - M_energy * y[p + 1, 0]
                )

            for j in C:
                dj = float(demand[j].item())
                rj = float(ready[j].item())
                duj = float(due[j].item())
                svj = float(service[j].item())

                model.addConstr(s_start[p + 1] >= rj - M_time * (1 - y[p + 1, j]))
                model.addConstr(tardy[p + 1] >= s_start[p + 1] - duj - M_time * (1 - y[p + 1, j]))

                model.addConstr(t[p + 1] <= s_start[p + 1] + svj + M_time * (1 - y[p + 1, j]))
                model.addConstr(t[p + 1] >= s_start[p + 1] + svj - M_time * (1 - y[p + 1, j]))
                model.addConstr(load[p + 1] <= load[p] - dj + M_load * (1 - y[p + 1, j]))
                model.addConstr(load[p + 1] >= load[p] - dj - M_load * (1 - y[p + 1, j]))
                model.addConstr(load[p] >= dj - M_load * (1 - y[p + 1, j]))

            if not is_simple:
                # Match env action feasibility: serving a customer must pay flight
                # energy, early-arrival hover energy, and still keep enough
                # battery for an economical return to depot.
                for i in V:
                    for j in C:
                        if (i, j) not in allowed_arcs:
                            continue
                        dj = float(demand[j].item())
                        for s in S:
                            # Align with env.check_energy_feasible: return leg always uses economical speed (V_NORM, idx=0).
                            required_const = e0[(i, j, s)] + e0[(j, 0, 0)] - e1[(j, 0, 0)] * dj + E_SAFE_MARGIN
                            required_coeff = e1[(i, j, s)] + e1[(j, 0, 0)]
                            model.addConstr(
                                energy[p]
                                >= required_const
                                + required_coeff * load[p]
                                + wait[p + 1] * hover_energy_norm
                                - M_energy * (1 - z[p, i, j, s])
                            )

                # Match env._return_mask(): a direct return must be feasible
                # under the actual selected speed. Energy is reset only after
                # this feasible return action reaches the depot.
                for i in C:
                    for s in S:
                        required_return = e0[(i, 0, s)] + e1[(i, 0, s)] * load[p] + E_SAFE_MARGIN
                        model.addConstr(energy[p] >= required_return - M_energy * (1 - z[p, i, 0, s]))

                model.addConstr(energy[p + 1] >= E_SAFE_MARGIN)

        # Objective: same structure as environment (then re-evaluate in env)
        if is_simple:
            phys_expr = gp.quicksum(5.0 * f_time[p] + wait[p + 1] for p in steps)
        else:
            phys_expr = gp.quicksum(
                pref1 * 10.0 * (f_energy[p] + hover_energy_norm * wait[p + 1]) + pref0 * 5.0 * f_time[p]
                for p in steps
            )
        pen_expr = gp.quicksum(tardy[p] for p in P)
        total_expr = PHYS_WEIGHT * phys_expr + PEN_WEIGHT * pen_expr
        model.setObjective(total_expr, GRB.MINIMIZE)

        # Optional MIP starts from DRL/ALNS/OR-Tools routes. These do not alter
        # constraints/objective; they only give Gurobi strong incumbents early.
        if warm_start_actions is not None and len(warm_start_actions) > 0:
            try:
                if warm_start_actions and isinstance(warm_start_actions[0], (list, tuple)):
                    warm_starts = warm_start_actions
                else:
                    warm_starts = [warm_start_actions]

                model.NumStart = len(warm_starts)
                for start_idx, action_seq in enumerate(warm_starts):
                    if not action_seq:
                        continue
                    model.Params.StartNumber = start_idx
                    for p in P:
                        for i in V:
                            y[p, i].Start = 0.0
                        t[p].Start = 0.0
                        s_start[p].Start = 0.0
                        wait[p].Start = 0.0
                        tardy[p].Start = 0.0
                        load[p].Start = 1.0
                        all_served[p].Start = 1.0 if required_count == 0 else 0.0
                        if not is_simple:
                            energy[p].Start = 1.0
                    for p in steps:
                        arr[p].Start = 0.0
                        f_time[p].Start = 0.0
                        if not is_simple:
                            f_energy[p].Start = 0.0
                        for i, j in allowed_arcs:
                            for s in S:
                                z[p, i, j, s].Start = 0.0
                                if not is_simple:
                                    q[p, i, j, s].Start = 0.0

                    curr_node = 0
                    curr_t = start_time
                    curr_load_val = start_load
                    curr_energy_val = start_energy
                    served_required = set()
                    y[0, curr_node].Start = 1.0
                    t[0].Start = curr_t
                    s_start[0].Start = curr_t
                    wait[0].Start = 0.0
                    tardy[0].Start = 0.0
                    load[0].Start = curr_load_val
                    if not is_simple:
                        energy[0].Start = curr_energy_val
                    clean_actions = [int(action) for action in action_seq][:L]
                    padded_actions = clean_actions + [N * 2] * max(0, L - len(clean_actions))
                    for p, action in enumerate(padded_actions[:L]):
                        if action < N * 2:
                            nxt_node = action // 2 + 1
                        else:
                            nxt_node = 0
                        speed_idx = 0 if is_simple else action % 2
                        if (curr_node, nxt_node) in allowed_arcs and speed_idx in S:
                            z[p, curr_node, nxt_node, speed_idx].Start = 1.0
                            y[p + 1, nxt_node].Start = 1.0

                            flight_time = t_norm[(curr_node, nxt_node, speed_idx)]
                            arrival_time = curr_t + flight_time
                            arr[p].Start = arrival_time
                            f_time[p].Start = flight_time

                            if not is_simple:
                                q[p, curr_node, nxt_node, speed_idx].Start = curr_load_val
                                flight_energy = (
                                    e0[(curr_node, nxt_node, speed_idx)]
                                    + e1[(curr_node, nxt_node, speed_idx)] * curr_load_val
                                )
                                f_energy[p].Start = flight_energy
                            else:
                                flight_energy = 0.0

                            if nxt_node == 0:
                                service_start = arrival_time
                                wait_time = 0.0
                                tardy_val = 0.0
                                next_t = arrival_time
                                next_load = 1.0
                                next_energy = 1.0
                            else:
                                rj = float(ready[nxt_node].item())
                                duj = float(due[nxt_node].item())
                                svj = float(service[nxt_node].item())
                                dj = float(demand[nxt_node].item())
                                service_start = max(arrival_time, rj)
                                wait_time = max(0.0, service_start - arrival_time)
                                tardy_val = max(0.0, service_start - duj)
                                next_t = service_start + svj
                                next_load = max(0.0, curr_load_val - dj)
                                if not is_simple:
                                    next_energy = curr_energy_val - flight_energy - wait_time * hover_energy_norm
                                else:
                                    next_energy = curr_energy_val
                                if nxt_node in required_customers:
                                    served_required.add(nxt_node)

                            s_start[p + 1].Start = service_start
                            wait[p + 1].Start = wait_time
                            tardy[p + 1].Start = tardy_val
                            t[p + 1].Start = next_t
                            load[p + 1].Start = next_load
                            all_served[p + 1].Start = (
                                1.0 if required_count == 0 or len(served_required) >= required_count else 0.0
                            )
                            if not is_simple:
                                energy[p + 1].Start = next_energy

                            curr_node = nxt_node
                            curr_t = next_t
                            curr_load_val = next_load
                            curr_energy_val = next_energy
                        else:
                            break
            except Exception:
                # Keep solver robust even if MIP-start parsing fails.
                pass

        model.optimize()
        elapsed = time.perf_counter() - t0

        status = int(model.Status)
        accepted_status = {GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL, GRB.INTERRUPTED}
        if status not in accepted_status or model.SolCount <= 0:
            return {
                "status": status,
                "elapsed_s": elapsed,
                "actions": [],
                "mip_obj": math.inf,
                "obj_bound": math.inf,
                "mip_gap": math.inf,
                "sol_count": int(model.SolCount),
                "max_returns_used": int(max_returns),
                "env_eval": {
                    "actions": [],
                    "cost": math.inf,
                    "phys": math.inf,
                    "pen": math.inf,
                    "unvisit": math.inf,
                    "feasible": False,
                    "invalid_action": None,
                    "invalid_reason": "no incumbent solution",
                },
            }

        # Decode action sequence from selected arcs
        actions = []
        for p in steps:
            selected = None
            for i, j in allowed_arcs:
                for s in S:
                    if z[p, i, j, s].X > 0.5:
                        selected = (i, j, s)
                        break
                if selected is not None:
                    break
            if selected is None:
                continue

            _, j, s = selected
            if j == 0:
                act = N * 2 + s
            else:
                act = (j - 1) * 2 + s
            actions.append(int(act))

        env_eval = evaluate_actions_with_env(
            nodes=nodes,
            pref=pref,
            depot=depot,
            actions=actions,
            visited_mask=visited_mask,
            curr_pos=curr_pos,
            curr_time=curr_time,
            curr_load=curr_load,
            curr_energy=curr_energy,
            scenario=self.scenario,
        )

        return {
            "status": status,
            "elapsed_s": elapsed,
            "actions": actions,
            "mip_obj": float(model.ObjVal) if model.SolCount > 0 else math.inf,
            "obj_bound": float(model.ObjBound) if model.SolCount > 0 else math.inf,
            "mip_gap": float(model.MIPGap) if model.SolCount > 0 else math.inf,
            "sol_count": int(model.SolCount),
            "max_returns_used": int(max_returns),
            "env_eval": env_eval,
        }


# Backward-compatible alias with a less misleading name. The same step-indexed
# MILP can be run on N50, but large instances usually produce incumbents rather
# than proven optima within practical time limits.
GurobiVRPTWSolver = GurobiN20Solver
