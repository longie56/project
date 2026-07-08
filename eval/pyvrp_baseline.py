import math
import time

import torch

from config import GRID_SCALE, MAX_TIME, V_NORM
from eval.route_baseline_utils import (
    append_missing_actions_to_eval,
    customer_routes_to_eval,
    priority_to_actions,
    route_buckets_to_priority,
    sort_route_buckets_by_time,
)

try:
    from pyvrp import Model
    from pyvrp.stop import MaxRuntime
except Exception:  # pragma: no cover - runtime dependency
    Model = None
    MaxRuntime = None


class PyVRPBaselineSolver:
    """
    PyVRP baseline under a unified environment re-evaluation protocol.

    PyVRP solves a multi-vehicle capacitated VRPTW approximation using a norm-speed
    duration matrix. The resulting client routes are then replayed in VRPTWEnv via
    the shared route-order evaluator for a fair comparison to the learning methods.
    """

    def __init__(
        self,
        time_limit_s=10.0,
        num_vehicles=None,
        vehicle_candidates=None,
        route_time_limit=3.0,
        scenario="full",
        share_candidate_time=False,
    ):
        self.time_limit_s = float(time_limit_s)
        self.num_vehicles = num_vehicles
        self.vehicle_candidates = vehicle_candidates
        self.route_time_limit = float(route_time_limit)
        self.scenario = str(scenario).lower()
        self.share_candidate_time = bool(share_candidate_time)

    def _check_dependency(self):
        if Model is None or MaxRuntime is None:
            raise RuntimeError("PyVRP is not available. Please install `pyvrp` in the project venv.")

    def _annotate_proxy_result(self, result):
        result["exact_physical_model"] = False
        result["final_scored_by_env"] = True
        result["modeling_note"] = (
            "PyVRP solves a capacitated VRPTW proxy with norm-speed durations; "
            "VRPTWEnv replay performs the final speed, return, energy, and "
            "time-window scoring."
        )
        return result

    def _solve_for_vehicle_count(
        self,
        nodes,
        pref,
        depot,
        active_customers,
        visited_mask,
        curr_pos,
        curr_time,
        curr_load,
        curr_energy,
        num_vehicles,
        time_limit_s=None,
    ):
        problem_size = len(active_customers)
        model = Model()
        depot_xy = depot[0].detach().cpu()
        depot_node = model.add_depot(
            x=float(depot_xy[0].item()),
            y=float(depot_xy[1].item()),
            tw_early=0,
            tw_late=int(round(self.route_time_limit * 100000)),
        )
        model.add_vehicle_type(
            num_available=num_vehicles,
            capacity=int(round(float(curr_load[0].item()) * 1000)),
            start_depot=depot_node,
            end_depot=depot_node,
            tw_early=0,
            tw_late=int(round(self.route_time_limit * 100000)),
        )

        locations = [depot_node]
        index_to_customer = {}
        for pos, customer_idx in enumerate(active_customers, start=1):
            node = nodes[0, customer_idx].detach().cpu()
            client = model.add_client(
                x=float(node[0].item()),
                y=float(node[1].item()),
                delivery=int(round(float(node[2].item()) * 1000)),
                service_duration=int(round(float(node[5].item()) * 100000)),
                tw_early=int(round(float(node[3].item()) * 100000)),
                tw_late=int(round(max(float(node[3].item()), float(node[4].item())) * 100000)),
            )
            locations.append(client)
            index_to_customer[pos] = customer_idx

        for frm_idx, frm in enumerate(locations):
            frm_xy = depot_xy if frm_idx == 0 else nodes[0, index_to_customer[frm_idx], :2].detach().cpu()
            frm_service = 0.0 if frm_idx == 0 else float(nodes[0, index_to_customer[frm_idx], 5].item())
            for to_idx, to in enumerate(locations):
                if frm_idx == to_idx:
                    model.add_edge(frm, to, distance=0, duration=0)
                    continue
                to_xy = depot_xy if to_idx == 0 else nodes[0, index_to_customer[to_idx], :2].detach().cpu()
                dist = torch.norm(frm_xy - to_xy).item() * GRID_SCALE
                duration = dist / V_NORM / MAX_TIME
                model.add_edge(
                    frm,
                    to,
                    distance=int(round(dist)),
                    duration=int(round((duration + frm_service) * 100000)),
                )

        solve_time_limit = self.time_limit_s if time_limit_s is None else float(time_limit_s)
        result = model.solve(stop=MaxRuntime(max(1.0, solve_time_limit)), seed=42, display=False)
        best = result.best
        route_buckets = []
        if best is not None:
            for route in best.routes():
                visits = list(route.visits())
                if visits:
                    route_buckets.append([index_to_customer[int(idx)] for idx in visits])

        if not route_buckets:
            route_buckets = [[idx] for idx in sorted(active_customers, key=lambda c: float(nodes[0, c, 4].item()))]
            solver_status = "fallback"
        else:
            solver_status = "success"

        candidate_buckets = [route_buckets]
        sorted_buckets = sort_route_buckets_by_time(nodes, route_buckets)
        if sorted_buckets != route_buckets:
            candidate_buckets.append(sorted_buckets)

        env_eval = None
        best_buckets = route_buckets
        best_priority = None
        eval_candidates = []
        for buckets in candidate_buckets:
            priority = route_buckets_to_priority(nodes, buckets)
            for mode_policy in ("greedy", "fast"):
                route_eval = customer_routes_to_eval(
                    nodes=nodes,
                    pref=pref,
                    depot=depot,
                    route_buckets=buckets,
                    visited_mask=visited_mask,
                    curr_pos=curr_pos,
                    curr_time=curr_time,
                    curr_load=curr_load,
                    curr_energy=curr_energy,
                    scenario=self.scenario,
                    mode_policy=mode_policy,
                )
                route_eval["mode_policy"] = mode_policy
                eval_candidates.append(route_eval)

                priority_eval = priority_to_actions(
                    nodes=nodes,
                    pref=pref,
                    depot=depot,
                    priority=priority,
                    visited_mask=visited_mask,
                    curr_pos=curr_pos,
                    curr_time=curr_time,
                    curr_load=curr_load,
                    curr_energy=curr_energy,
                    scenario=self.scenario,
                    mode_policy=mode_policy,
                )
                priority_eval["priority"] = priority
                priority_eval["route_buckets"] = buckets
                priority_eval["mode_policy"] = mode_policy
                eval_candidates.append(priority_eval)

        due_priority = sorted(active_customers, key=lambda c: float(nodes[0, c, 4].item()))
        due_eval = priority_to_actions(
            nodes=nodes,
            pref=pref,
            depot=depot,
            priority=due_priority,
            visited_mask=visited_mask,
            curr_pos=curr_pos,
            curr_time=curr_time,
            curr_load=curr_load,
            curr_energy=curr_energy,
            scenario=self.scenario,
        )
        due_eval["priority"] = due_priority
        due_eval["route_buckets"] = [[idx] for idx in due_priority]
        eval_candidates.append(due_eval)
        due_fast_eval = priority_to_actions(
            nodes=nodes,
            pref=pref,
            depot=depot,
            priority=due_priority,
            visited_mask=visited_mask,
            curr_pos=curr_pos,
            curr_time=curr_time,
            curr_load=curr_load,
            curr_energy=curr_energy,
            scenario=self.scenario,
            mode_policy="fast",
        )
        due_fast_eval["priority"] = due_priority
        due_fast_eval["route_buckets"] = [[idx] for idx in due_priority]
        due_fast_eval["mode_policy"] = "fast"
        eval_candidates.append(due_fast_eval)

        for candidate_eval in eval_candidates:
            if not candidate_eval["feasible"]:
                repaired_eval = append_missing_actions_to_eval(
                    nodes=nodes,
                    pref=pref,
                    depot=depot,
                    actions=candidate_eval["actions"],
                    visited_mask=visited_mask,
                    curr_pos=curr_pos,
                    curr_time=curr_time,
                    curr_load=curr_load,
                    curr_energy=curr_energy,
                    scenario=self.scenario,
                    mode_policy=candidate_eval.get("mode_policy", "greedy"),
                )
                repaired_eval["priority"] = candidate_eval.get("priority")
                repaired_eval["route_buckets"] = candidate_eval.get("route_buckets", route_buckets)
                repaired_eval["mode_policy"] = candidate_eval.get("mode_policy", "greedy")
                repaired_eval["repair_applied"] = True
                candidate_eval = repaired_eval
            else:
                candidate_eval["repair_applied"] = False

            if env_eval is None:
                env_eval = candidate_eval
                best_buckets = candidate_eval.get("route_buckets", route_buckets)
                best_priority = candidate_eval.get("priority")
                continue
            current_key = (not env_eval["feasible"], env_eval["unvisit"], env_eval["cost"])
            candidate_key = (not candidate_eval["feasible"], candidate_eval["unvisit"], candidate_eval["cost"])
            if candidate_key < current_key:
                env_eval = candidate_eval
                best_buckets = candidate_eval.get("route_buckets", route_buckets)
                best_priority = candidate_eval.get("priority")

        route_buckets = best_buckets
        priority = best_priority if best_priority is not None else route_buckets_to_priority(nodes, route_buckets)
        env_eval["priority"] = priority
        env_eval["route_buckets"] = route_buckets
        env_eval["solver_status"] = solver_status
        env_eval["num_vehicles"] = num_vehicles
        return self._annotate_proxy_result(env_eval)

    def solve_instance(
        self,
        nodes,
        pref,
        depot,
        visited_mask=None,
        curr_pos=None,
        curr_time=None,
        curr_load=None,
        curr_energy=None,
    ):
        self._check_dependency()
        t0 = time.perf_counter()
        batch_size, n_nodes, _ = nodes.shape
        if batch_size != 1:
            raise ValueError("PyVRPBaselineSolver only supports batch size 1.")

        if visited_mask is None:
            visited_mask = torch.zeros(batch_size, n_nodes, dtype=torch.bool, device=nodes.device)
        if curr_load is None:
            curr_load = torch.ones(batch_size, device=nodes.device)

        active_customers = [idx for idx in range(n_nodes) if not bool(visited_mask[0, idx].item())]
        if not active_customers:
            result = priority_to_actions(
                nodes, pref, depot, [], visited_mask, curr_pos, curr_time, curr_load, curr_energy, scenario=self.scenario
            )
            result["elapsed_s"] = time.perf_counter() - t0
            result["priority"] = []
            result["route_buckets"] = []
            result["solver_status"] = "empty"
            return self._annotate_proxy_result(result)

        problem_size = len(active_customers)
        default_vehicle = min(problem_size, max(2, int(math.ceil(problem_size / 3))))
        if self.vehicle_candidates:
            candidates = []
            for cand in self.vehicle_candidates:
                cand = int(cand)
                if 1 <= cand <= problem_size and cand not in candidates:
                    candidates.append(cand)
        else:
            num_vehicles = self.num_vehicles or default_vehicle
            candidates = [int(num_vehicles)]

        per_candidate_time = self.time_limit_s
        if self.share_candidate_time and len(candidates) > 1:
            per_candidate_time = max(1.0, self.time_limit_s / len(candidates))

        best_eval = None
        for cand in candidates:
            env_eval = self._solve_for_vehicle_count(
                nodes,
                pref,
                depot,
                active_customers,
                visited_mask,
                curr_pos,
                curr_time,
                curr_load,
                curr_energy,
                cand,
                time_limit_s=per_candidate_time,
            )
            # Across vehicle counts, prioritize true environment feasibility first.
            # A lower raw cost with unvisited customers is not a stronger baseline.
            if best_eval is None:
                best_eval = env_eval
                continue
            best_key = (not best_eval["feasible"], best_eval["unvisit"], best_eval["cost"])
            eval_key = (not env_eval["feasible"], env_eval["unvisit"], env_eval["cost"])
            if eval_key < best_key:
                best_eval = env_eval

        env_eval = best_eval
        env_eval["elapsed_s"] = time.perf_counter() - t0
        env_eval["per_candidate_time_limit_s"] = per_candidate_time
        if self.vehicle_candidates:
            env_eval["vehicle_candidates"] = [int(x) for x in candidates]
        env_eval["share_candidate_time"] = self.share_candidate_time
        return self._annotate_proxy_result(env_eval)
