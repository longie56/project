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
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2
except Exception:  # pragma: no cover - runtime dependency
    pywrapcp = None
    routing_enums_pb2 = None


class ORToolsBaselineSolver:
    """
    Uses OR-Tools to build customer routes under simplified capacity/time-window constraints,
    then replays the resulting customer priority in VRPTWEnv for fair scoring.
    """

    def __init__(
        self,
        time_limit_s=5.0,
        num_vehicles=None,
        vehicle_candidates=None,
        first_solution_candidates=None,
        first_solution_strategy="PATH_CHEAPEST_ARC",
        local_search_metaheuristic="GUIDED_LOCAL_SEARCH",
        waiting_slack=2.0,
        route_time_limit=3.0,
        scenario="full",
        share_candidate_time=False,
    ):
        self.time_limit_s = float(time_limit_s)
        self.num_vehicles = num_vehicles
        self.vehicle_candidates = vehicle_candidates
        self.first_solution_candidates = first_solution_candidates
        self.first_solution_strategy = str(first_solution_strategy)
        self.local_search_metaheuristic = str(local_search_metaheuristic)
        self.waiting_slack = float(waiting_slack)
        self.route_time_limit = float(route_time_limit)
        self.scenario = str(scenario).lower()
        self.share_candidate_time = bool(share_candidate_time)

    def _check_dependency(self):
        if pywrapcp is None or routing_enums_pb2 is None:
            raise RuntimeError("OR-Tools is not available. Please install `ortools` in the project venv.")

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
        vehicle_candidates = self.vehicle_candidates
        first_solution_candidates = self.first_solution_candidates or [self.first_solution_strategy]
        if vehicle_candidates or len(first_solution_candidates) > 1:
            best = None
            vehicles = vehicle_candidates or [self.num_vehicles]
            combo_count = max(1, len(vehicles) * len(first_solution_candidates))
            per_candidate_time = self.time_limit_s
            if self.share_candidate_time and combo_count > 1:
                per_candidate_time = max(1.0, self.time_limit_s / combo_count)
            for num_vehicles in vehicles:
                for first_solution in first_solution_candidates:
                    result = self._solve_instance_once(
                        nodes=nodes,
                        pref=pref,
                        depot=depot,
                        visited_mask=visited_mask,
                        curr_pos=curr_pos,
                        curr_time=curr_time,
                        curr_load=curr_load,
                        curr_energy=curr_energy,
                        forced_num_vehicles=None if num_vehicles is None else int(num_vehicles),
                        first_solution_strategy=first_solution,
                        time_limit_s=per_candidate_time,
                    )
                    if best is None:
                        best = result
                        continue
                    best_key = (not best["feasible"], best["unvisit"], best["cost"])
                    result_key = (not result["feasible"], result["unvisit"], result["cost"])
                    if result_key < best_key:
                        best = result
            best["elapsed_s"] = time.perf_counter() - t0
            best["per_candidate_time_limit_s"] = per_candidate_time
            if vehicle_candidates:
                best["vehicle_candidates"] = [int(x) for x in vehicle_candidates]
            best["first_solution_candidates"] = [str(x) for x in first_solution_candidates]
            return best

        result = self._solve_instance_once(
            nodes=nodes,
            pref=pref,
            depot=depot,
            visited_mask=visited_mask,
            curr_pos=curr_pos,
            curr_time=curr_time,
            curr_load=curr_load,
            curr_energy=curr_energy,
            forced_num_vehicles=self.num_vehicles,
            first_solution_strategy=self.first_solution_strategy,
        )
        result["elapsed_s"] = time.perf_counter() - t0
        return result

    def _solve_instance_once(
        self,
        nodes,
        pref,
        depot,
        visited_mask=None,
        curr_pos=None,
        curr_time=None,
        curr_load=None,
        curr_energy=None,
        forced_num_vehicles=None,
        first_solution_strategy=None,
        time_limit_s=None,
    ):
        batch_size, n_nodes, _ = nodes.shape
        if batch_size != 1:
            raise ValueError("ORToolsBaselineSolver only supports batch size 1.")

        if visited_mask is None:
            visited_mask = torch.zeros(batch_size, n_nodes, dtype=torch.bool, device=nodes.device)
        if curr_load is None:
            curr_load = torch.ones(batch_size, device=nodes.device)

        active_customers = [idx for idx in range(n_nodes) if not bool(visited_mask[0, idx].item())]
        if not active_customers:
            result = priority_to_actions(
                nodes, pref, depot, [], visited_mask, curr_pos, curr_time, curr_load, curr_energy, scenario=self.scenario
            )
            result["priority"] = []
            result["route_buckets"] = []
            return result

        index_to_customer = {pos + 1: customer for pos, customer in enumerate(active_customers)}
        customer_to_index = {customer: pos + 1 for pos, customer in enumerate(active_customers)}
        problem_size = len(active_customers)
        num_vehicles = forced_num_vehicles or min(problem_size, max(2, int(math.ceil(problem_size / 3))))
        num_vehicles = max(1, min(problem_size, int(num_vehicles)))

        manager = pywrapcp.RoutingIndexManager(problem_size + 1, num_vehicles, 0)
        routing = pywrapcp.RoutingModel(manager)

        xy = [depot[0].detach().cpu()]
        xy.extend(nodes[0, active_customers, :2].detach().cpu())
        demands = [0.0]
        demands.extend(nodes[0, active_customers, 2].detach().cpu().tolist())
        ready = [0.0]
        ready.extend(nodes[0, active_customers, 3].detach().cpu().tolist())
        due = [0.0]
        due.extend(nodes[0, active_customers, 4].detach().cpu().tolist())
        service = [0.0]
        service.extend(nodes[0, active_customers, 5].detach().cpu().tolist())

        def distance_cb(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            dist = torch.norm(xy[from_node] - xy[to_node]).item() * GRID_SCALE
            return int(round(dist))

        transit_idx = routing.RegisterTransitCallback(distance_cb)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_idx)

        def demand_cb(from_index):
            from_node = manager.IndexToNode(from_index)
            return int(round(demands[from_node] * 1000))

        demand_idx = routing.RegisterUnaryTransitCallback(demand_cb)
        routing.AddDimensionWithVehicleCapacity(
            demand_idx,
            0,
            [int(round(float(curr_load[0].item()) * 1000)) for _ in range(num_vehicles)],
            True,
            "Capacity",
        )

        def time_cb(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            dist = torch.norm(xy[from_node] - xy[to_node]).item() * GRID_SCALE
            travel = dist / V_NORM / MAX_TIME
            return int(round((travel + service[from_node]) * 100000))

        time_idx = routing.RegisterTransitCallback(time_cb)
        routing.AddDimension(
            time_idx,
            int(round(self.waiting_slack * 100000)),
            int(round(self.route_time_limit * 100000)),
            False,
            "Time",
        )
        time_dim = routing.GetDimensionOrDie("Time")

        for customer in active_customers:
            idx = customer_to_index[customer]
            routing_index = manager.NodeToIndex(idx)
            time_dim.CumulVar(routing_index).SetRange(
                int(round(ready[idx] * 100000)),
                int(round(max(ready[idx], due[idx]) * 100000)),
            )

        search = pywrapcp.DefaultRoutingSearchParameters()
        first_solution = getattr(
            routing_enums_pb2.FirstSolutionStrategy,
            str(first_solution_strategy or self.first_solution_strategy),
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC,
        )
        local_search = getattr(
            routing_enums_pb2.LocalSearchMetaheuristic,
            self.local_search_metaheuristic,
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH,
        )
        search.first_solution_strategy = first_solution
        search.local_search_metaheuristic = local_search
        solve_time_limit = self.time_limit_s if time_limit_s is None else float(time_limit_s)
        search.time_limit.seconds = max(1, int(math.ceil(solve_time_limit)))

        solution = routing.SolveWithParameters(search)
        route_buckets = []
        if solution is not None:
            for vehicle_id in range(num_vehicles):
                index = routing.Start(vehicle_id)
                route = []
                while not routing.IsEnd(index):
                    node_idx = manager.IndexToNode(index)
                    if node_idx != 0:
                        route.append(index_to_customer[node_idx])
                    index = solution.Value(routing.NextVar(index))
                if route:
                    route_buckets.append(route)

        if not route_buckets:
            route_buckets = [[idx] for idx in sorted(active_customers, key=lambda c: float(nodes[0, c, 4].item()))]

        candidate_buckets = [route_buckets]
        sorted_buckets = sort_route_buckets_by_time(nodes, route_buckets)
        if sorted_buckets != route_buckets:
            candidate_buckets.append(sorted_buckets)

        result = None
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

        for env_eval in eval_candidates:
            if not env_eval["feasible"]:
                repaired_eval = append_missing_actions_to_eval(
                    nodes=nodes,
                    pref=pref,
                    depot=depot,
                    actions=env_eval["actions"],
                    visited_mask=visited_mask,
                    curr_pos=curr_pos,
                    curr_time=curr_time,
                    curr_load=curr_load,
                    curr_energy=curr_energy,
                    scenario=self.scenario,
                    mode_policy=env_eval.get("mode_policy", "greedy"),
                )
                repaired_eval["priority"] = env_eval.get("priority")
                repaired_eval["route_buckets"] = env_eval.get("route_buckets", route_buckets)
                repaired_eval["mode_policy"] = env_eval.get("mode_policy", "greedy")
                repaired_eval["repair_applied"] = True
                env_eval = repaired_eval
            else:
                env_eval["repair_applied"] = False

            if result is None:
                result = env_eval
                best_buckets = env_eval.get("route_buckets", route_buckets)
                best_priority = env_eval.get("priority")
                continue
            current_key = (not result["feasible"], result["unvisit"], result["cost"])
            eval_key = (not env_eval["feasible"], env_eval["unvisit"], env_eval["cost"])
            if eval_key < current_key:
                result = env_eval
                best_buckets = env_eval.get("route_buckets", route_buckets)
                best_priority = env_eval.get("priority")

        route_buckets = best_buckets
        priority = best_priority if best_priority is not None else route_buckets_to_priority(nodes, route_buckets)
        result["priority"] = priority
        result["route_buckets"] = route_buckets
        result["num_vehicles"] = num_vehicles
        result["first_solution_strategy"] = str(first_solution_strategy or self.first_solution_strategy)
        result["solver_status"] = "success" if solution is not None else "fallback"
        return result
