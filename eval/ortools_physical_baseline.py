import math
import time

import torch

from config import (
    E_MAX,
    GRID_SCALE,
    MAX_PAYLOAD_KG,
    MAX_TIME,
    P_EMPTY_FAST,
    P_EMPTY_NORM,
    P_GAMMA,
    PEN_WEIGHT,
    PHYS_WEIGHT,
    V_FAST,
    V_NORM,
)
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


class ORToolsPhysicalBaselineSolver:
    """
    OR-Tools baseline with a physics-aware routing objective.

    The RoutingModel cannot exactly represent the project's full environment:
    a single UAV performs serial depot returns, energy depends on the remaining
    load, and speed is a per-action mode decision. This baseline therefore uses
    OR-Tools for a stronger physical-cost route-order search, then replays the
    chosen order in VRPTWEnv for the final score.
    """

    TIME_SCALE = 100000
    COST_SCALE = 100000

    def __init__(
        self,
        time_limit_s=5.0,
        num_vehicles=None,
        vehicle_candidates=None,
        first_solution_candidates=None,
        local_search_candidates=None,
        first_solution_strategy="PATH_CHEAPEST_ARC",
        local_search_metaheuristic="GUIDED_LOCAL_SEARCH",
        waiting_slack=2.0,
        route_time_limit=3.0,
        scenario="full",
        routing_mode="serial_order",
        share_candidate_time=False,
    ):
        self.time_limit_s = float(time_limit_s)
        self.num_vehicles = num_vehicles
        self.vehicle_candidates = vehicle_candidates
        self.first_solution_candidates = first_solution_candidates
        self.local_search_candidates = local_search_candidates
        self.first_solution_strategy = str(first_solution_strategy)
        self.local_search_metaheuristic = str(local_search_metaheuristic)
        self.waiting_slack = float(waiting_slack)
        self.route_time_limit = float(route_time_limit)
        self.scenario = str(scenario).lower()
        self.routing_mode = str(routing_mode).lower()
        if self.routing_mode not in {"serial_order", "parallel_routes"}:
            raise ValueError(f"Unsupported OR-Tools physical routing_mode: {routing_mode}")
        self.share_candidate_time = bool(share_candidate_time)

    def _check_dependency(self):
        if pywrapcp is None or routing_enums_pb2 is None:
            raise RuntimeError("OR-Tools is not available. Please install `ortools` in the project venv.")

    def _annotate_proxy_result(self, result):
        result["exact_physical_model"] = False
        result["final_scored_by_env"] = True
        result["modeling_note"] = (
            "OR-Tools searches a physics-aware proxy route/order; VRPTWEnv replay "
            "performs the final speed, return, energy, and time-window scoring."
        )
        return result

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
        local_search_candidates = self.local_search_candidates or [self.local_search_metaheuristic]
        if vehicle_candidates or len(first_solution_candidates) > 1 or len(local_search_candidates) > 1:
            best = None
            vehicles = vehicle_candidates or [self.num_vehicles]
            combo_count = max(1, len(vehicles) * len(first_solution_candidates) * len(local_search_candidates))
            per_candidate_time = self.time_limit_s
            if self.share_candidate_time and combo_count > 1:
                per_candidate_time = max(1.0, self.time_limit_s / combo_count)
            for num_vehicles in vehicles:
                for first_solution in first_solution_candidates:
                    for local_search in local_search_candidates:
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
                            local_search_metaheuristic=local_search,
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
            best["local_search_candidates"] = [str(x) for x in local_search_candidates]
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
            local_search_metaheuristic=self.local_search_metaheuristic,
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
        local_search_metaheuristic=None,
        time_limit_s=None,
    ):
        batch_size, n_nodes, _ = nodes.shape
        if batch_size != 1:
            raise ValueError("ORToolsPhysicalBaselineSolver only supports batch size 1.")

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
            return self._annotate_proxy_result(result)

        if self.routing_mode == "serial_order":
            return self._solve_serial_order(
                nodes=nodes,
                pref=pref,
                depot=depot,
                active_customers=active_customers,
                visited_mask=visited_mask,
                curr_pos=curr_pos,
                curr_time=curr_time,
                curr_load=curr_load,
                curr_energy=curr_energy,
                first_solution_strategy=first_solution_strategy,
                local_search_metaheuristic=local_search_metaheuristic,
                time_limit_s=time_limit_s,
            )

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
        due = [self.route_time_limit]
        due.extend(nodes[0, active_customers, 4].detach().cpu().tolist())
        service = [0.0]
        service.extend(nodes[0, active_customers, 5].detach().cpu().tolist())
        pref_time = float(pref[0, 0].detach().cpu().item())
        pref_energy = float(pref[0, 1].detach().cpu().item())
        load_proxy = float(curr_load[0].detach().cpu().item())

        def _dist(from_node, to_node):
            return torch.norm(xy[from_node] - xy[to_node]).item() * GRID_SCALE

        def _step_phys_cost(from_node, to_node, speed):
            dist = _dist(from_node, to_node)
            t_flight = dist / speed
            t_norm = t_flight / MAX_TIME
            p_empty = P_EMPTY_FAST if abs(speed - V_FAST) < 1e-9 else P_EMPTY_NORM
            energy = (p_empty + P_GAMMA * load_proxy * MAX_PAYLOAD_KG) * t_flight
            phys = (energy / E_MAX) * 10.0 * pref_energy + t_norm * 5.0 * pref_time
            return PHYS_WEIGHT * phys

        def physical_cost_cb(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            normal_cost = _step_phys_cost(from_node, to_node, V_NORM)
            fast_cost = _step_phys_cost(from_node, to_node, V_FAST)
            return int(round(min(normal_cost, fast_cost) * self.COST_SCALE))

        cost_idx = routing.RegisterTransitCallback(physical_cost_cb)
        routing.SetArcCostEvaluatorOfAllVehicles(cost_idx)

        def demand_cb(from_index):
            from_node = manager.IndexToNode(from_index)
            return int(round(demands[from_node] * 1000))

        demand_idx = routing.RegisterUnaryTransitCallback(demand_cb)
        routing.AddDimensionWithVehicleCapacity(
            demand_idx,
            0,
            [int(round(load_proxy * 1000)) for _ in range(num_vehicles)],
            True,
            "Capacity",
        )

        def time_cb(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            # Speed is a real action in the environment. Use the faster physical
            # travel lower bound in OR-Tools and let VRPTWEnv choose the replay speed.
            travel = _dist(from_node, to_node) / V_FAST / MAX_TIME
            return int(round((travel + service[from_node]) * self.TIME_SCALE))

        time_idx = routing.RegisterTransitCallback(time_cb)
        routing.AddDimension(
            time_idx,
            int(round(self.waiting_slack * self.TIME_SCALE)),
            int(round(self.route_time_limit * self.TIME_SCALE)),
            False,
            "Time",
        )
        time_dim = routing.GetDimensionOrDie("Time")

        for customer in active_customers:
            idx = customer_to_index[customer]
            routing_index = manager.NodeToIndex(idx)
            time_dim.CumulVar(routing_index).SetMin(int(round(ready[idx] * self.TIME_SCALE)))
            soft_due = int(round(max(ready[idx], due[idx]) * self.TIME_SCALE))
            time_dim.SetCumulVarSoftUpperBound(
                routing_index,
                soft_due,
                max(1, int(round(PEN_WEIGHT * self.COST_SCALE / self.TIME_SCALE))),
            )

        search = pywrapcp.DefaultRoutingSearchParameters()
        first_solution = getattr(
            routing_enums_pb2.FirstSolutionStrategy,
            str(first_solution_strategy or self.first_solution_strategy),
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC,
        )
        local_search = getattr(
            routing_enums_pb2.LocalSearchMetaheuristic,
            str(local_search_metaheuristic or self.local_search_metaheuristic),
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

        result = self._evaluate_route_bucket_candidates(
            nodes=nodes,
            pref=pref,
            depot=depot,
            route_buckets=route_buckets,
            visited_mask=visited_mask,
            curr_pos=curr_pos,
            curr_time=curr_time,
            curr_load=curr_load,
            curr_energy=curr_energy,
            active_customers=active_customers,
        )
        result["num_vehicles"] = num_vehicles
        result["first_solution_strategy"] = str(first_solution_strategy or self.first_solution_strategy)
        result["local_search_metaheuristic"] = str(local_search_metaheuristic or self.local_search_metaheuristic)
        result["solver_status"] = "success" if solution is not None else "fallback"
        result["objective_type"] = "physical_arc_cost_soft_tw"
        result["routing_mode"] = self.routing_mode
        return self._annotate_proxy_result(result)

    def _solve_serial_order(
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
        first_solution_strategy=None,
        local_search_metaheuristic=None,
        time_limit_s=None,
    ):
        index_to_customer = {pos + 1: customer for pos, customer in enumerate(active_customers)}
        customer_to_index = {customer: pos + 1 for pos, customer in enumerate(active_customers)}
        problem_size = len(active_customers)

        manager = pywrapcp.RoutingIndexManager(problem_size + 1, 1, 0)
        routing = pywrapcp.RoutingModel(manager)

        xy = [depot[0].detach().cpu()]
        xy.extend(nodes[0, active_customers, :2].detach().cpu())
        ready = [0.0]
        ready.extend(nodes[0, active_customers, 3].detach().cpu().tolist())
        due = [self.route_time_limit]
        due.extend(nodes[0, active_customers, 4].detach().cpu().tolist())
        service = [0.0]
        service.extend(nodes[0, active_customers, 5].detach().cpu().tolist())
        pref_time = float(pref[0, 0].detach().cpu().item())
        pref_energy = float(pref[0, 1].detach().cpu().item())
        load_proxy = float(curr_load[0].detach().cpu().item())

        def _dist(from_node, to_node):
            return torch.norm(xy[from_node] - xy[to_node]).item() * GRID_SCALE

        def _step_phys_cost(from_node, to_node, speed):
            dist = _dist(from_node, to_node)
            t_flight = dist / speed
            t_norm = t_flight / MAX_TIME
            p_empty = P_EMPTY_FAST if abs(speed - V_FAST) < 1e-9 else P_EMPTY_NORM
            energy = (p_empty + P_GAMMA * load_proxy * MAX_PAYLOAD_KG) * t_flight
            phys = (energy / E_MAX) * 10.0 * pref_energy + t_norm * 5.0 * pref_time
            return PHYS_WEIGHT * phys

        def physical_cost_cb(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            normal_cost = _step_phys_cost(from_node, to_node, V_NORM)
            fast_cost = _step_phys_cost(from_node, to_node, V_FAST)
            return int(round(min(normal_cost, fast_cost) * self.COST_SCALE))

        cost_idx = routing.RegisterTransitCallback(physical_cost_cb)
        routing.SetArcCostEvaluatorOfAllVehicles(cost_idx)

        def time_cb(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            # This is a lower-bound schedule for one serial UAV. The final replay
            # performs depot returns and speed selection in the real environment.
            travel = _dist(from_node, to_node) / V_FAST / MAX_TIME
            return int(round((travel + service[from_node]) * self.TIME_SCALE))

        time_idx = routing.RegisterTransitCallback(time_cb)
        routing.AddDimension(
            time_idx,
            int(round(self.waiting_slack * self.TIME_SCALE)),
            int(round(self.route_time_limit * self.TIME_SCALE)),
            False,
            "Time",
        )
        time_dim = routing.GetDimensionOrDie("Time")

        for customer in active_customers:
            idx = customer_to_index[customer]
            routing_index = manager.NodeToIndex(idx)
            time_dim.CumulVar(routing_index).SetMin(int(round(ready[idx] * self.TIME_SCALE)))
            soft_due = int(round(max(ready[idx], due[idx]) * self.TIME_SCALE))
            time_dim.SetCumulVarSoftUpperBound(
                routing_index,
                soft_due,
                max(1, int(round(PEN_WEIGHT * self.COST_SCALE / self.TIME_SCALE))),
            )

        search = pywrapcp.DefaultRoutingSearchParameters()
        first_solution = getattr(
            routing_enums_pb2.FirstSolutionStrategy,
            str(first_solution_strategy or self.first_solution_strategy),
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC,
        )
        local_search = getattr(
            routing_enums_pb2.LocalSearchMetaheuristic,
            str(local_search_metaheuristic or self.local_search_metaheuristic),
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH,
        )
        search.first_solution_strategy = first_solution
        search.local_search_metaheuristic = local_search
        solve_time_limit = self.time_limit_s if time_limit_s is None else float(time_limit_s)
        search.time_limit.seconds = max(1, int(math.ceil(solve_time_limit)))

        solution = routing.SolveWithParameters(search)
        priority = []
        if solution is not None:
            index = routing.Start(0)
            while not routing.IsEnd(index):
                node_idx = manager.IndexToNode(index)
                if node_idx != 0:
                    priority.append(index_to_customer[node_idx])
                index = solution.Value(routing.NextVar(index))

        if not priority:
            priority = sorted(active_customers, key=lambda c: float(nodes[0, c, 4].item()))

        route_buckets = [priority]
        result = self._evaluate_route_bucket_candidates(
            nodes=nodes,
            pref=pref,
            depot=depot,
            route_buckets=route_buckets,
            visited_mask=visited_mask,
            curr_pos=curr_pos,
            curr_time=curr_time,
            curr_load=curr_load,
            curr_energy=curr_energy,
            active_customers=active_customers,
        )
        # In serial_order mode OR-Tools only searches one global customer
        # priority chain. Depot returns are inserted later by VRPTWEnv replay,
        # so reporting this as one vehicle/route would be misleading.
        result["num_vehicles"] = None
        result["num_priority_chains"] = 1
        result["first_solution_strategy"] = str(first_solution_strategy or self.first_solution_strategy)
        result["local_search_metaheuristic"] = str(local_search_metaheuristic or self.local_search_metaheuristic)
        result["solver_status"] = "success" if solution is not None else "fallback"
        result["objective_type"] = "serial_physical_priority_soft_tw"
        result["routing_mode"] = self.routing_mode
        return self._annotate_proxy_result(result)

    def _evaluate_route_bucket_candidates(
        self,
        nodes,
        pref,
        depot,
        route_buckets,
        visited_mask,
        curr_pos,
        curr_time,
        curr_load,
        curr_energy,
        active_customers,
    ):
        candidate_buckets = [route_buckets]
        sorted_buckets = sort_route_buckets_by_time(nodes, route_buckets)
        if sorted_buckets != route_buckets:
            candidate_buckets.append(sorted_buckets)

        eval_candidates = []
        for buckets in candidate_buckets:
            priority = route_buckets_to_priority(nodes, buckets)
            for mode_policy in ("greedy", "fast", "normal"):
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
                route_eval["priority"] = priority
                route_eval["route_buckets"] = buckets
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
        for mode_policy in ("greedy", "fast", "normal"):
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
                mode_policy=mode_policy,
            )
            due_eval["priority"] = due_priority
            due_eval["route_buckets"] = [[idx] for idx in due_priority]
            due_eval["mode_policy"] = mode_policy
            eval_candidates.append(due_eval)

        result = None
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
                continue
            current_key = (not result["feasible"], result["unvisit"], result["cost"])
            eval_key = (not env_eval["feasible"], env_eval["unvisit"], env_eval["cost"])
            if eval_key < current_key:
                result = env_eval

        return result
