"""Independent physical greedy/ALNS and disclosed solver-proxy adapters.

Every method produces an explicit loading, flight and return action stream.
Only ``replay_actions`` assigns the final feasibility and v3 objective.
OR-Tools, PyVRP and Gurobi search route-order proxies; their native objective
and status are retained separately and must never be described as exact v3.
"""

from __future__ import annotations

import math
import random
import time

import torch

from env.revision_v3_gu_physics import EnergyScenario, SCENARIOS
from revision_v3.common_eval import replay_actions
from revision_v3.environment import RouteEnvironment
from revision_v3.spec import BUDGET_J, Q_MAX_KG, SIDE_M, SPEEDS_M_S


def _one(nodes, depot, pref):
    if nodes.ndim == 3:
        nodes = nodes[0]
    if depot.ndim == 2:
        depot = depot[0]
    if pref.ndim == 2:
        pref = pref[0]
    return nodes.detach().cpu(), depot.detach().cpu(), pref.detach().cpu()


def _key(report):
    m = report["metrics"]
    return (not report["feasible"], int(m["unserved"]), float(m["cost"]))


def _anneal_energy(report):
    """Scalar acceptance score respecting the same feasibility priority as _key."""
    m = report["metrics"]
    return (1000.0*float(not report["feasible"])
            + 100.0*float(m["unserved"]) + float(m["cost"]))


def _ordered_unique(order, n):
    result = [int(x) for x in order]
    if len(result) != n or set(result) != set(range(n)):
        raise ValueError("Solver order must contain each customer exactly once")
    return result


def _choose_flight(env: RouteEnvironment, priority, speed_labels=None):
    mask = env.raw_mask()[0]
    ranks = {customer: pos for pos, customer in enumerate(priority)}
    options = [env._customer_options(speed) for speed in SPEEDS_M_S]
    alpha, beta = [float(x) for x in env.pref[0]]
    candidates = []
    for i in range(env.N):
        if not bool(env.loaded[0, i]):
            continue
        for mode, speed in enumerate(SPEEDS_M_S):
            action = 2*i + mode
            if not bool(mask[action]):
                continue
            _, travel, _, energy, hover, finish = options[mode]
            tardiness = max(0.0, float(
                finish[0, i] - env.nodes[0, i, 5] - env.nodes[0, i, 4]
            ))
            if env.objective_variant == "C1":
                increment = (alpha*(float(travel[0, i])+tardiness)/3600
                             + beta*float(energy[0, i]+hover[0, i])/BUDGET_J)
            else:
                increment = (alpha*float(travel[0, i])/3600
                             + beta*float(energy[0, i]+hover[0, i])/BUDGET_J
                             + tardiness/3600)
            preferred_mode = None if speed_labels is None else speed_labels.get(i)
            mode_penalty = 0 if preferred_mode is None or preferred_mode == mode else 1
            candidates.append((ranks[i], mode_penalty, increment, action))
    return min(candidates)[3] if candidates else None


def construct_actions(nodes, depot, pref, order, *, max_manifest=None,
                      reserve_fraction=0.0, objective_variant="C0",
                      energy_scenario: EnergyScenario | None = None,
                      speed_labels=None, return_mode_preference=None,
                      wait_schedule=None, return_diagnostics=False):
    """Feasible-action constructive decoder with actual parcel warehouse stock.

    ``order`` is the route-order prior. It can come from a learned policy,
    physical ALNS, OR-Tools, PyVRP or Gurobi. The decoder always uses the same
    hard action masks and explicitly returns undelivered cargo to stock.
    """
    nodes, depot, pref = _one(nodes, depot, pref)
    order = _ordered_unique(order, nodes.size(0))
    env = RouteEnvironment(nodes[None], depot[None], pref[None],
                           reserve_fraction=reserve_fraction,
                           objective_variant=objective_variant,
                           energy_scenario=energy_scenario)
    actions = []
    wait_schedule = dict(wait_schedule or {})
    if any(int(sortie) < 0 or int(count) != count or count < 0
           for sortie, count in wait_schedule.items()):
        raise ValueError("Wait schedule must map nonnegative sortie indexes to counts")
    scheduled_waits_used = {}
    # Sensitivity scenarios can need many one-customer sorties and warehouse
    # waits. The earlier 5*N+10 guard could truncate a physically legal N100
    # construction after 510 actions; a prior reserve/efficiency pilot used 662.
    action_limit_hit = False
    for _ in range(10*env.N+32):
        if bool(env.complete()[0]) or bool(env.done()[0]):
            break
        mask = env.raw_mask()[0]
        if bool(env.at_depot()[0]):
            added = 0
            for i in order:
                if max_manifest is not None and added >= max_manifest:
                    break
                load = env.load_offset+i
                if not bool(mask[load]):
                    continue
                snapshot = env.snapshot()
                env.step(torch.tensor([load]))
                if _choose_flight(env, order, speed_labels) is None:
                    env.load_snapshot(snapshot)
                    continue
                actions.append(load)
                added += 1
                mask = env.raw_mask()[0]
            if not bool(env.loaded.any()):
                if bool(env.raw_mask()[0, env.wait_action]):
                    env.step(torch.tensor([env.wait_action]))
                    actions.append(env.wait_action)
                    continue
                break
        flight = _choose_flight(env, order, speed_labels)
        if flight is not None:
            sortie = int(env.return_count[0])
            wanted = int(wait_schedule.get(sortie, 0))
            used = scheduled_waits_used.get(sortie, 0)
            if used < wanted and bool(env.at_depot()[0]) and bool(
                env.raw_mask()[0, env.wait_action]
            ):
                env.step(torch.tensor([env.wait_action]))
                actions.append(env.wait_action)
                scheduled_waits_used[sortie] = used + 1
                continue
            env.step(torch.tensor([flight]))
            actions.append(flight)
        else:
            if bool(env.at_depot()[0]) and bool(env.raw_mask()[0, env.wait_action]):
                env.step(torch.tensor([env.wait_action]))
                actions.append(env.wait_action)
                continue
            back_options = [env.return_offset+m for m in range(2)
                            if bool(mask[env.return_offset+m])]
            if not back_options:
                break
            # The return mode is selected against the same time-energy objective.
            distance = float(torch.linalg.vector_norm(env.pos[0]-depot)*SIDE_M)
            alpha, beta = [float(x) for x in pref]
            back = min(back_options, key=lambda a: (
                0 if return_mode_preference is None
                or a-env.return_offset == return_mode_preference else 1,
                alpha*distance/SPEEDS_M_S[a-env.return_offset]/3600
                + beta*float(env.cruise_j(
                    distance, SPEEDS_M_S[a-env.return_offset], env.load_kg[0]))/BUDGET_J
            ))
            env.step(torch.tensor([back]))
            actions.append(back)
    else:
        action_limit_hit = not bool(env.complete()[0]) and not bool(env.done()[0])
    if not bool(env.at_depot()[0]) and not bool(env.complete()[0]):
        mask = env.raw_mask()[0]
        back = next((env.return_offset+m for m in range(2)
                     if bool(mask[env.return_offset+m])), None)
        if back is not None:
            actions.append(back)
    if return_diagnostics:
        return actions, {"construction_action_limit_hit": action_limit_hit}
    return actions


def score_order(nodes, depot, pref, order, *, max_manifest=None,
                reserve_fraction=0.0, provenance="physical_greedy",
                objective_variant="C0",
                energy_scenario: EnergyScenario | None = None,
                speed_labels=None, return_mode_preference=None,
                wait_schedule=None):
    nodes, depot, pref = _one(nodes, depot, pref)
    actions, construction = construct_actions(nodes, depot, pref, order,
                                max_manifest=max_manifest,
                                reserve_fraction=reserve_fraction,
                                objective_variant=objective_variant,
                                energy_scenario=energy_scenario,
                                speed_labels=speed_labels,
                                return_mode_preference=return_mode_preference,
                                wait_schedule=wait_schedule,
                                return_diagnostics=True)
    result = replay_actions(nodes, depot, pref, actions,
                            reserve_fraction=reserve_fraction, provenance=provenance,
                            objective_variant=objective_variant,
                            energy_scenario=energy_scenario)
    result["order"] = list(order)
    result.update(construction)
    result["speed_labels"] = dict(speed_labels or {})
    result["return_mode_preference"] = return_mode_preference
    result["wait_schedule"] = dict(wait_schedule or {})
    result["max_manifest"] = max_manifest
    result["manifest_limit_rule"] = (
        "capacity_and_physical_action_masks_only" if max_manifest is None
        else "explicit_customer_count_cap"
    )
    result["exact_physical_model"] = True
    result["native_objective"] = None
    return result


def greedy(nodes, depot, pref, *, reserve_fraction=0.0,
           objective_variant="C0", time_limit_s=None,
           energy_scenario: EnergyScenario | None = None,
           max_candidates=4):
    nodes, depot, pref = _one(nodes, depot, pref)
    started = time.perf_counter()
    candidates = [
        sorted(range(nodes.size(0)), key=lambda i: float(nodes[i, 4])),
        sorted(range(nodes.size(0)), key=lambda i: float(nodes[i, 3])),
        sorted(range(nodes.size(0)), key=lambda i: float(torch.linalg.vector_norm(nodes[i, :2]-depot))),
        list(range(nodes.size(0))),
    ]
    results = []
    if max_candidates < 1:
        raise ValueError("At least one constructor candidate is required")
    for order in candidates[:max_candidates]:
        if results and time_limit_s is not None and time.perf_counter()-started >= time_limit_s:
            break
        results.append(score_order(nodes, depot, pref, order,
                                   reserve_fraction=reserve_fraction,
                                   provenance="physical_greedy",
                                   objective_variant=objective_variant,
                                   energy_scenario=energy_scenario))
    best = min(results, key=_key)
    best["candidate_orders"] = len(results)
    best["elapsed_s"] = time.perf_counter()-started
    return best


def _weighted_operator(weights, rng):
    names = list(weights)
    return rng.choices(names, weights=[weights[name] for name in names], k=1)[0]


def _destroy_positions(order, nodes, depot, count, operator, rng):
    n = len(order)
    if operator == "random":
        return sorted(rng.sample(range(n), count), reverse=True)
    if operator == "related":
        anchor = nodes[order[rng.randrange(n)], :2]
        positions = sorted(range(n), key=lambda pos: float(
            torch.linalg.vector_norm(nodes[order[pos], :2]-anchor)))
        return sorted(positions[:count], reverse=True)
    if operator == "long_edge":
        scores = []
        for pos, customer in enumerate(order):
            previous = depot if pos == 0 else nodes[order[pos-1], :2]
            following = depot if pos == n-1 else nodes[order[pos+1], :2]
            point = nodes[customer, :2]
            score = (torch.linalg.vector_norm(point-previous)
                     + torch.linalg.vector_norm(point-following)
                     - torch.linalg.vector_norm(following-previous))
            scores.append((float(score), pos))
        return sorted((pos for _, pos in sorted(scores, reverse=True)[:count]), reverse=True)
    raise ValueError(f"Unknown destroy operator: {operator}")


def _repair_order(order, removed, nodes, depot, operator, rng):
    if operator == "due_first":
        removed.sort(key=lambda i: float(nodes[i, 4]))
    elif operator == "random_insertion":
        rng.shuffle(removed)
    elif operator == "spatial_insertion":
        rng.shuffle(removed)
    else:
        raise ValueError(f"Unknown repair operator: {operator}")
    for item in removed:
        if operator == "random_insertion":
            order.insert(rng.randrange(len(order)+1), item)
            continue
        positions = rng.sample(range(len(order)+1), min(5, len(order)+1))
        position = min(positions, key=lambda at: (
            float(torch.linalg.vector_norm(
                nodes[item, :2] - (depot if at == 0 else nodes[order[at-1], :2])))
            + (float(torch.linalg.vector_norm(
                nodes[item, :2] - nodes[order[at], :2])) if at < len(order) else 0)
        ))
        order.insert(position, item)
    return order


def alns(nodes, depot, pref, *, time_limit_s=30.0, seed=42,
         reserve_fraction=0.0, initial_order=None, objective_variant="C0",
         energy_scenario: EnergyScenario | None = None,
         search_loaded_wait=False):
    """Destroy/reinsert ALNS; default has no neural seed."""
    nodes, depot, pref = _one(nodes, depot, pref)
    n = nodes.size(0)
    rng = random.Random(seed)
    begun = time.perf_counter()
    if initial_order is None:
        best = greedy(nodes, depot, pref, reserve_fraction=reserve_fraction,
                      objective_variant=objective_variant, time_limit_s=time_limit_s,
                      energy_scenario=energy_scenario,
                      max_candidates=1 if time_limit_s < 30 else 4)
    else:
        best = score_order(nodes, depot, pref,
                           _ordered_unique(initial_order, nodes.size(0)),
                           reserve_fraction=reserve_fraction,
                           provenance="neural_warm_start_physical_alns",
                           objective_variant=objective_variant,
                           energy_scenario=energy_scenario)
    initial_incumbent_cost = float(best["metrics"]["cost"])
    estimated_proposal_s = max(0.01, time.perf_counter()-begun)
    incumbent = best
    accepted = 0
    iterations = 0
    destroy_weights = {"random": 1.0, "related": 1.0, "long_edge": 1.0}
    repair_weights = {"spatial_insertion": 1.0, "due_first": 1.0,
                      "random_insertion": 1.0}
    operator_counts = {f"{d}+{r}": 0 for d in destroy_weights for r in repair_weights}
    speed_neighborhood_proposals = 0
    wait_neighborhood_proposals = 0
    stopped_for_remaining_budget = False
    # Leave time for result packaging and the common external physical replay.
    # The caller charges both to the same wall-clock comparison budget.
    search_deadline_s = max(0.0, time_limit_s-max(0.25, 0.03*time_limit_s))
    while time.perf_counter()-begun < search_deadline_s:
        remaining_s = search_deadline_s-(time.perf_counter()-begun)
        if remaining_s < 1.2*estimated_proposal_s:
            stopped_for_remaining_budget = True
            break
        order = list(incumbent["order"])
        n = len(order)
        destroy = rng.randint(1, max(1, math.ceil(n*0.2)))
        destroy_op = _weighted_operator(destroy_weights, rng)
        repair_op = _weighted_operator(repair_weights, rng)
        operator_counts[f"{destroy_op}+{repair_op}"] += 1
        removed_at = _destroy_positions(order, nodes, depot, destroy, destroy_op, rng)
        removed = [order.pop(pos) for pos in removed_at]
        order = _repair_order(order, removed, nodes, depot, repair_op, rng)
        speed_labels = dict(incumbent.get("speed_labels") or {})
        return_mode_preference = incumbent.get("return_mode_preference")
        wait_schedule = dict(incumbent.get("wait_schedule") or {})
        if iterations % 2 == 0:
            customer = rng.randrange(n)
            if customer not in speed_labels:
                delivered = next((event for event in incumbent["events"]
                                  if event["kind"] == "deliver"
                                  and event["order"] == customer), None)
                speed_labels[customer] = (
                    0 if delivered is not None and
                    delivered["speed_m_s"] == SPEEDS_M_S[1] else 1
                )
            else:
                speed_labels[customer] = 1-speed_labels[customer]
            speed_neighborhood_proposals += 1
        if iterations % 5 == 4:
            return_mode_preference = (
                rng.randrange(2) if return_mode_preference is None
                else 1-return_mode_preference
            )
            speed_neighborhood_proposals += 1
        if search_loaded_wait and iterations % 3 == 0:
            sortie_count = max(1, int(incumbent["metrics"]["depot_returns"]))
            sortie = rng.randrange(sortie_count)
            current_waits = int(wait_schedule.get(sortie, 0))
            # A local +/-1 move can grow beyond two useful waits. The only
            # finite cap is the number of orders, since each ground-wait
            # action advances to another service-opening event.
            new_waits = max(0, min(n, current_waits+rng.choice((-1, 1))))
            if new_waits == current_waits:
                new_waits = min(n, current_waits+1)
            if new_waits:
                wait_schedule[sortie] = new_waits
            else:
                wait_schedule.pop(sortie, None)
            wait_neighborhood_proposals += 1
        proposal_started = time.perf_counter()
        proposal = score_order(nodes, depot, pref, order,
                               reserve_fraction=reserve_fraction,
                               provenance="standalone_physical_alns",
                               objective_variant=objective_variant,
                               energy_scenario=energy_scenario,
                               speed_labels=speed_labels,
                               return_mode_preference=return_mode_preference,
                               wait_schedule=wait_schedule)
        estimated_proposal_s = (
            0.5*estimated_proposal_s + 0.5*(time.perf_counter()-proposal_started)
        )
        iterations += 1
        improved_best = _key(proposal) < _key(best)
        if improved_best:
            best = proposal
        delta = _anneal_energy(proposal) - _anneal_energy(incumbent)
        temperature = max(0.02, 0.5*(1-(time.perf_counter()-begun)/time_limit_s))
        improved_incumbent = _key(proposal) < _key(incumbent)
        accept = improved_incumbent or rng.random() < math.exp(-max(0, delta)/temperature)
        if accept:
            incumbent = proposal
            accepted += 1
        reward = 5.0 if improved_best else (2.0 if improved_incumbent else (0.5 if accept else 0.0))
        destroy_weights[destroy_op] = 0.8*destroy_weights[destroy_op] + 0.2*(1+reward)
        repair_weights[repair_op] = 0.8*repair_weights[repair_op] + 0.2*(1+reward)
    best["algorithm"] = (
        ("standalone_physical_alns_waitaware" if search_loaded_wait
         else "standalone_physical_alns") if initial_order is None else
        ("neural_warm_start_physical_alns_waitaware" if search_loaded_wait
         else "neural_warm_start_physical_alns")
    )
    best["search_seed"] = seed
    best["initial_incumbent_cost"] = initial_incumbent_cost
    best["search_iterations"] = iterations
    best["accepted_moves"] = accepted
    best["destroy_operator_weights"] = destroy_weights
    best["repair_operator_weights"] = repair_weights
    best["operator_counts"] = operator_counts
    best["speed_neighborhood_proposals"] = speed_neighborhood_proposals
    best["wait_neighborhood_proposals"] = wait_neighborhood_proposals
    best["search_loaded_wait"] = bool(search_loaded_wait)
    best["stopped_for_remaining_budget"] = stopped_for_remaining_budget
    best["elapsed_s"] = time.perf_counter()-begun
    best["time_limit_s"] = time_limit_s
    best["algorithm_config"] = {
        "search_deadline_s": search_deadline_s,
        "initial_constructor_max_candidates": (
            None if initial_order is not None else
            (1 if time_limit_s < 30 else 4)),
        "destroy_count_max_fraction": 0.2,
        "destroy_operators": list(destroy_weights),
        "repair_operators": list(repair_weights),
        "operator_weight_retention": 0.8,
        "operator_weight_reward_scale": 0.2,
        "acceptance_temperature_start": 0.5,
        "acceptance_temperature_floor": 0.02,
        "acceptance_infeasible_penalty": 1000.0,
        "acceptance_per_unserved_penalty": 100.0,
        "speed_customer_move_every_iterations": 2,
        "return_speed_move_every_iterations": 5,
        "loaded_wait_move_every_iterations": 3 if search_loaded_wait else None,
        "loaded_wait_max_events_per_sortie": n if search_loaded_wait else None,
        "warm_started_from_neural": initial_order is not None,
    }
    return best


def _pareto_vector(report):
    metrics = report["metrics"]
    if not report["feasible"]:
        return (1e8 + 1e6*int(metrics["unserved"]),)*3
    return (float(metrics["flight_time_s"]),
            float(metrics["total_energy_j"])/100.0,
            float(metrics["tardiness_s"]))


def _dominates(a, b):
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))


def _pareto_ranks(vectors):
    remaining = set(range(len(vectors)))
    ranks = [0]*len(vectors)
    rank = 0
    while remaining:
        front = [i for i in remaining
                 if not any(_dominates(vectors[j], vectors[i])
                            for j in remaining if j != i)]
        if not front:
            front = list(remaining)
        for i in front:
            ranks[i] = rank
        remaining.difference_update(front)
        rank += 1
    return ranks


def _crowding(vectors, ranks):
    crowd = [0.0]*len(vectors)
    for rank in set(ranks):
        front = [i for i, value in enumerate(ranks) if value == rank]
        if len(front) <= 2:
            for i in front:
                crowd[i] = float("inf")
            continue
        for objective in range(3):
            ordered = sorted(front, key=lambda i: vectors[i][objective])
            lo = vectors[ordered[0]][objective]
            hi = vectors[ordered[-1]][objective]
            crowd[ordered[0]] = crowd[ordered[-1]] = float("inf")
            if hi <= lo:
                continue
            for pos in range(1, len(ordered)-1):
                if math.isfinite(crowd[ordered[pos]]):
                    crowd[ordered[pos]] += (
                        vectors[ordered[pos+1]][objective]
                        - vectors[ordered[pos-1]][objective]
                    )/(hi-lo)
    return crowd


def _order_crossover(left, right, rng):
    n = len(left)
    lo, hi = sorted(rng.sample(range(n), 2))
    child = [None]*n
    child[lo:hi+1] = left[lo:hi+1]
    tail = [i for i in right if i not in child]
    positions = list(range(hi+1, n)) + list(range(lo))
    for position, item in zip(positions, tail):
        child[position] = item
    return child


def nsga2(nodes, depot, pref, *, time_limit_s=30.0, seed=42,
          population_size=16, reserve_fraction=0.0,
          objective_variant="C0", energy_scenario: EnergyScenario | None = None,
          search_loaded_wait=False):
    """Independent order-and-speed NSGA-II with complete v3 physical replays."""
    nodes, depot, pref = _one(nodes, depot, pref)
    if population_size < 4:
        raise ValueError("NSGA-II population must contain at least four individuals")
    n = nodes.size(0)
    rng = random.Random(seed)
    begun = time.perf_counter()
    # Ranking the final archive and packaging the result also consumes the
    # shared wall-clock budget. Reserve a small tail for those operations.
    packaging_reserve_s = (2.0 if time_limit_s <= 10.0
                           else max(3.0, 0.15*time_limit_s))
    search_deadline_s = max(0.0, time_limit_s-packaging_reserve_s)
    # The four-route constructor can consume almost all of a five-second
    # budget before NSGA-II evaluates one generation. Use one seed route for
    # that tier, as the standalone ALNS already does.
    initial_candidates = 1 if time_limit_s < 30 else 4
    best = greedy(nodes, depot, pref, reserve_fraction=reserve_fraction,
                  objective_variant=objective_variant,
                  time_limit_s=time_limit_s, energy_scenario=energy_scenario,
                  max_candidates=initial_candidates)
    initial_incumbent_cost = float(best["metrics"]["cost"])
    initial_sortie_count = max(1, int(best["metrics"]["depot_returns"]))
    initial = list(range(nodes.size(0)))
    population = [
        {"order": initial, "speed_labels": {}, "return_mode_preference": None,
         "wait_schedule": {}},
        {"order": sorted(initial, key=lambda i: float(nodes[i, 3])),
         "speed_labels": {}, "return_mode_preference": None,
         "wait_schedule": {}},
        {"order": sorted(initial, key=lambda i: float(nodes[i, 4])),
         "speed_labels": {}, "return_mode_preference": None,
         "wait_schedule": {}},
    ]
    while len(population) < population_size:
        permutation = initial.copy()
        rng.shuffle(permutation)
        labels = {i: rng.randrange(2) for i in initial if rng.random() < 0.15}
        wait_schedule = (
            {rng.randrange(initial_sortie_count): rng.choice((1, 2))}
            if search_loaded_wait and rng.random() < 0.4 else {}
        )
        population.append({"order": permutation, "speed_labels": labels,
                           "return_mode_preference": rng.choice((None, 0, 1)),
                           "wait_schedule": wait_schedule})
    archive = [best]
    evaluated = 1
    speed_genetic_mutations = 0
    wait_genetic_mutations = 0
    wait_schedule_evaluations = 0
    while time.perf_counter()-begun < search_deadline_s:
        reports = []
        for genome in population:
            if time.perf_counter()-begun >= search_deadline_s:
                break
            if genome["wait_schedule"]:
                wait_schedule_evaluations += 1
            report = score_order(nodes, depot, pref, genome["order"],
                                 reserve_fraction=reserve_fraction,
                                 provenance="standalone_nsga2_v3",
                                 objective_variant=objective_variant,
                                 energy_scenario=energy_scenario,
                                 speed_labels=genome["speed_labels"],
                                 return_mode_preference=genome["return_mode_preference"],
                                 wait_schedule=genome["wait_schedule"])
            reports.append(report)
            evaluated += 1
            if best is None or _key(report) < _key(best):
                best = report
        if not reports:
            break
        archive.extend(reports)
        vectors = [_pareto_vector(r) for r in reports]
        ranks = _pareto_ranks(vectors)
        crowd = _crowding(vectors, ranks)
        chosen = sorted(range(len(reports)),
                        key=lambda i: (ranks[i], -crowd[i]))
        parents = [{"order": list(reports[i]["order"]),
                    "speed_labels": dict(reports[i]["speed_labels"]),
                    "return_mode_preference": reports[i]["return_mode_preference"],
                    "wait_schedule": dict(reports[i]["wait_schedule"])}
                   for i in chosen[:max(2, len(chosen)//2)]]
        if len(parents) < 2:
            break
        population = [{"order": parent["order"].copy(),
                       "speed_labels": parent["speed_labels"].copy(),
                       "return_mode_preference": parent["return_mode_preference"],
                       "wait_schedule": parent["wait_schedule"].copy()}
                      for parent in parents]
        while len(population) < population_size:
            left, right = rng.sample(parents, 2)
            child = _order_crossover(left["order"], right["order"], rng)
            if rng.random() < 0.7:
                a, b = rng.sample(range(len(child)), 2)
                child[a], child[b] = child[b], child[a]
            labels = {}
            for customer in set(left["speed_labels"]) | set(right["speed_labels"]):
                parent = left if rng.random() < 0.5 else right
                if customer in parent["speed_labels"]:
                    labels[customer] = parent["speed_labels"][customer]
            if rng.random() < 0.7:
                customer = rng.randrange(n)
                labels[customer] = (1-labels[customer] if customer in labels
                                    else rng.randrange(2))
                speed_genetic_mutations += 1
            if rng.random() < 0.1 and labels:
                labels.pop(rng.choice(tuple(labels)))
            return_mode = (left["return_mode_preference"] if rng.random() < 0.5
                           else right["return_mode_preference"])
            if rng.random() < 0.2:
                return_mode = rng.randrange(2)
                speed_genetic_mutations += 1
            wait_schedule = {}
            if search_loaded_wait:
                for sortie in set(left["wait_schedule"]) | set(right["wait_schedule"]):
                    donor = left if rng.random() < 0.5 else right
                    if sortie in donor["wait_schedule"]:
                        wait_schedule[sortie] = donor["wait_schedule"][sortie]
                if rng.random() < 0.5:
                    sortie = rng.randrange(max(1, int(best["metrics"]["depot_returns"])))
                    current_waits = int(wait_schedule.get(sortie, 0))
                    new_waits = max(0, min(n, current_waits+rng.choice((-1, 1))))
                    if new_waits == current_waits:
                        new_waits = min(n, current_waits+1)
                    if new_waits:
                        wait_schedule[sortie] = new_waits
                    else:
                        wait_schedule.pop(sortie, None)
                    wait_genetic_mutations += 1
            population.append({"order": child, "speed_labels": labels,
                               "return_mode_preference": return_mode,
                               "wait_schedule": wait_schedule})
    archive_vectors = [_pareto_vector(r) for r in archive]
    best["algorithm"] = (
        "standalone_nsga2_v3_waitaware" if search_loaded_wait
        else "standalone_nsga2_v3"
    )
    best["search_seed"] = seed
    best["initial_incumbent_cost"] = initial_incumbent_cost
    best["evaluations"] = evaluated
    best["archive_nondominated_count"] = sum(r == 0 for r in _pareto_ranks(archive_vectors))
    best["population_size"] = population_size
    best["speed_genetic_mutations"] = speed_genetic_mutations
    best["wait_genetic_mutations"] = wait_genetic_mutations
    best["wait_schedule_evaluations"] = wait_schedule_evaluations
    best["search_loaded_wait"] = bool(search_loaded_wait)
    best["elapsed_s"] = time.perf_counter()-begun
    best["time_limit_s"] = time_limit_s
    best["algorithm_config"] = {
        "population_size": population_size,
        "initial_constructor_max_candidates": initial_candidates,
        "search_deadline_s": search_deadline_s,
        "order_swap_probability": 0.7,
        "speed_mutation_probability": 0.7,
        "speed_label_deletion_probability": 0.1,
        "return_speed_mutation_probability": 0.2,
        "loaded_wait_initialization_probability": 0.4 if search_loaded_wait else None,
        "loaded_wait_mutation_probability": 0.5 if search_loaded_wait else None,
        "loaded_wait_max_events_per_sortie": n if search_loaded_wait else None,
        "seeded_with_neural_solution": False,
    }
    return best


def _distance_matrix(nodes, depot):
    xy = torch.cat([depot[None], nodes[:, :2]], dim=0)
    return torch.cdist(xy[None], xy[None])[0].tolist()


def _proxy_cost_matrix(distance, pref, energy_scenario, representative_load):
    """Precompute the disclosed route-order surrogate before native search."""
    metres = torch.as_tensor(distance, dtype=torch.float64) * SIDE_M
    alpha, beta = [float(v) for v in pref]
    by_speed = [
        alpha*metres/s/3600
        + beta*energy_scenario.cruise_j(metres, s, representative_load)/BUDGET_J
        for s in SPEEDS_M_S
    ]
    return torch.minimum(by_speed[0], by_speed[1]).tolist()


def ortools_order(nodes, depot, pref, *, time_limit_s=30.0,
                  energy_scenario: EnergyScenario | None = None):
    try:
        import ortools
        from ortools.constraint_solver import pywrapcp, routing_enums_pb2
    except ImportError as exc:
        raise RuntimeError("Install ortools to run the v3 OR-Tools proxy") from exc
    nodes, depot, pref = _one(nodes, depot, pref)
    energy_scenario = energy_scenario or SCENARIOS["gu_nominal"]
    n = nodes.size(0)
    distance = _distance_matrix(nodes, depot)
    manager = pywrapcp.RoutingIndexManager(n+1, 1, 0)
    routing = pywrapcp.RoutingModel(manager)
    representative_load = min(Q_MAX_KG/2, float(nodes[:, 2].sum())/max(1, n))
    arc_cost = _proxy_cost_matrix(distance, pref, energy_scenario, representative_load)
    def arc(from_index, to_index):
        i, j = manager.IndexToNode(from_index), manager.IndexToNode(to_index)
        return max(1, round(arc_cost[i][j]*1_000_000)) if i != j else 0
    transit = routing.RegisterTransitCallback(arc)
    routing.SetArcCostEvaluatorOfAllVehicles(transit)
    settings = pywrapcp.DefaultRoutingSearchParameters()
    settings.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    settings.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    settings.time_limit.seconds = max(1, math.ceil(time_limit_s))
    solution = routing.SolveWithParameters(settings)
    if solution is None:
        return None, {"status": "no_solution", "native_objective": None,
                      "native_solver": "ortools",
                      "native_solver_version": ortools.__version__}
    index = routing.Start(0)
    order = []
    while not routing.IsEnd(index):
        node = manager.IndexToNode(index)
        if node:
            order.append(node-1)
        index = solution.Value(routing.NextVar(index))
    return order, {"status": "proxy_solution", "native_objective": solution.ObjectiveValue(),
                   "native_solver": "ortools",
                   "native_solver_version": ortools.__version__,
                   "proxy": "single-vehicle TSP order; representative cargo mass; no serial depot reset"}


def pyvrp_order(nodes, depot, pref, *, time_limit_s=30.0, seed=42,
                energy_scenario: EnergyScenario | None = None):
    try:
        import pyvrp
        from pyvrp import Model
        from pyvrp.stop import MaxRuntime
    except ImportError as exc:
        raise RuntimeError("Install pyvrp to run the v3 PyVRP proxy") from exc
    nodes, depot, pref = _one(nodes, depot, pref)
    energy_scenario = energy_scenario or SCENARIOS["gu_nominal"]
    n = nodes.size(0)
    model = Model()
    origin_location = model.add_location(x=0, y=0)
    origin = model.add_depot(origin_location)
    model.add_vehicle_type(num_available=1, capacity=int(Q_MAX_KG*1000*n),
                           start_depot=origin, end_depot=origin)
    client_locations = []
    for i in range(n):
        location = model.add_location(
            x=int(round((float(nodes[i, 0])-float(depot[0]))*100000)),
            y=int(round((float(nodes[i, 1])-float(depot[1]))*100000)),
        )
        model.add_client(location, delivery=int(round(float(nodes[i, 2])*1000)))
        client_locations.append(location)
    locations = [origin_location] + client_locations
    distance = _distance_matrix(nodes, depot)
    arc_cost = _proxy_cost_matrix(distance, pref, energy_scenario, 0.0)
    for i, frm in enumerate(locations):
        for j, to in enumerate(locations):
            if i == j:
                continue
            metres = distance[i][j]*SIDE_M
            model.add_edge(frm, to, distance=max(1, round(arc_cost[i][j]*1_000_000)),
                           duration=max(1, round(metres/SPEEDS_M_S[1])))
    result = model.solve(stop=MaxRuntime(max(1.0, time_limit_s)), seed=seed, display=False)
    if result.best is None:
        return None, {"status": "no_solution", "native_objective": None,
                      "native_solver": "pyvrp",
                      "native_solver_version": getattr(pyvrp, "__version__", "unknown")}
    routes = list(result.best.routes())
    order = [int(activity.idx) for route in routes for activity in route
             if activity.is_client()]
    return order, {"status": "proxy_solution", "native_objective": float(result.cost()),
                   "native_solver": "pyvrp",
                   "native_solver_version": getattr(pyvrp, "__version__", "unknown"),
                   "proxy": "single giant-route order; empty-load edge energy; no sortie battery"}


def gurobi_order(nodes, depot, pref, *, time_limit_s=30.0, threads=1,
                 energy_scenario: EnergyScenario | None = None):
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError as exc:
        raise RuntimeError("Install and license gurobipy to run the v3 Gurobi proxy") from exc
    nodes, depot, pref = _one(nodes, depot, pref)
    energy_scenario = energy_scenario or SCENARIOS["gu_nominal"]
    n = nodes.size(0)
    distance = _distance_matrix(nodes, depot)
    model = gp.Model("v3_order_proxy_not_exact_physics")
    model.Params.OutputFlag = 0
    model.Params.TimeLimit = time_limit_s
    model.Params.Threads = threads
    x = model.addVars(n+1, n+1, vtype=GRB.BINARY, name="arc")
    u = model.addVars(range(1, n+1), lb=1, ub=n, name="mtz")
    for i in range(n+1):
        x[i, i].ub = 0
        model.addConstr(gp.quicksum(x[i, j] for j in range(n+1)) == 1)
        model.addConstr(gp.quicksum(x[j, i] for j in range(n+1)) == 1)
    for i in range(1, n+1):
        for j in range(1, n+1):
            if i != j:
                model.addConstr(u[i]-u[j]+n*x[i, j] <= n-1)
    arc_cost = _proxy_cost_matrix(distance, pref, energy_scenario, 0.0)
    costs = {}
    for i in range(n+1):
        for j in range(n+1):
            if i == j:
                continue
            costs[i, j] = arc_cost[i][j]
    model.setObjective(gp.quicksum(costs[i, j]*x[i, j] for i, j in costs), GRB.MINIMIZE)
    try:
        model.optimize()
    except gp.GurobiError as exc:
        if exc.errno == GRB.Error.OUT_OF_MEMORY:
            return None, {"status": "observed_out_of_memory", "error_code": exc.errno,
                          "error_message": str(exc), "native_objective": None,
                          "native_bound": None, "native_mip_gap": None,
                          "native_solver": "gurobi",
                          "native_solver_version": ".".join(
                              map(str, gp.gurobi.version())),
                          "proxy": "TSP order; empty-load energy; no capacity, wait, recharge or sorties"}
        raise
    termination = {
        GRB.OPTIMAL: "optimal_proxy",
        GRB.TIME_LIMIT: "time_limit_proxy",
        GRB.MEM_LIMIT: "memory_limit_proxy",
        GRB.INFEASIBLE: "infeasible_proxy",
        GRB.INF_OR_UNBD: "infeasible_or_unbounded_proxy",
    }.get(model.Status, "other_proxy_termination")
    details = {"status_code": int(model.Status), "status": termination,
               "native_solver": "gurobi",
               "native_solver_version": ".".join(map(str, gp.gurobi.version())),
               "has_incumbent": bool(model.SolCount),
               "native_objective": float(model.ObjVal) if model.SolCount else None,
               "native_bound": float(model.ObjBound) if model.Status not in (GRB.INFEASIBLE, GRB.INF_OR_UNBD) else None,
               "native_mip_gap": float(model.MIPGap) if model.SolCount else None,
               "proxy": "TSP order; empty-load energy; no capacity, wait, recharge or sorties"}
    if not model.SolCount:
        return None, details
    order = []
    current = 0
    for _ in range(n):
        successors = [j for j in range(1, n+1) if x[current, j].X > 0.5]
        if len(successors) != 1:
            return None, {**details, "status": "invalid_incumbent"}
        current = successors[0]
        order.append(current-1)
    return order, details


def solve_proxy(method, nodes, depot, pref, *, time_limit_s=30.0, seed=42,
                reserve_fraction=0.0, objective_variant="C0",
                energy_scenario: EnergyScenario | None = None):
    nodes, depot, pref = _one(nodes, depot, pref)
    started = time.perf_counter()
    # The shared budget includes native optimization, physical construction,
    # and common replay. Reserve time for the latter two rather than handing
    # the entire budget to the native solver and then always timing out.
    repair_reserve_s = max(2.0, 0.15*time_limit_s)
    native_budget_s = max(1.0, time_limit_s-repair_reserve_s)
    if method == "ortools":
        order, details = ortools_order(nodes, depot, pref, time_limit_s=native_budget_s,
                                      energy_scenario=energy_scenario)
    elif method == "pyvrp":
        order, details = pyvrp_order(nodes, depot, pref, time_limit_s=native_budget_s, seed=seed,
                                     energy_scenario=energy_scenario)
    elif method == "gurobi":
        try:
            order, details = gurobi_order(nodes, depot, pref, time_limit_s=native_budget_s,
                                          energy_scenario=energy_scenario)
        except Exception as exc:
            if type(exc).__name__ != "GurobiError":
                raise
            code = getattr(exc, "errno", None)
            try:
                import gurobipy as gp
                solver_version = ".".join(map(str, gp.gurobi.version()))
            except Exception:
                solver_version = "unknown"
            order, details = None, {
                "status": ("observed_out_of_memory" if code == 10001
                           else "native_solver_error"),
                "error_code": code, "error_message": str(exc),
                "native_solver": "gurobi",
                "native_solver_version": solver_version,
                "native_objective": None, "native_bound": None,
                "native_mip_gap": None,
            }
    else:
        raise ValueError(f"Unknown proxy method: {method}")
    native_elapsed = time.perf_counter()-started
    details["requested_total_budget_s"] = time_limit_s
    details["native_solver_budget_s"] = native_budget_s
    details["reserved_for_physical_repair_s"] = repair_reserve_s
    if order is None:
        return {"method": method, "native": details, "feasible": False,
                "failure_reason": details.get("status", "native_solver_no_valid_order"),
                "native_elapsed_s": native_elapsed,
                "total_elapsed_s": native_elapsed, "exact_physical_model": False,
                "final_scored_by_v3_environment": False,
                "native_objective_scope": "route_order_surrogate_only",
                "native_bound_gap_scope": "route_order_surrogate_only"}
    result = score_order(nodes, depot, pref, order,
                         reserve_fraction=reserve_fraction,
                         provenance=f"{method}_proxy_order_v3_physical_repair",
                         objective_variant=objective_variant,
                         energy_scenario=energy_scenario)
    result["method"] = method
    result["native"] = details
    # Native bounds and gaps describe the route-order proxy only. A bound on
    # the complete physical task requires a separate relaxation proof.
    result["native_elapsed_s"] = native_elapsed
    result["total_elapsed_s"] = time.perf_counter()-started
    result["exact_physical_model"] = False
    result["final_scored_by_v3_environment"] = True
    result["native_objective_scope"] = "route_order_surrogate_only"
    result["native_bound_gap_scope"] = "route_order_surrogate_only"
    result["within_total_budget"] = result["total_elapsed_s"] <= time_limit_s
    return result
