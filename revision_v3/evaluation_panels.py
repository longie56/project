"""Prospectively specified test prefixes; never select instances by outcomes."""


def validate_test_panel(args, specification):
    panels = specification.get("evaluation_panels", {})
    if args.evaluation_panel not in panels:
        raise ValueError("Evaluation panel is absent from the frozen specification")
    panel = panels[args.evaluation_panel]
    if (args.n_nodes not in panel["scales"] or args.method not in panel["methods"]
            or args.preference_set != panel["preference_set"]
            or args.budget_tier not in panel["budget_tiers"]):
        raise ValueError("Method, scale, preference grid or budget differs from the panel")
    if args.start != 0 or args.count is not None:
        raise ValueError("Test slices must come from the named frozen panel, not CLI overrides")
    if args.objective_variant == "C1" and not (
            args.evaluation_panel == "main" and args.n_nodes == 20):
        raise ValueError("The compact C1 objective is registered only for the N20 main panel")
    if panel.get("nominal_only") and (
            args.energy_scenario != "gu_nominal" or args.reserve_fraction != 0
            or args.time_window_scale != 1 or args.objective_variant != "C0"):
        raise ValueError("This diagnostic panel requires nominal C0 physics")
    if args.search_seed not in specification["baseline_search_seeds"]:
        raise ValueError("Search seed is outside the frozen search-seed list")
    if args.method in ("neural", "neural_analytic_speed", "neural_fixed_slow",
                       "neural_fixed_fast") and args.search_seed != specification[
            "baseline_search_seeds"][0]:
        raise ValueError("Neural inference uses one canonical search seed; variability comes from training seeds")
    point = (args.energy_scenario, args.reserve_fraction, args.time_window_scale)
    if args.evaluation_panel == "sensitivity":
        allowed = {(p["energy_scenario"], p["reserve_fraction"], p["time_window_scale"])
                   for values in specification["panels"].values() for p in values}
        if point not in allowed or args.objective_variant != "C0":
            raise ValueError("Sensitivity requires a preregistered single-factor C0 point")
    elif point != ("gu_nominal", 0.0, 1.0):
        raise ValueError("Physical perturbations require the sensitivity panel")
    return panel["instances_per_kind"]
