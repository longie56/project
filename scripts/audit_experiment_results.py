import argparse
import glob
import json
import math
import os
from collections import Counter


METHOD_LABELS = {
    "drl": "PC-EGA-POMO greedy",
    "alns": "PC-EGA-POMO + ALNS",
    "ortools": "OR-Tools",
    "ortools_physical": "OR-Tools physical",
    "pyvrp": "PyVRP",
    "lkh3": "LKH-3",
    "gurobi": "Gurobi-MILP",
}


REQUIRED_CONFIG_KEYS = [
    "instances",
    "n_nodes",
    "problem_type",
    "scenario",
    "tightness",
    "checkpoint",
]


REQUIRED_METHOD_METRICS = [
    "avg_cost",
    "avg_phys",
    "avg_pen",
    "avg_unvisit",
    "feasible_rate",
]


def _safe_float(value):
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _add(findings, level, code, message):
    findings.append({"level": level, "code": code, "message": message})


def _method_time(method, payload):
    if method == "alns" and payload.get("avg_total_time_s") is not None:
        return _safe_float(payload.get("avg_total_time_s"))
    return _safe_float(payload.get("avg_time_s"))


def _summary_methods(obj):
    summary = obj.get("summary", {})
    return {
        key: value
        for key, value in summary.items()
        if isinstance(value, dict) and "avg_cost" in value
    }


def _flat_methods_from_rows(obj):
    rows = obj.get("rows", [])
    methods = {}
    if not rows:
        return methods

    candidates = {
        "drl": ("drl_cost", "drl_time_s", "drl_feasible"),
        "alns": ("alns_cost", "alns_time_s", "alns_feasible"),
        "gurobi": ("gurobi_cost_env", "gurobi_time_s", "gurobi_feasible_env"),
    }
    for method, (cost_key, time_key, feasible_key) in candidates.items():
        costs = [_safe_float(row.get(cost_key)) for row in rows]
        costs = [x for x in costs if x is not None]
        if not costs:
            continue
        times = [_safe_float(row.get(time_key)) for row in rows]
        times = [x for x in times if x is not None]
        feasible = [row.get(feasible_key) for row in rows if row.get(feasible_key) is not None]
        payload = {
            "avg_cost": sum(costs) / len(costs),
            "avg_time_s": sum(times) / len(times) if times else None,
            "feasible_rate": (
                sum(float(bool(x)) for x in feasible) / len(feasible)
                if feasible
                else None
            ),
        }
        if method == "alns":
            total_times = [_safe_float(row.get("total_time_s")) for row in rows]
            total_times = [x for x in total_times if x is not None]
            if total_times:
                payload["avg_total_time_s"] = sum(total_times) / len(total_times)
        for metric_key, row_key in [
            ("avg_phys", f"{method}_phys" if method != "gurobi" else "gurobi_phys_env"),
            ("avg_pen", f"{method}_pen" if method != "gurobi" else "gurobi_pen_env"),
            ("avg_unvisit", f"{method}_unvisit" if method != "gurobi" else "gurobi_unvisit_env"),
        ]:
            values = [_safe_float(row.get(row_key)) for row in rows]
            values = [x for x in values if x is not None]
            if values:
                payload[metric_key] = sum(values) / len(values)
        methods[method] = payload
    return methods


def _method_payloads(obj):
    methods = _summary_methods(obj)
    if methods:
        return methods
    return _flat_methods_from_rows(obj)


def _audit_config(obj, path, findings):
    config = obj.get("config", {})
    if not isinstance(config, dict):
        _add(findings, "FAIL", "config_missing", "Result JSON has no config object.")
        return
    for key in REQUIRED_CONFIG_KEYS:
        if key not in config or config.get(key) in (None, ""):
            if key == "checkpoint" and not any(k.startswith("gurobi") for k in config):
                _add(findings, "WARN", "config_checkpoint_missing", "No checkpoint recorded.")
            elif key != "checkpoint":
                _add(findings, "WARN", "config_field_missing", f"Missing config field: {key}.")
    rows = obj.get("rows", [])
    declared_instances = config.get("instances")
    if declared_instances is not None and rows:
        try:
            declared_instances = int(declared_instances)
        except (TypeError, ValueError):
            declared_instances = None
        if declared_instances is not None and declared_instances != len(rows):
            _add(
                findings,
                "FAIL",
                "row_count_mismatch",
                f"config.instances={declared_instances}, but rows={len(rows)}.",
            )
    if "solomon" in os.path.normpath(path).lower() and config.get("dataset") != "solomon":
        _add(
            findings,
            "WARN",
            "dataset_name_ambiguous",
            "Path mentions Solomon but config does not record dataset='solomon'.",
        )
    problem_type = str(config.get("problem_type", "")).upper()
    instances = config.get("instances")
    try:
        instances = int(instances)
    except (TypeError, ValueError):
        instances = None
    if problem_type in {"MIX", "MIXED", "R+C+RC", "R_C_RC", "R,C,RC", "R_C_RC_MIXED"}:
        if instances is not None and instances < 3:
            _add(
                findings,
                "WARN",
                "mixed_sample_too_small",
                (
                    "MIXED generation allocates instances across R/C/RC; "
                    "instances < 3 cannot cover all three types."
                ),
            )


def _audit_method_metrics(methods, findings):
    if not methods:
        _add(findings, "FAIL", "methods_missing", "No method summaries or flat row methods found.")
        return
    for method, payload in methods.items():
        missing = [key for key in REQUIRED_METHOD_METRICS if payload.get(key) is None]
        if missing:
            _add(
                findings,
                "WARN",
                "method_metrics_incomplete",
                f"{METHOD_LABELS.get(method, method)} missing metrics: {', '.join(missing)}.",
            )
        feasible_rate = _safe_float(payload.get("feasible_rate"))
        unvisit = _safe_float(payload.get("avg_unvisit"))
        if feasible_rate is not None and feasible_rate < 1.0:
            _add(
                findings,
                "WARN",
                "feasibility_below_one",
                f"{METHOD_LABELS.get(method, method)} feasible_rate={feasible_rate:.4f}; rank feasibility-first.",
            )
        if unvisit is not None and unvisit > 0:
            _add(
                findings,
                "WARN",
                "unvisited_positive",
                f"{METHOD_LABELS.get(method, method)} avg_unvisit={unvisit:.4f}; do not compare by cost alone.",
            )


def _audit_distribution_stats(obj, methods, findings):
    config = obj.get("config", {})
    instances = config.get("instances")
    try:
        instances = int(instances)
    except (TypeError, ValueError):
        instances = None
    if instances is None or instances <= 1:
        return
    required = ["std_cost", "median_cost", "p90_cost"]
    for method, payload in methods.items():
        missing = [key for key in required if payload.get(key) is None]
        if missing:
            _add(
                findings,
                "WARN",
                "distribution_stats_missing",
                (
                    f"{METHOD_LABELS.get(method, method)} missing distribution stats: "
                    f"{', '.join(missing)}."
                ),
            )


def _audit_realtime(methods, budget_s, tolerance_s, findings):
    if budget_s is None:
        _add(
            findings,
            "WARN",
            "realtime_budget_missing",
            "Realtime protocol inferred, but no budget was provided or found in config.",
        )
        return
    for method, payload in methods.items():
        observed = _method_time(method, payload)
        if observed is None:
            _add(
                findings,
                "WARN",
                "time_missing",
                f"{METHOD_LABELS.get(method, method)} has no observed wall time.",
            )
            continue
        if observed > budget_s + tolerance_s:
            _add(
                findings,
                "FAIL",
                "realtime_budget_exceeded",
                (
                    f"{METHOD_LABELS.get(method, method)} observed time {observed:.3f}s "
                    f"exceeds realtime budget {budget_s:.3f}s + tolerance {tolerance_s:.3f}s."
                ),
            )


def _audit_gurobi(obj, findings):
    rows = obj.get("rows", [])
    if not rows:
        return
    gurobi_rows = [row for row in rows if "gurobi_status" in row]
    if not gurobi_rows:
        return
    status_counts = Counter(row.get("gurobi_status") for row in gurobi_rows)
    optimal_count = status_counts.get(2, 0)
    if optimal_count < len(gurobi_rows):
        _add(
            findings,
            "WARN",
            "gurobi_not_all_optimal",
            (
                f"Gurobi optimal rows {optimal_count}/{len(gurobi_rows)}; "
                "report incumbent/bound, not exact optimum."
            ),
        )
    finite_bound = sum(1 for row in gurobi_rows if bool(row.get("gurobi_bound_finite")))
    if finite_bound < len(gurobi_rows):
        _add(
            findings,
            "WARN",
            "gurobi_bound_missing",
            f"Gurobi finite objective bound rows {finite_bound}/{len(gurobi_rows)}.",
        )
    max_gap = None
    for row in gurobi_rows:
        gap = _safe_float(row.get("gurobi_mip_gap"))
        if gap is not None:
            max_gap = gap if max_gap is None else max(max_gap, gap)
    if max_gap is not None and max_gap > 0.01:
        _add(
            findings,
            "WARN",
            "gurobi_gap_large",
            f"Maximum Gurobi MIP gap is {max_gap:.6f}; not a tight exact reference.",
        )
    summary = obj.get("summary", {})
    if summary.get("exact_optimality_claim_allowed") is False:
        _add(
            findings,
            "WARN",
            "exact_claim_disallowed",
            "Summary explicitly marks exact_optimality_claim_allowed=false.",
        )


def _infer_protocol(path, obj, requested_protocol):
    if requested_protocol != "auto":
        return requested_protocol
    name = os.path.basename(path).lower()
    if "quality" in name:
        return "quality"
    if "realtime" in name or "strict" in name or "dynamic_insert_5s" in name:
        return "realtime"
    rows = obj.get("rows", [])
    if any("gurobi_status" in row for row in rows):
        return "milp"
    return "static"


def _infer_realtime_budget(obj, explicit_budget):
    if explicit_budget is not None:
        return explicit_budget
    config = obj.get("config", {})
    for key in ("total_time_budget", "alns_time_limit", "ortools_time_limit", "pyvrp_time_limit"):
        value = _safe_float(config.get(key))
        if value is not None and value > 0:
            return value
    return None


def audit_file(path, protocol, realtime_budget_s, tolerance_s):
    obj = _load_json(path)
    if "results" in obj and "config" not in obj and "summary" not in obj:
        return None
    protocol = _infer_protocol(path, obj, protocol)
    realtime_budget_s = _infer_realtime_budget(obj, realtime_budget_s)
    findings = []
    _audit_config(obj, path, findings)
    methods = _method_payloads(obj)
    _audit_method_metrics(methods, findings)
    _audit_distribution_stats(obj, methods, findings)
    if protocol == "realtime":
        _audit_realtime(methods, realtime_budget_s, tolerance_s, findings)
    _audit_gurobi(obj, findings)
    if not findings:
        _add(findings, "PASS", "protocol_clean", "No audit findings.")
    return {"path": path, "protocol": protocol, "findings": findings}


def _markdown(results):
    lines = []
    for result in results:
        lines.append(f"## {result['path']}")
        lines.append("")
        for item in result["findings"]:
            lines.append(f"- {item['level']}: `{item['code']}` - {item['message']}")
        lines.append("")
    return "\n".join(lines)


def _print_markdown(results):
    print(_markdown(results))


def main():
    parser = argparse.ArgumentParser(
        description="Audit experiment result JSON files against the reporting protocol."
    )
    parser.add_argument("json_paths", nargs="+")
    parser.add_argument(
        "--protocol",
        choices=["auto", "static", "realtime", "quality", "milp"],
        default="auto",
    )
    parser.add_argument("--realtime-budget-s", type=float, default=None)
    parser.add_argument("--time-tolerance-s", type=float, default=0.05)
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-md", default="")
    parser.add_argument("--fail-on", choices=["never", "fail", "warn"], default="never")
    args = parser.parse_args()

    json_paths = []
    for pattern in args.json_paths:
        matches = glob.glob(pattern)
        if matches:
            json_paths.extend(sorted(matches))
        else:
            json_paths.append(pattern)

    results = []
    for path in json_paths:
        result = audit_file(path, args.protocol, args.realtime_budget_s, args.time_tolerance_s)
        if result is not None:
            results.append(result)
    if not results:
        raise SystemExit("No raw experiment result JSON files found.")
    _print_markdown(results)

    if args.output_json:
        out_dir = os.path.dirname(os.path.abspath(args.output_json))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump({"results": results}, f, ensure_ascii=False, indent=2)
    if args.output_md:
        out_dir = os.path.dirname(os.path.abspath(args.output_md))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.output_md, "w", encoding="utf-8") as f:
            f.write(_markdown(results) + "\n")

    levels = {item["level"] for result in results for item in result["findings"]}
    if args.fail_on == "fail" and "FAIL" in levels:
        raise SystemExit(1)
    if args.fail_on == "warn" and ({"FAIL", "WARN"} & levels):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
