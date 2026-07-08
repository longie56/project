import argparse
import glob
import json
import math
import os


METHOD_LABELS = {
    "gurobi": "Gurobi-MILP incumbent",
    "ortools": "OR-Tools",
    "ortools_physical": "OR-Tools (physical)",
    "pyvrp": "PyVRP",
    "drl": "PC-EGA-POMO, greedy",
    "alns": "PC-EGA-POMO + ALNS",
    "am": "Attention Model (AM)",
    "lkh3": "LKH-3",
}


TYPE_ORDER = ["R", "C", "RC", "MIXED"]


def _safe_float(value):
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _fmt_value(value, digits=4):
    value = _safe_float(value)
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def _fmt_gap(cost, ref_cost):
    cost = _safe_float(cost)
    ref_cost = _safe_float(ref_cost)
    if cost is None or ref_cost is None or ref_cost <= 0:
        return "-"
    return f"{(cost - ref_cost) / ref_cost * 100.0:.2f}%"


def _fmt_time(seconds):
    seconds = _safe_float(seconds)
    if seconds is None:
        return "-"
    if seconds < 1:
        return "<<1s" if seconds < 0.5 else f"{seconds:.2f}s"
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = seconds / 60.0
    if minutes < 60:
        return f"{minutes:.1f}m"
    return f"{minutes / 60.0:.1f}h"


def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _expand_paths(patterns):
    paths = []
    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            paths.extend(sorted(matches))
        else:
            paths.append(pattern)
    return paths


def _summary_methods(obj):
    summary = obj.get("summary", {})
    return {
        key: value
        for key, value in summary.items()
        if isinstance(value, dict) and "avg_cost" in value
    }


def _method_time(method, payload):
    if method == "alns" and payload.get("avg_total_time_s") is not None:
        return payload.get("avg_total_time_s")
    return payload.get("avg_time_s")


def _type_sort_key(problem_type):
    problem_type = str(problem_type).upper()
    if problem_type in TYPE_ORDER:
        return (TYPE_ORDER.index(problem_type), problem_type)
    return (len(TYPE_ORDER), problem_type)


def _read_result(path):
    obj = _load_json(path)
    if "results" in obj and "summary" not in obj:
        return None
    config = obj.get("config", {})
    problem_type = str(config.get("problem_type") or os.path.basename(path)).upper()
    return {
        "path": path,
        "problem_type": problem_type,
        "instances": config.get("instances", obj.get("summary", {}).get("instances")),
        "config": config,
        "methods": _summary_methods(obj),
    }


def _collect(paths, methods):
    results = []
    for path in paths:
        result = _read_result(path)
        if result is not None and result["methods"]:
            results.append(result)
    results.sort(key=lambda item: _type_sort_key(item["problem_type"]))

    if methods:
        selected_methods = methods
    else:
        seen = []
        for result in results:
            for method in result["methods"]:
                if method not in seen:
                    seen.append(method)
        selected_methods = seen
    return results, selected_methods


def _build_detailed_table(results, methods, title, include_stats=False):
    if include_stats:
        header = (
            "| Type | Method | n | Cost | Cost Std | Cost Median | Cost P90 | "
            "Gap vs type best | Phys | Penalty | Unvisit | Time | Feasible |"
        )
        sep = "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    else:
        header = "| Type | Method | n | Cost | Gap vs type best | Phys | Penalty | Unvisit | Time | Feasible |"
        sep = "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"
    lines = [f"**{title}**", "", header, sep]
    for result in results:
        available = [
            result["methods"][method]
            for method in methods
            if method in result["methods"] and _safe_float(result["methods"][method].get("avg_cost")) is not None
        ]
        ref_cost = min((_safe_float(item.get("avg_cost")) for item in available), default=None)
        for method in methods:
            if method not in result["methods"]:
                continue
            payload = result["methods"][method]
            row_values = {
                "ptype": result["problem_type"],
                "method": METHOD_LABELS.get(method, method),
                "n": result["instances"] if result["instances"] is not None else "-",
                "cost": _fmt_value(payload.get("avg_cost")),
                "cost_std": _fmt_value(payload.get("std_cost")),
                "cost_median": _fmt_value(payload.get("median_cost")),
                "cost_p90": _fmt_value(payload.get("p90_cost")),
                "gap": _fmt_gap(payload.get("avg_cost"), ref_cost),
                "phys": _fmt_value(payload.get("avg_phys")),
                "pen": _fmt_value(payload.get("avg_pen")),
                "unvisit": _fmt_value(payload.get("avg_unvisit")),
                "time": _fmt_time(_method_time(method, payload)),
                "feas": _fmt_value(payload.get("feasible_rate"), 3),
            }
            if include_stats:
                lines.append(
                    "| {ptype} | {method} | {n} | {cost} | {cost_std} | {cost_median} | {cost_p90} | "
                    "{gap} | {phys} | {pen} | {unvisit} | {time} | {feas} |".format(**row_values)
                )
            else:
                lines.append(
                    "| {ptype} | {method} | {n} | {cost} | {gap} | {phys} | {pen} | "
                    "{unvisit} | {time} | {feas} |".format(**row_values)
                )
    return "\n".join(lines)


def _build_compact_table(results, methods, title):
    type_map = {result["problem_type"]: result for result in results}
    ordered_types = [ptype for ptype in TYPE_ORDER if ptype in type_map]
    ordered_types.extend(
        ptype for ptype in sorted(type_map, key=_type_sort_key) if ptype not in ordered_types
    )
    header = ["Method"]
    for ptype in ordered_types:
        header.extend([f"{ptype} Cost", f"{ptype} Gap", f"{ptype} Time", f"{ptype} Feas"])
    lines = [
        f"**{title}**",
        "",
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] + ["---:"] * (len(header) - 1)) + "|",
    ]
    for method in methods:
        row = [METHOD_LABELS.get(method, method)]
        emitted = False
        for ptype in ordered_types:
            result = type_map[ptype]
            available = [
                result["methods"][name]
                for name in methods
                if name in result["methods"] and _safe_float(result["methods"][name].get("avg_cost")) is not None
            ]
            ref_cost = min((_safe_float(item.get("avg_cost")) for item in available), default=None)
            payload = result["methods"].get(method)
            if payload is None:
                row.extend(["-", "-", "-", "-"])
                continue
            emitted = True
            row.extend(
                [
                    _fmt_value(payload.get("avg_cost")),
                    _fmt_gap(payload.get("avg_cost"), ref_cost),
                    _fmt_time(_method_time(method, payload)),
                    _fmt_value(payload.get("feasible_rate"), 3),
                ]
            )
        if emitted:
            lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Build one cross-type table from multiple experiment result JSON files."
    )
    parser.add_argument("json_paths", nargs="+")
    parser.add_argument("--methods", default="")
    parser.add_argument("--title", default="By-type experiment results")
    parser.add_argument("--format", choices=["detailed", "compact"], default="detailed")
    parser.add_argument("--include-stats", action="store_true")
    parser.add_argument("--output-md", default="")
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()

    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    paths = _expand_paths(args.json_paths)
    results, selected_methods = _collect(paths, methods)
    if not results:
        raise SystemExit("No experiment result JSON files found.")
    if args.format == "compact":
        table = _build_compact_table(results, selected_methods, args.title)
    else:
        table = _build_detailed_table(
            results,
            selected_methods,
            args.title,
            include_stats=args.include_stats,
        )
    print(table)

    if args.output_md:
        out_dir = os.path.dirname(os.path.abspath(args.output_md))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.output_md, "w", encoding="utf-8") as f:
            f.write(table + "\n")
    if args.output_json:
        out_dir = os.path.dirname(os.path.abspath(args.output_json))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "methods": selected_methods,
                    "results": results,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )


if __name__ == "__main__":
    main()
