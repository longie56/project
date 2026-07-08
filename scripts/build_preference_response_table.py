import argparse
import glob
import json
import math
from pathlib import Path


METHOD_LABELS = {
    "drl": "PC-EGA-POMO",
    "alns": "PC-EGA-POMO+ALNS",
}


def load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def fmt(value, digits=4):
    if value is None:
        return "-"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "-"
    if not math.isfinite(value):
        return "-"
    return f"{value:.{digits}f}"


def action_fast_ratio(actions, n_nodes):
    if not actions:
        return None
    customer_actions = [int(a) for a in actions if int(a) < 2 * n_nodes]
    if not customer_actions:
        return None
    fast = sum(1 for a in customer_actions if a % 2 == 1)
    return fast / len(customer_actions)


def row_fast_ratio(rows, method, n_nodes):
    ratios = []
    for row in rows:
        payload = row.get(method, {})
        actions = payload.get("actions") or payload.get("route_actions")
        ratio = action_fast_ratio(actions, n_nodes)
        if ratio is not None:
            ratios.append(ratio)
    return sum(ratios) / len(ratios) if ratios else None


def summarize_file(path, methods):
    obj = load(path)
    config = obj.get("config", {})
    pref = config.get("pref") or [config.get("pref_alpha"), config.get("pref_beta")]
    n_nodes = int(config.get("n_nodes", 20))
    ptype = str(config.get("problem_type", "")).upper()
    summary = obj.get("summary", {})
    rows = obj.get("rows", [])
    out = []
    for method in methods:
        payload = summary.get(method)
        if not isinstance(payload, dict):
            continue
        out.append(
            {
                "type": ptype,
                "pref_time": pref[0] if pref else None,
                "pref_energy": pref[1] if pref and len(pref) > 1 else None,
                "method": method,
                "method_label": METHOD_LABELS.get(method, method),
                "cost": payload.get("avg_cost"),
                "phys": payload.get("avg_phys"),
                "penalty": payload.get("avg_pen"),
                "unvisit": payload.get("avg_unvisit"),
                "feasible": payload.get("feasible_rate"),
                "time_s": payload.get("avg_total_time_s", payload.get("avg_time_s")),
                "fast_ratio": row_fast_ratio(rows, method, n_nodes),
            }
        )
    return out


def md_table(rows):
    headers = [
        "Type",
        "Pref(time,energy)",
        "Method",
        "Cost",
        "Phys",
        "Penalty",
        "Unvisit",
        "Feasible",
        "Fast ratio",
        "Time",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["type"],
                    f"({fmt(row['pref_time'], 1)}, {fmt(row['pref_energy'], 1)})",
                    row["method_label"],
                    fmt(row["cost"]),
                    fmt(row["phys"]),
                    fmt(row["penalty"]),
                    fmt(row["unvisit"]),
                    fmt(row["feasible"], 3),
                    fmt(row["fast_ratio"], 3),
                    (f"{float(row['time_s']):.1f}s" if row["time_s"] is not None and float(row["time_s"]) >= 1 else "<1s"),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("json_paths", nargs="+")
    parser.add_argument("--methods", default="drl,alns")
    parser.add_argument("--output-md", default="tables/n20_preference_response/preference_response_summary.md")
    parser.add_argument("--output-json", default="tables/n20_preference_response/preference_response_summary.json")
    args = parser.parse_args()

    methods = [x.strip() for x in args.methods.split(",") if x.strip()]
    paths = []
    for pattern in args.json_paths:
        matches = glob.glob(pattern)
        paths.extend(matches if matches else [pattern])

    rows = []
    for path in paths:
        rows.extend(summarize_file(path, methods))
    rows.sort(key=lambda r: (r["type"], float(r["pref_time"] or 0), r["method"]))

    output_md = Path(args.output_md)
    output_json = Path(args.output_json)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("# Preference response summary\n\n" + md_table(rows) + "\n", encoding="utf-8")
    output_json.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {output_md}")
    print(f"Saved {output_json}")


if __name__ == "__main__":
    main()
