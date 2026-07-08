import argparse
import glob
import json
import math
from pathlib import Path


VARIANT_LABELS = {
    "film": "FiLM-only",
    "full": "Full PC-EGA-POMO",
    "no_film": "w/o FiLM",
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


def fmt_time(value):
    if value is None:
        return "-"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "-"
    if not math.isfinite(value):
        return "-"
    if value < 1:
        return "<1s"
    return f"{value:.1f}s"


def action_fast_ratio(actions, n_nodes):
    if not actions:
        return None
    customer_actions = [int(a) for a in actions if int(a) < 2 * n_nodes]
    if not customer_actions:
        return None
    return sum(1 for action in customer_actions if action % 2 == 1) / len(customer_actions)


def row_fast_ratio(rows, method, n_nodes):
    ratios = []
    for row in rows:
        payload = row.get(method, {})
        actions = payload.get("actions") or payload.get("route_actions")
        ratio = action_fast_ratio(actions, n_nodes)
        if ratio is not None:
            ratios.append(ratio)
    return sum(ratios) / len(ratios) if ratios else None


def infer_variant(path, obj):
    name = Path(path).as_posix().lower()
    config_variant = str(obj.get("config", {}).get("variant", "")).lower()
    for variant in ["no_film", "film", "full"]:
        if config_variant == variant or f"_{variant}_" in name or f"/{variant}/" in name:
            return variant
    return config_variant or "film"


def summarize_file(path, methods):
    obj = load(path)
    config = obj.get("config", {})
    pref = config.get("pref") or [config.get("pref_alpha"), config.get("pref_beta")]
    n_nodes = int(config.get("n_nodes", 20))
    ptype = str(config.get("problem_type", "")).upper()
    variant = infer_variant(path, obj)
    rows = obj.get("rows", [])
    summary = obj.get("summary", {})
    out = []
    for method in methods:
        payload = summary.get(method)
        if not isinstance(payload, dict):
            continue
        out.append(
            {
                "type": ptype,
                "variant": variant,
                "variant_label": VARIANT_LABELS.get(variant, variant),
                "pref_time": pref[0] if pref else None,
                "pref_energy": pref[1] if pref and len(pref) > 1 else None,
                "method": method,
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
        "Variant",
        "Pref(time,energy)",
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
                    row["variant_label"],
                    f"({fmt(row['pref_time'], 1)}, {fmt(row['pref_energy'], 1)})",
                    fmt(row["cost"]),
                    fmt(row["phys"]),
                    fmt(row["penalty"]),
                    fmt(row["unvisit"]),
                    fmt(row["feasible"], 3),
                    fmt(row["fast_ratio"], 3),
                    fmt_time(row["time_s"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("json_paths", nargs="+")
    parser.add_argument("--methods", default="drl")
    parser.add_argument("--output-md", default="tables/n20_film_response/film_response_summary.md")
    parser.add_argument("--output-json", default="tables/n20_film_response/film_response_summary.json")
    args = parser.parse_args()

    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    paths = []
    for pattern in args.json_paths:
        matches = glob.glob(pattern)
        paths.extend(matches if matches else [pattern])

    rows = []
    for path in paths:
        obj = load(path)
        if not isinstance(obj, dict) or "summary" not in obj or "config" not in obj:
            print(f"Skip non-experiment JSON: {path}")
            continue
        rows.extend(summarize_file(path, methods))

    rows.sort(
        key=lambda row: (
            {"R": 0, "C": 1, "RC": 2}.get(str(row.get("type", "")).upper(), 99),
            str(row.get("variant", "")),
            float(row.get("pref_time") or 0.0),
        )
    )

    output_md = Path(args.output_md)
    output_json = Path(args.output_json)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("# FiLM preference response summary\n\n" + md_table(rows) + "\n", encoding="utf-8")
    output_json.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {output_md}")
    print(f"Saved {output_json}")


if __name__ == "__main__":
    main()
