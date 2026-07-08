import argparse
import glob
import json
import math
from pathlib import Path


VARIANT_LABELS = {
    "film": "w/o EGA and mode mask",
    "ega": "w/o mode mask",
    "no_film": "w/o FiLM",
    "no_action_features": "w/o action physical features",
    "full": "Full PC-EGA-POMO",
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


def infer_variant(path, obj):
    name = Path(path).as_posix().lower()
    for variant in ["no_action_features", "no_film", "film", "ega", "full"]:
        if f"/{variant}/" in name or f"_{variant}_" in name or name.endswith(f"{variant}.json"):
            return variant
    checkpoint = str(obj.get("config", {}).get("checkpoint", "")).lower()
    for variant in ["no_action_features", "no_film", "film", "ega", "full"]:
        if variant in checkpoint:
            return variant
    return Path(path).parent.name


def summarize(path, methods):
    obj = load(path)
    if not isinstance(obj, dict):
        print(f"Skip non-experiment JSON: {path}")
        return []
    if "summary" not in obj or "config" not in obj:
        print(f"Skip JSON without experiment summary/config: {path}")
        return []
    variant = infer_variant(path, obj)
    ptype = str(obj.get("config", {}).get("problem_type", "")).upper()
    rows = []
    summary = obj.get("summary", {})
    for method in methods:
        payload = summary.get(method)
        if not isinstance(payload, dict):
            continue
        rows.append(
            {
                "variant": variant,
                "variant_label": VARIANT_LABELS.get(variant, variant),
                "type": ptype,
                "method": method,
                "cost": payload.get("avg_cost"),
                "phys": payload.get("avg_phys"),
                "penalty": payload.get("avg_pen"),
                "unvisit": payload.get("avg_unvisit"),
                "feasible": payload.get("feasible_rate"),
                "time_s": payload.get("avg_total_time_s", payload.get("avg_time_s")),
            }
        )
    return rows


def add_gaps(rows):
    best_by_type_method = {}
    for row in rows:
        key = (row["type"], row["method"])
        cost = row["cost"]
        if cost is None:
            continue
        best_by_type_method[key] = min(float(cost), best_by_type_method.get(key, float("inf")))
    for row in rows:
        best = best_by_type_method.get((row["type"], row["method"]))
        if not best or not row["cost"]:
            row["gap"] = None
        else:
            row["gap"] = (float(row["cost"]) - best) / best * 100.0


def md_table(rows):
    headers = ["Type", "Variant", "Method", "Cost", "Gap", "Phys", "Penalty", "Unvisit", "Feasible", "Time"]
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
                    row["method"],
                    fmt(row["cost"]),
                    fmt(row["gap"], 2) + "%" if row["gap"] is not None else "-",
                    fmt(row["phys"]),
                    fmt(row["penalty"]),
                    fmt(row["unvisit"]),
                    fmt(row["feasible"], 3),
                    f"{float(row['time_s']):.1f}s" if row["time_s"] is not None and float(row["time_s"]) >= 1 else "<1s",
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("json_paths", nargs="+")
    parser.add_argument("--methods", default="drl,alns")
    parser.add_argument("--output-md", default="tables/n20_ablation/ablation_summary.md")
    parser.add_argument("--output-json", default="tables/n20_ablation/ablation_summary.json")
    args = parser.parse_args()

    methods = [x.strip() for x in args.methods.split(",") if x.strip()]
    paths = []
    for pattern in args.json_paths:
        matches = glob.glob(pattern)
        paths.extend(matches if matches else [pattern])

    rows = []
    for path in paths:
        if Path(path).name.startswith("smoke_"):
            print(f"Skip smoke JSON: {path}")
            continue
        rows.extend(summarize(path, methods))
    add_gaps(rows)
    variant_order = {"film": 0, "ega": 1, "no_film": 2, "no_action_features": 3, "full": 4}
    rows.sort(key=lambda r: (r["type"], r["method"], variant_order.get(r["variant"], 99)))

    output_md = Path(args.output_md)
    output_json = Path(args.output_json)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("# Ablation summary\n\n" + md_table(rows) + "\n", encoding="utf-8")
    output_json.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {output_md}")
    print(f"Saved {output_json}")


if __name__ == "__main__":
    main()
