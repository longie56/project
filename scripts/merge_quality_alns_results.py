import argparse
import json
import math
from pathlib import Path


def load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n")


def close_list(a, b, tol=1e-6):
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        try:
            if abs(float(x) - float(y)) > tol:
                return False
        except (TypeError, ValueError):
            return False
    return True


def validate_same_instances(base, update, base_path, update_path):
    base_cfg = base.get("config", {})
    update_cfg = update.get("config", {})
    keys = ["instances", "n_nodes", "problem_type", "pref_mode", "pref_random_low", "pref_random_high"]
    mismatches = []
    for key in keys:
        if str(base_cfg.get(key)) != str(update_cfg.get(key)):
            mismatches.append(f"{key}: {base_cfg.get(key)!r} != {update_cfg.get(key)!r}")
    if mismatches:
        raise ValueError(
            f"Config mismatch between {base_path} and {update_path}: " + "; ".join(mismatches)
        )

    base_rows = base.get("rows", [])
    update_rows = update.get("rows", [])
    if len(base_rows) != len(update_rows):
        raise ValueError(f"Row count mismatch: {len(base_rows)} != {len(update_rows)}")

    for idx, (base_row, update_row) in enumerate(zip(base_rows, update_rows)):
        if int(base_row.get("instance", -1)) != int(update_row.get("instance", -2)):
            raise ValueError(f"Instance id mismatch at row {idx}.")
        if not close_list(base_row.get("pref", []), update_row.get("pref", [])):
            raise ValueError(f"Preference vector mismatch at row {idx}.")
        if "alns" not in update_row:
            raise ValueError(f"Update row {idx} does not contain an alns payload.")
    if "alns" not in update.get("summary", {}):
        raise ValueError("Update summary does not contain an alns payload.")


def merge_alns(base, update, note):
    base["summary"]["alns"] = update["summary"]["alns"]
    for base_row, update_row in zip(base["rows"], update["rows"]):
        base_row["alns"] = update_row["alns"]
    base.setdefault("config", {})["alns_quality_update"] = {
        "source": str(update.get("config", {}).get("output_json", "")),
        "alns_time_limit": update.get("config", {}).get("alns_time_limit"),
        "alns_repair_mode": update.get("config", {}).get("alns_repair_mode"),
        "alns_restart_rounds": update.get("config", {}).get("alns_restart_rounds"),
        "note": note,
    }
    return base


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-json", required=True)
    parser.add_argument("--alns-json", required=True)
    parser.add_argument("--output-json", default="")
    parser.add_argument(
        "--note",
        default="Updated PC-EGA-POMO+ALNS only for the long-budget quality protocol.",
    )
    args = parser.parse_args()

    base_path = Path(args.base_json)
    update_path = Path(args.alns_json)
    output_path = Path(args.output_json) if args.output_json else base_path

    base = load(base_path)
    update = load(update_path)
    validate_same_instances(base, update, base_path, update_path)
    old_cost = base.get("summary", {}).get("alns", {}).get("avg_cost")
    new_cost = update.get("summary", {}).get("alns", {}).get("avg_cost")
    merged = merge_alns(base, update, args.note)
    save(output_path, merged)

    print(f"Saved {output_path}")
    if old_cost is not None and new_cost is not None and math.isfinite(float(old_cost)):
        delta = (float(old_cost) - float(new_cost)) / float(old_cost) * 100.0
        print(f"ALNS avg_cost: {float(old_cost):.4f} -> {float(new_cost):.4f} ({delta:.2f}% improvement)")


if __name__ == "__main__":
    main()
