import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "figures"

# Keep the original prefix for compatibility with the manuscript, and also
# export a clearer ablation-only filename.
OUT_PREFIXES = [
    OUT_DIR / "Fig7_ablation_preference",
    OUT_DIR / "Fig7_ablation_study",
]
OUT_CSVS = [
    OUT_DIR / "Fig7_ablation_preference_sources.csv",
    OUT_DIR / "Fig7_ablation_study_sources.csv",
]

ABLATION_JSON = ROOT / "tables/n20_ablation/ablation_summary.json"

FULL_BLUE = "#1f4e79"
GRAY_DARK = "#606060"
GRAY_MID = "#858585"
GRAY_LIGHT = "#adadad"
GRID = "#eeeeee"

ABLATION_ORDER = [
    ("full", "Full model\n(PC-EGA-POMO)"),
    ("no_film", "w/o FiLM"),
    ("no_action_features", "w/o physical\nfeatures"),
    ("film", "w/o EGA"),
    ("ega", "w/o speed\nbias"),
]

ABLATION_COLORS = {
    "full": FULL_BLUE,
    "no_film": GRAY_LIGHT,
    "no_action_features": "#9f9f9f",
    "film": GRAY_MID,
    "ega": GRAY_DARK,
}


def load_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def finite_mean(values):
    clean = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return sum(clean) / len(clean) if clean else None


def configure_matplotlib():
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "black",
            "axes.linewidth": 0.8,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 3.5,
            "ytick.major.size": 3.5,
            "legend.frameon": False,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_ablation():
    rows = load_json(ABLATION_JSON)
    grouped = defaultdict(lambda: {"cost": [], "feasible": []})
    for row in rows:
        if row.get("method") != "drl":
            continue
        variant = str(row.get("variant", "")).lower()
        grouped[variant]["cost"].append(row.get("cost"))
        grouped[variant]["feasible"].append(row.get("feasible"))

    records = []
    for variant, label in ABLATION_ORDER:
        payload = grouped.get(variant, {})
        records.append(
            {
                "variant": variant,
                "label": label,
                "cost": finite_mean(payload.get("cost", [])),
                "feasible": finite_mean(payload.get("feasible", [])),
                "available": bool(payload),
            }
        )

    full_cost = next((row["cost"] for row in records if row["variant"] == "full"), None)
    for row in records:
        if full_cost and row["cost"] is not None:
            row["delta_pct"] = (row["cost"] - full_cost) / full_cost * 100.0
        else:
            row["delta_pct"] = None
    return records


def plot(records):
    configure_matplotlib()
    import matplotlib.pyplot as plt

    labels = [row["label"] for row in records]
    costs = np.array([row["cost"] if row["cost"] is not None else np.nan for row in records], dtype=float)
    y = np.arange(len(records))
    colors = [ABLATION_COLORS[row["variant"]] for row in records]

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    bars = ax.barh(
        y,
        costs,
        height=0.62,
        color=colors,
        edgecolor="#333333",
        linewidth=0.45,
    )

    full_cost = records[0]["cost"]
    if full_cost is not None:
        ax.axvline(full_cost, color=FULL_BLUE, linestyle="--", linewidth=1.0, alpha=0.65)

    xmax = float(np.nanmax(costs)) * 1.28
    for row, bar in zip(records, bars):
        x = bar.get_width()
        if not row["available"] or not math.isfinite(float(x)):
            continue
        delta = row["delta_pct"]
        delta_text = "0.0%" if abs(delta or 0.0) < 0.05 else f"+{delta:.1f}%"
        ax.text(
            x + xmax * 0.012,
            bar.get_y() + bar.get_height() / 2,
            f"{x:.2f}  ({delta_text})",
            ha="left",
            va="center",
            fontsize=9.2,
            color="#222222",
        )

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9.6)
    ax.invert_yaxis()
    ax.set_xlabel("Average Cost", fontsize=12)
    ax.set_xlim(0, xmax)
    ax.tick_params(axis="x", labelsize=10)
    ax.grid(True, axis="x", color=GRID, linewidth=0.5, alpha=0.75)
    ax.set_axisbelow(True)
    # The paired source CSV keeps feasible-rate values. The visual focuses on
    # cost degradation because all tested variants remained feasible.
    fig.tight_layout(rect=(0.03, 0.02, 0.98, 0.98))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for prefix in OUT_PREFIXES:
        fig.savefig(prefix.with_suffix(".png"), dpi=600)
        fig.savefig(prefix.with_suffix(".svg"))
        fig.savefig(prefix.with_suffix(".pdf"))
        fig.savefig(prefix.with_suffix(".tiff"), dpi=600)
    plt.close(fig)


def write_sources(records):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for out_csv in OUT_CSVS:
        with out_csv.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["variant", "label", "cost", "delta_pct_vs_full", "feasible", "source"])
            for row in records:
                writer.writerow(
                    [
                        row["variant"],
                        row["label"].replace("\n", " "),
                        row["cost"],
                        row["delta_pct"],
                        row["feasible"],
                        ABLATION_JSON,
                    ]
                )


def main():
    records = load_ablation()
    plot(records)
    write_sources(records)
    for prefix in OUT_PREFIXES:
        print(f"Saved {prefix.with_suffix('.png')}")
        print(f"Saved {prefix.with_suffix('.svg')}")
        print(f"Saved {prefix.with_suffix('.pdf')}")
        print(f"Saved {prefix.with_suffix('.tiff')}")
    for out_csv in OUT_CSVS:
        print(f"Saved {out_csv}")


if __name__ == "__main__":
    main()
