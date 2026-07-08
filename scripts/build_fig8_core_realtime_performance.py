import csv
import json
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "figures"
OUT_DIR.mkdir(exist_ok=True)

SCALES = [20, 50, 100]
TYPES = ["R", "C", "RC"]
METHODS = [
    ("alns", "PC-EGA-POMO+ALNS"),
    ("drl", "PC-EGA-POMO"),
    ("ortools_physical", "OR-Tools"),
    ("pyvrp", "PyVRP"),
]
COLORS = {
    "alns": "#1f4e79",
    "drl": "#7faed0",
    "ortools_physical": "#7f7f7f",
    "pyvrp": "#c9c9c9",
}
HATCHES = {
    "alns": "",
    "drl": "",
    "ortools_physical": "",
    "pyvrp": "",
}


def load_summary(scale):
    path = ROOT / "tables" / f"n{scale}_by_type_random" / f"n{scale}_by_type_summary.json"
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    return {
        str(result.get("problem_type", "")).upper(): result.get("methods", {})
        for result in obj.get("results", [])
    }


def finite(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def build_rows():
    rows = []
    for scale in SCALES:
        by_type = load_summary(scale)
        for ptype in TYPES:
            methods = by_type.get(ptype, {})
            costs = [
                finite(methods.get(method, {}).get("avg_cost"))
                for method, _ in METHODS
            ]
            best = min(value for value in costs if value is not None)
            for method, label in METHODS:
                payload = methods.get(method, {})
                cost = finite(payload.get("avg_cost"))
                gap = (cost - best) / best * 100.0 if cost is not None and best > 0 else None
                time_s = payload.get("avg_total_time_s")
                if time_s is None:
                    time_s = payload.get("avg_time_s")
                rows.append(
                    {
                        "scale": f"N{scale}",
                        "type": ptype,
                        "group": f"N{scale}-{ptype}",
                        "method": method,
                        "method_label": label,
                        "cost": cost,
                        "gap_pct": gap,
                        "feasible_rate": finite(payload.get("feasible_rate")),
                        "time_s": finite(time_s),
                        "unvisit": finite(payload.get("avg_unvisit")),
                    }
                )
    return rows


def matrix(rows, key):
    groups = [f"N{scale}-{ptype}" for scale in SCALES for ptype in TYPES]
    result = {method: [] for method, _ in METHODS}
    by_group_method = {(row["group"], row["method"]): row for row in rows}
    for group in groups:
        for method, _ in METHODS:
            value = by_group_method.get((group, method), {}).get(key)
            result[method].append(np.nan if value is None else value)
    return groups, result


def style_axes(ax, tag, ylabel):
    ax.set_ylabel(ylabel, fontsize=11)
    ax.text(
        -0.065,
        1.035,
        tag,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
        ha="left",
    )
    # Word's SVG renderer can draw transparent grid lines above bars.
    # Keep these bar panels grid-free for reliable manuscript insertion.
    ax.grid(False)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#222222")
    ax.spines["bottom"].set_color("#222222")
    ax.spines["left"].set_linewidth(0.75)
    ax.spines["bottom"].set_linewidth(0.75)
    ax.tick_params(axis="both", width=0.8, length=3.2, color="#222222")


def plot_grouped_bars(ax, groups, values, ylabel, tag, ylim=None, log=False):
    x = np.arange(len(groups))
    width = 0.18
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(METHODS))
    log_baseline = 0.08
    for offset, (method, label) in zip(offsets, METHODS):
        vals = np.array(values[method], dtype=float)
        bar_kwargs = {
            "width": width,
            "label": label,
            "color": COLORS[method],
            "edgecolor": "#4a4a4a",
            "linewidth": 0.22,
            "hatch": HATCHES[method],
            "clip_on": True,
            "zorder": 2,
        }
        if log:
            heights = np.maximum(vals - log_baseline, 1e-6)
            heights[~np.isfinite(vals)] = np.nan
            ax.bar(x + offset, heights, bottom=log_baseline, **bar_kwargs)
        else:
            ax.bar(x + offset, vals, **bar_kwargs)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=35, ha="right", fontsize=9)
    ax.margins(x=0.015)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if log:
        ax.set_yscale("log")
        ax.set_ylim(log_baseline, 25)
        ax.set_yticks([0.1, 0.3, 1, 3, 10, 20])
        ax.get_yaxis().set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    style_axes(ax, tag, ylabel)


def save_source(rows):
    out = OUT_DIR / "Fig8_core_realtime_performance_sources.csv"
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "scale",
                "type",
                "group",
                "method_label",
                "cost",
                "gap_pct",
                "feasible_rate",
                "time_s",
                "unvisit",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in writer.fieldnames})


def main():
    rows = build_rows()
    save_source(rows)

    mpl.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
        }
    )

    groups, gap_values = matrix(rows, "gap_pct")
    _, feasible_values = matrix(rows, "feasible_rate")
    _, time_values = matrix(rows, "time_s")

    fig, axes = plt.subplots(3, 1, figsize=(13.2, 7.5), sharex=False)
    plot_grouped_bars(axes[0], groups, gap_values, "Cost Gap (%)", "(a)", ylim=(0, 390))
    plot_grouped_bars(axes[1], groups, feasible_values, "Feasible Rate", "(b)", ylim=(0, 1.08))
    plot_grouped_bars(axes[2], groups, time_values, "Runtime (s, log scale)", "(c)", log=True)
    axes[2].set_xlabel("Instance group", fontsize=11)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.99),
        handlelength=1.25,
        columnspacing=1.5,
        borderaxespad=0.0,
    )
    fig.tight_layout(rect=(0.025, 0.025, 0.995, 0.905), h_pad=0.82)

    for ext in ["png", "svg", "pdf", "tiff"]:
        path = OUT_DIR / f"Fig8_core_realtime_performance.{ext}"
        kwargs = {"bbox_inches": "tight"}
        if ext in {"png", "tiff"}:
            kwargs["dpi"] = 600
        fig.savefig(path, **kwargs)
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
