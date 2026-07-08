import csv
import json
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
QUALITY_ROOT = ROOT / "tables" / "quality_protocol_random_paired"
OUT_DIR = ROOT / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PREFIX = OUT_DIR / "Fig11_quality_protocol_performance"
OUT_CSV = OUT_DIR / "Fig11_quality_protocol_performance_sources.csv"

SCALES = [20, 50, 100]
TYPES = ["R", "C", "RC"]
METHODS = [
    ("alns", "PC-EGA-POMO+ALNS"),
    ("drl", "PC-EGA-POMO"),
    ("gurobi", "Gurobi Incumbent"),
    ("ortools_physical", "OR-Tools"),
    ("pyvrp", "PyVRP"),
]

COLORS = {
    "alns": "#1f4e79",
    "drl": "#7faed0",
    "gurobi": "#4c4c4c",
    "ortools_physical": "#8f8f8f",
    "pyvrp": "#c9c9c9",
}
HATCHES = {
    "alns": "",
    "drl": "",
    "gurobi": "",
    "ortools_physical": "",
    "pyvrp": "",
}


def configure_matplotlib():
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 10,
            "axes.labelsize": 11,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#222222",
            "axes.linewidth": 0.8,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
        }
    )


def load_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def finite(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def mean(values):
    clean = [float(v) for v in values if finite(v) is not None]
    return sum(clean) / len(clean) if clean else None


def classical_path(scale, ptype):
    return QUALITY_ROOT / f"n{scale}" / f"n{scale}_{ptype.lower()}_random_seed42_5_classical.json"


def gurobi_path(scale, ptype):
    return QUALITY_ROOT / f"n{scale}" / f"n{scale}_{ptype.lower()}_random_seed42_5_gurobi.json"


def load_classical_payloads(scale, ptype):
    path = classical_path(scale, ptype)
    obj = load_json(path)
    payloads = {}
    summary = obj.get("summary", {})
    for method in ["alns", "drl", "ortools_physical", "pyvrp"]:
        payload = summary.get(method, {})
        time_s = payload.get("avg_total_time_s", payload.get("avg_time_s"))
        payloads[method] = {
            "cost": finite(payload.get("avg_cost")),
            "phys": finite(payload.get("avg_phys")),
            "penalty": finite(payload.get("avg_pen")),
            "unvisit": finite(payload.get("avg_unvisit")),
            "feasible_rate": finite(payload.get("feasible_rate")),
            "time_s": finite(time_s),
            "source": str(path.relative_to(ROOT)),
            "oom_no_incumbent": False,
        }
    return payloads


def load_gurobi_payload(scale, ptype):
    path = gurobi_path(scale, ptype)
    obj = load_json(path)
    rows = obj.get("rows", [])
    summary = obj.get("summary", {})
    costs = [row.get("gurobi_cost_env") for row in rows]
    phys = [row.get("gurobi_phys_env") for row in rows]
    penalties = [row.get("gurobi_pen_env") for row in rows]
    unvisits = [row.get("gurobi_unvisit_env") for row in rows]
    feasible_values = [float(bool(row.get("gurobi_feasible_env"))) for row in rows]
    avg_cost = mean(costs)
    feasible_rate = mean(feasible_values)
    if feasible_rate is None:
        feasible_rate = finite(summary.get("gurobi_feasible_rate_env"))
    time_s = finite(summary.get("avg_gurobi_time_s"))
    oom_no_incumbent = avg_cost is None and (feasible_rate == 0.0 or scale == 100)
    return {
        "cost": avg_cost,
        "phys": mean(phys),
        "penalty": mean(penalties),
        "unvisit": mean(unvisits),
        "feasible_rate": feasible_rate,
        "time_s": time_s,
        "source": str(path.relative_to(ROOT)),
        "oom_no_incumbent": oom_no_incumbent,
    }


def build_rows():
    rows = []
    for scale in SCALES:
        for ptype in TYPES:
            payloads = load_classical_payloads(scale, ptype)
            payloads["gurobi"] = load_gurobi_payload(scale, ptype)
            finite_costs = [p["cost"] for p in payloads.values() if p["cost"] is not None]
            best = min(finite_costs)
            for method, label in METHODS:
                payload = payloads[method]
                cost = payload["cost"]
                gap_pct = (cost - best) / best * 100.0 if cost is not None and best > 0 else None
                rows.append(
                    {
                        "scale": f"N{scale}",
                        "scale_int": scale,
                        "type": ptype,
                        "group": f"N{scale}-{ptype}",
                        "method": method,
                        "method_label": label,
                        "cost": cost,
                        "gap_pct": gap_pct,
                        "phys": payload["phys"],
                        "penalty": payload["penalty"],
                        "unvisit": payload["unvisit"],
                        "feasible_rate": payload["feasible_rate"],
                        "time_s": payload["time_s"],
                        "oom_no_incumbent": payload["oom_no_incumbent"],
                        "source": payload["source"],
                    }
                )
    return rows


def save_source(rows):
    with OUT_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        fieldnames = [
            "scale",
            "type",
            "group",
            "method",
            "method_label",
            "cost",
            "gap_pct",
            "phys",
            "penalty",
            "unvisit",
            "feasible_rate",
            "time_s",
            "oom_no_incumbent",
            "source",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def matrix(rows, key):
    groups = [f"N{scale}-{ptype}" for scale in SCALES for ptype in TYPES]
    result = {method: [] for method, _ in METHODS}
    by_key = {(row["group"], row["method"]): row for row in rows}
    for group in groups:
        for method, _ in METHODS:
            value = by_key.get((group, method), {}).get(key)
            result[method].append(np.nan if value is None else float(value))
    return groups, result


def oom_positions(rows):
    groups = [f"N{scale}-{ptype}" for scale in SCALES for ptype in TYPES]
    positions = []
    by_key = {(row["group"], row["method"]): row for row in rows}
    for group_idx, group in enumerate(groups):
        for method_idx, (method, _label) in enumerate(METHODS):
            row = by_key.get((group, method), {})
            if row.get("oom_no_incumbent"):
                positions.append((group_idx, method_idx, row))
    return positions


def style_axes(ax, tag, ylabel):
    ax.set_ylabel(ylabel, fontsize=11)
    ax.text(
        -0.045,
        1.04,
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


def plot_grouped(ax, groups, values, ylabel, tag, ylim=None, log=False):
    x = np.arange(len(groups))
    width = 0.145
    offsets = np.linspace(-2 * width, 2 * width, len(METHODS))
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
    ax.set_xticklabels(groups, rotation=35, ha="right", fontsize=8.5)
    ax.margins(x=0.015)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if log:
        ax.set_yscale("log")
        ax.set_ylim(log_baseline, 10000)
        ax.set_yticks([0.1, 1, 10, 100, 1000, 5000])
        ax.get_yaxis().set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    style_axes(ax, tag, ylabel)


def annotate_oom(ax, rows, y):
    width = 0.145
    offsets = np.linspace(-2 * width, 2 * width, len(METHODS))
    for group_idx, method_idx, _row in oom_positions(rows):
        ax.text(
            group_idx + offsets[method_idx],
            y,
            "OOM",
            rotation=90,
            ha="center",
            va="bottom",
            fontsize=7.8,
            color="#333333",
        )


def main():
    configure_matplotlib()
    rows = build_rows()
    save_source(rows)
    groups, gap_values = matrix(rows, "gap_pct")
    _, feas_values = matrix(rows, "feasible_rate")
    _, time_values = matrix(rows, "time_s")

    finite_gaps = [row["gap_pct"] for row in rows if row["gap_pct"] is not None]
    gap_top = max(100.0, min(max(finite_gaps) * 1.08, 480.0))

    fig, axes = plt.subplots(3, 1, figsize=(13.6, 7.75), sharex=False)
    plot_grouped(axes[0], groups, gap_values, "Cost Gap (%)", "(a)", ylim=(0, gap_top))
    annotate_oom(axes[0], rows, gap_top * 0.82)
    plot_grouped(axes[1], groups, feas_values, "Feasible Rate", "(b)", ylim=(0, 1.08))
    plot_grouped(axes[2], groups, time_values, "Runtime (s, log scale)", "(c)", log=True)
    axes[2].set_xlabel("Instance group", fontsize=11)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, 0.99),
        handlelength=1.25,
        columnspacing=1.25,
        borderaxespad=0.0,
    )
    fig.tight_layout(rect=(0.022, 0.025, 0.995, 0.905), h_pad=0.82)

    for ext in ["png", "svg", "pdf", "tiff"]:
        path = OUT_PREFIX.with_suffix(f".{ext}")
        kwargs = {}
        if ext in {"png", "tiff"}:
            kwargs["dpi"] = 600
        fig.savefig(path, **kwargs)
        print(f"Saved {path}")
    print(f"Saved {OUT_CSV}")


if __name__ == "__main__":
    main()
