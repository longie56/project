import csv
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import E_MAX, GRID_SCALE, MAX_PAYLOAD_KG, MAX_TIME, P_EMPTY_FAST, P_EMPTY_NORM, P_GAMMA, P_HOVER, V_FAST, V_NORM
from data.solomon_generator import SolomonGenerator

DATA_DIR = ROOT / "tables" / "n20_preference_pareto_11pt"
OUT_DIR = ROOT / "figures"
OUT_PNG = OUT_DIR / "Fig9_preference_pareto_front.png"
OUT_SVG = OUT_DIR / "Fig9_preference_pareto_front.svg"
OUT_PDF = OUT_DIR / "Fig9_preference_pareto_front.pdf"
OUT_TIFF = OUT_DIR / "Fig9_preference_pareto_front.tiff"
OUT_CSV = OUT_DIR / "Fig9_preference_pareto_front_sources.csv"

FULL_BLUE = "#1f4e79"
MID_BLUE = "#5d7fa3"
LIGHT_GRAY = "#d9d9d9"
GRID = "#eeeeee"


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
            "legend.frameon": False,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def finite_mean(values):
    clean = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return sum(clean) / len(clean) if clean else None


def regenerate_instances(config):
    seed = int(config.get("seed", 42))
    instances = int(config.get("instances", 100))
    n_nodes = int(config.get("n_nodes", 20))
    ptype = str(config.get("problem_type", "R")).upper()
    tightness = float(config.get("tightness", 0.7))

    torch.manual_seed(seed)
    random.seed(seed)
    nodes, depot, visited = SolomonGenerator.generate(
        batch_size=instances,
        n_nodes=n_nodes,
        problem_type=ptype,
        tightness=tightness,
        device=torch.device("cpu"),
    )
    return nodes, depot, visited


def replay_raw_metrics(nodes, depot, actions):
    n_nodes = nodes.size(0)
    pos = depot.clone()
    time = torch.tensor(0.0)
    load = torch.tensor(1.0)
    raw_energy = 0.0
    raw_penalty = 0.0
    fast_count = 0
    econ_count = 0
    visited = torch.zeros(n_nodes, dtype=torch.bool)

    for action_value in actions:
        action = int(action_value)
        is_customer = action < n_nodes * 2
        node_idx = min(action // 2, n_nodes - 1)
        is_fast = bool(action % 2 == 1)
        speed = V_FAST if is_fast else V_NORM
        p_empty = P_EMPTY_FAST if is_fast else P_EMPTY_NORM
        if is_fast:
            fast_count += 1
        else:
            econ_count += 1

        target = nodes[node_idx, :2] if is_customer else depot
        dist = torch.norm(target - pos) * GRID_SCALE
        t_flight = dist / speed
        t_norm = t_flight / MAX_TIME
        arrival = time + t_norm
        e_flight = (p_empty + P_GAMMA * load * MAX_PAYLOAD_KG) * t_flight

        if is_customer:
            ready = nodes[node_idx, 3]
            due = nodes[node_idx, 4]
            service = nodes[node_idx, 5]
            start = torch.maximum(arrival, ready)
            wait_delta = torch.relu(start - arrival)
            wait_energy = P_HOVER * wait_delta * MAX_TIME
            raw_energy += float((e_flight + wait_energy).item())
            raw_penalty += float(torch.relu(start - due).item())
            time = start + service
            load = torch.clamp(load - nodes[node_idx, 2], min=0.0)
            visited[node_idx] = True
        else:
            raw_energy += float(e_flight.item())
            time = arrival
            load = torch.tensor(1.0)

        pos = target.clone()

    return {
        "raw_energy": raw_energy,
        "raw_energy_norm": raw_energy / E_MAX,
        "raw_penalty": raw_penalty,
        "fast_count": fast_count,
        "econ_count": econ_count,
        "unvisited": int((~visited).sum().item()),
    }


def collect_records():
    records = []
    for path in sorted(DATA_DIR.glob("n20_*_pref_*.json")):
        obj = load_json(path)
        cfg = obj.get("config", {})
        rows = obj.get("rows", [])
        if not rows:
            continue
        nodes, depot, _visited = regenerate_instances(cfg)
        pref = cfg.get("pref") or [None, None]
        per_row = []
        for row in rows:
            idx = int(row.get("instance", 0))
            actions = row.get("drl", {}).get("route_actions", [])
            if not actions:
                continue
            per_row.append(replay_raw_metrics(nodes[idx], depot[idx], actions))
        if not per_row:
            continue
        records.append(
            {
                "type": str(cfg.get("problem_type", "")).upper(),
                "lambda_time": float(pref[0]),
                "lambda_energy": float(pref[1]),
                "raw_energy_norm": finite_mean([r["raw_energy_norm"] for r in per_row]),
                "raw_penalty": finite_mean([r["raw_penalty"] for r in per_row]),
                "fast_ratio": finite_mean(
                    [
                        r["fast_count"] / max(1, r["fast_count"] + r["econ_count"])
                        for r in per_row
                    ]
                ),
                "unvisited": finite_mean([r["unvisited"] for r in per_row]),
                "source": str(path.relative_to(ROOT)),
            }
        )
    if not records:
        raise FileNotFoundError(
            f"No 11-point preference JSON files found in {DATA_DIR}. "
            "Run the preference sweep first."
        )
    return records


def aggregate(records):
    grouped = defaultdict(lambda: defaultdict(list))
    for row in records:
        key = round(row["lambda_time"], 6)
        for metric in ["raw_energy_norm", "raw_penalty", "fast_ratio", "unvisited"]:
            grouped[key][metric].append(row[metric])
        grouped[key]["lambda_energy"].append(row["lambda_energy"])

    agg = []
    for key in sorted(grouped):
        payload = grouped[key]
        agg.append(
            {
                "lambda_time": key,
                "lambda_energy": finite_mean(payload["lambda_energy"]),
                "raw_energy_norm": finite_mean(payload["raw_energy_norm"]),
                "raw_penalty": finite_mean(payload["raw_penalty"]),
                "fast_ratio": finite_mean(payload["fast_ratio"]),
                "unvisited": finite_mean(payload["unvisited"]),
            }
        )
    return agg


def pareto_flags(rows):
    flags = []
    for i, row in enumerate(rows):
        dominated = False
        for j, other in enumerate(rows):
            if i == j:
                continue
            energy_ok = other["raw_energy_norm"] <= row["raw_energy_norm"]
            penalty_ok = other["raw_penalty"] <= row["raw_penalty"]
            strictly_better = (
                other["raw_energy_norm"] < row["raw_energy_norm"]
                or other["raw_penalty"] < row["raw_penalty"]
            )
            if energy_ok and penalty_ok and strictly_better:
                dominated = True
                break
        flags.append(not dominated)
    return flags


def plot(records, agg):
    configure_matplotlib()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.4, 5.35))

    ax.scatter(
        [r["raw_energy_norm"] for r in records],
        [r["raw_penalty"] for r in records],
        s=22,
        color="#b8b8b8",
        alpha=0.38,
        edgecolor="none",
        label="Type-level observations",
        zorder=1,
    )

    xs = np.array([row["raw_energy_norm"] for row in agg], dtype=float)
    ys = np.array([row["raw_penalty"] for row in agg], dtype=float)
    pareto = pareto_flags(agg)
    front = sorted([row for row, keep in zip(agg, pareto) if keep], key=lambda r: r["raw_energy_norm"])
    ax.plot(
        [row["raw_energy_norm"] for row in front],
        [row["raw_penalty"] for row in front],
        color=FULL_BLUE,
        linewidth=2.1,
        label="Pareto-efficient subset",
        zorder=3,
    )
    ax.scatter(
        xs,
        ys,
        s=40,
        color="#9fb8d4",
        edgecolor="white",
        linewidth=0.6,
        label="11 preference means",
        zorder=4,
    )
    ax.scatter(
        [row["raw_energy_norm"] for row in front],
        [row["raw_penalty"] for row in front],
        s=58,
        color=FULL_BLUE,
        edgecolor="white",
        linewidth=0.8,
        zorder=5,
    )

    for idx, row in enumerate(agg):
        if idx not in {0, 1, 5, 6, 7, 10}:
            continue
        dx = 0.014 * (max(xs) - min(xs) if len(xs) > 1 else 1.0)
        dy = 0.03 * (max(ys) - min(ys) if len(ys) > 1 else 1.0)
        if idx in {0, 1}:
            dy *= -1.4
        label = rf"$\lambda_T$={row['lambda_time']:.1f}"
        ax.annotate(
            label,
            (row["raw_energy_norm"], row["raw_penalty"]),
            xytext=(row["raw_energy_norm"] + dx, row["raw_penalty"] + dy),
            fontsize=8.8,
            color=FULL_BLUE if pareto[idx] else "#666666",
        )

    if len(front) >= 2:
        for i in range(len(front) - 1):
            ax.annotate(
                "",
                xy=(front[i + 1]["raw_energy_norm"], front[i + 1]["raw_penalty"]),
                xytext=(front[i]["raw_energy_norm"], front[i]["raw_penalty"]),
                arrowprops=dict(arrowstyle="->", color=MID_BLUE, lw=0.8, alpha=0.55),
            )

    ax.set_xlabel(r"Raw Energy Consumption ($E/E_{\max}$)", fontsize=13)
    ax.set_ylabel("Raw Tardiness Penalty", fontsize=13)
    ax.tick_params(labelsize=11)
    ax.grid(True, color=GRID, linewidth=0.5, alpha=0.75)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=9.5)
    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=600)
    fig.savefig(OUT_SVG)
    fig.savefig(OUT_PDF)
    fig.savefig(OUT_TIFF, dpi=600)
    plt.close(fig)


def write_sources(records, agg):
    flags = pareto_flags(agg)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "level",
                "type",
                "lambda_time",
                "lambda_energy",
                "raw_energy_norm",
                "raw_penalty",
                "fast_ratio",
                "unvisited",
                "source",
            ]
        )
        for row in records:
            writer.writerow(
                [
                    "type",
                    row["type"],
                    row["lambda_time"],
                    row["lambda_energy"],
                    row["raw_energy_norm"],
                    row["raw_penalty"],
                    row["fast_ratio"],
                    row["unvisited"],
                    row["source"],
                ]
            )
        for row, keep in zip(agg, flags):
            writer.writerow(
                [
                    "average",
                    "R/C/RC",
                    row["lambda_time"],
                    row["lambda_energy"],
                    row["raw_energy_norm"],
                    row["raw_penalty"],
                    row["fast_ratio"],
                    row["unvisited"],
                    f"aggregate of type-level rows; pareto_efficient={keep}",
                ]
            )


def main():
    records = collect_records()
    agg = aggregate(records)
    plot(records, agg)
    write_sources(records, agg)
    print(f"Saved {OUT_PNG}")
    print(f"Saved {OUT_SVG}")
    print(f"Saved {OUT_PDF}")
    print(f"Saved {OUT_TIFF}")
    print(f"Saved {OUT_CSV}")


if __name__ == "__main__":
    main()
