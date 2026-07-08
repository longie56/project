import csv
import json
import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import DEVICE, E_SAFE_MARGIN
from data.solomon_generator import SolomonGenerator
from env.vrp_env import VRPTWEnv


DATA_DIR = ROOT / "tables" / "n100_by_type_random"
FALLBACK_JSON = DATA_DIR / "n100_c_random_seed42_50.json"
OUT_DIR = ROOT / "figures"
OUT_PNG = OUT_DIR / "Fig10_n100_route_visualization.png"
OUT_SVG = OUT_DIR / "Fig10_n100_route_visualization.svg"
OUT_PDF = OUT_DIR / "Fig10_n100_route_visualization.pdf"
OUT_TIFF = OUT_DIR / "Fig10_n100_route_visualization.tiff"
OUT_CSV = OUT_DIR / "Fig10_n100_route_visualization_sources.csv"

BLUE = "#1f4e79"
FAST_RED = "#b22222"
ORANGE = "#c65f21"
PYVRP_PURPLE = "#7a5c91"
NODE_GRAY = "#c9c9c9"
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


def count_speed_modes(actions, n_nodes):
    econ = 0
    fast = 0
    for action_value in actions:
        action = int(action_value)
        if action % 2 == 1:
            fast += 1
        else:
            econ += 1
    total = max(1, econ + fast)
    fast_ratio = fast / total
    mix_balance = 1.0 - abs(fast_ratio - 0.5) * 2.0
    return econ, fast, mix_balance


def choose_instance(obj):
    best = None
    n_nodes = int(obj.get("config", {}).get("n_nodes", 100))
    for row in obj.get("rows", []):
        ortools = row.get("ortools_physical", {})
        pyvrp = row.get("pyvrp", {})
        ours = row.get("drl", {})
        alns = row.get("alns", {})
        if not (
            ortools.get("actions")
            and pyvrp.get("actions")
            and ours.get("route_actions")
            and alns.get("actions")
        ):
            continue
        econ, fast, mix_balance = count_speed_modes(ours.get("route_actions", []), n_nodes)
        if not (econ > 0 and fast > 0):
            continue
        score = (
            float(ortools.get("unvisit") or 0.0) >= 10.0,
            float(pyvrp.get("unvisit") or 0.0) >= 10.0,
            mix_balance >= 0.35,
            0 if ortools.get("feasible") else 1,
            0 if pyvrp.get("feasible") else 1,
            float(ortools.get("unvisit") or 0.0) + float(pyvrp.get("unvisit") or 0.0),
            mix_balance,
            float(ortools.get("pen") or 0.0) + float(pyvrp.get("pen") or 0.0),
            float(ortools.get("cost") or 0.0) + float(pyvrp.get("cost") or 0.0),
        )
        if best is None or score > best[0]:
            best = (score, row, econ, fast, mix_balance)
    if best is None:
        raise ValueError("No rows found for route visualization.")
    return best[1], {"econ": best[2], "fast": best[3], "mix_balance": best[4]}


def choose_across_files():
    best = None
    for path in sorted(DATA_DIR.glob("n100_*_random_seed42_50.json")):
        obj = load_json(path)
        row, speed_info = choose_instance(obj)
        ortools = row.get("ortools_physical", {})
        pyvrp = row.get("pyvrp", {})
        score = (
            float(ortools.get("unvisit") or 0.0) >= 10.0,
            float(pyvrp.get("unvisit") or 0.0) >= 10.0,
            speed_info["mix_balance"] >= 0.35,
            float(ortools.get("unvisit") or 0.0) + float(pyvrp.get("unvisit") or 0.0),
            speed_info["mix_balance"],
            float(ortools.get("cost") or 0.0) + float(pyvrp.get("cost") or 0.0),
        )
        if best is None or score > best[0]:
            best = (score, path, obj, row, speed_info)
    if best is not None:
        return best[1], best[2], best[3], best[4]
    obj = load_json(FALLBACK_JSON)
    row, speed_info = choose_instance(obj)
    return FALLBACK_JSON, obj, row, speed_info


def regenerate_instance(config, target_idx):
    seed = int(config.get("seed", 42))
    instances = int(config.get("instances", target_idx + 1))
    n_nodes = int(config.get("n_nodes", 100))
    ptype = str(config.get("problem_type", "C")).upper()
    tightness = float(config.get("tightness", 0.7))

    torch.manual_seed(seed)
    random.seed(seed)
    nodes, depot, visited = SolomonGenerator.generate(
        batch_size=instances,
        n_nodes=n_nodes,
        problem_type=ptype,
        tightness=tightness,
        device=DEVICE,
    )
    if str(config.get("pref_mode", "random")).lower() == "random":
        low = float(config.get("pref_random_low", 0.1))
        high = float(config.get("pref_random_high", 0.9))
        alpha = torch.rand(instances, 1, device=DEVICE) * (high - low) + low
        pref = torch.cat([alpha, 1.0 - alpha], dim=1)
    else:
        raw_pref = torch.tensor([config.get("pref", [0.5, 0.5])], dtype=torch.float32, device=DEVICE)
        pref = raw_pref.expand(instances, -1)
        pref = pref / pref.sum(dim=-1, keepdim=True).clamp_min(1e-9)
    return (
        nodes[target_idx : target_idx + 1],
        depot[target_idx : target_idx + 1],
        visited[target_idx : target_idx + 1],
        pref[target_idx : target_idx + 1],
    )


def replay(nodes, depot, visited, pref, actions):
    env = VRPTWEnv(nodes.clone(), pref.clone(), depot.clone(), scenario="full")
    env.load_state(
        visited.clone(),
        depot.clone(),
        torch.zeros(1, device=DEVICE),
        torch.ones(1, device=DEVICE),
        torch.ones(1, device=DEVICE),
    )
    n_nodes = nodes.size(1)
    segments = []
    for step, action_value in enumerate(actions):
        action = int(action_value)
        start = env.pos[0].detach().cpu().numpy().tolist()
        speed_fast = bool(action % 2 == 1)
        is_customer = action < n_nodes * 2
        target_node = int(action // 2) if is_customer else None
        if is_customer:
            end = nodes[0, target_node, :2].detach().cpu().numpy().tolist()
        else:
            end = depot[0].detach().cpu().numpy().tolist()
        energy_before = float(env.energy[0].detach().cpu().item())
        visited_before = env.visited.clone()
        env.step(torch.tensor([action], dtype=torch.long, device=DEVICE))
        energy_after = float(env.energy[0].detach().cpu().item())
        segments.append(
            {
                "step": step,
                "action": action,
                "is_customer": is_customer,
                "target_node": target_node,
                "fast": speed_fast,
                "x0": start[0],
                "y0": start[1],
                "x1": end[0],
                "y1": end[1],
                "energy_before": energy_before,
                "energy_after": energy_after,
                "new_visit": bool((env.visited & ~visited_before).any().item()),
            }
        )
        if bool(env._compute_done()[0].detach().cpu().item()):
            break
    unvisited = (~env.visited[0]).detach().cpu().numpy().nonzero()[0].tolist()
    return segments, unvisited, float(env.energy[0].detach().cpu().item())


def draw_route(ax, nodes_xy, depot_xy, segments, unvisited, panel_label, method_label, route_kind, metrics):
    ax.scatter(nodes_xy[:, 0], nodes_xy[:, 1], s=14, color=NODE_GRAY, alpha=0.65, edgecolors="none")
    if unvisited:
        uv = nodes_xy[unvisited]
        ax.scatter(
            uv[:, 0],
            uv[:, 1],
            s=40,
            marker="x",
            color=FAST_RED,
            linewidths=1.2,
            label="Unvisited customers",
            zorder=5,
        )
    ax.scatter(
        [depot_xy[0]],
        [depot_xy[1]],
        s=110,
        marker="*",
        color="black",
        edgecolor="white",
        linewidth=0.5,
        label="Depot",
        zorder=6,
    )

    for seg in segments:
        if route_kind == "ours":
            color = FAST_RED if seg["fast"] else BLUE
            linestyle = "--" if seg["fast"] else "-"
            alpha = 0.62 if seg["fast"] else 0.74
            linewidth = 0.95 if seg["fast"] else 1.05
        else:
            color = PYVRP_PURPLE if route_kind == "pyvrp" else ORANGE
            linestyle = "-"
            alpha = 0.45
            linewidth = 0.8
        ax.plot(
            [seg["x0"], seg["x1"]],
            [seg["y0"], seg["y1"]],
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            alpha=alpha,
            zorder=2,
        )

    feasible_rate = 1.0 if bool(metrics.get("feasible")) else 0.0
    subtitle = (
        f"Cost={metrics.get('cost'):.2f}, Feasible={feasible_rate:.2f}, "
        f"Unvisited={metrics.get('unvisit'):.0f}"
    )
    ax.text(
        -0.02,
        1.105,
        f"{panel_label} {method_label}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10.6,
        fontweight="bold",
        color="#111111",
        clip_on=False,
    )
    ax.text(
        -0.02,
        1.045,
        subtitle,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.4,
        color="#333333",
        clip_on=False,
    )
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color=GRID, linewidth=0.45, alpha=0.65)
    ax.tick_params(labelsize=9)


def plot(source_path, obj, row, nodes, depot, pref, route_records):
    configure_matplotlib()
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    nodes_xy = nodes[0, :, :2].detach().cpu().numpy()
    depot_xy = depot[0].detach().cpu().numpy()

    fig, axes = plt.subplots(2, 2, figsize=(12.2, 11.0), sharex=True, sharey=True)
    axes_flat = axes.ravel()
    for ax, record in zip(axes_flat, route_records):
        draw_route(
            ax,
            nodes_xy,
            depot_xy,
            record["segments"],
            record["unvisited"],
            record["panel"],
            record["label"],
            record["kind"],
            row[record["method"]],
        )
        ax.set_xlabel("x-coordinate", fontsize=9.5)
        ax.set_ylabel("y-coordinate", fontsize=9.5)
        ax.tick_params(axis="both", which="both", labelleft=True, labelbottom=True, labelsize=8.5)

    handles = [
        Line2D([0], [0], color=ORANGE, lw=1.4, label="OR-Tools route"),
        Line2D([0], [0], color=PYVRP_PURPLE, lw=1.4, label="PyVRP route"),
        Line2D([0], [0], color=FAST_RED, marker="x", lw=0, label="Unvisited"),
        Line2D([0], [0], color="black", marker="*", lw=0, markersize=10, label="Depot"),
    ]
    own_segments = [seg for record in route_records if record["kind"] == "ours" for seg in record["segments"]]
    if any(not seg["fast"] for seg in own_segments):
        handles.insert(1, Line2D([0], [0], color=BLUE, lw=1.6, label="Ours: economical speed"))
    if any(seg["fast"] for seg in own_segments):
        insert_at = 2 if any(not seg["fast"] for seg in own_segments) else 1
        handles.insert(insert_at, Line2D([0], [0], color=FAST_RED, lw=1.4, linestyle="--", label="Ours: fast speed"))
    fig.legend(handles=handles, loc="upper center", ncol=6, fontsize=9.2, bbox_to_anchor=(0.5, 0.995))
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.945), w_pad=1.15, h_pad=2.1)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=600)
    fig.savefig(OUT_SVG)
    fig.savefig(OUT_PDF)
    fig.savefig(OUT_TIFF, dpi=600)
    plt.close(fig)


def write_sources(source_path, obj, row, route_records):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["section", "key", "value"])
        writer.writerow(["source_json", "path", str(source_path.relative_to(ROOT))])
        writer.writerow(["instance", "problem_type", obj.get("config", {}).get("problem_type")])
        writer.writerow(["instance", "index", row.get("instance")])
        writer.writerow(["instance", "pref", json.dumps(row.get("pref"))])
        for method in ["ortools_physical", "pyvrp", "drl", "alns"]:
            payload = row.get(method, {})
            for key in ["cost", "phys", "pen", "unvisit", "feasible", "elapsed_s"]:
                writer.writerow([method, key, payload.get(key)])
        for record in route_records:
            writer.writerow([record["method"], "unvisited_nodes", json.dumps(record["unvisited"])])
        writer.writerow([])
        writer.writerow(["method", "step", "action", "fast", "x0", "y0", "x1", "y1", "energy_before", "energy_after"])
        for record in route_records:
            for seg in record["segments"]:
                writer.writerow(
                    [
                        record["method"],
                        seg["step"],
                        seg["action"],
                        seg["fast"],
                        seg["x0"],
                        seg["y0"],
                        seg["x1"],
                        seg["y1"],
                        seg["energy_before"],
                        seg["energy_after"],
                    ]
                )


def main():
    source_path, obj, row, speed_info = choose_across_files()
    nodes, depot, visited, pref = regenerate_instance(obj.get("config", {}), int(row["instance"]))
    stored_pref = row.get("pref", [])
    regenerated_pref = [float(pref[0, 0].item()), float(pref[0, 1].item())]
    if stored_pref and max(abs(float(a) - float(b)) for a, b in zip(stored_pref, regenerated_pref)) > 1e-4:
        print(
            "WARNING: regenerated preference does not exactly match stored preference; "
            "the route geometry is still generated from the same config."
        )
    method_specs = [
        {
            "panel": "(a)",
            "label": "OR-Tools",
            "method": "ortools_physical",
            "action_key": "actions",
            "kind": "ortools",
        },
        {
            "panel": "(b)",
            "label": "PyVRP",
            "method": "pyvrp",
            "action_key": "actions",
            "kind": "pyvrp",
        },
        {
            "panel": "(c)",
            "label": "PC-EGA-POMO",
            "method": "drl",
            "action_key": "route_actions",
            "kind": "ours",
        },
        {
            "panel": "(d)",
            "label": "PC-EGA-POMO+ALNS",
            "method": "alns",
            "action_key": "actions",
            "kind": "ours",
        },
    ]
    route_records = []
    for spec in method_specs:
        segments, unvisited, terminal_energy = replay(
            nodes, depot, visited, pref, row[spec["method"]][spec["action_key"]]
        )
        route_records.append(
            {
                **spec,
                "segments": segments,
                "unvisited": unvisited,
                "terminal_energy": terminal_energy,
            }
        )
        if spec["method"] in {"ortools_physical", "pyvrp"} and terminal_energy < E_SAFE_MARGIN:
            print(f"{spec['label']} terminal energy below safe margin: {terminal_energy:.4f}")

    plot(source_path, obj, row, nodes, depot, pref, route_records)
    write_sources(source_path, obj, row, route_records)
    print(f"Selected N100 instance {row['instance']} from {source_path}")
    print(f"Ours speed modes: economical={speed_info['econ']}, fast={speed_info['fast']}")
    print(
        "; ".join(
            f"{record['label']} unvisited: {len(record['unvisited'])}"
            for record in route_records
        )
    )
    print(f"Saved {OUT_PNG}")
    print(f"Saved {OUT_SVG}")
    print(f"Saved {OUT_PDF}")
    print(f"Saved {OUT_TIFF}")
    print(f"Saved {OUT_CSV}")


if __name__ == "__main__":
    main()
