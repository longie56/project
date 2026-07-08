import csv
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "figures"
OUT_PNG = OUT_DIR / "FigX_training_convergence_SCI.png"
OUT_SVG = OUT_DIR / "FigX_training_convergence_SCI.svg"
OUT_PDF = OUT_DIR / "FigX_training_convergence_SCI.pdf"
OUT_TIFF = OUT_DIR / "FigX_training_convergence_SCI.tiff"
OUT_CSV = OUT_DIR / "FigX_training_convergence_SCI_sources.csv"

SMOOTHING_WINDOW = 5
BAND_WINDOW = 5

RUNS = [
    {
        "label": "N20",
        "path": ROOT / "checkpoints/n20_full_100_smooth/full_epoch100.pt",
        "color": "#1f4e79",
        "linestyle": "--",
        "linewidth": 1.4,
    },
    {
        "label": "N50",
        "path": ROOT / "checkpoints/n50_fromscratch_mixed_random_100e_2060/full_epoch100.pt",
        "color": "#1f4e79",
        "linestyle": ":",
        "linewidth": 1.5,
    },
    {
        "label": "N100",
        "path": ROOT / "n100_bita_backup_20260608_174051/full_epoch100.pt",
        "color": "#1f4e79",
        "linestyle": "-",
        "linewidth": 2.0,
    },
]


def as_float_array(values):
    return np.asarray([float(value) for value in values if value is not None], dtype=float)


def rolling_mean(values, window):
    if len(values) == 0:
        return values
    out = np.empty_like(values, dtype=float)
    for idx in range(len(values)):
        start = max(0, idx - window + 1)
        out[idx] = np.mean(values[start : idx + 1])
    return out


def rolling_std(values, window):
    if len(values) == 0:
        return values
    out = np.empty_like(values, dtype=float)
    for idx in range(len(values)):
        start = max(0, idx - window + 1)
        segment = values[start : idx + 1]
        out[idx] = np.std(segment, ddof=0) if len(segment) > 1 else 0.0
    return out


def load_run(spec):
    path = spec["path"]
    if not path.exists():
        return {**spec, "missing": True}

    ckpt = torch.load(path, map_location="cpu")
    val_scores = as_float_array(ckpt.get("val_scores", []))
    feasible = as_float_array(ckpt.get("val_feasible", []))
    epochs = np.arange(1, len(val_scores) + 1, dtype=float)
    feasible_epochs = np.arange(1, len(feasible) + 1, dtype=float)

    return {
        **spec,
        "missing": False,
        "checkpoint_epoch": ckpt.get("epoch"),
        "best_epoch": ckpt.get("best_epoch"),
        "best_val_score": ckpt.get("best_val_score"),
        "epochs": epochs,
        "val_scores_raw": val_scores,
        "val_scores_smooth": rolling_mean(val_scores, SMOOTHING_WINDOW),
        "val_scores_band": rolling_std(val_scores, BAND_WINDOW),
        "feasible_epochs": feasible_epochs,
        "feasible_raw": feasible,
        "feasible_smooth": rolling_mean(feasible, SMOOTHING_WINDOW),
        "feasible_band": rolling_std(feasible, BAND_WINDOW),
    }


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


def plot_metric(ax, runs, metric, ylabel, panel_label, ylim=None, legend=False):
    handles = []
    for run in runs:
        x_key = "epochs" if metric == "val_scores" else "feasible_epochs"
        raw_key = f"{metric}_raw"
        smooth_key = f"{metric}_smooth"
        band_key = f"{metric}_band"
        x = run[x_key]
        raw = run[raw_key]
        smooth = run[smooth_key]
        band = run[band_key]
        color = run["color"]

        ax.plot(
            x,
            raw,
            color=color,
            linestyle=run["linestyle"],
            linewidth=0.8,
            alpha=0.20,
            zorder=1,
        )
        ax.fill_between(
            x,
            smooth - band,
            smooth + band,
            color=color,
            alpha=0.08,
            linewidth=0,
            zorder=0,
        )
        line = ax.plot(
            x,
            smooth,
            label=run["label"],
            color=color,
            linestyle=run["linestyle"],
            linewidth=run["linewidth"],
            zorder=2,
        )[0]
        handles.append(line)

    ax.set_xlim(-4, 104)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_xlabel("Training Epochs", fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.grid(True, color="#eeeeee", linewidth=0.45, alpha=0.55)
    ax.text(
        -0.07,
        1.035,
        panel_label,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        ha="left",
        va="top",
        clip_on=False,
    )
    ax.tick_params(axis="both", labelsize=11)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if legend:
        ax.legend(loc="upper right", fontsize=11)
    return handles


def add_feasible_inset(ax, runs):
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

    inset = inset_axes(ax, width="38%", height="38%", loc="lower right", borderpad=1.2)
    for run in runs:
        inset.plot(
            run["feasible_epochs"],
            run["feasible_smooth"],
            color=run["color"],
            linestyle=run["linestyle"],
            linewidth=max(1.1, run["linewidth"] - 0.3),
            zorder=2,
        )
    inset.set_xlim(0, 15)
    inset.set_ylim(0.955, 1.001)
    inset.set_xticks([0, 5, 10, 15])
    inset.set_yticks([0.96, 0.98, 1.00])
    inset.grid(True, color="#eeeeee", linewidth=0.4, alpha=0.55)
    inset.tick_params(axis="both", labelsize=8, direction="in")
    mark_inset(
        ax,
        inset,
        loc1=2,
        loc2=4,
        fc="none",
        ec="#9a9a9a",
        lw=0.65,
        alpha=0.75,
    )


def plot(runs):
    configure_matplotlib()
    import matplotlib.pyplot as plt

    usable = [run for run in runs if not run.get("missing") and len(run["val_scores_raw"]) > 0]
    if not usable:
        raise SystemExit("No usable training histories found.")

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.1))
    handles = plot_metric(
        axes[0],
        usable,
        metric="val_scores",
        ylabel="Validation Cost",
        panel_label="(a)",
        legend=False,
    )
    plot_metric(
        axes[1],
        usable,
        metric="feasible",
        ylabel="Feasible Rate",
        panel_label="(b)",
        ylim=(0.94, 1.005),
        legend=False,
    )
    add_feasible_inset(axes[1], usable)

    fig.legend(
        handles,
        [run["label"] for run in usable],
        loc="upper center",
        ncol=3,
        fontsize=11,
        bbox_to_anchor=(0.5, 1.01),
        handlelength=3.0,
        frameon=False,
    )
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.16, top=0.84, wspace=0.22)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=600)
    fig.savefig(OUT_SVG)
    fig.savefig(OUT_PDF)
    fig.savefig(OUT_TIFF, dpi=600)
    plt.close(fig)


def write_sources(runs):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "label",
                "checkpoint",
                "missing",
                "checkpoint_epoch",
                "best_epoch",
                "best_val_score",
                "points",
                "smoothing_window",
                "uncertainty_band",
            ]
        )
        for run in runs:
            writer.writerow(
                [
                    run["label"],
                    run["path"],
                    run.get("missing", False),
                    run.get("checkpoint_epoch", ""),
                    run.get("best_epoch", ""),
                    run.get("best_val_score", ""),
                    len(run.get("val_scores_raw", [])),
                    SMOOTHING_WINDOW,
                    "rolling std within the same run; no multi-seed std available",
                ]
            )


def main():
    runs = [load_run(spec) for spec in RUNS]
    plot(runs)
    write_sources(runs)
    print(f"Saved {OUT_PNG}")
    print(f"Saved {OUT_SVG}")
    print(f"Saved {OUT_PDF}")
    print(f"Saved {OUT_TIFF}")
    print(f"Saved {OUT_CSV}")


if __name__ == "__main__":
    main()
