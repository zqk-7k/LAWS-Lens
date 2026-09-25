#!/usr/bin/env python3
"""Build the seed-aggregated GWTC candidate-stability figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "source_data" / "fig5_reference_shortlists.csv"
OUTPUT = HERE / "figures" / "fig_real_gwtc34_search"

COLORS = {"GWTC-3": "#168A72", "GWTC-4.1 O4a": "#7B5AA6"}

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 7.4,
        "axes.titlesize": 8.7,
        "axes.titleweight": "bold",
        "axes.labelsize": 8.0,
        "axes.labelweight": "bold",
        "axes.linewidth": 0.75,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 6.6,
        "legend.fontsize": 7.0,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
)


def as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.34,
        1.07,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.0,
        fontweight="bold",
    )


def plot_catalog(ax: plt.Axes, frame: pd.DataFrame, title: str, label: str) -> None:
    frame = frame.sort_values(["median_rank", "pair_key"]).head(10)
    plot = frame.iloc[::-1].reset_index(drop=True)
    y = np.arange(len(plot))
    median = plot["median_rank"].to_numpy(dtype=float)
    lower = median - plot["min_rank"].to_numpy(dtype=float)
    upper = plot["max_rank"].to_numpy(dtype=float) - median
    color = COLORS[str(frame["catalog"].iloc[0])]

    ax.axvspan(0.5, 10.5, color="#EAF3EF", alpha=0.72, lw=0, zorder=0)
    ax.axvline(10.5, color="#83908A", lw=0.7, ls=(0, (3, 2)), zorder=1)
    ax.errorbar(
        median,
        y,
        xerr=np.vstack([lower, upper]),
        fmt="none",
        ecolor="#66717B",
        elinewidth=0.75,
        capsize=1.7,
        zorder=2,
    )
    for index, row in plot.iterrows():
        filled = as_bool(row["full_waveform_scored"])
        ax.scatter(
            row["median_rank"],
            index,
            s=30,
            facecolor=color if filled else "white",
            edgecolor=color,
            linewidth=0.95,
            zorder=3,
        )

    ax.set_yticks(y, plot["display_pair"])
    ax.set_xlabel("Rank across training seeds")
    ax.set_title(title, pad=6)
    ax.grid(axis="x", color="#D9DEE2", linewidth=0.55, alpha=0.8)
    maximum = max(12, int(plot["max_rank"].max()) + 1)
    ax.set_xlim(0.5, maximum + 0.5)
    ax.set_xticks([value for value in (1, 3, 5, 7, 10, 12) if value <= maximum])
    ax.text(
        0.97,
        0.035,
        "top 10",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.1,
        color="#4F665B",
    )
    panel_label(ax, label)


def main() -> None:
    data = pd.read_csv(SOURCE)
    required = {
        "catalog",
        "pair_key",
        "median_rank",
        "min_rank",
        "max_rank",
        "full_waveform_scored",
        "display_pair",
    }
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    fig, axes = plt.subplots(1, 2, figsize=(7.25, 3.58))
    plt.subplots_adjust(left=0.18, right=0.985, bottom=0.17, top=0.79, wspace=0.47)
    plot_catalog(
        axes[0], data[data["catalog"].eq("GWTC-3")], "GWTC-3 candidate stability", "a"
    )
    plot_catalog(
        axes[1],
        data[data["catalog"].eq("GWTC-4.1 O4a")],
        "GWTC-4.1 O4a candidate stability",
        "b",
    )
    fig.legend(
        handles=[
            Line2D([], [], marker="o", color="#4E5963", markerfacecolor="#4E5963", lw=0, label="Full waveform score"),
            Line2D([], [], marker="o", color="#4E5963", markerfacecolor="white", lw=0, label="Waveform term set to zero"),
            Line2D([], [], color="#66717B", lw=0.8, marker="|", label="Three-seed rank range"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.56, 0.975),
        ncol=3,
        columnspacing=1.5,
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(OUTPUT.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(OUTPUT.with_suffix(".png"), dpi=400, bbox_inches="tight")
    fig.savefig(OUTPUT.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
