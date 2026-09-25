#!/usr/bin/env python3
"""Plot a compact PE-overlap and apparent-magnification audit for Top-10 pairs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


AUDIT = Path(__file__).resolve().parent
INPUT = AUDIT / "top10_expanded_pe_audit.csv"
OUTPUT_PDF = AUDIT / "top10_pe_amplitude_overview.pdf"
OUTPUT_PNG = AUDIT / "top10_pe_amplitude_overview.png"

PARAMETERS = (
    ("chirp_mass", r"$\mathcal{M}_c^{\rm det}$"),
    ("mass_ratio", r"$q$"),
    ("chi_eff", r"$\chi_{\rm eff}$"),
    ("mass_1", r"$m_1^{\rm det}$"),
    ("mass_2", r"$m_2^{\rm det}$"),
    ("chi_p", r"$\chi_p$"),
    ("theta_jn", r"$\theta_{JN}$"),
)


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 9,
            "ytick.labelsize": 8,
            "legend.fontsize": 9,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def main() -> None:
    style()
    data = pd.read_csv(INPUT)
    cmap = LinearSegmentedColormap.from_list(
        "posterior_overlap", ["#F5F7F8", "#BCD7DE", "#4E91A4", "#176B79"]
    )
    fig = plt.figure(figsize=(11.2, 10.8))
    grid = fig.add_gridspec(
        2, 2, width_ratios=(1.85, 1.0), height_ratios=(1, 1),
        left=0.24, right=0.96, bottom=0.19, top=0.90, hspace=0.30, wspace=0.13
    )
    image = None
    for row, (deployment, title) in enumerate(
        (("gwtc3", "GWTC-3 / O3"), ("gwtc4", "GWTC-4.1 / O4a"))
    ):
        subset = data[data.deployment == deployment].sort_values("rank").reset_index(drop=True)
        heat_axis = fig.add_subplot(grid[row, 0])
        ratio_axis = fig.add_subplot(grid[row, 1], sharey=heat_axis)
        matrix = np.column_stack(
            [subset[f"{name}_bhattacharyya_coefficient"].to_numpy(float) for name, _ in PARAMETERS]
        )
        image = heat_axis.imshow(matrix, vmin=0, vmax=1, cmap=cmap, aspect="auto")
        for y in range(len(subset)):
            for x in range(len(PARAMETERS)):
                value = matrix[y, x]
                heat_axis.text(
                    x, y, f"{value:.2f}", ha="center", va="center",
                    color="white" if value >= 0.64 else "#263238", fontsize=7.2
                )
        labels = [f"{int(r.rank):02d}  {r.event_i} / {r.event_j}" for r in subset.itertuples()]
        heat_axis.set_yticks(np.arange(len(subset)), labels)
        heat_axis.set_xticks(np.arange(len(PARAMETERS)), [label for _, label in PARAMETERS])
        heat_axis.set_title(f"{title}: marginal posterior overlap", loc="left", fontweight="bold")
        heat_axis.tick_params(axis="x", length=0)
        heat_axis.tick_params(axis="y", length=0, pad=5)
        heat_axis.spines[:].set_visible(False)

        passed = subset.current_intrinsic_3sigma_consistent.astype(bool).to_numpy()
        med = subset.apparent_mu_ratio_i_over_j_median.to_numpy(float)
        low = subset.apparent_mu_ratio_i_over_j_q05.to_numpy(float)
        high = subset.apparent_mu_ratio_i_over_j_q95.to_numpy(float)
        y = np.arange(len(subset))
        for index in range(len(subset)):
            color = "#176B79" if passed[index] else "#D2693C"
            marker = "o" if passed[index] else "x"
            ratio_axis.errorbar(
                med[index], y[index],
                xerr=[[med[index] - low[index]], [high[index] - med[index]]],
                fmt=marker, ms=5.2, mew=1.2, color=color, ecolor=color,
                elinewidth=0.9, capsize=2.0, zorder=3
            )
        ratio_axis.axvline(1.0, color="#6F7780", ls="--", lw=0.9, zorder=0)
        ratio_axis.set_xscale("log")
        finite_low = max(float(np.nanmin(low)) * 0.75, 0.015)
        finite_high = min(float(np.nanmax(high)) * 1.35, 80.0)
        ratio_axis.set_xlim(finite_low, finite_high)
        ratio_axis.set_title(r"Apparent $|\mu_i|/|\mu_j|$", loc="left", fontweight="bold")
        ratio_axis.set_xlabel("Median and 90% posterior interval (log scale)", fontweight="bold")
        ratio_axis.tick_params(axis="y", left=False, labelleft=False)
        ratio_axis.grid(axis="x", color="#DCE1E4", lw=0.55, which="both")
        ratio_axis.spines[["top", "right", "left"]].set_visible(False)

    fig.suptitle(
        "Expanded posterior and apparent-magnification audit of the displayed Top-10 pairs",
        fontsize=14, fontweight="bold", y=0.965
    )
    fig.text(
        0.5, 0.93,
        "Bhattacharyya coefficient: 0 = disjoint, 1 = identical. Magnification ratios are derived from independent luminosity-distance posteriors.",
        ha="center", fontsize=9
    )
    if image is not None:
        color_axis = fig.add_axes([0.24, 0.072, 0.43, 0.016])
        colorbar = fig.colorbar(image, cax=color_axis, orientation="horizontal")
        colorbar.set_label("Marginal posterior Bhattacharyya coefficient", fontweight="bold")
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#176B79", markeredgecolor="#176B79", label=r"Current $D_{\max}\leq3$"),
        Line2D([0], [0], marker="x", color="#D2693C", linestyle="none", label=r"Current $D_{\max}>3$"),
    ]
    fig.legend(handles=handles, loc="lower right", bbox_to_anchor=(0.96, 0.064), ncol=2)
    fig.text(
        0.96, 0.014,
        "Distance-derived ratios are descriptive and do not include a lens-population prior, detector response or selection effects.",
        ha="right", fontsize=8.2
    )
    fig.savefig(OUTPUT_PDF, bbox_inches="tight")
    fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(OUTPUT_PDF)
    print(OUTPUT_PNG)


if __name__ == "__main__":
    main()
