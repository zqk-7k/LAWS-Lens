#!/usr/bin/env python3
"""Draw the combined scaling and fixed-budget triage figure for the main text."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parents[1]
DATA = (
    PROJECT_ROOT
    / "server_pull"
    / "sequence_mitigation_sandbox_20260706_stable"
    / "extracted"
    / "results"
    / "sequence_mitigation_sandbox_20260706_stable"
)
FIG_DIR = ROOT / "figures"

rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 7.2,
        "axes.labelsize": 7.4,
        "axes.titlesize": 8.0,
        "axes.titleweight": "bold",
        "axes.labelweight": "bold",
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "legend.fontsize": 6.7,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

C_TIME = "#d95f02"
C_WAVE = "#0072B2"
C_TRUE = "#159895"
C_FALSE = "#c9d1db"
C_LIMIT = "#5f6670"
C_GRID = "#dfe3e8"
C_TEXT = "#2b2b2b"


def load_tables():
    return (
        pd.read_csv(DATA / "method_summary.csv"),
        pd.read_csv(DATA / "fixed_budget_shortlist_by_method.csv"),
    )


def draw_density(ax, summary):
    for method, label, color, marker in [
        ("time-delay + sky-localization", "Time-delay + sky-localization", C_TIME, "^"),
        (
            "calibrated surrogate waveform embedding only",
            "Surrogate waveform embedding",
            C_WAVE,
            "o",
        ),
    ]:
        sub = summary[
            (summary["experiment"] == "density_scan") & (summary["method"] == method)
        ].sort_values("background_density_events_per_yr")
        x = sub["background_density_events_per_yr"].to_numpy()
        y = sub["R10_mean"].to_numpy()
        lo = sub["R10_iqr_low"].to_numpy()
        hi = sub["R10_iqr_high"].to_numpy()
        ax.plot(x, y, marker=marker, ms=4.2, lw=1.4, color=color, label=label, zorder=3)
        ax.fill_between(x, lo, hi, color=color, alpha=0.15, lw=0, zorder=2)

    ax.set_xscale("log")
    ax.set_ylim(-0.04, 1.06)
    ax.set_xlabel(r"Event density (yr$^{-1}$)")
    ax.set_ylabel(r"Companion retrieval $R@10$")
    ax.set_title("Specified-query retrieval")
    ax.grid(True, color=C_GRID, lw=0.6)
    ax.legend(loc="lower left", frameon=False, handlelength=1.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.text(-0.18, 1.08, "a", transform=ax.transAxes, fontsize=11, fontweight="bold")


def draw_scale(ax, summary):
    wave = summary[
        (summary["experiment"] == "rarity_fixed_budget_scan")
        & (summary["method"] == "calibrated surrogate waveform embedding only")
        & (summary["budget"] == 50)
        & (summary["lens_fraction"].isin([1e-3, 1e-4]))
    ].copy()
    wave["all_pairs"] = wave["n_events_mean"] * (wave["n_events_mean"] - 1) / 2
    wave = wave.sort_values("lens_fraction", ascending=False)

    ypos = np.arange(len(wave))[::-1]
    for y, (_, row) in zip(ypos, wave.iterrows()):
        all_pairs = row["all_pairs"]
        ax.hlines(y, 50, all_pairs, color="#cfd5dd", lw=5.5, zorder=1)
        ax.scatter([all_pairs], [y], s=90, color="#a9b4c0", edgecolor="white", lw=0.7, zorder=3)
        ax.scatter([50], [y], s=90, color=C_TRUE, edgecolor="white", lw=0.7, zorder=4)
        ax.annotate(
            "",
            xy=(65, y),
            xytext=(all_pairs / 2.5, y),
            arrowprops=dict(arrowstyle="-|>", color="#b65d2a", lw=1.1),
            zorder=5,
        )
        ax.text(72, y + 0.13, "top 50", ha="left", va="center", color=C_TRUE, fontsize=6.9)
        ax.text(
            all_pairs * 1.16,
            y + 0.10,
            rf"${all_pairs:.1e}$ pairs",
            ha="left",
            va="center",
            fontsize=7.0,
            color=C_TEXT,
        )
        ax.text(
            all_pairs * 1.16,
            y - 0.12,
            "300 true pairs",
            ha="left",
            va="center",
            fontsize=6.7,
            color="#66707a",
        )

    ax.set_xscale("log")
    ax.set_xlim(20, 3e12)
    ax.set_ylim(-0.45, len(wave) - 0.55 + 1)
    ax.set_yticks(ypos)
    ax.set_yticklabels([r"lens fraction $10^{-3}$", r"lens fraction $10^{-4}$"], fontweight="bold")
    ax.set_xlabel("Number of unordered pairs")
    ax.set_title("Catalog scale to top 50 budget")
    ax.grid(axis="x", color=C_GRID, lw=0.6)
    ax.tick_params(axis="y", length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.text(-0.16, 1.08, "b", transform=ax.transAxes, fontsize=11, fontweight="bold")


def draw_composition(ax, summary):
    sub = summary[
        (summary["experiment"] == "rarity_fixed_budget_scan")
        & (summary["method"] == "calibrated surrogate waveform embedding only")
        & (summary["lens_fraction"] == 1e-3)
    ].sort_values("budget")
    budgets = sub["budget"].astype(int).to_numpy()
    true_mean = sub["true_recovered_mean"].to_numpy()
    false_mean = sub["false_candidates_mean"].to_numpy()
    precision = sub["precision_mean"].to_numpy()

    y = np.arange(len(budgets))
    ax.barh(y, true_mean, color=C_TRUE, height=0.58, label="Recovered true pairs", zorder=3)
    ax.barh(
        y,
        false_mean,
        left=true_mean,
        color=C_FALSE,
        edgecolor="#aeb7c2",
        linewidth=0.5,
        height=0.58,
        label="False candidates",
        zorder=2,
    )
    for yi, budget, true_count, prec in zip(y, budgets, true_mean, precision):
        ax.text(max(true_count / 2, 4), yi, f"{true_count:.1f}", ha="center", va="center",
                color="white", fontsize=6.9, fontweight="bold")
        ax.text(budget + 10, yi, f"P={prec:.2f}", ha="left", va="center", fontsize=6.8,
                color=C_TEXT)

    ax.set_yticks(y)
    ax.set_yticklabels([f"top {b}" for b in budgets])
    ax.set_xlim(0, 560)
    ax.set_xlabel("Pairs in shortlist")
    ax.set_title(r"Shortlist composition, lens fraction $10^{-3}$")
    ax.grid(axis="x", color=C_GRID, lw=0.6, zorder=0)
    ax.legend(loc="lower right", frameon=False, handlelength=1.1)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.text(-0.18, 1.07, "c", transform=ax.transAxes, fontsize=11, fontweight="bold")


def draw_yield(ax, rows):
    budgets = np.array([50, 100, 200, 500])
    rng = np.random.default_rng(10)
    for method, color, marker, label, zorder in [
        (
            "calibrated surrogate waveform embedding only",
            C_WAVE,
            "o",
            "Surrogate waveform embedding",
            3,
        ),
        ("time-delay + sky-localization", C_TIME, "^", "Time-delay + sky-localization", 2),
    ]:
        sub = rows[
            (rows["method"] == method)
            & (rows["lens_fraction"] == 1e-3)
            & (rows["budget"].isin(budgets))
        ]
        means, q1, q3 = [], [], []
        for budget in budgets:
            vals = sub[sub["budget"] == budget]["true_recovered"].to_numpy()
            xs = budget * (1 + rng.normal(0, 0.015, len(vals)))
            ax.scatter(xs, vals, s=12, color=color, alpha=0.25, edgecolor="none", zorder=zorder)
            means.append(vals.mean())
            q1.append(np.quantile(vals, 0.25))
            q3.append(np.quantile(vals, 0.75))
        ax.plot(budgets, means, marker=marker, ms=4.2, lw=1.4, color=color, label=label, zorder=5)
        ax.fill_between(budgets, q1, q3, color=color, alpha=0.13, lw=0, zorder=1)

    ax.plot(budgets, np.minimum(budgets, 300), ls="--", color=C_LIMIT, lw=1.1,
            label="Budget-limited maximum")
    ax.set_xscale("log")
    ax.set_xticks(budgets)
    ax.set_xticklabels([str(b) for b in budgets])
    ax.set_ylim(-5, 320)
    ax.set_xlabel("Shortlist budget B")
    ax.set_ylabel("Recovered true pairs (of 300)")
    ax.set_title(r"True-pair yield, lens fraction $10^{-3}$")
    ax.grid(True, color=C_GRID, lw=0.6)
    ax.legend(loc="upper left", frameon=False, handlelength=1.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.text(-0.18, 1.07, "d", transform=ax.transAxes, fontsize=11, fontweight="bold")


def main():
    summary, rows = load_tables()
    fig = plt.figure(figsize=(7.05, 5.55))
    grid = fig.add_gridspec(
        2,
        2,
        height_ratios=[1.0, 1.05],
        width_ratios=[1.08, 1.0],
        hspace=0.50,
        wspace=0.34,
    )
    draw_density(fig.add_subplot(grid[0, 0]), summary)
    draw_scale(fig.add_subplot(grid[0, 1]), summary)
    draw_composition(fig.add_subplot(grid[1, 0]), summary)
    draw_yield(fig.add_subplot(grid[1, 1]), rows)

    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / "fig_sequence_mitigation_sandbox_stable.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig_sequence_mitigation_sandbox_stable.png", bbox_inches="tight", dpi=600)
    plt.close(fig)


if __name__ == "__main__":
    main()
