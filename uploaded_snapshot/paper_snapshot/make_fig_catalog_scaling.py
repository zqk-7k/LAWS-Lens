"""Draw the three-panel catalog-scaling figure for the main manuscript."""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.ticker import LogLocator, NullFormatter


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "source_data"
OUT = ROOT / "figures" / "fig_catalog_scaling_tradeoff"


def require_times_new_roman() -> None:
    path = font_manager.findfont("Times New Roman", fallback_to_default=False)
    if not path:
        raise RuntimeError("Times New Roman is required for this figure.")


require_times_new_roman()
mpl.rcParams.update(
    {
        "font.family": "Times New Roman",
        "mathtext.fontset": "stix",
        "font.size": 7.5,
        "axes.titlesize": 9.0,
        "axes.titleweight": "bold",
        "axes.labelsize": 8.2,
        "axes.labelweight": "bold",
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3.2,
        "ytick.major.size": 3.2,
        "legend.fontsize": 7.0,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.dpi": 600,
    }
)


COLORS = {
    "candidate": "#4477AA",
    "exact": "#222222",
    "hnsw": "#009988",
    "fidelity": "#AA3377",
    "neutral": "#7A7A7A",
    "grid": "#D8D8D8",
    "shade": "#F1F1F1",
    "selected": "#EE7733",
}


def clean_axis(ax: plt.Axes, grid_axis: str | None = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid_axis:
        ax.grid(axis=grid_axis, color=COLORS["grid"], linewidth=0.55, alpha=0.75)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.13,
        1.05,
        label,
        transform=ax.transAxes,
        fontsize=9.0,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


scaling = pd.read_csv(DATA / "speed_retrieval_scaling_summary.csv")
budget = pd.read_csv(DATA / "speed_waveform_candidate_budget_summary.csv")


fig = plt.figure(figsize=(7.16, 2.62), facecolor="white")
gs = fig.add_gridspec(
    1,
    3,
    wspace=0.50,
    left=0.075,
    right=0.985,
    top=0.885,
    bottom=0.205,
)
ax_a = fig.add_subplot(gs[0, 0])
ax_b = fig.add_subplot(gs[0, 1])
ax_c = fig.add_subplot(gs[0, 2])


# Panel a: analytical number of directed edges sent to physical evidence.
n_grid = np.logspace(3, 6, 240)
k_selected = 200
exhaustive_edges = n_grid * (n_grid - 1)
sparse_edges = n_grid * k_selected
ax_a.plot(
    n_grid,
    exhaustive_edges,
    color=COLORS["exact"],
    linewidth=1.35,
    label=r"Exhaustive, $N(N-1)$",
)
ax_a.plot(
    n_grid,
    sparse_edges,
    color=COLORS["candidate"],
    linewidth=1.35,
    label=rf"Candidate table, $NK_{{\rm cand}}$",
)
n_native = 9000
native_full = n_native * (n_native - 1)
native_sparse = n_native * k_selected
ax_a.scatter(
    [n_native, n_native],
    [native_full, native_sparse],
    s=24,
    color=[COLORS["exact"], COLORS["candidate"]],
    edgecolor="white",
    linewidth=0.5,
    zorder=4,
)
ax_a.annotate(
    "45-fold reduction\nat N = 9,000",
    xy=(n_native, native_sparse),
    xytext=(1.8e4, 3.5e6),
    ha="left",
    va="center",
    fontsize=6.8,
    fontweight="bold",
    arrowprops={"arrowstyle": "-", "lw": 0.75, "color": COLORS["candidate"]},
)
ax_a.set_xscale("log")
ax_a.set_yscale("log")
ax_a.set_xlim(8e2, 1.3e6)
ax_a.set_ylim(5e4, 2e12)
ax_a.set_xlabel("Catalog size, N")
ax_a.set_ylabel("Directed event-pair scores")
ax_a.set_title("Physical-evidence workload", loc="left", pad=4)
ax_a.legend(loc="upper left", borderaxespad=0.2, fontsize=6.6)
ax_a.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
ax_a.xaxis.set_minor_formatter(NullFormatter())
clean_axis(ax_a, "y")
panel_label(ax_a, "a")


# Panel b: exact versus HNSW retrieval scaling.
hnsw = scaling[
    (scaling["method"] == "HNSW")
    & (scaling["k"] == 200)
    & (scaling["ef_search"] == 512)
].sort_values("n_events")
exact = scaling[scaling["method"] == "exact_cosine"].sort_values("n_events")

ax_b.axvspan(9500, 1.25e6, color=COLORS["shade"], zorder=0)
for data, color, marker, label in [
    (exact, COLORS["exact"], "s", "Exhaustive cosine"),
    (hnsw, COLORS["hnsw"], "o", r"HNSW, $K_{\rm cand}=200$"),
]:
    y = data["total_s_median"].to_numpy()
    yerr = np.vstack(
        [
            y - data["total_s_q25"].to_numpy(),
            data["total_s_q75"].to_numpy() - y,
        ]
    )
    ax_b.errorbar(
        data["n_events"],
        y,
        yerr=yerr,
        color=color,
        marker=marker,
        markersize=4.2,
        linewidth=1.3,
        elinewidth=0.7,
        capsize=2.0,
        label=label,
        zorder=3,
    )
ax_b.text(
    0.97,
    0.06,
    r"$N>9{,}000$: indexing-only stress test",
    transform=ax_b.transAxes,
    ha="right",
    va="bottom",
    fontsize=7.0,
    color="#1F1F1F",
    fontweight="bold",
    bbox={"boxstyle": "square,pad=0.15", "facecolor": "white", "edgecolor": "none", "alpha": 0.88},
)
ax_b.set_xscale("log")
ax_b.set_yscale("log")
ax_b.set_xlim(8e2, 1.3e6)
ax_b.set_ylim(5e-3, 60)
ax_b.set_xlabel("Catalog size, N")
ax_b.set_ylabel("Candidate-generation time (s)")
ax_b.set_title("Candidate-generation scaling", loc="left", pad=4)
ax_b.legend(loc="upper left", borderaxespad=0.2)
ax_b.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
ax_b.xaxis.set_minor_formatter(NullFormatter())
clean_axis(ax_b, "y")
panel_label(ax_b, "b")


# Panel c: waveform candidate retention without any old physical reranking.
budget = budget.sort_values("k")
ax_c.plot(
    budget["k"],
    budget["candidate_companion_recall"],
    color=COLORS["candidate"],
    marker="o",
    markersize=4.0,
    markeredgecolor="white",
    markeredgewidth=0.45,
    linewidth=1.35,
    label="True-companion retention",
    zorder=3,
)
ax_c.plot(
    budget["k"],
    budget["ann_top10_fidelity"],
    color=COLORS["fidelity"],
    marker="s",
    markersize=3.7,
    markeredgecolor="white",
    markeredgewidth=0.45,
    linewidth=1.2,
    label="HNSW top-10 fidelity",
    zorder=3,
)
selected = budget[budget["k"] == k_selected].iloc[0]
ax_c.scatter(
    [k_selected],
    [selected["candidate_companion_recall"]],
    marker="o",
    s=48,
    facecolor="none",
    edgecolor=COLORS["selected"],
    linewidth=1.4,
    zorder=4,
)
ax_c.annotate(
    r"$K_{\rm cand}=200$: 97.47%",
    xy=(k_selected, selected["candidate_companion_recall"]),
    xytext=(72, 0.925),
    fontsize=6.8,
    fontweight="bold",
    color="#1F1F1F",
    arrowprops={"arrowstyle": "-", "lw": 0.7, "color": COLORS["selected"]},
    bbox={"boxstyle": "square,pad=0.10", "facecolor": "white", "edgecolor": "none", "alpha": 0.88},
)
ax_c.set_xscale("log")
ax_c.set_xlim(8, 1250)
ax_c.set_ylim(0.85, 1.005)
ax_c.set_xlabel(r"Waveform candidate budget, $K_{\rm cand}$")
ax_c.set_ylabel("Retained fraction")
ax_c.set_title("Waveform candidate retention", loc="left", pad=4)
ax_c.legend(loc="lower right", borderaxespad=0.25, fontsize=6.4)
clean_axis(ax_c, "y")
panel_label(ax_c, "c")


for suffix, kwargs in {
    ".pdf": {},
    ".svg": {},
    ".png": {"dpi": 600},
    ".tiff": {"dpi": 600},
}.items():
    fig.savefig(OUT.with_suffix(suffix), bbox_inches="tight", facecolor="white", **kwargs)

plt.close(fig)
