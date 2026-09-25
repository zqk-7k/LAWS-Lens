from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "source_data" / "current_v93"
FIGURES = ROOT / "figures"
FULL_WIDTH = 183.0 / 25.4

COLORS = {
    "background": "#CDD3D8",
    "background_edge": "#89959F",
    "sis": "#356FA8",
    "pm": "#D8752D",
    "waveform": "#0072B2",
    "time": "#D55E00",
    "sky": "#009E73",
    "ranking": "#7A5195",
    "neutral": "#737373",
    "candidate": "#1B8A78",
    "wave_a": "#2C7FB8",
    "wave_b": "#F28E2B",
    "fail": "#B54A32",
}
DOMAIN_LABELS = {
    "ET3": "ET-3",
    "gwtc3": "O3-noise test",
    "gwtc4": "O4a-noise test",
}
METHODS = ["waveform_only", "time_only", "sky_only", "three_channel"]
METHOD_LABELS = {
    "waveform_only": "Waveform evidence",
    "time_only": "Time-delay evidence",
    "sky_only": "Sky Bayes factor",
    "three_channel": "Catalog ranking score",
}
METHOD_COLORS = {
    "waveform_only": COLORS["waveform"],
    "time_only": COLORS["time"],
    "sky_only": COLORS["sky"],
    "three_channel": COLORS["ranking"],
}
METHOD_MARKERS = {
    "waveform_only": "o",
    "time_only": "s",
    "sky_only": "^",
    "three_channel": "D",
}
FIRST_SEED = {
    "ET3": "202607251",
    "gwtc3": "202607241",
    "gwtc4": "202607241",
}
DISPLAY_EVENT_NAMES = {
    "GW170809": "GW170809_082821",
    "GW170814": "GW170814_103043",
    "GW170104": "GW170104_101158",
}
WAVEFORM_EXAMPLES = {
    "gwtc3": ("GW170809_082821", "GW170814_103043"),
    "gwtc4": ("GW230707_124047", "GW230709_122727"),
}


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 7.6,
            "axes.labelsize": 8.0,
            "axes.labelweight": "bold",
            "axes.titlesize": 8.6,
            "axes.titleweight": "bold",
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 6.9,
            "axes.linewidth": 0.75,
            "xtick.major.width": 0.75,
            "ytick.major.width": 0.75,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def panel_label(ax: plt.Axes, label: str, x: float = -0.14, y: float = 1.04) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.5,
        fontweight="bold",
    )


def export(fig: plt.Figure, stem: str) -> None:
    target = FIGURES / stem
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(target.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(target.with_suffix(".png"), dpi=400, bbox_inches="tight")
    fig.savefig(
        target.with_suffix(".tiff"),
        dpi=600,
        bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)


def plot_channel_separation() -> None:
    density = pd.read_csv(DATA / "channel_separation_density.csv")
    specs = {
        "waveform": {
            "title": "Waveform evidence",
            "xlabel": r"$\mathbf{Z}_{\rm wf}$",
            "limits": (-12.0, 14.0),
            "ticks": [-12, -5, 0, 5, 10, 14],
            "left": True,
            "right": False,
        },
        "time": {
            "title": "Time-delay evidence",
            "xlabel": r"$\mathbf{Z}_{\rm time}$",
            "limits": (-7.0, 20.0),
            "ticks": [-7, 0, 5, 10, 15, 20],
            "left": True,
            "right": True,
        },
        "sky": {
            "title": "Common-sky evidence",
            "xlabel": r"$\mathbf{Z}_{\rm sky}$",
            "limits": (-8.0, 9.0),
            "ticks": [-8, -4, 0, 4, 8],
            "left": True,
            "right": False,
        },
        "ranking": {
            "title": "Catalog ranking score",
            "xlabel": r"$\mathbf{S}/\|\boldsymbol{\lambda}\|_2$",
            "limits": (-12.0, 20.0),
            "ticks": [-12, -5, 0, 5, 10, 15, 20],
            "left": True,
            "right": True,
        },
    }
    bases = {"ET3": 2.0, "gwtc3": 1.0, "gwtc4": 0.0}
    fig, axes = plt.subplots(
        1,
        4,
        figsize=(FULL_WIDTH, 3.28),
        sharey=True,
        gridspec_kw={"wspace": 0.23},
    )
    for panel_index, (score, spec) in enumerate(specs.items()):
        ax = axes[panel_index]
        for domain, base in bases.items():
            ax.axhline(base, color="#B9C1C7", lw=0.55, zorder=0)
            for pair_class in ("Non-companion", "SIS", "PM"):
                subset = density[
                    (density["domain"] == domain)
                    & (density["score"] == score)
                    & (density["pair_class"] == pair_class)
                ]
                x = subset["x"].to_numpy()
                mean = subset["density_mean"].to_numpy()
                low = subset["density_min"].to_numpy()
                high = subset["density_max"].to_numpy()
                if pair_class == "Non-companion":
                    ax.fill_between(
                        x,
                        base,
                        base + 0.68 * mean,
                        facecolor=COLORS["background"],
                        edgecolor=COLORS["background_edge"],
                        linewidth=0.55,
                        alpha=0.82,
                        zorder=1,
                    )
                else:
                    color = COLORS["sis"] if pair_class == "SIS" else COLORS["pm"]
                    linestyle = "-" if pair_class == "SIS" else "--"
                    ax.fill_between(
                        x,
                        base + 0.68 * low,
                        base + 0.68 * high,
                        color=color,
                        alpha=0.10,
                        linewidth=0,
                        zorder=2,
                    )
                    ax.plot(
                        x,
                        base + 0.68 * mean,
                        color=color,
                        linestyle=linestyle,
                        linewidth=1.35,
                        zorder=3,
                    )
        ax.axvline(
            0,
            color=COLORS["neutral"],
            lw=0.65,
            linestyle=(0, (2, 2)),
            alpha=0.75,
        )
        ax.set_xlim(spec["limits"])
        ax.set_ylim(-0.04, 2.78)
        ax.set_title(spec["title"], pad=7)
        ax.set_xlabel(spec["xlabel"], labelpad=5)
        ax.set_xticks(spec["ticks"])
        labels = [f"{value:g}" for value in spec["ticks"]]
        if spec["left"]:
            labels[0] = rf"$\leq${spec['ticks'][0]:g}"
        if spec["right"]:
            labels[-1] = rf"$\geq${spec['ticks'][-1]:g}"
        ax.set_xticklabels(labels)
        ax.spines["left"].set_visible(panel_index == 0)
        panel_label(ax, chr(ord("a") + panel_index))
    axes[0].set_yticks([0, 1, 2])
    axes[0].set_yticklabels(
        ["O4a-noise test", "O3-noise test", "ET-3"]
    )
    axes[0].set_ylabel("Held-out test domain", labelpad=8)
    for ax in axes[1:]:
        ax.tick_params(axis="y", left=False, labelleft=False)
    fig.legend(
        handles=[
            Patch(
                facecolor=COLORS["background"],
                edgecolor=COLORS["background_edge"],
                label="Non-companion pairs",
            ),
            Line2D([0], [0], color=COLORS["sis"], lw=1.5, label="SIS companions"),
            Line2D(
                [0],
                [0],
                color=COLORS["pm"],
                lw=1.5,
                ls="--",
                label="PM companions",
            ),
            Line2D(
                [0],
                [0],
                color=COLORS["neutral"],
                lw=0.7,
                ls=(0, (2, 2)),
                label="Neutral evidence",
            ),
        ],
        loc="upper center",
        bbox_to_anchor=(0.54, 0.985),
        ncol=4,
        handlelength=2.5,
        columnspacing=1.25,
    )
    fig.subplots_adjust(left=0.175, right=0.98, bottom=0.18, top=0.78)
    export(fig, "fig_channel_separation_corner")


def plot_retrieval_curves() -> None:
    frame = pd.read_csv(DATA / "retrieval_curves_per_seed.csv", dtype={"seed": str})
    selected = pd.concat(
        [
            frame[
                (frame["domain"] == domain)
                & (frame["seed"] == seed)
            ]
            for domain, seed in FIRST_SEED.items()
        ],
        ignore_index=True,
    )
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(FULL_WIDTH, 2.58),
        gridspec_kw={"wspace": 0.31},
    )
    for panel_index, (domain, ax) in enumerate(
        zip(("ET3", "gwtc3", "gwtc4"), axes)
    ):
        subset = selected[selected["domain"] == domain]
        for method in METHODS:
            values = subset[subset["method"] == method].sort_values("rank_cutoff")
            ax.plot(
                values["rank_cutoff"],
                values["recall"],
                color=METHOD_COLORS[method],
                marker=METHOD_MARKERS[method],
                markersize=3.3,
                linewidth=1.35,
            )
        ax.set_xscale("log")
        ax.set_xlim(0.9, 55)
        ax.set_ylim(0, 1.03)
        ax.set_xticks([1, 2, 5, 10, 20, 50])
        ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
        ax.set_xlabel(r"Rank cutoff, $K$")
        if panel_index == 0:
            ax.set_ylabel(r"Companion recall, R@$K$")
        ax.set_title(DOMAIN_LABELS[domain], pad=6)
        panel_label(ax, chr(ord("a") + panel_index), x=-0.16)
    fig.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color=METHOD_COLORS[method],
                marker=METHOD_MARKERS[method],
                linewidth=1.35,
                markersize=3.5,
                label=METHOD_LABELS[method],
            )
            for method in METHODS
        ],
        loc="upper center",
        bbox_to_anchor=(0.515, 0.995),
        ncol=4,
        columnspacing=1.15,
        handlelength=2.1,
    )
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.20, top=0.78)
    export(fig, "fig_retrieval_pair_curves_current")


def display_event(event: str) -> str:
    return DISPLAY_EVENT_NAMES.get(str(event), str(event))


def pair_label(row: object) -> str:
    return (
        f"{int(row.consensus_rank):02d}  "
        f"{display_event(row.event_i)} / {display_event(row.event_j)}"
    )


def plot_measured_strain(
    ax: plt.Axes, waveforms: pd.DataFrame, domain: str
) -> None:
    time = waveforms["time_from_event_gps_s"].to_numpy(dtype=float)
    pair = WAVEFORM_EXAMPLES[domain]
    rows = [
        (pair[0], "H1", 3.0, COLORS["wave_a"]),
        (pair[0], "L1", 2.0, COLORS["wave_a"]),
        (pair[1], "H1", 1.0, COLORS["wave_b"]),
        (pair[1], "L1", 0.0, COLORS["wave_b"]),
    ]
    values = np.vstack(
        [
            waveforms[f"{event}__{detector}"].to_numpy(dtype=float)
            for event, detector, _, _ in rows
        ]
    )
    scale = 0.32 / max(float(np.quantile(np.abs(values), 0.995)), 1e-12)
    for event, detector, offset, color in rows:
        trace = waveforms[f"{event}__{detector}"].to_numpy(dtype=float)
        ax.plot(time, offset + scale * trace, color=color, lw=0.32, alpha=0.88)
    ax.axvline(0, color="#5A5A5A", ls=(0, (2, 2)), lw=0.60)
    ax.set_xlim(-0.80, 0.12)
    ax.set_ylim(-0.45, 3.45)
    ax.set_yticks(
        [3, 2, 1, 0],
        [
            f"{pair[0]} H1",
            f"{pair[0]} L1",
            f"{pair[1]} H1",
            f"{pair[1]} L1",
        ],
    )
    ax.tick_params(axis="y", length=0, labelsize=4.7, pad=2)
    ax.set_xticks([-0.75, -0.50, -0.25, 0.0])
    ax.tick_params(axis="x", labelsize=5.7)
    ax.set_xlabel("Time from catalog trigger (s)", fontsize=6.4, labelpad=1)
    ax.set_title("Measured strain (noise retained)", fontsize=7.3, pad=4)
    ax.spines["left"].set_visible(False)


def plot_reconstructed_waveform(
    ax: plt.Axes, waveforms: pd.DataFrame, domain: str
) -> None:
    time = waveforms["time_from_peak_s"].to_numpy(dtype=float)
    pair = WAVEFORM_EXAMPLES[domain]
    rows = [
        (pair[0], 1.0, COLORS["wave_a"]),
        (pair[1], 0.0, COLORS["wave_b"]),
    ]
    for event, offset, color in rows:
        trace = waveforms[event].to_numpy(dtype=float)
        ax.plot(time, offset + 0.38 * trace, color=color, lw=0.60, alpha=0.96)
    ax.axvline(0, color="#5A5A5A", ls=(0, (2, 2)), lw=0.65)
    ax.set_xlim(-0.80, 0.08)
    ax.set_ylim(-0.48, 1.48)
    ax.set_yticks([1, 0], [pair[0], pair[1]])
    ax.tick_params(axis="y", length=0, labelsize=5.0, pad=2)
    ax.set_xticks([-0.75, -0.50, -0.25, 0.0])
    ax.tick_params(axis="x", labelsize=5.7)
    ax.set_xlabel("Time from reconstructed peak (s)", fontsize=6.4, labelpad=1)
    ax.set_title("Noise-free PE reconstruction", fontsize=7.3, pad=4)
    ax.text(
        0.98,
        0.04,
        r"normalized $h_+$",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.5,
        color="#4F4F4F",
    )
    ax.spines["left"].set_visible(False)


def plot_real_candidates() -> None:
    candidates = pd.read_csv(DATA / "consensus_real_candidates_top10_v93.csv")
    measured_waveforms = pd.read_csv(DATA / "representative_waveform_inputs.csv")
    reconstructed_waveforms = pd.read_csv(DATA / "reconstructed_waveform_inputs.csv")
    distance_columns = [
        "chirp_mass_standardized_posterior_distance",
        "mass_ratio_standardized_posterior_distance",
        "chi_eff_standardized_posterior_distance",
    ]
    distance_labels = [
        r"$D_{\mathcal{M}_c}$",
        r"$D_q$",
        r"$D_{\chi_{\rm eff}}$",
    ]
    cmap = LinearSegmentedColormap.from_list(
        "pe_distance",
        ["#F7FBFC", "#D8EBF0", "#9EC9D7", "#5B99B3", "#1F5D78"],
    )
    fig = plt.figure(figsize=(FULL_WIDTH, 6.05))
    top = fig.add_gridspec(
        1,
        2,
        left=0.205,
        right=0.985,
        bottom=0.680,
        top=0.955,
        wspace=0.88,
    )
    bottom = fig.add_gridspec(
        1,
        4,
        width_ratios=[1.18, 1.28, 1.18, 1.28],
        left=0.060,
        right=0.985,
        bottom=0.125,
        top=0.585,
        wspace=0.54,
    )
    image = None
    for domain_index, domain in enumerate(("gwtc3", "gwtc4")):
        subset = candidates[candidates["deployment"] == domain].sort_values(
            "consensus_rank"
        )
        y = np.arange(len(subset))
        example = set(WAVEFORM_EXAMPLES[domain])
        example_mask = subset.apply(
            lambda row: {
                display_event(row["event_i"]),
                display_event(row["event_j"]),
            }
            == example,
            axis=1,
        ).to_numpy(dtype=bool)
        score_ax = fig.add_subplot(top[0, domain_index])
        score_max = float(subset["final_score_mean"].max())
        score_ax.hlines(
            y,
            0,
            subset["final_score_mean"],
            color="#C7D3D8",
            linewidth=0.75,
        )
        score_ax.scatter(
            subset["final_score_mean"],
            y,
            s=24,
            facecolor=COLORS["candidate"],
            edgecolor="white",
            linewidth=0.55,
            zorder=3,
        )
        if example_mask.any():
            selected_y = int(np.flatnonzero(example_mask)[0])
            score_ax.scatter(
                [subset.iloc[selected_y]["final_score_mean"]],
                [selected_y],
                s=39,
                facecolor=COLORS["candidate"],
                edgecolor=COLORS["wave_b"],
                linewidth=1.25,
                zorder=4,
            )
        score_ax.set_yticks(y, [pair_label(row) for row in subset.itertuples()])
        score_ax.set_ylim(len(subset) - 0.4, -0.6)
        score_ax.set_xlim(0, score_max + max(0.10, 0.06 * score_max))
        score_ax.set_xlabel(r"Mean catalog ranking score, $\overline{S}_{ij}$")
        score_ax.set_title(
            "GWTC-3 consensus shortlist"
            if domain == "gwtc3"
            else "GWTC-4.1 O4a consensus shortlist",
            pad=6,
        )
        score_ax.tick_params(axis="y", length=0, pad=3, labelsize=5.45)
        score_ax.tick_params(axis="x", labelsize=6.4)
        panel_label(score_ax, chr(ord("a") + domain_index), x=-0.69)

        pe_ax = fig.add_subplot(bottom[0, 2 * domain_index])
        values = subset[distance_columns].to_numpy(dtype=float)
        display = np.clip(values, 0, 6)
        image = pe_ax.imshow(
            display,
            cmap=cmap,
            norm=Normalize(vmin=0, vmax=6),
            aspect="auto",
            interpolation="nearest",
        )
        pe_ax.set_xticks(np.arange(3), distance_labels)
        pe_ax.set_yticks(
            np.arange(len(subset)),
            [str(int(value)) for value in subset["consensus_rank"]],
        )
        pe_ax.tick_params(axis="x", length=0, labelsize=8.2, pad=4)
        pe_ax.tick_params(axis="y", length=0, labelsize=7.1, pad=2)
        pe_ax.set_ylabel("Consensus rank", fontsize=7.8)
        pe_ax.set_title("Intrinsic PE audit", fontsize=8.3, pad=5)
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                value = values[i, j]
                text = f"{value:.1f}" if value < 10 else r"$>9.9$"
                pe_ax.text(
                    j,
                    i,
                    text,
                    ha="center",
                    va="center",
                    fontsize=6.4,
                    color="white" if display[i, j] > 4 else "#1A1A1A",
                )
        if example_mask.any():
            selected_y = int(np.flatnonzero(example_mask)[0])
            pe_ax.add_patch(
                Rectangle(
                    (-0.5, selected_y - 0.5),
                    3,
                    1,
                    fill=False,
                    edgecolor=COLORS["wave_b"],
                    linewidth=1.15,
                    clip_on=False,
                )
            )
        panel_label(pe_ax, chr(ord("c") + 2 * domain_index), x=-0.30)
        for spine in pe_ax.spines.values():
            spine.set_visible(False)

        waveform_grid = bottom[0, 2 * domain_index + 1].subgridspec(
            2, 1, hspace=0.72
        )
        measured_ax = fig.add_subplot(waveform_grid[0, 0])
        plot_measured_strain(measured_ax, measured_waveforms, domain)
        panel_label(measured_ax, chr(ord("d") + 2 * domain_index), x=-0.48)

        reconstructed_ax = fig.add_subplot(waveform_grid[1, 0])
        plot_reconstructed_waveform(
            reconstructed_ax, reconstructed_waveforms, domain
        )

    colorbar_axis = fig.add_axes([0.39, 0.035, 0.23, 0.012])
    colorbar = fig.colorbar(image, cax=colorbar_axis, orientation="horizontal")
    colorbar.set_label(
        r"Standardized posterior distance, $D_x$",
        fontsize=7.2,
        labelpad=2,
    )
    colorbar.set_ticks([0, 3, 6])
    colorbar.ax.tick_params(labelsize=6.4, length=2, pad=1)
    export(fig, "fig_real_candidate_pe_audit")


def plot_topb() -> None:
    frame = pd.read_csv(DATA / "topb_summary.csv")
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(FULL_WIDTH, 4.20),
        gridspec_kw={"hspace": 0.42, "wspace": 0.28},
    )
    for domain, color, marker in [
        ("gwtc3", COLORS["waveform"], "o"),
        ("gwtc4", COLORS["time"], "s"),
    ]:
        subset = frame[frame["domain"] == domain].sort_values("budget")
        label = "O3-noise test" if domain == "gwtc3" else "O4a-noise test"
        for ax, metric in [
            (axes[0, 0], "true_pairs"),
            (axes[0, 1], "false_pairs"),
            (axes[1, 0], "precision"),
            (axes[1, 1], "recall"),
        ]:
            ax.plot(
                subset["budget"],
                subset[f"{metric}_mean"],
                color=color,
                marker=marker,
                markersize=3.5,
                lw=1.35,
                label=label,
            )
            ax.fill_between(
                subset["budget"],
                subset[f"{metric}_min"],
                subset[f"{metric}_max"],
                color=color,
                alpha=0.12,
                linewidth=0,
            )
    titles = [
        "True companions retained",
        "False-pair burden",
        "Pair precision",
        "Pair recall",
    ]
    for index, (ax, title) in enumerate(zip(axes.flat, titles)):
        ax.set_xscale("log")
        ax.set_xticks([10, 50, 100, 500, 1000])
        ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
        ax.set_xlabel(r"Global shortlist budget, $B$")
        ax.set_title(title, pad=5)
        ax.grid(axis="y", color="#D8D8D8", lw=0.55, alpha=0.7)
        panel_label(ax, chr(ord("a") + index), x=-0.12)
    axes[0, 0].set_ylabel("Number of true pairs")
    axes[0, 1].set_ylabel("Number of false pairs")
    axes[1, 0].set_ylabel("Precision")
    axes[1, 1].set_ylabel("Recall")
    axes[0, 0].legend(loc="upper left")
    export(fig, "fig_topb_triage")


def plot_catalog_scope_pe() -> None:
    pe = pd.read_csv(DATA / "real_catalog_pe_enrichment_v93.csv")
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(FULL_WIDTH, 2.75),
        gridspec_kw={"wspace": 0.40},
    )
    categories = ["GWTC-3", "GWTC-4.1 O4a"]
    x = np.arange(2)
    available = np.array([63, 84])
    strict = np.array([53, 74])
    axes[0].bar(
        x,
        available,
        width=0.62,
        facecolor="white",
        edgecolor="#9AA4AA",
        linewidth=1.0,
        label="Availability-driven",
    )
    axes[0].bar(
        x,
        strict,
        width=0.42,
        color=[COLORS["waveform"], COLORS["time"]],
        alpha=0.88,
        label="Strict waveform scope",
    )
    axes[0].set_xticks(x, categories)
    axes[0].set_ylabel("Number of events")
    axes[0].set_title("Real-catalog deployment scope")
    axes[0].text(0, 24, "1,953 → 1,378 pairs", ha="center", color="white", fontsize=7.1)
    axes[0].text(1, 34, "3,486 → 2,701 pairs", ha="center", color="white", fontsize=7.1)
    axes[0].legend(loc="upper left")
    panel_label(axes[0], "a")

    for domain, color, marker in [
        ("gwtc3", COLORS["waveform"], "o"),
        ("gwtc4", COLORS["time"], "s"),
    ]:
        subset = pe[pe["domain"] == domain].sort_values("cutoff")
        label = "GWTC-3" if domain == "gwtc3" else "GWTC-4.1 O4a"
        axes[1].plot(
            subset["cutoff"],
            subset["pass_fraction"],
            color=color,
            marker=marker,
            lw=1.35,
            markersize=4,
            label=label,
        )
        axes[1].axhline(
            subset["baseline_fraction"].iloc[0],
            color=color,
            lw=0.8,
            ls=(0, (2, 2)),
            alpha=0.75,
        )
    axes[1].set_xscale("log")
    axes[1].set_xticks([5, 10, 20, 50, 100])
    axes[1].get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    axes[1].set_ylim(0, 1.04)
    axes[1].set_xlabel("Nested consensus shortlist size")
    axes[1].set_ylabel(r"Fraction with $D_{\max}\leq3$")
    axes[1].set_title("Independent intrinsic-parameter audit")
    axes[1].legend(loc="lower left")
    axes[1].grid(axis="y", color="#D8D8D8", lw=0.55, alpha=0.7)
    panel_label(axes[1], "b")
    export(fig, "fig_supp_catalog_scope_pe")


def plot_sky_resolution_audit() -> None:
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(FULL_WIDTH, 2.72),
        gridspec_kw={"wspace": 0.37},
    )
    for index, (domain, color, title) in enumerate(
        [
            ("gwtc3", COLORS["waveform"], "GWTC-3 pair table"),
            ("gwtc4", COLORS["time"], "GWTC-4.1 O4a pair table"),
        ]
    ):
        frame = pd.read_parquet(DATA / f"{domain}_sky_resolution_all_pairs_v93.parquet")
        stable = ~frame["sign_flip_nside32_1024"].astype(bool)
        ax = axes[index]
        ax.scatter(
            frame.loc[stable, "sky_log_bf_nside32"],
            frame.loc[stable, "sky_log_bf_nside1024"],
            s=7,
            color="#A9B1B6",
            alpha=0.32,
            linewidth=0,
            rasterized=True,
        )
        ax.scatter(
            frame.loc[~stable, "sky_log_bf_nside32"],
            frame.loc[~stable, "sky_log_bf_nside1024"],
            s=9,
            color=color,
            alpha=0.62,
            linewidth=0,
            rasterized=True,
            label="Sign flip",
        )
        limits = (-4.5, 4.5)
        ax.plot(limits, limits, color="#4D4D4D", lw=0.8, ls=(0, (3, 2)))
        ax.axhline(0, color="#8A8A8A", lw=0.6)
        ax.axvline(0, color="#8A8A8A", lw=0.6)
        ax.set_xlim(limits)
        ax.set_ylim(limits)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(r"$\log B_{\rm sky}$ at $N_{\rm side}=32$")
        if index == 0:
            ax.set_ylabel(r"$\log B_{\rm sky}$ at $N_{\rm side}=1024$")
        ax.set_title(title, pad=5)
        fraction = 100 * float((~stable).mean())
        ax.text(
            0.04,
            0.96,
            f"{int((~stable).sum())}/{len(frame)} sign flips\n({fraction:.1f}%)",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=6.9,
            fontweight="bold",
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.86,
                "pad": 1.5,
            },
        )
        panel_label(ax, chr(ord("a") + index), x=-0.17)

    audit = pd.read_csv(DATA / "gwtc3_old_failed_pairs_v93_audit.csv")
    nsides = np.array([32, 64, 128, 256, 512, 1024])
    ax = axes[2]
    line_colors = [COLORS["fail"], COLORS["ranking"], COLORS["sky"]]
    for row, color in zip(audit.itertuples(), line_colors):
        values = np.array(
            [
                row.sky_log_bf_nside32,
                row.sky_log_bf_nside64,
                row.sky_log_bf_nside128,
                row.sky_log_bf_nside256,
                row.sky_log_bf_nside512,
                row.sky_log_bf_nside1024,
            ]
        )
        ax.plot(
            nsides,
            values,
            color=color,
            marker="o",
            markersize=3.4,
            lw=1.3,
            label=(
                f"Old rank {int(row.old_consensus_rank_nside32)}"
                f" → new rank {int(row.new_consensus_rank_nside1024)}"
            ),
        )
    ax.axhline(0, color="#606060", lw=0.8, ls=(0, (3, 2)))
    ax.set_xscale("log", base=2)
    ax.set_xticks(nsides, [str(value) for value in nsides])
    ax.set_xlabel(r"HEALPix resolution, $N_{\rm side}$")
    ax.set_ylabel(r"$\log B_{\rm sky}$")
    ax.set_title("Previously anomalous O3 candidates", pad=5)
    legend_labels = [
        r"Old 1 $\rightarrow$ new 50",
        r"Old 3 $\rightarrow$ new 8",
        r"Old 10 $\rightarrow$ new 16",
    ]
    ax.legend(
        ax.lines[:3],
        legend_labels,
        loc="upper right",
        fontsize=6.5,
        handlelength=1.8,
    )
    ax.grid(axis="y", color="#D8D8D8", lw=0.55, alpha=0.7)
    panel_label(ax, "c", x=-0.17, y=1.13)
    export(fig, "fig_supp_sky_resolution_audit")


def plot_seed_stability() -> None:
    metrics = pd.read_csv(DATA / "retrieval_metrics_per_seed.csv")
    metrics = metrics[metrics["method"] == "three_channel"].copy()
    weights = pd.read_csv(DATA / "selected_weights_all_v93.csv")
    weights = weights[weights["objective"] == "retrieval"].copy()

    domains = ["ET3", "gwtc3", "gwtc4"]
    labels = ["ET-3", "O3 noise", "O4a noise"]
    colors = [COLORS["ranking"], COLORS["waveform"], COLORS["time"]]
    x = np.arange(len(domains), dtype=float)

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(FULL_WIDTH, 2.65),
        gridspec_kw={"wspace": 0.42},
    )

    for panel_index, metric in enumerate(("r_at_1", "r_at_10")):
        ax = axes[panel_index]
        for domain_index, (domain, color) in enumerate(zip(domains, colors)):
            values = metrics.loc[metrics["domain"] == domain, metric].to_numpy()
            offsets = np.linspace(-0.07, 0.07, len(values))
            ax.scatter(
                np.full(len(values), x[domain_index]) + offsets,
                values,
                s=24,
                facecolor="white",
                edgecolor=color,
                linewidth=0.9,
                zorder=3,
            )
            ax.errorbar(
                x[domain_index],
                values.mean(),
                yerr=values.std(ddof=1),
                fmt="o",
                ms=4.8,
                color=color,
                capsize=2.5,
                lw=1.0,
                zorder=4,
            )
        ax.set_xticks(x, labels)
        ax.set_ylim(0, 1.04)
        ax.set_ylabel(
            r"$\mathrm{R}@1$" if metric == "r_at_1" else r"$\mathrm{R}@10$"
        )
        ax.grid(axis="y", color="#D8D8D8", lw=0.55, alpha=0.75)
        panel_label(ax, chr(ord("a") + panel_index), x=-0.18)

    ax = axes[2]
    channel_order = ["waveform", "time", "sky"]
    bottoms = np.zeros(len(domains))
    channel_colors = {
        "waveform": COLORS["waveform"],
        "time": COLORS["time"],
        "sky": COLORS["sky"],
    }
    for channel in channel_order:
        means = []
        for domain in domains:
            subset = weights.loc[weights["domain"] == domain, channel_order]
            normalized = subset.div(subset.sum(axis=1), axis=0)
            means.append(float(normalized[channel].mean()))
        ax.bar(
            x,
            means,
            bottom=bottoms,
            width=0.62,
            color=channel_colors[channel],
            edgecolor="white",
            linewidth=0.55,
            label=channel.capitalize(),
        )
        bottoms += np.asarray(means)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Mean normalized coefficient")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.17),
        ncol=3,
        handlelength=1.0,
        columnspacing=0.9,
    )
    panel_label(ax, "c", x=-0.18)
    export(fig, "fig_supp_seed_stability")


def main() -> None:
    configure_style()
    # Figures 2--4 use the audited single-deployment score linkage, ROC/FPR
    # panels and posterior-overlap display introduced on 2026-08-05.
    # Keep their source-data-driven builder authoritative so this legacy
    # publication entry point cannot overwrite them with the earlier layouts.
    from make_main_figures_2_4_20260805 import main as plot_main_figures_2_4

    plot_main_figures_2_4()
    plot_topb()
    plot_catalog_scope_pe()
    plot_seed_stability()
    plot_sky_resolution_audit()


if __name__ == "__main__":
    main()
