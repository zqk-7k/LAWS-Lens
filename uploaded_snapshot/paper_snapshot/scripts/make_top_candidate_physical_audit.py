from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "source_data"
OUT = ROOT / "figures" / "fig_top_candidate_physical_audit"

PARAMETERS = [
    ("chirp_mass", r"$\mathcal{M}_c$"),
    ("mass_ratio", r"$q$"),
    ("chi_eff", r"$\chi_{\rm eff}$"),
    ("luminosity_distance", r"$d_L$"),
]

C_EVENT_1 = "#2F6B8A"
C_EVENT_2 = "#D87528"
C_ACCENT = "#B54A32"
C_GRID = "#D8DDE3"
C_TEXT = "#202427"
C_MUTED = "#5F666B"
CMAP = LinearSegmentedColormap.from_list(
    "posterior_overlap", ["#F3F5F6", "#C5DCE7", "#6FA5BF", "#2F6B8A"]
)


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 7.4,
            "axes.titlesize": 8.6,
            "axes.labelsize": 8.0,
            "axes.labelweight": "bold",
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
        }
    )


def panel_label(ax: mpl.axes.Axes, label: str, x: float = -0.10) -> None:
    ax.text(
        x,
        1.06,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.0,
        fontweight="bold",
        color=C_TEXT,
    )


def compact_event(event: str) -> str:
    return event.replace("GW", "", 1).split("_", 1)[0]


def pair_overlaps(summary: pd.DataFrame, catalog: str) -> pd.DataFrame:
    frame = summary[
        (summary["catalog"] == catalog)
        & summary["parameter"].isin([name for name, _ in PARAMETERS])
    ].drop_duplicates(["rank", "event_i", "event_j", "parameter"])
    matrix = frame.pivot(
        index=["rank", "event_i", "event_j"],
        columns="parameter",
        values="pair_overlap_coefficient",
    ).reset_index()
    matrix = matrix.sort_values("rank").reset_index(drop=True)
    intrinsic = ["chirp_mass", "mass_ratio", "chi_eff"]
    matrix["minimum_intrinsic_overlap"] = matrix[intrinsic].min(axis=1)
    return matrix


def plot_overlap_heatmap(
    ax: mpl.axes.Axes,
    matrix: pd.DataFrame,
    title: str,
    label: str,
) -> int:
    values = matrix[[name for name, _ in PARAMETERS]].to_numpy(dtype=float)
    image = ax.imshow(values, vmin=0.0, vmax=1.0, cmap=CMAP, aspect="auto")
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            color = "white" if value >= 0.58 else C_TEXT
            ax.text(column, row, f"{value:.2f}", ha="center", va="center", color=color, fontsize=7.0)

    selected_index = int(matrix["minimum_intrinsic_overlap"].idxmax())
    selected_row = matrix.index.get_loc(selected_index)
    ax.add_patch(
        Rectangle(
            (-0.49, selected_row - 0.49),
            3.0 - 0.02,
            0.98,
            fill=False,
            edgecolor=C_ACCENT,
            linewidth=1.3,
        )
    )
    ax.set_xticks(np.arange(len(PARAMETERS)), [label_text for _, label_text in PARAMETERS])
    ax.set_yticks(np.arange(len(matrix)), [f"Rank {int(rank)}" for rank in matrix["rank"]])
    ax.get_yticklabels()[selected_row].set_color(C_ACCENT)
    ax.get_yticklabels()[selected_row].set_fontweight("bold")
    ax.tick_params(length=0)
    ax.set_title(title, pad=6, fontweight="bold")
    for spine in ax.spines.values():
        spine.set_visible(False)
    panel_label(ax, label)
    return int(matrix.iloc[selected_row]["rank"]), image


def available_detectors(
    detector_audit: pd.DataFrame,
    catalog: str,
    event: str,
) -> list[str]:
    frame = detector_audit[
        (detector_audit["catalog"] == catalog)
        & (detector_audit["event"] == event)
        & detector_audit["usable_for_display"].astype(bool)
    ]
    return [detector for detector in ("H1", "L1") if detector in set(frame["detector"])]


def plot_waveform_pair(
    ax: mpl.axes.Axes,
    arrays: np.lib.npyio.NpzFile,
    detector_audit: pd.DataFrame,
    pair: pd.Series,
    title: str,
    label: str,
) -> None:
    tracks: list[tuple[str, str, np.ndarray, np.ndarray, str]] = []
    detector_sets: list[set[str]] = []
    for event, color in ((pair.event_i, C_EVENT_1), (pair.event_j, C_EVENT_2)):
        detectors = available_detectors(detector_audit, pair.catalog, event)
        detector_sets.append(set(detectors))
        for detector in detectors:
            time = np.asarray(arrays[f"{event}_{detector}_time"], dtype=float)
            strain = np.asarray(arrays[f"{event}_{detector}_white"], dtype=float)
            keep = (time >= -0.8) & (time <= 0.15)
            time = time[keep]
            strain = np.nan_to_num(strain[keep], nan=0.0, posinf=0.0, neginf=0.0)
            scale = np.quantile(np.abs(strain), 0.995)
            normalized = np.clip(strain / max(scale, 1e-12), -1.25, 1.25)
            tracks.append((event, detector, time, normalized, color))

    offsets = np.arange(len(tracks) - 1, -1, -1, dtype=float)
    for offset, (event, detector, time, normalized, color) in zip(offsets, tracks):
        ax.axhline(offset, color=C_GRID, lw=0.45, zorder=0)
        ax.plot(time[::2], offset + 0.34 * normalized[::2], color=color, lw=0.58, alpha=0.96)
        ax.text(
            -0.955,
            offset,
            f"{compact_event(event)}  {detector}",
            ha="left",
            va="center",
            color=color,
            fontsize=6.2,
        )

    common = detector_sets[0] & detector_sets[1]
    ax.axvline(0.0, color="#6B7280", lw=0.8, ls=(0, (3, 2)), zorder=0)
    ax.set_xlim(-0.98, 0.15)
    ax.set_ylim(-0.55, max(len(tracks) - 0.45, 0.55))
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Time from trigger (s)")
    ax.set_title(title, pad=6, fontweight="bold")
    ax.text(
        0.985,
        0.955,
        f"waveform embedding score = {pair.waveform_score:.3f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        color=C_MUTED,
        fontsize=6.6,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 1.2},
    )
    if not common:
        ax.text(
            0.985,
            0.06,
            "No common finite detector data",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            color=C_ACCENT,
            fontsize=6.5,
            fontweight="bold",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 1.2},
        )
    panel_label(ax, label)


def main() -> None:
    configure_style()
    scores = pd.read_csv(DATA / "top5_candidate_score_records.csv")
    summary = pd.read_csv(DATA / "top5_pe_summary.csv")
    detector_audit = pd.read_csv(DATA / "top5_detector_window_audit.csv")
    arrays = np.load(DATA / "top5_whitened_strain_windows.npz")

    o3 = pair_overlaps(summary, "GWTC-3 O3")
    o4 = pair_overlaps(summary, "GWTC-4.1 O4a")

    width = 183.0 / 25.4
    height = 111.0 / 25.4
    fig = plt.figure(figsize=(width, height))
    grid = fig.add_gridspec(
        2,
        2,
        height_ratios=[0.92, 1.08],
        left=0.09,
        right=0.965,
        bottom=0.12,
        top=0.93,
        hspace=0.58,
        wspace=0.36,
    )

    ax_a = fig.add_subplot(grid[0, 0])
    o3_rank, image = plot_overlap_heatmap(ax_a, o3, "GWTC-3 top five posterior overlap", "a")
    ax_b = fig.add_subplot(grid[0, 1])
    o4_rank, _ = plot_overlap_heatmap(ax_b, o4, "GWTC-4.1 O4a top five posterior overlap", "b")

    colorbar_ax = fig.add_axes([0.968, 0.555, 0.012, 0.315])
    colorbar = fig.colorbar(image, cax=colorbar_ax)
    colorbar.set_label("Posterior overlap coefficient", fontweight="bold", fontsize=7.0)
    colorbar.ax.tick_params(labelsize=6.3, width=0.6, length=2.5)

    o3_pair = scores[(scores["catalog"] == "GWTC-3 O3") & (scores["rank"] == o3_rank)].iloc[0]
    o4_pair = scores[(scores["catalog"] == "GWTC-4.1 O4a") & (scores["rank"] == o4_rank)].iloc[0]
    ax_c = fig.add_subplot(grid[1, 0])
    plot_waveform_pair(
        ax_c,
        arrays,
        detector_audit,
        o3_pair,
        f"GWTC-3 rank {o3_rank}: strongest intrinsic screen",
        "c",
    )
    ax_d = fig.add_subplot(grid[1, 1])
    plot_waveform_pair(
        ax_d,
        arrays,
        detector_audit,
        o4_pair,
        f"GWTC-4.1 O4a rank {o4_rank}: strongest intrinsic screen",
        "d",
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".png"), dpi=400, bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
