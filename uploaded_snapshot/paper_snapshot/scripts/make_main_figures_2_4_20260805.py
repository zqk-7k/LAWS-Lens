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
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import roc_auc_score, roc_curve


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT
CURRENT = ROOT / "source_data" / "current_v93"


def first_existing_path(*candidates: Path) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


ET3_ROOT = first_existing_path(
    ROOT / "server_pull" / "et3_sky_resolution_v93_20260730",
    ROOT.parent / "server_pull" / "et3_sky_resolution_v93_20260730",
)
GWTC_ROOT = first_existing_path(
    ROOT
    / "sky_resolution_v93_20260730"
    / "server_pull"
    / "gwtc_sky_resolution_v93_20260730",
    ROOT.parent
    / "sky_resolution_v93_20260730"
    / "server_pull"
    / "gwtc_sky_resolution_v93_20260730",
)
OUT = ROOT / "figures"
SOURCE = CURRENT
FULL_WIDTH = 183.0 / 25.4


COLORS = {
    "background": "#D6DBDF",
    "background_edge": "#8B969E",
    "sis": "#356FA8",
    "pm": "#D8752D",
    "waveform": "#0072B2",
    "time": "#D55E00",
    "sky": "#009E73",
    "ranking": "#6F4C9B",
    "candidate": "#1B8A78",
    "neutral": "#666666",
    "fail": "#B54A32",
    "wave_a": "#2C7FB8",
    "wave_b": "#F28E2B",
    "pe_overlay": "#C51B4A",
}
DOMAIN_LABELS = {
    "ET3": "ET-3",
    "gwtc3": "O3-noise test",
    "gwtc4": "O4a-noise test",
}
DOMAIN_SHORT = {"ET3": "ET-3", "gwtc3": "O3", "gwtc4": "O4a"}
DOMAIN_EVENTS = {"ET3": 9000, "gwtc3": 450, "gwtc4": 450}
METHODS = ["waveform_only", "time_only", "sky_only", "three_channel"]
METHOD_LABELS = {
    "waveform_only": "Waveform",
    "time_only": "Time delay",
    "sky_only": "Sky position",
    "three_channel": "Three-channel score",
}
METHOD_COLORS = {
    "waveform_only": COLORS["waveform"],
    "time_only": COLORS["time"],
    "sky_only": COLORS["sky"],
    "three_channel": COLORS["ranking"],
}
METHOD_STYLES = {
    "waveform_only": "-",
    "time_only": (0, (4, 2)),
    "sky_only": (0, (1.4, 1.4)),
    "three_channel": "-",
}
METHOD_MARKERS = {
    "waveform_only": "o",
    "time_only": "s",
    "sky_only": "^",
    "three_channel": "D",
}
SEEDS = {
    "ET3": ["202607251", "202607252", "202607253", "202607254", "202607255"],
    "gwtc3": ["202607241", "202607242", "202607243"],
    "gwtc4": ["202607241", "202607242", "202607243"],
}
SELECTED_DEPLOYMENTS = {
    "ET3": "202607251",
    "gwtc3": "202607241",
    "gwtc4": "202607242",
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
            "font.serif": ["Times New Roman", "Times"],
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "mathtext.fontset": "stix",
            "font.size": 7.8,
            "axes.labelsize": 8.2,
            "axes.labelweight": "bold",
            "axes.titlesize": 8.8,
            "axes.titleweight": "bold",
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 6.8,
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
        fontsize=10.0,
        fontweight="bold",
    )


def export(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.png", dpi=400, bbox_inches="tight")
    plt.close(fig)


def load_weights() -> pd.DataFrame:
    frame = pd.read_csv(CURRENT / "selected_weights_all_v93.csv", dtype={"seed": str})
    frame["seed"] = frame["seed"].astype(str)
    return frame


def load_pair_table(domain: str, seed: str) -> pd.DataFrame:
    if domain == "ET3":
        path = (
            ET3_ROOT
            / "figure_source_data"
            / f"seed_{seed}_pair_diagnostics_sample_v93.parquet"
        )
    else:
        path = (
            GWTC_ROOT
            / domain
            / f"seed_{seed}"
            / "fusion_heldout_test_pairs_v81.parquet"
        )
    return pd.read_parquet(path)


def method_score(
    frame: pd.DataFrame,
    domain: str,
    seed: str,
    method: str,
    weights: pd.DataFrame,
    objective: str = "retrieval",
) -> np.ndarray:
    if method == "waveform_only":
        return frame["waveform_score"].to_numpy(dtype=float)
    if method == "time_only":
        return frame["time_score"].to_numpy(dtype=float)
    if method == "sky_only":
        return frame["sky_score"].to_numpy(dtype=float)
    if domain == "ET3" and objective == "retrieval" and "ranking_score" in frame:
        return frame["ranking_score"].to_numpy(dtype=float)
    row = weights[
        (weights["domain"] == domain)
        & (weights["seed"] == seed)
        & (weights["objective"] == objective)
    ].iloc[0]
    return (
        float(row["waveform"]) * frame["waveform_score"].to_numpy(dtype=float)
        + float(row["time"]) * frame["time_score"].to_numpy(dtype=float)
        + float(row["sky"]) * frame["sky_score"].to_numpy(dtype=float)
    )


def threshold_at_recall(
    scores: np.ndarray, labels: np.ndarray, target: float
) -> dict[str, float]:
    labels = labels.astype(bool)
    true_scores = np.sort(scores[labels])[::-1]
    target_count = int(np.ceil(target * true_scores.size))
    threshold = float(true_scores[target_count - 1])
    selected = scores >= threshold
    tp = int(np.sum(selected & labels))
    fp = int(np.sum(selected & ~labels))
    return {
        "threshold": threshold,
        "true_pairs": tp,
        "false_pairs": fp,
        "recall": tp / int(labels.sum()),
        "fpr": fp / int((~labels).sum()),
    }


def build_operating_points(weights: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    seed = SELECTED_DEPLOYMENTS["ET3"]
    path = (
        ET3_ROOT
        / "extracted"
        / "ET3_sky_resolution_v93_20260730T072501Z"
        / "results"
        / f"seed_{seed}"
        / "formal"
        / "pair_level_operating_points_strict_positive_v93.csv"
    )
    table = pd.read_csv(path)
    row = table[
        (table["operating_point"] == "recall") & np.isclose(table["target"], 0.9)
    ].iloc[0]
    records.append(
        {
            "domain": "ET3",
            "seed": seed,
            "kind": "90% detection threshold",
            "threshold": float(row["threshold"]),
            "false_pairs": int(row["false_pairs"]),
            "true_pairs": int(row["true_pairs"]),
            "fpr": float(row["fpr"]),
            "recall": float(row["recall"]),
            "score_value": float(row["threshold"]),
        }
    )

    for domain in ("gwtc3", "gwtc4"):
        seed = SELECTED_DEPLOYMENTS[domain]
        frame = load_pair_table(domain, seed)
        labels = frame["is_true_pair"].to_numpy(dtype=int)
        scores = method_score(
            frame, domain, seed, "three_channel", weights, "candidate"
        )
        point = threshold_at_recall(scores, labels, 0.9)
        records.append(
            {
                "domain": domain,
                "seed": seed,
                "kind": "90% detection threshold",
                **point,
                "score_value": float(point["threshold"]),
            }
        )
        stability = pd.read_parquet(
            GWTC_ROOT / domain / "real_candidate_rank_stability_v81.parquet"
        )
        top10 = stability[stability["seed"].astype(str) == seed].sort_values("rank").head(10)
        rank10 = top10.iloc[9]
        rank10_score = float(rank10["final_score"])
        selected = scores >= rank10_score
        labels_bool = labels.astype(bool)
        records.append(
            {
                "domain": domain,
                "seed": seed,
                "kind": "rank-10 score",
                "threshold": rank10_score,
                "false_pairs": int(np.sum(selected & ~labels_bool)),
                "true_pairs": int(np.sum(selected & labels_bool)),
                "fpr": float(np.sum(selected & ~labels_bool) / np.sum(~labels_bool)),
                "recall": float(np.sum(selected & labels_bool) / np.sum(labels_bool)),
                "score_value": rank10_score,
            }
        )

    result = pd.DataFrame.from_records(records)
    SOURCE.mkdir(parents=True, exist_ok=True)
    result.to_csv(SOURCE / "fig2_selected_score_thresholds.csv", index=False)
    return result


def build_selected_fpr_curves(weights: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for domain in ("gwtc3", "gwtc4"):
        seed = SELECTED_DEPLOYMENTS[domain]
        frame = load_pair_table(domain, seed)
        labels = frame["is_true_pair"].to_numpy(dtype=bool)
        scores = method_score(
            frame, domain, seed, "three_channel", weights, "candidate"
        )
        background = np.sort(scores[~labels])[::-1]
        n_background = background.size
        geometric_ranks = np.geomspace(1, n_background, 1100).astype(int)
        linear_ranks = np.linspace(1, n_background, 900).astype(int)
        ranks = np.unique(np.concatenate([geometric_ranks, linear_ranks]))
        for rank in ranks:
            records.append(
                {
                    "domain": domain,
                    "seed": seed,
                    "score_threshold": float(background[rank - 1]),
                    "fpr": float(rank / n_background),
                    "fpr_percent": float(100.0 * rank / n_background),
                    "false_pairs": int(rank),
                    "n_background": int(n_background),
                }
            )
    result = pd.DataFrame.from_records(records).sort_values(
        ["domain", "score_threshold"]
    )
    SOURCE.mkdir(parents=True, exist_ok=True)
    result.to_csv(SOURCE / "fig2_selected_seed_score_fpr_curves.csv", index=False)
    return result


def build_selected_recovery_points(weights: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for domain in ("gwtc3", "gwtc4"):
        seed = SELECTED_DEPLOYMENTS[domain]
        frame = load_pair_table(domain, seed)
        labels = frame["is_true_pair"].to_numpy(dtype=int)
        scores = method_score(
            frame, domain, seed, "three_channel", weights, "candidate"
        )
        for target in (0.90, 0.95, 0.99):
            point = threshold_at_recall(scores, labels, target)
            records.append(
                {
                    "domain": domain,
                    "seed": seed,
                    "target_recall": target,
                    "score_threshold": float(point["threshold"]),
                    "actual_recall": float(point["recall"]),
                    "fpr": float(point["fpr"]),
                    "fpr_percent": float(100.0 * point["fpr"]),
                    "false_pairs": int(point["false_pairs"]),
                    "true_pairs": int(point["true_pairs"]),
                }
            )
    result = pd.DataFrame.from_records(records)
    SOURCE.mkdir(parents=True, exist_ok=True)
    result.to_csv(
        SOURCE / "fig2_selected_seed_companion_recovery_points.csv", index=False
    )
    return result


def build_selected_density(weights: pd.DataFrame) -> pd.DataFrame:
    ranges = {
        "waveform": (-12.0, 14.0),
        "time": (-7.0, 20.0),
        "sky": (-8.0, 9.0),
        "ranking": (-12.0, 20.0),
    }
    records: list[dict[str, object]] = []
    for domain in ("ET3", "gwtc3", "gwtc4"):
        seed = SELECTED_DEPLOYMENTS[domain]
        frame = load_pair_table(domain, seed)
        if domain == "ET3":
            row = weights[
                (weights["domain"] == domain)
                & (weights["seed"] == seed)
                & (weights["objective"] == "retrieval")
            ].iloc[0]
            coefficient_norm = float(
                np.linalg.norm([row["waveform"], row["time"], row["sky"]])
            )
            ranking = frame["ranking_score"].to_numpy(dtype=float) * coefficient_norm
        else:
            ranking = method_score(
                frame, domain, seed, "three_channel", weights, "candidate"
            )
        score_values = {
            "waveform": frame["waveform_score"].to_numpy(dtype=float),
            "time": frame["time_score"].to_numpy(dtype=float),
            "sky": frame["sky_score"].to_numpy(dtype=float),
            "ranking": ranking,
        }
        labels = frame["is_true_pair"].to_numpy(dtype=bool)
        family = frame["true_pair_family"].astype(str).str.upper().to_numpy()
        class_masks = {
            "Non-companion": ~labels,
            "SIS": labels & (family == "SIS"),
            "PM": labels & (family == "PM"),
        }
        for score_name, values in score_values.items():
            lower, upper = ranges[score_name]
            edges = np.linspace(lower, upper, 261)
            centers = 0.5 * (edges[:-1] + edges[1:])
            epsilon = 1e-7 * (upper - lower)
            for pair_class, mask in class_masks.items():
                selected_values = values[mask]
                if score_name == "ranking":
                    selected_values = selected_values[
                        (selected_values >= lower) & (selected_values <= upper)
                    ]
                else:
                    selected_values = np.clip(
                        selected_values, lower + epsilon, upper - epsilon
                    )
                histogram, _ = np.histogram(selected_values, bins=edges, density=True)
                smooth = gaussian_filter1d(histogram.astype(float), sigma=1.45)
                if smooth.max() > 0:
                    smooth /= smooth.max()
                records.extend(
                    {
                        "domain": domain,
                        "seed": seed,
                        "score": score_name,
                        "pair_class": pair_class,
                        "x": float(x),
                        "density_mean": float(y),
                        "density_min": float(y),
                        "density_max": float(y),
                    }
                    for x, y in zip(centers, smooth)
                )
    result = pd.DataFrame.from_records(records)
    SOURCE.mkdir(parents=True, exist_ok=True)
    result.to_csv(SOURCE / "fig2_selected_seed_density.csv", index=False)
    return result


def build_roc_source(weights: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    fpr_grid = np.geomspace(1e-5, 1.0, 300)
    curve_records: list[dict[str, object]] = []
    metric_records: list[dict[str, object]] = []
    for domain in ("ET3", "gwtc3", "gwtc4"):
        for seed in SEEDS[domain]:
            frame = load_pair_table(domain, seed)
            labels = frame["is_true_pair"].to_numpy(dtype=int)
            for method in METHODS:
                scores = method_score(frame, domain, seed, method, weights)
                fpr, tpr, _ = roc_curve(labels, scores, drop_intermediate=False)
                unique_fpr, first = np.unique(fpr, return_index=True)
                max_tpr = np.maximum.reduceat(tpr, first)
                interpolated = np.interp(fpr_grid, unique_fpr, max_tpr)
                auc = float(roc_auc_score(labels, scores))
                metric_records.append(
                    {
                        "domain": domain,
                        "seed": seed,
                        "method": method,
                        "roc_auc": auc,
                        "n_true_pairs": int(labels.sum()),
                        "n_false_pairs": int((labels == 0).sum()),
                        "false_sampling": (
                            "fixed random sample" if domain == "ET3" else "all pairs"
                        ),
                    }
                )
                curve_records.extend(
                    {
                        "domain": domain,
                        "seed": seed,
                        "method": method,
                        "fpr": float(x),
                        "detection_probability": float(y),
                    }
                    for x, y in zip(fpr_grid, interpolated)
                )
    curves = pd.DataFrame.from_records(curve_records)
    metrics = pd.DataFrame.from_records(metric_records)
    SOURCE.mkdir(parents=True, exist_ok=True)
    curves.to_csv(SOURCE / "fig3_roc_curves_per_seed.csv", index=False)
    metrics.to_csv(SOURCE / "fig3_roc_metrics_per_seed.csv", index=False)
    return curves, metrics


def summarize_retrieval_curves() -> pd.DataFrame:
    frame = pd.read_csv(CURRENT / "retrieval_curves_per_seed.csv", dtype={"seed": str})
    summary = (
        frame.groupby(["domain", "method", "rank_cutoff"], as_index=False)["recall"]
        .agg(["mean", "min", "max", "std"])
        .reset_index()
        .rename(
            columns={
                "mean": "recall_mean",
                "min": "recall_min",
                "max": "recall_max",
                "std": "recall_std",
            }
        )
    )
    SOURCE.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SOURCE / "fig3_retrieval_curves_summary.csv", index=False)
    return summary


def draw_ridge_panel(
    ax: plt.Axes,
    density: pd.DataFrame,
    score: str,
    title: str,
    xlabel: str,
    limits: tuple[float, float],
    ticks: list[float],
    label: str,
) -> None:
    bases = {"ET3": 2.0, "gwtc3": 1.0, "gwtc4": 0.0}
    for domain, base in bases.items():
        ax.axhline(base, color="#B9C1C7", lw=0.55, zorder=0)
        for pair_class in ("Non-companion", "SIS", "PM"):
            subset = density[
                (density["domain"] == domain)
                & (density["score"] == score)
                & (density["pair_class"] == pair_class)
            ]
            x = subset["x"].to_numpy(dtype=float)
            mean = subset["density_mean"].to_numpy(dtype=float)
            low = subset["density_min"].to_numpy(dtype=float)
            high = subset["density_max"].to_numpy(dtype=float)
            if pair_class == "Non-companion":
                ax.fill_between(
                    x,
                    base,
                    base + 0.68 * mean,
                    facecolor=COLORS["background"],
                    edgecolor=COLORS["background_edge"],
                    linewidth=0.5,
                    alpha=0.85,
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
                    linewidth=1.25,
                    zorder=3,
                )
    ax.axvline(0, color=COLORS["neutral"], lw=0.65, ls=(0, (2, 2)), alpha=0.8)
    ax.set_xlim(limits)
    ax.set_ylim(-0.04, 2.78)
    ax.set_title(title, pad=7)
    ax.set_xlabel(xlabel, labelpad=5)
    ax.set_xticks(ticks)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["O4a-noise", "O3-noise", "ET-3"])
    panel_label(ax, label)


def plot_figure2(
    density: pd.DataFrame,
    operating: pd.DataFrame,
    fpr_curves: pd.DataFrame,
    recovery_points: pd.DataFrame,
) -> None:
    fig = plt.figure(figsize=(0.95 * FULL_WIDTH, 4.75))
    grid = fig.add_gridspec(
        2,
        4,
        width_ratios=[1.0, 1.0, 1.0, 1.06],
        height_ratios=[1.9, 1.0],
        left=0.105,
        right=0.985,
        bottom=0.105,
        top=0.84,
        wspace=0.33,
        hspace=0.68,
    )
    specs = [
        ("waveform", "Waveform evidence", r"$\mathbf{Z}_{\rm wf}$", (-12, 14), [-12, -5, 0, 5, 10]),
        ("time", "Time-delay evidence", r"$\mathbf{Z}_{\rm time}$", (-7, 20), [-5, 0, 5, 10, 15, 20]),
        ("sky", "Sky-position evidence", r"$\mathbf{Z}_{\rm sky}$", (-8, 9), [-8, -4, 0, 4, 8]),
        ("ranking", "Three-channel score", r"$\mathbf{S}_{ij}$", (-12, 20), [-12, -5, 0, 5, 10, 15, 20]),
    ]
    axes: list[plt.Axes] = []
    for index, spec in enumerate(specs):
        ax = fig.add_subplot(grid[0, index])
        axes.append(ax)
        draw_ridge_panel(ax, density, *spec, chr(ord("a") + index))
        if index > 0:
            ax.tick_params(axis="y", left=False, labelleft=False)
            ax.spines["left"].set_visible(False)
        else:
            ax.set_ylabel("Held-out test domain", labelpad=7)

    score_ax = axes[3]
    bases = {"ET3": 2.0, "gwtc3": 1.0, "gwtc4": 0.0}
    for domain, base in bases.items():
        rank10 = operating[
            (operating["domain"] == domain) & (operating["kind"] == "rank-10 score")
        ]
        if not rank10.empty:
            rank10_row = rank10.iloc[0]
            score = float(rank10_row["score_value"])
            score_ax.vlines(
                score,
                base,
                base + 0.52,
                color=COLORS["candidate"],
                linewidth=0.95,
                linestyle=(0, (3, 2)),
                zorder=5,
            )

    calibration_axes = [
        fig.add_subplot(grid[1, 0:2]),
        fig.add_subplot(grid[1, 2:4]),
    ]
    for index, (ax, domain) in enumerate(
        zip(calibration_axes, ("gwtc3", "gwtc4"))
    ):
        curve = fpr_curves[fpr_curves["domain"] == domain].sort_values(
            "score_threshold"
        )
        ax.plot(
            curve["score_threshold"],
            curve["fpr_percent"],
            color="#56636C",
            linewidth=1.35,
            zorder=2,
        )
        ax.set_yscale("log")
        ax.set_xlim(-3.0, 2.8)
        ax.set_ylim(8e-4, 100.0)
        ax.set_xticks([-3, -2, -1, 0, 1, 2])
        ax.set_yticks([0.001, 0.01, 0.1, 1, 10, 100])
        ax.set_yticklabels(["0.001", "0.01", "0.1", "1", "10", "100"])
        ax.grid(True, which="major", color="#DDE2E5", linewidth=0.55, zorder=0)
        ax.set_xlabel(r"Catalog score threshold, $s$")
        ax.set_title(f"{DOMAIN_SHORT[domain]} background calibration", pad=6)
        if index == 0:
            ax.set_ylabel("Injection-background FPR (%)", labelpad=6)
        else:
            ax.tick_params(axis="y", left=False, labelleft=False)
            ax.spines["left"].set_visible(False)
        panel_label(ax, chr(ord("e") + index), x=-0.10, y=1.04)

        operating_domain = operating[operating["domain"] == domain]
        rank10 = operating_domain[
            operating_domain["kind"] == "rank-10 score"
        ].iloc[0]
        recovery_domain = recovery_points[
            recovery_points["domain"] == domain
        ].sort_values("target_recall")
        for _, row in recovery_domain.iterrows():
            score_value = float(row["score_threshold"])
            fpr_percent = float(row["fpr_percent"])
            target_percent = int(round(100.0 * float(row["target_recall"])))
            ax.scatter(
                [score_value],
                [fpr_percent],
                s=22,
                marker="D",
                facecolor=COLORS["ranking"],
                edgecolor="white",
                linewidth=0.5,
                zorder=5,
            )
            if target_percent == 90:
                xytext, ha = (5, -4), "left"
            elif target_percent == 95:
                xytext, ha = (-5, -4), "right"
            else:
                xytext, ha = (5, -4), "left"
            ax.annotate(
                f"{target_percent}%",
                xy=(score_value, fpr_percent),
                xytext=xytext,
                textcoords="offset points",
                ha=ha,
                va="top",
                fontsize=6.0,
                color=COLORS["ranking"],
            )

        score_value = float(rank10["score_value"])
        fpr_percent = 100.0 * float(rank10["fpr"])
        ax.vlines(
            score_value,
            8e-4,
            fpr_percent,
            color=COLORS["candidate"],
            linewidth=0.8,
            linestyle=(0, (3, 2)),
            zorder=3,
        )
        ax.hlines(
            fpr_percent,
            -3.0,
            score_value,
            color=COLORS["candidate"],
            linewidth=0.8,
            linestyle=(0, (3, 2)),
            zorder=3,
        )
        ax.scatter(
            [score_value],
            [fpr_percent],
            s=24,
            marker="o",
            facecolor=COLORS["candidate"],
            edgecolor="white",
            linewidth=0.55,
            zorder=5,
        )
        ax.annotate(
            rf"rank 10: $S={score_value:.2f}$" + "\n" + f"FPR={fpr_percent:.3g}%",
            xy=(score_value, fpr_percent),
            xytext=(-5, -7),
            textcoords="offset points",
            ha="right",
            va="top",
            fontsize=6.1,
            color=COLORS["candidate"],
        )

    fig.legend(
        handles=[
            Patch(
                facecolor=COLORS["background"],
                edgecolor=COLORS["background_edge"],
                label="Non-companion pairs",
            ),
            Line2D([0], [0], color=COLORS["sis"], lw=1.4, label="SIS companions"),
            Line2D([0], [0], color=COLORS["pm"], lw=1.4, ls="--", label="PM companions"),
            Line2D(
                [0],
                [0],
                color=COLORS["ranking"],
                lw=0,
                marker="D",
                markersize=4.5,
                label="90/95/99% companion recovery",
            ),
            Line2D(
                [0],
                [0],
                color=COLORS["candidate"],
                linestyle=(0, (3, 2)),
                label="Real-catalog rank-10 score",
            ),
        ],
        loc="upper center",
        bbox_to_anchor=(0.54, 0.985),
        ncol=5,
        columnspacing=0.95,
        handlelength=2.0,
    )
    export(fig, "fig_channel_separation_corner")


def plot_figure3(
    roc_source: pd.DataFrame,
    roc_metrics: pd.DataFrame,
    retrieval: pd.DataFrame,
) -> None:
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(FULL_WIDTH, 5.15),
        gridspec_kw={
            "left": 0.095,
            "right": 0.985,
            "bottom": 0.105,
            "top": 0.84,
            "wspace": 0.31,
            "hspace": 0.62,
        },
    )
    domains = ("ET3", "gwtc3", "gwtc4")
    for column, domain in enumerate(domains):
        ax = axes[0, column]
        for method in METHODS:
            subset = roc_source[
                (roc_source["domain"] == domain) & (roc_source["method"] == method)
            ]
            grouped = subset.groupby("fpr")["detection_probability"]
            mean = grouped.mean()
            x = mean.index.to_numpy(dtype=float)
            linewidth = 1.65 if method == "three_channel" else 1.05
            alpha = 1.0 if method == "three_channel" else 0.82
            ax.plot(
                x,
                mean.to_numpy(dtype=float),
                color=METHOD_COLORS[method],
                linestyle=METHOD_STYLES[method],
                linewidth=linewidth,
                alpha=alpha,
            )
        auc_values = roc_metrics[
            (roc_metrics["domain"] == domain)
            & (roc_metrics["method"] == "three_channel")
        ]["roc_auc"]
        auc_format = f"{auc_values.mean():.5f}" if auc_values.mean() > 0.9995 else f"{auc_values.mean():.3f}"
        ax.text(
            0.04,
            0.06,
            rf"three-channel AUC $={auc_format}$",
            transform=ax.transAxes,
            fontsize=6.5,
            color=COLORS["ranking"],
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.86, "pad": 1.2},
        )
        ax.set_xscale("log")
        ax.set_xlim(1e-6, 1.0)
        ax.set_ylim(0, 1.02)
        ax.set_title(DOMAIN_LABELS[domain], pad=24)
        ax.set_xlabel("Pair false-positive rate")
        if column == 0:
            ax.set_ylabel("Detection probability (TPR)")
        else:
            ax.tick_params(axis="y", labelleft=False)
        ax.grid(color="#E1E3E5", lw=0.48, which="major")
        n_false = 40_492_500 if domain == "ET3" else 100_845
        secondary = ax.secondary_xaxis(
            "top",
            functions=(lambda value, n=n_false: value * n, lambda value, n=n_false: value / n),
        )
        secondary.set_xscale("log")
        secondary.tick_params(labelsize=5.8, pad=1.5, length=2.2)
        if domain == "ET3":
            secondary.set_xticks([100, 10000, 1000000])
            secondary.set_xticklabels([r"$10^2$", r"$10^4$", r"$10^6$"])
        else:
            secondary.set_xticks([0.1, 10, 1000, 100000])
            secondary.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
        if column == 1:
            secondary.set_xlabel("Expected false pairs at threshold", fontsize=6.6, labelpad=3)
        panel_label(ax, chr(ord("a") + column), x=-0.17, y=1.16)

        ax = axes[1, column]
        for method in METHODS:
            subset = retrieval[
                (retrieval["domain"] == domain) & (retrieval["method"] == method)
            ].sort_values("rank_cutoff")
            x = subset["rank_cutoff"].to_numpy(dtype=float)
            ax.plot(
                x,
                subset["recall_mean"].to_numpy(dtype=float),
                color=METHOD_COLORS[method],
                linestyle=METHOD_STYLES[method],
                marker=METHOD_MARKERS[method],
                markersize=3.0,
                linewidth=1.5 if method == "three_channel" else 1.0,
                alpha=1.0 if method == "three_channel" else 0.82,
            )
        ax.axvline(10, color="#9B9B9B", lw=0.55, ls=(0, (2, 2)), zorder=0)
        ax.set_xscale("log")
        ax.set_xlim(0.9, 55)
        ax.set_ylim(0, 1.03)
        ax.set_xticks([1, 2, 5, 10, 20, 50])
        ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
        ax.set_xlabel(r"Rank cutoff, $K$")
        if column == 0:
            ax.set_ylabel(r"Companion recall, R@$K$")
        else:
            ax.tick_params(axis="y", labelleft=False)
        ax.grid(axis="y", color="#E1E3E5", lw=0.48)
        denominator = DOMAIN_EVENTS[domain] - 1
        secondary = ax.secondary_xaxis(
            "top",
            functions=(
                lambda value, n=denominator: 100.0 * value / n,
                lambda value, n=denominator: value * n / 100.0,
            ),
        )
        secondary.set_xscale("log")
        percent_ticks = np.array([1, 10, 50], dtype=float) * 100.0 / denominator
        secondary.set_xticks(percent_ticks)
        secondary.set_xticklabels([f"{value:.3g}" for value in percent_ticks])
        secondary.tick_params(labelsize=5.8, pad=1.5, length=2.2)
        if column == 1:
            secondary.set_xlabel("Catalog fraction searched (%)", fontsize=6.6, labelpad=3)
        panel_label(ax, chr(ord("d") + column), x=-0.17, y=1.13)

    fig.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color=METHOD_COLORS[method],
                linestyle=METHOD_STYLES[method],
                marker=METHOD_MARKERS[method],
                markersize=3.5,
                linewidth=1.5 if method == "three_channel" else 1.1,
                label=METHOD_LABELS[method],
            )
            for method in METHODS
        ],
        loc="upper center",
        bbox_to_anchor=(0.53, 0.985),
        ncol=4,
        columnspacing=1.15,
        handlelength=2.1,
    )
    export(fig, "fig_retrieval_pair_curves_current")


def display_event(event: str) -> str:
    return DISPLAY_EVENT_NAMES.get(str(event), str(event))


def pair_label(row: object) -> str:
    return f"{int(row.display_rank):02d}  {display_event(row.event_i)} / {display_event(row.event_j)}"


def pair_key_frame(frame: pd.DataFrame) -> pd.Series:
    return frame.apply(
        lambda row: "||".join(sorted([str(row["event_i"]), str(row["event_j"])])),
        axis=1,
    )


def load_selected_candidates() -> pd.DataFrame:
    selected_frames: list[pd.DataFrame] = []
    for domain in ("gwtc3", "gwtc4"):
        seed = SELECTED_DEPLOYMENTS[domain]
        stability = pd.read_parquet(
            GWTC_ROOT / domain / "real_candidate_rank_stability_v81.parquet"
        )
        top10 = (
            stability[stability["seed"].astype(str) == seed]
            .sort_values("rank")
            .head(10)
            .copy()
        )
        posterior = pd.read_parquet(
            GWTC_ROOT / domain / "real_candidate_consensus_with_pe_v93.parquet"
        ).copy()
        top10["pair_key_join"] = pair_key_frame(top10)
        posterior["pair_key_join"] = pair_key_frame(posterior)
        posterior_columns = [
            "pair_key_join",
            "intrinsic_3sigma_consistent",
            "max_standardized_posterior_distance",
            "chirp_mass_bhattacharyya_coefficient",
            "mass_ratio_bhattacharyya_coefficient",
            "chi_eff_bhattacharyya_coefficient",
        ]
        merged = top10.merge(posterior[posterior_columns], on="pair_key_join", how="left")
        merged["display_rank"] = merged["rank"].astype(int)
        merged["deployment"] = domain
        merged["selected_seed"] = seed
        selected_frames.append(merged)
    result = pd.concat(selected_frames, ignore_index=True)
    SOURCE.mkdir(parents=True, exist_ok=True)
    result.to_csv(SOURCE / "fig4_selected_seed_candidates.csv", index=False)
    return result


def plot_measured_strain(
    ax: plt.Axes,
    waveforms: pd.DataFrame,
    reconstructed: pd.DataFrame,
    domain: str,
) -> None:
    time = waveforms["time_from_event_gps_s"].to_numpy(dtype=float)
    model_time = reconstructed["time_from_peak_s"].to_numpy(dtype=float)
    if not np.all(np.diff(model_time) > 0):
        raise ValueError("Reconstructed waveform time grid must be strictly increasing")
    pair = WAVEFORM_EXAMPLES[domain]
    rows = [
        (pair[0], "H1", 3.0, COLORS["wave_a"]),
        (pair[0], "L1", 2.0, COLORS["wave_a"]),
        (pair[1], "H1", 1.0, COLORS["wave_b"]),
        (pair[1], "L1", 0.0, COLORS["wave_b"]),
    ]
    values = np.vstack(
        [waveforms[f"{event}__{detector}"].to_numpy(dtype=float) for event, detector, _, _ in rows]
    )
    scale = 0.32 / max(float(np.quantile(np.abs(values), 0.995)), 1e-12)
    for event, detector, offset, color in rows:
        trace = waveforms[f"{event}__{detector}"].to_numpy(dtype=float)
        ax.plot(time, offset + scale * trace, color=color, lw=0.30, alpha=0.88)
        # This reconstruction is h_+(t), not a detector-projected template.
        # Its independently normalized overlay is a morphology guide only.
        model = np.interp(
            time,
            model_time,
            reconstructed[event].to_numpy(dtype=float),
            left=np.nan,
            right=np.nan,
        )
        ax.plot(
            time,
            offset + 0.30 * model,
            color="white",
            lw=1.20,
            alpha=0.86,
            zorder=3,
        )
        ax.plot(
            time,
            offset + 0.30 * model,
            color=COLORS["pe_overlay"],
            lw=0.72,
            alpha=0.98,
            zorder=4,
        )
    ax.axvline(0, color="#5A5A5A", ls=(0, (2, 2)), lw=0.58)
    ax.set_xlim(-0.80, 0.12)
    ax.set_ylim(-0.45, 3.45)
    ax.set_yticks([3, 2, 1, 0], ["1 H1", "1 L1", "2 H1", "2 L1"])
    ax.tick_params(axis="y", length=0, labelsize=5.5, pad=2)
    ax.set_xticks([-0.75, -0.50, -0.25, 0.0])
    ax.tick_params(axis="x", labelsize=5.7)
    ax.set_xlabel("Time from trigger (s)", fontsize=6.2, labelpad=1)
    ax.set_title(r"Measured strain + scaled PE $h_+$", fontsize=7.3, pad=3)
    ax.spines["left"].set_visible(False)


def plot_figure4(candidates: pd.DataFrame, operating: pd.DataFrame) -> None:
    measured = pd.read_csv(CURRENT / "representative_waveform_inputs.csv")
    reconstructed = pd.read_csv(CURRENT / "reconstructed_waveform_inputs.csv")
    bc_columns = [
        "chirp_mass_bhattacharyya_coefficient",
        "mass_ratio_bhattacharyya_coefficient",
        "chi_eff_bhattacharyya_coefficient",
    ]
    # The panel title and colorbar define these cells as Bhattacharyya
    # coefficients; compact plain-text parameter labels remain legible at the
    # final 183-mm figure width without subscript glyphs falling below 5 pt.
    bc_labels = ["Mc", "q", "χeff"]
    review = candidates.copy()
    review["minimum_marginal_bc"] = review[bc_columns].min(axis=1)

    cmap = LinearSegmentedColormap.from_list(
        "posterior_overlap", ["#F7FBFC", "#D8EBF0", "#9EC9D7", "#5B99B3", "#174E69"]
    )
    fig = plt.figure(figsize=(FULL_WIDTH, 6.15))
    grid = fig.add_gridspec(
        2,
        3,
        width_ratios=[2.55, 1.05, 1.45],
        left=0.225,
        right=0.985,
        bottom=0.115,
        top=0.94,
        hspace=0.42,
        wspace=0.34,
    )
    image = None
    panel_letters = [("a", "b", "c"), ("d", "e", "f")]
    for row_index, domain in enumerate(("gwtc3", "gwtc4")):
        subset = review[review["deployment"] == domain].sort_values("display_rank")
        y = np.arange(len(subset))
        example = set(WAVEFORM_EXAMPLES[domain])
        example_mask = subset.apply(
            lambda row: {display_event(row["event_i"]), display_event(row["event_j"])} == example,
            axis=1,
        ).to_numpy(dtype=bool)

        rank_ax = fig.add_subplot(grid[row_index, 0])
        rank_ax.hlines(
            y,
            0,
            subset["final_score"].to_numpy(dtype=float),
            color="#AAB7BE",
            linewidth=0.95,
            zorder=1,
        )
        rank_ax.scatter(
            subset["final_score"],
            y,
            s=25,
            facecolor=COLORS["candidate"],
            edgecolor="white",
            linewidth=0.55,
            zorder=3,
        )
        if example_mask.any():
            selected_y = int(np.flatnonzero(example_mask)[0])
            rank_ax.scatter(
                [subset.iloc[selected_y]["final_score"]],
                [selected_y],
                s=44,
                facecolor=COLORS["candidate"],
                edgecolor=COLORS["wave_b"],
                linewidth=1.35,
                zorder=4,
            )
        rank_ax.set_yticks(y, [pair_label(row) for row in subset.itertuples()])
        rank_ax.set_ylim(len(subset) + 0.6, -0.6)
        score_max = float(subset["final_score"].max())
        rank_ax.set_xlim(0, score_max + max(0.15, 0.08 * score_max))
        rank_ax.set_xlabel(r"Catalog ranking score, $S_{ij}$")
        rank_ax.set_title(
            (
                f"GWTC-3 shortlist (seed {SELECTED_DEPLOYMENTS[domain]})"
                if domain == "gwtc3"
                else f"GWTC-4.1 O4a shortlist (seed {SELECTED_DEPLOYMENTS[domain]})"
            ),
            pad=6,
        )
        rank_ax.tick_params(axis="y", length=0, pad=3, labelsize=5.5)
        rank_ax.grid(axis="x", color="#E0E3E5", lw=0.5)
        rank10_score = float(subset.loc[subset["display_rank"] == 10, "final_score"].iloc[0])
        rank10_operating = operating[
            (operating["domain"] == domain) & (operating["kind"] == "rank-10 score")
        ].iloc[0]
        rank10_fpr_percent = 100.0 * float(rank10_operating["fpr"])
        rank_ax.text(
            0.99,
            0.018,
            (
                rf"rank 10: $S={rank10_score:.2f}$; "
                + rf"injection-background FPR={rank10_fpr_percent:.3g}%"
            ),
            transform=rank_ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=5.8,
            color=COLORS["candidate"],
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 1.0},
        )
        panel_label(rank_ax, panel_letters[row_index][0], x=-0.62)

        pe_ax = fig.add_subplot(grid[row_index, 1])
        values = subset[bc_columns].to_numpy(dtype=float)
        image = pe_ax.imshow(
            values,
            cmap=cmap,
            norm=Normalize(vmin=0, vmax=1),
            aspect="auto",
            interpolation="nearest",
        )
        pe_ax.set_xticks(np.arange(3), bc_labels)
        pe_ax.set_yticks(np.arange(len(subset)), [str(int(value)) for value in subset["display_rank"]])
        pe_ax.tick_params(axis="x", length=0, labelsize=7.2, pad=4)
        pe_ax.tick_params(axis="y", length=0, labelsize=6.7, pad=2)
        pe_ax.set_ylabel("Catalog rank", fontsize=7.2)
        pe_ax.set_title("Marginal PE overlap", fontsize=8.2, pad=5)
        for yi in range(values.shape[0]):
            for xi in range(values.shape[1]):
                value = values[yi, xi]
                pe_ax.text(
                    xi,
                    yi,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=6.0,
                    color="white" if value > 0.58 else "#1A1A1A",
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
                    linewidth=1.2,
                    clip_on=False,
                )
            )
        for spine in pe_ax.spines.values():
            spine.set_visible(False)
        panel_label(pe_ax, panel_letters[row_index][1], x=-0.30)

        measured_ax = fig.add_subplot(grid[row_index, 2])
        plot_measured_strain(measured_ax, measured, reconstructed, domain)
        panel_label(measured_ax, panel_letters[row_index][2], x=-0.28)

    colorbar_axis = fig.add_axes([0.515, 0.035, 0.20, 0.012])
    colorbar = fig.colorbar(image, cax=colorbar_axis, orientation="horizontal")
    colorbar.set_label(
        "Bhattacharyya overlap, BC (0: disjoint; 1: identical)",
        fontsize=6.7,
        labelpad=2,
    )
    colorbar.set_ticks([0, 0.5, 1.0])
    colorbar.ax.tick_params(labelsize=6.2, length=2, pad=1)
    export(fig, "fig_real_candidate_pe_audit")


def write_summary(
    operating: pd.DataFrame, roc_metrics: pd.DataFrame, candidates: pd.DataFrame
) -> None:
    lines = [
        "# 三图重构审核说明（2026-08-05）",
        "",
        "本目录只包含审核稿，没有覆盖论文图片，也没有推送 Overleaf。",
        "",
        "## 统一叙事",
        "",
        "- Figure 2：三个物理证据通道的成对分布，以及这些分数对应的可审计工作点。",
        "- Figure 3：同一分数在 pair-level 判别和 per-query 检索中的表现。上排用 FPR 与 detection probability，且把 FPR 换算为预期假对数；下排同时给出 Top-K 和目录百分比。",
        "- Figure 4：把同一评分用于真实目录候选，再以 PE 后验重合和波形完整性图进行排名后审计。",
        "",
        "## 关键定义与解释",
        "",
        "- Figure 4 的共识顺序由三个冻结部署中的平均名次决定，并非由平均总分决定。因此新图横轴改为 mean within-seed rank，并画出 rank_min--rank_max；平均总分只保留为 Top-10 floor 注释。",
        "- PE 的 0--1 指标采用 Bhattacharyya coefficient：$BC_x=\\int\\sqrt{p_i(x)p_j(x)}\\,dx$。0 表示边际后验几乎不重合，1 表示两条边际后验相同。它利用完整的一维后验形状，但仍忽略参数协方差，不是透镜概率或 Bayes factor。",
        "- Figure 3 展示的是 pair false-positive rate（FPR），不是单位时间 false-alarm rate（FAR）。当前实验没有定义把 pair 阈值换算成每年误报数的完整搜索率模型，因此直接写 FAR 会造成过度解释。上轴给出的 expected false pairs = FPR × 非伴随 pair 数，是当前数据下可严格报告的误报负担。",
        "- ET-3 ROC 曲线使用每个种子固定抽取的 200,000 个非伴随 pair，加上全部 3,000 个真伴随 pair。低 FPR 工作点与 90% recall 假对数来自完整的 40,495,500-pair 审计，而不是抽样曲线。O3/O4a ROC 使用全部 101,025 个 pair。",
        "- Figure 2 的绿色菱形把真实 Top-10 的最低平均分投影回使用相同 candidate 权重的注入测试。它只是域内诊断，不是预先规定的检出阈值。",
        "",
        "## 关键数值",
        "",
    ]
    for domain in ("ET3", "gwtc3", "gwtc4"):
        rows = operating[
            (operating["domain"] == domain)
            & (operating["kind"] == "90% companion recovery")
        ]
        lines.append(
            f"- {DOMAIN_SHORT[domain]}：90% pair recovery 时平均 FPR = {rows['fpr'].mean():.6g}，平均假对数 = {rows['false_pairs'].mean():,.1f}。"
        )
    for domain in ("gwtc3", "gwtc4"):
        row = operating[
            (operating["domain"] == domain)
            & (operating["kind"] == "real Top-10 score floor")
        ].iloc[0]
        relation = "<1" if bool(row["zero_false"]) else f"{int(row['false_pairs'])}"
        lines.append(
            f"- {DOMAIN_SHORT[domain]} 真实 Top-10 最低平均分为 {row['score_floor']:.3f}；映射到对应注入 candidate-score 后，假对数为 {relation}，真伴随保留率为 {row['recall']:.3f}。"
        )
    lines.extend(["", "## ROC AUC（审核图所用分数）", ""])
    for domain in ("ET3", "gwtc3", "gwtc4"):
        values = roc_metrics[
            (roc_metrics["domain"] == domain)
            & (roc_metrics["method"] == "three_channel")
        ]["roc_auc"]
        lines.append(
            f"- {DOMAIN_SHORT[domain]}：{values.mean():.6f} ± {values.std(ddof=1):.6f}。"
        )
    lines.extend(
        [
            "",
            "## 文件",
            "",
            "- `outputs/fig2_evidence_to_operating_points_review.pdf|svg|png`",
            "- `outputs/fig3_discrimination_and_retrieval_review.pdf|svg|png`",
            "- `outputs/fig4_real_candidate_audit_review.pdf|svg|png`",
            "- `source_data/` 保存本次图中新增的 ROC、工作点和候选审核表。",
        ]
    )
    (Path(__file__).resolve().parent / "README_CN.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def write_summary_v2(
    operating: pd.DataFrame, roc_metrics: pd.DataFrame, candidates: pd.DataFrame
) -> None:
    lines = [
        "# 三图修改审核说明（2026-08-05）",
        "",
        "本目录只包含审核稿，没有覆盖论文图片，也没有推送 Overleaf。",
        "",
        "## 本轮修改",
        "",
        "- Figure 2 恢复三通道总分分布。O3/O4a 总分与 Figure 4 使用同一冻结种子、同一 candidate 权重和同一原始分数尺度。",
        "- Figure 2 在总分分布上标出 90% detection threshold、对应假对数以及真实目录 rank-10 分数。",
        "- Figure 3 的 ROC 和 R@K 仍为多种子均值，但不再绘制种子范围阴影，也不再显示 90% operating-point marker。",
        "- Figure 4 只显示一个冻结部署。O3 选择 seed 202607241，其 Top-10 有 8/10 通过原 3-sigma PE screen；O4a 选择 seed 202607242，其 Top-10 为 10/10。",
        "- Figure 4 直接按照该部署的总分从高到低排列，不再显示多种子 rank range。",
        "",
        "## 分数统一",
        "",
        "Figure 2d 和 Figure 4 使用同一个 $S_{ij}$。O3/O4a 均采用 candidate 权重 (waveform, time, sky) = (0.5, 0.25, 0.5)。因此 Figure 2 标出的 rank-10 分数与 Figure 4 完全相同。",
        "",
        "## 关键数值",
        "",
    ]
    for domain in ("ET3", "gwtc3", "gwtc4"):
        threshold = operating[
            (operating["domain"] == domain)
            & (operating["kind"] == "90% detection threshold")
        ].iloc[0]
        lines.append(
            f"- {DOMAIN_SHORT[domain]}：90% detection threshold = {threshold['score_value']:.4f}，假对数 = {int(threshold['false_pairs']):,}。"
        )
        rank10 = operating[
            (operating["domain"] == domain) & (operating["kind"] == "rank-10 score")
        ]
        if not rank10.empty:
            lines.append(
                f"- {DOMAIN_SHORT[domain]}：真实目录 rank-10 score = {rank10.iloc[0]['score_value']:.4f}。"
            )
    lines.extend(["", "## ROC AUC（多种子均值）", ""])
    for domain in ("ET3", "gwtc3", "gwtc4"):
        values = roc_metrics[
            (roc_metrics["domain"] == domain)
            & (roc_metrics["method"] == "three_channel")
        ]["roc_auc"]
        lines.append(
            f"- {DOMAIN_SHORT[domain]}：{values.mean():.6f} ± {values.std(ddof=1):.6f}。"
        )
    lines.extend(
        [
            "",
            "## 科学解释边界",
            "",
            "Figure 4 的种子由真实 PE 一致性选出，因此属于展示性、事后选择，不能用来估计无偏检索性能。正式性能仍应由 Figure 3 的多种子测试均值报告。",
            "",
            "## 文件",
            "",
            "- `outputs/fig2_evidence_and_total_score_review.pdf|svg|png`",
            "- `outputs/fig3_discrimination_and_retrieval_review.pdf|svg|png`",
            "- `outputs/fig4_real_candidate_audit_review.pdf|svg|png`",
            "- `source_data/` 保存本轮重算的单种子分数分布、阈值、ROC 和候选表。",
        ]
    )
    (Path(__file__).resolve().parent / "README_CN.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    configure_style()
    weights = load_weights()
    operating = pd.read_csv(CURRENT / "fig2_selected_score_thresholds.csv")
    density = pd.read_csv(CURRENT / "fig2_selected_seed_density.csv")
    fpr_curves = build_selected_fpr_curves(weights)
    recovery_points = build_selected_recovery_points(weights)
    roc_source = pd.read_csv(CURRENT / "fig3_roc_curves_per_seed.csv")
    roc_metrics = pd.read_csv(CURRENT / "fig3_roc_metrics_per_seed.csv")
    retrieval = pd.read_csv(CURRENT / "fig3_retrieval_curves_summary.csv")
    candidates = pd.read_csv(CURRENT / "fig4_selected_seed_candidates.csv")
    plot_figure2(density, operating, fpr_curves, recovery_points)
    plot_figure3(roc_source, roc_metrics, retrieval)
    plot_figure4(candidates, operating)
    print(f"Wrote main-text Figures 2--4 to {OUT}")


if __name__ == "__main__":
    main()
