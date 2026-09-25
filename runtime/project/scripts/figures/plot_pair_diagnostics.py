from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def save_all(fig, stem: Path) -> list[str]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ["png", "pdf"]:
        path = stem.with_suffix(f".{ext}")
        fig.savefig(path, dpi=400, bbox_inches="tight")
        paths.append(str(path))
    return paths


def plot_gwtc_overlay(pair_csv: Path, out_stem: Path) -> dict:
    df = pd.read_csv(pair_csv)
    seed = int(df["seed"].min())
    background = df[(df["seed"] == seed) & (df["pair_class"] == "background_pair")].copy()
    true_pairs = df[df["pair_class"] == "true_injected_pair"].copy()

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.scatter(
        background["delta_t_days"].clip(lower=1e-5),
        background["sky_norm_sep"],
        s=8,
        c="0.72",
        alpha=0.32,
        linewidths=0,
        label=f"GWTC-3 null background pairs (seed {seed})",
    )
    sc = ax.scatter(
        true_pairs["delta_t_days"].clip(lower=1e-5),
        true_pairs["sky_norm_sep"],
        s=22,
        c=true_pairs["seed"],
        cmap="viridis",
        alpha=0.88,
        linewidths=0.25,
        edgecolors="white",
        label="synthetic injected lensed pairs",
    )

    marker_pair = background[
        ((background["event_i"].str.contains("GW170104", na=False)) & (background["event_j"].str.contains("GW170814", na=False)))
        | ((background["event_i"].str.contains("GW170814", na=False)) & (background["event_j"].str.contains("GW170104", na=False)))
    ]
    marker_found = bool(len(marker_pair))
    if marker_found:
        row = marker_pair.iloc[0]
        ax.scatter(
            [max(float(row["delta_t_days"]), 1e-5)],
            [float(row["sky_norm_sep"])],
            marker="*",
            s=140,
            c="#d62728",
            edgecolors="black",
            linewidths=0.5,
            label="GW170104--GW170814",
            zorder=5,
        )

    ax.set_xscale("log")
    ax.set_xlabel("Observed time separation, |Delta t| (days)")
    ax.set_ylabel("Observed sky normalized separation")
    ax.grid(True, which="both", alpha=0.18)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    cbar = fig.colorbar(sc, ax=ax, pad=0.012)
    cbar.set_label("Injection seed")
    ax.text(
        0.01,
        -0.20,
        "GWTC background is treated as a null catalog; injected pairs are synthetic.",
        transform=ax.transAxes,
        fontsize=8,
        va="top",
    )
    paths = save_all(fig, out_stem)
    plt.close(fig)
    return {
        "figure": "A",
        "paths": paths,
        "background_pairs_plotted": int(len(background)),
        "true_injected_pairs_plotted": int(len(true_pairs)),
        "gw170104_gw170814_marker_found": marker_found,
    }


def plot_et3_channel_corner(pair_parquet: Path, out_stem: Path, max_false: int, seed: int) -> dict:
    df = pd.read_parquet(pair_parquet)
    true_df = df[df["is_true_pair"]].copy()
    false_df = df[~df["is_true_pair"]].copy()
    if len(false_df) > max_false:
        false_df = false_df.sample(max_false, random_state=seed)

    false_x = 0.5 * (false_df["time_score_i_to_j"].to_numpy(np.float32) + false_df["time_score_j_to_i"].to_numpy(np.float32))
    false_y = false_df["sky_log_overlap"].to_numpy(np.float32)
    sis = true_df[true_df["lens_type"] == "SIS"]
    pm = true_df[true_df["lens_type"] == "PM"]
    sis_x = 0.5 * (sis["time_score_i_to_j"].to_numpy(np.float32) + sis["time_score_j_to_i"].to_numpy(np.float32))
    sis_y = sis["sky_log_overlap"].to_numpy(np.float32)
    pm_x = 0.5 * (pm["time_score_i_to_j"].to_numpy(np.float32) + pm["time_score_j_to_i"].to_numpy(np.float32))
    pm_y = pm["sky_log_overlap"].to_numpy(np.float32)

    fig = plt.figure(figsize=(6.8, 6.2))
    gs = fig.add_gridspec(2, 2, width_ratios=(4, 1), height_ratios=(1, 4), hspace=0.05, wspace=0.05)
    ax_histx = fig.add_subplot(gs[0, 0])
    ax = fig.add_subplot(gs[1, 0], sharex=ax_histx)
    ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)

    ax.scatter(false_x, false_y, s=4, c="0.72", alpha=0.14, linewidths=0, label="sampled false pairs")
    ax.scatter(sis_x, sis_y, s=12, c="#1f77b4", alpha=0.72, linewidths=0, label="true SIS pairs")
    ax.scatter(pm_x, pm_y, s=12, c="#d62728", alpha=0.72, linewidths=0, label="true PM pairs")
    ax.set_xlabel("Liao time-score component (row-z, pair mean)")
    ax.set_ylabel("Observed sky log-overlap")
    ax.grid(True, alpha=0.18)
    ax.legend(frameon=False, fontsize=8, loc="best")

    bins_x = np.linspace(
        np.nanpercentile(np.concatenate([false_x, sis_x, pm_x]), 0.5),
        np.nanpercentile(np.concatenate([false_x, sis_x, pm_x]), 99.5),
        80,
    )
    bins_y = np.linspace(
        np.nanpercentile(np.concatenate([false_y, sis_y, pm_y]), 0.5),
        np.nanpercentile(np.concatenate([false_y, sis_y, pm_y]), 99.5),
        80,
    )
    ax_histx.hist(false_x, bins=bins_x, color="0.72", alpha=0.55, density=True)
    ax_histx.hist(sis_x, bins=bins_x, histtype="step", color="#1f77b4", linewidth=1.2, density=True)
    ax_histx.hist(pm_x, bins=bins_x, histtype="step", color="#d62728", linewidth=1.2, density=True)
    ax_histy.hist(false_y, bins=bins_y, orientation="horizontal", color="0.72", alpha=0.55, density=True)
    ax_histy.hist(sis_y, bins=bins_y, orientation="horizontal", histtype="step", color="#1f77b4", linewidth=1.2, density=True)
    ax_histy.hist(pm_y, bins=bins_y, orientation="horizontal", histtype="step", color="#d62728", linewidth=1.2, density=True)
    ax_histx.tick_params(labelbottom=False)
    ax_histy.tick_params(labelleft=False)
    ax_histx.set_ylabel("density", fontsize=8)
    ax_histy.set_xlabel("density", fontsize=8)
    ax.text(
        0.01,
        -0.18,
        "False pairs are sampled diagnostics, not the full 40.5M unordered ET-3 pair space.",
        transform=ax.transAxes,
        fontsize=8,
        va="top",
    )

    paths = save_all(fig, out_stem)
    plt.close(fig)
    return {
        "figure": "B",
        "paths": paths,
        "false_pairs_plotted": int(len(false_df)),
        "true_sis_pairs_plotted": int(len(sis)),
        "true_pm_pairs_plotted": int(len(pm)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot pair-level diagnostic demo figures.")
    parser.add_argument("--gwtc-pairs", default=str(REPO_ROOT / "data" / "gwtc_injection_pair_diagnostics.csv"))
    parser.add_argument("--et3-pairs", default=str(REPO_ROOT / "runs" / "et3_liao_realistic_p1_p2_rerank_20260616" / "pair_level_diagnostics" / "et3_pair_diagnostics.parquet"))
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "figures_pair_diagnostics"))
    parser.add_argument("--max-false-b", type=int, default=120_000)
    parser.add_argument("--seed", type=int, default=20260624)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    results = []
    results.append(plot_gwtc_overlay(Path(args.gwtc_pairs), out_dir / "figA_gwtc_real_background_injected_overlay"))
    results.append(plot_et3_channel_corner(Path(args.et3_pairs), out_dir / "figB_et3_channel_separability_corner", args.max_false_b, args.seed))
    summary_path = out_dir / "pair_diagnostic_figures_summary.json"
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps({"summary": str(summary_path), "figures": results}, indent=2), flush=True)


if __name__ == "__main__":
    main()
