#!/usr/bin/env python3
"""Frozen-candidate diagnostics; no fitting, selection, or score changes."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import mcwf_development_20260905 as dev
import mcwf_encoder_body_20260906 as body
import mcwf_fresh_confirmation_20260906 as fresh


def diagnostics(root):
    contract = fresh.verify_freeze(root)
    directory = fresh.data_root(root)
    rows, per_event, resource = [], [], []
    for cs in fresh.CATALOG_SEEDS:
        cat = directory / f"gwtc3/catalog_{cs}"
        events = pd.read_parquet(cat / "event_manifest.parquet")
        full = np.stack([np.load(cat / f"events/{uid}_full24.npy") for uid in events.event_uid])
        truth = np.log(events.mc_det.to_numpy(float))
        for ms in body.MODEL_SEEDS:
            checkpoint = root / f"models/{contract['O3_config']}/gwtc3/seed_{ms}/validation_selected_model.pt"
            p, z, _ = body.encode(checkpoint, full)
            position = np.clip((truth-body.tf.LOG_EDGES[0])/np.diff(body.tf.LOG_EDGES)[0], 0, 64-1e-9)
            bi = position.astype(int)
            cdf = np.c_[np.zeros(len(p)), np.cumsum(p, -1)]
            pit = cdf[np.arange(len(p)), bi]+(position-bi)*p[np.arange(len(p)), bi]
            pred = p @ body.tf.LOG_CENTERS
            values = np.maximum(np.linalg.eigvalsh(np.cov(z, rowvar=False)), 0)
            fraction = values[values > 0]/values.sum()
            rows.append({"deployment": "gwtc3", "catalog_seed": cs, "model_seed": ms,
                "logMc_MAE": float(np.abs(pred-truth).mean()), "logMc_bias": float((pred-truth).mean()),
                "central50_coverage": float(np.mean((pit >= .25) & (pit <= .75))),
                "central90_coverage": float(np.mean((pit >= .05) & (pit <= .95))),
                "logMc_Spearman": float(spearmanr(pred, truth).statistic),
                "effective_rank": float(np.exp(-np.sum(fraction*np.log(fraction))))})
            per_event.append(pd.DataFrame({"event_uid": events.event_uid, "catalog_seed": cs, "model_seed": ms,
                "true_Mc": np.exp(truth), "predicted_geometric_mean_Mc": np.exp(pred),
                "predictive_PIT": pit, "target_network_SNR": events.target_network_snr}))
            np.savez_compressed(cat / f"model_{ms}/new_encoder_predictions.npz", p=p, embedding=z)
    dev.csv_write(directory / "NEW_ENCODER_PHYSICAL_DIAGNOSTICS.csv", pd.DataFrame(rows))
    dev.csv_write(directory / "NEW_ENCODER_EVENT_PREDICTIONS.csv", pd.concat(per_event, ignore_index=True))
    for path in sorted((root / "models").glob("*/*/seed_*/training_history.csv")):
        f = pd.read_csv(path)
        resource.append({"stage": "training", "config": path.parents[2].name, "deployment": path.parents[1].name,
            "seed": path.parent.name, "epochs": len(f), "timed_seconds": f.seconds.sum(),
            "GPU_peak_bytes": int(f.gpu_peak_bytes.max()), "timing_scope": "sum timed training/validation epoch loops; excludes feature extraction"})
    for dep in ("gwtc3", "gwtc4"):
        events = pd.concat([pd.read_parquet(directory / f"{dep}/catalog_{cs}/event_manifest.parquet") for cs in fresh.CATALOG_SEEDS])
        resource.append({"stage": "BAYESTAR", "config": "frozen_conditional_protocol", "deployment": dep,
            "events": len(events), "timed_seconds": events.map_seconds.sum(),
            "P50_seconds_per_event": events.map_seconds.median(), "P90_seconds_per_event": events.map_seconds.quantile(.9),
            "max_seconds_per_event": events.map_seconds.max(), "fallback_count": int(events.sky_fallback.sum()),
            "timing_scope": "sum per-event wall times in parallel workers, not end-to-end wall time"})
    dev.csv_write(root / "tables/RESOURCE_AND_TIMING.csv", pd.DataFrame(resource))


def figures(root):
    directory = fresh.data_root(root)
    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 9, "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    colors = ("#6C757D", "#237A69")
    fig, axes = plt.subplots(2, 3, figsize=(9.4, 5.3), constrained_layout=True)
    for i, dep in enumerate(("gwtc3", "gwtc4")):
        f = pd.read_csv(directory / f"{dep}/metrics_all.csv")
        for j, (metric, ylabel) in enumerate((("macro_r_at_10", "Companion R@10"), ("average_precision", "Pair AUPRC"),
                                              ("false_at_recall_0p9", "False pairs at 90% recall"))):
            ax = axes[i, j]
            for x, variant in enumerate(("BASELINE", "CANDIDATE")):
                g = f.loc[f.variant.eq(variant) & f.method.eq("C_fixed")]
                for k, cs in enumerate(fresh.CATALOG_SEEDS):
                    values = g.loc[g.catalog_seed.eq(cs), metric].to_numpy()
                    ax.scatter(x+(k-1)*.08+np.linspace(-.022, .022, len(values)), values,
                               color=colors[x], marker=("o", "s", "^")[k], s=18, alpha=.65)
                model_means = g.groupby("model_seed")[metric].mean()
                ax.errorbar(x, model_means.mean(), model_means.std(), color=colors[x], fmt="_", markersize=15, capsize=4)
            ax.set_xticks([0, 1], ["Baseline", "Candidate"])
            ax.set_ylabel(ylabel)
            ax.set_title("O3" if dep == "gwtc3" else "O4a: unchanged by construction")
            if j < 2:
                ax.set_ylim(0, 1)
            ax.text(-.17, 1.03, "abcdef"[3*i+j], transform=ax.transAxes, weight="bold", fontsize=12)
    fig.suptitle("Fresh source/noise test, conditional on reused lens environments", fontsize=11)
    for ext in ("pdf", "png"):
        fig.savefig(root / f"figures/fig_fresh_confirmation.{ext}", dpi=300)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(7.3, 3.3), constrained_layout=True)
    for ax, code in zip(axes, ("CFIX-baseline", "MAIN-O3-MCWF-ENCODER-v2-RUNSPEC")):
        f = pd.read_parquet(root / f"final_results/gwtc3/{code}/C_fixed/all_pairs.parquet")
        ax.hexbin(f.waveform_score_mean, f.pe_mc_bhattacharyya_coefficient,
                  gridsize=35, mincnt=1, cmap="Greys", bins="log")
        ax.scatter(f.head(10).waveform_score_mean, f.head(10).pe_mc_bhattacharyya_coefficient,
                   facecolor="#237A69", edgecolor="white", s=28, linewidth=.5, label="Consensus C-fixed Top-10")
        ax.axhline(.1, color="#AF4157", lw=.8, ls=":")
        ax.set(xlabel="Mean waveform score", ylabel="Public chirp-mass posterior BC", ylim=(-.025, 1.025),
               title="Baseline" if code == "CFIX-baseline" else "Candidate")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", frameon=False)
    for ext in ("pdf", "png"):
        fig.savefig(root / f"figures/fig_waveform_Mc_correlation.{ext}", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    a = p.parse_args()
    diagnostics(a.root)
    figures(a.root)
