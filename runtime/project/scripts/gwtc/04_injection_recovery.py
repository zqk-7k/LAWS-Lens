from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from matchgw.aux_priors import a90_to_sigma_rad, observed_sky_pair_features, radec_from_unit, unit_from_radec
from scripts.gwtc import config

liao = importlib.import_module("scripts.experiments.88_liao_realistic_p1_p2_rerank")


SECONDS_PER_DAY = 86400.0
DEFAULT_K = (10, 20, 50)
DEFAULT_SEEDS = tuple(range(20260617, 20260627))
SCORE_NAMES = ("time_lr", "sky_log_overlap", "sky_step", "combined_time_sky")


def load_observables(catalog: str) -> pd.DataFrame:
    path = config.GWTC3_OBSERVABLES_CSV if catalog == "gwtc3" else config.GWTC5_OBSERVABLES_CSV
    df = pd.read_csv(path)
    required = ["event_name", "gps_trigger_time", "ra_median", "dec_median", "sky_area_90_deg2", "network_snr"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"{path} missing columns: {missing}")
    return df.dropna(subset=required).reset_index(drop=True)


def tangent_perturb(ra: np.ndarray, dec: np.ndarray, sigma: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    true_vec = unit_from_radec(ra, dec)
    z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    ref = np.tile(z_axis, (len(true_vec), 1))
    ref[np.abs(true_vec @ z_axis) > 0.95] = x_axis
    e1 = np.cross(ref, true_vec)
    e1 /= np.maximum(np.linalg.norm(e1, axis=1, keepdims=True), 1e-12)
    e2 = np.cross(true_vec, e1)
    e2 /= np.maximum(np.linalg.norm(e2, axis=1, keepdims=True), 1e-12)
    offset = rng.normal(0.0, sigma)[:, None] * e1 + rng.normal(0.0, sigma)[:, None] * e2
    obs_vec = true_vec + offset
    obs_vec /= np.maximum(np.linalg.norm(obs_vec, axis=1, keepdims=True), 1e-12)
    return radec_from_unit(obs_vec)


def row_z_matrix(score: np.ndarray) -> np.ndarray:
    out = score.astype(np.float32).copy()
    np.fill_diagonal(out, np.nan)
    mu = np.nanmean(out, axis=1, keepdims=True)
    sd = np.nanstd(out, axis=1, keepdims=True)
    out = (out - mu) / np.maximum(sd, 1e-8)
    np.fill_diagonal(out, -np.inf)
    return out.astype(np.float32)


def build_score_matrices(obs: pd.DataFrame) -> dict[str, np.ndarray]:
    time_obs = pd.DataFrame({
        "trigger_time_obs": obs["gps_trigger_time"].to_numpy(dtype=np.float64),
        "snr": obs["network_snr"].to_numpy(dtype=np.float64),
    })
    sky_obs = pd.DataFrame({
        "ra_obs": obs["ra_median"].to_numpy(dtype=np.float64),
        "dec_obs": obs["dec_median"].to_numpy(dtype=np.float64),
        "sky_area90_deg2": obs["sky_area_90_deg2"].to_numpy(dtype=np.float64),
    })
    sky_obs["sky_sigma_rad"] = a90_to_sigma_rad(sky_obs["sky_area90_deg2"].to_numpy(dtype=np.float64))

    gt_none = np.full(len(obs), -1, dtype=np.int32)
    prior = liao.fit_time_lr_from_liao("LIGO", time_obs, gt_none)
    time_lr = liao.time_lr_score_matrix(time_obs, prior)
    sky = observed_sky_pair_features(sky_obs)

    time_z = row_z_matrix(time_lr)
    sky_z = row_z_matrix(sky["sky_log_overlap"])
    return {
        "time_lr": time_lr,
        "sky_log_overlap": sky["sky_log_overlap"],
        "sky_step": sky["sky_step_weight"],
        "combined_time_sky": time_z + sky_z,
    }


def draw_liao_delay_ratio(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    cfg = liao.LIAO_PRIOR_CONFIG["LIGO"]
    delays, ratios = liao.extract_liao_delay_snr_pairs(cfg["image_csv"], cfg["snr_threshold"])
    if len(delays) == 0:
        raise RuntimeError("No usable Liao delay/SNR-ratio samples")
    idx = rng.integers(0, len(delays), size=n)
    return delays[idx].astype(np.float64), ratios[idx].astype(np.float64)


def make_injected_catalog(background: pd.DataFrame, k: int, seed: int) -> tuple[pd.DataFrame, list[tuple[int, int]]]:
    rng = np.random.default_rng(seed)
    base_idx = rng.integers(0, len(background), size=k)
    base = background.iloc[base_idx].reset_index(drop=True)
    delays_days, snr_ratios = draw_liao_delay_ratio(rng, k)

    t_min = float(background["gps_trigger_time"].min())
    t_max = float(background["gps_trigger_time"].max())
    span = max(t_max - t_min, 30.0 * SECONDS_PER_DAY)
    start_times = rng.uniform(t_min - 0.05 * span, t_max + 0.05 * span, size=k)
    second_times = start_times + delays_days * SECONDS_PER_DAY

    a90_1 = np.clip(
        base["sky_area_90_deg2"].to_numpy(dtype=np.float64) * rng.lognormal(0.0, 0.20, size=k),
        1.0,
        2.0e4,
    )
    a90_2 = np.clip(
        base["sky_area_90_deg2"].to_numpy(dtype=np.float64) * rng.lognormal(0.0, 0.20, size=k),
        1.0,
        2.0e4,
    )
    sigma_1 = a90_to_sigma_rad(a90_1)
    sigma_2 = a90_to_sigma_rad(a90_2)
    true_ra = base["ra_median"].to_numpy(dtype=np.float64)
    true_dec = base["dec_median"].to_numpy(dtype=np.float64)
    ra_1, dec_1 = tangent_perturb(true_ra, true_dec, sigma_1, rng)
    ra_2, dec_2 = tangent_perturb(true_ra, true_dec, sigma_2, rng)

    snr_base = np.maximum(base["network_snr"].to_numpy(dtype=np.float64), 5.0)
    snr_1 = snr_base * rng.lognormal(0.0, 0.10, size=k)
    snr_2 = np.maximum(snr_1 / np.maximum(snr_ratios, 1.0), 5.0)

    rows = []
    pairs: list[tuple[int, int]] = []
    offset = len(background)
    for idx in range(k):
        name = str(base.loc[idx, "event_name"])
        rows.append({
            "event_name": f"inj_{seed}_{idx:03d}a_from_{name}",
            "gps_trigger_time": start_times[idx],
            "ra_median": ra_1[idx],
            "dec_median": dec_1[idx],
            "sky_area_90_deg2": a90_1[idx],
            "network_snr": snr_1[idx],
            "injected_pair_id": idx,
        })
        rows.append({
            "event_name": f"inj_{seed}_{idx:03d}b_from_{name}",
            "gps_trigger_time": second_times[idx],
            "ra_median": ra_2[idx],
            "dec_median": dec_2[idx],
            "sky_area_90_deg2": a90_2[idx],
            "network_snr": snr_2[idx],
            "injected_pair_id": idx,
        })
        pairs.append((offset + 2 * idx, offset + 2 * idx + 1))

    bg = background.copy()
    bg["injected_pair_id"] = np.nan
    injected = pd.DataFrame(rows)
    mixed = pd.concat([bg, injected], ignore_index=True, sort=False)
    return mixed, pairs


def target_rank(score: np.ndarray, query: int, target: int) -> int:
    target_score = float(score[query, target])
    row = score[query]
    return int(np.sum(row > target_score) + 1)


def evaluate_one(catalog: str, background: pd.DataFrame, k: int, seed: int) -> list[dict]:
    mixed, pairs = make_injected_catalog(background, k, seed)
    scores = build_score_matrices(mixed)
    records = []
    for score_name in SCORE_NAMES:
        score = scores[score_name]
        ranks = []
        for left, right in pairs:
            ranks.append(target_rank(score, left, right))
            ranks.append(target_rank(score, right, left))
        ranks_arr = np.asarray(ranks, dtype=np.int32)
        records.append({
            "catalog": catalog,
            "k_injected_pairs": k,
            "seed": seed,
            "score": score_name,
            "queries": int(len(ranks_arr)),
            "median_rank": float(np.median(ranks_arr)),
            "mean_rank": float(np.mean(ranks_arr)),
            "mrr": float(np.mean(1.0 / ranks_arr)),
            "recall_at_1": float(np.mean(ranks_arr <= 1)),
            "recall_at_5": float(np.mean(ranks_arr <= 5)),
            "recall_at_10": float(np.mean(ranks_arr <= 10)),
            "recall_at_50": float(np.mean(ranks_arr <= 50)),
        })
    return records


def summarize(records: pd.DataFrame) -> pd.DataFrame:
    metrics = ["median_rank", "mean_rank", "mrr", "recall_at_1", "recall_at_5", "recall_at_10", "recall_at_50"]
    grouped = records.groupby(["catalog", "k_injected_pairs", "score"])[metrics].agg(["mean", "std"])
    grouped.columns = [f"{metric}_{stat}" for metric, stat in grouped.columns]
    return grouped.reset_index().sort_values(["catalog", "k_injected_pairs", "score"])


def save_figure(summary: pd.DataFrame) -> None:
    config.FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.5), sharey=True)
    for ax, catalog in zip(axes, ["gwtc3", "gwtc5"]):
        sub = summary[summary["catalog"] == catalog]
        for score in SCORE_NAMES:
            line = sub[sub["score"] == score].sort_values("k_injected_pairs")
            if line.empty:
                continue
            ax.errorbar(
                line["k_injected_pairs"],
                line["recall_at_10_mean"],
                yerr=line["recall_at_10_std"].fillna(0.0),
                marker="o",
                capsize=3,
                label=score,
            )
        ax.set_title(catalog.upper())
        ax.set_xlabel("injected lensed pairs")
        ax.set_ylim(0.0, 1.02)
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("injected-pair recall@10")
    axes[1].legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    for ext in ["png", "pdf"]:
        plt.savefig(config.FIGURE_DIR / f"fig_gwtc_injection_recovery.{ext}", dpi=400)
    plt.close(fig)


def append_results(summary: pd.DataFrame, records_path: Path, summary_path: Path) -> None:
    lines = [
        "",
        "## Injection recovery",
        "",
        "Synthetic lensed pairs were injected into the real GWTC observable catalogs. Source sky/A90/SNR are sampled from the real catalog distribution, time delays and SNR ratios are sampled from the Liao LIGO lensing prior, and each pair shares the same latent sky with independent 2D Gaussian observed-sky errors.",
        "",
        f"Per-seed records: `{records_path}`",
        f"Averaged summary: `{summary_path}`",
        "",
        "| catalog | injected pairs | score | median rank | MRR | R@1 | R@5 | R@10 | R@50 |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['catalog']} | {int(row['k_injected_pairs'])} | {row['score']} | "
            f"{row['median_rank_mean']:.2f}+/-{row['median_rank_std']:.2f} | "
            f"{row['mrr_mean']:.4f}+/-{row['mrr_std']:.4f} | "
            f"{row['recall_at_1_mean']:.3f}+/-{row['recall_at_1_std']:.3f} | "
            f"{row['recall_at_5_mean']:.3f}+/-{row['recall_at_5_std']:.3f} | "
            f"{row['recall_at_10_mean']:.3f}+/-{row['recall_at_10_std']:.3f} | "
            f"{row['recall_at_50_mean']:.3f}+/-{row['recall_at_50_std']:.3f} |"
        )
    lines += [
        "",
        "Figure written under `figures_gwtc/fig_gwtc_injection_recovery.{png,pdf}`.",
    ]
    result_path = config.SCRIPT_DIR / "RESULTS_gwtc.md"
    existing = result_path.read_text(encoding="utf-8") if result_path.exists() else "# GWTC real-data LensGraph physical reranker results\n"
    marker = "\n## Injection recovery\n"
    if marker in existing:
        existing = existing.split(marker)[0].rstrip() + "\n"
    result_path.write_text(existing.rstrip() + "\n" + "\n".join(lines) + "\n", encoding="utf-8")


def parse_int_list(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalogs", default="gwtc3,gwtc5", help="comma-separated list from: gwtc3,gwtc5")
    parser.add_argument("--k-list", default=",".join(str(k) for k in DEFAULT_K))
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    args = parser.parse_args()

    catalogs = [item.strip() for item in args.catalogs.split(",") if item.strip()]
    k_list = parse_int_list(args.k_list)
    seeds = parse_int_list(args.seeds)

    all_records: list[dict] = []
    for catalog in catalogs:
        background = load_observables(catalog)
        for k in k_list:
            for seed in seeds:
                print(f"catalog={catalog} k={k} seed={seed}", flush=True)
                all_records.extend(evaluate_one(catalog, background, k, seed))

    records = pd.DataFrame(all_records)
    summary = summarize(records)
    records_path = config.DATA_DIR / "gwtc_injection_recovery_records.csv"
    summary_path = config.DATA_DIR / "gwtc_injection_recovery_summary.csv"
    records.to_csv(records_path, index=False)
    summary.to_csv(summary_path, index=False)
    save_figure(summary)
    append_results(summary, records_path, summary_path)
    print(summary.to_string(index=False), flush=True)
    print(f"Wrote {records_path}", flush=True)
    print(f"Wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
