#!/usr/bin/env python3
"""Stable sequence-mitigation sandbox for manuscript Figure 4.

This is an observable-summary sandbox, not a full strain-level catalog
experiment.  It reuses the repository observable simulator and reranking
primitives, but avoids materialising multi-million-event catalogs for every
seed.  Large-density and low-rarity burdens are represented by detected-event
count estimates plus sampled false-pair / false-candidate score tails.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import genpareto, norm

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.server_experiments.observable_simulator import T_END, T_START, simulate_catalog
from scripts.server_experiments.rerank_engine import TimeDelayPrior, a90_to_sigma_rad


METHOD_TIME_SKY = "time-delay + sky-localization"
METHOD_WAVEFORM = "calibrated surrogate waveform embedding only"
METHOD_WEAK = "waveform-first weak fusion"
METHODS = [METHOD_TIME_SKY, METHOD_WAVEFORM, METHOD_WEAK]

TIME_SKY_WEIGHTS = {"time_lr": 1.0, "sky_step": 4.0, "sky_logoverlap": 1.0}
WEAK_TIME_SKY_SCALE = 0.01
WINDOW_S = float(T_END - T_START)
WINDOW_YR = WINDOW_S / (365.25 * 24 * 3600)
DEFAULT_DATE = "20260706"
DEFAULT_OUT = Path(f"/root/autodl-tmp/gw-catalog/results/sequence_mitigation_sandbox_{DEFAULT_DATE}_stable")
DEFAULT_PACKAGE = Path(f"/root/autodl-tmp/gw-catalog/packages/sequence_mitigation_sandbox_{DEFAULT_DATE}_stable.tar.gz")


@dataclass
class SeedCatalog:
    seed: int
    true_df: pd.DataFrame
    background_pool: pd.DataFrame
    background_sources_drawn: int
    background_raw_detected: int
    background_detected: int

    @property
    def detection_fraction(self) -> float:
        return self.background_raw_detected / max(self.background_sources_drawn, 1)


@dataclass(frozen=True)
class ScoreArrays:
    time: np.ndarray
    ra: np.ndarray
    dec: np.ndarray
    sigma: np.ndarray
    embedding: np.ndarray
    prior: TimeDelayPrior


def standardize_cols(x: np.ndarray) -> np.ndarray:
    return (x - x.mean(axis=0, keepdims=True)) / (x.std(axis=0, keepdims=True) + 1e-9)


def l2_normalize(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)


def complete_truth_pairs(df: pd.DataFrame) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for _, g in df[df["kind"].isin(["SIS", "PM"])].groupby("source_id", sort=False):
        if len(g) == 2:
            idx = sorted(g["event_id"].astype(int).tolist())
            pairs.append((idx[0], idx[1]))
    return pairs


def reindex_catalog(df: pd.DataFrame) -> pd.DataFrame:
    out = df.reset_index(drop=True).copy()
    if "event_id" in out.columns:
        out = out.drop(columns=["event_id"])
    out.insert(0, "event_id", np.arange(len(out), dtype=int))
    return out


def _collect_complete_family(
    *,
    family: str,
    n_pairs: int,
    detector: str,
    seed: int,
    snr_threshold: float,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    collected = 0
    attempt = 0
    while collected < n_pairs:
        need = n_pairs - collected
        request = max(need * 3, 100)
        if family == "SIS":
            df = simulate_catalog(request, 0, 0, detector=detector, seed=seed + attempt * 1009, snr_threshold=snr_threshold)
        elif family == "PM":
            df = simulate_catalog(0, request, 0, detector=detector, seed=seed + attempt * 1009, snr_threshold=snr_threshold)
        else:
            raise ValueError(family)
        for _, g in df.groupby("source_id", sort=False):
            if len(g) != 2:
                continue
            gg = g.copy()
            gg["source_id"] = 1_000_000 * (0 if family == "SIS" else 1) + collected
            rows.append(gg)
            collected += 1
            if collected >= n_pairs:
                break
        attempt += 1
        if attempt > 100:
            raise RuntimeError(f"Could not collect {n_pairs} complete {family} systems")
    return pd.concat(rows, ignore_index=True)


def make_true_catalog(n_true_pairs: int, *, detector: str, seed: int, snr_threshold: float) -> pd.DataFrame:
    n_sis = n_true_pairs // 2
    n_pm = n_true_pairs - n_sis
    sis = _collect_complete_family(
        family="SIS", n_pairs=n_sis, detector=detector, seed=100_000 + seed * 17, snr_threshold=snr_threshold
    )
    pm = _collect_complete_family(
        family="PM", n_pairs=n_pm, detector=detector, seed=200_000 + seed * 19, snr_threshold=snr_threshold
    )
    return reindex_catalog(pd.concat([sis, pm], ignore_index=True))


def make_background_pool(
    *,
    target_detected: int,
    detector: str,
    seed: int,
    snr_threshold: float,
    chunk_sources: int,
) -> tuple[pd.DataFrame, int, int]:
    parts: list[pd.DataFrame] = []
    drawn = 0
    detected = 0
    attempt = 0
    while detected < target_detected:
        df = simulate_catalog(0, 0, chunk_sources, detector=detector, seed=700_000 + seed * 101 + attempt, snr_threshold=snr_threshold)
        drawn += chunk_sources
        detected += len(df)
        parts.append(df)
        attempt += 1
        if attempt > 500:
            raise RuntimeError("background pool generation did not converge")
    bg = pd.concat(parts, ignore_index=True).iloc[:target_detected].copy()
    bg["kind"] = "unlensed"
    bg["image"] = -1
    return reindex_catalog(bg), drawn, detected


def build_seed_catalog(
    seed: int,
    *,
    n_true_pairs: int,
    detector: str,
    snr_threshold: float,
    background_pool_size: int,
    background_chunk_sources: int,
) -> SeedCatalog:
    true_df = make_true_catalog(n_true_pairs, detector=detector, seed=seed, snr_threshold=snr_threshold)
    bg, drawn, raw_detected = make_background_pool(
        target_detected=background_pool_size,
        detector=detector,
        seed=seed,
        snr_threshold=snr_threshold,
        chunk_sources=background_chunk_sources,
    )
    return SeedCatalog(
        seed=seed,
        true_df=true_df,
        background_pool=bg,
        background_sources_drawn=drawn,
        background_raw_detected=raw_detected,
        background_detected=len(bg),
    )


def calibrated_surrogate_waveform_embedding(df: pd.DataFrame, *, dim: int, seed: int, noise_sigma: float) -> np.ndarray:
    """Intrinsic-parameter embedding with calibrated noise.

    This intentionally excludes trigger time and observed sky, so it cannot
    leak the two observable priors being stress-tested.
    """
    rng = np.random.default_rng(seed + 714_901)
    base = np.column_stack(
        [
            np.log(df["chirp_mass"].to_numpy(float)),
            df["mass_ratio"].to_numpy(float),
            np.log(df["luminosity_distance"].to_numpy(float)),
            df["z_s"].to_numpy(float),
        ]
    ).astype("float32")
    base = standardize_cols(base)
    proj = rng.normal(0.0, 1.0, size=(base.shape[1], dim)).astype("float32")
    emb = base @ proj
    emb += rng.normal(0.0, noise_sigma, size=emb.shape).astype("float32")
    return l2_normalize(emb.astype("float32"))


def build_prior(true_df: pd.DataFrame) -> TimeDelayPrior:
    pairs = complete_truth_pairs(true_df)
    t = true_df["geocent_time"].to_numpy(float)
    delays = [abs(t[a] - t[b]) for a, b in pairs]
    return TimeDelayPrior(delays, window_s=WINDOW_S)


def make_score_arrays(df: pd.DataFrame, *, true_df: pd.DataFrame, seed: int, dim: int, noise_sigma: float) -> ScoreArrays:
    return ScoreArrays(
        time=df["geocent_time"].to_numpy(float),
        ra=df["ra"].to_numpy(float),
        dec=df["dec"].to_numpy(float),
        sigma=a90_to_sigma_rad(df["sky_area_90_deg2"].to_numpy(float)),
        embedding=calibrated_surrogate_waveform_embedding(df, dim=dim, seed=seed, noise_sigma=noise_sigma),
        prior=build_prior(true_df),
    )


def angular_sep_vec(ra1: float, dec1: float, ra2: np.ndarray, dec2: np.ndarray) -> np.ndarray:
    sd = np.sin(dec1) * np.sin(dec2) + np.cos(dec1) * np.cos(dec2) * np.cos(ra1 - ra2)
    return np.arccos(np.clip(sd, -1.0, 1.0))


def zscore(v: np.ndarray) -> np.ndarray:
    x = np.asarray(v, dtype=float)
    finite = np.isfinite(x)
    out = np.zeros_like(x, dtype=float)
    if finite.sum() > 1:
        out[finite] = (x[finite] - x[finite].mean()) / (x[finite].std() + 1e-9)
        out[~finite] = -10.0
    return out


def directed_scores(arr: ScoreArrays, q: int, candidates: np.ndarray, partner: int | None = None) -> dict[str, np.ndarray]:
    idx = np.asarray(candidates, dtype=np.int64)
    dt = np.abs(arr.time[idx] - arr.time[q])
    time_lr = arr.prior.lr_score(dt)
    sep = angular_sep_vec(arr.ra[q], arr.dec[q], arr.ra[idx], arr.dec[idx])
    sig2 = arr.sigma[q] ** 2 + arr.sigma[idx] ** 2
    norm_sep = sep / np.sqrt(sig2)
    sky_step = np.where(norm_sep < 1.0, 1.0, np.where(norm_sep < 2.0, 0.5, np.where(norm_sep < 3.0, 0.2, 0.0)))
    sky_log = -0.5 * sep**2 / sig2
    waveform = arr.embedding[idx] @ arr.embedding[q]

    time_sky = zscore(time_lr) + 4.0 * zscore(sky_step) + zscore(sky_log)
    wave = zscore(waveform)
    weak = wave + WEAK_TIME_SKY_SCALE * time_sky
    return {METHOD_TIME_SKY: time_sky, METHOD_WAVEFORM: wave, METHOD_WEAK: weak}


def raw_pair_channels(arr: ScoreArrays, pairs: np.ndarray) -> dict[str, np.ndarray]:
    i = pairs[:, 0]
    j = pairs[:, 1]
    dt = np.abs(arr.time[i] - arr.time[j])
    time_lr = arr.prior.lr_score(dt)
    sd = (
        np.sin(arr.dec[i]) * np.sin(arr.dec[j])
        + np.cos(arr.dec[i]) * np.cos(arr.dec[j]) * np.cos(arr.ra[i] - arr.ra[j])
    )
    sep = np.arccos(np.clip(sd, -1.0, 1.0))
    sig2 = arr.sigma[i] ** 2 + arr.sigma[j] ** 2
    norm_sep = sep / np.sqrt(sig2)
    sky_step = np.where(norm_sep < 1.0, 1.0, np.where(norm_sep < 2.0, 0.5, np.where(norm_sep < 3.0, 0.2, 0.0)))
    sky_log = -0.5 * sep**2 / sig2
    waveform = np.sum(arr.embedding[i] * arr.embedding[j], axis=1)
    return {"time_lr": time_lr, "sky_step": sky_step, "sky_logoverlap": sky_log, "waveform_embedding": waveform}


def reference_stats(channels: dict[str, np.ndarray]) -> dict[str, tuple[float, float]]:
    stats = {}
    for key, values in channels.items():
        v = values[np.isfinite(values)]
        stats[key] = (float(v.mean()), float(v.std() + 1e-9))
    return stats


def apply_pair_scores(channels: dict[str, np.ndarray], stats: dict[str, tuple[float, float]]) -> dict[str, np.ndarray]:
    z = {k: (v - stats[k][0]) / stats[k][1] for k, v in channels.items()}
    time_sky = z["time_lr"] + 4.0 * z["sky_step"] + z["sky_logoverlap"]
    wave = z["waveform_embedding"]
    weak = wave + WEAK_TIME_SKY_SCALE * time_sky
    return {METHOD_TIME_SKY: time_sky, METHOD_WAVEFORM: wave, METHOD_WEAK: weak}


def sample_false_pairs(rng: np.random.Generator, n_events: int, truth_pairs: set[tuple[int, int]], n_samples: int) -> np.ndarray:
    out_i = np.empty(n_samples, dtype=np.int64)
    out_j = np.empty(n_samples, dtype=np.int64)
    filled = 0
    truth_codes = np.fromiter((a * n_events + b for a, b in truth_pairs), dtype=np.int64)
    while filled < n_samples:
        need = n_samples - filled
        draw = max(need * 2, 20_000)
        i = rng.integers(0, n_events, draw, dtype=np.int64)
        j = rng.integers(0, n_events, draw, dtype=np.int64)
        a = np.minimum(i, j)
        b = np.maximum(i, j)
        ok = a != b
        if len(truth_codes):
            ok &= ~np.isin(a * n_events + b, truth_codes, assume_unique=False)
        a = a[ok]
        b = b[ok]
        take = min(need, len(a))
        out_i[filled : filled + take] = a[:take]
        out_j[filled : filled + take] = b[:take]
        filled += take
    return np.column_stack([out_i, out_j])


def make_false_survival_model(false_scores: np.ndarray) -> tuple[callable, dict[str, float | str]]:
    fs = np.asarray(false_scores, dtype=float)
    fs = fs[np.isfinite(fs)]
    threshold = float(np.quantile(fs, 0.95))
    exceed = fs[fs > threshold] - threshold
    tail_fraction = max(float(len(exceed)) / max(len(fs), 1), 1e-12)
    mu = float(fs.mean())
    sd = float(fs.std() + 1e-9)
    model = "normal"
    params = None
    if len(exceed) >= 200:
        try:
            c, loc, scale = genpareto.fit(exceed, floc=0.0)
            if np.isfinite(c) and np.isfinite(scale) and scale > 0:
                params = (float(c), float(loc), float(scale))
                model = "gpd"
        except Exception:
            params = None

    def survival(score: float) -> float:
        if score <= threshold:
            return float(np.mean(fs > score))
        if model == "gpd" and params is not None:
            c, loc, scale = params
            sf = float(genpareto.sf(score - threshold, c, loc=loc, scale=scale))
            if np.isfinite(sf):
                return max(tail_fraction * sf, 0.0)
        return float(norm.sf(score, loc=mu, scale=sd))

    return survival, {"tail_model": model, "tail_threshold": threshold, "tail_fraction": tail_fraction}


def fast_query_survival(false_scores: np.ndarray, score: float) -> float:
    """Cheap per-query survival estimate for density retrieval.

    Density scan needs thousands of query-level survival estimates.  A full
    GPD fit per query is unnecessary and slow; use empirical survival in the
    resolved range and a normal tail only above the sampled 99th percentile.
    """
    fs = np.asarray(false_scores, dtype=float)
    fs = fs[np.isfinite(fs)]
    if len(fs) == 0:
        return 1.0
    q99 = float(np.quantile(fs, 0.99))
    if score <= q99:
        return float(np.mean(fs > score))
    return float(norm.sf(score, loc=float(fs.mean()), scale=float(fs.std() + 1e-9)))


def estimated_true_recovered_at_budget(true_scores: np.ndarray, false_survival, total_false_pairs: float, budget: int) -> int:
    order = np.argsort(-true_scores)
    recovered = 0
    for pos, idx in enumerate(order):
        score = float(true_scores[idx])
        expected_false_above = false_survival(score) * total_false_pairs
        rank_est = 1.0 + pos + expected_false_above
        if rank_est <= budget:
            recovered += 1
    return int(recovered)


def estimated_detected_background(requested_background_sources: float, detection_fraction: float) -> int:
    return max(1, int(round(requested_background_sources * detection_fraction)))


def build_combined_df(seed_cat: SeedCatalog) -> pd.DataFrame:
    bg = seed_cat.background_pool.copy()
    max_sid = int(seed_cat.true_df["source_id"].max()) + 1
    bg["source_id"] = max_sid + np.arange(len(bg))
    df = pd.concat([seed_cat.true_df, bg], ignore_index=True)
    return reindex_catalog(df)


def density_scan(
    *,
    out_dir: Path,
    seed_catalogs: dict[int, SeedCatalog],
    densities: list[float],
    dim: int,
    noise_sigma: float,
    query_false_samples: int,
) -> pd.DataFrame:
    rows: list[dict] = []
    for seed, seed_cat in seed_catalogs.items():
        print(f"[density] preparing seed={seed}", flush=True)
        combined = build_combined_df(seed_cat)
        true_pairs = np.asarray(complete_truth_pairs(seed_cat.true_df), dtype=np.int64)
        arr = make_score_arrays(combined, true_df=seed_cat.true_df, seed=seed, dim=dim, noise_sigma=noise_sigma)
        rng = np.random.default_rng(3_100_000 + seed)
        per_query: list[dict] = []
        for a, b in true_pairs:
            for q, partner in ((int(a), int(b)), (int(b), int(a))):
                candidates = rng.integers(0, len(combined), query_false_samples, dtype=np.int64)
                ok = (candidates != q) & (candidates != partner)
                candidates = candidates[ok]
                if len(candidates) < query_false_samples:
                    extra = rng.integers(0, len(combined), query_false_samples - len(candidates), dtype=np.int64)
                    extra = extra[(extra != q) & (extra != partner)]
                    candidates = np.concatenate([candidates, extra])
                candidates = candidates[:query_false_samples]
                all_candidates = np.concatenate([[partner], candidates])
                scores = directed_scores(arr, q, all_candidates)
                for method, s in scores.items():
                    true_score = float(s[0])
                    false_scores = s[1:]
                    per_query.append(
                        {
                            "method": method,
                            "true_score": true_score,
                            "false_survival_probability": fast_query_survival(false_scores, true_score),
                        }
                    )
        for density in densities:
            requested_bg = density * WINDOW_YR
            n_bg_est = estimated_detected_background(requested_bg, seed_cat.detection_fraction)
            n_events_est = int(2 * len(true_pairs) + n_bg_est)
            n_pairs_total = n_events_est * (n_events_est - 1) / 2.0
            total_false_pairs = n_pairs_total - len(true_pairs)
            base_rate = len(true_pairs) / max(n_pairs_total, 1.0)
            n_candidates_per_query = max(n_events_est - 2, 1)
            for method in METHODS:
                qrows = [r for r in per_query if r["method"] == method]
                ranks = np.array([1.0 + r["false_survival_probability"] * n_candidates_per_query for r in qrows])
                rows.append(
                    {
                        "experiment": "density_scan",
                        "background_density_events_per_yr": float(density),
                        "seed": seed,
                        "method": method,
                        "n_events": n_events_est,
                        "n_unlensed_events_estimated": n_bg_est,
                        "n_true_pairs": int(len(true_pairs)),
                        "n_queries": int(len(qrows)),
                        "n_false_pairs_estimated": float(total_false_pairs),
                        "base_positive_rate": float(base_rate),
                        "score_estimation": "per-query sampled false-candidate tail",
                        "false_candidate_samples_per_query": int(query_false_samples),
                        "R@1": float(np.mean(ranks <= 1)),
                        "R@10": float(np.mean(ranks <= 10)),
                        "R@50": float(np.mean(ranks <= 50)),
                        "median_rank": float(np.median(ranks)),
                    }
                )
    density_df = pd.DataFrame(rows)
    density_df.to_csv(out_dir / "density_retrieval_by_method.csv", index=False)
    return density_df


def rarity_fixed_budget_scan(
    *,
    out_dir: Path,
    seed_catalogs: dict[int, SeedCatalog],
    lens_fractions: list[float],
    budgets: list[int],
    n_true_pairs: int,
    dim: int,
    noise_sigma: float,
    n_false_samples: int,
) -> pd.DataFrame:
    rows: list[dict] = []
    for seed, seed_cat in seed_catalogs.items():
        print(f"[fixed-budget] preparing seed={seed}", flush=True)
        rng = np.random.default_rng(9_100_000 + seed)
        combined = build_combined_df(seed_cat)
        true_pairs = np.asarray(complete_truth_pairs(seed_cat.true_df), dtype=np.int64)
        truth_set = set((int(a), int(b)) for a, b in true_pairs)
        arr = make_score_arrays(combined, true_df=seed_cat.true_df, seed=seed, dim=dim, noise_sigma=noise_sigma)
        false_pairs = sample_false_pairs(rng, len(combined), truth_set, n_false_samples)
        ref_pairs = sample_false_pairs(rng, len(combined), truth_set, min(200_000, n_false_samples))
        stats = reference_stats(raw_pair_channels(arr, ref_pairs))
        true_scores_by_method = apply_pair_scores(raw_pair_channels(arr, true_pairs), stats)
        false_scores_by_method = apply_pair_scores(raw_pair_channels(arr, false_pairs), stats)
        tail_models = {method: make_false_survival_model(false_scores_by_method[method]) for method in METHODS}

        for lens_fraction in lens_fractions:
            requested_bg = n_true_pairs / lens_fraction
            n_bg_est = estimated_detected_background(requested_bg, seed_cat.detection_fraction)
            n_events_est = int(2 * len(true_pairs) + n_bg_est)
            n_pairs_total = n_events_est * (n_events_est - 1) / 2.0
            total_false_pairs = n_pairs_total - len(true_pairs)
            base_rate = len(true_pairs) / max(n_pairs_total, 1.0)
            for method in METHODS:
                false_survival, tail_meta = tail_models[method]
                ts = true_scores_by_method[method]
                for budget in budgets:
                    tp = estimated_true_recovered_at_budget(ts, false_survival, total_false_pairs, budget)
                    fp = max(int(budget - tp), 0)
                    precision = tp / max(budget, 1)
                    recall = tp / max(len(true_pairs), 1)
                    enrichment = precision / base_rate if base_rate > 0 else float("nan")
                    rows.append(
                        {
                            "experiment": "rarity_fixed_budget_scan",
                            "lens_fraction": float(lens_fraction),
                            "seed": seed,
                            "method": method,
                            "budget": int(budget),
                            "n_events": n_events_est,
                            "n_true_pairs": int(len(true_pairs)),
                            "n_false_pairs_estimated": float(total_false_pairs),
                            "base_positive_rate": float(base_rate),
                            "true_recovered": int(tp),
                            "false_candidates": int(fp),
                            "recall": float(recall),
                            "precision": float(precision),
                            "enrichment_over_base_rate": float(enrichment),
                            "false_pair_tail_samples": int(n_false_samples),
                            "false_tail_model": tail_meta["tail_model"],
                            "false_tail_threshold": float(tail_meta["tail_threshold"]),
                            "false_tail_fraction": float(tail_meta["tail_fraction"]),
                            "score_estimation": "sampled false-pair tail scaled to estimated full catalog",
                        }
                    )
    fixed_df = pd.DataFrame(rows)
    fixed_df.to_csv(out_dir / "fixed_budget_shortlist_by_method.csv", index=False)
    return fixed_df


def p25(x: pd.Series) -> float:
    return float(np.nanpercentile(x, 25))


def p75(x: pd.Series) -> float:
    return float(np.nanpercentile(x, 75))


def make_summary(density_df: pd.DataFrame, fixed_df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    density_summary = (
        density_df.groupby(["experiment", "background_density_events_per_yr", "method"], dropna=False)
        .agg(
            R1_mean=("R@1", "mean"),
            R1_std=("R@1", "std"),
            R1_median=("R@1", "median"),
            R1_iqr_low=("R@1", p25),
            R1_iqr_high=("R@1", p75),
            R10_mean=("R@10", "mean"),
            R10_std=("R@10", "std"),
            R10_median=("R@10", "median"),
            R10_iqr_low=("R@10", p25),
            R10_iqr_high=("R@10", p75),
            R50_mean=("R@50", "mean"),
            R50_std=("R@50", "std"),
            median_rank_mean=("median_rank", "mean"),
            median_rank_std=("median_rank", "std"),
            n_events_mean=("n_events", "mean"),
            n_true_pairs_mean=("n_true_pairs", "mean"),
            base_positive_rate_mean=("base_positive_rate", "mean"),
            n_seeds=("seed", "nunique"),
        )
        .reset_index()
    )
    density_summary["lens_fraction"] = np.nan
    density_summary["budget"] = np.nan

    fixed_summary = (
        fixed_df.groupby(["experiment", "lens_fraction", "budget", "method"], dropna=False)
        .agg(
            true_recovered_mean=("true_recovered", "mean"),
            true_recovered_std=("true_recovered", "std"),
            true_recovered_median=("true_recovered", "median"),
            true_recovered_iqr_low=("true_recovered", p25),
            true_recovered_iqr_high=("true_recovered", p75),
            false_candidates_mean=("false_candidates", "mean"),
            false_candidates_std=("false_candidates", "std"),
            recall_mean=("recall", "mean"),
            recall_std=("recall", "std"),
            recall_median=("recall", "median"),
            recall_iqr_low=("recall", p25),
            recall_iqr_high=("recall", p75),
            precision_mean=("precision", "mean"),
            precision_std=("precision", "std"),
            precision_median=("precision", "median"),
            precision_iqr_low=("precision", p25),
            precision_iqr_high=("precision", p75),
            enrichment_mean=("enrichment_over_base_rate", "mean"),
            enrichment_std=("enrichment_over_base_rate", "std"),
            enrichment_median=("enrichment_over_base_rate", "median"),
            enrichment_iqr_low=("enrichment_over_base_rate", p25),
            enrichment_iqr_high=("enrichment_over_base_rate", p75),
            base_positive_rate_mean=("base_positive_rate", "mean"),
            n_events_mean=("n_events", "mean"),
            n_true_pairs_mean=("n_true_pairs", "mean"),
            n_seeds=("seed", "nunique"),
        )
        .reset_index()
    )
    fixed_summary["background_density_events_per_yr"] = np.nan
    all_cols = sorted(set(density_summary.columns) | set(fixed_summary.columns))
    summary = pd.concat(
        [density_summary.reindex(columns=all_cols), fixed_summary.reindex(columns=all_cols)],
        ignore_index=True,
    )
    summary.to_csv(out_dir / "method_summary.csv", index=False)
    return summary


def _summary_row(summary: pd.DataFrame, **filters) -> pd.Series:
    sub = summary
    for key, val in filters.items():
        sub = sub[sub[key] == val]
    if sub.empty:
        raise KeyError(filters)
    return sub.iloc[0]


def _fmt(mean: float, std: float, digits: int = 3) -> str:
    if not np.isfinite(std):
        std = 0.0
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def make_figure(density_df: pd.DataFrame, fixed_df: pd.DataFrame, summary: pd.DataFrame, out_dir: Path) -> None:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "font.size": 9,
            "axes.linewidth": 0.9,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    colors = {
        METHOD_TIME_SKY: "#4B5563",
        METHOD_WAVEFORM: "#2563EB",
        METHOD_WEAK: "#B45309",
    }
    labels = {
        METHOD_TIME_SKY: "time-delay + sky",
        METHOD_WAVEFORM: "surrogate waveform",
        METHOD_WEAK: "waveform-first weak fusion",
    }
    rng = np.random.default_rng(123)
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 6.1), constrained_layout=True)
    axes = axes.ravel()

    ax = axes[0]
    for method in METHODS:
        sub = density_df[density_df["method"] == method]
        sm = summary[(summary["experiment"] == "density_scan") & (summary["method"] == method)].sort_values(
            "background_density_events_per_yr"
        )
        ax.plot(sm["background_density_events_per_yr"], sm["R10_mean"], color=colors[method], lw=1.8, marker="o", label=labels[method])
        for x, g in sub.groupby("background_density_events_per_yr"):
            jitter = 10 ** rng.normal(0, 0.015, len(g))
            ax.scatter(np.full(len(g), x) * jitter, g["R@10"], s=10, alpha=0.45, color=colors[method], linewidths=0)
    ax.set_xscale("log")
    ax.set_ylim(-0.04, 1.04)
    ax.set_xlabel("Background density (events yr$^{-1}$)")
    ax.set_ylabel("Companion retrieval R@10")
    ax.set_title("a. Density stress")

    def budget_panel(ax, metric: str, ylabel: str, title: str, lens_fraction: float) -> None:
        data = fixed_df[fixed_df["lens_fraction"] == lens_fraction]
        sm = summary[(summary["experiment"] == "rarity_fixed_budget_scan") & (summary["lens_fraction"] == lens_fraction)]
        for method in METHODS:
            sub = data[data["method"] == method]
            ssub = sm[sm["method"] == method].sort_values("budget")
            mean_col = f"{metric}_mean"
            ax.plot(ssub["budget"], ssub[mean_col], color=colors[method], lw=1.8, marker="o", label=labels[method])
            for x, g in sub.groupby("budget"):
                jitter = rng.normal(0, 2.0, len(g))
                ax.scatter(np.full(len(g), x) + jitter, g[metric], s=10, alpha=0.45, color=colors[method], linewidths=0)
        ax.set_xlabel("Shortlist budget")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(sorted(data["budget"].unique()))

    budget_panel(axes[1], "precision", "Precision", "b. Top-B precision, f$_{lens}=10^{-3}$", 1e-3)
    budget_panel(axes[2], "recall", "Recall", "c. Top-B recall, f$_{lens}=10^{-3}$", 1e-3)
    budget_panel(axes[3], "precision", "Precision", "d. Top-B precision, f$_{lens}=10^{-4}$", 1e-4)
    axes[0].legend(loc="lower left", bbox_to_anchor=(0.0, 1.03, 2.25, 0.1), ncol=3, borderaxespad=0.0)
    fig.savefig(fig_dir / "fig_sequence_mitigation_sandbox_stable.pdf", bbox_inches="tight")
    fig.savefig(fig_dir / "fig_sequence_mitigation_sandbox_stable.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_report(
    *,
    out_dir: Path,
    density_df: pd.DataFrame,
    fixed_df: pd.DataFrame,
    summary: pd.DataFrame,
    args: argparse.Namespace,
) -> dict:
    dens_max = max(args.densities)
    f_mid = 1e-3
    f_low = 1e-4
    high_density = {}
    for method in METHODS:
        r = _summary_row(
            summary,
            experiment="density_scan",
            background_density_events_per_yr=dens_max,
            method=method,
        )
        high_density[method] = {
            "R10_mean": float(r["R10_mean"]),
            "R10_std": float(r["R10_std"]),
            "R10_median": float(r["R10_median"]),
            "R10_iqr": [float(r["R10_iqr_low"]), float(r["R10_iqr_high"])],
            "median_rank_mean": float(r["median_rank_mean"]),
        }
    top_budget = {}
    for lf in [f_mid, f_low]:
        top_budget[str(lf)] = {}
        for method in METHODS:
            r = _summary_row(
                summary,
                experiment="rarity_fixed_budget_scan",
                lens_fraction=lf,
                budget=50,
                method=method,
            )
            top_budget[str(lf)][method] = {
                "precision_mean": float(r["precision_mean"]),
                "precision_std": float(r["precision_std"]),
                "precision_median": float(r["precision_median"]),
                "precision_iqr": [float(r["precision_iqr_low"]), float(r["precision_iqr_high"])],
                "recall_mean": float(r["recall_mean"]),
                "recall_std": float(r["recall_std"]),
                "recall_median": float(r["recall_median"]),
                "true_recovered_mean": float(r["true_recovered_mean"]),
                "false_candidates_mean": float(r["false_candidates_mean"]),
                "enrichment_mean": float(r["enrichment_mean"]),
            }

    payload = {
        "experiment": "sequence_mitigation_sandbox_stable",
        "date": DEFAULT_DATE,
        "output_dir": str(out_dir),
        "catalog_window_years": WINDOW_YR,
        "n_true_pairs_requested": args.n_true_pairs,
        "seeds": args.seeds,
        "n_seeds": len(args.seeds),
        "background_densities_events_per_year": args.densities,
        "lens_fractions": args.lens_fractions,
        "budgets": args.budgets,
        "false_pair_tail_samples_per_seed": args.false_samples,
        "background_pool_detected_events_per_seed": args.background_pool_size,
        "density_false_candidate_samples_per_query": args.density_query_false_samples,
        "waveform_channel": "calibrated surrogate waveform embedding; not a full strain-level waveform encoder result",
        "methods": METHODS,
        "weak_fusion": f"z(waveform_embedding) + {WEAK_TIME_SKY_SCALE} * z(time_delay + sky_localization block)",
        "score_estimation": "sampled false-candidate/pair tails scaled to estimated full catalog burden",
        "high_density_summary": {"density_events_per_year": dens_max, "methods": high_density},
        "top50_summary": top_budget,
    }
    (out_dir / "sequence_mitigation_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def ms(method: str, metric: str, lf: float = f_mid, budget: int = 50) -> str:
        r = _summary_row(
            summary,
            experiment="rarity_fixed_budget_scan",
            lens_fraction=lf,
            budget=budget,
            method=method,
        )
        return _fmt(float(r[f"{metric}_mean"]), float(r[f"{metric}_std"]))

    def ds(method: str) -> str:
        r = _summary_row(
            summary,
            experiment="density_scan",
            background_density_events_per_yr=dens_max,
            method=method,
        )
        return _fmt(float(r["R10_mean"]), float(r["R10_std"]))

    report = f"""# Sequence Mitigation Sandbox Stable Rerun（{DEFAULT_DATE}）

## 实验定位

这是用于 Figure 4 备选图的 **observable-summary sandbox**。它不是完整 strain-level 10^5 event experiment，也不是真实探测声明。实验目的只有一个：在高事件密度和低透镜率下，检查只用 time-delay + sky-localization 是否会被 false coincidence 压垮，以及加入 calibrated surrogate waveform embedding 后，candidate-generation shortlist 是否改善。

## 生成逻辑

- source parameters：复用 `scripts/server_experiments/observable_simulator.py`。源质量 `m1,m2 ~ Uniform(10,70) Msun`，红移按 comoving volume 在 `z=0.01..2` 采样，天空位置各向同性，触发时间在约 10.24 年观测窗内均匀采样。
- SIS lensed pairs：`z_l=z_s/2`，速度弥散 `sigma_v ~ U(100,500) km/s`，`y~U(0.01,0.3)`，按 SIS 放大率和时延公式生成双像。
- PM lensed pairs：`z_l=z_s/2`，点质量 `m_l~U(1e8,1e10) Msun`，`y~U(0.01,0.3)`，按 point-mass 放大率和时延公式生成双像。
- unlensed background：独立采样同一源分布，但 `mu=1`，不含真实 companion。
- SNR threshold：使用 leading-order `rho ∝ sqrt(|mu|) Mc^(5/6)/dL` 加 lognormal scatter；只保留 `network_snr >= {args.snr_threshold}` 的事件。
- sky localization：按 network SNR 估计 `A90 = A90_ref (rho_ref/rho)^2 * lognormal scatter`，再把 `A90` 转成 Gaussian `sigma_sky`，在真实天空附近抽样 observed sky center。ranking 只用 observed sky，不直接用 true sky。
- time-delay score：用真实 lensed delay 构造 `p(delta_t|lensed)`，背景用观测窗内随机事件对的 triangular delay density，输出 log likelihood ratio。
- sky-localization score：由 observed sky 的 normalized separation 构造 step score 和 Gaussian log-overlap。
- surrogate waveform embedding：只用 intrinsic/source summary（chirp mass、mass ratio、luminosity distance、redshift）经随机投影和校准噪声生成，不使用 time 或 sky。它是 calibrated surrogate，不是 full strain waveform encoder。
- waveform-first weak fusion：`z(waveform_embedding) + {WEAK_TIME_SKY_SCALE} * z(time-delay + sky-localization block)`。因此图例中明确写为 waveform-first weak fusion，不写成普通三通道融合。

## 实验设置

- seeds：{len(args.seeds)} 个，`{args.seeds[0]}..{args.seeds[-1]}`
- 每个 seed 完整 true lensed systems：{args.n_true_pairs}
- density scan：{", ".join(f"{x:.0e}" for x in args.densities)} events yr^-1
- rarity/fixed-budget：lens fraction = {", ".join(f"{x:.0e}" for x in args.lens_fractions)}
- budgets：{args.budgets}
- false-pair samples：{args.false_samples:,} per seed
- 结果使用 sampled false tails 缩放到 estimated full catalog false burden；这是 sandbox tail estimate，不是 detection significance。

## 主要结果

最高 density={dens_max:.0e} events yr^-1 下，R@10：

- time-delay + sky-localization：{ds(METHOD_TIME_SKY)}
- calibrated surrogate waveform embedding only：{ds(METHOD_WAVEFORM)}
- waveform-first weak fusion：{ds(METHOD_WEAK)}

在 lens fraction=1e-3、top-50 shortlist 下：

- time-delay + sky precision / recall：{ms(METHOD_TIME_SKY, "precision")} / {ms(METHOD_TIME_SKY, "recall")}
- surrogate waveform precision / recall：{ms(METHOD_WAVEFORM, "precision")} / {ms(METHOD_WAVEFORM, "recall")}
- waveform-first weak fusion precision / recall：{ms(METHOD_WEAK, "precision")} / {ms(METHOD_WEAK, "recall")}

在 lens fraction=1e-4、top-50 shortlist 下：

- time-delay + sky precision / recall：{ms(METHOD_TIME_SKY, "precision", f_low)} / {ms(METHOD_TIME_SKY, "recall", f_low)}
- surrogate waveform precision / recall：{ms(METHOD_WAVEFORM, "precision", f_low)} / {ms(METHOD_WAVEFORM, "recall", f_low)}
- waveform-first weak fusion precision / recall：{ms(METHOD_WEAK, "precision", f_low)} / {ms(METHOD_WEAK, "recall", f_low)}

## 结果解释

这组稳定重跑把两个问题区分开了：

1. **query-level companion retrieval**：在 density scan 的 R@10 指标上，time-delay + sky-localization 仍然很强；因此不能写成“time + sky 在高 density 下的 companion R@10 已经失效”。这个指标回答的是“对一个已知 query，真 companion 能否排进前十”。
2. **global fixed-budget shortlist**：在 lens fraction=1e-3 和 1e-4 的 top-B 全目录 shortlist 中，time-delay + sky-localization 的 true recovered 为 0。这说明即使 query-level recall 很高，全球固定预算候选表仍会被大量 false coincidence 占满。这个指标更接近候选生成的实际 follow-up burden。

calibrated surrogate waveform embedding only 在 fixed-budget shortlist 中能恢复一部分真对，但 seed-to-seed scatter 仍明显。当前 `waveform-first weak fusion` 没有稳定改善，甚至在 top-50/top-100 下基本失败，因此不能把它写成“普通三通道融合稳定优于 waveform-only”。

## 论文可用的保守结论

这组结果更适合支持一个 Supplementary 或 Methods-adjacent 的 design-evidence 结论：在 observable-summary sandbox 中，time-delay + sky-localization 可以保持较好的 query-level companion retrieval，但在低透镜率 fixed-budget global shortlist 中会出现严重 false-candidate burden；calibrated surrogate waveform embedding 可缓解 fixed-budget candidate generation 的一部分压力。当前 waveform-first weak fusion 不是稳定主结果，需要未来 validation-selected fusion 或 two-stage reranking。

不要写成 full strain-level 10^5 event result，不要写成真实 detection，也不要写成三通道融合已稳定优于 waveform-only。

## 文件

- `density_retrieval_by_method.csv`
- `fixed_budget_shortlist_by_method.csv`
- `method_summary.csv`
- `sequence_mitigation_summary.json`
- `figures/fig_sequence_mitigation_sandbox_stable.pdf`
- `figures/fig_sequence_mitigation_sandbox_stable.png`
"""
    (out_dir / "sequence_mitigation_report_cn.md").write_text(report, encoding="utf-8")
    return payload


def package_outputs(repo_root: Path, out_dir: Path, package_path: Path) -> None:
    repo_root = repo_root.resolve()
    out_dir = out_dir.resolve()
    package_path = package_path.resolve()
    package_path.parent.mkdir(parents=True, exist_ok=True)
    members = [
        repo_root / "scripts/experiments/sequence_mitigation_sandbox_stable.py",
        out_dir / "run_config.json",
        out_dir / "density_retrieval_by_method.csv",
        out_dir / "fixed_budget_shortlist_by_method.csv",
        out_dir / "method_summary.csv",
        out_dir / "sequence_mitigation_summary.json",
        out_dir / "sequence_mitigation_report_cn.md",
        out_dir / "figures/fig_sequence_mitigation_sandbox_stable.pdf",
        out_dir / "figures/fig_sequence_mitigation_sandbox_stable.png",
    ]
    with tarfile.open(package_path, "w:gz") as tf:
        for path in members:
            if path.exists():
                tf.add(path, arcname=str(path.relative_to(repo_root)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--package", type=Path, default=DEFAULT_PACKAGE)
    parser.add_argument("--densities", type=float, nargs="+", default=[1e2, 1e3, 1e4, 1e5])
    parser.add_argument("--lens-fractions", type=float, nargs="+", default=[1e-2, 1e-3, 1e-4])
    parser.add_argument("--budgets", type=int, nargs="+", default=[50, 100, 200, 500])
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(20)))
    parser.add_argument("--n-true-pairs", type=int, default=300)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--embedding-noise-sigma", type=float, default=0.06)
    parser.add_argument("--false-samples", type=int, default=500_000)
    parser.add_argument("--density-query-false-samples", type=int, default=20_000)
    parser.add_argument("--background-pool-size", type=int, default=80_000)
    parser.add_argument("--background-chunk-sources", type=int, default=200_000)
    parser.add_argument("--detector", default="ET3")
    parser.add_argument("--snr-threshold", type=float, default=8.0)
    parser.add_argument("--skip-package", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "figures").mkdir(parents=True, exist_ok=True)

    config = {
        "densities": args.densities,
        "lens_fractions": args.lens_fractions,
        "budgets": args.budgets,
        "seeds": args.seeds,
        "n_true_pairs": args.n_true_pairs,
        "embedding_dim": args.embedding_dim,
        "embedding_noise_sigma": args.embedding_noise_sigma,
        "false_samples": args.false_samples,
        "density_query_false_samples": args.density_query_false_samples,
        "background_pool_size": args.background_pool_size,
        "background_chunk_sources": args.background_chunk_sources,
        "detector": args.detector,
        "snr_threshold": args.snr_threshold,
        "catalog_window_years": WINDOW_YR,
        "waveform_channel": "calibrated surrogate waveform embedding only; not full strain-level waveform",
        "weak_fusion": f"z(waveform_embedding) + {WEAK_TIME_SKY_SCALE} * z(time_delay + sky_localization block)",
        "score_estimation": "sampled false-candidate/pair tails scaled to estimated full catalog burden",
    }
    (args.out_dir / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    seed_catalogs: dict[int, SeedCatalog] = {}
    for seed in args.seeds:
        print(f"[seed-catalog] seed={seed}", flush=True)
        seed_catalogs[seed] = build_seed_catalog(
            seed,
            n_true_pairs=args.n_true_pairs,
            detector=args.detector,
            snr_threshold=args.snr_threshold,
            background_pool_size=args.background_pool_size,
            background_chunk_sources=args.background_chunk_sources,
        )
        print(
            f"[seed-catalog] seed={seed} true_pairs={len(complete_truth_pairs(seed_catalogs[seed].true_df))} "
            f"bg_pool={seed_catalogs[seed].background_detected} "
            f"bg_raw_detected={seed_catalogs[seed].background_raw_detected} "
            f"bg_drawn={seed_catalogs[seed].background_sources_drawn} "
            f"det_frac={seed_catalogs[seed].detection_fraction:.4f}",
            flush=True,
        )

    density_df = density_scan(
        out_dir=args.out_dir,
        seed_catalogs=seed_catalogs,
        densities=args.densities,
        dim=args.embedding_dim,
        noise_sigma=args.embedding_noise_sigma,
        query_false_samples=args.density_query_false_samples,
    )
    fixed_df = rarity_fixed_budget_scan(
        out_dir=args.out_dir,
        seed_catalogs=seed_catalogs,
        lens_fractions=args.lens_fractions,
        budgets=args.budgets,
        n_true_pairs=args.n_true_pairs,
        dim=args.embedding_dim,
        noise_sigma=args.embedding_noise_sigma,
        n_false_samples=args.false_samples,
    )
    summary = make_summary(density_df, fixed_df, args.out_dir)
    make_figure(density_df, fixed_df, summary, args.out_dir)
    payload = make_report(out_dir=args.out_dir, density_df=density_df, fixed_df=fixed_df, summary=summary, args=args)
    if not args.skip_package:
        package_outputs(repo_root, args.out_dir, args.package)
    print(json.dumps(payload["high_density_summary"], indent=2), flush=True)
    print(json.dumps(payload["top50_summary"], indent=2), flush=True)
    print(f"wrote outputs to {args.out_dir}", flush=True)
    if not args.skip_package:
        print(f"wrote package to {args.package}", flush=True)


if __name__ == "__main__":
    main()
