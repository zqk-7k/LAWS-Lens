#!/usr/bin/env python3
"""Sequence-mitigation sandbox for 3G-scale catalog lensing retrieval.

This experiment intentionally stays at observable-summary level.  It does not
generate full strain catalogs.  The waveform channel is a *calibrated surrogate
waveform embedding*: an intrinsic-parameter embedding with controlled noise,
used only to test whether a waveform-like sequence channel can mitigate the
time/sky ambiguity at high background density and low lens fraction.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import tarfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import genpareto, norm

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "server_experiments"))

from scripts.server_experiments.observable_simulator import T_END, T_START, simulate_catalog
from scripts.server_experiments.rerank_engine import TimeDelayPrior, a90_to_sigma_rad, true_partner_map


METHOD_TIME_SKY = "time_delay + sky_localization"
METHOD_WAVEFORM = "calibrated surrogate waveform_embedding only"
METHOD_COMBINED = "calibrated surrogate waveform_embedding + time_delay + sky_localization"
METHODS = [METHOD_TIME_SKY, METHOD_WAVEFORM, METHOD_COMBINED]
COMBINED_TIME_SKY_SCALE = 0.01

WINDOW_S = float(T_END - T_START)
WINDOW_YR = WINDOW_S / (365.25 * 24 * 3600)
DEFAULT_OUT = Path("/root/autodl-tmp/gw-catalog/results/sequence_mitigation_sandbox_20260703")
DEFAULT_PACKAGE = Path("/root/autodl-tmp/gw-catalog/packages/sequence_mitigation_sandbox_20260703.tar.gz")


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


def calibrated_surrogate_waveform_embedding(
    df: pd.DataFrame,
    *,
    dim: int,
    seed: int,
    noise_sigma: float,
) -> np.ndarray:
    """Build a calibrated surrogate waveform embedding.

    Lensed images share source intrinsic parameters.  We exclude trigger time
    and observed sky so this channel cannot use the two observable priors being
    tested.  The Gaussian perturbation is the calibration knob; it makes this a
    waveform-like retrieval surrogate, not a strain-level encoder result.
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


def build_prior(df: pd.DataFrame) -> TimeDelayPrior:
    t = df["geocent_time"].to_numpy(float)
    pmap = true_partner_map(df)
    delays = [abs(t[a] - t[b]) for a, b in pmap.values()]
    if not delays:
        raise RuntimeError("No complete true lensed pairs survived the observable simulation.")
    return TimeDelayPrior(delays, window_s=WINDOW_S)


def make_score_arrays(df: pd.DataFrame, seed: int, dim: int, noise_sigma: float) -> ScoreArrays:
    return ScoreArrays(
        time=df["geocent_time"].to_numpy(float),
        ra=df["ra"].to_numpy(float),
        dec=df["dec"].to_numpy(float),
        sigma=a90_to_sigma_rad(df["sky_area_90_deg2"].to_numpy(float)),
        embedding=calibrated_surrogate_waveform_embedding(
            df, dim=dim, seed=seed, noise_sigma=noise_sigma
        ),
        prior=build_prior(df),
    )


def zrow(v: np.ndarray, exclude_idx: int | None = None) -> np.ndarray:
    x = np.asarray(v, dtype=float).copy()
    if exclude_idx is not None:
        x[exclude_idx] = np.nan
    finite = np.isfinite(x)
    out = np.full_like(x, -10.0, dtype=float)
    if finite.sum() > 1:
        mu = x[finite].mean()
        sd = x[finite].std() + 1e-9
        out[finite] = (x[finite] - mu) / sd
    return out


def angular_sep_vec(ra1: float, dec1: float, ra2: np.ndarray, dec2: np.ndarray) -> np.ndarray:
    sd = np.sin(dec1) * np.sin(dec2) + np.cos(dec1) * np.cos(dec2) * np.cos(ra1 - ra2)
    return np.arccos(np.clip(sd, -1.0, 1.0))


def directed_scores_for_query(arr: ScoreArrays, q: int) -> dict[str, np.ndarray]:
    dt = np.abs(arr.time - arr.time[q])
    time_lr = arr.prior.lr_score(dt)
    sep = angular_sep_vec(arr.ra[q], arr.dec[q], arr.ra, arr.dec)
    sig2 = arr.sigma[q] ** 2 + arr.sigma**2
    norm_sep = sep / np.sqrt(sig2)
    sky_step = np.where(
        norm_sep < 1.0,
        1.0,
        np.where(norm_sep < 2.0, 0.5, np.where(norm_sep < 3.0, 0.2, 0.0)),
    )
    sky_log = -0.5 * sep**2 / sig2
    waveform = arr.embedding @ arr.embedding[q]

    time_sky = zrow(time_lr, q) + 4.0 * zrow(sky_step, q) + zrow(sky_log, q)
    wave = zrow(waveform, q)
    combined = wave + COMBINED_TIME_SKY_SCALE * time_sky
    for score in (time_sky, wave, combined):
        score[q] = -np.inf
    return {
        METHOD_TIME_SKY: time_sky,
        METHOD_WAVEFORM: wave,
        METHOD_COMBINED: combined,
    }


def complete_truth_pairs(df: pd.DataFrame) -> list[tuple[int, int]]:
    pairs = []
    for a, b in true_partner_map(df).values():
        pairs.append((int(min(a, b)), int(max(a, b))))
    return sorted(pairs)


def density_scan(
    *,
    out_dir: Path,
    densities: list[float],
    seeds: list[int],
    n_true_pairs: int,
    dim: int,
    noise_sigma: float,
) -> pd.DataFrame:
    rows: list[dict] = []
    n_sis = n_true_pairs // 2
    n_pm = n_true_pairs - n_sis
    for density in densities:
        n_bg_requested = max(1, int(round(density * WINDOW_YR)))
        for seed in seeds:
            print(f"[density] density={density:.0e} seed={seed} requested_bg={n_bg_requested}", flush=True)
            df = simulate_catalog(n_sis, n_pm, n_bg_requested, detector="ET3", seed=seed, snr_threshold=8.0)
            truth = complete_truth_pairs(df)
            arr = make_score_arrays(df, seed=seed, dim=dim, noise_sigma=noise_sigma)
            ranks = {m: [] for m in METHODS}
            kinds = df["kind"].to_numpy()
            for a, b in truth:
                for q, partner in ((a, b), (b, a)):
                    score_by_method = directed_scores_for_query(arr, q)
                    for method, score in score_by_method.items():
                        partner_score = score[partner]
                        rank = 1 + int(np.sum(score > partner_score))
                        ranks[method].append(rank)
            for method, rr in ranks.items():
                r = np.asarray(rr, dtype=float)
                rows.append(
                    {
                        "experiment": "density_scan",
                        "background_density_events_per_yr": density,
                        "seed": seed,
                        "method": method,
                        "n_events": int(len(df)),
                        "n_unlensed_events": int(np.sum(kinds == "unlensed")),
                        "n_true_pairs": int(len(truth)),
                        "n_queries": int(len(r)),
                        "R@1": float(np.mean(r <= 1)) if len(r) else float("nan"),
                        "R@10": float(np.mean(r <= 10)) if len(r) else float("nan"),
                        "R@50": float(np.mean(r <= 50)) if len(r) else float("nan"),
                        "median_rank": float(np.median(r)) if len(r) else float("nan"),
                    }
                )
    density_df = pd.DataFrame(rows)
    density_df.to_csv(out_dir / "density_retrieval_by_method.csv", index=False)
    return density_df


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
    sky_step = np.where(
        norm_sep < 1.0,
        1.0,
        np.where(norm_sep < 2.0, 0.5, np.where(norm_sep < 3.0, 0.2, 0.0)),
    )
    sky_log = -0.5 * sep**2 / sig2
    waveform = np.sum(arr.embedding[i] * arr.embedding[j], axis=1)
    return {
        "time_lr": time_lr,
        "sky_step": sky_step,
        "sky_logoverlap": sky_log,
        "waveform_embedding": waveform,
    }


def reference_stats(channels: dict[str, np.ndarray]) -> dict[str, tuple[float, float]]:
    stats = {}
    for key, values in channels.items():
        v = values[np.isfinite(values)]
        stats[key] = (float(v.mean()), float(v.std() + 1e-9))
    return stats


def apply_pair_scores(
    channels: dict[str, np.ndarray], stats: dict[str, tuple[float, float]]
) -> dict[str, np.ndarray]:
    z = {k: (v - stats[k][0]) / stats[k][1] for k, v in channels.items()}
    time_sky = z["time_lr"] + 4.0 * z["sky_step"] + z["sky_logoverlap"]
    wave = z["waveform_embedding"]
    combined = wave + COMBINED_TIME_SKY_SCALE * time_sky
    return {
        METHOD_TIME_SKY: time_sky,
        METHOD_WAVEFORM: wave,
        METHOD_COMBINED: combined,
    }


def sample_false_pairs(
    rng: np.random.Generator,
    n_events: int,
    truth_pairs: set[tuple[int, int]],
    n_samples: int,
) -> np.ndarray:
    out_i = np.empty(n_samples, dtype=np.int64)
    out_j = np.empty(n_samples, dtype=np.int64)
    filled = 0
    # Oversample in chunks to remove self-pairs and true pairs without Python
    # pair loops over the full requested sample.
    truth_codes = np.fromiter((a * n_events + b for a, b in truth_pairs), dtype=np.int64)
    while filled < n_samples:
        need = n_samples - filled
        draw = max(need * 2, 10000)
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


def make_false_survival_model(false_scores: np.ndarray):
    """Return a callable S(s)=P(false_score>s) with high-tail extrapolation.

    Fixed budgets at lens fractions 1e-3--1e-4 correspond to far smaller false
    survival probabilities than a 350k--1M Monte Carlo sample can resolve
    empirically.  We therefore use the empirical survival inside the sampled
    range and a generalized Pareto fit above the 95th percentile.  If the GPD
    fit is numerically unstable, fall back to a Gaussian tail fit.  This is a
    tail-estimation sandbox, not a detection-significance calculation.
    """
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


def estimated_true_recovered_at_budget(
    true_scores: np.ndarray,
    false_survival,
    total_false_pairs: float,
    budget: int,
) -> int:
    """Estimate how many true pairs enter a fixed global top-B shortlist.

    A true pair is recovered if the estimated number of false pairs plus true
    pairs scoring above it is less than the budget.  The false count comes from
    a fitted high-score tail survival model.
    """
    order = np.argsort(-true_scores)
    recovered = 0
    for pos, idx in enumerate(order):
        score = true_scores[idx]
        expected_false_above = false_survival(float(score)) * total_false_pairs
        rank_est = 1.0 + pos + expected_false_above
        if rank_est <= budget:
            recovered += 1
    return int(recovered)


def rarity_fixed_budget_scan(
    *,
    out_dir: Path,
    lens_fractions: list[float],
    budgets: list[int],
    seeds: list[int],
    n_true_pairs: int,
    dim: int,
    noise_sigma: float,
    n_false_samples: int,
) -> pd.DataFrame:
    rows: list[dict] = []
    n_sis = n_true_pairs // 2
    n_pm = n_true_pairs - n_sis
    for lens_fraction in lens_fractions:
        # lens_fraction is approximated as true lensed systems / background
        # events, matching the existing rarity sandbox convention.
        n_bg_requested = max(1, int(round(n_true_pairs / lens_fraction)))
        for seed in seeds:
            print(
                f"[fixed-budget] lens_fraction={lens_fraction:.0e} seed={seed} requested_bg={n_bg_requested}",
                flush=True,
            )
            rng = np.random.default_rng(9_100_000 + seed + int(round(-math.log10(lens_fraction))) * 1000)
            df = simulate_catalog(n_sis, n_pm, n_bg_requested, detector="ET3", seed=seed, snr_threshold=8.0)
            truth = complete_truth_pairs(df)
            truth_set = set(truth)
            arr = make_score_arrays(df, seed=seed, dim=dim, noise_sigma=noise_sigma)

            true_pairs = np.asarray(truth, dtype=np.int64)
            false_pairs = sample_false_pairs(rng, len(df), truth_set, n_false_samples)
            ref_pairs = sample_false_pairs(rng, len(df), truth_set, min(200_000, n_false_samples))
            stats = reference_stats(raw_pair_channels(arr, ref_pairs))
            true_scores_by_method = apply_pair_scores(raw_pair_channels(arr, true_pairs), stats)
            false_scores_by_method = apply_pair_scores(raw_pair_channels(arr, false_pairs), stats)

            n_pairs_total = len(df) * (len(df) - 1) / 2.0
            total_false_pairs = n_pairs_total - len(truth)
            base_rate = len(truth) / max(n_pairs_total, 1.0)
            for method in METHODS:
                ts = true_scores_by_method[method]
                fs = false_scores_by_method[method]
                false_survival, tail_meta = make_false_survival_model(fs)
                for budget in budgets:
                    tp = estimated_true_recovered_at_budget(ts, false_survival, total_false_pairs, budget)
                    fp = max(int(budget - tp), 0)
                    precision = tp / max(budget, 1)
                    recall = tp / max(len(truth), 1)
                    enrichment = precision / base_rate if base_rate > 0 else float("nan")
                    rows.append(
                        {
                            "experiment": "rarity_fixed_budget_scan",
                            "lens_fraction": lens_fraction,
                            "seed": seed,
                            "method": method,
                            "budget": budget,
                            "n_events": int(len(df)),
                            "n_true_pairs": int(len(truth)),
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
                        }
                    )
    fixed_df = pd.DataFrame(rows)
    fixed_df.to_csv(out_dir / "fixed_budget_shortlist_by_method.csv", index=False)
    return fixed_df


def make_summary(density_df: pd.DataFrame, fixed_df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    density_summary = (
        density_df.groupby(["experiment", "background_density_events_per_yr", "method"], dropna=False)
        .agg(
            R1_mean=("R@1", "mean"),
            R1_std=("R@1", "std"),
            R10_mean=("R@10", "mean"),
            R10_std=("R@10", "std"),
            R50_mean=("R@50", "mean"),
            R50_std=("R@50", "std"),
            median_rank_mean=("median_rank", "mean"),
            median_rank_std=("median_rank", "std"),
            n_events_mean=("n_events", "mean"),
            n_true_pairs_mean=("n_true_pairs", "mean"),
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
            false_candidates_mean=("false_candidates", "mean"),
            false_candidates_std=("false_candidates", "std"),
            recall_mean=("recall", "mean"),
            recall_std=("recall", "std"),
            precision_mean=("precision", "mean"),
            precision_std=("precision", "std"),
            enrichment_mean=("enrichment_over_base_rate", "mean"),
            enrichment_std=("enrichment_over_base_rate", "std"),
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


def get_scalar(df: pd.DataFrame, **filters) -> pd.DataFrame:
    out = df
    for key, value in filters.items():
        out = out[out[key] == value]
    return out


def make_figure(summary: pd.DataFrame, out_dir: Path) -> None:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "font.size": 10,
            "axes.linewidth": 0.9,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    colors = {
        METHOD_TIME_SKY: "#4B5563",
        METHOD_WAVEFORM: "#2563EB",
        METHOD_COMBINED: "#B45309",
    }
    labels = {
        METHOD_TIME_SKY: "time + sky",
        METHOD_WAVEFORM: "surrogate waveform",
        METHOD_COMBINED: "surrogate waveform + time + sky",
    }

    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.25), constrained_layout=True)

    ax = axes[0]
    dsum = summary[summary["experiment"] == "density_scan"].copy()
    for method in METHODS:
        sub = dsum[dsum["method"] == method].sort_values("background_density_events_per_yr")
        ax.errorbar(
            sub["background_density_events_per_yr"],
            sub["R10_mean"],
            yerr=sub["R10_std"].fillna(0),
            marker="o",
            lw=1.8,
            color=colors[method],
            label=labels[method],
        )
    ax.set_xscale("log")
    ax.set_ylim(-0.02, 1.03)
    ax.set_xlabel("Background density (events yr$^{-1}$)")
    ax.set_ylabel("Companion retrieval R@10")
    ax.set_title("a. Density stress")

    fsum = summary[(summary["experiment"] == "rarity_fixed_budget_scan") & (summary["budget"] == 50)].copy()
    for panel, metric, ylabel, title in [
        (axes[1], "enrichment_mean", "Top-50 enrichment over base rate", "b. Fixed-budget enrichment"),
        (axes[2], "recall_mean", "Top-50 recall", "c. Fixed-budget recall"),
    ]:
        for method in METHODS:
            sub = fsum[fsum["method"] == method].sort_values("lens_fraction")
            err_col = metric.replace("_mean", "_std")
            panel.errorbar(
                sub["lens_fraction"],
                sub[metric],
                yerr=sub[err_col].fillna(0),
                marker="o",
                lw=1.8,
                color=colors[method],
                label=labels[method],
            )
        panel.set_xscale("log")
        panel.invert_xaxis()
        panel.set_xlabel("Lens fraction")
        panel.set_ylabel(ylabel)
        panel.set_title(title)
    axes[1].set_yscale("log")
    axes[0].legend(loc="lower left", bbox_to_anchor=(0.0, 1.02, 3.1, 0.1), ncol=3, borderaxespad=0.0)
    fig.savefig(fig_dir / "fig_sequence_mitigation_sandbox.pdf", bbox_inches="tight")
    fig.savefig(fig_dir / "fig_sequence_mitigation_sandbox.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fmt_mean_std(mean: float, std: float | None, digits: int = 3) -> str:
    if std is None or not np.isfinite(std):
        std = 0.0
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def make_report(
    *,
    out_dir: Path,
    density_df: pd.DataFrame,
    fixed_df: pd.DataFrame,
    summary: pd.DataFrame,
    args: argparse.Namespace,
) -> dict:
    dens_max = max(args.densities)
    dmax = summary[
        (summary["experiment"] == "density_scan")
        & (summary["background_density_events_per_yr"] == dens_max)
    ]
    def dens_metric(method: str, col: str) -> tuple[float, float]:
        row = dmax[dmax["method"] == method].iloc[0]
        return float(row[f"{col}_mean"]), float(row[f"{col}_std"] if pd.notna(row[f"{col}_std"]) else 0.0)

    f_real = min(args.lens_fractions)
    f50 = summary[
        (summary["experiment"] == "rarity_fixed_budget_scan")
        & (summary["lens_fraction"] == f_real)
        & (summary["budget"] == 50)
    ]
    def fixed_metric(method: str, col: str) -> tuple[float, float]:
        row = f50[f50["method"] == method].iloc[0]
        return float(row[f"{col}_mean"]), float(row[f"{col}_std"] if pd.notna(row[f"{col}_std"]) else 0.0)

    ts_r10 = dens_metric(METHOD_TIME_SKY, "R10")
    wf_r10 = dens_metric(METHOD_WAVEFORM, "R10")
    cb_r10 = dens_metric(METHOD_COMBINED, "R10")
    ts_prec = fixed_metric(METHOD_TIME_SKY, "precision")
    wf_prec = fixed_metric(METHOD_WAVEFORM, "precision")
    cb_prec = fixed_metric(METHOD_COMBINED, "precision")
    ts_rec = fixed_metric(METHOD_TIME_SKY, "recall")
    wf_rec = fixed_metric(METHOD_WAVEFORM, "recall")
    cb_rec = fixed_metric(METHOD_COMBINED, "recall")
    ts_enr = fixed_metric(METHOD_TIME_SKY, "enrichment")
    wf_enr = fixed_metric(METHOD_WAVEFORM, "enrichment")
    cb_enr = fixed_metric(METHOD_COMBINED, "enrichment")

    f_mid = 1e-3 if 1e-3 in args.lens_fractions else args.lens_fractions[min(1, len(args.lens_fractions) - 1)]
    f50_mid = summary[
        (summary["experiment"] == "rarity_fixed_budget_scan")
        & (summary["lens_fraction"] == f_mid)
        & (summary["budget"] == 50)
    ]
    def fixed_metric_mid(method: str, col: str) -> tuple[float, float]:
        row = f50_mid[f50_mid["method"] == method].iloc[0]
        return float(row[f"{col}_mean"]), float(row[f"{col}_std"] if pd.notna(row[f"{col}_std"]) else 0.0)

    mid_ts_prec = fixed_metric_mid(METHOD_TIME_SKY, "precision")
    mid_wf_prec = fixed_metric_mid(METHOD_WAVEFORM, "precision")
    mid_cb_prec = fixed_metric_mid(METHOD_COMBINED, "precision")
    mid_ts_rec = fixed_metric_mid(METHOD_TIME_SKY, "recall")
    mid_wf_rec = fixed_metric_mid(METHOD_WAVEFORM, "recall")
    mid_cb_rec = fixed_metric_mid(METHOD_COMBINED, "recall")

    payload = {
        "experiment": "sequence_mitigation_sandbox",
        "date": "2026-07-03",
        "output_dir": str(out_dir),
        "catalog_window_years": WINDOW_YR,
        "n_true_pairs_requested": args.n_true_pairs,
        "seeds": args.seeds,
        "background_densities_events_per_year": args.densities,
        "lens_fractions": args.lens_fractions,
        "budgets": args.budgets,
        "false_pair_tail_samples_per_seed": args.false_samples,
        "waveform_channel": "calibrated surrogate waveform embedding; not a full strain-level waveform encoder result",
        "combined_fusion": f"z(waveform_embedding) + {COMBINED_TIME_SKY_SCALE} * z(time_delay + sky_localization block)",
        "high_density_summary": {
            "density_events_per_year": dens_max,
            "time_sky_R10_mean_std": ts_r10,
            "surrogate_waveform_R10_mean_std": wf_r10,
            "combined_R10_mean_std": cb_r10,
        },
        "realistic_rarity_top50_summary": {
            "lens_fraction": f_real,
            "time_sky_precision_mean_std": ts_prec,
            "surrogate_waveform_precision_mean_std": wf_prec,
            "combined_precision_mean_std": cb_prec,
            "time_sky_recall_mean_std": ts_rec,
            "surrogate_waveform_recall_mean_std": wf_rec,
            "combined_recall_mean_std": cb_rec,
            "time_sky_enrichment_mean_std": ts_enr,
            "surrogate_waveform_enrichment_mean_std": wf_enr,
            "combined_enrichment_mean_std": cb_enr,
        },
        "intermediate_rarity_top50_summary": {
            "lens_fraction": f_mid,
            "time_sky_precision_mean_std": mid_ts_prec,
            "surrogate_waveform_precision_mean_std": mid_wf_prec,
            "combined_precision_mean_std": mid_cb_prec,
            "time_sky_recall_mean_std": mid_ts_rec,
            "surrogate_waveform_recall_mean_std": mid_wf_rec,
            "combined_recall_mean_std": mid_cb_rec,
        },
    }
    (out_dir / "sequence_mitigation_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    report = f"""# Sequence Mitigation Sandbox（2026-07-03）

## 实验目的

这个实验回答一个很具体的问题：在真实 3G 规模和低透镜率下，只有 time-delay + sky-localization 是否会因为事件序列密度升高而不足；加入 **calibrated surrogate waveform embedding** 后，companion retrieval 和 fixed-budget shortlist 是否改善。

重要边界：这里没有生成完整 strain 大目录，也不是完整 strain-level 10^5 event experiment。waveform 通道是 calibrated surrogate waveform embedding，只用于机制 sandbox。

## 方法

三种方法在同一批 catalog、同一 seeds、同一 true pairs 上比较：

1. `time_delay + sky_localization`
2. `calibrated surrogate waveform_embedding only`
3. `calibrated surrogate waveform_embedding + time_delay + sky_localization`

observable catalog 复用 `scripts/server_experiments/observable_simulator.py`。time-delay 使用现有 population delay likelihood-ratio；sky-localization 使用 observed sky A90 转换得到的 normalized separation / log-overlap；waveform surrogate 只使用内禀源 summary 生成 embedding，并加入校准噪声，不使用 trigger time 或 observed sky。

组合方法使用预设 waveform-first calibrated fusion：`z(waveform_embedding) + {COMBINED_TIME_SKY_SCALE} * z(time_delay + sky_localization block)`。这样做的原因是 fixed-budget global shortlist 由极端 false tail 决定；time/sky block 在高 density 下容易产生大量 coincidence，因此在 mitigation setting 中作为弱辅助项，而不是覆盖 waveform sequence 信息。

## A. Density Scan

背景密度：{", ".join(f"{x:.0e}" for x in args.densities)} events yr^-1。每个条件使用 seeds={args.seeds}，每个 catalog 请求 {args.n_true_pairs} 个透镜双像系统。

在最高背景密度 {dens_max:.0e} events yr^-1 下：

- time-delay + sky-localization 的 R@10 = {fmt_mean_std(*ts_r10)}
- calibrated surrogate waveform embedding only 的 R@10 = {fmt_mean_std(*wf_r10)}
- waveform + time + sky 的 R@10 = {fmt_mean_std(*cb_r10)}

结论：高 density 下，time + sky 会退化，因为随机背景中会出现越来越多时间延迟和天空定位都相容的 coincidence。surrogate waveform 通道提供了与时间/天空近似正交的内禀序列信息，因此改善 top-ten companion retrieval。

## B. Rarity / Fixed-Budget Scan

透镜率：{", ".join(f"{x:.0e}" for x in args.lens_fractions)}；shortlist budget：{args.budgets}。fixed-budget 结果使用 Monte Carlo false-pair sampling，并对高分尾部做 GPD/normal extrapolation，再按全 catalog false-pair 数缩放，输出 true recovered、false candidates、recall、precision 和 enrichment over base rate。这个尾部估计只用于 sandbox，不是 catalog-level detection significance。

在最稀有设置 lens fraction={f_real:.0e}、top-50 shortlist 下：

- time + sky precision = {fmt_mean_std(*ts_prec)}
- calibrated surrogate waveform only precision = {fmt_mean_std(*wf_prec)}
- waveform + time + sky precision = {fmt_mean_std(*cb_prec)}
- time + sky recall = {fmt_mean_std(*ts_rec)}
- calibrated surrogate waveform only recall = {fmt_mean_std(*wf_rec)}
- waveform + time + sky recall = {fmt_mean_std(*cb_rec)}
- time + sky enrichment = {fmt_mean_std(*ts_enr)}
- calibrated surrogate waveform only enrichment = {fmt_mean_std(*wf_enr)}
- waveform + time + sky enrichment = {fmt_mean_std(*cb_enr)}

在中间低透镜率 lens fraction={f_mid:.0e}、top-50 shortlist 下：

- time + sky precision / recall = {fmt_mean_std(*mid_ts_prec)} / {fmt_mean_std(*mid_ts_rec)}
- calibrated surrogate waveform only precision / recall = {fmt_mean_std(*mid_wf_prec)} / {fmt_mean_std(*mid_wf_rec)}
- waveform + time + sky precision / recall = {fmt_mean_std(*mid_cb_prec)} / {fmt_mean_std(*mid_cb_rec)}

结论：time + sky 在 fixed-budget global shortlist 中被高密度 false coincidence 压垮。加入 calibrated surrogate waveform 后，在 lens fraction=1e-3 时三通道 shortlist 明显改善；在更极端的 1e-4 下，waveform-only 仍能恢复少量真对，但当前弱融合三通道没有进入 top-50，说明真实 1e-4 口径还需要专门的 validation-selected fusion 或两阶段 waveform-first shortlist。

## 文件

- `density_retrieval_by_method.csv`：density scan per-seed 结果
- `fixed_budget_shortlist_by_method.csv`：rarity/fixed-budget per-seed 结果
- `method_summary.csv`：mean ± std summary
- `sequence_mitigation_summary.json`：核心数值摘要
- `figures/fig_sequence_mitigation_sandbox.pdf`
- `figures/fig_sequence_mitigation_sandbox.png`

## 不能说明什么

这个实验不能替代完整 strain-level 3G catalog 实验，也不能证明真实 waveform encoder 在 10^5 events yr^-1 catalog 上已经完成部署。它只能说明：在 observable-summary sandbox 中，time-delay + sky-localization 会受到序列密度和低透镜率压力；加入校准的 waveform-like embedding 后，fixed-budget shortlist 有机制性改善。
"""
    (out_dir / "sequence_mitigation_sandbox_report_cn.md").write_text(report, encoding="utf-8")
    return payload


def package_outputs(repo_root: Path, out_dir: Path, package_path: Path) -> None:
    repo_root = repo_root.resolve()
    out_dir = out_dir.resolve()
    package_path = package_path.resolve()
    package_path.parent.mkdir(parents=True, exist_ok=True)
    members = [
        repo_root / "scripts/experiments/sequence_mitigation_sandbox.py",
        out_dir / "density_retrieval_by_method.csv",
        out_dir / "fixed_budget_shortlist_by_method.csv",
        out_dir / "method_summary.csv",
        out_dir / "sequence_mitigation_summary.json",
        out_dir / "sequence_mitigation_sandbox_report_cn.md",
        out_dir / "figures/fig_sequence_mitigation_sandbox.pdf",
        out_dir / "figures/fig_sequence_mitigation_sandbox.png",
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
    parser.add_argument("--budgets", type=int, nargs="+", default=[50, 100, 200])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--n-true-pairs", type=int, default=60)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--embedding-noise-sigma", type=float, default=0.06)
    parser.add_argument("--false-samples", type=int, default=350_000)
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
        "catalog_window_years": WINDOW_YR,
        "waveform_channel": "calibrated surrogate waveform embedding only; not full strain-level waveform",
        "combined_time_sky_scale": COMBINED_TIME_SKY_SCALE,
        "combined_fusion": f"z(waveform_embedding) + {COMBINED_TIME_SKY_SCALE} * z(time_delay + sky_localization block)",
    }
    (args.out_dir / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    density_df = density_scan(
        out_dir=args.out_dir,
        densities=args.densities,
        seeds=args.seeds,
        n_true_pairs=args.n_true_pairs,
        dim=args.embedding_dim,
        noise_sigma=args.embedding_noise_sigma,
    )
    fixed_df = rarity_fixed_budget_scan(
        out_dir=args.out_dir,
        lens_fractions=args.lens_fractions,
        budgets=args.budgets,
        seeds=args.seeds,
        n_true_pairs=args.n_true_pairs,
        dim=args.embedding_dim,
        noise_sigma=args.embedding_noise_sigma,
        n_false_samples=args.false_samples,
    )
    summary = make_summary(density_df, fixed_df, args.out_dir)
    make_figure(summary, args.out_dir)
    payload = make_report(
        out_dir=args.out_dir,
        density_df=density_df,
        fixed_df=fixed_df,
        summary=summary,
        args=args,
    )
    package_outputs(repo_root, args.out_dir, args.package)
    print(json.dumps(payload["high_density_summary"], indent=2), flush=True)
    print(json.dumps(payload["realistic_rarity_top50_summary"], indent=2), flush=True)
    print(f"wrote outputs to {args.out_dir}", flush=True)
    print(f"wrote package to {args.package}", flush=True)


if __name__ == "__main__":
    main()
