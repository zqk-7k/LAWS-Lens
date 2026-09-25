#!/usr/bin/env python3
"""Fast, read-only audit of synthetic-vs-real sky-score domain shift.

This script deliberately does not perform parameter estimation and does not
modify the v9.3 products.  It compares the existing O3/O4a synthetic sky
validation/test distributions with public-PE real-catalog sky scores at
multiple HEALPix resolutions.  The purpose is to separate:

1. a sky-posterior morphology/domain mismatch at the same Nside=64; and
2. a numerical resolution effect within the real public-PE maps.

The outputs are exploratory diagnostics, not a replacement ranking.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, spearmanr, wasserstein_distance


DEPLOYMENTS = ("gwtc3", "gwtc4")
NSIDES = (64, 512, 1024)
QUANTILES = (0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 0.995, 0.999, 1.0)
TAIL_QUANTILES = (0.90, 0.95, 0.99, 0.995, 0.999)
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 2026082306


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def finite(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    return arr[np.isfinite(arr)]


def canonical_pair(a: object, b: object) -> str:
    x, y = str(a), str(b)
    return f"{x}--{y}" if x <= y else f"{y}--{x}"


def pair_name_columns(frame: pd.DataFrame) -> tuple[str, str]:
    candidates = (
        ("event_i", "event_j"),
        ("event_1", "event_2"),
        ("event_a", "event_b"),
        ("name_i", "name_j"),
    )
    for left, right in candidates:
        if left in frame.columns and right in frame.columns:
            return left, right
    raise KeyError(f"Could not identify event-pair columns: {frame.columns.tolist()}")


def add_pair_key(frame: pd.DataFrame) -> pd.DataFrame:
    left, right = pair_name_columns(frame)
    out = frame.copy()
    out["pair_key"] = [canonical_pair(a, b) for a, b in zip(out[left], out[right])]
    return out


def quantile_record(values: Iterable[float], **labels: object) -> dict[str, object]:
    arr = finite(values)
    record: dict[str, object] = dict(labels)
    record["n"] = int(arr.size)
    record["mean"] = float(np.mean(arr)) if arr.size else math.nan
    record["std"] = float(np.std(arr, ddof=1)) if arr.size > 1 else math.nan
    for q in QUANTILES:
        key = "min" if q == 0 else "max" if q == 1 else f"q{1000 * q:04.0f}"
        record[key] = float(np.quantile(arr, q)) if arr.size else math.nan
    return record


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def strict_pair_keys(seed_dir: Path) -> set[str]:
    preferred = seed_dir / "real_pair_scores_strict_h1l1_bbh_candidate_three_channel_strict_positive_v81.parquet"
    candidates = [preferred] if preferred.exists() else []
    candidates.extend(sorted(seed_dir.glob("real_pair_scores_strict_h1l1_bbh*.parquet")))
    for path in candidates:
        frame = pd.read_parquet(path)
        try:
            keyed = add_pair_key(frame)
        except KeyError:
            continue
        return set(keyed["pair_key"].astype(str))

    features = pd.read_parquet(seed_dir / "real_pair_features_unified_sky_v81.parquet")
    keyed = add_pair_key(features)
    strict_columns = [
        c
        for c in keyed.columns
        if "strict" in c.lower() and ("pair" in c.lower() or "h1l1" in c.lower())
    ]
    if strict_columns:
        mask = keyed[strict_columns[0]].fillna(False).astype(bool)
        return set(keyed.loc[mask, "pair_key"].astype(str))
    return set(keyed["pair_key"].astype(str))


def score_column(frame: pd.DataFrame) -> str:
    for name in ("sky_log_bayes_factor_raw", "sky_score", "sky_log_bf"):
        if name in frame.columns:
            return name
    raise KeyError(f"No synthetic sky-score column in {frame.columns.tolist()}")


def bool_true_pair(frame: pd.DataFrame) -> np.ndarray:
    for name in ("is_true_pair", "true_pair", "is_companion"):
        if name in frame.columns:
            return frame[name].fillna(False).astype(bool).to_numpy()
    raise KeyError("No true-pair label found")


def vertex_bootstrap_tail(
    frame: pd.DataFrame,
    score_col: str,
    threshold: float,
    replicates: int,
    seed: int,
) -> tuple[float, float]:
    left, right = pair_name_columns(frame)
    events = sorted(set(frame[left].astype(str)) | set(frame[right].astype(str)))
    event_to_idx = {event: idx for idx, event in enumerate(events)}
    ii = frame[left].astype(str).map(event_to_idx).to_numpy(dtype=np.int32)
    jj = frame[right].astype(str).map(event_to_idx).to_numpy(dtype=np.int32)
    exceed = (frame[score_col].to_numpy(dtype=np.float64) > threshold).astype(np.float64)
    rng = np.random.default_rng(seed)
    rates: list[np.ndarray] = []
    batch = 200
    probability = np.full(len(events), 1.0 / len(events), dtype=np.float64)
    for start in range(0, replicates, batch):
        size = min(batch, replicates - start)
        counts = rng.multinomial(len(events), probability, size=size).astype(np.float64)
        weights = counts[:, ii] * counts[:, jj]
        denominator = weights.sum(axis=1)
        numerator = weights @ exceed
        valid = denominator > 0
        rates.append(np.divide(numerator, denominator, out=np.full(size, np.nan), where=valid))
    values = finite(np.concatenate(rates))
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def top_overlap(a: np.ndarray, b: np.ndarray, k: int) -> float:
    if a.size == 0:
        return math.nan
    k = min(k, a.size)
    left = set(np.argpartition(-a, k - 1)[:k].tolist())
    right = set(np.argpartition(-b, k - 1)[:k].tolist())
    return len(left & right) / k


def metric_column(frame: pd.DataFrame, metric: str, real: bool) -> str | None:
    choices = {
        "area90_deg2": ("area90_nside1024_deg2", "area90_deg2", "sky_area90_deg2") if real else ("area90_deg2", "sky_area90_deg2"),
        "kl_to_uniform_nats": ("kl_to_uniform_nats", "sky_kl_nats"),
        "effective_area_deg2": ("effective_area_deg2", "sky_effective_area_deg2"),
        "entropy_nats": ("entropy_nats", "sky_entropy_nats"),
    }
    for name in choices[metric]:
        if name in frame.columns:
            return name
    return None


def load_inputs(root: Path) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    dict[str, dict[str, object]],
    list[Path],
    dict[str, dict[str, np.ndarray]],
]:
    pair_stats: list[dict[str, object]] = []
    event_stats: list[dict[str, object]] = []
    shift_tests: list[dict[str, object]] = []
    tail_rows: list[dict[str, object]] = []
    resolution_rows: list[dict[str, object]] = []
    headline: dict[str, dict[str, object]] = {}
    used_paths: list[Path] = []
    plot_data: dict[str, dict[str, np.ndarray]] = {}

    for deployment_index, deployment in enumerate(DEPLOYMENTS):
        dep_root = root / deployment
        seed_dirs = sorted(path for path in dep_root.glob("seed_*") if path.is_dir())
        if not seed_dirs:
            raise FileNotFoundError(f"No seed directories below {dep_root}")
        resolution_path = dep_root / "real_sky_resolution_convergence_all_pairs_v93.parquet"
        resolution_all = add_pair_key(pd.read_parquet(resolution_path))
        used_paths.append(resolution_path)
        dep_plot: dict[str, list[np.ndarray] | np.ndarray] = {
            "synthetic_null": [],
            "synthetic_true": [],
            "synthetic_area90": [],
        }

        for seed_index, seed_dir in enumerate(seed_dirs):
            seed = seed_dir.name.removeprefix("seed_")
            strict_keys = strict_pair_keys(seed_dir)
            real = resolution_all.loc[resolution_all["pair_key"].isin(strict_keys)].copy()
            if real.empty:
                raise RuntimeError(f"Strict real-pair merge produced zero rows for {deployment}/{seed}")

            synthetic_frames: dict[str, pd.DataFrame] = {}
            for split, filename in (
                ("validation", "fusion_validation_pairs_v81.parquet"),
                ("test", "fusion_heldout_test_pairs_v81.parquet"),
            ):
                path = seed_dir / filename
                frame = pd.read_parquet(path)
                used_paths.append(path)
                synthetic_frames[split] = frame
                col = score_column(frame)
                true_mask = bool_true_pair(frame)
                for pair_class, mask in (("true_companion", true_mask), ("non_companion", ~true_mask)):
                    values = finite(frame.loc[mask, col])
                    pair_stats.append(
                        quantile_record(
                            values,
                            deployment=deployment,
                            seed=seed,
                            source="synthetic_injection",
                            split=split,
                            pair_class=pair_class,
                            nside=64,
                        )
                    )
                    if split == "test":
                        key = "synthetic_true" if pair_class == "true_companion" else "synthetic_null"
                        assert isinstance(dep_plot[key], list)
                        dep_plot[key].append(values)

            for nside in NSIDES:
                col = f"sky_log_bf_nside{nside}"
                values = finite(real[col])
                pair_stats.append(
                    quantile_record(
                        values,
                        deployment=deployment,
                        seed=seed,
                        source="real_gwtc",
                        split="strict_h1l1_bbh_catalog",
                        pair_class="unknown_real_pair",
                        nside=nside,
                    )
                )

            validation = synthetic_frames["validation"]
            test = synthetic_frames["test"]
            vcol, tcol = score_column(validation), score_column(test)
            validation_null = finite(validation.loc[~bool_true_pair(validation), vcol])
            test_null = finite(test.loc[~bool_true_pair(test), tcol])
            for nside in NSIDES:
                real_col = f"sky_log_bf_nside{nside}"
                real_values = finite(real[real_col])
                ks = ks_2samp(validation_null, real_values, alternative="two-sided", method="auto")
                shift_tests.append(
                    {
                        "deployment": deployment,
                        "seed": seed,
                        "comparison": f"validation_injection_null_nside64_vs_real_nside{nside}",
                        "n_reference": int(validation_null.size),
                        "n_target": int(real_values.size),
                        "ks_statistic": float(ks.statistic),
                        "ks_pvalue": float(ks.pvalue),
                        "wasserstein_nats": float(wasserstein_distance(validation_null, real_values)),
                        "median_shift_nats": float(np.median(real_values) - np.median(validation_null)),
                        "q99_shift_nats": float(np.quantile(real_values, 0.99) - np.quantile(validation_null, 0.99)),
                    }
                )

                for q in TAIL_QUANTILES:
                    threshold = float(np.quantile(validation_null, q))
                    reference_rate = float(np.mean(validation_null > threshold))
                    for target_name, target_values, target_frame, target_col in (
                        ("heldout_injection_null_nside64", test_null, None, None),
                        (f"real_strict_pairs_nside{nside}", real_values, real, real_col),
                    ):
                        observed_rate = float(np.mean(target_values > threshold))
                        ci_low = ci_high = math.nan
                        if target_frame is not None and target_col is not None:
                            ci_low, ci_high = vertex_bootstrap_tail(
                                target_frame,
                                target_col,
                                threshold,
                                BOOTSTRAP_REPLICATES,
                                BOOTSTRAP_SEED + deployment_index * 10000 + seed_index * 100 + nside + int(q * 1000),
                            )
                        tail_rows.append(
                            {
                                "deployment": deployment,
                                "seed": seed,
                                "reference": "validation_injection_null_nside64",
                                "target": target_name,
                                "reference_quantile": q,
                                "threshold_nats": threshold,
                                "reference_exceedance_rate": reference_rate,
                                "target_n": int(target_values.size),
                                "target_exceedance_count": int(np.sum(target_values > threshold)),
                                "target_exceedance_rate": observed_rate,
                                "target_event_block_bootstrap_ci_low": ci_low,
                                "target_event_block_bootstrap_ci_high": ci_high,
                                "tail_inflation_ratio": observed_rate / reference_rate if reference_rate > 0 else math.inf,
                            }
                        )

            for low, high in ((64, 512), (512, 1024), (64, 1024)):
                left_col, right_col = f"sky_log_bf_nside{low}", f"sky_log_bf_nside{high}"
                left_values = real[left_col].to_numpy(dtype=np.float64)
                right_values = real[right_col].to_numpy(dtype=np.float64)
                valid = np.isfinite(left_values) & np.isfinite(right_values)
                left_values, right_values = left_values[valid], right_values[valid]
                delta = np.abs(right_values - left_values)
                sign_flip = np.signbit(left_values) != np.signbit(right_values)
                rho = spearmanr(left_values, right_values).statistic
                resolution_rows.append(
                    {
                        "deployment": deployment,
                        "seed": seed,
                        "comparison": f"nside{low}_to_nside{high}",
                        "n_pairs": int(delta.size),
                        "median_abs_delta_nats": float(np.median(delta)),
                        "q90_abs_delta_nats": float(np.quantile(delta, 0.90)),
                        "q95_abs_delta_nats": float(np.quantile(delta, 0.95)),
                        "q99_abs_delta_nats": float(np.quantile(delta, 0.99)),
                        "max_abs_delta_nats": float(np.max(delta)),
                        "sign_flip_count": int(sign_flip.sum()),
                        "sign_flip_fraction": float(sign_flip.mean()),
                        "spearman": float(rho),
                        "top10_overlap": top_overlap(left_values, right_values, 10),
                        "top20_overlap": top_overlap(left_values, right_values, 20),
                        "top50_overlap": top_overlap(left_values, right_values, 50),
                    }
                )

            synthetic_events: list[pd.DataFrame] = []
            for split, filename in (
                ("validation", "synthetic_validation_event_sky_diagnostics_v81.parquet"),
                ("test", "synthetic_test_event_sky_diagnostics_v81.parquet"),
            ):
                path = seed_dir / filename
                frame = pd.read_parquet(path)
                frame = frame.assign(_split=split)
                synthetic_events.append(frame)
                used_paths.append(path)
            synthetic_event = pd.concat(synthetic_events, ignore_index=True)
            real_event_path = seed_dir / "real_event_sky_diagnostics_v81.parquet"
            real_event = pd.read_parquet(real_event_path)
            used_paths.append(real_event_path)

            for metric in ("area90_deg2", "kl_to_uniform_nats", "effective_area_deg2", "entropy_nats"):
                synthetic_col = metric_column(synthetic_event, metric, real=False)
                real_col = metric_column(real_event, metric, real=True)
                if synthetic_col is None or real_col is None:
                    continue
                synthetic_values = finite(synthetic_event[synthetic_col])
                real_values = finite(real_event[real_col])
                event_stats.append(
                    quantile_record(
                        synthetic_values,
                        deployment=deployment,
                        seed=seed,
                        source="synthetic_injection",
                        metric=metric,
                        native_analysis_nside=64,
                    )
                )
                event_stats.append(
                    quantile_record(
                        real_values,
                        deployment=deployment,
                        seed=seed,
                        source="real_public_pe",
                        metric=metric,
                        native_analysis_nside=1024,
                    )
                )
                ks = ks_2samp(synthetic_values, real_values, alternative="two-sided", method="auto")
                shift_tests.append(
                    {
                        "deployment": deployment,
                        "seed": seed,
                        "comparison": f"event_metric_synthetic_vs_real:{metric}",
                        "n_reference": int(synthetic_values.size),
                        "n_target": int(real_values.size),
                        "ks_statistic": float(ks.statistic),
                        "ks_pvalue": float(ks.pvalue),
                        "wasserstein_nats": float(wasserstein_distance(synthetic_values, real_values)),
                        "median_shift_nats": float(np.median(real_values) - np.median(synthetic_values)),
                        "q99_shift_nats": float(np.quantile(real_values, 0.99) - np.quantile(synthetic_values, 0.99)),
                    }
                )
                if metric == "area90_deg2":
                    assert isinstance(dep_plot["synthetic_area90"], list)
                    dep_plot["synthetic_area90"].append(synthetic_values)
                    if seed_index == 0:
                        dep_plot["real_area90"] = real_values

            if seed_index == 0:
                for nside in NSIDES:
                    dep_plot[f"real_nside{nside}"] = finite(real[f"sky_log_bf_nside{nside}"])

        plot_data[deployment] = {
            key: np.concatenate(value) if isinstance(value, list) else value
            for key, value in dep_plot.items()
        }

        dep_tail = pd.DataFrame(tail_rows)
        dep_shift = pd.DataFrame(shift_tests)
        q99 = dep_tail.loc[
            (dep_tail["deployment"] == deployment)
            & (dep_tail["reference_quantile"] == 0.99)
            & (dep_tail["target"] == "real_strict_pairs_nside64"),
            "tail_inflation_ratio",
        ]
        q999 = dep_tail.loc[
            (dep_tail["deployment"] == deployment)
            & (dep_tail["reference_quantile"] == 0.999)
            & (dep_tail["target"] == "real_strict_pairs_nside64"),
            "tail_inflation_ratio",
        ]
        same_resolution = dep_shift.loc[
            (dep_shift["deployment"] == deployment)
            & (dep_shift["comparison"] == "validation_injection_null_nside64_vs_real_nside64")
        ]
        headline[deployment] = {
            "n_seeds": len(seed_dirs),
            "strict_real_pairs_per_seed": int(
                pd.DataFrame(pair_stats)
                .loc[
                    (pd.DataFrame(pair_stats)["deployment"] == deployment)
                    & (pd.DataFrame(pair_stats)["source"] == "real_gwtc")
                    & (pd.DataFrame(pair_stats)["nside"] == 512),
                    "n",
                ]
                .iloc[0]
            ),
            "same_nside64_ks_mean": float(same_resolution["ks_statistic"].mean()),
            "same_nside64_ks_pvalue_max": float(same_resolution["ks_pvalue"].max()),
            "same_nside64_q99_shift_nats_mean": float(same_resolution["q99_shift_nats"].mean()),
            "real_tail_inflation_at_injection_q99_mean": float(q99.mean()),
            "real_tail_inflation_at_injection_q999_mean": float(q999.mean()),
            "domain_shift_flag": bool(
                (same_resolution["ks_pvalue"].max() < 0.01)
                and ((q99.mean() > 2.0) or (q999.mean() > 2.0))
            ),
        }

    return pair_stats, event_stats, shift_tests, tail_rows, resolution_rows, headline, used_paths, plot_data


def aggregate_seed_summary(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    numeric = [
        c
        for c in frame.columns
        if c not in group_columns + ["seed"] and pd.api.types.is_numeric_dtype(frame[c])
    ]
    rows: list[dict[str, object]] = []
    for keys, group in frame.groupby(group_columns, dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_columns, keys))
        row["n_seeds"] = int(group["seed"].nunique()) if "seed" in group else 1
        for col in numeric:
            values = finite(group[col])
            row[f"{col}_mean"] = float(np.mean(values)) if values.size else math.nan
            row[f"{col}_std"] = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def plot_results(plot_data: dict[str, dict[str, np.ndarray]], tail: pd.DataFrame, output: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 8.5,
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "axes.linewidth": 0.8,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    colors = {
        "synthetic_null": "#6B7280",
        "synthetic_true": "#0072B2",
        "real_nside64": "#E69F00",
        "real_nside512": "#D55E00",
        "real_nside1024": "#8B1A1A",
    }
    labels = {
        "synthetic_null": "Injection non-companion (Nside 64)",
        "synthetic_true": "Injection companion (Nside 64)",
        "real_nside64": "Real pairs (Nside 64)",
        "real_nside512": "Real pairs (Nside 512)",
        "real_nside1024": "Real pairs (Nside 1024)",
    }
    fig, axes = plt.subplots(2, 3, figsize=(10.8, 6.4), constrained_layout=True)
    panel_letters = iter("abcdef")
    for row, deployment in enumerate(DEPLOYMENTS):
        data = plot_data[deployment]
        ax = axes[row, 0]
        all_values = np.concatenate([finite(data[key]) for key in labels])
        lo, hi = np.quantile(all_values, [0.005, 0.995])
        for key in labels:
            values = np.sort(finite(data[key]))
            values = values[(values >= lo) & (values <= hi)]
            if not values.size:
                continue
            y = np.arange(1, values.size + 1) / values.size
            ax.plot(values, y, lw=1.2, color=colors[key], label=labels[key])
        ax.set_xlabel(r"Sky evidence $Z_{\rm sky}$ [nats]")
        ax.set_ylabel("Empirical CDF")
        ax.set_title(f"{next(panel_letters)}  {deployment.upper()}: central distribution", loc="left")
        ax.grid(alpha=0.18, linewidth=0.5)
        if row == 0:
            ax.legend(fontsize=6.8, loc="upper left")

        ax = axes[row, 1]
        for key in labels:
            values = np.sort(finite(data[key]))
            if not values.size:
                continue
            survival = (values.size - np.arange(values.size)) / values.size
            start = max(0, int(0.80 * values.size))
            ax.plot(values[start:], survival[start:], lw=1.2, color=colors[key], label=labels[key])
        ax.set_yscale("log")
        ax.set_xlabel(r"Sky evidence $Z_{\rm sky}$ [nats]")
        ax.set_ylabel("Survival probability")
        ax.set_title(f"{next(panel_letters)}  Positive-tail comparison", loc="left")
        ax.grid(alpha=0.18, linewidth=0.5)

        ax = axes[row, 2]
        for key, color, label in (
            ("synthetic_area90", "#0072B2", "Injection sky posterior"),
            ("real_area90", "#D55E00", "Public-PE sky posterior"),
        ):
            values = np.sort(finite(data[key]))
            values = values[values > 0]
            y = np.arange(1, values.size + 1) / values.size
            ax.plot(values, y, lw=1.4, color=color, label=label)
        ax.set_xscale("log")
        ax.set_xlabel(r"$A_{90}$ [deg$^2$]")
        ax.set_ylabel("Empirical CDF")
        ax.set_title(f"{next(panel_letters)}  Event-level localization", loc="left")
        ax.grid(alpha=0.18, linewidth=0.5)
        if row == 0:
            ax.legend(fontsize=7.0, loc="lower right")

    fig.suptitle(
        "Fast sky-domain-shift audit: injection posteriors versus real public-PE maps",
        fontsize=11,
        fontweight="bold",
    )
    figure_dir = output / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_dir / "fig_sky_domain_shift_fast_audit.pdf", bbox_inches="tight")
    fig.savefig(figure_dir / "fig_sky_domain_shift_fast_audit.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    selected = tail.loc[
        (tail["reference_quantile"].isin([0.99, 0.999]))
        & tail["target"].str.startswith("real_strict_pairs")
    ].copy()
    selected["nside"] = selected["target"].str.extract(r"nside(\d+)").astype(int)
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.2), constrained_layout=True)
    for ax, deployment, letter in zip(axes, DEPLOYMENTS, "ab"):
        subset = selected.loc[selected["deployment"] == deployment]
        for q, marker, color in ((0.99, "o", "#0072B2"), (0.999, "s", "#D55E00")):
            qset = subset.loc[subset["reference_quantile"] == q]
            for nside in NSIDES:
                values = qset.loc[qset["nside"] == nside, "tail_inflation_ratio"].to_numpy(float)
                x = np.full(values.size, nside, dtype=float)
                ax.scatter(x, values, color=color, marker=marker, s=24, alpha=0.75)
                if values.size:
                    ax.plot([nside - 10, nside + 10], [np.mean(values)] * 2, color=color, lw=2)
            ax.plot([], [], color=color, marker=marker, linestyle="none", label=f"Injection-null q{100*q:g}")
        ax.axhline(1.0, color="black", lw=0.8, ls="--")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(NSIDES, [str(x) for x in NSIDES])
        ax.set_xlabel("Real-map scoring Nside")
        ax.set_ylabel("Real-pair tail inflation ratio")
        ax.set_title(f"{letter}  {deployment.upper()}", loc="left")
        ax.grid(alpha=0.18, linewidth=0.5)
        ax.legend(fontsize=7, loc="best")
    fig.savefig(figure_dir / "fig_real_null_tail_inflation.pdf", bbox_inches="tight")
    fig.savefig(figure_dir / "fig_real_null_tail_inflation.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_report(
    output: Path,
    headline: dict[str, dict[str, object]],
    resolution: pd.DataFrame,
    tail: pd.DataFrame,
) -> None:
    lines = [
        "# 天空后验域差异快速审计报告",
        "",
        f"生成时间：{datetime.now(timezone.utc).isoformat()}",
        "",
        "## 1. 实验定位",
        "",
        "本实验是只读、独立的快速诊断，不修改或替代 v9.3，不重新训练 encoder，也不使用真实候选调节参数。目标是检查 O3/O4a 注入天空后验与真实 GWTC 公开 PE 天空后验是否属于不同统计分布，以及这种差异是否会使真实目录中的天空高分假对多于注入验证阶段的预期。",
        "",
        "需要区分两个问题：",
        "",
        "1. **后验域差异**：在相同 Nside=64 下比较注入 non-companion 与真实 pair。",
        "2. **数值分辨率效应**：只对真实公开 PE 图比较 Nside=64、512 和 1024。",
        "",
        "本实验中的真实 GWTC pair 没有透镜真值标签，因此 real-pair 分布只能作为经验目录背景诊断，不能把其中每一对都断言为非透镜。鉴于真实强透镜率极低，这仍是合理的背景近似，但结论按探索性结果表述。",
        "",
        "## 2. 核心结果",
        "",
        "| catalog | strict real pairs | same-Nside KS | q99 shift [nats] | real-tail inflation at injection q99 | at q99.9 | domain-shift flag |",
        "|---|---:|---:|---:|---:|---:|:---:|",
    ]
    for deployment in DEPLOYMENTS:
        item = headline[deployment]
        lines.append(
            f"| {deployment.upper()} | {item['strict_real_pairs_per_seed']} | {item['same_nside64_ks_mean']:.4f} | "
            f"{item['same_nside64_q99_shift_nats_mean']:.4f} | {item['real_tail_inflation_at_injection_q99_mean']:.3f} | "
            f"{item['real_tail_inflation_at_injection_q999_mean']:.3f} | {'YES' if item['domain_shift_flag'] else 'NO'} |"
        )
    lines.extend(
        [
            "",
            "判定规则在运行前写入脚本：相同 Nside=64 的两样本 KS 检验在全部 seed 上达到 p<0.01，且 q99 或 q99.9 尾部膨胀均值大于 2，才标记为明显 domain shift。该标记是工程与统计诊断，不是候选显著性。",
            "",
            "## 3. 分辨率效应",
            "",
            "下表给出真实 strict-scope pair 的跨分辨率变化；这些数字回答粗像素本身造成多大影响。",
            "",
            "| catalog | comparison | median |Z difference| | q99 | max | sign flips | Spearman | Top-20 overlap |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    grouped = resolution.groupby(["deployment", "comparison"], sort=True).mean(numeric_only=True).reset_index()
    for _, row in grouped.iterrows():
        lines.append(
            f"| {str(row['deployment']).upper()} | {row['comparison']} | {row['median_abs_delta_nats']:.5g} | "
            f"{row['q99_abs_delta_nats']:.5g} | {row['max_abs_delta_nats']:.5g} | {row['sign_flip_count']:.2f} | "
            f"{row['spearman']:.6f} | {row['top20_overlap']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## 4. 科学解释",
            "",
            "- 如果相同 Nside=64 下已经出现显著尾部差异，问题不能只归因于 64 与 512/1024 的像素分辨率，而说明注入后验形态、面积、网络/SNR 条件或模板构造与真实公开 PE 后验不匹配。",
            "- 如果真实 Nside=64 到 512 的排名或符号大幅变化，则低分辨率还叠加了独立的数值偏差。",
            "- 注入 validation 上选择出的 sky 权重只在注入 null 尾部与真实目录背景具有可比性时才可直接部署；否则需要 map-matched synthetic validation，不能用真实候选反向调权。",
            "- A90、KL 和 effective area 的差异用于定位域差异来自定位宽度还是后验形态。entropy 跨 Nside 会受像素数影响，仅作辅助列。",
            "",
            "## 5. 下一步",
            "",
            "下一步应使用 O3/O4a 各自的真实公开 PE map 库，在 Nside=512 下构造 run/network/SNR/A90 分层匹配的 synthetic sky posteriors；validation 只用 synthetic systems 重新选择权重，随后冻结并在 held-out synthetic test 与真实目录各运行一次。快速相干天空 PE pilot 只用于验证这种模板匹配近似，不是本报告的前提。",
            "",
            "## 6. 输出说明",
            "",
            "- `pair_score_distribution_per_seed.csv`：全部 pair 分布分位数。",
            "- `null_tail_exceedance_per_seed.csv`：注入阈值在 held-out 注入和真实目录中的超阈值率。",
            "- `distribution_shift_tests.csv`：KS/Wasserstein 和分位数偏移。",
            "- `real_resolution_effects_per_seed.csv`：真实 64/512/1024 分辨率变化。",
            "- `event_sky_distribution_per_seed.csv`：A90、KL、effective area 和 entropy。",
            "- `figures/`：分布及尾部膨胀图。",
            "",
            "最终状态：`HOLD_FOR_AUTHOR_REVIEW_NO_ADOPTION_NO_OVERWRITE`。",
        ]
    )
    (output / "sky_domain_shift_fast_report_cn.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("/root/autodl-tmp/gw-catalog"),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--no-package", action="store_true")
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    input_root = project_root / "results" / "gwtc_sky_resolution_v93_20260730"
    output = args.output or (
        project_root / "results" / f"sky_domain_shift_fast_v101_20260823_{utc_stamp()}"
    )
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.mkdir(parents=True)

    contract = {
        "experiment": "sky_domain_shift_fast_v101",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "EXPLORATORY_READ_ONLY",
        "input_v93_root": str(input_root),
        "historical_results_modified": False,
        "encoder_retrained": False,
        "real_candidates_used_for_calibration": False,
        "synthetic_map_nside": 64,
        "real_map_nsides_compared": list(NSIDES),
        "primary_question": "Do synthetic injection sky nulls match the empirical real-catalog sky-pair background?",
        "domain_shift_flag_rule": "same-Nside64 KS p<0.01 for all seeds and mean q99 or q99.9 tail inflation >2",
        "real_pair_uncertainty": {
            "method": "event-vertex block bootstrap",
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
        },
        "terminal_state": "HOLD_FOR_AUTHOR_REVIEW_NO_ADOPTION_NO_OVERWRITE",
    }
    (output / "analysis_contract.json").write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")

    (
        pair_stats,
        event_stats,
        shift_tests,
        tail_rows,
        resolution_rows,
        headline,
        used_paths,
        plot_data,
    ) = load_inputs(input_root)

    pair_frame = pd.DataFrame(pair_stats)
    event_frame = pd.DataFrame(event_stats)
    shift_frame = pd.DataFrame(shift_tests)
    tail_frame = pd.DataFrame(tail_rows)
    resolution_frame = pd.DataFrame(resolution_rows)

    write_csv(pair_frame, output / "pair_score_distribution_per_seed.csv")
    write_csv(
        aggregate_seed_summary(pair_frame, ["deployment", "source", "split", "pair_class", "nside"]),
        output / "pair_score_distribution_summary.csv",
    )
    write_csv(event_frame, output / "event_sky_distribution_per_seed.csv")
    write_csv(
        aggregate_seed_summary(event_frame, ["deployment", "source", "metric", "native_analysis_nside"]),
        output / "event_sky_distribution_summary.csv",
    )
    write_csv(shift_frame, output / "distribution_shift_tests.csv")
    write_csv(tail_frame, output / "null_tail_exceedance_per_seed.csv")
    write_csv(
        aggregate_seed_summary(tail_frame, ["deployment", "reference", "target", "reference_quantile"]),
        output / "null_tail_exceedance_summary.csv",
    )
    write_csv(resolution_frame, output / "real_resolution_effects_per_seed.csv")
    write_csv(
        aggregate_seed_summary(resolution_frame, ["deployment", "comparison"]),
        output / "real_resolution_effects_summary.csv",
    )
    pair_frame.to_parquet(output / "pair_score_distribution_per_seed.parquet", index=False)
    tail_frame.to_parquet(output / "null_tail_exceedance_per_seed.parquet", index=False)

    summary = {
        "experiment": "sky_domain_shift_fast_v101",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "headline": headline,
        "interpretation_boundary": {
            "real_pair_labels_known": False,
            "real_catalog_used_as_empirical_null": True,
            "is_new_candidate_ranking": False,
            "is_full_parameter_estimation": False,
            "is_detection_claim": False,
        },
        "next_stage": "Nside512 map-matched synthetic validation and held-out test",
        "terminal_state": "HOLD_FOR_AUTHOR_REVIEW_NO_ADOPTION_NO_OVERWRITE",
    }
    (output / "sky_domain_shift_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    plot_results(plot_data, tail_frame, output)
    make_report(output, headline, resolution_frame, tail_frame)

    scripts_dir = output / "scripts"
    scripts_dir.mkdir()
    shutil.copy2(Path(__file__).resolve(), scripts_dir / Path(__file__).name)
    input_manifest_rows = []
    for path in sorted(set(used_paths)):
        input_manifest_rows.append(
            {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "mtime_utc": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
                "sha256": sha256_file(path),
            }
        )
    write_csv(pd.DataFrame(input_manifest_rows), output / "input_manifest.csv")

    manifest_dir = output / "manifest"
    manifest_dir.mkdir()
    output_files = [p for p in output.rglob("*") if p.is_file() and "manifest/files_sha256.csv" not in str(p)]
    manifest_rows = [
        {
            "relative_path": str(path.relative_to(output)),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(output_files)
    ]
    write_csv(pd.DataFrame(manifest_rows), manifest_dir / "files_sha256.csv")

    latest = project_root / "results" / "sky_domain_shift_fast_v101_LATEST.txt"
    latest.write_text(str(output) + "\n", encoding="utf-8")

    package_path = None
    package_sha = None
    if not args.no_package:
        packages = project_root / "packages"
        packages.mkdir(exist_ok=True)
        package_path = packages / f"{output.name}.tar.gz"
        with tarfile.open(package_path, "w:gz") as archive:
            archive.add(output, arcname=output.name)
        package_sha = sha256_file(package_path)
        (package_path.with_suffix(package_path.suffix + ".sha256")).write_text(
            f"{package_sha}  {package_path.name}\n", encoding="ascii"
        )

    print(json.dumps({"output": str(output), "package": str(package_path) if package_path else None, "package_sha256": package_sha, "headline": headline}, indent=2))


if __name__ == "__main__":
    main()
