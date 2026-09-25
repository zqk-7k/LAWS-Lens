from __future__ import annotations

import argparse
import gc
import importlib
import importlib.util
import json
import math
import os
import sys
import tarfile
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from matchgw.aux_priors.observed_sky import DETECTOR_SKY_SCENARIOS, build_observed_sky_table  # noqa: E402
from scripts.sky.sky_posterior_overlap import angular_separation_rad  # noqa: E402


ET3_MATCH_ROOT = Path("/root/autodl-tmp/createdata/et3_10000_20260616_1006_match_root")
ET3_ENCODER_DIR = REPO_ROOT / "runs" / "et3_fresh50_full_catalog_20260616" / "fresh_mixed_encoders" / "et3_noisy_mixed_sis_pm_ep50"
DEFAULT_OUT = REPO_ROOT / "results" / "et3_sky_localization_sensitivity_20260712_mainline_fixed"
DEFAULT_PACKAGE = REPO_ROOT / "packages" / "et3_sky_localization_sensitivity_20260712_mainline_fixed.tar.gz"
DEFAULT_SKY_SEEDS = (2026071201, 2026071202, 2026071203)
LAMBDA_GRID = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0)
ROW_CHUNK = 64
MAINLINE_WEIGHTS = {"waveform": 0.25, "time": 0.25, "sky": 4.0}
MAINLINE_LAMBDA = MAINLINE_WEIGHTS["sky"] / MAINLINE_WEIGHTS["waveform"]
MAINLINE_METHOD = "waveform_time_sky_mainline_fixed"
MAINLINE_WEIGHT_SOURCE = "runs/unified_sky_posterior_overlap_20260708/et3/et3_unified_sky_summary.json"
TIMES_NEW_ROMAN_PATH = Path("/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf")

if TIMES_NEW_ROMAN_PATH.exists():
    font_manager.fontManager.addfont(TIMES_NEW_ROMAN_PATH)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        value = float(obj)
        return value if math.isfinite(value) else None
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=json_default) + "\n", encoding="utf-8")


def evict_file_cache(path: Path) -> None:
    """Release mmap-backed pages after a one-time read in the 2-GiB CPU container."""
    if not path.exists() or not hasattr(os, "posix_fadvise"):
        return
    try:
        with path.open("rb") as handle:
            os.posix_fadvise(handle.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    except OSError:
        pass


def scenario_grid() -> list[dict[str, Any]]:
    baseline = {"a90_ref_deg2": 100.0, "scatter": 0.35, "clip_min_deg2": 20.0, "clip_max_deg2": 1000.0}
    rows: list[dict[str, Any]] = [{"scenario_id": "baseline", "scan_axis": "baseline", **baseline}]
    for value in (25.0, 50.0, 200.0, 400.0):
        rows.append({"scenario_id": f"a90ref_{value:g}", "scan_axis": "a90_ref", **baseline, "a90_ref_deg2": value})
    for value in (0.0, 0.15, 0.60, 0.90):
        rows.append({"scenario_id": f"scatter_{value:g}", "scan_axis": "scatter", **baseline, "scatter": value})
    for value in (1.0, 5.0, 50.0, 100.0):
        rows.append({"scenario_id": f"clipmin_{value:g}", "scan_axis": "clip_min", **baseline, "clip_min_deg2": value})
    for value in (200.0, 500.0, 2000.0, 5000.0):
        rows.append({"scenario_id": f"clipmax_{value:g}", "scan_axis": "clip_max", **baseline, "clip_max_deg2": value})
    rows.extend(
        [
            {"scenario_id": "joint_optimistic", "scan_axis": "joint", "a90_ref_deg2": 25.0, "scatter": 0.15, "clip_min_deg2": 1.0, "clip_max_deg2": 200.0},
            {"scenario_id": "joint_narrow", "scan_axis": "joint", "a90_ref_deg2": 50.0, "scatter": 0.20, "clip_min_deg2": 5.0, "clip_max_deg2": 500.0},
            {"scenario_id": "joint_pessimistic", "scan_axis": "joint", "a90_ref_deg2": 400.0, "scatter": 0.80, "clip_min_deg2": 100.0, "clip_max_deg2": 5000.0},
            {"scenario_id": "joint_weak_clip", "scan_axis": "joint", "a90_ref_deg2": 100.0, "scatter": 0.35, "clip_min_deg2": 0.1, "clip_max_deg2": 5000.0},
        ]
    )
    return rows


def scenario_object(params: dict[str, Any]):
    base = DETECTOR_SKY_SCENARIOS["ET_TRIANGLE"]
    return replace(
        base,
        label=f"ET triangle sensitivity {params['scenario_id']}",
        a90_ref_deg2=float(params["a90_ref_deg2"]),
        clip_min_deg2=float(params["clip_min_deg2"]),
        clip_max_deg2=float(params["clip_max_deg2"]),
        lognormal_sigma=float(params["scatter"]),
    )


def build_sky(raw: pd.DataFrame, timing: pd.DataFrame, params: dict[str, Any], seed: int) -> pd.DataFrame:
    return build_observed_sky_table(
        raw,
        timing,
        scenario_object(params),
        np.random.default_rng(seed),
        sampling="tangent_2d_gaussian",
    )


def load_et3_without_model():
    base = importlib.import_module("scripts.experiments.80_mixed_sis_pm_catalog_modality_compare")
    base.ROOTS[("SIS", "ET3")] = ET3_MATCH_ROOT
    base.ROOTS[("PM", "ET3")] = ET3_MATCH_ROOT
    cfg = base.make_cfg("ET3", "noisy", ET3_ENCODER_DIR)

    def npy_length(path: Path, limit: int) -> int:
        array = np.load(path, mmap_mode="r")
        length = min(len(array), limit)
        del array
        return length

    splits = {}
    for index, family in enumerate(("SIS", "PM")):
        source = ET3_MATCH_ROOT / f"{family}_data_0222"
        n_lensed = npy_length(source / f"{family}_data_strain_1.npy", cfg.lensed_limit)
        n_unlensed = npy_length(
            ET3_MATCH_ROOT / "Unlensed_data_0222" / "unlensed_data_strain.npy",
            cfg.unlensed_limit,
        )
        splits[family] = base.split_indices(n_lensed, cfg.seed + index)
        splits[f"{family}_U"] = base.split_indices(n_unlensed, cfg.seed + 100 + index)

    def metadata(split: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        pair_id = 0
        for family in ("SIS", "PM"):
            lensed_idx = splits[family][split]
            start_pair = pair_id
            for original in lensed_idx:
                rows.append({"family": family, "tag": "L1", "pair_id": pair_id, "source_index": int(original)})
                pair_id += 1
            for local, original in enumerate(lensed_idx):
                rows.append({"family": family, "tag": "L2", "pair_id": start_pair + local, "source_index": int(original)})
            for original in splits[f"{family}_U"][split]:
                rows.append({"family": family, "tag": "U", "pair_id": -1, "source_index": int(original)})
        return rows

    def pack(split: str):
        meta = metadata(split)
        raw = base.mixed_obs_frame("ET3", split, splits, "raw")
        timing = base.mixed_obs_frame("ET3", split, splits, "time")
        return meta, raw, timing, base.ground_truth(meta)

    return base, pack("val"), pack("test")


def row_z(raw: np.ndarray, query_indices: np.ndarray) -> np.ndarray:
    arr = np.asarray(raw, dtype=np.float32).copy()
    arr[np.arange(len(query_indices)), query_indices] = np.nan
    mean = np.nanmean(arr, axis=1, keepdims=True)
    std = np.nanstd(arr, axis=1, keepdims=True)
    out = (arr - mean) / np.maximum(std, 1e-8)
    out[~np.isfinite(out)] = 0.0
    out[np.arange(len(query_indices)), query_indices] = 0.0
    return out.astype(np.float32, copy=False)


def build_base_rows(
    path: Path,
    waveform_scores_path: Path,
    timing: pd.DataFrame,
    gt: np.ndarray,
    prior: dict[str, Any],
) -> tuple[np.ndarray, Path]:
    queries = np.where(gt >= 0)[0].astype(np.int32)
    n = len(gt)
    if path.exists():
        return queries, path
    waveform = np.load(waveform_scores_path, mmap_mode="r")
    out = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(len(queries), n))
    trigger = timing["trigger_time_obs"].to_numpy(dtype=np.float64)
    edges = np.asarray(prior["edges"], dtype=np.float64)
    lr = np.asarray(prior["lr"], dtype=np.float32)
    for start in range(0, len(queries), ROW_CHUNK):
        stop = min(start + ROW_CHUNK, len(queries))
        q = queries[start:stop]
        wf_z = row_z(waveform[q], q)
        dt_days = np.abs(trigger[q, None] - trigger[None, :]) / 86400.0
        bins = np.searchsorted(edges, np.log10(np.maximum(dt_days, 1e-6)), side="right") - 1
        bins = np.clip(bins, 0, len(lr) - 1)
        time_z = row_z(lr[bins], q)
        out[start:stop] = wf_z + time_z
    out.flush()
    del out, waveform
    gc.collect()
    return queries, path


def write_sky_row_files(
    sky: pd.DataFrame,
    queries: np.ndarray,
    cosine_path: Path,
    bayes_path: Path,
) -> None:
    n = len(sky)
    cosine = np.lib.format.open_memmap(cosine_path, mode="w+", dtype=np.float32, shape=(len(queries), n))
    bayes = np.lib.format.open_memmap(bayes_path, mode="w+", dtype=np.float32, shape=(len(queries), n))
    ra = sky["ra_obs"].to_numpy(dtype=np.float64)
    dec = sky["dec_obs"].to_numpy(dtype=np.float64)
    sigma = np.maximum(sky["sky_sigma_rad"].to_numpy(dtype=np.float64), 1e-12)
    for start in range(0, len(queries), ROW_CHUNK):
        stop = min(start + ROW_CHUNK, len(queries))
        q = queries[start:stop]
        theta = angular_separation_rad(ra[q, None], dec[q, None], ra[None, :], dec[None, :])
        si = sigma[q, None]
        sj = sigma[None, :]
        var = si * si + sj * sj
        sep = -(theta * theta) / (2.0 * var)
        cosine[start:stop] = row_z(np.log(2.0 * si * sj / var) + sep, q)
        bayes[start:stop] = row_z(math.log(2.0) - np.log(var) + sep, q)
    cosine.flush()
    bayes.flush()
    del cosine, bayes
    gc.collect()


def ranks_for_lambda(
    base_path: Path | None,
    sky_path: Path,
    queries: np.ndarray,
    gt: np.ndarray,
    lambda_sky: float,
) -> np.ndarray:
    sky = np.load(sky_path, mmap_mode="r")
    base = np.load(base_path, mmap_mode="r") if base_path is not None else None
    ranks = np.empty(len(queries), dtype=np.int32)
    for start in range(0, len(queries), ROW_CHUNK):
        stop = min(start + ROW_CHUNK, len(queries))
        q = queries[start:stop]
        score = np.asarray(sky[start:stop], dtype=np.float32).copy() * float(lambda_sky)
        if base is not None:
            score += np.asarray(base[start:stop], dtype=np.float32)
        score[np.arange(len(q)), q] = -np.inf
        partners = gt[q].astype(np.int32)
        true_score = score[np.arange(len(q)), partners]
        ranks[start:stop] = 1 + np.sum(score > true_score[:, None], axis=1)
    return ranks


def metrics_from_ranks(fresh, ranks: np.ndarray, queries: np.ndarray, gt: np.ndarray, meta: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    family = np.asarray([str(meta[int(q)]["family"]).upper() for q in queries])
    return fresh.metrics_from_ranks(ranks.astype(np.int32), family, len(gt))


def select_lambda(fresh, base_path: Path, sky_path: Path, queries: np.ndarray, gt: np.ndarray, meta: list[dict[str, Any]]) -> tuple[float, pd.DataFrame]:
    rows = []
    best_lambda = 0.0
    best_key = (-1.0, -1.0, -1.0)
    for value in LAMBDA_GRID:
        ranks = ranks_for_lambda(base_path, sky_path, queries, gt, value)
        metrics = metrics_from_ranks(fresh, ranks, queries, gt, meta)
        key = (metrics["overall"]["r@10"], metrics["overall"]["r@5"], metrics["overall"]["r@1"])
        rows.append(
            {
                "lambda_sky": value,
                "overall_r_at_1": metrics["overall"]["r@1"],
                "overall_r_at_5": metrics["overall"]["r@5"],
                "overall_r_at_10": metrics["overall"]["r@10"],
                "sis_r_at_10": metrics["SIS"]["r@10"],
                "pm_r_at_10": metrics["PM"]["r@10"],
            }
        )
        if key > best_key:
            best_key = key
            best_lambda = float(value)
    return best_lambda, pd.DataFrame(rows)


def flatten_metrics(
    metrics: dict[str, dict[str, Any]],
    params: dict[str, Any],
    sky_seed: int,
    score_type: str,
    method: str,
    lambda_sky: float,
    selection: str,
) -> list[dict[str, Any]]:
    rows = []
    for subset, values in metrics.items():
        rows.append(
            {
                **params,
                "sky_seed": int(sky_seed),
                "score_type": score_type,
                "method": method,
                "lambda_sky": float(lambda_sky),
                "selection": selection,
                "subset": subset,
                "r_at_1": values["r@1"],
                "r_at_5": values["r@5"],
                "r_at_10": values["r@10"],
                "r_at_50": values["r@50"],
                "median_true_rank": values["median_true_rank"],
                "n_queries": values["valid"],
            }
        )
    return rows


def sample_false_pairs(gt: np.ndarray, count: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    left: list[np.ndarray] = []
    right: list[np.ndarray] = []
    remaining = count
    while remaining:
        size = max(10_000, remaining * 2)
        a = rng.integers(0, len(gt), size=size, dtype=np.int32)
        b = rng.integers(0, len(gt), size=size, dtype=np.int32)
        i = np.minimum(a, b)
        j = np.maximum(a, b)
        keep = (i != j) & (gt[i] != j)
        take = min(remaining, int(keep.sum()))
        left.append(i[keep][:take])
        right.append(j[keep][:take])
        remaining -= take
    return np.concatenate(left), np.concatenate(right)


def direct_pair_score(sky: pd.DataFrame, i: np.ndarray, j: np.ndarray, score_type: str) -> np.ndarray:
    ra = sky["ra_obs"].to_numpy(dtype=np.float64)
    dec = sky["dec_obs"].to_numpy(dtype=np.float64)
    sigma = np.maximum(sky["sky_sigma_rad"].to_numpy(dtype=np.float64), 1e-12)
    theta = angular_separation_rad(ra[i], dec[i], ra[j], dec[j])
    si = sigma[i]
    sj = sigma[j]
    var = si * si + sj * sj
    sep = -(theta * theta) / (2.0 * var)
    if score_type == "log_cosine":
        return np.log(2.0 * si * sj / var) + sep
    return math.log(2.0) - np.log(var) + sep


def diagnostic_row(
    sky: pd.DataFrame,
    gt: np.ndarray,
    false_i: np.ndarray,
    false_j: np.ndarray,
    score_type: str,
) -> dict[str, Any]:
    true_i = np.asarray([i for i, partner in enumerate(gt) if partner > i], dtype=np.int32)
    true_j = gt[true_i].astype(np.int32)
    true_score = direct_pair_score(sky, true_i, true_j, score_type)
    false_score = direct_pair_score(sky, false_i, false_j, score_type)
    area = sky["sky_area90_deg2"].to_numpy(dtype=np.float64)
    false_area = np.sqrt(area[false_i] * area[false_j])
    corr = spearmanr(np.log10(false_area), false_score, nan_policy="omit").statistic
    q05 = float(np.quantile(true_score, 0.05))
    return {
        "n_true_pairs": int(len(true_score)),
        "n_false_pairs_sampled": int(len(false_score)),
        "true_score_median": float(np.median(true_score)),
        "true_score_q05": q05,
        "false_score_median": float(np.median(false_score)),
        "false_score_q99": float(np.quantile(false_score, 0.99)),
        "false_score_q999": float(np.quantile(false_score, 0.999)),
        "false_fraction_above_true_q05": float(np.mean(false_score >= q05)),
        "false_score_vs_log10_geometric_area_spearman": float(corr) if np.isfinite(corr) else None,
    }


def summarize(per_seed: pd.DataFrame) -> pd.DataFrame:
    group = ["scenario_id", "scan_axis", "a90_ref_deg2", "scatter", "clip_min_deg2", "clip_max_deg2", "score_type", "method", "selection", "subset"]
    rows = []
    for key, frame in per_seed.groupby(group, dropna=False, sort=False):
        fixed = dict(zip(group, key))
        for metric in ("r_at_1", "r_at_5", "r_at_10", "r_at_50", "median_true_rank", "lambda_sky"):
            values = frame[metric].to_numpy(dtype=np.float64)
            rows.append(
                {
                    **fixed,
                    "metric": metric,
                    "n_sky_seeds": int(len(values)),
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                    "median": float(np.median(values)),
                    "q25": float(np.quantile(values, 0.25)),
                    "q75": float(np.quantile(values, 0.75)),
                    "min": float(values.min()),
                    "max": float(values.max()),
                }
            )
    return pd.DataFrame(rows)


def axis_data(metrics: pd.DataFrame, axis: str) -> pd.DataFrame:
    baseline = metrics[metrics["scenario_id"] == "baseline"].copy()
    baseline["scan_axis"] = axis
    return pd.concat([baseline, metrics[metrics["scan_axis"] == axis]], ignore_index=True).drop_duplicates(
        ["scenario_id", "sky_seed", "score_type", "method", "subset"]
    )


def make_figures(out_root: Path, metrics: pd.DataFrame, diagnostics: pd.DataFrame) -> None:
    fig_dir = out_root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 10,
            "mathtext.fontset": "custom",
            "mathtext.rm": "Times New Roman",
            "mathtext.it": "Times New Roman:italic",
            "mathtext.bf": "Times New Roman:bold",
            "mathtext.cal": "Times New Roman",
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    specs = [
        ("a90_ref", "a90_ref_deg2", r"$A_{90,\mathrm{ref}}$ (deg$^2$)"),
        ("scatter", "scatter", "Lognormal scatter"),
        ("clip_min", "clip_min_deg2", r"Minimum $A_{90}$ (deg$^2$)"),
        ("clip_max", "clip_max_deg2", r"Maximum $A_{90}$ (deg$^2$)"),
    ]
    colors = {"log_cosine": "#4C78A8", "log_sky_bayes_factor": "#E45756"}
    labels = {"log_cosine": "Cosine overlap", "log_sky_bayes_factor": "Sky Bayes factor"}
    styles = {
        "log_cosine": {"linestyle": "-", "marker": "o", "markerfacecolor": "#4C78A8"},
        "log_sky_bayes_factor": {"linestyle": "--", "marker": "s", "markerfacecolor": "white"},
    }
    for metric, suffix in (("r_at_10", "r10"), ("r_at_1", "r1")):
        plotted = metrics[(metrics["method"] == MAINLINE_METHOD) & (metrics["subset"] == "overall")][metric]
        span = float(plotted.max() - plotted.min())
        pad = max(0.001, 0.30 * span)
        y_limits = (max(0.0, float(plotted.min()) - pad), min(1.0005, float(plotted.max()) + pad))
        fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.6))
        for ax, (axis, xcol, xlabel) in zip(axes.flat, specs):
            frame = axis_data(metrics, axis)
            frame = frame[(frame["method"] == MAINLINE_METHOD) & (frame["subset"] == "overall")]
            for score_type in colors:
                part = frame[frame["score_type"] == score_type]
                stats = part.groupby(xcol)[metric].agg(["mean", "std"]).reset_index().sort_values(xcol)
                if stats.empty:
                    continue
                ax.plot(
                    stats[xcol], stats["mean"], color=colors[score_type], label=labels[score_type],
                    linewidth=1.7, markersize=5.5, markeredgewidth=1.2, **styles[score_type],
                )
                ax.fill_between(
                    stats[xcol].to_numpy(float),
                    (stats["mean"] - stats["std"]).to_numpy(float),
                    (stats["mean"] + stats["std"]).to_numpy(float),
                    color=colors[score_type], alpha=0.14,
                )
                ax.scatter(
                    part[xcol], part[metric], edgecolor=colors[score_type],
                    facecolor=styles[score_type]["markerfacecolor"], marker=styles[score_type]["marker"],
                    s=20, alpha=0.55, linewidth=0.8,
                )
            ax.set_xlabel(xlabel)
            ax.set_ylabel(metric.replace("_", "@").replace("r@at@", "R@"))
            ax.set_ylim(*y_limits)
            ax.grid(alpha=0.2)
            if axis != "scatter":
                ax.set_xscale("log")
        axes[0, 0].legend(frameon=False)
        fig.suptitle("ET-3 sky-localization sensitivity", fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        for ext in ("pdf", "png"):
            fig.savefig(fig_dir / f"fig_et3_sky_sensitivity_{suffix}.{ext}", dpi=300, bbox_inches="tight")
        plt.close(fig)

    diag = diagnostics[diagnostics["split"] == "test"]
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0))
    for score_type in colors:
        part = diag[diag["score_type"] == score_type]
        for ax, ycol in (
            (axes[0], "false_fraction_above_true_q05"),
            (axes[1], "false_score_vs_log10_geometric_area_spearman"),
        ):
            ax.scatter(
                part["a90_median_deg2"], part[ycol], s=24, alpha=0.65,
                edgecolor=colors[score_type], facecolor=styles[score_type]["markerfacecolor"],
                marker=styles[score_type]["marker"], linewidth=0.9, label=labels[score_type],
            )
    axes[0].set_xlabel(r"Median $A_{90}$ (deg$^2$)")
    axes[0].set_ylabel("False pairs above true-pair 5th percentile")
    axes[1].set_xlabel(r"Median $A_{90}$ (deg$^2$)")
    axes[1].set_ylabel("False score–area Spearman correlation")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[1].set_xscale("log")
    axes[0].legend(frameon=False)
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(fig_dir / f"fig_et3_sky_score_area_audit.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_report(out_root: Path, summary: pd.DataFrame, diagnostics: pd.DataFrame) -> None:
    baseline = summary[
        (summary["scenario_id"] == "baseline")
        & (summary["method"] == MAINLINE_METHOD)
        & (summary["subset"] == "overall")
        & summary["metric"].isin(["r_at_1", "r_at_10"])
    ][["score_type", "metric", "mean", "std", "q25", "q75"]]
    overall = summary[
        (summary["method"] == MAINLINE_METHOD)
        & (summary["subset"] == "overall")
        & summary["metric"].isin(["r_at_1", "r_at_10"])
    ]
    ranges = (
        overall.groupby(["score_type", "metric"])["mean"]
        .agg(["min", "max"])
        .reset_index()
    )
    diag_mean = (
        diagnostics.groupby(["scenario_id", "score_type"], as_index=False)
        .agg(
            a90_median_deg2=("a90_median_deg2", "mean"),
            fraction_at_min_clip=("fraction_at_min_clip", "mean"),
            false_tail=("false_fraction_above_true_q05", "mean"),
        )
    )
    tail = diag_mean.pivot(index="scenario_id", columns="score_type", values="false_tail")
    tail = tail.loc[
        [name for name in ("baseline", "a90ref_400", "clipmin_1", "joint_pessimistic", "joint_weak_clip") if name in tail.index]
    ].reset_index()
    tail["relative_reduction_bayes"] = 1.0 - tail["log_sky_bayes_factor"] / tail["log_cosine"]
    base_diag = diag_mean[(diag_mean["scenario_id"] == "baseline") & (diag_mean["score_type"] == "log_cosine")].iloc[0]
    text = "\n".join(
        [
            "# ET-3 天空定位敏感性扫描报告",
            "",
            "该实验复用现有 ET-3 waveform/time score、split 和 encoder，不重新生成 strain、不重新训练网络。",
            "",
            "基准参数为 A90_ref=100 deg2、lognormal scatter=0.35、clip=[20,1000] deg2。扫描采用三个 common-random-number sky realizations。主线融合权重固定为 waveform=0.25、time=0.25、sky=4.0，来源于 2026-07-08 unified-sky validation 结果；等比例写法为 waveform + time + 16*sky。所有 R@K 只在 test catalog 计算。",
            "",
            "面积敏感分数定义为 B_sky=4π∫p_i p_j dΩ；局部二维高斯近似下 log B=log2-log(σ_i²+σ_j²)-θ²/[2(σ_i²+σ_j²)]。它与 log-cosine 使用相同中心一致性项，但保留绝对定位面积惩罚。",
            "",
            "## 基准结果",
            "",
            baseline.to_markdown(index=False),
            "",
            "## 全扫描范围",
            "",
            ranges.to_markdown(index=False),
            "",
            f"基准 test catalog 的 A90 中位数为 {base_diag['a90_median_deg2']:.3f} deg2；{base_diag['fraction_at_min_clip']:.1%} 的事件落在 20 deg2 下限，说明下限设置比上限更能影响当前高 SNR ET-3 catalog。",
            "",
            "## False-pair tail",
            "",
            "下表给出固定随机 false-pair 样本中，分数高于 true-pair 第 5 百分位数的比例。relative_reduction_bayes>0 表示面积敏感 Bayes factor 减少了高分 false pairs。",
            "",
            tail.to_markdown(index=False),
            "",
            "## 解释边界",
            "",
            "该结果说明现有 ET-3 companion retrieval 对所扫描的 observed-sky proxy 参数总体稳定，但它不是完整 ET parameter-estimation sky-map 实验。面积敏感 Bayes factor 对 R@K 的改变很小，主要收益体现在部分场景的 false-pair tail；不能将该 tail 样本解释为 detection FAR。",
            "",
            "false-pair tail 使用固定背景 pair 样本做机制诊断，不是 detection FAR。本实验是 observed-sky proxy sensitivity study，不是真实 ET parameter-estimation sky-map 分析。",
        ]
    )
    (out_root / "et3_sky_localization_sensitivity_report_cn.md").write_text(text + "\n", encoding="utf-8")


def aggregate(out_root: Path) -> None:
    files = sorted((out_root / "per_scenario").glob("*/metrics.csv"))
    if not files:
        return
    metrics = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    diagnostics = pd.concat([pd.read_csv(path.parent / "diagnostics.csv") for path in files], ignore_index=True)
    metrics.to_csv(out_root / "et3_sky_sensitivity_per_seed.csv", index=False)
    diagnostics.to_csv(out_root / "et3_sky_score_diagnostics_per_seed.csv", index=False)
    summary = summarize(metrics)
    summary.to_csv(out_root / "et3_sky_sensitivity_summary.csv", index=False)
    make_figures(out_root, metrics, diagnostics)
    write_report(out_root, summary, diagnostics)
    config = json.loads((out_root / "run_config.json").read_text())
    write_json(
        out_root / "et3_sky_sensitivity_summary.json",
        {
            "status": "complete" if len(files) == int(config["planned_runs"]) else "partial",
            "completed_runs": len(files),
            "planned_runs": int(config["planned_runs"]),
            "baseline": asdict(DETECTOR_SKY_SCENARIOS["ET_TRIANGLE"]),
            "sky_seeds": config["sky_seeds"],
            "n_scenarios": len(config["scenarios"]),
            "false_pairs_sampled_per_run": int(config["false_samples"]),
            "mainline_weights": MAINLINE_WEIGHTS,
            "mainline_weight_source": MAINLINE_WEIGHT_SOURCE,
            "score_definitions": {
                "log_cosine": "log(2*sigma_i*sigma_j/(sigma_i^2+sigma_j^2)) - theta^2/(2*(sigma_i^2+sigma_j^2))",
                "log_sky_bayes_factor": "log(2) - log(sigma_i^2+sigma_j^2) - theta^2/(2*(sigma_i^2+sigma_j^2))",
            },
            "memory_strategy": "query-row streaming; no 9000x9000 sky matrix retained",
        },
    )


def package_outputs(out_root: Path, package: Path) -> None:
    package.parent.mkdir(parents=True, exist_ok=True)
    scripts = [
        REPO_ROOT / "scripts" / "experiments" / "105_et3_sky_localization_sensitivity.py",
        REPO_ROOT / "matchgw" / "aux_priors" / "observed_sky.py",
        REPO_ROOT / "scripts" / "sky" / "sky_posterior_overlap.py",
    ]
    with tarfile.open(package, "w:gz") as tar:
        for path in scripts:
            tar.add(path, arcname=str(path.relative_to(REPO_ROOT)))
        for path in out_root.rglob("*"):
            if path.is_file() and "cache" not in path.relative_to(out_root).parts:
                tar.add(path, arcname=str(Path("results") / out_root.name / path.relative_to(out_root)))


def run(args: argparse.Namespace) -> None:
    out_root = args.out_root
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "per_scenario").mkdir(exist_ok=True)
    cache = out_root / "cache"
    cache.mkdir(exist_ok=True)
    scenarios = scenario_grid()[: args.max_scenarios] if args.max_scenarios else scenario_grid()
    write_json(
        out_root / "run_config.json",
        {
            "sky_seeds": args.sky_seeds,
            "scenarios": scenarios,
            "planned_runs": len(scenarios) * len(args.sky_seeds),
            "false_samples": args.false_samples,
            "mainline_weights": MAINLINE_WEIGHTS,
            "mainline_lambda_after_common_scaling": MAINLINE_LAMBDA,
            "mainline_weight_source": MAINLINE_WEIGHT_SOURCE,
            "weight_protocol": "frozen before sensitivity scan; no per-scenario retuning",
            "gpu_required": False,
            "cgroup_memory_strategy": "stream only valid query rows in chunks",
        },
    )

    base, val, test = load_et3_without_model()
    _, _, val_time, val_gt = val
    test_meta, test_raw, test_time, test_gt = test
    liao = load_module("liao_sensitivity_stream", REPO_ROOT / "scripts" / "experiments" / "88_liao_realistic_p1_p2_rerank.py")
    fresh = importlib.import_module("scripts.experiments.84_fresh50_full_catalog_ranking")
    prior = liao.fit_time_lr_from_liao("ET3", val_time, val_gt, seed=20260712)
    test_queries, test_base_path = build_base_rows(cache / "test_waveform_time_base.npy", ET3_ENCODER_DIR / "test_scores.npy", test_time, test_gt, prior)
    evict_file_cache(ET3_ENCODER_DIR / "test_scores.npy")

    for sky_seed in args.sky_seeds:
        ordered = sorted(scenarios, key=lambda item: item["scenario_id"] != "baseline")
        for scenario in ordered:
            run_id = f"{scenario['scenario_id']}__seed_{sky_seed}"
            run_dir = out_root / "per_scenario" / run_id
            marker = run_dir / "complete.json"
            if marker.exists():
                continue
            run_dir.mkdir(parents=True, exist_ok=True)
            started = time.perf_counter()
            print(f"SKY_SCAN {run_id}", flush=True)
            test_sky = build_sky(test_raw, test_time, scenario, sky_seed + 20_000)
            test_cos_path = cache / f"{run_id}_test_cos.npy"
            test_bf_path = cache / f"{run_id}_test_bf.npy"
            write_sky_row_files(test_sky, test_queries, test_cos_path, test_bf_path)
            false_i, false_j = sample_false_pairs(test_gt, args.false_samples, sky_seed + 30_000)
            metric_rows: list[dict[str, Any]] = []
            diagnostic_rows = []
            for score_type, path in (("log_cosine", test_cos_path), ("log_sky_bayes_factor", test_bf_path)):
                sky_ranks = ranks_for_lambda(None, path, test_queries, test_gt, 1.0)
                sky_metrics = metrics_from_ranks(fresh, sky_ranks, test_queries, test_gt, test_meta)
                metric_rows.extend(flatten_metrics(sky_metrics, scenario, sky_seed, score_type, "sky_only", 1.0, "none"))
                fixed_ranks = ranks_for_lambda(test_base_path, path, test_queries, test_gt, MAINLINE_LAMBDA)
                fixed_metrics = metrics_from_ranks(fresh, fixed_ranks, test_queries, test_gt, test_meta)
                metric_rows.extend(
                    flatten_metrics(
                        fixed_metrics, scenario, sky_seed, score_type,
                        MAINLINE_METHOD, MAINLINE_LAMBDA, "frozen_unified_mainline_validation",
                    )
                )
                diagnostic_rows.append(
                    {
                        **scenario,
                        "sky_seed": sky_seed,
                        "split": "test",
                        "score_type": score_type,
                        "a90_median_deg2": float(test_sky["sky_area90_deg2"].median()),
                        "a90_p10_deg2": float(test_sky["sky_area90_deg2"].quantile(0.1)),
                        "a90_p90_deg2": float(test_sky["sky_area90_deg2"].quantile(0.9)),
                        "fraction_at_min_clip": float(np.mean(np.isclose(test_sky["sky_area90_deg2"], scenario["clip_min_deg2"]))),
                        "fraction_at_max_clip": float(np.mean(np.isclose(test_sky["sky_area90_deg2"], scenario["clip_max_deg2"]))),
                        **diagnostic_row(test_sky, test_gt, false_i, false_j, score_type),
                    }
                )
            pd.DataFrame(metric_rows).to_csv(run_dir / "metrics.csv", index=False)
            pd.DataFrame(diagnostic_rows).to_csv(run_dir / "diagnostics.csv", index=False)
            write_json(
                run_dir / "weight_protocol.json",
                {
                    "weights": MAINLINE_WEIGHTS,
                    "equivalent_lambda": MAINLINE_LAMBDA,
                    "source": MAINLINE_WEIGHT_SOURCE,
                    "retuned_for_scenario": False,
                },
            )
            write_json(
                marker,
                {
                    "scenario": scenario,
                    "sky_seed": sky_seed,
                    "mainline_weights": MAINLINE_WEIGHTS,
                    "equivalent_lambda": MAINLINE_LAMBDA,
                    "elapsed_seconds": time.perf_counter() - started,
                    "status": "complete",
                },
            )
            test_cos_path.unlink(missing_ok=True)
            test_bf_path.unlink(missing_ok=True)
            del test_sky
            gc.collect()
    aggregate(out_root)
    package_outputs(out_root, args.package_path)
    print(json.dumps({"status": "complete", "out_root": str(out_root), "package": str(args.package_path)}, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--package-path", type=Path, default=DEFAULT_PACKAGE)
    parser.add_argument("--sky-seeds", type=int, nargs="+", default=list(DEFAULT_SKY_SEEDS))
    parser.add_argument("--false-samples", type=int, default=200_000)
    parser.add_argument("--max-scenarios", type=int, default=None)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
