from __future__ import annotations

import importlib
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


runner = importlib.import_module("scripts.experiments.92_ligo_h1l1_full_experiment_runner")
fresh, liao, _pdf = runner.configure_modules()

OUT_DIR = runner.RERANK_ROOT / "stage8_true_sky_oracle_combinations"
GRID = [0.25, 0.5, 1.0, 2.0, 4.0]
DETECTOR = "LIGO"
MODE = "noisy"
TRUE_SKY_A90_DEG2 = 1.0


def true_sky_table(raw_obs: pd.DataFrame, a90_deg2: float = TRUE_SKY_A90_DEG2) -> pd.DataFrame:
    """Build an oracle sky table from true ra/dec.

    This is an upper-bound ablation. It intentionally bypasses observed-sky
    center sampling, so it must not be mixed with deployable observed-sky
    results in the main table.
    """
    sigma = liao.sky_sigma_from_a90_deg2(np.full(len(raw_obs), float(a90_deg2), dtype=np.float64))
    return pd.DataFrame({
        "event_id": np.arange(len(raw_obs), dtype=np.int64),
        "scenario": "TRUE_SKY_ORACLE",
        "sky_model": "oracle_true_ra_dec_fixed_a90",
        "sky_sampling": "none_true_center",
        "snr_for_sky_mode": "none",
        "snr_for_sky": np.nan,
        "a90_ref_deg2": float(a90_deg2),
        "ra_true": raw_obs["ra"].to_numpy(dtype=np.float64),
        "dec_true": raw_obs["dec"].to_numpy(dtype=np.float64),
        "ra_obs": raw_obs["ra"].to_numpy(dtype=np.float64),
        "dec_obs": raw_obs["dec"].to_numpy(dtype=np.float64),
        "sky_area90_deg2": float(a90_deg2),
        "sky_sigma_rad": sigma,
        "uses_h1l1_timing": False,
        "uses_antenna_pattern_localization": False,
        "uses_healpix_skymap": False,
    })


def true_sky_score_matrices(raw_obs: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sky = true_sky_table(raw_obs)
    features = liao.observed_sky_pair_features(sky, chunk_rows=liao.CHUNK_ROWS)
    true_sep_score = -features["sky_sep_obs"]
    np.fill_diagonal(true_sep_score, -np.inf)
    return true_sep_score, features["sky_step_weight"], features["sky_log_overlap"]


def add_rows(rows: list[dict], variant: str, metrics: dict[str, dict], diag: dict, extra: dict | None = None) -> None:
    extra = extra or {}
    for subset, values in metrics.items():
        rows.append({
            "detector": DETECTOR,
            "data_mode": MODE,
            "stage": "stage8_true_sky_oracle_combinations",
            "variant": variant,
            "subset": subset,
            **diag,
            **extra,
            **values,
        })


def score_from_weights(components: dict[str, np.ndarray], keys: list[str], weights: dict[str, float]) -> np.ndarray:
    score = np.zeros_like(components[keys[0]], dtype=np.float32)
    for key in keys:
        score = score + float(weights[key]) * components[key]
    np.fill_diagonal(score, -np.inf)
    return score


def select_weights(
    val_components: dict[str, np.ndarray],
    val_gt: np.ndarray,
    val_meta: list[dict],
    keys: list[str],
) -> tuple[dict[str, float], dict]:
    fixed = {"waveform": 1.0} if "waveform" in keys else {}
    tune_keys = [key for key in keys if key not in fixed]
    best_weights = {key: 1.0 for key in keys}
    best_weights.update(fixed)
    best_metrics = None
    best_key = (-1.0, -1.0, -1.0, -1.0)
    for values in itertools.product(GRID, repeat=len(tune_keys)):
        weights = dict(fixed)
        weights.update(dict(zip(tune_keys, values)))
        score = score_from_weights(val_components, keys, weights)
        metrics = liao.evaluate_score(score, val_gt, val_meta)
        rank_key = (
            metrics["overall"]["r@10"],
            metrics["overall"]["r@5"],
            metrics["overall"]["r@1"],
            metrics["overall"]["top_1pct"],
        )
        if rank_key > best_key:
            best_key = rank_key
            best_weights = weights
            best_metrics = metrics
    return best_weights, best_metrics


def run() -> pd.DataFrame:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    print("STAGE8_TRUE_SKY_ORACLE_COMBINATIONS", DETECTOR, MODE, flush=True)

    loaded = liao.load_job(DETECTOR, MODE)
    cfg = loaded["cfg"]
    val_ds, val_raw, val_time, val_gt, val_scores = loaded["val"]
    test_ds, test_raw, test_time, test_gt, test_scores = loaded["test"]

    prior = liao.fit_time_lr_from_liao(DETECTOR, val_time, val_gt)
    val_sep, val_step, val_log = true_sky_score_matrices(val_raw)
    test_sep, test_step, test_log = true_sky_score_matrices(test_raw)

    val_components = {
        "waveform": liao.row_z(val_scores),
        "raw_time": liao.row_z(liao.raw_time_score_matrix(val_time)),
        "liao_time_lr": liao.row_z(liao.time_lr_score_matrix(val_time, prior)),
        "true_sky_sep": liao.row_z(val_sep),
        "true_sky_step": liao.row_z(val_step),
        "true_sky_log_overlap": liao.row_z(val_log),
    }
    test_components = {
        "waveform": liao.row_z(test_scores),
        "raw_time": liao.row_z(liao.raw_time_score_matrix(test_time)),
        "liao_time_lr": liao.row_z(liao.time_lr_score_matrix(test_time, prior)),
        "true_sky_sep": liao.row_z(test_sep),
        "true_sky_step": liao.row_z(test_step),
        "true_sky_log_overlap": liao.row_z(test_log),
    }

    diag = liao.base_diag(test_ds, cfg, {
        "liao_label": prior["liao_label"],
        "liao_delay_count": prior["liao_delay_count"],
        "sky_label": "oracle true ra/dec fixed A90",
        "sky_scenario": "TRUE_SKY_ORACLE",
        "true_sky_a90_deg2": TRUE_SKY_A90_DEG2,
        "weight_grid": str(GRID),
        "deployment_note": "upper_bound_only_not_deployable",
    })

    variants = [
        ("waveform_only", ["waveform"]),
        ("raw_time_only", ["raw_time"]),
        ("liao_time_lr_only", ["liao_time_lr"]),
        ("true_sky_sep_only", ["true_sky_sep"]),
        ("true_sky_step_only", ["true_sky_step"]),
        ("true_sky_log_overlap_only", ["true_sky_log_overlap"]),
        ("waveform_plus_true_sky_sep", ["waveform", "true_sky_sep"]),
        ("waveform_plus_true_sky_step", ["waveform", "true_sky_step"]),
        ("waveform_plus_true_sky_log_overlap", ["waveform", "true_sky_log_overlap"]),
        ("raw_time_plus_true_sky_step", ["raw_time", "true_sky_step"]),
        ("raw_time_plus_true_sky_log_overlap", ["raw_time", "true_sky_log_overlap"]),
        ("liao_time_lr_plus_true_sky_step", ["liao_time_lr", "true_sky_step"]),
        ("liao_time_lr_plus_true_sky_log_overlap", ["liao_time_lr", "true_sky_log_overlap"]),
        ("waveform_plus_raw_time_plus_true_sky_step", ["waveform", "raw_time", "true_sky_step"]),
        ("waveform_plus_liao_time_lr_plus_true_sky_step", ["waveform", "liao_time_lr", "true_sky_step"]),
        ("waveform_plus_liao_time_lr_plus_true_sky_log_overlap", ["waveform", "liao_time_lr", "true_sky_log_overlap"]),
    ]

    for variant, keys in variants:
        print("STAGE8_VARIANT", variant, flush=True)
        if len(keys) == 1:
            weights = {keys[0]: 1.0}
            val_metric = liao.evaluate_score(val_components[keys[0]], val_gt, val_ds.meta)
        else:
            weights, val_metric = select_weights(val_components, val_gt, val_ds.meta, keys)
        test_score = score_from_weights(test_components, keys, weights)
        extra = {
            "component_keys": ",".join(keys),
            "val_selected_r@1": val_metric["overall"]["r@1"],
            "val_selected_r@5": val_metric["overall"]["r@5"],
            "val_selected_r@10": val_metric["overall"]["r@10"],
        }
        for key in ["waveform", "raw_time", "liao_time_lr", "true_sky_sep", "true_sky_step", "true_sky_log_overlap"]:
            if key in weights:
                extra[f"lambda_{key}"] = float(weights[key])
        add_rows(rows, variant, liao.evaluate_score(test_score, test_gt, test_ds.meta), diag, extra)
        pd.DataFrame(rows).to_csv(OUT_DIR / "stage8_true_sky_oracle_combinations_partial.csv", index=False)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "stage8_true_sky_oracle_combinations_summary.csv", index=False)
    return df


if __name__ == "__main__":
    run()
