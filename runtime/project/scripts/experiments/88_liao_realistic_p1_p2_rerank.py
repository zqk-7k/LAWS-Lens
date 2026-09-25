from __future__ import annotations

import argparse
import ast
import importlib
import json
import math
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from matchgw.aux_priors import (
    DETECTOR_SKY_SCENARIOS,
    build_observed_sky_table,
    observed_sky_pair_features,
    public_observed_sky_features,
    rank_feature_matrices,
    scenario_for_detector,
    select_best_weighted_lambdas,
    time_step_score_matrix,
)
from matchgw.aux_priors.observed_sky import with_a90_ref

base = importlib.import_module("scripts.experiments.80_mixed_sis_pm_catalog_modality_compare")
hard = importlib.import_module("scripts.experiments.81_time_matched_hard_negative_mixed_catalog")
fresh = importlib.import_module("scripts.experiments.84_fresh50_full_catalog_ranking")

OUT_ROOT = Path("runs/liao_realistic_p1_p2_rerank_20260612")
DOC_ROOT = Path("docs")
ENCODER_ROOT = Path("runs/fresh50_full_catalog_ranking_20260611/fresh_mixed_encoders")
GW_LMC_ROOT = Path("/root/autodl-tmp/GW-LMC")
FAMILIES = ["SIS", "PM"]
JOBS = [("ET", "noisy"), ("LIGO", "noisy")]
CHUNK_ROWS = 48
EPS = 1e-8
SECONDS_PER_DAY = 86400.0
DEG2_TO_SR_EQUIV = (math.pi / 180.0) ** 2


LIAO_PRIOR_CONFIG = {
    "ET": {
        "label": "GW-LMC ET BBH Any_Detected_SNR8",
        "image_csv": GW_LMC_ROOT / "ET/BBH/Any_Detected_SNR8/BBH_ET_Any_Detected_SNR8_ImageParams.csv",
        "snr_threshold": 8.0,
    },
    "ET3": {
        "label": "GW-LMC ET BBH Any_Detected_SNR8",
        "image_csv": GW_LMC_ROOT / "ET/BBH/Any_Detected_SNR8/BBH_ET_Any_Detected_SNR8_ImageParams.csv",
        "snr_threshold": 8.0,
    },
    "LIGO": {
        "label": "GW-LMC 2.5PLUS BBH Any_Detected_SNR1",
        "image_csv": GW_LMC_ROOT / "2.5PLUS/BBH/Any_Detected_SNR1/BBH_2.5PLUS_Any_Detected_SNR1_ImageParams.csv",
        "snr_threshold": 1.0,
    },
}

OBSERVED_SKY_CONFIG = {
    "ET": {
        "scenario": "ET_SINGLE",
        "label": DETECTOR_SKY_SCENARIOS["ET_SINGLE"].label,
        "a90_ref_deg2": DETECTOR_SKY_SCENARIOS["ET_SINGLE"].a90_ref_deg2,
    },
    "ET3": {
        "scenario": "ET_TRIANGLE",
        "label": DETECTOR_SKY_SCENARIOS["ET_TRIANGLE"].label,
        "a90_ref_deg2": DETECTOR_SKY_SCENARIOS["ET_TRIANGLE"].a90_ref_deg2,
    },
    "LIGO": {
        "scenario": "LIGO_HL",
        "label": DETECTOR_SKY_SCENARIOS["LIGO_HL"].label,
        "a90_ref_deg2": DETECTOR_SKY_SCENARIOS["LIGO_HL"].a90_ref_deg2,
    },
}

SKY_A90_SWEEP = {
    "ET": [100.0, 300.0, 1000.0],
    "ET3": [50.0, 100.0, 300.0],
    "LIGO": [50.0, 100.0, 200.0],
}


def fmt(x) -> str:
    if pd.isna(x):
        return ""
    if isinstance(x, str):
        return x
    x = float(x)
    if abs(x - round(x)) < 1e-9 and abs(x) >= 10:
        return str(int(round(x)))
    return f"{x:.4f}".rstrip("0").rstrip(".")


def md_table(rows: list[dict], headers: list[str]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] + ["---:" for _ in headers[1:]]) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")
    return "\n".join(lines)


def parse_list(value) -> list[float]:
    if pd.isna(value):
        return []
    try:
        parsed = ast.literal_eval(str(value))
    except Exception:
        return []
    return [float(x) for x in parsed]


def extract_liao_detected_pair_delays_days(image_csv: Path, snr_threshold: float) -> np.ndarray:
    df = pd.read_csv(image_csv)
    delays = []
    for _, row in df.iterrows():
        ds = parse_list(row.get("img_delays_days"))
        snrs = parse_list(row.get("img_snrs"))
        if len(ds) != len(snrs) or len(ds) < 2:
            continue
        keep = [idx for idx, snr in enumerate(snrs) if snr >= snr_threshold]
        if len(keep) < 2:
            continue
        for a_pos, a in enumerate(keep):
            for b in keep[a_pos + 1:]:
                dt = abs(ds[b] - ds[a])
                if np.isfinite(dt) and dt > 0:
                    delays.append(dt)
    return np.asarray(delays, dtype=np.float64)


def family_by_index(meta: list[dict], rows: np.ndarray) -> np.ndarray:
    return np.asarray([meta[int(row)]["family"] for row in rows])


def row_z(score: np.ndarray) -> np.ndarray:
    out = score.astype(np.float32).copy()
    np.fill_diagonal(out, np.nan)
    mu = np.nanmean(out, axis=1, keepdims=True)
    sd = np.nanstd(out, axis=1, keepdims=True)
    out = (out - mu) / np.maximum(sd, EPS)
    np.fill_diagonal(out, -np.inf)
    return out.astype(np.float32)


def load_model_only(cfg, arrays=None):
    model_path = cfg.out_dir / "model.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing fresh50 model: {model_path}")
    in_channels = None
    if arrays is not None:
        in_channels = base.data_mod.prepared_channel_count(arrays[FAMILIES[0]].l1[0], cfg)
    model = base.build_model(cfg, in_channels=in_channels)
    ckpt = torch.load(model_path, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)
    return model


def load_job(detector: str, mode: str):
    encoder_dir = ENCODER_ROOT / f"{detector.lower()}_{mode}_mixed_sis_pm_ep50"
    cfg = base.make_cfg(detector, mode, encoder_dir)
    arrays = {family: base.FamilyArrays(family, base.ROOTS[(family, detector)], mode) for family in FAMILIES}
    splits = {}
    for index, family in enumerate(FAMILIES):
        splits[family] = base.split_indices(len(arrays[family].l1), cfg.seed + index)
        splits[f"{family}_U"] = base.split_indices(len(arrays[family].unlensed), cfg.seed + 100 + index)
    model = load_model_only(cfg, arrays)
    val_ds, val_raw, val_time, val_gt, val_emb, val_scores = base.split_pack(detector, "val", cfg, arrays, splits, model)
    test_ds, test_raw, test_time, test_gt, test_emb, test_scores = base.split_pack(detector, "test", cfg, arrays, splits, model)
    return {
        "cfg": cfg,
        "val": (val_ds, val_raw, val_time, val_gt, val_scores),
        "test": (test_ds, test_raw, test_time, test_gt, test_scores),
    }


def raw_time_score_matrix(time_obs: pd.DataFrame) -> np.ndarray:
    t = time_obs["trigger_time_obs"].to_numpy(dtype=np.float64)
    n = len(t)
    out = np.empty((n, n), dtype=np.float32)
    for start in range(0, n, CHUNK_ROWS):
        rows = slice(start, min(start + CHUNK_ROWS, n))
        dt_days = np.abs(t[rows, None] - t[None, :]) / SECONDS_PER_DAY
        out[rows] = (-np.log1p(dt_days)).astype(np.float32)
    np.fill_diagonal(out, -np.inf)
    return out


def fit_time_lr_from_liao(detector: str, val_time: pd.DataFrame, val_gt: np.ndarray, seed: int = 20260612) -> dict:
    cfg = LIAO_PRIOR_CONFIG[detector]
    lensed_days = extract_liao_detected_pair_delays_days(cfg["image_csv"], cfg["snr_threshold"])
    if len(lensed_days) < 20:
        raise RuntimeError(f"Too few Liao delays for {detector}: {len(lensed_days)}")

    rng = np.random.default_rng(seed)
    t = val_time["trigger_time_obs"].to_numpy(dtype=np.float64)
    n = len(t)
    sample_n = min(2_000_000, n * 500)
    a = rng.integers(0, n, size=sample_n, dtype=np.int32)
    c = rng.integers(0, n, size=sample_n, dtype=np.int32)
    bad = a == c
    if bad.any():
        c[bad] = (c[bad] + 1) % n
    valid = hard.valid_queries(val_gt)
    mate = np.full(n, -1, dtype=np.int32)
    mate[valid] = val_gt[valid].astype(np.int32)
    same_pair = mate[a] == c
    if same_pair.any():
        c[same_pair] = (c[same_pair] + 7) % n
    random_days = np.abs(t[a] - t[c]) / SECONDS_PER_DAY
    random_days = random_days[np.isfinite(random_days) & (random_days > 0)]

    x_l = np.log10(np.maximum(lensed_days, 1e-6))
    x_u = np.log10(np.maximum(random_days, 1e-6))
    lo = min(-6.0, float(np.percentile(x_l, 0.1)), float(np.percentile(x_u, 0.1)))
    hi = max(6.0, float(np.percentile(x_l, 99.9)), float(np.percentile(x_u, 99.9)))
    edges = np.linspace(lo, hi, 181)
    hist_l, _ = np.histogram(x_l, bins=edges)
    hist_u, _ = np.histogram(x_u, bins=edges)
    alpha = 1.0
    p_l = (hist_l + alpha) / float(hist_l.sum() + alpha * len(hist_l))
    p_u = (hist_u + alpha) / float(hist_u.sum() + alpha * len(hist_u))
    lr = np.log(p_l) - np.log(p_u)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return {
        "detector": detector,
        "liao_label": cfg["label"],
        "liao_image_csv": str(cfg["image_csv"]),
        "liao_snr_threshold": float(cfg["snr_threshold"]),
        "liao_delay_count": int(len(lensed_days)),
        "liao_delay_median_days": float(np.median(lensed_days)),
        "liao_delay_p90_days": float(np.percentile(lensed_days, 90)),
        "random_delay_count": int(len(random_days)),
        "random_delay_median_days": float(np.median(random_days)),
        "edges": edges,
        "centers": centers,
        "lr": lr.astype(np.float32),
    }


def time_lr_score_matrix(time_obs: pd.DataFrame, prior: dict) -> np.ndarray:
    t = time_obs["trigger_time_obs"].to_numpy(dtype=np.float64)
    edges = prior["edges"]
    lr = prior["lr"]
    n = len(t)
    out = np.empty((n, n), dtype=np.float32)
    for start in range(0, n, CHUNK_ROWS):
        rows = slice(start, min(start + CHUNK_ROWS, n))
        dt_days = np.abs(t[rows, None] - t[None, :]) / SECONDS_PER_DAY
        x = np.log10(np.maximum(dt_days, 1e-6))
        idx = np.searchsorted(edges, x, side="right") - 1
        idx = np.clip(idx, 0, len(lr) - 1)
        out[rows] = lr[idx].astype(np.float32)
    np.fill_diagonal(out, -np.inf)
    return out


def extract_liao_delay_snr_pairs(image_csv: Path, snr_threshold: float) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(image_csv)
    delays = []
    ratios = []
    for _, row in df.iterrows():
        ds = parse_list(row.get("img_delays_days"))
        snrs = parse_list(row.get("img_snrs"))
        if len(ds) != len(snrs) or len(ds) < 2:
            continue
        keep = [idx for idx, snr in enumerate(snrs) if snr >= snr_threshold]
        if len(keep) < 2:
            continue
        for a_pos, a in enumerate(keep):
            for b in keep[a_pos + 1:]:
                dt = abs(ds[b] - ds[a])
                s1 = max(float(snrs[a]), EPS)
                s2 = max(float(snrs[b]), EPS)
                ratio = max(s1, s2) / max(min(s1, s2), EPS)
                if np.isfinite(dt) and dt > 0 and np.isfinite(ratio):
                    delays.append(dt)
                    ratios.append(ratio)
    return np.asarray(delays, dtype=np.float64), np.asarray(ratios, dtype=np.float64)


def raw_snr_ratio_score_matrix(time_obs: pd.DataFrame) -> np.ndarray:
    snr = np.maximum(time_obs["snr"].to_numpy(dtype=np.float64), EPS)
    log_snr = np.log(snr)
    n = len(snr)
    out = np.empty((n, n), dtype=np.float32)
    for start in range(0, n, CHUNK_ROWS):
        rows = slice(start, min(start + CHUNK_ROWS, n))
        # Raw baseline: favor similar observed SNRs. The Liao 2D prior below is the more physical version.
        out[rows] = (-np.abs(log_snr[rows, None] - log_snr[None, :])).astype(np.float32)
    np.fill_diagonal(out, -np.inf)
    return out


def fit_amp_time_lr_from_liao(detector: str, val_time: pd.DataFrame, val_gt: np.ndarray, seed: int = 20260612) -> dict:
    cfg = LIAO_PRIOR_CONFIG[detector]
    lensed_days, lensed_ratios = extract_liao_delay_snr_pairs(cfg["image_csv"], cfg["snr_threshold"])
    if len(lensed_days) < 20:
        raise RuntimeError(f"Too few Liao delay/SNR-ratio pairs for {detector}: {len(lensed_days)}")

    rng = np.random.default_rng(seed + 91)
    t = val_time["trigger_time_obs"].to_numpy(dtype=np.float64)
    snr = np.maximum(val_time["snr"].to_numpy(dtype=np.float64), EPS)
    n = len(t)
    sample_n = min(2_000_000, n * 500)
    a = rng.integers(0, n, size=sample_n, dtype=np.int32)
    c = rng.integers(0, n, size=sample_n, dtype=np.int32)
    bad = a == c
    if bad.any():
        c[bad] = (c[bad] + 1) % n
    valid = hard.valid_queries(val_gt)
    mate = np.full(n, -1, dtype=np.int32)
    mate[valid] = val_gt[valid].astype(np.int32)
    same_pair = mate[a] == c
    if same_pair.any():
        c[same_pair] = (c[same_pair] + 11) % n
    random_days = np.abs(t[a] - t[c]) / SECONDS_PER_DAY
    random_ratio = np.maximum(snr[a], snr[c]) / np.maximum(np.minimum(snr[a], snr[c]), EPS)
    ok = np.isfinite(random_days) & (random_days > 0) & np.isfinite(random_ratio)
    random_days = random_days[ok]
    random_ratio = random_ratio[ok]

    x_l = np.log10(np.maximum(lensed_days, 1e-6))
    y_l = np.log(np.maximum(lensed_ratios, 1.0))
    x_u = np.log10(np.maximum(random_days, 1e-6))
    y_u = np.log(np.maximum(random_ratio, 1.0))
    x_lo = min(-6.0, float(np.percentile(x_l, 0.1)), float(np.percentile(x_u, 0.1)))
    x_hi = max(6.0, float(np.percentile(x_l, 99.9)), float(np.percentile(x_u, 99.9)))
    y_hi = max(float(np.percentile(y_l, 99.9)), float(np.percentile(y_u, 99.9)), 1.0)
    x_edges = np.linspace(x_lo, x_hi, 121)
    y_edges = np.linspace(0.0, y_hi, 81)
    hist_l, _, _ = np.histogram2d(x_l, y_l, bins=[x_edges, y_edges])
    hist_u, _, _ = np.histogram2d(x_u, y_u, bins=[x_edges, y_edges])
    alpha = 1.0
    p_l = (hist_l + alpha) / float(hist_l.sum() + alpha * hist_l.size)
    p_u = (hist_u + alpha) / float(hist_u.sum() + alpha * hist_u.size)
    lr = np.log(p_l) - np.log(p_u)
    return {
        "detector": detector,
        "liao_label": cfg["label"],
        "liao_image_csv": str(cfg["image_csv"]),
        "liao_snr_threshold": float(cfg["snr_threshold"]),
        "liao_pair_count": int(len(lensed_days)),
        "liao_snr_ratio_median": float(np.median(lensed_ratios)),
        "liao_snr_ratio_p90": float(np.percentile(lensed_ratios, 90)),
        "random_snr_ratio_median": float(np.median(random_ratio)),
        "x_edges": x_edges,
        "y_edges": y_edges,
        "lr": lr.astype(np.float32),
    }


def amp_time_lr_score_matrix(time_obs: pd.DataFrame, prior: dict) -> np.ndarray:
    t = time_obs["trigger_time_obs"].to_numpy(dtype=np.float64)
    snr = np.maximum(time_obs["snr"].to_numpy(dtype=np.float64), EPS)
    log_snr = np.log(snr)
    x_edges = prior["x_edges"]
    y_edges = prior["y_edges"]
    lr = prior["lr"]
    n = len(t)
    out = np.empty((n, n), dtype=np.float32)
    for start in range(0, n, CHUNK_ROWS):
        rows = slice(start, min(start + CHUNK_ROWS, n))
        dt_days = np.abs(t[rows, None] - t[None, :]) / SECONDS_PER_DAY
        x = np.log10(np.maximum(dt_days, 1e-6))
        y = np.abs(log_snr[rows, None] - log_snr[None, :])
        xi = np.searchsorted(x_edges, x, side="right") - 1
        yi = np.searchsorted(y_edges, y, side="right") - 1
        xi = np.clip(xi, 0, lr.shape[0] - 1)
        yi = np.clip(yi, 0, lr.shape[1] - 1)
        out[rows] = lr[xi, yi].astype(np.float32)
    np.fill_diagonal(out, -np.inf)
    return out


def unit_from_radec(ra: np.ndarray, dec: np.ndarray) -> np.ndarray:
    return np.column_stack([
        np.cos(dec) * np.cos(ra),
        np.cos(dec) * np.sin(ra),
        np.sin(dec),
    ]).astype(np.float64)


def radec_from_unit(vec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    v = vec / np.maximum(np.linalg.norm(vec, axis=1, keepdims=True), EPS)
    ra = np.mod(np.arctan2(v[:, 1], v[:, 0]), 2.0 * np.pi)
    dec = np.arcsin(np.clip(v[:, 2], -1.0, 1.0))
    return ra.astype(np.float64), dec.astype(np.float64)


def angular_sep_unit(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.arccos(np.clip(np.sum(a * b, axis=-1), -1.0, 1.0))


def sky_sigma_from_a90_deg2(a90_deg2: np.ndarray) -> np.ndarray:
    a90_rad2 = a90_deg2 * DEG2_TO_SR_EQUIV
    return np.sqrt(a90_rad2 / (2.0 * np.pi * np.log(10.0))).astype(np.float64)


def make_observed_sky(
    detector: str,
    raw_obs: pd.DataFrame,
    time_obs: pd.DataFrame,
    seed: int,
    a90_ref_deg2: float | None = None,
    scenario_name: str | None = None,
    sampling: str = "tangent_2d_gaussian",
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    scenario = scenario_for_detector(detector, scenario_name)
    scenario = with_a90_ref(scenario, a90_ref_deg2)
    return build_observed_sky_table(raw_obs, time_obs, scenario, rng, sampling=sampling)


def observed_sky_score_matrices(sky_obs: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features = observed_sky_pair_features(public_observed_sky_features(sky_obs), chunk_rows=CHUNK_ROWS)
    return features["sky_step_weight"], features["sky_gaussian_weight"], features["sky_log_overlap"]


def evaluate_score(score: np.ndarray, gt: np.ndarray, meta: list[dict]) -> dict[str, dict]:
    rows, ranks = fresh.ranks_from_score_matrix(score, gt)
    return fresh.metrics_from_ranks(ranks, family_by_index(meta, rows), len(gt))


def add_rows(rows: list[dict], detector: str, mode: str, stage: str, variant: str, metrics: dict[str, dict], diag: dict, extra: dict | None = None) -> None:
    extra = extra or {}
    for subset, values in metrics.items():
        rows.append({
            "detector": detector,
            "data_mode": mode,
            "stage": stage,
            "variant": variant,
            "subset": subset,
            **diag,
            **values,
            **extra,
        })


def base_diag(ds, cfg, extra: dict | None = None) -> dict:
    extra = extra or {}
    return {
        "catalog": "fresh50_mixed_SIS_PM_unlensed_full_catalog",
        "candidate_kind": "full_catalog",
        "query_total": int(len([x for x in ds.meta if x.get("tag") in {"L1", "L2"}])),
        **fresh.full_catalog_composition(ds.meta),
        "epochs": int(cfg.epochs),
        "backbone": cfg.backbone,
        "preprocess": cfg.preprocess,
        **extra,
    }


def select_best_lambda(val_components: dict[str, np.ndarray], val_gt: np.ndarray, val_meta: list[dict], lambdas: list[float], prior_key: str) -> tuple[float, dict]:
    best_lam = lambdas[0]
    best_metrics = None
    best_key = (-1.0, -1.0, -1.0)
    for lam in lambdas:
        score = val_components["waveform"] + lam * val_components[prior_key]
        np.fill_diagonal(score, -np.inf)
        metrics = evaluate_score(score, val_gt, val_meta)
        key = (
            metrics["overall"]["r@10"],
            metrics["overall"]["r@5"],
            metrics["overall"]["r@1"],
        )
        if key > best_key:
            best_key = key
            best_lam = lam
            best_metrics = metrics
    return best_lam, best_metrics


def stage0_baseline() -> pd.DataFrame:
    out_dir = OUT_ROOT / "stage0_baseline"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    lambdas = [0.25, 0.5, 1.0, 2.0, 4.0]
    for detector, mode in JOBS:
        print("STAGE0", detector, mode, flush=True)
        loaded = load_job(detector, mode)
        cfg = loaded["cfg"]
        val_ds, _, val_time, val_gt, val_scores = loaded["val"]
        test_ds, _, test_time, test_gt, test_scores = loaded["test"]
        val_components = {
            "waveform": row_z(val_scores),
            "raw_time": row_z(raw_time_score_matrix(val_time)),
        }
        test_components = {
            "waveform": row_z(test_scores),
            "raw_time": row_z(raw_time_score_matrix(test_time)),
        }
        lam, val_metric = select_best_lambda(val_components, val_gt, val_ds.meta, lambdas, "raw_time")
        diag = base_diag(test_ds, cfg)
        add_rows(rows, detector, mode, "stage0_baseline", "waveform_only", evaluate_score(test_components["waveform"], test_gt, test_ds.meta), diag)
        add_rows(rows, detector, mode, "stage0_baseline", "raw_time_only", evaluate_score(test_components["raw_time"], test_gt, test_ds.meta), diag)
        fused = test_components["waveform"] + lam * test_components["raw_time"]
        np.fill_diagonal(fused, -np.inf)
        add_rows(rows, detector, mode, "stage0_baseline", "waveform_plus_raw_time_val_selected", evaluate_score(fused, test_gt, test_ds.meta), diag, {
            "lambda_time": lam,
            "val_selected_r@10": val_metric["overall"]["r@10"],
        })
        pd.DataFrame(rows).to_csv(out_dir / "stage0_baseline_partial.csv", index=False)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "stage0_baseline_summary.csv", index=False)
    write_stage_doc("stage0_baseline", df, out_dir, notes=[
        "Stage0 只使用当前数据自身的 waveform similarity 与 raw trigger_time_obs 时间差。",
        "该阶段不使用 GW-LMC/Liao prior、不使用 observed sky、不使用 SNR ratio。",
        "waveform_plus_raw_time 的 lambda 在 validation full catalog 上选择，然后应用到 test full catalog。",
    ])
    return df


def stage1_liao_time() -> pd.DataFrame:
    out_dir = OUT_ROOT / "stage1_liao_time_lr"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    prior_rows = []
    lambdas = [0.25, 0.5, 1.0, 2.0, 4.0]
    for detector, mode in JOBS:
        print("STAGE1_LIAO_TIME", detector, mode, flush=True)
        loaded = load_job(detector, mode)
        cfg = loaded["cfg"]
        val_ds, _, val_time, val_gt, val_scores = loaded["val"]
        test_ds, _, test_time, test_gt, test_scores = loaded["test"]
        prior = fit_time_lr_from_liao(detector, val_time, val_gt)
        lensed_days = extract_liao_detected_pair_delays_days(
            LIAO_PRIOR_CONFIG[detector]["image_csv"],
            LIAO_PRIOR_CONFIG[detector]["snr_threshold"],
        )
        prior_rows.append({k: v for k, v in prior.items() if not isinstance(v, np.ndarray)})
        val_components = {
            "waveform": row_z(val_scores),
            "liao_time_step": row_z(time_step_score_matrix(val_time, lensed_days, chunk_rows=CHUNK_ROWS)),
            "liao_time_lr": row_z(time_lr_score_matrix(val_time, prior)),
        }
        test_components = {
            "waveform": row_z(test_scores),
            "liao_time_step": row_z(time_step_score_matrix(test_time, lensed_days, chunk_rows=CHUNK_ROWS)),
            "liao_time_lr": row_z(time_lr_score_matrix(test_time, prior)),
        }
        lam, val_metric = select_best_lambda(val_components, val_gt, val_ds.meta, lambdas, "liao_time_lr")
        diag = base_diag(test_ds, cfg, {
            "liao_label": prior["liao_label"],
            "liao_delay_count": prior["liao_delay_count"],
            "liao_delay_median_days": prior["liao_delay_median_days"],
            "liao_delay_p90_days": prior["liao_delay_p90_days"],
            "random_delay_median_days": prior["random_delay_median_days"],
        })
        add_rows(rows, detector, mode, "stage1_liao_time_lr", "liao_time_step_only", evaluate_score(test_components["liao_time_step"], test_gt, test_ds.meta), diag)
        add_rows(rows, detector, mode, "stage1_liao_time_lr", "liao_time_lr_only", evaluate_score(test_components["liao_time_lr"], test_gt, test_ds.meta), diag)
        step_lam, step_val_metric = select_best_lambda(val_components, val_gt, val_ds.meta, lambdas, "liao_time_step")
        step_fused = test_components["waveform"] + step_lam * test_components["liao_time_step"]
        np.fill_diagonal(step_fused, -np.inf)
        add_rows(rows, detector, mode, "stage1_liao_time_lr", "waveform_plus_liao_time_step_val_selected", evaluate_score(step_fused, test_gt, test_ds.meta), diag, {
            "lambda_liao_time_step": step_lam,
            "val_selected_r@10": step_val_metric["overall"]["r@10"],
        })
        fused = test_components["waveform"] + lam * test_components["liao_time_lr"]
        np.fill_diagonal(fused, -np.inf)
        add_rows(rows, detector, mode, "stage1_liao_time_lr", "waveform_plus_liao_time_lr_val_selected", evaluate_score(fused, test_gt, test_ds.meta), diag, {
            "lambda_liao_time": lam,
            "val_selected_r@10": val_metric["overall"]["r@10"],
        })
        pd.DataFrame(rows).to_csv(out_dir / "stage1_liao_time_lr_partial.csv", index=False)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "stage1_liao_time_lr_summary.csv", index=False)
    pd.DataFrame(prior_rows).to_csv(out_dir / "liao_time_prior_diagnostics.csv", index=False)
    write_stage_doc("stage1_liao_time_lr", df, out_dir, notes=[
        "Stage1 只新增 GW-LMC/Liao time-delay step 与 likelihood-ratio prior。",
        "本阶段不使用 observed sky，不使用 SNR ratio，不使用候选图项。",
        "time step 使用 Liao 检测多图像时延分布的 q10-q90/q05-q95/q01-q99 分位区间。",
        "p(delta_t|lensed) 来自 GW-LMC ImageParams 中检测到的多图像 pair time delay；p(delta_t|random) 来自当前 validation catalog 随机 pair。",
        "lambda 在 validation full catalog 上选择，然后应用到 test full catalog。",
    ], prior_csv=out_dir / "liao_time_prior_diagnostics.csv")
    return df


def stage2_observed_sky() -> pd.DataFrame:
    out_dir = OUT_ROOT / "stage2_observed_sky"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    sky_diag_rows = []
    lambdas = [0.25, 0.5, 1.0, 2.0, 4.0]
    for detector, mode in JOBS:
        print("STAGE2_OBSERVED_SKY", detector, mode, flush=True)
        loaded = load_job(detector, mode)
        cfg = loaded["cfg"]
        val_ds, val_raw, val_time, val_gt, val_scores = loaded["val"]
        test_ds, test_raw, test_time, test_gt, test_scores = loaded["test"]
        val_sky = make_observed_sky(detector, val_raw, val_time, seed=301000 + (0 if detector == "ET" else 1))
        test_sky = make_observed_sky(detector, test_raw, test_time, seed=302000 + (0 if detector == "ET" else 1))
        val_sky.to_csv(out_dir / f"{detector}_{mode}_val_observed_sky_audit.csv", index=False)
        test_sky.to_csv(out_dir / f"{detector}_{mode}_test_observed_sky_audit.csv", index=False)
        val_step, val_gauss, val_log = observed_sky_score_matrices(val_sky)
        test_step, test_gauss, test_log = observed_sky_score_matrices(test_sky)
        sky_diag_rows.append({
            "detector": detector,
            "sky_label": OBSERVED_SKY_CONFIG[detector]["label"],
            "scenario": str(test_sky["scenario"].iloc[0]),
            "sky_model": str(test_sky["sky_model"].iloc[0]),
            "sky_sampling": str(test_sky["sky_sampling"].iloc[0]),
            "snr_for_sky_mode": str(test_sky["snr_for_sky_mode"].iloc[0]),
            "uses_h1l1_timing": bool(test_sky["uses_h1l1_timing"].iloc[0]),
            "uses_antenna_pattern_localization": bool(test_sky["uses_antenna_pattern_localization"].iloc[0]),
            "uses_healpix_skymap": bool(test_sky["uses_healpix_skymap"].iloc[0]),
            "a90_ref_deg2": OBSERVED_SKY_CONFIG[detector]["a90_ref_deg2"],
            "test_a90_median_deg2": float(np.median(test_sky["sky_area90_deg2"])),
            "test_a90_p90_deg2": float(np.percentile(test_sky["sky_area90_deg2"], 90)),
            "test_sigma_median_rad": float(np.median(test_sky["sky_sigma_rad"])),
        })
        val_components = {
            "waveform": row_z(val_scores),
            "sky_step": row_z(val_step),
            "sky_log_overlap": row_z(val_log),
        }
        test_components = {
            "waveform": row_z(test_scores),
            "sky_step": row_z(test_step),
            "sky_log_overlap": row_z(test_log),
        }
        diag = base_diag(test_ds, cfg, {
            "observed_sky_label": OBSERVED_SKY_CONFIG[detector]["label"],
            "a90_ref_deg2": OBSERVED_SKY_CONFIG[detector]["a90_ref_deg2"],
            "test_a90_median_deg2": float(np.median(test_sky["sky_area90_deg2"])),
            "test_a90_p90_deg2": float(np.percentile(test_sky["sky_area90_deg2"], 90)),
        })
        add_rows(rows, detector, mode, "stage2_observed_sky", "observed_sky_step_only", evaluate_score(test_components["sky_step"], test_gt, test_ds.meta), diag)
        add_rows(rows, detector, mode, "stage2_observed_sky", "observed_sky_log_overlap_only", evaluate_score(test_components["sky_log_overlap"], test_gt, test_ds.meta), diag)
        for key, variant_name, lambda_col in [
            ("sky_step", "waveform_plus_observed_sky_step_val_selected", "lambda_sky_step"),
            ("sky_log_overlap", "waveform_plus_observed_sky_log_overlap_val_selected", "lambda_sky_log_overlap"),
        ]:
            lam, val_metric = select_best_lambda(val_components, val_gt, val_ds.meta, lambdas, key)
            fused = test_components["waveform"] + lam * test_components[key]
            np.fill_diagonal(fused, -np.inf)
            add_rows(rows, detector, mode, "stage2_observed_sky", variant_name, evaluate_score(fused, test_gt, test_ds.meta), diag, {
                lambda_col: lam,
                "val_selected_r@10": val_metric["overall"]["r@10"],
            })
        for a90_ref in SKY_A90_SWEEP[detector]:
            val_sky_sweep = make_observed_sky(detector, val_raw, val_time, seed=303000 + int(a90_ref) + (0 if detector == "ET" else 10000), a90_ref_deg2=a90_ref)
            test_sky_sweep = make_observed_sky(detector, test_raw, test_time, seed=304000 + int(a90_ref) + (0 if detector == "ET" else 10000), a90_ref_deg2=a90_ref)
            val_step_sweep, _, val_log_sweep = observed_sky_score_matrices(val_sky_sweep)
            test_step_sweep, _, test_log_sweep = observed_sky_score_matrices(test_sky_sweep)
            sweep_val_components = {
                "waveform": val_components["waveform"],
                "sky_step": row_z(val_step_sweep),
                "sky_log_overlap": row_z(val_log_sweep),
            }
            sweep_test_components = {
                "waveform": test_components["waveform"],
                "sky_step": row_z(test_step_sweep),
                "sky_log_overlap": row_z(test_log_sweep),
            }
            sweep_diag = dict(diag)
            sweep_diag.update({
                "observed_sky_label": f"A90 sweep {a90_ref:g} deg2",
                "a90_ref_deg2": float(a90_ref),
                "test_a90_median_deg2": float(np.median(test_sky_sweep["sky_area90_deg2"])),
                "test_a90_p90_deg2": float(np.percentile(test_sky_sweep["sky_area90_deg2"], 90)),
            })
            for key, suffix, lambda_col in [
                ("sky_step", "step", "lambda_sky_step"),
                ("sky_log_overlap", "gaussian_log_overlap", "lambda_sky_log_overlap"),
            ]:
                lam, val_metric = select_best_lambda(sweep_val_components, val_gt, val_ds.meta, lambdas, key)
                fused = sweep_test_components["waveform"] + lam * sweep_test_components[key]
                np.fill_diagonal(fused, -np.inf)
                add_rows(rows, detector, mode, "stage2_observed_sky", f"a90_{fmt(a90_ref)}_{suffix}_val_selected", evaluate_score(fused, test_gt, test_ds.meta), sweep_diag, {
                    lambda_col: lam,
                    "val_selected_r@10": val_metric["overall"]["r@10"],
                })
        pd.DataFrame(rows).to_csv(out_dir / "stage2_observed_sky_partial.csv", index=False)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "stage2_observed_sky_summary.csv", index=False)
    pd.DataFrame(sky_diag_rows).to_csv(out_dir / "observed_sky_diagnostics.csv", index=False)
    write_stage_doc("stage2_observed_sky", df, out_dir, notes=[
        "Stage2 只新增 observed sky posterior，不使用 Liao time LR、不使用 SNR ratio、不使用候选图。",
        "true ra/dec 只用于模拟观测中心 ra_obs/dec_obs 和 sky_area90；rerank 输入只使用 observed sky posterior 特征。",
        "分别测试 observed sky step 和 observed sky gaussian/log-overlap。",
        "A90 sweep 按任务文档测试 ET=100/300/1000 deg2 与 LIGO=50/100/200 deg2。",
        "lambda 在 validation full catalog 上选择，然后应用到 test full catalog。",
    ], prior_csv=out_dir / "observed_sky_diagnostics.csv")
    return df


def select_best_two_lambdas(
    val_components: dict[str, np.ndarray],
    val_gt: np.ndarray,
    val_meta: list[dict],
    time_key: str,
    sky_key: str,
    lambda_time_grid: list[float],
    lambda_sky_grid: list[float],
) -> tuple[float, float, dict]:
    best_lt = lambda_time_grid[0]
    best_ls = lambda_sky_grid[0]
    best_metrics = None
    best_key = (-1.0, -1.0, -1.0, -1.0)
    for lt in lambda_time_grid:
        for ls in lambda_sky_grid:
            score = val_components["waveform"] + lt * val_components[time_key] + ls * val_components[sky_key]
            np.fill_diagonal(score, -np.inf)
            metrics = evaluate_score(score, val_gt, val_meta)
            key = (
                metrics["overall"]["r@10"],
                metrics["overall"]["r@5"],
                metrics["overall"]["r@1"],
                metrics["overall"]["top_1pct"],
            )
            if key > best_key:
                best_key = key
                best_lt = lt
                best_ls = ls
                best_metrics = metrics
    return best_lt, best_ls, best_metrics


def stage3_liao_time_plus_observed_sky() -> pd.DataFrame:
    out_dir = OUT_ROOT / "stage3_liao_time_plus_observed_sky"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    diag_rows = []
    lambda_time_grid = [0.25, 0.5, 1.0, 2.0, 4.0]
    lambda_sky_grid = [0.25, 0.5, 1.0, 2.0, 4.0]
    for detector, mode in JOBS:
        print("STAGE3_LIAO_TIME_PLUS_OBSERVED_SKY", detector, mode, flush=True)
        loaded = load_job(detector, mode)
        cfg = loaded["cfg"]
        val_ds, val_raw, val_time, val_gt, val_scores = loaded["val"]
        test_ds, test_raw, test_time, test_gt, test_scores = loaded["test"]

        prior = fit_time_lr_from_liao(detector, val_time, val_gt)
        val_sky = make_observed_sky(detector, val_raw, val_time, seed=401000 + (0 if detector == "ET" else 1))
        test_sky = make_observed_sky(detector, test_raw, test_time, seed=402000 + (0 if detector == "ET" else 1))
        val_step, _, val_log = observed_sky_score_matrices(val_sky)
        test_step, _, test_log = observed_sky_score_matrices(test_sky)

        val_components = {
            "waveform": row_z(val_scores),
            "liao_time_lr": row_z(time_lr_score_matrix(val_time, prior)),
            "sky_step": row_z(val_step),
            "sky_log_overlap": row_z(val_log),
        }
        test_components = {
            "waveform": row_z(test_scores),
            "liao_time_lr": row_z(time_lr_score_matrix(test_time, prior)),
            "sky_step": row_z(test_step),
            "sky_log_overlap": row_z(test_log),
        }
        diag_rows.append({
            "detector": detector,
            "liao_label": prior["liao_label"],
            "liao_delay_count": prior["liao_delay_count"],
            "liao_delay_median_days": prior["liao_delay_median_days"],
            "liao_delay_p90_days": prior["liao_delay_p90_days"],
            "sky_label": OBSERVED_SKY_CONFIG[detector]["label"],
            "a90_ref_deg2": OBSERVED_SKY_CONFIG[detector]["a90_ref_deg2"],
            "test_a90_median_deg2": float(np.median(test_sky["sky_area90_deg2"])),
            "test_a90_p90_deg2": float(np.percentile(test_sky["sky_area90_deg2"], 90)),
        })
        diag = base_diag(test_ds, cfg, {
            "liao_label": prior["liao_label"],
            "liao_delay_count": prior["liao_delay_count"],
            "liao_delay_median_days": prior["liao_delay_median_days"],
            "liao_delay_p90_days": prior["liao_delay_p90_days"],
            "observed_sky_label": OBSERVED_SKY_CONFIG[detector]["label"],
            "a90_ref_deg2": OBSERVED_SKY_CONFIG[detector]["a90_ref_deg2"],
            "test_a90_median_deg2": float(np.median(test_sky["sky_area90_deg2"])),
            "test_a90_p90_deg2": float(np.percentile(test_sky["sky_area90_deg2"], 90)),
        })

        for sky_key, variant_name in [
            ("sky_step", "waveform_plus_liao_time_lr_plus_observed_sky_step_val_selected"),
            ("sky_log_overlap", "waveform_plus_liao_time_lr_plus_observed_sky_log_overlap_val_selected"),
        ]:
            lt, ls, val_metric = select_best_two_lambdas(
                val_components, val_gt, val_ds.meta, "liao_time_lr", sky_key, lambda_time_grid, lambda_sky_grid
            )
            fused = test_components["waveform"] + lt * test_components["liao_time_lr"] + ls * test_components[sky_key]
            np.fill_diagonal(fused, -np.inf)
            add_rows(rows, detector, mode, "stage3_liao_time_plus_observed_sky", variant_name, evaluate_score(fused, test_gt, test_ds.meta), diag, {
                "lambda_liao_time": lt,
                "lambda_observed_sky": ls,
                "val_selected_r@10": val_metric["overall"]["r@10"],
            })
        pd.DataFrame(rows).to_csv(out_dir / "stage3_liao_time_plus_observed_sky_partial.csv", index=False)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "stage3_liao_time_plus_observed_sky_summary.csv", index=False)
    pd.DataFrame(diag_rows).to_csv(out_dir / "stage3_prior_sky_diagnostics.csv", index=False)
    write_stage_doc("stage3_liao_time_plus_observed_sky", df, out_dir, notes=[
        "Stage3 只融合 Liao time-delay LR 与 observed sky posterior，不使用 SNR ratio、不使用候选图。",
        "本阶段用于回答：Liao 时间先验与 observed sky 是否互补。",
        "分别测试 observed sky step 与 observed sky gaussian/log-overlap 两种空间特征。",
        "lambda_time 与 lambda_sky 均在 validation full catalog 上网格选择，然后应用到 test full catalog。",
    ], prior_csv=out_dir / "stage3_prior_sky_diagnostics.csv")
    return df


def select_best_multi_lambdas(
    val_components: dict[str, np.ndarray],
    val_gt: np.ndarray,
    val_meta: list[dict],
    prior_keys: list[str],
    grid: list[float],
) -> tuple[dict[str, float], dict]:
    best_lams = {key: 0.0 for key in prior_keys}
    best_metrics = evaluate_score(val_components["waveform"], val_gt, val_meta)

    def metric_key(metrics: dict) -> tuple[float, float, float, float]:
        return (
            metrics["overall"]["r@10"],
            metrics["overall"]["r@5"],
            metrics["overall"]["r@1"],
            metrics["overall"]["top_1pct"],
        )

    # Greedy coordinate selection keeps Stage4/5 practical on full-catalog matrices.
    # It changes only the lambda search strategy, not the tested feature set.
    for _ in range(2):
        improved = False
        for prior_key in prior_keys:
            current_best_lam = best_lams[prior_key]
            current_best_metrics = best_metrics
            current_best_key = metric_key(best_metrics)
            for lam in grid:
                trial_lams = dict(best_lams)
                trial_lams[prior_key] = lam
                score = val_components["waveform"].copy()
                for key, value in trial_lams.items():
                    if value != 0.0:
                        score = score + value * val_components[key]
                np.fill_diagonal(score, -np.inf)
                metrics = evaluate_score(score, val_gt, val_meta)
                key_tuple = metric_key(metrics)
                if key_tuple > current_best_key:
                    current_best_key = key_tuple
                    current_best_lam = lam
                    current_best_metrics = metrics
            if current_best_lam != best_lams[prior_key]:
                improved = True
            best_lams[prior_key] = current_best_lam
            best_metrics = current_best_metrics
        if not improved:
            break
    return best_lams, best_metrics


def stage4_snr_amplitude_prior() -> pd.DataFrame:
    out_dir = OUT_ROOT / "stage4_snr_amplitude_prior"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    diag_rows = []
    grid = [0.25, 0.5, 1.0, 2.0, 4.0]
    for detector, mode in JOBS:
        print("STAGE4_SNR_AMPLITUDE_PRIOR", detector, mode, flush=True)
        loaded = load_job(detector, mode)
        print("STAGE4_LOADED", detector, mode, flush=True)
        cfg = loaded["cfg"]
        val_ds, val_raw, val_time, val_gt, val_scores = loaded["val"]
        test_ds, test_raw, test_time, test_gt, test_scores = loaded["test"]
        time_prior = fit_time_lr_from_liao(detector, val_time, val_gt)
        print("STAGE4_TIME_PRIOR", detector, mode, flush=True)
        amp_prior = fit_amp_time_lr_from_liao(detector, val_time, val_gt)
        print("STAGE4_AMP_PRIOR", detector, mode, flush=True)
        val_sky = make_observed_sky(detector, val_raw, val_time, seed=501000 + (0 if detector == "ET" else 1))
        test_sky = make_observed_sky(detector, test_raw, test_time, seed=502000 + (0 if detector == "ET" else 1))
        _, _, val_log = observed_sky_score_matrices(val_sky)
        _, _, test_log = observed_sky_score_matrices(test_sky)
        print("STAGE4_SKY_MATRICES", detector, mode, flush=True)

        val_components = {
            "waveform": row_z(val_scores),
            "liao_time_lr": row_z(time_lr_score_matrix(val_time, time_prior)),
            "sky_log_overlap": row_z(val_log),
            "raw_snr_ratio": row_z(raw_snr_ratio_score_matrix(val_time)),
            "amp_time_lr": row_z(amp_time_lr_score_matrix(val_time, amp_prior)),
        }
        test_components = {
            "waveform": row_z(test_scores),
            "liao_time_lr": row_z(time_lr_score_matrix(test_time, time_prior)),
            "sky_log_overlap": row_z(test_log),
            "raw_snr_ratio": row_z(raw_snr_ratio_score_matrix(test_time)),
            "amp_time_lr": row_z(amp_time_lr_score_matrix(test_time, amp_prior)),
        }
        print("STAGE4_COMPONENTS", detector, mode, flush=True)
        diag_rows.append({k: v for k, v in amp_prior.items() if not isinstance(v, np.ndarray)})
        diag = base_diag(test_ds, cfg, {
            "liao_label": time_prior["liao_label"],
            "observed_sky_label": OBSERVED_SKY_CONFIG[detector]["label"],
            "amp_prior_pair_count": amp_prior["liao_pair_count"],
            "liao_snr_ratio_median": amp_prior["liao_snr_ratio_median"],
            "liao_snr_ratio_p90": amp_prior["liao_snr_ratio_p90"],
            "random_snr_ratio_median": amp_prior["random_snr_ratio_median"],
        })
        variants = [
            ("waveform_plus_time_lr_plus_sky_log_overlap", ["liao_time_lr", "sky_log_overlap"]),
            ("plus_raw_snr_ratio", ["liao_time_lr", "sky_log_overlap", "raw_snr_ratio"]),
            ("plus_amp_time_2d_lr", ["liao_time_lr", "sky_log_overlap", "amp_time_lr"]),
        ]
        for variant_name, keys in variants:
            print("STAGE4_SELECT", detector, variant_name, flush=True)
            lams, val_metric = select_best_multi_lambdas(val_components, val_gt, val_ds.meta, keys, grid)
            print("STAGE4_EVAL", detector, variant_name, lams, flush=True)
            score = test_components["waveform"].copy()
            for key in keys:
                if lams[key] != 0.0:
                    score = score + lams[key] * test_components[key]
            np.fill_diagonal(score, -np.inf)
            extra = {f"lambda_{key}": value for key, value in lams.items()}
            extra["val_selected_r@10"] = val_metric["overall"]["r@10"]
            add_rows(rows, detector, mode, "stage4_snr_amplitude_prior", variant_name, evaluate_score(score, test_gt, test_ds.meta), diag, extra)
        pd.DataFrame(rows).to_csv(out_dir / "stage4_snr_amplitude_prior_partial.csv", index=False)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "stage4_snr_amplitude_prior_summary.csv", index=False)
    pd.DataFrame(diag_rows).to_csv(out_dir / "amp_time_prior_diagnostics.csv", index=False)
    write_stage_doc("stage4_snr_amplitude_prior", df, out_dir, notes=[
        "Stage4 在 Stage3 的 waveform + Liao time LR + observed sky gaussian/log-overlap 基础上，只新增 SNR/amplitude 信息。",
        "A1 使用 raw SNR ratio baseline：两个事件 observed SNR 越接近，分数越高。",
        "A2 使用 GW-LMC/Liao 的二维 time-SNR ratio likelihood-ratio prior，即 p(delta_t, R_snr|lensed) / p(delta_t, R_snr|random)。",
        "所有 lambda 都在 validation full catalog 上选择，然后应用到 test full catalog。",
    ], prior_csv=out_dir / "amp_time_prior_diagnostics.csv")
    return df


def valid_pair_sample(
    components: dict[str, np.ndarray],
    gt: np.ndarray,
    feature_keys: list[str],
    seed: int,
    negatives_per_positive: int = 30,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    valid = hard.valid_queries(gt)
    rows = np.where(valid)[0]
    x_parts = []
    y_parts = []
    n = len(gt)
    waveform = components["waveform"]
    for i in rows:
        pos = int(gt[i])
        cand = [pos]
        hard_order = np.argsort(-waveform[i])[:negatives_per_positive]
        negs = [int(j) for j in hard_order if int(j) != i and int(j) != pos]
        while len(negs) < negatives_per_positive:
            j = int(rng.integers(0, n))
            if j != i and j != pos:
                negs.append(j)
        cand.extend(negs[:negatives_per_positive])
        feats = np.column_stack([components[key][i, cand] for key in feature_keys])
        feats = np.nan_to_num(feats, nan=0.0, posinf=10.0, neginf=-10.0)
        x_parts.append(feats)
        y_parts.append(np.asarray([1] + [0] * negatives_per_positive, dtype=np.int8))
    return np.vstack(x_parts).astype(np.float32), np.concatenate(y_parts)


def model_score_matrix(model, components: dict[str, np.ndarray], feature_keys: list[str]) -> np.ndarray:
    n = components[feature_keys[0]].shape[0]
    out = np.empty((n, n), dtype=np.float32)
    for start in range(0, n, CHUNK_ROWS):
        rows = slice(start, min(start + CHUNK_ROWS, n))
        feats = np.stack([components[key][rows].reshape(-1) for key in feature_keys], axis=1).astype(np.float32)
        feats = np.nan_to_num(feats, nan=0.0, posinf=10.0, neginf=-10.0)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="X does not have valid feature names.*")
            pred = model.predict_proba(feats)[:, 1].reshape(rows.stop - rows.start, n)
        out[rows] = pred.astype(np.float32)
    np.fill_diagonal(out, -np.inf)
    return out


def maybe_lightgbm_model():
    try:
        from lightgbm import LGBMClassifier
    except Exception:
        return None
    return LGBMClassifier(
        n_estimators=180,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        class_weight="balanced",
        random_state=20260612,
        n_jobs=4,
        verbose=-1,
    )


def load_stage4_lambdas(detector: str) -> dict[str, float]:
    summary = OUT_ROOT / "stage4_snr_amplitude_prior" / "stage4_snr_amplitude_prior_summary.csv"
    defaults = {
        "liao_time_lr": 1.0 if detector in {"ET", "ET3"} else 2.0,
        "sky_log_overlap": 4.0,
        "amp_time_lr": 0.0 if detector in {"ET", "ET3"} else 0.25,
        "raw_snr_ratio": 0.0,
    }
    if not summary.exists():
        return defaults
    df = pd.read_csv(summary)
    sub = df[(df["detector"] == detector) & (df["subset"] == "overall") & (df["variant"] == "plus_amp_time_2d_lr")]
    if sub.empty:
        return defaults
    row = sub.iloc[0]
    return {
        "liao_time_lr": float(row.get("lambda_liao_time_lr", defaults["liao_time_lr"])),
        "sky_log_overlap": float(row.get("lambda_sky_log_overlap", defaults["sky_log_overlap"])),
        "amp_time_lr": float(row.get("lambda_amp_time_lr", defaults["amp_time_lr"])),
        "raw_snr_ratio": 0.0,
    }


def stage5_reranker_model_compare() -> pd.DataFrame:
    out_dir = OUT_ROOT / "stage5_reranker_model_compare"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    feature_keys = [
        "waveform",
        "waveform_reciprocal_rank",
        "waveform_margin",
        "liao_time_lr",
        "sky_norm_sep",
        "sky_log_overlap",
        "amp_time_lr",
    ]
    weighted_prior_keys = ["waveform_reciprocal_rank", "waveform_margin", "liao_time_lr", "sky_norm_sep", "sky_log_overlap", "amp_time_lr"]
    weight_grid = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]
    for detector, mode in JOBS:
        print("STAGE5_RERANKER_MODEL_COMPARE", detector, mode, flush=True)
        loaded = load_job(detector, mode)
        cfg = loaded["cfg"]
        val_ds, val_raw, val_time, val_gt, val_scores = loaded["val"]
        test_ds, test_raw, test_time, test_gt, test_scores = loaded["test"]
        time_prior = fit_time_lr_from_liao(detector, val_time, val_gt)
        amp_prior = fit_amp_time_lr_from_liao(detector, val_time, val_gt)
        val_sky = make_observed_sky(detector, val_raw, val_time, seed=601000 + (0 if detector == "ET" else 1))
        test_sky = make_observed_sky(detector, test_raw, test_time, seed=602000 + (0 if detector == "ET" else 1))
        val_sky_features = observed_sky_pair_features(val_sky, chunk_rows=CHUNK_ROWS)
        test_sky_features = observed_sky_pair_features(test_sky, chunk_rows=CHUNK_ROWS)
        val_rank_features = rank_feature_matrices(val_scores, chunk_rows=CHUNK_ROWS)
        test_rank_features = rank_feature_matrices(test_scores, chunk_rows=CHUNK_ROWS)
        val_components = {
            "waveform": row_z(val_scores),
            "waveform_reciprocal_rank": row_z(val_rank_features["waveform_reciprocal_rank"]),
            "waveform_margin": row_z(val_rank_features["waveform_margin"]),
            "liao_time_lr": row_z(time_lr_score_matrix(val_time, time_prior)),
            "sky_norm_sep": row_z(-val_sky_features["sky_norm_sep"]),
            "sky_log_overlap": row_z(val_sky_features["sky_log_overlap"]),
            "amp_time_lr": row_z(amp_time_lr_score_matrix(val_time, amp_prior)),
            "raw_snr_ratio": row_z(raw_snr_ratio_score_matrix(val_time)),
        }
        test_components = {
            "waveform": row_z(test_scores),
            "waveform_reciprocal_rank": row_z(test_rank_features["waveform_reciprocal_rank"]),
            "waveform_margin": row_z(test_rank_features["waveform_margin"]),
            "liao_time_lr": row_z(time_lr_score_matrix(test_time, time_prior)),
            "sky_norm_sep": row_z(-test_sky_features["sky_norm_sep"]),
            "sky_log_overlap": row_z(test_sky_features["sky_log_overlap"]),
            "amp_time_lr": row_z(amp_time_lr_score_matrix(test_time, amp_prior)),
            "raw_snr_ratio": row_z(raw_snr_ratio_score_matrix(test_time)),
        }
        diag = base_diag(test_ds, cfg, {"feature_set": "+".join(feature_keys)})
        lams = load_stage4_lambdas(detector)
        weighted = test_components["waveform"].copy()
        for key in ["liao_time_lr", "sky_log_overlap", "amp_time_lr", "raw_snr_ratio"]:
            if lams[key] != 0.0:
                weighted = weighted + lams[key] * test_components[key]
        np.fill_diagonal(weighted, -np.inf)
        add_rows(rows, detector, mode, "stage5_reranker_model_compare", "weighted_sum_stage4_lambdas", evaluate_score(weighted, test_gt, test_ds.meta), diag, {f"lambda_{k}": v for k, v in lams.items()})
        selected_lams, val_metric = select_best_weighted_lambdas(
            val_components,
            weighted_prior_keys,
            weight_grid,
            lambda score: evaluate_score(score, val_gt, val_ds.meta),
            base_key="waveform",
            max_joint_keys=3,
        )
        selected_weighted = test_components["waveform"].copy()
        for key, value in selected_lams.items():
            if value != 0.0:
                selected_weighted = selected_weighted + value * test_components[key]
        np.fill_diagonal(selected_weighted, -np.inf)
        add_rows(rows, detector, mode, "stage5_reranker_model_compare", "weighted_sum_val_selected_extensible", evaluate_score(selected_weighted, test_gt, test_ds.meta), diag, {
            **{f"lambda_{k}": v for k, v in selected_lams.items()},
            "val_selected_r@10": val_metric["overall"]["r@10"],
        })

        x_train, y_train = valid_pair_sample(val_components, val_gt, feature_keys, seed=7000 + (0 if detector == "ET" else 1))
        models = {
            "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs")),
            "hgb": HistGradientBoostingClassifier(max_iter=140, learning_rate=0.06, max_leaf_nodes=31, l2_regularization=0.01, random_state=20260612),
            "mlp_tabular": make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-4, learning_rate_init=1e-3, max_iter=220, early_stopping=True, random_state=20260612)),
        }
        lgbm = maybe_lightgbm_model()
        if lgbm is not None:
            models["lightgbm"] = lgbm
        for name, model in models.items():
            print("FIT", detector, name, x_train.shape, flush=True)
            t0 = time.perf_counter()
            model.fit(x_train, y_train)
            score = model_score_matrix(model, test_components, feature_keys)
            add_rows(rows, detector, mode, "stage5_reranker_model_compare", name, evaluate_score(score, test_gt, test_ds.meta), diag, {
                "train_pairs": int(len(y_train)),
                "train_positive": int(y_train.sum()),
                "fit_predict_elapsed_s": float(time.perf_counter() - t0),
            })
        pd.DataFrame(rows).to_csv(out_dir / "stage5_reranker_model_compare_partial.csv", index=False)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "stage5_reranker_model_compare_summary.csv", index=False)
    write_stage_doc("stage5_reranker_model_compare", df, out_dir, notes=[
        "Stage5 固定同一组可扩展 pair features，只比较 rerank 模型。",
        "特征为 waveform、waveform reciprocal rank、waveform margin、Liao time LR、sky norm sep、observed sky log-overlap、amp-time 2D LR。",
        "weighted_sum_val_selected_extensible 使用统一辅助参数模块，并在 validation full catalog 上自动搜索加权修正。",
        "模型比较默认包括 weighted-sum、logistic regression、HGB、MLP，以及环境中可用的 LightGBM；RandomForest/ExtraTrees 不作为 full-catalog 默认项，避免 O(N^2) 推理成本过高。",
        "监督 reranker 在 validation catalog 上用正样本加 hard negatives 训练，然后在 test full catalog 上逐块推理。",
        "该阶段用于回答：固定物理特征后，线性/非线性表格 reranker 是否优于可解释 weighted-sum。",
    ])
    return df


class DSU:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def graph_metrics_from_score(score: np.ndarray, gt: np.ndarray, meta: list[dict], topk: int) -> dict:
    n = len(gt)
    valid = hard.valid_queries(gt)
    dsu = DSU(n)
    for i in np.where(valid)[0]:
        order = np.argsort(-score[i])[:topk]
        for j in order:
            if int(j) != int(i):
                dsu.union(int(i), int(j))
    comps: dict[int, list[int]] = {}
    for i in range(n):
        comps.setdefault(dsu.find(i), []).append(i)
    true_pairs = set()
    for i in np.where(valid)[0]:
        j = int(gt[i])
        if i < j:
            true_pairs.add((int(i), j))
    recovered = 0
    for i, j in true_pairs:
        if dsu.find(i) == dsu.find(j):
            recovered += 1
    pred_components = [nodes for nodes in comps.values() if len(nodes) > 1]
    hit_components = 0
    purities = []
    for nodes in pred_components:
        node_set = set(nodes)
        pair_hits = sum(1 for i, j in true_pairs if i in node_set and j in node_set)
        if pair_hits:
            hit_components += 1
        lensed_nodes = sum(1 for idx in nodes if meta[idx].get("tag") in {"L1", "L2"})
        purities.append(lensed_nodes / max(len(nodes), 1))
    return {
        "topk_edges": int(topk),
        "true_systems": int(len(true_pairs)),
        "pred_components": int(len(pred_components)),
        "system_recall": float(recovered / max(len(true_pairs), 1)),
        "system_precision": float(hit_components / max(len(pred_components), 1)),
        "mean_component_purity": float(np.mean(purities) if purities else 0.0),
        "mean_component_size": float(np.mean([len(x) for x in pred_components]) if pred_components else 0.0),
    }


def stage6_catalog_graph_discovery() -> pd.DataFrame:
    out_dir = OUT_ROOT / "stage6_catalog_graph_discovery"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    feature_keys = ["waveform", "liao_time_lr", "sky_log_overlap", "amp_time_lr", "raw_snr_ratio"]
    for detector, mode in JOBS:
        print("STAGE6_CATALOG_GRAPH_DISCOVERY", detector, mode, flush=True)
        loaded = load_job(detector, mode)
        cfg = loaded["cfg"]
        val_ds, val_raw, val_time, val_gt, val_scores = loaded["val"]
        test_ds, test_raw, test_time, test_gt, test_scores = loaded["test"]
        time_prior = fit_time_lr_from_liao(detector, val_time, val_gt)
        amp_prior = fit_amp_time_lr_from_liao(detector, val_time, val_gt)
        val_sky = make_observed_sky(detector, val_raw, val_time, seed=801000 + (0 if detector == "ET" else 1))
        test_sky = make_observed_sky(detector, test_raw, test_time, seed=802000 + (0 if detector == "ET" else 1))
        _, _, val_log = observed_sky_score_matrices(val_sky)
        _, _, test_log = observed_sky_score_matrices(test_sky)
        val_components = {
            "waveform": row_z(val_scores),
            "liao_time_lr": row_z(time_lr_score_matrix(val_time, time_prior)),
            "sky_log_overlap": row_z(val_log),
            "amp_time_lr": row_z(amp_time_lr_score_matrix(val_time, amp_prior)),
            "raw_snr_ratio": row_z(raw_snr_ratio_score_matrix(val_time)),
        }
        test_components = {
            "waveform": row_z(test_scores),
            "liao_time_lr": row_z(time_lr_score_matrix(test_time, time_prior)),
            "sky_log_overlap": row_z(test_log),
            "amp_time_lr": row_z(amp_time_lr_score_matrix(test_time, amp_prior)),
            "raw_snr_ratio": row_z(raw_snr_ratio_score_matrix(test_time)),
        }
        lams = load_stage4_lambdas(detector)
        weighted = test_components["waveform"].copy()
        for key in feature_keys[1:]:
            if lams[key] != 0.0:
                weighted = weighted + lams[key] * test_components[key]
        np.fill_diagonal(weighted, -np.inf)
        scores = {"waveform_only": test_components["waveform"], "weighted_sum_best_features": weighted}
        x_train, y_train = valid_pair_sample(val_components, val_gt, feature_keys, seed=9000 + (0 if detector == "ET" else 1))
        hgb = HistGradientBoostingClassifier(max_iter=140, learning_rate=0.06, max_leaf_nodes=31, l2_regularization=0.01, random_state=20260612)
        hgb.fit(x_train, y_train)
        scores["hgb"] = model_score_matrix(hgb, test_components, feature_keys)
        for name, score in scores.items():
            for topk in [1, 2, 5]:
                gm = graph_metrics_from_score(score, test_gt, test_ds.meta, topk)
                rows.append({
                    "detector": detector,
                    "data_mode": mode,
                    "stage": "stage6_catalog_graph_discovery",
                    "pair_scorer": name,
                    **base_diag(test_ds, cfg),
                    **gm,
                })
        pd.DataFrame(rows).to_csv(out_dir / "stage6_catalog_graph_discovery_partial.csv", index=False)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "stage6_catalog_graph_discovery_summary.csv", index=False)
    write_graph_doc("stage6_catalog_graph_discovery", df, out_dir)
    return df


def write_stage_doc(stage: str, df: pd.DataFrame, out_dir: Path, notes: list[str], prior_csv: Path | None = None) -> None:
    DOC_ROOT.mkdir(parents=True, exist_ok=True)
    doc = DOC_ROOT / f"{stage}_report_20260612_cn.md"
    parts = [f"# {stage} 实验报告", "", "生成时间：2026-06-12", ""]
    parts += ["## 输出位置", ""]
    parts.append(md_table([
        {"项目": "结果目录", "路径": f"`{out_dir}`"},
        {"项目": "结果 CSV", "路径": f"`{next(out_dir.glob('*summary.csv'))}`"},
        *([{"项目": "prior 诊断 CSV", "路径": f"`{prior_csv}`"}] if prior_csv else []),
    ], ["项目", "路径"]))
    parts += ["", "## 实验说明", ""]
    parts.extend([f"- {note}" for note in notes])
    parts += ["", "## Overall 结果", ""]
    rows = []
    for _, r in df[df["subset"] == "overall"].iterrows():
        rows.append({
            "detector": r.detector,
            "variant": r.variant,
            "R@1": fmt(r["r@1"]),
            "R@5": fmt(r["r@5"]),
            "R@10": fmt(r["r@10"]),
            "Top1%": fmt(r["top_1pct"]),
            "Top5%": fmt(r["top_5pct"]),
            "Top10%": fmt(r["top_10pct"]),
            "Median rank": fmt(r["median_true_rank"]),
            "lambda": fmt(r.get("lambda_time", r.get("lambda_liao_time", np.nan))),
        })
    parts.append(md_table(rows, ["detector", "variant", "R@1", "R@5", "R@10", "Top1%", "Top5%", "Top10%", "Median rank", "lambda"]))
    parts += ["", "## SIS / PM 分解", ""]
    rows = []
    for _, r in df[df["subset"].isin(["SIS", "PM"])].iterrows():
        rows.append({
            "detector": r.detector,
            "subset": r.subset,
            "variant": r.variant,
            "R@1": fmt(r["r@1"]),
            "R@5": fmt(r["r@5"]),
            "R@10": fmt(r["r@10"]),
            "Top1%": fmt(r["top_1pct"]),
            "Median rank": fmt(r["median_true_rank"]),
        })
    parts.append(md_table(rows, ["detector", "subset", "variant", "R@1", "R@5", "R@10", "Top1%", "Median rank"]))
    if prior_csv:
        prior = pd.read_csv(prior_csv)
        if "liao_label" in prior.columns and "sky_label" in prior.columns:
            parts += ["", "## Liao time + observed sky 诊断", ""]
            prior_rows = []
            for _, r in prior.iterrows():
                prior_rows.append({
                    "detector": r.detector,
                    "liao_label": r.liao_label,
                    "liao_delay_count": fmt(r.liao_delay_count),
                    "liao_delay_median_days": fmt(r.liao_delay_median_days),
                    "liao_delay_p90_days": fmt(r.liao_delay_p90_days),
                    "sky_label": r.sky_label,
                    "a90_ref_deg2": fmt(r.a90_ref_deg2),
                    "test_a90_median_deg2": fmt(r.test_a90_median_deg2),
                    "test_a90_p90_deg2": fmt(r.test_a90_p90_deg2),
                })
            parts.append(md_table(prior_rows, ["detector", "liao_label", "liao_delay_count", "liao_delay_median_days", "liao_delay_p90_days", "sky_label", "a90_ref_deg2", "test_a90_median_deg2", "test_a90_p90_deg2"]))
        elif "liao_label" in prior.columns:
            parts += ["", "## Liao prior 诊断", ""]
            prior_rows = []
            for _, r in prior.iterrows():
                delay_count = r.get("liao_delay_count", r.get("liao_pair_count", np.nan))
                delay_median = r.get("liao_delay_median_days", np.nan)
                delay_p90 = r.get("liao_delay_p90_days", np.nan)
                random_delay_median = r.get("random_delay_median_days", np.nan)
                prior_rows.append({
                    "detector": r.detector,
                    "liao_label": r.liao_label,
                    "liao_delay_count": fmt(delay_count),
                    "liao_delay_median_days": fmt(delay_median),
                    "liao_delay_p90_days": fmt(delay_p90),
                    "random_delay_median_days": fmt(random_delay_median),
                })
            parts.append(md_table(prior_rows, ["detector", "liao_label", "liao_delay_count", "liao_delay_median_days", "liao_delay_p90_days", "random_delay_median_days"]))
        elif "sky_label" in prior.columns:
            parts += ["", "## Observed sky 诊断", ""]
            prior_rows = []
            for _, r in prior.iterrows():
                prior_rows.append({
                    "detector": r.detector,
                    "sky_label": r.sky_label,
                    "a90_ref_deg2": fmt(r.a90_ref_deg2),
                    "test_a90_median_deg2": fmt(r.test_a90_median_deg2),
                    "test_a90_p90_deg2": fmt(r.test_a90_p90_deg2),
                    "test_sigma_median_rad": fmt(r.test_sigma_median_rad),
                })
            parts.append(md_table(prior_rows, ["detector", "sky_label", "a90_ref_deg2", "test_a90_median_deg2", "test_a90_p90_deg2", "test_sigma_median_rad"]))
    doc.write_text("\n".join(parts), encoding="utf-8")


def write_graph_doc(stage: str, df: pd.DataFrame, out_dir: Path) -> None:
    DOC_ROOT.mkdir(parents=True, exist_ok=True)
    doc = DOC_ROOT / f"{stage}_report_20260612_cn.md"
    parts = [f"# {stage} 实验报告", "", "生成时间：2026-06-12", ""]
    parts += ["## 输出位置", ""]
    parts.append(md_table([
        {"项目": "结果目录", "路径": f"`{out_dir}`"},
        {"项目": "结果 CSV", "路径": f"`{out_dir / 'stage6_catalog_graph_discovery_summary.csv'}`"},
    ], ["项目", "路径"]))
    parts += ["", "## 实验说明", ""]
    parts += [
        "- Stage6 不再只看 pair-level R@K，而是把 pair scorer 生成的高分边转成 catalog graph。",
        "- 每个 query 连接 top-k candidate，形成无向图连通分量，用于近似 catalog-level lensed-system discovery。",
        "- 当前实现比较 waveform_only、weighted_sum_best_features 和 HGB 三种 pair scorer。",
        "- system_recall 表示真实双像是否落在同一连通分量；system_precision 表示预测连通分量中有多少包含真实双像。",
    ]
    parts += ["", "## Catalog Graph 结果", ""]
    rows = []
    for _, r in df.iterrows():
        rows.append({
            "detector": r.detector,
            "pair_scorer": r.pair_scorer,
            "topk_edges": fmt(r.topk_edges),
            "true_systems": fmt(r.true_systems),
            "pred_components": fmt(r.pred_components),
            "system_precision": fmt(r.system_precision),
            "system_recall": fmt(r.system_recall),
            "purity": fmt(r.mean_component_purity),
            "mean_size": fmt(r.mean_component_size),
        })
    parts.append(md_table(rows, ["detector", "pair_scorer", "topk_edges", "true_systems", "pred_components", "system_precision", "system_recall", "purity", "mean_size"]))
    doc.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=[
        "stage0_baseline",
        "stage1_liao_time_lr",
        "stage2_observed_sky",
        "stage3_liao_time_plus_observed_sky",
        "stage4_snr_amplitude_prior",
        "stage5_reranker_model_compare",
        "stage6_catalog_graph_discovery",
        "all_p0",
        "all_p1_p2",
    ], required=True)
    args = parser.parse_args()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    if args.stage in {"stage0_baseline", "all_p0"}:
        stage0_baseline()
    if args.stage in {"stage1_liao_time_lr", "all_p0"}:
        stage1_liao_time()
    if args.stage in {"stage2_observed_sky", "all_p0"}:
        stage2_observed_sky()
    if args.stage in {"stage3_liao_time_plus_observed_sky", "all_p0"}:
        stage3_liao_time_plus_observed_sky()
    if args.stage in {"stage4_snr_amplitude_prior", "all_p1_p2"}:
        stage4_snr_amplitude_prior()
    if args.stage in {"stage5_reranker_model_compare", "all_p1_p2"}:
        stage5_reranker_model_compare()
    if args.stage in {"stage6_catalog_graph_discovery", "all_p1_p2"}:
        stage6_catalog_graph_discovery()
    (OUT_ROOT / f"{args.stage}_protocol_summary.json").write_text(json.dumps({
        "stage": args.stage,
        "elapsed_s": float(time.perf_counter() - t0),
        "jobs": JOBS,
        "note": "Staged realistic rerank. Each stage changes one factor for ablation comparison.",
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
