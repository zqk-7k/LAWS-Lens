from __future__ import annotations

import argparse
import ast
import importlib
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

base = importlib.import_module("scripts.experiments.80_mixed_sis_pm_catalog_modality_compare")
hard = importlib.import_module("scripts.experiments.81_time_matched_hard_negative_mixed_catalog")
fresh = importlib.import_module("scripts.experiments.84_fresh50_full_catalog_ranking")

OUT_ROOT = Path("runs/liao_realistic_staged_rerank_20260612")
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
    "LIGO": {
        "label": "GW-LMC 2.5PLUS BBH Any_Detected_SNR1",
        "image_csv": GW_LMC_ROOT / "2.5PLUS/BBH/Any_Detected_SNR1/BBH_2.5PLUS_Any_Detected_SNR1_ImageParams.csv",
        "snr_threshold": 1.0,
    },
}

OBSERVED_SKY_CONFIG = {
    "ET": {
        "label": "ET single-site baseline A90=300 deg2",
        "a90_ref_deg2": 300.0,
        "rho_ref": 12.0,
        "clip_min_deg2": 50.0,
        "clip_max_deg2": 2000.0,
        "lognormal_sigma": 0.35,
    },
    "LIGO": {
        "label": "LIGO/2.5G HL-like baseline A90=100 deg2",
        "a90_ref_deg2": 100.0,
        "rho_ref": 12.0,
        "clip_min_deg2": 10.0,
        "clip_max_deg2": 500.0,
        "lognormal_sigma": 0.35,
    },
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


def load_model_only(cfg):
    model_path = cfg.out_dir / "model.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing fresh50 model: {model_path}")
    model = base.build_model(cfg)
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
    model = load_model_only(cfg)
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


def make_observed_sky(detector: str, raw_obs: pd.DataFrame, time_obs: pd.DataFrame, seed: int) -> pd.DataFrame:
    cfg = OBSERVED_SKY_CONFIG[detector]
    rng = np.random.default_rng(seed)
    snr = time_obs["snr"].to_numpy(dtype=np.float64)
    jitter = rng.lognormal(mean=0.0, sigma=cfg["lognormal_sigma"], size=len(time_obs))
    a90 = cfg["a90_ref_deg2"] * (cfg["rho_ref"] / np.maximum(snr, 1.0)) ** 2 * jitter
    a90 = np.clip(a90, cfg["clip_min_deg2"], cfg["clip_max_deg2"])
    sigma = sky_sigma_from_a90_deg2(a90)
    true_vec = unit_from_radec(raw_obs["ra"].to_numpy(dtype=np.float64), raw_obs["dec"].to_numpy(dtype=np.float64))
    noise = rng.normal(size=true_vec.shape)
    noise -= np.sum(noise * true_vec, axis=1, keepdims=True) * true_vec
    noise_norm = noise / np.maximum(np.linalg.norm(noise, axis=1, keepdims=True), EPS)
    radial = rng.normal(loc=0.0, scale=sigma)
    obs_vec = true_vec * np.cos(radial)[:, None] + noise_norm * np.sin(radial)[:, None]
    obs_vec = obs_vec / np.maximum(np.linalg.norm(obs_vec, axis=1, keepdims=True), EPS)
    ra_obs, dec_obs = radec_from_unit(obs_vec)
    return pd.DataFrame({
        "ra_obs": ra_obs,
        "dec_obs": dec_obs,
        "sky_area90_deg2": a90,
        "sky_sigma_rad": sigma,
    })


def observed_sky_score_matrices(sky_obs: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vec = unit_from_radec(sky_obs["ra_obs"].to_numpy(dtype=np.float64), sky_obs["dec_obs"].to_numpy(dtype=np.float64))
    sigma = sky_obs["sky_sigma_rad"].to_numpy(dtype=np.float64)
    n = len(sky_obs)
    step = np.empty((n, n), dtype=np.float32)
    gauss_weight = np.empty((n, n), dtype=np.float32)
    log_overlap = np.empty((n, n), dtype=np.float32)
    for start in range(0, n, CHUNK_ROWS):
        rows = slice(start, min(start + CHUNK_ROWS, n))
        sep = angular_sep_unit(vec[rows, None, :], vec[None, :, :])
        sigma_ij = np.sqrt(sigma[rows, None] ** 2 + sigma[None, :] ** 2)
        d = sep / np.maximum(sigma_ij, EPS)
        st = np.full(d.shape, -0.5, dtype=np.float32)
        st[d <= 3.03] = 0.1
        st[d <= 2.15] = 0.5
        st[d <= 1.18] = 1.0
        var = np.maximum(sigma_ij ** 2, EPS)
        step[rows] = st
        gauss_weight[rows] = np.exp(-0.5 * d * d).astype(np.float32)
        log_overlap[rows] = (-np.log(2.0 * np.pi * var) - (sep * sep) / (2.0 * var)).astype(np.float32)
    np.fill_diagonal(step, -np.inf)
    np.fill_diagonal(gauss_weight, -np.inf)
    np.fill_diagonal(log_overlap, -np.inf)
    return step, gauss_weight, log_overlap


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
        prior_rows.append({k: v for k, v in prior.items() if not isinstance(v, np.ndarray)})
        val_components = {
            "waveform": row_z(val_scores),
            "liao_time_lr": row_z(time_lr_score_matrix(val_time, prior)),
        }
        test_components = {
            "waveform": row_z(test_scores),
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
        add_rows(rows, detector, mode, "stage1_liao_time_lr", "liao_time_lr_only", evaluate_score(test_components["liao_time_lr"], test_gt, test_ds.meta), diag)
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
        "Stage1 只新增 GW-LMC/Liao time-delay likelihood-ratio prior。",
        "本阶段不使用 observed sky，不使用 SNR ratio，不使用候选图项。",
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
        val_step, val_gauss, val_log = observed_sky_score_matrices(val_sky)
        test_step, test_gauss, test_log = observed_sky_score_matrices(test_sky)
        sky_diag_rows.append({
            "detector": detector,
            "sky_label": OBSERVED_SKY_CONFIG[detector]["label"],
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
        pd.DataFrame(rows).to_csv(out_dir / "stage2_observed_sky_partial.csv", index=False)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "stage2_observed_sky_summary.csv", index=False)
    pd.DataFrame(sky_diag_rows).to_csv(out_dir / "observed_sky_diagnostics.csv", index=False)
    write_stage_doc("stage2_observed_sky", df, out_dir, notes=[
        "Stage2 只新增 observed sky posterior，不使用 Liao time LR、不使用 SNR ratio、不使用候选图。",
        "true ra/dec 只用于模拟观测中心 ra_obs/dec_obs 和 sky_area90；rerank 输入只使用 observed sky posterior 特征。",
        "分别测试 observed sky step 和 observed sky gaussian/log-overlap。",
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
                prior_rows.append({
                    "detector": r.detector,
                    "liao_label": r.liao_label,
                    "liao_delay_count": fmt(r.liao_delay_count),
                    "liao_delay_median_days": fmt(r.liao_delay_median_days),
                    "liao_delay_p90_days": fmt(r.liao_delay_p90_days),
                    "random_delay_median_days": fmt(r.random_delay_median_days),
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=[
        "stage0_baseline",
        "stage1_liao_time_lr",
        "stage2_observed_sky",
        "stage3_liao_time_plus_observed_sky",
        "all_p0",
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
    (OUT_ROOT / f"{args.stage}_protocol_summary.json").write_text(json.dumps({
        "stage": args.stage,
        "elapsed_s": float(time.perf_counter() - t0),
        "jobs": JOBS,
        "note": "Staged realistic rerank. Each stage changes one factor for ablation comparison.",
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
