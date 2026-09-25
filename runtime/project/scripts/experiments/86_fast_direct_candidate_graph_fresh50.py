from __future__ import annotations

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

OUT_ROOT = Path("runs/fast_direct_candidate_graph_fresh50_20260612")
ENCODER_ROOT = Path("runs/fresh50_full_catalog_ranking_20260611/fresh_mixed_encoders")
JOBS = [("ET", "noisy"), ("LIGO", "noisy")]
FAMILIES = ["SIS", "PM"]
EPS = 1e-8

FUSION_WEIGHTS = [
    ("waveform_only", (1.0, 0.0, 0.0)),
    ("time_only", (0.0, 1.0, 0.0)),
    ("waveform_time_0p5", (1.0, 0.5, 0.0)),
    ("waveform_time_1p0", (1.0, 1.0, 0.0)),
    ("waveform_time_2p0", (1.0, 2.0, 0.0)),
    ("waveform_time_predsky_0p5", (1.0, 1.0, 0.5)),
    ("waveform_time_predsky_1p0", (1.0, 1.0, 1.0)),
]

GRAPH_PARAMS = [
    ("no_graph", (0.0, 0.0, 0.0, 0.0)),
    ("reciprocal_0p5", (0.5, 0.0, 0.0, 0.0)),
    ("mutual50_0p5", (0.5, 0.5, 0.0, 0.0)),
    ("mutual50_100", (0.5, 0.5, 0.25, 0.0)),
    ("hub_0p5", (0.5, 0.5, 0.25, 0.5)),
]


def family_by_index(meta: list[dict], rows: np.ndarray) -> np.ndarray:
    return np.asarray([meta[int(row)]["family"] for row in rows])


def row_z(x: np.ndarray) -> np.ndarray:
    out = x.astype(np.float32).copy()
    np.fill_diagonal(out, np.nan)
    mu = np.nanmean(out, axis=1, keepdims=True)
    sd = np.nanstd(out, axis=1, keepdims=True)
    out = (out - mu) / np.maximum(sd, EPS)
    np.fill_diagonal(out, -np.inf)
    return out.astype(np.float32)


def global_z(x: np.ndarray) -> np.ndarray:
    finite = np.isfinite(x)
    vals = x[finite]
    if len(vals) == 0:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - float(np.mean(vals))) / max(float(np.std(vals)), EPS)).astype(np.float32)


def rank_matrix(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(-scores, axis=1)
    ranks = np.empty(scores.shape, dtype=np.uint16)
    ranks[np.arange(scores.shape[0])[:, None], order] = np.arange(1, scores.shape[1] + 1, dtype=np.uint16)
    return ranks


def metrics(score: np.ndarray, gt: np.ndarray, meta: list[dict]) -> dict[str, dict]:
    rows, ranks = fresh.ranks_from_score_matrix(score, gt)
    return fresh.metrics_from_ranks(ranks, family_by_index(meta, rows), len(gt))


def load_split(detector: str, mode: str):
    encoder_dir = ENCODER_ROOT / f"{detector.lower()}_{mode}_mixed_sis_pm_ep50"
    cfg = base.make_cfg(detector, mode, encoder_dir)
    model_path = cfg.out_dir / "model.pt"
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    arrays = {family: base.FamilyArrays(family, base.ROOTS[(family, detector)], mode) for family in FAMILIES}
    splits = {}
    for index, family in enumerate(FAMILIES):
        splits[family] = base.split_indices(len(arrays[family].l1), cfg.seed + index)
        splits[f"{family}_U"] = base.split_indices(len(arrays[family].unlensed), cfg.seed + 100 + index)
    model = base.build_model(cfg)
    ckpt = torch.load(model_path, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)
    train_ds, train_raw, _, _, train_emb, _ = base.split_pack(detector, "train", cfg, arrays, splits, model)
    val_ds, val_raw, val_time, val_gt, val_emb, val_scores = base.split_pack(detector, "val", cfg, arrays, splits, model)
    test_ds, test_raw, test_time, test_gt, test_emb, test_scores = base.split_pack(detector, "test", cfg, arrays, splits, model)
    sky_model, sky_sigma, sky_mean_err, sky_med_err = base.fit_sky_predictor(train_raw, train_emb, val_raw, val_emb)
    return cfg, {
        "val": (val_ds, val_raw, val_time, val_gt, val_scores, base.normalize_vectors(sky_model.predict(val_emb))),
        "test": (test_ds, test_raw, test_time, test_gt, test_scores, base.normalize_vectors(sky_model.predict(test_emb))),
        "sky_sigma": sky_sigma,
        "sky_mean_err": sky_mean_err,
        "sky_med_err": sky_med_err,
    }


def component_scores(raw: pd.DataFrame, time_obs: pd.DataFrame, scores: np.ndarray, sky_mu: np.ndarray, sky_sigma: float) -> dict[str, np.ndarray]:
    waveform = scores.astype(np.float32).copy()
    np.fill_diagonal(waveform, -np.inf)
    time_score = base.direct_scores("time_only", raw, time_obs, scores, sky_mu, sky_sigma)
    sky_score = base.direct_scores("predicted_sky_overlap_only", raw, time_obs, scores, sky_mu, sky_sigma)
    return {
        "waveform": row_z(waveform),
        "time": row_z(time_score),
        "predsky": row_z(sky_score),
    }


def fused_score(components: dict[str, np.ndarray], weights: tuple[float, float, float]) -> np.ndarray:
    ww, wt, ws = weights
    score = np.zeros_like(components["waveform"], dtype=np.float32)
    if ww:
        score += ww * components["waveform"]
    if wt:
        score += wt * components["time"]
    if ws:
        score += ws * components["predsky"]
    np.fill_diagonal(score, -np.inf)
    return score.astype(np.float32)


def graph_score(base_score: np.ndarray, params: tuple[float, float, float, float]) -> np.ndarray:
    alpha, gamma50, gamma100, beta = params
    if alpha == gamma50 == gamma100 == beta == 0.0:
        return base_score.astype(np.float32).copy()
    ranks = rank_matrix(base_score)
    reciprocal = (1.0 / np.maximum(ranks.astype(np.float32), 1.0)) + (1.0 / np.maximum(ranks.T.astype(np.float32), 1.0))
    mutual50 = ((ranks <= 50) & (ranks.T <= 50)).astype(np.float32)
    mutual100 = ((ranks <= 100) & (ranks.T <= 100)).astype(np.float32)
    indeg50 = np.sum(ranks <= 50, axis=0).astype(np.float32)
    indeg100 = np.sum(ranks <= 100, axis=0).astype(np.float32)
    hub = np.log1p(indeg50)[None, :] + 0.5 * np.log1p(indeg100)[None, :]
    out = (
        base_score
        + alpha * global_z(reciprocal)
        + gamma50 * mutual50
        + gamma100 * mutual100
        - beta * global_z(hub)
    ).astype(np.float32)
    np.fill_diagonal(out, -np.inf)
    return out


def add_rows(rows: list[dict], detector: str, mode: str, variant: str, stage: str, metric_rows: dict[str, dict], diag: dict, extra: dict):
    for subset, values in metric_rows.items():
        rows.append({
            "detector": detector,
            "data_mode": mode,
            "catalog": "fresh50_mixed_SIS_PM_unlensed_full_catalog",
            "candidate_kind": "full_catalog",
            "subset": subset,
            "variant": variant,
            "stage": stage,
            **diag,
            **values,
            **extra,
        })


def run_one(detector: str, mode: str) -> list[dict]:
    out_dir = OUT_ROOT / f"{detector.lower()}_{mode}_fast_graph"
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg, packed = load_split(detector, mode)
    val_ds, val_raw, val_time, val_gt, val_scores, val_sky_mu = packed["val"]
    test_ds, test_raw, test_time, test_gt, test_scores, test_sky_mu = packed["test"]
    sky_sigma = packed["sky_sigma"]
    diag = {
        "query_total": int(len(hard.valid_queries(test_gt))),
        **fresh.full_catalog_composition(test_ds.meta),
        "epochs": int(cfg.epochs),
        "backbone": cfg.backbone,
        "preprocess": cfg.preprocess,
        "sky_sigma_rad": sky_sigma,
        "sky_val_mean_angular_error_rad": packed["sky_mean_err"],
        "sky_val_median_angular_error_rad": packed["sky_med_err"],
    }
    print("COMPONENTS", detector, mode, "val", flush=True)
    val_components = component_scores(val_raw, val_time, val_scores, val_sky_mu, sky_sigma)
    print("COMPONENTS", detector, mode, "test", flush=True)
    test_components = component_scores(test_raw, test_time, test_scores, test_sky_mu, sky_sigma)
    rows = []
    fusion_candidates = []
    for fusion_name, weights in FUSION_WEIGHTS:
        print("VAL_FUSION", detector, mode, fusion_name, weights, flush=True)
        val_base = fused_score(val_components, weights)
        base_metrics = metrics(val_base, val_gt, val_ds.meta)
        fusion_candidates.append((base_metrics["overall"]["r@10"], base_metrics["overall"]["r@5"], base_metrics["overall"]["r@1"], fusion_name, weights, "no_graph", GRAPH_PARAMS[0][1], base_metrics))
    fusion_candidates.sort(reverse=True, key=lambda x: x[:3])
    candidates = fusion_candidates[:]
    for _, _, _, fusion_name, weights, _, _, _ in fusion_candidates[:2]:
        val_base = fused_score(val_components, weights)
        for graph_name, params in GRAPH_PARAMS[1:]:
            print("VAL_GRAPH", detector, mode, fusion_name, graph_name, params, flush=True)
            val_graph = graph_score(val_base, params)
            graph_metrics = metrics(val_graph, val_gt, val_ds.meta)
            candidates.append((graph_metrics["overall"]["r@10"], graph_metrics["overall"]["r@5"], graph_metrics["overall"]["r@1"], fusion_name, weights, graph_name, params, graph_metrics))
    candidates.sort(reverse=True, key=lambda x: x[:3])
    top = candidates[:8]
    for rank, (_, _, _, fusion_name, weights, graph_name, params, val_metrics) in enumerate(top, 1):
        print("EVAL_TEST", detector, mode, rank, fusion_name, graph_name, weights, params, flush=True)
        test_base = fused_score(test_components, weights)
        test_graph = graph_score(test_base, params)
        test_metrics = metrics(test_graph, test_gt, test_ds.meta)
        add_rows(rows, detector, mode, f"{fusion_name}__{graph_name}", "validation_selected_fast_candidate_graph", test_metrics, diag, {
            "selection_rank": rank,
            "w_waveform": weights[0],
            "w_time": weights[1],
            "w_predsky": weights[2],
            "graph_alpha": params[0],
            "graph_gamma50": params[1],
            "graph_gamma100": params[2],
            "graph_beta": params[3],
            "val_r@1": val_metrics["overall"]["r@1"],
            "val_r@5": val_metrics["overall"]["r@5"],
            "val_r@10": val_metrics["overall"]["r@10"],
            "val_top_1pct": val_metrics["overall"]["top_1pct"],
        })
    pd.DataFrame(rows).to_csv(out_dir / "fast_candidate_graph_summary.csv", index=False)
    return rows


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    rows = []
    for detector, mode in JOBS:
        print("RUN_FAST_GRAPH", detector, mode, flush=True)
        rows.extend(run_one(detector, mode))
        pd.DataFrame(rows).to_csv(OUT_ROOT / "fast_candidate_graph_summary_partial.csv", index=False)
    df = pd.DataFrame(rows)
    df.to_csv(OUT_ROOT / "fast_candidate_graph_summary.csv", index=False)
    (OUT_ROOT / "protocol_summary.json").write_text(json.dumps({
        "elapsed_s": float(time.perf_counter() - start),
        "jobs": JOBS,
        "note": "Fast validation-selected graph reranking using direct waveform/time/predicted-sky score fusion plus reciprocal-rank, mutual-topK, and hub-penalty terms.",
    }, indent=2), encoding="utf-8")
    print(df.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
