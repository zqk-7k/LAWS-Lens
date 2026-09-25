from __future__ import annotations

import importlib
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

base = importlib.import_module("scripts.experiments.80_mixed_sis_pm_catalog_modality_compare")
hard = importlib.import_module("scripts.experiments.81_time_matched_hard_negative_mixed_catalog")

OUT_ROOT = Path("runs/fresh50_full_catalog_ranking_20260611")
ENCODER_ROOT = OUT_ROOT / "fresh_mixed_encoders"
JOBS = [("ET", "pure"), ("ET", "noisy"), ("LIGO", "pure"), ("LIGO", "noisy")]
FAMILIES = ["SIS", "PM"]
DIRECT_VARIANTS = [
    "waveform_only",
    "time_only",
    "true_sky_overlap_only",
    "predicted_sky_overlap_only",
]
RERANK_VARIANTS = [
    "waveform_plus_time",
    "waveform_plus_predicted_sky_overlap",
    "waveform_plus_time_plus_predicted_sky_overlap",
    "waveform_plus_true_sky_overlap",
    "waveform_plus_time_plus_true_sky_overlap",
]
CHUNK_ROWS = 8


def family_by_index(meta: list[dict], rows: np.ndarray) -> np.ndarray:
    return np.asarray([meta[int(row)]["family"] for row in rows])


def full_catalog_composition(meta: list[dict]) -> dict:
    out = {"catalog_total": int(len(meta))}
    for family in FAMILIES:
        prefix = family.lower()
        out[f"{prefix}_lensed_images"] = int(sum(1 for item in meta if item["family"] == family and item["tag"] in {"L1", "L2"}))
        out[f"{prefix}_unlensed"] = int(sum(1 for item in meta if item["family"] == family and item["tag"] == "U"))
    out["total_lensed_images"] = int(out["sis_lensed_images"] + out["pm_lensed_images"])
    out["total_unlensed"] = int(out["sis_unlensed"] + out["pm_unlensed"])
    return out


def metrics_from_ranks(ranks: np.ndarray, query_family: np.ndarray, catalog_total: int) -> dict[str, dict]:
    def one(mask: np.ndarray) -> dict:
        selected = ranks[mask]
        if len(selected) == 0:
            return {}
        out = {
            "valid": int(len(selected)),
            "r@1": float(np.mean(selected <= 1)),
            "r@5": float(np.mean(selected <= 5)),
            "r@10": float(np.mean(selected <= 10)),
            "r@50": float(np.mean(selected <= 50)),
            "r@100": float(np.mean(selected <= 100)),
            "r@500": float(np.mean(selected <= 500)),
            "median_true_rank": float(np.median(selected)),
        }
        usable = max(catalog_total - 1, 1)
        for pct in (1, 5, 10):
            k = max(1, int(math.ceil(usable * pct / 100.0)))
            out[f"top_{pct}pct_k"] = int(k)
            out[f"top_{pct}pct"] = float(np.mean(selected <= k))
        return out

    overall = one(np.ones(len(ranks), dtype=bool))
    sis = one(query_family == "SIS")
    pm = one(query_family == "PM")
    macro = {"valid": int(overall.get("valid", 0))}
    for key in ["r@1", "r@5", "r@10", "r@50", "r@100", "r@500", "top_1pct", "top_5pct", "top_10pct", "median_true_rank"]:
        vals = [group[key] for group in (sis, pm) if key in group]
        if vals:
            macro[key] = float(np.mean(vals))
    for pct in (1, 5, 10):
        macro[f"top_{pct}pct_k"] = overall.get(f"top_{pct}pct_k", np.nan)
    return {"overall": overall, "SIS": sis, "PM": pm, "macro": macro}


def ranks_from_score_matrix(score_matrix: np.ndarray, gt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scores = score_matrix.copy()
    np.fill_diagonal(scores, -np.inf)
    rows = hard.valid_queries(gt)
    true_scores = scores[rows, gt[rows].astype(int)]
    ranks = 1 + np.sum(scores[rows] > true_scores[:, None], axis=1)
    return rows, ranks.astype(np.int32)


def direct_full_scores(
    variant: str,
    raw_obs: pd.DataFrame,
    time_obs: pd.DataFrame,
    scores: np.ndarray,
    sky_mu: np.ndarray,
    sky_sigma: float,
) -> np.ndarray:
    if variant == "waveform_only":
        out = scores.astype(np.float32).copy()
        np.fill_diagonal(out, -np.inf)
        return out
    return base.direct_scores(variant, raw_obs, time_obs, scores, sky_mu, sky_sigma)


def full_rerank_ranks(
    variant: str,
    clf,
    raw_obs: pd.DataFrame,
    time_obs: pd.DataFrame,
    sky_mu: np.ndarray,
    sky_sigma: float,
    scores: np.ndarray,
    ranks: np.ndarray,
    gt: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    query_rows = hard.valid_queries(gt)
    cols = np.arange(len(gt), dtype=np.int32)
    out_ranks = []
    for start in range(0, len(query_rows), CHUNK_ROWS):
        rows = query_rows[start:start + CHUNK_ROWS]
        a = np.repeat(rows, len(cols)).astype(np.int32)
        c = np.tile(cols, len(rows)).astype(np.int32)
        features = hard.rerank_features(variant, raw_obs, time_obs, sky_mu, sky_sigma, scores, ranks, a, c)
        pred = clf.predict_proba(features)[:, 1].reshape(len(rows), len(cols))
        pred[np.arange(len(rows)), rows] = -np.inf
        true_scores = pred[np.arange(len(rows)), gt[rows].astype(int)]
        out_ranks.extend((1 + np.sum(pred > true_scores[:, None], axis=1)).tolist())
    return query_rows, np.asarray(out_ranks, dtype=np.int32)


def add_group_rows(
    rows: list[dict],
    detector: str,
    mode: str,
    variant: str,
    stage: str,
    metrics: dict[str, dict],
    diag: dict,
    extra: dict | None = None,
) -> None:
    extra = extra or {}
    for subset, values in metrics.items():
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
    encoder_dir = ENCODER_ROOT / f"{detector.lower()}_{mode}_mixed_sis_pm_ep50"
    out_dir = OUT_ROOT / f"{detector.lower()}_{mode}_full_catalog"
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = base.make_cfg(detector, mode, encoder_dir)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    arrays = {family: base.FamilyArrays(family, base.ROOTS[(family, detector)], mode) for family in FAMILIES}
    splits = {}
    for index, family in enumerate(FAMILIES):
        splits[family] = base.split_indices(len(arrays[family].l1), cfg.seed + index)
        splits[f"{family}_U"] = base.split_indices(len(arrays[family].unlensed), cfg.seed + 100 + index)

    model, train_info = base.train_or_load_encoder(cfg, arrays, splits)
    train_ds, train_raw, _, _, train_emb, _ = base.split_pack(detector, "train", cfg, arrays, splits, model)
    val_ds, val_raw, val_time, val_gt, val_emb, val_scores = base.split_pack(detector, "val", cfg, arrays, splits, model)
    test_ds, test_raw, test_time, test_gt, test_emb, test_scores = base.split_pack(detector, "test", cfg, arrays, splits, model)
    val_ranks = base.row_ranks(val_scores)
    test_ranks = base.row_ranks(test_scores)
    sky_model, sky_sigma, sky_mean_err, sky_med_err = base.fit_sky_predictor(train_raw, train_emb, val_raw, val_emb)
    val_sky_mu = base.normalize_vectors(sky_model.predict(val_emb))
    test_sky_mu = base.normalize_vectors(sky_model.predict(test_emb))

    val_hard, val_hard_diag = hard.make_candidate_lists(val_time, val_ds.meta, val_gt, "hard", hard.N_NEG, seed=91001)
    diag = {
        "query_total": int(len(hard.valid_queries(test_gt))),
        **full_catalog_composition(test_ds.meta),
        "epochs": int(cfg.epochs),
        "backbone": cfg.backbone,
        "preprocess": cfg.preprocess,
        "train_s": train_info.get("train_s", np.nan),
        "mean_epoch_s": train_info.get("mean_epoch_s", np.nan),
        "sky_sigma_rad": sky_sigma,
        "sky_val_mean_angular_error_rad": sky_mean_err,
        "sky_val_median_angular_error_rad": sky_med_err,
        **{f"val_hard_{key}": value for key, value in val_hard_diag.items()},
    }
    all_rows = []
    for variant in DIRECT_VARIANTS:
        print("DIRECT_FRESH50_FULL", detector, mode, variant, flush=True)
        score_matrix = direct_full_scores(variant, test_raw, test_time, test_scores, test_sky_mu, sky_sigma)
        query_rows, metric_ranks = ranks_from_score_matrix(score_matrix, test_gt)
        query_family = family_by_index(test_ds.meta, query_rows)
        metrics = metrics_from_ranks(metric_ranks, query_family, len(test_gt))
        add_group_rows(all_rows, detector, mode, variant, "direct_full_catalog", metrics, diag)
        pd.DataFrame(all_rows).to_csv(out_dir / "fresh50_full_catalog_partial.csv", index=False)

    for index, variant in enumerate(RERANK_VARIANTS):
        print("RERANK_FRESH50_FULL", detector, mode, variant, flush=True)
        clf, auc, n_train, n_pos = hard.train_reranker(
            variant, val_raw, val_time, val_sky_mu, sky_sigma, val_scores, val_ranks, val_hard, seed=93000 + index
        )
        query_rows, metric_ranks = full_rerank_ranks(
            variant, clf, test_raw, test_time, test_sky_mu, sky_sigma, test_scores, test_ranks, test_gt
        )
        query_family = family_by_index(test_ds.meta, query_rows)
        metrics = metrics_from_ranks(metric_ranks, query_family, len(test_gt))
        add_group_rows(
            all_rows,
            detector,
            mode,
            variant,
            "hard_trained_full_catalog_rerank",
            metrics,
            diag,
            {"val_auc_hard_sampled": auc, "train_examples": n_train, "train_positive": n_pos},
        )
        pd.DataFrame(all_rows).to_csv(out_dir / "fresh50_full_catalog_partial.csv", index=False)
    pd.DataFrame(all_rows).to_csv(out_dir / "fresh50_full_catalog_summary.csv", index=False)
    return all_rows


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    all_rows = []
    for detector, mode in JOBS:
        print("RUN_FRESH50_FULL_CATALOG", detector, mode, flush=True)
        all_rows.extend(run_one(detector, mode))
        pd.DataFrame(all_rows).to_csv(OUT_ROOT / "fresh50_full_catalog_summary_partial.csv", index=False)
    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_ROOT / "fresh50_full_catalog_summary.csv", index=False)
    for metric in ["r@1", "r@5", "r@10", "top_1pct", "top_5pct", "top_10pct"]:
        df.pivot_table(
            index=["detector", "data_mode", "subset"],
            columns="variant",
            values=metric,
            aggfunc="first",
        ).to_csv(OUT_ROOT / f"{metric.replace('@', '')}_pivot.csv")
    (OUT_ROOT / "protocol_summary.json").write_text(json.dumps({
        "elapsed_s": float(time.perf_counter() - t0),
        "epochs": 50,
        "candidate_kind": "full_catalog",
        "note": "Fresh 50-epoch mixed encoders are trained in a new output root, then evaluated against full SIS+PM+unlensed catalogs.",
    }, indent=2), encoding="utf-8")
    print(df.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
