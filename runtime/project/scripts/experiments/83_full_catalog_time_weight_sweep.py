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

OUT_ROOT = Path("runs/full_catalog_time_weight_sweep_20260611")
BASE_OUT_ROOT = Path("runs/mixed_sis_pm_catalog_modality_compare_20260609")
JOBS = [("ET", "pure"), ("ET", "noisy"), ("LIGO", "pure"), ("LIGO", "noisy")]
FAMILIES = ["SIS", "PM"]
LAMBDAS = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0]
CAPS = [None, 1.0, 2.0, 3.0]
CHUNK_ROWS = 32
EPS = 1e-8


def full_catalog_composition(meta: list[dict]) -> dict:
    out = {"catalog_total": int(len(meta))}
    for family in FAMILIES:
        prefix = family.lower()
        out[f"{prefix}_lensed_images"] = int(sum(1 for item in meta if item["family"] == family and item["tag"] in {"L1", "L2"}))
        out[f"{prefix}_unlensed"] = int(sum(1 for item in meta if item["family"] == family and item["tag"] == "U"))
    out["total_lensed_images"] = int(out["sis_lensed_images"] + out["pm_lensed_images"])
    out["total_unlensed"] = int(out["sis_unlensed"] + out["pm_unlensed"])
    return out


def family_by_index(meta: list[dict], rows: np.ndarray) -> np.ndarray:
    return np.asarray([meta[int(row)]["family"] for row in rows])


def row_standardize(values: np.ndarray, self_cols: np.ndarray) -> np.ndarray:
    out = values.astype(np.float32, copy=True)
    finite = np.isfinite(out)
    finite[np.arange(len(self_cols)), self_cols] = False
    masked = np.where(finite, out, np.nan)
    mean = np.nanmean(masked, axis=1, keepdims=True)
    std = np.nanstd(masked, axis=1, keepdims=True)
    out = (out - mean) / np.maximum(std, EPS)
    out[np.arange(len(self_cols)), self_cols] = -np.inf
    return out.astype(np.float32)


def time_score_rows(time_obs: pd.DataFrame, rows: np.ndarray, n: int) -> np.ndarray:
    times = time_obs["trigger_time_obs"].to_numpy(dtype=np.float64)
    cols = np.arange(n, dtype=np.int32)
    dt = np.abs(times[rows][:, None] - times[cols][None, :])
    return (-np.log10(dt + 1.0)).astype(np.float32)


def ranks_for_lambda(
    scores: np.ndarray,
    time_obs: pd.DataFrame,
    gt: np.ndarray,
    query_rows: np.ndarray,
    lambda_time: float,
    cap: float | None,
) -> np.ndarray:
    out = []
    n = len(gt)
    for start in range(0, len(query_rows), CHUNK_ROWS):
        rows = query_rows[start:start + CHUNK_ROWS]
        wave_z = row_standardize(scores[rows], rows)
        time_z = row_standardize(time_score_rows(time_obs, rows, n), rows)
        if cap is not None:
            time_z = np.clip(time_z, -float(cap), float(cap))
            time_z[np.arange(len(rows)), rows] = -np.inf
        fused = wave_z + float(lambda_time) * time_z
        fused[np.arange(len(rows)), rows] = -np.inf
        true_scores = fused[np.arange(len(rows)), gt[rows].astype(int)]
        out.extend((1 + np.sum(fused > true_scores[:, None], axis=1)).tolist())
    return np.asarray(out, dtype=np.int32)


def ranks_for_time_only(time_obs: pd.DataFrame, gt: np.ndarray, query_rows: np.ndarray) -> np.ndarray:
    out = []
    n = len(gt)
    for start in range(0, len(query_rows), CHUNK_ROWS):
        rows = query_rows[start:start + CHUNK_ROWS]
        values = time_score_rows(time_obs, rows, n)
        values[np.arange(len(rows)), rows] = -np.inf
        true_scores = values[np.arange(len(rows)), gt[rows].astype(int)]
        out.extend((1 + np.sum(values > true_scores[:, None], axis=1)).tolist())
    return np.asarray(out, dtype=np.int32)


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


def add_rows(rows: list[dict], detector: str, mode: str, variant: str, metrics: dict[str, dict], diag: dict, params: dict) -> None:
    for subset, values in metrics.items():
        rows.append({
            "detector": detector,
            "data_mode": mode,
            "catalog": "mixed_SIS_PM_unlensed_full_catalog",
            "candidate_kind": "full_catalog",
            "subset": subset,
            "variant": variant,
            **diag,
            **params,
            **values,
        })


def run_one(detector: str, mode: str) -> list[dict]:
    base_dir = BASE_OUT_ROOT / f"{detector.lower()}_{mode}_mixed_sis_pm_ep50"
    out_dir = OUT_ROOT / f"{detector.lower()}_{mode}_lambda_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = base.make_cfg(detector, mode, base_dir)
    arrays = {family: base.FamilyArrays(family, base.ROOTS[(family, detector)], mode) for family in FAMILIES}
    splits = {}
    for index, family in enumerate(FAMILIES):
        splits[family] = base.split_indices(len(arrays[family].l1), cfg.seed + index)
        splits[f"{family}_U"] = base.split_indices(len(arrays[family].unlensed), cfg.seed + 100 + index)
    _, _, test_time, test_gt, _, _ = base.split_pack(detector, "test", cfg, arrays, splits, model=None)
    test_scores = np.load(base_dir / "test_scores.npy").astype(np.float32)
    test_ds = base.MixedEvaluationSet(arrays, splits, "test", cfg)
    query_rows = hard.valid_queries(test_gt)
    query_family = family_by_index(test_ds.meta, query_rows)
    diag = {
        "query_total": int(len(query_rows)),
        **full_catalog_composition(test_ds.meta),
    }
    rows = []

    print("TIME_ONLY", detector, mode, flush=True)
    time_ranks = ranks_for_time_only(test_time, test_gt, query_rows)
    add_rows(rows, detector, mode, "time_only", metrics_from_ranks(time_ranks, query_family, len(test_gt)), diag, {"lambda_time": np.nan, "time_cap": np.nan})

    for cap in CAPS:
        for lambda_time in LAMBDAS:
            variant = "waveform_only" if lambda_time == 0.0 else "waveform_plus_capped_time" if cap is not None else "waveform_plus_time"
            print("SWEEP", detector, mode, "lambda", lambda_time, "cap", cap, flush=True)
            ranks = ranks_for_lambda(test_scores, test_time, test_gt, query_rows, lambda_time, cap)
            add_rows(
                rows,
                detector,
                mode,
                variant,
                metrics_from_ranks(ranks, query_family, len(test_gt)),
                diag,
                {"lambda_time": float(lambda_time), "time_cap": np.nan if cap is None else float(cap)},
            )
            pd.DataFrame(rows).to_csv(out_dir / "lambda_sweep_partial.csv", index=False)
    pd.DataFrame(rows).to_csv(out_dir / "lambda_sweep_summary.csv", index=False)
    return rows


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    all_rows = []
    for detector, mode in JOBS:
        print("RUN_LAMBDA_SWEEP", detector, mode, flush=True)
        all_rows.extend(run_one(detector, mode))
        pd.DataFrame(all_rows).to_csv(OUT_ROOT / "lambda_sweep_summary_partial.csv", index=False)
    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_ROOT / "lambda_sweep_summary.csv", index=False)
    for metric in ["r@1", "r@5", "r@10", "top_1pct", "top_5pct", "top_10pct"]:
        df.pivot_table(
            index=["detector", "data_mode", "subset", "variant", "lambda_time", "time_cap"],
            values=metric,
            aggfunc="first",
        ).to_csv(OUT_ROOT / f"{metric.replace('@', '')}_pivot.csv")
    (OUT_ROOT / "protocol_summary.json").write_text(json.dumps({
        "elapsed_s": float(time.perf_counter() - t0),
        "lambdas": LAMBDAS,
        "caps": CAPS,
        "note": "Full-catalog linear fusion sweep: z(waveform_score) + lambda_time * clip(z(time_score), +/- cap). The catalog includes SIS, PM, and unlensed events.",
    }, indent=2), encoding="utf-8")
    print(df.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
