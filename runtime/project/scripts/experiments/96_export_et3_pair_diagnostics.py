from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from matchgw.aux_priors import observed_sky_pair_features, public_observed_sky_features

runner = importlib.import_module("scripts.experiments.90_et3_full_experiment_runner")
fresh, liao, _pdf = runner.configure_modules()
stage7 = importlib.import_module("scripts.experiments.91_et3_modality_combinations")

SECONDS_PER_DAY = 86400.0
STAGE7_SUMMARY = runner.RERANK_ROOT / "stage7_modality_combinations" / "stage7_modality_combinations_summary.csv"
MAIN_VARIANT = "waveform_plus_liao_time_lr_plus_observed_sky_step"


def pair_code(i: np.ndarray, j: np.ndarray, n: int) -> np.ndarray:
    left = np.minimum(i, j).astype(np.int64)
    right = np.maximum(i, j).astype(np.int64)
    return left * np.int64(n) + right


def decode_code(code: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    left = (code // np.int64(n)).astype(np.int32)
    right = (code % np.int64(n)).astype(np.int32)
    return left, right


def rank_matrix(score: np.ndarray) -> np.ndarray:
    order = np.argsort(-score, axis=1)
    ranks = np.empty_like(order, dtype=np.int32)
    ranks[np.arange(score.shape[0])[:, None], order] = np.arange(1, score.shape[1] + 1, dtype=np.int32)
    return ranks


def event_label(meta: list[dict], idx: int) -> str:
    item = meta[int(idx)]
    pair = item.get("pair_id", -1)
    source = item.get("source_index", -1)
    if item["tag"] in {"L1", "L2"}:
        return f"{item['family']}_{item['tag']}_pair{int(pair):04d}_src{int(source):05d}"
    return f"{item['family']}_U_src{int(source):05d}"


def true_pair_table(gt: np.ndarray, meta: list[dict], n: int) -> tuple[np.ndarray, np.ndarray, set[int]]:
    rows = np.flatnonzero(gt >= 0).astype(np.int32)
    left = []
    right = []
    for i in rows:
        j = int(gt[int(i)])
        if int(i) < j:
            left.append(int(i))
            right.append(j)
    left_arr = np.asarray(left, dtype=np.int32)
    right_arr = np.asarray(right, dtype=np.int32)
    codes = set(int(x) for x in pair_code(left_arr, right_arr, n))
    if len(left_arr) != 3000:
        print(f"WARNING expected 3000 true unordered ET3 test pairs, found {len(left_arr)}", flush=True)
    return left_arr, right_arr, codes


def random_false_pairs(n: int, true_codes: set[int], count: int, seed: int) -> tuple[np.ndarray, np.ndarray, set[int]]:
    rng = np.random.default_rng(seed)
    selected: set[int] = set()
    while len(selected) < count:
        need = count - len(selected)
        batch = max(need * 3, 10000)
        i = rng.integers(0, n, size=batch, dtype=np.int32)
        j = rng.integers(0, n, size=batch, dtype=np.int32)
        keep = i != j
        codes = pair_code(i[keep], j[keep], n)
        for code in np.unique(codes):
            c = int(code)
            if c not in true_codes and c not in selected:
                selected.add(c)
                if len(selected) >= count:
                    break
    codes_arr = np.asarray(sorted(selected), dtype=np.int64)
    left, right = decode_code(codes_arr, n)
    return left, right, selected


def hard_false_pairs(
    combined_score: np.ndarray,
    true_codes: set[int],
    exclude_codes: set[int],
    count: int,
    chunk_rows: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    n = combined_score.shape[0]
    pools = []
    blocked = true_codes | exclude_codes
    for start in range(0, n - 1, chunk_rows):
        end = min(start + chunk_rows, n - 1)
        left_parts = []
        right_parts = []
        for i in range(start, end):
            js = np.arange(i + 1, n, dtype=np.int32)
            left_parts.append(np.full(len(js), i, dtype=np.int32))
            right_parts.append(js)
        if not left_parts:
            continue
        left = np.concatenate(left_parts)
        right = np.concatenate(right_parts)
        codes = pair_code(left, right, n)
        keep = np.fromiter((int(code) not in blocked for code in codes), dtype=bool, count=len(codes))
        if not keep.any():
            continue
        left = left[keep]
        right = right[keep]
        vals = np.maximum(combined_score[left, right], combined_score[right, left])
        vals = vals.astype(np.float32)
        finite = np.isfinite(vals)
        left = left[finite]
        right = right[finite]
        vals = vals[finite]
        if len(vals) == 0:
            continue
        local_k = min(count, len(vals))
        idx = np.argpartition(vals, -local_k)[-local_k:]
        pools.append(pd.DataFrame({"left": left[idx], "right": right[idx], "score": vals[idx]}))
        print(f"HARD_FALSE_POOL rows={start}:{end} kept={len(idx)}", flush=True)
    if not pools:
        raise RuntimeError("No hard false candidates found")
    pool = pd.concat(pools, ignore_index=True)
    pool["code"] = pair_code(pool["left"].to_numpy(np.int32), pool["right"].to_numpy(np.int32), n)
    pool = pool.sort_values("score", ascending=False).drop_duplicates("code", keep="first")
    top = pool.head(count)
    if len(top) < count:
        print(f"WARNING requested {count} hard false pairs, only found {len(top)}", flush=True)
    return top["left"].to_numpy(np.int32), top["right"].to_numpy(np.int32)


def pair_class(meta_i: dict, meta_j: dict, is_true: bool) -> str:
    if is_true:
        return f"true_{str(meta_i['family']).lower()}_pair"
    lensed_i = meta_i["tag"] in {"L1", "L2"}
    lensed_j = meta_j["tag"] in {"L1", "L2"}
    if lensed_i and lensed_j:
        return "lensed_nonpartner_false"
    if not lensed_i and not lensed_j:
        return "unlensed_unlensed_false"
    return "lensed_unlensed_false"


def build_rows(
    left: np.ndarray,
    right: np.ndarray,
    sampling_strategy: str,
    true_codes: set[int],
    meta: list[dict],
    test_time: pd.DataFrame,
    waveform: np.ndarray,
    time_score: np.ndarray,
    sky_score: np.ndarray,
    sky_norm_sep: np.ndarray,
    sky_log_overlap: np.ndarray,
    waveform_ranks: np.ndarray,
    combined: np.ndarray,
    combined_ranks: np.ndarray,
) -> pd.DataFrame:
    n = len(meta)
    codes = pair_code(left, right, n)
    is_true = np.fromiter((int(code) in true_codes for code in codes), dtype=bool, count=len(codes))
    t = test_time["trigger_time_obs"].to_numpy(dtype=np.float64)

    lens_type = []
    classes = []
    events_i = []
    events_j = []
    for i, j, flag in zip(left, right, is_true):
        mi = meta[int(i)]
        mj = meta[int(j)]
        lens_type.append(str(mi["family"]) if flag else "none")
        classes.append(pair_class(mi, mj, bool(flag)))
        events_i.append(event_label(meta, int(i)))
        events_j.append(event_label(meta, int(j)))

    return pd.DataFrame({
        "catalog": "ET3",
        "split": "test",
        "pair_i": left.astype(np.int32),
        "pair_j": right.astype(np.int32),
        "event_i": events_i,
        "event_j": events_j,
        "is_true_pair": is_true,
        "lens_type": lens_type,
        "pair_class": classes,
        "sampling_strategy": sampling_strategy,
        "delta_t_obs_days": (np.abs(t[left] - t[right]) / SECONDS_PER_DAY).astype(np.float32),
        "time_score_i_to_j": time_score[left, right],
        "time_score_j_to_i": time_score[right, left],
        "sky_score_i_to_j": sky_score[left, right],
        "sky_score_j_to_i": sky_score[right, left],
        "sky_norm_sep": sky_norm_sep[left, right],
        "sky_log_overlap": sky_log_overlap[left, right],
        "waveform_score_i_to_j": waveform[left, right],
        "waveform_score_j_to_i": waveform[right, left],
        "waveform_rank_i_to_j": waveform_ranks[left, right],
        "waveform_rank_j_to_i": waveform_ranks[right, left],
        "combined_score_i_to_j": combined[left, right],
        "combined_score_j_to_i": combined[right, left],
        "combined_score_mean": ((combined[left, right] + combined[right, left]) * 0.5).astype(np.float32),
        "combined_score_max": np.maximum(combined[left, right], combined[right, left]).astype(np.float32),
    })


def metrics_subset(metrics: dict[str, dict], variant: str) -> dict:
    overall = metrics["overall"]
    return {
        "variant": variant,
        "r@1": float(overall["r@1"]),
        "r@5": float(overall["r@5"]),
        "r@10": float(overall["r@10"]),
        "median_true_rank": float(overall["median_true_rank"]),
        "valid": int(overall["valid"]),
    }


def validate_metrics(waveform_metrics: dict, combined_metrics: dict, mismatch_out: Path) -> None:
    summary = pd.read_csv(STAGE7_SUMMARY)
    checks = []
    for variant, observed in [("waveform_only", waveform_metrics), (MAIN_VARIANT, combined_metrics)]:
        ref = summary[(summary["variant"] == variant) & (summary["subset"] == "overall")].iloc[0]
        for col in ["r@1", "r@5", "r@10", "median_true_rank", "valid"]:
            checks.append({
                "variant": variant,
                "metric": col,
                "new": observed[col],
                "reference": float(ref[col]),
                "abs_diff": abs(float(observed[col]) - float(ref[col])),
            })
    check_df = pd.DataFrame(checks)
    bad = check_df[check_df["abs_diff"] > 1e-8]
    if not bad.empty:
        mismatch_out.parent.mkdir(parents=True, exist_ok=True)
        check_df.to_csv(mismatch_out, index=False)
        raise RuntimeError(f"ET3 pair diagnostic metrics do not reproduce stage7 summary; wrote {mismatch_out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export ET3 pair-level diagnostics for NC-style mechanism figures.")
    parser.add_argument("--random-false-pairs", type=int, default=200_000)
    parser.add_argument("--hard-false-pairs", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=20260624)
    parser.add_argument("--out", default=str(runner.RERANK_ROOT / "pair_level_diagnostics" / "et3_pair_diagnostics.parquet"))
    parser.add_argument("--metrics-out", default=str(runner.RERANK_ROOT / "pair_level_diagnostics" / "et3_pair_diagnostics_metrics.json"))
    parser.add_argument("--mismatch-out", default=str(runner.RERANK_ROOT / "pair_level_diagnostics" / "et3_pair_diagnostics_mismatch_report.csv"))
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    print("LOAD_ET3_STAGE7_INPUTS", flush=True)
    loaded = liao.load_job("ET3", "noisy")
    val_ds, val_raw, val_time, val_gt, val_scores = loaded["val"]
    test_ds, test_raw, test_time, test_gt, test_scores = loaded["test"]

    print("BUILD_ET3_COMPONENTS", flush=True)
    prior = liao.fit_time_lr_from_liao("ET3", val_time, val_gt)
    test_sky = liao.make_observed_sky("ET3", test_raw, test_time, seed=702000)
    sky_features = observed_sky_pair_features(public_observed_sky_features(test_sky), chunk_rows=liao.CHUNK_ROWS)

    waveform = liao.row_z(test_scores)
    time_score = liao.row_z(liao.time_lr_score_matrix(test_time, prior))
    sky_score = liao.row_z(sky_features["sky_step_weight"])
    sky_log_overlap = sky_features["sky_log_overlap"].astype(np.float32)
    sky_norm_sep = sky_features["sky_norm_sep"].astype(np.float32)
    combined = stage7.score_from_weights(
        {"waveform": waveform, "liao_time_lr": time_score, "observed_sky_step": sky_score},
        ["waveform", "liao_time_lr", "observed_sky_step"],
        {"waveform": 1.0, "liao_time_lr": 1.0, "observed_sky_step": 0.25},
    )

    print("VALIDATE_ET3_RECALL", flush=True)
    waveform_metrics = metrics_subset(liao.evaluate_score(waveform, test_gt, test_ds.meta), "waveform_only")
    combined_metrics = metrics_subset(liao.evaluate_score(combined, test_gt, test_ds.meta), MAIN_VARIANT)
    validate_metrics(waveform_metrics, combined_metrics, Path(args.mismatch_out))

    n = len(test_ds.meta)
    true_left, true_right, true_codes = true_pair_table(test_gt, test_ds.meta, n)
    random_left, random_right, random_codes = random_false_pairs(n, true_codes, args.random_false_pairs, args.seed)
    hard_left, hard_right = hard_false_pairs(combined, true_codes, random_codes, args.hard_false_pairs)

    print("BUILD_RANK_MATRICES", flush=True)
    waveform_ranks = rank_matrix(waveform)
    combined_ranks = rank_matrix(combined)

    print("ASSEMBLE_PAIR_TABLE", flush=True)
    frames = [
        build_rows(true_left, true_right, "all_true_pairs", true_codes, test_ds.meta, test_time, waveform, time_score, sky_score, sky_norm_sep, sky_log_overlap, waveform_ranks, combined, combined_ranks),
        build_rows(random_left, random_right, "random_false_pairs", true_codes, test_ds.meta, test_time, waveform, time_score, sky_score, sky_norm_sep, sky_log_overlap, waveform_ranks, combined, combined_ranks),
        build_rows(hard_left, hard_right, "hard_false_pairs_top_combined", true_codes, test_ds.meta, test_time, waveform, time_score, sky_score, sky_norm_sep, sky_log_overlap, waveform_ranks, combined, combined_ranks),
    ]
    df = pd.concat(frames, ignore_index=True)
    df.to_parquet(out, index=False)

    summary = {
        "path": str(out),
        "rows": int(len(df)),
        "true_pairs": int(df["is_true_pair"].sum()),
        "false_pairs": int((~df["is_true_pair"]).sum()),
        "sampling_strategy_counts": {str(k): int(v) for k, v in df["sampling_strategy"].value_counts().sort_index().items()},
        "pair_class_counts": {str(k): int(v) for k, v in df["pair_class"].value_counts().sort_index().items()},
        "metrics": {"waveform_only": waveform_metrics, MAIN_VARIANT: combined_metrics},
        "stage7_summary": str(STAGE7_SUMMARY),
        "weights": {"waveform": 1.0, "liao_time_lr": 1.0, "observed_sky_step": 0.25},
    }
    Path(args.metrics_out).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
