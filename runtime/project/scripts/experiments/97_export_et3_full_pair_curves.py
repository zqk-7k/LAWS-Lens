from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from matchgw.aux_priors import observed_sky_pair_features, public_observed_sky_features


runner = importlib.import_module("scripts.experiments.90_et3_full_experiment_runner")
fresh, liao, _pdf = runner.configure_modules()
stage7 = importlib.import_module("scripts.experiments.91_et3_modality_combinations")


MAIN_VARIANT = "waveform_plus_liao_time_lr_plus_observed_sky_step"
STAGE7_SUMMARY = runner.RERANK_ROOT / "stage7_modality_combinations" / "stage7_modality_combinations_summary.csv"
SECONDS_PER_DAY = 86400.0


def event_label(meta: list[dict], idx: int) -> str:
    item = meta[int(idx)]
    pair = item.get("pair_id", -1)
    source = item.get("source_index", -1)
    if item["tag"] in {"L1", "L2"}:
        return f"{item['family']}_{item['tag']}_pair{int(pair):04d}_src{int(source):05d}"
    return f"{item['family']}_U_src{int(source):05d}"


def validate_directed_stage7(waveform: np.ndarray, combined: np.ndarray, gt: np.ndarray, meta: list[dict]) -> dict:
    summary = pd.read_csv(STAGE7_SUMMARY)
    observed = {
        "waveform_only": liao.evaluate_score(waveform, gt, meta)["overall"],
        MAIN_VARIANT: liao.evaluate_score(combined, gt, meta)["overall"],
    }
    checks = []
    for variant, metrics in observed.items():
        ref = summary[(summary["variant"] == variant) & (summary["subset"] == "overall")].iloc[0]
        for key in ["r@1", "r@5", "r@10", "median_true_rank", "valid"]:
            checks.append({
                "variant": variant,
                "metric": key,
                "computed": float(metrics[key]),
                "reference": float(ref[key]),
                "abs_diff": abs(float(metrics[key]) - float(ref[key])),
            })
    bad = [row for row in checks if row["abs_diff"] > 1e-8]
    if bad:
        raise RuntimeError(f"Directed stage7 score mismatch: {bad[:5]}")
    return {
        variant: {
            "r@1": float(metrics["r@1"]),
            "r@5": float(metrics["r@5"]),
            "r@10": float(metrics["r@10"]),
            "median_true_rank": float(metrics["median_true_rank"]),
            "valid": int(metrics["valid"]),
        }
        for variant, metrics in observed.items()
    }


def true_pair_maps(gt: np.ndarray, meta: list[dict]) -> tuple[dict[int, tuple[int, str]], int, dict[str, int]]:
    true_by_left: dict[int, tuple[int, str]] = {}
    counts = {"SIS": 0, "PM": 0}
    for i, j in enumerate(gt):
        j = int(j)
        if j >= 0 and i < j:
            family = str(meta[i]["family"])
            true_by_left[int(i)] = (j, family)
            counts[family] += 1
    return true_by_left, sum(counts.values()), counts


def parquet_schema() -> pa.Schema:
    return pa.schema([
        ("pair_i", pa.int32()),
        ("pair_j", pa.int32()),
        ("event_i", pa.string()),
        ("event_j", pa.string()),
        ("is_true_pair", pa.int8()),
        ("lens_type", pa.string()),
        ("waveform_score", pa.float32()),
        ("time_score", pa.float32()),
        ("sky_score", pa.float32()),
        ("final_score", pa.float32()),
    ])


def write_full_pair_table(
    out_path: Path,
    meta: list[dict],
    gt: np.ndarray,
    waveform: np.ndarray,
    time_score: np.ndarray,
    sky_score: np.ndarray,
    final_directed: np.ndarray,
    row_chunk: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    n = len(meta)
    n_pairs = n * (n - 1) // 2
    scores_all = np.empty(n_pairs, dtype=np.float32)
    labels_all = np.zeros(n_pairs, dtype=np.bool_)
    group_all = np.zeros(n_pairs, dtype=np.uint8)  # 0 false/background, 1 SIS true, 2 PM true

    events = np.asarray([event_label(meta, idx) for idx in range(n)], dtype=object)
    true_by_left, n_true, true_counts = true_pair_maps(gt, meta)
    schema = parquet_schema()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    cursor = 0
    writer = pq.ParquetWriter(out_path, schema=schema, compression="zstd", use_dictionary=True)
    try:
        for start in range(0, n - 1, row_chunk):
            end = min(start + row_chunk, n - 1)
            left_parts = []
            right_parts = []
            for i in range(start, end):
                js = np.arange(i + 1, n, dtype=np.int32)
                left_parts.append(np.full(len(js), i, dtype=np.int32))
                right_parts.append(js)
            left = np.concatenate(left_parts)
            right = np.concatenate(right_parts)
            m = len(left)

            wf = np.maximum(waveform[left, right], waveform[right, left]).astype(np.float32)
            tm = np.maximum(time_score[left, right], time_score[right, left]).astype(np.float32)
            sk = np.maximum(sky_score[left, right], sky_score[right, left]).astype(np.float32)
            fs = np.maximum(final_directed[left, right], final_directed[right, left]).astype(np.float32)

            is_true = np.zeros(m, dtype=np.int8)
            group = np.zeros(m, dtype=np.uint8)
            lens_type = np.full(m, "background", dtype=object)
            for i in range(start, end):
                item = true_by_left.get(i)
                if item is None:
                    continue
                j, family = item
                local = np.flatnonzero((left == i) & (right == j))
                if len(local) == 1:
                    pos = int(local[0])
                    is_true[pos] = 1
                    lens_type[pos] = family
                    group[pos] = 1 if family == "SIS" else 2

            scores_all[cursor:cursor + m] = fs
            labels_all[cursor:cursor + m] = is_true.astype(bool)
            group_all[cursor:cursor + m] = group
            cursor += m

            table = pa.Table.from_pydict({
                "pair_i": left,
                "pair_j": right,
                "event_i": events[left],
                "event_j": events[right],
                "is_true_pair": is_true,
                "lens_type": lens_type,
                "waveform_score": wf,
                "time_score": tm,
                "sky_score": sk,
                "final_score": fs,
            }, schema=schema)
            writer.write_table(table)
            print(f"WROTE_PAIR_ROWS left={start}:{end} rows={m} cursor={cursor}/{n_pairs}", flush=True)
    finally:
        writer.close()

    if cursor != n_pairs:
        raise RuntimeError(f"Pair cursor mismatch: {cursor} != {n_pairs}")
    if int(labels_all.sum()) != n_true:
        raise RuntimeError(f"True pair label mismatch: {int(labels_all.sum())} != {n_true}")
    meta_out = {
        "n_events": int(n),
        "n_pairs": int(n_pairs),
        "n_true_pairs": int(n_true),
        "n_false_pairs": int(n_pairs - n_true),
        "true_pair_counts": {k: int(v) for k, v in true_counts.items()},
    }
    return scores_all, labels_all, group_all, meta_out


def precision_recall_curve_compat(labels: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        return precision_recall_curve(labels, scores, drop_intermediate=True)
    except TypeError:
        return precision_recall_curve(labels, scores)


def summarize_low_fpr(
    scores: np.ndarray,
    labels: np.ndarray,
    group: np.ndarray,
    targets: list[float],
) -> list[dict]:
    n_pos = int(labels.sum())
    n_neg = int(len(labels) - n_pos)
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    sorted_group = group[order]
    tp = np.cumsum(sorted_labels, dtype=np.int64)
    fp = np.cumsum(~sorted_labels, dtype=np.int64)
    sis_tp = np.cumsum(sorted_group == 1, dtype=np.int64)
    pm_tp = np.cumsum(sorted_group == 2, dtype=np.int64)
    sis_total = int(np.sum(group == 1))
    pm_total = int(np.sum(group == 2))

    rows = []
    for target in targets:
        max_fp = int(math.floor(target * n_neg))
        valid = np.flatnonzero(fp <= max_fp)
        idx = int(valid[-1]) if len(valid) else 0
        true_count = int(tp[idx])
        false_count = int(fp[idx])
        selected = true_count + false_count
        rows.append({
            "target_fpr": float(target),
            "threshold": float(sorted_scores[idx]),
            "fpr": float(false_count / n_neg),
            "tpr": float(true_count / n_pos),
            "precision": float(true_count / selected) if selected else 0.0,
            "false_pair_count": false_count,
            "true_pair_count": true_count,
            "sis_recall": float(sis_tp[idx] / sis_total) if sis_total else None,
            "pm_recall": float(pm_tp[idx] / pm_total) if pm_total else None,
        })
    return rows


def summarize_recall_targets(scores: np.ndarray, labels: np.ndarray, targets: list[float]) -> list[dict]:
    n_pos = int(labels.sum())
    n_neg = int(len(labels) - n_pos)
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    tp = np.cumsum(sorted_labels, dtype=np.int64)
    fp = np.cumsum(~sorted_labels, dtype=np.int64)
    rows = []
    for target in targets:
        needed = int(math.ceil(target * n_pos))
        idxs = np.flatnonzero(tp >= needed)
        idx = int(idxs[0]) if len(idxs) else len(tp) - 1
        true_count = int(tp[idx])
        false_count = int(fp[idx])
        selected = true_count + false_count
        rows.append({
            "target_recall": float(target),
            "threshold": float(sorted_scores[idx]),
            "recall": float(true_count / n_pos),
            "fpr": float(false_count / n_neg),
            "precision": float(true_count / selected) if selected else 0.0,
            "false_candidates": false_count,
            "true_candidates": true_count,
            "total_candidates": selected,
        })
    return rows


def group_metrics(scores: np.ndarray, group: np.ndarray) -> dict:
    out = {}
    for code, name in [(1, "SIS"), (2, "PM")]:
        mask = (group == 0) | (group == code)
        labels = group[mask] == code
        out[name] = {
            "n_true": int(labels.sum()),
            "n_false": int((~labels).sum()),
            "roc_auc": float(roc_auc_score(labels, scores[mask])),
            "average_precision": float(average_precision_score(labels, scores[mask])),
            "base_positive_rate": float(labels.mean()),
        }
    return out


def save_curves(
    out_dir: Path,
    labels: np.ndarray,
    scores: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, float, float]:
    roc_auc = float(roc_auc_score(labels, scores))
    auprc = float(average_precision_score(labels, scores))
    fpr, tpr, roc_thr = roc_curve(labels, scores, drop_intermediate=True)
    precision, recall, pr_thr = precision_recall_curve_compat(labels, scores)

    roc_df = pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": roc_thr})
    pr_threshold = np.empty(len(precision), dtype=np.float64)
    pr_threshold[:] = np.nan
    pr_threshold[:len(pr_thr)] = pr_thr
    pr_df = pd.DataFrame({"precision": precision, "recall": recall, "threshold": pr_threshold})
    roc_df.to_csv(out_dir / "et3_full_roc.csv", index=False)
    pr_df.to_csv(out_dir / "et3_full_pr.csv", index=False)
    return roc_df, pr_df, roc_auc, auprc


def save_figure(fig_path: Path, roc_df: pd.DataFrame, pr_df: pd.DataFrame, roc_auc: float, auprc: float, base_rate: float) -> None:
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 12,
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10,
    })
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))
    ax = axes[0]
    ax.plot(roc_df["fpr"], roc_df["tpr"], color="#1f77b4", lw=1.7, label=f"AUC = {roc_auc:.4f}")
    ax.set_xscale("symlog", linthresh=1e-6, linscale=0.7)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Full unordered pair-level ROC")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False, loc="lower right")

    ax = axes[1]
    ax.plot(pr_df["recall"], pr_df["precision"], color="#d62728", lw=1.7, label=f"AUPRC = {auprc:.4f}")
    ax.axhline(base_rate, color="0.45", lw=1.0, ls="--", label=f"base rate = {base_rate:.2e}")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Full unordered pair-level PR")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, bbox_inches="tight")
    fig.savefig(fig_path.with_suffix(".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export complete ET-3 unordered pair-level ROC/PR diagnostics.")
    parser.add_argument("--out-dir", default="results/pair_level_full_curves")
    parser.add_argument("--figure", default="figures/fig_et3_full_pair_roc_pr.pdf")
    parser.add_argument("--row-chunk", type=int, default=64)
    args = parser.parse_args()

    t0 = time.perf_counter()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table_path = out_dir / "et3_pair_scores_full.parquet"
    final_alias = out_dir / "et3_pair_scores_final_full.parquet"

    print("LOAD_STAGE7_ET3_INPUTS", flush=True)
    loaded = liao.load_job("ET3", "noisy")
    val_ds, val_raw, val_time, val_gt, val_scores = loaded["val"]
    test_ds, test_raw, test_time, test_gt, test_scores = loaded["test"]

    print("BUILD_STAGE7_COMPONENTS", flush=True)
    prior = liao.fit_time_lr_from_liao("ET3", val_time, val_gt)
    test_sky = liao.make_observed_sky("ET3", test_raw, test_time, seed=702000)
    sky_features = observed_sky_pair_features(public_observed_sky_features(test_sky), chunk_rows=liao.CHUNK_ROWS)

    waveform = liao.row_z(test_scores)
    time_score = liao.row_z(liao.time_lr_score_matrix(test_time, prior))
    sky_score = liao.row_z(sky_features["sky_step_weight"])
    final_directed = stage7.score_from_weights(
        {"waveform": waveform, "liao_time_lr": time_score, "observed_sky_step": sky_score},
        ["waveform", "liao_time_lr", "observed_sky_step"],
        {"waveform": 1.0, "liao_time_lr": 1.0, "observed_sky_step": 0.25},
    )

    print("VALIDATE_DIRECTED_STAGE7", flush=True)
    directed_metrics = validate_directed_stage7(waveform, final_directed, test_gt, test_ds.meta)

    print("WRITE_FULL_UNORDERED_PAIR_TABLE", flush=True)
    scores, labels, group, count_meta = write_full_pair_table(
        table_path,
        test_ds.meta,
        test_gt,
        waveform,
        time_score,
        sky_score,
        final_directed,
        row_chunk=args.row_chunk,
    )
    if final_alias.exists() or final_alias.is_symlink():
        final_alias.unlink()
    try:
        final_alias.symlink_to(table_path.name)
    except OSError:
        final_alias.write_bytes(table_path.read_bytes())

    expected_pairs = count_meta["n_events"] * (count_meta["n_events"] - 1) // 2
    if count_meta["n_pairs"] != expected_pairs:
        raise RuntimeError(f"Unexpected n_pairs: {count_meta['n_pairs']} != {expected_pairs}")

    print("COMPUTE_FULL_CURVES", flush=True)
    roc_df, pr_df, roc_auc, auprc = save_curves(out_dir, labels, scores)
    base_rate = float(labels.mean())
    low_fpr = summarize_low_fpr(scores, labels, group, [1e-2, 1e-3, 1e-4, 1e-5, 1e-6])
    recall_targets = summarize_recall_targets(scores, labels, [0.1, 0.5, 0.9])
    subgroup = group_metrics(scores, group)

    print("SAVE_FIGURE", flush=True)
    fig_path = Path(args.figure)
    save_figure(fig_path, roc_df, pr_df, roc_auc, auprc, base_rate)

    metrics = {
        **count_meta,
        "n_false_pairs": int((~labels).sum()),
        "roc_auc": roc_auc,
        "average_precision_auprc": auprc,
        "base_positive_rate": base_rate,
        "score_definition": {
            "stage7_variant": MAIN_VARIANT,
            "directed_components": {
                "waveform": "row_z(test_scores)",
                "time_score": "row_z(liao_time_lr_score_matrix(test_time, validation_fitted_liao_prior))",
                "sky_score": "row_z(observed_sky_step_weight, observed sky seed=702000)",
            },
            "validation_selected_weights": {
                "waveform": 1.0,
                "liao_time_lr": 1.0,
                "observed_sky_step": 0.25,
            },
            "pair_level_aggregation": "max_directed",
            "final_score": "max_i_j(waveform + 1.0*time + 0.25*sky, reverse direction)",
        },
        "paths": {
            "pair_scores_full": str(table_path),
            "pair_scores_final_full": str(final_alias),
            "roc_csv": str(out_dir / "et3_full_roc.csv"),
            "pr_csv": str(out_dir / "et3_full_pr.csv"),
            "figure_pdf": str(fig_path),
            "figure_png": str(fig_path.with_suffix(".png")),
            "stage7_summary": str(STAGE7_SUMMARY),
        },
        "directed_stage7_validation": directed_metrics,
        "low_fpr": low_fpr,
        "recall_targets": recall_targets,
        "subgroup_metrics": subgroup,
        "curve_rows": {
            "roc": int(len(roc_df)),
            "pr": int(len(pr_df)),
        },
        "elapsed_s": float(time.perf_counter() - t0),
    }
    metrics_path = out_dir / "et3_full_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
