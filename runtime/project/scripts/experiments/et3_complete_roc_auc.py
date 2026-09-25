from __future__ import annotations

import gc
import importlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_auc_score, roc_curve


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

ET3_ROOT = Path("/root/autodl-tmp/createdata/et3_10000_20260616_1006_match_root")
FRESH_ROOT = Path("runs/et3_fresh50_full_catalog_20260616")
OUT_DIR = Path("runs/et3_fresh50_full_catalog_20260616/complete_roc_auc_9000_event_test")
FAMILIES = ["SIS", "PM"]
MAX_SAVED_ROC_POINTS = 20_000


def make_test_gt(mode: str) -> tuple[np.ndarray, list[dict]]:
    base = importlib.import_module("scripts.experiments.80_mixed_sis_pm_catalog_modality_compare")
    base.ROOTS[("SIS", "ET3")] = ET3_ROOT
    base.ROOTS[("PM", "ET3")] = ET3_ROOT
    cfg = base.make_cfg("ET3", mode, FRESH_ROOT / "fresh_mixed_encoders" / f"et3_{mode}_mixed_sis_pm_ep50")
    arrays = {family: base.FamilyArrays(family, ET3_ROOT, mode) for family in FAMILIES}
    splits = {}
    for index, family in enumerate(FAMILIES):
        splits[family] = base.split_indices(len(arrays[family].l1), cfg.seed + index)
        splits[f"{family}_U"] = base.split_indices(len(arrays[family].unlensed), cfg.seed + 100 + index)
    ds = base.MixedEvaluationSet(arrays, splits, "test", cfg)
    return base.ground_truth(ds.meta), ds.meta


def offdiag_flatten(scores: np.ndarray) -> np.ndarray:
    n = scores.shape[0]
    keep = ~np.eye(n, dtype=bool)
    return np.asarray(scores[keep], dtype=np.float32)


def positive_flat_indices(gt: np.ndarray) -> np.ndarray:
    n = len(gt)
    rows = np.flatnonzero(gt >= 0).astype(np.int64)
    cols = gt[rows].astype(np.int64)
    # Index in row-major matrix with the diagonal removed from each row.
    return (rows * (n - 1) + cols - (cols > rows)).astype(np.int64)


def downsample_curve(fpr: np.ndarray, tpr: np.ndarray, thresholds: np.ndarray) -> tuple[pd.DataFrame, int, bool]:
    total = int(len(fpr))
    if total <= MAX_SAVED_ROC_POINTS:
        return pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thresholds}), total, False
    linear = np.linspace(0, total - 1, MAX_SAVED_ROC_POINTS, dtype=np.int64)
    low_fpr = np.flatnonzero(fpr <= 0.01)
    if len(low_fpr) > 0:
        low_pick = low_fpr[np.linspace(0, len(low_fpr) - 1, min(5000, len(low_fpr)), dtype=np.int64)]
        idx = np.unique(np.concatenate([linear, low_pick, [0, total - 1]])).astype(np.int64)
    else:
        idx = np.unique(np.concatenate([linear, [0, total - 1]])).astype(np.int64)
    return pd.DataFrame({"fpr": fpr[idx], "tpr": tpr[idx], "threshold": thresholds[idx]}), total, True


def run_mode(mode: str) -> dict:
    score_path = FRESH_ROOT / "fresh_mixed_encoders" / f"et3_{mode}_mixed_sis_pm_ep50" / "test_scores.npy"
    scores = np.load(score_path, mmap_mode="r")
    gt, meta = make_test_gt(mode)
    n = int(scores.shape[0])
    flat_scores = offdiag_flatten(scores)
    labels = np.zeros(flat_scores.shape[0], dtype=bool)
    pos_idx = positive_flat_indices(gt)
    labels[pos_idx] = True

    fpr, tpr, thresholds = roc_curve(labels, flat_scores, drop_intermediate=True)
    roc_auc = float(roc_auc_score(labels, flat_scores))
    curve_auc = float(auc(fpr, tpr))

    curve, roc_points_total, curve_downsampled = downsample_curve(fpr, tpr, thresholds)
    curve_path = OUT_DIR / f"et3_{mode}_9000_event_complete_waveform_roc_curve_downsampled.csv"
    curve.to_csv(curve_path, index=False)

    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.plot(curve["fpr"], curve["tpr"], lw=1.7, label=f"AUC = {roc_auc:.6f}")
    ax.plot([0, 1], [0, 1], color="0.45", lw=1.0, ls="--")
    ax.set_xlabel("False positive rate", fontweight="bold")
    ax.set_ylabel("True positive rate", fontweight="bold")
    ax.set_title(f"ET-3 {mode}: complete ROC on 9,000-event test catalog", fontweight="bold")
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"et3_{mode}_9000_event_complete_waveform_roc.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUT_DIR / f"et3_{mode}_9000_event_complete_waveform_roc.pdf", bbox_inches="tight")
    plt.close(fig)

    by_tag = pd.Series([m["tag"] for m in meta]).value_counts().to_dict()
    by_family = pd.Series([m["family"] for m in meta]).value_counts().to_dict()
    summary = {
        "mode": mode,
        "score_source": str(score_path),
        "catalog_events": n,
        "directed_candidate_pairs_excluding_self": int(labels.size),
        "positive_directed_pairs": int(labels.sum()),
        "negative_directed_pairs": int((~labels).sum()),
        "roc_auc": roc_auc,
        "curve_auc": curve_auc,
        "roc_points_total": int(roc_points_total),
        "roc_points_saved": int(len(curve)),
        "curve_downsampled": bool(curve_downsampled),
        "curve_csv": str(curve_path),
        "family_counts": {str(k): int(v) for k, v in by_family.items()},
        "tag_counts": {str(k): int(v) for k, v in by_tag.items()},
        "note": "ROC uses every directed query-candidate pair in the 9,000-event test catalog, excluding self-pairs; positives are true lensed image mates.",
    }
    del scores, flat_scores, labels, fpr, tpr, thresholds
    gc.collect()
    return summary


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summaries = [run_mode("noisy"), run_mode("pure")]
    df = pd.DataFrame(summaries)
    df.to_csv(OUT_DIR / "et3_9000_event_complete_waveform_roc_auc_summary.csv", index=False)
    (OUT_DIR / "et3_9000_event_complete_waveform_roc_auc_summary.json").write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
