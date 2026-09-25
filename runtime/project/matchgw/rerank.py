from __future__ import annotations

import numpy as np
import pandas as pd

from .calibration import LogisticCalibrator, calibration_metrics
from .matching import candidate_rows

# 候选边校准使用的轻量特征：原始相似度、双向 rank、是否互为 top1、行内 margin。
FEATURE_COLUMNS = [
    "score",
    "rank_inv_sum",
    "rank_inv_min",
    "is_mutual_top1",
    "margin_min",
    "margin_mean",
]


def candidate_feature_frame(scores: np.ndarray, gt_partner: np.ndarray, params: dict, matcher: str = "mw") -> pd.DataFrame:
    # 把 Top-K 候选边转成表格特征。这个表既可训练 calibrator，也可导出给论文分析。
    rows = candidate_rows(scores, gt_partner, params, matcher=matcher)
    df = pd.DataFrame(rows)
    if df.empty:
        for col in FEATURE_COLUMNS + ["p_hat", "tier"]:
            df[col] = []
        return df
    df["rank_inv_sum"] = 1.0 / df["rank_i"].astype(float) + 1.0 / df["rank_j"].astype(float)
    df["rank_inv_min"] = np.minimum(1.0 / df["rank_i"].astype(float), 1.0 / df["rank_j"].astype(float))
    df["is_mutual_top1"] = ((df["rank_i"] == 1) & (df["rank_j"] == 1)).astype(float)
    df["margin_min"] = np.minimum(df["margin_i"].astype(float), df["margin_j"].astype(float))
    df["margin_mean"] = 0.5 * (df["margin_i"].astype(float) + df["margin_j"].astype(float))
    return df


def fit_pair_calibrator(df: pd.DataFrame, cfg) -> LogisticCalibrator:
    # 在验证候选边上拟合逻辑回归校准器，把相似度/排名特征变成可解释的 p_hat。
    calibrator = LogisticCalibrator(l2=cfg.calibration_l2, lr=cfg.calibration_lr, max_iter=cfg.calibration_iters)
    if df.empty:
        return calibrator.fit(np.zeros((0, len(FEATURE_COLUMNS)), dtype=np.float32), np.zeros((0,), dtype=np.float32))
    x = df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y = df["is_true"].to_numpy(dtype=np.float32)
    return calibrator.fit(x, y)


def apply_pair_calibrator(df: pd.DataFrame, calibrator: LogisticCalibrator, p_low: float, p_high: float) -> pd.DataFrame:
    # 按 p_hat 分 Tier：tier1 高可信、tier2 待跟进、tier3 低优先级。
    df = df.copy()
    if df.empty:
        df["p_hat"] = []
        df["tier"] = []
        return df
    df["p_hat"] = calibrator.predict_proba(df[FEATURE_COLUMNS].to_numpy(dtype=np.float32))
    df["tier"] = np.where(df["p_hat"] >= p_high, "tier1", np.where(df["p_hat"] >= p_low, "tier2", "tier3"))
    return df.sort_values(["p_hat", "score"], ascending=False).reset_index(drop=True)


def candidate_system_metrics(df: pd.DataFrame, n_events: int, gt_partner: np.ndarray, p_low: float, p_high: float) -> dict[str, float]:
    # 论文中的 follow-up reduction / compression factor / tier recall 都在这里计算。
    true_pairs = {tuple(sorted((i, int(j)))) for i, j in enumerate(gt_partner) if j >= 0 and i < j}
    total_true = len(true_pairs)
    exhaustive = n_events * (n_events - 1) / 2.0
    out: dict[str, float] = {
        "candidate_edges": int(len(df)),
        "exhaustive_edges": float(exhaustive),
        "followup_reduction": float(1.0 - len(df) / max(exhaustive, 1.0)),
        "compression_factor": float(exhaustive / max(len(df), 1)),
        "candidate_pair_recall": float(df["is_true"].sum() / max(total_true, 1)) if not df.empty else 0.0,
    }
    for name, mask in {
        "tier1": df["p_hat"] >= p_high if not df.empty else np.array([], dtype=bool),
        "tier12": df["p_hat"] >= p_low if not df.empty else np.array([], dtype=bool),
        "tier3": df["p_hat"] < p_low if not df.empty else np.array([], dtype=bool),
    }.items():
        sub = df.loc[mask] if not df.empty else df
        positives = float(sub["is_true"].sum()) if not sub.empty else 0.0
        out[f"{name}_edges"] = int(len(sub))
        out[f"{name}_precision"] = float(positives / max(len(sub), 1))
        out[f"{name}_recall"] = float(positives / max(total_true, 1))
    out["tier3_discarded_fraction"] = float(out["tier3_edges"] / max(len(df), 1))
    return out


def calibrated_candidate_report(scores: np.ndarray, gt_partner: np.ndarray, params: dict, calibrator: LogisticCalibrator | None, cfg, matcher: str = "mw") -> tuple[pd.DataFrame, dict[str, float]]:
    df = candidate_feature_frame(scores, gt_partner, params, matcher=matcher)
    if calibrator is None:
        calibrator = fit_pair_calibrator(df, cfg)
    df = apply_pair_calibrator(df, calibrator, cfg.p_low, cfg.p_high)
    labels = df["is_true"].to_numpy(dtype=np.float32) if not df.empty else np.zeros((0,), dtype=np.float32)
    probs = df["p_hat"].to_numpy(dtype=np.float32) if not df.empty else np.zeros((0,), dtype=np.float32)
    metrics = {
        **candidate_system_metrics(df, len(scores), gt_partner, cfg.p_low, cfg.p_high),
        **{f"cal_{k}": v for k, v in calibration_metrics(probs, labels, cfg.calibration_bins).items()},
    }
    return df, metrics
