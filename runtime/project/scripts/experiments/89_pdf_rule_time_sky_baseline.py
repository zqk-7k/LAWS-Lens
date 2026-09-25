from __future__ import annotations

import importlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

liao = importlib.import_module("scripts.experiments.88_liao_realistic_p1_p2_rerank")

OUT_DIR = Path("runs/liao_realistic_p1_p2_rerank_20260612/stage2b_pdf_rule_time_sky_baseline")
DOC_PATH = Path("docs/stage2b_pdf_rule_time_sky_baseline_report_20260615_cn.md")

PDF_SKY_AREA_THRESHOLDS_DEG2 = {
    "ET": 5000.0,   # PDF: 1 detector, within 5000 deg2 -> weight 1 else 0
    "ET3": 1000.0,  # ET three-arm approximation: tighter than single-ET hard mask, still not a true skymap
    "LIGO": 500.0,  # PDF: 2 detectors, within 500 deg2 -> weight 1 else 0
}


def pdf_sky_hard_mask_matrix(sky_obs: pd.DataFrame, area_threshold_deg2: float) -> np.ndarray:
    vec = liao.unit_from_radec(
        sky_obs["ra_obs"].to_numpy(dtype=np.float64),
        sky_obs["dec_obs"].to_numpy(dtype=np.float64),
    )
    n = len(sky_obs)
    theta_threshold_deg = math.sqrt(area_threshold_deg2 / math.pi)
    theta_threshold_rad = math.radians(theta_threshold_deg)
    out = np.empty((n, n), dtype=np.float32)
    for start in range(0, n, liao.CHUNK_ROWS):
        rows = slice(start, min(start + liao.CHUNK_ROWS, n))
        sep = liao.angular_sep_unit(vec[rows, None, :], vec[None, :, :])
        out[rows] = (sep <= theta_threshold_rad).astype(np.float32)
    np.fill_diagonal(out, -np.inf)
    return out


def add_rows(rows: list[dict], detector: str, mode: str, variant: str, metrics: dict[str, dict], diag: dict, extra: dict | None = None) -> None:
    extra = extra or {}
    for subset, values in metrics.items():
        rows.append({
            "detector": detector,
            "data_mode": mode,
            "stage": "stage2b_pdf_rule_time_sky_baseline",
            "variant": variant,
            "subset": subset,
            **diag,
            **values,
            **extra,
        })


def select_two_lambdas(components, gt, meta, k1, k2, grid):
    best_l1 = grid[0]
    best_l2 = grid[0]
    best_metrics = None
    best_key = (-1.0, -1.0, -1.0, -1.0)
    for l1 in grid:
        for l2 in grid:
            score = components["waveform"] + l1 * components[k1] + l2 * components[k2]
            np.fill_diagonal(score, -np.inf)
            metrics = liao.evaluate_score(score, gt, meta)
            key = (
                metrics["overall"]["r@10"],
                metrics["overall"]["r@5"],
                metrics["overall"]["r@1"],
                metrics["overall"]["top_1pct"],
            )
            if key > best_key:
                best_key = key
                best_l1 = l1
                best_l2 = l2
                best_metrics = metrics
    return best_l1, best_l2, best_metrics


def write_doc(df: pd.DataFrame) -> None:
    DOC_PATH.parent.mkdir(parents=True, exist_ok=True)
    parts = [
        "# Stage2b PDF 原始空间/时间赋权 baseline",
        "",
        "生成时间：2026-06-15",
        "",
        "## 实验目的",
        "",
        "本阶段专门测试 `透镜识别流程.pdf` 第三部分“后续处理”中的原始赋权思路，作为 naive baseline：",
        "",
        "- 空间位置赋权：按探测器数量给一个硬面积阈值，阈值内权重 1，否则 0。",
        "- 时间差赋权：用 Liao / GW-LMC time-delay prior 对照。",
        "",
        "注意：该阶段不是当前推荐主方法。推荐主方法仍是 Stage3 `waveform + Liao time LR + observed sky step`。",
        "",
        "## PDF 规则映射",
        "",
        "| detector | PDF detector-count mapping | area threshold | implementation |",
        "|---|---:|---:|---|",
        "| ET | 1 detector | 5000 deg2 | observed center angular area `pi theta^2 <= 5000` |",
        "| LIGO | 2 detectors | 500 deg2 | observed center angular area `pi theta^2 <= 500` |",
        "",
        "这里仍然不直接使用 true sky 做 pair feature；先生成 `ra_obs/dec_obs/sky_area90`，再对 observed center 计算硬阈值 mask。",
        "",
        "## Overall 结果",
        "",
    ]
    overall = df[df["subset"].eq("overall")].copy()
    overall = overall.sort_values(["detector", "r@10", "r@1"], ascending=[True, False, False])
    rows = []
    for _, r in overall.iterrows():
        rows.append({
            "detector": r.detector,
            "variant": r.variant,
            "R@1": liao.fmt(r["r@1"]),
            "R@5": liao.fmt(r["r@5"]),
            "R@10": liao.fmt(r["r@10"]),
            "Top1%": liao.fmt(r["top_1pct"]),
            "Median rank": liao.fmt(r["median_true_rank"]),
            "lambda_sky": liao.fmt(r.get("lambda_pdf_sky", np.nan)),
            "lambda_time": liao.fmt(r.get("lambda_liao_time_lr", np.nan)),
        })
    parts.append(liao.md_table(rows, ["detector", "variant", "R@1", "R@5", "R@10", "Top1%", "Median rank", "lambda_sky", "lambda_time"]))
    parts += [
        "",
        "## SIS / PM 分解",
        "",
    ]
    sub = df[df["subset"].isin(["SIS", "PM"])].copy()
    sub = sub.sort_values(["detector", "subset", "r@10", "r@1"], ascending=[True, True, False, False])
    rows = []
    for _, r in sub.iterrows():
        rows.append({
            "detector": r.detector,
            "subset": r.subset,
            "variant": r.variant,
            "R@1": liao.fmt(r["r@1"]),
            "R@5": liao.fmt(r["r@5"]),
            "R@10": liao.fmt(r["r@10"]),
            "Top1%": liao.fmt(r["top_1pct"]),
            "Median rank": liao.fmt(r["median_true_rank"]),
        })
    parts.append(liao.md_table(rows, ["detector", "subset", "variant", "R@1", "R@5", "R@10", "Top1%", "Median rank"]))
    parts += [
        "",
        "## 初步解读",
        "",
        "- PDF 原始空间规则是硬 mask，不能区分阈值内 pair 的 posterior 重叠强弱。",
        "- 当前 realistic sky step 使用 `d_sky = theta / sqrt(sigma_i^2 + sigma_j^2)`，会考虑每个事件自身定位误差，因此物理上更合理。",
        "- 如果 PDF hard mask 明显弱于 Stage3，说明后处理需要从固定面积阈值升级为 posterior-aware sky weighting。",
    ]
    DOC_PATH.write_text("\n".join(parts), encoding="utf-8")


def run() -> pd.DataFrame:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    grid = [0.25, 0.5, 1.0, 2.0, 4.0]
    for detector, mode in liao.JOBS:
        print("STAGE2B_PDF_RULE", detector, mode, flush=True)
        loaded = liao.load_job(detector, mode)
        cfg = loaded["cfg"]
        val_ds, val_raw, val_time, val_gt, val_scores = loaded["val"]
        test_ds, test_raw, test_time, test_gt, test_scores = loaded["test"]

        time_prior = liao.fit_time_lr_from_liao(detector, val_time, val_gt)
        val_sky = liao.make_observed_sky(detector, val_raw, val_time, seed=901000 + (0 if detector == "ET" else 1))
        test_sky = liao.make_observed_sky(detector, test_raw, test_time, seed=902000 + (0 if detector == "ET" else 1))
        area_thr = PDF_SKY_AREA_THRESHOLDS_DEG2[detector]
        theta_thr = math.sqrt(area_thr / math.pi)

        val_components = {
            "waveform": liao.row_z(val_scores),
            "pdf_sky_hard_mask": liao.row_z(pdf_sky_hard_mask_matrix(val_sky, area_thr)),
            "liao_time_lr": liao.row_z(liao.time_lr_score_matrix(val_time, time_prior)),
        }
        test_components = {
            "waveform": liao.row_z(test_scores),
            "pdf_sky_hard_mask": liao.row_z(pdf_sky_hard_mask_matrix(test_sky, area_thr)),
            "liao_time_lr": liao.row_z(liao.time_lr_score_matrix(test_time, time_prior)),
        }
        diag = liao.base_diag(test_ds, cfg, {
            "pdf_area_threshold_deg2": float(area_thr),
            "pdf_equiv_theta_threshold_deg": float(theta_thr),
            "observed_sky_label": liao.OBSERVED_SKY_CONFIG[detector]["label"],
            "test_a90_median_deg2": float(np.median(test_sky["sky_area90_deg2"])),
            "test_a90_p90_deg2": float(np.percentile(test_sky["sky_area90_deg2"], 90)),
            "liao_label": time_prior["liao_label"],
        })
        add_rows(rows, detector, mode, "pdf_sky_hard_mask_only", liao.evaluate_score(test_components["pdf_sky_hard_mask"], test_gt, test_ds.meta), diag)
        add_rows(rows, detector, mode, "liao_time_lr_only", liao.evaluate_score(test_components["liao_time_lr"], test_gt, test_ds.meta), diag)

        sky_lam, sky_val = liao.select_best_lambda(val_components, val_gt, val_ds.meta, grid, "pdf_sky_hard_mask")
        sky_score = test_components["waveform"] + sky_lam * test_components["pdf_sky_hard_mask"]
        np.fill_diagonal(sky_score, -np.inf)
        add_rows(rows, detector, mode, "waveform_plus_pdf_sky_hard_mask_val_selected", liao.evaluate_score(sky_score, test_gt, test_ds.meta), diag, {
            "lambda_pdf_sky": sky_lam,
            "val_selected_r@10": sky_val["overall"]["r@10"],
        })

        time_lam, time_val = liao.select_best_lambda(val_components, val_gt, val_ds.meta, grid, "liao_time_lr")
        time_score = test_components["waveform"] + time_lam * test_components["liao_time_lr"]
        np.fill_diagonal(time_score, -np.inf)
        add_rows(rows, detector, mode, "waveform_plus_liao_time_lr_val_selected", liao.evaluate_score(time_score, test_gt, test_ds.meta), diag, {
            "lambda_liao_time_lr": time_lam,
            "val_selected_r@10": time_val["overall"]["r@10"],
        })

        sky_lam2, time_lam2, val_metric = select_two_lambdas(
            val_components, val_gt, val_ds.meta, "pdf_sky_hard_mask", "liao_time_lr", grid
        )
        both = test_components["waveform"] + sky_lam2 * test_components["pdf_sky_hard_mask"] + time_lam2 * test_components["liao_time_lr"]
        np.fill_diagonal(both, -np.inf)
        add_rows(rows, detector, mode, "waveform_plus_pdf_sky_hard_mask_plus_liao_time_lr_val_selected", liao.evaluate_score(both, test_gt, test_ds.meta), diag, {
            "lambda_pdf_sky": sky_lam2,
            "lambda_liao_time_lr": time_lam2,
            "val_selected_r@10": val_metric["overall"]["r@10"],
        })
        pd.DataFrame(rows).to_csv(OUT_DIR / "stage2b_pdf_rule_time_sky_baseline_partial.csv", index=False)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "stage2b_pdf_rule_time_sky_baseline_summary.csv", index=False)
    write_doc(df)
    return df


def main() -> None:
    t0 = time.perf_counter()
    run()
    (OUT_DIR.parent / "stage2b_pdf_rule_time_sky_baseline_protocol_summary.json").write_text(json.dumps({
        "stage": "stage2b_pdf_rule_time_sky_baseline",
        "elapsed_s": float(time.perf_counter() - t0),
        "pdf_sky_area_thresholds_deg2": PDF_SKY_AREA_THRESHOLDS_DEG2,
        "note": "Naive baseline for the post-processing rules in 透镜识别流程.pdf.",
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
