from __future__ import annotations

import importlib
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


runner = importlib.import_module("scripts.experiments.90_et3_full_experiment_runner")
fresh, liao, _pdf = runner.configure_modules()

OUT_DIR = runner.RERANK_ROOT / "stage7_modality_combinations"
GRID = [0.25, 0.5, 1.0, 2.0, 4.0]


def add_rows(rows: list[dict], variant: str, metrics: dict[str, dict], diag: dict, extra: dict | None = None) -> None:
    extra = extra or {}
    for subset, values in metrics.items():
        rows.append({
            "detector": "ET3",
            "data_mode": "noisy",
            "stage": "stage7_modality_combinations",
            "variant": variant,
            "subset": subset,
            **diag,
            **extra,
            **values,
        })


def score_from_weights(components: dict[str, np.ndarray], keys: list[str], weights: dict[str, float]) -> np.ndarray:
    score = np.zeros_like(components[keys[0]], dtype=np.float32)
    for key in keys:
        score = score + float(weights[key]) * components[key]
    np.fill_diagonal(score, -np.inf)
    return score


def select_weights(
    val_components: dict[str, np.ndarray],
    val_gt: np.ndarray,
    val_meta: list[dict],
    keys: list[str],
) -> tuple[dict[str, float], dict]:
    fixed = {"waveform": 1.0} if "waveform" in keys else {}
    tune_keys = [key for key in keys if key not in fixed]
    best_weights = {key: 1.0 for key in keys}
    best_weights.update(fixed)
    best_metrics = None
    best_key = (-1.0, -1.0, -1.0, -1.0)
    for values in itertools.product(GRID, repeat=len(tune_keys)):
        weights = dict(fixed)
        weights.update(dict(zip(tune_keys, values)))
        score = score_from_weights(val_components, keys, weights)
        metrics = liao.evaluate_score(score, val_gt, val_meta)
        rank_key = (
            metrics["overall"]["r@10"],
            metrics["overall"]["r@5"],
            metrics["overall"]["r@1"],
            metrics["overall"]["top_1pct"],
        )
        if rank_key > best_key:
            best_key = rank_key
            best_weights = weights
            best_metrics = metrics
    return best_weights, best_metrics


def run() -> pd.DataFrame:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    print("STAGE7_MODALITY_COMBINATIONS ET3 noisy", flush=True)
    loaded = liao.load_job("ET3", "noisy")
    cfg = loaded["cfg"]
    val_ds, val_raw, val_time, val_gt, val_scores = loaded["val"]
    test_ds, test_raw, test_time, test_gt, test_scores = loaded["test"]

    prior = liao.fit_time_lr_from_liao("ET3", val_time, val_gt)
    val_sky = liao.make_observed_sky("ET3", val_raw, val_time, seed=701000)
    test_sky = liao.make_observed_sky("ET3", test_raw, test_time, seed=702000)
    val_step, _, val_log = liao.observed_sky_score_matrices(val_sky)
    test_step, _, test_log = liao.observed_sky_score_matrices(test_sky)

    val_components = {
        "waveform": liao.row_z(val_scores),
        "raw_time": liao.row_z(liao.raw_time_score_matrix(val_time)),
        "liao_time_lr": liao.row_z(liao.time_lr_score_matrix(val_time, prior)),
        "observed_sky_step": liao.row_z(val_step),
        "observed_sky_log_overlap": liao.row_z(val_log),
    }
    test_components = {
        "waveform": liao.row_z(test_scores),
        "raw_time": liao.row_z(liao.raw_time_score_matrix(test_time)),
        "liao_time_lr": liao.row_z(liao.time_lr_score_matrix(test_time, prior)),
        "observed_sky_step": liao.row_z(test_step),
        "observed_sky_log_overlap": liao.row_z(test_log),
    }

    diag = liao.base_diag(test_ds, cfg, {
        "liao_label": prior["liao_label"],
        "liao_delay_count": prior["liao_delay_count"],
        "observed_sky_label": liao.OBSERVED_SKY_CONFIG["ET3"]["label"],
        "a90_ref_deg2": liao.OBSERVED_SKY_CONFIG["ET3"]["a90_ref_deg2"],
        "test_a90_median_deg2": float(np.median(test_sky["sky_area90_deg2"])),
        "test_a90_p90_deg2": float(np.percentile(test_sky["sky_area90_deg2"], 90)),
        "weight_grid": str(GRID),
    })

    variants = [
        ("waveform_only", ["waveform"]),
        ("raw_time_only", ["raw_time"]),
        ("liao_time_lr_only", ["liao_time_lr"]),
        ("observed_sky_step_only", ["observed_sky_step"]),
        ("observed_sky_log_overlap_only", ["observed_sky_log_overlap"]),
        ("waveform_plus_raw_time", ["waveform", "raw_time"]),
        ("waveform_plus_liao_time_lr", ["waveform", "liao_time_lr"]),
        ("waveform_plus_observed_sky_step", ["waveform", "observed_sky_step"]),
        ("waveform_plus_observed_sky_log_overlap", ["waveform", "observed_sky_log_overlap"]),
        ("raw_time_plus_observed_sky_step", ["raw_time", "observed_sky_step"]),
        ("raw_time_plus_observed_sky_log_overlap", ["raw_time", "observed_sky_log_overlap"]),
        ("liao_time_lr_plus_observed_sky_step", ["liao_time_lr", "observed_sky_step"]),
        ("liao_time_lr_plus_observed_sky_log_overlap", ["liao_time_lr", "observed_sky_log_overlap"]),
        ("waveform_plus_raw_time_plus_observed_sky_step", ["waveform", "raw_time", "observed_sky_step"]),
        ("waveform_plus_raw_time_plus_observed_sky_log_overlap", ["waveform", "raw_time", "observed_sky_log_overlap"]),
        ("waveform_plus_liao_time_lr_plus_observed_sky_step", ["waveform", "liao_time_lr", "observed_sky_step"]),
        ("waveform_plus_liao_time_lr_plus_observed_sky_log_overlap", ["waveform", "liao_time_lr", "observed_sky_log_overlap"]),
    ]

    for variant, keys in variants:
        print("STAGE7_VARIANT", variant, flush=True)
        if len(keys) == 1:
            weights = {keys[0]: 1.0}
            val_metric = liao.evaluate_score(val_components[keys[0]], val_gt, val_ds.meta)
        else:
            weights, val_metric = select_weights(val_components, val_gt, val_ds.meta, keys)
        test_score = score_from_weights(test_components, keys, weights)
        extra = {
            "component_keys": ",".join(keys),
            "val_selected_r@1": val_metric["overall"]["r@1"],
            "val_selected_r@5": val_metric["overall"]["r@5"],
            "val_selected_r@10": val_metric["overall"]["r@10"],
        }
        for key in ["waveform", "raw_time", "liao_time_lr", "observed_sky_step", "observed_sky_log_overlap"]:
            if key in weights:
                extra[f"lambda_{key}"] = float(weights[key])
        add_rows(rows, variant, liao.evaluate_score(test_score, test_gt, test_ds.meta), diag, extra)
        pd.DataFrame(rows).to_csv(OUT_DIR / "stage7_modality_combinations_partial.csv", index=False)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "stage7_modality_combinations_summary.csv", index=False)
    write_doc(df)
    return df


def fmt(value) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.4f}".rstrip("0").rstrip(".")


def write_doc(df: pd.DataFrame) -> None:
    doc = OUT_DIR / "stage7_modality_combinations_report_cn.md"
    overall = df[df["subset"].eq("overall")].sort_values(["r@10", "r@1"], ascending=False)
    sispm = df[df["subset"].isin(["SIS", "PM"])].sort_values(["variant", "subset"])
    lines = [
        "# ET3 SIS/PM modality 组合实验",
        "",
        "本实验基于 ET3 noisy full-catalog 缓存结果，不重新训练 encoder；系统比较 waveform、time、observed sky 及其组合。",
        "",
        "## Overall 排名",
        "",
        "| variant | R@1 | R@5 | R@10 | Top1% | median rank | weights |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for _, r in overall.iterrows():
        weights = []
        for key in ["waveform", "raw_time", "liao_time_lr", "observed_sky_step", "observed_sky_log_overlap"]:
            col = f"lambda_{key}"
            if col in r and pd.notna(r[col]):
                weights.append(f"{key}={fmt(r[col])}")
        lines.append(
            f"| {r.variant} | {fmt(r['r@1'])} | {fmt(r['r@5'])} | {fmt(r['r@10'])} | "
            f"{fmt(r['top_1pct'])} | {fmt(r['median_true_rank'])} | {', '.join(weights)} |"
        )
    lines += [
        "",
        "## SIS / PM 分解",
        "",
        "| variant | subset | R@1 | R@5 | R@10 | Top1% | median rank |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for _, r in sispm.iterrows():
        lines.append(
            f"| {r.variant} | {r.subset} | {fmt(r['r@1'])} | {fmt(r['r@5'])} | "
            f"{fmt(r['r@10'])} | {fmt(r['top_1pct'])} | {fmt(r['median_true_rank'])} |"
        )
    doc.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    run()
