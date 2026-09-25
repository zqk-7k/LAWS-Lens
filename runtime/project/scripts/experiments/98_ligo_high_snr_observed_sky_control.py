from __future__ import annotations

import importlib
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


base = importlib.import_module("scripts.experiments.80_mixed_sis_pm_catalog_modality_compare")
fresh = importlib.import_module("scripts.experiments.84_fresh50_full_catalog_ranking")
liao = importlib.import_module("scripts.experiments.88_liao_realistic_p1_p2_rerank")


DATA_ROOT = Path("data_generation/ligo_et_snr_like_2000_matchroots/LIGO")
WAVEFORM_ROOT = Path("runs/ligo_et_snr_like_2000_full_catalog_20260623")
OUT_DIR = Path("runs/ligo_high_snr_observed_sky_control_20260625")
GRID = [0.25, 0.5, 1.0, 2.0, 4.0]
DETECTOR = "LIGO"
MODES = ["noisy", "pure"]
MAIN_VARIANTS = [
    ("waveform_only", ["waveform"]),
    ("liao_time_lr_only", ["liao_time_lr"]),
    ("observed_sky_step_only", ["observed_sky_step"]),
    ("waveform_plus_liao_time_lr", ["waveform", "liao_time_lr"]),
    ("waveform_plus_observed_sky_step", ["waveform", "observed_sky_step"]),
    ("liao_time_lr_plus_observed_sky_step", ["liao_time_lr", "observed_sky_step"]),
    ("waveform_plus_liao_time_lr_plus_observed_sky_step", ["waveform", "liao_time_lr", "observed_sky_step"]),
]


def configure_modules() -> None:
    base.ROOTS[("SIS", DETECTOR)] = DATA_ROOT
    base.ROOTS[("PM", DETECTOR)] = DATA_ROOT
    fresh.OUT_ROOT = WAVEFORM_ROOT
    fresh.ENCODER_ROOT = WAVEFORM_ROOT / "fresh_mixed_encoders"
    fresh.JOBS = [(DETECTOR, mode) for mode in MODES]
    liao.OUT_ROOT = OUT_DIR
    liao.ENCODER_ROOT = fresh.ENCODER_ROOT
    liao.JOBS = [(DETECTOR, mode) for mode in MODES]


def score_from_weights(components: dict[str, np.ndarray], keys: list[str], weights: dict[str, float]) -> np.ndarray:
    score = np.zeros_like(components[keys[0]], dtype=np.float32)
    for key in keys:
        score = score + float(weights[key]) * components[key]
    np.fill_diagonal(score, -np.inf)
    return score.astype(np.float32)


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
    if best_metrics is None:
        raise RuntimeError(f"no validation metrics selected for {keys}")
    return best_weights, best_metrics


def add_rows(
    rows: list[dict],
    mode: str,
    variant: str,
    metrics: dict[str, dict],
    diag: dict,
    extra: dict | None = None,
) -> None:
    extra = extra or {}
    for subset, values in metrics.items():
        rows.append({
            "detector": DETECTOR,
            "data_mode": mode,
            "experiment": "ligo_h1l1_high_snr_observed_sky_control",
            "stage": "observed_sky_proxy_modality_combinations",
            "variant": variant,
            "subset": subset,
            **diag,
            **extra,
            **values,
        })


def sky_diag_rows(mode: str, split: str, sky: pd.DataFrame, time_obs: pd.DataFrame) -> dict:
    snr = time_obs["snr"].to_numpy(dtype=np.float64)
    a90 = sky["sky_area90_deg2"].to_numpy(dtype=np.float64)
    sigma = sky["sky_sigma_rad"].to_numpy(dtype=np.float64)
    return {
        "detector": DETECTOR,
        "data_mode": mode,
        "split": split,
        "n_events": int(len(sky)),
        "sky_scenario": str(sky["scenario"].iloc[0]),
        "sky_model": str(sky["sky_model"].iloc[0]),
        "sky_sampling": str(sky["sky_sampling"].iloc[0]),
        "snr_for_sky_mode": str(sky["snr_for_sky_mode"].iloc[0]),
        "snr_median": float(np.median(snr)),
        "snr_q10": float(np.percentile(snr, 10)),
        "snr_q90": float(np.percentile(snr, 90)),
        "snr_q99": float(np.percentile(snr, 99)),
        "a90_median_deg2": float(np.median(a90)),
        "a90_p90_deg2": float(np.percentile(a90, 90)),
        "a90_q10_deg2": float(np.percentile(a90, 10)),
        "a90_q99_deg2": float(np.percentile(a90, 99)),
        "sky_sigma_median_rad": float(np.median(sigma)),
        "uses_true_sky_as_input": False,
        "true_sky_usage": "simulation_only_to_sample_observed_ra_dec",
    }


def fmt(value) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, str):
        return value
    return f"{float(value):.4f}".rstrip("0").rstrip(".")


def run_mode(mode: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    out_mode = OUT_DIR / mode
    out_mode.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    sky_rows: list[dict] = []

    print("LIGO_HIGH_SNR_OBSERVED_SKY_CONTROL", mode, flush=True)
    loaded = liao.load_job(DETECTOR, mode)
    cfg = loaded["cfg"]
    val_ds, val_raw, val_time, val_gt, val_scores = loaded["val"]
    test_ds, test_raw, test_time, test_gt, test_scores = loaded["test"]

    prior = liao.fit_time_lr_from_liao(DETECTOR, val_time, val_gt)
    seed_offset = 0 if mode == "noisy" else 1000
    val_sky = liao.make_observed_sky(DETECTOR, val_raw, val_time, seed=731000 + seed_offset)
    test_sky = liao.make_observed_sky(DETECTOR, test_raw, test_time, seed=732000 + seed_offset)
    val_sky.to_csv(out_mode / f"{DETECTOR}_{mode}_val_observed_sky_audit.csv", index=False)
    test_sky.to_csv(out_mode / f"{DETECTOR}_{mode}_test_observed_sky_audit.csv", index=False)
    sky_rows.append(sky_diag_rows(mode, "val", val_sky, val_time))
    sky_rows.append(sky_diag_rows(mode, "test", test_sky, test_time))

    val_step, _, _ = liao.observed_sky_score_matrices(val_sky)
    test_step, _, _ = liao.observed_sky_score_matrices(test_sky)

    val_components = {
        "waveform": liao.row_z(val_scores),
        "liao_time_lr": liao.row_z(liao.time_lr_score_matrix(val_time, prior)),
        "observed_sky_step": liao.row_z(val_step),
    }
    test_components = {
        "waveform": liao.row_z(test_scores),
        "liao_time_lr": liao.row_z(liao.time_lr_score_matrix(test_time, prior)),
        "observed_sky_step": liao.row_z(test_step),
    }

    diag = liao.base_diag(test_ds, cfg, {
        "catalog": "ligo_et_snr_like_2000_mixed_SIS_PM_unlensed_full_catalog",
        "candidate_kind": "full_catalog",
        "control_label": "H1-L1 high-SNR control at ET-like SNR",
        "data_root": str(DATA_ROOT),
        "waveform_cache_root": str(WAVEFORM_ROOT),
        "liao_label": prior["liao_label"],
        "liao_delay_count": prior["liao_delay_count"],
        "observed_sky_label": liao.OBSERVED_SKY_CONFIG[DETECTOR]["label"],
        "sky_scenario": "LIGO_HL",
        "a90_ref_deg2": liao.OBSERVED_SKY_CONFIG[DETECTOR]["a90_ref_deg2"],
        "test_snr_median": sky_rows[-1]["snr_median"],
        "test_snr_q90": sky_rows[-1]["snr_q90"],
        "test_a90_median_deg2": sky_rows[-1]["a90_median_deg2"],
        "test_a90_p90_deg2": sky_rows[-1]["a90_p90_deg2"],
        "weight_grid": str(GRID),
        "sky_input_policy": "observed sky proxy only; true ra/dec used only to sample noisy observed sky center",
    })

    for variant, keys in MAIN_VARIANTS:
        print("OBSERVED_SKY_VARIANT", mode, variant, flush=True)
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
        for key in ["waveform", "liao_time_lr", "observed_sky_step"]:
            if key in weights:
                extra[f"lambda_{key}"] = float(weights[key])
        add_rows(rows, mode, variant, liao.evaluate_score(test_score, test_gt, test_ds.meta), diag, extra)
        pd.DataFrame(rows).to_csv(out_mode / "observed_sky_control_partial.csv", index=False)

    df = pd.DataFrame(rows)
    sky_df = pd.DataFrame(sky_rows)
    df.to_csv(out_mode / "observed_sky_control_summary.csv", index=False)
    sky_df.to_csv(out_mode / "observed_sky_a90_snr_diagnostics.csv", index=False)
    return df, sky_df


def maybe_load_reference(path: Path, label: str, source_kind: str) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    keep = df[
        df["subset"].eq("overall")
        & df["variant"].isin([
            "waveform_only",
            "liao_time_lr_only",
            "observed_sky_step_only",
            "waveform_plus_liao_time_lr_plus_observed_sky_step",
        ])
    ].copy()
    if keep.empty:
        return keep
    keep["comparison_dataset"] = label
    keep["source_kind"] = source_kind
    return keep


def write_comparison(high: pd.DataFrame) -> pd.DataFrame:
    refs = [
        maybe_load_reference(
            Path("runs/et3_liao_realistic_p1_p2_rerank_20260616/stage7_modality_combinations/stage7_modality_combinations_summary.csv"),
            "ET-3 noisy main",
            "main",
        ),
        maybe_load_reference(
            Path("runs/ligo_h1l1_liao_realistic_p1_p2_rerank_20260617/stage7_modality_combinations/stage7_modality_combinations_summary.csv"),
            "LIGO H1-L1 low-SNR noisy",
            "baseline",
        ),
    ]
    high_overall = high[
        high["subset"].eq("overall")
        & high["variant"].isin([
            "waveform_only",
            "liao_time_lr_only",
            "observed_sky_step_only",
            "waveform_plus_liao_time_lr_plus_observed_sky_step",
        ])
    ].copy()
    high_overall["comparison_dataset"] = np.where(
        high_overall["data_mode"].eq("pure"),
        "LIGO H1-L1 high-SNR pure diagnostic",
        "LIGO H1-L1 high-SNR noisy control",
    )
    high_overall["source_kind"] = "control"
    combined = pd.concat([x for x in [*refs, high_overall] if not x.empty], ignore_index=True, sort=False)
    if combined.empty:
        return combined
    cols = [
        "comparison_dataset",
        "source_kind",
        "detector",
        "data_mode",
        "variant",
        "r@1",
        "r@5",
        "r@10",
        "top_1pct",
        "median_true_rank",
        "test_snr_median",
        "test_snr_q90",
        "test_a90_median_deg2",
        "test_a90_p90_deg2",
    ]
    available = [col for col in cols if col in combined.columns]
    comparison = combined[available].copy()
    comparison.to_csv(OUT_DIR / "et3_ligo_low_snr_high_snr_observed_sky_comparison.csv", index=False)
    return comparison


def write_report(summary: pd.DataFrame, sky_diag: pd.DataFrame, comparison: pd.DataFrame) -> None:
    lines = [
        "# LIGO H1-L1 high-SNR observed-sky proxy control",
        "",
        f"Generated: {pd.Timestamp.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "## Scope",
        "",
        "This reruns the LIGO ET-like SNR 2000 pilot under the corrected observed-sky proxy protocol. It reuses the existing waveform encoders and cached similarity matrices from `runs/ligo_et_snr_like_2000_full_catalog_20260623`; it does not use predicted-sky overlap or true-sky overlap as a main result.",
        "",
        "True sky coordinates are used only inside the simulator to draw a noisy observed sky center. Ranking inputs are limited to waveform similarity, observed trigger time through the Liao time-delay likelihood-ratio score, and the observed-sky proxy (`ra_obs`, `dec_obs`, `sky_area90_deg2`, `sky_sigma_rad`).",
        "",
        "## Observed-sky / SNR diagnostics",
        "",
        "| mode | split | n events | SNR median | SNR q90 | A90 median deg2 | A90 p90 deg2 | scenario |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for _, row in sky_diag.iterrows():
        lines.append(
            f"| {row['data_mode']} | {row['split']} | {int(row['n_events'])} | {fmt(row['snr_median'])} | {fmt(row['snr_q90'])} | "
            f"{fmt(row['a90_median_deg2'])} | {fmt(row['a90_p90_deg2'])} | {row['sky_scenario']} |"
        )

    lines += [
        "",
        "## Main variants",
        "",
        "| mode | subset | variant | R@1 | R@5 | R@10 | Top1% | median rank | weights |",
        "|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    main = summary[summary["subset"].isin(["overall", "SIS", "PM"])].copy()
    for _, row in main.sort_values(["data_mode", "variant", "subset"]).iterrows():
        weights = []
        for key in ["waveform", "liao_time_lr", "observed_sky_step"]:
            col = f"lambda_{key}"
            if col in row and pd.notna(row[col]):
                weights.append(f"{key}={fmt(row[col])}")
        lines.append(
            f"| {row['data_mode']} | {row['subset']} | {row['variant']} | {fmt(row['r@1'])} | "
            f"{fmt(row['r@5'])} | {fmt(row['r@10'])} | {fmt(row['top_1pct'])} | "
            f"{fmt(row['median_true_rank'])} | {', '.join(weights)} |"
        )

    if not comparison.empty:
        lines += [
            "",
            "## ET-3 / low-SNR LIGO / high-SNR LIGO comparison",
            "",
            "| dataset | mode | variant | R@1 | R@10 | Top1% | median rank |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
        for _, row in comparison.sort_values(["comparison_dataset", "variant"]).iterrows():
            lines.append(
                f"| {row['comparison_dataset']} | {row.get('data_mode', '')} | {row['variant']} | "
                f"{fmt(row['r@1'])} | {fmt(row['r@10'])} | {fmt(row['top_1pct'])} | {fmt(row['median_true_rank'])} |"
            )

    lines += [
        "",
        "## Files",
        "",
        "- Summary: `observed_sky_control_summary.csv`",
        "- A90/SNR diagnostics: `observed_sky_a90_snr_diagnostics.csv`",
        "- Observed-sky audit tables: `{mode}/LIGO_{mode}_{split}_observed_sky_audit.csv`",
        "- Comparison table: `et3_ligo_low_snr_high_snr_observed_sky_comparison.csv`",
    ]
    (OUT_DIR / "ligo_high_snr_observed_sky_control_report_cn.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    t0 = time.perf_counter()
    configure_modules()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_summary = []
    all_sky = []
    for mode in MODES:
        df, sky = run_mode(mode)
        all_summary.append(df)
        all_sky.append(sky)
    summary = pd.concat(all_summary, ignore_index=True)
    sky_diag = pd.concat(all_sky, ignore_index=True)
    summary.to_csv(OUT_DIR / "observed_sky_control_summary.csv", index=False)
    sky_diag.to_csv(OUT_DIR / "observed_sky_a90_snr_diagnostics.csv", index=False)
    comparison = write_comparison(summary)
    write_report(summary, sky_diag, comparison)
    protocol = {
        "elapsed_s": float(time.perf_counter() - t0),
        "data_root": str(DATA_ROOT),
        "waveform_cache_root": str(WAVEFORM_ROOT),
        "out_dir": str(OUT_DIR),
        "modes": MODES,
        "variants": [{"name": name, "components": keys} for name, keys in MAIN_VARIANTS],
        "weight_grid": GRID,
        "sky_protocol": "observed-sky proxy only; no predicted sky overlap and no true sky overlap in main variants",
        "sky_scenario": "LIGO_HL",
    }
    (OUT_DIR / "protocol_summary.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print(summary.to_string(index=False), flush=True)
    print(f"WROTE {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
