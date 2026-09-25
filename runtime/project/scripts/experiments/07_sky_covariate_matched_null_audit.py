#!/usr/bin/env python3
"""Covariate-matched synthetic-null audit for the O3/O4a sky channel.

The script uses existing v9.3 products only.  It repeatedly maps each strict
real-catalog event to a distinct synthetic validation event with similar sky
area, sky information, SNR, and (where applicable) observing run.  It then
compares the induced synthetic non-companion pair-score tail with the real
catalog tail at the same Nside=64.  This identifies how much of the observed
domain shift is explained by event-level covariates, without fitting on real
candidate ranks or changing any v9.3 score.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import tarfile
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance


DEPLOYMENTS = ("gwtc3", "gwtc4")
REPLICATES = 500
NEIGHBOURS = 16
BASE_SEED = 2026082307
MATCHING_SCHEMES = {
    "run_only_random": (),
    "localization": ("log_area90", "kl"),
    "localization_and_snr": ("log_area90", "kl", "log_snr"),
}


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def canonical_pair(a: object, b: object) -> str:
    x, y = str(a), str(b)
    return f"{x}--{y}" if x <= y else f"{y}--{x}"


def pair_columns(frame: pd.DataFrame) -> tuple[str, str]:
    for left, right in (("event_i", "event_j"), ("event_1", "event_2"), ("event_a", "event_b")):
        if left in frame and right in frame:
            return left, right
    raise KeyError(frame.columns.tolist())


def add_pair_key(frame: pd.DataFrame) -> pd.DataFrame:
    left, right = pair_columns(frame)
    out = frame.copy()
    out["pair_key"] = [canonical_pair(a, b) for a, b in zip(out[left], out[right])]
    return out


def strict_pairs(seed_dir: Path, resolution: pd.DataFrame) -> pd.DataFrame:
    score_path = seed_dir / "real_pair_scores_strict_h1l1_bbh_candidate_three_channel_strict_positive_v81.parquet"
    if not score_path.exists():
        matches = sorted(seed_dir.glob("real_pair_scores_strict_h1l1_bbh*.parquet"))
        if not matches:
            raise FileNotFoundError(f"No strict pair table in {seed_dir}")
        score_path = matches[0]
    strict = add_pair_key(pd.read_parquet(score_path))
    keys = set(strict["pair_key"])
    return resolution.loc[resolution["pair_key"].isin(keys)].copy()


def infer_run(gps: float, deployment: str) -> str:
    if deployment == "gwtc4":
        return "O4a"
    boundaries = (
        ("O1", 1126051217.0, 1137254417.0),
        ("O2", 1164556817.0, 1187733618.0),
        ("O3a", 1238166018.0, 1253977218.0),
        ("O3b", 1256655618.0, 1269363618.0),
    )
    for name, start, end in boundaries:
        if start <= gps <= end:
            return name
    return "outside_run"


def find_column(frame: pd.DataFrame, names: tuple[str, ...]) -> str:
    for name in names:
        if name in frame:
            return name
    raise KeyError(f"None of {names} in {frame.columns.tolist()}")


def load_event_tables(
    project: Path,
    v93_seed: Path,
    v7_seed: Path,
    deployment: str,
    split: str,
    strict_event_names: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame, list[Path]]:
    split_tag = "val" if split == "validation" else "test"
    synthetic_path = v7_seed / "results" / f"mixed_{split_tag}_synthetic_events_v7.parquet"
    diagnostic_path = v93_seed / f"synthetic_{split}_event_sky_diagnostics_v81.parquet"
    synthetic = pd.read_parquet(synthetic_path)
    diagnostic = pd.read_parquet(diagnostic_path)
    synthetic = synthetic.merge(diagnostic, on="idx", how="inner", validate="one_to_one", suffixes=("", "_diag"))
    synthetic["run"] = [infer_run(float(gps), deployment) for gps in synthetic["gps_obs"]]
    area_col = find_column(synthetic, ("area90_deg2", "sky_area90_deg2"))
    kl_col = find_column(synthetic, ("kl_to_uniform_nats", "sky_kl_nats"))
    synthetic = synthetic.assign(
        log_area90=np.log(np.maximum(synthetic[area_col].to_numpy(float), 1e-6)),
        kl=synthetic[kl_col].to_numpy(float),
        log_snr=np.log(np.maximum(synthetic["snr"].to_numpy(float), 1e-6)),
    )

    real_diag_path = v93_seed / "real_event_sky_diagnostics_v81.parquet"
    real = pd.read_parquet(real_diag_path)
    source_run = (
        project / "runs" / "real_gwtc_lensing_search_20260625"
        if deployment == "gwtc3"
        else project / "runs" / "real_gwtc34_lensing_search_20260629_full_o4"
    )
    manifest_path = source_run / "data" / "event_manifest.csv"
    manifest = pd.read_csv(manifest_path)
    real = real.merge(
        manifest[["event_name", "run", "network_snr", "detectors_available"]],
        on="event_name",
        how="left",
        validate="one_to_one",
    )
    real = real.loc[real["event_name"].astype(str).isin(strict_event_names)].copy()
    real_area_col = find_column(real, ("area90_nside1024_deg2", "area90_deg2", "sky_area90_deg2"))
    real_kl_col = find_column(real, ("kl_to_uniform_nats", "sky_kl_nats"))
    real = real.assign(
        log_area90=np.log(np.maximum(real[real_area_col].to_numpy(float), 1e-6)),
        kl=real[real_kl_col].to_numpy(float),
        log_snr=np.log(np.maximum(real["network_snr"].to_numpy(float), 1e-6)),
    )
    return synthetic, real, [synthetic_path, diagnostic_path, real_diag_path, manifest_path]


def score_matrix(pair_frame: pd.DataFrame, n_events: int) -> np.ndarray:
    score_col = find_column(pair_frame, ("sky_log_bayes_factor_raw", "sky_score", "sky_log_bf"))
    matrix = np.full((n_events, n_events), np.nan, dtype=np.float64)
    ii = pair_frame["idx_i"].to_numpy(np.int32)
    jj = pair_frame["idx_j"].to_numpy(np.int32)
    values = pair_frame[score_col].to_numpy(float)
    matrix[ii, jj] = values
    matrix[jj, ii] = values
    np.fill_diagonal(matrix, np.nan)
    return matrix


def robust_scale(synthetic: pd.DataFrame, real: pd.DataFrame, features: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    if not features:
        return np.empty((len(synthetic), 0)), np.empty((len(real), 0))
    combined = np.vstack([synthetic[list(features)].to_numpy(float), real[list(features)].to_numpy(float)])
    median = np.nanmedian(combined, axis=0)
    q25, q75 = np.nanquantile(combined, [0.25, 0.75], axis=0)
    scale = np.maximum(q75 - q25, 1e-6)
    return (
        (synthetic[list(features)].to_numpy(float) - median) / scale,
        (real[list(features)].to_numpy(float) - median) / scale,
    )


def candidate_lists(
    synthetic: pd.DataFrame,
    real: pd.DataFrame,
    features: tuple[str, ...],
    neighbours: int,
) -> tuple[list[np.ndarray], list[np.ndarray], pd.DataFrame]:
    synthetic_x, real_x = robust_scale(synthetic, real, features)
    candidates: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    audit_rows: list[dict[str, object]] = []
    for real_index, row in real.reset_index(drop=True).iterrows():
        same_run = np.flatnonzero(synthetic["run"].astype(str).to_numpy() == str(row["run"]))
        pool = same_run if same_run.size >= neighbours else np.arange(len(synthetic))
        if features:
            distances = np.linalg.norm(synthetic_x[pool] - real_x[real_index], axis=1)
            order = np.argsort(distances)[: min(neighbours, pool.size)]
            selected = pool[order]
            selected_distance = distances[order]
            positive = selected_distance[selected_distance > 0]
            temperature = float(np.median(positive)) if positive.size else 1.0
            weights = np.exp(-0.5 * (selected_distance / max(temperature, 1e-6)) ** 2)
        else:
            selected = pool
            selected_distance = np.zeros(pool.size)
            weights = np.ones(pool.size)
        weights = weights / weights.sum()
        candidates.append(selected.astype(np.int32))
        probabilities.append(weights.astype(np.float64))
        audit_rows.append(
            {
                "event_name": row["event_name"],
                "run": row["run"],
                "candidate_pool_size": int(pool.size),
                "neighbours_used": int(selected.size),
                "nearest_distance": float(selected_distance[0]),
                "farthest_selected_distance": float(selected_distance[-1]),
            }
        )
    return candidates, probabilities, pd.DataFrame(audit_rows)


def draw_unique_mapping(
    candidates: list[np.ndarray],
    probabilities: list[np.ndarray],
    rng: np.random.Generator,
) -> np.ndarray:
    order = np.argsort([len(x) for x in candidates])
    mapping = np.full(len(candidates), -1, dtype=np.int32)
    used: set[int] = set()
    for real_index in order:
        choices = candidates[real_index]
        probs = probabilities[real_index]
        available = np.array([idx not in used for idx in choices], dtype=bool)
        if np.any(available):
            local_choices = choices[available]
            local_probs = probs[available]
            local_probs = local_probs / local_probs.sum()
        else:
            local_choices = choices
            local_probs = probs
        chosen = int(rng.choice(local_choices, p=local_probs))
        mapping[real_index] = chosen
        used.add(chosen)
    return mapping


def summarize(values: np.ndarray) -> dict[str, float]:
    values = values[np.isfinite(values)]
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
        "median": float(np.median(values)),
        "q025": float(np.quantile(values, 0.025)),
        "q25": float(np.quantile(values, 0.25)),
        "q75": float(np.quantile(values, 0.75)),
        "q975": float(np.quantile(values, 0.975)),
    }


def run_seed(
    project: Path,
    v93_root: Path,
    deployment: str,
    seed_dir: Path,
    resolution: pd.DataFrame,
    output: Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[Path], dict[str, np.ndarray]]:
    seed = seed_dir.name.removeprefix("seed_")
    v7_seed = project / "results" / "real_noise_injection_v7_peak2s_formal_20260722" / deployment / seed_dir.name
    strict = strict_pairs(seed_dir, resolution)
    left, right = pair_columns(strict)
    strict_events = sorted(set(strict[left].astype(str)) | set(strict[right].astype(str)))
    strict_event_set = set(strict_events)
    real_index = {event: idx for idx, event in enumerate(strict_events)}
    pair_ii = strict[left].astype(str).map(real_index).to_numpy(np.int32)
    pair_jj = strict[right].astype(str).map(real_index).to_numpy(np.int32)
    real_scores = strict["sky_log_bf_nside64"].to_numpy(float)

    pair_path = seed_dir / "fusion_validation_pairs_v81.parquet"
    pair_frame = pd.read_parquet(pair_path)
    true_mask = pair_frame["is_true_pair"].fillna(False).astype(bool).to_numpy()
    score_col = find_column(pair_frame, ("sky_log_bayes_factor_raw", "sky_score", "sky_log_bf"))
    validation_null = pair_frame.loc[~true_mask, score_col].to_numpy(float)
    thresholds = {q: float(np.quantile(validation_null, q)) for q in (0.99, 0.999)}

    synthetic, real, input_paths = load_event_tables(
        project, seed_dir, v7_seed, deployment, "validation", strict_event_set
    )
    if set(real["event_name"].astype(str)) != strict_event_set:
        missing = strict_event_set - set(real["event_name"].astype(str))
        raise RuntimeError(f"Missing strict real-event diagnostics: {sorted(missing)}")
    real = real.set_index("event_name").loc[strict_events].reset_index()
    matrix = score_matrix(pair_frame, len(synthetic))
    synthetic_pair_id = synthetic["pair_id"].to_numpy(np.int64)
    synthetic_tag = synthetic["tag"].astype(str).to_numpy()

    replicate_rows: list[dict[str, object]] = []
    matching_rows: list[dict[str, object]] = []
    plot_distributions: dict[str, np.ndarray] = {"real": real_scores, "injection_null": validation_null}
    rng = np.random.default_rng(BASE_SEED + int(seed[-3:]))
    for scheme, features in MATCHING_SCHEMES.items():
        candidates, probabilities, match_audit = candidate_lists(synthetic, real, features, NEIGHBOURS)
        match_audit.insert(0, "scheme", scheme)
        match_audit.insert(0, "seed", seed)
        match_audit.insert(0, "deployment", deployment)
        matching_rows.extend(match_audit.to_dict("records"))
        retained_for_plot: list[np.ndarray] = []
        for replicate in range(REPLICATES):
            mapping = draw_unique_mapping(candidates, probabilities, rng)
            si, sj = mapping[pair_ii], mapping[pair_jj]
            values = matrix[si, sj]
            accidental_true = (
                (synthetic_pair_id[si] == synthetic_pair_id[sj])
                & (synthetic_tag[si] != "U")
                & (synthetic_tag[sj] != "U")
            )
            valid = np.isfinite(values) & ~accidental_true
            values = values[valid]
            if replicate < 20:
                retained_for_plot.append(values)
            ks = ks_2samp(values, real_scores)
            row: dict[str, object] = {
                "deployment": deployment,
                "seed": seed,
                "scheme": scheme,
                "replicate": replicate,
                "n_pairs": int(values.size),
                "accidental_true_pairs_removed": int(accidental_true.sum()),
                "ks_statistic_vs_real_nside64": float(ks.statistic),
                "ks_pvalue_vs_real_nside64": float(ks.pvalue),
                "wasserstein_vs_real_nside64_nats": float(wasserstein_distance(values, real_scores)),
                "matched_q99_nats": float(np.quantile(values, 0.99)),
                "real_q99_nats": float(np.quantile(real_scores, 0.99)),
                "q99_gap_to_real_nats": float(np.quantile(real_scores, 0.99) - np.quantile(values, 0.99)),
            }
            for q, threshold in thresholds.items():
                rate = float(np.mean(values > threshold))
                reference_rate = float(np.mean(validation_null > threshold))
                row[f"exceedance_at_validation_q{1000*q:04.0f}"] = rate
                row[f"tail_inflation_at_validation_q{1000*q:04.0f}"] = rate / reference_rate
            replicate_rows.append(row)
        plot_distributions[scheme] = np.concatenate(retained_for_plot)
    return replicate_rows, matching_rows, [pair_path, *input_paths], plot_distributions


def aggregate(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    metric_columns = [
        c for c in frame.columns if c not in {"deployment", "seed", "scheme", "replicate"} and pd.api.types.is_numeric_dtype(frame[c])
    ]
    for keys, group in frame.groupby(["deployment", "seed", "scheme"], sort=True):
        row: dict[str, object] = dict(zip(("deployment", "seed", "scheme"), keys))
        row["n_replicates"] = len(group)
        for column in metric_columns:
            stats = summarize(group[column].to_numpy(float))
            for stat_name, value in stats.items():
                row[f"{column}_{stat_name}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def plot(summary: pd.DataFrame, plot_data: dict[str, dict[str, np.ndarray]], output: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 8.5,
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "legend.frameon": False,
            "pdf.fonttype": 42,
        }
    )
    colors = {
        "run_only_random": "#999999",
        "localization": "#0072B2",
        "localization_and_snr": "#D55E00",
        "real": "#111111",
    }
    labels = {
        "run_only_random": "Run-matched random",
        "localization": r"Matched $A_{90}$ + KL",
        "localization_and_snr": r"Matched $A_{90}$ + KL + SNR",
        "real": "Real strict pairs",
    }
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.0), constrained_layout=True)
    for row, deployment in enumerate(DEPLOYMENTS):
        ax = axes[row, 0]
        data = plot_data[deployment]
        for key in ("run_only_random", "localization", "localization_and_snr", "real"):
            values = np.sort(data[key][np.isfinite(data[key])])
            start = int(0.80 * len(values))
            survival = (len(values) - np.arange(len(values))) / len(values)
            ax.plot(values[start:], survival[start:], color=colors[key], lw=1.25, label=labels[key])
        ax.set_yscale("log")
        ax.set_xlabel(r"Sky evidence $Z_{\rm sky}$ at Nside 64 [nats]")
        ax.set_ylabel("Survival probability")
        ax.set_title(f"{'ac'[row]}  {deployment.upper()}: matched null tail", loc="left")
        ax.grid(alpha=0.18, linewidth=0.5)
        if row == 0:
            ax.legend(fontsize=7, loc="best")

        ax = axes[row, 1]
        subset = summary.loc[summary["deployment"] == deployment]
        for idx, scheme in enumerate(MATCHING_SCHEMES):
            values = subset.loc[subset["scheme"] == scheme, "tail_inflation_at_validation_q0990_median"].to_numpy(float)
            x = np.full(values.size, idx, dtype=float) + np.linspace(-0.08, 0.08, values.size)
            ax.scatter(x, values, color=colors[scheme], s=28, alpha=0.8)
            if values.size:
                ax.plot([idx - 0.2, idx + 0.2], [np.mean(values)] * 2, color=colors[scheme], lw=2)
        real_rate_rows = subset.loc[subset["scheme"] == "run_only_random"]
        if len(real_rate_rows):
            # The observed real inflation is stored outside this table; derive from real and threshold in report.
            pass
        ax.axhline(1.0, color="black", ls="--", lw=0.8)
        ax.set_yscale("log")
        ax.set_xticks(range(len(MATCHING_SCHEMES)), ["Run", "$A_{90}$+KL", "+SNR"])
        ax.set_ylabel("Matched-null tail inflation at validation q99")
        ax.set_title(f"{'bd'[row]}  Event-covariate contribution", loc="left")
        ax.grid(alpha=0.18, linewidth=0.5)
    figure_dir = output / "figures"
    figure_dir.mkdir(parents=True)
    fig.savefig(figure_dir / "fig_covariate_matched_sky_null.pdf", bbox_inches="tight")
    fig.savefig(figure_dir / "fig_covariate_matched_sky_null.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def report(summary: pd.DataFrame, output: Path) -> dict[str, object]:
    headline: dict[str, object] = {}
    lines = [
        "# 天空背景协变量匹配快速审计",
        "",
        f"生成时间：{datetime.now(timezone.utc).isoformat()}",
        "",
        "## 目的",
        "",
        "第一层审计发现，真实 GWTC pair 在相同 Nside=64 下超过注入验证背景 q99 阈值的比例仍高出约一个数量级。本实验不改变天空图或候选排名，而是把每个真实事件匹配到具有相近 observing run、A90、KL 和 SNR 的 synthetic validation event，检查可观测协变量差异能解释多少正尾错配。",
        "",
        "每个 seed 进行 500 次事件级匹配重采样；一个 replicate 内尽量不重复使用 synthetic event，并删除偶然抽到真实 synthetic companion 的 pair。真实候选排名和 PE 一致性均不进入匹配。",
        "",
        "## 结果",
        "",
        "| catalog | matching | q99 tail inflation, median [95% interval] | KS vs real, median | q99 gap to real [nats] |",
        "|---|---|---:|---:|---:|",
    ]
    for deployment in DEPLOYMENTS:
        headline[deployment] = {}
        for scheme in MATCHING_SCHEMES:
            subset = summary.loc[(summary["deployment"] == deployment) & (summary["scheme"] == scheme)]
            inflation = subset["tail_inflation_at_validation_q0990_median"].to_numpy(float)
            low = subset["tail_inflation_at_validation_q0990_q025"].to_numpy(float)
            high = subset["tail_inflation_at_validation_q0990_q975"].to_numpy(float)
            ks = subset["ks_statistic_vs_real_nside64_median"].to_numpy(float)
            gap = subset["q99_gap_to_real_nats_median"].to_numpy(float)
            lines.append(
                f"| {deployment.upper()} | {scheme} | {np.mean(inflation):.3f} "
                f"[{np.mean(low):.3f}, {np.mean(high):.3f}] | {np.mean(ks):.3f} | {np.mean(gap):.3f} |"
            )
            headline[deployment][scheme] = {
                "q99_tail_inflation_mean_of_seed_medians": float(np.mean(inflation)),
                "ks_vs_real_mean_of_seed_medians": float(np.mean(ks)),
                "q99_gap_to_real_nats_mean_of_seed_medians": float(np.mean(gap)),
            }
    lines.extend(
        [
            "",
            "## 解释规则",
            "",
            "- 若按 A90/KL/SNR 匹配后 synthetic null 的尾部明显变重并接近真实目录，说明旧 validation 主要因事件定位质量分布不匹配而低估假对尾部。",
            "- 若匹配后仍与真实目录相差很大，剩余差异更可能来自地图形态、多峰/细长弧结构、旋转插值、PE pipeline 或有限模板重复使用；这需要 Nside=512 map-level 构造与快速相干 PE pilot 继续验证。",
            "- 本实验使用真实事件的 A90/KL/SNR 只做事后域差异诊断，不能据此选择正式融合权重，也不产生新候选表。",
            "",
            "最终状态：`HOLD_FOR_AUTHOR_REVIEW_NO_ADOPTION_NO_OVERWRITE`。",
        ]
    )
    (output / "sky_covariate_matched_null_report_cn.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return headline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path("/root/autodl-tmp/gw-catalog"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    project = args.project_root.resolve()
    v93 = project / "results" / "gwtc_sky_resolution_v93_20260730"
    output = (args.output or project / "results" / f"sky_covariate_matched_null_v102_20260823_{stamp()}").resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    contract = {
        "experiment": "sky_covariate_matched_null_v102",
        "status": "EXPLORATORY_READ_ONLY",
        "matching_schemes": {name: list(features) for name, features in MATCHING_SCHEMES.items()},
        "replicates_per_seed": REPLICATES,
        "nearest_neighbours": NEIGHBOURS,
        "real_candidate_rank_used": False,
        "real_pe_consistency_used": False,
        "historical_results_modified": False,
        "formal_weight_selection": False,
        "terminal_state": "HOLD_FOR_AUTHOR_REVIEW_NO_ADOPTION_NO_OVERWRITE",
    }
    (output / "analysis_contract.json").write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")

    replicate_rows: list[dict[str, object]] = []
    matching_rows: list[dict[str, object]] = []
    used_paths: list[Path] = []
    plot_data: dict[str, dict[str, np.ndarray]] = {}
    for deployment in DEPLOYMENTS:
        resolution_path = v93 / deployment / "real_sky_resolution_convergence_all_pairs_v93.parquet"
        resolution = add_pair_key(pd.read_parquet(resolution_path))
        used_paths.append(resolution_path)
        dep_plot: dict[str, list[np.ndarray]] = {}
        for seed_dir in sorted((v93 / deployment).glob("seed_*")):
            rows, audits, paths, data = run_seed(project, v93, deployment, seed_dir, resolution, output)
            replicate_rows.extend(rows)
            matching_rows.extend(audits)
            used_paths.extend(paths)
            for key, values in data.items():
                dep_plot.setdefault(key, []).append(values)
        plot_data[deployment] = {key: np.concatenate(values) for key, values in dep_plot.items()}

    replicate_frame = pd.DataFrame(replicate_rows)
    matching_frame = pd.DataFrame(matching_rows)
    summary = aggregate(replicate_frame)
    write_csv(replicate_frame, output / "covariate_matched_null_per_replicate.csv")
    write_csv(summary, output / "covariate_matched_null_summary.csv")
    write_csv(matching_frame, output / "event_matching_audit.csv")
    replicate_frame.to_parquet(output / "covariate_matched_null_per_replicate.parquet", index=False)
    plot(summary, plot_data, output)
    headline = report(summary, output)
    payload = {
        "experiment": "sky_covariate_matched_null_v102",
        "headline": headline,
        "scientific_boundary": "diagnostic only; no weight selection and no candidate reranking",
        "next_stage": "Nside512 map-level matched posterior construction",
        "terminal_state": "HOLD_FOR_AUTHOR_REVIEW_NO_ADOPTION_NO_OVERWRITE",
    }
    (output / "sky_covariate_matched_null_summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    scripts = output / "scripts"
    scripts.mkdir()
    shutil.copy2(Path(__file__).resolve(), scripts / Path(__file__).name)
    input_manifest = pd.DataFrame(
        [
            {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in sorted(set(used_paths))
        ]
    )
    write_csv(input_manifest, output / "input_manifest.csv")
    manifest = output / "manifest"
    manifest.mkdir()
    files = [path for path in output.rglob("*") if path.is_file()]
    write_csv(
        pd.DataFrame(
            [
                {"relative_path": str(path.relative_to(output)), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
                for path in sorted(files)
            ]
        ),
        manifest / "files_sha256.csv",
    )
    latest = project / "results" / "sky_covariate_matched_null_v102_LATEST.txt"
    latest.write_text(str(output) + "\n", encoding="utf-8")
    package = project / "packages" / f"{output.name}.tar.gz"
    with tarfile.open(package, "w:gz") as archive:
        archive.add(output, arcname=output.name)
    package_sha = sha256_file(package)
    package.with_suffix(package.suffix + ".sha256").write_text(f"{package_sha}  {package.name}\n", encoding="ascii")
    print(json.dumps({"output": str(output), "package": str(package), "sha256": package_sha, "headline": headline}, indent=2))


if __name__ == "__main__":
    main()
