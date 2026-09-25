#!/usr/bin/env python3
"""Re-fuse existing ET-3 scores with an absolute observed-sky Bayes factor.

This script does not retrain the encoder or regenerate strain. Waveform
similarity is globally calibrated on validation pairs, time is the frozen
Liao/GW-LMC likelihood ratio, and sky uses common-source evidence relative to
an isotropic prior. No test-catalog row standardization is used.
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
ENCODER = REPO / "runs/et3_fresh50_full_catalog_20260616/fresh_mixed_encoders/et3_noisy_mixed_sis_pm_ep50"
DEFAULT_OUT = REPO / "results/real_noise_injection_v6_unified_physics_20260721/et3_unified_sky_bf_refusion"
WEIGHT_GRID = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)
SKY_SEEDS = (202607081, 202607082, 202607083)


def module_from(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, payload: Any) -> None:
    def default(value: Any):
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value) if np.isfinite(value) else None
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, Path):
            return str(value)
        return str(value)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=default) + "\n", encoding="utf-8")


def load_inputs():
    sensitivity = module_from(
        REPO / "scripts/experiments/105_et3_sky_localization_sensitivity.py",
        "et3_sensitivity_for_v6_refusion",
    )
    liao = module_from(
        REPO / "scripts/experiments/88_liao_realistic_p1_p2_rerank.py",
        "liao_for_v6_et3_refusion",
    )
    _, val, test = sensitivity.load_et3_without_model()
    return sensitivity, liao, val, test


def sample_waveform_scores(score: np.ndarray, gt: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    true_i = np.asarray([i for i, j in enumerate(gt) if j > i], dtype=np.int32)
    true = np.asarray(score[true_i, gt[true_i].astype(np.int32)], dtype=np.float64)
    rng = np.random.default_rng(seed)
    chunks, remaining = [], 1_000_000
    while remaining:
        count = min(max(remaining * 2, 100_000), 2_000_000)
        i = rng.integers(0, len(gt), size=count, dtype=np.int32)
        j = rng.integers(0, len(gt), size=count, dtype=np.int32)
        keep = (i != j) & (gt[i] != j)
        values = np.asarray(score[i[keep], j[keep]], dtype=np.float64)
        values = values[np.isfinite(values)]
        take = min(remaining, len(values))
        chunks.append(values[:take])
        remaining -= take
    return true, np.concatenate(chunks)


def build_waveform_components(out: Path, val_gt: np.ndarray, physical) -> dict[str, Any]:
    config_path = out / "waveform_global_likelihood_ratio_validation.json"
    paths = {split: out / f"cache/{split}_waveform_log_lr.npy" for split in ("val", "test")}
    if config_path.exists() and all(path.exists() for path in paths.values()):
        return json.loads(config_path.read_text())
    val_raw = np.load(ENCODER / "val_scores.npy", mmap_mode="r")
    signal, null = sample_waveform_scores(val_raw, val_gt, 202607081)
    config = physical.fit_score_likelihood_ratio(signal, null, grid_size=2048)
    config.update(
        definition="global validation-frozen log p(cosine|companion)/p(cosine|non-companion)",
        validation_true_pairs=len(signal),
        validation_false_pairs_sampled=len(null),
        row_standardization=False,
    )
    write_json(config_path, config)
    for split, path in paths.items():
        raw = np.load(ENCODER / f"{split}_scores.npy", mmap_mode="r")
        target = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=raw.shape)
        for start in range(0, len(raw), 64):
            stop = min(start + 64, len(raw))
            target[start:stop] = physical.apply_score_likelihood_ratio(
                np.asarray(raw[start:stop], dtype=np.float64), config
            ).astype(np.float32)
        np.fill_diagonal(target, 0.0)
        target.flush()
        del target
    return config


def build_time_components(
    out: Path,
    liao,
    val_time: pd.DataFrame,
    val_gt: np.ndarray,
    test_time: pd.DataFrame,
) -> dict[str, Any]:
    config_path = out / "time_delay_likelihood_ratio_validation.json"
    paths = {split: out / f"cache/{split}_time_log_lr.npy" for split in ("val", "test")}
    if config_path.exists() and all(path.exists() for path in paths.values()):
        return json.loads(config_path.read_text())
    prior = liao.fit_time_lr_from_liao("ET3", val_time, val_gt, seed=202607082)
    write_json(
        config_path,
        {
            **prior,
            "fit_split": "ET-3 validation catalog",
            "row_standardization": False,
        },
    )
    for split, timing in (("val", val_time), ("test", test_time)):
        score = liao.time_lr_score_matrix(timing, prior).astype(np.float32)
        np.fill_diagonal(score, 0.0)
        np.save(paths[split], score)
        del score
    return json.loads(config_path.read_text())


def sky_rows(sky: pd.DataFrame, start: int, stop: int) -> np.ndarray:
    from scripts.sky.sky_posterior_overlap import angular_separation_rad

    ra = sky.ra_obs.to_numpy(float)
    dec = sky.dec_obs.to_numpy(float)
    sigma = np.maximum(sky.sky_sigma_rad.to_numpy(float), 1e-12)
    theta = angular_separation_rad(ra[start:stop, None], dec[start:stop, None], ra[None, :], dec[None, :])
    variance = sigma[start:stop, None] ** 2 + sigma[None, :] ** 2
    return (math.log(2.0) - np.log(variance) - theta**2 / (2.0 * variance)).astype(np.float32)


def build_sky_component(
    out: Path,
    sensitivity,
    raw: pd.DataFrame,
    timing: pd.DataFrame,
    split: str,
    seed: int,
) -> tuple[Path, pd.DataFrame]:
    path = out / f"cache/{split}_sky_log_bf_seed_{seed}.npy"
    audit = out / f"{split}_observed_sky_seed_{seed}.csv"
    scenario = {
        "scenario_id": "ET_TRIANGLE_baseline",
        "a90_ref_deg2": 100.0,
        "scatter": 0.35,
        "clip_min_deg2": 20.0,
        "clip_max_deg2": 1000.0,
    }
    sky = pd.read_csv(audit) if audit.exists() else sensitivity.build_sky(raw, timing, scenario, seed)
    if not audit.exists():
        sky.to_csv(audit, index=False)
    if not path.exists():
        target = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(len(sky), len(sky)))
        for start in range(0, len(sky), 64):
            stop = min(start + 64, len(sky))
            target[start:stop] = sky_rows(sky, start, stop)
        np.fill_diagonal(target, 0.0)
        target.flush()
        del target
    return path, sky


def valid_queries(gt: np.ndarray) -> np.ndarray:
    return np.flatnonzero(np.asarray(gt) >= 0).astype(np.int32)


def grid(mode: str) -> list[dict[str, float]]:
    output = []
    for waveform, time, sky in itertools.product(WEIGHT_GRID, repeat=3):
        if waveform == time == sky == 0:
            continue
        if mode == "positive" and min(waveform, time, sky) <= 0:
            continue
        if mode == "time_sky" and (waveform != 0 or min(time, sky) <= 0):
            continue
        output.append({"waveform": waveform, "time": time, "sky": sky})
    return output


def rank_weights(paths: list[Path], gt: np.ndarray, weights: list[dict[str, float]]) -> np.ndarray:
    components = [np.load(path, mmap_mode="r") for path in paths]
    queries = valid_queries(gt)
    partners = gt[queries].astype(np.int64)
    result = np.empty((len(weights), len(queries)), dtype=np.int32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch = 16 if device.type == "cuda" else 2
    weight_array = np.asarray([[w["waveform"], w["time"], w["sky"]] for w in weights], np.float32)
    for start in range(0, len(queries), 128):
        stop = min(start + 128, len(queries))
        q = queries[start:stop]
        p = partners[start:stop]
        block = torch.from_numpy(np.stack([np.asarray(x[q], np.float32) for x in components])).to(device)
        local = torch.arange(len(q), device=device)
        q_t = torch.from_numpy(q.astype(np.int64)).to(device)
        p_t = torch.from_numpy(p).to(device)
        for left in range(0, len(weights), batch):
            right = min(left + batch, len(weights))
            w = torch.from_numpy(weight_array[left:right]).to(device)
            score = torch.einsum("wc,cqn->wqn", w, block)
            score[:, local, q_t] = -torch.inf
            truth = score[:, local, p_t]
            result[left:right, start:stop] = (1 + (score > truth[:, :, None]).sum(-1)).cpu().numpy()
            del score, truth
        del block
    return result


def metrics(ranks: np.ndarray, weights: list[dict[str, float]], meta: list[dict], gt: np.ndarray) -> pd.DataFrame:
    queries = valid_queries(gt)
    family = np.asarray([str(meta[int(i)]["family"]).upper() for i in queries])
    rows = []
    for index, weight in enumerate(weights):
        row: dict[str, Any] = {**weight}
        for label, keep in (("overall", np.ones(len(queries), bool)), ("sis", family == "SIS"), ("pm", family == "PM")):
            values = ranks[index, keep]
            for k in (1, 5, 10):
                row[f"{label}_r_at_{k}"] = float(np.mean(values <= k))
            row[f"{label}_median_rank"] = float(np.median(values))
        row["min_family_r_at_10"] = min(row["sis_r_at_10"], row["pm_r_at_10"])
        row["weight_l2"] = math.sqrt(sum(value * value for value in weight.values()))
        rows.append(row)
    return pd.DataFrame(rows)


def select(frame: pd.DataFrame) -> dict[str, float]:
    best = frame.sort_values(
        ["overall_r_at_10", "min_family_r_at_10", "overall_r_at_1", "overall_r_at_5", "overall_median_rank", "weight_l2"],
        ascending=[False, False, False, False, True, True],
    ).iloc[0]
    return {name: float(best[name]) for name in ("waveform", "time", "sky")}


def evaluate(
    ranks: np.ndarray,
    methods: list[str],
    weights: list[dict[str, float]],
    meta: list[dict],
    gt: np.ndarray,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    queries = valid_queries(gt)
    family = np.asarray([str(meta[int(i)]["family"]).upper() for i in queries])
    metric_rows, query_rows = [], []
    for method_index, (method, weight) in enumerate(zip(methods, weights)):
        values = ranks[method_index]
        for subset, keep in (("overall", np.ones(len(values), bool)), ("SIS", family == "SIS"), ("PM", family == "PM")):
            metric_rows.append(
                {
                    "sky_seed": seed,
                    "method": method,
                    "subset": subset,
                    **{f"r_at_{k}": float(np.mean(values[keep] <= k)) for k in (1, 5, 10)},
                    "median_rank": float(np.median(values[keep])),
                    "n_queries": int(keep.sum()),
                    **{f"weight_{key}": value for key, value in weight.items()},
                }
            )
        for q, p, fam, rank in zip(queries, gt[queries], family, values):
            query_rows.append(
                {
                    "sky_seed": seed,
                    "method": method,
                    "query_index": int(q),
                    "partner_index": int(p),
                    "family": fam,
                    "system_id": f"{fam}:{min(int(q), int(p))}-{max(int(q), int(p))}",
                    "query_rank": int(rank),
                }
            )
    return pd.DataFrame(metric_rows), pd.DataFrame(query_rows)


def bootstrap(query: pd.DataFrame, draws: int) -> pd.DataFrame:
    rows = []
    for (seed, method), group in query.groupby(["sky_seed", "method"]):
        system = group.groupby(["family", "system_id"], as_index=False).agg(
            r1=("query_rank", lambda x: np.mean(np.asarray(x) <= 1)),
            r10=("query_rank", lambda x: np.mean(np.asarray(x) <= 10)),
        )
        rng = np.random.default_rng(int(seed) + sum(map(ord, method)))
        for column, metric in (("r1", "r_at_1"), ("r10", "r_at_10")):
            samples = np.zeros(draws)
            for _, part in system.groupby("family"):
                values = part[column].to_numpy(float)
                for start in range(0, draws, 100):
                    stop = min(start + 100, draws)
                    index = rng.integers(0, len(values), size=(stop - start, len(values)))
                    samples[start:stop] += 0.5 * values[index].mean(1)
            rows.append(
                {
                    "sky_seed": seed,
                    "method": method,
                    "metric": metric,
                    "point_estimate": float(group.query_rank.le(1 if metric == "r_at_1" else 10).mean()),
                    "ci95_low": float(np.quantile(samples, 0.025)),
                    "ci95_high": float(np.quantile(samples, 0.975)),
                    "bootstrap_draws": draws,
                    "bootstrap_unit": "lensed_system_stratified_by_family",
                    "n_systems": int(len(system)),
                }
            )
    return pd.DataFrame(rows)


def figure(metric_frame: pd.DataFrame, out: Path) -> None:
    overall = metric_frame[metric_frame.subset == "overall"]
    methods = ["waveform_only", "time_sky_validation_selected", "three_channel_validation_selected", "three_channel_strictly_positive"]
    labels = ["Waveform", "Time + sky", "Three-channel", "Three-channel\npositive"]
    colors = ["#4C78A8", "#72B7B2", "#E45756", "#F2CF5B"]
    with plt.rc_context({"font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif"], "font.size": 8.5, "axes.labelweight": "bold", "axes.titleweight": "bold", "pdf.fonttype": 42}):
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.25), sharey=True)
        for ax, column, title in zip(axes, ("r_at_1", "r_at_10"), ("First-rank recall", "Top-ten recall")):
            values = [overall.loc[overall.method == method, column].to_numpy(float) for method in methods]
            x = np.arange(len(methods))
            ax.bar(x, [v.mean() for v in values], yerr=[v.std(ddof=1) for v in values], color=colors, capsize=3, alpha=0.85)
            for index, points in enumerate(values):
                ax.scatter(np.full(len(points), index), points, facecolors="white", edgecolors="black", s=22, zorder=3)
            ax.set_xticks(x, labels, rotation=18, ha="right")
            ax.set_title(title)
            ax.set_ylim(0, 1)
            ax.grid(axis="y", alpha=0.2)
        axes[0].set_ylabel("Companion retrieval recall")
        fig.suptitle("ET-3 refusion with absolute observed-sky evidence", fontweight="bold")
        fig.tight_layout()
        (out / "figures").mkdir(exist_ok=True)
        fig.savefig(out / "figures/fig_et3_sky_bf_refusion.pdf", dpi=300)
        fig.savefig(out / "figures/fig_et3_sky_bf_refusion.png", dpi=240)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--sky-seeds", nargs="+", type=int, default=list(SKY_SEEDS))
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "cache").mkdir(exist_ok=True)
    sensitivity, liao, val, test = load_inputs()
    val_meta, val_raw, val_time, val_gt = val
    test_meta, test_raw, test_time, test_gt = test
    physical = module_from(REPO / "scripts/real_search/physical_common.py", "physical_for_et3_refusion")
    waveform_config = build_waveform_components(args.out, val_gt, physical)
    time_config = build_time_components(args.out, liao, val_time, val_gt, test_time)
    val_base = [args.out / "cache/val_waveform_log_lr.npy", args.out / "cache/val_time_log_lr.npy"]
    test_base = [args.out / "cache/test_waveform_log_lr.npy", args.out / "cache/test_time_log_lr.npy"]
    metric_frames, query_frames, weight_rows = [], [], []
    for seed in args.sky_seeds:
        val_sky_path, val_sky = build_sky_component(args.out, sensitivity, val_raw, val_time, "val", seed)
        test_sky_path, test_sky = build_sky_component(args.out, sensitivity, test_raw, test_time, "test", seed + 10_000)
        weights_all = grid("free")
        validation_ranks = rank_weights(val_base + [val_sky_path], val_gt, weights_all)
        validation_metrics = metrics(validation_ranks, weights_all, val_meta, val_gt)
        validation_metrics.to_csv(args.out / f"validation_weight_grid_seed_{seed}.csv", index=False)
        selected = select(validation_metrics)
        positive = select(validation_metrics[(validation_metrics.waveform > 0) & (validation_metrics.time > 0) & (validation_metrics.sky > 0)])
        time_sky = select(validation_metrics[(validation_metrics.waveform == 0) & (validation_metrics.time > 0) & (validation_metrics.sky > 0)])
        methods = ["waveform_only", "time_only", "sky_bayes_factor_only", "time_sky_validation_selected", "three_channel_validation_selected", "three_channel_strictly_positive", "three_channel_equal_evidence"]
        method_weights = [
            {"waveform": 1.0, "time": 0.0, "sky": 0.0},
            {"waveform": 0.0, "time": 1.0, "sky": 0.0},
            {"waveform": 0.0, "time": 0.0, "sky": 1.0},
            time_sky,
            selected,
            positive,
            {"waveform": 1.0, "time": 1.0, "sky": 1.0},
        ]
        test_ranks = rank_weights(test_base + [test_sky_path], test_gt, method_weights)
        metric_frame, query_frame = evaluate(test_ranks, methods, method_weights, test_meta, test_gt, seed)
        metric_frames.append(metric_frame)
        query_frames.append(query_frame)
        weight_rows.append(
            {
                "sky_seed": seed,
                "three_channel_validation_selected": selected,
                "three_channel_strictly_positive": positive,
                "time_sky_validation_selected": time_sky,
                "validation_events": len(val_gt),
                "test_events": len(test_gt),
                "validation_queries": len(valid_queries(val_gt)),
                "test_queries": len(valid_queries(test_gt)),
                "val_a90_median_deg2": float(val_sky.sky_area90_deg2.median()),
                "test_a90_median_deg2": float(test_sky.sky_area90_deg2.median()),
            }
        )
    metric_frame = pd.concat(metric_frames, ignore_index=True)
    query_frame = pd.concat(query_frames, ignore_index=True)
    metric_frame.to_csv(args.out / "et3_sky_bf_refusion_metrics_per_seed.csv", index=False)
    query_frame.to_parquet(args.out / "et3_sky_bf_refusion_query_ranks.parquet", index=False)
    bootstrap(query_frame, args.bootstrap_draws).to_csv(args.out / "et3_sky_bf_refusion_system_bootstrap_95ci.csv", index=False)
    summary = metric_frame.groupby(["method", "subset"], as_index=False).agg(
        r_at_1_mean=("r_at_1", "mean"), r_at_1_std=("r_at_1", "std"),
        r_at_10_mean=("r_at_10", "mean"), r_at_10_std=("r_at_10", "std"),
        median_rank_mean=("median_rank", "mean"), n_sky_realizations=("sky_seed", "nunique"),
    )
    summary.to_csv(args.out / "et3_sky_bf_refusion_summary.csv", index=False)
    write_json(args.out / "et3_sky_bf_refusion_weights.json", weight_rows)
    write_json(
        args.out / "et3_sky_bf_refusion_run_config.json",
        {
            "status": "complete",
            "encoder_retrained": False,
            "waveform_source": str(ENCODER / "test_scores.npy"),
            "waveform_calibration": waveform_config,
            "time_calibration": time_config,
            "sky_definition": "log(2)-log(sigma_i^2+sigma_j^2)-theta^2/[2(sigma_i^2+sigma_j^2)]",
            "sky_prior": "isotropic solid-angle prior",
            "row_standardization": False,
            "weights_selected_on": "ET-3 validation catalog only",
            "test_used_for_tuning": False,
            "sky_seeds": args.sky_seeds,
            "limitation": "Observed-sky Gaussian posterior proxy, not full ET parameter-estimation HEALPix posteriors.",
        },
    )
    figure(metric_frame, args.out)
    report = "\n".join(
        [
            "# ET-3 absolute sky-Bayes-factor refusion",
            "",
            "The existing ET-3 encoder is reused. Waveform similarity is globally calibrated on validation pairs, time is the validation-frozen Liao/GW-LMC delay LR, and sky is the absolute common-source Bayes factor of the observed-sky Gaussian posterior proxy. No channel is row-standardized in the test catalog.",
            "",
            summary[summary.subset == "overall"].to_markdown(index=False),
            "",
            "This is an observed-sky proxy experiment, not full ET parameter estimation, a false-alarm significance, or a detection claim.",
            "",
        ]
    )
    (args.out / "et3_sky_bf_refusion_report.md").write_text(report, encoding="utf-8")
    print(json.dumps({"status": "complete", "out": str(args.out)}, indent=2))


if __name__ == "__main__":
    main()
