from __future__ import annotations

import argparse
import gc
import importlib
import importlib.util
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.experiments.mainline_uncertainty_common import (  # noqa: E402
    across_seed_summary,
    bootstrap_system_ci,
    full_unordered_pair_metrics,
    metric_rows,
    ranks_from_components,
    seed_everything,
    split_integrity_audit,
    write_json,
)
from scripts.sky.sky_posterior_overlap import gaussian_log_cosine_overlap_matrix  # noqa: E402


ET3_MATCH_ROOT = Path("/root/autodl-tmp/createdata/et3_10000_20260616_1006_match_root")
DEFAULT_OUT = REPO_ROOT / "results" / "mainline_uncertainty_20260713" / "et3"
DEFAULT_SEEDS = (202607101, 202607102, 202607103, 202607104, 202607105)
FAMILIES = ("SIS", "PM")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save_split_manifest(path: Path, splits: dict[str, dict[str, np.ndarray]]) -> None:
    arrays = {f"{family}_{split}": values for family, parts in splits.items() for split, values in parts.items()}
    np.savez_compressed(path, **arrays)


def train_or_load_seed(base, seed_dir: Path, seed: int, epochs: int, num_workers: int):
    encoder_dir = seed_dir / "encoder"
    encoder_dir.mkdir(parents=True, exist_ok=True)
    cfg = base.make_cfg("ET3", "noisy", encoder_dir)
    cfg.seed = int(seed)
    cfg.epochs = int(epochs)
    cfg.num_workers = int(num_workers)
    arrays = {family: base.FamilyArrays(family, ET3_MATCH_ROOT, "noisy") for family in FAMILIES}
    splits: dict[str, dict[str, np.ndarray]] = {}
    for index, family in enumerate(FAMILIES):
        # Split by source-system index. Both images of a doublet therefore remain
        # together, while each training seed receives an independent system split.
        splits[family] = base.split_indices(len(arrays[family].l1), cfg.seed + index)
        splits[f"{family}_U"] = base.split_indices(len(arrays[family].unlensed), cfg.seed + 100 + index)
    save_split_manifest(seed_dir / "split_indices.npz", splits)
    split_audit = split_integrity_audit(splits)
    if not split_audit["all_disjoint"]:
        raise RuntimeError("ET-3 train/validation/test system split leakage detected")
    write_json(seed_dir / "split_integrity_audit.json", split_audit)
    seed_everything(seed)
    model, train_info = base.train_or_load_encoder(cfg, arrays, splits)
    summary_path = encoder_dir / "waveform_summary.json"
    if not summary_path.exists():
        write_json(
            summary_path,
            {
                "config": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(cfg).items()},
                "splits": {k: {s: int(len(v)) for s, v in part.items()} for k, part in splits.items()},
                "timing": {k: v for k, v in train_info.items() if k != "history"},
                "uncertainty_run": True,
            },
        )
    return cfg, arrays, splits, model, train_info


def run_seed(
    seed: int,
    out_root: Path,
    bootstrap_draws: int,
    pair_metrics: bool,
    epochs: int,
    num_workers: int,
) -> dict[str, Any]:
    seed_dir = out_root / f"seed_{seed}"
    complete = seed_dir / "seed_summary.json"
    if complete.exists():
        print(f"ET3 seed {seed}: complete, reusing", flush=True)
        return json.loads(complete.read_text(encoding="utf-8"))
    seed_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    base = importlib.import_module("scripts.experiments.80_mixed_sis_pm_catalog_modality_compare")
    fresh = importlib.import_module("scripts.experiments.84_fresh50_full_catalog_ranking")
    liao = load_module(
        f"liao_uncertainty_{seed}",
        REPO_ROOT / "scripts" / "experiments" / "88_liao_realistic_p1_p2_rerank.py",
    )
    unified = load_module(
        f"unified_uncertainty_{seed}",
        REPO_ROOT / "scripts" / "experiments" / "102_unified_sky_formal_pipeline.py",
    )
    base.ROOTS[("SIS", "ET3")] = ET3_MATCH_ROOT
    base.ROOTS[("PM", "ET3")] = ET3_MATCH_ROOT
    liao.base.ROOTS[("SIS", "ET3")] = ET3_MATCH_ROOT
    liao.base.ROOTS[("PM", "ET3")] = ET3_MATCH_ROOT

    cfg, arrays, splits, model, train_info = train_or_load_seed(
        base, seed_dir, seed, epochs, num_workers
    )
    val_ds, val_raw, val_time, val_gt, _, val_scores = base.split_pack(
        "ET3", "val", cfg, arrays, splits, model
    )
    test_ds, test_raw, test_time, test_gt, _, test_scores = base.split_pack(
        "ET3", "test", cfg, arrays, splits, model
    )
    prior = liao.fit_time_lr_from_liao("ET3", val_time, val_gt, seed=seed + 1100)
    val_sky = liao.make_observed_sky("ET3", val_raw, val_time, seed=seed + 2100)
    test_sky = liao.make_observed_sky("ET3", test_raw, test_time, seed=seed + 3100)
    val_sky.to_parquet(seed_dir / "val_observed_sky.parquet", index=False)
    test_sky.to_parquet(seed_dir / "test_observed_sky.parquet", index=False)

    print(f"ET3 seed {seed}: unified sky matrices", flush=True)
    val_sky_score = gaussian_log_cosine_overlap_matrix(val_sky, chunk_rows=64, diagonal=0.0)
    test_sky_score = gaussian_log_cosine_overlap_matrix(test_sky, chunk_rows=64, diagonal=0.0)
    val_time_score = liao.time_lr_score_matrix(val_time, prior)
    test_time_score = liao.time_lr_score_matrix(test_time, prior)
    val_components = {
        "waveform": unified.row_z_zero_diag(val_scores),
        "time": unified.row_z_zero_diag(val_time_score),
        "sky": unified.row_z_zero_diag(val_sky_score),
    }
    test_components = {
        "waveform": unified.row_z_zero_diag(test_scores),
        "time": unified.row_z_zero_diag(test_time_score),
        "sky": unified.row_z_zero_diag(test_sky_score),
    }
    del val_sky_score, test_sky_score, val_time_score, test_time_score
    gc.collect()

    fixed = {
        "waveform_only": {"waveform": 1.0},
        "time_delay_only": {"time": 1.0},
        "unified_sky_only": {"sky": 1.0},
    }
    selected = {
        "waveform_time": ("waveform", "time"),
        "waveform_sky": ("waveform", "sky"),
        "time_sky": ("time", "sky"),
        "waveform_time_sky": ("waveform", "time", "sky"),
    }
    methods: dict[str, dict[str, float]] = dict(fixed)
    weight_payload: dict[str, Any] = {name: {"selected_weights": weights, "selection": "fixed"} for name, weights in fixed.items()}
    for name, channels in selected.items():
        print(f"ET3 seed {seed}: validation weight selection {name}", flush=True)
        weights, grid = unified.select_et3_weights(val_components, channels, val_gt, val_ds.meta, fresh)
        grid.to_csv(seed_dir / f"weight_grid_{name}.csv", index=False)
        methods[name] = weights
        weight_payload[name] = {
            "selected_weights": weights,
            "selection": "validation_only",
            "validation_best": grid.iloc[0].to_dict(),
        }
    write_json(seed_dir / "selected_weights.json", weight_payload)

    query_frames = []
    final_directed = None
    for name, weights in methods.items():
        frame, _ = ranks_from_components(
            test_components,
            weights,
            test_gt,
            test_ds.meta,
            deployment="ET3",
            seed=seed,
            method=name,
        )
        query_frames.append(frame)
        if name == "waveform_time_sky" and pair_metrics:
            final_directed = unified.score_matrix_from_components(test_components, weights)
    query = pd.concat(query_frames, ignore_index=True)
    query.to_parquet(seed_dir / "query_ranks.parquet", index=False)
    point = metric_rows(query)
    point.to_csv(seed_dir / "retrieval_metrics.csv", index=False)
    bootstrap = bootstrap_system_ci(query, draws=bootstrap_draws, seed=seed + 5100)
    bootstrap.to_csv(seed_dir / "bootstrap_95ci.csv", index=False)

    pair_summary = None
    if pair_metrics:
        if final_directed is None:
            raise RuntimeError("Final directed score was not built")
        print(f"ET3 seed {seed}: exact 40,495,500 unordered-pair metrics", flush=True)
        pair_summary, operating = full_unordered_pair_metrics(
            final_directed, test_gt, test_ds.meta, aggregation="max"
        )
        pair_summary.update({"deployment": "ET3", "seed": int(seed), "method": "waveform_time_sky"})
        write_json(seed_dir / "pair_level_metrics.json", pair_summary)
        operating.insert(0, "seed", int(seed))
        operating.insert(0, "deployment", "ET3")
        operating.to_csv(seed_dir / "pair_level_operating_points.csv", index=False)
        mean_summary, mean_operating = full_unordered_pair_metrics(
            final_directed, test_gt, test_ds.meta, aggregation="mean"
        )
        mean_summary.update(
            {"deployment": "ET3", "seed": int(seed), "method": "waveform_time_sky_mean_sensitivity"}
        )
        write_json(seed_dir / "pair_level_metrics_mean_sensitivity.json", mean_summary)
        mean_operating.insert(0, "seed", int(seed))
        mean_operating.insert(0, "deployment", "ET3")
        mean_operating.to_csv(seed_dir / "pair_level_operating_points_mean_sensitivity.csv", index=False)

    summary = {
        "deployment": "ET3",
        "seed": int(seed),
        "status": "complete",
        "method": "latest unified posterior-overlap",
        "n_test_events": int(len(test_gt)),
        "n_test_queries": int(np.sum(test_gt >= 0)),
        "n_test_true_systems": int(np.sum(test_gt >= 0) // 2),
        "training_epochs": int(cfg.epochs),
        "training_seed": int(cfg.seed),
        "train_seconds": float(train_info.get("train_s", 0.0)),
        "sky_seed_validation": int(seed + 2100),
        "sky_seed_test": int(seed + 3100),
        "time_prior_seed": int(seed + 1100),
        "base_detector_noise_policy": "fixed generated ET-3 reservoir; independent source-system split, model initialization and stochastic training augmentation per seed",
        "selected_weights": weight_payload,
        "pair_level": pair_summary,
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    write_json(complete, summary)
    del val_components, test_components, final_directed, model, arrays
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


def aggregate(out_root: Path, seeds: list[int]) -> None:
    query = pd.concat([pd.read_parquet(out_root / f"seed_{seed}" / "query_ranks.parquet") for seed in seeds], ignore_index=True)
    query.to_parquet(out_root / "et3_query_ranks_all_seeds.parquet", index=False)
    per_seed = metric_rows(query)
    per_seed.to_csv(out_root / "et3_retrieval_metrics_per_seed.csv", index=False)
    across = across_seed_summary(per_seed)
    across.to_csv(out_root / "et3_retrieval_metrics_across_seed_summary.csv", index=False)
    boot = pd.concat([pd.read_csv(out_root / f"seed_{seed}" / "bootstrap_95ci.csv") for seed in seeds], ignore_index=True)
    boot.to_csv(out_root / "et3_bootstrap_95ci_per_seed.csv", index=False)
    pair_json = []
    pair_ops = []
    for seed in seeds:
        path = out_root / f"seed_{seed}" / "pair_level_metrics.json"
        if path.exists():
            pair_json.append(json.loads(path.read_text(encoding="utf-8")))
            pair_ops.append(
                pd.read_csv(out_root / f"seed_{seed}" / "pair_level_operating_points.csv")
                .assign(method="waveform_time_sky")
            )
            sensitivity_path = out_root / f"seed_{seed}" / "pair_level_metrics_mean_sensitivity.json"
            if sensitivity_path.exists():
                pair_json.append(json.loads(sensitivity_path.read_text(encoding="utf-8")))
                pair_ops.append(
                    pd.read_csv(out_root / f"seed_{seed}" / "pair_level_operating_points_mean_sensitivity.csv")
                    .assign(method="waveform_time_sky_mean_sensitivity")
                )
    if pair_json:
        pair_df = pd.DataFrame(pair_json)
        pair_df.to_csv(out_root / "et3_pair_level_metrics_per_seed.csv", index=False)
        pair_summary = across_seed_summary(
            pair_df.rename(columns={"n_events": "n_queries"}).assign(
                deployment="ET3", subset="unordered_pairs", n_systems=3000
            )[
                ["deployment", "seed", "method", "subset", "n_queries", "n_systems", "roc_auc", "average_precision", "base_positive_rate"]
            ]
        )
        pair_summary.to_csv(out_root / "et3_pair_level_metrics_across_seed_summary.csv", index=False)
        operating_all = pd.concat(pair_ops, ignore_index=True)
        operating_all.to_csv(out_root / "et3_pair_level_operating_points_all_seeds.csv", index=False)
        summary_rows = []
        value_columns = [
            "threshold", "false_pairs", "true_pairs", "fpr", "recall", "precision", "sis_recall", "pm_recall"
        ]
        for key, frame in operating_all.groupby(
            ["deployment", "method", "operating_point", "target"], sort=False
        ):
            for metric in value_columns:
                values = frame[metric].dropna().to_numpy(dtype=np.float64)
                if not len(values):
                    continue
                summary_rows.append(
                    {
                        "deployment": key[0],
                        "method": key[1],
                        "operating_point": key[2],
                        "target": float(key[3]),
                        "metric": metric,
                        "n_seeds": int(len(values)),
                        "mean": float(values.mean()),
                        "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                        "median": float(np.median(values)),
                        "q25": float(np.quantile(values, 0.25)),
                        "q75": float(np.quantile(values, 0.75)),
                    }
                )
        pd.DataFrame(summary_rows).to_csv(
            out_root / "et3_pair_level_operating_points_across_seed_summary.csv", index=False
        )
    write_json(
        out_root / "et3_uncertainty_summary.json",
        {
            "status": "complete",
            "deployment": "ET3",
            "seeds": seeds,
            "n_seeds": len(seeds),
            "method": "waveform + GW-LMC time-delay LR + unified Gaussian posterior log-cosine overlap",
            "weight_selection": "independently on each seed validation catalog",
            "split_policy": "independent source-system split per seed; both directed images remain in the same split",
            "bootstrap_unit": "lensed system, family-stratified",
            "pair_level_scope": "all 40,495,500 unordered test-catalog pairs per seed; max aggregation is primary and mean aggregation is SI sensitivity; summary metrics retained, full tables not retained",
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--skip-pair-metrics", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("CUDA is unavailable; refusing to launch the formal multi-seed retraining on CPU")
    args.out_root.mkdir(parents=True, exist_ok=True)
    write_json(
        args.out_root / "run_config.json",
        {
            "seeds": args.seeds,
            "bootstrap_draws": args.bootstrap_draws,
            "pair_metrics": not args.skip_pair_metrics,
            "et3_match_root": ET3_MATCH_ROOT,
            "training_epochs": args.epochs,
            "num_workers": args.num_workers,
        },
    )
    completed = []
    for seed in args.seeds:
        run_seed(
            int(seed),
            args.out_root,
            args.bootstrap_draws,
            not args.skip_pair_metrics,
            args.epochs,
            args.num_workers,
        )
        completed.append(int(seed))
        aggregate(args.out_root, completed)
    print(json.dumps({"status": "complete", "out_root": str(args.out_root), "seeds": completed}, indent=2), flush=True)


if __name__ == "__main__":
    main()
