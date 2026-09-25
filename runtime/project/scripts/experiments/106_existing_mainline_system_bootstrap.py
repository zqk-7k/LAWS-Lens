from __future__ import annotations

import argparse
import gc
import importlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.experiments.mainline_uncertainty_common import (  # noqa: E402
    bootstrap_system_ci,
    metric_rows,
    ranks_from_components,
    write_json,
)
from scripts.sky.sky_posterior_overlap import gaussian_log_cosine_overlap_matrix  # noqa: E402


ET3_MATCH_ROOT = Path("/root/autodl-tmp/createdata/et3_10000_20260616_1006_match_root")
ET3_ENCODER_ROOT = REPO_ROOT / "runs" / "et3_fresh50_full_catalog_20260616" / "fresh_mixed_encoders"
O3_SOURCE = REPO_ROOT / "runs" / "real_gwtc_lensing_search_20260625"
O4_SOURCE = REPO_ROOT / "runs" / "real_gwtc34_lensing_search_20260629_full_o4"
DEFAULT_OUT = REPO_ROOT / "results" / "mainline_uncertainty_20260713" / "existing_model_bootstrap"
MODEL_SEED = 42


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def et3_query_ranks() -> tuple[pd.DataFrame, dict[str, Any]]:
    liao = load_module(
        "bootstrap_liao_mainline",
        REPO_ROOT / "scripts" / "experiments" / "88_liao_realistic_p1_p2_rerank.py",
    )
    unified = load_module(
        "bootstrap_unified_mainline",
        REPO_ROOT / "scripts" / "experiments" / "102_unified_sky_formal_pipeline.py",
    )
    fresh = importlib.import_module("scripts.experiments.84_fresh50_full_catalog_ranking")
    liao.base.ROOTS[("SIS", "ET3")] = ET3_MATCH_ROOT
    liao.base.ROOTS[("PM", "ET3")] = ET3_MATCH_ROOT
    liao.ENCODER_ROOT = ET3_ENCODER_ROOT
    loaded = liao.load_job("ET3", "noisy")
    val_ds, val_raw, val_time, val_gt, val_scores = loaded["val"]
    test_ds, test_raw, test_time, test_gt, test_scores = loaded["test"]

    prior = liao.fit_time_lr_from_liao("ET3", val_time, val_gt, seed=20260708)
    val_sky = liao.make_observed_sky("ET3", val_raw, val_time, seed=801000)
    test_sky = liao.make_observed_sky("ET3", test_raw, test_time, seed=802000)
    val_components = {
        "waveform": unified.row_z_zero_diag(val_scores),
        "time": unified.row_z_zero_diag(liao.time_lr_score_matrix(val_time, prior)),
        "sky": unified.row_z_zero_diag(
            gaussian_log_cosine_overlap_matrix(val_sky, chunk_rows=64, diagonal=0.0)
        ),
    }
    test_components = {
        "waveform": unified.row_z_zero_diag(test_scores),
        "time": unified.row_z_zero_diag(liao.time_lr_score_matrix(test_time, prior)),
        "sky": unified.row_z_zero_diag(
            gaussian_log_cosine_overlap_matrix(test_sky, chunk_rows=64, diagonal=0.0)
        ),
    }
    final_weights, grid = unified.select_et3_weights(
        val_components, ("waveform", "time", "sky"), val_gt, val_ds.meta, fresh
    )
    time_sky_weights, _ = unified.select_et3_weights(
        val_components, ("time", "sky"), val_gt, val_ds.meta, fresh
    )
    methods = {
        "waveform_only": {"waveform": 1.0},
        "time_sky": time_sky_weights,
        "waveform_time_sky": final_weights,
    }
    frames = [
        ranks_from_components(
            test_components,
            weights,
            test_gt,
            test_ds.meta,
            deployment="ET3",
            seed=MODEL_SEED,
            method=method,
        )[0]
        for method, weights in methods.items()
    ]
    del val_components, test_components
    gc.collect()
    return pd.concat(frames, ignore_index=True), {
        "model": str(ET3_ENCODER_ROOT / "et3_noisy_mixed_sis_pm_ep50" / "model.pt"),
        "split_seed": MODEL_SEED,
        "selected_weights": final_weights,
        "time_sky_weights": time_sky_weights,
        "validation_best": grid.iloc[0].to_dict(),
        "n_test_events": int(len(test_gt)),
    }


def gwtc_query_ranks(deployment: str, source: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    uncertainty = load_module(
        f"bootstrap_uncertainty_{deployment}",
        REPO_ROOT / "scripts" / "real_search" / "16_gwtc_unified_sky_uncertainty.py",
    )
    deploy = load_module(
        f"bootstrap_deploy_{deployment}",
        REPO_ROOT / "scripts" / "real_search" / "15_gwtc34_real_deployment.py",
    )
    delay_prior = deploy.liao_delay_samples()
    val = pd.concat(
        [
            uncertainty.build_split_pair_table(
                deploy, source, family, 20, 600, "val", delay_prior, 20260708
            )
            for family in ("SIS", "PM")
        ],
        ignore_index=True,
    )
    test = pd.concat(
        [
            uncertainty.build_split_pair_table(
                deploy, source, family, 20, 600, "test", delay_prior, 20260709
            )
            for family in ("SIS", "PM")
        ],
        ignore_index=True,
    )
    final_weights, grid = deploy.select_weights(
        val, force_waveform_zero=False, require_all_positive=False
    )
    time_sky_weights, _ = deploy.select_weights(
        val, force_waveform_zero=True, require_time_sky_positive=False
    )
    methods = {
        "waveform_only": {"waveform": 1.0, "time": 0.0, "sky": 0.0},
        "time_sky": time_sky_weights,
        "waveform_time_sky": final_weights,
    }
    query = pd.concat(
        [
            uncertainty.query_ranks_from_pairs(
                deploy, test, weights, deployment, MODEL_SEED, method
            )
            for method, weights in methods.items()
        ],
        ignore_index=True,
    )
    return query, {
        "source_run": str(source),
        "split_seed": MODEL_SEED,
        "selected_weights": final_weights,
        "time_sky_weights": time_sky_weights,
        "validation_best": grid.iloc[0].to_dict(),
        "n_validation_unordered_pairs": int(len(val)),
        "n_test_unordered_pairs": int(len(test)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_query = []
    provenance: dict[str, Any] = {}
    for deployment, source in (("ET3", None), ("GWTC3", O3_SOURCE), ("GWTC4", O4_SOURCE)):
        print(f"Reconstructing fixed-model held-out ranks: {deployment}", flush=True)
        if deployment == "ET3":
            query, info = et3_query_ranks()
        else:
            query, info = gwtc_query_ranks(deployment, source)
        query.to_parquet(args.out_dir / f"{deployment.lower()}_query_ranks.parquet", index=False)
        all_query.append(query)
        provenance[deployment] = info

    query = pd.concat(all_query, ignore_index=True)
    point = metric_rows(query)
    bootstrap = bootstrap_system_ci(query, draws=args.bootstrap_draws, seed=20260713)
    query.to_parquet(args.out_dir / "existing_model_query_ranks_all.parquet", index=False)
    point.to_csv(args.out_dir / "existing_model_retrieval_metrics.csv", index=False)
    bootstrap.to_csv(args.out_dir / "existing_model_system_bootstrap_95ci.csv", index=False)
    write_json(
        args.out_dir / "existing_model_bootstrap_summary.json",
        {
            "status": "complete",
            "scope": "fixed existing model and fixed held-out catalog; no retraining or new strain generation",
            "bootstrap_unit": "lensed system; both directed queries retained together; SIS/PM stratified",
            "bootstrap_draws": int(args.bootstrap_draws),
            "provenance": provenance,
            "primary_metrics": point[
                (point["method"] == "waveform_time_sky") & (point["subset"] == "overall")
            ].to_dict(orient="records"),
        },
    )
    print(json.dumps({"status": "complete", "out_dir": str(args.out_dir)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
