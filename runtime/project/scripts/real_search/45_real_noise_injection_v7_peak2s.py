#!/usr/bin/env python3
"""Formal mixed-population O3/O4a real-noise deployment (v7).

This run supersedes the invalid legacy waveform deployments. It uses physical
H1/L1 source strain, one run-matched unified encoder, one mixed validation/test
catalog, global evidence calibration, and validation-only model/weight/Gate
selection. Real GWTC pairs and PE posteriors are never used for tuning.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


# Fixed before importing the training/evaluation module. Every train,
# validation, held-out test, and real-event input receives the same tail crop:
# [GPS-1.75 s, GPS+0.25 s] at 2048 Hz, i.e. 4096 samples.
os.environ["GW_WAVEFORM_INPUT_SAMPLES"] = "4096"

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.experiments.mainline_uncertainty_common import (
    across_seed_summary,
    bootstrap_system_ci,
    split_integrity_audit,
)
from scripts.real_search.physical_common import write_json
from scripts.real_search.unified_v7_common import (
    FAMILIES,
    apply_waveform_channel,
    build_mixed_pair_table,
    evaluation_tables,
    fit_waveform_channel,
    load_mixed_data,
    score_real_catalog,
    select_fusion_weights,
    waveform_gate,
)


DEFAULT_OUT = REPO / "results/real_noise_injection_v7_peak2s_formal_20260722"
V6_ROOT = REPO / "results/real_noise_injection_v6_unified_physics_20260721"
DEFAULT_DEVELOPMENT_SELECTION = (
    REPO
    / "results/real_noise_injection_v7_peak2s_20260722"
    / "development/frozen_selection_v7_peak2s.json"
)
V5_ROOT = REPO / "results/real_noise_injection_v5_physical_source_20260721"
# Seed 202607231 was used for development and is intentionally excluded from
# the formal run. These source-system splits are fresh and remain unopened
# until the backbone and all hyperparameters have been frozen.
DEFAULT_SEEDS = (202607241, 202607242, 202607243)
VARIANTS_PER_SOURCE = 8
PRETRAIN_EPOCHS = 16
ADAPT_EPOCHS = 60


def module_from(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


v3 = module_from(REPO / "scripts/experiments/20_real_noise_injection_v3_physical.py", "v3_for_v7")
v5 = module_from(REPO / "scripts/real_search/35_real_noise_injection_v5_physical_source.py", "v5_for_v7")


def run_checked(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        subprocess.run(command, cwd=REPO, stdout=handle, stderr=subprocess.STDOUT, check=True)


def ensure_training_material(seed_dir: Path, source_bank: Path, seed: int, samples: int) -> None:
    for family in FAMILIES:
        root = seed_dir / "data/real_noise_injections" / f"multinoise_{family.lower()}_train_v{VARIANTS_PER_SOURCE}"
        if (root / "multinoise_summary.json").exists():
            continue
        run_checked(
            [
                sys.executable,
                str(REPO / "scripts/real_search/36_materialize_multinoise_v5.py"),
                "--seed-root",
                str(seed_dir),
                "--source-bank",
                str(source_bank),
                "--family",
                family,
                "--seed",
                str(seed),
                "--samples",
                str(samples),
                "--variants-per-source",
                str(VARIANTS_PER_SOURCE),
            ],
            seed_dir / "logs" / f"materialize_multinoise_{family.lower()}_v7.log",
        )


def train_unified(
    seed_dir: Path,
    source_bank: Path,
    seed: int,
    samples: int,
    batch_size: int,
    aux_weight: float,
    q_loss_weight: float,
    backbone: str,
) -> Path:
    tag = str(aux_weight).replace(".", "p")
    q_tag = str(q_loss_weight).replace(".", "p")
    out = seed_dir / "waveform_gate" / f"unified_{backbone}_peak2s_4096_aux_{tag}_q_{q_tag}_v7"
    checkpoint = out / "validation_selected_model.pt"
    if checkpoint.exists() and (out / "summary.json").exists():
        return checkpoint
    run_checked(
        [
            sys.executable,
            str(REPO / "scripts/real_search/37_unified_intrinsic_multitask_pilot.py"),
            "--seed-root",
            str(seed_dir),
            "--source-bank",
            str(source_bank),
            "--seed",
            str(seed),
            "--samples",
            str(samples),
            "--pretrain-epochs",
            str(PRETRAIN_EPOCHS),
            "--adapt-epochs",
            str(ADAPT_EPOCHS),
            "--batch-size",
            str(batch_size),
            "--aux-weight",
            str(aux_weight),
            "--q-loss-weight",
            str(q_loss_weight),
            "--backbone",
            backbone,
            "--out-dir",
            str(out),
        ],
        seed_dir / "logs" / "train_unified_v7.log",
    )
    return checkpoint


def historical_rank(frame: pd.DataFrame, event_a: str, event_b: str) -> dict[str, Any]:
    keep = ((frame.event_i == event_a) & (frame.event_j == event_b)) | (
        (frame.event_i == event_b) & (frame.event_j == event_a)
    )
    if not keep.any():
        return {"found": False}
    row = frame.loc[keep].iloc[0]
    return {
        "found": True,
        "rank": int(row["rank"]),
        "final_score": float(row["final_score"]),
        "strict_h1l1_bbh_pair": bool(row["strict_h1l1_bbh_pair"]),
    }


def audit_time_calibration(shared: Path) -> None:
    output = shared / "time_delay_bandwidth_sensitivity_v7.json"
    if output.exists():
        return
    lens = np.load(shared / "time_delay_lensed_samples_days.npy")
    null = np.load(shared / "time_delay_null_samples_days.npy")
    fits = {}
    for scale in (0.5, 1.0, 2.0):
        fits[str(scale)] = v3.fit_time_likelihood_ratio(
            lens,
            null,
            grid_size=2048,
            bandwidth_scale=scale,
        )
    quantiles = np.geomspace(
        max(float(np.quantile(lens, 0.001)), 1e-6),
        float(np.quantile(np.concatenate([lens, null]), 0.999)),
        512,
    )
    scores = {
        key: v3.apply_time_likelihood_ratio(quantiles, fit)
        for key, fit in fits.items()
    }
    write_json(
        output,
        {
            "primary_bandwidth_scale": 1.0,
            "sensitivity_scales": [0.5, 1.0, 2.0],
            "lens_samples": int(len(lens)),
            "null_samples": int(len(null)),
            "max_abs_score_difference_scale_0p5_vs_1": float(np.max(np.abs(scores["0.5"] - scores["1.0"]))),
            "max_abs_score_difference_scale_2_vs_1": float(np.max(np.abs(scores["2.0"] - scores["1.0"]))),
            "median_abs_score_difference_scale_0p5_vs_1": float(np.median(np.abs(scores["0.5"] - scores["1.0"]))),
            "median_abs_score_difference_scale_2_vs_1": float(np.median(np.abs(scores["2.0"] - scores["1.0"]))),
            "lookup_delays_days": quantiles,
            "scores_by_bandwidth": scores,
            "interpretation": "The primary lookup is the scale=1 Gaussian-KDE log likelihood ratio. The other frozen bandwidths are a sensitivity audit, not additional tuning choices.",
        },
    )


def run_seed(
    deployment: str,
    seed: int,
    out_root: Path,
    samples: int,
    batch_size: int,
    aux_weight: float,
    q_loss_weight: float,
    backbone: str,
    bootstrap_draws: int,
) -> dict[str, Any]:
    seed_dir = out_root / deployment.lower() / f"seed_{seed}"
    complete = seed_dir / "seed_summary_v7.json"
    if complete.exists():
        return json.loads(complete.read_text(encoding="utf-8"))
    started = time.perf_counter()
    source_run = v3.SOURCES[deployment]
    shared = V5_ROOT / deployment.lower() / "shared"
    v3.ensure_link(out_root / deployment.lower() / "shared", shared)
    source_bank = shared / "physical_h1l1_source_bank"
    if not (source_bank / "physical_source_bank_summary.json").exists():
        raise FileNotFoundError(f"Missing audited v5 physical source bank: {source_bank}")
    v3.prepare_seed_layout(seed_dir, source_run)
    # The physical injections, run-matched off-source noise, PSDs, SNR draws,
    # and source-system split are unchanged from v6. Reuse those immutable
    # arrays and change only the model-visible fixed waveform crop.
    prior_data = V6_ROOT / deployment.lower() / f"seed_{seed}" / "data/real_noise_injections"
    current_data = seed_dir / "data/real_noise_injections"
    if not current_data.exists() and not current_data.is_symlink() and prior_data.exists():
        current_data.symlink_to(prior_data.resolve(), target_is_directory=True)
    schedule = v3.build_live_schedule(deployment, source_run, shared)
    v3.build_noise_bank(
        source_run,
        shared,
        20260721 + (0 if deployment == "GWTC3" else 1000),
    )
    v3.build_sky_template_library(source_run, shared)
    raw_delays, _, _ = v3.liao_delay_ratio_samples()
    write_json(
        shared / "time_delay_prior_source_audit_v7.json",
        {
            "source": "GW-LMC 2.5PLUS BBH Any_Detected_SNR8",
            "raw_detectable_doublet_delay_count": int(len(raw_delays)),
            "raw_delay_days_quantiles": {
                str(q): float(np.quantile(raw_delays, q))
                for q in (0.0, 0.01, 0.1, 0.5, 0.9, 0.99, 1.0)
            },
            "limitation": "The 100,000 signal draws used by the KDE are bootstrap/exposure-conditioned draws from this finite external catalog; they are not 100,000 independent lens systems. Bandwidth sensitivity is reported separately.",
        },
    )
    v3.build_time_calibration(
        shared,
        schedule,
        raw_delays,
        20260722 + (0 if deployment == "GWTC3" else 1000),
    )
    audit_time_calibration(shared)
    v5.materialize_dataset_v5(seed_dir, shared, deployment, seed + 100, samples)
    ensure_training_material(seed_dir, source_bank, seed, samples)

    data = load_mixed_data(seed_dir, source_bank, seed, samples)
    split_audit = split_integrity_audit(
        {
            f"{family}_lensed": data[family].parts["lensed"]
            for family in FAMILIES
        }
        | {"unlensed": data[FAMILIES[0]].parts["unlensed"]}
    )
    if not split_audit["all_disjoint"]:
        raise RuntimeError("Train/validation/test source-system leakage detected")
    write_json(seed_dir / "results/split_integrity_audit_v7.json", split_audit)

    checkpoint = train_unified(
        seed_dir,
        source_bank,
        seed,
        samples,
        batch_size,
        aux_weight,
        q_loss_weight,
        backbone,
    )
    time_calibration = json.loads((shared / "time_delay_likelihood_ratio.json").read_text(encoding="utf-8"))
    validation = build_mixed_pair_table(
        seed_dir,
        shared,
        source_bank,
        checkpoint,
        seed,
        samples,
        "val",
        time_calibration,
        seed + 2000,
    )
    waveform_config, waveform_grid = fit_waveform_channel(
        validation,
        allow_q_feature=q_loss_weight > 0,
    )
    validation = apply_waveform_channel(validation, waveform_config)
    waveform_grid.to_csv(seed_dir / "results/waveform_intrinsic_grid_validation_v7.csv", index=False)
    write_json(seed_dir / "results/waveform_channel_calibration_v7.json", waveform_config)
    validation.to_parquet(seed_dir / "results/fusion_validation_pairs_v7.parquet", index=False)

    retrieval_free, grid_retrieval_free = select_fusion_weights(validation, objective="retrieval")
    retrieval_positive, grid_retrieval_positive = select_fusion_weights(
        validation, require_all_positive=True, objective="retrieval"
    )
    candidate_free, grid_candidate_free = select_fusion_weights(validation, objective="candidate")
    candidate_positive, grid_candidate_positive = select_fusion_weights(
        validation, require_all_positive=True, objective="candidate"
    )
    time_sky_retrieval, grid_time_sky_retrieval = select_fusion_weights(
        validation, waveform_zero=True, objective="retrieval"
    )
    time_sky_candidate, grid_time_sky_candidate = select_fusion_weights(
        validation, waveform_zero=True, objective="candidate"
    )
    for name, grid in {
        "fusion_weight_grid_retrieval_unconstrained_v7.csv": grid_retrieval_free,
        "fusion_weight_grid_retrieval_strictly_positive_v7.csv": grid_retrieval_positive,
        "fusion_weight_grid_candidate_unconstrained_v7.csv": grid_candidate_free,
        "fusion_weight_grid_candidate_strictly_positive_v7.csv": grid_candidate_positive,
        "fusion_weight_grid_time_sky_retrieval_v7.csv": grid_time_sky_retrieval,
        "fusion_weight_grid_time_sky_candidate_v7.csv": grid_time_sky_candidate,
    }.items():
        grid.to_csv(seed_dir / "results" / name, index=False)
    gate = waveform_gate(validation, candidate_free, time_sky_candidate, seed=seed)
    write_json(seed_dir / "results/waveform_gate_validation_v7.json", gate)
    weights = {
        "selection_split": "synthetic validation systems only",
        "retrieval_objective": "macro R@10, minimum-family R@10, pair AUPRC, precision at 50% recall, macro R@1, then lower L2 norm",
        "candidate_objective": "pair AUPRC, precision at 50% recall, macro R@10, minimum-family R@10, macro R@1, then lower L2 norm",
        "retrieval_three_channel_unconstrained": retrieval_free,
        "retrieval_three_channel_strictly_positive": retrieval_positive,
        "candidate_three_channel_unconstrained": candidate_free,
        "candidate_three_channel_strictly_positive": candidate_positive,
        "time_sky_retrieval_selected": time_sky_retrieval,
        "time_sky_candidate_selected": time_sky_candidate,
        "three_channel_equal_evidence": {"waveform": 1.0, "time": 1.0, "sky": 1.0},
    }
    write_json(seed_dir / "results/selected_weights_v7.json", weights)

    # Only now instantiate/evaluate the held-out test catalog.
    test = build_mixed_pair_table(
        seed_dir,
        shared,
        source_bank,
        checkpoint,
        seed,
        samples,
        "test",
        time_calibration,
        seed + 3000,
    )
    test = apply_waveform_channel(test, waveform_config)
    test.to_parquet(seed_dir / "results/fusion_heldout_test_pairs_v7.parquet", index=False)
    methods = {
        "waveform_only": {"waveform": 1.0, "time": 0.0, "sky": 0.0},
        "time_only": {"waveform": 0.0, "time": 1.0, "sky": 0.0},
        "sky_bayes_factor_only": {"waveform": 0.0, "time": 0.0, "sky": 1.0},
        "time_sky_retrieval_selected": time_sky_retrieval,
        "time_sky_candidate_selected": time_sky_candidate,
        "retrieval_three_channel_unconstrained": retrieval_free,
        "retrieval_three_channel_strictly_positive": retrieval_positive,
        "candidate_three_channel_unconstrained": candidate_free,
        "candidate_three_channel_strictly_positive": candidate_positive,
        "three_channel_equal_evidence": {"waveform": 1.0, "time": 1.0, "sky": 1.0},
    }
    query, retrieval, pair = evaluation_tables(test, methods, deployment, seed)
    query.to_parquet(seed_dir / "results/heldout_test_query_ranks_v7.parquet", index=False)
    retrieval.to_csv(seed_dir / "results/heldout_test_retrieval_metrics_v7.csv", index=False)
    pair.to_csv(seed_dir / "results/heldout_test_pair_level_metrics_v7.csv", index=False)
    bootstrap = bootstrap_system_ci(query, draws=bootstrap_draws, seed=seed + 4000)
    bootstrap.to_csv(seed_dir / "results/heldout_test_bootstrap_95ci_v7.csv", index=False)

    validation_embedding = np.load(seed_dir / "results/mixed_val_unified_embeddings_v7.npy")
    real_methods = {
        "retrieval_three_channel_unconstrained": retrieval_free,
        "retrieval_three_channel_strictly_positive": retrieval_positive,
        "candidate_three_channel_unconstrained": candidate_free,
        "candidate_three_channel_strictly_positive": candidate_positive,
        "three_channel_equal_evidence": {"waveform": 1.0, "time": 1.0, "sky": 1.0},
        "time_sky_retrieval_selected": time_sky_retrieval,
        "time_sky_candidate_selected": time_sky_candidate,
    }
    real, deployment_audit = score_real_catalog(
        deployment,
        seed_dir,
        shared,
        real_methods,
        checkpoint,
        waveform_config,
        time_calibration,
        validation_embedding,
        seed,
    )
    waveform_eligible = bool(gate["passed_for_primary_deployment"] and deployment_audit["passed"])
    if waveform_eligible and candidate_free["waveform"] > 0:
        primary = "candidate_three_channel_unconstrained"
        reason = "Waveform passed the validation discrimination/incremental-utility gates and the real-event non-collapse audit; the candidate-generation objective selected positive waveform weight."
    else:
        primary = "time_sky_candidate_selected"
        reason = "Predeclared fallback: waveform failed a validation/deployment gate or received zero validation weight."
    result_dir = seed_dir / "results"
    for scope in ("strict", "all"):
        source = real[primary][scope]
        source.to_parquet(result_dir / f"real_pair_scores_{scope}_primary_v7.parquet", index=False)
        source.head(100).to_csv(result_dir / f"candidate_shortlist_{scope}_primary_v7.csv", index=False)
    strict = real[primary]["strict"]
    all_catalog = real[primary]["all"]
    summary = {
        "status": "complete",
        "deployment": deployment,
        "seed": int(seed),
        "samples_per_gwlmc_group": int(samples),
        "single_mixed_validation_and_test_catalog": True,
        "auxiliary_weight_fixed_before_formal_test": float(aux_weight),
        "q_loss_weight_fixed_before_formal_test": float(q_loss_weight),
        "backbone_fixed_before_formal_test": backbone,
        "checkpoint": str(checkpoint),
        "validation_gate": gate,
        "real_waveform_deployment_audit": deployment_audit,
        "waveform_eligible_for_primary": waveform_eligible,
        "primary_method": primary,
        "primary_method_reason": reason,
        "weights": weights,
        "strict_primary_top_pair": strict.iloc[0].to_dict() if len(strict) else {},
        "all_catalog_primary_top_pair": all_catalog.iloc[0].to_dict() if len(all_catalog) else {},
        "gw170104_gw170814": historical_rank(all_catalog, "GW170104", "GW170814") if deployment == "GWTC3" else {"found": False},
        "pe_used_for_tuning": False,
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    write_json(complete, summary)
    return summary


def aggregate(out_root: Path, deployment: str, seeds: list[int]) -> None:
    root = out_root / deployment.lower()
    query = pd.concat(
        [pd.read_parquet(root / f"seed_{seed}/results/heldout_test_query_ranks_v7.parquet") for seed in seeds],
        ignore_index=True,
    )
    retrieval = pd.concat(
        [pd.read_csv(root / f"seed_{seed}/results/heldout_test_retrieval_metrics_v7.csv") for seed in seeds],
        ignore_index=True,
    )
    pair = pd.concat(
        [pd.read_csv(root / f"seed_{seed}/results/heldout_test_pair_level_metrics_v7.csv") for seed in seeds],
        ignore_index=True,
    )
    bootstrap = pd.concat(
        [pd.read_csv(root / f"seed_{seed}/results/heldout_test_bootstrap_95ci_v7.csv") for seed in seeds],
        ignore_index=True,
    )
    query.to_parquet(root / "heldout_test_query_ranks_all_seeds_v7.parquet", index=False)
    retrieval.to_csv(root / "heldout_test_retrieval_metrics_per_seed_v7.csv", index=False)
    across_seed_summary(retrieval).to_csv(root / "heldout_test_retrieval_metrics_across_seed_v7.csv", index=False)
    pair.to_csv(root / "heldout_test_pair_level_metrics_per_seed_v7.csv", index=False)
    pair_for_summary = pair.copy()
    pair_for_summary["subset"] = "overall"
    across_seed_summary(pair_for_summary).to_csv(
        root / "heldout_test_pair_level_metrics_across_seed_v7.csv", index=False
    )
    bootstrap.to_csv(root / "heldout_test_bootstrap_95ci_per_seed_v7.csv", index=False)
    summaries = [json.loads((root / f"seed_{seed}/seed_summary_v7.json").read_text(encoding="utf-8")) for seed in seeds]
    write_json(
        root / "deployment_summary_v7.json",
        {
            "deployment": deployment,
            "seeds": seeds,
            "all_seeds_complete": len(summaries) == len(seeds),
            "waveform_primary_seed_count": int(sum(x["waveform_eligible_for_primary"] for x in summaries)),
            "primary_methods": {str(x["seed"]): x["primary_method"] for x in summaries},
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--deployments", nargs="+", choices=("GWTC3", "GWTC4"), default=["GWTC3", "GWTC4"])
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--samples-per-group", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--aux-weight", type=float, required=True)
    parser.add_argument("--q-loss-weight", type=float, required=True)
    parser.add_argument(
        "--backbone",
        choices=("trigger_spectrogram", "inceptiontime", "inception_attention"),
        required=True,
    )
    parser.add_argument(
        "--development-selection",
        type=Path,
        default=DEFAULT_DEVELOPMENT_SELECTION,
    )
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if not args.development_selection.exists():
        raise FileNotFoundError(
            "Formal held-out evaluation requires a completed validation-only "
            f"development selection artifact: {args.development_selection}"
        )
    development_selection = json.loads(
        args.development_selection.read_text(encoding="utf-8")
    )
    if not np.isclose(
        float(development_selection["selected_aux_weight"]), args.aux_weight
    ) or not np.isclose(
        float(development_selection["selected_q_loss_weight"]), args.q_loss_weight
    ):
        raise ValueError(
            "Requested formal auxiliary settings do not match the frozen "
            "validation-only development selection"
        )
    if str(development_selection.get("selected_backbone", "")) != args.backbone:
        raise ValueError(
            "Requested backbone does not match the frozen v7 validation-only "
            "development selection"
        )
    args.out_root.mkdir(parents=True, exist_ok=True)
    config = {
        "protocol": "real_noise_injection_v7_peak2s",
        "status": "running",
        "deployments": args.deployments,
        "seeds": args.seeds,
        "samples_per_gwlmc_group": args.samples_per_group,
        "group_mapping": {
            "SIS": "GW-LMC smooth/non-subhalo compatibility slot",
            "PM": "GW-LMC subhalo-present compatibility slot",
        },
        "single_unified_encoder": True,
        "single_mixed_catalog": True,
        "waveform_input_window": {
            "definition": "fixed tail crop from the common preprocessed 24 s array",
            "gps_interval": "[GPS-1.75 s, GPS+0.25 s]",
            "sample_rate_hz": 2048,
            "samples": 4096,
            "selection_status": "specified before this formal run; not selected on held-out test or real PE",
        },
        "unchanged_from_v6": [
            "physical H1/L1 source strain",
            "run-matched GWOSC off-source noise and PSD whitening",
            "empirical detected-event SNR proposal",
            "loss, augmentation, epoch budget, and auxiliary weights",
            "time-delay likelihood ratio and sky common-source Bayes factor",
        ],
        "changed_from_v6_and_frozen_on_development_validation": [
            "4096-sample (2 s) model input ending 0.25 s after GPS",
            "InceptionTime backbone selected from development validation only",
            "fresh formal source-system splits and training seeds",
        ],
        "development_seed_excluded_from_formal_results": 202607231,
        "clean_pretrain_epochs": PRETRAIN_EPOCHS,
        "real_noise_adaptation_epochs": ADAPT_EPOCHS,
        "aux_weight_development_selected": args.aux_weight,
        "q_loss_weight_development_selected": args.q_loss_weight,
        "backbone_development_selected": args.backbone,
        "development_selection_artifact": str(args.development_selection),
        "development_selection": development_selection,
        "primary_channels": [
            "calibrated waveform evidence",
            "run-exposure-conditioned time-delay log likelihood ratio",
            "HEALPix common-source sky log Bayes factor",
        ],
        "snr_in_final_score": False,
        "real_catalog_or_pe_used_for_tuning": False,
        "primary_catalog": "strict complete H1/L1 non-OOD BBH subset",
        "full_catalog": "supplementary, neutral waveform evidence for missing waveform pairs",
    }
    write_json(args.out_root / "run_config_v7.json", config)
    for deployment in args.deployments:
        completed = []
        for seed in args.seeds:
            run_seed(
                deployment,
                seed,
                args.out_root,
                args.samples_per_group,
                args.batch_size,
                args.aux_weight,
                args.q_loss_weight,
                args.backbone,
                args.bootstrap_draws,
            )
            completed.append(seed)
            aggregate(args.out_root, deployment, completed)
    config["status"] = "complete"
    write_json(args.out_root / "run_config_v7.json", config)
    print(json.dumps({"status": "complete", "out_root": str(args.out_root)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
