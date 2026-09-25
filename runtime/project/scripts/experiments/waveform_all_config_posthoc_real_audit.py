#!/usr/bin/env python3
"""Post-hoc real-catalog audit of every trained G2/G3/G4 waveform variant.

This script is deliberately outside the registered G0-G9 selection path.  It
does not retrain a model or alter the failed G7 decision.  All real-catalog PE
and literature fields are attached only after scores and ranks are frozen.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT = Path("/root/autodl-tmp/gw-catalog")
SOURCE_EXPERIMENT = PROJECT / "results/waveform_domain_multiscale_exploratory_20260903_20260903T063500Z"
BASELINE = PROJECT / "results/bayestar_injection_sky_pe_20260901_20260901T102000Z"
BASELINE_PACKAGE = PROJECT / "packages/bayestar_injection_sky_pe_20260901_20260901T102000Z_deliverables.tar.gz"
BASELINE_PACKAGE_SHA256 = "3ea3cf13992a5154e0dd757ab6087b73344cf28eb0ccf5e4a8a903b0e83850af"
FINAL_STATUS = "POSTHOC_DIAGNOSTIC_HOLD_NO_SELECTION_NO_ADOPTION_NO_OVERWRITE"
DEPLOYMENTS = ("gwtc3", "gwtc4")
MODEL_SEEDS = (202609031, 202609032, 202609033)
EVALUATION_SEEDS = (202607241, 202607242, 202607243)


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ORCH = load_module(SOURCE_EXPERIMENT / "scripts/waveform_domain_multiscale_exploratory.py", "posthoc_orchestrator")
TRAIN = load_module(SOURCE_EXPERIMENT / "scripts/waveform_multiscale_train.py", "posthoc_trainer")
BASE = ORCH.BASE


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    temporary.replace(path)


def table_markdown(frame: pd.DataFrame, digits: int = 4) -> str:
    shown = frame.copy()
    for column in shown.select_dtypes(include=[np.number]).columns:
        shown[column] = shown[column].map(lambda value: "" if pd.isna(value) else f"{value:.{digits}f}")
    return shown.to_markdown(index=False)


def experiment_registry() -> list[dict[str, Any]]:
    root = SOURCE_EXPERIMENT / "models"
    single = 202609031
    return [
        {
            "stage": "G2/G3/G4 reference", "config_id": "D1-2s_base_single_seed",
            "change": "2 s base model", "seed_paths": [(single, 202607241, root / "G2/D1-2s_base/{deployment}/seed_202609031/validation_selected_model.pt")],
        },
        {
            "stage": "G2", "config_id": "D1-2plus8s_base",
            "change": "2 s + 8 s input branches", "seed_paths": [(single, 202607241, root / "G2/D1-2plus8s_base/{deployment}/seed_202609031/validation_selected_model.pt")],
        },
        {
            "stage": "G2", "config_id": "D1-2plus16s_base",
            "change": "2 s + 16 s input branches", "seed_paths": [(single, 202607241, root / "G2/D1-2plus16s_base/{deployment}/seed_202609031/validation_selected_model.pt")],
        },
        {
            "stage": "G3", "config_id": "D1-2s_H",
            "change": "2 s + physics hard negatives", "seed_paths": [(single, 202607241, root / "G3/D1-2s_H/{deployment}/seed_202609031/validation_selected_model.pt")],
        },
        {
            "stage": "G4", "config_id": "D1-2s_U",
            "change": "2 s + chirp-mass uncertainty head", "seed_paths": [(single, 202607241, root / "G4/D1-2s_U/{deployment}/seed_202609031/validation_selected_model.pt")],
        },
        {
            "stage": "G4", "config_id": "D1-2s_E",
            "change": "2 s + embedding regularization", "seed_paths": [(single, 202607241, root / "G4/D1-2s_E/{deployment}/seed_202609031/validation_selected_model.pt")],
        },
        {
            "stage": "G5-G7", "config_id": "D1-2s_base_three_seed",
            "change": "selected 2 s base repeated over three seeds",
            "seed_paths": [
                (seed, eval_seed, root / f"final/{{deployment}}/seed_{seed}/validation_selected_model.pt")
                for seed, eval_seed in zip(MODEL_SEEDS, EVALUATION_SEEDS, strict=True)
            ],
        },
    ]


def stage_design_table() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "stage": "G0", "comparison": "historical C-fixed baseline",
            "relationship_to_previous": "frozen reference",
            "selection_rule": "BAYESTAR baseline identity and package SHA-256 must reproduce",
            "real_GWTC_PE_or_official_used_for_selection": False,
        },
        {
            "stage": "G2", "comparison": "2 s vs 2+8 s vs 2+16 s",
            "relationship_to_previous": "one-factor window ablation from the same D1 base",
            "selection_rule": "both runs validation-noninferior (R@10 margin 0.02; AUPRC margin 0.005), then low-mass R@10, overall R@10/AUPRC, shortest near-tied window",
            "real_GWTC_PE_or_official_used_for_selection": False,
        },
        {
            "stage": "G3", "comparison": "selected G2 model without vs with hard negatives",
            "relationship_to_previous": "adds one training-data factor to the G2 winner",
            "selection_rule": "both runs validation-noninferior and mean Top-100 catastrophic-mass false fraction reduced by at least 10%",
            "real_GWTC_PE_or_official_used_for_selection": False,
        },
        {
            "stage": "G4", "comparison": "base vs U and base vs E; U+E only if U and E separately pass",
            "relationship_to_previous": "separate architecture ablations on the G3 winner",
            "selection_rule": "U: noninferior, log-Mc MAE improves at least 5%, 90% coverage in [0.75,0.98]; E: noninferior and effective rank improves at least 10%",
            "real_GWTC_PE_or_official_used_for_selection": False,
        },
        {
            "stage": "G5-G6", "comparison": "repeat the unique selected configuration over three seeds and calibrate by run",
            "relationship_to_previous": "no new feature; replication and validation-only calibration",
            "selection_rule": "report all seeds; O3 and O4a waveform calibrations are fit separately on validation",
            "real_GWTC_PE_or_official_used_for_selection": False,
        },
        {
            "stage": "G7", "comparison": "new three-seed waveform vs frozen C-fixed waveform baseline",
            "relationship_to_previous": "one-time locked-test gate",
            "selection_rule": "both runs mean R@10/AUPRC noninferior, at least 2/3 seeds noninferior, and F50/F90 no more than about 10% worse",
            "real_GWTC_PE_or_official_used_for_selection": False,
        },
        {
            "stage": "post-hoc addendum", "comparison": "every trained checkpoint on real GWTC PE and official-stage fields",
            "relationship_to_previous": "descriptive audit only; no promotion or retraining",
            "selection_rule": "not eligible for model selection; preserves original G7 FAIL",
            "real_GWTC_PE_or_official_used_for_selection": False,
        },
    ])


def checkpoint_path(template: Path, deployment: str) -> Path:
    return Path(str(template).format(deployment=deployment))


def metric_row(
    stage: str,
    config_id: str,
    change: str,
    deployment: str,
    model_seed: int,
    evaluation_seed: int,
    split: str,
    method: str,
    frame: pd.DataFrame,
    scores: np.ndarray,
) -> dict[str, Any]:
    metrics = BASE.full_metrics(frame, scores)
    return {
        "stage": stage, "config_id": config_id, "change": change,
        "deployment": deployment, "model_seed": model_seed, "evaluation_seed": evaluation_seed,
        "split": split, "method": method, "n_events": int(frame.event_count.iloc[0]),
        "n_pairs": len(frame), "n_true_pairs": int(frame.is_true_pair.sum()), **metrics,
    }


def attach_pair_audit(frame: pd.DataFrame, deployment: str) -> pd.DataFrame:
    audit = pd.read_parquet(BASELINE / f"results/{deployment}/real_consensus_with_pe_official_C_fixed.parquet")
    excluded = {
        "consensus_rank", "method", "seed_count", "rank_mean", "rank_sd", "rank_min", "rank_max",
        "final_score_mean", "waveform_score_mean", "time_score_mean", "sky_raw_log_bf_mean",
        "sky_bc_mean", "sky_j50_mean", "sky_j90_mean", "waveform_contribution_mean",
        "time_contribution_mean", "sky_contribution_mean",
    }
    columns = ["pair_key"] + [column for column in audit.columns if column not in excluded and column not in {"pair_key", "event_i", "event_j"}]
    return frame.merge(audit[columns], on="pair_key", how="left", validate="one_to_one")


def budget_rows(
    stage: str,
    config_id: str,
    change: str,
    deployment: str,
    method: str,
    n_seeds: int,
    frame: pd.DataFrame,
) -> list[dict[str, Any]]:
    rows = []
    for budget in (10, 20, 50, 100):
        part = frame.head(budget)
        mc_bc = pd.to_numeric(part.chirp_mass_bhattacharyya_coefficient, errors="coerce")
        mc_d = pd.to_numeric(part.chirp_mass_standardized_posterior_distance, errors="coerce")
        dmax = pd.to_numeric(part.max_standardized_posterior_distance, errors="coerce")
        rows.append({
            "stage": stage, "config_id": config_id, "change": change,
            "deployment": deployment, "method": method, "n_model_seeds": n_seeds,
            "budget": budget, "n_pairs": len(part),
            "chirp_mass_BC_ge_0p5": int((mc_bc >= 0.5).sum()),
            "median_chirp_mass_BC": float(mc_bc.median()),
            "chirp_mass_D_le_3": int((mc_d <= 3).sum()),
            "Dmax_le_3": int((dmax <= 3).sum()),
            "catastrophic_chirp_mass": int(((mc_bc < 0.1) | (mc_d > 5)).sum()),
            "official_frontend_overlap": int(part.official_frontend_overlap.fillna(False).astype(bool).sum()),
            "public_hanabi_table_overlap": int(part.public_hanabi_table_overlap.fillna(False).astype(bool).sum()),
            "official_po_ml_fpp_available": int(pd.to_numeric(part.official_po_ml_fpp, errors="coerce").notna().sum()),
            "official_po_phazap_fpp_available": int(pd.to_numeric(part.official_po_phazap_fpp, errors="coerce").notna().sum()),
            "interpretation": "post-hoc descriptive audit; not a model-selection metric",
        })
    return rows


def prepare_contract(root: Path) -> None:
    source_g7 = json.loads((SOURCE_EXPERIMENT / "contracts/G7_LOCKED_TEST_COMPLETE.json").read_text(encoding="utf-8"))
    if source_g7.get("decision") != "FAIL_STOP_BEFORE_G8":
        raise RuntimeError("Unexpected source G7 state")
    if sha256_file(BASELINE_PACKAGE) != BASELINE_PACKAGE_SHA256:
        raise RuntimeError("BAYESTAR baseline package hash mismatch")
    registry_rows = []
    for experiment in experiment_registry():
        for deployment in DEPLOYMENTS:
            for model_seed, evaluation_seed, template in experiment["seed_paths"]:
                checkpoint = checkpoint_path(template, deployment)
                if not checkpoint.exists():
                    raise FileNotFoundError(checkpoint)
                registry_rows.append({
                    "stage": experiment["stage"], "config_id": experiment["config_id"],
                    "change": experiment["change"], "deployment": deployment,
                    "model_seed": model_seed, "evaluation_seed": evaluation_seed,
                    "checkpoint": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint),
                })
    write_csv(root / "contracts/CONFIG_REGISTRY.csv", pd.DataFrame(registry_rows))
    write_json(root / "contracts/POSTHOC_ANALYSIS_CONTRACT.json", {
        "created_utc": utc_stamp(),
        "status": FINAL_STATUS,
        "source_experiment": str(SOURCE_EXPERIMENT),
        "source_G7_decision": source_g7["decision"],
        "purpose": "Describe real-catalog PE and official-stage behavior for every already-trained variant",
        "selection_or_retraining_allowed": False,
        "changes_to_source_gate_allowed": False,
        "time_channel": "Frozen one-dimensional Z_time from BAYESTAR C-fixed baseline",
        "sky_channel": "Frozen event-level BAYESTAR Nside=512 Z_sky from C-fixed baseline",
        "fusion_weights": "Frozen per-seed v9.3 weights used by C-fixed; no retuning",
        "waveform_calibration": "Refit independently for each variant using its matching frozen validation split only",
        "real_PE_or_official_fields_used_in_scoring": False,
        "baseline_package_sha256": BASELINE_PACKAGE_SHA256,
        "not_authorized_claims": ["model selection", "replacement of C-fixed", "lensing detection", "official-candidate truth labels"],
    })


def run(root: Path) -> None:
    for directory in ("contracts", "configs", "results", "tables", "figures", "reports", "scripts", "manifest", "logs"):
        (root / directory).mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__), root / "scripts" / Path(__file__).name)
    prepare_contract(root)
    design = stage_design_table()
    write_csv(root / "tables/STAGE_DESIGN_AND_SELECTION_RULES.csv", design)

    metric_rows: list[dict[str, Any]] = []
    budget_output: list[dict[str, Any]] = []
    top_output: list[pd.DataFrame] = []
    correlation_rows: list[dict[str, Any]] = []
    known_pair_rows: list[dict[str, Any]] = []
    invariant_rows: list[dict[str, Any]] = []
    known_pairs = {
        "GW190924_021846--GW191105_143521",
        "GW190412--GW191204_171526",
        "GW190924_021846--GW190930_133541",
    }

    frozen_metrics = pd.read_csv(BASELINE / "results/locked_test_retrieval_metrics_per_seed.csv")
    for source_method, method in (("waveform_only", "waveform_only"), ("BAYESTAR_C_fixed", "C_fixed_fusion")):
        selected = frozen_metrics.loc[frozen_metrics.method.eq(source_method)]
        for row in selected.to_dict("records"):
            metric_rows.append({
                "stage": "G0", "config_id": "historical_C_fixed", "change": "frozen BAYESTAR C-fixed reference",
                "deployment": row["deployment"], "model_seed": np.nan, "evaluation_seed": row["seed"],
                "split": "locked_test_frozen_reference", "method": method,
                **{
                    key: value for key, value in row.items()
                    if key not in {"stage", "config_id", "change", "deployment", "model_seed",
                                   "evaluation_seed", "seed", "split", "method"}
                },
            })
    for deployment in DEPLOYMENTS:
        frozen_real = pd.read_parquet(BASELINE / f"results/{deployment}/real_consensus_with_pe_official_C_fixed.parquet")
        frozen_real = frozen_real.sort_values("consensus_rank", kind="stable").copy()
        frozen_real.insert(0, "deployment", deployment)
        frozen_real.insert(0, "change", "frozen BAYESTAR C-fixed reference")
        frozen_real.insert(0, "config_id", "historical_C_fixed")
        frozen_real.insert(0, "stage", "G0")
        shown = frozen_real.head(100).copy()
        shown.insert(4, "ranking_method", "C_fixed_fusion")
        top_output.append(shown)
        budget_output.extend(budget_rows(
            "G0", "historical_C_fixed", "frozen BAYESTAR C-fixed reference",
            deployment, "C_fixed_fusion", 3, frozen_real,
        ))
        tracked = frozen_real.loc[frozen_real.pair_key.astype(str).isin(known_pairs)]
        for row in tracked.itertuples(index=False):
            known_pair_rows.append({
                "stage": "G0", "config_id": "historical_C_fixed", "deployment": deployment,
                "method": "C_fixed_fusion", "pair_key": row.pair_key, "rank": row.consensus_rank,
                "waveform_score_mean": row.waveform_score_mean,
                "chirp_mass_BC": row.chirp_mass_bhattacharyya_coefficient,
                "chirp_mass_D": row.chirp_mass_standardized_posterior_distance,
                "Dmax": row.max_standardized_posterior_distance,
                "official_frontend_overlap": row.official_frontend_overlap,
                "public_hanabi_table_overlap": row.public_hanabi_table_overlap,
                "official_po_ml_fpp": row.official_po_ml_fpp,
                "official_po_phazap_fpp": row.official_po_phazap_fpp,
                "official_tier": row.official_tier,
                "official_fast_golum_stage": row.official_fast_golum_stage,
                "official_hanabi_stage": row.official_hanabi_stage,
            })

    for experiment in experiment_registry():
        stage = experiment["stage"]
        config_id = experiment["config_id"]
        change = experiment["change"]
        for deployment in DEPLOYMENTS:
            ranked_by_method: dict[str, list[pd.DataFrame]] = {"waveform_only": [], "C_fixed_fusion": []}
            for model_seed, evaluation_seed, template in experiment["seed_paths"]:
                checkpoint = checkpoint_path(template, deployment)
                validation_plan = BASE.retained_event_plan(deployment, evaluation_seed, "validation")
                validation = BASE.subset_pair_table(deployment, evaluation_seed, "validation", validation_plan, allow_test_scores=False)
                validation_events = ORCH.event_array_for_plan(deployment, evaluation_seed, "validation", validation_plan)
                embedding, mean, sigma = TRAIN.encode_full24(checkpoint, validation_events)
                components = ORCH.component_values(validation, embedding, mean, sigma)
                calibration, grid = ORCH.fit_waveform_calibration(validation, components)
                raw, waveform_score = ORCH.apply_waveform_calibration(components, calibration)
                validation = ORCH.enrich_waveform_frame(validation, components, raw, waveform_score)
                validation = ORCH.attach_frozen_time_sky(
                    validation,
                    pd.read_parquet(BASELINE / f"results/{deployment}/seed_{evaluation_seed}/validation_pair_scores_bayestar_sky.parquet"),
                )
                weights = BASE.FROZEN_V93_WEIGHTS[deployment][evaluation_seed]
                metric_rows.append(metric_row(stage, config_id, change, deployment, model_seed, evaluation_seed, "validation", "waveform_only", validation, waveform_score))
                metric_rows.append(metric_row(stage, config_id, change, deployment, model_seed, evaluation_seed, "validation", "C_fixed_fusion", validation, BASE.score_vector(validation, weights)))

                config_directory = root / f"configs/{config_id}/{deployment}/seed_{model_seed}"
                write_json(config_directory / "waveform_calibration.json", calibration)
                write_csv(config_directory / "waveform_component_grid.csv", grid)

                test_plan = BASE.retained_event_plan(deployment, evaluation_seed, "test")
                test = BASE.subset_pair_table(deployment, evaluation_seed, "test", test_plan, allow_test_scores=True)
                test_events = ORCH.event_array_for_plan(deployment, evaluation_seed, "test", test_plan)
                embedding, mean, sigma = TRAIN.encode_full24(checkpoint, test_events)
                components = ORCH.component_values(test, embedding, mean, sigma)
                raw, waveform_score = ORCH.apply_waveform_calibration(components, calibration)
                test = ORCH.enrich_waveform_frame(test, components, raw, waveform_score)
                test = ORCH.attach_frozen_time_sky(
                    test,
                    pd.read_parquet(BASELINE / f"results/{deployment}/seed_{evaluation_seed}/test_pair_scores_bayestar_sky.parquet"),
                )
                test_reference = pd.read_parquet(
                    BASELINE / f"results/{deployment}/seed_{evaluation_seed}/test_pair_scores_bayestar_sky.parquet"
                )
                if not (
                    np.array_equal(test.idx_i.to_numpy(), test_reference.idx_i.to_numpy())
                    and np.array_equal(test.idx_j.to_numpy(), test_reference.idx_j.to_numpy())
                ):
                    raise RuntimeError("Locked-test pair order changed during post-hoc audit")
                for column in ("time_score", "sky_raw_log_bf"):
                    current_values = test[column].to_numpy(dtype=float)
                    reference_values = test_reference[column].to_numpy(dtype=float)
                    invariant_rows.append({
                        "stage": stage, "config_id": config_id, "deployment": deployment,
                        "model_seed": model_seed, "evaluation_seed": evaluation_seed,
                        "dataset": "locked_test", "channel": column,
                        "n_pairs": len(test),
                        "max_abs_difference": float(np.max(np.abs(current_values - reference_values))),
                    })
                metric_rows.append(metric_row(stage, config_id, change, deployment, model_seed, evaluation_seed, "locked_test_posthoc", "waveform_only", test, waveform_score))
                metric_rows.append(metric_row(stage, config_id, change, deployment, model_seed, evaluation_seed, "locked_test_posthoc", "C_fixed_fusion", test, BASE.score_vector(test, weights)))

                real = pd.read_parquet(BASELINE / f"results/{deployment}/seed_{evaluation_seed}/real_strict_pair_scores_C_fixed.parquet")
                embedding, mean, sigma = TRAIN.encode_full24(checkpoint, ORCH.real_full24(deployment))
                components = ORCH.component_values(real, embedding, mean, sigma)
                raw, waveform_score = ORCH.apply_waveform_calibration(components, calibration)
                real = ORCH.enrich_waveform_frame(real, components, raw, waveform_score)
                real_reference = pd.read_parquet(
                    BASELINE / f"results/{deployment}/seed_{evaluation_seed}/real_strict_pair_scores_C_fixed.parquet"
                )
                for column in ("time_score", "sky_raw_log_bf"):
                    reference = real[["pair_key", column]].merge(
                        real_reference[["pair_key", column]], on="pair_key", suffixes=("_new", "_reference"),
                        validate="one_to_one",
                    )
                    invariant_rows.append({
                        "stage": stage, "config_id": config_id, "deployment": deployment,
                        "model_seed": model_seed, "evaluation_seed": evaluation_seed,
                        "dataset": "real_catalog", "channel": column,
                        "n_pairs": len(reference),
                        "max_abs_difference": float(np.max(np.abs(
                            reference[f"{column}_new"].to_numpy(dtype=float)
                            - reference[f"{column}_reference"].to_numpy(dtype=float)
                        ))),
                    })
                ranked_by_method["waveform_only"].append(BASE.rank_real(
                    real, {"waveform": 1.0, "time": 0.0, "sky": 0.0},
                    f"{config_id}_posthoc_waveform_only", model_seed,
                ))
                ranked_by_method["C_fixed_fusion"].append(BASE.rank_real(
                    real, weights, f"{config_id}_posthoc_C_fixed_fusion", model_seed,
                ))

            for method, frames in ranked_by_method.items():
                consensus = BASE.consensus_real(frames, f"{config_id}_{method}")
                consensus = attach_pair_audit(consensus, deployment)
                consensus.insert(0, "deployment", deployment)
                consensus.insert(0, "change", change)
                consensus.insert(0, "config_id", config_id)
                consensus.insert(0, "stage", stage)
                out = root / f"results/{config_id}/{deployment}"
                out.mkdir(parents=True, exist_ok=True)
                consensus.to_parquet(out / f"real_all_pairs_{method}_with_PE_official.parquet", index=False)
                write_csv(out / f"real_top100_{method}_with_PE_official.csv", consensus.head(100))
                shown = consensus.head(100).copy()
                shown.insert(4, "ranking_method", method)
                top_output.append(shown)
                budget_output.extend(budget_rows(stage, config_id, change, deployment, method, len(frames), consensus))
                valid_bc = consensus[["waveform_score_mean", "chirp_mass_bhattacharyya_coefficient"]].dropna()
                valid_d = consensus[["waveform_score_mean", "chirp_mass_standardized_posterior_distance"]].dropna()
                correlation_rows.append({
                    "stage": stage, "config_id": config_id, "deployment": deployment, "method": method,
                    "n_pairs_with_PE": len(valid_bc),
                    "spearman_waveform_vs_BC_Mc": float(valid_bc.corr(method="spearman").iloc[0, 1]) if len(valid_bc) > 2 else np.nan,
                    "spearman_waveform_vs_negative_D_Mc": float(valid_d.assign(negative_D=-valid_d.iloc[:, 1])[["waveform_score_mean", "negative_D"]].corr(method="spearman").iloc[0, 1]) if len(valid_d) > 2 else np.nan,
                    "interpretation": "descriptive only; pairs share events",
                })
                tracked = consensus.loc[consensus.pair_key.astype(str).isin(known_pairs)]
                for row in tracked.itertuples(index=False):
                    known_pair_rows.append({
                        "stage": stage, "config_id": config_id, "deployment": deployment, "method": method,
                        "pair_key": row.pair_key, "rank": row.consensus_rank,
                        "waveform_score_mean": row.waveform_score_mean,
                        "chirp_mass_BC": row.chirp_mass_bhattacharyya_coefficient,
                        "chirp_mass_D": row.chirp_mass_standardized_posterior_distance,
                        "Dmax": row.max_standardized_posterior_distance,
                        "official_frontend_overlap": row.official_frontend_overlap,
                        "public_hanabi_table_overlap": row.public_hanabi_table_overlap,
                        "official_po_ml_fpp": row.official_po_ml_fpp,
                        "official_po_phazap_fpp": row.official_po_phazap_fpp,
                        "official_tier": row.official_tier,
                        "official_fast_golum_stage": row.official_fast_golum_stage,
                        "official_hanabi_stage": row.official_hanabi_stage,
                    })

    metrics = pd.DataFrame(metric_rows)
    write_csv(root / "tables/RETRIEVAL_METRICS_PER_CONFIG_SEED.csv", metrics)
    summary_rows = []
    keys = ["stage", "config_id", "change", "deployment", "split", "method"]
    metric_columns = ["overall_r_at_1", "overall_r_at_10", "average_precision", "false_at_recall_0p5", "false_at_recall_0p9"]
    for group, part in metrics.groupby(keys, sort=False):
        row = dict(zip(keys, group, strict=True)); row["n_seeds"] = len(part)
        for column in metric_columns:
            row[f"{column}_mean"] = float(part[column].mean())
            row[f"{column}_sd"] = float(part[column].std(ddof=1)) if len(part) > 1 else np.nan
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    write_csv(root / "tables/RETRIEVAL_METRICS_SUMMARY.csv", summary)
    budgets = pd.DataFrame(budget_output)
    write_csv(root / "tables/REAL_PE_OFFICIAL_BUDGET_BY_CONFIG.csv", budgets)
    all_top = pd.concat(top_output, ignore_index=True, sort=False)
    write_csv(root / "tables/REAL_TOP100_ALL_CONFIGS.csv", all_top)
    for budget in (10, 20, 50):
        write_csv(
            root / f"tables/REAL_TOP{budget}_ALL_CONFIGS.csv",
            all_top.loc[pd.to_numeric(all_top.consensus_rank, errors="coerce") <= budget],
        )
    write_csv(root / "tables/REAL_WAVEFORM_PE_CORRELATION_BY_CONFIG.csv", pd.DataFrame(correlation_rows))
    write_csv(root / "tables/KNOWN_FAILURE_PAIR_RANKS_BY_CONFIG.csv", pd.DataFrame(known_pair_rows))
    invariants = pd.DataFrame(invariant_rows)
    write_csv(root / "tables/FROZEN_TIME_SKY_INVARIANCE.csv", invariants)
    if not invariants.max_abs_difference.eq(0.0).all():
        raise RuntimeError("Frozen time/sky invariance audit failed")

    make_figure(root, summary, budgets)
    write_report(root, summary, budgets, pd.DataFrame(known_pair_rows), design)
    write_json(root / "contracts/FINAL_STATUS.json", {
        "status": FINAL_STATUS, "completed_utc": utc_stamp(),
        "source_G7_decision_unchanged": "FAIL_STOP_BEFORE_G8",
        "used_for_model_selection": False, "historical_outputs_overwritten": False, "paper_modified": False,
    })


def make_figure(root: Path, summary: pd.DataFrame, budgets: pd.DataFrame) -> None:
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"], "font.size": 8,
        "axes.labelweight": "bold", "axes.titleweight": "bold", "legend.frameon": False,
    })
    test = summary.loc[summary.split.isin(["locked_test_frozen_reference", "locked_test_posthoc"])]
    pe = budgets.loc[budgets.budget.eq(10) & budgets.method.eq("C_fixed_fusion")]
    configs = list(dict.fromkeys(test.config_id))
    colors = {"gwtc3": "#0072B2", "gwtc4": "#D55E00"}
    labels = {"gwtc3": "O3", "gwtc4": "O4a"}
    x = np.arange(len(configs))
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.0), constrained_layout=True)
    for deployment in DEPLOYMENTS:
        part = test.loc[test.deployment.eq(deployment) & test.method.eq("waveform_only")].set_index("config_id").reindex(configs)
        axes[0, 0].plot(x, part.overall_r_at_10_mean, marker="o", color=colors[deployment], label=labels[deployment])
        fusion = test.loc[test.deployment.eq(deployment) & test.method.eq("C_fixed_fusion")].set_index("config_id").reindex(configs)
        axes[0, 1].plot(x, fusion.overall_r_at_10_mean, marker="o", color=colors[deployment], label=labels[deployment])
        real = pe.loc[pe.deployment.eq(deployment)].set_index("config_id").reindex(configs)
        axes[1, 0].plot(x, real.Dmax_le_3 / real.n_pairs, marker="o", color=colors[deployment], label=labels[deployment])
        axes[1, 1].plot(x, real.official_frontend_overlap, marker="o", color=colors[deployment], label=labels[deployment])
    panels = (
        (axes[0, 0], "Post-hoc waveform-only R@10", "R@10"),
        (axes[0, 1], "Post-hoc C-fixed-fusion R@10", "R@10"),
        (axes[1, 0], "Real Top-10 PE consistency", "Fraction with $D_{max}\\leq3$"),
        (axes[1, 1], "Real Top-10 official-front-end overlap", "Number of pairs"),
    )
    short = [value.replace("D1-", "").replace("_single_seed", "").replace("_three_seed", " (3s)") for value in configs]
    for letter, (ax, title, ylabel) in zip("abcd", panels):
        ax.set_xticks(x, short, rotation=28, ha="right")
        ax.set_title(title, loc="left"); ax.set_ylabel(ylabel); ax.grid(alpha=0.2, linewidth=0.5)
        ax.text(-0.12, 1.04, letter, transform=ax.transAxes, fontsize=10, fontweight="bold")
    axes[0, 0].legend(loc="best")
    fig.suptitle("All-config post-hoc audit; not used for model selection", fontsize=10, fontweight="bold")
    fig.savefig(root / "figures/fig_all_config_posthoc_recall_PE_official.pdf", bbox_inches="tight")
    fig.savefig(root / "figures/fig_all_config_posthoc_recall_PE_official.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_report(
    root: Path,
    summary: pd.DataFrame,
    budgets: pd.DataFrame,
    known: pd.DataFrame,
    design: pd.DataFrame,
) -> None:
    test = summary.loc[summary.split.isin(["locked_test_frozen_reference", "locked_test_posthoc"]), [
        "stage", "config_id", "deployment", "method", "n_seeds", "overall_r_at_1_mean",
        "overall_r_at_10_mean", "average_precision_mean", "false_at_recall_0p5_mean", "false_at_recall_0p9_mean",
    ]]
    top10 = budgets.loc[budgets.budget.eq(10), [
        "stage", "config_id", "deployment", "method", "n_model_seeds", "chirp_mass_BC_ge_0p5",
        "chirp_mass_D_le_3", "Dmax_le_3", "catastrophic_chirp_mass", "official_frontend_overlap",
        "public_hanabi_table_overlap", "official_po_ml_fpp_available", "official_po_phazap_fpp_available",
    ]]
    report = f"""# 全配置真实 GWTC、PE 与官方阶段事后审计

**状态：** `{FINAL_STATUS}`  
**源实验 G7：** `FAIL_STOP_BEFORE_G8`，本审计不改变该结论。

## 1. 审计边界

本补充回答“每个 G2/G3/G4 配置在真实 GWTC 上表现怎样”。所有 checkpoint
在原实验结束后一次性读取；不重新训练、不新增配置、不修改冻结的
`Z_time`、BAYESTAR `Nside=512` `Z_sky` 或 C-fixed 逐 seed 权重。每个变体只用
其对应 validation 拟合 waveform 校准。真实 PE、官方候选和公开 Hanabi 表字段
只在排名完成后附加，未进入分数。

因此这些结果适合解释模型失败模式，但不能用于事后选择某个变体、改判 G7，
也不能证明透镜探测。官方候选重合不是已知真阳性率。

## 2. 到底修改了什么

本研究计划只修改 waveform 侧：输入窗口、hard negatives、chirp-mass
不确定度 head、embedding regularization、encoder 训练和 waveform 校准。
时间与天空保持 BAYESTAR C-fixed 基线不变。`C_fixed_fusion` 表示把每个新
waveform score 接到冻结 C-fixed 的 `Z_time`、`Z_sky` 与逐 seed v9.3 权重上；
不是早期 C，也不是 C-retuned。

## 3. 阶段是累加还是单因素比较

{table_markdown(design, 3)}

G2 是同一基线上的窗口单因素比较。G3 只在 G2 胜出配置上增加 hard
negatives。G4 的 U 与 E 分别加到 G3 胜出配置上；只有两者各自通过才允许
训练 U+E。本次 G2 最终保留 2 s、G3 未保留 hard negatives、G4 的 U/E 均未
晋级，因此 G5--G7 是 2 s base 的三 seed 重复，不是把所有改动叠加起来。

## 4. 每个配置的注入检索

下表中的 G2/G3/G4 变体只有一个训练 seed；`D1-2s_base_three_seed` 才有三个
seed。由于 locked test 已在原 G7 打开，本表明确标为 post-hoc，不能作为新的
确认性模型选择。

{table_markdown(test, 4)}

## 5. 每个配置的真实 Top-10 PE 与官方重合

{table_markdown(top10, 3)}

`BC` 和 `D` 是公开 PE 的描述性物理一致性检查。官方 PO/ML、PO/Phazap、
Tier、Fast-GOLUM 和 Hanabi 字段按冻结本地表合并；不存在的机器可读数值保持
NA，不推断、不补写。

## 6. 已知异常 pair

{table_markdown(known, 4)}

## 7. 文件

- 全部逐 seed 注入指标：`tables/RETRIEVAL_METRICS_PER_CONFIG_SEED.csv`
- 注入汇总：`tables/RETRIEVAL_METRICS_SUMMARY.csv`
- Top-10/20/50/100 PE/官方预算：`tables/REAL_PE_OFFICIAL_BUDGET_BY_CONFIG.csv`
    - 所有配置真实 Top-100：`tables/REAL_TOP100_ALL_CONFIGS.csv`
    - 阶段关系和预注册比较规则：`tables/STAGE_DESIGN_AND_SELECTION_RULES.csv`
    - time/sky 逐 pair 不变性：`tables/FROZEN_TIME_SKY_INVARIANCE.csv`
- 各配置完整真实 pair 排名：`results/<config>/<deployment>/`
- 已知异常 pair：`tables/KNOWN_FAILURE_PAIR_RANKS_BY_CONFIG.csv`
- 分数与 PE 相关：`tables/REAL_WAVEFORM_PE_CORRELATION_BY_CONFIG.csv`
- 不可变合同：`contracts/POSTHOC_ANALYSIS_CONTRACT.json`
"""
    (root / "reports/ALL_CONFIG_POSTHOC_REAL_AUDIT_CN.md").write_text(report, encoding="utf-8")
    (root / "README_CN.md").write_text(
        "# 全配置 post-hoc 审计\n\n"
        "本结果不参与模型选择，不改变原 G7 FAIL。详见 `reports/ALL_CONFIG_POSTHOC_REAL_AUDIT_CN.md`。\n",
        encoding="utf-8",
    )


def package(root: Path, package_dir: Path) -> tuple[Path, str]:
    allowed = ("contracts", "configs", "results", "tables", "figures", "reports", "scripts", "logs")
    files = sorted(
        path for directory in allowed for path in (root / directory).glob("**/*")
        if path.is_file() and path.suffix not in {".npy", ".npz", ".h5", ".hdf5", ".pt"}
    )
    if (root / "README_CN.md").exists():
        files.append(root / "README_CN.md")
    manifest = pd.DataFrame({
        "relative_path": [str(path.relative_to(root)) for path in files],
        "size_bytes": [path.stat().st_size for path in files],
        "sha256": [sha256_file(path) for path in files],
    })
    write_csv(root / "manifest/SHA256SUMS.csv", manifest)
    files.append(root / "manifest/SHA256SUMS.csv")
    package_dir.mkdir(parents=True, exist_ok=True)
    target = package_dir / f"{root.name}_deliverables.tar.gz"
    with tarfile.open(target, "w:gz", compresslevel=6) as archive:
        for path in sorted(set(files)):
            archive.add(path, arcname=str(Path(root.name) / path.relative_to(root)), recursive=False)
    digest = sha256_file(target)
    target.with_suffix(target.suffix + ".sha256").write_text(f"{digest}  {target.name}\n", encoding="ascii")
    return target, digest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--package-dir", type=Path, default=PROJECT / "packages")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise RuntimeError(f"Refusing to overwrite non-empty output: {args.output}")
    run(args.output)
    target, digest = package(args.output, args.package_dir)
    print(json.dumps({"output": str(args.output), "package": str(target), "sha256": digest, "status": FINAL_STATUS}, indent=2))


if __name__ == "__main__":
    main()
