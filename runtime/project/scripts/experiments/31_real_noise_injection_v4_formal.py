#!/usr/bin/env python3
"""Formal multi-seed physical real-noise deployment.

This wrapper freezes the dual-scale architecture selected on the separate
202607211 development validation split.  It reuses the v3 physical data,
time-delay, sky-Bayes-factor, metric, and catalog-scoring implementation while
replacing the failed InceptionTime waveform trainer with the predeclared
full-24-s plus one-second trigger-aligned spectrogram encoder.

The held-out test set is evaluated only after checkpoint selection.  A failed
waveform gate forces the primary real-catalog result to time+sky; the positive
three-channel result remains an explicitly labelled sensitivity analysis.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from matchgw.data import peak_flip_channels, zscore_channels
from scripts.real_search.physical_common import effective_rank, write_json


DEFAULT_OUT = REPO / "results" / "real_noise_injection_v4_physical_20260721"
DEFAULT_SEEDS = (202607221, 202607222, 202607223)
ARCHITECTURE = "dual"
ADAPTATION = "teacher"
PRETRAIN_EPOCHS = 12
VARIANTS_PER_SOURCE = 8


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


v3 = load_module("physical_v3", REPO / "scripts" / "experiments" / "20_real_noise_injection_v3_physical.py")
specmod = load_module("physical_spectrogram", REPO / "scripts" / "experiments" / "26_spectrogram_encoder_pilot.py")


def selected_dir(seed_dir: Path, family: str) -> Path:
    return seed_dir / "waveform_gate" / f"{family.lower()}_{ARCHITECTURE}_{ADAPTATION}_multinoise_curriculum_pilot"


def expected_dir(seed_dir: Path, family: str, epochs: int) -> Path:
    return seed_dir / "waveform_gate" / f"{family.lower()}_physical_full24_ep{epochs}"


def run_checked(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        subprocess.run(command, cwd=REPO, stdout=handle, stderr=subprocess.STDOUT, check=True)


def train_selected_family(
    seed_dir: Path,
    family: str,
    seed: int,
    samples: int,
    epochs: int,
    batch_size: int,
) -> dict[str, Any]:
    target = expected_dir(seed_dir, family, epochs)
    summary_path = target / "summary.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))

    multinoise_marker = (
        seed_dir
        / "data"
        / "real_noise_injections"
        / f"multinoise_{family.lower()}_train_v{VARIANTS_PER_SOURCE}"
        / "multinoise_summary.json"
    )
    if not multinoise_marker.exists():
        run_checked(
            [
                sys.executable,
                str(REPO / "scripts" / "experiments" / "29_materialize_multinoise_training.py"),
                "--seed-root",
                str(seed_dir),
                "--family",
                family,
                "--seed",
                str(seed),
                "--samples",
                str(samples),
                "--variants-per-source",
                str(VARIANTS_PER_SOURCE),
            ],
            seed_dir / "logs" / f"materialize_multinoise_{family.lower()}.log",
        )

    selected_summary = selected_dir(seed_dir, family) / "multinoise_curriculum_summary.json"
    if not selected_summary.exists():
        run_checked(
            [
                sys.executable,
                str(REPO / "scripts" / "experiments" / "30_multinoise_curriculum_pilot.py"),
                "--seed-root",
                str(seed_dir),
                "--family",
                family,
                "--seed",
                str(seed),
                "--samples",
                str(samples),
                "--variants-per-source",
                str(VARIANTS_PER_SOURCE),
                "--pretrain-epochs",
                str(PRETRAIN_EPOCHS),
                "--adapt-epochs",
                str(epochs),
                "--batch-size",
                str(batch_size),
                "--architecture",
                ARCHITECTURE,
                "--adaptation",
                ADAPTATION,
            ],
            seed_dir / "logs" / f"train_{family.lower()}_{ARCHITECTURE}_{ADAPTATION}.log",
        )

    selected = json.loads(selected_summary.read_text(encoding="utf-8"))
    source_checkpoint = Path(selected["checkpoint"])
    if not source_checkpoint.is_absolute():
        source_checkpoint = REPO / source_checkpoint
    target.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_checkpoint, target / "model.pt")
    payload = {
        **selected,
        "formal_architecture_frozen_before_formal_seeds": True,
        "architecture_selection_development_seed": 202607211,
        "heldout_test_evaluated_during_training": False,
        "formal_checkpoint": str(target / "model.pt"),
    }
    write_json(summary_path, payload)
    return payload


def load_selected_model(path: Path, data_root: Path, samples: int, family: str):
    del data_root, samples, family
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = specmod.TriggerAlignedDualScaleSpectrogramEncoder(base_channels=24)
    model.load_state_dict(payload["model_state"])
    model = model.cuda().eval()
    cfg = type(
        "EmbeddingConfig",
        (),
        {
            "eval_batch_size": 16,
            "amp": True,
            "amp_dtype": "bf16",
            "num_workers": 0,
            "pin_memory": True,
        },
    )()
    return model, cfg


@torch.no_grad()
def embed_values(model: torch.nn.Module, values: np.ndarray, batch_size: int = 16) -> np.ndarray:
    rows: list[np.ndarray] = []
    model.eval()
    for start in range(0, len(values), batch_size):
        prepared = [zscore_channels(peak_flip_channels(np.asarray(x, dtype=np.float32))) for x in values[start : start + batch_size]]
        tensor = torch.from_numpy(np.stack(prepared)).cuda(non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            rows.append(model(tensor).float().cpu().numpy())
    return np.concatenate(rows).astype(np.float32)


def bootstrap_effective_rank_floor(values: np.ndarray, sample_size: int, seed: int, draws: int = 300) -> dict[str, float]:
    x = np.asarray(values, dtype=np.float32)
    if len(x) < sample_size:
        sample_size = len(x)
    rng = np.random.default_rng(seed)
    ranks = np.asarray(
        [effective_rank(x[rng.choice(len(x), size=sample_size, replace=False)]) for _ in range(draws)],
        dtype=np.float64,
    )
    return {
        "validation_reference_draws": int(draws),
        "validation_reference_sample_size": int(sample_size),
        "validation_effective_rank_q01": float(np.quantile(ranks, 0.01)),
        "validation_effective_rank_median": float(np.median(ranks)),
        "validation_effective_rank_q99": float(np.quantile(ranks, 0.99)),
    }


def embed_real_events_selected(
    deployment: str,
    seed_dir: Path,
    shared_dir: Path,
    epochs: int,
    samples: int,
    calibrations: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    emb_path = seed_dir / "features" / "real_waveform_embeddings_physical.parquet"
    pair_path = seed_dir / "features" / "real_waveform_similarity_physical.parquet"
    if emb_path.exists() and pair_path.exists():
        return pd.read_parquet(emb_path), pd.read_parquet(pair_path)

    inputs, audit = v3.build_real_preprocessed_inputs(deployment, v3.SOURCES[deployment], shared_dir)
    available = audit["strict_h1l1_preprocessing_pass"].to_numpy(dtype=bool)
    root = seed_dir / "data" / "real_noise_injections" / "matchroots" / "LIGO"
    by_family: dict[str, np.ndarray] = {}
    reference: dict[str, Any] = {}
    for family in v3.FAMILIES:
        checkpoint = expected_dir(seed_dir, family, epochs) / "model.pt"
        model, _ = load_selected_model(checkpoint, root, samples, family)
        current = embed_values(model, np.asarray(inputs[available]), batch_size=12)
        full = np.full((len(audit), current.shape[1]), np.nan, dtype=np.float32)
        full[available] = current
        by_family[family] = full

        validation_embeddings = []
        for target_family in v3.FAMILIES:
            path = seed_dir / "results" / f"{target_family.lower()}_val_{family.lower()}_embeddings.npy"
            if path.exists():
                validation_embeddings.append(np.load(path))
        if validation_embeddings:
            validation_values = np.concatenate(validation_embeddings)
            reference[family] = bootstrap_effective_rank_floor(
                validation_values,
                int(available.sum()),
                seed=27100 + (0 if family == "SIS" else 1000),
            )
        del model
        torch.cuda.empty_cache()

    emb = audit.copy()
    for family, values in by_family.items():
        emb[f"{family.lower()}_embedding"] = [row.tolist() if np.isfinite(row).all() else None for row in values]
    emb.to_parquet(emb_path, index=False)

    ii, jj = np.triu_indices(len(emb), k=1)
    both = available[ii] & available[jj]
    raw: dict[str, np.ndarray] = {}
    evidence: dict[str, np.ndarray] = {}
    for family in v3.FAMILIES:
        values = np.full(len(ii), np.nan, dtype=np.float32)
        values[both] = np.sum(by_family[family][ii[both]] * by_family[family][jj[both]], axis=1)
        raw[family] = values
        calibrated = np.zeros(len(ii), dtype=np.float32)
        calibrated[both] = v3.apply_score_likelihood_ratio(values[both], calibrations[family])
        evidence[family] = calibrated
    mixture = np.zeros(len(ii), dtype=np.float32)
    mixture[both] = (np.logaddexp(evidence["SIS"][both], evidence["PM"][both]) - np.log(2.0)).astype(np.float32)
    pairs = pd.DataFrame(
        {
            "idx_i": ii.astype(np.int32),
            "idx_j": jj.astype(np.int32),
            "event_i": emb["event_name"].to_numpy()[ii],
            "event_j": emb["event_name"].to_numpy()[jj],
            "waveform_available": both,
            "waveform_raw_sis": raw["SIS"],
            "waveform_raw_pm": raw["PM"],
            "waveform_log_bf_sis": evidence["SIS"],
            "waveform_log_bf_pm": evidence["PM"],
            "waveform_score": mixture,
        }
    )
    pairs.to_parquet(pair_path, index=False)

    family_audits: dict[str, Any] = {}
    reference_passes = []
    for family in v3.FAMILIES:
        rank = effective_rank(by_family[family][available])
        floor = reference.get(family, {}).get("validation_effective_rank_q01", 0.0)
        passed = bool(rank >= floor)
        reference_passes.append(passed)
        family_audits[family] = {
            "real_effective_rank": rank,
            **reference.get(family, {}),
            "passes_validation_q01_floor": passed,
        }
    valid_scores = mixture[both]
    deployment_audit = {
        "deployment": deployment,
        "strict_h1l1_events": int(available.sum()),
        "strict_h1l1_pairs": int(both.sum()),
        "family_embedding_audits": family_audits,
        "reference_calibrated_effective_rank_pass": bool(all(reference_passes)),
        "embedding_effective_rank_combined": effective_rank(np.concatenate([by_family[f][available] for f in v3.FAMILIES])),
        "embedding_effective_rank_sis": effective_rank(by_family["SIS"][available]),
        "embedding_effective_rank_pm": effective_rank(by_family["PM"][available]),
        "waveform_score_min": float(np.min(valid_scores)) if len(valid_scores) else None,
        "waveform_score_max": float(np.max(valid_scores)) if len(valid_scores) else None,
        "waveform_score_std": float(np.std(valid_scores)) if len(valid_scores) else None,
        "waveform_score_unique_rounded_1e6": int(len(np.unique(np.round(valid_scores, 6)))) if len(valid_scores) else 0,
        "constant_score_failure": bool(len(valid_scores) == 0 or np.std(valid_scores) < 1e-6),
        "silent_zero_fill": False,
        "audit_rule": "Each real family embedding effective rank must exceed the 1st percentile of equally sized resamples from the frozen synthetic validation embeddings; score must be nonconstant.",
    }
    write_json(seed_dir / "results" / "real_waveform_deployment_audit_physical.json", deployment_audit)
    return emb, pairs


def choose_primary_and_export(seed_dir: Path) -> dict[str, Any]:
    summary_path = seed_dir / "seed_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    audit = summary["real_deployment_audit"]
    gate_pass = bool(summary["waveform_gate1"]["passed"])
    audit_pass = bool(
        audit.get("reference_calibrated_effective_rank_pass", False)
        and not audit.get("constant_score_failure", True)
    )
    free = summary["weights"]["unconstrained"]
    waveform_eligible = gate_pass and audit_pass
    if waveform_eligible and float(free["waveform"]) > 0:
        primary = "three_channel_unconstrained"
        reason = "Waveform passed held-out Gate-1 and the real-deployment embedding audit, and validation selected a positive waveform weight."
    else:
        primary = "time_sky_validation_selected"
        reason = "Waveform failed Gate-1/deployment audit or validation assigned zero waveform weight; time+sky is the predeclared fallback."

    results = seed_dir / "results"
    primary_frames: dict[str, pd.DataFrame] = {}
    for scope in ("strict_h1l1_bbh", "all_catalog"):
        source = results / f"real_pair_scores_{scope}_{primary}.parquet"
        target = results / f"real_pair_scores_{scope}_primary.parquet"
        shutil.copy2(source, target)
        frame = pd.read_parquet(source)
        primary_frames[scope] = frame
        frame.head(100).to_csv(results / f"candidate_shortlist_{scope}_primary.csv", index=False)
    summary.update(
        {
            "deployment_passed": waveform_eligible,
            "primary_method_policy": primary,
            "primary_method_reason": reason,
            "waveform_gate_passed": gate_pass,
            "real_embedding_audit_passed": audit_pass,
            "positive_three_channel_is_supplementary_when_primary_falls_back": primary == "time_sky_validation_selected",
            "strict_primary_top_pair": primary_frames["strict_h1l1_bbh"].iloc[0].to_dict() if len(primary_frames["strict_h1l1_bbh"]) else {},
        }
    )
    if summary.get("deployment") == "GWTC3":
        summary["gw170104_gw170814"] = v3._historical_pair_rank(
            primary_frames["all_catalog"], "GW170104", "GW170814"
        )
    write_json(summary_path, summary)
    return summary


def aggregate_primary(out_root: Path, deployment: str, seeds: list[int]) -> None:
    root = out_root / deployment.lower()
    frames = []
    summaries = []
    for seed in seeds:
        seed_dir = root / f"seed_{seed}"
        summaries.append(json.loads((seed_dir / "seed_summary.json").read_text(encoding="utf-8")))
        for scope in ("strict_h1l1_bbh", "all_catalog"):
            frame = pd.read_parquet(seed_dir / "results" / f"real_pair_scores_{scope}_primary.parquet")
            keep = frame[["event_i", "event_j", "rank", "final_score", "waveform_available", "strict_h1l1_bbh_pair", "pair_has_ood"]].copy()
            keep["seed"] = seed
            keep["scope"] = scope
            keep["pair_key"] = keep.apply(lambda row: "--".join(sorted((str(row.event_i), str(row.event_j)))), axis=1)
            frames.append(keep)
    all_rows = pd.concat(frames, ignore_index=True)
    all_rows.to_parquet(root / "real_candidate_primary_ranks_all_seeds.parquet", index=False)
    stability = all_rows.groupby(["scope", "pair_key"], as_index=False).agg(
        event_i=("event_i", "first"),
        event_j=("event_j", "first"),
        median_rank=("rank", "median"),
        q25_rank=("rank", lambda x: float(np.quantile(x, 0.25))),
        q75_rank=("rank", lambda x: float(np.quantile(x, 0.75))),
        min_rank=("rank", "min"),
        max_rank=("rank", "max"),
        top10_frequency=("rank", lambda x: float(np.mean(np.asarray(x) <= 10))),
        waveform_available=("waveform_available", "all"),
        pair_has_ood=("pair_has_ood", "any"),
    ).sort_values(["scope", "median_rank", "q75_rank", "min_rank"])
    stability.to_csv(root / "real_candidate_primary_rank_stability.csv", index=False)
    write_json(
        root / "formal_primary_summary.json",
        {
            "deployment": deployment,
            "seeds": seeds,
            "primary_methods": [x["primary_method_policy"] for x in summaries],
            "all_waveform_gates_passed": bool(all(x["waveform_gate_passed"] for x in summaries)),
            "all_real_embedding_audits_passed": bool(all(x["real_embedding_audit_passed"] for x in summaries)),
            "interpretation": "Primary is waveform+time+sky only when both frozen gates pass; otherwise primary is time+sky and positive three-channel remains supplementary.",
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--deployments", nargs="+", choices=("GWTC3", "GWTC4"), default=["GWTC3", "GWTC4"])
    parser.add_argument("--samples-per-family", type=int, default=600)
    parser.add_argument("--adapt-epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for formal waveform training")

    args.out_root.mkdir(parents=True, exist_ok=True)
    write_json(
        args.out_root / "run_config.json",
        {
            "protocol": "real_noise_injection_v4_physical_dual_scale_formal",
            "formal_seeds": args.seeds,
            "development_seed_excluded": 202607211,
            "deployments": args.deployments,
            "samples_per_family": args.samples_per_family,
            "training_variants_per_source": VARIANTS_PER_SOURCE,
            "pretrain_epochs": PRETRAIN_EPOCHS,
            "adapt_epochs": args.adapt_epochs,
            "architecture": "full 24-s log-STFT GroupNorm CNN plus fixed 1-s trigger-aligned log-STFT branch",
            "channels": ["calibrated waveform log evidence", "run-conditioned time-delay log likelihood ratio", "HEALPix common-source sky log Bayes factor"],
            "snr_amplitude_in_final_score": False,
            "real_catalog_used_for_tuning": False,
            "primary_fallback": "time+sky when waveform Gate-1 or real embedding audit fails",
            "strict_primary_catalog": "complete H1-L1, non-OOD BBH pairs",
        },
    )

    v3.train_family = train_selected_family
    v3.load_checkpoint_model = load_selected_model
    v3.embed_real_events_v3 = embed_real_events_selected

    for deployment in args.deployments:
        shared = args.out_root / deployment.lower() / "shared"
        schedule = v3.build_live_schedule(deployment, v3.SOURCES[deployment], shared)
        v3.build_noise_bank(v3.SOURCES[deployment], shared, 20260721 + (0 if deployment == "GWTC3" else 1000))
        v3.build_sky_template_library(v3.SOURCES[deployment], shared)
        raw_delays, _, _ = v3.liao_delay_ratio_samples()
        v3.build_time_calibration(shared, schedule, raw_delays, 20260722 + (0 if deployment == "GWTC3" else 1000))
        complete_seeds: list[int] = []
        for seed in args.seeds:
            seed_dir = args.out_root / deployment.lower() / f"seed_{seed}"
            v3.prepare_seed_layout(seed_dir, v3.SOURCES[deployment])
            v3.run_seed(
                deployment,
                seed,
                args.out_root,
                args.samples_per_family,
                args.adapt_epochs,
                args.batch_size,
                args.bootstrap_draws,
            )
            choose_primary_and_export(seed_dir)
            complete_seeds.append(seed)
            v3.aggregate_deployment(args.out_root, deployment, complete_seeds)
            aggregate_primary(args.out_root, deployment, complete_seeds)
    print(json.dumps({"status": "complete", "out_root": str(args.out_root)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
