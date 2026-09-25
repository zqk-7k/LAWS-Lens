#!/usr/bin/env python3
"""GWTC sky-resolution v9.3 rerun.

The waveform and time-delay channels, system splits, encoder checkpoints,
calibrations, fusion grid, and validation-only selection rules are inherited
unchanged from unified-sky v8.1.  This script changes only the HEALPix
resolution used to evaluate the common-source sky Bayes factor:

* synthetic O3/O4a real-noise injections stay at their native Nside=64;
* real GWTC PE maps are ranked at common Nside=1024;
* real-pair sky scores are also exported at Nside 32--1024 for convergence
  auditing.

The v8.1-compatible output filenames are deliberate: downstream evaluation
functions are reused without modification.  Every output directory contains a
v9.3 contract that records the changed resolution.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import healpy as hp
import numpy as np
import pandas as pd
import torch


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.real_search.common import read_probability_map
from scripts.real_search.unified_sky_v81 import (
    MAIN_SKY_COLUMN,
    attach_main_sky_score,
    pair_feature_frame,
    write_json,
)
from scripts.real_search.unified_sky_v8 import (
    event_map_diagnostics,
    normalize_probability_maps,
)


V81_SCRIPT = REPO / "scripts/experiments/121_gwtc_sky_v81_pipeline.py"
V7_ROOT = REPO / "results/real_noise_injection_v7_peak2s_formal_20260722"
V81_ROOT = REPO / "results/unified_sky_v81_20260725"
DEFAULT_OUTPUT = REPO / "results/gwtc_sky_resolution_v93_20260730"
INJECTION_NSIDE = 64
REAL_RANKING_NSIDE = 1024
REAL_AUDIT_NSIDES = (32, 64, 128, 256, 512, 1024)
DEPLOYMENTS = {
    "gwtc3": {"source_key": "GWTC3", "seeds": (202607241, 202607242, 202607243)},
    "gwtc4": {"source_key": "GWTC4", "seeds": (202607241, 202607242, 202607243)},
}


def load_module(path: Path, name: str):
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


v81 = load_module(V81_SCRIPT, "gwtc_v81_for_sky_resolution_v93")
v7 = v81.v7
v81.COMMON_NSIDE = INJECTION_NSIDE


def pair_key(event_i: str, event_j: str) -> str:
    return "||".join(sorted((str(event_i), str(event_j))))


def load_real_maps(
    primary: pd.DataFrame,
    target_nside: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    maps: list[np.ndarray] = []
    audits: list[dict[str, Any]] = []
    for index, row in primary.iterrows():
        probability, metadata = read_probability_map(
            row,
            target_nside=target_nside,
        )
        maps.append(probability)
        audits.append(
            {
                "idx": int(index),
                "event_name": str(row["event_name"]),
                "target_nside": int(target_nside),
                **metadata,
            }
        )
    return np.stack(maps).astype(np.float32), pd.DataFrame(audits)


def log_bayes_factor_matrix(maps: np.ndarray) -> np.ndarray:
    """Evaluate log[Npix * sum(P_i P_j)] without HPD diagnostics."""
    probability = normalize_probability_maps(maps)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensor = torch.as_tensor(
        probability,
        dtype=torch.float32,
        device=device,
    )
    raw = tensor @ tensor.T
    log_bf = torch.log(
        torch.clamp(raw * probability.shape[1], min=1e-30)
    )
    output = log_bf.float().cpu().numpy()
    del tensor, raw, log_bf
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return 0.5 * (output + output.T)


def precompute_real_sky(
    deployment: str,
    source_key: str,
    output_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    deployment_root = output_root / deployment
    deployment_root.mkdir(parents=True, exist_ok=True)
    primary = v7.v3.primary_manifest(v7.v3.SOURCES[source_key])
    first_seed = DEPLOYMENTS[deployment]["seeds"][0]
    base_pairs = v81._base_real_pairs(
        V7_ROOT / deployment / f"seed_{first_seed}"
    )
    pair_indices = base_pairs[
        ["idx_i", "idx_j", "event_i", "event_j"]
    ].copy()
    pair_indices["pair_key"] = [
        pair_key(left, right)
        for left, right in zip(
            pair_indices["event_i"],
            pair_indices["event_j"],
        )
    ]
    ii = pair_indices["idx_i"].to_numpy(dtype=np.int32)
    jj = pair_indices["idx_j"].to_numpy(dtype=np.int32)

    convergence = pair_indices.copy()
    formal_sky: pd.DataFrame | None = None
    formal_event_diagnostics: pd.DataFrame | None = None
    formal_source_audit: pd.DataFrame | None = None
    area32: pd.DataFrame | None = None

    for nside in REAL_AUDIT_NSIDES:
        print(f"[real-sky] {deployment}: load Nside={nside}", flush=True)
        maps, source_audit = load_real_maps(primary, nside)
        if nside == REAL_RANKING_NSIDE:
            print(
                f"[real-sky] {deployment}: score Nside={nside} "
                "(formal ranking + diagnostics)",
                flush=True,
            )
            formal_sky, formal_event_diagnostics = pair_feature_frame(
                maps,
                idx_i=ii,
                idx_j=jj,
            )
            formal_event_diagnostics["event_name"] = (
                primary["event_name"].astype(str).to_numpy()
            )
            formal_source_audit = source_audit.copy()
            values = formal_sky[MAIN_SKY_COLUMN].to_numpy(dtype=np.float64)
        else:
            print(
                f"[real-sky] {deployment}: score Nside={nside}",
                flush=True,
            )
            matrix = log_bayes_factor_matrix(maps)
            values = matrix[ii, jj].astype(np.float64)
            del matrix
            if nside == 32:
                diagnostics, levels, mask90 = event_map_diagnostics(
                    maps,
                    event_names=primary["event_name"].astype(str).to_numpy(),
                )
                area32 = diagnostics[["idx", "event_name", "area90_deg2"]].rename(
                    columns={"area90_deg2": "area90_nside32_deg2"}
                )
                del levels, mask90
        convergence[f"sky_log_bf_nside{nside}"] = values
        del maps, source_audit, values
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if (
        formal_sky is None
        or formal_event_diagnostics is None
        or formal_source_audit is None
        or area32 is None
    ):
        raise RuntimeError("Formal real-sky products were not constructed")

    high = convergence[
        [
            "sky_log_bf_nside256",
            "sky_log_bf_nside512",
            "sky_log_bf_nside1024",
        ]
    ].to_numpy(dtype=np.float64)
    convergence["positive_at_all_high_resolutions"] = np.all(high > 0.0, axis=1)
    convergence["negative_at_all_high_resolutions"] = np.all(high < 0.0, axis=1)
    convergence["high_resolution_sign_stable"] = (
        convergence["positive_at_all_high_resolutions"]
        | convergence["negative_at_all_high_resolutions"]
    )
    convergence["sign_flip_nside32_1024"] = (
        np.sign(convergence["sky_log_bf_nside32"])
        != np.sign(convergence["sky_log_bf_nside1024"])
    )
    convergence["abs_delta_nside512_1024"] = np.abs(
        convergence["sky_log_bf_nside512"]
        - convergence["sky_log_bf_nside1024"]
    )
    convergence.to_parquet(
        deployment_root
        / "real_sky_resolution_convergence_all_pairs_v93.parquet",
        index=False,
    )

    event_audit = formal_event_diagnostics.merge(
        area32,
        on=["idx", "event_name"],
        how="left",
        validate="one_to_one",
    ).merge(
        formal_source_audit[
            [
                "idx",
                "event_name",
                "source_format",
                "source_group",
                "source_nside",
                "source_ordering",
                "target_nside",
            ]
        ],
        on=["idx", "event_name"],
        how="left",
        validate="one_to_one",
    )
    event_audit = event_audit.rename(
        columns={"area90_deg2": "area90_nside1024_deg2"}
    )
    event_audit["area90_ratio_nside32_over_1024"] = (
        event_audit["area90_nside32_deg2"]
        / np.maximum(event_audit["area90_nside1024_deg2"], 1e-12)
    )
    event_audit.to_parquet(
        deployment_root / "real_event_sky_resolution_audit_v93.parquet",
        index=False,
    )
    formal_source_audit.to_csv(
        deployment_root / "real_map_source_audit_v93.csv",
        index=False,
    )

    delta = convergence["abs_delta_nside512_1024"].to_numpy(dtype=np.float64)
    write_json(
        deployment_root / "real_sky_resolution_convergence_summary_v93.json",
        {
            "deployment": deployment,
            "n_events": int(len(primary)),
            "n_unordered_pairs": int(len(convergence)),
            "formal_ranking_nside": REAL_RANKING_NSIDE,
            "audit_nsides": list(REAL_AUDIT_NSIDES),
            "sign_flips_nside32_to_1024": int(
                convergence["sign_flip_nside32_1024"].sum()
            ),
            "high_resolution_sign_unstable": int(
                (~convergence["high_resolution_sign_stable"]).sum()
            ),
            "abs_delta_nside512_1024": {
                "median": float(np.median(delta)),
                "q90": float(np.quantile(delta, 0.90)),
                "q99": float(np.quantile(delta, 0.99)),
                "max": float(np.max(delta)),
            },
        },
    )
    return primary, formal_sky, event_audit, convergence


def cached_rank_real_factory(
    formal_sky: pd.DataFrame,
    event_audit: pd.DataFrame,
):
    """Return the v8.1 ranking routine with a cached Nside=1024 sky table."""

    def rank_real(
        seed_dir: Path,
        output_seed: Path,
        methods: dict[str, dict[str, float]],
        primary: pd.DataFrame,
        _unused_maps: np.ndarray,
    ) -> dict[str, dict[str, pd.DataFrame]]:
        base = v81._base_real_pairs(seed_dir)
        real = attach_main_sky_score(base, formal_sky)
        event_audit.to_parquet(
            output_seed / "real_event_sky_diagnostics_v81.parquet",
            index=False,
        )
        real.to_parquet(
            output_seed / "real_pair_features_unified_sky_v81.parquet",
            index=False,
        )
        outputs: dict[str, dict[str, pd.DataFrame]] = {}
        for method, weights in methods.items():
            all_catalog = v7._rank_real_pairs(real, weights, method)
            strict = v7._rank_real_pairs(
                real[real["strict_h1l1_bbh_pair"]].copy(),
                weights,
                method,
            )
            all_catalog.to_parquet(
                output_seed
                / f"real_pair_scores_all_catalog_{method}_v81.parquet",
                index=False,
            )
            strict.to_parquet(
                output_seed
                / f"real_pair_scores_strict_h1l1_bbh_{method}_v81.parquet",
                index=False,
            )
            all_catalog.head(100).to_csv(
                output_seed
                / f"candidate_shortlist_all_catalog_{method}_v81.csv",
                index=False,
            )
            strict.head(100).to_csv(
                output_seed
                / f"candidate_shortlist_strict_h1l1_bbh_{method}_v81.csv",
                index=False,
            )
            outputs[method] = {"all": all_catalog, "strict": strict}
        return outputs

    return rank_real


def verify_frozen_channels(
    deployment: str,
    seeds: tuple[int, ...],
    output_root: Path,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        old = pd.read_csv(
            V81_ROOT
            / deployment
            / f"seed_{seed}"
            / "heldout_test_retrieval_metrics_v81.csv"
        )
        new = pd.read_csv(
            output_root
            / deployment
            / f"seed_{seed}"
            / "heldout_test_retrieval_metrics_v81.csv"
        )
        for method in ("waveform_only", "time_only"):
            for subset in ("overall", "sis", "pm"):
                old_row = old[
                    (old["method"] == method) & (old["subset"] == subset)
                ].iloc[0]
                new_row = new[
                    (new["method"] == method) & (new["subset"] == subset)
                ].iloc[0]
                for metric in ("r_at_1", "r_at_5", "r_at_10", "median_rank"):
                    delta = float(new_row[metric] - old_row[metric])
                    rows.append(
                        {
                            "deployment": deployment,
                            "seed": seed,
                            "method": method,
                            "subset": subset,
                            "metric": metric,
                            "old_value": float(old_row[metric]),
                            "new_value": float(new_row[metric]),
                            "delta": delta,
                            "exact_match": bool(abs(delta) <= 1e-12),
                        }
                    )
    audit = pd.DataFrame(rows)
    audit.to_csv(
        output_root
        / deployment
        / "frozen_waveform_time_reproduction_v93.csv",
        index=False,
    )
    if not audit["exact_match"].all():
        mismatch = audit.loc[~audit["exact_match"]]
        raise RuntimeError(
            f"Frozen waveform/time reproduction failed:\n{mismatch}"
        )
    return audit


def run_deployment(
    deployment: str,
    output_root: Path,
    selected_seed: int | None,
) -> None:
    config = DEPLOYMENTS[deployment]
    primary, formal_sky, event_audit, convergence = precompute_real_sky(
        deployment,
        config["source_key"],
        output_root,
    )
    original_rank_real = v81._rank_real
    v81._rank_real = cached_rank_real_factory(formal_sky, event_audit)
    try:
        seeds = (
            (selected_seed,)
            if selected_seed is not None
            else config["seeds"]
        )
        for seed in seeds:
            print(
                f"[seed] {deployment} seed={seed}: "
                f"injection Nside={INJECTION_NSIDE}, "
                f"real Nside={REAL_RANKING_NSIDE}",
                flush=True,
            )
            output_seed = output_root / deployment / f"seed_{seed}"
            v81.run_seed(
                deployment,
                config["source_key"],
                seed,
                output_root,
                primary,
                np.empty((0, 0), dtype=np.float32),
                event_audit,
            )
            write_json(
                output_seed / "analysis_contract_v93.json",
                {
                    "version": "gwtc_sky_resolution_v93",
                    "only_changed_component": "HEALPix sky resolution",
                    "injection_nside": INJECTION_NSIDE,
                    "real_ranking_nside": REAL_RANKING_NSIDE,
                    "real_resolution_audit_nsides": list(REAL_AUDIT_NSIDES),
                    "sky_formula": (
                        "log B_sky = log[Npix * sum_k(P_i[k] P_j[k])]"
                    ),
                    "waveform_frozen": True,
                    "time_delay_frozen": True,
                    "splits_frozen": True,
                    "weight_grid_and_selection_frozen": True,
                    "selection_split": "synthetic validation systems only",
                    "heldout_test_used_for_selection": False,
                    "real_catalog_used_for_selection": False,
                    "pe_used_for_selection": False,
                },
            )
        if selected_seed is None:
            v81.aggregate(deployment, config["seeds"], output_root)
            verify_frozen_channels(
                deployment,
                config["seeds"],
                output_root,
            )
    finally:
        v81._rank_real = original_rank_real
    del formal_sky, event_audit, convergence
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--deployment",
        choices=["gwtc3", "gwtc4", "all"],
        default="all",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--allow-existing-output",
        action="store_true",
        help="Permit resuming into an existing output directory.",
    )
    args = parser.parse_args()
    if args.output_root.exists() and not args.allow_existing_output:
        raise FileExistsError(
            f"Output already exists: {args.output_root}; "
            "use --allow-existing-output only for an intentional resume"
        )
    args.output_root.mkdir(parents=True, exist_ok=True)
    contract_path = args.output_root / "analysis_contract_v93.json"
    if not (args.allow_existing_output and contract_path.exists()):
        write_json(
            contract_path,
            {
                "version": "gwtc_sky_resolution_v93",
                "created_utc": "2026-07-30",
                "deployments": (
                    list(DEPLOYMENTS)
                    if args.deployment == "all"
                    else [args.deployment]
                ),
                "et3_rerun": False,
                "injection_nside": INJECTION_NSIDE,
                "real_ranking_nside": REAL_RANKING_NSIDE,
                "real_resolution_audit_nsides": list(REAL_AUDIT_NSIDES),
                "source_v7": str(V7_ROOT),
                "source_v81": str(V81_ROOT),
                "frozen_components": [
                    "waveform encoder and calibrated waveform score",
                    "time-delay evidence",
                    "train/validation/test system split",
                    "three training seeds",
                    "fusion grid and validation-only selection rule",
                    "strict H1-L1 BBH catalog definition",
                    "PE audit definition",
                ],
            },
        )
    deployments = (
        list(DEPLOYMENTS)
        if args.deployment == "all"
        else [args.deployment]
    )
    for deployment in deployments:
        run_deployment(deployment, args.output_root, args.seed)
    print("[complete] sky-resolution v9.3 main rerun finished", flush=True)


if __name__ == "__main__":
    main()
