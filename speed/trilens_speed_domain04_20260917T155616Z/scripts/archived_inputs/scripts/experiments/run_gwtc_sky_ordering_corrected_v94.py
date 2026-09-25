#!/usr/bin/env python3
"""Independent ordering-corrected rerun of the GWTC sky-resolution v9.3 flow.

This script never writes into the historical v9.3, v8.1, or v7 directories.
It freezes waveform/time features and all system splits, rebuilds only the sky
products after correctly decoding PE HDF5 NESTED metadata, then reruns the
validation-only fusion selection, held-out evaluation, and real-catalog rank.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import healpy as hp
import numpy as np
import pandas as pd
import torch


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.real_search.common import (
    choose_h5_skymap_group,
    resize_probability_map,
    sanitize_probability_map,
)
from scripts.real_search.physical_common import posterior_area90


V93_SCRIPT = REPO / "scripts/experiments/run_gwtc_sky_resolution_v93.py"
DEFAULT_OUTPUT = REPO / "results/gwtc_sky_ordering_corrected_v94_20260830"
VERSION = "gwtc_sky_ordering_corrected_v94"


def load_module(path: Path, name: str):
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


v93 = load_module(V93_SCRIPT, "v93_for_ordering_corrected_v94")
v81 = v93.v81
v7 = v93.v7
v3 = v7.v3


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def parse_hdf5_bool_strict(raw: Any) -> bool:
    """Decode scalar HDF5 booleans including arrays such as [b'True']."""
    values = np.asarray(raw)
    if values.size != 1:
        raise ValueError(f"Expected scalar bool metadata, got shape={values.shape}")
    value = values.reshape(-1)[0]
    if isinstance(value, (bytes, np.bytes_)):
        text = value.decode(errors="strict")
    else:
        text = str(value)
    normalized = text.strip().lower()
    if normalized in {"1", "true", "t", "yes", "y"}:
        return True
    if normalized in {"0", "false", "f", "no", "n"}:
        return False
    raise ValueError(f"Unrecognized HDF5 bool metadata: {raw!r}")


def corrected_read_probability_map(
    row: pd.Series,
    target_nside: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    path = Path(str(row["sky_map_path"]))
    if not path.is_absolute():
        path = REPO / path
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() not in {".h5", ".hdf5"}:
        raise ValueError(f"v9.3 primary map is not PE HDF5: {path}")

    group = choose_h5_skymap_group(path)
    if group is None:
        raise FileNotFoundError(f"No /skymap/data group in {path}")
    manifest_group = str(row.get("sky_map_group", "")).strip()
    if manifest_group and manifest_group.lower() != "nan" and manifest_group != group:
        raise ValueError(
            f"Manifest/loader PE group mismatch for {row['event_name']}: "
            f"{manifest_group!r} != {group!r}"
        )

    with h5py.File(path, "r") as handle:
        probability = np.asarray(
            handle[f"{group}/skymap/data"][:], dtype=np.float64
        ).reshape(-1)
        nest_key = f"{group}/skymap/meta_data/nest"
        if nest_key not in handle:
            raise KeyError(f"Missing required ordering metadata: {nest_key}")
        raw_nest = handle[nest_key][()]
        nested = parse_hdf5_bool_strict(raw_nest)
        origin_key = f"{group}/skymap/meta_data/origin"
        origin = (
            np.asarray(handle[origin_key][()]).reshape(-1)[0]
            if origin_key in handle
            else ""
        )
        if isinstance(origin, (bytes, np.bytes_)):
            origin = origin.decode(errors="ignore")

    probability = sanitize_probability_map(probability)
    source_nside = hp.npix2nside(len(probability))
    if nested:
        probability = hp.reorder(probability, n2r=True)
    probability = resize_probability_map(
        probability,
        source_nside,
        int(target_nside),
    )
    metadata = {
        "source_format": "pe_hdf5_skymap",
        "source_group": group,
        "source_nside": int(source_nside),
        "source_ordering": "NESTED" if nested else "RING",
        "output_ordering": "RING",
        "coordinate_frame": "celestial_equatorial_ra_dec",
        "source_origin": str(origin),
        "target_nside": int(target_nside),
        "ordering_metadata_raw": repr(raw_nest),
        "ordering_conversion_applied": bool(nested),
    }
    return probability.astype(np.float32), metadata


def fixed_load_real_maps(
    primary: pd.DataFrame,
    target_nside: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    maps: list[np.ndarray] = []
    audits: list[dict[str, Any]] = []
    for index, row in primary.iterrows():
        probability, metadata = corrected_read_probability_map(
            row,
            target_nside=target_nside,
        )
        maps.append(probability)
        audits.append(
            {
                "idx": int(index),
                "event_name": str(row["event_name"]),
                **metadata,
            }
        )
    stacked = np.stack(maps).astype(np.float32)
    normalization_error = np.max(
        np.abs(stacked.sum(axis=1, dtype=np.float64) - 1.0)
    )
    if not np.isfinite(stacked).all() or np.any(stacked < 0.0):
        raise RuntimeError("Corrected real map bank contains invalid values")
    if normalization_error > 2e-6:
        raise RuntimeError(
            f"Corrected real map normalization error={normalization_error}"
        )
    return stacked, pd.DataFrame(audits)


def build_corrected_template_library(
    deployment: str,
    source_key: str,
    output_root: Path,
) -> tuple[np.ndarray, pd.DataFrame]:
    root = output_root / "corrected_sky_inputs" / deployment / "shared"
    root.mkdir(parents=True, exist_ok=True)
    map_path = root / "real_pe_sky_templates_nside64_ordering_corrected_v94.npy"
    manifest_path = root / "real_pe_sky_template_manifest_ordering_corrected_v94.csv"
    primary = v3.primary_manifest(v3.SOURCES[source_key])
    maps: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    for index, row in primary.iterrows():
        probability, metadata = corrected_read_probability_map(row, target_nside=64)
        maps.append(probability)
        rows.append(
            {
                "event_name": str(row["event_name"]),
                "usable": True,
                "template_index": int(index),
                "network_snr": float(row.get("network_snr", np.nan)),
                "area90_deg2": float(posterior_area90(probability)),
                **metadata,
            }
        )
    stacked = np.stack(maps).astype(np.float32)
    frame = pd.DataFrame(rows)
    np.save(map_path, stacked)
    frame.to_csv(manifest_path, index=False)
    write_json(
        root / "real_pe_sky_template_summary_ordering_corrected_v94.json",
        {
            "deployment": deployment,
            "n_templates": int(len(stacked)),
            "nside": 64,
            "npix": int(stacked.shape[1]),
            "all_source_ordering": sorted(frame["source_ordering"].unique().tolist()),
            "all_output_ordering": sorted(frame["output_ordering"].unique().tolist()),
            "all_ordering_conversions_applied": bool(
                frame["ordering_conversion_applied"].all()
            ),
            "area90_median_deg2": float(frame["area90_deg2"].median()),
            "area90_p90_deg2": float(frame["area90_deg2"].quantile(0.9)),
            "map_sha256": sha256_file(map_path),
            "manifest_sha256": sha256_file(manifest_path),
        },
    )
    return stacked, frame


def rebuild_injection_maps(
    deployment: str,
    source_key: str,
    seeds: tuple[int, ...],
    output_root: Path,
) -> pd.DataFrame:
    templates, template_frame = build_corrected_template_library(
        deployment,
        source_key,
        output_root,
    )
    template_frame = template_frame.sort_values("template_index")
    template_snr = template_frame["network_snr"].to_numpy(dtype=np.float64)
    template_names = template_frame["event_name"].astype(str).to_numpy()
    template_area = template_frame["area90_deg2"].to_numpy(dtype=np.float64)
    rows: list[dict[str, Any]] = []

    for seed in seeds:
        source_results = v93.V7_ROOT / deployment / f"seed_{seed}" / "results"
        output_results = (
            output_root
            / "corrected_sky_inputs"
            / deployment
            / f"seed_{seed}"
            / "results"
        )
        output_results.mkdir(parents=True, exist_ok=True)
        for split, offset in (("val", 2000), ("test", 3000)):
            source_event_path = source_results / f"mixed_{split}_synthetic_events_v7.parquet"
            source_events = pd.read_parquet(source_event_path).reset_index(drop=True)
            rng = np.random.default_rng(seed + offset)
            maps: list[np.ndarray] = []
            corrected_events: list[dict[str, Any]] = []
            selected_indices: list[int] = []
            replay_proposed_indices: list[int] = []
            for event_index, event in source_events.iterrows():
                replay_proposed_index = v3._choose_sky_template(
                    float(event["snr"]),
                    template_snr,
                    rng,
                )
                # Template identity is part of the frozen event-level sky
                # realization.  Correcting the map probabilities can alter
                # the RNG consumption inside weighted pixel sampling, so a
                # global RNG replay alone may drift on later events.
                template_index = int(event["sky_template_index"])
                if template_index < 0 or template_index >= len(templates):
                    raise IndexError(
                        f"Frozen template index {template_index} is invalid"
                    )
                probability, anchor = v3.rotate_probability_map_to_true_position(
                    np.asarray(templates[template_index]),
                    float(event["ra_true"]),
                    float(event["dec_true"]),
                    rng,
                )
                maps.append(probability.astype(np.float32))
                selected_indices.append(template_index)
                replay_proposed_indices.append(replay_proposed_index)
                payload = event.to_dict()
                payload.update(
                    {
                        "idx": int(event_index),
                        "sky_template_index": int(template_index),
                        "sky_template_event": str(template_names[template_index]),
                        "sky_template_area90_deg2": float(template_area[template_index]),
                        "sky_anchor_pixel": int(anchor),
                        "sky_ordering": "RING",
                        "sky_nside": 64,
                        "source_template_ordering_corrected": True,
                    }
                )
                corrected_events.append(payload)

            stacked = np.stack(maps).astype(np.float32)
            map_path = (
                output_results
                / f"mixed_{split}_synthetic_sky_posteriors_nside64_ordering_corrected_v94.npy"
            )
            event_path = (
                output_results
                / f"mixed_{split}_synthetic_events_ordering_corrected_v94.parquet"
            )
            np.save(map_path, stacked)
            pd.DataFrame(corrected_events).to_parquet(event_path, index=False)
            old_indices = source_events["sky_template_index"].to_numpy(dtype=np.int64)
            new_indices = np.asarray(selected_indices, dtype=np.int64)
            replay_indices = np.asarray(replay_proposed_indices, dtype=np.int64)
            normalization_error = float(
                np.max(np.abs(stacked.sum(axis=1, dtype=np.float64) - 1.0))
            )
            if not np.isfinite(stacked).all() or np.any(stacked < 0.0):
                raise RuntimeError(f"Invalid corrected injection maps: {map_path}")
            if normalization_error > 2e-6:
                raise RuntimeError(
                    f"Corrected injection normalization error={normalization_error}"
                )
            rows.append(
                {
                    "deployment": deployment,
                    "seed": int(seed),
                    "split": split,
                    "n_events": int(len(stacked)),
                    "nside": 64,
                    "npix": int(stacked.shape[1]),
                    "source_event_table": str(source_event_path),
                    "corrected_map_path": str(map_path),
                    "corrected_event_path": str(event_path),
                    "observable_seed": int(seed + offset),
                    "used_template_selection_match_fraction": float(
                        np.mean(old_indices == new_indices)
                    ),
                    "unfrozen_rng_replay_match_fraction_diagnostic": float(
                        np.mean(old_indices == replay_indices)
                    ),
                    "max_abs_normalization_error": normalization_error,
                    "map_sha256": sha256_file(map_path),
                    "event_sha256": sha256_file(event_path),
                }
            )
            del stacked, maps
            gc.collect()
    audit = pd.DataFrame(rows)
    audit.to_csv(
        output_root
        / "corrected_sky_inputs"
        / deployment
        / "injection_map_rebuild_audit_v94.csv",
        index=False,
    )
    return audit


def corrected_pair_loader_factory(output_root: Path):
    def load_pair_sky(
        seed_dir: Path,
        split: str,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        deployment = seed_dir.parent.name
        seed_name = seed_dir.name
        map_path = (
            output_root
            / "corrected_sky_inputs"
            / deployment
            / seed_name
            / "results"
            / f"mixed_{split}_synthetic_sky_posteriors_nside64_ordering_corrected_v94.npy"
        )
        maps = np.load(map_path, mmap_mode="r")
        base_name = (
            "fusion_validation_pairs_v7.parquet"
            if split == "val"
            else "fusion_heldout_test_pairs_v7.parquet"
        )
        base = pd.read_parquet(
            seed_dir / "results" / base_name,
            columns=["idx_i", "idx_j"],
        )
        sky, diagnostics = v93.pair_feature_frame(
            maps,
            idx_i=base["idx_i"].to_numpy(dtype=np.int32),
            idx_j=base["idx_j"].to_numpy(dtype=np.int32),
        )
        return sky, diagnostics

    return load_pair_sky


def root_contract(output_root: Path) -> dict[str, Any]:
    historical_package = REPO / "packages/gwtc_sky_resolution_v93_20260730.tar.gz"
    return {
        "version": VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "historical_v93_read_only": True,
        "historical_v93_result_root": str(v93.DEFAULT_OUTPUT),
        "historical_v93_package": str(historical_package),
        "historical_v93_package_sha256_before_rerun": (
            sha256_file(historical_package) if historical_package.is_file() else None
        ),
        "et3_rerun": False,
        "injection_nside": 64,
        "real_ranking_nside": 1024,
        "real_resolution_audit_nsides": [32, 64, 128, 256, 512, 1024],
        "changed_components": [
            "strict decoding of PE HDF5 skymap/meta_data/nest",
            "NESTED-to-RING conversion before probability-mass resizing",
            "regenerated run-matched injection sky template libraries",
            "regenerated validation/test injection event sky maps",
            "recomputed sky evidence, validation-selected weights, held-out metrics, and real ranks",
        ],
        "frozen_components": [
            "waveform encoder and waveform pair evidence",
            "time-delay evidence",
            "train/validation/test source-system splits",
            "three model seeds",
            "observable sky random seeds",
            "fusion grid and validation-only selection rule",
            "strict H1-L1 BBH catalog definition",
            "PE audit definition",
            "sky Bayes-factor formula",
            "injection Nside=64",
            "real ranking Nside=1024",
        ],
        "selection_contract": {
            "weights_selected_on": "synthetic validation systems only",
            "heldout_test_used_for_selection": False,
            "real_catalog_used_for_selection": False,
            "pe_used_for_selection": False,
        },
        "no_overwrite": True,
    }


def rewrite_seed_contracts(output_root: Path) -> None:
    for deployment, config in v93.DEPLOYMENTS.items():
        for seed in config["seeds"]:
            seed_root = output_root / deployment / f"seed_{seed}"
            write_json(
                seed_root / "analysis_contract_ordering_corrected_v94.json",
                {
                    "version": VERSION,
                    "deployment": deployment,
                    "seed": int(seed),
                    "compatibility_output_filenames": "v81/v93 names retained for downstream readers",
                    "ordering_fix": "strict [b'True'] decoding and hp.reorder(n2r=True)",
                    "source_map_ordering_required": "NESTED",
                    "scoring_map_ordering": "RING",
                    "injection_nside": 64,
                    "real_ranking_nside": 1024,
                    "waveform_frozen": True,
                    "time_delay_frozen": True,
                    "splits_frozen": True,
                    "weight_grid_and_selection_rule_frozen": True,
                    "weights_reselected_on_corrected_validation_sky": True,
                    "heldout_test_used_for_selection": False,
                    "real_catalog_used_for_selection": False,
                    "pe_used_for_selection": False,
                },
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--deployment",
        choices=["gwtc3", "gwtc4", "all"],
        default="all",
    )
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--allow-existing-output", action="store_true")
    args = parser.parse_args()

    if args.output_root.exists() and not args.allow_existing_output:
        raise FileExistsError(
            f"Output already exists: {args.output_root}; pass --allow-existing-output only to resume"
        )
    args.output_root.mkdir(parents=True, exist_ok=True)
    contract_path = args.output_root / "analysis_contract_ordering_corrected_v94.json"
    if not contract_path.exists():
        write_json(contract_path, root_contract(args.output_root))
    compatibility_contract = args.output_root / "analysis_contract_v93.json"
    if not compatibility_contract.exists():
        payload = json.loads(contract_path.read_text(encoding="utf-8"))
        payload["compatibility_filename_only"] = True
        payload["authoritative_contract"] = contract_path.name
        write_json(compatibility_contract, payload)

    deployments = (
        list(v93.DEPLOYMENTS)
        if args.deployment == "all"
        else [args.deployment]
    )
    for deployment in deployments:
        config = v93.DEPLOYMENTS[deployment]
        rebuild_injection_maps(
            deployment,
            config["source_key"],
            config["seeds"],
            args.output_root,
        )
    if args.prepare_only:
        print("[hold] corrected sky inputs prepared; scoring not started", flush=True)
        return

    original_pair_loader = v81._load_pair_sky
    original_real_loader = v93.load_real_maps
    v81._load_pair_sky = corrected_pair_loader_factory(args.output_root)
    v93.load_real_maps = fixed_load_real_maps
    try:
        for deployment in deployments:
            v93.run_deployment(deployment, args.output_root, selected_seed=None)
    finally:
        v81._load_pair_sky = original_pair_loader
        v93.load_real_maps = original_real_loader

    rewrite_seed_contracts(args.output_root)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    historical_package = REPO / "packages/gwtc_sky_resolution_v93_20260730.tar.gz"
    contract["completed_utc"] = datetime.now(timezone.utc).isoformat()
    contract["historical_v93_package_sha256_after_rerun"] = (
        sha256_file(historical_package) if historical_package.is_file() else None
    )
    contract["historical_v93_package_unchanged"] = (
        contract["historical_v93_package_sha256_before_rerun"]
        == contract["historical_v93_package_sha256_after_rerun"]
    )
    write_json(contract_path, contract)
    compatibility_payload = dict(contract)
    compatibility_payload["compatibility_filename_only"] = True
    compatibility_payload["authoritative_contract"] = contract_path.name
    write_json(args.output_root / "analysis_contract_v93.json", compatibility_payload)
    print("[complete] ordering-corrected v9.4 rerun finished", flush=True)


if __name__ == "__main__":
    main()
