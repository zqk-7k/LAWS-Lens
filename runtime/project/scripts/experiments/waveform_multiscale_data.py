#!/usr/bin/env python3
"""Build leakage-controlled, run-matched waveform development data.

The development proposal deliberately balances detector-frame chirp-mass and
SNR strata. It is a coverage proposal for representation learning, not an
astrophysical population model. The frozen BAYESTAR validation and locked-test
catalogs remain the population-level evaluation sets.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import bilby
import h5py
import numpy as np
import pandas as pd


PROJECT = Path("/root/autodl-tmp/gw-catalog")
SOURCE_SCRIPT = PROJECT / "scripts/real_search/34_generate_physical_h1l1_source_bank.py"
V3_SCRIPT = PROJECT / "scripts/experiments/20_real_noise_injection_v3_physical.py"
OLD_SOURCE_ROOT = PROJECT / "results/real_noise_injection_v5_physical_source_20260721"
SOURCE_RUNS = {
    "gwtc3": PROJECT / "runs/real_gwtc_lensing_search_20260625",
    "gwtc4": PROJECT / "runs/real_gwtc34_lensing_search_20260629_full_o4",
}
BASELINE = PROJECT / "results/bayestar_injection_sky_pe_20260901_20260901T102000Z"
BASELINE_SEEDS = (202607241, 202607242, 202607243)
DEPLOYMENT_LABEL = {"gwtc3": "GWTC3", "gwtc4": "GWTC4"}
MASS_BINS = ((5.0, 10.0), (10.0, 20.0), (20.0, 40.0), (40.0, 80.0), (80.0, 200.0))
SNR_BINS = ((8.0, 10.0), (10.0, 12.0), (12.0, 20.0), (20.0, 40.0))
N_PER_FAMILY = 600
N_UNLENSED = 600
N_TRAIN_PER_FAMILY = 480
N_VAL_PER_FAMILY = 120
N_VAL_UNLENSED = 240
TRAIN_VARIANTS = 4
NOISE_TRAIN_REFS = 48
NOISE_VAL_REFS = 16
SEED = 202609030


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**32 - 1)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)


def chirp_mass(m1: float, m2: float) -> float:
    return float((m1 * m2) ** (3.0 / 5.0) / (m1 + m2) ** (1.0 / 5.0))


def masses_from_chirp_q(mc: float, q: float) -> tuple[float, float]:
    m1 = mc * (1.0 + q) ** (1.0 / 5.0) / q ** (3.0 / 5.0)
    return float(m1), float(q * m1)


def newtonian_duration_from_40hz(mc: float) -> float:
    solar_mass_seconds = 4.925490947e-6
    return float(5.0 / 256.0 * (math.pi * 40.0) ** (-8.0 / 3.0) * (mc * solar_mass_seconds) ** (-5.0 / 3.0))


def balanced_mass_draws(count: int, rng: np.random.Generator) -> list[tuple[float, float, int]]:
    assignments = np.arange(count) % len(MASS_BINS)
    rng.shuffle(assignments)
    output: list[tuple[float, float, int]] = []
    for bin_index in assignments:
        lo, hi = MASS_BINS[int(bin_index)]
        for _ in range(1000):
            mc = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
            q = float(rng.uniform(0.25, 1.0))
            m1, m2 = masses_from_chirp_q(mc, q)
            if 3.0 <= m2 <= m1 <= 300.0:
                output.append((m1, m2, int(bin_index)))
                break
        else:
            raise RuntimeError(f"Unable to draw masses in bin {lo}-{hi}")
    return output


def assign_mass_proposal(
    table: pd.DataFrame,
    selected_by_slot: dict[str, list[dict[str, Any]]],
    rng: np.random.Generator,
) -> pd.DataFrame:
    out = table.copy()
    out["waveform_parent_uid"] = ""
    out["mass_stratum"] = -1
    used: set[int] = set()
    for slot, selected in selected_by_slot.items():
        draws = balanced_mass_draws(len(selected), rng)
        for local_index, (item, (m1, m2, stratum)) in enumerate(zip(selected, draws, strict=True)):
            row_index = int(item["row_index"])
            used.add(row_index)
            uid = f"WDME-{slot}-{local_index:04d}-{stable_seed(slot, local_index, m1, m2):08x}"
            out.loc[row_index, ["m1_det", "m2_det", "waveform_parent_uid", "mass_stratum"]] = [m1, m2, uid, stratum]
    remaining = np.asarray([idx for idx in range(len(out)) if idx not in used], dtype=np.int64)
    draws = balanced_mass_draws(len(remaining), rng)
    for local_index, (row_index, (m1, m2, stratum)) in enumerate(zip(remaining, draws, strict=True)):
        uid = f"WDME-U-{local_index:06d}-{stable_seed('U', local_index, m1, m2):08x}"
        out.loc[int(row_index), ["m1_det", "m2_det", "waveform_parent_uid", "mass_stratum"]] = [m1, m2, uid, stratum]
    return out


def augment_source_metadata(path: Path, table: pd.DataFrame) -> None:
    frame = pd.read_parquet(path)
    lookup = table.set_index("gwlmc_row")
    frame["waveform_parent_uid"] = frame["gwlmc_row"].map(lookup["waveform_parent_uid"])
    frame["gwlmc_origin_row"] = frame["gwlmc_row"].map(lookup["gwlmc_origin_row"]).astype(int)
    frame["mass_stratum"] = frame["gwlmc_row"].map(lookup["mass_stratum"]).astype(int)
    frame["chirp_mass_detector"] = [chirp_mass(a, b) for a, b in zip(frame.mass_1_detector, frame.mass_2_detector)]
    frame["duration_from_40hz_s"] = frame.chirp_mass_detector.map(newtonian_duration_from_40hz)
    frame.to_parquet(path, index=False)


def frozen_evaluation_origin_rows() -> set[int]:
    """Return every GW-LMC row used by frozen validation or locked test."""
    rows: set[int] = set()
    for deployment in ("gwtc3", "gwtc4"):
        for seed in BASELINE_SEEDS:
            for split in ("validation", "test"):
                path = BASELINE / f"results/{deployment}/seed_{seed}/{split}_bayestar_event_metrics.csv"
                frame = pd.read_csv(path, usecols=["family", "gwlmc_row"])
                values = frame.loc[frame.family.isin(["SIS", "PM"]), "gwlmc_row"].dropna()
                rows.update(values.astype(int).tolist())
    return rows


def build_disjoint_development_table(
    table: pd.DataFrame,
    frozen_rows: set[int],
    minimum_rows_per_family: int,
    src: Any,
    schedule: list[Any],
    maximum_proposal_snr_ratio: float,
) -> pd.DataFrame:
    """Exclude frozen systems, then clone only unused lens environments if needed.

    A clone receives a new synthetic row identifier and later an independently
    drawn detector-frame mass pair and waveform-parent UID.  The retained
    ``gwlmc_origin_row`` identifies the reused lens environment for audit.
    """
    base = table.loc[~table.gwlmc_row.astype(int).isin(frozen_rows)].copy()
    base["gwlmc_origin_row"] = base.gwlmc_row.astype(int)
    parts = []
    next_identifier = int(table.gwlmc_row.max()) + 1
    for subhalo in (False, True):
        family = base.loc[base.lens_is_subhalo.astype(bool).eq(subhalo)].copy()
        feasible_indices = []
        feasibility_rng = np.random.default_rng(stable_seed(SEED, "feasibility", subhalo))
        for index, row in family.iterrows():
            images = src.choose_images(row)
            if images is None or images["proposal_snr_ratio"] > maximum_proposal_snr_ratio:
                continue
            if src.place_pair(images["delay_days"], schedule, feasibility_rng) is not None:
                feasible_indices.append(index)
        family = family.loc[feasible_indices].copy()
        if family.empty:
            raise RuntimeError(f"No development lens environments for subhalo={subhalo}")
        repeats = max(1, int(math.ceil(minimum_rows_per_family / len(family))))
        for repeat in range(repeats):
            clone = family.copy()
            if repeat:
                clone["gwlmc_row"] = np.arange(next_identifier, next_identifier + len(clone), dtype=np.int64)
                next_identifier += len(clone)
            parts.append(clone)
    out = pd.concat(parts, ignore_index=True)
    if set(out.gwlmc_origin_row.astype(int)) & frozen_rows:
        raise RuntimeError("Frozen GW-LMC origin entered the development table")
    return out


def generate_source_bank(deployment: str, output: Path) -> None:
    marker = output / "physical_source_bank_summary.json"
    if marker.exists():
        return
    src = load_module(SOURCE_SCRIPT, f"source_generator_{deployment}")
    rng = np.random.default_rng(stable_seed(SEED, deployment, "source"))
    bilby.core.utils.random.seed(stable_seed(SEED, deployment, "bilby"))
    original_table = src.load_tables(src.GW_LMC_ROOT)
    frozen_rows = frozen_evaluation_origin_rows()
    schedule_path = OLD_SOURCE_ROOT / deployment / "shared/h1l1_live_schedule.csv"
    schedule = src.load_schedule(schedule_path)
    table = build_disjoint_development_table(
        original_table, frozen_rows, N_PER_FAMILY, src, schedule, 4.0
    )
    selected_by_slot: dict[str, list[dict[str, Any]]] = {}
    for slot, (subhalo, _) in src.FAMILY_SLOTS.items():
        selected_by_slot[slot] = src.select_lensed_rows(
            table, subhalo, N_PER_FAMILY, schedule, rng, 4.0
        )
    table = assign_mass_proposal(table, selected_by_slot, rng)
    output.mkdir(parents=True, exist_ok=True)
    generator = src.build_waveform_generator()
    ifos = list(bilby.gw.detector.InterferometerList(["H1", "L1"]))
    summaries: dict[str, Any] = {}
    excluded: set[int] = set()
    for slot, (_, label) in src.FAMILY_SLOTS.items():
        selected = selected_by_slot[slot]
        excluded.update(int(item["row_index"]) for item in selected)
        _, summaries[slot] = src.generate_family(slot, label, selected, table, output, generator, ifos)
        augment_source_metadata(output / f"{slot}_data_0222/physical_source_pair_metadata.parquet", table)
    summaries["unlensed"] = src.generate_unlensed(
        table, N_UNLENSED, excluded, schedule, rng, output, generator, ifos
    )
    augment_source_metadata(output / "Unlensed_data_0222/physical_unlensed_source_metadata.parquet", table)
    payload = {
        "status": "complete",
        "deployment": deployment,
        "scientific_role": "coverage-balanced waveform training proposal; not an astrophysical population prior",
        "detectors": ["H1", "L1"],
        "waveform": "IMRPhenomXPHM",
        "duration_s": 24,
        "sample_rate_hz": 4096,
        "mass_proposal": {
            "parameter": "detector-frame chirp mass",
            "bins_msun": MASS_BINS,
            "allocation": "equal count per family and bin",
            "q_range": [0.25, 1.0],
        },
        "parent_definition": "unique generated intrinsic/extrinsic waveform parameter hash",
        "frozen_validation_test_gwlmc_origin_rows_excluded": len(frozen_rows),
        "selected_frozen_origin_overlap": int(sum(
            len(set(pd.read_parquet(output / f"{slot}_data_0222/physical_source_pair_metadata.parquet").gwlmc_origin_row.astype(int)) & frozen_rows)
            for slot in src.FAMILY_SLOTS
        )),
        "lens_environment_reuse_note": (
            "Only frozen-disjoint GW-LMC lens environments may repeat when the unused subhalo pool is smaller "
            "than 600; every generated source has independently drawn masses and a unique waveform_parent_uid."
        ),
        "summaries": summaries,
    }
    write_json(marker, payload)


def nonoverlapping_reference_starts(
    rows: pd.DataFrame,
    count: int,
    rng: np.random.Generator,
    duration: float,
) -> list[tuple[pd.Series, float]]:
    candidates: list[tuple[pd.Series, float]] = []
    by_event_intervals: dict[str, list[tuple[float, float]]] = {}
    attempts = 0
    while len(candidates) < count and attempts < count * 1000:
        attempts += 1
        row = rows.iloc[int(rng.integers(0, len(rows)))]
        lo = float(row.segment_start)
        hi = float(row.segment_end) - duration
        if hi <= lo:
            continue
        start = float(rng.uniform(lo, hi))
        interval = (start, start + duration)
        key = str(row.event_name)
        overlap = any(not (interval[1] <= a or interval[0] >= b) for a, b in by_event_intervals.get(key, []))
        if overlap:
            continue
        by_event_intervals.setdefault(key, []).append(interval)
        candidates.append((row, start))
    if len(candidates) != count:
        raise RuntimeError(f"Only {len(candidates)}/{count} independent reference windows")
    return candidates


def select_valid_noise_references(
    pool: pd.DataFrame,
    count: int,
    rng: np.random.Generator,
    duration: float,
    cache: Any,
    v3: Any,
) -> tuple[list[tuple[pd.Series, float, np.ndarray, np.ndarray, np.ndarray]], dict[str, int]]:
    """Select disjoint, finite H1/L1 windows before committing the noise bank."""
    selected: list[tuple[pd.Series, float, np.ndarray, np.ndarray, np.ndarray]] = []
    by_event_intervals: dict[str, list[tuple[float, float]]] = {}
    rejected = {
        "overlap": 0,
        "sample_rate": 0,
        "length": 0,
        "finite_fraction": 0,
        "psd": 0,
    }
    attempts = 0
    while len(selected) < count and attempts < count * 2000:
        attempts += 1
        row = pool.iloc[int(rng.integers(0, len(pool)))]
        lo = float(row.segment_start)
        hi = float(row.segment_end) - duration
        if hi <= lo:
            rejected["length"] += 1
            continue
        start = float(rng.uniform(lo, hi))
        interval = (start, start + duration)
        event_name = str(row.event_name)
        if any(not (interval[1] <= a or interval[0] >= b) for a, b in by_event_intervals.get(event_name, [])):
            rejected["overlap"] += 1
            continue

        channels = []
        reason: str | None = None
        for detector in v3.DETECTORS:
            values, gps_start, file_duration = cache.get(str(row[f"{detector}_path"]))
            sample_rate = len(values) / file_duration
            if abs(sample_rate - v3.RAW_SAMPLE_RATE) > 1e-3:
                reason = "sample_rate"
                break
            i0 = int(round((start - gps_start) * sample_rate))
            part = values[i0 : i0 + v3.NOISE_REFERENCE_SAMPLES]
            if len(part) != v3.NOISE_REFERENCE_SAMPLES:
                reason = "length"
                break
            if np.mean(np.isfinite(part)) < 0.999:
                reason = "finite_fraction"
                break
            channels.append(part)
        if reason is not None:
            rejected[reason] += 1
            continue

        reference = np.stack(channels).astype(np.float32)
        frequency, psd = v3.estimate_psd(reference)
        if (
            not np.all(np.isfinite(frequency))
            or not np.all(np.isfinite(psd))
            or np.any(psd <= 0)
        ):
            rejected["psd"] += 1
            continue
        by_event_intervals.setdefault(event_name, []).append(interval)
        selected.append((row, start, reference, frequency, psd))

    if len(selected) != count:
        raise RuntimeError(
            f"Only {len(selected)}/{count} valid independent noise references; "
            f"rejections={rejected}"
        )
    rejected["attempts"] = attempts
    return selected, rejected


def build_noise_partition(deployment: str, output: Path) -> None:
    summary_path = output / "noise_partition_summary.json"
    if summary_path.exists():
        return
    v3 = load_module(V3_SCRIPT, f"v3_noise_{deployment}")
    source_run = SOURCE_RUNS[deployment]
    segments = pd.read_parquet(source_run / "data/real_noise_injections/offsource_noise_segments.parquet")
    segments = segments[(segments.segment_end - segments.segment_start) >= v3.PSD_SECONDS + 2.0].copy()
    historical = pd.read_csv(OLD_SOURCE_ROOT / deployment / "shared/noise_bank_manifest.csv")
    excluded_events = set(historical.source_event.astype(str))
    segments = segments.loc[~segments.event_name.astype(str).isin(excluded_events)].reset_index(drop=True)
    event_names = sorted(segments.event_name.astype(str).unique())
    if len(event_names) < 8:
        raise RuntimeError(f"Only {len(event_names)} parent events remain after historical exclusion")
    ordered = sorted(event_names, key=lambda name: stable_seed(SEED, deployment, "noise-parent", name))
    n_val_events = max(4, int(round(0.25 * len(ordered))))
    val_events = set(ordered[:n_val_events])
    train_events = set(ordered[n_val_events:])
    if not train_events:
        raise RuntimeError("No training noise parents")
    cache = v3.HdfCache(source_run, max_files=12)
    output.mkdir(parents=True, exist_ok=True)
    all_rows: list[pd.DataFrame] = []
    rejection_audit: dict[str, dict[str, int]] = {}
    for split, parents, count in (("train", train_events, NOISE_TRAIN_REFS), ("validation", val_events, NOISE_VAL_REFS)):
        pool = segments.loc[segments.event_name.astype(str).isin(parents)].reset_index(drop=True)
        rng = np.random.default_rng(stable_seed(SEED, deployment, "noise", split))
        selected, rejection_audit[split] = select_valid_noise_references(
            pool, count, rng, v3.PSD_SECONDS, cache, v3
        )
        refs = np.lib.format.open_memmap(
            output / f"noise_reference_{split}.npy",
            mode="w+",
            dtype=np.float32,
            shape=(count, 2, v3.NOISE_REFERENCE_SAMPLES),
        )
        historical_psd_shape = np.load(
            OLD_SOURCE_ROOT / deployment / "shared/noise_psd_bank.npy", mmap_mode="r"
        ).shape
        psds = np.lib.format.open_memmap(
            output / f"noise_psd_{split}.npy",
            mode="w+",
            dtype=np.float64,
            shape=(count, 2, int(historical_psd_shape[-1])),
        )
        records = []
        frequency = None
        for index, (row, start, reference, frequency, psd) in enumerate(selected):
            refs[index] = reference
            psds[index] = psd
            records.append(
                {
                    "split": split,
                    "bank_index": index,
                    "parent_event": str(row.event_name),
                    "segment_kind": str(row.segment_kind),
                    "reference_start_gps": start,
                    "reference_end_gps": start + v3.PSD_SECONDS,
                    "finite_fraction": float(np.mean(np.isfinite(refs[index]))),
                }
            )
        refs.flush()
        psds.flush()
        np.save(output / f"noise_psd_frequency_{split}.npy", np.asarray(frequency, dtype=np.float64))
        frame = pd.DataFrame(records)
        frame.to_csv(output / f"noise_manifest_{split}.csv", index=False)
        all_rows.append(frame)
    merged = pd.concat(all_rows, ignore_index=True)
    train_parents = set(merged.loc[merged.split.eq("train"), "parent_event"])
    val_parents = set(merged.loc[merged.split.eq("validation"), "parent_event"])
    payload = {
        "status": "complete",
        "deployment": deployment,
        "historical_parent_event_overlap": int(len((train_parents | val_parents) & excluded_events)),
        "train_validation_parent_event_overlap": int(len(train_parents & val_parents)),
        "train_reference_count": NOISE_TRAIN_REFS,
        "validation_reference_count": NOISE_VAL_REFS,
        "reference_seconds": v3.PSD_SECONDS,
        "sample_rate_hz": v3.RAW_SAMPLE_RATE,
        "source": "public GWOSC run-matched H1/L1 off-source strain",
        "rejected_candidate_windows": rejection_audit,
    }
    write_json(summary_path, payload)
    if payload["historical_parent_event_overlap"] or payload["train_validation_parent_event_overlap"]:
        raise RuntimeError(f"Noise parent leakage: {payload}")


def stratified_split(metadata: pd.DataFrame, seed: int) -> tuple[np.ndarray, np.ndarray]:
    train: list[int] = []
    val: list[int] = []
    for _, part in metadata.groupby("mass_stratum"):
        indices = part.index.to_numpy(dtype=np.int64)
        rng = np.random.default_rng(stable_seed(seed, int(part.mass_stratum.iloc[0])))
        rng.shuffle(indices)
        n_val = int(round(len(indices) * N_VAL_PER_FAMILY / N_PER_FAMILY))
        val.extend(indices[:n_val].tolist())
        train.extend(indices[n_val:].tolist())
    if len(train) != N_TRAIN_PER_FAMILY or len(val) != N_VAL_PER_FAMILY:
        raise RuntimeError(f"Unexpected split sizes: {len(train)}/{len(val)}")
    return np.asarray(sorted(train)), np.asarray(sorted(val))


def snr_draw(bin_index: int, rng: np.random.Generator) -> float:
    lo, hi = SNR_BINS[int(bin_index)]
    return float(rng.uniform(lo, hi))


def draw_noise_window(refs: np.ndarray, rng: np.random.Generator, padded_samples: int) -> tuple[np.ndarray, int, int]:
    bank = int(rng.integers(0, len(refs)))
    offset = int(rng.integers(0, refs.shape[-1] - padded_samples + 1))
    return np.asarray(refs[bank, :, offset : offset + padded_samples], dtype=np.float32), bank, offset


def materialize_dataset(deployment: str, source_root: Path, noise_root: Path, output: Path) -> None:
    marker = output / "DATASET_COMPLETE.json"
    if marker.exists():
        return
    v3 = load_module(V3_SCRIPT, f"v3_materialize_{deployment}")
    output.mkdir(parents=True, exist_ok=True)
    split_contract: dict[str, Any] = {"deployment": deployment, "families": {}}
    rng = np.random.default_rng(stable_seed(SEED, deployment, "materialize"))
    refs = {
        split: np.load(noise_root / f"noise_reference_{split}.npy", mmap_mode="r")
        for split in ("train", "validation")
    }
    psds = {
        split: np.load(noise_root / f"noise_psd_{split}.npy", mmap_mode="r")
        for split in ("train", "validation")
    }
    freqs = {
        split: np.load(noise_root / f"noise_psd_frequency_{split}.npy")
        for split in ("train", "validation")
    }

    def inject(clean: np.ndarray, split: str, target_snr: float) -> tuple[np.ndarray, dict[str, Any]]:
        noise, bank, offset = draw_noise_window(refs[split], rng, v3.RAW_PADDED_SAMPLES)
        _, mixed, audit = v3._preprocess_injection(clean, noise, freqs[split], psds[split][bank], target_snr)
        return mixed, {**audit, "noise_bank_index": bank, "noise_offset_samples": offset}

    all_metadata: list[dict[str, Any]] = []
    for family in ("SIS", "PM"):
        src_dir = source_root / f"{family}_data_0222"
        meta = pd.read_parquet(src_dir / "physical_source_pair_metadata.parquet").reset_index(drop=True)
        train_ids, val_ids = stratified_split(meta, stable_seed(SEED, deployment, family, "split"))
        split_contract["families"][family] = {
            "train_source_indices": train_ids.tolist(),
            "validation_source_indices": val_ids.tolist(),
            "train_parent_uids": meta.loc[train_ids, "waveform_parent_uid"].tolist(),
            "validation_parent_uids": meta.loc[val_ids, "waveform_parent_uid"].tolist(),
        }
        clean_a = np.load(src_dir / f"{family}_h_strain_1.npy", mmap_mode="r")
        clean_b = np.load(src_dir / f"{family}_h_strain_2.npy", mmap_mode="r")
        train_rows = len(train_ids) * TRAIN_VARIANTS
        train_a = np.lib.format.open_memmap(output / f"{family.lower()}_train_a.npy", mode="w+", dtype=np.float32, shape=(train_rows, 2, v3.MODEL_SAMPLES))
        train_b = np.lib.format.open_memmap(output / f"{family.lower()}_train_b.npy", mode="w+", dtype=np.float32, shape=(train_rows, 2, v3.MODEL_SAMPLES))
        train_source = np.empty(train_rows, dtype=np.int32)
        train_records = []
        row_index = 0
        for source_index in train_ids:
            m = meta.iloc[int(source_index)]
            for variant in range(TRAIN_VARIANTS):
                bin_a = int((int(source_index) + 2 * variant) % len(SNR_BINS))
                bin_b = int((int(source_index) + 2 * variant + 1) % len(SNR_BINS))
                snr_a, snr_b = snr_draw(bin_a, rng), snr_draw(bin_b, rng)
                train_a[row_index], audit_a = inject(clean_a[int(source_index)], "train", snr_a)
                train_b[row_index], audit_b = inject(clean_b[int(source_index)], "train", snr_b)
                train_source[row_index] = int(source_index)
                train_records.append({
                    "deployment": deployment, "split": "train", "family": family,
                    "row_index": row_index, "source_index": int(source_index), "variant": variant,
                    "waveform_parent_uid": m.waveform_parent_uid, "mass_stratum": int(m.mass_stratum),
                    "mass_1_detector": float(m.mass_1_detector), "mass_2_detector": float(m.mass_2_detector),
                    "chirp_mass_detector": float(m.chirp_mass_detector), "duration_from_40hz_s": float(m.duration_from_40hz_s),
                    "target_snr_a": snr_a, "target_snr_b": snr_b, "snr_stratum_a": bin_a, "snr_stratum_b": bin_b,
                    **{f"a_{k}": val for k, val in audit_a.items()}, **{f"b_{k}": val for k, val in audit_b.items()},
                })
                row_index += 1
        train_a.flush(); train_b.flush()
        np.save(output / f"{family.lower()}_train_source_index.npy", train_source)
        pd.DataFrame(train_records).to_parquet(output / f"{family.lower()}_train_metadata.parquet", index=False)

        val_a = np.lib.format.open_memmap(output / f"{family.lower()}_validation_a.npy", mode="w+", dtype=np.float32, shape=(len(val_ids), 2, v3.MODEL_SAMPLES))
        val_b = np.lib.format.open_memmap(output / f"{family.lower()}_validation_b.npy", mode="w+", dtype=np.float32, shape=(len(val_ids), 2, v3.MODEL_SAMPLES))
        val_records = []
        for local_index, source_index in enumerate(val_ids):
            m = meta.iloc[int(source_index)]
            bin_a = int((2 * local_index) % len(SNR_BINS)); bin_b = int((2 * local_index + 1) % len(SNR_BINS))
            snr_a, snr_b = snr_draw(bin_a, rng), snr_draw(bin_b, rng)
            val_a[local_index], audit_a = inject(clean_a[int(source_index)], "validation", snr_a)
            val_b[local_index], audit_b = inject(clean_b[int(source_index)], "validation", snr_b)
            val_records.append({
                "deployment": deployment, "split": "validation", "family": family,
                "row_index": local_index, "source_index": int(source_index),
                "waveform_parent_uid": m.waveform_parent_uid, "mass_stratum": int(m.mass_stratum),
                "mass_1_detector": float(m.mass_1_detector), "mass_2_detector": float(m.mass_2_detector),
                "chirp_mass_detector": float(m.chirp_mass_detector), "duration_from_40hz_s": float(m.duration_from_40hz_s),
                "target_snr_a": snr_a, "target_snr_b": snr_b, "snr_stratum_a": bin_a, "snr_stratum_b": bin_b,
                **{f"a_{k}": val for k, val in audit_a.items()}, **{f"b_{k}": val for k, val in audit_b.items()},
            })
        val_a.flush(); val_b.flush()
        pd.DataFrame(val_records).to_parquet(output / f"{family.lower()}_validation_metadata.parquet", index=False)

    unlensed_src = source_root / "Unlensed_data_0222"
    unlensed_clean = np.load(unlensed_src / "unlensed_h_strain.npy", mmap_mode="r")
    unlensed_meta = pd.read_parquet(unlensed_src / "physical_unlensed_source_metadata.parquet").reset_index(drop=True)
    chosen = np.asarray(sorted(range(len(unlensed_meta)), key=lambda x: stable_seed(SEED, deployment, "unlensed", x))[:N_VAL_UNLENSED])
    val_u = np.lib.format.open_memmap(output / "unlensed_validation.npy", mode="w+", dtype=np.float32, shape=(len(chosen), 2, v3.MODEL_SAMPLES))
    u_records = []
    for local_index, source_index in enumerate(chosen):
        m = unlensed_meta.iloc[int(source_index)]
        snr_bin = int(local_index % len(SNR_BINS)); target = snr_draw(snr_bin, rng)
        val_u[local_index], audit = inject(unlensed_clean[int(source_index)], "validation", target)
        u_records.append({
            "deployment": deployment, "split": "validation", "family": "unlensed", "row_index": local_index,
            "source_index": int(source_index), "waveform_parent_uid": m.waveform_parent_uid,
            "mass_stratum": int(m.mass_stratum), "mass_1_detector": float(m.mass_1_detector),
            "mass_2_detector": float(m.mass_2_detector), "chirp_mass_detector": float(m.chirp_mass_detector),
            "duration_from_40hz_s": float(m.duration_from_40hz_s), "target_snr": target, "snr_stratum": snr_bin,
            **audit,
        })
    val_u.flush()
    pd.DataFrame(u_records).to_parquet(output / "unlensed_validation_metadata.parquet", index=False)

    train_parents = set().union(*(set(v["train_parent_uids"]) for v in split_contract["families"].values()))
    val_parents = set().union(*(set(v["validation_parent_uids"]) for v in split_contract["families"].values()))
    split_contract["source_parent_overlap_count"] = len(train_parents & val_parents)
    split_contract["locked_test_parent_namespace"] = "historical GW-LMC source-bank IDs; distinct from WDME parent UID namespace"
    write_json(output / "DATA_SPLIT_CONTRACT.json", split_contract)
    if split_contract["source_parent_overlap_count"]:
        raise RuntimeError("Training/validation source parent leakage")
    files = sorted(path for path in output.iterdir() if path.is_file())
    hashes = pd.DataFrame({"path": [str(path.relative_to(output)) for path in files], "sha256": [sha256_file(path) for path in files]})
    hashes.to_csv(output / "DATA_SHA256.csv", index=False)
    write_json(marker, {
        "status": "complete", "deployment": deployment, "train_variants": TRAIN_VARIANTS,
        "train_systems_per_family": N_TRAIN_PER_FAMILY, "validation_systems_per_family": N_VAL_PER_FAMILY,
        "validation_unlensed": N_VAL_UNLENSED, "prepared_sample_rate_hz": 2048,
        "prepared_duration_s": 24, "preprocessing": "frozen v7 PSD whitening, 40-580 Hz bandpass, anti-aliased 4096-to-2048 resampling",
        "snr_bins": SNR_BINS, "mass_bins_msun": MASS_BINS,
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--deployment", choices=("gwtc3", "gwtc4", "all"), default="all")
    args = parser.parse_args()
    deployments = ("gwtc3", "gwtc4") if args.deployment == "all" else (args.deployment,)
    for deployment in deployments:
        source_root = args.root / "data/source_banks" / deployment
        noise_root = args.root / "data/noise_banks" / deployment
        dataset_root = args.root / "data/development" / deployment
        generate_source_bank(deployment, source_root)
        build_noise_partition(deployment, noise_root)
        materialize_dataset(deployment, source_root, noise_root, dataset_root)


if __name__ == "__main__":
    main()
