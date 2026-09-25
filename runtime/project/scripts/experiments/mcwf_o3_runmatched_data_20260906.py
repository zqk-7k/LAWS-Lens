#!/usr/bin/env python3
"""O3-only noise replacement, preserving source waveforms/SNR/training split."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

import mcwf_development_20260905 as dev
import mcwf_encoder_body_20260906 as body
import mcwf_adaptive_psd_encoder_20260906 as adaptive
import mcwf_phasebank_20260905 as phase


def freeze(root):
    out = root / "contracts/O3_RUNMATCHED_NOISE_ONLY.json"
    if out.exists():
        return
    dev.json_write(out, {
        "config": "O3-RUNMATCHED-RAW",
        "newly_verified_issue": {"old_training_O1O2_references": 16, "old_training_total": 48,
                                 "old_validation_O1O2_references": 10, "old_validation_total": 16},
        "interpretation": "source provenance mismatch, not proof it explains every PE discordance",
        "change": "replace development noise/PSD with official-O3-event off-source segments only; keep source waveforms, parent split, individual SNR targets and number of variants",
        "network": "same RAW-PHASE-SOURCE, same losses/epochs/seeds/checkpoint/temperature rules",
        "new_noise_split": "parent-event-disjoint train/validation; windows disjoint from all historical and previous development256s noise references with16s guard",
        "noise_training_reference_count": 48, "noise_validation_reference_count": 16,
        "feature_bank": "same576 template grid and extraction, fixed median PSD now computed from O3-only TRAIN noise",
        "historical_injection_comparison": "unchanged original BAYESTAR catalogs remain mixed-run provenance; report honestly, do not call a new blind O3-only test",
        "real_or_official_labels_in_model": False,
        "real_PE_role": "development audit only",
        "unchanged": ["peak2s4096", "physical source parameters", "SNR targets", "one-dimensional time_score", "sky_score", "C-fixed weights", "real scopes"],
        "seed": 202609064, "no_historical_outputs_modified": True,
    })


def subtract_intervals(lo, hi, excluded):
    parts = [(lo, hi)]
    for start, stop in sorted(excluded):
        new = []
        for a, b in parts:
            if stop <= a or start >= b:
                new.append((a, b))
            else:
                if start > a:
                    new.append((a, start))
                if stop < b:
                    new.append((stop, b))
        parts = new
    return parts


def build_noise(root):
    out = root / "data/o3_only_noise"
    marker = out / "COMPLETE.json"
    if marker.exists():
        return out
    out.mkdir(parents=True, exist_ok=True)
    data = dev.module(dev.OLD / "scripts/waveform_multiscale_data.py", "mcwf_o3_data")
    v3 = data.load_module(data.V3_SCRIPT, "mcwf_o3_noise")
    source = dev.MAIN / "cache/source_run"
    input_cache = v3.HdfCache(source, max_files=8)
    events = pd.read_csv(dev.MAIN / "results/strict_h1l1_event_audit.csv")
    segment_rows = []
    for event in events.itertuples():
        if not event.strict_h1l1_preprocessing_pass:
            continue
        detector_rows = {item["detector"]: item for item in json.loads(event.detector_audit)}
        if set(detector_rows) != {"H1", "L1"}:
            raise RuntimeError("Strict event missing detector provenance")
        starts, ends = [], []
        for detector in ("H1", "L1"):
            values, start, duration = input_cache.get(detector_rows[detector]["path"])
            if abs(len(values) / duration - v3.RAW_SAMPLE_RATE) > 1e-3:
                raise RuntimeError("Noise source sample rate mismatch")
            starts.append(start)
            ends.append(start + duration)
        lo, hi = max(starts), min(ends)
        for kind, a, b in (("left", lo, min(hi, event.gps_time - 128)),
                           ("right", max(lo, event.gps_time + 128), hi)):
            if b - a >= v3.PSD_SECONDS + 2:
                segment_rows.append({"event_name": event.event_name, "segment_kind": "O3_offsource_" + kind,
                    "segment_start": a, "segment_end": b,
                    "H1_path": detector_rows["H1"]["path"], "L1_path": detector_rows["L1"]["path"]})
    segments = pd.DataFrame(segment_rows)
    dev.csv_write(out / "official_O3_full_strain_offsource_pool.csv", segments)
    dev.json_write(out / "SOURCE_POOL_EXPANSION.json", {
        "reason": "old55-event run directory supplied only40/48 valid independent references after exclusions; use all currently downloaded official-O3 strict62 files",
        "original_failure_preserved_in": "logs/o3_strict_runmatched_generation.log",
        "no_quality_or_independence_requirement_relaxed": True,
        "onsource_exclusion_seconds": 128, "source": str(source), "n_pool_segments": len(segments)})
    official = set(pd.read_csv(dev.MAIN / "cache/source_run/data/event_manifest.csv").event_name.astype(str))
    segments = segments.loc[segments.event_name.astype(str).isin(official)].copy()
    original = pd.read_csv(dev.ORCH.SOURCE_ROOT / "gwtc3/shared/noise_bank_manifest.csv")
    historical_parents = set(original.source_event.astype(str))
    segments = segments.loc[~segments.event_name.astype(str).isin(historical_parents)].copy()
    excluded = [(float(r.reference_start_gps) - 16, float(r.reference_start_gps) + float(r.reference_duration_s) + 16)
                for r in original.itertuples()]
    for split in ("train", "validation"):
        old = pd.read_csv(dev.OLD / f"data/noise_banks/gwtc3/noise_manifest_{split}.csv")
        excluded.extend((float(r.reference_start_gps) - 16, float(r.reference_end_gps) + 16) for r in old.itertuples())
    records = []
    for row in segments.to_dict("records"):
        for lo, hi in subtract_intervals(float(row["segment_start"]), float(row["segment_end"]), excluded):
            if hi - lo >= v3.PSD_SECONDS + 2:
                records.append({**row, "segment_start": lo, "segment_end": hi})
    pool = pd.DataFrame(records)
    names = sorted(set(pool.event_name.astype(str)), key=lambda x: data.stable_seed(202609064, "O3-noise", x))
    if len(names) < 8:
        raise RuntimeError(f"Insufficient O3-only independent noise parents: {len(names)}")
    validation = set(names[:max(4, round(len(names) * .25))])
    training = set(names) - validation
    cache = v3.HdfCache(source, max_files=8)
    manifests = []
    for split, parents, count in (("train", training, 48), ("validation", validation, 16)):
        rng = np.random.default_rng(data.stable_seed(202609064, split))
        selected, rejected = data.select_valid_noise_references(pool.loc[pool.event_name.astype(str).isin(parents)], count,
                            rng, v3.PSD_SECONDS, cache, v3)
        refs = np.lib.format.open_memmap(out / f"noise_reference_{split}.npy", mode="w+", dtype=np.float32,
                                        shape=(count, 2, v3.NOISE_REFERENCE_SAMPLES))
        psds, rows = [], []
        for i, (row, start, reference, freq, psd) in enumerate(selected):
            refs[i] = reference
            psds.append(psd)
            rows.append({"split": split, "bank_index": i, "parent_event": str(row.event_name),
                         "reference_start_gps": start, "reference_end_gps": start + v3.PSD_SECONDS,
                         "all_O3_scope": str(row.event_name) in official, "reused_reference_overlap": False})
        refs.flush()
        np.save(out / f"noise_psd_{split}.npy", np.stack(psds))
        np.save(out / f"noise_psd_frequency_{split}.npy", freq)
        manifest = pd.DataFrame(rows)
        dev.csv_write(out / f"noise_manifest_{split}.csv", manifest)
        manifests.append(manifest)
        dev.json_write(out / f"{split}_selection_audit.json", rejected)
    overlap = set(manifests[0].parent_event) & set(manifests[1].parent_event)
    if overlap:
        raise RuntimeError("O3 train/validation noise-parent overlap")
    dev.json_write(marker, {"all_noise_in_official_O3_event_scope": True, "train_validation_parent_overlap": 0,
                   "excluded_historical_parent_overlap": 0, "eligible_parent_count": len(names),
                   "new_reference_count": 64, "source": str(source)})
    return out


def prepare(root):
    freeze(root)
    noise_root = build_noise(root)
    out = root / "cache/runmatched/gwtc3"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "DATA_COMPLETE.json").exists():
        return
    data = dev.module(dev.OLD / "scripts/waveform_multiscale_data.py", "mcwf_o3_materialize")
    v3 = data.load_module(data.V3_SCRIPT, "mcwf_o3_injection")
    freq = np.load(noise_root / "noise_psd_frequency_train.npy")
    median = np.median(np.load(noise_root / "noise_psd_train.npy"), axis=0)
    bank = adaptive.whitened_bank(adaptive.spectra(root), freq, median)
    np.save(out / "quadrature_bank.npy", bank)
    audit = []
    for split in ("train", "validation"):
        raw_path = out / f"{split}_raw2s.npy"
        metadata_path = out / f"{split}_metadata.parquet"
        if not (raw_path.exists() and metadata_path.exists()):
            metadata = body.source_metadata("gwtc3", split)
            refs = np.load(noise_root / f"noise_reference_{split}.npy", mmap_mode="r")
            psds = np.load(noise_root / f"noise_psd_{split}.npy", mmap_mode="r")
            freq = np.load(noise_root / f"noise_psd_frequency_{split}.npy")
            raw = np.lib.format.open_memmap(raw_path, mode="w+", dtype=np.float16, shape=(len(metadata), 2, 4096))
            rng = np.random.default_rng(data.stable_seed(202609064, "regenerate", split))
            clean_cache = {}
            for family in ("sis", "pm"):
                for image, number in (("a", 1), ("b", 2)):
                    clean_cache[(family, image)] = np.load(dev.OLD / f"data/source_banks/gwtc3/{family.upper()}_data_0222/{family.upper()}_h_strain_{number}.npy", mmap_mode="r")
            new_rows = []
            for idx, row in enumerate(metadata.to_dict("records")):
                clean = np.asarray(clean_cache[(row["family_slot"], row["image"])][int(row["source_index"] )])
                noise, bi, offset = data.draw_noise_window(refs, rng, v3.RAW_PADDED_SAMPLES)
                _, mixed, info = v3._preprocess_injection(clean, noise, freq, psds[bi], float(row[f"target_snr_{row['image']}"]))
                if not np.isfinite(mixed).all():
                    raise RuntimeError("Nonfinite regenerated O3 injection")
                raw[idx] = dev.TRAIN.make_window_view(mixed, 2)
                new_rows.append({**row, "actual_noise_bank_index": bi, "actual_noise_offset_samples": offset,
                                 "actual_noise_split": split, "actual_noise_run": "O3", **{f"new_{k}": v for k, v in info.items()}})
                if idx % 500 == 0:
                    print(json.dumps({"O3_noise_regeneration": split, "done": idx, "total": len(metadata)}), flush=True)
            raw.flush()
            pd.DataFrame(new_rows).to_parquet(metadata_path, index=False)
        raw = np.load(raw_path, mmap_mode="r")
        feature_path = out / f"{split}_features.npy"
        if not feature_path.exists():
            features = phase.features(raw, bank)
            np.save(feature_path, features)
        audit.append({"split": split, "raw_sha256": dev.sha(raw_path), "features_sha256": dev.sha(feature_path),
                      "metadata_sha256": dev.sha(metadata_path), "events": len(raw)})
    dev.json_write(out / "DATA_COMPLETE.json", {"source_waveforms_changed": False, "noise_run": "O3_only",
                   "source_split_changed": False, "input_samples": 4096, "input_seconds": 2, "audit": audit})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    args = p.parse_args()
    prepare(args.root)
