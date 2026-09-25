#!/usr/bin/env python3
"""Repair a verified pre-scoring GPS overlap in a new confirmation directory."""
from datetime import datetime, timezone
import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

import mcwf_development_20260905 as dev
import mcwf_fresh_confirmation_20260906 as fresh
import mcwf_o3_runmatched_data_20260906 as rm


def conflicts(frame):
    kept, bad = [], []
    for row in frame.sort_values("bank_index").itertuples():
        interval = (float(row.start_gps), float(row.end_gps))
        if any(interval[0] < b and interval[1] > a for a, b in kept):
            bad.append(int(row.bank_index))
        else:
            kept.append(interval)
    return bad


def run(root):
    old = root / "confirmation"
    new = root / "confirmation_global_noise_fixed"
    if new.exists() or (root / "contracts/ACTIVE_CONFIRMATION.json").exists():
        raise RuntimeError("Repair directory/registry exists; refuse overwrite")
    if list(old.glob("**/*_pairs.parquet")):
        raise RuntimeError("Fresh scoring already exists; cannot call this a pre-scoring repair")
    _, data, v3, _ = fresh.modules()
    critical = list(old.glob("*/noise/*.csv")) + list(old.glob("*/catalog_*/source_systems.parquet"))
    before = {str(p): dev.sha(p) for p in critical}
    shutil.copytree(old, new)
    all_changes, independence = [], []
    for dep in ("gwtc3", "gwtc4"):
        noise = new / dep / "noise"
        manifest = pd.read_csv(noise / "noise_manifest.csv")
        bad = conflicts(manifest)
        historical = pd.read_csv(dev.ORCH.SOURCE_ROOT / dep / "shared/noise_bank_manifest.csv")
        excluded = [(float(r.reference_start_gps)-16, float(r.reference_start_gps)+float(r.reference_duration_s)+16) for r in historical.itertuples()]
        for split in ("train", "validation"):
            paths = [dev.OLD / f"data/noise_banks/{dep}/noise_manifest_{split}.csv"]
            if dep == "gwtc3":
                paths.append(root / f"data/o3_only_noise/noise_manifest_{split}.csv")
            for path in paths:
                f = pd.read_csv(path)
                excluded += [(float(r.reference_start_gps)-16, float(r.reference_end_gps)+16) for r in f.itertuples()]
        excluded += [(float(r.start_gps)-16, float(r.end_gps)+16) for r in manifest.itertuples()]
        if dep == "gwtc3":
            pool = pd.read_csv(root / "data/o3_only_noise/official_O3_full_strain_offsource_pool.csv")
            source_run = dev.MAIN / "cache/source_run"
        else:
            source_run = data.SOURCE_RUNS[dep]
            pool = pd.read_parquet(source_run / "data/real_noise_injections/offsource_noise_segments.parquet")
        cache = v3.HdfCache(source_run, max_files=8)
        refs = np.load(noise / "reference.npy", mmap_mode="r+")
        psds = np.load(noise / "psd.npy")
        changed_catalogs = set()
        for bank in bad:
            records = []
            for record in pool.to_dict("records"):
                for a, b in rm.subtract_intervals(float(record["segment_start"]), float(record["segment_end"]), excluded):
                    if b-a >= v3.PSD_SECONDS+2:
                        records.append({**record, "segment_start": a, "segment_end": b})
            selected, rejects = data.select_valid_noise_references(pd.DataFrame(records), 1,
                np.random.default_rng(data.stable_seed("global-GPS-repair", dep, bank, 202609080)), v3.PSD_SECONDS, cache, v3)
            record, start, reference, freq, psd = selected[0]
            oldrow = manifest.loc[manifest.bank_index.eq(bank)].iloc[0].to_dict()
            if any(start < b and start+v3.PSD_SECONDS > a for a, b in excluded):
                raise RuntimeError("Replacement still overlaps protected GPS intervals")
            refs[bank] = reference
            psds[bank] = psd
            select = manifest.bank_index.eq(bank)
            manifest.loc[select, ["parent_event", "start_gps", "end_gps", "H1_path", "L1_path"]] = [
                str(record.event_name), start, start+v3.PSD_SECONDS, str(record.H1_path), str(record.L1_path)]
            changed_catalogs.add(int(oldrow["catalog_seed"]))
            excluded.append((start-16, start+v3.PSD_SECONDS+16))
            all_changes.append({"deployment": dep, "bank_index": bank, "before": oldrow,
                "after": manifest.loc[select].iloc[0].to_dict(), "replacement_rejections": rejects})
        refs.flush()
        np.save(noise / "psd.npy", psds)
        dev.csv_write(noise / "noise_manifest.csv", manifest)
        if conflicts(manifest):
            raise RuntimeError("Remaining cross-file GPS overlap")
        invalidated = []
        for cs in fresh.CATALOG_SEEDS:
            directory = new / dep / f"catalog_{cs}"
            if not directory.exists():
                continue
            events = directory / "events"
            for marker in list(events.glob("FRESH_BBH_*.json")):
                records = json.loads(marker.read_text())
                if any(int(item["noise_bank_index"]) in bad for item in records):
                    invalidated.append(marker.name)
                    for item in records:
                        for suffix in ("_full24.npy", "_sky512.npy", ".fits"):
                            (events / (item["event_uid"]+suffix)).unlink(missing_ok=True)
                    marker.unlink()
            if cs in changed_catalogs:
                (directory / "GENERATION_COMPLETE.json").unlink(missing_ok=True)
                (directory / "event_manifest.parquet").unlink(missing_ok=True)
        dev.json_write(noise / "GLOBAL_GPS_REPAIR.json", {"replaced_banks": bad, "global_overlap_after": 0,
            "invalidated_completed_systems": invalidated, "old_generation_preserved_at": str(old / dep),
            "selection": "keep earlier bank_index; deterministic new reference excluding all historical/development and all original fresh intervals"})
        dev.json_write(noise / "COMPLETE.json", {"excluded_interval_overlap": 0, "independent_blocks": len(manifest),
            "source_run": str(source_run), "global_GPS_overlap_after": 0, "reference_sha256": dev.sha(noise / "reference.npy")})
        independence.append({"deployment": dep, "global_overlaps_before": len(bad), "global_overlaps_after": 0,
                              "blocks": len(manifest), "invalidated_systems": len(invalidated)})
    altered = [path for path, digest in before.items() if dev.sha(Path(path)) != digest]
    if altered:
        raise RuntimeError("Original generation inputs were modified")
    dev.json_write(root / "contracts/GLOBAL_GPS_NOISE_REPAIR.json", {
        "utc": datetime.now(timezone.utc).isoformat(), "root_cause": "legacy helper checked by event name, not physical GPS; GW190828 files overlap",
        "before_any_fresh_scores": True, "old_inputs_unchanged": True,
        "original_candidate_models_and_scientific_rules_unchanged": True,
        "old_directory": str(old), "corrected_directory": str(new), "changes": all_changes,
        "independence": independence, "old_critical_hashes": before})
    dev.json_write(root / "contracts/ACTIVE_CONFIRMATION.json", {"directory": new.name,
        "reason": "pre-scoring global-GPS independence repair; original generation archived without overwrite"})
    print(json.dumps(independence), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    a = p.parse_args()
    run(a.root)
