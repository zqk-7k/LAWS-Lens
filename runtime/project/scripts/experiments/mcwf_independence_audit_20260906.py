#!/usr/bin/env python3
"""Read-only global GPS and actual source-parameter independence audit."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

import mcwf_development_20260905 as dev
import mcwf_fresh_confirmation_20260906 as fresh


def audit(root):
    output = fresh.data_root(root)
    oldmass, oldnoise, newnoise, source, files = [], [], [], [], []
    for dep in ("gwtc3", "gwtc4"):
        for bank in (dev.ORCH.SOURCE_ROOT / dep / "shared/physical_h1l1_source_bank",
                     dev.OLD / f"data/source_banks/{dep}"):
            for path in sorted(bank.glob("*/*metadata.parquet")):
                f = pd.read_parquet(path)
                oldmass.append(f[["mass_1_detector", "mass_2_detector"]].to_numpy(float))
                files.append({"path": str(path), "sha256": dev.sha(path), "rows": len(f)})
        paths = [dev.ORCH.SOURCE_ROOT / dep / "shared/noise_bank_manifest.csv"]
        paths += [dev.OLD / f"data/noise_banks/{dep}/noise_manifest_{s}.csv" for s in ("train", "validation")]
        if dep == "gwtc3":
            paths += [root / f"data/o3_only_noise/noise_manifest_{s}.csv" for s in ("train", "validation")]
        for path in paths:
            f = pd.read_csv(path)
            a = f.reference_start_gps.to_numpy(float)
            b = (f.reference_end_gps.to_numpy(float) if "reference_end_gps" in f
                 else a + f.reference_duration_s.to_numpy(float))
            oldnoise += [{"start": s, "end": t, "path": str(path), "deployment": dep} for s, t in zip(a, b)]
            files.append({"path": str(path), "sha256": dev.sha(path), "rows": len(f)})
        noise = pd.read_csv(output / f"{dep}/noise/noise_manifest.csv")
        newnoise.append(noise.assign(deployment=dep))
        for cs in fresh.CATALOG_SEEDS:
            path = output / f"{dep}/catalog_{cs}/source_systems.parquet"
            source.append(pd.read_parquet(path).assign(deployment=dep, catalog_seed=cs))
            files.append({"path": str(path), "sha256": dev.sha(path), "rows": len(source[-1])})
    oldmass = np.concatenate(oldmass)
    source = pd.concat(source, ignore_index=True)
    noise = pd.concat(newnoise, ignore_index=True)
    conflicts = []
    for i, a in noise.iterrows():
        for j, b in noise.iloc[i+1:].iterrows():
            if a.start_gps < b.end_gps and a.end_gps > b.start_gps:
                conflicts.append({"type": "fresh_vs_fresh", "left": int(i), "right": int(j)})
        for b in oldnoise:
            if a.start_gps < b["end"]+16 and a.end_gps > b["start"]-16:
                conflicts.append({"type": "fresh_vs_development_with16s_guard", "left": int(i), "right": b})
    newmass = source[["m1_det", "m2_det"]].to_numpy(float)
    nearest, _ = cKDTree(oldmass).query(newmass, p=np.inf)
    repeated_old = int((nearest < 1e-10).sum())
    repeated_new = int(source.duplicated(["m1_det", "m2_det"]).sum())
    # Differing masses alone are sufficient to rule out identical full parents;
    # they do not establish a new independent lens-environment population.
    output_record = {
        "fresh_sources": len(source), "fresh_noise_blocks": len(noise),
        "old_metadata_rows_compared": len(oldmass), "old_noise_references_compared": len(oldnoise),
        "duplicate_source_uid": int(source.source_uid.duplicated().sum()),
        "fresh_duplicate_mass_pairs_exact": repeated_new,
        "old_parent_mass_pair_matches_at_1e_minus10_Msun_tolerance": repeated_old,
        "minimum_old_mass_pair_Linf_distance_Msun": float(nearest.min()),
        "global_GPS_conflicts": conflicts,
        "source_noise_independence_pass": not conflicts and repeated_old == 0 and repeated_new == 0 and not source.source_uid.duplicated().any(),
        "lens_environment_independence_claimed": False,
        "known_event_exclusion": "retains originating off-source pool onsource exclusion; no claim all unrecognized transients are absent",
    }
    dev.json_write(output / "GLOBAL_SOURCE_NOISE_INDEPENDENCE.json", output_record)
    dev.csv_write(output / "INDEPENDENCE_INPUT_MANIFEST.csv", pd.DataFrame(files))
    dev.csv_write(output / "GLOBAL_NOISE_BLOCKS.csv", noise)
    if not output_record["source_noise_independence_pass"]:
        raise RuntimeError("Global source/noise audit failed; do not claim independent confirmation")
    print(output_record, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    audit(p.parse_args().root)
