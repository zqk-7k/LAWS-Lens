#!/usr/bin/env python3
"""Add missing O4a apparent-distance diagnostics, never a ranking criterion."""
import argparse
import hashlib
from pathlib import Path

import pandas as pd

import mcwf_development_20260905 as dev


def run(root):
    pairs = dev.real_frame("gwtc4", dev.SEEDS[0])[["pair_key", "event_i", "event_j"]]
    names = sorted(set(pairs.event_i) | set(pairs.event_j))
    manifest = dev.BASE.primary_manifest("gwtc4").set_index("event_name")
    map_contract = pd.read_csv(dev.BAY / "results/gwtc4/real_public_pe_ordering_audit.csv").set_index("event_name")
    samples, rows = {}, []
    for name in names:
        row = manifest.loc[name].copy()
        fixed = map_contract.loc[name]
        if fixed.source_group != row.sky_map_group:
            raise RuntimeError("Frozen posterior-group manifests disagree")
        path = Path(fixed.source_path)
        row["sky_map_path"] = str(path)
        row["sky_map_internal_group"] = row.sky_map_group
        samples[name] = dev.MAINCODE.load_pe_samples(row)["luminosity_distance"]
        rows.append({"event_name": name, "path": str(path), "group": row.sky_map_internal_group,
                     "samples": len(samples[name]), "bytes": path.stat().st_size,
                     "distance_sample_sha256": hashlib.sha256(samples[name].tobytes()).hexdigest()})
    results = []
    for row in pairs.itertuples():
        stats = dev.MAINCODE.posterior_metrics(samples[row.event_i], samples[row.event_j])
        results.append({"pair_key": row.pair_key, **{f"pe_dl_app_{k}": v for k, v in stats.items()}})
    pd.DataFrame(results).to_parquet(root / "audit/gwtc4_additional_distance_pe.parquet", index=False)
    dev.csv_write(root / "audit/gwtc4_distance_exact_group_manifest.csv", pd.DataFrame(rows))
    dev.json_write(root / "audit/O4_DISTANCE_SUPPLEMENT_CONTRACT.json", {
        "events": len(names), "pairs": len(pairs), "group_selection": "frozen original event-manifest sky_map_group, no dynamic fallback",
        "statistic": "same O3 audit 256-bin BC over combined0.1-99.9percentile span; robust sigma=(q84-q16)/2",
        "physical_role": "apparent luminosity distance, magnification-dependent; NOT a common-source veto",
        "scoring_and_rank_changed": False, "existing_intrinsic_PE_metrics_changed": False})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    run(p.parse_args().root)
