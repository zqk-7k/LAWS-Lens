#!/usr/bin/env python3
"""Read-only inventory for a genuinely new confirmation population."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import mcwf_development_20260905 as dev
import mcwf_encoder_body_20260906 as body


def inventory(root):
    src = dev.module(dev.PROJECT / "scripts/real_search/34_generate_physical_h1l1_source_bank.py", "mcwf_fresh_source_inventory")
    data = dev.module(dev.OLD / "scripts/waveform_multiscale_data.py", "mcwf_fresh_data_inventory")
    table = src.load_tables(src.GW_LMC_ROOT)
    oldrows, oldids = set(), set()
    for dep in ("gwtc3", "gwtc4"):
        for bank in (dev.ORCH.SOURCE_ROOT / dep / "shared/physical_h1l1_source_bank", dev.OLD / f"data/source_banks/{dep}"):
            for file in bank.glob("*/*metadata.parquet"):
                frame = pd.read_parquet(file)
                if "gwlmc_origin_row" in frame:
                    oldrows.update(frame.gwlmc_origin_row.dropna().astype(int))
                elif "gwlmc_row" in frame:
                    oldrows.update(frame.gwlmc_row.dropna().astype(int))
                if "gwlmc_event_id" in frame:
                    oldids.update(frame.gwlmc_event_id.dropna().astype(int))
    available = table.loc[~table.gwlmc_row.isin(oldrows) & ~table.event_id.isin(oldids)].copy()
    results = []
    for dep in ("gwtc3", "gwtc4"):
        schedule = src.load_schedule(dev.ORCH.SOURCE_ROOT / dep / "shared/h1l1_live_schedule.csv")
        records = []
        for r in available.itertuples():
            row = available.loc[r.Index]
            images = src.choose_images(row)
            if images is None or images["proposal_snr_ratio"] > 4:
                continue
            rng = np.random.default_rng(data.stable_seed(202609060, dep, "fresh-feasibility", r.event_id))
            if src.place_pair(images["delay_days"], schedule, rng) is not None:
                records.append({"gwlmc_row": r.gwlmc_row, "event_id": r.event_id,
                                "is_subhalo": bool(row.lens_is_subhalo), "delay_days": images["delay_days"]})
        out = pd.DataFrame(records)
        dev.csv_write(root / f"audit/confirmation/{dep}_unused_feasible_systems.csv", out)
        source_run = data.SOURCE_RUNS[dep]
        segments = pd.read_parquet(source_run / "data/real_noise_injections/offsource_noise_segments.parquet")
        intervals = []
        hist = pd.read_csv(dev.ORCH.SOURCE_ROOT / dep / "shared/noise_bank_manifest.csv")
        for r in hist.to_dict("records"):
            intervals.append({"kind": "historical", **r})
        for split in ("train", "validation"):
            file = dev.OLD / f"data/noise_banks/{dep}/noise_manifest_{split}.csv"
            for r in pd.read_csv(file).to_dict("records"):
                intervals.append({"kind": "development_" + split, **r})
        dev.csv_write(root / f"audit/confirmation/{dep}_protected_noise_intervals.csv", pd.DataFrame(intervals))
        dev.csv_write(root / f"audit/confirmation/{dep}_available_offsource_segments.csv", segments)
        results.append({"deployment": dep, "full_gwlmc_rows": len(table), "excluded_row_count": len(oldrows),
                        "excluded_global_source_ids": len(oldids), "unseen_rows": len(available),
                        "feasible_unseen_smooth": int((~out.is_subhalo).sum()) if len(out) else 0,
                        "feasible_unseen_subhalo": int(out.is_subhalo.sum()) if len(out) else 0,
                        "offsource_segments": len(segments), "protected_noise_references": len(intervals),
                        "new_signal_or_test_scores_generated": False})
    dev.json_write(root / "audit/confirmation/FRESH_DATA_INVENTORY.json", results)
    print(results, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    args = p.parse_args()
    inventory(args.root)
