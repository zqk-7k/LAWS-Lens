#!/usr/bin/env python3
"""Reconstruct metadata in the exact cached ET-3 test-embedding order."""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        default="runs/p2a_ann_et3_noisy_meta_20260618/catalog_meta.csv",
    )
    args = ap.parse_args()

    base = importlib.import_module(
        "scripts.experiments.80_mixed_sis_pm_catalog_modality_compare"
    )
    detector, mode = "ET3", "noisy"
    encoder_dir = Path(
        "runs/et3_fresh50_full_catalog_20260616/fresh_mixed_encoders/"
        "et3_noisy_mixed_sis_pm_ep50"
    )
    cfg = base.make_cfg(detector, mode, encoder_dir)
    arrays = {
        family: base.FamilyArrays(family, base.ROOTS[(family, detector)], mode)
        for family in base.FAMILIES
    }
    splits = {}
    for index, family in enumerate(base.FAMILIES):
        splits[family] = base.split_indices(len(arrays[family].l1), cfg.seed + index)
        splits[f"{family}_U"] = base.split_indices(
            len(arrays[family].unlensed), cfg.seed + 100 + index
        )

    dataset = base.MixedEvaluationSet(arrays, splits, "test", cfg)
    raw = base.mixed_obs_frame(detector, "test", splits, "raw").reset_index(drop=True)
    timing = base.mixed_obs_frame(detector, "test", splits, "time").reset_index(drop=True)
    embedding_count = len(np.load(encoder_dir / "test_embeddings.npy", mmap_mode="r"))
    if not (len(dataset) == len(raw) == len(timing) == embedding_count):
        raise RuntimeError(
            f"ordering inputs disagree: dataset={len(dataset)}, raw={len(raw)}, "
            f"timing={len(timing)}, embeddings={embedding_count}"
        )

    rows = []
    for event_id, item in enumerate(dataset.meta):
        is_lensed = item["tag"] in {"L1", "L2"}
        # Lensed images share pair_id. Every unlensed event receives a unique ID.
        source_id = int(item["pair_id"]) if is_lensed else int(embedding_count + event_id)
        rows.append(
            {
                "event_id": event_id,
                "source_id": source_id,
                "kind": item["family"] if is_lensed else "unlensed",
                "geocent_time": float(timing.loc[event_id, "trigger_time_obs"]),
                "ra": float(raw.loc[event_id, "ra"]),
                "dec": float(raw.loc[event_id, "dec"]),
                "sky_area_90_deg2": np.nan,
                "network_snr": float(timing.loc[event_id, "snr"]),
                **item,
            }
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(out, index=False)
    paired = frame[frame["kind"].isin(["SIS", "PM"])].groupby("source_id").size()
    if not (paired == 2).all():
        raise RuntimeError("generated lensed source IDs are not exact pairs")
    print(
        f"saved {out}: events={len(frame)}, lensed_systems={len(paired)}, "
        f"unlensed={(frame['kind'] == 'unlensed').sum()}"
    )


if __name__ == "__main__":
    main()
