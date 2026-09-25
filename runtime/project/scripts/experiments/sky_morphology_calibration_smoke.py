#!/usr/bin/env python3
import importlib.util
import sys
from pathlib import Path

import pandas as pd


project = Path("/root/autodl-tmp/gw-catalog")
path = project / "scripts/experiments/sky_morphology_one_vs_two_stage_exploratory.py"
spec = importlib.util.spec_from_file_location("morph_cal_smoke", path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

root = project / "results/sky_morphology_one_vs_two_stage_exploratory_20260827_20260830T154500Z"
deployment = "gwtc3"
seed = 202607241
seed_root = root / "results" / deployment / f"seed_{seed}"
validation = pd.read_parquet(seed_root / "synthetic_validation_pair_morphology_nside512.parquet")
events = pd.read_parquet(module.input_seed_dir(deployment, seed) / "results/mixed_val_synthetic_events_v7.parquet")
partition = module.validation_system_partition(events, deployment, seed)
validation["calibration_subset"] = module.pair_partition(validation, partition)
event_features = pd.read_csv(seed_root / "synthetic_validation_event_sky_morphology_nside512.csv")
event_features["calibration_subset"] = event_features.idx.map(partition.set_index("idx").calibration_subset)
area = float(event_features.loc[event_features.calibration_subset == "fit", "area90_deg2"].median())
fit = validation.loc[validation.calibration_subset == "fit"].reset_index(drop=True)
tune = validation.loc[validation.calibration_subset == "tune"].reset_index(drop=True)
calibrator = module.fit_morphology_calibrator(fit, tune, area, deployment, seed)
scored = module.apply_morphology_calibrator(tune, calibrator)
print("COUNTS", validation.calibration_subset.value_counts().to_dict(), "AREA", area, flush=True)
print(
    "CELLS",
    {
        key: {
            field: value.get(field)
            for field in ("active", "n_fit_positive", "n_fit_negative", "n_tune_positive", "n_tune_negative", "reason")
        }
        for key, value in calibrator["cells"].items()
    },
    flush=True,
)
print("OOD", float(scored.sky_morph_ood.mean()), flush=True)
one, grid = module.select_one_stage(scored)
two, two_grid = module.select_two_stage(
    scored,
    module.FROZEN_V93_WEIGHTS[deployment][seed],
    one["positive_cap"],
    one["negative_cap"],
)
print("ONE", one, flush=True)
print("TWO", two, flush=True)
print("SMOKE_CALIBRATION_OK", len(grid), len(two_grid), flush=True)
