from __future__ import annotations

import importlib
from pathlib import Path


base = importlib.import_module("scripts.experiments.80_mixed_sis_pm_catalog_modality_compare")
fresh = importlib.import_module("scripts.experiments.84_fresh50_full_catalog_ranking")

DATA_ROOT = Path("data_generation/ligo_et_snr_like_2000_matchroots/LIGO")
OUT_ROOT = Path("runs/ligo_et_snr_like_2000_full_catalog_20260623")

base.ROOTS[("SIS", "LIGO")] = DATA_ROOT
base.ROOTS[("PM", "LIGO")] = DATA_ROOT

fresh.OUT_ROOT = OUT_ROOT
fresh.ENCODER_ROOT = OUT_ROOT / "fresh_mixed_encoders"
fresh.JOBS = [("LIGO", "noisy"), ("LIGO", "pure")]


if __name__ == "__main__":
    fresh.main()
