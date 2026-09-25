#!/usr/bin/env python3
"""Deterministic contract tests for the unified v8.1 sky statistic."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import healpy as hp
import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.real_search.unified_sky_v81 import (
    MAIN_SKY_COLUMN,
    attach_main_sky_score,
    pair_feature_frame,
    score_contract,
)


def main() -> None:
    nside = 8
    npix = hp.nside2npix(nside)
    uniform = np.full(npix, 1.0 / npix, dtype=np.float32)
    localized = np.full(npix, 1e-12, dtype=np.float32)
    localized[17] = 1.0
    localized /= localized.sum()
    maps = np.stack((uniform, uniform, localized, localized))
    pairs, _ = pair_feature_frame(
        maps,
        idx_i=np.asarray([0, 2]),
        idx_j=np.asarray([1, 3]),
        device="cpu",
    )
    uniform_log_bf = float(pairs.loc[0, MAIN_SKY_COLUMN])
    localized_log_bf = float(pairs.loc[1, MAIN_SKY_COLUMN])
    if abs(uniform_log_bf) > 1e-5:
        raise AssertionError(
            f"Uniform-vs-uniform must have log B_sky=0, got {uniform_log_bf}"
        )
    if localized_log_bf <= 0.0:
        raise AssertionError(
            "Identical localized maps must support a common sky position"
        )

    base = pd.DataFrame(
        {
            "idx_i": [0, 2],
            "idx_j": [1, 3],
            "waveform_score": [0.1, 0.2],
            "time_score": [0.3, 0.4],
        }
    )
    attached = attach_main_sky_score(base, pairs)
    if not np.array_equal(
        attached["sky_score"].to_numpy(),
        attached[MAIN_SKY_COLUMN].to_numpy(),
    ):
        raise AssertionError("Main sky score is not raw log B_sky")
    payload = {
        "status": "passed",
        "nside": nside,
        "uniform_log_bf": uniform_log_bf,
        "identical_localized_log_bf": localized_log_bf,
        "contract": score_contract(),
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
