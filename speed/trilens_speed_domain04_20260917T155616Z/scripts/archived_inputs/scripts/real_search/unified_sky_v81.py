#!/usr/bin/env python3
"""Frozen unified-sky v8.1 score contract.

ET-3, GWTC-3/O3 and GWTC-4.1/O4a all use the same main pair statistic:

    B_sky = N_pix * sum_k P_i[k] P_j[k]
    sky_score = log(B_sky)

The HEALPix maps may come from different measurement models, but the
probability normalization, common-source hypothesis and pair score are
identical.  O90 and cross-HPD remain diagnostic columns only.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from scripts.real_search.unified_sky_v8 import (
    FULL_SKY_DEG2,
    SKY_FEATURE_COLUMNS,
    change_nside_probability_mass,
    credible_levels,
    event_map_diagnostics,
    normalize_probability_maps,
    pair_feature_frame,
    pair_statistic_matrices,
    write_json,
)


MAIN_SKY_COLUMN = "sky_log_bayes_factor_raw"


def main_sky_matrix(matrices: dict[str, np.ndarray]) -> np.ndarray:
    score = np.asarray(matrices[MAIN_SKY_COLUMN], dtype=np.float32).copy()
    np.fill_diagonal(score, -np.inf)
    return score


def attach_main_sky_score(
    base: pd.DataFrame,
    sky: pd.DataFrame,
) -> pd.DataFrame:
    """Attach raw log B_sky while retaining all diagnostic sky statistics."""
    keys = ["idx_i", "idx_j"]
    payload = sky.copy()
    payload["sky_score"] = payload[MAIN_SKY_COLUMN].astype(np.float32)
    output = base.drop(
        columns=[
            column
            for column in base.columns
            if column.startswith("sky_")
            or column in {"sky_score", "sky_bayes_factor"}
        ],
        errors="ignore",
    ).merge(payload, on=keys, how="left", validate="one_to_one")
    output["sky_bayes_factor"] = np.exp(
        np.clip(output[MAIN_SKY_COLUMN], -80.0, 80.0)
    )
    return output


def score_contract() -> dict[str, Any]:
    return {
        "version": "unified_sky_v81",
        "main_score": "sky_score = log(B_sky)",
        "bayes_factor": "B_sky=N_pix*sum_k(P_i[k]*P_j[k])",
        "map_requirement": (
            "Each event-level HEALPix map is non-negative and normalized to "
            "unit probability mass under a uniform all-sky prior."
        ),
        "main_ranking_features": [MAIN_SKY_COLUMN],
        "diagnostic_only_features": ["sky_o90", "sky_cross_hpd"],
        "validation_fitted_sky_composite": False,
        "test_labels_used": False,
        "real_catalog_ranks_used": False,
    }
