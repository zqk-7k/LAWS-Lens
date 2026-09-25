"""Extensible auxiliary priors for catalog-level reranking."""

from .feature_builder import (
    observed_sky_pair_features,
    rank_feature_matrices,
    time_step_score_matrix,
)
from .observed_sky import (
    DETECTOR_SKY_SCENARIOS,
    build_observed_sky_table,
    public_observed_sky_features,
    scenario_for_detector,
    unit_from_radec,
    radec_from_unit,
    a90_to_sigma_rad,
)
from .scorer import select_best_weighted_lambdas

__all__ = [
    "DETECTOR_SKY_SCENARIOS",
    "a90_to_sigma_rad",
    "build_observed_sky_table",
    "observed_sky_pair_features",
    "public_observed_sky_features",
    "radec_from_unit",
    "rank_feature_matrices",
    "scenario_for_detector",
    "select_best_weighted_lambdas",
    "time_step_score_matrix",
    "unit_from_radec",
]
