# ARCHIVED: conditional annualized false-pair estimates

This directory is retained for historical provenance only. It does not supply the current Figure 4 or candidate tables. Current empirical estimates are in `../conditional_fpp/`, and fixed-endpoint model estimates are in `../fixed_endpoint_gpd/`; see `../C_SOURCE_INDEX.md`. Numerical files in this archive are unchanged.

The calculation below describes the earlier annualization exercise, not a calibrated operational FAR. No additional injections or model training were performed for that exercise.

For each run and model, the fraction `f_bg` counts simulated unrelated-pair scores greater than or equal to the threshold, divided by 100,845. The calculation excludes the 180 same-source pairs. All remaining cross-source pairs are retained, including their controlled source and amplitude selection.

`conditional_far_per_year = C(N,2) * f_bg / T_ref`, with N=62/74/86 and T_ref the first-to-last span of the frozen injection calendar in years of 365.25 days. This assumes transfer of the simulated unrelated-pair tail to the target catalog. It is not a calibrated operational FAR, a consensus-selection false-alarm probability, or a lensing probability.

- `normalization.csv`: GPS endpoints, calendar spans, merged HL live durations, event/pair counts, and normalization factors.
- `threshold_curves.csv`: exact background counts on the displayed threshold grid, with both calendar and live-time normalizations.
- `operating_points.csv`: thresholds and false counts at 50%/90% recovery of 180 true pairs, separately per model.
- `candidate_per_model.csv`: thresholds, integer counts, fractions and rates for all three models and each run's Top-100.
- `candidate_top100.csv` and `candidate_top20.csv`: model means and full model ranges, not confidence intervals.

The alternative live-time normalization is per HL live year, not per calendar year. Neither time basis has been validated as the effective exposure of the catalog-selection process. Model repeats do not create additional background exposure. Pair dependence is retained in the input scores, and no independent-binomial or Poisson significance conversion is performed.

Threshold curves stop at the largest observed unrelated score. One-count levels are numerical resolution, not confidence bounds. No score or candidate order is changed. Original exceedance-fraction source files remain available in the parent directory.
