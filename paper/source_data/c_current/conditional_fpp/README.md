# GWLR-FPP-01 source data

These files support the empirical part of current Figure 4 and the supplementary real-candidate background tables. The continuous B=4 GPD curve and candidate model estimates, including Main Table 1, are supplied separately by `../fixed_endpoint_gpd/`. See `../C_SOURCE_INDEX.md` for the complete mapping. These estimates supersede the earlier annualized background quantities for these displays. The method remains GWLR-UC-01 (C_PHYSICAL); rankings and model scores were not changed.

- `score_FPP_curves.csv`: every distinct mean-score threshold and its integer reference exceedance count.
- `recovery_operating_points.csv`: score thresholds for 50% and 90% global test-pair recovery, including ties.
- `candidate_top20_all_runs.csv`: frozen Top-20 rows for all three runs, source counts and deletion sensitivities.
- `efficiency_at_one_percent.csv`: complete-test pair counts below empirical conditional FPP 1%.

Each run uses 90 isolated validation sources and all 4005 unordered pairs. FPP is the empirical fraction k/4005 at the three-model mean score, not the mean of model-specific fractions. Zero empirical counts remain numeric zero in the archival CSV for reproducibility; their displayed FPP is a dash and the displayed count is 0/4005. No positive upper limit is inferred. Deletion sensitivities are not confidence intervals. Scores and their background are averaged identically. Effective background exposure and a real-population probability calibration were not established, so no annual FAR or lensing probability is supplied.

Source archive: `gwlr_fpp_injection_real_01_20260922T061000Z_r2_deliverables.tar.gz`

SHA-256: `6e57bda810b8744aca1b0c694365e1549cc89cec53e65e77f20073600819c907`
