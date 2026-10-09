# Current fixed-endpoint GPD source data

These exports supply the B=4 model curve and candidate estimates in Figure 4 and Main Table 1, together with supplementary endpoint sensitivity and diagnostics. The underlying experiment remains GWLR-UC-01 / C_PHYSICAL. Scores, rankings and model weights are unchanged.

- `figure4_curves.csv`: empirical and B=4 model curves and recovery-marker coordinates for Figure 4 a–c.
- `fits.csv`: run-specific fits above the 95th reference percentile for B=4, 4.5 and 5.
- `candidate_Top50_all_bounds.csv`: candidate scores, empirical counts and model estimates. Select B=4, ranks 1–20 for Figure 4 d–f; O4b ranks 1–10 for Main Table 1; all bounds, ranks 1–20 for supplementary model tables.
- `O3_Top50_PE_official_fixed_endpoints.csv`, `O4a_Top50_PE_official_fixed_endpoints.csv`, `O4b_Top50_PE_official_fixed_endpoints.csv`: candidate joins with PE and available official information.
- `heldout_checks.csv`, `deletions.csv`, `deletion_summary.csv`: predictive and deletion diagnostics, not proof of calibrated real-candidate significance.
- `tail02_gpd_fits.csv`, `tail02_gates.csv`, `tail02_HELDOUT_ENDPOINT_DIAGNOSTIC.csv`: free-endpoint diagnostic exports.

Each run uses three-model mean scores and 4,005 pairs from 90 isolated validation sources. The endpoint is an imposed modelling assumption, not a known physical maximum. Columns explicitly labelled percent are percentages; probability fields are fractions. Differences among B=4, 4.5 and 5 describe endpoint sensitivity, not confidence intervals. These estimates are not annual FAR or posterior lensing probabilities.

The empirical source data and integer counts are retained in `../conditional_fpp/`. Historical annualized and earlier tail-fit exports remain archived in `../conditional_far/` and `../tail_model/`.
