# Supplementary Figure S3: paired sky coverage and area

The summaries use all 450 test events (270 distinct source systems) in each of O3, O4a and O4b. No events are excluded. Calibration temperatures are frozen at 1.5 from validation, without fitting to test data.

- `sky_area_before_after_events.csv`: raw and calibrated 90% areas, identifiers, source weights, map SHA-256 hashes and normalization checks for 1,350 maps.
- `sky_area_before_after_summary.csv`: source-weighted coverage and unweighted event-area medians. `ratio_of_median_areas` is distinct from `median_area_ratio` (the median of event-wise ratios).
- `sky_truth_coverage_and_runtime.csv`: preserved earlier summary, used to verify raw-area medians and both coverage series.

Native NUNIQ probability densities are conservatively rasterized onto equal-area NESTED pixels at Nside=512. Finer cells are summed by parent and coarser cells are subdivided with constant density. The raw map is normalized. The calibrated probability mass is proportional to `max(P, 1e-300)^(1/1.5)` and renormalized. Area is the minimum integer pixel count reaching cumulative probability 0.9 after descending sorting, multiplied by pixel area in square degrees. Pixel ordering does not alter this area statistic.

Repeated equal-area pixel masses are compressed with integer multiplicities for computation. Full dense rasterization reproduces both area calculations for 12 test maps (four per run). All 1,350 raw event areas match their archived values within 1e-6 square degrees, and every map hash matches its archived event record. The medians include all events equally, preserving the original area summary convention. Coverage gives each source system total weight one (each of a doublet's maps contributes one half).

These are descriptive summaries of the fixed test catalogs, not independent population estimates. No maps, model parameters, pair scores or rankings were changed.
