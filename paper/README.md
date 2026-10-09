# Figure and table source data

These sources accompany the October manuscript. They supplement, rather than
replace, the models and primary experiment in the published September archives.

| Result | Source directory |
| --- | --- |
| Main run-specific scores, retrieval and candidate comparisons | source_data/c_current/ |
| Empirical background probabilities | source_data/c_current/conditional_fpp/ |
| Fixed-bound GPD estimates and sensitivity | source_data/c_current/fixed_endpoint_gpd/ |
| Updated five-size O3 runtime | source_data/runtime_curve_20261007/ |
| Joint-search composition and full ranking | source_data/xrun_global/ |
| O3 waveform display arrays | source_data/current_waveform_display/ |

The figure/table mapping is in source_data/c_current/C_SOURCE_INDEX.md. Historical
exports in that directory are identified in its archive section. Use the mapped
files, not an older file with a similar figure number.

The complete runtime evidence archive is
runtime_evidence/LAWS_O3_FULL_RUNTIME_CURVE_EVIDENCE.zip, SHA-256
`b5f2612121b62316c86860ddef4ad22b8dc7af01f7d0f49ca4c76a427107e8c5`.
Extract it into a new directory and supply that directory to
source_data/runtime_curve_20261007/plot_runtime_curve.py with --data-dir.
The plot script checks its manifest before reading the timing records.

The CSV sources provide the numerical summaries and `figures/` contains the
corresponding vector figures. The runtime plotting script is included. Layout
scripts for some other current figures remain to be added. See
`../docs/REPRODUCTION.md` for the available calculation and inference entry points.
