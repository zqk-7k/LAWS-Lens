# LAWS-Lens current source data: GWLR-UC-01 / C_PHYSICAL

This directory supersedes the historical plot inputs for the active C figures and tables. Historical directories remain archived and must not be silently pooled with these data. Three model seeds are 2026091721, 2026091722 and 2026091723.

## Plot mapping

The current Figure 1 is `../../figures/workflow_ai_multimodel.pdf`. It includes the later typography and score-label updates, including bold `(AI waveform score)`. Earlier assembly and symbol-coloring scripts generate intermediate layouts, not this final figure.

| Display | Data |
| --- | --- |
| Main Fig. 1 | `../../figures/workflow_ai_multimodel.pdf`; two network icons and ellipses summarize the four waveform branches and their outputs. The candidate inset reproduces Main Fig. 5a--c, with O3 ranks, PE overlaps and waveform traces. The waveform-score and fusion-weight symbols are red; the added waveform-score label is bold black. |
| Main Fig. 2 | fig2_density_curves.csv; channel_distributions.csv; first fixed 190-event subset, first model |
| Main Fig. 3 | roc_per_model.csv, recall_per_model.csv; n=190; model means only, without shaded bands |
| Main Fig. 4 (a–c) | fixed_endpoint_gpd/figure4_curves.csv (empirical and B=4 model curves and recovery markers); fixed_endpoint_gpd/fits.csv; conditional_fpp/score_FPP_curves.csv; conditional_fpp/recovery_operating_points.csv |
| Main Fig. 4 (d–f) | fixed_endpoint_gpd/candidate_Top50_all_bounds.csv, selecting B=4 and ranks 1–20; conditional_fpp/candidate_top20_all_runs.csv for empirical counts |
| Main Fig. 5 | O3_real_all.csv and O4a_real_all.csv, selecting ranks 1–10; fig5_O3_top10.csv and fig5_O4a_top10.csv are the corresponding display exports. Vertical references: fig5_simulation_fpp_thresholds.csv, derived from conditional_fpp/score_FPP_curves.csv. Waveforms: ../current_waveform_display/fig4_gwtc3_rank1_waveforms.csv (O3), fig5_O4a_rank1_waveforms.csv (O4a). PE morphology guides: fig5_rank1_PE_morphology_guides.csv; O4a provenance: fig5_rank1_provenance.json |
| Main Table 1 | O4b_real_all.csv, selecting ranks 1–10 for scores, contributions and PE; fixed_endpoint_gpd/O4b_Top50_PE_official_fixed_endpoints.csv and fixed_endpoint_gpd/candidate_Top50_all_bounds.csv for counts and B=4 model FPP |
| Extended Data Fig. 1 | ../../figures/waveform_compatibility_workflow.pdf; schematic waveform workflow |
| Extended Data Fig. 2 | roc_per_model.csv, recall_per_model.csv; n=450; model means without bands |
| Extended Data Fig. 3 | ../runtime_curve_20261007/plot_runtime_curve.py uses runtime_summary.csv and scaling_timings.csv in that directory: five nested samples of 5/15/30/46/62 events, with medians and minimum--maximum ranges from three initial-processing measurements and nine cached pair calculations per size. stage_timings.csv records event preparation. Earlier runtime_comparison/ and runtime_62/ inputs and make_fig_runtime_lines.py remain historical, not the active plotting entry. |
| Extended Data Fig. 4 | retrieval_metrics_per_seed.csv; split=test, methods waveform-short, waveform-FRT, waveform-OMC, waveform-new; cumulative waveform contributions |
| Supplementary Fig. S3 | sky_area_before_after_summary.csv; paired raw/calibrated coverage and median 90% sky areas. sky_area_before_after_events.csv contains all 1,350 event-map areas and map hashes. The original sky_truth_coverage_and_runtime.csv is retained as the archived raw-area and coverage check. |
| Supplementary Tables S2 / S3 | retrieval_summary_190.csv / retrieval_summary_450.csv; full channel metrics, means and sample SDs; the 450-event table is not duplicated in Extended Data |
| Supplementary Figs. S1–S6 | S1: fixed-budget precision; S2: injection SNRs; S3: sky coverage and area (see above); S4: model recall variation; S5: real-candidate PE agreement; S6: GPD endpoint sensitivity. Other inputs: test_events.csv; retrieval_metrics_per_seed.csv; real_PE_official_budget_summary.csv; fixed_endpoint_gpd/candidate_Top50_all_bounds.csv |
| Supplementary candidate / coefficient / numerical tables | O3_real_all.csv, O4a_real_all.csv, O4b_real_all.csv; selected_weights.csv; sky_resolution_summary.csv; sky_resolution_retrieval.csv; cluster_intervals.csv |
| Supplementary empirical FPP and efficiency tables | conditional_fpp/candidate_top20_all_runs.csv; conditional_fpp/efficiency_at_one_percent.csv |
| Supplementary fixed-endpoint GPD tables and sensitivity figure | fixed_endpoint_gpd/candidate_Top50_all_bounds.csv (ranks 1–20 for tables, rank 1 for the figure); fixed_endpoint_gpd/fits.csv; fixed_endpoint_gpd/heldout_checks.csv; fixed_endpoint_gpd/deletion_summary.csv. tail02_gpd_fits.csv, tail02_gates.csv and tail02_HELDOUT_ENDPOINT_DIAGNOSTIC.csv in the same directory supply the free-endpoint diagnostics. |

The source columns SIS and PM are inherited identifiers for GW-LMC environments without and with subhalos, not analytic SIS or point-mass injection populations.

## Evaluation convention

Each complete test catalog contains 180 doublets and 90 isolated sources (450 events). Every fixed smaller subset contains 35 doublets from each lens group and 50 isolated sources (190 events). The same 500 source-subset definitions are used across runs and models. Smaller-catalog curves average subsets within a model before model averaging. Main Figure 3 and Extended Data Figure 2 show mean curves without shaded bands. Sample SDs across the three models remain in the retrieval tables. The 500 overlapping subsets are not treated as independent experiments. Ties receive uniform-order expected inclusion probability. ROC curves use attainable thresholds and do not extrapolate below one null pair.

## Current background convention

Figure 4 uses purple for empirical FPP and orange for GPD estimates in both rows. In panels d–f, all nonzero empirical values use the same filled-circle marker. Unboxed black annotations identify zero-exceedance candidates by rank without assigning a probability coordinate. Panel e reads “Ranks 1–6: above all background scores”; panel f reads “Rank 1: above all background scores”. The grey dashed reference is 1%; the one-count dotted reference appears only in a–c. These display changes leave all source values unchanged.

Figure 5 daggers precede rank numbers and mark at least one published LVK screening FPP below 0.01, using official_po_fpp / official_ml_fpp for O3 and official_po_fpp / official_phazap_fpp for O4a. The marked ranks are O3: 1, 3, 4, 8, 9 and O4a: 1–9. These flags agree with official_frontend in the unchanged source tables and are distinct from the Figure 4 simulation-background FPP estimates.

Figure 5 a,d now also show purple dashed vertical lines at S=1.406159510297031 (O3) and S=2.063838029034593 (O4a). Each is the smallest tabulated validation-background score with empirical conditional FPP below 0.01, corresponding to 40/4005=0.009987515605493134. The adjacent lower tabulated scores have 41/4005 exceedances. Rounded display labels are 1.406 and 2.064; the full-precision thresholds and their selection rule are in fig5_simulation_fpp_thresholds.csv. These are simulation-based empirical references, not GPD fits, LVK selection thresholds, or new candidate cuts. Neither candidate rankings nor displayed scores changed. Hanabi comparisons do not determine the dagger flags.

Figure 4 uses the three-model mean score for both reference pairs and real candidates. Each run has 90 isolated validation sources and 4,005 unordered reference pairs. Empirical conditional FPP is the number of reference scores at least as high as the candidate score, divided by 4,005. It is not the mean of three model-specific fractions. Zero counts remain zero in raw exports but do not establish zero risk.

The GPD curve is fitted above the 95th reference percentile with fixed endpoint B=4. Main Table 1 and the orange model estimates in Figure 4 use this setting. The supplementary sensitivity comparison also reports B=4.5 and B=5. These bounds are modelling assumptions, not established score limits. Neither empirical nor model estimates are annual FAR, official-search FPP, or a posterior lensing probability. Probability columns contain fractions unless their name explicitly states percent; display percentages multiply fractions by 100.

The Figure 4 recovery markers are defined from the 180 true test pairs in each complete 450-event catalog, at 50% and 90% pair recovery. Their ordinates are evaluated against the 4,005-pair validation background. The 100,845 unrelated test pairs remain relevant to complete-catalog retrieval and false-pair counts, but are not the denominator of the current Figure 4 FPP curves.

## Historical exports retained for provenance

The following files are archived and are not the active source mapping for Figure 4, Figure 5 or Main Table 1:

- `fig4_threshold_curves_450.csv`, `fig4_operating_points_450.csv`, `fig4_threshold_provenance.json`, `candidate_background_per_model.csv` and `burden_per_model.csv`: earlier per-model background and burden displays.
- `conditional_far/`: earlier annualized estimates without a validated effective background exposure; not operational FAR.
- `tail_model/`: earlier exploratory tail fits; not the current B=4 candidate estimates.
- `fig5_PE_morphology_guides.csv`: earlier morphology-guide export. Use `fig5_rank1_PE_morphology_guides.csv` for the current figure.

Historical numerical files are preserved unchanged. Use the current mappings above rather than selecting a file solely by its figure number.

## Figure revisions and public records

Main Figure 5 was regenerated from the same inputs with two-line event labels and larger text. No candidate, score, PE overlap, waveform sample or LVK screening marker changed. The source table exports are provided here; the original layout script remains separate from the data exports. Candidate table edits alter font size and column spacing only. The 30 September display revision moves daggers from event names to before rank numbers, without changing any values or marked pairs.

Public software: https://github.com/zqk-7k/LAWS-Lens. Software DOI: https://doi.org/10.5281/zenodo.22949904; data/models DOI: https://doi.org/10.5281/zenodo.22949906. Both records are public, verified on 9 October 2026. The archived manuscript snapshot is afb86ff; later source data and runtime measurements are mapped separately in the repository's docs/VERSIONS.md.

## Source lineage

Original completion archive SHA-256: `bd5e31e619b0de31a7d605351fc06ff6be5cb828b59d5d0d667269d58e1865a5`.

Native-map archive SHA-256: `1287ef0c4a7850a2c3052a2c1c133cc89a96e65dbc16acdcf6f8e5d41ee90583`.

The reviewed delivery is `gwlr_unified_c_physical_20260918T134500Z_r1`. INPUT_SHA256.json records the local immutable inputs used by the figure replay. Representative190 raw pair CSVs and the replay/plot scripts remain in the local revision bundle rather than adding large duplicate tables or Python files to Overleaf.

Injection generation: training/scripts/c_data.py and source_generator.py. Physical source proposal: legacy_unified_ab.py and waveform_multiscale_data.py. Recovery: c_trigger.py. Calibration: completion/scripts/uab_scoring.py. Evaluation and consensus tie rules: uab_evaluate.py. Real PE audit: uab_real.py. Sky floor: uab_real.py and legacy_completion.py use 1e-300 before the log for real/injection overlaps.

These exports reproduce figures and table summaries, not full strain generation. Full regeneration additionally requires raw noise, original GW-LMC tables, source/subset plans, model checkpoints, recovery bank, PSDs, adapters and dependency versions. The published scientific archives and later source-data additions are identified in the repository version map. The current real-event timing benchmark and Supplementary Table S27 use the mapping in `../runtime_curve_20261007/README.md`.

## Joint-search supplement added on 2026-10-02

Supplementary Section 12 and Tables S28-S30 use a separately trained joint O3/O4 model ensemble and its exploratory all-pair ranking. Their provenance and source tables are in `../xrun_global/README.md`. This addition does not replace the run-specific main-experiment results mapped above. The previous release snapshot does not automatically contain the new joint-search supplement.
