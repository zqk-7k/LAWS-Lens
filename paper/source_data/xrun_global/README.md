# Supplementary global ranking: source mapping

This directory supports Supplementary Section 12 and Tables S28-S30. It is a separately trained joint-search extension, not a replacement for the run-specific main analysis or its FPP calibration.

## Frozen provenance

- Experiment: GWLR-XRUN-C-01, formal_joint_r1.
- Ranking extension: exploratory_global_ranking_20261002_r2.
- Original archive: GWLR_XRUN_C_01_EXPLORATORY_GLOBAL_RANKING.zip.
- Archive SHA-256: `fb3c48f7f84297414a0af15410c1222e07d63c2a65f04abdf0cb4c8aad997ff8`.
- Complete 24,531-pair CSV SHA-256: `d1f53076c4d7dd058a3d4c9ac549626f59bb1b713c5380ade231256989dedf25`.
- Model scores and weights were not changed when extending the ranking domain. Local verification reconstructed all global ranks from the three frozen seed score tables.

## Included source tables

| Manuscript content | Source |
| --- | --- |
| Run composition, Table S28 | GLOBAL_TOPK_STRATUM_COUNTS.csv; totals and support flags in the complete archived CSV |
| Global Top-20, Tables S29-S30 | GLOBAL_TOP20_PE_OFFICIAL.csv |
| Raw global Top-100 and per-model scores/ranks | GLOBAL_TOP100.csv |
| Rank sensitivity in Section 12.3 | GLOBAL_RANK_SENSITIVITY.csv |
| Ranking definitions and frozen channel weights | COLUMN_GUIDE.json |

The Top-20 PE/official table is a post-ranking join. Its BC columns use the frozen marginal comparisons also used for the six-stratum review. Same-run PE and official-stage fields come from the existing reviewed main-experiment audit. Cross-run marginal BCs come from the frozen common-grid public-PE comparison. Missing cross-run Dmax and official comparisons remain missing. No public posterior or official-stage field was used to construct the global ranking.

`global_rank` combines each model's ranks over all 24,531 pairs. `mean_score_global_rank` is an alternative ordering, not the primary rank. `original_stratum_rank` is retained only for reference. Unavailable official comparisons are not failures. The historic per-run empirical/GPD FPP values are not assigned to this global list.

The published September Zenodo archive predates this extension. The complete 24,531-pair ranking is included here as GLOBAL_ALL_24531_PAIRS_RANKED.csv, alongside the source tables used in the Supplementary Information.
