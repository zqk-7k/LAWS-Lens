from pathlib import Path
import json
import pandas as pd
import numpy as np
import mcwf_unified_waveform_20260906 as u
import mcwf_development_20260905 as dev
import mcwf_encoder_body_20260906 as body

root = Path('/root/autodl-tmp/gw-catalog/results/mcwf_unified_v3_20260906T234500Z')
for dep in u.DEPS:
    met = pd.read_csv(root / 'tables/RETRIEVAL_PER_SEED.csv')
    met = met.loc[(met.deployment == dep) & (met.split == 'test')]
    keys = ['macro_r_at_10', 'average_precision', 'false_at_recall_0p5', 'false_at_recall_0p9']
    print(dep, met.pivot(index=['seed','method'], columns='config', values=keys).to_string())
    top = pd.read_parquet(root / f'results/UNIFIED/{dep}/consensus_C_fixed_all_pairs_pe_official.parquet').head(10)
    print(top[['pair_key','pe_mc_bhattacharyya_coefficient','pe_mc_standardized_distance']].to_string(index=False))
    bad = top.loc[(top.pe_mc_bhattacharyya_coefficient < .1) | (top.pe_mc_standardized_distance > 5), 'pair_key']
    for ms, es in zip(body.MODEL_SEEDS,dev.SEEDS):
        f = pd.read_parquet(root / f'evaluation/{dep}/seed_{es}/real_pairs.parquet')
        cols = ['pair_key','previous_waveform_score','waveform_score','new_calibrated_waveform_score','new_encoder_cosine','new_mass_pred_logmc_i','new_mass_pred_logmc_j']
        print(es, f.loc[f.pair_key.isin(bad),cols].to_string(index=False))

alltab = pd.read_csv(u.PREV / 'tables/ALL_ABLATIONS_RECALL_PE_OFFICIAL_LEDGER.csv')
print(alltab.loc[(alltab.deployment=='gwtc4') & (alltab.method=='C_fixed'), ['config','rule','macro_r_at_10_mean','catastrophic_mc','BC_mc_ge_0p5','Dmax_le_3']].to_string(index=False))
