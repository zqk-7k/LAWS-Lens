#!/usr/bin/env python3
"""Inspect reused development true-pair failures, not unseen confirmation."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import mcwf_development_20260905 as dev
import mcwf_encoder_evaluate_20260906 as ev

ROOT=dev.PROJECT/'results/mcwf_unified_cross_domain_tail_20260906'
rows=[]
for dep,es in (('gwtc3',202607243),('gwtc4',202607241),('gwtc4',202607243)):
    f=pd.read_parquet(ROOT/f'evaluation/{dep}/seed_{es}/test_pairs.parquet').reset_index(drop=True)
    plan=dev.BASE.retained_event_plan(dep,es,'test')
    print(dep,es,'plan columns',list(plan),flush=True)
    frozen=dev.BASE.FROZEN_V93_WEIGHTS[dep][es]
    extra=frozen['time']*f.time_score.to_numpy()+frozen['sky']*f.sky_raw_log_bf.to_numpy()
    f['baseline_combined']=frozen['waveform']*f.previous_waveform_score+extra
    f['candidate_combined']=frozen['waveform']*f.waveform_score+extra
    f['waveform_delta']=f.waveform_score-f.previous_waveform_score
    mask=f.is_true_pair.to_numpy(dtype=bool)
    f['candidate_true_rank']=f.loc[mask,'candidate_combined'].rank(ascending=False,method='first')
    f['baseline_true_rank']=f.loc[mask,'baseline_combined'].rank(ascending=False,method='first')
    selected=f.loc[mask].nsmallest(8,'waveform_delta')
    for r in selected.to_dict('records'):
        record={'deployment':dep,'seed':es,**r}
        for side in ('i','j'):
            e=plan.iloc[int(r['idx_'+side])]
            for c,v in e.items():
                if any(k in c.lower() for k in ('mass','snr','system','noise','source','duration')):
                    record[side+'_'+c]=v
        rows.append(record)
        if r['waveform_delta']<-.05:
            print(json.dumps({k:v for k,v in record.items() if k in ('deployment','seed','idx_i','idx_j','waveform_delta','new_mass_predictive_BC','new_mass_pred_logmc_i','new_mass_pred_logmc_j','i_snr','j_snr','i_source_index','j_source_index')},default=str),flush=True)
    print(selected[['idx_i','idx_j','previous_waveform_score','waveform_score','waveform_delta','new_mass_predictive_BC','new_mass_pred_logmc_i','new_mass_pred_logmc_j','baseline_true_rank','candidate_true_rank']].to_string(index=False),flush=True)
out=ROOT/'audit/REUSED_TRUE_PAIR_PENALTY_DIAGNOSIS.csv'
dev.csv_write(out,pd.DataFrame(rows))
print('saved',out)
