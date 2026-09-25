#!/usr/bin/env python3
"""Validation-only audit of agreement as a mass-prediction reliability diagnostic."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='2'
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import spearmanr
import torch
import mcwf_development_20260905 as dev
import mcwf_encoder_body_20260906 as body
import mcwf_mass_tf_20260905 as tf
import mcwf_physical_mass_anchor_20260907 as anchor


def run(trained,root):
    if root.exists():
        raise RuntimeError('Independent audit directory required')
    root.mkdir(parents=True)
    rows=[]
    for dep in ('gwtc3','gwtc4'):
        x=np.load(trained/f'cache/finelag/{dep}/validation_features.npy')
        meta=pd.read_parquet(trained/f'cache/{dep}/validation_metadata.parquet')
        truth=np.log(meta.chirp_mass_detector.to_numpy(float))
        for ms in body.MODEL_SEEDS:
            folder=trained/f'models/RAW-PHASE-SOURCE/{dep}/seed_{ms}'
            ck=torch.load(folder/'validation_selected_model.pt',map_location='cpu',weights_only=False)
            a=np.load(folder/'development_validation.npz')
            logits=a['logits'].astype(float)/ck['temperature']
            neural=np.exp(logits-logsumexp(logits,axis=-1,keepdims=True))
            y=a['truth']
            grid=[]
            for t in anchor.GRID:
                p=anchor.profile(x,t)
                ce=float(-(y*np.log(p.clip(1e-30))).sum(-1).mean())
                grid.append({'temperature':t,'physical_validation_ce':ce})
            t=min(grid,key=lambda r:(r['physical_validation_ce'],r['temperature']))['temperature']
            physical=anchor.profile(x,t)
            agreement=np.sqrt(neural*physical).sum(-1)
            error=np.abs(neural@tf.LOG_CENTERS-truth)
            table=pd.DataFrame({'source':meta.waveform_parent_uid,'agreement':agreement,'absolute_logMc_error':error,
                'neural_mean':neural@tf.LOG_CENTERS,'physical_mean':physical@tf.LOG_CENTERS,'true_logMc':truth})
            dev.csv_write(root/f'{dep}_{ms}_validation_events.csv',table)
            dev.csv_write(root/f'{dep}_{ms}_temperature_grid.csv',pd.DataFrame(grid))
            q=np.quantile(agreement,[.25,.5,.75])
            bins=np.searchsorted(q,agreement,side='right')
            for k in range(4):
                keep=bins==k
                rows.append({'deployment':dep,'seed':ms,'agreement_quartile':k,'events':int(keep.sum()),
                    'physical_temperature':t,'mean_agreement':float(agreement[keep].mean()),
                    'mass_MAE':float(error[keep].mean()),'error_gt_0p5':float((error[keep]>.5).mean()),
                    'agreement_vs_negative_error_spearman':float(spearmanr(agreement,-error).statistic)})
    dev.csv_write(root/'VALIDATION_AGREEMENT_RELIABILITY.csv',pd.DataFrame(rows))
    dev.json_write(root/'CONTRACT.json',{'role':'validation-only mechanism diagnostic; no candidate rescoring',
        'trained_root':str(trained),'temperature_selection':'pure physical predictive CE on simulation validation',
        'agreement':'Bhattacharyya coefficient between neural and phase-profile mass predictions',
        'not_probability_of_correctness':True,'real_PE_or_test_read':False})
    print(pd.DataFrame(rows).to_json(orient='records'),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--trained-root',type=Path,required=True);p.add_argument('--root',type=Path,required=True)
    a=p.parse_args();run(a.trained_root,a.root)
