#!/usr/bin/env python3
"""Descriptive validation geometry audit, not a new selection or acceptance gate."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='2'
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score
import mcwf_development_20260905 as dev
import mcwf_encoder_body_20260906 as body
import mcwf_rankncontrast_20260907 as rnc


def run(root):
    rows=[]
    for dep in ('gwtc3','gwtc4'):
        meta=pd.read_parquet(root/f'cache/{dep}/validation_metadata.parquet')
        mass=np.log(meta.chirp_mass_detector.to_numpy(float))
        ii,jj=np.triu_indices(len(meta),1)
        group=meta.waveform_parent_uid.to_numpy()
        y=group[ii]==group[jj]
        d=np.abs(mass[ii]-mass[jj])
        for seed in body.MODEL_SEEDS:
            for name,base in (('WARM_EVENT_PSD',rnc.WARM),('RNC',root)):
                path=base/f'models/RAW-PHASE-SOURCE/{dep}/seed_{seed}/development_validation.npz'
                a=np.load(path)
                z=np.asarray(a['embedding'],float)
                z=z/np.linalg.norm(z,axis=1,keepdims=True)
                score=np.sum(z[ii]*z[jj],axis=1)
                cov=np.cov(z,rowvar=False)
                eigen=np.linalg.eigvalsh(cov).clip(0)
                eigen=eigen/eigen.sum()
                erank=float(np.exp(-np.sum(eigen[eigen>0]*np.log(eigen[eigen>0]))))
                rows.append({'deployment':dep,'model_seed':seed,'variant':name,'events':len(meta),
                    'source_count':meta.waveform_parent_uid.nunique(),'embedding_effective_rank':erank,
                    'cosine_vs_negative_logMc_gap_spearman_noncompanion':float(spearmanr(score[~y],-d[~y]).statistic),
                    'source_companion_pair_average_precision':float(average_precision_score(y,score)),
                    'companion_cosine_median':float(np.median(score[y])),
                    'noncompanion_cosine_median':float(np.median(score[~y])),
                    'large_mass_gap_false_cosine_q99':float(np.quantile(score[(~y)&(d>.5)],.99)),
                    'feature_sha256':dev.sha(path)})
    dev.csv_write(root/'audit/VALIDATION_GEOMETRY_DESCRIPTIVE.csv',pd.DataFrame(rows))
    dev.json_write(root/'audit/GEOMETRY_INTERPRETATION.json',{
        'not_new_gate':True,'selection_already_frozen':'validation CE + .2 SupCon + .2 RNC',
        'data':'development validation, not independent confirmation',
        'scope':'simulated intrinsic-mass ordering and source retrieval; not real PE or lens detection'})
    print(pd.DataFrame(rows).to_json(orient='records'),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    run(p.parse_args().root)
