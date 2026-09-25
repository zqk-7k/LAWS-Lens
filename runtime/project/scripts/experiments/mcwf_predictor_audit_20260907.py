#!/usr/bin/env python3
"""Read-only external diagnostic; public PE is never a predictor input."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='2'
import argparse
from datetime import datetime,timezone
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import torch
import mcwf_pe_frontend_extension_v2_20260907 as e
import mcwf_mass_tf_20260905 as tf
import mcwf_detector_mass_density_20260907 as detector

dev,body=e.dev,e.body


def mass_moments(a,ck):
    w=np.asarray(a['w'],float);m=np.asarray(a['m'],float)[:,:,0]
    mean=np.sum(w*m,1)
    variance=np.sum(w*(np.asarray(a['cov'],float)[:,:,0,0]+m*m),1)-mean*mean
    return mean*ck['ys'][0]+ck['ym'][0],variance.clip(1e-9)*ck['ys'][0]**2


def run(root):
    out=root/'audit'/('predictor_information_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    out.mkdir(parents=True,exist_ok=False);summaries=[]
    for dep in e.DEPS:
        pe=pd.read_parquet(root/f'audit/{dep}_frozen_external_reference.parquet')
        for ms,es in zip(body.MODEL_SEEDS,dev.SEEDS):
            f=pd.read_parquet(e.BASE/f'evaluation/{dep}/seed_{es}/real_pairs.parquet')
            i,j=f.idx_i.to_numpy(int),f.idx_j.to_numpy(int)
            merged=f[['pair_key','idx_i','idx_j']].merge(pe[['pair_key','pe_mc_bhattacharyya_coefficient','pe_mc_standardized_distance']],on='pair_key',validate='one_to_one')
            if not np.array_equal(merged.idx_i,i):raise RuntimeError('Pair order changed in PE join')
            for family in ('FROZEN_RNC','mixture_density','joint_source_density','highermode_density','detector_mass_density'):
                if family=='FROZEN_RNC':
                    a=e.predictions(dep,ms,es,'real');p=np.asarray(a['p'],float)
                    mean=p@tf.LOG_CENTERS;var=(p@(tf.LOG_CENTERS**2)-mean*mean).clip(1e-9)
                else:
                    path=root/f'{family}/predictions/{dep}/model_{ms}_eval_{es}/real.npz'
                    if not path.exists():continue
                    a=np.load(path);cp=root/f'{family}/models/{dep}/seed_{ms}/validation_selected_model.pt'
                    ck=torch.load(cp,weights_only=False,map_location='cpu')
                    if str(a['checkpoint_sha256'])!=dev.sha(cp):raise RuntimeError('Prediction checkpoint hash mismatch')
                    if family=='detector_mass_density':
                        p=np.asarray(a['p'],float);centers=np.asarray(detector.CENTERS)
                        mean=p@centers;var=(p@(centers**2)-mean*mean).clip(1e-9)
                    else:mean,var=mass_moments(a,ck)
                difference=abs(mean[i]-mean[j]);distance=difference/np.sqrt(var[i]+var[j])
                v=merged.copy();v['predictor']=family;v['predicted_logMc_i']=mean[i];v['predicted_logMc_j']=mean[j]
                v['predictive_logMc_sigma_i']=np.sqrt(var[i]);v['predictive_logMc_sigma_j']=np.sqrt(var[j]);v['predictive_Dmc']=distance
                v.to_parquet(out/f'{dep}_{es}_{family}.parquet',index=False)
                bc=v.pe_mc_bhattacharyya_coefficient.to_numpy(float);d=v.pe_mc_standardized_distance.to_numpy(float)
                valid=np.isfinite(distance)&np.isfinite(bc)&np.isfinite(d)
                summaries.append({'deployment':dep,'seed':es,'predictor':family,'pairs':int(valid.sum()),
                    'rho_negative_predictive_D_vs_PE_BC':float(spearmanr(-distance[valid],bc[valid]).statistic),
                    'rho_predictive_D_vs_PE_D':float(spearmanr(distance[valid],d[valid]).statistic),
                    'rho_negative_absolute_mean_gap_vs_PE_BC':float(spearmanr(-difference[valid],bc[valid]).statistic),
                    'scope_events':len(np.unique(np.r_[i,j]))})
    dev.csv_write(out/'PREDICTOR_INFORMATION.csv',pd.DataFrame(summaries))
    dev.json_write(out/'CONTRACT.json',{'created_utc':datetime.now(timezone.utc).isoformat(),
        'purpose':'read-only adaptive external diagnostic; not independent validation or a new score',
        'pair_dependence':'shared events; no ordinary independent-pair p-values',
        'predicted_D':'learned waveform predictive density moments,not public PE posterior',
        'no_reranking_or_selection':True})
    print(json.dumps({'predictor_audit':str(out),'rows':summaries}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    run(parser.parse_args().root)
