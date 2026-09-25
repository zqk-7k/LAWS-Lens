#!/usr/bin/env python3
"""Audit whether the fixed phasebank lag support truncates coherent peaks."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[k]='2'
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.signal import hilbert
import mcwf_development_20260905 as dev
import mcwf_encoder_body_20260906 as body
import mcwf_finelag_eventpsd_20260907 as psd
import mcwf_adaptive_psd_encoder_20260906 as adaptive
import mcwf_mass_tf_20260905 as tf

ROOT=dev.PROJECT/'results/mcwf_lag_support_audit_20260907'
TRAINED=dev.PROJECT/'results/mcwf_unified_finelag_eventpsd_encoder_20260907'


@torch.no_grad()
def audit(x,bank):
    x=torch.as_tensor(x,dtype=torch.float32,device='cuda')
    x=(x-x.mean(-1,keepdim=True))/x.std(-1,keepdim=True).clamp_min(1e-6)
    data=torch.fft.rfft(x,n=8192)
    kernels=torch.fft.rfft(torch.as_tensor(bank,device='cuda'),n=8192)
    lags=torch.arange(-1024,1025,device='cuda')
    indices=lags.remainder(8192)
    power=[]
    for d in range(2):
        correlation=torch.fft.irfft(data[d,None,None,:]*kernels[:,d].conj(),n=8192)[...,indices]
        power.append(correlation.square().sum(1))
    network=power[0]+F.max_pool1d(power[1][:,None],43,stride=1,padding=21)[:,0]
    outputs={}
    for half in (304,512,1024):
        sel=lags.abs()<=half
        values,where=network[:,sel].max(-1)
        best=int(values.argmax())
        lag=int(lags[sel][where[best]])
        outputs.update({f'peak_{half}':float(values[best]),f'lag_{half}':lag,
                        f'mc_{half}':float(np.exp(tf.LOG_CENTERS[best//9]))})
    return outputs


ROOT.mkdir(parents=True,exist_ok=False)
spectra=adaptive.spectra(TRAINED)
rows=[]
for dep in ('gwtc3','gwtc4'):
    meta=pd.read_parquet(TRAINED/f'cache/{dep}/validation_metadata.parquet')
    raw=np.load(TRAINED/f'cache/{dep}/validation_raw2s.npy',mmap_mode='r')
    noise=dev.OLD/f'data/noise_banks/{dep}'
    freq=np.load(noise/'noise_psd_frequency_validation.npy')
    bank=np.load(noise/'noise_psd_validation.npy',mmap_mode='r')
    selection=np.random.default_rng(202609072).choice(len(meta),64,replace=False)
    for idx in selection:
        event=meta.iloc[idx]
        bid=int(event.a_noise_bank_index if event.image=='a' else event.b_noise_bank_index)
        templates=adaptive.whitened_bank(spectra,freq,bank[bid])
        result=audit(np.asarray(raw[idx]),templates)
        rows.append({'deployment':dep,'split':'development_validation','event_index':int(idx),
            'true_mc':float(event.chirp_mass_detector),'noise_index':bid,**result})
    es=202607242 if dep=='gwtc3' else 202607243
    plan=dev.BASE.retained_event_plan(dep,es,'test')
    values=dev.ORCH.event_array_for_plan(dep,es,'test',plan)
    values=dev.TRAIN.make_window_view(values,2)
    freq,bank,ids,_=psd.psd_context(dep,es,'test')
    for idx in ((6,41) if dep=='gwtc3' else (70,105)):
        templates=adaptive.whitened_bank(spectra,freq,bank[ids[idx]])
        result=audit(values[idx],templates)
        rows.append({'deployment':dep,'split':'already_examined_failure_diagnostic','event_index':idx,
                     'true_mc':None,'noise_index':int(ids[idx]),**result})
    print(dep,flush=True)
f=pd.DataFrame(rows)
dev.csv_write(ROOT/'LAG_SUPPORT.csv',f)
summary=[]
for dep,g in f[f.split.eq('development_validation')].groupby('deployment'):
    summary.append({'deployment':dep,'events':len(g),
        'peak_gain_above_10pct_fraction':float((g.peak_1024>1.1*g.peak_304).mean()),
        'default_best_near_boundary_fraction':float((g.lag_304.abs()>=290).mean()),
        'absolute_logmass_error_median_304':float(np.median(abs(np.log(g.mc_304/g.true_mc)))),
        'absolute_logmass_error_median_1024':float(np.median(abs(np.log(g.mc_1024/g.true_mc))))})
dev.json_write(ROOT/'AUDIT.json',{'validation':summary,'only_diagnostic':True,
    'failure_examples_are_not_blind_test':True,'feature_bank_is_not_PE':True})
print(json.dumps(summary),flush=True)
