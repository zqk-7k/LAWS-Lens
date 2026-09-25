#!/usr/bin/env python3
"""Direct-dot and time-grid checks using development validation only."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[k]='2'
import argparse
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
import mcwf_development_20260905 as dev
import mcwf_finelag_features_20260907 as fine

p=argparse.ArgumentParser()
p.add_argument('--root',type=Path,required=True)
a=p.parse_args()
out=a.root/'audit/fine_lag_numerics'
out.mkdir(parents=True,exist_ok=False)
rows=[]
tests=[]
for dep in ('gwtc3','gwtc4'):
    conf=json.loads((a.root/f'cache/finelag/{dep}/COMPLETE.json').read_text())
    bank=np.load(conf['bankpath'])
    raw=np.load(a.root/f'cache/{dep}/validation_raw2s.npy',mmap_mode='r')
    ids=np.random.default_rng(202609071).choice(len(raw),128,replace=False)
    x=np.asarray(raw[ids],dtype=np.float32)
    fields={}
    for step in (1,2,4,16):
        start=time.perf_counter()
        fields[step]=fine.features(x,bank,step=step,network_radius=21)
        rows.append({'deployment':dep,'quantity':'feature_runtime','lag_step_samples':step,'seconds':time.perf_counter()-start})
    for low,high in ((16,4),(4,2),(2,1)):
        delta=fields[high]-fields[low]
        if np.min(delta)<-1e-4:
            raise RuntimeError('Refined maximization lost an allowed coarse peak')
        rows.append({'deployment':dep,'quantity':'log1p_power_resolution_delta','coarse_step':low,'fine_step':high,
            'minimum':float(delta.min()),'median':float(np.median(delta)),'q90':float(np.quantile(delta,.9)),
            'q99':float(np.quantile(delta,.99)),'maximum':float(delta.max()),'monotone_refinement':True})
    shifts=np.roll(x,4,axis=-1)
    for step in (1,16):
        shifted=fine.features(shifts,bank,step=step,network_radius=21)
        delta=np.abs(shifted-fields[step])
        rows.append({'deployment':dep,'quantity':'four_sample_shift_sensitivity','lag_step_samples':step,
                     'median':float(np.median(delta)),'q90':float(np.quantile(delta,.9)),'q99':float(np.quantile(delta,.99)),'maximum':float(delta.max())})
    values=torch.as_tensor(x[:2],device='cuda')
    values=(values-values.mean(-1,keepdim=True))/values.std(-1,keepdim=True)
    kernels=torch.as_tensor(bank[[0,288,575]],device='cuda')
    corr=torch.fft.irfft(torch.fft.rfft(values[:,None,:,None,:],n=8192)*torch.fft.rfft(kernels[None],n=8192).conj(),n=8192).cpu().numpy()
    padded=np.pad(values.cpu().numpy().astype(float),((0,0),(0,0),(304,304)))
    absolute=[]
    scaled=[]
    for lag in (-304,-8,0,8,304):
        truth=np.einsum('bdn,tdqn->btdq',padded[...,lag+304:lag+304+4096],bank[[0,288,575]].astype(float))
        error=np.abs(corr[...,lag%8192]-truth)
        absolute.extend(error.ravel())
        scaled.extend((error/(1+np.abs(truth))).ravel())
    passed=max(scaled)<1e-4
    tests.append({'deployment':dep,'float32_fft_direct_dot_max_abs':max(absolute),'max_scaled_error':max(scaled),'pass':passed,
        'inputs':'128 deterministic development-validation events per run, no test or real PE','validation_indices':ids.tolist()})
    if not passed:
        raise RuntimeError('FFT precision validation failed')
dev.csv_write(out/'TIME_GRID_CONVERGENCE.csv',pd.DataFrame(rows))
dev.json_write(out/'NUMERICAL_TESTS.json',{'pass':all(t['pass'] for t in tests),'tests':tests,'float64':fine.numerical_test(),
    'interpretation':'Feature-level numerical verification, not evidence of retrieval or real PE performance.'})
print(json.dumps({'numeric_pass':all(t['pass'] for t in tests),'output':str(out)}),flush=True)
