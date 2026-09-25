#!/usr/bin/env python3
"""Finer intrinsic waveform template grid without changing the 2s input."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='2'
import argparse
from datetime import datetime,timezone
from pathlib import Path
import shutil
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import signal
from pycbc.waveform import get_fd_waveform
import mcwf_pe_frontend_extension_v2_20260907 as e
import mcwf_adaptive_psd_encoder_20260906 as adaptive
import mcwf_finelag_eventpsd_20260907 as psd
import mcwf_mass_tf_20260905 as tf

dev,body=e.dev,e.body
torch.set_num_threads(2)
QGRID=(.125,.25,.375,.5,.625,.75,.875,1.)
SPINGRID=(-.75,-.5,-.25,0.,.25,.5,.75)
EXTRA=[(mc,q,s) for mc in np.exp(tf.LOG_CENTERS) for q in QGRID for s in SPINGRID
       if not(q in (.25,.5,1.) and s in(-.5,0.,.5))]
NFFT=8192;LAGS=304


def initialize(root):
    out=root/'dense_context';path=out/'contracts/FEATURES.json'
    if path.exists():return
    out.mkdir(parents=True,exist_ok=True)
    dev.json_write(path,{'created_utc':datetime.now(timezone.utc).isoformat(),
        'same_both_runs':True,'input':'unchanged H1L1 peak2s4096points,2048Hz,40-580Hz',
        'coarse_features':'original576templates, three detector/network profiles, unchanged caches',
        'extra_templates':len(EXTRA),'q_grid':QGRID,'aligned_equal_spin_grid':SPINGRID,
        'waveform':'IMRPhenomD,16s construction at30Hz;crop2s and physical40-580Hz band before matching',
        'features':'per-event PSD whitened orthonormal phase quadratures, zero-padded8192FFT, lag+-304samples step1, inter-detector lag radius21',
        'rationale':'test loss of intrinsic discrimination due to coarse q/spin grid; no new source parameters or real PE inputs',
        'output_order':'concatenate original1728 features then extra detector/template features; original network can be embedded exactly with new weights0',
        'storage':'extra unwhitened spectral bank once; perPSD dense filters temporary; event features cached once per split, no new strain bank',
        'not_PE':'coarse/dense match features have no claimed PE coverage, normalized evidence or minimal-match guarantee',
        'frozen':['time','sky','old encoders','outerC-fixedweights','scope','historical outputs'],
        'reference':'https://pycbc.org/pycbc/latest/html/pycbc.filter.html',
        'fft_unit_test':psd.fine.numerical_test()})
    (out/'scripts').mkdir(exist_ok=True);shutil.copy2(__file__,out/'scripts'/Path(__file__).name)


def spectra(root):
    initialize(root);path=root/'dense_context/cache/extra_spectra.npy'
    if path.exists():return np.load(path,mmap_mode='r')
    path.parent.mkdir(parents=True,exist_ok=True);bank=[]
    for k,(mc,q,spin) in enumerate(EXTRA):
        m1=mc*(1+q)**.2/q**.6
        hp,_=get_fd_waveform(approximant='IMRPhenomD',mass1=m1,mass2=m1*q,spin1z=spin,spin2z=spin,
                            delta_f=1/16,f_lower=30,f_final=580,distance=1000,inclination=0)
        hp.resize(adaptive.N//2+1);h=np.fft.irfft(np.asarray(hp),n=adaptive.N)
        peak=np.argmax(abs(signal.hilbert(h)));h=np.roll(h,14*2048-peak)
        bank.append(np.fft.rfft(h).astype(np.complex64))
    arr=np.stack(bank);np.save(path,arr)
    dev.json_write(path.with_suffix('.json'),{'sha256':dev.sha(path),'templates':len(arr),'parameters':[[float(v) for v in row] for row in EXTRA]})
    return np.load(path,mmap_mode='r')


@torch.no_grad()
def features(x,bank,batch=8):
    templates=torch.as_tensor(bank,dtype=torch.float32,device='cuda');kernels=torch.fft.rfft(templates,n=NFFT)
    indices=torch.arange(-LAGS,LAGS+1,device='cuda').remainder(NFFT);output=[]
    for start in range(0,len(x),batch):
        values=torch.as_tensor(np.asarray(x[start:start+batch,:,-4096:],np.float32),device='cuda')
        if values.shape[1:]!=(2,4096) or not torch.isfinite(values).all():raise RuntimeError('Invalid2s input')
        values=(values-values.mean(-1,keepdim=True))/values.std(-1,keepdim=True).clamp_min(1e-6);data=torch.fft.rfft(values,n=NFFT);power=[]
        for d in range(2):
            chunks=[]
            for k in range(0,len(bank),64):
                c=torch.fft.irfft(data[:,d,None,None,:]*kernels[None,k:k+64,d].conj(),n=NFFT)[...,indices]
                chunks.append(c.square().sum(2))
            power.append(torch.cat(chunks,1))
        net=(power[0]+F.max_pool1d(power[1],43,stride=1,padding=21)).amax(-1)
        output.append(torch.log1p(torch.stack([p.amax(-1) for p in power]+[net],1)).cpu().numpy())
    return np.concatenate(output)


def grouped(root,raw,freq,psds,ids,path):
    if path.exists():
        f=np.load(path)
        if len(f)!=len(raw):raise RuntimeError('Changed feature row count')
        return f
    path.parent.mkdir(parents=True,exist_ok=True);h=spectra(root);arr=np.empty((len(raw),3,len(EXTRA)),np.float32);rows=[]
    for pid in np.unique(ids):
        take=np.flatnonzero(ids==pid);tick=time.perf_counter();bank=adaptive.whitened_bank(h,freq,psds[int(pid)])
        gram=max(float(abs((bank*bank).sum(-1)-1).max()),float(abs((bank[:,:,0]*bank[:,:,1]).sum(-1)).max()))
        if gram>1e-4:raise RuntimeError('Invalid quadrature normalization')
        arr[take]=features(raw[take],bank);del bank
        rows.append({'PSD':int(pid),'events':len(take),'seconds':time.perf_counter()-tick,'gram_error':gram})
    if not np.isfinite(arr).all():raise RuntimeError('Nonfinite dense features')
    np.save(path,arr);dev.csv_write(path.with_suffix('.audit.csv'),pd.DataFrame(rows));dev.json_write(path.with_suffix('.sha256.json'),{'sha256':dev.sha(path),'events':len(raw),'seconds':sum(r['seconds'] for r in rows)})
    print(json.dumps({'dense_features':str(path),'events':len(raw),'seconds':sum(r['seconds'] for r in rows)}),flush=True)
    return arr


def development(root,dep,split):
    initialize(root);path=root/f'dense_context/features/{dep}/{split}.npy'
    if path.exists():return np.load(path,mmap_mode='r')
    raw=np.load(e.TRAINED/f'cache/{dep}/{split}_raw2s.npy',mmap_mode='r');meta=pd.read_parquet(e.TRAINED/f'cache/{dep}/{split}_metadata.parquet')
    ids=np.where(meta.image.eq('a'),meta.a_noise_bank_index,meta.b_noise_bank_index).astype(int);noise=dev.OLD/f'data/noise_banks/{dep}'
    return grouped(root,raw,np.load(noise/f'noise_psd_frequency_{split}.npy'),np.load(noise/f'noise_psd_{split}.npy',mmap_mode='r'),ids,path)


def deployment(root,dep,es,split):
    path=root/f'dense_context/features/{dep}/{es}_{split}.npy'
    if split=='real':path=root/f'dense_context/features/{dep}/real.npy'
    if path.exists():return np.load(path,mmap_mode='r')
    freq,psds,ids,_=psd.psd_context(dep,es,split)
    if split=='real':
        full,events=dev.real_inputs(dep);raw=np.asarray(full[events.strict_h1l1_preprocessing_pass.to_numpy(bool),:,-4096:])
    else:
        plan=dev.BASE.retained_event_plan(dep,es,split);full=dev.ORCH.event_array_for_plan(dep,es,split,plan);raw=np.asarray(full[...,-4096:])
    return grouped(root,raw,freq,psds,ids,path)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    for dep in e.DEPS:
        for split in ('validation','train'):development(a.root,dep,split)
