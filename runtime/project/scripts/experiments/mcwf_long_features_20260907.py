#!/usr/bin/env python3
"""Eight-second phase context alongside the untouched peak-two-second branch."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='2'
import argparse
from datetime import datetime,timezone
from pathlib import Path
import shutil
import time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import signal
import mcwf_pe_frontend_extension_v2_20260907 as e
import mcwf_adaptive_psd_encoder_20260906 as adaptive
import mcwf_finelag_eventpsd_20260907 as psd

dev,body=e.dev,e.body
torch.set_num_threads(2)
FS=2048;LENGTH=8*FS;NFFT=32768;LAGS=304


def initialize(root):
    out=root/'long_context'
    p=out/'contracts/PHASE_CONTEXT.json'
    if p.exists():return
    out.mkdir(parents=True,exist_ok=True)
    dev.json_write(p,{'created_utc':datetime.now(timezone.utc).isoformat(),
        'ablation':'retain current peak2s4096 raw branch AND2s template features; add8s context template correlations',
        'why':'some low-Mc inspirals extend outside2s; added context tests whether the observed mass ambiguity is an information-limit of the crop',
        'no_change_to_historical_input':'all prior results and2s caches are read-only; this is a separately named2plus8s exploration, not a silent replacement',
        'same_O3_O4a':True,'time_sky_outer_weights_scope_frozen':True,
        'template_bank':'same576 IMRPhenomD Mc/q/aligned-spin templates, event-specific offsourcePSD, phase quadratures;8s crop centered with merger0.25s before end',
        'lag_grid':{'range_samples':[-304,304],'step':1,'H1L1_radius':21},
        'feature_calculation':'linear FFT correlation with32768 zero padding, detector-wise input centering/SD scaling, normalized template quadratures',
        'new_storage':'features only;8s dense waveform batches temporary; reuse existing physical24s injections, no new training strain directory',
        'scientific_limit':'template-match features and learned predictions, not PE likelihoods',
        'reference':'https://pycbc.org/pycbc/latest/html/filter.html','unit_test':test_fft()})
    (out/'scripts').mkdir(exist_ok=True)
    shutil.copy2(__file__,out/'scripts'/Path(__file__).name)


def test_fft():
    rng=np.random.default_rng(202609076);x=rng.normal(size=LENGTH);h=rng.normal(size=LENGTH)
    c=np.fft.irfft(np.fft.rfft(x,NFFT)*np.fft.rfft(h,NFFT).conj(),NFFT)
    pad=np.pad(x,(LAGS,LAGS));lags=np.array([-304,-2,0,1,304])
    direct=np.array([pad[LAGS+k:LAGS+k+LENGTH]@h for k in lags])
    err=float(abs(direct-c[lags%NFFT]).max())
    if err>1e-9:raise RuntimeError('Long-context FFT/direct mismatch')
    return {'pass':True,'max_error':err}


@torch.no_grad()
def bank_for(h,f,psds):
    frequency=np.fft.rfftfreq(adaptive.N,1/FS)
    pp=np.array([np.interp(frequency,f,p) for p in psds])
    pp=np.maximum(pp,pp.max(axis=1,keepdims=True)*1e-20)
    sos=signal.butter(6,[40,580],fs=FS,btype='bandpass',output='sos')
    _,response=signal.sosfreqz(sos,worN=frequency,fs=FS)
    spectral=torch.as_tensor(h,device='cuda')[:,None,:]*torch.as_tensor(abs(response)**2/np.sqrt(pp),dtype=torch.float32,device='cuda')[None]
    white=torch.fft.irfft(spectral,n=adaptive.N,dim=-1)
    filt=torch.zeros(adaptive.N,device='cuda');filt[0]=filt[adaptive.N//2]=1;filt[1:adaptive.N//2]=2
    analytic=torch.fft.ifft(torch.fft.fft(white,dim=-1)*filt,dim=-1)
    offset=14*FS-int(7.75*FS)
    window=torch.as_tensor(signal.windows.tukey(LENGTH,.05),dtype=torch.float32,device='cuda')
    u=analytic.real[...,offset:offset+LENGTH]*window;v=analytic.imag[...,offset:offset+LENGTH]*window
    u-=u.mean(-1,keepdim=True);v-=v.mean(-1,keepdim=True)
    u=F.normalize(u,dim=-1);v-=(v*u).sum(-1,keepdim=True)*u;v=F.normalize(v,dim=-1)
    bank=torch.stack([u,v],2)
    gram=max(float(((bank*bank).sum(-1)-1).abs().max()),float((u*v).sum(-1).abs().max()))
    if gram>1e-4 or not torch.isfinite(bank).all():raise RuntimeError('Long quadrature normalization failure')
    return bank


@torch.no_grad()
def features(x,bank,batch=6):
    kernels=torch.fft.rfft(bank,n=NFFT)
    ix=torch.arange(-LAGS,LAGS+1,device='cuda').remainder(NFFT)
    output=[]
    for start in range(0,len(x),batch):
        values=torch.as_tensor(np.array(x[start:start+batch],dtype=np.float32,copy=True),device='cuda')
        if values.shape[1:]!=(2,LENGTH) or not torch.isfinite(values).all():raise RuntimeError('Invalid8s waveform')
        values=(values-values.mean(-1,keepdim=True))/values.std(-1,keepdim=True).clamp_min(1e-6)
        data=torch.fft.rfft(values,n=NFFT);power=[]
        for d in range(2):
            blocks=[]
            for k in range(0,len(bank),64):
                product=data[:,d,None,None,:]*kernels[None,k:k+64,d].conj()
                corr=torch.fft.irfft(product,n=NFFT)[...,ix]
                blocks.append(corr.square().sum(2))
            power.append(torch.cat(blocks,1))
        nearby=F.max_pool1d(power[1],43,stride=1,padding=21)
        net=(power[0]+nearby).amax(-1)
        f=torch.stack([p.amax(-1) for p in power]+[net],1)
        output.append(torch.log1p(f).reshape(len(values),3,64,3,3).cpu().numpy())
    return np.concatenate(output)


def grouped(root,raw,frequency,psds,ids,path):
    if path.exists():
        f=np.load(path)
        if len(f)!=len(raw):raise RuntimeError('Mismatched cached rows')
        return f
    path.parent.mkdir(parents=True,exist_ok=True)
    h=np.load(e.TRAINED/'cache/adaptive_psd/unwhitened_aligned_template_spectra.npy')
    result=np.empty((len(raw),3,64,3,3),np.float32);audit=[]
    for pid in np.unique(ids):
        keep=np.flatnonzero(ids==pid);start=time.perf_counter();bank=bank_for(h,frequency,psds[int(pid)])
        result[keep]=features(raw[keep],bank);del bank
        audit.append({'PSD_index':int(pid),'events':len(keep),'seconds':time.perf_counter()-start})
    if not np.isfinite(result).all():raise RuntimeError('Nonfinite features')
    np.save(path,result);dev.csv_write(path.with_suffix('.audit.csv'),pd.DataFrame(audit))
    dev.json_write(path.with_suffix('.sha256.json'),{'sha256':dev.sha(path),'event_count':len(raw),'seconds':sum(a['seconds'] for a in audit)})
    return result


def development(root,dep,split):
    initialize(root)
    path=root/f'long_context/features/{dep}/{split}.npy'
    if path.exists():return
    meta=pd.read_parquet(e.TRAINED/f'cache/{dep}/{split}_metadata.parquet')
    raw=np.empty((len(meta),2,LENGTH),np.float32)
    for family in ('sis','pm'):
        for image in ('a','b'):
            keep=(meta.family_slot==family)&(meta.image==image)
            src=np.load(dev.OLD/f'data/development/{dep}/{family}_{split}_{image}.npy',mmap_mode='r')
            raw[keep]=src[meta.loc[keep,'row_index'].to_numpy(int),:,-LENGTH:]
    noise=dev.OLD/f'data/noise_banks/{dep}'
    ids=np.where(meta.image.eq('a'),meta.a_noise_bank_index,meta.b_noise_bank_index).astype(int)
    grouped(root,raw,np.load(noise/f'noise_psd_frequency_{split}.npy'),np.load(noise/f'noise_psd_{split}.npy',mmap_mode='r'),ids,path)
    print({'long_development':dep,'split':split,'events':len(raw),'sha256':dev.sha(path)},flush=True)


def deployment(root,dep,es,split):
    initialize(root)
    path=root/f'long_context/features/{dep}/{es}_{split}.npy'
    if split=='real':path=root/f'long_context/features/{dep}/real.npy'
    if path.exists():return np.load(path)
    f,psds,ids,_=psd.psd_context(dep,es,split)
    if split=='real':
        full,events=dev.real_inputs(dep);valid=events.strict_h1l1_preprocessing_pass.to_numpy(bool);raw=np.asarray(full[valid,:,-LENGTH:])
    else:
        events=dev.BASE.retained_event_plan(dep,es,split)
        full=dev.ORCH.event_array_for_plan(dep,es,split,events);raw=np.asarray(full[...,-LENGTH:])
    return grouped(root,raw,f,psds,ids,path)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--split',choices=['validation','train'],required=True);a=p.parse_args()
    for d in e.DEPS:development(a.root,d,a.split)
