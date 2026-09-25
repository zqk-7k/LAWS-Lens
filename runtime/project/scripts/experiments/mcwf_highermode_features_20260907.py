#!/usr/bin/env python3
"""Higher-mode/precession waveform features, preserving the two-second input."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time
import numpy as np
import pandas as pd
from scipy import signal
from pycbc.waveform import get_fd_waveform
import mcwf_pe_frontend_extension_v2_20260907 as e
import mcwf_dense_features_20260907 as dense
import mcwf_adaptive_psd_encoder_20260906 as adaptive
import mcwf_finelag_eventpsd_20260907 as psd
import mcwf_mass_tf_20260905 as tf

dev = e.dev
PARAMETERS = [(float(mc), q, z, x, inclination)
    for mc in np.exp(tf.LOG_CENTERS) for q in (.25, .5, 1.)
    for z in (-.5, 0., .5) for x in (0., .4) for inclination in (.4, 1.2)]


def initialize(root):
    out = root / 'highermode_context'
    path = out / 'contracts/FEATURES.json'
    if path.exists():
        return
    dev.json_write(path, {'created_utc': datetime.now(timezone.utc).isoformat(),
        'same_O3_O4a': True, 'input': 'unchanged H1L1 peak2s4096points at2048Hz,physical40-580Hz',
        'waveform': 'IMRPhenomXPHM via PyCBC/LALSimulation,16s construction,f_lower=f_ref=30Hz,f_final580Hz',
        'mass_centers': np.exp(tf.LOG_CENTERS).tolist(), 'q': [.25,.5,1.],
        'spin1z_spin2z': [-.5,0.,.5], 'spin1x': [0.,.4], 'spin2x': 0.,
        'spin1y_spin2y': 0., 'inclination': [.4,1.2], 'projection': ['h_plus','h_cross'],
        'waveforms': len(PARAMETERS), 'projection_templates': 2*len(PARAMETERS),
        'alignment': 'same shift for plus/cross using joint analytic-signal envelope peak',
        'matching': 'frozen per-event PSD whitening; Hilbert quadrature powers; fine FFT lag grid and network lag radius unchanged',
        'limitation': 'Hilbert quadrature is a feature construction,not exact coalescence-phase marginalization for multimode signals; no PE/evidence claim',
        'storage': 'spectra once,features once per event; no full strain archive',
        'frozen': ['time','sky','outer weights','scope','historical outputs'],
        'references': ['https://arxiv.org/abs/2004.06503','https://pycbc.org/pycbc/latest/html/waveform.html'],
        'tests': psd.fine.numerical_test()})
    (out/'scripts').mkdir(exist_ok=True)
    shutil.copy2(__file__, out/'scripts'/Path(__file__).name)


def spectra(root):
    initialize(root)
    path = root/'highermode_context/cache/spectra.npy'
    if path.exists():
        info = json.loads(path.with_suffix('.json').read_text())
        if dev.sha(path) != info['sha256']:
            raise RuntimeError('Changed higher-mode spectral bank')
        return np.load(path, mmap_mode='r')
    path.parent.mkdir(parents=True, exist_ok=True)
    bank, rows = [], []
    started = time.perf_counter()
    for k, (mc,q,z,x,inc) in enumerate(PARAMETERS):
        m1 = mc*(1+q)**.2/q**.6
        hp,hc = get_fd_waveform(approximant='IMRPhenomXPHM',mass1=m1,mass2=m1*q,
            spin1x=x,spin1y=0,spin1z=z,spin2x=0,spin2y=0,spin2z=z,
            delta_f=1/16,f_lower=30,f_ref=30,f_final=580,distance=1000,inclination=inc)
        hp.resize(adaptive.N//2+1);hc.resize(adaptive.N//2+1)
        h = np.stack([np.fft.irfft(np.asarray(a), n=adaptive.N) for a in (hp,hc)])
        peak = np.argmax(np.sum(np.abs(signal.hilbert(h,axis=-1))**2,axis=0))
        h = np.roll(h,14*2048-int(peak),axis=-1)
        for p,name in enumerate(('plus','cross')):
            if not np.isfinite(h[p]).all() or np.linalg.norm(h[p]) == 0:
                raise RuntimeError('Invalid multimode waveform')
            bank.append(np.fft.rfft(h[p]).astype(np.complex64))
            rows.append({'mc':mc,'q':q,'spin_z':z,'spin1x':x,'inclination':inc,'polarization':name})
        if (k+1)%256==0:
            print(json.dumps({'XPHM_waveforms':k+1,'total':len(PARAMETERS),'seconds':time.perf_counter()-started}),flush=True)
    np.save(path,np.stack(bank))
    dev.csv_write(path.with_suffix('.parameters.csv'),pd.DataFrame(rows))
    dev.json_write(path.with_suffix('.json'),{'sha256':dev.sha(path),'projection_templates':len(bank),'seconds':time.perf_counter()-started})
    return np.load(path,mmap_mode='r')


def grouped(root,raw,freq,psds,ids,path):
    if path.exists():
        a=np.load(path,mmap_mode='r')
        if len(a)!=len(raw):
            raise RuntimeError('Feature row count changed')
        return a
    h=spectra(root);path.parent.mkdir(parents=True,exist_ok=True)
    a=np.empty((len(raw),3,len(h)),np.float32);rows=[]
    for pid in np.unique(ids):
        take=np.flatnonzero(ids==pid);started=time.perf_counter()
        bank=adaptive.whitened_bank(h,freq,psds[int(pid)])
        gram=max(float(abs((bank*bank).sum(-1)-1).max()),float(abs((bank[:,:,0]*bank[:,:,1]).sum(-1)).max()))
        if gram>1e-4:
            raise RuntimeError('Invalid multimode quadratures')
        a[take]=dense.features(raw[take],bank)
        del bank
        rows.append({'PSD':int(pid),'events':len(take),'seconds':time.perf_counter()-started,'gram_error':gram})
    if not np.isfinite(a).all():
        raise RuntimeError('Nonfinite higher-mode features')
    np.save(path,a)
    dev.csv_write(path.with_suffix('.audit.csv'),pd.DataFrame(rows))
    dev.json_write(path.with_suffix('.sha256.json'),{'sha256':dev.sha(path),'events':len(a),'seconds':sum(r['seconds'] for r in rows)})
    print(json.dumps({'highermode_features':str(path),'events':len(a),'seconds':sum(r['seconds'] for r in rows)}),flush=True)
    return a


def development(root,dep,split):
    noise=root/f'expanded_data/{dep}/noise';src=root/f'expanded_data/{dep}/{split}'
    if not (src/'COMPLETE.json').exists():
        raise RuntimeError('Expanded data incomplete')
    meta=pd.read_parquet(src/'event_metadata.parquet')
    return grouped(root,np.load(src/'raw2s.npy',mmap_mode='r'),np.load(noise/'frequency.npy'),
        np.load(noise/'psd.npy',mmap_mode='r'),meta.noise_bank_index.to_numpy(int),
        root/f'highermode_context/features/{dep}/{split}.npy')


def deployment(root,dep,es,split):
    path=root/f'highermode_context/features/{dep}/{es}_{split}.npy'
    if split=='real':
        path=root/f'highermode_context/features/{dep}/real.npy'
    if path.exists():
        return np.load(path,mmap_mode='r')
    freq,psds,ids,_=psd.psd_context(dep,es,split)
    if split=='real':
        full,events=dev.real_inputs(dep)
        raw=np.asarray(full[events.strict_h1l1_preprocessing_pass.to_numpy(bool),:,-4096:])
    else:
        plan=dev.BASE.retained_event_plan(dep,es,split)
        raw=np.asarray(dev.ORCH.event_array_for_plan(dep,es,split,plan)[...,-4096:])
    return grouped(root,raw,freq,psds,ids,path)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args()
    for dep in e.DEPS:
        for split in ('validation','train'):
            development(args.root,dep,split)
