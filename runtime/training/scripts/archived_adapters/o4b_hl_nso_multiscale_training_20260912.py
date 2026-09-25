#!/usr/bin/env python3
"""O4b adapters for the archived ordered-mass, multirate and conditional models.

Only waveform development inputs are read. No historical evaluation or real
catalog function is called. Module implementations are snapshotted and hashed.
"""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='2'
import argparse
import fcntl
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import pandas as pd
import torch

import o4b_hl_nso_data_training_20260912 as support

P=Path('/root/autodl-tmp/gw-catalog')
SEEDS=(2026091221,2026091222,2026091223)


def modules():
    sys.path.insert(0,str(P/'scripts/experiments'))
    import mcwf_multirate_features_20260908 as low
    import mcwf_multirate_train_20260908 as mult
    import mcwf_conditional_intrinsics_20260908 as cond
    return low,mult,cond


def initialize(root):
    low,mult,cond=modules()
    out=root/'waveform_feature_operator'
    for d in ('cache','contracts','tables'): (out/d).mkdir(parents=True,exist_ok=True)
    contract=root/'contracts/MULTISCALE_TRAINING_CONTRACT.json'
    records=[];snapshot=root/'scripts/archived_waveform_dependencies';snapshot.mkdir(exist_ok=True)
    for module in list(sys.modules.values()):
        filename=getattr(module,'__file__',None)
        if not filename:continue
        path=Path(filename)
        if path.suffix!='.py' or not path.is_relative_to(P):continue
        # Historical dependencies and this run's orchestration have separate freezes.
        if path.is_relative_to(root):continue
        relative=path.relative_to(P);destination=snapshot/relative
        digest=support.sha(path)
        if destination.exists() and support.sha(destination)!=digest:raise RuntimeError('Archived dependency changed')
        if not destination.exists():destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,destination)
        records.append({'upstream':str(path),'snapshot':str(destination),'sha256':digest})
    depfile=root/'manifests/MULTISCALE_CODE_DEPENDENCIES.json'
    if not depfile.exists():support.write(depfile,records)
    else:
        for row in json.loads(depfile.read_text()):
            if Path(row['upstream']).is_relative_to(root):continue
            if support.sha(row['upstream'])!=row['sha256']:raise RuntimeError('Frozen implementation changed')
    if not contract.exists():
        support.write(contract,{'utc':support.now(),'role':'required learned components of NEW-SCORE-ONLY, not a complete score pipeline by itself',
            'source':'auxiliary_data_v2;4096 train/512 validation waveform parents; source/noise/lens-environment disjoint',
            'short_matching':'same physical PSD whitening and IMRPhenomD253Mc x3q x3chi; 2s2048Hz; phase-quadrature H,L,network; +/-304samples; relative21samples; log1p;27x253',
            'long_matching':'same archived physical20-80Hz16s256Hz; +/-38samples; relative3;27x253; no long-window SD normalization',
            'model_seeds':SEEDS,'ordered_mass':'same Predictor;30epochs;AdamW2e-4;source-balanced batches; minimum validationCE; temperature grid0.5,0.75,1,1.25,1.5,2',
            'multirate':'same55-channel first convolution; corresponding ordered-Mc checkpoint warm start; new input weights exactly zero;15epochs;AdamW1e-4; epoch0 eligible',
            'conditional':'same103-96-96-272 residual head for eta16 xchi17 conditional on Mc253; frozen Mc marginal;25epochs;AdamW1e-3;epoch0 eligible',
            'prior':'one weight per simulated waveform source, not per view; archived smoothing/floors unchanged',
            'selection':'simulation validation CE only; no real PE or official overlap; no locked-test inference',
            'no_outer_score_mixture':True,'waveform_prior_is_not_PE':True,
            'pending_other_components':['legacy RAW-PHASE/FRT/OMC score ancestry','conditional penalty/increment calibration','BAYESTAR and time scoring','final validation weight freeze','heldout and real audit'],
            'numerical_gates':['template quadrature orthonormality1e-4','lowband canonical correlation>=0.999','warmstart logit difference<=1e-10','joint mass marginal invariance<=1e-6']})
        support.write(root/'contracts/MULTISCALE_TRAINING_FREEZE.json',{'sha256':support.sha(contract)})
    return low,mult,cond,out


def templates(root):
    low,mult,cond,out=initialize(root)
    path=out/'cache/short253_spectra.npy'
    if not path.exists():
        # These are deterministic physical templates, not event/test features.
        np.save(path,low.t.spectrum_bank())
        support.write(path.with_suffix('.json'),{'sha256':support.sha(path),'templates':2277,'source':'archived coarse and refined IMRPhenomD spectra, no observed PE inputs'})
    low.spectral_bank(out)
    print(json.dumps({'templates':'READY','root':str(out)}),flush=True)


def data(root,split,kind='ORDERED'):
    folder=root/f'waveform_features/auxiliary_v2/{split}'
    paths=[folder/'short.npy']+([folder/'long.npy'] if kind=='MULTIRATE' else [])
    for path in paths:
        if not path.with_suffix('.COMPLETE.json').exists():raise RuntimeError(f'Features incomplete:{path}')
    arrays=[np.load(path,mmap_mode='r') for path in paths]
    values=np.concatenate(arrays,axis=1) if len(arrays)>1 else np.array(arrays[0],copy=True)
    meta=pd.read_parquet(root/f'auxiliary_data_v2/{split}/event_metadata.parquet')
    if len(meta)!=len(values) or not np.array_equal(meta.row_index,np.arange(len(meta))):raise RuntimeError('Feature/event ordering mismatch')
    return values,meta


def features(root,split):
    low,mult,cond,out=initialize(root);templates(root)
    source=root/f'auxiliary_data_v2/{split}'
    if not (source/'COMPLETE.json').exists():raise RuntimeError('Auxiliary waveforms incomplete')
    meta=pd.read_parquet(source/'event_metadata.parquet');ids=meta.noise_bank_index.to_numpy(int)
    shared=root/'workspace/results/real_noise_injection_v5_physical_source_20260721/gwtc5/shared'
    freq=np.load(shared/'noise_psd_frequency.npy');psds=np.load(shared/'noise_psd_bank.npy',mmap_mode='r')
    folder=root/f'waveform_features/auxiliary_v2/{split}';folder.mkdir(parents=True,exist_ok=True)
    path=folder/'short.npy';marker=path.with_suffix('.COMPLETE.json')
    if not marker.exists():
        h=np.load(out/'cache/short253_spectra.npy',mmap_mode='r');raw=np.load(source/'raw2s.npy',mmap_mode='r')
        partial=path.with_suffix('.partial.npy');shape=(len(meta),27,253)
        x=np.lib.format.open_memmap(partial,mode='r+' if partial.exists() else 'w+',dtype=np.float32,shape=shape)
        progress=path.with_suffix('.progress.json');rows=json.loads(progress.read_text()) if progress.exists() else []
        done={row['PSD'] for row in rows}
        for pid in np.unique(ids):
            if int(pid) in done:continue
            support.disk_guard(root);tick=time.monotonic();keep=np.flatnonzero(ids==pid)
            bank=low.t.adaptive.whitened_bank(h,freq,psds[int(pid)])
            gram=max(float(abs((bank*bank).sum(-1)-1).max()),float(abs((bank[:,:,0]*bank[:,:,1]).sum(-1)).max()))
            if gram>1e-4:raise RuntimeError('Nonorthonormal short quadratures')
            f=low.t.fine.features(raw[keep],bank)
            values=f.reshape(len(f),3,253,3,3).transpose(0,1,3,4,2).reshape(len(f),27,253)
            if not np.isfinite(values).all():raise RuntimeError('Nonfinite matching features')
            x[keep]=values;x.flush();del bank
            rows.append({'PSD':int(pid),'events':len(keep),'seconds':time.monotonic()-tick,'gram_error':gram})
            support.write(progress,rows)
            print(json.dumps({'short_features':split,'PSD_complete':len(rows),'PSD_total':len(np.unique(ids)),**rows[-1]}),flush=True)
        del x;partial.rename(path)
        support.write(marker,{'sha256':support.sha(path),'events':len(meta),'shape':shape})
    low.grouped(out,np.load(source/'low16s.npy',mmap_mode='r'),freq,psds,ids,folder/'long.npy')


def train(root,stage,seed):
    low,mult,cond,out=initialize(root)
    old=low.old
    old.data=lambda unused,dep,split:data(root,split)
    # The O4b contract replaces old O3/O4a population descriptions, not algorithms.
    old.initialize=lambda unused:None
    mult.training_data=lambda unused,dep,split,kind:data(root,split,kind)
    if stage=='ordered':old.train(root,'gwtc5',seed,seed)
    elif stage=='multirate':
        mult.t.PREVIOUS=root
        mult.train_one(root,'gwtc5','MULTIRATE',seed,seed)
    elif stage=='conditional':
        cond.UPSTREAM=root
        cond.mult.training_data=mult.training_data
        cond.train_one(root,'gwtc5',seed,seed,'CONDITIONAL-ETA-CHI')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--stage',choices=['templates','features','ordered','multirate','conditional'],required=True)
    p.add_argument('--split',choices=['train','validation'],default='validation')
    p.add_argument('--seed',type=int,choices=SEEDS,default=SEEDS[0]);a=p.parse_args()
    lock=a.root/'contracts/GPU_TRAINING.lock';lock.parent.mkdir(exist_ok=True)
    with lock.open('a') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX)
        if a.stage=='templates':templates(a.root)
        elif a.stage=='features':features(a.root,a.split)
        else:train(a.root,a.stage,a.seed)
