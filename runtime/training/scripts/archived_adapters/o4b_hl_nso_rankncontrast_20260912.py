#!/usr/bin/env python3
"""O4b transfer training of the archived RAW-PHASE/RNC waveform component.

Only the archived O4a model parameters are used as initialization. All training,
normalization, source priors, temperature and checkpoint selection use O4b
simulation development data. No historical event ranks or public PE are read.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='2'
import argparse
import fcntl
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd
import torch

import o4b_hl_nso_data_training_20260912 as s
import o4b_hl_nso_multiscale_training_20260912 as shared_code

SEEDS=(2026091221,2026091222,2026091223)
OLD_SEEDS=(202609061,202609062,202609063)


def setup(root):
    shared_code.initialize(root)
    import mcwf_finelag_eventpsd_20260907 as psd
    import mcwf_finelag_encoder_20260907 as trainer
    import mcwf_rankncontrast_20260907 as rnc
    import mcwf_mass_tf_20260905 as tf
    out=root/'rankncontrast_component'
    for name in ('contracts','cache','models','scripts','manifests'):
        (out/name).mkdir(parents=True,exist_ok=True)
    records=[]
    for module in list(sys.modules.values()):
        filename=getattr(module,'__file__',None)
        if not filename:continue
        path=Path(filename)
        if path.suffix!='.py' or not path.is_relative_to(shared_code.P):continue
        dest=out/'scripts'/path.relative_to(shared_code.P)
        if dest.exists() and s.sha(dest)!=s.sha(path):raise RuntimeError('Frozen dependency changed')
        if not dest.exists():dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest)
        records.append({'source':str(path),'snapshot':str(dest),'sha256':s.sha(path)})
    cp=out/'contracts/FINE_LAG_TRAINING.json'
    if not cp.exists():
        s.write(cp,{'created_utc':s.now(),'code':'O4B-HL-NSO-01-RNC',
            'architecture':'archived RAW-PHASE-SOURCE;2s96D InceptionAttention +576-template phase responses ->128D embedding and64-bin logMc',
            'feature_implementation':psd.IMPLEMENTATION,'objective':'CE+sourceSupCon+RNC(logMc)',
            'initialization':'one-to-one archived O4a RNC checkpoints; transfer initialization only, no old ranks/calibration/prior used',
            'training_data':'auxiliary_data_v2 O4b:4096 independent waveform parents/32768views',
            'validation_data':'512 source-disjoint waveform parents/1024views; no locked test or publicPE',
            'epochs':50,'passes_per_epoch':4,'sources_per_batch':64,'views_per_source':2,
            'optimizer':'AdamW lr1e-4 weight_decay1e-4 cosine_eta_min1e-5;gradientclip5',
            'selection':'earliest minimum validation CE+0.2SupCon+0.2RNC',
            'temperature_grid':[.5,.75,1,1.25,1.5,2,3,4],
            'source_prior':'one weight per O4b waveform parent;not old O4a prior',
            'source_seed_mapping':dict(zip(map(str,SEEDS),OLD_SEEDS)),
            'numerical_test':rnc.tests(),'frozen_time_sky':True})
        s.write(out/'contracts/FREEZE.json',{'sha256':s.sha(cp)})
        s.write(out/'manifests/CODE_DEPENDENCIES.json',records)
    else:
        if s.sha(cp)!=json.loads((out/'contracts/FREEZE.json').read_text())['sha256']:
            raise RuntimeError('RNC training contract changed')
    trainer.b.PREVIOUS=out
    return out,psd,trainer,tf


def prepare(root):
    out,psd,trainer,tf=setup(root)
    cache=out/'cache/gwtc5';cache.mkdir(parents=True,exist_ok=True)
    features=out/'cache/finelag/gwtc5';features.mkdir(parents=True,exist_ok=True)
    targets=out/'cache/masstf/gwtc5';targets.mkdir(parents=True,exist_ok=True)
    shared=root/'workspace/results/real_noise_injection_v5_physical_source_20260721/gwtc5/shared'
    freq=np.load(shared/'noise_psd_frequency.npy')
    psds=np.load(shared/'noise_psd_bank.npy',mmap_mode='r')
    stats=[]
    for split in ('validation','train'):
        src=root/f'auxiliary_data_v2/{split}'
        if not (src/'COMPLETE.json').exists():raise RuntimeError('O4b auxiliary waveform data incomplete')
        meta=pd.read_parquet(src/'event_metadata.parquet')
        meta['waveform_parent_uid']=meta.source_uid
        meta['chirp_mass_detector']=meta.mc_det
        meta.to_parquet(cache/f'{split}_metadata.parquet',index=False)
        rawpath=cache/f'{split}_raw2s.npy'
        if not rawpath.exists():rawpath.symlink_to(src/'raw2s.npy')
        np.save(targets/f'{split}_targets.npy',tf.target_prob(np.log(meta.mc_det.to_numpy())))
        raw=np.load(rawpath,mmap_mode='r');path=features/f'{split}_features.npy'
        psd.extract_grouped(out,raw,freq,psds,meta.noise_bank_index.to_numpy(int),path)
        stats.append({'split':split,'events':len(meta),'waveform_parents':meta.source_uid.nunique(),
                      'raw_sha256':s.sha(rawpath),'features_sha256':s.sha(path)})
    tmeta=pd.read_parquet(cache/'train_metadata.parquet')
    yy=np.load(targets/'train_targets.npy')
    keep=~tmeta.source_uid.duplicated().to_numpy()
    prior=np.maximum(yy[keep].mean(0),1e-6);prior/=prior.sum()
    warmroot=out/'warm_initialization'
    records=[]
    for seed,oldseed in zip(SEEDS,OLD_SEEDS):
        src=shared_code.P/f'results/mcwf_unified_rankncontrast_encoder_20260907/models/RAW-PHASE-SOURCE/gwtc4/seed_{oldseed}/validation_selected_model.pt'
        dest=warmroot/f'models/RAW-PHASE-SOURCE/gwtc5/seed_{seed}/validation_selected_model.pt'
        if not dest.exists():
            ck=torch.load(src,weights_only=False,map_location='cpu')
            # Keep architecture weights, replace population-specific bookkeeping.
            ck.update(prior=prior,seed=seed,deployment='gwtc5',temperature=1.,
                      transfer_source=str(src),transfer_source_sha256=s.sha(src))
            dest.parent.mkdir(parents=True,exist_ok=True);torch.save(ck,dest)
        records.append({'model_seed':seed,'source_seed':oldseed,'source':str(src),
                        'source_sha256':s.sha(src),'adapted_initialization_sha256':s.sha(dest)})
    s.write(out/'manifests/TRANSFER_INITIALIZATIONS.json',records)
    s.write(features/'COMPLETE.json',{'warm_root':str(warmroot),'rankncontrast':True,
        'feature_implementation':psd.IMPLEMENTATION,
        'bankpath':str(out/'cache/adaptive_psd/unwhitened_aligned_template_spectra.npy'),
        'bankpath_is_provenance_only':True,'splits':stats})


def train(root,seed):
    out,psd,trainer,tf=setup(root)
    if not (out/'cache/finelag/gwtc5/COMPLETE.json').exists():raise RuntimeError('RNC input features incomplete')
    trainer.train(out,'gwtc5',seed)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--stage',choices=('prepare','train'),required=True)
    p.add_argument('--seed',type=int,choices=SEEDS,default=SEEDS[0]);a=p.parse_args()
    with (a.root/'contracts/GPU_TRAINING.lock').open('a') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX)
        if a.stage=='prepare':prepare(a.root)
        else:train(a.root,a.seed)
