#!/usr/bin/env python3
"""Same fine time grid, using each event's actual whitening PSD."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[k]='2'
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import torch
import mcwf_development_20260905 as dev
import mcwf_encoder_body_20260906 as body
import mcwf_mass_tf_20260905 as tf
import mcwf_adaptive_psd_encoder_20260906 as adaptive
import mcwf_finelag_encoder_20260907 as trainer
import mcwf_finelag_features_20260907 as fine

WARM=dev.PROJECT/'results/mcwf_unified_finelag_encoder_20260907'
IMPLEMENTATION='event_psd_sample_resolved_fft_v1'


def extract_grouped(root,raw,freq,psds,ids,path):
    previous=adaptive.phase.features
    try:
        adaptive.phase.features=fine.features
        return adaptive.extract_grouped(root,raw,freq,psds,ids,path)
    finally:
        adaptive.phase.features=previous


def initialize(root):
    path=root/'contracts/EVENT_PSD_FINE_LAG.json'
    if path.exists():
        return
    if root.exists():
        raise RuntimeError('Independent PSD-fine-lag output required')
    trainer.initialize(root)
    contract=json.loads((root/'contracts/FINE_LAG_TRAINING.json').read_text())
    contract.update(code='MCWF-FINELAG-EVENTPSD-v1',
        changed_mechanism='relative to fine-lag: each template is whitened by the exact off-source PSD used for that event; same network, loss,2sinput and numerical lag grid',
        normalization='new event-PSD feature mean/SD fit on train only',
        reason='An event-whitened waveform and a median-PSD-whitened template do not share the same frequency transfer function. Match PSDs instead of interpreting the mismatch as intrinsic mass disagreement.',
        warm_root=str(WARM),feature_implementation=IMPLEMENTATION,
        PSD_role='waveform preprocessing only, not an added ranking channel; never use on-source PE masses',
        bank_storage='unwhitened spectra once; temporary per-PSD bank, persist features only')
    dev.json_write(root/'contracts/FINE_LAG_TRAINING.json',contract)
    dev.json_write(path,contract)
    for p in (Path(__file__),Path(adaptive.__file__)):
        shutil.copy2(p,root/'scripts'/p.name)


def prepare(root,dep):
    cache=body.prepare(root,dep)
    out=root/f'cache/finelag/{dep}'
    if (out/'COMPLETE.json').exists():
        return
    noise=dev.OLD/f'data/noise_banks/{dep}'
    hashes=[]
    for split in ('train','validation'):
        meta=pd.read_parquet(cache/f'{split}_metadata.parquet')
        ids=np.where(meta.image.eq('a'),meta.a_noise_bank_index,meta.b_noise_bank_index).astype(int)
        freq=np.load(noise/f'noise_psd_frequency_{split}.npy')
        psd=np.load(noise/f'noise_psd_{split}.npy',mmap_mode='r')
        x=np.load(cache/f'{split}_raw2s.npy',mmap_mode='r')
        path=out/f'{split}_features.npy'
        extract_grouped(root,x,freq,psd,ids,path)
        hashes.append({'split':split,'features_sha256':dev.sha(path),'PSD_sha256':dev.sha(noise/f'noise_psd_{split}.npy'),
                       'event_count':len(x),'independent_PSDs':len(np.unique(ids))})
        print(json.dumps({'event_psd_prepared':dep,'split':split,'rows':len(x)}),flush=True)
    baseline=json.loads((WARM/f'cache/finelag/{dep}/COMPLETE.json').read_text())
    dev.json_write(out/'COMPLETE.json',{'feature_implementation':IMPLEMENTATION,'warm_root':str(WARM),
        'bankpath':baseline['bankpath'],'bankpath_is_compatibility_metadata_only':True,'PSD_features':hashes,
        'spectra_path':str(root/'cache/adaptive_psd/unwhitened_aligned_template_spectra.npy')})


def psd_context(dep,es,split):
    if split=='real':
        parent=body.PREVIOUS/f'cache/phasepsd/{dep}'
        packed=np.load(parent/'real_reference_PSDs.npz')
        manifest=pd.read_csv(parent/'real_psd_source_manifest.csv')
        _,events=dev.real_inputs(dep)
        names=manifest.event_name.drop_duplicates().to_numpy()
        expected=events.set_index('idx').loc[packed['idx'],'event_name'].to_numpy()
        if not np.array_equal(names,expected):
            raise RuntimeError('PSD/event-name mapping failed')
        valid=events.strict_h1l1_preprocessing_pass.to_numpy(bool)
        mapping={int(k):i for i,k in enumerate(packed['idx'])}
        indices=np.array([mapping[int(i)] for i in events.loc[valid,'idx']])
        return packed['frequency'],packed['psd'],indices,dev.sha(parent/'real_reference_PSDs.npz')
    plan=dev.BASE.retained_event_plan(dep,es,split)
    shared=dev.ORCH.SOURCE_ROOT/dep/'shared'
    return np.load(shared/'noise_psd_frequency.npy'),np.load(shared/'noise_psd_bank.npy',mmap_mode='r'),plan.parent_noise_bank.to_numpy(int),dev.sha(shared/'noise_psd_bank.npy')


def encode_context(root,checkpoint,full,frequency,psds,indices,feature_path):
    ck=torch.load(checkpoint,weights_only=False,map_location='cpu')
    if ck.get('feature_implementation')!=IMPLEMENTATION:
        raise RuntimeError('Wrong checkpoint for event-PSD feature pipeline')
    raw=dev.TRAIN.make_window_view(np.asarray(full,dtype=np.float32),2)
    x=extract_grouped(root,raw,frequency,psds,indices,feature_path)
    model=body.Encoder('RAW-PHASE-SOURCE').cuda().eval()
    model.load_state_dict(ck['model'])
    l,z=body.infer(model,(x-ck['mu'])/ck['sd'],raw)
    if 'independent_mass_component' in ck:
        mass_model=adaptive.phase.MassPhase().cuda().eval()
        mass_model.load_state_dict(ck['independent_mass_component'])
        l,_=tf.infer_logits(mass_model,(x-ck['independent_mass_mu'])/ck['independent_mass_sd'])
    lp=l.astype(float)/ck['temperature']
    lp-=np.logaddexp.reduce(lp,axis=-1,keepdims=True)
    p=np.exp(lp)
    if 'physical_mass_anchor' in ck:
        from mcwf_physical_mass_anchor_20260907 import combine
        p=combine(p,x,ck['physical_mass_anchor'])
    return p,z,ck


def input_prediction(root,dep,ms,es,config,split):
    cp=root/f'models/RAW-PHASE-SOURCE/{dep}/seed_{ms}/validation_selected_model.pt'
    out=root/f'cache/predictions/RAW-PHASE-SOURCE/{dep}/model_{ms}_eval_{es}'
    out.mkdir(parents=True,exist_ok=True)
    path=out/f'{split}.npz'
    if path.exists():
        saved=np.load(path)
        if str(saved['checkpoint_sha256'])!=dev.sha(cp):
            raise RuntimeError('Changed checkpoint')
        return saved['p'],saved['z']
    freq,psds,ids,psdhash=psd_context(dep,es,split)
    if split=='real':
        full,events=dev.real_inputs(dep)
        valid=events.strict_h1l1_preprocessing_pass.to_numpy(bool)
        values=np.asarray(full[valid])
        feature_path=root/f'cache/deployment_event_psd/{dep}/real_features.npy'
    else:
        events=dev.BASE.retained_event_plan(dep,es,split)
        values=dev.ORCH.event_array_for_plan(dep,es,split,events)
        feature_path=root/f'cache/deployment_event_psd/{dep}/{es}_{split}_features.npy'
        events.to_parquet(out/f'{split}_plan.parquet',index=False)
    pp,zz,_=encode_context(root,cp,values,freq,psds,ids,feature_path)
    if split=='real':
        probs=np.full((len(full),64),np.nan)
        emb=np.full((len(full),128),np.nan)
        probs[valid],emb[valid]=pp,zz
        table=events[['idx','event_name','strict_h1l1_preprocessing_pass']].copy()
        table['pred_logmc']=probs@tf.LOG_CENTERS
        table['pred_mc']=np.exp(table.pred_logmc)
        table['pred_logmc_std']=np.sqrt(np.maximum(probs@tf.LOG_CENTERS**2-table.pred_logmc.to_numpy()**2,0))
        dev.csv_write(out/'real_event_mass_predictions.csv',table)
    else:
        probs,emb=pp,zz
    np.savez_compressed(path,p=probs,z=emb,checkpoint_sha256=dev.sha(cp),PSD_sha256=psdhash)
    return probs,emb


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--deployment',choices=('gwtc3','gwtc4'),required=True)
    p.add_argument('--seed',type=int,choices=body.MODEL_SEEDS,required=True)
    a=p.parse_args()
    trainer.train(a.root,a.deployment,a.seed)
