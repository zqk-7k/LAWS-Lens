#!/usr/bin/env python3
"""Separate phase-feature mass prediction from raw source embedding learning."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[k]='2'
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import mcwf_development_20260905 as dev
import mcwf_encoder_body_20260906 as body
import mcwf_mass_tf_20260905 as tf
import mcwf_phasebank_20260905 as phase
import mcwf_finelag_encoder_20260907 as trainer

WARM=dev.PROJECT/'results/mcwf_unified_finelag_eventpsd_encoder_20260907'
torch.set_num_threads(2)


def initialize(root):
    if (root/'contracts/INDEPENDENT_MASS_COMPONENT.json').exists():
        return
    if root.exists():
        raise RuntimeError('Independent output required')
    trainer.initialize(root)
    contract=json.loads((root/'contracts/FINE_LAG_TRAINING.json').read_text())
    contract.update(code='MCWF-FINELAG-EVENTPSD-INDEPENDENT-MASS',created_utc=datetime.now(timezone.utc).isoformat(),
        changed_mechanism='Retain trained fine-lag event-PSD RAW source encoder and its embedding. Train a separate phase-feature-only MassPhase network by simulated-mass CE; no shared raw/source gradients into this mass predictor.',
        reason='Physical matched-filter feature maxima on diagnosed noisy examples remain near injected mass while the merged raw/source mass head shifts far away. Task-gradient audit shows no negative dot products; PCGrad is therefore not used.',
        same_across_runs=True,feature_implementation='event_psd_sample_resolved_fft_v1',
        source_encoder_root=str(WARM),mass_input='same 2s4096 phase-correlations at exact event PSD; no additional strain window, no real PE',
        mass_architecture='existing MassPhase:1728->256 LayerNorm SiLU Dropout0.15->128 SiLU->64',
        mass_training={'initialization':'independent seeded random','epochs':50,'lr':3e-4,'weight_decay':1e-4,
                       'cosine_min_lr':3e-5,'batch_sources':64,'views':2,'passes':4,'clip':5},
        selection='earliest minimum development-validation soft-label CE; temperature minimum CE on same validation',
        source_encoder_retrained_this_substep=False,source_encoder_already_trained_for_both_runs=True,
        diagnostic_roots=['results/mcwf_task_gradient_audit_20260907','results/mcwf_lag_support_audit_20260907'],
        new_component_is_not_PE_posterior=True,frozen='time,sky,C-fixed outer weights,real scope,historical data')
    dev.json_write(root/'contracts/FINE_LAG_TRAINING.json',contract)
    dev.json_write(root/'contracts/INDEPENDENT_MASS_COMPONENT.json',contract)
    for dep in ('gwtc3','gwtc4'):
        (root/'cache').mkdir(exist_ok=True)
        (root/f'cache/{dep}').symlink_to((WARM/f'cache/{dep}').resolve(),target_is_directory=True)
        out=root/f'cache/finelag/{dep}'
        out.mkdir(parents=True)
        for name in ('train_features.npy','validation_features.npy','COMPLETE.json'):
            (out/name).symlink_to((WARM/f'cache/finelag/{dep}/{name}').resolve())
    out=root/'cache/adaptive_psd'
    out.mkdir(parents=True)
    for name in ('unwhitened_aligned_template_spectra.npy','SPECTRA_SOURCE.json'):
        shutil.copy2(WARM/'cache/adaptive_psd'/name,out/name)
    shutil.copy2(__file__,root/'scripts'/Path(__file__).name)


def train(root,dep,seed):
    out=root/f'models/RAW-PHASE-SOURCE/{dep}/seed_{seed}'
    if (out/'COMPLETE.json').exists():
        return
    out.mkdir(parents=True,exist_ok=True)
    original=WARM/f'models/RAW-PHASE-SOURCE/{dep}/seed_{seed}/validation_selected_model.pt'
    ck=torch.load(original,weights_only=False,map_location='cpu')
    x=np.load(WARM/f'cache/finelag/{dep}/train_features.npy')
    vx=np.load(WARM/f'cache/finelag/{dep}/validation_features.npy')
    x=(x-ck['mu'])/ck['sd']
    vx=(vx-ck['mu'])/ck['sd']
    y=np.load(body.PREVIOUS/f'cache/masstf/{dep}/train_targets.npy')
    vy=np.load(body.PREVIOUS/f'cache/masstf/{dep}/validation_targets.npy')
    meta=pd.read_parquet(WARM/f'cache/{dep}/train_metadata.parquet')
    group,labels=pd.factorize(meta.waveform_parent_uid,sort=True)
    members=[np.flatnonzero(group==k) for k in range(len(labels))]
    dev.TRAIN.seed_everything(seed)
    model=phase.MassPhase().cuda()
    opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,50,eta_min=3e-5)
    history=[]
    best=float('inf')
    start_epoch=1
    if (out/'mass_resume.pt').exists():
        r=torch.load(out/'mass_resume.pt',weights_only=False,map_location='cpu')
        model.load_state_dict(r['state']);opt.load_state_dict(r['optimizer']);scheduler.load_state_dict(r['scheduler'])
        torch.set_rng_state(r['rng']);torch.cuda.set_rng_state_all(r['cuda_rng'])
        history,best,start_epoch=r['history'],r['best'],r['epoch']+1
    for epoch in range(start_epoch,51):
        start=time.perf_counter()
        model.train()
        losses=[]
        rng=np.random.default_rng(seed+epoch)
        for repeat in range(4):
            order=rng.permutation(len(labels))
            for first in range(0,len(order),64):
                groups=order[first:first+64]
                ids=np.stack([rng.choice(members[k],2,replace=False) for k in groups]).T.reshape(-1)
                xx=torch.as_tensor(x[ids],device='cuda')
                yy=torch.as_tensor(y[ids],device='cuda')
                opt.zero_grad(set_to_none=True)
                logits,_=model(xx)
                loss=-(F.log_softmax(logits,-1)*yy).sum(-1).mean()
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite mass CE')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
                losses.append(float(loss.detach()))
        scheduler.step()
        logits,_=tf.infer_logits(model,vx)
        lp=logits.astype(float)-np.logaddexp.reduce(logits.astype(float),axis=-1,keepdims=True)
        ce=float(-(vy*lp).sum(-1).mean())
        history.append({'epoch':epoch,'train_ce':float(np.mean(losses)),'validation_ce':ce,
            'validation_logMc_mae':float(np.mean(abs(np.exp(lp)@tf.LOG_CENTERS-vy@tf.LOG_CENTERS))),
            'seconds':time.perf_counter()-start})
        if ce<best:
            best=ce
            torch.save({'state':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},'epoch':epoch,'criterion':ce},out/'selected_mass_component.pt')
        torch.save({'state':model.state_dict(),'optimizer':opt.state_dict(),'scheduler':scheduler.state_dict(),
            'history':history,'best':best,'epoch':epoch,'rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()},out/'mass_resume.pt')
        dev.csv_write(out/'training_history.csv',pd.DataFrame(history))
        if epoch%10==0:
            print(json.dumps({'deployment':dep,'seed':seed,**history[-1]}),flush=True)
    mass=torch.load(out/'selected_mass_component.pt',weights_only=False,map_location='cpu')
    model.load_state_dict(mass['state'])
    logits,_=tf.infer_logits(model,vx)
    grid=[]
    for t in (.5,.75,1.,1.25,1.5,2.,3.,4.):
        lp=logits.astype(float)/t
        lp-=np.logaddexp.reduce(lp,axis=-1,keepdims=True)
        grid.append({'temperature':t,'ce':float(-(vy*lp).sum(-1).mean())})
    temp=min(grid,key=lambda q:(q['ce'],abs(q['temperature']-1)))['temperature']
    ck.update(independent_mass_component=mass['state'],source_temperature=ck['temperature'],temperature=temp,
        independent_mass_epoch=mass['epoch'],source_encoder_checkpoint=str(original),source_encoder_sha256=dev.sha(original),
        independent_mass_mu=ck['mu'],independent_mass_sd=ck['sd'])
    torch.save(ck,out/'validation_selected_model.pt')
    previous=np.load(WARM/f'models/RAW-PHASE-SOURCE/{dep}/seed_{seed}/development_validation.npz')
    np.savez_compressed(out/'development_validation.npz',logits=logits,embedding=previous['embedding'],truth=vy,group=previous['group'])
    dev.csv_write(out/'temperature_grid.csv',pd.DataFrame(grid))
    dev.json_write(out/'COMPLETE.json',{'checkpoint_sha256':dev.sha(out/'validation_selected_model.pt'),
        'mass_epochs':50,'mass_selected_epoch':mass['epoch'],'mass_temperature':temp,
        'source_embedding_unchanged_from_finelag_eventPSD':True,'real_PE_training':False})


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--deployment',choices=('gwtc3','gwtc4'))
    p.add_argument('--seed',type=int,choices=body.MODEL_SEEDS)
    a=p.parse_args()
    initialize(a.root)
    if a.deployment:
        train(a.root,a.deployment,a.seed)
