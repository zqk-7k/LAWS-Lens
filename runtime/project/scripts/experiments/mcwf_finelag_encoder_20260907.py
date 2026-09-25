#!/usr/bin/env python3
"""Shared fine-lag waveform encoder training for O3 and O4a."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
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
from torch import nn
import torch.nn.functional as F
import mcwf_development_20260905 as dev
import mcwf_encoder_body_20260906 as b
import mcwf_mass_tf_20260905 as tf
import mcwf_finelag_features_20260907 as fine

OLDROOT=dev.PROJECT/'results/main_o3_mcwf_encoder_v2_20260906T153600Z'
EPOCHS=50
torch.set_num_threads(2)


def initialize(root):
    path=root/'contracts/FINE_LAG_TRAINING.json'
    if path.exists():
        return
    root.mkdir(parents=True,exist_ok=True)
    dev.json_write(path,{'created_utc':datetime.now(timezone.utc).isoformat(),'code':'MCWF-FINELAG-FFT-v1',
        'changed_mechanism':'template-correlation time sampling16->1samples, float32 FFT; near-coincident H1L1 window covers physical10.013ms with21samples',
        'why':'Reused failure diagnostics show mass predictions jump under4-8sample shifts. Numerical timing resolution must be fixed before trying stronger evidence penalties.',
        'input':'unchanged peak2s4096,H1L1,2048Hz; no new time or sky channel inputs',
        'model':'same RAW-PHASE-SOURCE architecture in both runs; all parameters fine-tuned from each independent original seed',
        'objective':'soft64bin simulated logMc cross entropy + source supervised contrastive loss',
        'optimizer':{'class':'AdamW','lr':1e-4,'weight_decay':1e-4,'cosine_min_lr':1e-5,'epochs':EPOCHS,'sources_per_batch':64,'views_per_source':2,'passes_per_epoch':4,'gradient_norm_clip':5},
        'selection':'earliest minimum validation CE+0.2*source-contrastive; temperature by validation CE only',
        'normalization':'recompute phase-feature train mean and SD floor0.05 after time-resolution change; raw channel preprocessing unchanged',
        'frozen':'time,sky,outerweights,scope,all historical files',
        'PE_usage':'no real PE in training or calibration; repeated real-audit results are development feedback, not blind confirmation',
        'reference':'https://pycbc.org/pycbc/latest/html/filter.html',
        'limitation':'approximate phase-sensitive waveform features, not a new PE likelihood or SNR-based detection pipeline',
        'numerical_test':fine.numerical_test()})
    for p in (Path(__file__),Path(fine.__file__)):
        (root/'scripts').mkdir(exist_ok=True)
        shutil.copy2(p,root/'scripts'/p.name)


@torch.no_grad()
def encode(checkpoint,full24):
    ck=torch.load(checkpoint,weights_only=False,map_location='cpu')
    if ck.get('feature_implementation')!='sample_resolved_zero_padded_fft_v1':
        raise RuntimeError('Fine-lag encoder requires its own trained checkpoint')
    model=b.Encoder('RAW-PHASE-SOURCE').cuda().eval()
    model.load_state_dict(ck['model'])
    raw=dev.TRAIN.make_window_view(np.asarray(full24,dtype=np.float32),2)
    x=fine.features(raw,np.load(ck['bankpath']))
    l,z=b.infer(model,(x-ck['mu'])/ck['sd'],raw)
    lp=l.astype(float)/ck['temperature']
    lp-=np.logaddexp.reduce(lp,axis=-1,keepdims=True)
    return np.exp(lp),z,ck


def train(root,dep,seed):
    initialize(root)
    out=root/f'models/RAW-PHASE-SOURCE/{dep}/seed_{seed}'
    if (out/'COMPLETE.json').exists():
        return
    out.mkdir(parents=True,exist_ok=True)
    cache=root/f'cache/{dep}'
    feature_root=root/f'cache/finelag/{dep}'
    metadata=json.loads((feature_root/'COMPLETE.json').read_text())
    x=np.load(feature_root/'train_features.npy')
    vx=np.load(feature_root/'validation_features.npy')
    mu,sd=x.mean(0),x.std(0).clip(.05)
    x=(x-mu)/sd
    vx=(vx-mu)/sd
    y=np.load(b.PREVIOUS/f'cache/masstf/{dep}/train_targets.npy')
    vy=np.load(b.PREVIOUS/f'cache/masstf/{dep}/validation_targets.npy')
    meta=pd.read_parquet(cache/'train_metadata.parquet')
    vm=pd.read_parquet(cache/'validation_metadata.parquet')
    true_mass=np.log(meta.chirp_mass_detector.to_numpy(float))
    validation_mass=np.log(vm.chirp_mass_detector.to_numpy(float))
    group,labels=pd.factorize(meta.waveform_parent_uid,sort=True)
    vg,_=pd.factorize(vm.waveform_parent_uid,sort=True)
    members=[np.flatnonzero(group==k) for k in range(len(labels))]
    tx=np.load(cache/'train_raw2s.npy',mmap_mode='r')
    tv=np.load(cache/'validation_raw2s.npy',mmap_mode='r')
    warm=Path(metadata.get('warm_root',OLDROOT))/f'models/RAW-PHASE-SOURCE/{dep}/seed_{seed}/validation_selected_model.pt'
    ck=torch.load(warm,weights_only=False,map_location='cpu')
    dev.TRAIN.seed_everything(seed)
    model=b.Encoder('RAW-PHASE-SOURCE').cuda()
    model.load_state_dict(ck['model'])
    opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,EPOCHS,eta_min=1e-5)
    history=[]
    best=float('inf')
    start_epoch=1
    resume=out/'resume.pt'
    if resume.exists():
        r=torch.load(resume,weights_only=False,map_location='cpu')
        model.load_state_dict(r['model'])
        opt.load_state_dict(r['optimizer'])
        schedule.load_state_dict(r['scheduler'])
        torch.set_rng_state(r['rng'])
        torch.cuda.set_rng_state_all(r['cuda_rng'])
        history,best,start_epoch=r['history'],r['best'],r['epoch']+1
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(start_epoch,EPOCHS+1):
        start=time.perf_counter()
        rng=np.random.default_rng(seed+epoch)
        model.train()
        losses=[]
        gradient_audit=[]
        for repeat in range(4):
            order=rng.permutation(len(labels))
            for first in range(0,len(order),64):
                sources=order[first:first+64]
                ids=np.stack([rng.choice(members[k],2,replace=False) for k in sources]).T.reshape(-1)
                groups=torch.as_tensor(np.tile(sources,2),device='cuda')
                raw=torch.as_tensor(np.asarray(tx[ids]),dtype=torch.float32,device='cuda')
                xx=torch.as_tensor(x[ids],dtype=torch.float32,device='cuda')
                yy=torch.as_tensor(y[ids],device='cuda')
                opt.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    l,z=model(xx,raw)
                    ce=-(F.log_softmax(l.float(),-1)*yy).sum(-1).mean()
                    sc=b.source_contrastive(z,groups)
                    loss=ce+sc
                    if metadata.get('rankncontrast'):
                        from mcwf_rankncontrast_20260907 import rank_loss
                        rc=rank_loss(z,torch.as_tensor(true_mass[ids],device='cuda'))
                        loss=loss+rc
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite training objective')
                if metadata.get('gradient_mode')=='pcgrad':
                    from mcwf_pcgrad_20260907 import gradients
                    gradient_audit.append(gradients((ce,sc),model.parameters()))
                else:
                    loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(),5)
                opt.step()
                losses.append(float(loss.detach()))
        schedule.step()
        l,z=b.infer(model,vx,tv)
        lp=l.astype(float)-np.logaddexp.reduce(l.astype(float),axis=-1,keepdims=True)
        ce=float(-(vy*lp).sum(-1).mean())
        sc=float(b.source_contrastive(torch.as_tensor(z,device='cuda'),torch.as_tensor(vg,device='cuda')))
        criterion=ce+.2*sc
        rank_value=None
        if metadata.get('rankncontrast'):
            from mcwf_rankncontrast_20260907 import rank_loss
            rank_value=float(rank_loss(torch.as_tensor(z,device='cuda'),torch.as_tensor(validation_mass,device='cuda')))
            criterion+=.2*rank_value
        row={'epoch':epoch,'train_loss':float(np.mean(losses)),'validation_ce':ce,'validation_supcon':sc,
             'selection_loss':criterion,'validation_logMc_mae':float(np.mean(np.abs(np.exp(lp)@tf.LOG_CENTERS-vy@tf.LOG_CENTERS))),
             'seconds':time.perf_counter()-start,'gpu_peak_bytes':torch.cuda.max_memory_allocated()}
        history.append(row)
        if rank_value is not None:
            row['validation_rank_contrast']=rank_value
        if gradient_audit:
            row.update(gradient_conflict_fraction=float(np.mean([g['conflict'] for g in gradient_audit])),
                       gradient_cosine_mean=float(np.mean([g['cosine'] for g in gradient_audit])))
        if criterion<best:
            best=criterion
            payload={**ck,'model':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                'mu':mu,'sd':sd,'epoch':epoch,'criterion':criterion,'bankpath':metadata['bankpath'],
                'warm_start':str(warm),'warm_sha256':dev.sha(warm),'feature_implementation':metadata.get('feature_implementation','sample_resolved_zero_padded_fft_v1'),
                'training_feature_path':str(feature_root/'train_features.npy'),'training_feature_sha256':dev.sha(feature_root/'train_features.npy')}
            payload['rankncontrast']=bool(metadata.get('rankncontrast',False))
            torch.save(payload,out/'validation_selected_model.pt')
        torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'scheduler':schedule.state_dict(),
                    'history':history,'best':best,'epoch':epoch,'rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()},resume)
        dev.csv_write(out/'training_history.csv',pd.DataFrame(history))
        if epoch%5==0:
            print(json.dumps({'dep':dep,'seed':seed,**row}),flush=True)
    payload=torch.load(out/'validation_selected_model.pt',weights_only=False,map_location='cpu')
    model.load_state_dict(payload['model'])
    l,z=b.infer(model,vx,tv)
    grid=[]
    for t in (.5,.75,1.,1.25,1.5,2.,3.,4.):
        lp=l.astype(float)/t
        lp-=np.logaddexp.reduce(lp,axis=-1,keepdims=True)
        grid.append({'temperature':t,'ce':float(-(vy*lp).sum(-1).mean())})
    payload['temperature']=min(grid,key=lambda r:(r['ce'],abs(r['temperature']-1)))['temperature']
    torch.save(payload,out/'validation_selected_model.pt')
    np.savez_compressed(out/'development_validation.npz',logits=l,embedding=z,truth=vy,group=vg)
    dev.csv_write(out/'temperature_grid.csv',pd.DataFrame(grid))
    dev.json_write(out/'COMPLETE.json',{'checkpoint_sha256':dev.sha(out/'validation_selected_model.pt'),
        'encoder_body_trained':True,'epochs':EPOCHS,'selected_epoch':payload['epoch'],'temperature':payload['temperature'],
        'feature_implementation':payload['feature_implementation'],'real_PE_used_for_training':False})


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--deployment',choices=('gwtc3','gwtc4'),required=True)
    p.add_argument('--seed',type=int,choices=b.MODEL_SEEDS,required=True)
    a=p.parse_args()
    train(a.root,a.deployment,a.seed)
