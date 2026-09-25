#!/usr/bin/env python3
"""Two-second raw branch with two/eight-second coherent phase contexts."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='2'
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
from scipy.special import softmax
import mcwf_pe_frontend_extension_v2_20260907 as e
import mcwf_long_features_20260907 as lf
import mcwf_rankncontrast_20260907 as rnc
import mcwf_mass_tf_20260905 as tf

dev,body=e.dev,e.body
torch.set_num_threads(2)
EPOCHS=50


def model():
    m=body.Encoder('RAW-PHASE-SOURCE')
    m.phase.layers[1]=nn.Linear(3456,256)
    return m


def initialize(root):
    lf.initialize(root)
    path=root/'long_context/contracts/TRAINING.json'
    if path.exists():return
    dev.json_write(path,{'created_utc':datetime.now(timezone.utc).isoformat(),
        'architecture':'same raw2s InceptionAttention branch; phase MLP first affine input1728->3456 for concatenated2s+8s features; rest unchanged',
        'warm_start':'RNC-FRT encoder; original2s feature weights copied and8s weights initialized to zero; epoch0 is eligible reference',
        'loss':'CE_Mc+sourceSupCon+RNC(logMc)','selection':'minimum validation CE_Mc+0.2*SupCon+0.2*RNC, earliest tie',
        'normalization':'2s original train mean/SD unchanged;8s train-only mean/SD floor0.05',
        'epochs':50,'optimizer':'AdamW1e-4,weight_decay1e-4,cosine_min1e-5,64sources*2views,4passes,gradientclip5',
        'seeds':[202609091,202609092,202609093],'same_both_runs':True,
        'no_PE_or_official_inputs':True,'time_sky_scope_outer_weights_unchanged':True,
        'goal':'test additional waveform context, not alter physical scores or hide previous failures'})
    shutil.copy2(__file__,root/'long_context/scripts'/Path(__file__).name)


def arrays(root,dep,ck,split):
    short=np.load(e.TRAINED/f'cache/finelag/{dep}/{split}_features.npy')
    long=np.load(root/f'long_context/features/{dep}/{split}.npy')
    x=np.concatenate([((short-ck['mu'])/ck['sd']).reshape(len(short),-1),((long-ck['long_mu'])/ck['long_sd']).reshape(len(long),-1)],1)
    raw=np.load(e.TRAINED/f'cache/{dep}/{split}_raw2s.npy',mmap_mode='r')
    return x,raw


def train(root,dep,ms,seed):
    initialize(root)
    out=root/f'long_context/models/{dep}/seed_{ms}'
    if (out/'COMPLETE.json').exists():return
    out.mkdir(parents=True,exist_ok=True)
    for split in ('validation','train'):lf.development(root,dep,split)
    cp=e.TRAINED/f'models/RAW-PHASE-SOURCE/{dep}/seed_{ms}/validation_selected_model.pt'
    ck=torch.load(cp,weights_only=False,map_location='cpu')
    long=np.load(root/f'long_context/features/{dep}/train.npy');ck['long_mu']=long.mean(0);ck['long_sd']=long.std(0).clip(.05)
    dev.TRAIN.seed_everything(seed)
    net=model().cuda();state=ck['model'].copy();old=state['phase.layers.1.weight'];state['phase.layers.1.weight']=torch.cat([old,torch.zeros_like(old)],1);net.load_state_dict(state)
    x,raw=arrays(root,dep,ck,'train');vx,vr=arrays(root,dep,ck,'validation')
    meta=pd.read_parquet(e.TRAINED/f'cache/{dep}/train_metadata.parquet');vm=pd.read_parquet(e.TRAINED/f'cache/{dep}/validation_metadata.parquet')
    group,names=pd.factorize(meta.waveform_parent_uid,sort=True);vg,_=pd.factorize(vm.waveform_parent_uid,sort=True)
    members=[np.flatnonzero(group==k) for k in range(len(names))]
    label=np.log(meta.chirp_mass_detector.to_numpy(float));vl=np.log(vm.chirp_mass_detector.to_numpy(float))
    y=np.load(body.PREVIOUS/f'cache/masstf/{dep}/train_targets.npy');vy=np.load(body.PREVIOUS/f'cache/masstf/{dep}/validation_targets.npy')
    opt=torch.optim.AdamW(net.parameters(),lr=1e-4,weight_decay=1e-4);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,EPOCHS,eta_min=1e-5)
    best=float('inf');history=[];start=time.perf_counter();first=0;resume=out/'resume.pt'
    if resume.exists():
        r=torch.load(resume,weights_only=False,map_location='cpu');net.load_state_dict(r['model']);opt.load_state_dict(r['optimizer']);sched.load_state_dict(r['scheduler'])
        best,history,first=r['best'],r['history'],r['epoch']+1;torch.set_rng_state(r['rng']);torch.cuda.set_rng_state_all(r['cuda_rng'])
    for epoch in range(first,EPOCHS+1):
        tick=time.perf_counter();losses=[]
        if epoch>0:
            net.train();rng=np.random.default_rng(seed+epoch)
            for repeat in range(4):
                order=rng.permutation(len(names))
                for begin in range(0,len(order),64):
                    sources=order[begin:begin+64];ids=np.stack([rng.choice(members[k],2,replace=False) for k in sources]).T.reshape(-1)
                    xx=torch.as_tensor(x[ids],dtype=torch.float32,device='cuda');rr=torch.as_tensor(np.asarray(raw[ids]),dtype=torch.float32,device='cuda')
                    opt.zero_grad(set_to_none=True)
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        logits,z=net(xx,rr);ce=-(F.log_softmax(logits.float(),-1)*torch.as_tensor(y[ids],device='cuda')).sum(-1).mean()
                        sc=body.source_contrastive(z,torch.as_tensor(np.tile(sources,2),device='cuda'));rank=rnc.rank_loss(z,torch.as_tensor(label[ids],device='cuda'));loss=ce+sc+rank
                    if not torch.isfinite(loss):raise RuntimeError('Nonfinite long-context loss')
                    loss.backward();nn.utils.clip_grad_norm_(net.parameters(),5);opt.step();losses.append(float(loss.detach()))
            sched.step()
        logits,z=body.infer(net,vx,vr);p=softmax(logits.astype(float),1)
        ce=float(-(vy*np.log(p.clip(1e-30))).sum(-1).mean());sc=float(body.source_contrastive(torch.as_tensor(z,device='cuda'),torch.as_tensor(vg,device='cuda')))
        rank=float(rnc.rank_loss(torch.as_tensor(z,device='cuda'),torch.as_tensor(vl,device='cuda')));criterion=ce+.2*sc+.2*rank
        row={'epoch':epoch,'train_loss':float(np.mean(losses)) if losses else None,'CE':ce,'SupCon':sc,'RNC':rank,'criterion':criterion,
             'logMc_MAE':float(abs(p@tf.LOG_CENTERS-vl).mean()),'seconds':time.perf_counter()-tick};history.append(row)
        if criterion<best:
            best=criterion;torch.save({**ck,'model':{k:v.detach().cpu().clone() for k,v in net.state_dict().items()},'epoch':epoch,'criterion':criterion,
                'feature_implementation':'peak2plus8_event_psd_v1','new_seed':seed,'warm_sha256':dev.sha(cp)},out/'validation_selected_model.pt')
        torch.save({'model':net.state_dict(),'optimizer':opt.state_dict(),'scheduler':sched.state_dict(),'history':history,'best':best,'epoch':epoch,
                    'rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()},resume)
        dev.csv_write(out/'history.csv',pd.DataFrame(history))
        if epoch%10==0:print(json.dumps({'long_training':dep,'seed':seed,**row}),flush=True)
    chosen=torch.load(out/'validation_selected_model.pt',weights_only=False,map_location='cpu');net.load_state_dict(chosen['model']);logits,z=body.infer(net,vx,vr)
    grid=[]
    for t in (.5,.75,1.,1.25,1.5,2.,3.,4.):
        pp=softmax(logits.astype(float)/t,1);grid.append({'temperature':t,'CE':float(-(vy*np.log(pp.clip(1e-30))).sum(-1).mean())})
    chosen['temperature']=min(grid,key=lambda v:(v['CE'],abs(v['temperature']-1)))['temperature'];torch.save(chosen,out/'validation_selected_model.pt')
    p=softmax(logits.astype(float)/chosen['temperature'],1);np.savez_compressed(out/'validation_predictions.npz',p=p,z=z,truth=vy,group=vg)
    dev.csv_write(out/'temperature_grid.csv',pd.DataFrame(grid))
    st={'deployment':dep,'seed':seed,'epoch':chosen['epoch'],'temperature':chosen['temperature'],'validation_logMc_MAE':float(abs(p@tf.LOG_CENTERS-vl).mean()),
        'seconds':time.perf_counter()-start,'sha256':dev.sha(out/'validation_selected_model.pt')};dev.json_write(out/'COMPLETE.json',st);print(json.dumps({'long_trained':st}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    for dep in e.DEPS:
        for ms,seed in zip(body.MODEL_SEEDS,(202609091,202609092,202609093)):train(a.root,dep,ms,seed)
