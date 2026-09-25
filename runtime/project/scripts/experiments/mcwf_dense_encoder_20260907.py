#!/usr/bin/env python3
"""RNC retraining with additional fine q/spin waveform template features."""
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
from torch import nn
import torch.nn.functional as F
from scipy.special import softmax
import mcwf_pe_frontend_extension_v2_20260907 as e
import mcwf_dense_features_20260907 as feat
import mcwf_rankncontrast_20260907 as rnc
import mcwf_mass_tf_20260905 as tf

dev,body=e.dev,e.body
torch.set_num_threads(2)
EPOCHS=50
EXTRA=3*len(feat.EXTRA)


def model():
    m=body.Encoder('RAW-PHASE-SOURCE');m.phase.layers[1]=nn.Linear(1728+EXTRA,256);return m


def initialize(root):
    feat.initialize(root);path=root/'dense_context/contracts/TRAINING.json'
    if path.exists():return
    dev.json_write(path,{'created_utc':datetime.now(timezone.utc).isoformat(),
        'same_O3_O4a':True,'single_controlled_change':'finer q/spin template features added to original phaseMLP; same2s rawbranch and rest of network',
        'input_size':1728+EXTRA,'warm_start':'original coarse weights copied, new template weights zero,epoch0eligible; original RNC weights otherwise unchanged at initialization',
        'loss':'CE_Mc+sourceSupCon+RNC(logMc)','selection':'earliest minimum validationCE+0.2SupCon+0.2RNC',
        'optimizer':'same AdamW1e-4,weightdecay1e-4,cosine_min1e-5,50epochs,4passes,64sources*2views,clip5',
        'new_seeds':[202609121,202609122,202609123],'normalization':'original coarse train mean/SDfrozen; additional feature train mean/SDfloor0.05',
        'real_PE_and_official_used_in_training':False,'new_information_not_guaranteed':True,
        'frozen':['time','sky','outerweights','scope','old model files','historical outputs']})
    shutil.copy2(__file__,root/'dense_context/scripts'/Path(__file__).name)


def arrays(root,dep,ck,split):
    old=np.load(e.TRAINED/f'cache/finelag/{dep}/{split}_features.npy')
    extra=np.load(root/f'dense_context/features/{dep}/{split}.npy')
    x=np.concatenate([((old-ck['mu'])/ck['sd']).reshape(len(old),-1),((extra-ck['extra_mu'])/ck['extra_sd']).reshape(len(old),-1)],1)
    return x,np.load(e.TRAINED/f'cache/{dep}/{split}_raw2s.npy',mmap_mode='r')


def train(root,dep,ms,seed):
    initialize(root);out=root/f'dense_context/models/{dep}/seed_{ms}'
    if (out/'COMPLETE.json').exists():return
    out.mkdir(parents=True,exist_ok=True)
    for split in ('validation','train'):feat.development(root,dep,split)
    cp=e.TRAINED/f'models/RAW-PHASE-SOURCE/{dep}/seed_{ms}/validation_selected_model.pt';ck=torch.load(cp,weights_only=False,map_location='cpu')
    extra=np.load(root/f'dense_context/features/{dep}/train.npy');ck['extra_mu']=extra.mean(0);ck['extra_sd']=extra.std(0).clip(.05);del extra
    dev.TRAIN.seed_everything(seed);net=model().cuda();state=ck['model'].copy();w=state['phase.layers.1.weight'];state['phase.layers.1.weight']=torch.cat([w,torch.zeros((256,EXTRA),dtype=w.dtype)],1);net.load_state_dict(state)
    x,raw=arrays(root,dep,ck,'train');vx,vr=arrays(root,dep,ck,'validation')
    original=body.Encoder('RAW-PHASE-SOURCE').cuda().eval();original.load_state_dict(ck['model']);net.eval()
    with torch.no_grad():
        rr=torch.as_tensor(np.array(vr[:4],np.float32),device='cuda');xx=torch.as_tensor(vx[:4],dtype=torch.float32,device='cuda')
        p0,z0=original(xx[:,:1728],rr);p1,z1=net(xx,rr)
        identity_error=max(float((p0-p1).abs().max()),float((z0-z1).abs().max()))
    del original
    if identity_error>1e-4:raise RuntimeError('Dense warm-start identity failed')
    dev.json_write(out/'WARM_IDENTITY.json',{'float32_max_abs_error':identity_error,'pass':True})
    meta=pd.read_parquet(e.TRAINED/f'cache/{dep}/train_metadata.parquet');vm=pd.read_parquet(e.TRAINED/f'cache/{dep}/validation_metadata.parquet')
    group,names=pd.factorize(meta.waveform_parent_uid,sort=True);vg,_=pd.factorize(vm.waveform_parent_uid,sort=True);members=[np.flatnonzero(group==k) for k in range(len(names))]
    labels=np.log(meta.chirp_mass_detector.to_numpy(float));vl=np.log(vm.chirp_mass_detector.to_numpy(float))
    y=np.load(body.PREVIOUS/f'cache/masstf/{dep}/train_targets.npy');vy=np.load(body.PREVIOUS/f'cache/masstf/{dep}/validation_targets.npy')
    opt=torch.optim.AdamW(net.parameters(),lr=1e-4,weight_decay=1e-4);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,EPOCHS,eta_min=1e-5)
    best=float('inf');history=[];first=0;start=time.perf_counter();resume=out/'resume.pt'
    if resume.exists():
        r=torch.load(resume,weights_only=False,map_location='cpu');net.load_state_dict(r['model']);opt.load_state_dict(r['optimizer']);sched.load_state_dict(r['scheduler'])
        best,history,first=r['best'],r['history'],r['epoch']+1;torch.set_rng_state(r['rng']);torch.cuda.set_rng_state_all(r['cuda_rng'])
    for epoch in range(first,EPOCHS+1):
        tick=time.perf_counter();losses=[]
        if epoch>0:
            net.train();rng=np.random.default_rng(seed+epoch)
            for _ in range(4):
                order=rng.permutation(len(names))
                for begin in range(0,len(order),64):
                    sources=order[begin:begin+64];ids=np.stack([rng.choice(members[k],2,replace=False) for k in sources]).T.reshape(-1)
                    xx=torch.as_tensor(x[ids],dtype=torch.float32,device='cuda');rr=torch.as_tensor(np.asarray(raw[ids]),dtype=torch.float32,device='cuda');opt.zero_grad(set_to_none=True)
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        logits,z=net(xx,rr);ce=-(F.log_softmax(logits.float(),-1)*torch.as_tensor(y[ids],device='cuda')).sum(-1).mean()
                        sc=body.source_contrastive(z,torch.as_tensor(np.tile(sources,2),device='cuda'));rank=rnc.rank_loss(z,torch.as_tensor(labels[ids],device='cuda'));loss=ce+sc+rank
                    if not torch.isfinite(loss):raise RuntimeError('Nonfinite dense training loss')
                    loss.backward();nn.utils.clip_grad_norm_(net.parameters(),5);opt.step();losses.append(float(loss.detach()))
            sched.step()
        logits,z=body.infer(net,vx,vr);p=softmax(logits.astype(float),1)
        ce=float(-(vy*np.log(p.clip(1e-30))).sum(-1).mean());sc=float(body.source_contrastive(torch.as_tensor(z,device='cuda'),torch.as_tensor(vg,device='cuda')))
        rank=float(rnc.rank_loss(torch.as_tensor(z,device='cuda'),torch.as_tensor(vl,device='cuda')));criterion=ce+.2*sc+.2*rank
        row={'epoch':epoch,'train_loss':float(np.mean(losses)) if losses else None,'CE':ce,'SupCon':sc,'RNC':rank,'criterion':criterion,'logMc_MAE':float(abs(p@tf.LOG_CENTERS-vl).mean()),'seconds':time.perf_counter()-tick};history.append(row)
        if criterion<best:
            best=criterion;torch.save({**ck,'model':{k:v.detach().cpu().clone() for k,v in net.state_dict().items()},'epoch':epoch,'criterion':criterion,
                'feature_implementation':'dense_q_spin_peak2s_v1','new_seed':seed,'warm_sha256':dev.sha(cp)},out/'validation_selected_model.pt')
        torch.save({'model':net.state_dict(),'optimizer':opt.state_dict(),'scheduler':sched.state_dict(),'history':history,'best':best,'epoch':epoch,'rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()},resume)
        dev.csv_write(out/'history.csv',pd.DataFrame(history))
        if epoch%10==0:print(json.dumps({'dense_training':dep,'seed':seed,**row}),flush=True)
    chosen=torch.load(out/'validation_selected_model.pt',weights_only=False,map_location='cpu');net.load_state_dict(chosen['model']);logits,z=body.infer(net,vx,vr)
    grid=[]
    for t in (.5,.75,1.,1.25,1.5,2.,3.,4.):
        p=softmax(logits.astype(float)/t,1);grid.append({'temperature':t,'CE':float(-(vy*np.log(p.clip(1e-30))).sum(-1).mean())})
    chosen['temperature']=min(grid,key=lambda r:(r['CE'],abs(r['temperature']-1)))['temperature'];torch.save(chosen,out/'validation_selected_model.pt')
    p=softmax(logits.astype(float)/chosen['temperature'],1);np.savez_compressed(out/'validation_predictions.npz',p=p,z=z,group=vg,truth=vy)
    dev.csv_write(out/'temperature_grid.csv',pd.DataFrame(grid))
    result={'deployment':dep,'seed':seed,'selected_epoch':chosen['epoch'],'temperature':chosen['temperature'],'Mc_MAE':float(abs(p@tf.LOG_CENTERS-vl).mean()),'seconds':time.perf_counter()-start,'sha256':dev.sha(out/'validation_selected_model.pt')}
    dev.json_write(out/'COMPLETE.json',result);print(json.dumps({'dense_trained':result}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    for dep in e.DEPS:
        for ms,seed in zip(body.MODEL_SEEDS,(202609121,202609122,202609123)):train(a.root,dep,ms,seed)
