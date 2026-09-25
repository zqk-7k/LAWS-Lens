#!/usr/bin/env python3
"""Matched small-population control and multirate auxiliary mass predictors."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='2'
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import sys
import time
import numpy as np
import pandas as pd
from scipy.special import softmax
import torch
from torch import nn
import torch.nn.functional as F

P=Path('/root/autodl-tmp/gw-catalog')
sys.path.insert(0,str(P/'scripts/experiments'))
import mcwf_multirate_features_20260908 as m
import mcwf_multirate_deployment_20260908 as deployment
import mcwf_temporal_response_evaluate_20260908 as ev
t,old,dev,cf=m.t,m.old,m.dev,ev.cf
KINDS=('SMALL-CONTROL','MULTIRATE')
torch.set_num_threads(2)


class Predictor(old.Predictor):
    def __init__(self,kind):
        super().__init__()
        if kind=='MULTIRATE':self.first[0]=nn.Conv1d(55,48,9,padding=4)


def warm(kind,cp,device='cuda'):
    model=Predictor(kind).to(device)
    state={k:v.clone() for k,v in cp['model'].items()}
    if kind=='MULTIRATE':
        w=torch.zeros_like(model.first[0].weight,device='cpu')
        w[:,:27]=state['first.0.weight'][:,:27];w[:,-1]=state['first.0.weight'][:,-1]
        state['first.0.weight']=w
    model.load_state_dict(state)
    return model


def training_data(root,dep,split,kind):
    c=np.load(t.PREVIOUS/f'expanded_encoder/features/{dep}/{split}.npy',mmap_mode='r')
    f=np.load(t.PREVIOUS/f'fine_mass_context/features/{dep}/{split}.npy',mmap_mode='r')
    x=old.arrange(c,f)
    meta=pd.read_parquet(t.PREVIOUS/f'expanded_data/{dep}/{split}/event_metadata.parquet')
    if kind=='MULTIRATE':x=np.concatenate([x,np.load(root/f'features/{dep}/{split}.npy',mmap_mode='r')],1)
    if len(x)!=len(meta):raise RuntimeError('Data/model feature mapping')
    return x,meta


def train_one(root,dep,kind,slot,seed):
    out=root/f'models/{kind}/{dep}/seed_{slot}'
    if (out/'COMPLETE.json').exists():return
    out.mkdir(parents=True,exist_ok=True)
    cp=torch.load(t.PREVIOUS/f'ordered_mass_predictor/models/{dep}/seed_{slot}/validation_selected_model.pt',map_location='cpu',weights_only=False)
    x,tm=training_data(root,dep,'train',kind);v,vm=training_data(root,dep,'validation',kind)
    if set(tm.source_uid)&set(vm.source_uid) or set(tm.noise_bank_index)&set(vm.noise_bank_index):raise RuntimeError('Source/noise leakage')
    mu,sd=cp['mu'].copy(),cp['sd'].copy()
    if kind=='MULTIRATE':
        mu=np.concatenate([mu,x[:,27:].mean((0,2),keepdims=True)],1)
        sd=np.concatenate([sd,x[:,27:].std((0,2),keepdims=True).clip(.01)],1)
    x=(x-mu)/sd;v=(v-mu)/sd
    group,names=pd.factorize(tm.source_uid,sort=True)
    members=[np.flatnonzero(group==i) for i in range(len(names))]
    vg,_=pd.factorize(vm.source_uid,sort=True)
    truth=np.log(tm.mc_det.to_numpy(float));vt=np.log(vm.mc_det.to_numpy(float))
    y,vy=old.targets(truth),old.targets(vt)
    dev.TRAIN.seed_everything(seed);model=warm(kind,cp)
    if kind=='MULTIRATE':
        before=old.Predictor().double().eval();before.load_state_dict(cp['model'])
        after=warm(kind,cp,'cpu').double().eval()
        with torch.no_grad():
            test=torch.as_tensor(v[:4],dtype=torch.float64)
            difference=float((before(test[:,:27])-after(test)).abs().max())
        if difference>1e-10:raise RuntimeError('Warm-start changed original outputs')
        dev.json_write(out/'WARM_START_TEST.json',{'pass':True,'max_logit_difference':difference})
        del before,after
    xx=torch.as_tensor(x,device='cuda');yy=torch.as_tensor(y,device='cuda');del x
    opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,15,eta_min=1e-5)
    best=float('inf');history=[];first=0;started=time.perf_counter()
    if (out/'resume.pt').exists():
        r=torch.load(out/'resume.pt',map_location='cpu',weights_only=False)
        model.load_state_dict(r['model']);opt.load_state_dict(r['optimizer']);sch.load_state_dict(r['scheduler'])
        history,best,first=r['history'],r['best'],r['epoch']+1
        torch.set_rng_state(r['rng']);torch.cuda.set_rng_state_all(r['cuda_rng'])
    for epoch in range(first,16):
        losses=[]
        if epoch:
            rng=np.random.default_rng(seed+epoch);order=rng.permutation(len(names));model.train()
            for start in range(0,len(order),128):
                ids=np.concatenate([rng.choice(members[k],2,replace=False) for k in order[start:start+128]])
                opt.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    logits=model(xx[ids]);loss=-(F.log_softmax(logits.float(),-1)*yy[ids]).sum(-1).mean()
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
                loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();losses.append(float(loss.detach()))
            sch.step()
        logits=old.infer(model,v);prob=softmax(logits.astype(float),1)
        ce=float(-(vy*np.log(prob.clip(1e-30))).sum(-1).mean())
        row={'epoch':epoch,'validationCE':ce,'trainCE':float(np.mean(losses)) if losses else None,
             'logMc_MAE':float(abs(prob@old.LOG_CENTERS-vt).mean()),'seconds':time.perf_counter()-started}
        history.append(row)
        if ce<best:
            best=ce
            torch.save({'model':{k:z.detach().cpu().clone() for k,z in model.state_dict().items()},'mu':mu,'sd':sd,
                'kind':kind,'seed':seed,'slot':slot,'epoch':epoch,'CE':ce,'prior':cp['prior']},out/'selected.pt')
        torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'scheduler':sch.state_dict(),
            'history':history,'best':best,'epoch':epoch,'rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()},out/'resume.pt')
        dev.csv_write(out/'history.csv',pd.DataFrame(history));print(json.dumps({'training':[dep,kind,seed],**row}),flush=True)
    ck=torch.load(out/'selected.pt',map_location='cpu',weights_only=False);model.load_state_dict(ck['model'])
    logits=old.infer(model,v);grid=[]
    for temperature in (.5,.75,1.,1.25,1.5,2.):
        prob=softmax(logits.astype(float)/temperature,1)
        grid.append({'temperature':temperature,'CE':float(-(vy*np.log(prob.clip(1e-30))).sum(-1).mean())})
    ck['temperature']=min(grid,key=lambda r:(r['CE'],abs(r['temperature']-1)))['temperature'];torch.save(ck,out/'selected.pt')
    prob,boundary=old.probability(logits,ck['temperature'])
    np.savez_compressed(out/'development_predictions.npz',p=prob,outside=boundary,group=vg,truth=vt)
    cdf=np.c_[np.zeros(len(prob)),prob.cumsum(1)];pit=np.array([np.interp(a,old.EDGES,b) for a,b in zip(vt,cdf)])
    dev.csv_write(out/'temperature_grid.csv',pd.DataFrame(grid))
    dev.json_write(out/'COMPLETE.json',{'epoch':ck['epoch'],'temperature':ck['temperature'],
        'logMc_MAE':float(abs(prob@old.CENTERS-vt).mean()),'central90coverage':float(((pit>=.05)&(pit<=.95)).mean()),
        'training_sources':len(names),'development_sources':len(np.unique(vg)),'source_noise_overlap':0,
        'sha256':dev.sha(out/'selected.pt'),'seconds':time.perf_counter()-started})


def predict(root,dep,kind,slot,es,split):
    path=root/f'predictions/{kind}/{dep}/model_{slot}_eval_{es}/{split}.npz'
    if path.exists():return np.load(path)
    cp=root/f'models/{kind}/{dep}/seed_{slot}/selected.pt';ck=torch.load(cp,map_location='cpu',weights_only=False)
    x=t.old_features(dep,es,split)
    if kind=='MULTIRATE':x=np.concatenate([x,np.load(deployment.deployment_features(root,dep,es,split))],1)
    model=Predictor(kind).cuda().eval();model.load_state_dict(ck['model'])
    prob,ood=old.probability(old.infer(model,(x-ck['mu'])/ck['sd']),ck['temperature'])
    if split=='real':
        full,events=dev.real_inputs(dep);valid=events.strict_h1l1_preprocessing_pass.to_numpy(bool)
        pp,oo=np.full((len(full),512),np.nan),np.full(len(full),np.nan)
        pp[valid],oo[valid]=prob,ood;prob,ood=pp,oo
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,p=prob,outside=ood,checkpoint_sha256=dev.sha(cp))
    return np.load(path)


def select(root):
    if (root/'contracts/INTEGRATION_FROZEN.json').exists():raise RuntimeError('Already frozen')
    choices=[];grid=[];audits=[]
    for dep in t.DEPS:
        for kind in KINDS:
            for slot,es in zip(t.MODEL_SLOTS,t.SEEDS):
                spec,audit=ev.simulated_calibration(root,dep,kind,slot);audits+=audit
                f,penalty,increment,_=ev.pair_values(root,dep,kind,slot,es,'validation',spec)
                weights=cf.frozen_weights(dep,es);base=f.waveform_score.to_numpy(float)
                bm,bwm=cf.fast_metrics(f,cf.channels(f,base)@weights),cf.fast_metrics(f,base)
                rows=[]
                for gamma in ev.COEFFICIENTS:
                    for beta in ev.COEFFICIENTS:
                        z=base+gamma*penalty+beta*increment
                        metrics,wm=cf.fast_metrics(f,cf.channels(f,z)@weights),cf.fast_metrics(f,z)
                        row={'gamma':gamma,'beta':beta,'pass':cf.guard(metrics,bm) and cf.guard(wm,bwm),**metrics}
                        rows.append(row);grid.append({'deployment':dep,'kind':kind,'seed':es,**row,**{'waveform_'+k:v for k,v in wm.items()}})
                for policy in ('CANDIDATE','RETRIEVAL'):
                    def key(r):
                        primary=(r['false_at_recall_0p5'],r['false_at_recall_0p9'],-r['average_precision'],-r['macro_r_at_10'],-r['macro_r_at_1']) if policy=='CANDIDATE' else (-r['macro_r_at_10'],-r['macro_r_at_1'],-r['average_precision'],r['false_at_recall_0p5'],r['false_at_recall_0p9'])
                        return (*primary,r['gamma']**2+r['beta']**2,r['gamma'],r['beta'])
                    win=min((r for r in rows if r['pass']),key=key)
                    choices.append({'deployment':dep,'kind':kind,'slot':slot,'seed':es,'method':kind+'-'+policy,
                        'gamma':win['gamma'],'beta':win['beta'],'weights':weights.tolist(),'calibration':spec})
    dev.csv_write(root/'tables/VALIDATION_GRID.csv',pd.DataFrame(grid));dev.csv_write(root/'tables/CALIBRATION_AUDIT.csv',pd.DataFrame(audits))
    dev.csv_write(root/'tables/SELECTED_COEFFICIENTS.csv',pd.DataFrame([{k:v for k,v in r.items() if k not in ('weights','calibration')} for r in choices]))
    dev.json_write(root/'calibration/SELECTED.json',choices)
    dev.json_write(root/'contracts/INTEGRATION_FROZEN.json',{'utc':datetime.now(timezone.utc).isoformat(),
        'file':'calibration/SELECTED.json','sha256':dev.sha(root/'calibration/SELECTED.json'),'real_or_test_selection':False})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--stage',choices=['train','select','evaluate','real','assess'],required=True);a=parser.parse_args()
    t.predict=predict
    if a.stage=='train':
        target=a.root/'scripts/multirate_train.py'
        if target.exists() and dev.sha(target)!=dev.sha(Path(__file__)):raise RuntimeError('Frozen training source changed')
        if not target.exists():shutil.copy2(__file__,target)
        for dep in t.DEPS:
            for kind in KINDS:
                for slot,seed in zip(t.MODEL_SLOTS,t.TRAIN_SEEDS):train_one(a.root,dep,kind,slot,seed)
    elif a.stage=='select':select(a.root)
    else:getattr(ev,a.stage)(a.root)
