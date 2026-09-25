#!/usr/bin/env python3
"""Symmetric simulation-trained pair verifier on frozen waveform latents."""
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
from sklearn.isotonic import IsotonicRegression
import mcwf_pe_frontend_extension_v2_20260907 as e
import mcwf_pe_frontend_qprobe_20260907 as probe

dev,body,ev=e.dev,e.body,e.ev
torch.set_num_threads(2)
EPOCHS=50
BETAS=(0.,.0625,.125,.25,.5,1.,2.)


class Verifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers=nn.Sequential(nn.Linear(704,128),nn.SiLU(),nn.Dropout(.15),nn.Linear(128,32),nn.SiLU(),nn.Linear(32,1))

    def forward(self,a,b):
        return self.layers(torch.cat([(a-b).abs(),a*b],-1)).squeeze(-1)


def initialize(root):
    out=root/'pair_verifier';path=out/'contracts/TRAINING.json'
    if path.exists():return
    out.mkdir(parents=True,exist_ok=True)
    dev.json_write(path,{'created_utc':datetime.now(timezone.utc).isoformat(),
        'same_both_runs':True,'frozen_encoder':'current RNC raw2s+phase model;352 phase/raw/merged hidden features',
        'network':'symmetric [abs(h_i-h_j),h_i*h_j]704->128SiLU dropout0.15->32SiLU->1logit',
        'training_sources':960,'fit_views_per_source':8,'new_seeds':[202609111,202609112,202609113],
        'pairs':'each source4positive independent-view pairs/epoch;one negative per positive,half uniform,half sampled among32 nearest different-source true simulated logMc neighbors',
        'labels':'simulation source identity only; never real PE/FPP/Hanabi/eventidentity',
        'optimization':'AdamWlr0.001,weightdecay0.01,50epochs,cosine_min1e-5,batch512,clip5',
        'selection':'minimum balanced-class BCE on all240-source encoder validation pairs,earliest tie',
        'calibration':'isotonic on selected logits with equal positive/negative totalweight; finite-source floor; cap+-4;positiveOOD0',
        'scoring':'Z_FRT+beta*verifier_increment;beta from BAYESTAR simulation validationonly with existing guardrails and priority',
        'negative_mixture_correction':'training hard-negative mixture is not the catalog null; calibrate on uniform non-companion validation pairs, not training logits',
        'limitation':'encoder-validation sources reused for early stopping/calibration; exploratory development, not an independent PE likelihood or physicalBF',
        'no_fourth_channel':'increment stays inside waveform score;time,sky andouterC-fixedweights remain frozen',
        'references':['https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.loss.BCEWithLogitsLoss.html'],
        'engineering_not_literature_defaults':True})
    (out/'scripts').mkdir(exist_ok=True);shutil.copy2(__file__,out/'scripts'/Path(__file__).name)


@torch.no_grad()
def infer(model,x,i,j):
    model.eval();x=torch.as_tensor(x,dtype=torch.float32,device='cuda');out=[]
    for start in range(0,len(i),2048):
        a=torch.as_tensor(i[start:start+2048],device='cuda');b=torch.as_tensor(j[start:start+2048],device='cuda')
        out.append(model(x[a],x[b]).cpu().numpy())
    return np.concatenate(out)


def train(root,dep,ms,seed):
    initialize(root);out=root/f'pair_verifier/models/{dep}/seed_{ms}'
    if (out/'COMPLETE.json').exists():return
    out.mkdir(parents=True,exist_ok=True)
    trainpath=root/f'qprobe/models/{dep}/seed_{ms}/train_hidden.npy'
    tx=np.load(trainpath);vx=np.load(root/f'qprobe/models/{dep}/seed_{ms}/validation_hidden.npy')
    mu,sd=tx.mean(0),tx.std(0).clip(.05);tx=(tx-mu)/sd;vx=(vx-mu)/sd
    meta=pd.read_parquet(e.TRAINED/f'cache/{dep}/train_metadata.parquet');vm=pd.read_parquet(e.TRAINED/f'cache/{dep}/validation_metadata.parquet')
    groups,names=pd.factorize(meta.waveform_parent_uid,sort=True)
    if set(meta.waveform_parent_uid)&set(vm.waveform_parent_uid):raise RuntimeError('Train/validation source overlap')
    members=[np.flatnonzero(groups==k) for k in range(len(names))]
    masses=np.array([np.log(meta.chirp_mass_detector.iloc[ids[0]]) for ids in members]);distance=abs(masses[:,None]-masses[None,:]);np.fill_diagonal(distance,np.inf)
    neighbors=np.argsort(distance,axis=1,kind='stable')[:,:32]
    vi,vj=np.triu_indices(len(vx),1);vg=vm.waveform_parent_uid.to_numpy();vy=(vg[vi]==vg[vj]);vweights=np.where(vy,.5/vy.sum(),.5/(~vy).sum())
    dev.TRAIN.seed_everything(seed);net=Verifier().cuda();opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.01);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,EPOCHS,eta_min=1e-5)
    x=torch.as_tensor(tx,dtype=torch.float32,device='cuda');history=[];best=float('inf');start=time.perf_counter()
    resume=out/'resume.pt';first=1
    if resume.exists():
        r=torch.load(resume,weights_only=False,map_location='cpu');net.load_state_dict(r['model']);opt.load_state_dict(r['optimizer']);sched.load_state_dict(r['scheduler'])
        history,best,first=r['history'],r['best'],r['epoch']+1;torch.set_rng_state(r['rng']);torch.cuda.set_rng_state_all(r['cuda_rng'])
    for epoch in range(first,EPOCHS+1):
        rng=np.random.default_rng(seed+epoch);a=[];b=[];labels=[]
        for _ in range(4):
            for g in rng.permutation(len(names)):
                u,v=rng.choice(members[g],2,replace=False);a.append(u);b.append(v);labels.append(1.)
                if rng.random()<.5:other=int(rng.choice(neighbors[g]))
                else:other=(g+int(rng.integers(1,len(names))))%len(names)
                if other==g:raise RuntimeError('False negative source contamination')
                a.append(u);b.append(rng.choice(members[other]));labels.append(0.)
        a=np.array(a);b=np.array(b);labels=np.array(labels,np.float32);order=rng.permutation(len(a));net.train();losses=[]
        for begin in range(0,len(a),512):
            ids=order[begin:begin+512];opt.zero_grad(set_to_none=True);logits=net(x[a[ids]],x[b[ids]])
            loss=F.binary_cross_entropy_with_logits(logits,torch.as_tensor(labels[ids],device='cuda'))
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite pair loss')
            loss.backward();nn.utils.clip_grad_norm_(net.parameters(),5);opt.step();losses.append(float(loss.detach()))
        sched.step();vl=infer(net,vx,vi,vj);ce=float(np.sum(vweights*(np.logaddexp(0.,vl)-vy*vl)))
        history.append({'epoch':epoch,'train_BCE':float(np.mean(losses)),'validation_balanced_BCE':ce,'seconds_elapsed':time.perf_counter()-start})
        if ce<best:
            best=ce;torch.save({'model':{k:v.detach().cpu().clone() for k,v in net.state_dict().items()},'mu':mu,'sd':sd,'epoch':epoch,'seed':seed,'validation_BCE':ce},out/'validation_selected_model.pt')
        torch.save({'model':net.state_dict(),'optimizer':opt.state_dict(),'scheduler':sched.state_dict(),'history':history,'best':best,'epoch':epoch,
                    'rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()},resume)
        dev.csv_write(out/'history.csv',pd.DataFrame(history))
    ck=torch.load(out/'validation_selected_model.pt',weights_only=False,map_location='cpu');net.load_state_dict(ck['model']);vl=infer(net,vx,vi,vj)
    iso=IsotonicRegression(increasing=True,out_of_bounds='clip').fit(vl,vy.astype(float),sample_weight=vweights)
    floor=1/(vy.sum()+2);prob=np.clip(iso.y_thresholds_,floor,1-floor)
    cal={'knots':iso.X_thresholds_.tolist(),'loglr':(np.log(prob)-np.log1p(-prob)).tolist(),'fit_min':float(vl.min()),'fit_max':float(vl.max()),'floor':float(floor)}
    dev.json_write(out/'CALIBRATION.json',cal);np.savez_compressed(out/'validation_logits.npz',logits=vl,idx_i=vi,idx_j=vj,is_true_pair=vy)
    net.eval()
    with torch.no_grad():
        error=float((net(x[:5],x[5:10])-net(x[5:10],x[:5])).abs().max())
    if error>1e-7:raise RuntimeError('Pair swap asymmetry')
    result={'deployment':dep,'seed':seed,'selected_epoch':ck['epoch'],'balanced_BCE':ck['validation_BCE'],'symmetry_error':error,
            'seconds':time.perf_counter()-start,'train_feature_sha256':dev.sha(trainpath),'model_sha256':dev.sha(out/'validation_selected_model.pt')}
    dev.json_write(out/'COMPLETE.json',result);print(json.dumps({'pair_verifier_trained':result}),flush=True)


def raw_score(root,dep,ms,es,split):
    f=pd.read_parquet(e.BASE/f'evaluation/{dep}/seed_{es}/{split}_pairs.parquet')
    path=root/f'pair_verifier/predictions/{dep}/model_{ms}_eval_{es}/{split}.npz'
    if path.exists():
        a=np.load(path)
        if not np.array_equal(a['idx_i'],f.idx_i) or not np.array_equal(a['idx_j'],f.idx_j):raise RuntimeError('Pair cache ordering changed')
        return f,a['logits']
    if split=='real':
        full,events=dev.real_inputs(dep);valid=events.strict_h1l1_preprocessing_pass.to_numpy(bool);values=np.asarray(full[valid])
        feat=e.TRAINED/f'cache/deployment_event_psd/{dep}/real_features.npy'
    else:
        events=dev.BASE.retained_event_plan(dep,es,split);values=dev.ORCH.event_array_for_plan(dep,es,split,events)
        feat=e.TRAINED/f'cache/deployment_event_psd/{dep}/{es}_{split}_features.npy'
    hidden=probe.hidden_features(dep,ms,np.load(feat),dev.TRAIN.make_window_view(np.asarray(values,np.float32),2))
    if split=='real':
        x=np.full((len(full),352),np.nan);x[valid]=hidden;hidden=x
    ck=torch.load(root/f'pair_verifier/models/{dep}/seed_{ms}/validation_selected_model.pt',weights_only=False,map_location='cpu')
    net=Verifier().cuda().eval();net.load_state_dict(ck['model']);x=(hidden-ck['mu'])/ck['sd']
    raw=infer(net,x,f.idx_i.to_numpy(int),f.idx_j.to_numpy(int))
    if not np.isfinite(raw).all():raise RuntimeError('Invalid waveform pair features')
    path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,logits=raw,idx_i=f.idx_i,idx_j=f.idx_j)
    return f,raw


def score(f,raw,cal):
    val=np.interp(raw,cal['knots'],cal['loglr']);ood=(raw<cal['fit_min'])|(raw>cal['fit_max']);val=np.clip(np.where(ood,np.minimum(val,0),val),-4,4)
    return f.waveform_score.to_numpy(float)+cal['beta']*val,val,ood


def evaluate(root):
    trial=root/'trials/SYMMETRIC-PAIR-VERIFIER'
    if trial.exists():raise RuntimeError('Independent trial required')
    for n in ('contracts','calibration','tables','results','evaluation'):(trial/n).mkdir(parents=True)
    shutil.copy2(__file__,trial/'contracts'/Path(__file__).name)
    dev.json_write(trial/'contracts/RECIPE.json',{'training_contract_sha256':dev.sha(root/'pair_verifier/contracts/TRAINING.json'),
        'beta_grid':BETAS,'same_both_runs':True,'selection':'existing validation priority and FRT-relative guardrails, real not opened to select beta'})
    selected={};statuses=[];cache={}
    for dep in e.DEPS:
        for ms,es in zip(body.MODEL_SEEDS,dev.SEEDS):
            f,raw=raw_score(root,dep,ms,es,'validation');cache[dep,es]=(f,raw)
            cal=json.loads((root/f'pair_verifier/models/{dep}/seed_{ms}/CALIBRATION.json').read_text());bm=ev.metrics(f,f.waveform_score.to_numpy(float),dep,es)
            rows=[];choices=[]
            for beta in BETAS:
                cfg={**cal,'beta':beta};z,_,ood=score(f,raw,cfg);m=ev.metrics(f,z,dep,es);ok=ev.guard(m,bm)
                rows.append({'beta':beta,'pass':ok,'ood':float(ood.mean()),**{a+'_'+k:v for a,b in m.items() for k,v in b.items()}})
                if ok:choices.append((e.old_selection.objective(m,beta,1),cfg))
            out=trial/f'calibration/{dep}/seed_{es}';out.mkdir(parents=True);dev.csv_write(out/'validation_grid.csv',pd.DataFrame(rows))
            cfg=min(choices,key=lambda c:c[0])[1];selected[dep,es]=cfg;dev.json_write(out/'SELECTED_CONFIG.json',cfg)
            statuses.append({'deployment':dep,'seed':es,'beta':cfg['beta']})
    dev.json_write(trial/'contracts/SELECTED_FREEZE.json',{'configs':statuses,'real_used_for_beta_selection':False});print(json.dumps({'verifier_selected':statuses}),flush=True)
    rows=[];guards=[]
    for dep in e.DEPS:
        real={}
        for ms,es in zip(body.MODEL_SEEDS,dev.SEEDS):
            for split in ('validation','test','real'):
                f,raw=cache[dep,es] if split=='validation' else raw_score(root,dep,ms,es,split)
                z,val,ood=score(f,raw,selected[dep,es]);n=f.copy();n['FRT_baseline_waveform_score']=f.waveform_score;n['waveform_score']=z
                n['verifier_raw']=raw;n['verifier_increment']=val;n['verifier_ood']=ood
                out=trial/f'evaluation/{dep}/seed_{es}';out.mkdir(parents=True,exist_ok=True);n.to_parquet(out/f'{split}_pairs.parquet',index=False)
                if split=='real':real[es]=n
                else:
                    bm=ev.metrics(f,f.waveform_score.to_numpy(float),dep,es);m=ev.metrics(f,z,dep,es)
                    guards.append({'deployment':dep,'seed':es,'split':split,'pass':ev.guard(m,bm)})
                    for method in bm:
                        for config,metrics in [('FRT_BASELINE',bm[method]),('CANDIDATE',m[method])]:
                            rows.append({'deployment':dep,'seed':es,'split':split,'method':method,'config':config,**metrics})
        dev.save_evaluation(trial,'CANDIDATE',dep,real,pd.read_parquet(root/f'audit/{dep}_frozen_external_reference.parquet'))
    dev.csv_write(trial/'tables/RETRIEVAL_PER_SEED.csv',pd.DataFrame(rows));dev.csv_write(trial/'tables/REUSED_GUARDRAILS.csv',pd.DataFrame(guards));e.assess(root,trial)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    for dep in e.DEPS:
        for ms,seed in zip(body.MODEL_SEEDS,(202609111,202609112,202609113)):train(a.root,dep,ms,seed)
    evaluate(a.root)
