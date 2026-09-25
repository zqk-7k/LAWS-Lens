#!/usr/bin/env python3
"""Condition the unchanged mass-response model on the pre-window signal scale."""
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
import torch
from torch import nn
import torch.nn.functional as F
from scipy.special import softmax

P=Path('/root/autodl-tmp/gw-catalog');sys.path.insert(0,str(P/'scripts/experiments'))
import mcwf_temporal_response_20260908 as parent
import mcwf_temporal_response_evaluate_20260908 as ev

dev,old,cf=parent.dev,parent.old,ev.cf
PREVIOUS,PAIRS,EXTERNAL=parent.PREVIOUS,parent.PAIRS,parent.EXTERNAL
DEPS,SEEDS,MODEL_SLOTS,TRAIN_SEEDS=parent.DEPS,parent.SEEDS,parent.MODEL_SLOTS,parent.TRAIN_SEEDS
CONTROL=P/'results/mcwf_temporal_response_exploratory_20260908T142000Z'
STATUS=parent.STATUS
torch.set_num_threads(2)


def initialize(root):
    if root.exists():raise RuntimeError('Independent fresh directory required')
    for name in ('contracts','scripts','logs','models','features','predictions','calibration','evaluation','tables','reports','manifest'):
        (root/name).mkdir(parents=True)
    dev.json_write(root/'contracts/ANALYSIS_CONTRACT.json',{
        'id':'MCWF-NOISE-SCALE-02','utc':datetime.now(timezone.utc).isoformat(),'status':STATUS,
        'goal_achieved':False,'baseline':str(PAIRS),'matched_continued_training_control':str(CONTROL),
        'same_both_runs':True,'mechanism':'Recover two detector-window scale descriptors removed by the additional peak2s normalization; no temporal-response residuals added',
        'input':'unchanged27x253 ordered template response features plus log(std(x_H)),log(std(x_L)),each broadcast along mass axis',
        'x_definition':'existing full24s offsource-PSD whitened,bandpassed,antialias-resampled and robust-MAD-scaled data,then peak2s4096crop,before extra per2s normalization',
        'meaning':'Relative signal/noise-scale descriptors,NOT PSD-optimal SNR,not PE network_optimal_snr,not magnification ratios',
        'provenance':'physical_common.preprocess_24s ends with robust_scale_channels; fine_mass_features.features standardizes each2s window again',
        'training':'Same12288sourceparents/160noiseblocks,512developmentsources/32noiseblocks;30inputchannels incl coordinate;warm oldOMC,extra weightszero;15epochs AdamW1e-4;128sources*2views',
        'seeds':TRAIN_SEEDS,'selection':'sim-development CE including epoch0;same temperature grid as temporal control',
        'OOD':'Scale outside development per-detector min/max OR boundary predictive mass>.25 neutralizes the new pair increment;limits frozen without real inputs',
        'calibration':'same source/noise-half development fit and audit as temporal-control;validation-only coefficients;same candidate/retrieval priorities and coefficient grid',
        'frozen':['time','sky','outer C-fixed weights','scope','old encoder and predictor checkpoints','history','paper'],
        'real_labels_as_input':False,'real_feedback':'Adaptive development,not blind test;official candidates not lens truth',
        'target':'same joint Top10/20 PE/Mc+official nondegradation and gain requirements as MCWF-TEMPORAL-RESPONSE-01; independent new injection confirmation required',
        'not_added':'No new time,sky,SNR-ratio channel,PE parameter input,real-label training,template frequency-operator change or hard veto',
        'references':['https://arxiv.org/abs/gr-qc/0509116','https://arxiv.org/abs/2106.12594'],
        'reference_scope':'Motivation for noise-conditioned waveform inference; this simple conditioning model is not full PE or DINGO',
        'limitations':'SNR and population proposal shortcuts must be audited; existing test reused;no claim that scale descriptors alone measure intrinsic mass'})
    dev.csv_write(root/'manifest/INPUT_SHA256.csv',pd.DataFrame(parent.protected()))
    shutil.copy2(__file__,root/'scripts/noise_scale.py')
    dev.json_write(root/'contracts/FREEZE.json',{'contract_sha256':dev.sha(root/'contracts/ANALYSIS_CONTRACT.json'),
        'code_sha256':dev.sha(root/'scripts/noise_scale.py')})


def scales(root,dep,split,seed=0):
    path=root/f'features/{dep}'/(f'{seed}_{split}.npy' if seed else f'{split}.npy')
    if path.exists():return np.load(path)
    if split in ('development','train'):
        part='validation' if split=='development' else 'train'
        paths=[PREVIOUS/f'expanded_data/{dep}/{part}/raw2s.npy']
        if split=='train':paths.append(PREVIOUS/f'additional_population/expanded_data/{dep}/train/raw2s.npy')
        blocks=[]
        for p in paths:
            raw=np.load(p,mmap_mode='r')
            for start in range(0,len(raw),1024):
                x=np.asarray(raw[start:start+1024],np.float32)
                if not np.isfinite(x).all():raise RuntimeError('Nonfinite input')
                blocks.append(np.log(x.std(-1,ddof=1).clip(1e-6)))
        result=np.concatenate(blocks)
    else:
        if split=='real':
            full,events=dev.real_inputs(dep);full=full[events.strict_h1l1_preprocessing_pass.to_numpy(bool)]
        else:
            plan=dev.BASE.retained_event_plan(dep,seed,split)
            full=dev.ORCH.event_array_for_plan(dep,seed,split,plan)
        x=dev.TRAIN.make_window_view(np.asarray(full,np.float32),2)
        if not np.isfinite(x).all():raise RuntimeError('Nonfinite strict waveform')
        result=np.log(x.std(-1,ddof=1).clip(1e-6))
    path.parent.mkdir(parents=True,exist_ok=True);np.save(path,result.astype(np.float32))
    dev.json_write(path.with_suffix('.sha256.json'),{'sha256':dev.sha(path),'shape':list(result.shape)})
    return result


class Predictor(old.Predictor):
    def __init__(self):
        super().__init__();self.first[0]=nn.Conv1d(30,48,9,padding=4)


def concatenate(x,s):
    if len(x)!=len(s):raise RuntimeError('Scale rows not aligned')
    return np.concatenate([x,np.broadcast_to(s[:,:,None],(len(s),2,253))],1)


def train(root,dep,slot,seed):
    out=root/f'models/SCALE/{dep}/seed_{slot}'
    if (out/'COMPLETE.json').exists():return
    out.mkdir(parents=True,exist_ok=True)
    source=PREVIOUS/f'ordered_mass_predictor/models/{dep}/seed_{slot}/validation_selected_model.pt'
    ck=torch.load(source,map_location='cpu',weights_only=False)
    x,tm=old.data(PREVIOUS,dep,'train');v,vm=old.data(PREVIOUS,dep,'validation')
    s=scales(root,dep,'train');vs=scales(root,dep,'development')
    mu=np.concatenate([ck['mu'],s.mean(0)[None,:,None]],1)
    sd=np.concatenate([ck['sd'],s.std(0).clip(.01)[None,:,None]],1)
    x=(concatenate(x,s)-mu)/sd;v=(concatenate(v,vs)-mu)/sd
    if set(tm.source_uid)&set(vm.source_uid) or set(tm.noise_bank_index)&set(vm.noise_bank_index):raise RuntimeError('Source/noise overlap')
    group,names=pd.factorize(tm.source_uid,sort=True);vg,_=pd.factorize(vm.source_uid,sort=True)
    members=[np.flatnonzero(group==k) for k in range(len(names))]
    yt=old.targets(np.log(tm.mc_det.to_numpy(float)));truth=np.log(vm.mc_det.to_numpy(float));yv=old.targets(truth)
    dev.TRAIN.seed_everything(seed);model=Predictor().cuda()
    state={k:v.clone() for k,v in ck['model'].items()};w=torch.zeros_like(model.first[0].weight,device='cpu')
    w[:,:27]=state['first.0.weight'][:,:27];w[:,-1]=state['first.0.weight'][:,-1];state['first.0.weight']=w
    model.load_state_dict(state)
    xx=torch.as_tensor(x,device='cuda');yy=torch.as_tensor(yt,device='cuda');del x
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,15,eta_min=1e-5)
    history=[];best=float('inf');first=0;start=time.perf_counter()
    if (out/'resume.pt').exists():
        a=torch.load(out/'resume.pt',map_location='cpu',weights_only=False)
        model.load_state_dict(a['model']);optimizer.load_state_dict(a['optimizer']);scheduler.load_state_dict(a['scheduler'])
        history,best,first=a['history'],a['best'],a['epoch']+1
        torch.set_rng_state(a['rng']);torch.cuda.set_rng_state_all(a['cuda_rng'])
    for epoch in range(first,16):
        losses=[]
        if epoch:
            rng=np.random.default_rng(seed+epoch);order=rng.permutation(len(names));model.train()
            for k in range(0,len(order),128):
                ids=np.concatenate([rng.choice(members[g],2,replace=False) for g in order[k:k+128]])
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    z=model(xx[ids]);loss=-(F.log_softmax(z.float(),-1)*yy[ids]).sum(-1).mean()
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
                loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step();losses.append(float(loss.detach()))
            scheduler.step()
        logits=old.infer(model,v);p=softmax(logits.astype(float),1)
        ce=float(-(yv*np.log(p.clip(1e-30))).sum(-1).mean())
        row={'epoch':epoch,'validationCE':ce,'trainCE':float(np.mean(losses)) if losses else None,
             'logMc_MAE':float(abs(p@old.LOG_CENTERS-truth).mean()),'seconds':time.perf_counter()-start}
        history.append(row)
        if ce<best:
            best=ce
            torch.save({'model':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},'mu':mu,'sd':sd,
                        'prior':ck['prior'],'kind':'SCALE','epoch':epoch,'seed':seed,'slot':slot,
                        'scale_min':vs.min(0),'scale_max':vs.max(0)},out/'selected.pt')
        torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
                    'history':history,'epoch':epoch,'best':best,'rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()},out/'resume.pt')
        dev.csv_write(out/'history.csv',pd.DataFrame(history));print(json.dumps({'SCALE':[dep,seed],**row}),flush=True)
    c=torch.load(out/'selected.pt',map_location='cpu',weights_only=False);model.load_state_dict(c['model']);logits=old.infer(model,v)
    grid=[]
    for temp in (.5,.75,1.,1.25,1.5,2.):
        p=softmax(logits.astype(float)/temp,1);grid.append({'temperature':temp,'CE':float(-(yv*np.log(p.clip(1e-30))).sum(-1).mean())})
    c['temperature']=min(grid,key=lambda r:(r['CE'],abs(r['temperature']-1)))['temperature'];torch.save(c,out/'selected.pt')
    p,boundary=old.probability(logits,c['temperature'])
    np.savez_compressed(out/'development_predictions.npz',p=p,outside=boundary,group=vg,truth=truth)
    cdf=np.c_[np.zeros(len(p)),p.cumsum(1)]
    pit=np.array([np.interp(z,old.EDGES,row) for z,row in zip(truth,cdf)])
    dev.csv_write(out/'temperature_grid.csv',pd.DataFrame(grid))
    dev.json_write(out/'COMPLETE.json',{'epoch':c['epoch'],'temperature':c['temperature'],'logMc_MAE':float(abs(p@old.CENTERS-truth).mean()),
        'central90coverage':float(((pit>=.05)&(pit<=.95)).mean()),'training_sources':len(names),'development_sources':len(np.unique(vg)),
        'sha256':dev.sha(out/'selected.pt'),'seconds':time.perf_counter()-start})


def predict(root,dep,kind,slot,es,split):
    path=root/f'predictions/SCALE/{dep}/model_{slot}_eval_{es}/{split}.npz'
    if path.exists():return np.load(path)
    cp=root/f'models/SCALE/{dep}/seed_{slot}/selected.pt';ck=torch.load(cp,map_location='cpu',weights_only=False)
    x=parent.old_features(dep,es,split);s=scales(root,dep,split,es)
    model=Predictor().cuda().eval();model.load_state_dict(ck['model'])
    p,boundary=old.probability(old.infer(model,(concatenate(x,s)-ck['mu'])/ck['sd']),ck['temperature'])
    scale_ood=((s<ck['scale_min']-1e-6)|(s>ck['scale_max']+1e-6)).any(1)
    outside=np.where(scale_ood,1.,boundary)
    if split=='real':
        full,events=dev.real_inputs(dep);valid=events.strict_h1l1_preprocessing_pass.to_numpy(bool)
        pp,oo=np.full((len(full),512),np.nan),np.full(len(full),np.nan);pp[valid],oo[valid]=p,outside;p,outside=pp,oo
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,p=p,outside=outside,checkpoint_sha256=dev.sha(cp))
    return np.load(path)


def select(root):
    if (root/'contracts/INTEGRATION_FROZEN.json').exists():raise RuntimeError('Already frozen')
    choices=[];grid=[];audit=[]
    for dep in DEPS:
        for slot,es in zip(MODEL_SLOTS,SEEDS):
            spec,rows=ev.simulated_calibration(root,dep,'SCALE',slot);audit+=rows
            frame,penalty,increment,x=ev.pair_values(root,dep,'SCALE',slot,es,'validation',spec)
            w=cf.frozen_weights(dep,es);baseline=frame.waveform_score.to_numpy(float)
            bm=cf.fast_metrics(frame,cf.channels(frame,baseline)@w);bwm=cf.fast_metrics(frame,baseline);rows=[]
            for gamma in ev.COEFFICIENTS:
                for beta in ev.COEFFICIENTS:
                    score=baseline+gamma*penalty+beta*increment
                    m=cf.fast_metrics(frame,cf.channels(frame,score)@w);wm=cf.fast_metrics(frame,score)
                    good=cf.guard(m,bm) and cf.guard(wm,bwm)
                    row={'gamma':gamma,'beta':beta,'pass':good,**m};rows.append(row)
                    grid.append({'deployment':dep,'seed':es,**row,**{'waveform_'+k:v for k,v in wm.items()}})
            for policy in ('CANDIDATE','RETRIEVAL'):
                def key(r):
                    m=(r['false_at_recall_0p5'],r['false_at_recall_0p9'],-r['average_precision'],-r['macro_r_at_10'],-r['macro_r_at_1']) if policy=='CANDIDATE' else (-r['macro_r_at_10'],-r['macro_r_at_1'],-r['average_precision'],r['false_at_recall_0p5'],r['false_at_recall_0p9'])
                    return (*m,r['gamma']**2+r['beta']**2,r['gamma'],r['beta'])
                win=min((r for r in rows if r['pass']),key=key)
                choices.append({'deployment':dep,'kind':'SCALE','slot':slot,'seed':es,'method':'SCALE-'+policy,
                                'gamma':win['gamma'],'beta':win['beta'],'weights':w.tolist(),'calibration':spec})
    dev.csv_write(root/'tables/VALIDATION_GRID.csv',pd.DataFrame(grid));dev.csv_write(root/'tables/CALIBRATION_AUDIT.csv',pd.DataFrame(audit))
    dev.csv_write(root/'tables/SELECTED_COEFFICIENTS.csv',pd.DataFrame([{k:v for k,v in c.items() if k not in ('weights','calibration')} for c in choices]))
    dev.json_write(root/'calibration/SELECTED.json',choices)
    dev.json_write(root/'contracts/INTEGRATION_FROZEN.json',{'UTC':datetime.now(timezone.utc).isoformat(),
        'file':'calibration/SELECTED.json','sha256':dev.sha(root/'calibration/SELECTED.json'),'test_real_used_to_select':False})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--stage',choices=['initialize','train','select','evaluate','real','assess'],required=True)
    args=parser.parse_args()
    # Reuse the frozen evaluator with this module's prediction API,not its files or training code.
    ev.t=sys.modules[__name__]
    if args.stage=='initialize':initialize(args.root)
    elif args.stage=='train':
        for dep in DEPS:
            for slot,seed in zip(MODEL_SLOTS,TRAIN_SEEDS):train(args.root,dep,slot,seed)
    elif args.stage=='select':select(args.root)
    else:getattr(ev,args.stage)(args.root)
