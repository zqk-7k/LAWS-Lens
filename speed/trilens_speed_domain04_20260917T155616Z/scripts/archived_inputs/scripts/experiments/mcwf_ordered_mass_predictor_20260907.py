#!/usr/bin/env python3
"""Ordered template-response posterior head on the unchanged peak2s data."""
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
from scipy.ndimage import gaussian_filter1d
import mcwf_pe_frontend_extension_v2_20260907 as e
import mcwf_fine_mass_features_20260907 as fine
import mcwf_additional_population_features_20260907 as population

dev,body=e.dev,e.body
torch.set_num_threads(2)
SEEDS=(202609381,202609382,202609383)
EPOCHS=30
LOG_CENTERS=fine.FINE_LOG_CENTERS
NATIVE_EDGES=np.r_[np.log(5),.5*(LOG_CENTERS[:-1]+LOG_CENTERS[1:]),np.log(200)]
EDGES=np.linspace(np.log(5),np.log(200),513)
CENTERS=.5*(EDGES[:-1]+EDGES[1:])


class Predictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.first=nn.Sequential(nn.Conv1d(28,48,9,padding=4),nn.GroupNorm(8,48),nn.SiLU())
        self.middle=nn.Sequential(nn.Conv1d(48,48,9,padding=8,dilation=2),nn.GroupNorm(8,48),nn.SiLU(),
                                  nn.Dropout(.1),nn.Conv1d(48,48,9,padding=16,dilation=4),nn.GroupNorm(8,48))
        self.last=nn.Conv1d(48,1,1)
        self.register_buffer('coordinate',torch.linspace(-1,1,253)[None,None,:])

    def forward(self,x):
        y=self.first(torch.cat([x,self.coordinate.expand(len(x),-1,-1)],1))
        return self.last(F.silu(y+self.middle(y))).squeeze(1)


def arrange(coarse,refined):
    if len(coarse)!=len(refined):
        raise RuntimeError('Feature row mismatch')
    n=len(coarse)
    a=np.asarray(coarse).reshape(n,3,64,3,3)
    b=np.asarray(refined).reshape(n,3,189,3,3)
    out=np.empty((n,3,253,3,3),np.float32)
    out[:,:,::4]=a
    out[:,:,np.arange(253)%4!=0]=b
    if not np.array_equal(out[:,:,::4],a):
        raise RuntimeError('Ordered mass feature indexing mismatch')
    return out.transpose(0,1,3,4,2).reshape(n,27,253).copy()


def data(root,dep,split):
    c=np.load(root/f'expanded_encoder/features/{dep}/{split}.npy',mmap_mode='r')
    f=np.load(root/f'fine_mass_context/features/{dep}/{split}.npy',mmap_mode='r')
    x=arrange(c,f)
    if split=='train':
        c=np.load(root/f'additional_population/expanded_encoder/features/{dep}/train.npy',mmap_mode='r')
        f=np.load(root/f'additional_population/features/{dep}/fine.npy',mmap_mode='r')
        x=np.concatenate([x,arrange(c,f)],0)
    return x,population.metadata(root,dep,split)


def targets(truth):
    t=np.clip(truth,LOG_CENTERS[0],LOG_CENTERS[-1])
    upper=np.searchsorted(LOG_CENTERS,t,side='right').clip(1,len(LOG_CENTERS)-1)
    lower=upper-1
    fraction=(t-LOG_CENTERS[lower])/(LOG_CENTERS[upper]-LOG_CENTERS[lower])
    y=np.zeros((len(t),len(LOG_CENTERS)),np.float32)
    y[np.arange(len(t)),lower]=1-fraction
    y[np.arange(len(t)),upper]=fraction
    if not np.allclose(y.sum(1),1) or np.max(abs(y@LOG_CENTERS-t))>1e-6:
        raise RuntimeError('Interpolated target mass conservation failed')
    return y


def probability(logits,temperature):
    native=softmax(logits.astype(float)/temperature,1)
    cdf=np.c_[np.zeros(len(native)),np.cumsum(native,1)]
    p=np.diff(np.stack([np.interp(EDGES,NATIVE_EDGES,row) for row in cdf]),axis=1).clip(1e-300)
    p/=p.sum(1,keepdims=True)
    return p,native[:,[0,-1]].sum(1)


@torch.no_grad()
def infer(model,x):
    model.eval();out=[]
    for start in range(0,len(x),512):
        with torch.autocast('cuda',dtype=torch.bfloat16):
            z=model(torch.as_tensor(x[start:start+512],dtype=torch.float32,device='cuda'))
        out.append(z.float().cpu().numpy())
    return np.concatenate(out)


def initialize(root):
    folder=root/'ordered_mass_predictor';path=folder/'contracts/TRAINING.json'
    if path.exists():return
    dev.json_write(path,{'utc':datetime.now(timezone.utc).isoformat(),'same_O3_O4a':True,
        'input':'same40-580Hz2s4096H1L1;existing576+1701event-PSD phase templates,27responsechannels x253 orderedlogMc centers;no new strain/templates',
        'architectural_hypothesis':'local convolution along mass coordinate retains neighboring-template structure rather than flattening all responses into an MLP',
        'not_a_pure_architecture_only_control':'uses a253bin mass CE objective rather than prior4Gaussian continuousNLL; architecture and predictive-density representation jointly change, report as one method variant',
        'model':'28->48conv9,GN8,SiLU;residualconv9dilation2/dropout.1/conv9dilation4;1x1masslogits',
        'data':'same12288training sourceparents/160noise blocks;512developmentparents/32noise blocks',
        'target':'linear interpolation onto two nearest logMc centers;outside5.145..194.2Msun targets clip to endpoints and boundary rate is reported;no artificially narrow predictive interval is imposed',
        'training':'AdamW2e-4,WD1e-4,30epochs128sources*2views,clip5,cosine_min1e-5',
        'seeds':SEEDS,'selection':'minimum development253binCE;temperature .5,.75,1,1.25,1.5,2 bydevelopmentCE',
        'normalization':'training channel mean/SD shared across mass coordinate;no catalog-dependent rescaling',
        'output':'probability-mass-conserving CDF conversion to512logmass bins;boundary mass proxy is not an estimate of true outside-support probability',
        'calibration':'samefinite-source mass tail and bounded isotonic BC/prior-overlap features,calibrated ondevelopment only',
        'frozen':['time','sky','outerweights','scope','historicaloutputs'],
        'no_PE_official_ID_inputs':True,'not_full_PE':True,'fresh_confirmation_required':True})
    (folder/'scripts').mkdir(exist_ok=True);shutil.copy2(__file__,folder/'scripts'/Path(__file__).name)


def train(root,dep,ms,seed):
    initialize(root);out=root/f'ordered_mass_predictor/models/{dep}/seed_{ms}'
    if (out/'COMPLETE.json').exists():return
    out.mkdir(parents=True,exist_ok=True)
    x,tm=data(root,dep,'train');v,vm=data(root,dep,'validation')
    if set(tm.source_uid)&set(vm.source_uid) or set(tm.noise_bank_index)&set(vm.noise_bank_index):
        raise RuntimeError('Source/noise overlap')
    group,names=pd.factorize(tm.source_uid,sort=True);vg,_=pd.factorize(vm.source_uid,sort=True)
    truth=np.log(tm.mc_det.to_numpy(float));vt=np.log(vm.mc_det.to_numpy(float))
    y,vy=targets(truth),targets(vt)
    mu=x.mean((0,2),keepdims=True);sd=x.std((0,2),keepdims=True).clip(.01)
    x-=mu;x/=sd;v=(v-mu)/sd
    dev.json_write(out/'DATA_AUDIT.json',{'source_overlap':0,'noise_overlap':0,'training_sources':len(names),
        'validation_sources':len(np.unique(vg)),'training_target_edge_fraction':float(((truth<LOG_CENTERS[0])|(truth>LOG_CENTERS[-1])).mean()),
        'validation_target_edge_fraction':float(((vt<LOG_CENTERS[0])|(vt>LOG_CENTERS[-1])).mean())})
    xx=torch.as_tensor(x,dtype=torch.float32,device='cuda');yy=torch.as_tensor(y,device='cuda')
    members=[np.flatnonzero(group==k) for k in range(len(names))]
    dev.TRAIN.seed_everything(seed);model=Predictor().cuda()
    opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,EPOCHS,eta_min=1e-5)
    history,best,first=[],float('inf'),1;started=time.perf_counter();resume=out/'resume.pt'
    if resume.exists():
        r=torch.load(resume,weights_only=False,map_location='cpu');model.load_state_dict(r['model']);opt.load_state_dict(r['optimizer']);sch.load_state_dict(r['scheduler'])
        history,best,first=r['history'],r['best'],r['epoch']+1;torch.set_rng_state(r['rng']);torch.cuda.set_rng_state_all(r['cuda_rng'])
    for epoch in range(first,EPOCHS+1):
        rng=np.random.default_rng(seed+epoch);order=rng.permutation(len(names));losses=[];model.train()
        for start in range(0,len(order),128):
            ids=np.stack([rng.choice(members[k],2,replace=False) for k in order[start:start+128]]).reshape(-1)
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                logit=model(xx[ids]);loss=-(F.log_softmax(logit.float(),-1)*yy[ids]).sum(-1).mean()
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite ordered mass loss')
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();losses.append(float(loss.detach()))
        sch.step();logits=infer(model,v);p=softmax(logits.astype(float),1)
        ce=float(-(vy*np.log(p.clip(1e-30))).sum(-1).mean())
        row={'epoch':epoch,'trainCE':float(np.mean(losses)),'validationCE':ce,'logMc_MAE':float(abs(p@LOG_CENTERS-vt).mean()),'seconds':time.perf_counter()-started}
        history.append(row)
        if ce<best:
            best=ce;torch.save({'model':{k:z.detach().cpu().clone() for k,z in model.state_dict().items()},'mu':mu,'sd':sd,
                               'epoch':epoch,'seed':seed,'CE':ce},out/'validation_selected_model.pt')
        torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'scheduler':sch.state_dict(),'history':history,'best':best,
                    'epoch':epoch,'rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()},resume)
        dev.csv_write(out/'history.csv',pd.DataFrame(history))
        if epoch==1 or epoch%10==0:print(json.dumps({'ordered_mass_training':[dep,seed],**row}),flush=True)
    ck=torch.load(out/'validation_selected_model.pt',weights_only=False,map_location='cpu');model.load_state_dict(ck['model']);logits=infer(model,v)
    grid=[]
    for temp in (.5,.75,1.,1.25,1.5,2.):
        p=softmax(logits.astype(float)/temp,1);grid.append({'temperature':temp,'CE':float(-(vy*np.log(p.clip(1e-30))).sum(-1).mean())})
    ck['temperature']=min(grid,key=lambda r:(r['CE'],abs(r['temperature']-1)))['temperature']
    p,boundary=probability(logits,ck['temperature'])
    prior=np.histogram(np.log(tm.drop_duplicates('source_uid').mc_det),EDGES)[0].astype(float)
    prior=gaussian_filter1d(prior,2,mode='nearest')+1e-3;prior/=prior.sum();ck['prior']=prior
    torch.save(ck,out/'validation_selected_model.pt')
    cdf=np.c_[np.zeros(len(p)),p.cumsum(1)]
    pit=np.array([np.interp(t,EDGES,c) for t,c in zip(vt,cdf)])
    np.savez_compressed(out/'validation_predictions.npz',p=p,outside=boundary,boundary_mass=boundary,group=vg,truth=vt)
    dev.csv_write(out/'temperature_grid.csv',pd.DataFrame(grid))
    report={'deployment':dep,'seed':seed,'epoch':ck['epoch'],'temperature':ck['temperature'],'logMc_MAE':float(abs(p@CENTERS-vt).mean()),
            'central90_coverage':float(((pit>=.05)&(pit<=.95)).mean()),'boundary_ood_rate':float((boundary>.25).mean()),
            'seconds':time.perf_counter()-started,'checkpoint_sha256':dev.sha(out/'validation_selected_model.pt')}
    dev.json_write(out/'COMPLETE.json',report);print(json.dumps({'ordered_mass_done':report}),flush=True)


def prediction(root,dep,ms,es,split):
    cp=root/f'ordered_mass_predictor/models/{dep}/seed_{ms}/validation_selected_model.pt'
    if not (cp.parent/'COMPLETE.json').exists():raise RuntimeError('Incomplete predictor')
    path=root/f'ordered_mass_predictor/predictions/{dep}/model_{ms}_eval_{es}/{split}.npz'
    if path.exists():
        a=np.load(path)
        if str(a['checkpoint_sha256'])!=dev.sha(cp):raise RuntimeError('Changed checkpoint')
        return a
    coarse=np.load(e.TRAINED/f'cache/deployment_event_psd/{dep}'/('real_features.npy' if split=='real' else f'{es}_{split}_features.npy'))
    refined=fine.deployment(root,dep,es,split);x=arrange(coarse,refined)
    ck=torch.load(cp,weights_only=False,map_location='cpu');model=Predictor().cuda().eval();model.load_state_dict(ck['model'])
    p,boundary=probability(infer(model,(x-ck['mu'])/ck['sd']),ck['temperature'])
    if split=='real':
        full,events=dev.real_inputs(dep);valid=events.strict_h1l1_preprocessing_pass.to_numpy(bool)
        pp,bb=np.full((len(full),512),np.nan),np.full(len(full),np.nan);pp[valid],bb[valid]=p,boundary;p,boundary=pp,bb
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,p=p,outside=boundary,boundary_mass=boundary,checkpoint_sha256=dev.sha(cp))
    return np.load(path)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    for dep in e.DEPS:
        for ms,seed in zip(body.MODEL_SEEDS,SEEDS):train(a.root,dep,ms,seed)
