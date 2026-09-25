#!/usr/bin/env python3
"""Mass-only predictive density from fine-mass two-detector waveform features."""
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
from scipy.ndimage import gaussian_filter1d
from scipy.special import ndtr,softmax
import mcwf_pe_frontend_extension_v2_20260907 as e
import mcwf_dense_features_20260907 as dense
import mcwf_fine_mass_density_20260907 as full_density
import mcwf_fine_mass_features_20260907 as fine
import mcwf_additional_population_features_20260907 as population

dev,body=e.dev,e.body
torch.set_num_threads(2)
SEEDS=(202609291,202609292,202609293)
EDGES=np.linspace(np.log(5),np.log(200),513)
CENTERS=.5*(EDGES[:-1]+EDGES[1:])
EPOCHS=20


class Predictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(15855,256),nn.LayerNorm(256),nn.SiLU(),nn.Dropout(.1),
            nn.Linear(256,128),nn.SiLU(),nn.Linear(128,12))

    def forward(self,x):
        a=self.net(x).reshape(-1,4,3)
        return a[...,0],a[...,1],F.softplus(a[...,2])+.03


def nll(params,truth,temperature=1.):
    w,m,s=params;s=s*np.sqrt(temperature)
    lp=-.5*((truth[:,None]-m)/s)**2-s.log()-.5*np.log(2*np.pi)
    return -torch.logsumexp(F.log_softmax(w,-1)+lp,-1).mean()


@torch.no_grad()
def infer(model,x,batch=1024):
    model.eval();out=[[],[],[]]
    for start in range(0,len(x),batch):
        a=model(torch.as_tensor(x[start:start+batch],dtype=torch.float32,device='cuda'))
        for output,values in zip(out,a):output.append(values.cpu().numpy())
    return tuple(np.concatenate(v) for v in out)


def development(root,dep,split):
    meta=population.metadata(root,dep,split)
    return population.inputs(root,dep,split),meta


def probability(params,ck,temperature):
    w,m,s=params;w=softmax(w.astype(float),1)
    m=m.astype(float)*ck['ys']+ck['ym'];s=s.astype(float)*ck['ys']*np.sqrt(temperature)
    cdf=(w[...,None]*ndtr((EDGES[None,None,:]-m[...,None])/s[...,None])).sum(1)
    p=np.diff(cdf,axis=1).clip(1e-300);retained=p.sum(1)
    p/=retained[:,None]
    return p,1-retained




def initialize(root):
    out=root/'large_population_mass_only';path=out/'contracts/TRAINING.json'
    if path.exists():return
    dev.json_write(path,{'created_utc':datetime.now(timezone.utc).isoformat(),'same_O3_O4a':True,
        'input':'unchanged H1L1peak2s4096points40-580Hz;15855 original+denseqspin+fineMcphasefeatures',
        'data':'12288train=4096old+8192newsourceparents;160trainnoiseblocks;frozen512developmentvalidationparentsand32noiseblocks;all historicaldataunchanged',
        'architecture':'15855->256LN/SiLU/dropout0.1->128SiLU->4Gaussian(logMc)components',
        'comparison':'samefineMcfeatures,3percentstandardizedscale_floor,mass-onlyNLL;12288/128*20=1920updates equals old4096/128*60;epochs20frozen before new fitting',
        'loss':'source-balanced1DcontinuousmassmixtureNLL',
        'training':'20epochs,128sources*2views,AdamW2e-4,WD1e-4,clip5,cosine_min1e-5',
        'selection':'earliestminimum512sourcevalidationNLL;globalvariance-temperaturebyvalidationNLL',
        'temperature_grid':[.5,.75,1,1.25,1.5,2],'seeds':SEEDS,'scale_floor':.03,
        'prior':'one histogram entry pertraining source;logMc5-200,512bins,smoothing2bins,pseudocount1e-3/bin',
        'interpretation':'waveform predictive density under simulator,not fullPE or a proper lensingBayesfactor',
        'frozen':['time','sky','outerweights','scope','historicalresults'],'no_PE_official_or_ID_inputs':True,
        'fresh_confirmation_required':True})
    (out/'scripts').mkdir(exist_ok=True);shutil.copy2(__file__,out/'scripts'/Path(__file__).name)


def train(root,dep,ms,seed):
    initialize(root);out=root/f'large_population_mass_only/models/{dep}/seed_{ms}'
    if (out/'COMPLETE.json').exists():return
    out.mkdir(parents=True,exist_ok=True);x,tm=development(root,dep,'train');v,vm=development(root,dep,'validation')
    if set(tm.source_uid)&set(vm.source_uid) or set(tm.noise_bank_index)&set(vm.noise_bank_index):raise RuntimeError('Splitoverlap')
    group,names=pd.factorize(tm.source_uid,sort=True);vg,vnames=pd.factorize(vm.source_uid,sort=True)
    mu,sd=x.mean(0),x.std(0).clip(1e-4);x,v=(x-mu)/sd,(v-mu)/sd
    truth=np.log(tm.mc_det.to_numpy(float));vt=np.log(vm.mc_det.to_numpy(float));ym,ys=truth.mean(),truth.std()
    yy=torch.as_tensor((truth-ym)/ys,dtype=torch.float32,device='cuda');vy=torch.as_tensor((vt-ym)/ys,dtype=torch.float32,device='cuda')
    xx=torch.as_tensor(x,device='cuda');vx=torch.as_tensor(v,device='cuda');members=[np.flatnonzero(group==g) for g in range(len(names))]
    dev.TRAIN.seed_everything(seed);model=Predictor().cuda();opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,EPOCHS,eta_min=1e-5);best=np.inf;history=[];start=time.perf_counter();first=1
    resume=out/'resume.pt'
    if resume.exists():
        r=torch.load(resume,weights_only=False,map_location='cpu');model.load_state_dict(r['model']);opt.load_state_dict(r['optimizer']);sch.load_state_dict(r['scheduler'])
        best,history,first=r['best'],r['history'],r['epoch']+1;torch.set_rng_state(r['rng']);torch.cuda.set_rng_state_all(r['cuda_rng'])
    for epoch in range(first,EPOCHS+1):
        model.train();rng=np.random.default_rng(seed+epoch);order=rng.permutation(len(names));losses=[]
        for begin in range(0,len(order),128):
            take=np.stack([rng.choice(members[g],2,replace=False) for g in order[begin:begin+128]]).reshape(-1)
            indices=take;opt.zero_grad(set_to_none=True);loss=nll(model(xx[indices]),yy[indices])
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite mass-only NLL')
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();losses.append(float(loss.detach()))
        sch.step();model.eval()
        with torch.no_grad():vn=float(nll(model(vx),vy))
        history.append({'epoch':epoch,'trainNLL':float(np.mean(losses)),'validationNLL':vn})
        if vn<best:
            best=vn;torch.save({'model':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                'mu':mu,'sd':sd,'ym':ym,'ys':ys,'epoch':epoch,'seed':seed,'NLL':vn},out/'validation_selected_model.pt')
        torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'scheduler':sch.state_dict(),'epoch':epoch,
            'history':history,'best':best,'rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()},resume)
        if epoch%20==0:print(json.dumps({'fine_mass_only_training':dep,'seed':seed,**history[-1]}),flush=True)
    ck=torch.load(out/'validation_selected_model.pt',weights_only=False,map_location='cpu');model.load_state_dict(ck['model']);model.eval()
    params=infer(model,v);grid=[]
    with torch.no_grad():
        pars=model(vx)
        for t in (.5,.75,1.,1.25,1.5,2.):grid.append({'temperature':t,'NLL':float(nll(pars,vy,t))})
    ck['temperature']=min(grid,key=lambda r:(r['NLL'],abs(r['temperature']-1)))['temperature']
    p,ood=probability(params,ck,ck['temperature'])
    unique=tm.drop_duplicates('source_uid');prior=np.histogram(np.log(unique.mc_det),bins=EDGES)[0].astype(float)
    prior=gaussian_filter1d(prior,2,mode='nearest')+1e-3;prior/=prior.sum();ck['prior']=prior
    torch.save(ck,out/'validation_selected_model.pt')
    np.savez_compressed(out/'validation_predictions.npz',p=p,outside=ood,group=vg,truth=vt)
    dev.csv_write(out/'history.csv',pd.DataFrame(history));dev.csv_write(out/'temperature_grid.csv',pd.DataFrame(grid))
    report={'deployment':dep,'seed':seed,'epoch':ck['epoch'],'temperature':ck['temperature'],
        'logMc_MAE':float(abs(p@CENTERS-vt).mean()),'seconds':time.perf_counter()-start,
        'checkpoint_sha256':dev.sha(out/'validation_selected_model.pt')}

    dev.json_write(out/'COMPLETE.json',report);print(json.dumps({'fine_mass_only_done':report}),flush=True)


def prediction(root,dep,ms,es,split):
    cp=root/f'large_population_mass_only/models/{dep}/seed_{ms}/validation_selected_model.pt'
    if not (cp.parent/'COMPLETE.json').exists():raise RuntimeError('Trainingincomplete')
    path=root/f'large_population_mass_only/predictions/{dep}/model_{ms}_eval_{es}/{split}.npz'
    if path.exists():
        a=np.load(path)
        if str(a['checkpoint_sha256'])!=dev.sha(cp):raise RuntimeError('Changedcheckpoint')
        return a
    ck=torch.load(cp,weights_only=False,map_location='cpu');prefix=e.TRAINED/f'cache/deployment_event_psd/{dep}'
    coarse=np.load(prefix/('real_features.npy' if split=='real' else f'{es}_{split}_features.npy'));extra=dense.deployment(root,dep,es,split)
    refined=fine.deployment(root,dep,es,split)
    x=np.concatenate([coarse.reshape(len(coarse),-1),extra.reshape(len(extra),-1),refined.reshape(len(refined),-1)],1)
    model=Predictor().cuda().eval();model.load_state_dict(ck['model'])
    p,ood=probability(infer(model,(x-ck['mu'])/ck['sd']),ck,ck['temperature'])
    a={'p':p,'outside':ood}

    if split=='real':
        full,events=dev.real_inputs(dep);valid=events.strict_h1l1_preprocessing_pass.to_numpy(bool);old=a
        a={k:np.full((len(full),*v.shape[1:]),np.nan) for k,v in old.items()}
        for k,v in old.items():a[k][valid]=v
    path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,**a,checkpoint_sha256=dev.sha(cp));return np.load(path)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    for dep in e.DEPS:
        population.run(a.root,dep)
        for ms,seed in zip(body.MODEL_SEEDS,SEEDS):train(a.root,dep,ms,seed)
