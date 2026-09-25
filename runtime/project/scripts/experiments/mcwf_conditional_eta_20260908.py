#!/usr/bin/env python3
"""Conditional symmetric-mass-ratio evidence with a frozen mass predictor."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='2'
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.special import softmax
import torch
from torch import nn
import torch.nn.functional as F

P=Path('/root/autodl-tmp/gw-catalog')
sys.path.insert(0,str(P/'scripts/experiments'))
import mcwf_multirate_train_20260908 as mult
import mcwf_temporal_response_evaluate_20260908 as ev
t,old,dev,cf=mult.t,mult.old,mult.dev,ev.cf
UPSTREAM=P/'results/mcwf_multirate_lowband_exploratory_20260908T145551Z'
ETA=np.linspace(.05,.25,16)
KIND='CONDITIONAL-ETA'


def targets(value,centers):
    value=np.clip(value,centers[0],centers[-1]);hi=np.searchsorted(centers,value,side='right').clip(1,len(centers)-1);lo=hi-1
    w=(value-centers[lo])/(centers[hi]-centers[lo])
    out=np.zeros((len(value),len(centers)),np.float32);out[np.arange(len(out)),lo]=1-w;out[np.arange(len(out)),hi]=w
    if np.max(abs(out@centers-value))>1e-6:raise RuntimeError('Soft-target first moment failure')
    return out


class Head(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Linear(103,96),nn.SiLU(),nn.Dropout(.1),nn.Linear(96,96),nn.SiLU(),nn.Linear(96,16))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
    def forward(self,x):return self.net(x)


@torch.no_grad()
def representations(x,checkpoint,truth=None,batch=128):
    backbone=mult.Predictor('MULTIRATE').cuda().eval();backbone.load_state_dict(checkpoint['model'])
    coordinate=backbone.coordinate
    outputs=[]
    if truth is not None:
        weights=targets(truth,old.LOG_CENTERS);hi=np.searchsorted(old.LOG_CENTERS,truth,side='right').clip(1,252);lo=hi-1
    for start in range(0,len(x),batch):
        raw=torch.as_tensor(np.array(x[start:start+batch],np.float32),device='cuda')
        z=(raw-torch.as_tensor(checkpoint['mu'],device='cuda'))/torch.as_tensor(checkpoint['sd'],device='cuda')
        y=backbone.first(torch.cat([z,coordinate.expand(len(z),-1,-1)],1))
        hidden=F.silu(y+backbone.middle(y))
        rep=torch.cat([z,hidden,coordinate.expand(len(z),-1,-1)],1).transpose(1,2)
        if truth is None:outputs.append(rep.cpu().numpy())
        else:
            rows=torch.arange(len(z),device='cuda');l=torch.as_tensor(lo[start:start+len(z)],device='cuda');h=l+1
            w=torch.as_tensor(weights[start:start+len(z)][np.arange(len(z)),hi[start:start+len(z)]],device='cuda')[:,None]
            outputs.append(((1-w)*rep[rows,l]+w*rep[rows,h]).cpu().numpy())
    return np.concatenate(outputs)


def prior_conditional(meta):
    sources=meta.drop_duplicates('source_uid')
    q=sources.m2_det.to_numpy()/sources.m1_det.to_numpy();eta=q/(1+q)**2
    ym=targets(np.log(sources.mc_det.to_numpy()),old.LOG_CENTERS);yq=targets(eta,ETA)
    counts=gaussian_filter(ym.T@yq,sigma=(2.,1.),mode='nearest')+1e-6
    return counts/counts.sum(1,keepdims=True)


def interpolate_rows(values,masses,centers=old.LOG_CENTERS):
    hi=np.searchsorted(centers,masses,side='right').clip(1,len(centers)-1);lo=hi-1
    w=((masses-centers[lo])/(centers[hi]-centers[lo])).clip(0,1)
    return (1-w[:,None])*values[lo]+w[:,None]*values[hi]


def initialize(root):
    if root.exists():raise RuntimeError('Fresh independent root required')
    for folder in ('contracts','scripts','logs','models','predictions','calibration','evaluation','tables','reports','manifest','figures','cache'):(root/folder).mkdir(parents=True)
    dev.json_write(root/'contracts/ANALYSIS_CONTRACT.json',{
        'id':'MCWF-CONDITIONAL-ETA-08','UTC':datetime.now(timezone.utc).isoformat(),'status':t.STATUS,'goal_achieved':False,
        'same_both_runs':True,'upstream':str(UPSTREAM),'why':'A precise chirp mass does not determine mass ratio;use additional PN waveform-shape information without allowing the new head to change Mc predictions',
        'parameter':'eta=q/(1+q)^2,16binsfrom0.05to0.25;source truth only fromsimulation',
        'model':'freeze MULTIRATE mass backbone;train103-to96-to96-to16conditional eta residual head,25epochs;no gradientbackbone;zero final layer starts at conditional training-population prior',
        'features':'54standardized responsechannels+48frozenhidden+masscoordinate,interpolated at trueMc only fortraining and evaluated at every mass grid point forprediction',
        'training':'4096sourceparents and96noiseblocks/8views;512developmentparents/32noiseblocks;uniformsourcebatches2views;AdamW1e-3WD1e-4,clip5,cosine1e-5,minimumdevelopmentCE includingepoch0',
        'seeds':[202609881,202609882,202609883],
        'temperature_grid':[.5,.75,1.,1.25,1.5,2.],'prior':'one weight/source;Gaussian-smoothed253x16 histogram sigma=(2,1),floor1e-6;conditional prior marginalized exactly to frozenMc prior',
        'joint_prediction':'p(Mc,eta|x)=frozen_p(Mc|x)*p_new(eta|Mc,x);eta marginalization must recover frozenMc to1e-6',
        'extra_feature':'log(sum p_i(M,eta)p_j(M,eta)/(prior_M prior_eta_given_M)) -log(sum p_i(M)p_j(M)/prior_M)',
        'unit_test':'uninformative eta=conditionalprior gives extra_feature=0;mass marginal unchanged;prior normalization',
        'interpretation':'predictive-common-parameter-overlap proxy,not truePE Bayesfactor;calibratedonlyonsimulation,notindependent fourthchannel',
        'integration':'retained OMC waveform+beta*boundedconditional-eta increment;no new standaloneMc positive bonus;time/sky/outerweights fixed',
        'selection':'samecandidateandretrievalsimulatedvalidationpriorities;beta0,.0625,.125,.25,.5,1,2,4;guardsbothwaveform/fusion;zero option',
        'calibration':'source/noise-disjoint development halves,isotonic ratio minus its valueat0,cap4nats;positive OOD ->0;newMc conflict at finite-source tail<.05 prohibitspositiveeta support',
        'external':'adaptive real-catalog development;PE/official appendedonlyafterselection;no official labels,PE,eventID rankinginput',
        'frozen':['mass checkpoints','OMC baseline','one-dimensionaltime','sky_raw_log_bf','outerweights','scope','history','paper'],
        'references':['https://arxiv.org/abs/gr-qc/9402014','https://arxiv.org/abs/1612.01474'],
        'reference_limits':'physicalparametercovariance andpredictiveuncertainty motivation;not proof ofourapproximation'})
    protected=t.protected()+[{'path':str(p),'sha256':dev.sha(p),'bytes':p.stat().st_size} for p in (UPSTREAM/'models/MULTIRATE').glob('*/*/selected.pt')]
    dev.csv_write(root/'manifest/INPUT_SHA256.csv',pd.DataFrame(protected));shutil.copy2(__file__,root/'scripts/conditional_eta.py')
    dev.json_write(root/'contracts/FREEZE.json',{'contract_sha256':dev.sha(root/'contracts/ANALYSIS_CONTRACT.json'),'code_sha256':dev.sha(Path(__file__))})
    rng=np.random.default_rng(202609880)
    p=rng.dirichlet(np.ones(512),size=8);prior=rng.dirichlet(np.ones(512));cond=rng.dirichlet(np.ones(16),size=512)
    joint=p[:,:,None]*cond
    a,b=np.triu_indices(len(p),1)
    numerator=(joint[a]*joint[b]/(prior[:,None]*cond)).sum((1,2));denominator=(p[a]*p[b]/prior).sum(1)
    error=float(abs(np.log(numerator)-np.log(denominator)).max())
    if error>1e-12:raise RuntimeError('Uninformative eta supplies nonzero evidence')
    dev.json_write(root/'contracts/ETA_ALGEBRA_TEST.json',{'pass':True,'uninformative_conditional_log_overlap_max_abs':error,
        'mass_marginal_max_abs':float(abs(joint.sum(-1)-p).max())})


@torch.no_grad()
def infer(head,x,batch=2048):
    head.eval();out=[]
    for start in range(0,len(x),batch):out.append(head(torch.as_tensor(x[start:start+batch],device='cuda')).cpu().numpy())
    return np.concatenate(out)


def train_one(root,dep,slot,seed):
    folder=root/f'models/{KIND}/{dep}/seed_{slot}'
    if (folder/'COMPLETE.json').exists():return
    folder.mkdir(parents=True,exist_ok=True)
    cp_path=UPSTREAM/f'models/MULTIRATE/{dep}/seed_{slot}/selected.pt';before=dev.sha(cp_path)
    cp=torch.load(cp_path,map_location='cpu',weights_only=False)
    train,tm=mult.training_data(UPSTREAM,dep,'train','MULTIRATE');val,vm=mult.training_data(UPSTREAM,dep,'validation','MULTIRATE')
    if set(tm.source_uid)&set(vm.source_uid) or set(tm.noise_bank_index)&set(vm.noise_bank_index):raise RuntimeError('Source/noise overlap')
    trlog=np.log(tm.mc_det.to_numpy(float));vlog=np.log(vm.mc_det.to_numpy(float))
    x=representations(train,cp,trlog);v=representations(val,cp,vlog);del train,val
    mu=x.mean(0,keepdims=True);sd=x.std(0,keepdims=True).clip(.01);x=(x-mu)/sd;v=(v-mu)/sd
    prior=prior_conditional(tm)
    tx=np.log(interpolate_rows(prior,trlog).clip(1e-12));vx=np.log(interpolate_rows(prior,vlog).clip(1e-12))
    tq=tm.m2_det.to_numpy()/tm.m1_det.to_numpy();vq=vm.m2_det.to_numpy()/vm.m1_det.to_numpy()
    y,vy=targets(tq/(1+tq)**2,ETA),targets(vq/(1+vq)**2,ETA)
    group,names=pd.factorize(tm.source_uid,sort=True);members=[np.flatnonzero(group==i) for i in range(len(names))]
    vg,_=pd.factorize(vm.source_uid,sort=True)
    dev.TRAIN.seed_everything(seed);head=Head().cuda()
    xx,yy,tt=[torch.as_tensor(a,device='cuda') for a in (x,y,tx)]
    opt=torch.optim.AdamW(head.parameters(),lr=1e-3,weight_decay=1e-4);sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,25,eta_min=1e-5)
    history=[];best=float('inf');started=time.perf_counter()
    for epoch in range(26):
        losses=[]
        if epoch:
            head.train();rng=np.random.default_rng(seed+epoch);order=rng.permutation(len(names))
            for start in range(0,len(order),128):
                ids=np.concatenate([rng.choice(members[k],2,replace=False) for k in order[start:start+128]])
                opt.zero_grad(set_to_none=True);logits=head(xx[ids])+tt[ids]
                loss=-(F.log_softmax(logits,-1)*yy[ids]).sum(-1).mean()
                loss.backward();nn.utils.clip_grad_norm_(head.parameters(),5);opt.step();losses.append(float(loss.detach()))
            sch.step()
        residual=infer(head,v);prob=softmax(residual+vx,-1)
        ce=float(-(vy*np.log(prob.clip(1e-30))).sum(-1).mean())
        row={'epoch':epoch,'validation_conditional_eta_CE':ce,'eta_MAE':float(abs(prob@ETA-vq/(1+vq)**2).mean()),'seconds':time.perf_counter()-started}
        history.append(row)
        if ce<best:
            best=ce
            torch.save({'model':{k:z.detach().cpu().clone() for k,z in head.state_dict().items()},'mu':mu,'sd':sd,'conditional_eta_prior':prior,
                'prior':cp['prior'],'epoch':epoch,'CE':ce,'seed':seed,'mass_checkpoint':str(cp_path),'mass_checkpoint_sha256':before},folder/'selected.pt')
        dev.csv_write(folder/'history.csv',pd.DataFrame(history));print(json.dumps({'eta_training':[dep,slot],**row}),flush=True)
    ck=torch.load(folder/'selected.pt',map_location='cpu',weights_only=False);head.load_state_dict(ck['model']);residual=infer(head,v);grid=[]
    for temp in (.5,.75,1.,1.25,1.5,2.):
        prob=softmax(residual/temp+vx,-1);grid.append({'temperature':temp,'CE':float(-(vy*np.log(prob.clip(1e-30))).sum(-1).mean())})
    ck['temperature']=min(grid,key=lambda a:(a['CE'],abs(a['temperature']-1)))['temperature'];torch.save(ck,folder/'selected.pt')
    dev.csv_write(folder/'temperature_grid.csv',pd.DataFrame(grid))
    if dev.sha(cp_path)!=before:raise RuntimeError('Frozen mass backbone changed')
    dev.json_write(folder/'COMPLETE.json',{'epoch':ck['epoch'],'CE':ck['CE'],'prior_CE':history[0]['validation_conditional_eta_CE'],
        'temperature':ck['temperature'],'training_sources':len(names),'development_sources':len(np.unique(vg)),
        'source_noise_overlap':0,'backbone_hash_unchanged':True,'sha256':dev.sha(folder/'selected.pt'),'seconds':time.perf_counter()-started})


def prediction(root,dep,slot,es,split):
    path=root/f'predictions/{dep}/{slot}_{es}_{split}.npz'
    if path.exists():return np.load(path)
    folder=root/f'models/{KIND}/{dep}/seed_{slot}';ck=torch.load(folder/'selected.pt',map_location='cpu',weights_only=False)
    cp=torch.load(ck['mass_checkpoint'],map_location='cpu',weights_only=False)
    if dev.sha(Path(ck['mass_checkpoint']))!=ck['mass_checkpoint_sha256']:raise RuntimeError('Mass checkpoint hash')
    if split=='development':
        x,meta=mult.training_data(UPSTREAM,dep,'validation','MULTIRATE')
        mass=dict(np.load(Path(ck['mass_checkpoint']).parent/'development_predictions.npz'))
    else:
        x=np.concatenate([t.old_features(dep,es,split),np.load(UPSTREAM/f'features/{dep}'/('real.npy' if split=='real' else f'{es}_{split}.npy'))],1)
        mass=dict(np.load(UPSTREAM/f'predictions/MULTIRATE/{dep}/model_{slot}_eval_{es}/{split}.npz'))
    rep=representations(x,cp);head=Head().cuda().eval();head.load_state_dict(ck['model'])
    residual=infer(head,((rep-ck['mu'])/ck['sd']).reshape(-1,103)).reshape(len(rep),253,16)
    peta=softmax(residual/ck['temperature']+np.log(ck['conditional_eta_prior'].clip(1e-12))[None],-1)
    hi=np.searchsorted(old.LOG_CENTERS,old.CENTERS,side='right').clip(1,252);lo=hi-1;w=((old.CENTERS-old.LOG_CENTERS[lo])/(old.LOG_CENTERS[hi]-old.LOG_CENTERS[lo])).clip(0,1)
    peta=(1-w[None,:,None])*peta[:,lo]+w[None,:,None]*peta[:,hi]
    if split=='real':
        full,events=dev.real_inputs(dep);valid=events.strict_h1l1_preprocessing_pass.to_numpy(bool)
        out=np.full((len(full),512,16),np.nan,np.float32);out[valid]=peta;peta=out
    joint=mass['p'][:,:,None]*peta
    finite=np.isfinite(mass['p']).all(1)
    error=float(abs(joint[finite].sum(-1)-mass['p'][finite]).max())
    if error>1e-6:raise RuntimeError('Conditional eta head changed mass marginal')
    extra={k:mass[k] for k in ('group','truth') if k in mass}
    path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,joint=joint.astype(np.float32),p=mass['p'],outside=mass['outside'],**extra)
    dev.json_write(path.with_suffix('.json'),{'frozen_mass_marginal_max_difference':error,'upstream_checkpoint':ck['mass_checkpoint_sha256']})
    return np.load(path)


def pair_features(root,dep,slot,a,i,j):
    ck=torch.load(root/f'models/{KIND}/{dep}/seed_{slot}/selected.pt',map_location='cpu',weights_only=False)
    prior=ck['prior'];cond=interpolate_rows(ck['conditional_eta_prior'],old.CENTERS)
    joint_prior=prior[:,None]*cond
    if not np.allclose(joint_prior.sum(1),prior,atol=1e-10):raise RuntimeError('Conditional prior changed mass prior')
    def gram(x,weight=None):
        z=torch.as_tensor(np.asarray(x,dtype=np.float64),device='cuda').flatten(1)
        if weight is not None:z=z/torch.as_tensor(np.sqrt(weight).reshape(1,-1),device='cuda')
        return (z@z.T).cpu().numpy()[i,j]
    joint=a['joint'];p=a['p']
    massbf=gram(p,prior).clip(1e-300);jointbf=gram(joint,joint_prior).clip(1e-300)
    bc=gram(np.sqrt(joint.astype(float)));mbc=gram(np.sqrt(p))
    return {'conditional_eta_overlap':np.log(jointbf)-np.log(massbf),'joint_BC':bc,'mass_BC':mbc.clip(1e-15,1),
        'ood':(a['outside'][i]>.25)|(a['outside'][j]>.25)}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--stage',choices=['initialize','train','development'],required=True);a=p.parse_args()
    if a.stage=='initialize':initialize(a.root)
    else:
        for dep in t.DEPS:
            for slot,seed in zip(t.MODEL_SLOTS,(202609881,202609882,202609883)):
                if a.stage=='train':train_one(a.root,dep,slot,seed)
                else:prediction(a.root,dep,slot,0,'development')
