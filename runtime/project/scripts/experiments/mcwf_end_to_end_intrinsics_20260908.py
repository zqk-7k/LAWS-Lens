#!/usr/bin/env python3
"""End-to-end shared mass/intrinsic representation with a matched mass-only control."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='2'
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace
import numpy as np
import pandas as pd
from scipy.special import softmax
import torch
from torch import nn
import torch.nn.functional as F
P=Path('/root/autodl-tmp/gw-catalog');sys.path.insert(0,str(P/'scripts/experiments'))
import mcwf_conditional_intrinsics_20260908 as labels
import mcwf_joint_intrinsics_evidence_20260908 as score
import mcwf_multirate_train_20260908 as mult
import mcwf_temporal_response_evaluate_20260908 as ev
t,old,dev,cf=ev.t,ev.t.old,ev.dev,ev.cf
UPSTREAM=P/'results/mcwf_multirate_lowband_exploratory_20260908T145551Z'
KINDS=('END2END-MASS','END2END-CHI','END2END-ETA-CHI')
SEEDS=(202610011,202610012,202610013)
TEMPS=(.5,.75,1.,1.25,1.5,2.)
EPOCHS=20
torch.set_num_threads(2)


def dimension(kind):return 1 if kind.endswith('MASS') else 272 if 'ETA' in kind else 17


def truth_and_prior(meta,kind):
    if kind.endswith('MASS'):return np.ones((len(meta),1),np.float32),np.ones((253,1),np.float32)
    original='CONDITIONAL-ETA-CHI' if 'ETA' in kind else 'CONDITIONAL-CHI'
    return labels.labels(meta,original).astype(np.float32),labels.conditional_prior(meta,original).astype(np.float32)


class Predictor(mult.Predictor):
    def __init__(self,kind):
        super().__init__('MULTIRATE');self.kind=kind
        self.conditional=nn.Sequential(nn.Conv1d(48,64,1),nn.SiLU(),nn.Dropout(.1),nn.Conv1d(64,dimension(kind),1))
        nn.init.zeros_(self.conditional[-1].weight);nn.init.zeros_(self.conditional[-1].bias)

    def forward(self,x):
        h=self.first(torch.cat([x,self.coordinate.expand(len(x),-1,-1)],1))
        h=F.silu(h+self.middle(h))
        return self.last(h).squeeze(1),self.conditional(h).transpose(1,2)


@torch.no_grad()
def infer(model,x):
    model.eval();mass=[];conditional=[]
    for start in range(0,len(x),128):
        with torch.autocast('cuda',dtype=torch.bfloat16):m,c=model(torch.as_tensor(x[start:start+128],device='cuda'))
        mass.append(m.float().cpu().numpy());conditional.append(c.float().cpu().numpy())
    return np.concatenate(mass),np.concatenate(conditional)


def ce(m,c,ym,yc,prior,tm=1.,tc=1.):
    pm=softmax(m.astype(float)/tm,1);pc=softmax(c.astype(float)/tc+np.log(prior.clip(1e-12))[None],-1)
    a=float(-(ym*np.log(pm.clip(1e-30))).sum(1).mean())
    b=float(-(ym[:,:,None]*yc[:,None,:]*np.log(pc.clip(1e-30))).sum((1,2)).mean())
    return a,b


def initialize(root):
    if root.exists():raise RuntimeError('Independent root required')
    for name in ('contracts','scripts','logs','models','predictions','calibration','tables','reports','manifest','evaluation','figures','cache'):(root/name).mkdir(parents=True)
    contract={'id':'MCWF-END-TO-END-INTRINSICS-23','UTC':datetime.now(timezone.utc).isoformat(),'status':t.STATUS,'goal_achieved':False,
      'same_algorithm_both_runs':True,'hypothesis':'Conditional-only heads cannot change a wrong mass representation. Jointproperclassification loss lets shared waveform features learn mass/spin/eta phase correlations.',
      'data':'Same4096simulatedtrainparents96noiseblocks8views;512developmentparents32noiseblocks;frozen06 low20-80Hz16s plusoldpeak2s40-580Hz responsefeatures.',
      'arms':KINDS,'control':'END2END-MASS hasidentical warm06 backbone and20epoch budget,conditionaldimension1.',
      'architecture':'Warm06shared55->48 mass-axisCNN;conditional1x1conv48->64->D,SiLU/dropout.1;conditional lastlayerzero,initialconditionaltrainingprior. Unlike10,shared backbone receives intrinsicgradients.',
      'loss':'CE_mass + E_soft_true_mass[CE_conditional_eta_chi],a singlejointproperlogscore;alltruthfromsimulation;noPE,officialorIDinputs.',
      'training':{'seeds':SEEDS,'epochs':EPOCHS,'optimizer':'AdamW1e-4,WD1e-4,cosine1e-5,clip5','batch':'128sources2views',
        'checkpoint':'minimumdevelopmentjointCE amongepochs with massCE<=epoch0massCE+.01;includeepoch0',
        'temperature':TEMPS,'temperature_rule':'massTbyminimumdevelopmentmassCE,conditionalTbyminimumconditionalCE;no real/test tuning.'},
      'prior':'Onevotepertrain source,conditionalpriorfrom10samegridandGaussian smoothing;empiricalsimulatednotastrophysical prior.',
      'density':'p(M,theta|waveform)=p(M|waveform)*p(theta|M,waveform);bothtrainedjointly;probabilitymass conserving M-gridconversion andconditionalinterpolation.',
      'calibration':'Same12fulljointBC/PRIOR empiricalLR;ADD/REPLACE oldOMCaux,validation-only coefficients,unchangedguards andbothpriorities;no newglobal realcoefficient selection inthisround.',
      'frozen':['originalencoders','time_score','sky_raw_log_bf','outer C-fixed weights','scope','history','paper'],
      'changed_channels':['waveform'],'outer_weights_changed':False,'limitations':'NeuralpredictivedensitynotphysicalPE;developmentreuseandadaptive realfeedbackdisclosed;newsource/noiseconfirmationrequired.',
      'references':['https://arxiv.org/abs/gr-qc/9402014','https://arxiv.org/abs/0909.2867'],
      'reference_scope':'Physicalmotivationforphasecorrelations,not validation of architecture orhyperparameters.'}
    dev.json_write(root/'contracts/ANALYSIS_CONTRACT.json',contract)
    rows=t.protected()
    for p in (UPSTREAM/'models/MULTIRATE').glob('*/*/selected.pt'):rows.append({'path':str(p),'sha256':dev.sha(p),'bytes':p.stat().st_size})
    dev.csv_write(root/'manifest/INPUT_SHA256.csv',pd.DataFrame(rows));shutil.copy2(__file__,root/'scripts/end_to_end_intrinsics.py')
    tests=[]
    torch.manual_seed(202610010)
    for kind in KINDS:
        m=Predictor(kind).double().eval();base=mult.Predictor('MULTIRATE').double().eval()
        base.load_state_dict({k:v for k,v in m.state_dict().items() if not k.startswith('conditional.')})
        with torch.no_grad():
            x=torch.randn(3,54,253,dtype=torch.float64);a,b=m(x);difference=float(abs(a-base(x)).max())
        if difference!=0 or float(abs(b).max())!=0:raise RuntimeError('Warm model algebra test')
        tests.append({'kind':kind,'zero_head_mass_difference':difference,'conditional_dimensions':dimension(kind),'pass':True})
    dev.json_write(root/'contracts/ALGEBRA_TESTS.json',tests)
    dev.json_write(root/'contracts/START_FREEZE.json',{'contract_sha256':dev.sha(root/'contracts/ANALYSIS_CONTRACT.json'),'code_sha256':dev.sha(Path(__file__))})


def train_one(root,dep,kind,slot,seed):
    out=root/f'models/{kind}/{dep}/seed_{slot}'
    if (out/'COMPLETE.json').exists():return
    out.mkdir(parents=True,exist_ok=True)
    cp=torch.load(UPSTREAM/f'models/MULTIRATE/{dep}/seed_{slot}/selected.pt',map_location='cpu',weights_only=False)
    x,meta=mult.training_data(UPSTREAM,dep,'train','MULTIRATE');v,vm=mult.training_data(UPSTREAM,dep,'validation','MULTIRATE')
    if set(meta.source_uid)&set(vm.source_uid) or set(meta.noise_bank_index)&set(vm.noise_bank_index):raise RuntimeError('Source/noise overlap')
    ym=old.targets(np.log(meta.mc_det));vy=old.targets(np.log(vm.mc_det))
    yc,prior=truth_and_prior(meta,kind);vc,_=truth_and_prior(vm,kind)
    x=(x-cp['mu'])/cp['sd'];v=(v-cp['mu'])/cp['sd']
    g,names=pd.factorize(meta.source_uid,sort=True);members=[np.flatnonzero(g==i) for i in range(len(names))]
    dev.TRAIN.seed_everything(seed);model=Predictor(kind).cuda()
    missing,unexpected=model.load_state_dict(cp['model'],strict=False)
    if unexpected or any(not k.startswith('conditional.') for k in missing):raise RuntimeError('Warm state mismatch')
    xx,yy,zz=[torch.as_tensor(a,device='cuda') for a in (x,ym,yc)];pr=torch.as_tensor(np.log(prior.clip(1e-12)),device='cuda');del x
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,EPOCHS,eta_min=1e-5)
    history=[];best=float('inf');started=time.perf_counter();initial=None
    for epoch in range(EPOCHS+1):
        if epoch:
            rng=np.random.default_rng(seed+epoch);order=rng.permutation(len(names));model.train()
            for start in range(0,len(order),128):
                ids=np.concatenate([rng.choice(members[k],2,replace=False) for k in order[start:start+128]])
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    m,c=model(xx[ids]);lm=-(yy[ids]*F.log_softmax(m.float(),1)).sum(1)
                    lc=-(yy[ids,:,None]*zz[ids,None,:]*F.log_softmax(c.float()+pr[None],-1)).sum((1,2))
                    loss=(lm+lc).mean()
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite jointloss')
                loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step()
            scheduler.step()
        m,c=infer(model,v);a,b=ce(m,c,vy,vc,prior)
        if initial is None:initial=a
        row={'epoch':epoch,'massCE':a,'conditionalCE':b,'jointCE':a+b,'mass_guard':a<=initial+.01,'seconds':time.perf_counter()-started};history.append(row)
        if row['mass_guard'] and a+b<best:
            best=a+b;torch.save({'model':{k:z.detach().cpu().clone() for k,z in model.state_dict().items()},'mu':cp['mu'],'sd':cp['sd'],
                'prior':cp['prior'],'conditional_prior':prior,'kind':kind,'seed':seed,'slot':slot,'epoch':epoch,'jointCE':best},out/'selected.pt')
        dev.csv_write(out/'history.csv',pd.DataFrame(history));print(json.dumps({'train':[dep,kind,slot],**row}),flush=True)
    ck=torch.load(out/'selected.pt',map_location='cpu',weights_only=False);model.load_state_dict(ck['model']);m,c=infer(model,v)
    massgrid=[{'T':temp,'CE':ce(m,c,vy,vc,prior,temp,1)[0]} for temp in TEMPS]
    conditionalgrid=[{'T':temp,'CE':ce(m,c,vy,vc,prior,1,temp)[1]} for temp in TEMPS]
    ck['mass_temperature']=min(massgrid,key=lambda r:(r['CE'],abs(r['T']-1)))['T']
    ck['conditional_temperature']=min(conditionalgrid,key=lambda r:(r['CE'],abs(r['T']-1)))['T'];torch.save(ck,out/'selected.pt')
    dev.json_write(out/'TEMPERATURES.json',{'mass':massgrid,'conditional':conditionalgrid})
    dev.json_write(out/'COMPLETE.json',{'kind':kind,'epoch':ck['epoch'],'jointCE':best,'training_sources':len(names),'development_sources':vm.source_uid.nunique(),
        'mass_temperature':ck['mass_temperature'],'conditional_temperature':ck['conditional_temperature'],'source_noise_overlap':0,'seconds':time.perf_counter()-started,'sha256':dev.sha(out/'selected.pt')})


def prediction(root,dep,slot,seed,split,kind):
    dest=root/f'predictions/{kind}/{dep}/{slot}_{seed}_{split}.npz'
    if dest.exists():return np.load(dest)
    ck=torch.load(root/f'models/{kind}/{dep}/seed_{slot}/selected.pt',map_location='cpu',weights_only=False)
    if split=='development':x,meta=mult.training_data(UPSTREAM,dep,'validation','MULTIRATE')
    else:
        file='real.npy' if split=='real' else f'{seed}_{split}.npy'
        x=np.concatenate([t.old_features(dep,seed,split),np.load(UPSTREAM/f'features/{dep}/{file}')],1)
    model=Predictor(kind).cuda().eval();model.load_state_dict(ck['model']);m,c=infer(model,(x-ck['mu'])/ck['sd'])
    p,outside=old.probability(m,ck['mass_temperature'])
    cond=softmax(c.astype(float)/ck['conditional_temperature']+np.log(ck['conditional_prior'].clip(1e-12))[None],-1)
    hi=np.searchsorted(old.LOG_CENTERS,old.CENTERS,side='right').clip(1,252);lo=hi-1
    w=((old.CENTERS-old.LOG_CENTERS[lo])/(old.LOG_CENTERS[hi]-old.LOG_CENTERS[lo])).clip(0,1)
    cond=cond[:,lo]*(1-w[None,:,None])+cond[:,hi]*w[None,:,None]
    joint=p[:,:,None]*cond
    if abs(joint.sum(-1)-p).max()>1e-6:raise RuntimeError('Massmarginalnonconservation')
    if split=='real':
        full,events=dev.real_inputs(dep);valid=events.strict_h1l1_preprocessing_pass.to_numpy(bool)
        a,b,z=np.full((len(full),512),np.nan),np.full((len(full),512,dimension(kind)),np.nan),np.full(len(full),np.nan)
        a[valid],b[valid],z[valid]=p,joint,outside;p,joint,outside=a,b,z
    dest.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest,p=p,joint=joint.astype(np.float32),outside=outside)
    return np.load(dest)


def features(root,dep,slot,seed,split,kind):
    dest=root/f'cache/{kind}/{dep}/{slot}_{seed}_{split}.npz'
    if dest.exists():return dict(np.load(dest))
    a=prediction(root,dep,slot,seed,split,kind)
    ck=torch.load(root/f'models/{kind}/{dep}/seed_{slot}/selected.pt',map_location='cpu',weights_only=False)
    cond=labels.original.interpolate_rows(ck['conditional_prior'],old.CENTERS)
    prior=ck['prior'][:,None]*cond
    p=torch.as_tensor(a['joint'],device='cuda',dtype=torch.float64).flatten(1)
    z=p/torch.as_tensor(np.sqrt(prior).reshape(1,-1),device='cuda')
    bf=(z@z.T).cpu().numpy().clip(1e-300);bc=(torch.sqrt(p)@torch.sqrt(p).T).cpu().numpy().clip(1e-15,1)
    result={'joint_logbf':np.log(bf),'joint_logbc':np.log(bc),'joint_bc':bc,'outside':a['outside']}
    dest.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest,**result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--stage',required=True,choices=['initialize','train','select','evaluate','real','assess']);a=p.parse_args()
    score.features=features;score.model=SimpleNamespace(KINDS=KINDS);ev.scored=score.scored
    if a.stage=='initialize':initialize(a.root)
    elif a.stage=='train':
        for dep in t.DEPS:
            for kind in KINDS:
                for slot,seed in zip(t.MODEL_SLOTS,SEEDS):train_one(a.root,dep,kind,slot,seed)
    elif a.stage=='select':score.select(a.root)
    else:getattr(ev,a.stage)(a.root)
