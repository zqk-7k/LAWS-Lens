#!/usr/bin/env python3
"""Low-band long-duration time-frequency residual, with a matched continuation."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='2'
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

P=Path('/root/autodl-tmp/gw-catalog');sys.path.insert(0,str(P/'scripts/experiments'))
import mcwf_multirate_train_20260908 as mult
import mcwf_temporal_response_evaluate_20260908 as ev
t,old,dev,cf=ev.t,ev.t.old,ev.dev,ev.cf
UPSTREAM=P/'results/mcwf_multirate_lowband_exploratory_20260908T145551Z'
KINDS=('LOW-CONTINUE','LOW-TF')
SEEDS=(202609981,202609982,202609983)
EPOCHS=15
torch.set_num_threads(2)


class Predictor(nn.Module):
    def __init__(self,kind):
        super().__init__();self.kind=kind;self.base=mult.Predictor('MULTIRATE')
        if kind=='LOW-TF':
            self.register_buffer('window',torch.hann_window(256))
            self.register_buffer('frequency',torch.linspace(-1,1,61).view(1,1,61,1).expand(1,1,61,121))
            self.register_buffer('clock',torch.linspace(-1,1,121).view(1,1,1,121).expand(1,1,61,121))
            layers=[];channels=4
            for width,kernel in ((16,5),(32,3),(48,3)):
                layers += [nn.Conv2d(channels,width,kernel,stride=2,padding=kernel//2),nn.GroupNorm(8,width),nn.SiLU()]
                channels=width
            self.spectral=nn.Sequential(*layers,nn.AdaptiveAvgPool2d((4,8)))
            self.head=nn.Sequential(nn.Flatten(),nn.Linear(48*4*8,128),nn.SiLU(),nn.Dropout(.1),nn.Linear(128,253))
            nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)

    def transform(self,raw):
        z=torch.stft(raw.float().flatten(0,1),n_fft=256,hop_length=32,window=self.window,
            normalized=True,center=False,return_complex=True)
        power=torch.log1p(z.abs().square()).reshape(len(raw),2,129,121)[:,:,20:81]
        return torch.cat([power,self.frequency.expand(len(raw),-1,-1,-1),self.clock.expand(len(raw),-1,-1,-1)],1)

    def forward(self,x,raw):
        logits=self.base(x)
        if self.kind=='LOW-CONTINUE':return logits
        with torch.autocast(device_type=raw.device.type,enabled=False):feature=self.transform(raw)
        return logits.float()+2*torch.tanh(self.head(self.spectral(feature)).float())


@torch.no_grad()
def infer(model,x,raw):
    model.eval();out=[]
    for start in range(0,len(x),256):
        a=torch.as_tensor(np.asarray(x[start:start+256],np.float32),device='cuda')
        b=torch.as_tensor(np.asarray(raw[start:start+256],np.float32),device='cuda')
        with torch.autocast('cuda',dtype=torch.bfloat16):logits=model(a,b)
        out.append(logits.float().cpu().numpy())
    return np.concatenate(out)


def numerical_tests():
    torch.manual_seed(202609980)
    raw=torch.randn(3,2,4096);x=torch.randn(3,54,253)
    model=Predictor('LOW-TF').eval()
    a=model.transform(raw);b=model.transform(-raw)
    if a.shape!=(3,4,61,121) or not torch.allclose(a,b,atol=1e-6):raise RuntimeError('STFT shape/sign')
    w=np.hanning(257)[:-1];r=raw.numpy()[0,0]
    frames=np.stack([r[i:i+256]*w for i in range(0,4096-256+1,32)],1)
    reference=np.log1p(abs(np.fft.rfft(frames,axis=0)/16)**2)[20:81]
    error=float(abs(reference-a.numpy()[0,0]).max())
    if error>1e-6:raise RuntimeError('Independent STFT mismatch')
    with torch.no_grad():delta=float(abs(model(x,raw)-model.base(x)).max())
    if delta!=0:raise RuntimeError('Initial residual changed base')
    return {'pass':True,'shape':list(a.shape),'CPU_FFT_STFT_max_abs_error':error,'zero_residual_max_abs_difference':delta,
            'power_sign_invariant':True,'physical_sampling_rate':256,'frequency_bins_Hz':[20,80],
            'first_frame_center_s_before_peak_end':-15.5,'last_frame_center_s_before_peak_end':-.5}


def initialize(root):
    if root.exists():raise RuntimeError('Independent fresh directory required')
    for name in ('contracts','scripts','logs','models','predictions','calibration','tables','reports','manifest','evaluation','figures'):(root/name).mkdir(parents=True)
    contract={'id':'MCWF-LOWBAND-TIMEFREQUENCY-19','UTC':datetime.now(timezone.utc).isoformat(),'status':t.STATUS,
        'goal_achieved':False,'baseline':str(t.PAIRS),'upstream':str(UPSTREAM),'same_algorithm_both_runs':True,
        'frozen':['original peak2s4096 input','old encoder outputs','time_score','sky_raw_log_bf','outer C-fixed weights','scope','historical files','paper'],
        'changed_channels':['waveform'],'outer_weights_changed':False,
        'rationale':'Matched-filter peak powers discard the early chirp time-frequency trajectory. Existing TF study used only2s40-580Hz;this adds16s20-80Hz at256Hz.',
        'data':'Previously replay-verified physical source/noise low16s4096 cache from06;no new strain/noise samples,4096trainingparents96noiseblocks8views,512developmentparents32noiseblocks.',
        'model':'Warm06 MULTIRATE55channel1d backbone;LOW-TF adds log1p Hann256 normalized STFT,hop32,centerfalse,20..80Hz;H1/L1plusfixedtime/frequencycoordinates;conv16/32/48,GN,SiLU,adaptive4x8,1536->128->253;bounded2tanh residual,zero-init.',
        'control':'LOW-CONTINUE: same06 warmstart andsame trainingbudget/data;noTFbranch. Both variants may selectepoch0.',
        'training':{'seeds':SEEDS,'epochs':EPOCHS,'batch':'128sources,2views/source','optimizer':'AdamW lr1e-4 WD1e-4 cosine1e-5 clip5',
                    'checkpoint':'minimum simulateddevelopment CE','temperature':[.5,.75,1,1.25,1.5,2],
                    'source_labels':'detector-frame chirpmass fromsimulation only;no PE or official labels'},
        'calibration':'same06 source/noise-disjoint development halves;finite-reference mass conflict and isotonic prior-overlap LR;positiveOOD neutral,clip+-4;coefficients chosenvalidation only.',
        'selection':'candidate:F50,F90,-AP,-R10,-R1;retrieval:-R10,-R1,-AP,F50,F90;tiesminimumcoeffnorm. Same frozen0..2grid andperseed guards.',
        'limitations':'Dataanddevelopmentsets reused. Neural masspredictive density isnot publicPE;no guarantee ofrecoveredchirptrack;realfeedbackadaptive,notblindconfirmation. Literaturedoesnotvalidatechosenengineeringhyperparameters.',
        'references':['https://docs.pytorch.org/docs/stable/generated/torch.stft.html','https://arxiv.org/abs/2011.10425']}
    dev.json_write(root/'contracts/ANALYSIS_CONTRACT.json',contract)
    rows=t.protected()
    for dep in t.DEPS:
        for split in ('train','validation'):
            p=UPSTREAM/f'cache/{dep}/{split}/low16s.npy'
            rows.append({'path':str(p),'sha256':dev.sha(p),'bytes':p.stat().st_size})
        for slot in t.MODEL_SLOTS:
            p=UPSTREAM/f'models/MULTIRATE/{dep}/seed_{slot}/selected.pt'
            rows.append({'path':str(p),'sha256':dev.sha(p),'bytes':p.stat().st_size})
    dev.csv_write(root/'manifest/INPUT_SHA256.csv',pd.DataFrame(rows))
    dev.json_write(root/'contracts/NUMERICAL_TESTS.json',numerical_tests())
    shutil.copy2(__file__,root/'scripts/lowband_timefrequency.py')
    dev.json_write(root/'contracts/START_FREEZE.json',{'contract_sha256':dev.sha(root/'contracts/ANALYSIS_CONTRACT.json'),'code_sha256':dev.sha(Path(__file__))})


def train_one(root,dep,kind,slot,seed):
    out=root/f'models/{kind}/{dep}/seed_{slot}'
    if (out/'COMPLETE.json').exists():return
    out.mkdir(parents=True,exist_ok=True)
    cp=torch.load(UPSTREAM/f'models/MULTIRATE/{dep}/seed_{slot}/selected.pt',map_location='cpu',weights_only=False)
    x,tm=mult.training_data(UPSTREAM,dep,'train','MULTIRATE');v,vm=mult.training_data(UPSTREAM,dep,'validation','MULTIRATE')
    raw=np.load(UPSTREAM/f'cache/{dep}/train/low16s.npy',mmap_mode='r')
    rv=np.load(UPSTREAM/f'cache/{dep}/validation/low16s.npy',mmap_mode='r')
    if len(raw)!=len(tm) or len(rv)!=len(vm):raise RuntimeError('Lowband index mismatch')
    if set(tm.source_uid)&set(vm.source_uid) or set(tm.noise_bank_index)&set(vm.noise_bank_index):raise RuntimeError('Training/development leakage')
    mu,sd=cp['mu'],cp['sd'];x=(x-mu)/sd;v=(v-mu)/sd
    g,names=pd.factorize(tm.source_uid,sort=True);vg,_=pd.factorize(vm.source_uid,sort=True)
    members=[np.flatnonzero(g==j) for j in range(len(names))]
    truth,vt=np.log(tm.mc_det.to_numpy(float)),np.log(vm.mc_det.to_numpy(float))
    y,vy=old.targets(truth),old.targets(vt)
    dev.TRAIN.seed_everything(seed);model=Predictor(kind).cuda();model.base.load_state_dict(cp['model'])
    xx=torch.as_tensor(x,device='cuda');yy=torch.as_tensor(y,device='cuda');rr=torch.as_tensor(np.asarray(raw,np.float32),device='cuda');del x
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,EPOCHS,eta_min=1e-5)
    best=float('inf');history=[];first=0;started=time.perf_counter()
    if (out/'resume.pt').exists():
        r=torch.load(out/'resume.pt',map_location='cpu',weights_only=False);model.load_state_dict(r['model'])
        optimizer.load_state_dict(r['optimizer']);scheduler.load_state_dict(r['scheduler'])
        history,best,first=r['history'],r['best'],r['epoch']+1
        torch.set_rng_state(r['rng']);torch.cuda.set_rng_state_all(r['cuda_rng'])
    for epoch in range(first,EPOCHS+1):
        losses=[]
        if epoch:
            rng=np.random.default_rng(seed+epoch);order=rng.permutation(len(names));model.train()
            for start in range(0,len(order),128):
                ids=np.concatenate([rng.choice(members[k],2,replace=False) for k in order[start:start+128]])
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    logits=model(xx[ids],rr[ids]);loss=-(F.log_softmax(logits.float(),-1)*yy[ids]).sum(-1).mean()
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
                loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step();losses.append(float(loss.detach()))
            scheduler.step()
        logits=infer(model,v,rv);prob=softmax(logits.astype(float),1)
        ce=float(-(vy*np.log(prob.clip(1e-30))).sum(-1).mean())
        row={'epoch':epoch,'validationCE':ce,'trainCE':float(np.mean(losses)) if losses else None,
             'logMc_MAE':float(abs(prob@old.LOG_CENTERS-vt).mean()),'seconds':time.perf_counter()-started}
        history.append(row)
        if ce<best:
            best=ce;torch.save({'model':{k:z.detach().cpu().clone() for k,z in model.state_dict().items()},'mu':mu,'sd':sd,
                'kind':kind,'seed':seed,'slot':slot,'epoch':epoch,'CE':ce,'prior':cp['prior']},out/'selected.pt')
        torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
            'history':history,'best':best,'epoch':epoch,'rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()},out/'resume.pt')
        dev.csv_write(out/'history.csv',pd.DataFrame(history));print(json.dumps({'training':[dep,kind,seed],**row}),flush=True)
    ck=torch.load(out/'selected.pt',map_location='cpu',weights_only=False);model.load_state_dict(ck['model'])
    logits=infer(model,v,rv);grid=[]
    for temp in (.5,.75,1.,1.25,1.5,2.):
        p=softmax(logits.astype(float)/temp,1)
        grid.append({'temperature':temp,'CE':float(-(vy*np.log(p.clip(1e-30))).sum(-1).mean())})
    ck['temperature']=min(grid,key=lambda r:(r['CE'],abs(r['temperature']-1)))['temperature'];torch.save(ck,out/'selected.pt')
    prob,boundary=old.probability(logits,ck['temperature'])
    np.savez_compressed(out/'development_predictions.npz',p=prob,outside=boundary,group=vg,truth=vt)
    cdf=np.c_[np.zeros(len(prob)),prob.cumsum(1)];pit=np.array([np.interp(a,old.EDGES,b) for a,b in zip(vt,cdf)])
    dev.csv_write(out/'temperature_grid.csv',pd.DataFrame(grid))
    dev.json_write(out/'COMPLETE.json',{'epoch':ck['epoch'],'temperature':ck['temperature'],'logMc_MAE':float(abs(prob@old.CENTERS-vt).mean()),
        'central90coverage':float(((pit>=.05)&(pit<=.95)).mean()),'training_sources':len(names),'development_sources':len(np.unique(vg)),
        'source_noise_overlap':0,'sha256':dev.sha(out/'selected.pt'),'seconds':time.perf_counter()-started,'peak_gpu_memory_bytes':torch.cuda.max_memory_allocated()})


def predict(root,dep,kind,slot,es,split):
    dest=root/f'predictions/{kind}/{dep}/model_{slot}_eval_{es}/{split}.npz'
    if dest.exists():return np.load(dest)
    cp=root/f'models/{kind}/{dep}/seed_{slot}/selected.pt';ck=torch.load(cp,map_location='cpu',weights_only=False)
    x=np.concatenate([t.old_features(dep,es,split),np.load(UPSTREAM/f'features/{dep}/real.npy' if split=='real' else UPSTREAM/f'features/{dep}/{es}_{split}.npy')],1)
    raw=np.load(UPSTREAM/f'cache/deployment/{dep}/real/low16s.npy' if split=='real' else UPSTREAM/f'cache/deployment/{dep}/{es}_{split}/low16s.npy',mmap_mode='r')
    if len(x)!=len(raw):raise RuntimeError('Deployment index mismatch')
    model=Predictor(kind).cuda().eval();model.load_state_dict(ck['model'])
    prob,ood=old.probability(infer(model,(x-ck['mu'])/ck['sd'],raw),ck['temperature'])
    if split=='real':
        full,events=dev.real_inputs(dep);valid=events.strict_h1l1_preprocessing_pass.to_numpy(bool)
        pp,oo=np.full((len(full),512),np.nan),np.full(len(full),np.nan);pp[valid],oo[valid]=prob,ood;prob,ood=pp,oo
    dest.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest,p=prob,outside=ood,checkpoint_sha256=dev.sha(cp))
    return np.load(dest)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--stage',choices=['initialize','train','select','evaluate','real','assess'],required=True);a=p.parse_args()
    t.predict=predict;mult.KINDS=KINDS
    if a.stage=='initialize':initialize(a.root)
    elif a.stage=='train':
        for dep in t.DEPS:
            for kind in KINDS:
                for slot,seed in zip(t.MODEL_SLOTS,SEEDS):train_one(a.root,dep,kind,slot,seed)
    elif a.stage=='select':mult.select(a.root)
    else:getattr(ev,a.stage)(a.root)
