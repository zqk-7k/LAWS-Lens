#!/usr/bin/env python3
"""Conditional intrinsic-grid predictors on frozen peak2s waveform features."""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '2'
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from scipy.special import softmax
from sklearn.isotonic import IsotonicRegression

import mcwf_omc_ensemble_extension_20260907 as e
import mcwf_omc_architecture_extension_20260907 as arch
import mcwf_ordered_mass_predictor_20260907 as mass

dev = e.dev
LAST = dev.PROJECT / 'results/mcwf_omc_ensemble_exploratory_20260907T141320Z'
GLOBAL = LAST / 'architectures/GLOBAL'
SEEDS = (202609711, 202609712, 202609713)
Q = np.linspace(.05, 1., 9)
CHI = np.linspace(-.95, .95, 9)
SHAPE = (253, 9, 9)
torch.set_num_threads(2)
torch.backends.cuda.matmul.allow_tf32 = False


def initialize(root, kind):
    e.initialize(root)
    assert dev.sha(dev.PROJECT / 'packages/mcwf_omc_ensemble_exploratory_20260907T141320Z_deliverables.tar.gz') == '71e82d796b04494f1b86240a27aeb8bda0b1319fd91df37865b6db1f70ffac5f'
    comp = GLOBAL / 'trials/MIXTURE-PRIOR/diagnostic_export'
    shutil.copytree(comp / 'results/DIAGNOSTIC/gwtc4', root / 'comparators/O4a_GLOBAL_PRIOR')
    shutil.copy2(comp / 'CHOICES.json', root / 'comparators/GLOBAL_PRIOR_CHOICES.json')
    records = [{'path': str(p), 'sha256': dev.sha(p)} for folder in (comp, GLOBAL/'models')
               for p in sorted(folder.rglob('*')) if p.is_file() and '__pycache__' not in p.parts]
    dev.csv_write(root / 'manifest/GLOBAL_PRIOR_PROTECTED.csv', pd.DataFrame(records))
    dev.json_write(root / 'contracts/INTRINSIC_GRID_CONTRACT.json', {
        'utc': datetime.now(timezone.utc).isoformat(), 'kind': kind, 'seeds': SEEDS,
        'baseline': 'MCWF-UNIFIED-OMC-DEVCONF', 'additional_comparator': str(comp),
        'hypothesis': 'A correlated predictive distribution over Mc,q,chi_eff can distinguish some mass-compatible unrelated signals; no guarantee of official overlap.',
        'representation': 'p(logMc|waveform) p(q,chi_eff|logMc,waveform),253x9x9 probability masses',
        'training_truth': 'simulated detector-frame Mc,m2/m1,and mass-weighted aligned spin components,not real PE',
        'features': 'unchanged27x253 template responses from2s4096H1L1;run-matched existing source/noise data',
        'training': 'same12288 sourceparents/160noise blocks;512 disjoint development parents/32noise blocks;128sources x2views;40epochs;AdamW;clip5',
        'models': {'CONDITIONAL': 'frozen prior GLOBAL encoder and Mc marginal;train new conditional q/chi head at2e-4',
                   'JOINT': 'warmstart prior GLOBAL;train encoder,Mc and conditional q/chi together at2e-5'},
        'loss': 'trilinearly interpolated joint-grid categorical NLL;equal weight per source and view',
        'checkpoint': 'lowest development joint NLL;no real PE/official checkpoint selection',
        'temperature': 'conditional head T .5,.75,1,1.25,1.5,2,3 selected on development NLL;Mc temperature retained from parent',
        'training_prior': 'one vote per independent source,trilinear histogram,Gaussian sigma1gridcell,+1e-5 smoothing;not a fitted astrophysical population',
        'arms': ['JOINT-PRIOR','JOINT-BC','MASS-Q-PRIOR','MASS-PRIOR'],
        'new_score': 'current OMC waveform score + gamma*finite mass compatibility penalty + beta*bounded simulation-calibrated predictive-overlap evidence',
        'gamma_grid': [0,.125,.25,.5,1], 'beta_grid': [0,.0625,.125,.25,.5,1,2,4],
        'cap': 4, 'support': 'development feature support;no positive reward outside support or below mass-reference p=.05;mass endpoint occupancy>.25 disables new terms',
        'target': 'both runs improve PE and both official budget counts relative to OMC;O4a must additionally not lose GLOBAL-PRIOR Top10/20 PE/official gains',
        'simulation_guard': {'R10_drop_max': .02, 'AP_drop_max': .005, 'F50_F90_ratio_max': 1.1},
        'selection_disclosure': 'Real PE and official outcomes used for adaptive development selection only,never network targets or score inputs;reused test is development,not blind validation',
        'fresh_confirmation': 'a candidate must pass new source/noise-disjoint retrieval confirmation;this cannot make adaptive real results independent',
        'frozen': ['time','sky','outer C-fixed weights','2s4096','strictscope','historical outputs'],
        'not_full_PE': True, 'not_lensing_Bayes_factor': True,
        'references': ['https://arxiv.org/abs/1807.07062','https://arxiv.org/abs/2008.03312'],
        'references_scope': 'Motivation only;this discretized surrogate and its empirical calibration are not DINGO or full PE.',
        'final_status': 'HOLD_FOR_AUTHOR_REVIEW_NO_ADOPTION_NO_OVERWRITE'})
    shutil.copy2(__file__, root / 'scripts' / Path(__file__).name)


def truth(meta):
    m1, m2 = meta.m1_det.to_numpy(float), meta.m2_det.to_numpy(float)
    assert np.all(m1 >= m2) and np.all(m2 > 0)
    q = m2 / m1
    chi = (meta.a1.to_numpy()*np.cos(meta.tilt1.to_numpy()) + q*meta.a2.to_numpy()*np.cos(meta.tilt2.to_numpy()))/(1+q)
    return np.column_stack([np.log(meta.mc_det.to_numpy()), q, chi])


def targets(t):
    choices = []
    for v, centers in zip(t.T, (mass.LOG_CENTERS, Q, CHI)):
        v = v.clip(centers[0], centers[-1])
        hi = np.searchsorted(centers, v, side='right').clip(1, len(centers)-1)
        lo = hi - 1
        f = (v-centers[lo])/(centers[hi]-centers[lo])
        choices.append(((lo, 1-f), (hi, f)))
    indices, weights = [], []
    for m, wm in choices[0]:
        for q, wq in choices[1]:
            for s, ws in choices[2]:
                indices.append((m*9+q)*9+s)
                weights.append(wm*wq*ws)
    ii, ww = np.stack(indices, 1), np.stack(weights, 1).astype(np.float32)
    assert np.allclose(ww.sum(1), 1)
    return ii, ww


class Predictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = arch.Predictor('GLOBAL')
        self.conditional = nn.Sequential(nn.Conv1d(64,64,5,padding=2),nn.GroupNorm(8,64),nn.SiLU(),
                                         nn.Dropout(.1),nn.Conv1d(64,81,1))

    def forward(self, x, mass_temperature=1., conditional_temperature=1.):
        b = self.body
        h = b.local(b.first(torch.cat([x,b.coordinate.expand(len(x),-1,-1)],1)))
        h = b.global_context(h.transpose(1,2)).transpose(1,2)
        m = F.log_softmax(b.last(h).squeeze(1).float()/mass_temperature,1)
        c = F.log_softmax(self.conditional(h).transpose(1,2).float()/conditional_temperature,-1)
        return (m[:,:,None]+c).flatten(1)


@torch.no_grad()
def infer(model, x, mt, ct=1., batch=128):
    model.eval()
    out=[]
    for start in range(0,len(x),batch):
        with torch.autocast('cuda',dtype=torch.bfloat16):
            v=model(torch.as_tensor(x[start:start+batch],device='cuda',dtype=torch.float32),mt,ct)
        out.append(v.cpu().numpy())
    return np.concatenate(out)


def nll(logp, ii, ww):
    return float(-(np.take_along_axis(logp,ii,axis=1)*ww).sum(1).mean())


def train(root, dep, ms, seed, kind):
    out = root / f'models/{dep}/seed_{ms}'
    if (out/'COMPLETE.json').exists(): return
    out.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter()
    x, tm=mass.data(e.PREVIOUS,dep,'train');v,vm=mass.data(e.PREVIOUS,dep,'validation')
    assert not set(tm.source_uid)&set(vm.source_uid)
    assert not set(tm.noise_bank_index)&set(vm.noise_bank_index)
    ck=torch.load(GLOBAL/f'models/{dep}/seed_{ms}/validation_selected_model.pt',weights_only=False,map_location='cpu')
    x=(x-ck['mu'])/ck['sd'];v=(v-ck['mu'])/ck['sd']
    tt,vt=truth(tm),truth(vm)
    ti,tw=targets(tt);vi,vw=targets(vt)
    group,names=pd.factorize(tm.source_uid,sort=True);vg,_=pd.factorize(vm.source_uid,sort=True)
    members=[np.flatnonzero(group==k) for k in range(len(names))]
    dev.TRAIN.seed_everything(seed)
    model=Predictor().cuda();model.body.load_state_dict(ck['model'])
    if kind=='CONDITIONAL':
        for p in model.body.parameters():p.requires_grad_(False)
    lr=2e-4 if kind=='CONDITIONAL' else 2e-5
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=lr,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,40,eta_min=lr/10)
    xx=torch.as_tensor(x,device='cuda');ii=torch.as_tensor(ti,device='cuda');ww=torch.as_tensor(tw,device='cuda')
    history=[];best=float('inf');mt=ck['temperature']
    for epoch in range(1,41):
        model.train()
        if kind=='CONDITIONAL':model.body.eval()
        rng=np.random.default_rng(seed+epoch);order=rng.permutation(len(names));losses=[]
        for start in range(0,len(order),128):
            ix=np.stack([rng.choice(members[k],2,replace=False) for k in order[start:start+128]]).reshape(-1)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                logp=model(xx[ix],mt)
                loss=-(logp.gather(1,ii[ix])*ww[ix]).sum(1).mean()
            assert torch.isfinite(loss)
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step()
            losses.append(float(loss.detach()))
        scheduler.step();lp=infer(model,v,mt);score=nll(lp,vi,vw)
        row={'epoch':epoch,'train_NLL':float(np.mean(losses)),'validation_NLL':score,'seconds':time.perf_counter()-started}
        history.append(row)
        if score<best:
            best=score
            torch.save({'model':{k:z.detach().cpu().clone() for k,z in model.state_dict().items()},
                        'mu':ck['mu'],'sd':ck['sd'],'mass_temperature':mt,'epoch':epoch,'seed':seed,'kind':kind},out/'validation_selected_model.pt')
        dev.csv_write(out/'history.csv',pd.DataFrame(history))
        if epoch%10==0:print(json.dumps({'train':[kind,dep,seed],**row}),flush=True)
    ck2=torch.load(out/'validation_selected_model.pt',weights_only=False,map_location='cpu');model.load_state_dict(ck2['model'])
    grid=[]
    for t in (.5,.75,1.,1.25,1.5,2.,3.):
        grid.append({'temperature':t,'NLL':nll(infer(model,v,mt,t),vi,vw)})
    ct=min(grid,key=lambda r:(r['NLL'],abs(r['temperature']-1)))['temperature'];ck2['conditional_temperature']=ct
    p=np.exp(infer(model,v,mt,ct)).astype(np.float32);p/=p.sum(1,keepdims=True)
    unique=~tm.source_uid.duplicated().to_numpy()
    prior=np.bincount(ti[unique].reshape(-1),weights=tw[unique].reshape(-1),minlength=np.prod(SHAPE)).reshape(SHAPE)
    prior=gaussian_filter(prior.astype(float),1,mode='nearest')+1e-5;prior/=prior.sum();ck2['prior']=prior.astype(np.float32)
    torch.save(ck2,out/'validation_selected_model.pt')
    np.savez_compressed(out/'development_predictions.npz',p=p,group=vg,truth=vt)
    marginal=p.reshape(-1,*SHAPE)
    error=np.column_stack([marginal.sum((2,3))@mass.LOG_CENTERS-vt[:,0],
                           marginal.sum((1,3))@Q-vt[:,1],marginal.sum((1,2))@CHI-vt[:,2]])
    report={'kind':kind,'deployment':dep,'seed':seed,'epoch':ck2['epoch'],'conditional_temperature':ct,
            'development_joint_NLL':nll(np.log(p.clip(1e-30)),vi,vw),
            'logMc_MAE':float(abs(error[:,0]).mean()),'q_MAE':float(abs(error[:,1]).mean()),'chi_eff_MAE':float(abs(error[:,2]).mean()),
            'source_overlap':0,'noise_overlap':0,'training_sources':len(names),'development_sources':len(np.unique(vg)),
            'seconds':time.perf_counter()-started,'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated(),
            'checkpoint_sha256':dev.sha(out/'validation_selected_model.pt')}
    dev.csv_write(out/'temperature_grid.csv',pd.DataFrame(grid));dev.json_write(out/'COMPLETE.json',report)
    print(json.dumps({'training_complete':report}),flush=True)


def prediction(root,dep,ms,es,split):
    cp=root/f'models/{dep}/seed_{ms}/validation_selected_model.pt'
    if split=='development':return dict(np.load(cp.parent/'development_predictions.npz'))
    cache=root/f'cache/predictions/{dep}/{ms}_{es}_{split}.npz'
    if cache.exists():return dict(np.load(cache))
    coarse=np.load(e.old.TRAINED/f'cache/deployment_event_psd/{dep}'/('real_features.npy' if split=='real' else f'{es}_{split}_features.npy'))
    fine=np.load(e.PREVIOUS/f'fine_mass_context/features/{dep}'/('real.npy' if split=='real' else f'{es}_{split}.npy'))
    x=mass.arrange(coarse,fine);ck=torch.load(cp,weights_only=False,map_location='cpu')
    model=Predictor().cuda().eval();model.load_state_dict(ck['model'])
    p=np.exp(infer(model,(x-ck['mu'])/ck['sd'],ck['mass_temperature'],ck['conditional_temperature'])).astype(np.float32)
    p/=p.sum(1,keepdims=True)
    if split=='real':
        full,events=dev.real_inputs(dep);valid=events.strict_h1l1_preprocessing_pass.to_numpy(bool)
        pp=np.full((len(full),p.shape[1]),np.nan,np.float32);pp[valid]=p;p=pp
    cache.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(cache,p=p)
    return {'p':p}


def ensemble(root,dep,es,split):
    total=None
    for ms in e.body.MODEL_SEEDS:
        a=prediction(root,dep,ms,es,split)
        if total is None:total=a['p'].astype(float)
        else:total+=a['p']
    return {**a,'p':total/3}


def pair_features(a, i, j, prior, arm):
    raw=a['p'].reshape(-1,*SHAPE)
    m=raw.sum((2,3));cdf=np.c_[np.zeros(len(m)),m.cumsum(1)]
    pm=np.diff(np.stack([np.interp(mass.EDGES,mass.NATIVE_EDGES,v) for v in cdf]),1).clip(1e-300)
    pm/=pm.sum(1,keepdims=True)
    bc=np.sqrt(pm[i]*pm[j]).sum(1).clip(1e-300,1.)
    outside=m[:,[0,-1]].sum(1)
    if arm=='MASS-Q-PRIOR':p=raw.sum(3).reshape(len(raw),-1);prior=prior.sum(2).reshape(-1)
    elif arm=='MASS-PRIOR':p=m;prior=prior.sum((1,2))
    else:p=raw.reshape(len(raw),-1);prior=prior.reshape(-1)
    valid=np.isfinite(p).all(1)
    p=np.where(valid[:,None],p,0.)
    transformed=np.sqrt(p) if arm=='JOINT-BC' else p/np.sqrt(prior.clip(1e-30))[None]
    t=torch.as_tensor(transformed,device='cuda',dtype=torch.float32)
    kernel=(t@t.T).cpu().numpy().astype(float)
    ov=kernel[i,j].clip(1e-300)
    feature=ov.clip(1e-300,1.) if arm=='JOINT-BC' else np.log(ov)
    return {'bc':bc,'prior_overlap':feature,'ood':(np.maximum(outside[i],outside[j])>.25)|~valid[i]|~valid[j]}


def get_prior(root,dep):
    p=[torch.load(root/f'models/{dep}/seed_{ms}/validation_selected_model.pt',weights_only=False,map_location='cpu')['prior'] for ms in e.body.MODEL_SEEDS]
    assert np.array_equal(p[0],p[1]) and np.array_equal(p[0],p[2])
    return p[0].astype(float)


def calibrate(root,dep,es,arm):
    path=root/f'contracts/{dep}_{arm}_CALIBRATION.json'
    if path.exists():return json.loads(path.read_text())
    a=ensemble(root,dep,es,'development');i,j=np.triu_indices(len(a['p']),1)
    y=a['group'][i]==a['group'][j];x=pair_features(a,i,j,get_prior(root,dep),arm)
    spec={'mass_reference':np.sort(-np.log(x['bc'][y])).tolist(),'source_pairs':int(y.sum()),'score_mode':'LATEST_OMC_PLUS'}
    weights=np.where(y,.5/y.sum(),.5/(~y).sum())
    iso=IsotonicRegression(increasing=True,out_of_bounds='clip').fit(x['prior_overlap'],y,sample_weight=weights)
    p=iso.y_thresholds_.clip(1/(y.sum()+2),1-1/(y.sum()+2))
    spec['prior_overlap']={'knots':iso.X_thresholds_.tolist(),'loglr':(np.log(p)-np.log1p(-p)).tolist(),
                           'minimum':float(x['prior_overlap'].min()),'maximum':float(x['prior_overlap'].max())}
    dev.json_write(path,spec);return spec


def values(root,dep,es,split,arm):
    f=pd.read_parquet(e.TRIAL/f'evaluation/{dep}/seed_{es}/{split}_pairs.parquet')
    cache=root/f'cache/pair_features/{arm}/{dep}_{es}_{split}.npz'
    if cache.exists():return f,dict(np.load(cache))
    a=ensemble(root,dep,es,split)
    x=pair_features(a,f.idx_i.to_numpy(int),f.idx_j.to_numpy(int),get_prior(root,dep),arm)
    cache.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(cache,**x)
    return f,x


def score(f,x,spec,arm):
    if spec.get('unchanged_baseline'):return f.waveform_score.to_numpy(float),np.zeros(len(f)),np.zeros(len(f))
    return e.evaluate.score(f,x,spec,'PRIOR')


def run(root,arm):
    e.values=lambda r,d,s,sp:values(r,d,s,sp,arm)
    e.calibrate=lambda r,d,s:calibrate(r,d,s,arm)
    e.score=score;e.GAMMAS=(0.,.125,.25,.5,1.);e.BETAS=(0.,.0625,.125,.25,.5,1.,2.,4.)
    original=e.batch_target
    comparator=pd.read_csv(root/'comparators/O4a_GLOBAL_PRIOR/pe_official_budget.csv')
    def target(combos,pools,pe,base,dep):
        out=original(combos,pools,pe,base,dep)
        if dep=='gwtc4':
            for b in (10,20):
                row=comparator[(comparator.seed.astype(str)=='consensus')&(comparator.method=='C_fixed')&(comparator.budget==b)].iloc[0]
                for key in ('BC_mc_ge_0p5','Dmax_le_3','official_frontend','official_hanabi','median_BC_mc'):
                    out['target_pass']&=out[f'Top{b}_{key}']>=row[key]-1e-12
                out['target_pass']&=out[f'Top{b}_catastrophic_mc']<=row.catastrophic_mc
        return out
    e.batch_target=target;e.run(root,arm)


def tests():
    t=np.array([[np.log(20),.5,.1],[np.log(40),1.,-.5]])
    i,w=targets(t)
    centers=np.array(np.meshgrid(mass.LOG_CENTERS,Q,CHI,indexing='ij')).reshape(3,-1).T
    assert np.allclose((centers[i]*w[:,:,None]).sum(1),t,atol=1e-7)
    p=np.full((3,np.prod(SHAPE)),1/np.prod(SHAPE))
    a={'p':p};x=pair_features(a,np.array([0,1]),np.array([1,0]),np.full(SHAPE,1/np.prod(SHAPE)),'JOINT-PRIOR')
    assert np.allclose(x['prior_overlap'],0,atol=1e-5) and x['bc'][0]==x['bc'][1]
    print('intrinsic-grid target conservation,uniform neutrality,pair swap: PASS',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--kind',choices=('CONDITIONAL','JOINT'),default='CONDITIONAL')
    p.add_argument('--phase',choices=('init','train','evaluate','test'),required=True);p.add_argument('--arm',choices=('JOINT-PRIOR','JOINT-BC','MASS-Q-PRIOR','MASS-PRIOR'),default='JOINT-PRIOR')
    a=p.parse_args()
    if a.phase=='init':initialize(a.root,a.kind)
    elif a.phase=='train':
        for dep in e.old.DEPS:
            for ms,seed in zip(e.body.MODEL_SEEDS,SEEDS):train(a.root,dep,ms,seed,a.kind)
    elif a.phase=='test':tests()
    else:run(a.root,a.arm)
