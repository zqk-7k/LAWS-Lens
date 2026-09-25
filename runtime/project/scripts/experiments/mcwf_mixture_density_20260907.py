#!/usr/bin/env python3
"""Waveform-only conditional intrinsic density, not strain-level Bayesian PE."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
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
from scipy.special import expit
import mcwf_pe_frontend_extension_v2_20260907 as e
import mcwf_dense_features_20260907 as dense

dev, body = e.dev, e.body
torch.set_num_threads(2)
SEEDS = (202609141, 202609142, 202609143)
K, D, EPOCHS = 4, 3, 60


class Density(nn.Module):
    def __init__(self, dim=10752):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, 256), nn.LayerNorm(256), nn.SiLU(), nn.Dropout(.1),
            nn.Linear(256, 128), nn.SiLU(), nn.Linear(128, K*10))

    def forward(self, x):
        a = self.net(x).reshape(-1, K, 10)
        logits, mean = a[..., 0], a[..., 1:4]
        scale = torch.zeros((*a.shape[:2], 3, 3), device=a.device, dtype=a.dtype)
        rows, cols = torch.tril_indices(3, 3, device=a.device)
        scale[..., rows, cols] = a[..., 4:10]
        ii = torch.arange(3, device=a.device)
        scale[..., ii, ii] = F.softplus(scale[..., ii, ii])+.03
        return logits, mean, scale


def logprob(params, truth, temperature=1.):
    logits, mean, scale = params
    distribution = torch.distributions.MultivariateNormal(mean, scale_tril=scale*np.sqrt(temperature))
    return torch.logsumexp(F.log_softmax(logits, -1)+distribution.log_prob(truth[:, None]), -1)


def initialize(root):
    out = root / 'mixture_density'
    path = out / 'contracts/TRAINING.json'
    if path.exists():
        return
    dev.json_write(path, {'created_utc': datetime.now(timezone.utc).isoformat(), 'same_O3_O4a': True,
        'data': 'new4096training/512validation source parents; source/noise blocks disjoint; frozen expanded_data contract',
        'input': 'H1L1 peak2s4096points physical40-580Hz; event-PSD phase-quadrature log1p power from3584IMRPhenomD templates; no metadata as network inputs',
        'network': '10752->256 LayerNorm SiLU dropout0.1->128 SiLU->4 full-covariance3D Gaussian components',
        'targets': ['log_detector_frame_chirp_mass', 'logit_mass_ratio', 'atanh_chi_eff'],
        'normalization': 'training feature and target means/SD only',
        'numeric_scale_floor': .03, 'component_count': K, 'seeds': SEEDS,
        'optimizer': 'AdamW2e-4,weightdecay1e-4,cosine_min1e-5,60epochs,128sources*2views,batch256,clip5',
        'selection': 'earliest minimum512source validation mixture NLL; global covariance multiplier by same validation NLL from0.5,0.75,1,1.25,1.5,2',
        'loss': 'source-balanced joint mixture negative log likelihood, no real labels, no pair labels needed for network training',
        'interpretation': 'supervised predictive density under the development simulator,not calibrated astrophysical PE or a proper Bayes factor',
        'pair_statistic': 'analytic mixture product integral in standardized target coordinates; marginalMc and joint3D checked separately then calibrated as correlated ranking features',
        'frozen': ['originalC-fixed encoder','RNC-FRT encoder','time','sky','outerweights','scope','historical outputs'],
        'references': ['https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/bishop-ncrg-94-004.pdf',
            'https://docs.pytorch.org/docs/stable/distributions.html'],
        'fresh_confirmation_required': True})
    (out / 'scripts').mkdir(exist_ok=True)
    shutil.copy2(__file__, out / 'scripts' / Path(__file__).name)


def inputs(root, dep, split):
    a = np.load(root / f'expanded_encoder/features/{dep}/{split}.npy')
    b = np.load(root / f'mixture_density/features/{dep}/{split}_extra.npy')
    return np.concatenate([a.reshape(len(a), -1), b.reshape(len(b), -1)], 1)


def targets(meta):
    q = np.asarray(meta.m2_det/meta.m1_det, float)
    chi = (meta.a1*np.cos(meta.tilt1)+q*meta.a2*np.cos(meta.tilt2))/(1+q)
    return np.column_stack([np.log(meta.mc_det), np.log(q.clip(1e-5, 1-1e-5)/(1-q).clip(1e-5)),
        np.arctanh(np.asarray(chi).clip(-.99999, .99999))])


@torch.no_grad()
def infer(model, values, temperature=1., batch=512):
    model.eval()
    ww, mm, ss = [], [], []
    for start in range(0, len(values), batch):
        a = torch.as_tensor(values[start:start+batch], dtype=torch.float32, device='cuda')
        logits, mean, scale = model(a)
        ww.append(F.softmax(logits, -1).cpu().numpy())
        mm.append(mean.cpu().numpy())
        ss.append((scale@scale.transpose(-1, -2)*temperature).cpu().numpy())
    return {'w': np.concatenate(ww), 'm': np.concatenate(mm), 'cov': np.concatenate(ss)}


def train(root, dep, ms, seed):
    initialize(root)
    out = root / f'mixture_density/models/{dep}/seed_{ms}'
    if (out / 'COMPLETE.json').exists():
        return
    out.mkdir(parents=True, exist_ok=True)
    tm = pd.read_parquet(root / f'expanded_data/{dep}/train/event_metadata.parquet')
    vm = pd.read_parquet(root / f'expanded_data/{dep}/validation/event_metadata.parquet')
    if set(tm.source_uid)&set(vm.source_uid) or set(tm.noise_bank_index)&set(vm.noise_bank_index):
        raise RuntimeError('Source/noise overlap')
    group, names = pd.factorize(tm.source_uid, sort=True)
    vg, vn = pd.factorize(vm.source_uid, sort=True)
    x, v = inputs(root, dep, 'train'), inputs(root, dep, 'validation')
    mu, sd = x.mean(0), np.maximum(x.std(0), 1e-4)
    x, v = (x-mu)/sd, (v-mu)/sd
    yy, vv = targets(tm), targets(vm)
    ym, ys = yy.mean(0), np.maximum(yy.std(0), 1e-4)
    yt = torch.as_tensor((yy-ym)/ys, dtype=torch.float32, device='cuda')
    vy = torch.as_tensor((vv-ym)/ys, dtype=torch.float32, device='cuda')
    xx, vx = torch.as_tensor(x, device='cuda'), torch.as_tensor(v, device='cuda')
    members = [np.flatnonzero(group == g) for g in range(len(names))]
    dev.TRAIN.seed_everything(seed)
    model = Density().cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS, eta_min=1e-5)
    history, best, first = [], np.inf, 1
    started = time.perf_counter()
    resume = out / 'resume.pt'
    if resume.exists():
        r = torch.load(resume, weights_only=False, map_location='cpu')
        model.load_state_dict(r['model']); opt.load_state_dict(r['optimizer']); sched.load_state_dict(r['scheduler'])
        history, best, first = r['history'], r['best'], r['epoch']+1
        torch.set_rng_state(r['rng']); torch.cuda.set_rng_state_all(r['cuda_rng'])
    for epoch in range(first, EPOCHS+1):
        model.train()
        rng = np.random.default_rng(seed+epoch)
        order, losses = rng.permutation(len(names)), []
        for begin in range(0, len(order), 128):
            source = order[begin:begin+128]
            idx = np.stack([rng.choice(members[g], 2, replace=False) for g in source]).reshape(-1)
            opt.zero_grad(set_to_none=True)
            loss = -logprob(model(xx[idx]), yt[idx]).mean()
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite density NLL')
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5); opt.step()
            losses.append(float(loss.detach()))
        sched.step(); model.eval()
        with torch.no_grad():
            val = float(-logprob(model(vx), vy).mean())
        history.append({'epoch': epoch, 'train_nll': float(np.mean(losses)), 'validation_nll': val})
        if val < best:
            best = val
            torch.save({'model': {k: v.detach().cpu().clone() for k,v in model.state_dict().items()},
                'mu': mu, 'sd': sd, 'ym': ym, 'ys': ys, 'epoch': epoch, 'seed': seed, 'validation_nll': val},
                out / 'validation_selected_model.pt')
        torch.save({'model': model.state_dict(), 'optimizer': opt.state_dict(), 'scheduler': sched.state_dict(),
            'epoch': epoch, 'history': history, 'best': best, 'rng': torch.get_rng_state(),
            'cuda_rng': torch.cuda.get_rng_state_all()}, resume)
        if epoch % 10 == 0:
            print(json.dumps({'mixture_training': dep, 'seed': seed, **history[-1]}), flush=True)
    ck = torch.load(out / 'validation_selected_model.pt', weights_only=False, map_location='cpu')
    model.load_state_dict(ck['model']); model.eval()
    grid = []
    with torch.no_grad():
        par = model(vx)
        for temperature in (.5, .75, 1., 1.25, 1.5, 2.):
            grid.append({'temperature': temperature, 'nll': float(-logprob(par, vy, temperature).mean())})
    ck['temperature'] = min(grid, key=lambda r: (r['nll'], abs(r['temperature']-1)))['temperature']
    torch.save(ck, out / 'validation_selected_model.pt')
    prediction = infer(model, v, ck['temperature'])
    rng = np.random.default_rng(seed)
    draws = []
    for index in range(len(v)):
        probability = prediction['w'][index].astype(float)
        probability /= probability.sum()
        a = rng.choice(K, size=1024, p=probability)
        g = rng.standard_normal((1024, 3))
        sample = prediction['m'][index,a]+np.einsum('nij,nj->ni',np.linalg.cholesky(prediction['cov'][index,a]),g)
        draws.append(sample*ys+ym)
    draws = np.stack(draws)
    diagnostic = []
    for k, name in enumerate(('logMc', 'logitq', 'atanh_chi_eff')):
        median = np.median(draws[:,:,k],1)
        row = {'parameter': name, 'MAE': float(abs(median-vv[:,k]).mean()), 'bias': float((median-vv[:,k]).mean())}
        for level in (.5,.9):
            lo, hi = np.quantile(draws[:,:,k],[(1-level)/2,(1+level)/2],axis=1)
            row['coverage_'+str(level)] = float(((vv[:,k]>=lo)&(vv[:,k]<=hi)).mean())
        diagnostic.append(row)
    np.savez_compressed(out/'validation_predictions.npz', **prediction, group=vg, truth=(vv-ym)/ys)
    dev.csv_write(out/'history.csv',pd.DataFrame(history)); dev.csv_write(out/'temperature_grid.csv',pd.DataFrame(grid))
    dev.csv_write(out/'predictive_coverage.csv',pd.DataFrame(diagnostic))
    report = {'deployment':dep,'seed':seed,'epoch':ck['epoch'],'temperature':ck['temperature'],
        'NLL':ck['validation_nll'],'seconds':time.perf_counter()-started,'checkpoint_sha256':dev.sha(out/'validation_selected_model.pt'),
        'independent_validation_sources':len(vn),'diagnostics':diagnostic}
    dev.json_write(out/'COMPLETE.json',report)
    print(json.dumps({'mixture_training_done':report}),flush=True)


def prediction(root,dep,ms,es,split):
    cp=root/f'mixture_density/models/{dep}/seed_{ms}/validation_selected_model.pt'
    if not (cp.parent/'COMPLETE.json').exists():
        raise RuntimeError('Density training incomplete')
    path=root/f'mixture_density/predictions/{dep}/model_{ms}_eval_{es}/{split}.npz'
    if path.exists():
        a=np.load(path)
        if str(a['checkpoint_sha256'])!=dev.sha(cp):
            raise RuntimeError('Changed density checkpoint')
        return a
    ck=torch.load(cp,weights_only=False,map_location='cpu')
    feature_path=e.TRAINED/f'cache/deployment_event_psd/{dep}'/('real_features.npy' if split=='real' else f'{es}_{split}_features.npy')
    coarse=np.load(feature_path); extra=dense.deployment(root,dep,es,split)
    x=np.concatenate([coarse.reshape(len(coarse),-1),extra.reshape(len(extra),-1)],1)
    model=Density().cuda().eval(); model.load_state_dict(ck['model'])
    a=infer(model,(x-ck['mu'])/ck['sd'],ck['temperature'])
    if split=='real':
        full,events=dev.real_inputs(dep); valid=events.strict_h1l1_preprocessing_pass.to_numpy(bool)
        valid_predictions=a
        a={k:np.full((len(full),*v.shape[1:]),np.nan) for k,v in valid_predictions.items()}
        for k,v in valid_predictions.items(): a[k][valid]=v
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,**a,checkpoint_sha256=dev.sha(cp))
    return np.load(path)


@torch.no_grad()
def pair_overlap(a,i,j,dimensions=(0,1,2),batch=2048):
    w=torch.as_tensor(a['w'],dtype=torch.float64,device='cuda')
    m=torch.as_tensor(a['m'],dtype=torch.float64,device='cuda')[...,list(dimensions)]
    c=torch.as_tensor(a['cov'],dtype=torch.float64,device='cuda')[...,list(dimensions),:][...,list(dimensions)]
    out=[]
    for start in range(0,len(i),batch):
        ii,jj=i[start:start+batch],j[start:start+batch]
        delta=m[ii,:,None,:]-m[jj,None,:,:]
        cov=c[ii,:,None,:,:]+c[jj,None,:,:,:]
        chol=torch.linalg.cholesky(cov)
        solved=torch.linalg.solve_triangular(chol,delta[...,None],upper=False).squeeze(-1)
        logn=-.5*(len(dimensions)*np.log(2*np.pi)+2*chol.diagonal(dim1=-2,dim2=-1).log().sum(-1)+solved.square().sum(-1))
        val=w[ii,:,None].clamp_min(1e-300).log()+w[jj,None,:].clamp_min(1e-300).log()+logn
        out.append(torch.logsumexp(val.flatten(1),-1).cpu().numpy())
    return np.concatenate(out)


def tests():
    a={'w':np.ones((2,1)),'m':np.array([[[0.,0.,0.]],[[1.,2.,3.]]]),'cov':np.tile(np.eye(3),(2,1,1,1))}
    full=pair_overlap(a,np.array([0]),np.array([1]))[0]
    expected=-1.5*np.log(4*np.pi)-14/4
    marginal=pair_overlap(a,np.array([0]),np.array([1]),(0,))[0]
    symmetric=pair_overlap(a,np.array([1]),np.array([0]))[0]
    if abs(full-expected)>1e-10 or abs(marginal-(-.5*np.log(4*np.pi)-.25))>1e-10 or full!=symmetric:
        raise RuntimeError('Gaussian overlap or symmetry unit test failed')
    return {'gaussian_exact_error':abs(full-expected),'swap_exact_error':abs(full-symmetric),'marginal_exact':True}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    initialize(a.root);dev.json_write(a.root/'mixture_density/contracts/UNIT_TESTS.json',tests())
    for dep in e.DEPS:
        for ms,seed in zip(body.MODEL_SEEDS,SEEDS):train(a.root,dep,ms,seed)
