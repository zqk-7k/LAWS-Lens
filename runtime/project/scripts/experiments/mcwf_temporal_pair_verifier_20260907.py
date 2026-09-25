#!/usr/bin/env python3
"""Paired temporal/global readout control with a frozen waveform backbone."""
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
from sklearn.isotonic import IsotonicRegression
import mcwf_pe_frontend_extension_v2_20260907 as e
import mcwf_temporal_pair_features_20260907 as latent
import mcwf_finite_reference_tail_20260907 as tail

dev, body, ev = e.dev, e.body, e.ev
torch.set_num_threads(2)
EPOCHS = 25
SEEDS = (202609361, 202609362, 202609363)
BETAS = (0., .0625, .125, .25, .5, 1., 2.)


class Verifier(nn.Module):
    def __init__(self, temporal=True):
        super().__init__()
        self.temporal = temporal
        self.project = nn.Sequential(nn.LayerNorm(128), nn.Linear(128, 32), nn.SiLU())
        self.compare = nn.Sequential(nn.Linear(64, 32), nn.SiLU())
        self.global_project = nn.Sequential(nn.Linear(192, 32), nn.LayerNorm(32), nn.SiLU())
        self.head = nn.Sequential(nn.Linear(128, 64), nn.SiLU(), nn.Dropout(.1), nn.Linear(64, 1))
        position = torch.arange(64, dtype=torch.float32)[:, None]
        frequency = torch.exp(torch.arange(0, 32, 2)*(-np.log(10000.)/32))
        pe = torch.zeros(64, 32)
        pe[:, 0::2], pe[:, 1::2] = torch.sin(position*frequency), torch.cos(position*frequency)
        self.register_buffer('position', .1*pe)

    def direction(self, a, b):
        context = F.scaled_dot_product_attention(a[:, None], b[:, None], b[:, None], dropout_p=0.)[:, 0]
        d = self.compare(torch.cat([(a-context).abs(), a*context], -1))
        return torch.cat([d.mean(1), d.amax(1)], -1)

    def forward(self, a, b, ga, gb):
        if not self.temporal:
            a, b = a.mean(1, keepdim=True).expand(-1, 64, -1), b.mean(1, keepdim=True).expand(-1, 64, -1)
        a, b = self.project(a)+self.position, self.project(b)+self.position
        aligned = .5*(self.direction(a, b)+self.direction(b, a))
        ga, gb = self.global_project(ga), self.global_project(gb)
        return self.head(torch.cat([aligned, (ga-gb).abs(), ga*gb], -1)).squeeze(-1)


def initialize(root):
    folder = root / 'temporal_pair'
    path = folder / 'contracts/TRAINING.json'
    if path.exists():
        return
    dev.json_write(path, {
        'utc': datetime.now(timezone.utc).isoformat(), 'same_both_runs': True,
        'input': 'unchanged peak2s4096 H1L1,40-580Hz; frozen larger-population encoder before pooling',
        'features': '128channels x64 temporal bins before global readout; frozen128D embedding plus64 predictive mass probabilities',
        'controlled_arms': ['POOL-CONTROL', 'TEMPORAL'],
        'control': 'identical parameter counts/seeds/data/optimizer; pool-control averages tokens over time before the identical pair head',
        'pair_head': 'shared128->32 token projection, fixed sinusoidal positions, symmetric two-direction scaled-dot-product cross-attention; mean/max residual-product readout plus symmetric global features',
        'labels': 'simulation source identity;4 independent a/b positives per training source per epoch;equal negative count,half uniform,half among32 nearest frozen source latents with abs(logMc gap)>=0.15',
        'sources': '4096 training parents/96 noise blocks,512 validation parents/32 disjoint noise blocks;frozen backbone had12288 training parents but no validation parents',
        'seeds': SEEDS, 'optimizer': 'AdamW3e-4,weightdecay.01,25epochs,cosine_min1e-5,batch256,gradientclip5',
        'early_stop': 'earliest minimum balanced BCE on all512 validation companions plus a fixed source-labeled sample of16384 random non-companion pairs;no real labels',
        'calibration': 'selected-model logits for all512-source validation unorderedpairs;equal total classweight isotonic,finite512-system floor,cap+-4,positiveOOD0;correlated pairs are not independent systems',
        'score': 'Z_FRT+beta*new pair-verifier evidence;positive increments require frozenRNC mass-tail>=.05',
        'beta_grid': BETAS, 'frozen': ['time', 'sky', 'outerweights', 'scope', 'historical files'],
        'interpretation': 'engineering ablation,not fullPE or properBF; source validation reused adaptively;new independent confirmation required',
        'references': ['https://arxiv.org/abs/1706.03762', 'https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html'],
        'reference_limit': 'attention implementation motivation only;no literature claim for our architecture or tolerances'})
    (folder / 'scripts').mkdir(exist_ok=True)
    shutil.copy2(__file__, folder / 'scripts' / Path(__file__).name)


@torch.no_grad()
def infer(net, x, g, i, j):
    net.eval()
    x = torch.as_tensor(x, dtype=torch.float16, device='cuda')
    g = torch.as_tensor(g, dtype=torch.float32, device='cuda')
    result = []
    for start in range(0, len(i), 512):
        a, b = i[start:start+512], j[start:start+512]
        with torch.autocast('cuda', dtype=torch.bfloat16):
            v = net(x[a].float(), x[b].float(), g[a], g[b])
        result.append(v.float().cpu().numpy())
    return np.concatenate(result)


def train(root, dep, ms, seed, arm):
    initialize(root)
    out = root / f'temporal_pair/models/{arm}/{dep}/seed_{ms}'
    if (out / 'COMPLETE.json').exists():
        return
    out.mkdir(parents=True, exist_ok=True)
    tx, tg = latent.features(root, dep, ms, 0, 'train')
    vx, vg = latent.features(root, dep, ms, 0, 'development')
    meta = pd.read_parquet(root / f'expanded_data/{dep}/train/event_metadata.parquet')
    vm = pd.read_parquet(root / f'expanded_data/{dep}/validation/event_metadata.parquet')
    if set(meta.source_uid)&set(vm.source_uid) or set(meta.noise_bank_index)&set(vm.noise_bank_index):
        raise RuntimeError('Source/noise split overlap')
    groups, names = pd.factorize(meta.source_uid, sort=True)
    members = [np.flatnonzero(groups == k) for k in range(len(names))]
    image = meta.image.to_numpy()
    image_a = [np.flatnonzero((groups == k)&(image == 'a')) for k in range(len(names))]
    image_b = [np.flatnonzero((groups == k)&(image == 'b')) for k in range(len(names))]
    if any(len(a) != 4 or len(b) != 4 for a, b in zip(image_a, image_b)):
        raise RuntimeError('Expected four independent noise views per image')
    mass = np.array([np.log(meta.mc_det.iloc[ids[0]]) for ids in members])
    centers = np.stack([tg[ids, :128].mean(0) for ids in members])
    centers /= np.linalg.norm(centers, axis=1, keepdims=True).clip(1e-12)
    sim = centers @ centers.T
    sim[abs(mass[:, None]-mass[None, :]) < .15] = -np.inf
    neighbors = np.argsort(-sim, axis=1, kind='stable')[:, :32]
    if not np.isfinite(np.take_along_axis(sim, neighbors, 1)).all():
        raise RuntimeError('Insufficient training hard negatives')
    del sim
    mu, sd = tg.mean(0), tg.std(0).clip(.02)
    tg, vg = (tg-mu)/sd, (vg-mu)/sd
    i, j = np.triu_indices(len(vx), 1)
    truth = vm.source_uid.to_numpy()
    vy = truth[i] == truth[j]
    rng = np.random.default_rng(202609360)
    select = np.r_[np.flatnonzero(vy), rng.choice(np.flatnonzero(~vy), 16384, replace=False)]
    si, sj, sy = i[select], j[select], vy[select]
    sw = np.where(sy, .5/sy.sum(), .5/(~sy).sum())
    dev.json_write(out / 'DATA_AUDIT.json', {'training_sources': len(names), 'validation_sources': len(set(truth)),
        'source_overlap': 0, 'noise_overlap': 0, 'validation_selection_pairs': len(select),
        'same_validation_subset_for_all_arms_and_seeds': True})
    dev.TRAIN.seed_everything(seed)
    net = Verifier(arm == 'TEMPORAL').cuda()
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS, eta_min=1e-5)
    x = torch.as_tensor(np.asarray(tx), dtype=torch.float16, device='cuda')
    g = torch.as_tensor(tg, dtype=torch.float32, device='cuda')
    vxx = torch.as_tensor(np.asarray(vx), dtype=torch.float16, device='cuda')
    history, best, first = [], float('inf'), 1
    resume = out / 'resume.pt'
    if resume.exists():
        ck = torch.load(resume, weights_only=False, map_location='cpu')
        net.load_state_dict(ck['model']); opt.load_state_dict(ck['optimizer']); sched.load_state_dict(ck['scheduler'])
        history, best, first = ck['history'], ck['best'], ck['epoch']+1
        torch.set_rng_state(ck['rng']); torch.cuda.set_rng_state_all(ck['cuda_rng'])
    started = time.perf_counter()
    for epoch in range(first, EPOCHS+1):
        rng = np.random.default_rng(seed+epoch)
        aa, bb, yy = [], [], []
        for _ in range(4):
            for k in rng.permutation(len(names)):
                u, v = int(rng.choice(image_a[k])), int(rng.choice(image_b[k]))
                aa.append(u); bb.append(v); yy.append(1.)
                other = int(rng.choice(neighbors[k])) if rng.random() < .5 else (k+int(rng.integers(1, len(names)))) % len(names)
                aa.append(u); bb.append(int(rng.choice(members[other]))); yy.append(0.)
        aa, bb, yy = np.array(aa), np.array(bb), np.array(yy, np.float32)
        order = rng.permutation(len(aa))
        net.train(); losses = []
        for start in range(0, len(aa), 256):
            ids = order[start:start+256]
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                pred = net(x[aa[ids]].float(), x[bb[ids]].float(), g[aa[ids]], g[bb[ids]])
                loss = F.binary_cross_entropy_with_logits(pred.float(), torch.as_tensor(yy[ids], device='cuda'))
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite temporal pair loss')
            loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 5); opt.step()
            losses.append(float(loss.detach()))
        sched.step()
        logits = infer(net, vxx, vg, si, sj)
        ce = float(np.sum(sw*(np.logaddexp(0., logits)-sy*logits)))
        row = {'epoch': epoch, 'training_BCE': float(np.mean(losses)), 'validation_balanced_BCE': ce,
               'elapsed_seconds': time.perf_counter()-started}
        history.append(row)
        if ce < best:
            best = ce
            torch.save({'model': {k:v.detach().cpu().clone() for k,v in net.state_dict().items()}, 'mu':mu, 'sd':sd,
                        'arm':arm, 'seed':seed, 'epoch':epoch, 'validation_BCE':ce}, out / 'validation_selected_model.pt')
        torch.save({'model':net.state_dict(), 'optimizer':opt.state_dict(), 'scheduler':sched.state_dict(),
                    'history':history, 'best':best, 'epoch':epoch, 'rng':torch.get_rng_state(),
                    'cuda_rng':torch.cuda.get_rng_state_all()}, resume)
        dev.csv_write(out / 'history.csv', pd.DataFrame(history))
        if epoch == 1 or epoch % 5 == 0:
            print(json.dumps({'temporal_pair_training': [arm,dep,seed], **row}), flush=True)
    ck = torch.load(out / 'validation_selected_model.pt', weights_only=False, map_location='cpu')
    net.load_state_dict(ck['model'])
    logits = infer(net, vxx, vg, i, j)
    weight = np.where(vy, .5/vy.sum(), .5/(~vy).sum())
    iso = IsotonicRegression(increasing=True, out_of_bounds='clip').fit(logits, vy, sample_weight=weight)
    floor = 1/(vy.sum()+2)
    prob = iso.y_thresholds_.clip(floor, 1-floor)
    dev.json_write(out / 'CALIBRATION.json', {'knots':iso.X_thresholds_.tolist(), 'loglr':(np.log(prob)-np.log1p(-prob)).tolist(),
        'fit_min':float(logits.min()), 'fit_max':float(logits.max()), 'independent_positive_sources':int(vy.sum()),
        'dependent_null_pairs':int((~vy).sum())})
    np.savez_compressed(out / 'validation_logits.npz', logits=logits, idx_i=i, idx_j=j, is_true_pair=vy)
    net.eval()
    with torch.no_grad():
        error = float((net(x[:5].float(),x[5:10].float(),g[:5],g[5:10])-net(x[5:10].float(),x[:5].float(),g[5:10],g[:5])).abs().max())
    if error > 1e-6:
        raise RuntimeError('Pair swap asymmetry')
    info = {'arm':arm, 'deployment':dep, 'seed':seed, 'selected_epoch':ck['epoch'],
            'validation_BCE':ck['validation_BCE'], 'swap_max_error':error,
            'seconds':time.perf_counter()-started, 'model_sha256':dev.sha(out / 'validation_selected_model.pt')}
    dev.json_write(out / 'COMPLETE.json', info)
    print(json.dumps({'temporal_pair_done':info}), flush=True)


def values(root, dep, ms, es, split, arm):
    f = pd.read_parquet(e.BASE / f'evaluation/{dep}/seed_{es}/{split}_pairs.parquet')
    cp = root / f'temporal_pair/models/{arm}/{dep}/seed_{ms}/validation_selected_model.pt'
    if not (cp.parent / 'COMPLETE.json').exists():
        raise RuntimeError('Incomplete pair verifier')
    path = root / f'temporal_pair/predictions/{arm}/{dep}/model_{ms}_eval_{es}/{split}.npz'
    if path.exists():
        a = np.load(path)
        if str(a['checkpoint_sha256']) != dev.sha(cp) or not np.array_equal(a['idx_i'], f.idx_i) or not np.array_equal(a['idx_j'], f.idx_j):
            raise RuntimeError('Prediction provenance mismatch')
        return f, a['logits']
    x, g = latent.features(root, dep, ms, es, split)
    ck = torch.load(cp, weights_only=False, map_location='cpu')
    model = Verifier(arm == 'TEMPORAL').cuda().eval()
    model.load_state_dict(ck['model'])
    v = infer(model, x, (g-ck['mu'])/ck['sd'], f.idx_i.to_numpy(int), f.idx_j.to_numpy(int))
    if not np.isfinite(v).all():
        raise RuntimeError('Nonfinite strict-scope pair logits')
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, logits=v, idx_i=f.idx_i, idx_j=f.idx_j, checkpoint_sha256=dev.sha(cp))
    return f, v


def score(f, raw, spec):
    value = np.interp(raw, spec['knots'], spec['loglr']).clip(-4.,4.)
    ood = (raw < spec['fit_min']) | (raw > spec['fit_max'])
    p = tail.tail_probability(-np.log(f.new_mass_predictive_BC.to_numpy(float).clip(1e-12)), spec['mass_reference'])
    value = np.where(ood | (p < .05), np.minimum(value,0.), value)
    return f.waveform_score.to_numpy(float)+spec['beta']*value, value, ood


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    a = p.parse_args()
    for ms, seed in zip(body.MODEL_SEEDS, SEEDS):
        for dep in e.DEPS:
            for arm in ('POOL-CONTROL','TEMPORAL'):
                train(a.root, dep, ms, seed, arm)
