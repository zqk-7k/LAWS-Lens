#!/usr/bin/env python3
"""Joint mass-axis CNN and symmetric source-identity contrastive representation."""
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
from scipy.special import softmax
from sklearn.isotonic import IsotonicRegression
import mcwf_omc_ensemble_extension_20260907 as e
import mcwf_omc_architecture_extension_20260907 as arch
import mcwf_ordered_mass_predictor_20260907 as omc

dev = e.dev
SEEDS = (202609601, 202609602, 202609603)
EPOCHS = 15
torch.set_num_threads(2)


class Predictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = arch.Predictor('WIDE')
        self.source_head = nn.Sequential(nn.Linear(96+253, 256), nn.SiLU(), nn.Dropout(.1), nn.Linear(256, 128))

    def forward(self, x):
        b = self.backbone
        h = b.local(b.first(torch.cat([x, b.coordinate.expand(len(x), -1, -1)], 1)))
        logits = b.last(h).squeeze(1).float()
        p = F.softmax(logits, -1)
        pooled = (h.float() * p[:, None, :]).sum(-1)
        z = F.normalize(self.source_head(torch.cat([pooled, p], -1)).float(), dim=-1)
        return logits, z


@torch.no_grad()
def infer(model, x):
    model.eval()
    logits, features = [], []
    for start in range(0, len(x), 256):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            p, z = model(torch.as_tensor(x[start:start+256], dtype=torch.float32, device='cuda'))
        logits.append(p.cpu().numpy())
        features.append(z.cpu().numpy())
    return np.concatenate(logits), np.concatenate(features)


def source_loss(z):
    sim = z.float() @ z.float().T / .1
    sim = sim.masked_fill(torch.eye(len(z), dtype=torch.bool, device=z.device), -torch.inf)
    positive = torch.arange(len(z), device=z.device) ^ 1
    return (torch.logsumexp(sim, -1) - sim[torch.arange(len(z), device=z.device), positive]).mean()


def initialize(root, parent):
    e.initialize(root)
    dev.json_write(root / 'contracts/SOURCE_CONSISTENCY_ADDENDUM.json', {
        'utc': datetime.now(timezone.utc).isoformat(), 'parent_model_root': str(parent),
        'start': 'Warm start each WIDE checkpoint; leave parent and old main encoders untouched',
        'features': 'Same2s waveform and27x253 ordered template response input',
        'source_head': 'Posterior-weighted96feature pooling plus253mass probabilities ->256SiLU/dropout.1 ->128L2embedding',
        'loss': 'Interpolated mass cross entropy +.25 symmetric same-source two-view InfoNCE,temperature.1',
        'training': '15epochs,AdamWlr2e-5WD1e-4,128sources x2 independent views,clip5,cosine min2e-6',
        'selection': 'minimum development massCE +.25 development paired InfoNCE on deterministic128source batches',
        'temperature': 'Mass CE grid same as OMC after checkpoint selection',
        'seeds': SEEDS, 'no_real_PE_or_official_training_targets': True,
        'calibration': 'source cosine via512 source-pair,class-balanced isotonic;positive reward only with supported mass compatibility;cap4',
        'same_O3_O4a': True, 'time_sky_outer_weights_frozen': True,
        'pair_selection_uses_real_development': True, 'references': ['https://arxiv.org/abs/2004.11362'],
        'not_full_PE': True})
    shutil.copy2(__file__, root / 'scripts' / Path(__file__).name)


def train(root, parent, dep, ms, seed):
    out = root / f'models/{dep}/seed_{ms}'
    if (out / 'COMPLETE.json').exists():
        return
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    x, tm = omc.data(e.PREVIOUS, dep, 'train')
    v, vm = omc.data(e.PREVIOUS, dep, 'validation')
    assert not set(tm.source_uid) & set(vm.source_uid)
    assert not set(tm.noise_bank_index) & set(vm.noise_bank_index)
    groups, names = pd.factorize(tm.source_uid, sort=True)
    vg, _ = pd.factorize(vm.source_uid, sort=True)
    vt = np.log(vm.mc_det.to_numpy(float))
    y, vy = omc.targets(np.log(tm.mc_det.to_numpy(float))), omc.targets(vt)
    ck0 = torch.load(parent / f'models/{dep}/seed_{ms}/validation_selected_model.pt', weights_only=False, map_location='cpu')
    mu, sd = ck0['mu'], ck0['sd']
    x -= mu
    x /= sd
    v = (v-mu)/sd
    xx, yy = torch.as_tensor(x, dtype=torch.float32, device='cuda'), torch.as_tensor(y, device='cuda')
    members = [np.flatnonzero(groups == k) for k in range(len(names))]
    vmembers = [np.flatnonzero(vg == k) for k in np.unique(vg)]
    vid = np.stack([r[:2] for r in vmembers]).reshape(-1)
    dev.TRAIN.seed_everything(seed)
    model = Predictor().cuda()
    model.backbone.load_state_dict(ck0['model'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS, eta_min=2e-6)
    history, best = [], float('inf')
    for epoch in range(1, EPOCHS+1):
        rng = np.random.default_rng(seed+epoch)
        order = rng.permutation(len(names))
        model.train()
        losses = []
        for start in range(0, len(order), 128):
            ix = np.stack([rng.choice(members[k], 2, replace=False) for k in order[start:start+128]]).reshape(-1)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits, z = model(xx[ix])
                ce = -(F.log_softmax(logits, -1)*yy[ix]).sum(-1).mean()
                loss = ce + .25*source_loss(z)
            assert torch.isfinite(loss)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            losses.append(float(loss.detach()))
        scheduler.step()
        logits, z = infer(model, v)
        p = softmax(logits.astype(float), 1)
        ce = float(-(vy*np.log(p.clip(1e-30))).sum(-1).mean())
        nce = np.mean([float(source_loss(torch.as_tensor(z[vid[k:k+256]], device='cuda'))) for k in range(0, len(vid), 256)])
        criterion = ce + .25*nce
        row = {'epoch': epoch, 'loss': float(np.mean(losses)), 'validation_CE': ce, 'validation_NCE': float(nce),
               'criterion': float(criterion), 'logMc_MAE': float(abs(p@omc.LOG_CENTERS-vt).mean()), 'seconds': time.perf_counter()-started}
        history.append(row)
        if criterion < best:
            best = criterion
            torch.save({'model': {k: v.detach().cpu().clone() for k,v in model.state_dict().items()},
                        'epoch': epoch, 'mu': mu, 'sd': sd, 'prior': ck0['prior'], 'seed': seed}, out / 'validation_selected_model.pt')
        dev.csv_write(out / 'history.csv', pd.DataFrame(history))
        if epoch % 5 == 0:
            print(json.dumps({'joint_training': [dep, ms], **row}), flush=True)
    ck = torch.load(out / 'validation_selected_model.pt', weights_only=False, map_location='cpu')
    model.load_state_dict(ck['model'])
    logits, z = infer(model, v)
    grid = []
    for t in (.5, .75, 1., 1.25, 1.5, 2.):
        p = softmax(logits.astype(float)/t, 1)
        grid.append({'temperature': t, 'CE': float(-(vy*np.log(p.clip(1e-30))).sum(-1).mean())})
    ck['temperature'] = min(grid, key=lambda r: (r['CE'], abs(r['temperature']-1)))['temperature']
    p, outside = omc.probability(logits, ck['temperature'])
    np.savez_compressed(out / 'validation_predictions.npz', p=p, outside=outside, z=z, group=vg, truth=vt)
    torch.save(ck, out / 'validation_selected_model.pt')
    dev.csv_write(out / 'temperature_grid.csv', pd.DataFrame(grid))
    report = {'deployment': dep, 'seed': seed, 'selected_epoch': ck['epoch'], 'temperature': ck['temperature'],
              'logMc_MAE': float(abs(p@omc.CENTERS-vt).mean()), 'source_overlap': 0, 'noise_overlap': 0,
              'seconds': time.perf_counter()-started, 'checkpoint_sha256': dev.sha(out / 'validation_selected_model.pt')}
    dev.json_write(out / 'COMPLETE.json', report)
    print(json.dumps({'joint_complete': report}), flush=True)


def prediction(root, dep, es, split):
    ms = dict(zip(dev.SEEDS, e.body.MODEL_SEEDS))[es]
    out = root / f'models/{dep}/seed_{ms}'
    if split == 'development':
        return dict(np.load(out / 'validation_predictions.npz'))
    path = root / f'cache/predictions/{dep}/{es}_{split}.npz'
    if path.exists():
        return dict(np.load(path))
    ck = torch.load(out / 'validation_selected_model.pt', weights_only=False, map_location='cpu')
    coarse = np.load(e.old.TRAINED / f'cache/deployment_event_psd/{dep}' / ('real_features.npy' if split == 'real' else f'{es}_{split}_features.npy'))
    refined = np.load(e.PREVIOUS / f'fine_mass_context/features/{dep}' / ('real.npy' if split == 'real' else f'{es}_{split}.npy'))
    x = omc.arrange(coarse, refined)
    model = Predictor().cuda().eval()
    model.load_state_dict(ck['model'])
    logits, z = infer(model, (x-ck['mu'])/ck['sd'])
    p, outside = omc.probability(logits, ck['temperature'])
    if split == 'real':
        full, events = dev.real_inputs(dep)
        valid = events.strict_h1l1_preprocessing_pass.to_numpy(bool)
        pp, bb, zz = np.full((len(full), 512), np.nan), np.full(len(full), np.nan), np.full((len(full), 128), np.nan)
        pp[valid], bb[valid], zz[valid] = p, outside, z
        p, outside, z = pp, bb, zz
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, p=p, outside=outside, z=z)
    return dict(np.load(path))


def pair_features(a, i, j, prior):
    x = e.evaluate.pair_features(a, i, j, prior)
    x['cosine'] = np.sum(a['z'][i]*a['z'][j], 1)
    return x


def calibration(root, dep, es):
    path = root / f'contracts/{dep}_{es}_SOURCE_CALIBRATION.json'
    if path.exists():
        return json.loads(path.read_text())
    a = prediction(root, dep, es, 'development')
    i, j = np.triu_indices(len(a['p']), 1)
    y = a['group'][i] == a['group'][j]
    x = pair_features(a, i, j, e.prior(dep))
    w = np.where(y, .5/y.sum(), .5/(~y).sum())
    spec = {'mass_reference': np.sort(-np.log(x['bc'][y])).tolist(), 'source_pairs': int(y.sum()), 'noise_blocks': 32}
    for name in ('bc', 'prior_overlap', 'cosine'):
        iso = IsotonicRegression(increasing=True, out_of_bounds='clip').fit(x[name], y, sample_weight=w)
        p = iso.y_thresholds_.clip(1/(y.sum()+2), 1-1/(y.sum()+2))
        spec[name] = {'knots': iso.X_thresholds_.tolist(), 'loglr': (np.log(p)-np.log1p(-p)).tolist(),
                      'minimum': float(x[name].min()), 'maximum': float(x[name].max())}
    dev.json_write(path, spec)
    return spec


def values(root, dep, es, split):
    f = pd.read_parquet(e.TRIAL / f'evaluation/{dep}/seed_{es}/{split}_pairs.parquet')
    a = prediction(root, dep, es, split)
    return f, pair_features(a, f.idx_i.to_numpy(int), f.idx_j.to_numpy(int), e.prior(dep))


def run(root, arm):
    e.values, e.calibrate = values, calibration
    original_score = e.score
    def source_score(f, x, spec, ignored_arm):
        if spec.get('unchanged_baseline'):
            return original_score(f, x, spec, ignored_arm)
        if arm == 'SOURCE-COSINE':
            xx = {**x, 'prior_overlap': x['cosine']}
            ss = {**spec, 'prior_overlap': spec['cosine']}
        else:
            xx, ss = x, spec
        return original_score(f, xx, ss, 'MIXTURE-PRIOR')
    e.score = source_score
    e.run(root, arm)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--parent', type=Path)
    p.add_argument('--phase', choices=('init', 'train', 'evaluate'), required=True)
    p.add_argument('--arm', choices=('SOURCE-COSINE', 'SOURCE-MASS'), default='SOURCE-COSINE')
    a = p.parse_args()
    if a.phase == 'init': initialize(a.root, a.parent)
    elif a.phase == 'train':
        for dep in e.old.DEPS:
            for ms, seed in zip(e.body.MODEL_SEEDS, SEEDS): train(a.root, a.parent, dep, ms, seed)
    else: run(a.root, a.arm)
