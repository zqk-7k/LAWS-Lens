#!/usr/bin/env python3
"""Continuous conditional mass density with probability-conserving output bins."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
import argparse
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import pandas as pd
from scipy.special import ndtr, logsumexp, softmax
import torch
from torch import nn
import torch.nn.functional as F

P = Path('/root/autodl-tmp/gw-catalog')
sys.path.insert(0, str(P / 'scripts/experiments'))
import mcwf_multirate_train_20260908 as mult
import mcwf_conditional_intrinsics_20260908 as intrinsic
import mcwf_nodup_reliability_20260909 as reliability
n = reliability.n
LOW = P / 'results/mcwf_multirate_lowband_exploratory_20260908T145551Z'
EDGES = mult.old.EDGES
CENTERS = mult.old.LOG_CENTERS
DX = float(np.median(np.diff(CENTERS)))
TRAIN_SEEDS = (2026090951, 2026090952, 2026090953)
EPOCHS = 30
ROOT = None
torch.set_num_threads(2)


class Predictor(mult.Predictor):
    def __init__(self):
        super().__init__('MULTIRATE')
        self.density = nn.Conv1d(48, 2, 1)
        nn.init.zeros_(self.density.weight)
        nn.init.zeros_(self.density.bias)
        self.density.bias.data[1] = np.log(.5)
        self.register_buffer('mass_centers', torch.as_tensor(CENTERS, dtype=torch.float32)[None])

    def forward(self, x):
        h = self.first(torch.cat([x, self.coordinate.expand(len(x), -1, -1)], 1))
        h = F.silu(h + self.middle(h))
        logits = self.last(h).squeeze(1)
        offset, logsigma = self.density(h).unbind(1)
        mean = self.mass_centers + .5 * DX * torch.tanh(offset)
        sigma = DX * torch.exp(logsigma.clamp(-2., 2.))
        return logits, mean, sigma


def nll(logits, mean, sigma, truth):
    logweight = F.log_softmax(logits.float(), -1)
    mean, sigma = mean.float(), sigma.float()
    lognormal = -.5 * ((truth[:, None] - mean) / sigma)**2 - torch.log(sigma) - .5 * np.log(2 * np.pi)
    lo, hi = (EDGES[0] - mean) / sigma, (EDGES[-1] - mean) / sigma
    component_mass = (torch.special.ndtr(hi) - torch.special.ndtr(lo)).clamp_min(1e-20)
    lognorm = torch.logsumexp(logweight + torch.log(component_mass), -1)
    return -(torch.logsumexp(logweight + lognormal, -1) - lognorm).mean()


@torch.no_grad()
def infer_arrays(model, x, batch=128):
    model.eval()
    result = [[], [], []]
    for start in range(0, len(x), batch):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out = model(torch.as_tensor(x[start:start + batch], dtype=torch.float32, device='cuda'))
        for dst, src in zip(result, out):
            dst.append(src.float().cpu().numpy())
    return [np.concatenate(v).astype(float) for v in result]


def numpy_nll(a, truth, temperature=1.):
    logits, mean, sigma = a
    logweight = logits / temperature - logsumexp(logits / temperature, axis=1, keepdims=True)
    lognormal = -.5 * ((truth[:, None] - mean) / sigma)**2 - np.log(sigma) - .5 * np.log(2 * np.pi)
    mass = (ndtr((EDGES[-1] - mean) / sigma) - ndtr((EDGES[0] - mean) / sigma)).clip(1e-300)
    return float(np.mean(-logsumexp(logweight + lognormal, axis=1) + logsumexp(logweight + np.log(mass), axis=1)))


def probability(a, temperature=1.):
    logits, mean, sigma = a
    weights = softmax(logits / temperature, axis=1)
    output = []
    for start in range(0, len(logits), 32):
        cdf = ndtr((EDGES[None, None] - mean[start:start + 32, :, None]) / sigma[start:start + 32, :, None])
        p = np.sum(np.diff(cdf, axis=-1) * weights[start:start + 32, :, None], axis=1).clip(1e-300)
        p /= p.sum(1, keepdims=True)
        output.append(p)
    p = np.concatenate(output)
    if not np.allclose(p.sum(1), 1, atol=1e-12) or np.any(p < 0):
        raise RuntimeError('Mass probability conservation failed')
    return p, weights[:, [0, -1]].sum(1)


def freeze():
    if ROOT.exists():
        raise RuntimeError('New independent directory required')
    for name in ('contracts', 'models', 'predictions', 'calibration', 'configs', 'tables', 'audit',
                 'reports', 'scripts', 'logs', 'manifest', 'results', 'figures', 'cache'):
        (ROOT / name).mkdir(parents=True)
    contract = {'id': 'MCWF-NODUP-SUBGRID-DENSITY-03', 'utc': n.utc(), 'status': n.STATUS,
        'goal_achieved': False, 'same_both_runs': True,
        'reason': 'Precision-aware calibration did not resolve the case. Test continuous conditional mass density instead of quantized class labels; no public-PE width used as target.',
        'parent': str(LOW), 'input': 'Unchanged54x253 response features from peak2s and low20-80Hz16s branches.',
        'architecture': 'Warm mass-axis CNN, mixture weights plus two local heads per253component: mean=center+0.5grid*tanh(offset);sigma=grid*exp(clip(logsigma,-2,2)).',
        'density': 'Gaussian mixture truncated and normalized on logMc in log5..log200. Continuous NLL trained,512 output masses obtained by CDF integration,not point sampling.',
        'bounds': 'Offsets keep local component identity;scale bounds tied only to existing grid spacing for numerical stability,not chosen from real case. Boundary rates and coverage reported.',
        'training': {'seeds': list(TRAIN_SEEDS), 'epochs': EPOCHS, 'optimizer': 'AdamW1e-4 WD1e-4 cosine1e-5 clip5',
                     'batch': '128 independent sources,2 of8views/source', 'selection': 'minimum simulationdevelopment continuousNLL including epoch0',
                     'temperature': [.5, .75, 1., 1.25, 1.5, 2.], 'temperature_choice': 'minimum development continuous NLL,not PE'},
        'data': '4096trainingparents96noiseblocks;512developmentparents32noiseblocks. Same source/noise-disjoint split,development reuse explicitly acknowledged.',
        'joint': 'Replace old mass marginal with NEW mass;retain normalized frozen conditional eta/chi given Mc. No adding separate old or new marginal mass scores.',
        'scoring': 'New COS-only calibration + gamma*newjoint finite-source lower-tail + beta*ONE newjoint empiricalLR;all prior mass scores and totalalpha removed.',
        'coefficient_grid': {'gamma': [0., .25, .5, 1., 2., 4., 8.], 'beta': [0., .0625, .125, .25, .5, 1., 2., 4.]},
        'coefficient_selection': 'simulation validation F50,F90,-AP,-R10,-R1;guard both waveform+fusion;fixed upstream weights;no PE/official selection',
        'frozen': ['time', 'sky', 'oldembedding', 'oldcheckpoints', 'scope', 'paper', 'historicaloutputs'],
        'limitation': 'Neural predictive density,not fullPE. Selected cases and reusedtests are adaptive development,not a new blind confirmation.',
        'refs': ['https://www.microsoft.com/en-us/research/publication/mixture-density-networks/',
                 'https://arxiv.org/abs/gr-qc/9402014'],
        'reference_limits': 'MDN probability representation and physical mass-phase motivation,not validation of chosen CNN or bounds.'}
    n.write_json(ROOT / 'contracts/ANALYSIS_CONTRACT.json', contract)
    shutil.copy2(__file__, ROOT / 'scripts/subgrid_density.py')
    n.write_json(ROOT / 'contracts/START_FREEZE.json', {'UTC': n.utc(), 'contract_sha256': n.sha(ROOT / 'contracts/ANALYSIS_CONTRACT.json'),
                 'script_sha256': n.sha(Path(__file__))})
    rows = pd.read_csv(P / 'results/mcwf_nodup_mass_reliability_02_20260909T154100Z/manifest/INPUT_SHA256.csv')
    n.write_csv(ROOT / 'manifest/INPUT_SHA256.csv', rows)
    rng = np.random.default_rng(2026090950)
    a = [rng.normal(size=(3, 253)), np.tile(CENTERS, (3, 1)), np.full((3, 253), DX)]
    truth = rng.uniform(np.log(6), np.log(100), 3)
    expected = numpy_nll(a, truth)
    actual = float(nll(*[torch.as_tensor(v, dtype=torch.float64) for v in a], torch.as_tensor(truth)))
    p, _ = probability(a)
    error = abs(expected - actual)
    if error > 2e-6:
        raise RuntimeError('Torch/NumPy mixture likelihood mismatch')
    n.write_json(ROOT / 'audit/DENSITY_ALGEBRA.json', {'pass': True, 'NLL_difference': error,
        'max_normalization_error': float(abs(p.sum(1) - 1).max()), 'C_source': 'independent NumPy float64 evaluation'})


def train_one(dep, slot, seed):
    out = ROOT / f'models/{dep}/seed_{slot}'
    if (out / 'COMPLETE.json').exists():
        return
    out.mkdir(parents=True, exist_ok=True)
    cp_path = LOW / f'models/MULTIRATE/{dep}/seed_{slot}/selected.pt'
    cp = torch.load(cp_path, map_location='cpu', weights_only=False)
    x, meta = mult.training_data(LOW, dep, 'train', 'MULTIRATE')
    v, vm = mult.training_data(LOW, dep, 'validation', 'MULTIRATE')
    if set(meta.source_uid) & set(vm.source_uid) or set(meta.noise_bank_index) & set(vm.noise_bank_index):
        raise RuntimeError('Training/source noise overlap')
    x, v = (x - cp['mu']) / cp['sd'], (v - cp['mu']) / cp['sd']
    truth, vt = np.log(meta.mc_det.to_numpy()), np.log(vm.mc_det.to_numpy())
    n.dev.TRAIN.seed_everything(seed)
    model = Predictor().cuda()
    missing, extra = model.load_state_dict(cp['model'], strict=False)
    if extra or set(missing) != {'mass_centers', 'density.weight', 'density.bias'}:
        raise RuntimeError('Warm state mapping error')
    xx, yy = torch.as_tensor(x, device='cuda'), torch.as_tensor(truth, dtype=torch.float32, device='cuda')
    del x
    groups, names = pd.factorize(meta.source_uid, sort=True)
    members = [np.flatnonzero(groups == k) for k in range(len(names))]
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS, eta_min=1e-5)
    best, history, start = float('inf'), [], time.perf_counter()
    first = 0
    resume = out / 'resume.pt'
    if resume.exists():
        state = torch.load(resume, map_location='cpu', weights_only=False)
        model.load_state_dict(state['model'])
        opt.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        best, history, first = state['best'], state['history'], state['epoch'] + 1
        torch.set_rng_state(state['rng'])
        torch.cuda.set_rng_state_all(state['cuda_rng'])
    for epoch in range(first, EPOCHS + 1):
        if epoch:
            model.train()
            rng = np.random.default_rng(seed + epoch)
            order = rng.permutation(len(names))
            for lo in range(0, len(order), 128):
                idx = np.concatenate([rng.choice(members[k], 2, replace=False) for k in order[lo:lo + 128]])
                opt.zero_grad(set_to_none=True)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    parameters = model(xx[idx])
                loss = nll(*parameters, yy[idx])
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite continuous density loss')
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5)
                opt.step()
            scheduler.step()
        arrays = infer_arrays(model, v)
        value = numpy_nll(arrays, vt)
        row = {'epoch': epoch, 'validation_NLL': value, 'seconds': time.perf_counter() - start}
        history.append(row)
        if value < best:
            best = value
            torch.save({'model': {k: z.detach().cpu().clone() for k, z in model.state_dict().items()},
                        'mu': cp['mu'], 'sd': cp['sd'], 'prior': cp['prior'], 'epoch': epoch, 'NLL': best,
                        'seed': seed, 'parent_sha256': n.sha(cp_path)}, out / 'selected.pt')
        torch.save({'model': model.state_dict(), 'optimizer': opt.state_dict(), 'scheduler': scheduler.state_dict(),
                    'best': best, 'history': history, 'epoch': epoch, 'rng': torch.get_rng_state(),
                    'cuda_rng': torch.cuda.get_rng_state_all()}, resume)
        n.write_csv(out / 'HISTORY.csv', history)
        print('SUBGRID', dep, slot, row, flush=True)
    cp = torch.load(out / 'selected.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(cp['model'])
    arrays = infer_arrays(model, v)
    grid = [{'temperature': temp, 'NLL': numpy_nll(arrays, vt, temp)} for temp in (.5, .75, 1., 1.25, 1.5, 2.)]
    cp['temperature'] = min(grid, key=lambda r: (r['NLL'], abs(r['temperature'] - 1)))['temperature']
    torch.save(cp, out / 'selected.pt')
    p, outside = probability(arrays, cp['temperature'])
    cdf = np.c_[np.zeros(len(p)), p.cumsum(1)]
    pit = np.array([np.interp(a, EDGES, b) for a, b in zip(vt, cdf)])
    np.savez_compressed(out / 'development_predictions.npz', p=p, outside=outside, truth=vt,
                        group=pd.factorize(vm.source_uid, sort=True)[0])
    n.write_csv(out / 'TEMPERATURE_GRID.csv', grid)
    n.write_json(out / 'COMPLETE.json', {'deployment': dep, 'slot': slot, 'epoch': cp['epoch'], 'NLL': cp['NLL'],
        'temperature': cp['temperature'], 'central90coverage': float(((pit >= .05) & (pit <= .95)).mean()),
        'central50coverage': float(((pit >= .25) & (pit <= .75)).mean()),
        'mean_absolute_logmass_error': float(np.mean(abs(p @ mult.old.CENTERS - vt))),
        'training_sources': len(names), 'development_sources': vm.source_uid.nunique(),
        'no_source_noise_overlap': True, 'selected_sha256': n.sha(out / 'selected.pt'),
        'seconds': time.perf_counter() - start})


def event_features(dep, seed, split, catalog):
    if split == 'development':
        return mult.training_data(LOW, dep, 'validation', 'MULTIRATE')[0]
    if catalog is None:
        file = 'real.npy' if split == 'real' else f'{seed}_{split}.npy'
        return np.concatenate([n.t.old_features(dep, seed, split), np.load(LOW / f'features/{dep}' / file)], 1)
    folder = n.FRESH / f'confirmation/{dep}/catalog_{catalog}'
    return np.concatenate([mult.old.arrange(np.load(folder / 'coarse_event_features.npy'),
                           np.load(folder / 'fine_event_features.npy')), np.load(folder / 'low_event_features.npy')], 1)


def predict(dep, seed, split, catalog=None):
    slot = n.recipes()[dep, seed]['slot']
    tag = split if catalog is None else f'{split}_{catalog}'
    path = ROOT / f'predictions/{dep}/{slot}_{tag}.npz'
    if path.exists():
        return dict(np.load(path))
    cp = torch.load(ROOT / f'models/{dep}/seed_{slot}/selected.pt', map_location='cpu', weights_only=False)
    x = event_features(dep, seed, split, catalog)
    model = Predictor().cuda().eval()
    model.load_state_dict(cp['model'])
    p, outside = probability(infer_arrays(model, (x - cp['mu']) / cp['sd']), cp['temperature'])
    if split == 'real':
        full, events = n.dev.real_inputs(dep)
        valid = events.strict_h1l1_preprocessing_pass.to_numpy(bool)
        pp, oo = np.full((len(full), 512), np.nan), np.full(len(full), np.nan)
        pp[valid], oo[valid] = p, outside
        p, outside = pp, oo
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, p=p, outside=outside)
    return {'p': p, 'outside': outside}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--stage', choices=('freeze', 'train'), required=True)
    a = parser.parse_args()
    ROOT = a.root
    if a.stage == 'freeze':
        freeze()
    else:
        for dep in n.DEPS:
            for slot, seed in zip(n.t.MODEL_SLOTS, TRAIN_SEEDS):
                train_one(dep, slot, seed)
