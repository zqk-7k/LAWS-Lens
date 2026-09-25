#!/usr/bin/env python3
"""Symmetric waveform-response verifier; no PE labels or legacy mass score."""
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
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.isotonic import IsotonicRegression

P = Path('/root/autodl-tmp/gw-catalog')
sys.path.insert(0, str(P / 'scripts/experiments'))
import mcwf_subgrid_density_20260909 as d
r, n, mult = d.reliability, d.n, d.mult
ROOT = None
SEEDS = (2026090971, 2026090972, 2026090973)
METHODS = ('NODUP-DIRECT-REPLAY', 'PAIR-RESPONSE-ONLY', 'PAIR-RESPONSE-VAL')


class Verifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('coordinate', torch.linspace(-1, 1, 253)[None, None])
        self.first = nn.Sequential(nn.Conv1d(55, 48, 7, padding=3), nn.SiLU())
        self.middle = nn.Sequential(nn.Conv1d(48, 48, 7, padding=3), nn.SiLU(),
                                    nn.Conv1d(48, 32, 7, padding=6, dilation=2), nn.SiLU())
        self.compare = nn.Sequential(nn.Conv1d(96, 48, 7, padding=3), nn.SiLU(),
            nn.Conv1d(48, 24, 7, padding=6, dilation=2), nn.SiLU())
        self.final = nn.Sequential(nn.Linear(48, 32), nn.SiLU(), nn.Linear(32, 1))

    def encode(self, x):
        return self.middle(self.first(torch.cat([x, self.coordinate.expand(len(x), -1, -1)], 1)))

    def compare_encoded(self, a, b):
        h = self.compare(torch.cat([.5 * (a + b), abs(a - b), a * b], 1))
        return self.final(torch.cat([h.mean(-1), h.amax(-1)], -1)).squeeze(-1)

    def forward(self, a, b):
        return self.compare_encoded(self.encode(a), self.encode(b))


def members(meta):
    group, names = pd.factorize(meta.source_uid, sort=True)
    a, b, mass = [], [], []
    for k in range(len(names)):
        ids = np.flatnonzero(group == k)
        ai = ids[meta.image.to_numpy()[ids] == 'a']
        bi = ids[meta.image.to_numpy()[ids] == 'b']
        if not len(ai) or not len(bi):
            raise RuntimeError('Missing independent image view')
        a.append(ai)
        b.append(bi)
        mass.append(float(np.log(meta.iloc[ids[0]].mc_det)))
    return a, b, np.array(mass), group


def sample_pairs(meta, seed, count, hard=True):
    rng = np.random.default_rng(seed)
    aa, bb, mc, _ = members(meta)
    groups = len(aa)
    left = rng.integers(groups, size=count)
    right = rng.integers(groups - 1, size=count)
    right += right >= left
    positive = np.arange(count) < count // 2
    right[positive] = left[positive]
    if hard:
        for row in range(count // 2, count // 2 + count // 8):
            delta = abs(mc - mc[left[row]])
            candidates = np.flatnonzero((delta >= .02) & (delta <= .35))
            if len(candidates):
                right[row] = rng.choice(candidates)
    i = np.array([rng.choice(aa[k]) for k in left])
    j = np.array([rng.choice(bb[k]) for k in right])
    permutation = rng.permutation(count)
    return i[permutation], j[permutation], positive[permutation].astype(np.float32)


@torch.no_grad()
def encoded(model, values):
    model.eval()
    out = []
    for lo in range(0, len(values), 128):
        x = torch.as_tensor(values[lo:lo + 128], dtype=torch.float32, device='cuda')
        with torch.autocast('cuda', dtype=torch.bfloat16):
            h = model.encode(x)
        out.append(h.float())
    return torch.cat(out)


@torch.no_grad()
def logits(model, h, i, j):
    out = []
    for lo in range(0, len(i), 512):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            a = model.compare_encoded(h[i[lo:lo + 512]], h[j[lo:lo + 512]])
        out.append(a.float().cpu().numpy())
    return np.concatenate(out).astype(float)


def freeze():
    if ROOT.exists():
        raise RuntimeError('Independent directory required')
    for folder in ('contracts', 'models', 'predictions', 'calibration', 'configs', 'tables', 'audit',
                   'reports', 'scripts', 'logs', 'manifest', 'results', 'figures', 'cache'):
        (ROOT / folder).mkdir(parents=True)
    n.write_json(ROOT / 'contracts/ANALYSIS_CONTRACT.json', {
        'id': 'MCWF-NODUP-PAIR-RESPONSE-05', 'UTC': n.utc(), 'status': n.STATUS,
        'same_both_runs': True, 'goal_achieved': False,
        'motivation': 'Compare paired waveform response patterns directly;independent marginal posterior overlap may discard discriminating information.',
        'inputs': '54x253 frozen waveform-template powers:peak2s40-580Hz andlow16s20-80Hz;no timing,sky,PE,legacyMc/q,official labels or event IDs as features',
        'architecture': 'Shared48/32-channel CNN then symmetricmean/absolute difference/product,48/24pairCNN,mean/max pooling,32unitMLP',
        'training': {'seeds': list(SEEDS), 'epochs': 30, 'pairs_per_epoch': 16384, 'batch': 256,
                     'positive': 'same independent simulatedsource,oppositeimage and independentnoise realization',
                     'negative': '75%random different source,25%near logMc .02-.35;onlysimulatedtruth',
                     'optimizer': 'AdamW lr2e-4 WD1e-4 cosine2e-5 clip5',
                     'source_population': '4096trainparents;developmentmodelselectionusesonlysource/noisehashfold1',
                     'selection': 'lowestbalancedrandom-null validationBCE;epoch0included',
                     'densityfit': 'hashfold0independentfromnetworkvalidation;sourcepairweighted isotonic LR'} ,
        'score': 'One calibrated verifier waveformscore replaces theentire waveformchannel;no duplicatecosine/jointscore,no total-alpha blend',
        'OOD': 'fitlogitrange;positiveOOD0,negativeboundedretained;caplog(1+independentfittrueparents)',
        'outerweights': 'unchangedpure upstream perseedweights',
        'frozen': ['time_score', 'sky_raw_log_bf', 'oldencoder', 'scope', 'historicalresults', 'paper'],
        'adaptive_development': True, 'no_blind_confirmation_claim': True,
        'no_real_or_official_selection': True,
        'refs': ['https://arxiv.org/abs/1506.02169', 'https://arxiv.org/abs/gr-qc/9402014'],
        'ref_limits': 'Calibrated discriminative ratio andwaveformmassinformation motivation;not proof thisnetwork approximatesfullphysical likelihood or PE.'})
    shutil.copy2(__file__, ROOT / 'scripts/pair_response_verifier.py')
    n.write_json(ROOT / 'contracts/START_FREEZE.json', {'UTC': n.utc(),
        'contract_sha256': n.sha(ROOT / 'contracts/ANALYSIS_CONTRACT.json'), 'script_sha256': n.sha(Path(__file__))})
    base = P / 'results/mcwf_nodup_mass_reliability_02_20260909T154100Z/manifest/INPUT_SHA256.csv'
    n.write_csv(ROOT / 'manifest/INPUT_SHA256.csv', pd.read_csv(base))


def train(dep, slot, training_seed):
    out = ROOT / f'models/{dep}/seed_{slot}'
    if (out / 'COMPLETE.json').exists():
        return
    out.mkdir(parents=True, exist_ok=True)
    x, meta = mult.training_data(d.LOW, dep, 'train', 'MULTIRATE')
    v, vm = mult.training_data(d.LOW, dep, 'validation', 'MULTIRATE')
    vm['deployment'] = dep
    fold = r.source_fold(vm)
    keep = fold == 1
    v, vm = v[keep], vm[keep].reset_index(drop=True)
    if set(meta.source_uid) & set(vm.source_uid) or set(meta.noise_bank_index) & set(vm.noise_bank_index):
        raise RuntimeError('Source/noise isolation violation')
    mu, sd = x.mean((0, 2), keepdims=True), x.std((0, 2), keepdims=True).clip(.01)
    x, v = (x - mu) / sd, (v - mu) / sd
    vi, vj, vy = sample_pairs(vm, 2026090980, 8192, hard=False)
    n.dev.TRAIN.seed_everything(training_seed)
    model = Verifier().cuda()
    xx = torch.as_tensor(x, device='cuda')
    del x
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 30, eta_min=2e-5)
    history, best, start = [], float('inf'), time.perf_counter()
    for epoch in range(31):
        if epoch:
            ii, jj, yy = sample_pairs(meta, training_seed + epoch, 16384)
            model.train()
            for lo in range(0, len(ii), 256):
                ids = slice(lo, lo + 256)
                opt.zero_grad(set_to_none=True)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    pred = model(xx[ii[ids]], xx[jj[ids]])
                loss = F.binary_cross_entropy_with_logits(pred.float(), torch.as_tensor(yy[ids], device='cuda'))
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite training objective')
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5)
                opt.step()
            scheduler.step()
        hh = encoded(model, v)
        val = logits(model, hh, vi, vj)
        value = float(np.mean(np.logaddexp(0., val) - vy * val))
        row = {'epoch': epoch, 'validation_BCE': value, 'seconds': time.perf_counter() - start}
        history.append(row)
        if value < best:
            best = value
            torch.save({'model': {k: a.detach().cpu().clone() for k, a in model.state_dict().items()},
                        'mu': mu, 'sd': sd, 'epoch': epoch, 'seed': training_seed, 'validation_BCE': value}, out / 'selected.pt')
        n.write_csv(out / 'HISTORY.csv', history)
        if epoch % 5 == 0:
            print('PAIR_RESPONSE', dep, slot, row, flush=True)
    ck = torch.load(out / 'selected.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(ck['model'])
    h = encoded(model, v[:16])
    i, j = np.triu_indices(16, 1)
    error = float(abs(logits(model, h, i, j) - logits(model, h, j, i)).max())
    if error != 0:
        raise RuntimeError('Pair exchange changed score')
    n.write_json(out / 'COMPLETE.json', {'deployment': dep, 'slot': slot, 'epoch': ck['epoch'],
        'validation_BCE': best, 'seconds': time.perf_counter() - start, 'train_sources': meta.source_uid.nunique(),
        'validation_sources': vm.source_uid.nunique(), 'swap_error': error, 'checkpoint_sha256': n.sha(out / 'selected.pt')})


def response_values(dep, seed, split, i, j, catalog=None):
    slot = n.recipes()[dep, seed]['slot']
    ck = torch.load(ROOT / f'models/{dep}/seed_{slot}/selected.pt', map_location='cpu', weights_only=False)
    x = d.event_features(dep, seed, split, catalog)
    if split == 'real':
        _, events = n.dev.real_inputs(dep)
        valid = events.strict_h1l1_preprocessing_pass.to_numpy(bool)
        inverse = np.full(len(valid), -1, int)
        inverse[valid] = np.arange(valid.sum())
        i, j = inverse[i], inverse[j]
        if (i < 0).any() or (j < 0).any():
            raise RuntimeError('Invalid detector event entered strict scope')
    model = Verifier().cuda().eval()
    model.load_state_dict(ck['model'])
    h = encoded(model, (x - ck['mu']) / ck['sd'])
    return logits(model, h, i, j)


def load_panel(dep, seed, split, catalog=None):
    # Retain existing predictive-mass diagnostics for exports, not score inputs.
    f = r.load_panel(dep, seed, split, catalog)
    tag = split if catalog is None else f'{split}_{catalog}'
    path = ROOT / f'predictions/{dep}/{seed}_{tag}.npz'
    if path.exists():
        values = np.load(path)['values']
    else:
        values = response_values(dep, seed, split, f.idx_i.to_numpy(int), f.idx_j.to_numpy(int), catalog)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, values=values)
    if len(values) != len(f):
        raise RuntimeError('Pair count changed')
    f['pair_response_logit'] = values
    return f


def infer(frame, c):
    if c['method'] == METHODS[0]:
        return r.BASE_INFER(frame, {**c, 'method': 'NODUP-DIRECT'})
    spec = c['pair_calibration']
    value = frame.pair_response_logit.to_numpy(float)
    result = np.interp(value, spec['knots'], spec['loglr'])
    clipped = abs(result) > spec['cap']
    result = result.clip(-spec['cap'], spec['cap'])
    ood = (value < spec['minimum']) | (value > spec['maximum'])
    result[ood] = np.minimum(result[ood], 0.)
    return c.get('response_scale', 1.) * result, ood, clipped


def freeze_score():
    path = ROOT / 'contracts/SCORING_ADDENDUM.json'
    if path.exists():
        raise RuntimeError('Scoring addendum already frozen')
    n.write_json(path, {'UTC': n.utc(), 'methods': list(METHODS),
        'fixed': 'One calibrated waveform verifier, original outerweights unchanged',
        'sensitivity': 'Scale onlynewwaveformLR .25,.5,1,2,4,selectedperseedon simulationvalidation only',
        'selection': 'Existing waveform/fusionguard thenF50,F90,-AP,-R10,-R1,tieclosestscale1 thenlexicographic',
        'not_total_score_blend': True, 'no_old_waveform_or_massscore_added': True,
        'no_new_test_or_real_result_seen': True,
        'export_fix': 'Retain frozenmass diagnosticsforaudit only;notinputs toverifier'})
    shutil.copy2(__file__, ROOT / 'scripts/pair_response_verifier_evaluate.py')
    n.write_json(ROOT / 'contracts/SCORING_ADDENDUM_FREEZE.json', {'UTC': n.utc(),
        'contract_sha256': n.sha(path), 'script_sha256': n.sha(Path(__file__))})


def calibrate():
    base = json.loads((r.PRIOR / 'configs/SELECTED_CONFIGURATIONS.json').read_text())
    original = {(c['deployment'], c['seed']): c for c in base if c['method'] == 'NODUP-DIRECT'}
    if not (ROOT / 'contracts/SCORING_ADDENDUM_FREEZE.json').exists():
        raise RuntimeError('Missing scoring preregistration')
    configs, grid = [], []
    for dep in n.DEPS:
        meta = pd.read_parquet(n.t.PREVIOUS / f'expanded_data/{dep}/validation/event_metadata.parquet')
        meta['deployment'] = dep
        fold = r.source_fold(meta)
        ids = np.flatnonzero(fold == 0)
        a, b = np.triu_indices(len(ids), 1)
        i, j = ids[a], ids[b]
        group = meta.source_uid.to_numpy(str)
        y = group[i] == group[j]
        keys = pd.Series([':'.join(sorted((a, b))) for a, b in zip(group[i], group[j])])
        w = keys.map(1 / keys.value_counts()).to_numpy()
        w[y] *= .5 / w[y].sum()
        w[~y] *= .5 / w[~y].sum()
        for seed in n.SEEDS:
            value = response_values(dep, seed, 'development', i, j)
            iso = IsotonicRegression(increasing=True, out_of_bounds='clip').fit(value, y, sample_weight=w)
            pp = iso.y_thresholds_.clip(1 / (y.sum() + 2), 1 - 1 / (y.sum() + 2))
            spec = {'knots': iso.X_thresholds_.tolist(), 'loglr': (np.log(pp) - np.log1p(-pp)).tolist(),
                    'minimum': float(value.min()), 'maximum': float(value.max()), 'cap': float(np.log(y.sum() + 1)),
                    'independent_fit_sources': int(y.sum()), 'source_pair_weighted': True,
                    'network_validation_disjoint': True}
            c = original[dep, seed]
            common = {**c, 'pair_calibration': spec}
            configs += [{**c, 'method': METHODS[0]}, {**common, 'method': METHODS[1], 'response_scale': 1.}]
            f = load_panel(dep, seed, 'validation')
            baseline = n.cf.fast_metrics(f, f.PATH875_final_score.to_numpy())
            basewave = n.cf.fast_metrics(f, f.PATH875_waveform.to_numpy())
            options = []
            for scale in (.25, .5, 1., 2., 4.):
                z, oo, cl = infer(f, {**common, 'method': METHODS[2], 'response_scale': scale})
                m = n.cf.fast_metrics(f, n.cf.channels(f, z) @ np.asarray(c['weights']))
                wm = n.cf.fast_metrics(f, z)
                row = {'scale': scale, 'guard': n.cf.guard(m, baseline) and n.cf.guard(wm, basewave), **m}
                options.append(row)
                grid.append({'deployment': dep, 'seed': seed, **row})
            eligible = [a for a in options if a['guard']]
            best = min(eligible or options, key=lambda a: (a['false_at_recall_0p5'], a['false_at_recall_0p9'],
                -a['average_precision'], -a['macro_r_at_10'], -a['macro_r_at_1'], abs(np.log(a['scale'])), a['scale']))
            configs.append({**common, 'method': METHODS[2], 'response_scale': best['scale'],
                            'tune_guard': bool(eligible), 'tune_metrics': best})
            n.write_json(ROOT / f'calibration/{dep}_{seed}.json', spec)
            print('PAIR_CALIBRATION', dep, seed, y.sum(), flush=True)
    n.write_json(ROOT / 'configs/SELECTED_CONFIGURATIONS.json', configs)
    n.write_csv(ROOT / 'tables/PAIR_VALIDATION_SCALE.csv', grid)
    n.write_json(ROOT / 'contracts/CONFIGURATIONS_FROZEN.json', {'UTC': n.utc(),
        'file': 'configs/SELECTED_CONFIGURATIONS.json',
        'sha256': n.sha(ROOT / 'configs/SELECTED_CONFIGURATIONS.json'), 'no_test_or_real_selection': True,
        'no_total_alpha': True, 'new_waveform_is_single_calibrated_pair_classifier': True})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--stage', choices=('freeze', 'freeze_score', 'train', 'calibrate', 'evaluate', 'real'), required=True)
    args = parser.parse_args()
    ROOT = r.ROOT = args.root
    r.install()
    n.METHODS, n.load_panel, n.infer = METHODS, load_panel, infer
    if args.stage in ('freeze', 'freeze_score'):
        globals()[args.stage]()
    elif args.stage == 'train':
        for dep in n.DEPS:
            for slot, seed in zip(n.t.MODEL_SLOTS, SEEDS):
                train(dep, slot, seed)
    elif args.stage == 'calibrate':
        calibrate()
    else:
        n.run(ROOT, args.stage)
