#!/usr/bin/env python3
"""Conditional spin/eta inference with an immutable chirp-mass marginal."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
import argparse
from datetime import datetime, timezone
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

P = Path('/root/autodl-tmp/gw-catalog')
sys.path.insert(0, str(P / 'scripts/experiments'))
import mcwf_conditional_eta_20260908 as original
import mcwf_temporal_response_evaluate_20260908 as ev
mult, t, old, dev, cf = original.mult, original.t, original.old, original.dev, original.cf
UPSTREAM = original.UPSTREAM
KINDS = ('CONDITIONAL-CHI', 'CONDITIONAL-ETA-CHI')
CHI = np.linspace(-1., 1., 17)
ETA = original.ETA
KIND = KINDS[0]


def physical_truth(meta):
    m1, m2 = meta.m1_det.to_numpy(float), meta.m2_det.to_numpy(float)
    q = m2 / m1
    chi = (m1 * meta.a1.to_numpy() * np.cos(meta.tilt1.to_numpy()) +
           m2 * meta.a2.to_numpy() * np.cos(meta.tilt2.to_numpy())) / (m1 + m2)
    if (m2 > m1).any() or (abs(chi) > 1 + 1e-12).any():
        raise RuntimeError('Unphysical source labels')
    return q / (1 + q)**2, chi


def labels(meta, kind):
    eta, chi = physical_truth(meta)
    yc = original.targets(chi, CHI)
    if kind == KINDS[0]:
        return yc
    return (original.targets(eta, ETA)[:, :, None] * yc[:, None, :]).reshape(len(meta), -1)


def conditional_prior(meta, kind):
    sources = meta.drop_duplicates('source_uid')
    ym = original.targets(np.log(sources.mc_det.to_numpy()), old.LOG_CENTERS)
    counts = ym.T @ labels(sources, kind)
    if kind == KINDS[0]:
        counts = gaussian_filter(counts, (2., 1.), mode='nearest')
    else:
        counts = gaussian_filter(counts.reshape(253, 16, 17), (2., 1., 1.), mode='nearest').reshape(253, -1)
    counts = counts + 1e-6
    return counts / counts.sum(1, keepdims=True)


class Head(nn.Module):
    def __init__(self, output):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(103, 96), nn.SiLU(), nn.Dropout(.1),
                                 nn.Linear(96, 96), nn.SiLU(), nn.Linear(96, output))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x)


def initialize(root):
    if root.exists():
        raise RuntimeError('New independent directory required')
    for folder in ('contracts', 'scripts', 'logs', 'models', 'predictions', 'calibration',
                   'evaluation', 'tables', 'reports', 'manifest', 'figures', 'cache'):
        (root / folder).mkdir(parents=True)
    contract = {
        'id': 'MCWF-CONDITIONAL-INTRINSICS-10', 'UTC': datetime.now(timezone.utc).isoformat(),
        'status': t.STATUS, 'goal_achieved': False, 'same_both_runs': True,
        'upstream': str(UPSTREAM), 'controls': list(KINDS),
        'physical_motivation': 'Mass ratio and aligned effective spin contribute to waveform phasing and are correlated with Mc; test their conditional predictive information without modifying the Mc marginal.',
        'truth': 'eta=q/(1+q)^2;chi_eff=(m1*a1*cos(tilt1)+m2*a2*cos(tilt2))/(m1+m2),from simulated source metadata only. chi_eff is approximately conserved,not an exact precession invariant.',
        'bins': {'eta': ETA.tolist(), 'chi_eff': CHI.tolist()},
        'architecture': 'Frozen MULTIRATE mass network;103-to96-to96-to17 or272 residual conditional classifier. Last layer starts at zero (conditional population prior).',
        'training': '4096 source parents,96noiseblocks,8views/source;512developmentparents/32noiseblocks;25epochs,AdamW1e-3,WD1e-4,clip5,cosine1e-5,2views/source/batch,minimumdevelopmentCE includingepoch0.',
        'seeds': [202609901, 202609902, 202609903],
        'prior': 'One weight per independent source. Smoothed counts: mass sigma2,other axes sigma1;floor1e-6. Empirical simulation prior,not public PE or an astrophysical population measurement.',
        'temperature_grid': [.5, .75, 1., 1.25, 1.5, 2.],
        'joint': 'p(M,theta|x)=frozen_p(M|x)*p(theta|M,x);must exactly marginalize to frozen Mc. Joint eta+chi is a single correlated term,not the sum of eta-only and chi-only evidence.',
        'score_feature': 'log(overlap_joint/prior_joint)-log(overlap_M/prior_M);uninformative conditional head contributes zero.',
        'calibration': 'Simulation-only noise/source-disjoint fit/audit halves;class-balanced isotonic LR,subtract LR at feature0,clip[-4,4],positiveOOD=>0,mass finite-source tail<.05 prohibits positive support.',
        'integration': 'Retained OMC waveform + beta*conditional increment, beta0,.0625,.125,.25,.5,1,2,4;candidate/retrieval priorities and guards unchanged;zero always included.',
        'limitations': 'Predictive-overlap proxy,not physical PE or a proper Bayes factor;development reused for checkpoint selection and calibration audit. Finite simulated population and approximate templates limit interpretation.',
        'frozen': ['Mc checkpoints', 'OMC baseline', 'one-dimensional time', 'sky_raw_log_bf', 'outer C-fixed weights', 'scope', 'history', 'paper'],
        'external': 'Adaptive real-catalog development;real PE and official labels are appended only after selection;not ranking inputs. No claim of independent blind validation.',
        'references': ['https://arxiv.org/abs/gr-qc/9402014', 'https://arxiv.org/abs/0909.2867'],
        'reference_scope': 'Physical phasing/spin motivation only,not validation of this neural approximation.'}
    dev.json_write(root / 'contracts/ANALYSIS_CONTRACT.json', contract)
    paths = list((UPSTREAM / 'models/MULTIRATE').glob('*/*/selected.pt'))
    protected = t.protected() + [{'path': str(p), 'sha256': dev.sha(p), 'bytes': p.stat().st_size} for p in paths]
    dev.csv_write(root / 'manifest/INPUT_SHA256.csv', pd.DataFrame(protected))
    shutil.copy2(__file__, root / 'scripts/conditional_intrinsics.py')
    tests = []
    rng = np.random.default_rng(202609900)
    for dim in (17, 272):
        pm = rng.dirichlet(np.ones(512), size=4)
        prior = rng.dirichlet(np.ones(512))
        cond = rng.dirichlet(np.ones(dim), size=512)
        joint = pm[:, :, None] * cond
        i, j = np.triu_indices(4, 1)
        a = (joint[i] * joint[j] / (prior[:, None] * cond)).sum((1, 2))
        b = (pm[i] * pm[j] / prior).sum(1)
        error = float(abs(np.log(a) - np.log(b)).max())
        if error > 1e-12 or abs(joint.sum(-1) - pm).max() > 1e-12:
            raise RuntimeError('Conditional evidence algebra failure')
        tests.append({'conditional_dimensions': dim, 'uninformative_increment_max_abs': error, 'pass': True})
    dev.json_write(root / 'contracts/ALGEBRA_TESTS.json', tests)
    dev.json_write(root / 'contracts/FREEZE.json', {'contract_sha256': dev.sha(root / 'contracts/ANALYSIS_CONTRACT.json'), 'code_sha256': dev.sha(Path(__file__))})


def train_one(root, dep, slot, seed, kind):
    folder = root / f'models/{kind}/{dep}/seed_{slot}'
    if (folder / 'COMPLETE.json').exists():
        return
    folder.mkdir(parents=True, exist_ok=True)
    cp_path = UPSTREAM / f'models/MULTIRATE/{dep}/seed_{slot}/selected.pt'
    before = dev.sha(cp_path)
    cp = torch.load(cp_path, map_location='cpu', weights_only=False)
    train, tm = mult.training_data(UPSTREAM, dep, 'train', 'MULTIRATE')
    val, vm = mult.training_data(UPSTREAM, dep, 'validation', 'MULTIRATE')
    if set(tm.source_uid) & set(vm.source_uid) or set(tm.noise_bank_index) & set(vm.noise_bank_index):
        raise RuntimeError('Source or noise overlap')
    trlog, vlog = np.log(tm.mc_det.to_numpy(float)), np.log(vm.mc_det.to_numpy(float))
    x = original.representations(train, cp, trlog)
    v = original.representations(val, cp, vlog)
    del train, val
    mu, sd = x.mean(0, keepdims=True), x.std(0, keepdims=True).clip(.01)
    x, v = (x - mu) / sd, (v - mu) / sd
    prior = conditional_prior(tm, kind)
    tx = np.log(original.interpolate_rows(prior, trlog).clip(1e-12))
    vx = np.log(original.interpolate_rows(prior, vlog).clip(1e-12))
    y, vy = labels(tm, kind), labels(vm, kind)
    group, names = pd.factorize(tm.source_uid, sort=True)
    members = [np.flatnonzero(group == k) for k in range(len(names))]
    dev.TRAIN.seed_everything(seed)
    head = Head(y.shape[1]).cuda()
    xx, yy, tt = [torch.as_tensor(a, device='cuda') for a in (x, y, tx)]
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 25, eta_min=1e-5)
    history, best, started = [], float('inf'), time.perf_counter()
    for epoch in range(26):
        if epoch:
            head.train()
            rng = np.random.default_rng(seed + epoch)
            order = rng.permutation(len(names))
            for start in range(0, len(order), 128):
                ids = np.concatenate([rng.choice(members[k], 2, replace=False) for k in order[start:start+128]])
                optimizer.zero_grad(set_to_none=True)
                loss = -(F.log_softmax(head(xx[ids]) + tt[ids], -1) * yy[ids]).sum(-1).mean()
                loss.backward()
                nn.utils.clip_grad_norm_(head.parameters(), 5)
                optimizer.step()
            scheduler.step()
        residual = original.infer(head, v)
        prob = softmax(residual + vx, -1)
        ce = float(-(vy * np.log(prob.clip(1e-30))).sum(-1).mean())
        row = {'epoch': epoch, 'conditional_CE': ce, 'seconds': time.perf_counter() - started}
        history.append(row)
        if ce < best:
            best = ce
            torch.save({'model': {k: z.detach().cpu().clone() for k, z in head.state_dict().items()},
                        'mu': mu, 'sd': sd, 'conditional_prior': prior, 'conditional_dimensions': y.shape[1],
                        'prior': cp['prior'], 'epoch': epoch, 'CE': ce, 'seed': seed, 'kind': kind,
                        'mass_checkpoint': str(cp_path), 'mass_checkpoint_sha256': before}, folder / 'selected.pt')
        dev.csv_write(folder / 'history.csv', pd.DataFrame(history))
        print({'training': [kind, dep, slot], **row}, flush=True)
    ck = torch.load(folder / 'selected.pt', map_location='cpu', weights_only=False)
    head.load_state_dict(ck['model'])
    residual = original.infer(head, v)
    grid = [{'temperature': temp, 'CE': float(-(vy * np.log(softmax(residual / temp + vx, -1).clip(1e-30))).sum(-1).mean())}
            for temp in (.5, .75, 1., 1.25, 1.5, 2.)]
    ck['temperature'] = min(grid, key=lambda r: (r['CE'], abs(r['temperature'] - 1)))['temperature']
    torch.save(ck, folder / 'selected.pt')
    dev.csv_write(folder / 'temperature_grid.csv', pd.DataFrame(grid))
    if dev.sha(cp_path) != before:
        raise RuntimeError('Frozen mass model changed')
    dev.json_write(folder / 'COMPLETE.json', {'kind': kind, 'epoch': ck['epoch'], 'CE': ck['CE'],
                   'prior_CE': history[0]['conditional_CE'], 'temperature': ck['temperature'],
                   'source_noise_overlap': 0, 'training_sources': len(names), 'development_sources': vm.source_uid.nunique(),
                   'backbone_unchanged': True, 'seconds': time.perf_counter() - started, 'sha256': dev.sha(folder / 'selected.pt')})


def prediction(root, dep, slot, es, split):
    path = root / f'predictions/{KIND}/{dep}/{slot}_{es}_{split}.npz'
    if path.exists():
        return np.load(path)
    folder = root / f'models/{KIND}/{dep}/seed_{slot}'
    ck = torch.load(folder / 'selected.pt', map_location='cpu', weights_only=False)
    cp = torch.load(ck['mass_checkpoint'], map_location='cpu', weights_only=False)
    if dev.sha(Path(ck['mass_checkpoint'])) != ck['mass_checkpoint_sha256']:
        raise RuntimeError('Frozen mass checkpoint hash mismatch')
    if split == 'development':
        x, meta = mult.training_data(UPSTREAM, dep, 'validation', 'MULTIRATE')
        mass = dict(np.load(Path(ck['mass_checkpoint']).parent / 'development_predictions.npz'))
    else:
        filename = 'real.npy' if split == 'real' else f'{es}_{split}.npy'
        x = np.concatenate([t.old_features(dep, es, split), np.load(UPSTREAM / f'features/{dep}' / filename)], 1)
        mass = dict(np.load(UPSTREAM / f'predictions/MULTIRATE/{dep}/model_{slot}_eval_{es}/{split}.npz'))
    rep = original.representations(x, cp)
    head = Head(ck['conditional_dimensions']).cuda().eval()
    head.load_state_dict(ck['model'])
    residual = original.infer(head, ((rep - ck['mu']) / ck['sd']).reshape(-1, 103)).reshape(len(rep), 253, -1)
    conditional = softmax(residual / ck['temperature'] + np.log(ck['conditional_prior'].clip(1e-12))[None], -1)
    hi = np.searchsorted(old.LOG_CENTERS, old.CENTERS, side='right').clip(1, 252)
    lo = hi - 1
    w = ((old.CENTERS - old.LOG_CENTERS[lo]) / (old.LOG_CENTERS[hi] - old.LOG_CENTERS[lo])).clip(0, 1)
    conditional = (1 - w[None, :, None]) * conditional[:, lo] + w[None, :, None] * conditional[:, hi]
    if split == 'real':
        full, events = dev.real_inputs(dep)
        valid = events.strict_h1l1_preprocessing_pass.to_numpy(bool)
        out = np.full((len(full), 512, ck['conditional_dimensions']), np.nan, np.float32)
        out[valid] = conditional
        conditional = out
    joint = mass['p'][:, :, None] * conditional
    finite = np.isfinite(mass['p']).all(1)
    error = float(abs(joint[finite].sum(-1) - mass['p'][finite]).max())
    if error > 1e-6:
        raise RuntimeError('Mass marginal changed')
    extra = {k: mass[k] for k in ('group', 'truth') if k in mass}
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, joint=joint.astype(np.float32), p=mass['p'], outside=mass['outside'], **extra)
    dev.json_write(path.with_suffix('.json'), {'mass_marginal_max_abs': error, 'kind': KIND,
                   'upstream_checkpoint': ck['mass_checkpoint_sha256'], 'conditional_dimensions': ck['conditional_dimensions']})
    return np.load(path)


def pair_features(root, dep, slot, a, i, j):
    ck = torch.load(root / f'models/{KIND}/{dep}/seed_{slot}/selected.pt', map_location='cpu', weights_only=False)
    prior = ck['prior']
    cond = original.interpolate_rows(ck['conditional_prior'], old.CENTERS)
    joint_prior = prior[:, None] * cond
    if not np.allclose(joint_prior.sum(1), prior, atol=1e-10):
        raise RuntimeError('Joint prior normalization')
    def gram(x, weight=None):
        z = torch.as_tensor(np.asarray(x, dtype=np.float64), device='cuda').flatten(1)
        if weight is not None:
            z = z / torch.as_tensor(np.sqrt(weight).reshape(1, -1), device='cuda')
        return (z @ z.T).cpu().numpy()[i, j]
    massbf = gram(a['p'], prior).clip(1e-300)
    jointbf = gram(a['joint'], joint_prior).clip(1e-300)
    # API field is shared with the eta-only calibration; metadata names actual theta.
    return {'conditional_eta_overlap': np.log(jointbf) - np.log(massbf),
            'joint_BC': gram(np.sqrt(a['joint'].astype(float))),
            'mass_BC': gram(np.sqrt(a['p'])).clip(1e-15, 1),
            'ood': (a['outside'][i] > .25) | (a['outside'][j] > .25)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--stage', choices=['initialize', 'train', 'development'], required=True)
    args = parser.parse_args()
    if args.stage == 'initialize':
        initialize(args.root)
    else:
        for KIND in KINDS:
            for dep in t.DEPS:
                for slot, seed in zip(t.MODEL_SLOTS, (202609901, 202609902, 202609903)):
                    if args.stage == 'train':
                        train_one(args.root, dep, slot, seed, KIND)
                    else:
                        prediction(args.root, dep, slot, 0, 'development')
