#!/usr/bin/env python3
"""Use exactly one continuous-mass joint density in NODUP waveform scoring."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
import torch

P = Path('/root/autodl-tmp/gw-catalog')
sys.path.insert(0, str(P / 'scripts/experiments'))
import mcwf_subgrid_density_20260909 as density
r = density.reliability
n = r.n
METHODS = ('NODUP-DIRECT-REPLAY', 'SUBGRID-JOINT-FIXED', 'SUBGRID-JOINT-VAL')
ROOT = None
CACHE = {}
PREDICTOR = None


def matrices(dep, seed, split, catalog=None):
    key = (dep, seed, split, catalog)
    if key in CACHE:
        return CACHE[key]
    slot = n.recipes()[dep, seed]['slot']
    tag = split if catalog is None else f'{split}_{catalog}'
    dest = ROOT / f'cache/{dep}/{seed}_{tag}.npz'
    if dest.exists():
        CACHE[key] = dict(np.load(dest))
        return CACHE[key]
    mass = density.predict(dep, seed, split, catalog)
    if split == 'development':
        path = r.INTR / f'predictions/CONDITIONAL-ETA-CHI/{dep}/{slot}_0_development.npz'
    else:
        path = r.prediction_path(dep, seed, split, catalog)
    old = np.load(path)
    joint = old['joint'].astype(float)
    prior_cp = torch.load(r.INTR / f'models/CONDITIONAL-ETA-CHI/{dep}/seed_{slot}/selected.pt',
                          map_location='cpu', weights_only=False)
    cond_prior = r.physical.original.interpolate_rows(prior_cp['conditional_prior'], r.physical.old.CENTERS)
    valid = np.isfinite(mass['p']).all(1)
    conditional = joint[valid]
    normalization = conditional.sum(-1, keepdims=True)
    fallback = normalization[..., 0] <= 1e-250
    conditional = conditional / normalization.clip(1e-250)
    conditional[fallback] = np.broadcast_to(cond_prior, conditional.shape)[fallback]
    conditional /= conditional.sum(-1, keepdims=True)
    newjoint = mass['p'][valid, :, None] * conditional
    if abs(newjoint.sum(-1) - mass['p'][valid]).max() > 1e-10:
        raise RuntimeError('New mass marginal not conserved')
    z = torch.as_tensor(np.sqrt(newjoint.reshape(len(newjoint), -1)), dtype=torch.float64, device='cuda')
    active_bc = (z @ z.T).cpu().numpy().clip(1e-300, 1.)
    bc = np.full((len(valid), len(valid)), np.nan)
    ids = np.flatnonzero(valid)
    bc[np.ix_(ids, ids)] = active_bc
    result = {'bc': bc, 'logbc': np.log(bc), 'outside': mass['outside'],
              'summary': r.mass_summary(mass['p']),
              'conditional_fallback_mass_max': float(np.sum(mass['p'][valid] * fallback, axis=1).max())}
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest, **result)
    CACHE[key] = result
    return result


def load_panel(dep, seed, split, catalog=None):
    f = r.BASE_LOAD(dep, seed, split, catalog)
    a = matrices(dep, seed, split, catalog)
    i, j = f.idx_i.to_numpy(int), f.idx_j.to_numpy(int)
    for field in ('joint_BC', 'joint_logbc', 'joint_ood', 'joint_penalty', 'joint_increment'):
        f['parent_' + field] = f[field].to_numpy()
    f['joint_BC'], f['joint_logbc'] = a['bc'][i, j], a['logbc'][i, j]
    f['joint_ood'] = (a['outside'][i] > .25) | (a['outside'][j] > .25)
    x = r.pair_features(a['summary'], i, j, f.joint_logbc.to_numpy())
    for k, col in enumerate(r.FEATURES):
        f[col] = x[:, k]
    return f


def calibrate_one(dep, seed):
    slot = n.recipes()[dep, seed]['slot']
    meta = pd.read_parquet(n.t.PREVIOUS / f'expanded_data/{dep}/validation/event_metadata.parquet')
    meta['deployment'] = dep
    fold = r.source_fold(meta)
    a = matrices(dep, seed, 'development')
    ids = np.flatnonzero(fold == 0)
    i, j = np.triu_indices(len(ids), 1)
    i, j = ids[i], ids[j]
    groups = meta.source_uid.to_numpy(str)
    y = groups[i] == groups[j]
    keys = np.array([':'.join(sorted((a, b))) for a, b in zip(groups[i], groups[j])])
    count = pd.Series(keys).value_counts()
    weight = pd.Series(keys).map(1 / count).to_numpy()
    weight[y] *= .5 / weight[y].sum()
    weight[~y] *= .5 / weight[~y].sum()
    value = a['logbc'][i, j]
    iso = IsotonicRegression(increasing=True, out_of_bounds='clip').fit(value, y, sample_weight=weight)
    prob = iso.y_thresholds_.clip(1 / (y.sum() + 2), 1 - 1 / (y.sum() + 2))
    spec = {'knots': iso.X_thresholds_.tolist(), 'loglr': (np.log(prob) - np.log1p(-prob)).tolist(),
            'minimum': float(value.min()), 'maximum': float(value.max()),
            'reference': np.sort(-value[y]).tolist(), 'fit_source_systems': int(y.sum()),
            'fit_tune_source_noise_disjoint': True, 'conditional_fallback_mass_max': float(a['conditional_fallback_mass_max'])}
    n.write_json(ROOT / f'calibration/{dep}_{seed}_JOINT.json', spec)
    audit = []
    for side in (0, 1):
        ii = np.flatnonzero(fold == side)
        x, z = np.triu_indices(len(ii), 1)
        x, z = ii[x], ii[z]
        truth = groups[x] == groups[z]
        tail = n.ev.tail.tail_probability(-a['logbc'][x, z], spec['reference'])
        audit.append({'side': side, 'systems': int(truth.sum()),
                      'true_tail_below_0p05': float((tail[truth] < .05).mean()),
                      'source_noise_overlap': 0})
    n.write_csv(ROOT / f'calibration/{dep}_{seed}_TAIL_AUDIT.csv', audit)
    return spec


def infer(frame, c):
    if c['method'] == METHODS[0]:
        f = frame.copy()
        for field in ('joint_BC', 'joint_logbc', 'joint_ood', 'joint_penalty', 'joint_increment'):
            f[field] = f['parent_' + field]
        return r.BASE_INFER(f, {**c, 'method': 'NODUP-DIRECT'})
    spec = c['joint_calibration_subgrid']
    value = frame.joint_logbc.to_numpy(float)
    tail = n.ev.tail.tail_probability(-value, spec['reference'])
    penalty = np.minimum(np.log(tail / .05), 0.)
    increment = np.interp(value, spec['knots'], spec['loglr']).clip(-4., 4.)
    outside = frame.joint_ood.to_numpy(bool) | (value < spec['minimum']) | (value > spec['maximum'])
    increment[outside | (tail < .05)] = np.minimum(increment[outside | (tail < .05)], 0.)
    endpoint = frame.joint_ood.to_numpy(bool)
    increment[endpoint], penalty[endpoint] = 0., 0.
    cos, co, cl = n.apply_model(frame, c['cosine_calibration'])
    return cos + c['gamma'] * penalty + c['beta'] * increment, outside | co, cl | (abs(increment) >= 4)


def freeze_score():
    path = ROOT / 'contracts/SCORING_PROTOCOL.json'
    if path.exists():
        raise RuntimeError('Already frozen')
    n.write_json(path, {'UTC': n.utc(), 'methods': list(METHODS),
        'density_model': 'New continuous mass;no old marginal mass penalty or old encoder Mc/q input',
        'conditional': 'Retain frozen p(eta,chi|Mc,waveform),normalize at everymass. Only bins with float underflow use frozen simulated conditional prior,record probability mass affected.',
        'calibration': 'source/noise fitfold inherited202609850,source-pair weighted isotonic jointBC LR;finite independent true-source ranktail;clipincrement[-4,4];positiveOOD0;endpointOODneutral',
        'fixed_control': 'pure upstream outerweights and original joint gamma/beta,NEWcalibration not oldscores',
        'retuned': 'gamma0,.25,.5,1,2,4,8 beta0,.0625,.125,.25,.5,1,2,4;existingwaveform/fusionguards;candidatepriority;tiesnormthenlexicographic',
        'no_raw_physics_change': True, 'real_labels_inputs': False, 'real_selection': False,
        'adaptive_development': True})
    shutil.copy2(__file__, ROOT / 'scripts/subgrid_evaluate.py')
    n.write_json(ROOT / 'contracts/SCORING_FREEZE.json', {'UTC': n.utc(), 'script_sha256': n.sha(Path(__file__)),
                 'contract_sha256': n.sha(path)})


def calibrate():
    originals = json.loads((r.PRIOR / 'configs/SELECTED_CONFIGURATIONS.json').read_text())
    original = {(c['deployment'], c['seed']): c for c in originals if c['method'] == 'NODUP-DIRECT'}
    configs, grid = [], []
    for dep in n.DEPS:
        for seed in n.SEEDS:
            spec = calibrate_one(dep, seed)
            old = original[dep, seed]
            configs.append({**old, 'method': METHODS[0]})
            common = {**old, 'joint_calibration_subgrid': spec}
            configs.append({**common, 'method': METHODS[1]})
            f = load_panel(dep, seed, 'validation')
            weights = np.asarray(old['weights'])
            bm, bwm = n.cf.fast_metrics(f, f.PATH875_final_score.to_numpy()), n.cf.fast_metrics(f, f.PATH875_waveform.to_numpy())
            options = []
            for gamma in (0., .25, .5, 1., 2., 4., 8.):
                for beta in (0., .0625, .125, .25, .5, 1., 2., 4.):
                    c = {**common, 'method': METHODS[2], 'gamma': gamma, 'beta': beta}
                    z, oo, cl = infer(f, c)
                    m, wm = n.cf.fast_metrics(f, n.cf.channels(f, z) @ weights), n.cf.fast_metrics(f, z)
                    row = {'gamma': gamma, 'beta': beta, 'guard': n.cf.guard(m, bm) and n.cf.guard(wm, bwm), **m}
                    options.append(row)
                    grid.append({'deployment': dep, 'seed': seed, **row, **{'waveform_' + k: v for k, v in wm.items()}})
            eligible = [r for r in options if r['guard']]
            win = min(eligible or options, key=lambda r: (r['false_at_recall_0p5'], r['false_at_recall_0p9'],
                      -r['average_precision'], -r['macro_r_at_10'], -r['macro_r_at_1'], r['gamma']**2 + r['beta']**2, r['gamma'], r['beta']))
            configs.append({**common, 'method': METHODS[2], 'gamma': win['gamma'], 'beta': win['beta'],
                            'tune_guard': bool(eligible), 'tune_metrics': win})
            print('SUBGRID_SELECTION', dep, seed, win, flush=True)
    n.write_json(ROOT / 'configs/SELECTED_CONFIGURATIONS.json', configs)
    n.write_csv(ROOT / 'tables/SUBGRID_VALIDATION_GRID.csv', grid)
    n.write_json(ROOT / 'contracts/CONFIGURATIONS_FROZEN.json', {'UTC': n.utc(),
        'file': 'configs/SELECTED_CONFIGURATIONS.json', 'sha256': n.sha(ROOT / 'configs/SELECTED_CONFIGURATIONS.json'),
        'real_or_test_selection': False, 'no_total_alpha': True})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--stage', choices=('freeze_score', 'calibrate', 'evaluate', 'real'), required=True)
    args = parser.parse_args()
    ROOT = density.ROOT = r.ROOT = args.root
    r.install()
    n.METHODS, n.infer, n.load_panel = METHODS, infer, load_panel
    if args.stage in ('freeze_score', 'calibrate'):
        globals()[args.stage]()
    else:
        n.run(ROOT, args.stage)
