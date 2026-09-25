#!/usr/bin/env python3
"""Finite mass-only predictive evidence; all external labels are audit-only."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import torch
from sklearn.isotonic import IsotonicRegression
import mcwf_pe_frontend_extension_v2_20260907 as e
import mcwf_band_consistency_mass_20260907 as mass_model
import mcwf_finite_reference_tail_20260907 as tail

dev, body, ev = e.dev, e.body, e.ev
GAMMAS = (0., .125, .25, .5, 1., 2., 4.)
BETAS = (0., .0625, .125, .25, .5, 1., 2.)


def pair_features(a, i, j, prior):
    bc, overlap = np.empty(len(i)), np.empty(len(i))
    for start in range(0, len(i), 2048):
        sl = slice(start, start + 2048)
        product = a['p'][i[sl]] * a['p'][j[sl]]
        bc[sl] = np.sqrt(product).sum(1)
        overlap[sl] = (product / prior).sum(1)
    return {'bc': bc.clip(1e-15, 1.),
            'prior_overlap': np.log(overlap.clip(1e-300)),
            'ood': (a['outside'][i] > .25) | (a['outside'][j] > .25)}


def values(root, dep, ms, es, split):
    cp = root / f'{mass_model.root_name()}/models/{dep}/seed_{ms}/validation_selected_model.pt'
    ck = torch.load(cp, weights_only=False, map_location='cpu')
    if split == 'development':
        a = np.load(cp.parent / 'validation_predictions.npz')
        i, j = np.triu_indices(len(a['p']), 1)
        f = pd.DataFrame({'idx_i': i, 'idx_j': j,
                          'is_true_pair': a['group'][i] == a['group'][j]})
    else:
        a = mass_model.prediction(root, dep, ms, es, split)
        f = pd.read_parquet(e.BASE / f'evaluation/{dep}/seed_{es}/{split}_pairs.parquet')
        i, j = f.idx_i.to_numpy(int), f.idx_j.to_numpy(int)
    return f, pair_features(a, i, j, ck['prior'])


def calibrate(root, dep, ms):
    path = root / f'{mass_model.root_name()}/calibration/{dep}/seed_{ms}/CALIBRATION.json'
    if path.exists():
        return json.loads(path.read_text())
    f, x = values(root, dep, ms, 0, 'development')
    y = f.is_true_pair.to_numpy(bool)
    weight = np.where(y, .5 / y.sum(), .5 / (~y).sum())
    spec = {'mass_reference': np.sort(-np.log(x['bc'][y])).tolist(),
            'source_pairs': int(y.sum()), 'noise_blocks': 32}
    for feature in ('bc', 'prior_overlap'):
        iso = IsotonicRegression(increasing=True, out_of_bounds='clip').fit(
            x[feature], y, sample_weight=weight)
        p = iso.y_thresholds_.clip(1 / (y.sum() + 2), 1 - 1 / (y.sum() + 2))
        spec[feature] = {'knots': iso.X_thresholds_.tolist(),
                         'loglr': (np.log(p) - np.log1p(-p)).tolist(),
                         'minimum': float(x[feature].min()),
                         'maximum': float(x[feature].max())}
    dev.json_write(path, spec)
    return spec


def score(f, x, spec, arm):
    reference_p = tail.tail_probability(-np.log(x['bc']), spec['mass_reference'])
    penalty = np.minimum(np.log(reference_p / .05), 0.)
    penalty = np.where(x['ood'], 0., penalty)
    increment = np.zeros(len(f))
    if arm != 'TAIL':
        feature = 'bc' if arm == 'BC' else 'prior_overlap'
        cal = spec[feature]
        raw = np.interp(x[feature], cal['knots'], cal['loglr'])
        ood = x['ood'] | (x[feature] < cal['minimum']) | (x[feature] > cal['maximum'])
        # Positive rewards require both learned mass compatibility and calibration support.
        raw = np.where(ood | (reference_p < .05), np.minimum(raw, 0.), raw)
        increment = np.where(x['ood'], 0., np.clip(raw, -4, 4))
    return f.waveform_score.to_numpy(float) + spec['gamma'] * penalty + spec['beta'] * increment, penalty, increment


def run(root, arm):
    trial = root / f'trials/BAND-CONSISTENCY-MASS-{mass_model.MODE.upper()}-{arm}'
    if trial.exists():
        raise RuntimeError('Independent trial required')
    for name in ('contracts', 'calibration', 'tables', 'evaluation', 'results'):
        (trial / name).mkdir(parents=True)
    shutil.copy2(__file__, trial / 'contracts' / Path(__file__).name)
    dev.json_write(trial / 'contracts/RECIPE.json', {
        'created_utc': datetime.now(timezone.utc).isoformat(), 'same_O3_O4a': True,
        'arm': arm, 'pooling': mass_model.MODE, 'model': 'source-balanced4Gaussian logMc density;control1728 fullpower features versus4032augmented subbandresidual+dof features;same2s;notPE',
        'formula': 'Z_FRT + gamma*finite_mass_tail + beta*bounded_calibrated_mass_evidence',
        'gamma_grid': GAMMAS, 'beta_grid': (0.,) if arm == 'TAIL' else BETAS,
        'calibration': '512 independent source pairs; all source-pair negatives are dependent, with 32 noise blocks',
        'prior_overlap': 'sum predictive_p_i*predictive_p_j/training_prior; used only as a calibrated feature, not a proper PE Bayes factor',
        'OOD': 'no new positive reward outside calibration or incompatible mass; outside5-200Msun probability>25percent makes all new terms neutral',
        'selection': 'BAYESTAR validation only, original priority and guardrails',
        'frozen': ['time', 'sky', 'scope', 'outer C-fixed weights', 'historical outputs'],
        'real_used_to_select': False, 'PE_official_ID_inputs': False,
        'real_audit_is_adaptive_development': True, 'fresh_confirmation_required': True})
    selected, cache, states = {}, {}, []
    for dep in e.DEPS:
        for ms, es in zip(body.MODEL_SEEDS, dev.SEEDS):
            cal = calibrate(root, dep, ms)
            f, x = values(root, dep, ms, es, 'validation')
            cache[dep, es] = f, x
            bm = ev.metrics(f, f.waveform_score.to_numpy(float), dep, es)
            rows, choices = [], []
            for gamma in GAMMAS:
                for beta in ((0.,) if arm == 'TAIL' else BETAS):
                    spec = {**cal, 'gamma': gamma, 'beta': beta}
                    z, _, _ = score(f, x, spec, arm)
                    mm = ev.metrics(f, z, dep, es)
                    ok = ev.guard(mm, bm)
                    rows.append({'gamma': gamma, 'beta': beta, 'pass': ok,
                                 **{a + '_' + k: v for a, b in mm.items() for k, v in b.items()}})
                    if ok:
                        choices.append((e.old_selection.objective(mm, gamma, beta), spec))
            spec = min(choices, key=lambda a: a[0])[1]
            selected[dep, es] = spec
            out = trial / f'calibration/{dep}/seed_{es}'
            out.mkdir(parents=True)
            dev.csv_write(out / 'validation_grid.csv', pd.DataFrame(rows))
            dev.json_write(out / 'SELECTED_CONFIG.json', spec)
            states.append({'deployment': dep, 'seed': es, 'gamma': spec['gamma'], 'beta': spec['beta']})
    dev.json_write(trial / 'contracts/SELECTED_FREEZE.json', {'configs': states, 'real_used_to_select': False})
    print(json.dumps({'fine_mass_only_selected': arm, 'configs': states}), flush=True)
    rows, guards = [], []
    for dep in e.DEPS:
        real = {}
        for ms, es in zip(body.MODEL_SEEDS, dev.SEEDS):
            for split in ('validation', 'test', 'real'):
                f, x = cache[dep, es] if split == 'validation' else values(root, dep, ms, es, split)
                z, penalty, increment = score(f, x, selected[dep, es], arm)
                n = f.rename(columns={k: 'baseline_' + k for k in (
                    'final_score', 'rank', 'method', 'waveform_contribution', 'time_contribution', 'sky_contribution') if k in f}).copy()
                n['FRT_baseline_waveform_score'] = f.waveform_score
                n['waveform_score'] = z
                n['new_mass_tail_penalty'] = penalty
                n['new_mass_evidence'] = increment
                for key, value in x.items():
                    n['fine_mass_' + key] = value
                for col in ('time_score', 'sky_raw_log_bf'):
                    assert np.array_equal(n[col], f[col])
                out = trial / f'evaluation/{dep}/seed_{es}'
                out.mkdir(parents=True, exist_ok=True)
                n.to_parquet(out / f'{split}_pairs.parquet', index=False)
                if split == 'real':
                    real[es] = n
                else:
                    bm = ev.metrics(f, f.waveform_score.to_numpy(float), dep, es)
                    mm = ev.metrics(f, z, dep, es)
                    guards.append({'deployment': dep, 'seed': es, 'split': split, 'pass': ev.guard(mm, bm)})
                    for method in mm:
                        for config, metrics in [('FRT_BASELINE', bm[method]), ('CANDIDATE', mm[method])]:
                            rows.append({'deployment': dep, 'seed': es, 'split': split, 'method': method, 'config': config, **metrics})
        dev.save_evaluation(trial, 'CANDIDATE', dep, real, pd.read_parquet(root / f'audit/{dep}_frozen_external_reference.parquet'))
    dev.csv_write(trial / 'tables/RETRIEVAL_PER_SEED.csv', pd.DataFrame(rows))
    dev.csv_write(trial / 'tables/REUSED_GUARDRAILS.csv', pd.DataFrame(guards))
    e.assess(root, trial)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    a = p.parse_args()
    for mode in ('control', 'augmented'):
        mass_model.MODE = mode
        for arm in ('TAIL', 'BC', 'PRIOR'):
            run(a.root, arm)


