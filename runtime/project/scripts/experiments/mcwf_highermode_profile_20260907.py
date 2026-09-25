#!/usr/bin/env python3
"""Finer q/spin matched-waveform profiles, simulation-calibrated only."""
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
from scipy.special import logsumexp
from sklearn.isotonic import IsotonicRegression
import mcwf_pe_frontend_extension_v2_20260907 as e
import mcwf_highermode_features_20260907 as higher
import mcwf_mass_tf_20260905 as tf
import torch
import mcwf_physical_profile_20260907 as coarse

dev, body, ev = e.dev, e.body, e.ev


def combine(old, extra):
    return extra[:, 2].reshape(len(extra),64,72)


def distribution(x, temperature):
    power = np.expm1(np.asarray(x, float)).reshape(len(x), -1)
    logits = (power-power.max(1, keepdims=True))/(2*temperature)
    lp = logits-logsumexp(logits, axis=1, keepdims=True)
    return lp, logsumexp(lp.reshape(len(x), 64, 72), axis=2)


def pair_features(x, i, j, temperature):
    lp, lm = distribution(x, temperature)
    source = logsumexp(lp.reshape(len(x),2304,2),axis=2)
    result = []
    for first in range(0, len(i), 512):
        a, b = i[first:first+512], j[first:first+512]
        full = np.log(2304)+logsumexp(source[a]+source[b], axis=1)
        mass = np.log(64)+logsumexp(lm[a]+lm[b], axis=1)
        result.append(np.stack([full, full-mass], 1))
    return np.concatenate(result)


def fit(root, dep):
    path = root / f'highermode_profile/{dep}/FIT.json'
    if path.exists():
        return json.loads(path.read_text())
    old = None
    x = combine(old, higher.development(root, dep, 'validation'))
    meta = pd.read_parquet(root/f'expanded_data/{dep}/validation/event_metadata.parquet')
    target = tf.target_prob(np.log(meta.mc_det.to_numpy(float)))
    grid = []
    for t in coarse.TEMPERATURES:
        _, lm = distribution(x, t)
        grid.append({'temperature': t, 'Mc_CE': float(-(target*lm).sum(1).mean())})
    t = min(grid, key=lambda row: (row['Mc_CE'], row['temperature']))['temperature']
    
    i, j = np.triu_indices(len(meta), 1)
    y = meta.source_uid.to_numpy()[i] == meta.source_uid.to_numpy()[j]
    take = np.r_[np.flatnonzero(y),np.random.default_rng(202609211).choice(np.flatnonzero(~y),100000,replace=False)]
    i,j,y=i[take],j[take],y[take]
    raw = pair_features(x, i, j, t)
    specs = {}
    for column, kind in enumerate(coarse.KINDS):
        v = raw[:, column]
        weights = np.where(y, .5/y.sum(), .5/(~y).sum())
        iso = IsotonicRegression(increasing=True, out_of_bounds='clip').fit(v, y.astype(float), sample_weight=weights)
        floor = 1/(y.sum()+2)
        prob = np.clip(iso.y_thresholds_, floor, 1-floor)
        specs[kind] = {'temperature': t, 'kind': kind, 'column': column, 'knots': iso.X_thresholds_.tolist(),
            'loglr': (np.log(prob)-np.log1p(-prob)).tolist(), 'fit_min': float(v.min()), 'fit_max': float(v.max())}
    dev.json_write(path, specs)
    dev.csv_write(path.parent / 'TEMPERATURE_GRID.csv', pd.DataFrame(grid))
    print(json.dumps({'highermode_profile_fit': dep, 'temperature': t, 'CE': min(r['Mc_CE'] for r in grid)}), flush=True)
    return specs


def get(root, dep, es, split, temperature):
    f = pd.read_parquet(e.BASE / f'evaluation/{dep}/seed_{es}/{split}_pairs.parquet')
    extra = higher.deployment(root, dep, es, split)
    i, j = f.idx_i.to_numpy(int), f.idx_j.to_numpy(int)
    if split == 'real':
        old = np.load(e.TRAINED / f'cache/deployment_event_psd/{dep}/real_features.npy')
        _, events = dev.real_inputs(dep)
        valid = events.strict_h1l1_preprocessing_pass.to_numpy(bool)
        remap = np.full(len(events), -1, int)
        remap[valid] = np.arange(valid.sum())
        i, j = remap[i], remap[j]
        if (i < 0).any() or (j < 0).any():
            raise RuntimeError('Missing valid H1L1 data')
    else:
        old = np.load(e.TRAINED / f'cache/deployment_event_psd/{dep}/{es}_{split}_features.npy')
    return f, pair_features(combine(old, extra), i, j, temperature)


def run(root):
    trial = root / 'trials/HIGHER-MODE-PHYSICAL-PROFILE'
    if trial.exists():
        raise RuntimeError('Independent trial required')
    for name in ('contracts', 'calibration', 'tables', 'evaluation', 'results'):
        (trial / name).mkdir(parents=True)
    shutil.copy2(__file__, trial / 'contracts' / Path(__file__).name)
    dev.json_write(trial / 'contracts/RECIPE.json', {
        'created_utc': datetime.now(timezone.utc).isoformat(), 'same_O3_O4a': True,
        'single_changed_mechanism': 'use XPHM64Mc*3q*3alignedspin*2precessionspin*2inclination profiles; marginalize plus/cross projection per event before source compatibility;unchanged2s input',
        'temperature_grid': coarse.TEMPERATURES, 'kinds': coarse.KINDS, 'beta_grid': coarse.BETAS,
        'calibration': '512newsource companions,100kfixedrandomnulls,32sharednoiseblocks; isotoniccap+-4positiveOOD0;not100k independent systems',
        'selection': 'BAYESTAR validation only, same objective and FRT-relative guardrails',
        'limits': 'maximized phase/time profiles are NOT physical PE posteriors,properBF,or a template-bank minimal-match guarantee; Mc temperature calibration does not establish q/spin coverage; not exact multimode phase marginalization',
        'frozen': ['time', 'sky', 'outerweights', 'scope', 'encoders', 'historical data'],
        'no_real_PE_or_official_features': True})
    selected, cache, states = {}, {}, []
    for dep in e.DEPS:
        cal = fit(root, dep)
        for es in dev.SEEDS:
            f, x = get(root, dep, es, 'validation', cal[coarse.KINDS[0]]['temperature'])
            cache[dep, es] = f, x
            bm = ev.metrics(f, f.waveform_score.to_numpy(float), dep, es)
            rows, choices = [], []
            for kind in coarse.KINDS:
                for beta in coarse.BETAS:
                    spec = {**cal[kind], 'beta': beta}
                    z, _, ood = coarse.score(f, x, spec)
                    mm = ev.metrics(f, z, dep, es)
                    ok = ev.guard(mm, bm)
                    rows.append({'kind': kind, 'beta': beta, 'pass': ok, 'ood': float(ood.mean()),
                        **{a+'_'+k: v for a, b in mm.items() for k, v in b.items()}})
                    if ok:
                        choices.append((e.old_selection.objective(mm, beta, coarse.KINDS.index(kind)), spec))
            out = trial / f'calibration/{dep}/seed_{es}'
            out.mkdir(parents=True)
            dev.csv_write(out / 'validation_grid.csv', pd.DataFrame(rows))
            spec = min(choices, key=lambda x: x[0])[1]
            selected[dep, es] = spec
            dev.json_write(out / 'SELECTED_CONFIG.json', spec)
            states.append({'deployment': dep, 'seed': es, 'kind': spec['kind'], 'beta': spec['beta']})
    dev.json_write(trial / 'contracts/SELECTED_FREEZE.json', {'configs': states, 'real_used_to_select': False})
    print(json.dumps({'highermode_profile_selected': states}), flush=True)
    rows, guards = [], []
    for dep in e.DEPS:
        real = {}
        for es in dev.SEEDS:
            cfg = selected[dep, es]
            for split in ('validation', 'test', 'real'):
                f, x = cache[dep, es] if split == 'validation' else get(root, dep, es, split, cfg['temperature'])
                z, inc, ood = coarse.score(f, x, cfg)
                n = f.copy()
                n['FRT_baseline_waveform_score'] = f.waveform_score
                n['waveform_score'], n['highermode_profile_increment'], n['highermode_profile_ood'] = z, inc, ood
                for k, kind in enumerate(coarse.KINDS):
                    n['highermode_profile_'+kind] = x[:, k]
                out = trial / f'evaluation/{dep}/seed_{es}'
                out.mkdir(parents=True, exist_ok=True)
                n.to_parquet(out / f'{split}_pairs.parquet', index=False)
                if split == 'real':
                    real[es] = n
                else:
                    bm, mm = ev.metrics(f, f.waveform_score.to_numpy(float), dep, es), ev.metrics(f, z, dep, es)
                    guards.append({'deployment': dep, 'seed': es, 'split': split, 'pass': ev.guard(mm, bm)})
                    for method in bm:
                        for config, m in [('FRT_BASELINE', bm[method]), ('CANDIDATE', mm[method])]:
                            rows.append({'deployment': dep, 'seed': es, 'split': split, 'method': method, 'config': config, **m})
        dev.save_evaluation(trial, 'CANDIDATE', dep, real, pd.read_parquet(root / f'audit/{dep}_frozen_external_reference.parquet'))
    dev.csv_write(trial / 'tables/RETRIEVAL_PER_SEED.csv', pd.DataFrame(rows))
    dev.csv_write(trial / 'tables/REUSED_GUARDRAILS.csv', pd.DataFrame(guards))
    e.assess(root, trial)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    a = p.parse_args()
    run(a.root)
