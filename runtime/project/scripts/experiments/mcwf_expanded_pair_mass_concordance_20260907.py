#!/usr/bin/env python3
"""Pair-verifier evidence conditioned on independent waveform mass predictions."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
import argparse
from datetime import datetime, timezone
from pathlib import Path
import json
import shutil
import numpy as np
import pandas as pd
import mcwf_pe_frontend_extension_v2_20260907 as e
import mcwf_expanded_pair_verifier_20260907 as verifier
import mcwf_predictive_mass_pool_evaluate_20260907 as mass
import mcwf_finite_reference_tail_20260907 as tail

dev, body, ev = e.dev, e.body, e.ev
GAMMAS = (0., .125, .25, .5, 1., 2., 4.)
BETAS = (0., .0625, .125, .25, .5, 1., 2.)
POWERS = (0., 1., 2.)


def get(root, dep, ms, es, split):
    mass.mass_model.MODE = 'mixture'
    f, x = mass.values(root, dep, ms, es, split)
    vf, logits = verifier.raw_score(root, dep, ms, es, split)
    if not np.array_equal(f.idx_i, vf.idx_i) or not np.array_equal(f.idx_j, vf.idx_j):
        raise RuntimeError('Pair alignment mismatch')
    return f, {**x, 'logits': logits}


def score(f, x, spec):
    cal = spec['verifier']
    raw = np.interp(x['logits'], cal['knots'], cal['loglr'])
    ood = x['ood'] | (x['logits'] < cal['fit_min']) | (x['logits'] > cal['fit_max'])
    p = tail.tail_probability(-np.log(x['bc']), spec['mass_reference'])
    oldp = tail.tail_probability(-np.log(f.new_mass_predictive_BC.to_numpy(float).clip(1e-12)), spec['old_mass_reference'])
    gate = np.minimum(p, oldp) ** spec['power']
    inc = np.minimum(raw, 0.) + np.maximum(raw, 0.) * gate
    inc = np.clip(np.where(ood | (oldp < .05), np.minimum(inc, 0.), inc), -4., 4.)
    penalty = np.minimum(np.log(p / .05), 0.)
    penalty = np.where(x['ood'], 0., penalty)
    return f.waveform_score.to_numpy(float) + spec['gamma'] * penalty + spec['beta'] * inc, penalty, inc, gate, ood


def run(root):
    trial = root / 'trials/EXPANDED-PAIR-MASS-CONCORDANCE'
    if trial.exists():
        raise RuntimeError('Independent trial required')
    for name in ('contracts', 'calibration', 'tables', 'evaluation', 'results'):
        (trial / name).mkdir(parents=True)
    shutil.copy2(__file__, trial / 'contracts' / Path(__file__).name)
    dev.json_write(trial / 'contracts/RECIPE.json', {'created_utc': datetime.now(timezone.utc).isoformat(),
        'same_O3_O4a': True, 'same_2s_input': True,
        'formula': 'FRT+gamma*newMcfinitepenalty+beta*(negative_verifierLR+positive_verifierLR*mass_gate)',
        'mass_gate': 'min(newpredictiveMcBCtail,oldRNCpredictiveMcBCtail)^power',
        'models': 'frozen4096source-expandedpairverifierandheterogeneousmixtureMcmodels, no additional network training',
        'fit': 'verifier class-balanced512sourcevalidation; newMctail512sourcevalidation; oldMctailBAYESTARvalidationonepair/source',
        'gamma_grid': GAMMAS, 'beta_grid': BETAS, 'power_grid': POWERS,
        'selection': 'original BAYESTAR validation objective and guards only',
        'no_PE_official_ID_features': True, 'fresh_confirmation_required': True,
        'limits': 'correlated waveform predictors, not independent Bayes factors; real audit is adaptive development',
        'frozen': ['time', 'sky', 'outer weights', 'scope', 'historical outputs']})
    selected, cache, states = {}, {}, []
    for dep in e.DEPS:
        for ms, es in zip(body.MODEL_SEEDS, dev.SEEDS):
            mass.mass_model.MODE = 'mixture'
            ref = mass.calibrate(root, dep, ms)
            cal = json.loads((root / f'expanded_pair_verifier/models/{dep}/seed_{ms}/CALIBRATION.json').read_text())
            f, x = get(root, dep, ms, es, 'validation')
            cache[dep, es] = f, x
            base = ev.metrics(f, f.waveform_score.to_numpy(float), dep, es)
            fixed = {'verifier': cal, 'mass_reference': ref['mass_reference'],
                     'old_mass_reference': np.sort(-np.log(f.loc[f.is_true_pair, 'new_mass_predictive_BC'].to_numpy(float).clip(1e-12))).tolist()}
            rows, options = [], []
            for power in POWERS:
                for gamma in GAMMAS:
                    for beta in BETAS:
                        if beta == 0. and power != 0.:
                            continue
                        spec = {**fixed, 'power': power, 'gamma': gamma, 'beta': beta}
                        z, *_ = score(f, x, spec)
                        mm = ev.metrics(f, z, dep, es)
                        ok = ev.guard(mm, base)
                        rows.append({'gamma': gamma, 'beta': beta, 'power': power, 'pass': ok,
                                     **{a + '_' + k: v for a, b in mm.items() for k, v in b.items()}})
                        if ok:
                            options.append((e.old_selection.objective(mm, gamma, beta) + (power,), spec))
            spec = min(options, key=lambda a: a[0])[1]
            selected[dep, es] = spec
            out = trial / f'calibration/{dep}/seed_{es}'
            out.mkdir(parents=True)
            dev.csv_write(out / 'validation_grid.csv', pd.DataFrame(rows))
            dev.json_write(out / 'SELECTED_CONFIG.json', spec)
            states.append({'deployment': dep, 'seed': es, 'gamma': spec['gamma'], 'beta': spec['beta'], 'power': spec['power']})
    dev.json_write(trial / 'contracts/SELECTED_FREEZE.json', {'configs': states, 'real_used_to_select': False})
    print(json.dumps({'pair_mass_concordance_selection': states}), flush=True)
    rows, guards = [], []
    for dep in e.DEPS:
        real = {}
        for ms, es in zip(body.MODEL_SEEDS, dev.SEEDS):
            for split in ('validation', 'test', 'real'):
                f, x = cache[dep, es] if split == 'validation' else get(root, dep, ms, es, split)
                z, penalty, inc, gate, ood = score(f, x, selected[dep, es])
                n = f.rename(columns={c: 'baseline_' + c for c in ('final_score', 'rank', 'method', 'waveform_contribution', 'time_contribution', 'sky_contribution') if c in f}).copy()
                n['FRT_baseline_waveform_score'], n['waveform_score'] = f.waveform_score, z
                n['new_mass_penalty'], n['new_verifier_increment'] = penalty, inc
                n['mass_gate'], n['new_ood'] = gate, ood
                out = trial / f'evaluation/{dep}/seed_{es}'
                out.mkdir(parents=True, exist_ok=True)
                n.to_parquet(out / f'{split}_pairs.parquet', index=False)
                if split == 'real':
                    real[es] = n
                else:
                    bm, mm = ev.metrics(f, f.waveform_score.to_numpy(float), dep, es), ev.metrics(f, z, dep, es)
                    guards.append({'deployment': dep, 'seed': es, 'split': split, 'pass': ev.guard(mm, bm)})
                    for method in mm:
                        for config, m in [('FRT_BASELINE', bm[method]), ('CANDIDATE', mm[method])]:
                            rows.append({'deployment': dep, 'seed': es, 'split': split, 'method': method, 'config': config, **m})
        dev.save_evaluation(trial, 'CANDIDATE', dep, real, pd.read_parquet(root / f'audit/{dep}_frozen_external_reference.parquet'))
    dev.csv_write(trial / 'tables/RETRIEVAL_PER_SEED.csv', pd.DataFrame(rows))
    dev.csv_write(trial / 'tables/REUSED_GUARDRAILS.csv', pd.DataFrame(guards))
    e.assess(root, trial)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    run(p.parse_args().root)
