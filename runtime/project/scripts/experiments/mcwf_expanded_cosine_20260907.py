#!/usr/bin/env python3
"""Test additional source geometry from the expanded-data RNC encoders."""
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
from sklearn.isotonic import IsotonicRegression
import mcwf_pe_frontend_extension_v2_20260907 as e
import mcwf_expanded_encoder_20260907 as expanded

dev, body, ev = e.dev, e.body, e.ev
BETAS = (0., .0625, .125, .25, .5, 1., 2., 4.)


def unit(z):
    z = np.asarray(z, float)
    return z/np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-12)


def fit(root, dep, ms):
    a = np.load(root / f'expanded_encoder/models/{dep}/seed_{ms}/validation_predictions.npz')
    z = unit(a['z'])
    i, j = np.triu_indices(len(z), 1)
    raw = np.sum(z[i]*z[j], 1)
    y = a['group'][i] == a['group'][j]
    w = np.where(y, .5/y.sum(), .5/(~y).sum())
    iso = IsotonicRegression(increasing=True, out_of_bounds='clip').fit(raw, y.astype(float), sample_weight=w)
    floor = 1/(y.sum()+2)
    p = np.clip(iso.y_thresholds_, floor, 1-floor)
    return {'knots': iso.X_thresholds_.tolist(), 'loglr': (np.log(p)-np.log1p(-p)).tolist(),
        'fit_min': float(raw.min()), 'fit_max': float(raw.max()), 'source_reference': int(y.sum()),
        'checkpoint_sha256': dev.sha(root / f'expanded_encoder/models/{dep}/seed_{ms}/validation_selected_model.pt')}


def get(root, dep, ms, es, split):
    f = pd.read_parquet(e.BASE / f'evaluation/{dep}/seed_{es}/{split}_pairs.parquet')
    a = expanded.prediction(root, dep, ms, es, split)
    z = unit(a['z'])
    return f, np.sum(z[f.idx_i.to_numpy(int)]*z[f.idx_j.to_numpy(int)], 1)


def score(f, x, spec):
    inc = np.interp(x, spec['knots'], spec['loglr'])
    ood = (x < spec['fit_min']) | (x > spec['fit_max'])
    inc = np.clip(np.where(ood, np.minimum(inc, 0.), inc), -4, 4)
    return f.waveform_score.to_numpy(float)+spec['beta']*inc, inc, ood


def run(root):
    trial = root / 'trials/EXPANDED-RNC-COSINE-INCREMENT'
    if trial.exists():
        raise RuntimeError('Independent trial required')
    for name in ('contracts', 'calibration', 'tables', 'evaluation', 'results'):
        (trial / name).mkdir(parents=True)
    shutil.copy2(__file__, trial / 'contracts' / Path(__file__).name)
    dev.json_write(trial / 'contracts/RECIPE.json', {
        'created_utc': datetime.now(timezone.utc).isoformat(), 'same_O3_O4a': True,
        'formula': 'Z_FRT+beta*isotonic_LLR(new_RNC_cosine), bounded+-4,positiveOOD0',
        'new_information': '128D source-geometry representation trained on expanded independent source/noise population; original FRT mass guard retained',
        'calibration': '512 independent new developmentvalidation sources,one truepair/source,equal total positive/null classweights',
        'beta_grid': BETAS, 'selection': 'same BAYESTAR validation-only objective and FRT-relative guards',
        'physical_limit': 'correlated discriminative waveform increments,not independent Bayes factors or true PE likelihoods',
        'frozen': ['time', 'sky', 'outerweights', 'scope', 'old results'],
        'no_real_PE_or_official_features': True, 'fresh_confirmation_required': True})
    chosen, cache, states = {}, {}, []
    for dep in e.DEPS:
        for ms, es in zip(body.MODEL_SEEDS, dev.SEEDS):
            cal = fit(root, dep, ms)
            f, x = get(root, dep, ms, es, 'validation')
            cache[dep, es] = f, x
            bm = ev.metrics(f, f.waveform_score.to_numpy(float), dep, es)
            choices, rows = [], []
            for beta in BETAS:
                spec = {**cal, 'beta': beta}
                z, _, ood = score(f, x, spec)
                mm = ev.metrics(f, z, dep, es)
                ok = ev.guard(mm, bm)
                rows.append({'beta': beta, 'pass': ok, 'ood': float(ood.mean()),
                    **{a+'_'+k: v for a, b in mm.items() for k, v in b.items()}})
                if ok:
                    choices.append((e.old_selection.objective(mm, beta, 1.), spec))
            out = trial / f'calibration/{dep}/seed_{es}'
            out.mkdir(parents=True)
            dev.csv_write(out / 'validation_grid.csv', pd.DataFrame(rows))
            spec = min(choices, key=lambda x: x[0])[1]
            chosen[dep, es] = spec
            dev.json_write(out / 'SELECTED_CONFIG.json', spec)
            states.append({'deployment': dep, 'seed': es, 'beta': spec['beta']})
    dev.json_write(trial / 'contracts/SELECTED_FREEZE.json', {'configs': states, 'real_used_to_select': False})
    print(json.dumps({'expanded_cosine_selection': states}), flush=True)
    rows, guards = [], []
    for dep in e.DEPS:
        real = {}
        for ms, es in zip(body.MODEL_SEEDS, dev.SEEDS):
            for split in ('validation', 'test', 'real'):
                f, x = cache[dep, es] if split == 'validation' else get(root, dep, ms, es, split)
                z, inc, ood = score(f, x, chosen[dep, es])
                n = f.copy()
                n['FRT_baseline_waveform_score'] = f.waveform_score
                n['waveform_score'], n['expanded_cosine'], n['expanded_cosine_increment'], n['expanded_cosine_ood'] = z, x, inc, ood
                for col in ('time_score', 'sky_raw_log_bf'):
                    assert np.array_equal(n[col], f[col])
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
