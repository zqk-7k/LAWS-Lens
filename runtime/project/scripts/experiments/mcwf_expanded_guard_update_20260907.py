#!/usr/bin/env python3
"""Retain the original mass safeguard while auditing the new source population."""
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
import mcwf_finite_reference_tail_20260907 as tail

dev, body, ev = e.dev, e.body, e.ev
GAMMAS = (0., .125, .25, .5, 1., 2.)
BETAS = (0., .0625, .125, .25, .5, 1., 2.)


def pair_values(p, z, i, j):
    z = np.asarray(z, float)
    z /= np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-12)
    return {'mass': -np.log(np.sum(np.sqrt(p[i]*p[j]), 1).clip(1e-12)), 'cosine': np.sum(z[i]*z[j], 1)}


def fit(root, dep, ms):
    a = np.load(root / f'expanded_encoder/models/{dep}/seed_{ms}/validation_predictions.npz')
    i, j = np.triu_indices(len(a['p']), 1)
    y = a['group'][i] == a['group'][j]
    x = pair_values(a['p'], a['z'], i, j)
    weights = np.where(y, .5/y.sum(), .5/(~y).sum())
    iso = IsotonicRegression(increasing=True, out_of_bounds='clip').fit(x['cosine'], y.astype(float), sample_weight=weights)
    floor = 1/(y.sum()+2)
    p = np.clip(iso.y_thresholds_, floor, 1-floor)
    return {'mass_reference': np.sort(x['mass'][y]).tolist(), 'source_reference_count': int(y.sum()),
        'knots': iso.X_thresholds_.tolist(), 'loglr': (np.log(p)-np.log1p(-p)).tolist(),
        'fit_min': float(x['cosine'].min()), 'fit_max': float(x['cosine'].max()),
        'checkpoint_sha256': dev.sha(root / f'expanded_encoder/models/{dep}/seed_{ms}/validation_selected_model.pt')}


def get(root, dep, ms, es, split):
    f = pd.read_parquet(e.BASE / f'evaluation/{dep}/seed_{es}/{split}_pairs.parquet')
    a = expanded.prediction(root, dep, ms, es, split)
    return f, pair_values(a['p'], a['z'], f.idx_i.to_numpy(int), f.idx_j.to_numpy(int))


def score(f, x, spec):
    penalty = np.minimum(np.log(tail.tail_probability(x['mass'], spec['mass_reference'])/.05), 0.)
    raw = x['cosine']
    inc = np.interp(raw, spec['knots'], spec['loglr'])
    ood = (raw < spec['fit_min']) | (raw > spec['fit_max'])
    inc = np.clip(np.where(ood, np.minimum(inc, 0.), inc), -4, 4)
    return f.waveform_score.to_numpy(float)+spec['gamma']*penalty+spec['beta']*inc, penalty, inc, ood


def run(root):
    trial = root / 'trials/EXPANDED-INDEPENDENT-REFERENCE-GUARD'
    if trial.exists():
        raise RuntimeError('Independent trial required')
    for name in ('contracts', 'calibration', 'tables', 'evaluation', 'results'):
        (trial / name).mkdir(parents=True)
    shutil.copy2(__file__, trial / 'contracts' / Path(__file__).name)
    dev.json_write(trial / 'contracts/RECIPE.json', {
        'created_utc': datetime.now(timezone.utc).isoformat(), 'same_O3_O4a': True,
        'formula': 'Z_FRT+gamma*min(log(p_new_Mc_tail/0.05),0)+beta*bounded_calibrated_new_embedding_cosine',
        'mass_reference': '512 new independent development-validation sources,one true companion pair per source; original FRT and its older mass guard retained',
        'motivation': 'do not discard a working older waveform safeguard when updating the encoder; estimate the new finite reference tail from a larger independent source population',
        'calibration_scope': 'newdata are coverage-balanced waveforms with32shared validation noise blocks,NOT512independent noises or an astrophysical population; domain-shift coverage is not assumed',
        'gammas': GAMMAS, 'betas': BETAS,
        'selection': 'existing BAYESTAR validation only, original objective and unchanged FRT-relative guards',
        'interpretation': 'three underlying waveform models per run/seed:oldC-fixed,RNC-FRT,and expanded-dataRNC; correlated ranking evidence,not independentBFs or PE posteriors',
        'frozen': ['time', 'sky', 'outerweights', 'scope', 'all historical results'],
        'no_real_PE_or_official_features': True, 'fresh_confirmation_required': True})
    selected, cache, states = {}, {}, []
    for dep in e.DEPS:
        for ms, es in zip(body.MODEL_SEEDS, dev.SEEDS):
            cal = fit(root, dep, ms)
            f, x = get(root, dep, ms, es, 'validation')
            cache[dep, es] = f, x
            bm = ev.metrics(f, f.waveform_score.to_numpy(float), dep, es)
            rows, choices = [], []
            for gamma in GAMMAS:
                for beta in BETAS:
                    spec = {**cal, 'gamma': gamma, 'beta': beta}
                    z, *_ = score(f, x, spec)
                    mm = ev.metrics(f, z, dep, es)
                    ok = ev.guard(mm, bm)
                    rows.append({'gamma': gamma, 'beta': beta, 'pass': ok,
                        **{a+'_'+k: v for a, b in mm.items() for k, v in b.items()}})
                    if ok:
                        choices.append((e.old_selection.objective(mm, gamma, beta), spec))
            out = trial / f'calibration/{dep}/seed_{es}'
            out.mkdir(parents=True)
            dev.csv_write(out / 'validation_grid.csv', pd.DataFrame(rows))
            spec = min(choices, key=lambda x: x[0])[1]
            selected[dep, es] = spec
            dev.json_write(out / 'SELECTED_CONFIG.json', spec)
            states.append({'deployment': dep, 'seed': es, 'gamma': spec['gamma'], 'beta': spec['beta']})
    dev.json_write(trial / 'contracts/SELECTED_FREEZE.json', {'configs': states, 'real_used_to_select': False})
    print(json.dumps({'expanded_guard_selection': states}), flush=True)
    rows, guards = [], []
    for dep in e.DEPS:
        real = {}
        for ms, es in zip(body.MODEL_SEEDS, dev.SEEDS):
            for split in ('validation', 'test', 'real'):
                f, x = cache[dep, es] if split == 'validation' else get(root, dep, ms, es, split)
                z, penalty, inc, ood = score(f, x, selected[dep, es])
                n = f.copy()
                n['FRT_baseline_waveform_score'] = f.waveform_score
                n['waveform_score'], n['new_mass_guard'], n['new_cosine_increment'], n['new_cosine_ood'] = z, penalty, inc, ood
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
