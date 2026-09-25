#!/usr/bin/env python3
"""Concordance-gated dense waveform evidence; no time/sky or PE changes."""
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
import mcwf_pe_frontend_extension_v2_20260907 as e
import mcwf_finite_reference_tail_20260907 as tail

dev, body, ev = e.dev, e.body, e.ev
BETAS = (0., .0625, .125, .25, .5, 1., 2., 4.)
POWERS = (1., 2., 4.)


def get(root, dep, ms, es, split, kind):
    f = pd.read_parquet(e.BASE / f'evaluation/{dep}/seed_{es}/{split}_pairs.parquet')
    d = pd.read_parquet(root / f'trials/DENSE-PHYSICAL-INTRINSIC-PROFILE/evaluation/{dep}/seed_{es}/{split}_pairs.parquet')
    assert np.array_equal(f[['idx_i', 'idx_j']].to_numpy(), d[['idx_i', 'idx_j']].to_numpy())
    p = e.predictions(dep, ms, es, split)['p']
    i, j = f.idx_i.to_numpy(int), f.idx_j.to_numpy(int)
    return f, {'profile': d['dense_profile_'+kind].to_numpy(float),
        'new_mass': -np.log(np.sum(np.sqrt(p[i]*p[j]), 1).clip(1e-12)),
        'old_mass': f.waveform_abs_delta_logmc_std.to_numpy(float)}


def score(f, x, spec):
    raw = x['profile']
    value = np.interp(raw, spec['knots'], spec['loglr'])
    ood = (raw < spec['fit_min']) | (raw > spec['fit_max'])
    value = np.clip(np.where(ood, np.minimum(value, 0.), value), -4, 4)
    old = tail.tail_probability(x['old_mass'], spec['old_mass_reference'])
    new = tail.tail_probability(x['new_mass'], spec['new_mass_reference'])
    gate = np.minimum(old, new)**spec['power']
    increment = np.minimum(value, 0.) + np.maximum(value, 0.)*gate
    return f.waveform_score.to_numpy(float)+spec['beta']*increment, increment, gate, ood


def run(root, kind):
    trial = root / 'trials' / ('DENSE-CONCORDANT-FULL' if kind == 'full' else 'DENSE-CONCORDANT-CONDITIONAL')
    if trial.exists():
        raise RuntimeError('Independent trial required')
    for name in ('contracts', 'calibration', 'tables', 'evaluation', 'results'):
        (trial / name).mkdir(parents=True)
    shutil.copy2(__file__, trial / 'contracts' / Path(__file__).name)
    dev.json_write(trial / 'contracts/RECIPE.json', {
        'created_utc': datetime.now(timezone.utc).isoformat(), 'same_kind_all_O3_O4a_and_seeds': kind,
        'formula': 'Z_FRT+beta*(min(profileLLR,0)+max(profileLLR,0)*min(p_old_mass_tail,p_RNC_mass_tail)^power)',
        'motivation': 'do not grant full positive intrinsic-profile reward when independent old/new waveform mass predictions disagree across the pair',
        'tail_reference': 'one true pair per BAYESTAR validation source; nonconformity is old predicted logMc gap or new RNC massBC gap',
        'p_tail_is_not': 'not a PE posterior,lens probability,FPP,or proven conformal coverage under domain shift',
        'profile': 'same3584 waveform templates; simulation-calibrated bounded[-4,4] profile,positiveOOD0',
        'powers': POWERS, 'beta_grid': BETAS,
        'selection': 'BAYESTAR validation only, original deterministic objective, FRT-relative guards',
        'frozen': ['time', 'sky', 'outerweights', 'scope', 'encoders', 'historical outputs'],
        'real_PE_and_official': 'post-selection adaptive-development audit only'})
    chosen, states, cache = {}, [], {}
    for dep in e.DEPS:
        profile = json.loads((root / f'dense_profile/{dep}/FIT.json').read_text())[kind]
        for ms, es in zip(body.MODEL_SEEDS, dev.SEEDS):
            f, x = get(root, dep, ms, es, 'validation', kind)
            cache[dep, es] = f, x
            y = f.is_true_pair.to_numpy(bool)
            cal = {**profile, 'old_mass_reference': np.sort(x['old_mass'][y]).tolist(),
                'new_mass_reference': np.sort(x['new_mass'][y]).tolist()}
            bm = ev.metrics(f, f.waveform_score.to_numpy(float), dep, es)
            rows, choices = [], []
            for power in POWERS:
                for beta in BETAS:
                    spec = {**cal, 'power': power, 'beta': beta}
                    z, _, gate, ood = score(f, x, spec)
                    mm = ev.metrics(f, z, dep, es)
                    ok = ev.guard(mm, bm)
                    rows.append({'power': power, 'beta': beta, 'pass': ok, 'mean_gate': float(gate.mean()),
                        **{a+'_'+k: v for a, b in mm.items() for k, v in b.items()}})
                    if ok:
                        choices.append((e.old_selection.objective(mm, beta, power), spec))
            out = trial / f'calibration/{dep}/seed_{es}'
            out.mkdir(parents=True)
            dev.csv_write(out / 'validation_grid.csv', pd.DataFrame(rows))
            spec = min(choices, key=lambda x: x[0])[1]
            chosen[dep, es] = spec
            dev.json_write(out / 'SELECTED_CONFIG.json', spec)
            states.append({'deployment': dep, 'seed': es, 'kind': kind, 'power': spec['power'], 'beta': spec['beta']})
    dev.json_write(trial / 'contracts/SELECTED_FREEZE.json', {'configs': states, 'real_used_to_select': False})
    print(json.dumps({'concordant_dense_selection': states}), flush=True)
    rows, guards = [], []
    for dep in e.DEPS:
        real = {}
        for ms, es in zip(body.MODEL_SEEDS, dev.SEEDS):
            for split in ('validation', 'test', 'real'):
                f, x = cache[dep, es] if split == 'validation' else get(root, dep, ms, es, split, kind)
                z, inc, gate, ood = score(f, x, chosen[dep, es])
                n = f.copy()
                n['FRT_baseline_waveform_score'] = f.waveform_score
                n['waveform_score'], n['concordant_profile_increment'], n['mass_concordance_gate'], n['dense_profile_ood'] = z, inc, gate, ood
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
    p.add_argument('--kind', choices=['full', 'conditional_q_spin'], required=True)
    a = p.parse_args()
    run(a.root, a.kind)
