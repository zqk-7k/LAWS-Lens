#!/usr/bin/env python3
"""Independent mass discrimination plus source-morphology agreement."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='2'
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import mcwf_pe_frontend_extension_v2_20260907 as e
import mcwf_ordered_mass_evaluate_20260907 as mass
import mcwf_large_population_rnc_evaluate_20260907 as large
import mcwf_training_recipe_ensemble_evaluate_20260907 as ensemble
import mcwf_finite_reference_tail_20260907 as tail

dev,body,ev=e.dev,e.body,e.ev
MODE='large'
GAMMAS=large.GAMMAS
BETAS=large.BETAS


def calibration(root,dep,ms):
    component=large if MODE=='large' else ensemble
    c=component.calibration(root,dep,ms)
    m=mass.calibrate(root,dep,ms)
    c['mass_reference']=m['mass_reference']
    c['mass_model']='ordered_mass_predictor'
    c['source_model']=MODE
    return c


def values(root,dep,ms,es,split):
    f,m=mass.values(root,dep,ms,es,split)
    component=large if MODE=='large' else ensemble
    rf,x=component.values(root,dep,ms,es,split)
    for k in ('idx_i','idx_j'):
        if not np.array_equal(f[k],rf[k]):
            raise RuntimeError('Component event indexing mismatch')
    return f,{'bc':m['bc'],'cosine':x['cosine'],'mass_boundary_ood':m['ood']}


def score(f,x,spec):
    p=tail.tail_probability(-np.log(x['bc']),spec['mass_reference'])
    penalty=np.minimum(np.log(p/.05),0.)
    increment=np.interp(x['cosine'],spec['knots'],spec['loglr']).clip(-4.,4.)
    ood=x['mass_boundary_ood']|(x['cosine']<spec['minimum'])|(x['cosine']>spec['maximum'])
    increment=np.where(ood|(p<.05),np.minimum(increment,0.),increment)
    penalty=np.where(x['mass_boundary_ood'],0.,penalty)
    increment=np.where(x['mass_boundary_ood'],0.,increment)
    updated=spec['gamma']*penalty+spec['beta']*increment
    z=f.waveform_score.to_numpy(float)+updated
    if spec.get('replace_mass_guard',False):
        replacement=f.previous_waveform_score.to_numpy(float)+updated
        z=(1-spec['blend'])*f.waveform_score.to_numpy(float)+spec['blend']*replacement
    return z,penalty,increment,ood


def operating_points(arm):
    if arm == 'REPLACE-MASS-GUARD':
        yield {'gamma': 0., 'beta': 0., 'blend': 0., 'replace_mass_guard': True}
        for blend in (.25, .5, .75, 1.):
            for gamma in (.25, .5, 1., 2., 4.):
                for beta in (0., .125, .25, .5, 1.):
                    yield {'gamma': gamma, 'beta': beta, 'blend': blend, 'replace_mass_guard': True}
    else:
        for gamma in GAMMAS:
            for beta in ((0.,) if arm == 'TAIL' else BETAS):
                yield {'gamma': gamma, 'beta': beta, 'blend': 0., 'replace_mass_guard': False}


def run(root, arm):
    trial = root / f'trials/ORDERED-ANCHOR-RNC-{MODE.upper()}-{arm}'
    if trial.exists():
        raise RuntimeError('Independent trial required')
    for name in ('contracts', 'calibration', 'tables', 'evaluation', 'results'):
        (trial / name).mkdir(parents=True)
    shutil.copy2(__file__, trial / 'contracts' / Path(__file__).name)
    dev.json_write(trial / 'contracts/RECIPE.json', {
        'created_utc': datetime.now(timezone.utc).isoformat(), 'same_O3_O4a': True,
        'arm': arm, 'population': '12288 waveform-source parents,160 training noise blocks,unchanged512 development parents/32 noise blocks',
        'model': '253-position ordered mass-axis CNN plus frozen larger-population source RNC or uniform three-recipe kernel ensemble;no further model training',
        'source_kernel_mode': MODE,
        'mass_contract_sha256': dev.sha(root / 'ordered_mass_predictor/contracts/TRAINING.json'),
        'OOD': 'mass-posterior endpoint occupancy above25percent makes new terms neutral;this is not a fitted outside-support probability',
        'formula': 'increment arms: Z_FRT+gamma*finite_mass_tail+beta*bounded_isotonic_new_cosine; REPLACE-MASS-GUARD: (1-blend)*Z_FRT+blend*(Z_original_Cfixed+gamma*new_mass_tail+beta*new_cosine)',
        'gamma_grid': GAMMAS, 'beta_grid': (0.,) if arm == 'TAIL' else BETAS,
        'exact_operating_points': list(operating_points(arm)),
        'positive_support': 'new cosine inside development support and new mass finite-tail probability>=0.05',
        'interpretation': 'correlated discriminative waveform evidence,not fullPE or properBF; new mass tail is not FPP',
        'selection': 'same BAYESTAR validation-only objective and original per-seed guards',
        'frozen': ['time', 'sky', 'outerweights', 'strictscope', 'historicaloutputs'],
        'real_used_to_select': False, 'PE_official_ID_inputs': False,
        'real_audit_is_adaptive_development': True, 'fresh_confirmation_required': True})
    selected, cache, states = {}, {}, []
    for dep in e.DEPS:
        for ms, es in zip(body.MODEL_SEEDS, dev.SEEDS):
            cal = calibration(root, dep, ms)
            f, x = values(root, dep, ms, es, 'validation')
            cache[dep, es] = f, x
            bm = ev.metrics(f, f.waveform_score.to_numpy(float), dep, es)
            rows, choices = [], []
            for point in operating_points(arm):
                spec = {**cal, **point}
                z, _, _, _ = score(f, x, spec)
                mm = ev.metrics(f, z, dep, es)
                ok = ev.guard(mm, bm)
                rows.append({**point, 'pass': ok,
                             **{a + '_' + k: v for a, b in mm.items() for k, v in b.items()}})
                if ok:
                    choices.append((tuple(e.old_selection.objective(mm, point['gamma'], point['beta'])) + (point['blend'],), spec))
            spec = min(choices, key=lambda a: a[0])[1]
            selected[dep, es] = spec
            out = trial / f'calibration/{dep}/seed_{es}'
            out.mkdir(parents=True)
            dev.csv_write(out / 'validation_grid.csv', pd.DataFrame(rows))
            dev.json_write(out / 'SELECTED_CONFIG.json', spec)
            states.append({'deployment': dep, 'seed': es, 'gamma': spec['gamma'], 'beta': spec['beta'], 'blend': spec['blend']})
    dev.json_write(trial / 'contracts/SELECTED_FREEZE.json', {'configs': states, 'real_used_to_select': False})
    print(json.dumps({'large_population_rnc_selected': arm, 'configs': states}), flush=True)
    rows, guards = [], []
    for dep in e.DEPS:
        real = {}
        for ms, es in zip(body.MODEL_SEEDS, dev.SEEDS):
            for split in ('validation', 'test', 'real'):
                f, x = cache[dep, es] if split == 'validation' else values(root, dep, ms, es, split)
                z, penalty, inc, ood = score(f, x, selected[dep, es])
                n = f.rename(columns={k: 'baseline_' + k for k in (
                    'final_score', 'rank', 'method', 'waveform_contribution', 'time_contribution', 'sky_contribution') if k in f}).copy()
                n['FRT_baseline_waveform_score'] = f.waveform_score
                n['waveform_score'], n['new_mass_tail_penalty'], n['new_cosine_evidence'] = z, penalty, inc
                n['new_cosine_ood'] = ood
                for key, value in x.items():
                    n['ordered_anchor_' + key] = value
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    for mode in ('large', 'ensemble'):
        MODE = mode
        for arm in ('MASS-COSINE', 'REPLACE-MASS-GUARD'):
            run(args.root, arm)
