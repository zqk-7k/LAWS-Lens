#!/usr/bin/env python3
"""Independent waveform-only extension of the frozen RNC-FRT baseline."""
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
from scipy.special import softmax
from sklearn.isotonic import IsotonicRegression
import mcwf_development_20260905 as dev
import mcwf_encoder_body_20260906 as body
import mcwf_encoder_evaluate_20260906 as ev
import mcwf_unified_waveform_20260906 as old_selection

BASE = dev.PROJECT / 'results/mcwf_unified_rankncontrast_finite_tail_20260907'
TRAINED = dev.PROJECT / 'results/mcwf_unified_rankncontrast_encoder_20260907'
DEPS = ('gwtc3', 'gwtc4')
BETAS = (0., .0625, .125, .25, .5, 1., 2., 4.)
RECIPES = ('REF-RAW', 'REF-ISO', 'COS-ISO')


def predictions(dep, ms, es, split):
    return np.load(TRAINED / f'cache/predictions/RAW-PHASE-SOURCE/{dep}/model_{ms}_eval_{es}/{split}.npz')


def reference_prior(dep):
    meta = pd.read_parquet(TRAINED / f'cache/{dep}/train_metadata.parquet')
    target = np.load(body.PREVIOUS / f'cache/masstf/{dep}/train_targets.npy')
    groups = meta.waveform_parent_uid.to_numpy()
    prior = np.stack([target[groups == g].mean(0) for g in np.unique(groups)]).mean(0)
    prior = np.maximum(prior, 1e-6)
    return prior / prior.sum()


def features(p, z, i, j, prior):
    p = np.asarray(p, float)
    p /= p.sum(1, keepdims=True)
    z = np.asarray(z, float)
    z /= np.linalg.norm(z, axis=1, keepdims=True)
    return {
        'REF': np.log(np.maximum(np.sum(p[i] * p[j] / prior, axis=1), 1e-30)),
        'COS': np.sum(z[i] * z[j], axis=1),
    }


def frame_features(dep, ms, es, split, prior):
    f = pd.read_parquet(BASE / f'evaluation/{dep}/seed_{es}/{split}_pairs.parquet')
    a = predictions(dep, ms, es, split)
    x = features(a['p'], a['z'], f.idx_i.to_numpy(int), f.idx_j.to_numpy(int), prior)
    return f, x


def fit_reference(dep, ms, recipe, prior):
    a = np.load(TRAINED / f'models/RAW-PHASE-SOURCE/{dep}/seed_{ms}/development_validation.npz')
    tg = pd.read_csv(TRAINED / f'models/RAW-PHASE-SOURCE/{dep}/seed_{ms}/temperature_grid.csv')
    temperature = min(tg.to_dict('records'), key=lambda r: (r['ce'], abs(r['temperature'] - 1)))['temperature']
    p = softmax(a['logits'].astype(float) / temperature, axis=1)
    g = a['group']
    i, j = np.triu_indices(len(p), 1)
    y = g[i] == g[j]
    x = features(p, a['embedding'], i, j, prior)[recipe.split('-')[0]]
    spec = {'recipe': recipe, 'prior': prior.tolist(), 'temperature': temperature, 'fit_positive_systems': int(y.sum()),
            'fit_negative_pairs': int((~y).sum()), 'fit_min': float(x.min()), 'fit_max': float(x.max()),
            'fit_source': str(TRAINED / f'models/RAW-PHASE-SOURCE/{dep}/seed_{ms}/development_validation.npz')}
    if recipe.endswith('ISO'):
        # Equal total class weight removes the arbitrary generated pair prevalence.
        weight = np.where(y, .5 / y.sum(), .5 / (~y).sum())
        iso = IsotonicRegression(increasing=True, out_of_bounds='clip').fit(x, y.astype(float), sample_weight=weight)
        floor = 1. / (y.sum() + 2.)
        prob = np.clip(iso.y_thresholds_, floor, 1 - floor)
        spec.update(knots=iso.X_thresholds_.tolist(), loglr=(np.log(prob) - np.log1p(-prob)).tolist(), floor=floor)
    return spec


def increment(x, spec):
    raw = np.asarray(x[spec['recipe'].split('-')[0]], float)
    calibrated = np.interp(raw, spec['knots'], spec['loglr']) if 'knots' in spec else raw
    ood = (raw < spec['fit_min']) | (raw > spec['fit_max'])
    # Do not extrapolate a positive reward beyond finite simulation support.
    value = np.where(ood, np.minimum(calibrated, 0.), calibrated)
    return np.clip(value, -4., 4.), raw, ood


def score(f, x, spec):
    inc, raw, ood = increment(x, spec)
    return f.waveform_score.to_numpy(float) + spec['beta'] * inc, inc, raw, ood


def external_reference(dep):
    p = BASE / f'external_development_audit_v3/tables/{dep}_UNIFIED_C_fixed_all_pairs.csv'
    f = pd.read_csv(p)
    cols = ['pair_key', 'event_i', 'event_j'] + [c for c in f if c.startswith('pe_') or c.startswith('official_') or c == 'public_hanabi_table_overlap']
    return f[cols]


def initialize(root):
    if root.exists():
        raise RuntimeError('Independent empty output required')
    for name in ('contracts', 'scripts', 'trials', 'tables', 'audit', 'logs', 'manifest', 'reports', 'figures'):
        (root / name).mkdir(parents=True)
    budgets = pd.concat([pd.read_csv(BASE / f'results/UNIFIED/{d}/pe_official_budget.csv') for d in DEPS], ignore_index=True)
    dev.csv_write(root / 'contracts/BASELINE_BUDGETS.csv', budgets)
    contract = {
        'code': 'MCWF-PE-FRONTEND-EXTENSION', 'created_utc': datetime.now(timezone.utc).isoformat(),
        'baseline': str(BASE), 'archive_sha256': '241045c428091b1869c474f438c1c67ccca9f9eab674ab15d142216333243fe9',
        'scope': {'gwtc3': {'events': 62, 'pairs': 1891}, 'gwtc4': {'events': 74, 'pairs': 2701}},
        'frozen': ['time', 'sky', 'C-fixed outer weights', 'event scope', 'old and RNC-FRT checkpoints', 'historical outputs'],
        'same_across_runs': ['2s H1L1 input', 'feature recipe', 'calibration procedure', 'beta grid', 'validation selection order', 'PE and official acceptance criteria'],
        'allowed_run_differences': ['frozen checkpoints', 'simulation-fitted reference', 'validation-selected beta'],
        'recipes': RECIPES, 'beta_grid': BETAS, 'increment_cap': [-4., 4.],
        'fit': 'independent source-weighted training mass reference; 240-source encoder-development validation for incremental calibration, no real PE or official labels',
        'tune': 'existing BAYESTAR simulation validation; original deterministic F50/F90/AP/R10 priority relative to CURRENT FRT, not the older C-fixed baseline',
        'negative_and_positive_increment': 'retain current FRT; test predictive mass specificity or RNC geometry as one waveform increment; this is not an independent physical Bayes factor',
        'guardrails': {'R10_drop_max': .02, 'AP_drop_max': .005, 'F50_F90_ratio_max': 1.1},
        'minimum_real_development_target': {
            'budgets': [10, 20], 'methods': ['C_fixed'], 'each_run_separately': True,
            'catastrophic_mc_count': 'no increase', 'BC_mc_ge_0p5_count': 'no decrease', 'Dmax_le_3_count': 'no decrease',
            'median_BC_mc_drop_max': .02, 'official_frontend_increase_min': 1, 'official_hanabi_increase_min': 1,
            'waveform_only_Top10': 'zero catastrophic Mc, BC and Dmax counts not below current FRT'},
        'higher_target': 'minimum target plus at least one PE count improves per run, with no PE count deteriorating',
        'interpretation': 'real PE and official overlap explicitly used as adaptive development feedback, not blinded validation; official candidates are NOT true lens labels',
        'forbidden': ['event IDs as scoring features', 'PE or official labels as model inputs', 'hand boosting pairs', 'editing historical scores', 'selectively reporting seeds'],
        'fresh_confirmation': 'a NEW independent source/noise injection set after final selection; previous confirmation is reused development only',
        'final_status': 'HOLD_FOR_AUTHOR_REVIEW_NO_ADOPTION_NO_OVERWRITE',
        'references': ['https://arxiv.org/abs/1807.07062', 'https://arxiv.org/abs/2210.01189'],
        'reference_limit': 'references motivate posterior overlap and ordered embeddings, not performance or chosen engineering tolerances',
    }
    dev.json_write(root / 'contracts/EXPERIMENT_CONTRACT.json', contract)
    protected = []
    for folder in (BASE / 'contracts', BASE / 'calibration', BASE / 'evaluation', BASE / 'results', BASE / 'external_development_audit_v3/tables'):
        for p in sorted(folder.rglob('*')):
            if p.is_file():
                protected.append({'path': str(p), 'sha256': dev.sha(p), 'bytes': p.stat().st_size})
    for p in sorted(TRAINED.glob('models/*/*/*/validation_selected_model.pt')):
        protected.append({'path': str(p), 'sha256': dev.sha(p), 'bytes': p.stat().st_size})
    dev.csv_write(root / 'manifest/PROTECTED_INPUT_SHA256.csv', pd.DataFrame(protected).drop_duplicates('path'))
    shutil.copy2(__file__, root / 'scripts' / Path(__file__).name)
    for dep in DEPS:
        external_reference(dep).to_parquet(root / f'audit/{dep}_frozen_external_reference.parquet', index=False)
    print(json.dumps({'initialized': str(root), 'protected': len(protected)}), flush=True)


def assess(root, trial):
    base = pd.read_csv(root / 'contracts/BASELINE_BUDGETS.csv')
    new = pd.concat([pd.read_csv(trial / f'results/CANDIDATE/{d}/pe_official_budget.csv') for d in DEPS], ignore_index=True)
    checks = []
    for dep in DEPS:
        for b in (10, 20):
            cond = (base.deployment == dep) & (base.method == 'C_fixed') & (base.seed.astype(str) == 'consensus') & (base.budget == b)
            bc = base[cond].iloc[0]
            nc = new[(new.deployment == dep) & (new.method == 'C_fixed') & (new.seed.astype(str) == 'consensus') & (new.budget == b)].iloc[0]
            pe = nc.catastrophic_mc <= bc.catastrophic_mc and nc.BC_mc_ge_0p5 >= bc.BC_mc_ge_0p5 and nc.Dmax_le_3 >= bc.Dmax_le_3 and nc.median_BC_mc >= bc.median_BC_mc - .02
            official = nc.official_frontend >= bc.official_frontend + 1 and nc.official_hanabi >= bc.official_hanabi + 1
            checks.append({'deployment': dep, 'budget': b, 'PE_pass': bool(pe), 'official_pass': bool(official),
                           **{str(c): int(nc[c] - bc[c]) for c in ('catastrophic_mc', 'BC_mc_ge_0p5', 'Dmax_le_3', 'official_frontend', 'official_hanabi')}})
    wave = []
    for dep in DEPS:
        b = base[(base.deployment == dep) & (base.method == 'waveform_only') & (base.seed.astype(str) == 'consensus') & (base.budget == 10)].iloc[0]
        n = new[(new.deployment == dep) & (new.method == 'waveform_only') & (new.seed.astype(str) == 'consensus') & (new.budget == 10)].iloc[0]
        wave.append({'deployment': dep, 'pass': bool(n.catastrophic_mc == 0 and n.BC_mc_ge_0p5 >= b.BC_mc_ge_0p5 and n.Dmax_le_3 >= b.Dmax_le_3)})
    g = pd.read_csv(trial / 'tables/REUSED_GUARDRAILS.csv')
    result = {'minimum_target_pass': bool(all(x['PE_pass'] and x['official_pass'] for x in checks) and all(x['pass'] for x in wave) and g['pass'].all()),
              'real_development': checks, 'waveform_PE': wave, 'reused_injection_pass': bool(g['pass'].all()), 'fresh_confirmation_done': False}
    dev.json_write(trial / 'contracts/TARGET_AUDIT.json', result)
    dev.csv_write(trial / 'tables/PE_OFFICIAL_ALL.csv', new)
    print(json.dumps({'trial': trial.name, **result}), flush=True)


def run_trial(root, recipe, trial_id=None):
    trial = root / 'trials' / (trial_id or recipe)
    if trial.exists():
        raise RuntimeError('Do not overwrite a trial')
    for name in ('contracts', 'calibration', 'evaluation', 'tables', 'results'):
        (trial / name).mkdir(parents=True)
    shutil.copy2(__file__, trial / 'contracts' / Path(__file__).name)
    selected, priors = {}, {d: reference_prior(d) for d in DEPS}
    status = []
    for dep in DEPS:
        for ms, es in zip(body.MODEL_SEEDS, dev.SEEDS):
            spec = fit_reference(dep, ms, recipe, priors[dep])
            f, x = frame_features(dep, ms, es, 'validation', priors[dep])
            baseline = ev.metrics(f, f.waveform_score.to_numpy(float), dep, es)
            rows, options = [], []
            for beta in BETAS:
                cfg = {**spec, 'beta': beta, 'model_seed': ms, 'eval_seed': es, 'deployment': dep}
                z, inc, _, ood = score(f, x, cfg)
                m = ev.metrics(f, z, dep, es)
                ok = ev.guard(m, baseline)
                rows.append({'beta': beta, 'pass': bool(ok), 'ood_rate': float(ood.mean()),
                             **{method + '_' + k: v for method, mm in m.items() for k, v in mm.items()}})
                if ok:
                    options.append((old_selection.objective(m, beta, 1.), cfg))
            folder = trial / f'calibration/{dep}/seed_{es}'
            folder.mkdir(parents=True)
            dev.csv_write(folder / 'validation_grid.csv', pd.DataFrame(rows))
            choice = min(options, key=lambda p: p[0])[1]
            dev.json_write(folder / 'SELECTED_CONFIG.json', choice)
            selected[dep, es] = choice
            status.append({'deployment': dep, 'seed': es, 'beta': choice['beta'], 'config_sha256': dev.sha(folder / 'SELECTED_CONFIG.json')})
    dev.json_write(trial / 'contracts/SELECTED_FREEZE.json', {'configs': status, 'opened_test_or_real_during_selection': False})
    print(json.dumps({'trial_frozen': recipe, 'selected': status}), flush=True)
    metrics, guards = [], []
    for dep in DEPS:
        real = {}
        for ms, es in zip(body.MODEL_SEEDS, dev.SEEDS):
            for split in ('validation', 'test', 'real'):
                f, x = frame_features(dep, ms, es, split, priors[dep])
                z, inc, raw, ood = score(f, x, selected[dep, es])
                changed = f.copy()
                changed['FRT_baseline_waveform_score'] = f.waveform_score
                changed['waveform_score'] = z
                changed['extension_increment'] = inc
                changed['extension_raw'] = raw
                changed['extension_ood'] = ood
                for col in ('time_score', 'sky_raw_log_bf', 'idx_i', 'idx_j'):
                    assert np.array_equal(f[col], changed[col])
                out = trial / f'evaluation/{dep}/seed_{es}'
                out.mkdir(parents=True, exist_ok=True)
                changed.to_parquet(out / f'{split}_pairs.parquet', index=False)
                if split == 'real':
                    real[es] = changed
                else:
                    baseline = ev.metrics(f, f.waveform_score.to_numpy(float), dep, es)
                    candidate = ev.metrics(f, z, dep, es)
                    guards.append({'deployment': dep, 'seed': es, 'split': split, 'pass': ev.guard(candidate, baseline)})
                    for method in baseline:
                        for config, m in (('FRT_BASELINE', baseline[method]), ('CANDIDATE', candidate[method])):
                            metrics.append({'deployment': dep, 'seed': es, 'split': split, 'method': method, 'config': config, **m})
        pe = pd.read_parquet(root / f'audit/{dep}_frozen_external_reference.parquet')
        dev.save_evaluation(trial, 'CANDIDATE', dep, real, pe)
    dev.csv_write(trial / 'tables/RETRIEVAL_PER_SEED.csv', pd.DataFrame(metrics))
    dev.csv_write(trial / 'tables/REUSED_GUARDRAILS.csv', pd.DataFrame(guards))
    assess(root, trial)


def tests():
    prior = np.array([.2, .3, .5])
    p = np.array([prior, [.1, .8, .1], [.7, .2, .1]])
    z = np.eye(3)
    f = features(p, z, np.array([0, 1]), np.array([1, 2]), prior)
    assert abs(f['REF'][0]) < 1e-12
    rev = features(p, z, np.array([1, 2]), np.array([0, 1]), prior)
    assert np.allclose(f['REF'], rev['REF'])
    assert np.allclose(f['COS'], rev['COS'])
    print('PASS: neutral reference, pair symmetry, finite normalized probabilities', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path)
    p.add_argument('--phase', choices=('init', 'run', 'test'), required=True)
    p.add_argument('--recipe', choices=RECIPES)
    p.add_argument('--trial-id')
    args = p.parse_args()
    if args.phase == 'init': initialize(args.root)
    elif args.phase == 'test': tests()
    else: run_trial(args.root, args.recipe, args.trial_id)
