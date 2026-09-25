#!/usr/bin/env python3
"""Bounded removal of legacy mass scores and outer score blending."""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import resource
import shutil
import sys
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import spearmanr

P = Path('/root/autodl-tmp/gw-catalog')
sys.path.insert(0, str(P / 'scripts/experiments'))
import mcwf_temporal_response_evaluate_20260908 as ev

t, dev, cf = ev.t, ev.dev, ev.cf
ARCHIVE = P / 'results/mcwf_unified_path875_devconf_20260908T181500Z'
FRESH = P / 'results/mcwf_path25_fresh_confirmation_v2_20260908T172959Z'
JOINT = P / 'results/mcwf_joint_intrinsic_evidence_exploratory_20260908T155830Z'
OLD_CODE = 'MCWF-UNIFIED-PATH875-DEVCONF'
JOINT_CODE = 'JOINT-ETA-CHI-BC-ADD-CANDIDATE'
BASELINE = 'PATH875-ARCHIVED'
METHODS = ('NODUP-DIRECT', 'NODUP-JOINT-FIXED', 'NODUP-JOINT-VAL')
DEPS = ('gwtc3', 'gwtc4')
SEEDS = (202607241, 202607242, 202607243)
CATALOGS = (202609941, 202609942, 202609943)
REGULARIZATION = (0.01, 0.1, 1.0)
KEY_METRICS = ('macro_r_at_1', 'macro_r_at_5', 'macro_r_at_10',
               'average_precision', 'false_at_recall_0p5', 'false_at_recall_0p9')
STATUS = 'HOLD_FOR_AUTHOR_REVIEW_NO_ADOPTION_NO_OVERWRITE'


def utc():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def write_csv(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(value).to_csv(path, index=False, encoding='utf-8-sig')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def recipes():
    a = json.loads((FRESH / 'contracts/FRESH_FREEZE.json').read_text())
    return {(c['deployment'], c['seed']): c for c in a['recipes']}


def filename(split):
    return 'real_fusion_pairs.parquet' if split == 'real' else split + '_pairs.parquet'


def standard_paths(dep, seed, split):
    name = filename(split)
    return (cf.PAIRS / dep / f'seed_{seed}/{split}_pairs.parquet',
            JOINT / f'evaluation/{JOINT_CODE}/{dep}/seed_{seed}/{name}',
            ARCHIVE / f'development/evaluation/{OLD_CODE}/{dep}/seed_{seed}/{name}')


def align(frame, reference, real=False):
    keys = ['pair_key'] if real else ['idx_i', 'idx_j']
    if frame.duplicated(keys).any() or reference.duplicated(keys).any():
        raise RuntimeError('Duplicate pair identity')
    a, b = frame.set_index(keys), reference.set_index(keys)
    if len(a) != len(b) or set(a.index) != set(b.index):
        raise RuntimeError('Different pair scope')
    return a.loc[b.index].reset_index()


def load_panel(dep, seed, split, catalog=None):
    if catalog is None:
        old_path, joint_path, baseline_path = standard_paths(dep, seed, split)
        frame = pd.read_parquet(old_path)
        j = align(pd.read_parquet(joint_path), frame, split == 'real')
        baseline = align(pd.read_parquet(baseline_path), frame, split == 'real')
        aliases = {'joint_logbc': 'temporal_joint_statistic',
                   'joint_BC': 'temporal_joint_BC', 'joint_ood': 'temporal_joint_ood',
                   'joint_penalty': 'temporal_penalty', 'joint_increment': 'temporal_increment'}
    else:
        slot = recipes()[dep, seed]['slot']
        folder = FRESH / f'confirmation/{dep}/catalog_{catalog}/model_{slot}'
        frame = pd.read_parquet(folder / 'OMC_pairs.parquet')
        baseline = align(pd.read_parquet(folder / 'PATH25_pairs.parquet'), frame)
        j = baseline
        aliases = {'joint_logbc': 'joint_logBC', 'joint_BC': 'joint_BC',
                   'joint_ood': 'joint_ood', 'joint_penalty': 'joint_penalty',
                   'joint_increment': 'joint_increment'}
    for other in (j, baseline):
        for col in ('time_score', 'sky_raw_log_bf'):
            if not np.array_equal(frame[col].to_numpy(), other[col].to_numpy()):
                raise RuntimeError('Frozen physical score changed: ' + col)
    for target, source in aliases.items():
        frame[target] = j[source].to_numpy()
    frame['PATH875_waveform'] = baseline.waveform_score.to_numpy(float)
    frame['PATH875_final_score'] = baseline.final_score.to_numpy(float)
    frame['embedding_only'] = frame.waveform_embedding_cosine.to_numpy(float)
    finite = frame[['embedding_only', 'joint_logbc', 'joint_BC', 'joint_penalty',
                    'joint_increment', 'time_score', 'sky_raw_log_bf']].to_numpy(float)
    if not np.isfinite(finite).all():
        raise RuntimeError('Nonfinite required feature')
    if not np.allclose(np.exp(frame.joint_logbc), frame.joint_BC, rtol=1e-6, atol=1e-14):
        raise RuntimeError('Joint BC identity failed')
    return frame


def input_files():
    files = {FRESH / 'contracts/FRESH_FREEZE.json',
             ARCHIVE / 'contracts/DELIVERY_CONTRACT.json',
             ARCHIVE / 'tables/TOP_B_PE_OFFICIAL_COMPARISON.csv',
             JOINT / 'calibration/SELECTED.json',
             Path(cf.__file__), Path(dev.__file__), Path(dev.BASE.__file__)}
    for dep in DEPS:
        files.add(t.EXTERNAL / f'{dep}_external_reference.parquet')
        for seed in SEEDS:
            for split in ('validation', 'test', 'real'):
                files.update(standard_paths(dep, seed, split))
        for catalog in CATALOGS:
            folder = FRESH / f'confirmation/{dep}/catalog_{catalog}'
            files.add(folder / 'event_manifest.parquet')
            for c in recipes().values():
                if c['deployment'] == dep:
                    for name in ('OMC_pairs.parquet', 'PATH25_pairs.parquet'):
                        files.add(folder / f'model_{c["slot"]}/{name}')
    return sorted(files)


def initialize(root):
    if root.exists():
        raise RuntimeError('Output already exists; independent directory required')
    for name in ('contracts', 'configs', 'audit', 'results', 'tables', 'figures',
                 'reports', 'scripts', 'logs', 'manifest'):
        (root / name).mkdir(parents=True)
    contract = {
        'id': 'MCWF-NODUP-NOMIX-01', 'UTC': utc(), 'status': STATUS,
        'same_protocol_both_runs': True, 'seeds': list(SEEDS),
        'baseline': str(ARCHIVE), 'comparison': BASELINE,
        'methods': list(METHODS), 'primary_prespecified_method': 'NODUP-JOINT-VAL',
        'removed': ['old predicted Mc and q difference inputs and their composite calibration',
                    'RAW-PHASE mass tail', 'ordered-mass tail and calibrated increment',
                    'OMC anchor from candidate inference', 'old/new total-score mixing'],
        'retained': ['frozen original embedding cosine', 'frozen MULTIRATE mass prediction',
                     'frozen conditional eta/chi_eff head', 'single joint predictive BC',
                     'time_score', 'sky_raw_log_bf', 'strict scope', 'historical outputs'],
        'checkpoint_note': 'Auxiliary training heads remain in old checkpoint, but their outputs are not read by candidate scoring. An embedding can still encode physical mass information.',
        'direct': 'Zcos + frozen_gamma*T_joint + frozen_beta*I_joint. No old mass terms or total-score mixing. This is a deletion control, not independent evidence multiplication.',
        'joint': 'One jointly fitted nonnegative logistic score from (old embedding cosine, log joint predictive BC); replaces all old and new additive mass penalties/increments.',
        'calibration': 'Existing source/noise-disjoint fit/tune split of simulation validation; class-balanced, equal total weight per source-pair; nonnegative L2 logistic slopes; regularization chosen by tune log loss.',
        'regularization_grid': list(REGULARIZATION),
        'fit_minimum_positive_systems': 10,
        'OOD': 'Within fit box, cap abs(logit) at log(n_fit_positive_systems+1). COS box OOD is neutral. Joint box/previous joint-quality OOD falls back only to new COS calibration, never old mass score.',
        'fixed_weights': 'Pure upstream validation-selected normalized weights before PATH875, not PATH875 effective weights. No alpha in any candidate formula.',
        'retuned_weights': 'Positive simplex step .05, all channels >=.05; include exact pure-upstream weights. Tune simulation only. Priorities F50,F90,-AP,-R10,-R1; ties nearest pure-upstream weights then lexicographic.',
        'validation_guard': {'R10_drop_max': .02, 'AP_drop_max': .005, 'F50_F90_ratio_max': 1.1},
        'no_guard_pass': 'Keep best tune configuration marked FAIL for descriptive evaluation; do not restore old mass score or tune on test/real outcomes.',
        'real_acceptance': 'For EACH run and EACH Top10/20 budget: no decrease in Mc BC>=.5, median Mc BC, Dmax<=3, official 1pct or public Hanabi count; no increase in catastrophic Mc. Assess consensus and report every seed. No method selection using these outcomes.',
        'catastrophic': 'public Mc BC<.1 OR D_Mc>5',
        'real_scope': {'gwtc3_events': 62, 'gwtc3_pairs': 1891, 'gwtc4_events': 74, 'gwtc4_pairs': 2701},
        'test': 'Existing historical test and Sept8 fresh catalogs are now reused evaluation, not a new untouched test.',
        'selection_limit': 'Current round numbers use simulation fit/tune only; inherited neural models and chosen mechanism had real-development feedback. This ablation does not erase that history.',
        'neural_training': False, 'new_PE_or_Hanabi': False, 'ET3_or_paper_changes': False,
        'independent_evidence_claim': False, 'official_overlap_is_truth': False,
        'uncertainty': {'query_system_bootstrap': 10000, 'pair_source_bootstrap': 1000,
                        'null_pairs_dependent': True, 'real_PE_is_descriptive': True},
        'stop': 'One frozen bounded comparison; report FAIL as well as PASS. No adaptive retry to make PE/official counts pass.',
        'references': ['https://arxiv.org/abs/1506.02169',
                       'https://www.jmlr.org/papers/v11/cawley10a.html'],
        'reference_limits': 'Classifier-based density-ratio motivation and selection-bias caution, not proof these GW features/thresholds are optimal.'}
    write_json(root / 'contracts/ANALYSIS_CONTRACT.json', contract)
    shutil.copy2(__file__, root / 'scripts/nodup_experiment.py')
    protected = []
    for path in input_files():
        if not path.is_file():
            raise RuntimeError('Missing input: ' + str(path))
        protected.append({'path': str(path), 'sha256': sha(path), 'bytes': path.stat().st_size})
    # Preserve the inherited frozen-file list without duplicating large inputs.
    inherited = t.protected()
    present = {row['path'] for row in protected}
    protected.extend(row for row in inherited if row['path'] not in present)
    protected = pd.DataFrame(protected).drop_duplicates('path')
    for row in protected.itertuples():
        if sha(Path(row.path)) != row.sha256:
            raise RuntimeError('Input digest mismatch: ' + row.path)
    write_csv(root / 'manifest/INPUT_SHA256.csv', protected)
    inventory = []
    for dep in DEPS:
        for seed in SEEDS:
            plans = {s: dev.BASE.retained_event_plan(dep, seed, s) for s in ('validation', 'test')}
            a, b = plans['validation'], plans['test']
            if set(a.system_id) & set(b.system_id) or set(a.parent_noise_bank) & set(b.parent_noise_bank):
                raise RuntimeError('Inherited validation/test isolation failed')
            plan = cf.split_validation(a, seed)
            fit, tune = plan[plan.calibration_fold == 0], plan[plan.calibration_fold == 1]
            nfit = int((fit.groupby('system_id').size() == 2).sum())
            ntune = int((tune.groupby('system_id').size() == 2).sum())
            if min(nfit, ntune) < 10:
                raise RuntimeError('Insufficient calibration support')
            assert not set(fit.system_id) & set(tune.system_id)
            assert not set(fit.parent_noise_bank) & set(tune.parent_noise_bank)
            write_csv(root / f'audit/{dep}_{seed}_FIT_TUNE_PLAN.csv', plan)
            for split, p in plans.items():
                write_csv(root / f'audit/{dep}_{seed}_{split}_PLAN.csv', p)
            inventory.append({'deployment': dep, 'seed': seed, 'fit_positive_systems': nfit,
                              'tune_positive_systems': ntune, 'fit_events': len(fit), 'tune_events': len(tune),
                              'fit_tune_source_overlap': 0, 'fit_tune_noise_overlap': 0,
                              'val_test_source_overlap': 0, 'val_test_noise_overlap': 0,
                              'cross_noise_events_excluded': int((plan.calibration_fold == -1).sum())})
    write_csv(root / 'audit/SPLIT_SUPPORT.csv', inventory)
    write_json(root / 'contracts/START_FREEZE.json', {
        'UTC': utc(), 'contract_sha256': sha(root / 'contracts/ANALYSIS_CONTRACT.json'),
        'code_sha256': sha(root / 'scripts/nodup_experiment.py'), 'protected_files': len(protected)})
    print('INITIALIZED', root, flush=True)


def model_features(frame, family):
    cols = ['embedding_only'] if family == 'COS' else ['embedding_only', 'joint_logbc']
    return frame[cols].to_numpy(float)


def fit_model(frame, plan, family, reg):
    x, y = model_features(frame, family), frame.is_true_pair.to_numpy(float)
    weight = cf.balanced_weights(frame, plan)
    mu = np.sum(x * weight[:, None], axis=0)
    sd = np.maximum(np.sqrt(np.sum((x - mu)**2 * weight[:, None], axis=0)), 1e-6)
    z = (x - mu) / sd
    def loss(a):
        value = a[0] + z @ a[1:]
        objective = np.sum(weight * (np.logaddexp(0., value) - y * value))
        objective += .5 * reg * np.dot(a[1:], a[1:])
        residual = weight * (expit(value) - y)
        return objective, np.r_[residual.sum(), z.T @ residual + reg * a[1:]]
    result = minimize(loss, np.zeros(1 + x.shape[1]), jac=True, method='L-BFGS-B',
                      bounds=[(None, None)] + [(0., None)] * x.shape[1],
                      options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-9})
    if not result.success:
        raise RuntimeError('Calibration optimizer: ' + result.message)
    npos = plan.set_index('idx').loc[frame.loc[y.astype(bool), 'idx_i'], 'system_id'].nunique()
    return {'family': family, 'regularization': reg, 'mu': mu.tolist(), 'sd': sd.tolist(),
            'coef': result.x[1:].tolist(), 'intercept': float(result.x[0]),
            'minimum': x.min(0).tolist(), 'maximum': x.max(0).tolist(),
            'score_cap': float(np.log(npos + 1)), 'fit_positive_systems': int(npos),
            'optimizer_success': True, 'objective': float(result.fun)}


def apply_model(frame, spec, cosine_spec=None):
    x = model_features(frame, spec['family'])
    raw = spec['intercept'] + ((x - spec['mu']) / spec['sd']) @ np.asarray(spec['coef'])
    active = np.asarray(spec['coef']) > 1e-8
    ood = ((((x < np.asarray(spec['minimum']) - 1e-7) |
             (x > np.asarray(spec['maximum']) + 1e-7))) & active).any(1)
    if spec['family'] != 'COS':
        ood |= frame.joint_ood.to_numpy(bool)
    cap = spec['score_cap']
    score = raw.clip(-cap, cap)
    clipped = np.abs(raw) > cap
    if spec['family'] == 'COS':
        score[ood] = 0.
    else:
        if cosine_spec is None:
            raise RuntimeError('New cosine-only fallback required')
        fallback, _, _ = apply_model(frame, cosine_spec)
        score[ood] = fallback[ood]
    return score, ood, clipped & ~ood


def infer(frame, config):
    cosine, cosood, cosclip = apply_model(frame, config['cosine_calibration'])
    if config['method'] == 'NODUP-DIRECT':
        z = cosine + config['gamma'] * frame.joint_penalty.to_numpy(float)
        z += config['beta'] * frame.joint_increment.to_numpy(float)
        return z, frame.joint_ood.to_numpy(bool) | cosood, cosclip
    return apply_model(frame, config['joint_calibration'], config['cosine_calibration'])


def metric_key(row, anchor):
    m = row['metrics']
    return (m['false_at_recall_0p5'], m['false_at_recall_0p9'], -m['average_precision'],
            -m['macro_r_at_10'], -m['macro_r_at_1'],
            float(np.square(np.asarray(row['weights']) - anchor).sum()), *row['weights'])


def calibrate(root):
    freeze = root / 'contracts/CONFIGURATIONS_FROZEN.json'
    if freeze.exists():
        raise RuntimeError('Already frozen')
    choices, calibration_rows, grid_rows, tests = [], [], [], []
    for dep in DEPS:
        for seed in SEEDS:
            recipe = recipes()[dep, seed]
            full = load_panel(dep, seed, 'validation')
            plan = pd.read_csv(root / f'audit/{dep}_{seed}_FIT_TUNE_PLAN.csv')
            fit = cf.subset(full, plan[plan.calibration_fold == 0].idx)
            tune = cf.subset(full, plan[plan.calibration_fold == 1].idx)
            specs = {}
            for family in ('COS', 'JOINT'):
                options = []
                for reg in REGULARIZATION:
                    fit_frame = fit if family == 'COS' else fit[~fit.joint_ood].copy()
                    npos = int(fit_frame.is_true_pair.sum())
                    if npos < 10:
                        raise RuntimeError('Too few supported positive systems for joint calibration')
                    spec = fit_model(fit_frame, plan, family, reg)
                    value, ood, clipped = apply_model(tune, spec, specs.get('COS'))
                    w = cf.balanced_weights(tune, plan)
                    y = tune.is_true_pair.to_numpy(float)
                    objective = float(np.sum(w * (np.logaddexp(0., value) - y * value)))
                    calibration_rows.append({'deployment': dep, 'seed': seed, 'family': family,
                                             'regularization': reg, 'tune_logloss': objective,
                                             'tune_ood': float(ood.mean()), 'tune_clip': float(clipped.mean()),
                                             'coef': json.dumps(spec['coef']), 'fit_positive_systems': npos})
                    options.append(((objective, -reg), spec))
                specs[family] = min(options, key=lambda v: v[0])[1]
            anchor = np.asarray(recipe['upstream_weights'], float)
            anchor /= anchor.sum()
            base = cf.fast_metrics(tune, tune.PATH875_final_score.to_numpy(float))
            basewf = cf.fast_metrics(tune, tune.PATH875_waveform.to_numpy(float))
            common = {'deployment': dep, 'seed': seed, 'cosine_calibration': specs['COS'],
                      'joint_calibration': specs['JOINT'], 'weights': anchor.tolist(),
                      'gamma': recipe['joint_config']['gamma'], 'beta': recipe['joint_config']['beta']}
            for method in METHODS[:2]:
                c = {**common, 'method': method}
                z, _, _ = infer(tune, c)
                m = cf.fast_metrics(tune, cf.channels(tune, z) @ anchor)
                c['tune_guard'] = cf.guard(m, base) and cf.guard(cf.fast_metrics(tune, z), basewf)
                c['tune_metrics'] = m
                choices.append(c)
            c = {**common, 'method': METHODS[2]}
            z, _, _ = infer(tune, c)
            wm = cf.fast_metrics(tune, z)
            grid = [v for v in cf.weight_grid(dep, seed) if np.min(v) >= .05 - 1e-12]
            grid.append(anchor)
            grid = {tuple(np.round(w, 12)): w for w in grid}
            rows = []
            for w in grid.values():
                m = cf.fast_metrics(tune, cf.channels(tune, z) @ w)
                row = {'weights': w.tolist(), 'metrics': m,
                       'guard': cf.guard(m, base) and cf.guard(wm, basewf)}
                rows.append(row)
                grid_rows.append({'deployment': dep, 'seed': seed, 'guard': row['guard'],
                                  **dict(zip(('w_waveform', 'w_time', 'w_sky'), w)), **m})
            eligible = [r for r in rows if r['guard']]
            selected = min(eligible or rows, key=lambda row: metric_key(row, anchor))
            c.update(weights=selected['weights'], tune_guard=bool(eligible), tune_metrics=selected['metrics'])
            choices.append(c)
            # Perturb forbidden legacy/PE fields: candidate inference must be bit-identical.
            poison = tune.copy()
            forbidden = [k for k in poison if ('mass_' in k or 'logmc' in k or 'logitq' in k or
                         k in ('waveform_score', 'previous_waveform_score', 'FRT_baseline_waveform_score',
                               'PATH875_waveform', 'PATH875_final_score', 'new_encoder_cosine'))]
            for k in forbidden:
                poison[k] = -987654321.
            poison['pe_mc_bhattacharyya_coefficient'] = np.nan
            poison['official_po_fpp'] = -123.
            for config in choices[-3:]:
                a = infer(tune, config)[0]
                b = infer(poison, config)[0]
                if not np.array_equal(a, b):
                    raise RuntimeError('Forbidden old evidence or PE affects score')
                tests.append({'deployment': dep, 'seed': seed, 'method': config['method'],
                              'poisoned_legacy_columns': len(forbidden), 'max_score_difference': 0,
                              'no_outer_alpha': True, 'time_sky_raw_untouched': True})
            print('CALIBRATED', dep, seed, [c['tune_guard'] for c in choices[-3:]], flush=True)
    write_json(root / 'configs/SELECTED_CONFIGURATIONS.json', choices)
    write_csv(root / 'tables/CALIBRATION_GRID.csv', calibration_rows)
    write_csv(root / 'tables/VALIDATION_WEIGHT_GRID.csv', grid_rows)
    write_csv(root / 'audit/FORBIDDEN_INPUT_INVARIANCE.csv', tests)
    write_csv(root / 'tables/SELECTED_WEIGHTS.csv', [
        {'deployment': c['deployment'], 'seed': c['seed'], 'method': c['method'],
         'tune_guard': c['tune_guard'], **dict(zip(('waveform', 'time', 'sky'), c['weights'])),
         'cosine_coefficients': json.dumps(c['cosine_calibration']['coef']),
         'joint_coefficients': json.dumps(c['joint_calibration']['coef']),
         'gamma': c['gamma'] if c['method'] == METHODS[0] else None,
         'beta': c['beta'] if c['method'] == METHODS[0] else None} for c in choices])
    write_json(freeze, {'UTC': utc(), 'file': 'configs/SELECTED_CONFIGURATIONS.json',
                       'sha256': sha(root / 'configs/SELECTED_CONFIGURATIONS.json'),
                       'new_real_scores_or_external_labels_used': False,
                       'new_test_outcomes_used': False, 'alpha_search': False})


def selections(root):
    f = json.loads((root / 'contracts/CONFIGURATIONS_FROZEN.json').read_text())
    if sha(root / f['file']) != f['sha256']:
        raise RuntimeError('Frozen configurations changed')
    return json.loads((root / f['file']).read_text())


def query_rows(frame, scores):
    n = int(frame.event_count.iloc[0])
    i, j = frame.idx_i.to_numpy(int), frame.idx_j.to_numpy(int)
    matrix = np.full((n, n), -np.inf)
    matrix[i, j], matrix[j, i] = scores, scores
    true = np.flatnonzero(frame.is_true_pair.to_numpy(bool))
    rows = []
    for pos in true:
        for a, b in ((i[pos], j[pos]), (j[pos], i[pos])):
            r = 1 + int((matrix[a] > scores[pos]).sum())
            worst = int((matrix[a] >= scores[pos]).sum())
            rows.append({'query_idx': a, 'companion_idx': b, 'system_key': f'{i[pos]}:{j[pos]}',
                         'family': str(frame.true_pair_family.iloc[pos]), 'rank': r,
                         'pessimistic_rank': worst})
    return pd.DataFrame(rows)


def public_frame(frame, z, w, method):
    cols = ['idx_i', 'idx_j', 'event_i', 'event_j', 'pair_key', 'event_count', 'is_true_pair',
            'true_pair_family', 'delta_t_days', 'time_score', 'sky_raw_log_bf', 'sky_bc',
            'embedding_only', 'joint_logbc', 'joint_BC', 'joint_ood']
    out = frame[[c for c in cols if c in frame]].copy()
    out['waveform_score'] = z
    for col, values, weight in zip(('waveform', 'time', 'sky'),
                                   (z, frame.time_score, frame.sky_raw_log_bf), w):
        out[col + '_contribution'] = weight * values
    out['final_score'] = cf.channels(frame, z) @ np.asarray(w)
    out['method'] = method
    return out


def evaluate(root):
    if (root / 'contracts/EVALUATION_COMPLETE.json').exists():
        raise RuntimeError('Already evaluated')
    config = {(c['deployment'], c['seed'], c['method']): c for c in selections(root)}
    metrics, replay = [], []
    for dep in DEPS:
        for seed in SEEDS:
            recipe = recipes()[dep, seed]
            for split, catalog in [('validation', None), ('test', None)] + [('sept8_reused', c) for c in CATALOGS]:
                f = load_panel(dep, seed, split, catalog)
                basew = np.asarray(recipe['effective_weights'])
                bz = f.PATH875_waveform.to_numpy(float)
                bs = cf.channels(f, bz) @ basew
                error = float(abs(bs - f.PATH875_final_score.to_numpy(float)).max())
                if error > 1e-10:
                    raise RuntimeError('Baseline score replay failed')
                baselines = {'waveform': cf.fast_metrics(f, bz), 'fusion': cf.fast_metrics(f, bs)}
                panel = split if catalog is None else f'{split}_{catalog}'
                for method in (BASELINE, *METHODS):
                    if method == BASELINE:
                        z, w = bz, basew
                        ood = clipped = np.zeros(len(f), bool)
                    else:
                        c = config[dep, seed, method]
                        z, ood, clipped = infer(f, c)
                        w = np.asarray(c['weights'])
                    out = public_frame(f, z, w, method)
                    out['score_ood'] = ood
                    out['score_clipped'] = clipped
                    dest = root / f'results/{method}/{dep}/seed_{seed}/{panel}'
                    dest.mkdir(parents=True, exist_ok=True)
                    out.to_parquet(dest / 'pairs.parquet', index=False)
                    for mode, s in (('waveform', z), ('fusion', out.final_score.to_numpy(float))):
                        m = cf.fast_metrics(f, s)
                        ranks = query_rows(f, s)
                        for k in (1, 5, 10):
                            direct = ranks.assign(hit=ranks['rank'] <= k).groupby('family').hit.mean().mean()
                            if abs(direct - m[f'macro_r_at_{k}']) > 1e-12:
                                raise RuntimeError('Independent query-rank replay failed')
                        ranks.to_parquet(dest / f'{mode}_query_ranks.parquet', index=False)
                        metrics.append({'deployment': dep, 'seed': seed, 'method': method, 'split': split,
                                        'catalog': catalog, 'mode': mode, 'events': int(f.event_count.iloc[0]),
                                        'true_systems': int(f.is_true_pair.sum()), 'n_queries': len(ranks),
                                        'median_rank': float(ranks['rank'].median()),
                                        'OOD_rate': float(ood.mean()), 'clip_rate': float(clipped.mean()),
                                        'query_top10_tie_fraction': float(((ranks['rank'] <= 10) &
                                              (ranks.pessimistic_rank > 10)).mean()),
                                        'guard_pass': bool(cf.guard(m, baselines[mode])), **m})
                    for col in ('time_score', 'sky_raw_log_bf'):
                        assert np.array_equal(out[col], f[col])
                replay.append({'deployment': dep, 'seed': seed, 'panel': panel,
                               'baseline_max_abs_score_error': error, 'time_sky_max_abs_change': 0})
            print('EVALUATED', dep, seed, flush=True)
    write_csv(root / 'tables/RETRIEVAL_PER_SEED.csv', metrics)
    a = pd.DataFrame(metrics)
    cols = list(KEY_METRICS) + ['median_rank', 'pessimistic_macro_r_at_10', 'OOD_rate', 'clip_rate']
    # Average the three shared catalogs within each model before model-seed SD.
    per_model = a.groupby(['deployment', 'seed', 'method', 'split', 'mode'])[cols].mean().reset_index()
    write_csv(root / 'tables/RETRIEVAL_PER_MODEL.csv', per_model)
    summary = per_model.groupby(['deployment', 'method', 'split', 'mode'])[cols].agg(['mean', 'std']).reset_index()
    summary.columns = ['_'.join(c).rstrip('_') for c in summary.columns]
    write_csv(root / 'tables/RETRIEVAL_SUMMARY.csv', summary)
    write_csv(root / 'audit/BASELINE_AND_PHYSICAL_REPLAY.csv', replay)
    write_json(root / 'contracts/EVALUATION_COMPLETE.json', {'UTC': utc(), 'rows': len(a),
                    'baseline_reproduced': True, 'all_physical_score_differences': 0,
                    'new_independent_confirmation': False})


def join_external(frame, external):
    use = ['pair_key'] + [c for c in external if c not in frame]
    return frame.merge(external[use], on='pair_key', how='left', validate='one_to_one')


def real(root):
    if (root / 'contracts/REAL_COMPLETE.json').exists():
        raise RuntimeError('Already audited')
    if not (root / 'contracts/EVALUATION_COMPLETE.json').exists():
        raise RuntimeError('Injection evaluation must finish first')
    configs = {(c['deployment'], c['seed'], c['method']): c for c in selections(root)}
    budgets, per_seed, correlations, invariance = [], [], [], []
    for dep in DEPS:
        external = pd.read_parquet(t.EXTERNAL / f'{dep}_external_reference.parquet')
        rank_lists = {(method, mode): [] for method in (BASELINE, *METHODS) for mode in ('fusion', 'waveform')}
        for seed in SEEDS:
            f = load_panel(dep, seed, 'real')
            if len(f) != {'gwtc3': 1891, 'gwtc4': 2701}[dep]:
                raise RuntimeError('Real scope changed')
            for method in (BASELINE, *METHODS):
                if method == BASELINE:
                    z, w = f.PATH875_waveform.to_numpy(float), recipes()[dep, seed]['effective_weights']
                    ood = clip = np.zeros(len(f), bool)
                else:
                    c = configs[dep, seed, method]
                    z, ood, clip = infer(f, c)
                    w = c['weights']
                out = public_frame(f, z, w, method)
                out['score_ood'], out['score_clipped'] = ood, clip
                # No external columns are present when ranking is computed.
                assert not any(c.startswith(('pe_', 'official_')) for c in out)
                for mode in ('fusion', 'waveform'):
                    rank = dev.BASE.rank_real(out, cf.weights_dict(w), mode, seed)
                    rank_lists[method, mode].append(rank)
                    rank = join_external(rank, external)
                    dest = root / f'results/{method}/{dep}/seed_{seed}/real'
                    dest.mkdir(parents=True, exist_ok=True)
                    rank.to_parquet(dest / f'{mode}_all_pairs.parquet', index=False)
                    write_csv(dest / f'{mode}_top100.csv', rank.head(100))
                    if mode == 'fusion':
                        for b in (10, 20, 50, 100):
                            per_seed.append({'seed': seed, **dev.budget_row(rank, method, dep, mode, b)})
                merged = join_external(out, external)
                for label, sign in (('pe_mc_bhattacharyya_coefficient', 1), ('pe_mc_standardized_distance', -1)):
                    x = merged[label].to_numpy(float) * sign
                    good = np.isfinite(x) & np.isfinite(z)
                    rho = float(spearmanr(z[good], x[good]).statistic)
                    correlations.append({'deployment': dep, 'seed': seed, 'method': method,
                                         'PE_quantity': label, 'sign': sign,
                                         'spearman': rho if np.isfinite(rho) else None,
                                         'n_pairs': int(good.sum()), 'descriptive_only': True})
                invariance.append({'deployment': dep, 'seed': seed, 'method': method,
                                   'pairs': len(out), 'PE_joined_after_ranking': True,
                                   'real_score_ood': float(ood.mean()), 'real_score_clip': float(clip.mean())})
        for (method, mode), items in rank_lists.items():
            result = dev.BASE.consensus_real(items, mode)
            result = join_external(result, external)
            dest = root / f'results/{method}/{dep}/consensus'
            dest.mkdir(parents=True, exist_ok=True)
            result.to_parquet(dest / f'{mode}_all_pairs.parquet', index=False)
            write_csv(dest / f'{mode}_all_pairs.csv', result)
            for b in (10, 20, 50, 100):
                write_csv(dest / f'{mode}_top{b}.csv', result.head(b))
                if mode == 'fusion':
                    budgets.append(dev.budget_row(result, method, dep, mode, b))
        print('REAL_AUDITED', dep, flush=True)
    write_csv(root / 'tables/PE_OFFICIAL_BUDGETS.csv', budgets)
    write_csv(root / 'tables/PER_SEED_PE_OFFICIAL_BUDGETS.csv', per_seed)
    write_csv(root / 'tables/WAVEFORM_PE_CORRELATIONS.csv', correlations)
    write_csv(root / 'audit/REAL_SCOPE_AND_OOD.csv', invariance)
    b = pd.DataFrame(budgets)
    goals = []
    higher = ('BC_mc_ge_0p5', 'median_BC_mc', 'Dmax_le_3', 'official_frontend', 'official_hanabi')
    for method in METHODS:
        for dep in DEPS:
            for budget in (10, 20):
                old = b[(b.config == BASELINE) & (b.deployment == dep) & (b.budget == budget)].iloc[0]
                new = b[(b.config == method) & (b.deployment == dep) & (b.budget == budget)].iloc[0]
                tests = {c: bool(new[c] >= old[c] - 1e-12) for c in higher}
                tests['catastrophic_mc'] = bool(new.catastrophic_mc <= old.catastrophic_mc)
                goals.append({'method': method, 'deployment': dep, 'budget': budget,
                              'PE_and_official_no_loss': all(tests.values()),
                              **{c + '_pass': v for c, v in tests.items()},
                              **{c + '_delta': float(new[c] - old[c]) for c in (*higher, 'catastrophic_mc')}})
    write_csv(root / 'tables/PE_OFFICIAL_NO_LOSS_GATES.csv', goals)
    write_json(root / 'contracts/REAL_COMPLETE.json', {'UTC': utc(), 'no_reselection': True,
                    'all_arms_delivered': True, 'external_result_is_lensing_truth': False})


def run(root, stage):
    start = time.monotonic()
    try:
        globals()[stage](root)
    except Exception as exc:
        if root.exists():
            write_json(root / f'logs/{stage}_FAIL_{datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")}.json',
                       {'UTC': utc(), 'exception': repr(exc), 'wall_seconds': time.monotonic() - start})
        raise
    write_json(root / f'logs/{stage}_COMPLETE.json', {
        'UTC': utc(), 'wall_seconds': time.monotonic() - start,
        'max_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'CPU_user_seconds': resource.getrusage(resource.RUSAGE_SELF).ru_utime,
        'CPU_system_seconds': resource.getrusage(resource.RUSAGE_SELF).ru_stime})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--stage', choices=('initialize', 'calibrate', 'evaluate', 'real'), required=True)
    args = parser.parse_args()
    run(args.root, args.stage)
