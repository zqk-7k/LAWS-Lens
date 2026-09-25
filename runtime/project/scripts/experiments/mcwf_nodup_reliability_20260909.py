#!/usr/bin/env python3
"""Simulation-calibrated joint predictive compatibility, without old mass scores."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import pickle
import shutil
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.ensemble import HistGradientBoostingClassifier
import torch

P = Path('/root/autodl-tmp/gw-catalog')
sys.path.insert(0, str(P / 'scripts/experiments'))
import mcwf_conditional_intrinsics_20260908 as physical
sp = importlib.util.spec_from_file_location('nodup_core', P / 'scripts/experiments/mcwf_nodup_nomix_20260909T071108Z.py')
n = importlib.util.module_from_spec(sp)
sp.loader.exec_module(n)
INTR = P / 'results/mcwf_conditional_intrinsics_exploratory_20260908T154330Z'
PRIOR = P / 'results/mcwf_nodup_nomix_01_20260909T071108Z'
KINDS = ('LINEAR', 'TREE')
METHODS = ('NODUP-DIRECT-REPLAY', 'RELIABILITY-LINEAR', 'RELIABILITY-TREE')
FEATURES = ('log_joint_BC', 'negative_abs_delta_logMc', 'negative_standardized_delta',
            'log_pooled_width', 'mean_logMc', 'abs_log_width_ratio')
BETAS = (0., .0625, .125, .25, .5, 1., 2., 4., 8.)
ROOT = None
BASE_LOAD, BASE_INFER, BASE_EXPORT = n.load_panel, n.infer, n.public_frame
SUMMARY_CACHE, CLASSIFIER_CACHE = {}, {}


def source_fold(meta):
    banks = sorted(meta.noise_bank_index.unique(),
                   key=lambda b: hashlib.sha256(f'202609850:{meta.deployment.iloc[0]}:{b}'.encode()).hexdigest())
    folds = meta.noise_bank_index.map({b: k % 2 for k, b in enumerate(banks)}).to_numpy(int)
    group = meta.source_uid.to_numpy(str)
    mixed = meta.assign(fold=folds).groupby('source_uid').fold.nunique()
    folds[np.isin(group, mixed[mixed > 1].index)] = -1
    return folds


def mass_summary(p):
    valid = np.isfinite(p).all(1)
    out = np.full((len(p), 3), np.nan)
    for idx in np.flatnonzero(valid):
        prob = p[idx]
        if abs(prob.sum() - 1.) > 1e-5 or np.any(prob < 0):
            raise RuntimeError('Invalid predictive probability mass')
        qs = np.interp([.16, .5, .84], np.r_[0., prob.cumsum()], physical.old.EDGES)
        out[idx] = qs[1], max((qs[2] - qs[0]) / 2, .001), -np.sum(prob * np.log(prob.clip(1e-300)))
    return out


def pair_features(summary, i, j, logbc):
    mi, mj = summary[i, 0], summary[j, 0]
    si, sj = summary[i, 1], summary[j, 1]
    delta, width = abs(mi - mj), np.sqrt(si**2 + sj**2)
    x = np.c_[logbc, -delta, -delta / width, np.log(width), (mi + mj) / 2, abs(np.log(si / sj))]
    if not np.isfinite(x).all():
        raise RuntimeError('Nonfinite waveform-only predictor features')
    return x


def prediction_path(dep, seed, split, catalog):
    slot = n.recipes()[dep, seed]['slot']
    if catalog is not None:
        return n.FRESH / f'confirmation/{dep}/catalog_{catalog}/model_{slot}/joint_predictions.npz'
    return INTR / f'predictions/CONDITIONAL-ETA-CHI/{dep}/{slot}_{seed}_{split}.npz'


def load_panel(dep, seed, split, catalog=None):
    frame = BASE_LOAD(dep, seed, split, catalog)
    path = prediction_path(dep, seed, split, catalog)
    key = str(path)
    if key not in SUMMARY_CACHE:
        with np.load(path) as a:
            prob = a['p'] if 'p' in a else a['joint'].sum(-1)
            SUMMARY_CACHE[key] = mass_summary(prob)
    x = pair_features(SUMMARY_CACHE[key], frame.idx_i.to_numpy(int), frame.idx_j.to_numpy(int),
                      frame.joint_logbc.to_numpy(float))
    for k, col in enumerate(FEATURES):
        frame[col] = x[:, k]
    return frame


def development(dep, seed):
    slot = n.recipes()[dep, seed]['slot']
    meta = pd.read_parquet(n.t.PREVIOUS / f'expanded_data/{dep}/validation/event_metadata.parquet')
    meta['deployment'] = dep
    fold = source_fold(meta)
    path = INTR / f'predictions/CONDITIONAL-ETA-CHI/{dep}/{slot}_0_development.npz'
    with np.load(path) as a:
        p = a['p']
        if not np.allclose(a['truth'], np.log(meta.mc_det), atol=1e-12):
            raise RuntimeError('Predictive/source row alignment failed')
        sm = mass_summary(p)
    bc_path = n.JOINT / f'cache/CONDITIONAL-ETA-CHI/{dep}/{slot}_0_development.npz'
    with np.load(bc_path) as b:
        logbc = b['joint_logbc']
    parts = {}
    for side in (0, 1):
        ids = np.flatnonzero(fold == side)
        i, j = np.triu_indices(len(ids), 1)
        i, j = ids[i], ids[j]
        groups = meta.source_uid.to_numpy(str)
        y = groups[i] == groups[j]
        x = pair_features(sm, i, j, logbc[i, j])
        # Each independent unordered source-pair contributes equally within class.
        keys = np.array([':'.join(sorted((a, b))) for a, b in zip(groups[i], groups[j])])
        counts = pd.Series(keys).value_counts()
        weight = pd.Series(keys).map(1 / counts).to_numpy()
        weight[y] *= .5 / weight[y].sum()
        weight[~y] *= .5 / weight[~y].sum()
        parts[side] = (x, y, weight, i, j)
    source_sets = [set(meta.source_uid[fold == f]) for f in (0, 1)]
    assert not source_sets[0] & source_sets[1]
    assert not set(meta.noise_bank_index[fold == 0]) & set(meta.noise_bank_index[fold == 1])
    n.write_csv(ROOT / f'audit/{dep}_{seed}_RELIABILITY_DEVELOPMENT_SPLIT.csv',
                meta[['source_uid', 'noise_bank_index']].assign(fold=fold))
    return parts


def fit_linear(x, y, w, ridge):
    mu = np.sum(w[:, None] * x, axis=0)
    sd = np.sqrt(np.sum(w[:, None] * (x - mu)**2, axis=0)).clip(1e-6)
    z = (x - mu) / sd
    def loss(a):
        logits = a[0] + z @ a[1:]
        residual = w * (expit(logits) - y)
        return (float(np.dot(w, np.logaddexp(0., logits) - y * logits) + .5 * ridge * (a[1:] @ a[1:])),
                np.r_[residual.sum(), z.T @ residual + ridge * a[1:]])
    fit = minimize(loss, np.zeros(7), jac=True, method='L-BFGS-B',
                   bounds=[(None, None)] + [(0., None)] * 3 + [(None, None)] * 3,
                   options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8})
    if not fit.success:
        raise RuntimeError(str(fit.message))
    return {'kind': 'LINEAR', 'mu': mu, 'sd': sd, 'theta': fit.x}


def raw_predict(classifier, x):
    if classifier['kind'] == 'LINEAR':
        return classifier['theta'][0] + ((x - classifier['mu']) / classifier['sd']) @ classifier['theta'][1:]
    return classifier['estimator'].decision_function(x)


def fit_classifier(dep, seed, kind):
    x, y, w, _, _ = development(dep, seed)[0]
    xx, yy, ww, _, _ = development(dep, seed)[1]
    trials, models = [], []
    for ridge in (.001, .01, .1, 1.):
        if kind == 'LINEAR':
            classifier = fit_linear(x, y, w, ridge)
            complexity = 0
            options = [classifier]
        else:
            options = []
            for leaves in (3, 7):
                est = HistGradientBoostingClassifier(max_iter=100, learning_rate=.05, max_leaf_nodes=leaves,
                    min_samples_leaf=20, l2_regularization=ridge, early_stopping=False,
                    monotonic_cst=[1, 1, 1, 0, 0, 0], random_state=202609092)
                est.fit(x, y, sample_weight=w * len(x))
                options.append({'kind': kind, 'estimator': est, 'leaves': leaves})
        for classifier in options:
            logits = raw_predict(classifier, xx)
            logloss = float(np.sum(ww * (np.logaddexp(0., logits) - yy * logits)))
            trials.append({'ridge': ridge, 'leaves': classifier.get('leaves', 0), 'logloss': logloss})
            models.append(classifier)
    win = min(range(len(trials)), key=lambda k: (trials[k]['logloss'], trials[k]['leaves'], -trials[k]['ridge']))
    classifier = models[win]
    dest = ROOT / f'calibration/{kind}/{dep}/{seed}.pkl'
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open('xb') as handle:
        pickle.dump(classifier, handle)
    spec = {'file': str(dest.relative_to(ROOT)), 'sha256': n.sha(dest), 'kind': kind,
            'minimum': x.min(0).tolist(), 'maximum': x.max(0).tolist(),
            'cap': float(np.log(int(y.sum()) + 1)), 'fit_sources': int(y.sum()), 'tune_sources': int(yy.sum()),
            'selected': trials[win], 'development_reused_for_checkpoint': True}
    n.write_csv(ROOT / f'calibration/{kind}/{dep}/{seed}_GRID.csv', trials)
    return spec


def apply_classifier(x, spec):
    path = ROOT / spec['file']
    key = str(path)
    if key not in CLASSIFIER_CACHE:
        if n.sha(path) != spec['sha256']:
            raise RuntimeError('Classifier changed after freezing')
        with path.open('rb') as handle:
            CLASSIFIER_CACHE[key] = pickle.load(handle)
    raw = raw_predict(CLASSIFIER_CACHE[key], x)
    # OOD cannot grant a positive compatibility reward. Valid strong negative predictions
    # remain capped diagnostics, not a physical Bayes factor.
    outside = ((x < np.asarray(spec['minimum']) - 1e-7) | (x > np.asarray(spec['maximum']) + 1e-7)).any(1)
    clipped = np.abs(raw) > spec['cap']
    z = raw.clip(-spec['cap'], spec['cap'])
    z[outside] = np.minimum(z[outside], 0.)
    return z, outside, clipped


def infer(frame, c):
    if c['method'] == METHODS[0]:
        return BASE_INFER(frame, {**c, 'method': 'NODUP-DIRECT'})
    zcos, co, cc = n.apply_model(frame, c['cosine_calibration'])
    z, oo, cl = apply_classifier(frame[list(FEATURES)].to_numpy(float), c['reliability'])
    return zcos + c['beta'] * z, co | oo, cc | cl


def freeze():
    path = ROOT / 'contracts/RELIABILITY_PROTOCOL.json'
    if path.exists():
        raise RuntimeError('Protocol already frozen')
    contract = {'id': 'MCWF-NODUP-RELIABILITY-02', 'UTC': n.utc(), 'status': n.STATUS,
        'same_algorithm_both_runs': True, 'neural_training': False,
        'motivation': 'Similar broad predictive densities do not ensure precisely equal source mass. Jointly calibrate the predictive BC, center difference and precision rather than using public PE or artificially narrowing the density.',
        'features': list(FEATURES), 'methods': list(METHODS), 'primary': METHODS[-1],
        'prediction': 'Same frozen MULTIRATE mass density and conditional eta/chi head, probability marginal unchanged.',
        'scoring': 'Pure old embedding COS calibration + beta * ONE new joint predictive reliability logit. Replaces old joint tail/increment, old Mc/q terms and alpha total-score mixing; no independent-evidence-product claim.',
        'classifier': 'Nonnegative-slopes logistic or monotone histogram gradient boosting, proper balanced logloss selection; documented predeclared controls.',
        'classifier_grid': {'ridge': [.001, .01, .1, 1.], 'tree_leaves': [3, 7], 'tree_iterations': 100,
                            'tree_lr': .05, 'tree_min_leaf': 20},
        'fit_tune': '512 existing simulated source parents; source/noise-disjoint split by inherited hash202609850, cross-noise sources discarded, every source-pair totalweight1.',
        'support': 'fit feature box; positive OOD ->0;abs score cap log(nfit sources+1). No new real-based thresholds.',
        'beta_grid': list(BETAS), 'beta_selection': 'BAYESTAR simulation validation: F50,F90,-AP,-R10,-R1; waveform/fusion guardrails; tie smaller beta; report FAIL if none.',
        'outer_weights': 'Unchanged pure pre-PATH875 upstream weights. Time/sky raw values exactly frozen.',
        'real_gate': 'Compare both prior NODUP-DIRECT and PATH875: Top10/20 Mc-pass, medianMcBC,Dmax,official1pct,Hanabi cannot decrease; catastrophic count cannot rise. Critical pair no longer Top20. Failures retained.',
        'validation_limit': 'Adaptive real-development mechanism; real labels not model/calibrator inputs. Reused checkpoints and tests preclude blind confirmation claims.',
        'no_public_PE_or_official_training': True, 'no_event_specific_veto': True,
        'refs': ['https://arxiv.org/abs/gr-qc/9402014', 'https://arxiv.org/abs/1506.02169'],
        'ref_limits': 'Mass-phase physics and classifier-ratio motivation, not a proof of neural posterior accuracy.'}
    n.write_json(path, contract)
    shutil.copy2(__file__, ROOT / 'scripts/reliability_experiment.py')
    n.write_json(ROOT / 'contracts/RELIABILITY_PROTOCOL_SHA256.json',
                 {'contract': n.sha(path), 'script': n.sha(Path(__file__))})


def calibrate():
    if (ROOT / 'contracts/CONFIGURATIONS_FROZEN.json').exists():
        raise RuntimeError('Already frozen')
    originals = json.loads((PRIOR / 'configs/SELECTED_CONFIGURATIONS.json').read_text())
    original = {(c['deployment'], c['seed']): c for c in originals if c['method'] == 'NODUP-DIRECT'}
    configs, rows, support, invariance = [], [], [], []
    for dep in n.DEPS:
        for seed in n.SEEDS:
            c = original[dep, seed]
            configs.append({**c, 'method': METHODS[0]})
            frame = load_panel(dep, seed, 'validation')
            weights = np.asarray(c['weights'])
            base = n.cf.fast_metrics(frame, frame.PATH875_final_score.to_numpy(float))
            basewf = n.cf.fast_metrics(frame, frame.PATH875_waveform.to_numpy(float))
            for kind in KINDS:
                spec = fit_classifier(dep, seed, kind)
                common = {**c, 'method': 'RELIABILITY-' + kind, 'reliability': spec}
                options = []
                for beta in BETAS:
                    cc = {**common, 'beta': beta}
                    z, oo, cl = infer(frame, cc)
                    m, wm = n.cf.fast_metrics(frame, n.cf.channels(frame, z) @ weights), n.cf.fast_metrics(frame, z)
                    passed = n.cf.guard(m, base) and n.cf.guard(wm, basewf)
                    row = {'beta': beta, 'guard': passed, **m, **{'waveform_' + k: v for k, v in wm.items()}}
                    options.append(row)
                    rows.append({'deployment': dep, 'seed': seed, 'kind': kind, **row})
                eligible = [r for r in options if r['guard']]
                win = min(eligible or options, key=lambda r: (r['false_at_recall_0p5'], r['false_at_recall_0p9'],
                          -r['average_precision'], -r['macro_r_at_10'], -r['macro_r_at_1'], r['beta']))
                selected = {**common, 'beta': win['beta'], 'tune_guard': bool(eligible), 'tune_metrics': win}
                configs.append(selected)
                support.append({'deployment': dep, 'seed': seed, 'kind': kind, 'fit': spec['fit_sources'],
                                'tune': spec['tune_sources'], 'beta': win['beta'], 'guard': bool(eligible),
                                'classifier': str(spec['selected'])})
                poisoned = frame.copy()
                for col in frame:
                    if col not in FEATURES and col not in ('embedding_only',):
                        if pd.api.types.is_numeric_dtype(poisoned[col]):
                            poisoned[col] = -987654321.
                poisoned['pe_mc_bhattacharyya_coefficient'] = -123.
                poisoned['official_po_fpp'] = np.nan
                if not np.array_equal(infer(frame, selected)[0], infer(poisoned, selected)[0]):
                    raise RuntimeError('Forbidden metadata affects score')
                invariance.append({'deployment': dep, 'seed': seed, 'kind': kind, 'max_difference': 0})
            print('SELECTED', dep, seed, [(v['method'], v['beta'], v['tune_guard']) for v in configs[-3:]], flush=True)
    n.write_json(ROOT / 'configs/SELECTED_CONFIGURATIONS.json', configs)
    n.write_csv(ROOT / 'tables/RELIABILITY_WEIGHT_GRID.csv', rows)
    n.write_csv(ROOT / 'tables/RELIABILITY_SUPPORT.csv', support)
    n.write_csv(ROOT / 'audit/FORBIDDEN_INPUT_TESTS.csv', invariance)
    n.write_json(ROOT / 'contracts/CONFIGURATIONS_FROZEN.json', {'UTC': n.utc(),
        'file': 'configs/SELECTED_CONFIGURATIONS.json', 'sha256': n.sha(ROOT / 'configs/SELECTED_CONFIGURATIONS.json'),
        'real_or_test_selection': False, 'outer_total_blend': False})


def install():
    n.METHODS = METHODS
    n.load_panel, n.infer = load_panel, infer
    original_rank, original_consensus = n.dev.BASE.rank_real, n.dev.BASE.consensus_real
    def export(frame, z, w, method):
        result = BASE_EXPORT(frame, z, w, method)
        for key in (*FEATURES, 'sky_j50', 'sky_j90'):
            if key in frame:
                result[key] = frame[key].to_numpy()
        return result
    def rank(frame, weights, mode, seed):
        return original_rank(frame, {'waveform': 1., 'time': 0., 'sky': 0.} if mode == 'waveform' else weights, mode, seed)
    def consensus(items, method):
        out = original_consensus(items, method)
        cols = ['joint_BC', 'embedding_only', *FEATURES, 'score_ood', 'score_clipped']
        extra = pd.concat(items).groupby('pair_key')[cols].mean().add_suffix('_mean').reset_index()
        return out.merge(extra, on='pair_key', validate='one_to_one')
    n.public_frame, n.dev.BASE.rank_real, n.dev.BASE.consensus_real = export, rank, consensus


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--stage', choices=('freeze', 'calibrate', 'evaluate', 'real'), required=True)
    args = parser.parse_args()
    ROOT = args.root
    (ROOT / 'configs').mkdir(exist_ok=True)
    install()
    if args.stage in ('freeze', 'calibrate'):
        globals()[args.stage]()
    else:
        n.run(ROOT, args.stage)
