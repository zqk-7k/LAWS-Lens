#!/usr/bin/env python3
"""Conditional waveform calibration combining complementary learned features."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss
import sklearn
import mcwf_pe_frontend_extension_v2_20260907 as e
import mcwf_large_population_encoder_20260907 as encoder
import mcwf_large_population_rnc_evaluate_20260907 as rnc
import mcwf_expanded_calibration_20260907 as old_cal
import mcwf_highermode_evaluate_20260907 as higher
import mcwf_mass_tf_20260905 as tf
import mcwf_finite_reference_tail_20260907 as tail

dev, body, ev = e.dev, e.body, e.ev
BETAS = (0., .0625, .125, .25, .5, .75, 1.)
FEATURE_NAMES = ['Z_FRT', 'log_BC_new_RNC', 'cosine_new_RNC', 'HM_mass_product',
                 'HM_joint_minus_mass_product', 'minus_D_new_predicted_mass',
                 'max_predictive_entropy', 'abs_predictive_entropy_difference']


def features(f, a, h):
    i, j = f.idx_i.to_numpy(int), f.idx_j.to_numpy(int)
    x = rnc.features(a, i, j)
    p = np.asarray(a['p'], float)
    mean = p @ tf.LOG_CENTERS
    var = (p @ (tf.LOG_CENTERS ** 2) - mean ** 2).clip(1e-8)
    entropy = -np.sum(p * np.log(p.clip(1e-300)), axis=1)
    distance = np.abs(mean[i] - mean[j]) / np.sqrt(var[i] + var[j])
    result = np.column_stack([f.waveform_score.to_numpy(float), np.log(x['bc']), x['cosine'],
        h['mass'], h['conditional'], -distance, np.maximum(entropy[i], entropy[j]),
        np.abs(entropy[i] - entropy[j])])
    if not np.isfinite(result).all():
        raise RuntimeError('Invalid waveform calibration features')
    return result


def model(seed, columns):
    return HistGradientBoostingClassifier(learning_rate=.05, max_iter=100, max_leaf_nodes=4,
        max_depth=2, min_samples_leaf=40, l2_regularization=10., early_stopping=False,
        random_state=seed, monotonic_cst=([1] if columns == 1 else [1, 1, 1, 1, 1, 1, 0, 0]))


def weights(y):
    return np.where(y, 1., y.sum() / max(1, (~y).sum()))


def fit(root, dep, ms, es):
    out = root / f'large_population_stack/calibration/{dep}/seed_{es}'
    if (out / 'FIT.json').exists():
        return {k: joblib.load(out / f'{k}.joblib') for k in ('base', 'full')}, json.loads((out / 'FIT.json').read_text())
    out.mkdir(parents=True, exist_ok=True)
    f = old_cal.development(root, dep, ms, es)
    cp = root / f'large_population_encoder/models/{dep}/seed_{ms}/validation_selected_model.pt'
    if not (cp.parent / 'COMPLETE.json').exists():
        raise RuntimeError('Training incomplete')
    a = np.load(cp.parent / 'validation_predictions.npz')
    hc = np.load(root / f'highermode_density/calibration/{dep}/seed_{ms}/fit.npz')
    if not np.array_equal(hc['i'], f.idx_i) or not np.array_equal(hc['j'], f.idx_j):
        raise RuntimeError('Development pair ordering mismatch')
    x = features(f, a, hc)
    y = f.is_true_pair.to_numpy(bool)
    source = a['group']
    fold = np.random.default_rng(202609311).permutation(len(np.unique(source))) % 5
    fi, fj = fold[source[f.idx_i.to_numpy(int)]], fold[source[f.idx_j.to_numpy(int)]]
    rows = []
    for k in range(5):
        train, test = (fi != k) & (fj != k), (fi == k) & (fj == k)
        for kind, xx in [('base', x[:, :1]), ('full', x)]:
            clf = model(ms, xx.shape[1])
            clf.fit(xx[train], y[train], sample_weight=weights(y[train]))
            rows.append({'fold': k, 'kind': kind, 'train_source_pairs': int(y[train].sum()),
                'test_source_pairs': int(y[test].sum()), 'source_disjoint': True,
                'noise_blocks_shared_across_folds': True,
                'balanced_log_loss': float(log_loss(y[test], clf.predict_proba(xx[test])[:, 1], sample_weight=weights(y[test])))})
    fitted = {}
    for kind, xx in [('base', x[:, :1]), ('full', x)]:
        clf = model(ms, xx.shape[1])
        clf.fit(xx, y, sample_weight=weights(y))
        fitted[kind] = clf
        joblib.dump(clf, out / f'{kind}.joblib')
    spec = {'minimum': x.min(0).tolist(), 'maximum': x.max(0).tolist(),
            'mass_reference': np.sort(-x[y, 1]).tolist(), 'source_pairs': int(y.sum()), 'noise_blocks': 32,
            'new_encoder_sha256': dev.sha(cp), 'sklearn_version': sklearn.__version__,
            'feature_names': FEATURE_NAMES, 'folds_are_diagnostic_not_selection': True}
    dev.csv_write(out / 'SOURCE_FOLD_AUDIT.csv', pd.DataFrame(rows))
    dev.json_write(out / 'FIT.json', spec)
    print(json.dumps({'large_population_stack_fitted': dep, 'seed': es, 'source_pairs': int(y.sum())}), flush=True)
    return fitted, spec


def get(root, dep, ms, es, split):
    f, h = higher.get(root, dep, ms, es, split)
    a = encoder.prediction(root, dep, ms, es, split)
    return f, features(f, a, h)


def score(f, x, cal, spec, arm):
    full = cal['full'].decision_function(x)
    base = cal['base'].decision_function(x[:, :1]) if arm == 'CONDITIONAL' else f.waveform_score.to_numpy(float)
    delta = np.clip(full - base, -3., 3.)
    ood = ((x < np.asarray(spec['minimum'])) | (x > np.asarray(spec['maximum']))).any(1)
    mass_compatible = tail.tail_probability(-x[:, 1], spec['mass_reference']) >= .05
    delta = np.where(ood | ~mass_compatible, np.minimum(delta, 0.), delta)
    return f.waveform_score.to_numpy(float) + spec['beta'] * delta, delta, ood


def run(root, arm):
    trial = root / f'trials/LARGE-POPULATION-STACK-{arm}'
    if trial.exists():
        raise RuntimeError('Independent trial required')
    for name in ('contracts', 'tables', 'calibration', 'evaluation', 'results'):
        (trial / name).mkdir(parents=True)
    shutil.copy2(__file__, trial / 'contracts' / Path(__file__).name)
    dev.json_write(trial / 'contracts/RECIPE.json', {
        'created_utc': datetime.now(timezone.utc).isoformat(), 'same_O3_O4a': True,
        'features': FEATURE_NAMES, 'base_encoder': '12288-source RAW-PHASE-SOURCE RNC plus frozen higher-mode predictive density',
        'fit': '512 development source parents,32 independent validation noise blocks; all pair negatives share events; source-fold cross-validation is diagnostic,not noise-disjoint or model-unseen validation',
        'classifier': 'HGB100trees,maxdepth2,max4leaves,lr0.05,L2=10,minleaf40,monotone increasing infirst6compatibilityfeatures',
        'formula': 'Z_FRT+beta*clip(full_predictive_logLR-base_logLR,-3,3);CONDITIONAL base=fitted old-score logLR;RECALIBRATED base=oldZ_FRT',
        'positive_support': 'within development features ranges and new mass finite-tail compatibility>=0.05',
        'beta_grid': BETAS, 'selection': 'BAYESTAR validation only with original objective and guardrails',
        'relative_scale': 'waveform calibration can change relative evidence strength despite frozen outerweights; not independent Bayes-factor addition',
        'frozen': ['time', 'sky', 'outerweights', 'scope', 'oldmodels', 'historicalresults'],
        'PE_official_ID_inputs': False, 'adaptive_real_audit': True, 'fresh_confirmation_required': True,
        'reference': 'https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html'})
    selected, calibrators, cache, states = {}, {}, {}, []
    for dep in e.DEPS:
        for ms, es in zip(body.MODEL_SEEDS, dev.SEEDS):
            cal, spec = fit(root, dep, ms, es)
            calibrators[dep, es] = cal
            f, x = get(root, dep, ms, es, 'validation')
            cache[dep, es] = f, x
            bm = ev.metrics(f, f.waveform_score.to_numpy(float), dep, es)
            rows, choices = [], []
            for beta in BETAS:
                cfg = {**spec, 'beta': beta}
                z, _, _ = score(f, x, cal, cfg, arm)
                mm = ev.metrics(f, z, dep, es)
                ok = ev.guard(mm, bm)
                rows.append({'beta': beta, 'pass': ok, **{a + '_' + k: v for a, b in mm.items() for k, v in b.items()}})
                if ok:
                    choices.append((e.old_selection.objective(mm, 0., beta), cfg))
            spec = min(choices, key=lambda a: a[0])[1]
            selected[dep, es] = spec
            out = trial / f'calibration/{dep}/seed_{es}'
            out.mkdir(parents=True)
            dev.csv_write(out / 'validation_grid.csv', pd.DataFrame(rows))
            dev.json_write(out / 'SELECTED_CONFIG.json', spec)
            states.append({'deployment': dep, 'seed': es, 'beta': spec['beta']})
    dev.json_write(trial / 'contracts/SELECTED_FREEZE.json', {'configs': states, 'real_used_to_select': False})
    print(json.dumps({'large_stack_selected': arm, 'states': states}), flush=True)
    rows, guards = [], []
    for dep in e.DEPS:
        real = {}
        for ms, es in zip(body.MODEL_SEEDS, dev.SEEDS):
            for split in ('validation', 'test', 'real'):
                f, x = cache[dep, es] if split == 'validation' else get(root, dep, ms, es, split)
                z, inc, ood = score(f, x, calibrators[dep, es], selected[dep, es], arm)
                n = f.rename(columns={c: 'baseline_' + c for c in ('final_score', 'rank', 'method', 'waveform_contribution', 'time_contribution', 'sky_contribution') if c in f}).copy()
                n['FRT_baseline_waveform_score'], n['waveform_score'] = f.waveform_score, z
                n['stack_increment'], n['stack_ood'] = inc, ood
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
    for arm in ('CONDITIONAL', 'RECALIBRATED'):
        run(args.root, arm)
