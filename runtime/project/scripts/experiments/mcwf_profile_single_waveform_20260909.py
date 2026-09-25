#!/usr/bin/env python3
"""Calibrate shape and intrinsic consistency jointly, without additive double use."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
from pathlib import Path
import pickle
import shutil
import sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.ensemble import HistGradientBoostingClassifier

P = Path('/root/autodl-tmp/gw-catalog')
sys.path.insert(0, str(P / 'scripts/experiments'))
import mcwf_profile_hierarchical_calibration_20260909 as h
co, r, n = h.co, h.r, h.n
ROOT = None
ACTIVE_FIT = False
FEATURES = ('embedding_only',) + h.FEATURES
METHODS = ('NODUP-DIRECT-REPLAY', 'SINGLE-WF-LINEAR', 'SINGLE-WF-TREE',
           'SINGLE-WF-ACTIVE-LINEAR', 'SINGLE-WF-ACTIVE-TREE')
EMB = P / 'results/mcwf_nodup_latent_density_04_20260909T075600Z/embeddings'
PARENT = P / 'results/mcwf_nodup_hierarchical_profile_13_20260909T093520Z'


def development(dep, seed):
    h.ACTIVE_FIT = ACTIVE_FIT
    old = h.development(dep, seed)
    z = np.load(EMB / f'{dep}/{seed}_development.npy').astype(float)
    norm = np.linalg.norm(z, axis=1, keepdims=True)
    if not np.isfinite(z).all() or np.any(norm < 1e-12):
        raise RuntimeError('Invalid frozen base embedding')
    z = z / norm
    return {fold: (np.column_stack([np.sum(z[i] * z[j], axis=1), x]), y, w, i, j)
            for fold, (x, y, w, i, j) in old.items()}


def linear(x, y, w, ridge):
    mu = np.sum(w[:, None] * x, axis=0)
    sd = np.sqrt(np.sum(w[:, None] * (x - mu)**2, axis=0)).clip(1e-6)
    z = (x - mu) / sd
    def loss(a):
        logits = a[0] + z @ a[1:]
        residual = w * (expit(logits) - y)
        return (float(np.dot(w, np.logaddexp(0., logits) - y * logits)
                      + .5 * ridge * (a[1:] @ a[1:])),
                np.r_[residual.sum(), z.T @ residual + ridge * a[1:]])
    fit = minimize(loss, np.zeros(8), jac=True, method='L-BFGS-B',
                   bounds=[(None, None)] + [(0., None)] * 4 + [(None, None)] * 3,
                   options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8})
    if not fit.success:
        raise RuntimeError(str(fit.message))
    return {'kind': 'LINEAR', 'mu': mu, 'sd': sd, 'theta': fit.x}


def fit_classifier(dep, seed, kind):
    panels = development(dep, seed)
    x, y, w, _, _ = panels[0]
    xx, yy, ww, _, _ = panels[1]
    trials, models = [], []
    for ridge in (.001, .01, .1, 1.):
        if kind == 'LINEAR':
            options = [linear(x, y, w, ridge)]
        else:
            options = []
            for leaves in (3, 7):
                model = HistGradientBoostingClassifier(max_iter=100, learning_rate=.05,
                    max_leaf_nodes=leaves, min_samples_leaf=20, l2_regularization=ridge,
                    early_stopping=False, monotonic_cst=[1, 1, 1, 1, 0, 0, 0],
                    random_state=202609092)
                model.fit(x, y, sample_weight=w * len(x))
                options.append({'kind': kind, 'estimator': model, 'leaves': leaves})
        for model in options:
            logits = r.raw_predict(model, xx)
            value = float(np.sum(ww * (np.logaddexp(0., logits) - yy * logits)))
            trials.append({'ridge': ridge, 'leaves': model.get('leaves', 0), 'logloss': value})
            models.append(model)
    win = min(range(len(trials)), key=lambda i: (trials[i]['logloss'], trials[i]['leaves'], -trials[i]['ridge']))
    scope = 'active' if ACTIVE_FIT else 'global'
    dest = ROOT / f'calibration/{scope}/{kind}/{dep}/{seed}.pkl'
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open('xb') as file:
        pickle.dump(models[win], file)
    n.write_csv(dest.with_name(dest.stem + '_GRID.csv'), trials)
    return {'file': str(dest.relative_to(ROOT)), 'sha256': n.sha(dest), 'kind': kind,
            'minimum': x.min(0).tolist(), 'maximum': x.max(0).tolist(),
            'cap': float(np.log(int(y.sum()) + 1)), 'fit_sources': int(y.sum()),
            'tune_sources': int(yy.sum()), 'selected': trials[win],
            'development_reused_for_checkpoint': True}


def infer(frame, config):
    original = h.isolated.ORIGINALS[config['deployment'], config['seed']]
    base = co.BASE_INFER(frame, {**original, 'method': METHODS[0]})
    if config['method'] == METHODS[0]:
        return base
    output, outside, clipped = r.apply_classifier(frame[list(FEATURES)].to_numpy(float),
                                                 config['waveform_calibrator'])
    if config['active_only']:
        active = frame.profile_pair_active.to_numpy(bool)
        result = tuple(np.where(active, new, old) for new, old in zip((output, outside, clipped), base))
        if not np.array_equal(result[0][~active], base[0][~active]):
            raise RuntimeError('Inactive waveform score changed')
        return result
    return output, outside, clipped


def freeze():
    if ROOT.exists():
        raise RuntimeError('Independent directory required')
    for folder in ('contracts', 'configs', 'calibration', 'tables', 'audit', 'reports',
                   'scripts', 'logs', 'manifest', 'results', 'figures'):
        (ROOT / folder).mkdir(parents=True)
    n.write_json(ROOT / 'contracts/ANALYSIS_CONTRACT.json', {
        'id': 'MCWF-NODUP-SINGLE-WAVEFORM-14', 'UTC': n.utc(), 'status': n.STATUS,
        'parent': str(PARENT), 'goal_achieved': False, 'adaptive_development': True,
        'motivation': 'Embedding shape and intrinsic compatibility share waveform data. A single classifier learns their dependence rather than summing separate calibrated ratios.',
        'features': list(FEATURES), 'methods': list(METHODS), 'same_algorithm_both_runs': True,
        'fit': 'Source/noise-disjoint simulation fold0; one totalweight per unordered sourcepair and class-balanced training.',
        'selection': 'Fold1 proper balanced logloss; ridge .001,.01,.1,1; LINEAR or monotonic TREE3/7leaves100iterations. All four arms reported.',
        'score': 'Exactly ONE joint waveform classifier logit. No separate Zcos, mass LR, joint LR or global old/new blend added.',
        'monotonic': 'Nondecreasing in embedding cosine, massBC, conditionalBC and negative standardized mass distance.',
        'active_fallback': 'Inactive pairs use immutable NODUP-DIRECT coefficients; no candidate coefficient substitution.',
        'OOD': 'Fit feature box; positive reward0 outside; abs cap log(nfittrue+1).',
        'frozen': ['time', 'sky', 'outer weights', 'profile predictions', 'encoder', 'conditional eta/chi', 'scope', 'historical results'],
        'forbidden': ['old encoder Mc/q scores', 'total-score mixture', 'event-name veto', 'PE/official feature or weight selection'],
        'limitation': 'Existing validation and external catalogs reused. Not independent confirmation or physical common-source Bayes factor.'})
    inputs = pd.read_csv(PARENT / 'manifest/INPUT_SHA256.csv').to_dict('records')
    units = []
    for dep in n.DEPS:
        for seed in n.SEEDS:
            for split in ('development', 'validation'):
                path = EMB / f'{dep}/{seed}_{split}.npy'
                inputs.append({'path': str(path), 'bytes': path.stat().st_size, 'sha256': n.sha(path)})
            z = np.load(EMB / f'{dep}/{seed}_validation.npy').astype(float)
            z /= np.linalg.norm(z, axis=1, keepdims=True)
            frame = h.load_panel(dep, seed, 'validation')
            pred = np.sum(z[frame.idx_i] * z[frame.idx_j], axis=1)
            delta = float(abs(pred - frame.embedding_only.to_numpy()).max())
            if delta > 2e-6:
                raise RuntimeError('Frozen cosine representation mismatch')
            original = h.isolated.ORIGINALS[dep, seed]
            reference = infer(frame, {**original, 'method': METHODS[0]})[0]
            archived = pd.read_parquet(r.PRIOR / f'results/NODUP-DIRECT/{dep}/seed_{seed}/validation/pairs.parquet')
            if not np.array_equal(frame[['idx_i', 'idx_j']].to_numpy(), archived[['idx_i', 'idx_j']].to_numpy()):
                raise RuntimeError('Archived row alignment mismatch')
            error = float(abs(reference - archived.waveform_score.to_numpy()).max())
            if error > 1e-12:
                raise RuntimeError('Baseline changed')
            units.append({'deployment': dep, 'seed': seed, 'max_cosine_replay_difference': delta,
                          'max_archived_score_difference': error, 'old_parameter_head_executed': False})
    n.write_csv(ROOT / 'manifest/INPUT_SHA256.csv', inputs)
    n.write_csv(ROOT / 'audit/COSINE_AND_BASELINE_UNITS.csv', units)
    shutil.copy2(__file__, ROOT / 'scripts/profile_single_waveform.py')
    n.write_json(ROOT / 'contracts/START_FREEZE.json', {'UTC': n.utc(),
        'contract_sha256': n.sha(ROOT / 'contracts/ANALYSIS_CONTRACT.json'), 'script_sha256': n.sha(Path(__file__))})


def calibrate():
    global ACTIVE_FIT
    configs, rows = [], []
    for dep in n.DEPS:
        for seed in n.SEEDS:
            original = h.isolated.ORIGINALS[dep, seed]
            base = {**original, 'method': METHODS[0]}
            configs.append(base)
            frame = h.load_panel(dep, seed, 'validation')
            zbase = infer(frame, base)[0]
            fm = n.cf.fast_metrics(frame, n.cf.channels(frame, zbase) @ np.asarray(original['weights']))
            wm = n.cf.fast_metrics(frame, zbase)
            for active in (False, True):
                ACTIVE_FIT = active
                for kind in ('LINEAR', 'TREE'):
                    spec = fit_classifier(dep, seed, kind)
                    method = ('SINGLE-WF-ACTIVE-' if active else 'SINGLE-WF-') + kind
                    c = {**original, 'method': method, 'waveform_calibrator': spec, 'active_only': active}
                    z, oo, cl = infer(frame, c)
                    values = n.cf.fast_metrics(frame, n.cf.channels(frame, z) @ np.asarray(original['weights']))
                    wvalues = n.cf.fast_metrics(frame, z)
                    c.update(tune_guard=n.cf.guard(values, fm) and n.cf.guard(wvalues, wm), tune_metrics=values)
                    bad = frame.copy()
                    bad['pair_key'], bad['pe_mc_bhattacharyya_coefficient'], bad['official_po_fpp'] = 'ignored', -99., 1.
                    if not np.array_equal(z, infer(bad, c)[0]):
                        raise RuntimeError('Forbidden metadata affected score')
                    configs.append(c)
                    rows.append({'deployment': dep, 'seed': seed, 'method': method, 'tune_guard': c['tune_guard'],
                                 'calibration_logloss': spec['selected']['logloss'], 'fit_sources': spec['fit_sources'],
                                 'tune_sources': spec['tune_sources'], 'forbidden_input_delta': 0., **values})
                    print('SINGLE_WAVEFORM_SELECTION', dep, seed, method, spec['selected'], c['tune_guard'], flush=True)
    n.write_csv(ROOT / 'tables/SINGLE_WAVEFORM_CALIBRATION.csv', rows)
    n.write_json(ROOT / 'configs/SELECTED_CONFIGURATIONS.json', configs)
    n.write_json(ROOT / 'contracts/CONFIGURATIONS_FROZEN.json', {'UTC': n.utc(),
        'sha256': n.sha(ROOT / 'configs/SELECTED_CONFIGURATIONS.json'), 'no_real_or_test_selection': True})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--stage', choices=('freeze', 'calibrate', 'evaluate', 'real'), required=True)
    args = parser.parse_args()
    ROOT = h.ROOT = co.ROOT = r.ROOT = co.score.ROOT = args.root
    r.install()
    co.score.METHODS = METHODS
    co.score.matrices = co.matrices
    n.METHODS, n.load_panel, n.infer = METHODS, h.load_panel, infer
    old_export = n.public_frame
    old_consensus = n.dev.BASE.consensus_real
    def export(frame, z, weights, method):
        out = old_export(frame, z, weights, method)
        for name in h.FEATURES:
            out[name] = frame[name].to_numpy()
        return out
    def consensus(items, method):
        out = old_consensus(items, method)
        extra = pd.concat(items).groupby('pair_key')[list(h.FEATURES)].mean().add_suffix('_mean').reset_index()
        return out.merge(extra, on='pair_key', validate='one_to_one')
    n.public_frame, n.dev.BASE.consensus_real = export, consensus
    if args.stage in ('freeze', 'calibrate'):
        globals()[args.stage]()
    else:
        n.run(ROOT, args.stage)
