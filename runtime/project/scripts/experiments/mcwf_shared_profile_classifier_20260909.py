#!/usr/bin/env python3
"""Simulation-only feasibility calibration for the frozen shared-profile pilot."""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
import argparse
from pathlib import Path
import json
import shutil
import sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import average_precision_score, roc_auc_score

P = Path('/root/autodl-tmp/gw-catalog')
sys.path.insert(0, str(P/'scripts/experiments'))
import mcwf_shared_profile_coherence_pilot_20260909 as pilot
n = pilot.n
RIDGES = (.001, .01, .1, 1.)
ARMS = ('SEPARATE-PROFILE-GAP', 'SHARED-PROFILE-DEFICIT')
ROOT = DATA = None


def freeze():
    target = ROOT/'contracts/CLASSIFIER_CONTRACT.json'
    if target.exists():
        raise RuntimeError('Do not overwrite a classifier contract')
    if (ROOT/'tables/PAIR_RESULTS.parquet').exists():
        raise RuntimeError('Freeze classifier before seeing pilot outcome')
    n.write_json(target, {
        'UTC': n.utc(), 'stage': 'R50 simulation-only proper-score pilot',
        'arms': ARMS, 'ridges': RIDGES, 'truth_used_only_as_simulated_labels': True,
        'inputs': ['frozen embedding cosine', 'log minimum independent profile power',
                   'negative absolute fitted logMc gap OR negative log1p shared deficit'],
        'fit_fold': 0, 'selection_fold': 1, 'test_and_real_forbidden': True,
        'training_rows': 'true plus random_null only',
        'standardization': 'fit-fold class-balanced weighted mean and population SD',
        'coefficients': 'intercept free; standardized cosine/consistency coefficients >=0; strength free',
        'loss': 'weighted mean logistic cross entropy plus ridge/2 times slope squared norm',
        'weights': 'true pair1; null pair1/sqrt(source_i_degree*source_j_degree), classes normalized to mass0.5 each',
        'ridge_selection': 'lowest tune random-null balanced logloss; exact ties choose larger ridge',
        'hard_null': 'tune diagnostic only, neither fitting nor ridge selection',
        'gate': 'finite all480, unit pass, tune logloss new<=comparator in random/hard separately in bothruns averaged3encoders; strict improvement at leastone run',
        'bootstrap': {'draws': 2000, 'seed': 2026090951,
                      'unit': 'all tune source groups resampled together across methods and model seeds',
                      'true_pair_multiplier': 'source multiplicity',
                      'null_pair_multiplier': 'endpoint multiplicity product',
                      'limits': 'conditional fixed-fit pilot uncertainty, not post-selection or catalog evidence'},
        'no_arbitrary_threshold_search': True, 'no_old_Mc_q_score_or_total_blend': True,
        'same_method_both_runs': True, 'goal_achieved': False, 'status': n.STATUS})
    shutil.copy2(__file__, ROOT/'scripts/shared_profile_classifier.py')
    n.write_json(ROOT/'contracts/CLASSIFIER_FROZEN.json', {
        'UTC': n.utc(), 'contract_sha256': n.sha(target),
        'runtime_sha256': n.sha(Path(__file__))})


def base_weights(frame):
    true = frame.kind.eq('true').to_numpy()
    if not true.any() or true.all():
        raise RuntimeError('Both simulated classes required')
    degree = pd.concat([frame.loc[~true, 'source_i'], frame.loc[~true, 'source_j']]).value_counts()
    w = np.ones(len(frame))
    w[~true] = 1/np.sqrt(frame.loc[~true, 'source_i'].map(degree).to_numpy(float) *
                         frame.loc[~true, 'source_j'].map(degree).to_numpy(float))
    return true.astype(float), w


def balanced(y, w):
    result = np.zeros(len(y), float)
    for value in (0., 1.):
        take = y == value
        mass = w[take].sum()
        if mass <= 0:
            raise RuntimeError('Bootstrap class has zero mass')
        result[take] = .5*w[take]/mass
    return result


def loss(y, logits, weights):
    return float(np.dot(weights, np.logaddexp(0., logits)-y*logits))


def fit(x, y, w, ridge):
    mean = np.average(x, axis=0, weights=w)
    scale = np.sqrt(np.average((x-mean)**2, axis=0, weights=w))
    if not np.isfinite(x).all() or (scale <= 0).any():
        raise RuntimeError('Degenerate classifier inputs')
    design = np.column_stack([np.ones(len(x)), (x-mean)/scale])

    def objective(coef):
        z = design@coef
        probability = 1/(1+np.exp(-np.clip(z, -700., 700.)))
        gradient = design.T@(w*(probability-y))
        gradient[1:] += ridge*coef[1:]
        return loss(y, z, w)+.5*ridge*np.dot(coef[1:], coef[1:]), gradient

    result = minimize(objective, np.zeros(4), jac=True, method='L-BFGS-B',
                      bounds=[(None, None), (0., None), (None, None), (0., None)],
                      options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-8})
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError(f'Classifier optimizer failed: {result.message}')
    return {'ridge': ridge, 'mean': mean.tolist(), 'scale': scale.tolist(),
            'coefficients': result.x.tolist(), 'optimizer_message': str(result.message)}


def predict(x, spec):
    return np.column_stack([np.ones(len(x)), (x-spec['mean'])/spec['scale']])@spec['coefficients']


def calibrate():
    for filename in ('UNIT_PASS.json', 'CLASSIFIER_FROZEN.json', 'MEASUREMENT_COMPLETE.json'):
        if not (ROOT/'contracts'/filename).exists():
            raise RuntimeError(f'Missing prerequisite {filename}')
    frozen = json.loads((ROOT/'contracts/CLASSIFIER_FROZEN.json').read_text())
    if frozen['runtime_sha256'] != n.sha(Path(__file__)):
        raise RuntimeError('Frozen classifier code changed')
    frame = pd.read_parquet(ROOT/'tables/PAIR_RESULTS.parquet')
    if len(frame) != 480 or frame.status.ne('COMPLETE').any():
        raise RuntimeError('Every preselected profile pair must complete')
    configs = {}; results = []; tuning = []; predictions = []
    for dep in n.DEPS:
        panel = frame[frame.deployment.eq(dep)].reset_index(drop=True)
        train = panel.fold.eq(0)&panel.kind.ne('hard_null')
        validation = panel.fold.eq(1)&panel.kind.ne('hard_null')
        y, w0 = base_weights(panel[train]); w = balanced(y, w0)
        vy, vw0 = base_weights(panel[validation]); vw = balanced(vy, vw0)
        for seed in n.SEEDS:
            embedding = np.load(DATA/f'predictions/{dep}/{seed}_embedding.npy').astype(float)
            norms = np.linalg.norm(embedding, axis=1)
            if not np.isfinite(embedding).all() or (norms <= 0).any():
                raise RuntimeError('Invalid frozen waveform embedding')
            embedding /= norms[:, None]
            cosine = np.einsum('ij,ij->i', embedding[panel.idx_i], embedding[panel.idx_j])
            for arm in ARMS:
                consistency = (-panel.profile_logMc_gap.to_numpy() if arm==ARMS[0]
                               else -np.log1p(panel.deficit.to_numpy()))
                x = np.column_stack([cosine, np.log(panel.minimum_independent_power), consistency])
                candidates = []
                for ridge in RIDGES:
                    spec = fit(x[train], y, w, ridge)
                    value = loss(vy, predict(x[validation], spec), vw)
                    candidates.append((value, -ridge, spec))
                    tuning.append({'deployment': dep, 'seed': seed, 'arm': arm,
                                   'ridge': ridge, 'tune_random_logloss': value})
                _, _, selected = min(candidates, key=lambda r: (r[0], r[1]))
                configs[f'{dep}/{seed}/{arm}'] = selected
                logits = predict(x, selected)
                for kind in ('random_null', 'hard_null'):
                    mask = panel.fold.eq(1)&panel.kind.isin(('true', kind))
                    py, pw0 = base_weights(panel[mask]); pw = balanced(py, pw0)
                    z = logits[mask]
                    results.append({'deployment': dep, 'seed': seed, 'arm': arm, 'null_kind': kind,
                        'pairs': int(mask.sum()), 'positive_sources': int(py.sum()),
                        'ridge': selected['ridge'], 'balanced_logloss': loss(py, z, pw),
                        'balanced_ROC_AUC': roc_auc_score(py, z, sample_weight=pw),
                        'balanced_AP_not_catalog_AUPRC': average_precision_score(py, z, sample_weight=pw)})
                out = panel[['pair_id', 'deployment', 'fold', 'kind', 'source_i', 'source_j']].copy()
                out['seed'] = seed; out['arm'] = arm; out['logit'] = logits
                predictions.extend(out.to_dict('records'))
    metrics = pd.DataFrame(results)
    pred = pd.DataFrame(predictions)
    pred.to_parquet(ROOT/'tables/CLASSIFIER_PREDICTIONS.parquet', index=False)
    n.write_csv(ROOT/'tables/CLASSIFIER_METRICS_PER_SEED.csv', results)
    n.write_csv(ROOT/'tables/RIDGE_VALIDATION_PLATFORMS.csv', tuning)
    n.write_json(ROOT/'contracts/SELECTED_PILOT_CLASSIFIERS.json', configs)
    summary = metrics.groupby(['deployment', 'arm', 'null_kind']).balanced_logloss.agg(['mean', 'std']).reset_index()
    n.write_csv(ROOT/'tables/CLASSIFIER_LOGLOSS_SUMMARY.csv', summary.to_dict('records'))
    compare = summary.pivot(index=['deployment', 'null_kind'], columns='arm', values='mean')
    compare['new_minus_separate'] = compare[ARMS[1]]-compare[ARMS[0]]
    compare['nonworse'] = compare.new_minus_separate <= 0
    passed = bool(compare.nonworse.all() and (compare.new_minus_separate < 0).any())
    n.write_csv(ROOT/'tables/PILOT_GATE_COMPARISON.csv', compare.reset_index().to_dict('records'))
    boot = bootstrap(pred)
    n.write_csv(ROOT/'tables/PILOT_PAIRED_SOURCE_BOOTSTRAP.csv', boot)
    n.write_json(ROOT/'contracts/PILOT_GATE.json', {
        'UTC': n.utc(), 'gate': 'PASS' if passed else 'FAIL',
        'all_planned_pairs': len(frame), 'goal_achieved': False,
        'passed_means': 'Development feasibility only;not complete goal or authorization to adopt',
        'real_or_test_read': False, 'status': n.STATUS,
        'frozen_classifiers_sha256': n.sha(ROOT/'contracts/SELECTED_PILOT_CLASSIFIERS.json')})
    print('SHARED_PROFILE_PILOT_GATE', 'PASS' if passed else 'FAIL', flush=True)
    print(compare.to_string(), flush=True)


def bootstrap(pred):
    rows = []; rng = np.random.default_rng(2026090951)
    for dep in n.DEPS:
        subset = pred[pred.deployment.eq(dep)&pred.fold.eq(1)]
        unique = subset.drop_duplicates('pair_id').set_index('pair_id')
        sources = sorted(set(unique.source_i)|set(unique.source_j))
        lookup = {source: index for index, source in enumerate(sources)}
        draws = rng.multinomial(len(sources), np.full(len(sources), 1/len(sources)), size=2000)
        for kind in ('random_null', 'hard_null'):
            data = unique[unique.kind.isin(('true', kind))]
            y, w0 = base_weights(data)
            i = data.source_i.map(lookup).to_numpy(); j = data.source_j.map(lookup).to_numpy()
            losses = {}
            for arm in ARMS:
                seeds = []
                for seed in n.SEEDS:
                    z = subset[subset.arm.eq(arm)&subset.seed.eq(seed)].set_index('pair_id').loc[data.index, 'logit'].to_numpy()
                    seeds.append(np.logaddexp(0., z)-y*z)
                losses[arm] = np.mean(seeds, axis=0)
            differences = []
            for multiplicity in draws:
                multiplier = np.where(y==1, multiplicity[i], multiplicity[i]*multiplicity[j])
                weights = balanced(y, w0*multiplier)
                differences.append(float(weights@(losses[ARMS[1]]-losses[ARMS[0]])))
            rows.append({'deployment': dep, 'null_kind': kind, 'draws': len(differences),
                         'source_groups': len(sources), 'delta_logloss_mean': np.mean(differences),
                         'percentile95_low': np.quantile(differences, .025),
                         'percentile95_high': np.quantile(differences, .975),
                         'conditional_fixed_model_only': True})
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--stage', choices=('freeze', 'calibrate'), required=True)
    args = parser.parse_args(); ROOT, DATA = args.root, args.data_root
    globals()[args.stage]()
