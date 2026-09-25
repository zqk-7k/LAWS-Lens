#!/usr/bin/env python3
"""R66: one NODUP-anchored, population-weighted waveform classifier."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

P = Path('/root/autodl-tmp/gw-catalog')
sys.path.insert(0, str(P / 'scripts/experiments'))
import mcwf_global_branch_calibration_plan_20260910 as population
import mcwf_reference_regularized_catalog_20260910 as replay

n = replay.n
ARMS = ('ND-ANCHORED-LINEAR', 'ND-ANCHORED-MONOTONE-HINGES')
RIDGES = (1e-5, 3e-5, 1e-4, 3e-4, .001, .003, .01, .03, .1, .3, 1., 3., 10.)
BASE = P / 'results/mcwf_nodup_nomix_01_20260909T071108Z'
BOOTSTRAPS = 2000


def original_configs():
    return {(c['deployment'], c['seed']): c for c in json.loads(
        (BASE / 'configs/SELECTED_CONFIGURATIONS.json').read_text()) if c['method'] == 'NODUP-DIRECT'}


def nodup_terms(cosine, bc, endpoint, original, recipe):
    spec = recipe['joint_config']['calibration']
    bc = np.asarray(bc, float).clip(1e-15, 1.)
    logbc = np.log(bc)
    tail = n.ev.tail.tail_probability(-logbc, spec['joint_reference'])
    outside = endpoint | (logbc < spec['minimum']) | (logbc > spec['maximum'])
    penalty = np.minimum(np.log(tail / .05), 0.)
    increment = np.interp(logbc, spec['knots'], spec['loglr'])
    increment = np.where(outside | (tail < .05), np.minimum(increment, 0.), increment).clip(-4., 4.)
    penalty = np.where(endpoint, 0., penalty)
    increment = np.where(endpoint, 0., increment)
    f = pd.DataFrame({'embedding_only': cosine, 'joint_penalty': penalty,
                      'joint_increment': increment, 'joint_ood': outside})
    z, ood, clipped = n.infer(f, original)
    return z, ood, clipped


def raw_joint_bc(path, i, j):
    ids = np.unique(np.r_[i, j])
    with np.load(path) as bank:
        outside = bank['outside']
        # Keep the frozen NODUP mass convention; do not recalibrate its input.
        raw = bank['joint'][ids].astype(float).reshape(len(ids), -1)
    if np.any(raw < 0.) or not np.isfinite(raw).all():
        raise RuntimeError('Invalid frozen neural probability')
    sums = raw.sum(1)
    if np.any(sums <= 0.):
        raise RuntimeError('Empty frozen neural probability')
    np.sqrt(raw, out=raw)
    ii, jj = np.searchsorted(ids, i), np.searchsorted(ids, j)
    value = np.asarray([np.dot(raw[a], raw[b]) for a, b in zip(ii, jj)]).clip(1e-15, 1.)
    return value, (outside[i] > .25) | (outside[j] > .25), float(abs(sums - 1.).max())


def feature_matrix(z, power, deficit, spec):
    d = np.log1p(np.maximum(deficit, spec['deficit_support'][0]))
    x = np.column_stack([z, np.log(power), -d])
    if spec['arm'] == ARMS[1]:
        x = np.column_stack([x, *[-np.maximum(d - k, 0.) for k in spec['hinge_knots']]])
    return x


def objective(beta, design, labels, weights, center, ridge):
    score = design @ beta
    diff = beta - center
    value = np.dot(weights, population.binary_loss(labels, score)) + .5 * ridge * np.dot(diff, diff)
    grad = design.T @ (weights * (expit(score) - labels)) + ridge * diff
    return value, grad


def train_model(x, y, w, ridge):
    mean = np.average(x, axis=0, weights=w)
    scale = np.sqrt(np.average((x - mean)**2, axis=0, weights=w))
    if np.any(scale <= 0.) or not np.isfinite(x).all():
        raise RuntimeError('Degenerate conditional feature')
    design = np.column_stack([np.ones(len(x)), (x - mean) / scale])
    center = np.r_[mean[0], scale[0], np.zeros(x.shape[1] - 1)]
    bounds = [(None, None), (0., None), (None, None)] + [(0., None)] * (x.shape[1] - 2)
    result = minimize(objective, center.copy(), args=(design, y, w, center, ridge), jac=True,
                      method='L-BFGS-B', bounds=bounds,
                      options={'maxiter': 4000, 'ftol': 1e-13, 'gtol': 1e-9})
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError('Conditional optimizer failed: ' + str(result.message))
    return {'mean': mean.tolist(), 'scale': scale.tolist(), 'coefficients': result.x.tolist(),
            'ridge': ridge, 'center': center.tolist(), 'no_update': False,
            'optimizer_message': str(result.message),
            'raw_slopes': (result.x[1:] / scale).tolist(),
            'raw_intercept': float(result.x[0] - np.dot(result.x[1:] / scale, mean))}


def apply(z, power, deficit, spec):
    z = np.asarray(z, float)
    if spec.get('no_update', False):
        return z.copy(), np.zeros(len(z), bool), np.zeros(len(z), bool)
    x = feature_matrix(z, power, deficit, spec)
    raw = spec['coefficients'][0] + ((x - spec['mean']) / spec['scale']) @ np.asarray(spec['coefficients'][1:])
    lo, hi = spec['nodup_support']
    pl, ph = spec['minimum_power_support']
    # Exact fallback outside measured support; no extrapolated positive or negative evidence.
    ood = ((z < lo) | (z > hi) | (power < pl) | (power > ph) |
           (deficit > spec['deficit_support'][1]))
    value = raw.clip(-16., 16.)
    value[ood] = z[ood]
    return value, ood, (raw != value) & ~ood


def units():
    rng = np.random.default_rng(2026091066)
    rows = []
    for dim in (3, 5):
        x = rng.normal(size=(60, dim))
        w = rng.uniform(.0001, .005, len(x))
        y = rng.integers(0, 2, len(x))
        mean, scale = np.mean(x, 0), np.std(x, 0)
        a = np.column_stack([np.ones(len(x)), (x - mean) / scale])
        center = np.r_[mean[0], scale[0], np.zeros(dim - 1)]
        if np.max(abs(a @ center - x[:, 0])) > 1e-12:
            raise RuntimeError('NODUP reference coordinate transport failed')
        beta = center + rng.normal(0., .1, dim + 1)
        _, grad = objective(beta, a, y, w, center, .003)
        numeric = [(objective(beta + np.eye(dim + 1)[i] * 1e-5, a, y, w, center, .003)[0] -
                    objective(beta - np.eye(dim + 1)[i] * 1e-5, a, y, w, center, .003)[0]) / 2e-5 for i in range(dim + 1)]
        err = float(np.max(abs(np.asarray(numeric) - grad)))
        if err > 1e-8:
            raise RuntimeError('Objective gradient failed')
        rows.append({'dimension': dim, 'gradient_error': err, 'identity_transport': True})
    z = rng.normal(size=40)
    assert np.array_equal(apply(z, np.ones(40), np.ones(40), {'no_update': True})[0], z)
    return {'gradient_units': rows, 'identity_bit_exact': True,
            'input_API_no_PE_time_sky_or_old_heads': True}


def freeze(root, data, expansion, global_root, catalog):
    if root.exists():
        raise RuntimeError('Independent R66 output required')
    for folder in ('contracts', 'configs', 'tables', 'audit', 'reports', 'scripts', 'logs', 'manifest', 'figures'):
        (root / folder).mkdir(parents=True)
    contract = {'UTC': population.utc(), 'id': 'MCWF-NODUP-CONDITIONAL-DEFICIT-66',
        'status': n.STATUS, 'goal_achieved': False, 'adaptive_development': True,
        'data': str(data), 'expansion': str(expansion), 'global_population': str(global_root),
        'catalog_reference': str(catalog), 'arms': ARMS, 'ridge_grid': RIDGES,
        'question': 'Does shared-profile deficit add predictive information conditional on the current NODUP waveform score, instead of discarding that score or treating the deficit as independent evidence?',
        'features': 'NODUP waveform score, log minimum independent projection power, -log1p shared deficit. Monotone-hinges arm adds two negative hinge features at fit-true50% and90% log-deficit quantiles.',
        'model': 'ONE logistic waveform classifier; ridge toward exact NODUP in fit-standardized coordinates. Zero physical-feature coefficients and unit NODUP slope give the exact reference. Not an old/new total-score blend and no separate duplicate Mc/q head.',
        'monotonicity': 'Nonnegative coefficient for NODUP and each negative deficit coordinate; strength and intercept unconstrained. Increasing deficit cannot increase the supported raw score at fixed NODUP/power.',
        'population': 'Exactly R64 full-source-pair multiplicity and HT sampling weights, not class balancing within the eligible branch. Unchanged fallback cancels from global risk differences.',
        'fit': 'R22B source/noise-parent-disjoint fold0 only; per-run/model coefficients. Frozen shared projection measurements and original NODUP neural predictions/calibrations, no profile-updated mass density.',
        'selection': 'For each arm/run/model minimize fold1 global balanced risk contribution over fixed ridge grid plus exact identity. Tolerance1e-12; ties identity then stronger ridge.',
        'support': 'R55 eligible endpoints, power support and successful shared fit. NODUP support is fit min/max. Deficit below existing lower boundary saturates; above existing upper boundary or outside NODUP/power support EXACT NODUP fallback. Final supported score cap16. No unsupported veto.',
        'pilot_gate': 'Every run/model fold1 risk delta<=1e-12, strict mean improvement eachrun, and model-mean source-bootstrap one-sided95% upper<=0. Both runs required. Among passing arms minimize worst-run relative risk delta then mean then linear.',
        'bootstrap': {'draws': BOOTSTRAPS, 'seed': 2026091066, 'unit': 'full source doublet',
            'limits': 'Conditional repeated-development bootstrap; no independent noise/design variance or new blinded confirmation.'},
        'no_catalog_or_real_input_for_fit_selection': True,
        'later_full_goal': 'Original per-model/panel waveform+fusion guards, consensus+per-model Top10/20 PE/official nonloss, keypair outside every Top10. A pilot PASS does not establish the goal.',
        'frozen': ['encoder checkpoints', 'new learned Mc/eta/chi distributions', 'time', 'sky', 'outer weights', 'scope/splits', 'all previous results'],
        'limitations': 'Projection deficit is not normalized log likelihood or PE; logistic score is not a lensing Bayes factor. Population uses per-image SNR-scaled injections and reused lens environments. All ongoing real feedback is adaptive development.',
        'references': [{'url': 'https://arxiv.org/abs/1506.02169',
            'use': 'Joint discriminative calibration of correlated summaries; no theorem guarantees this finite GW classifier.'},
            {'url': 'https://www.jmlr.org/papers/v11/cawley10a.html',
             'use': 'Repeated validation use and selection bias must remain disclosed.'}]}
    population.write(root / 'contracts/ANALYSIS_CONTRACT.json', contract)
    files = [Path(__file__), Path(population.__file__), Path(replay.__file__), Path(n.__file__),
             BASE / 'configs/SELECTED_CONFIGURATIONS.json', n.FRESH / 'contracts/FRESH_FREEZE.json',
             expansion / 'configs/FROZEN_REFERENCE_CONFIGS.json', expansion / 'tables/PAIR_RESULTS.parquet',
             global_root / 'tables/POPULATION_WEIGHTED_PAIR_PLAN.parquet',
             global_root / 'tables/POPULATION_AUDIT.csv', global_root / 'contracts/PILOT_GATE.json',
             catalog / 'configs/SELECTED_CONFIGURATIONS.json']
    for r in pd.read_csv(global_root / 'manifest/INPUT_SHA256.csv').itertuples():
        files.append(Path(r.path))
    for dep in n.DEPS:
        for seed in n.SEEDS:
            slot = n.recipes()[dep, seed]['slot']
            files += [data / f'predictions/{dep}/{seed}_embedding.npy',
                      data / f'predictions/{dep}/{slot}_parent.npz',
                      replay.joint_path(dep, seed, 'validation'),
                      replay.archive_path(catalog, 'NODUP-DIRECT-REPLAY', dep, seed, 'validation')]
    population.snapshot(root, files)
    population.write(root / 'audit/NUMERICAL_UNITS.json', units())
    shutil.copy2(__file__, root / 'scripts/nodup_conditional_deficit.py')
    population.write(root / 'contracts/START_FREEZE.json', {'UTC': population.utc(),
        'runtime_sha256': population.sha(Path(__file__)),
        'contract_sha256': population.sha(root / 'contracts/ANALYSIS_CONTRACT.json'),
        'manifest_sha256': population.sha(root / 'manifest/INPUT_SHA256.csv')})
    print('CONDITIONAL_DEFICIT_FROZEN', root, flush=True)


def check(root):
    c = json.loads((root / 'contracts/START_FREEZE.json').read_text())
    for key, file in [('runtime_sha256', Path(__file__)),
                      ('contract_sha256', root / 'contracts/ANALYSIS_CONTRACT.json'),
                      ('manifest_sha256', root / 'manifest/INPUT_SHA256.csv')]:
        if c[key] != population.sha(file):
            raise RuntimeError('Frozen runtime or contract changed')
    rows = pd.read_csv(root / 'manifest/INPUT_SHA256.csv')
    for r in rows.itertuples():
        if population.sha(r.path) != r.sha256:
            raise RuntimeError('Frozen input changed ' + r.path)
    return len(rows)


def fit(root):
    count = check(root)
    if (root / 'contracts/PILOT_GATE.json').exists():
        raise RuntimeError('Completed R66 is immutable')
    tick = time.monotonic()
    c = json.loads((root / 'contracts/ANALYSIS_CONTRACT.json').read_text())
    data, exp, glob, cat = [Path(c[k]) for k in ('data', 'expansion', 'global_population', 'catalog_reference')]
    pairs = pd.read_parquet(exp / 'tables/PAIR_RESULTS.parquet')
    inventory = pd.read_csv(exp / 'tables/PLANNING_INVENTORY.csv')
    sealed = pd.read_parquet(glob / 'tables/POPULATION_WEIGHTED_PAIR_PLAN.parquet')
    refs = {(r['deployment'], r['seed']): r for r in json.loads((exp / 'configs/FROZEN_REFERENCE_CONFIGS.json').read_text())}
    originals = original_configs()
    grids, predictions, metrics, configs, checks, bootstrap, runboot = [], [], [], {}, [], [], {}
    for dep_index, dep in enumerate(n.DEPS):
        full, _, structures = population.population(data, dep, pairs, inventory)
        compare = sealed[sealed.deployment == dep].set_index('pair_id').loc[full.pair_id]
        for col in ('global_balanced_weight', 'global_HT_weight', 'full_sourcepair_multiplicity'):
            if not np.array_equal(full[col], compare[col]):
                raise RuntimeError('Population accounting differs from R64')
        ref = refs[dep, n.SEEDS[0]]
        f = full[full.minimum_independent_power.between(*ref['minimum_power_support']) & full.any_shared_converged].reset_index(drop=True)
        y, w = f.kind.eq('true').to_numpy(float), f.global_balanced_weight.to_numpy(float)
        train, tune = f.fold.eq(0).to_numpy(), f.fold.eq(1).to_numpy()
        power, deficit = f.minimum_independent_power.to_numpy(float), f.deficit.to_numpy(float)
        bweights = population.bootstrap_weights(f.loc[tune].reset_index(drop=True), *structures[1],
                                                np.random.default_rng(2026091066 + dep_index))
        for seed in n.SEEDS:
            original, recipe = originals[dep, seed], n.recipes()[dep, seed]
            slot = recipe['slot']
            embedding = np.load(data / f'predictions/{dep}/{seed}_embedding.npy').astype(float)
            embedding /= np.linalg.norm(embedding, axis=1, keepdims=True)
            i, j = f.idx_i.to_numpy(int), f.idx_j.to_numpy(int)
            cosine = np.sum(embedding[i] * embedding[j], axis=1)
            bc, end, masserror = raw_joint_bc(data / f'predictions/{dep}/{slot}_parent.npz', i, j)
            z, _, _ = nodup_terms(cosine, bc, end, original, recipe)
            old = pd.read_parquet(replay.archive_path(cat, 'NODUP-DIRECT-REPLAY', dep, seed, 'validation'))
            oi, oj = old.idx_i.to_numpy(int), old.idx_j.to_numpy(int)
            # Verify frozen equations against known validation output before any new fitting.
            with np.load(replay.joint_path(dep, seed, 'validation')) as bank:
                outside = bank['outside']
            rz, _, _ = nodup_terms(old.embedding_only.to_numpy(float), old.joint_BC.to_numpy(float),
                                  (outside[oi] > .25) | (outside[oj] > .25), original, recipe)
            error = float(abs(rz - old.waveform_score.to_numpy(float)).max())
            if error > 1e-10:
                raise RuntimeError('Frozen NODUP formula replay failed ' + str(error))
            checks.append({'deployment': dep, 'seed': seed, 'validation_replay_max_error': error,
                           'unchanged_neural_joint_mass_error': masserror})
            base = float(np.dot(w[tune], population.binary_loss(y[tune], z[tune])))
            arm_scores = {'NODUP': ({'no_update': True}, z)}
            for arm in ARMS:
                knots = np.quantile(np.log1p(np.maximum(deficit[train & (y == 1)], ref['deficit_support'][0])), [.5, .9]).tolist()
                shell = {'arm': arm, 'minimum_power_support': ref['minimum_power_support'],
                         'deficit_support': ref['deficit_support'], 'hinge_knots': knots if arm == ARMS[1] else [],
                         'nodup_support': [float(z[train].min()), float(z[train].max())],
                         'deployment': dep, 'seed': seed, 'weights': original['weights']}
                identity = {**shell, 'no_update': True, 'ridge': None}
                options = [(base, True, float('inf'), identity, z)]
                grids.append({'deployment': dep, 'seed': seed, 'arm': arm, 'ridge': None, 'no_update': True,
                              'risk': base, 'risk_delta': 0., 'ood_rate': 0.})
                x = feature_matrix(z, power, deficit, shell)
                for ridge in RIDGES:
                    spec = {**shell, **train_model(x[train], y[train], w[train], ridge)}
                    value, outside, _ = apply(z, power, deficit, spec)
                    risk = float(np.dot(w[tune], population.binary_loss(y[tune], value[tune])))
                    grids.append({'deployment': dep, 'seed': seed, 'arm': arm, 'ridge': ridge,
                                  'no_update': False, 'risk': risk, 'risk_delta': risk - base,
                                  'ood_rate': float(outside[tune].mean())})
                    options.append((risk, False, ridge, spec, value))
                best = min(v[0] for v in options)
                chosen = min([v for v in options if v[0] <= best + 1e-12], key=lambda v: (not v[1], -v[2]))
                spec, value = chosen[3:]
                arm_scores[arm] = (spec, value)
                configs[f'{dep}/{seed}/{arm}'] = spec
                if not spec['no_update']:
                    d = np.linspace(*spec['deficit_support'], 501)
                    zz = np.full(len(d), np.mean(spec['nodup_support']))
                    pp = np.full(len(d), np.sqrt(np.prod(spec['minimum_power_support'])))
                    curve = apply(zz, pp, d, spec)[0]
                    if np.max(np.diff(curve)) > 1e-10:
                        raise RuntimeError('Supported deficit monotonicity failed')
                print('CONDITIONAL_FIT', dep, seed, arm, 'ridge', spec['ridge'], 'delta', chosen[0] - base, flush=True)
            for arm, (spec, value) in arm_scores.items():
                for fold in (0, 1):
                    use = f.fold.eq(fold).to_numpy()
                    loss = float(np.dot(w[use], population.binary_loss(y[use], value[use])))
                    base_loss = float(np.dot(w[use], population.binary_loss(y[use], z[use])))
                    metrics.append({'deployment': dep, 'seed': seed, 'arm': arm, 'fold': fold,
                        'eligible_global_risk': loss, 'nodup_global_risk': base_loss,
                        'global_risk_delta': loss - base_loss, 'no_update': spec['no_update']})
                delta = population.binary_loss(y[tune], value[tune]) - population.binary_loss(y[tune], z[tune])
                draw = bweights @ delta
                runboot[dep, seed, arm] = draw
                bootstrap.append({'deployment': dep, 'seed': str(seed), 'arm': arm,
                    'delta_mean': float(np.dot(w[tune], delta)), 'ci_lower': float(np.quantile(draw, .025)),
                    'ci_upper': float(np.quantile(draw, .975)), 'upper95': float(np.quantile(draw, .95))})
                out = f[['pair_id', 'deployment', 'fold', 'kind', 'source_i', 'source_j', 'group_i', 'group_j',
                         'global_balanced_weight', 'global_HT_weight', 'deficit', 'minimum_independent_power']].copy()
                out['seed'], out['arm'], out['waveform_score'], out['NODUP_waveform'] = seed, arm, value, z
                out['joint_BC'], out['embedding_only'], out['joint_endpoint_ood'] = bc, cosine, end
                predictions.append(out)
    metric = pd.DataFrame(metrics)
    gates = []
    for arm in ARMS:
        for dep in n.DEPS:
            a = metric[(metric.deployment == dep) & (metric.arm == arm) & (metric.fold == 1)]
            draw = np.mean([runboot[dep, seed, arm] for seed in n.SEEDS], axis=0)
            upper = float(np.quantile(draw, .95))
            passed = bool((a.global_risk_delta <= 1e-12).all() and a.global_risk_delta.mean() < -1e-12 and upper <= 0.)
            gates.append({'deployment': dep, 'arm': arm, 'mean_delta': float(a.global_risk_delta.mean()),
                'relative_delta': float(a.global_risk_delta.mean() / a.nodup_global_risk.mean()),
                'upper95': upper, 'run_pass': passed})
            bootstrap.append({'deployment': dep, 'seed': 'MODEL_MEAN', 'arm': arm,
                'delta_mean': float(a.global_risk_delta.mean()), 'ci_lower': float(np.quantile(draw, .025)),
                'ci_upper': float(np.quantile(draw, .975)), 'upper95': upper})
    passing = [arm for arm in ARMS if all(g['run_pass'] for g in gates if g['arm'] == arm)]
    selected = min(passing, key=lambda arm: (max(g['relative_delta'] for g in gates if g['arm'] == arm),
                   np.mean([g['relative_delta'] for g in gates if g['arm'] == arm]), ARMS.index(arm))) if passing else None
    for name, rows in [('CALIBRATION_GRID', grids), ('CALIBRATION_METRICS', metrics),
                       ('SOURCE_BOOTSTRAP', bootstrap), ('PILOT_GATES', gates), ('NODUP_REPLAY_AUDIT', checks)]:
        population.csv(root / f'tables/{name}.csv', rows)
    pd.concat(predictions).to_parquet(root / 'tables/PREDICTIONS.parquet', index=False)
    population.write(root / 'configs/ALL_CONDITIONAL_MODELS.json', configs)
    verified = check(root)
    population.write(root / 'audit/INPUT_HASH_RECHECK.json', {'UTC': population.utc(), 'count': verified,
                                                           'initial_count': count, 'failures': 0})
    gate = {'UTC': population.utc(), 'gate': 'PASS' if selected else 'FAIL', 'selected_common_arm': selected,
            'passing_arms': passing, 'status': n.STATUS, 'goal_achieved': False,
            'catalog_or_real_scored': False, 'seconds': time.monotonic() - tick}
    population.write(root / 'contracts/PILOT_GATE.json', gate)
    print(json.dumps(gate, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--stage', choices=('freeze', 'fit'), required=True)
    for name in ('data', 'expansion', 'global-root', 'catalog'):
        parser.add_argument('--' + name, type=Path)
    a = parser.parse_args()
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=1)
    if a.stage == 'freeze':
        freeze(a.root, a.data, a.expansion, a.global_root, a.catalog)
    else:
        fit(a.root)
