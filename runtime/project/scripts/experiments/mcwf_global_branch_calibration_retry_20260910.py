#!/usr/bin/env python3
"""R64: simulation-only, population-weighted calibration of one waveform branch."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
from datetime import datetime, timezone
import hashlib
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
import mcwf_independent_predictive_calibration_20260909 as predictive

DEPS = ('gwtc3', 'gwtc4')
SEEDS = (202607241, 202607242, 202607243)
RIDGES = (1e-5, 1e-4, .001, .01, .1, 1., 10.)
ARMS = ('R62-FROZEN', 'ELIGIBILITY-OFFSET', 'GLOBAL-AFFINE')
BOOTSTRAPS = 2000
STATUS = 'HOLD_FOR_AUTHOR_REVIEW_NO_ADOPTION_NO_OVERWRITE'


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def plain(x):
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    raise TypeError(type(x).__name__)


def write(path, obj):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(obj, stream, indent=2, ensure_ascii=False, default=plain, allow_nan=False)


def csv(path, frame):
    if Path(path).exists():
        raise RuntimeError('Do not overwrite ' + str(path))
    pd.DataFrame(frame).to_csv(path, index=False, encoding='utf-8-sig')


def binary_loss(y, z):
    return np.logaddexp(0., z) - y * z


def objective(beta, design, y, weights, reference, ridge):
    z = design @ beta
    delta = beta - reference
    value = np.dot(weights, binary_loss(y, z)) + .5 * ridge * np.dot(delta, delta)
    grad = design.T @ (weights * (expit(z) - y)) + ridge * delta
    return value, grad


def fit_affine(z, y, weights, ridge):
    mean = float(np.average(z, weights=weights))
    scale = float(np.sqrt(np.average((z - mean)**2, weights=weights)))
    if not np.isfinite(z).all() or scale <= 0:
        raise RuntimeError('Invalid branch score')
    design = np.column_stack([np.ones(len(z)), (z - mean) / scale])
    reference = np.array([mean, scale])
    result = minimize(objective, reference.copy(), args=(design, y, weights, reference, ridge),
                      jac=True, method='L-BFGS-B', bounds=[(None, None), (1e-8, None)],
                      options={'maxiter': 4000, 'ftol': 1e-13, 'gtol': 1e-9})
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError('Affine optimizer: ' + str(result.message))
    return {'intercept': float(result.x[0] - result.x[1] * mean / scale),
            'slope': float(result.x[1] / scale), 'ridge': ridge, 'no_update': False,
            'standardized_coefficients': result.x.tolist(), 'mean': mean, 'scale': scale,
            'reference_coefficients': reference.tolist(), 'optimizer_message': str(result.message)}


def apply(z, deficit, spec, support):
    if spec.get('no_update'):
        return np.asarray(z, dtype=float).copy()
    out = np.clip(spec['intercept'] + spec['slope'] * np.asarray(z), -16., 16.)
    high = np.asarray(deficit) > support[1]
    out[high] = np.minimum(out[high], 0.)
    return out


def units():
    rng = np.random.default_rng(2026091064)
    d = np.column_stack([np.ones(50), rng.normal(size=50)])
    y = rng.integers(0, 2, 50)
    w = rng.uniform(.001, .01, 50)
    beta, center = np.array([.2, .7]), np.array([.1, 1.])
    value, grad = objective(beta, d, y, w, center, .03)
    numeric = [(objective(beta + np.eye(2)[k] * 1e-5, d, y, w, center, .03)[0] -
                objective(beta - np.eye(2)[k] * 1e-5, d, y, w, center, .03)[0]) / 2e-5 for k in range(2)]
    error = float(np.max(np.abs(grad - numeric)))
    assert error < 1e-8 and np.isfinite(value)
    z = np.array([-4., 0., 7., 12.])
    ident = {'no_update': True}
    assert np.array_equal(z, apply(z, [0., 1., 2., 3.], ident, [1., 2.]))
    out = apply(z, [0., 1., 2., 3.], {'slope': 2., 'intercept': 1.}, [1., 2.])
    assert np.array_equal(out, [-7., 1., 15., 0.])
    # The unchanged branch cancels; eligible risk is not absolute global risk.
    labels = rng.integers(0, 2, 100)
    old = rng.normal(size=100)
    new = old.copy(); new[:30] += .5
    allweights = rng.uniform(.001, .01, 100)
    full = np.dot(allweights, binary_loss(labels, new) - binary_loss(labels, old))
    active = np.dot(allweights[:30], binary_loss(labels[:30], new[:30]) - binary_loss(labels[:30], old[:30]))
    assert abs(full - active) < 1e-15
    # Four image pairs represent one source pair, even when only one is eligible.
    assert np.sum(np.full(4, .25)) == 1. and .25 < 1.
    return {'gradient_max_error': error, 'identity_bit_exact': True,
            'cap_and_high_D_neutral': True, 'unchanged_branch_risk_cancels': True,
            'full_image_multiplicity_not_eligible_multiplicity': True}


def snapshot(root, files):
    rows = []
    for p in sorted({Path(p).resolve() for p in files}):
        if not p.is_file():
            raise RuntimeError('Missing frozen input ' + str(p))
        rows.append({'path': str(p), 'sha256': sha(p), 'bytes': p.stat().st_size})
    csv(root / 'manifest/INPUT_SHA256.csv', rows)


def freeze(root, data, expansion, reference, catalog):
    if root.exists():
        raise RuntimeError('Independent R64 directory required')
    for name in ('contracts', 'configs', 'tables', 'audit', 'reports', 'scripts', 'manifest', 'logs', 'figures'):
        (root / name).mkdir(parents=True)
    contract = {'UTC': utc(), 'id': 'MCWF-GLOBAL-WAVEFORM-BRANCH-CALIBRATION-64',
        'status': STATUS, 'goal_achieved': False, 'adaptive_development': True,
        'data': str(data), 'expansion': str(expansion), 'reference': str(reference),
        'catalog_reference': str(catalog), 'arms': ARMS, 'ridge_grid': RIDGES,
        'motivation': 'The active shared-D classifier was class-balanced inside eligibility E. Its output competes with an unchanged fallback outside E. Audit global population calibration instead of pretending the old complete validation has enough active pairs.',
        'different_from_R23_R24': 'Earlier partition classifiers used at-least-one-valid profile state and marginal predictive features. This experiment uses both-valid endpoints, frozen projection-power support, shared-fit convergence, and the actual pair shared-fit deficit score from R62. Prior failed partition experiments remain failures.',
        'population': 'Each fold has its full source doublets as L. N comprises distinct source pairs with at least one image pair on distinct4096s noise parents. Each allowed image pair gets1/m_full for its source pair, BEFORE quality or power selection.',
        'sampling': 'Reuse R56 exact inclusion probabilities on quality-both pairs. E implies quality-both, so eligible L census and HT null estimates are available without fitting any new strain.',
        'global_risk': 'L weight1/(2*N_L), N weight1/(2*N_N*pi*m_full). No class rebalance inside E. Only E contribution is reported; full global risk difference equals this difference because fallback is unchanged.',
        'offset_control': 'Fit-only log[p(E|L)/p(E|N)] added to eligible R62. Approximate control: old R62 classifier is not an exact conditional LR and used a different within-quality multiplicity.',
        'affine': 'One positive-slope affine calibration of existing eligible R62 waveform score. Fitfold0 global weighted logistic CE plus ridge/2 distance from identity in fit-standardized score coordinates. No total blend.',
        'selection': 'Each run/model selects fold1 minimum eligible contribution to global balanced risk among affine grid and exact identity. Tolerance1e-12; ties identity, then stronger ridge. A single common algorithm for both runs; parameters remain perrun/model.',
        'pilot_gate': 'Every run/model fold1 point risk difference<=1e-12, strict mean gain eachrun, and eachrun model-mean source-bootstrap95% upper percentile<=0. Arms compared independently; if both pass choose affine by predeclared priority. Identity-only is not a gain.',
        'bootstrap': {'draws': BOOTSTRAPS, 'seed': 2026091064, 'unit': 'full source doublet',
            'rule': 'Resample512sources eachfold. True weight multiplied by source multiplicity. Null multiplied by product of endpoint multiplicities; full null-sourcepair denominator is recomputed on fixed adjacency. Keep HT design weights frozen.',
            'limits': 'Conditional repeated-validation bootstrap, not independent test, not noise-block/sampling-design variance, not new training seeds.'},
        'frozen': ['all encoder checkpoints', 'R62 physical branch features and eligibility', 'time', 'sky', 'outer weights', 'event/source/noise scope', 'R55 low-D plateau/high-D positive-neutral/cap16', 'all prior results'],
        'no_old_Mc_q_score_no_total_blend_no_real_selection': True,
        'support': 'No new physical support. Transform an existing bounded waveform score. Apply cap16 and high-D positive-neutral again after transform. Ineligible branch is exact NODUP.',
        'followup': 'Only a passing simulation arm may run complete archived injection/real replay in a new directory. Full original per-model and consensus PE/official and waveform/fusion guards remain required; calibration PASS is not user goal PASS.',
        'limitations': 'Repeated development and validation use. R22 independent source/noisefolds but reused GW-LMC lens environments and per-image targetSNR scaling. Not response-derived lens-rate inference or a physical Bayes factor.',
        'references': [{'url': 'https://arxiv.org/abs/1506.02169',
            'use': 'Calibrated discriminative likelihood-ratio estimation is methodological motivation only; finite affine classifier is not guaranteed a GW Bayes factor.'},
            {'url': 'https://www.jmlr.org/papers/v11/cawley10a.html',
             'use': 'Repeated model selection limits independent performance claims.'}]}
    write(root / 'contracts/ANALYSIS_CONTRACT.json', contract)
    files = [Path(__file__), Path(predictive.__file__), Path(predictive.mode.__file__),
             predictive.PARENT / 'calibration/PROFILE_PREDICTIVE.json',
             expansion / 'tables/PAIR_RESULTS.parquet', expansion / 'contracts/PAIR_PLAN.parquet',
             expansion / 'tables/PLANNING_INVENTORY.csv', expansion / 'configs/FROZEN_REFERENCE_CONFIGS.json',
             reference / 'tables/PREDICTIONS.parquet', reference / 'contracts/PILOT_GATE.json',
             reference / 'configs/ALL_SELECTED_CLASSIFIERS.json', catalog / 'contracts/ANALYSIS_CONTRACT.json',
             catalog / 'configs/SELECTED_CONFIGURATIONS.json']
    for dep in DEPS:
        files += [data / f'data/{dep}/event_metadata.parquet', data / f'data/{dep}/noise/noise_manifest.csv',
                  data / f'predictions/{dep}/ensemble_mass.npy']
        files += list((data / f'profile_events/{dep}').glob('*.json'))
    for module in list(sys.modules.values()):
        path = getattr(module, '__file__', None)
        if path and str(P / 'scripts/experiments') in path and Path(path).suffix == '.py':
            files.append(Path(path))
    snapshot(root, files)
    write(root / 'audit/UNITS.json', units())
    shutil.copy2(__file__, root / 'scripts/global_branch_calibration.py')
    write(root / 'contracts/START_FREEZE.json', {'UTC': utc(), 'runtime_sha256': sha(Path(__file__)),
          'contract_sha256': sha(root / 'contracts/ANALYSIS_CONTRACT.json'),
          'manifest_sha256': sha(root / 'manifest/INPUT_SHA256.csv')})
    print('GLOBAL_BRANCH_FROZEN', root, flush=True)


def check(root):
    freeze = json.loads((root / 'contracts/START_FREEZE.json').read_text())
    for name, p in [('runtime_sha256', Path(__file__)),
                    ('contract_sha256', root / 'contracts/ANALYSIS_CONTRACT.json'),
                    ('manifest_sha256', root / 'manifest/INPUT_SHA256.csv')]:
        if sha(p) != freeze[name]:
            raise RuntimeError('Frozen R64 input changed ' + str(p))
    manifest = pd.read_csv(root / 'manifest/INPUT_SHA256.csv')
    for r in manifest.itertuples():
        if sha(r.path) != r.sha256:
            raise RuntimeError('Frozen input changed ' + r.path)
    return len(manifest)


def population(data, dep, pairs, inventory):
    meta = pd.read_parquet(data / f'data/{dep}/event_metadata.parquet')
    if not np.array_equal(meta.row_index, np.arange(len(meta))):
        raise RuntimeError('Metadata row mapping changed')
    records = [json.loads(p.read_text()) for p in sorted((data / f'profile_events/{dep}').glob('*.json'))]
    if len(records) != len(meta):
        raise RuntimeError('Missing event profile')
    parent = np.load(data / f'predictions/{dep}/ensemble_mass.npy')
    prior = json.loads((predictive.PARENT / 'calibration/PROFILE_PREDICTIVE.json').read_text())[dep]['spec']
    quality, _ = predictive.mode.quality(parent, records, prior['parent_coverage'])
    noise = pd.read_csv(data / f'data/{dep}/noise/noise_manifest.csv').set_index('bank_index')
    chunk = meta.noise_bank_index.map(noise.parent_file_gps).to_numpy()
    if not np.isfinite(chunk).all():
        raise RuntimeError('Missing noise parent')
    if set(meta.source_uid[meta.fold == 0]) & set(meta.source_uid[meta.fold == 1]) or set(chunk[meta.fold == 0]) & set(chunk[meta.fold == 1]):
        raise RuntimeError('Source or parent-noise leakage')
    outputs, audits, structures = [], [], {}
    for fold in (0, 1):
        ids = np.flatnonzero(meta.fold.eq(fold))
        names = sorted(meta.source_uid.iloc[ids].unique())
        lookup = {name: k for k, name in enumerate(names)}
        if not (meta.iloc[ids].groupby('source_uid').size() == 2).all():
            raise RuntimeError('Expected complete source doublets')
        group = meta.source_uid.map(lookup).fillna(-1).to_numpy(int)
        i, j = np.triu_indices(len(ids), 1); i, j = ids[i], ids[j]
        truth = group[i] == group[j]
        null = ~truth & (chunk[i] != chunk[j])
        ni, nj = i[null], j[null]
        size = len(names)
        key = np.minimum(group[ni], group[nj]) * size + np.maximum(group[ni], group[nj])
        keys, multiplicity = np.unique(key, return_counts=True)
        full = dict(zip(keys, multiplicity))
        adjacency = np.zeros((size, size))
        a, b = keys // size, keys % size
        adjacency[a, b] = adjacency[b, a] = 1.
        qi, qj = quality[ni], quality[nj]
        inv = inventory[(inventory.deployment == dep) & (inventory.fold == fold)].iloc[0]
        true_q = {(int(a), int(b)) for a, b in zip(i[truth], j[truth]) if quality[a] and quality[b]}
        f = pairs[(pairs.deployment == dep) & (pairs.fold == fold)].copy()
        seen_true = set(map(tuple, f.loc[f.kind == 'true', ['idx_i', 'idx_j']].to_numpy(int)))
        if seen_true != true_q or (qi & qj).sum() != inv.null_population or quality[ids].sum() != inv.active_events:
            raise RuntimeError('R56 quality/census reconstruction failed')
        fi, fj = f.idx_i.to_numpy(int), f.idx_j.to_numpy(int)
        f['group_i'] = group[fi]; f['group_j'] = group[fj]
        f['full_sourcepair_multiplicity'] = [1 if t else int(full[min(a, b) * size + max(a, b)])
            for t, a, b in zip(f.kind.eq('true'), group[fi], group[fj])]
        f['full_sourcepair_weight'] = 1. / f.full_sourcepair_multiplicity
        f['global_HT_weight'] = f.full_sourcepair_weight / f.inclusion_probability
        f['full_true_systems'] = int(truth.sum())
        f['full_null_sourcepairs'] = len(keys)
        f['global_balanced_weight'] = np.where(f.kind.eq('true'),
            .5 / truth.sum(), .5 * f.global_HT_weight / len(keys))
        f['old_quality_HT_weight'] = f.HT_weight
        sums = np.bincount(np.searchsorted(keys, key), weights=1. / np.array([full[k] for k in key]))
        err = float(np.max(np.abs(sums - 1.)))
        if err > 1e-12:
            raise RuntimeError('Full source-pair group weight not1')
        audits.append({'deployment': dep, 'fold': fold, 'events': len(ids), 'sources': size,
            'true_systems': int(truth.sum()), 'null_image_pairs': len(ni), 'null_sourcepairs': len(keys),
            'quality_events': int(quality[ids].sum()), 'quality_true_census': len(true_q),
            'quality_null_image_pairs': int((qi & qj).sum()), 'sourcepair_weight_max_error': err,
            'sample_multiplicity_changed': int((f.full_sourcepair_multiplicity != f.source_pair_multiplicity).sum()),
            'source_fold_overlap': 0, 'noise_parent_fold_overlap': 0})
        structures[fold] = (names, adjacency)
        outputs.append(f)
    return pd.concat(outputs, ignore_index=True), audits, structures


def bootstrap_weights(f, names, adjacency, rng):
    size = len(names)
    counts = rng.multinomial(size, np.full(size, 1. / size), size=BOOTSTRAPS)
    gi, gj = f.group_i.to_numpy(int), f.group_j.to_numpy(int)
    null_den = .5 * np.einsum('bi,ij,bj->b', counts, adjacency, counts, optimize=True)
    if np.any(null_den <= 0):
        raise RuntimeError('Degenerate source bootstrap')
    y = f.kind.eq('true').to_numpy()
    weights = .5 * counts[:, gi] * counts[:, gj] * f.global_HT_weight.to_numpy()[None, :] / null_den[:, None]
    weights[:, y] = .5 * counts[:, gi[y]] / size
    return weights


def fit(root):
    count = check(root)
    if (root / 'contracts/PILOT_GATE.json').exists():
        raise RuntimeError('Completed R64 is immutable')
    start = time.monotonic()
    c = json.loads((root / 'contracts/ANALYSIS_CONTRACT.json').read_text())
    data, expansion, reference = [Path(c[k]) for k in ('data', 'expansion', 'reference')]
    pairs = pd.read_parquet(expansion / 'tables/PAIR_RESULTS.parquet')
    inventory = pd.read_csv(expansion / 'tables/PLANNING_INVENTORY.csv')
    predictions = pd.read_parquet(reference / 'tables/PREDICTIONS.parquet')
    base_arm = json.loads((reference / 'contracts/PILOT_GATE.json').read_text())['selected_common_arm']['arm']
    refs = {(r['deployment'], r['seed']): r for r in json.loads((expansion / 'configs/FROZEN_REFERENCE_CONFIGS.json').read_text())}
    allplans, audits, rates, grids, metrics, outpred, configs, bootstrap = [], [], [], [], [], [], {}, []
    run_boot = {}
    for dep_index, dep in enumerate(DEPS):
        full, audit_rows, structures = population(data, dep, pairs, inventory)
        audits += audit_rows
        ref = refs[dep, SEEDS[0]]
        eligible = full.minimum_independent_power.between(*ref['minimum_power_support']) & full.any_shared_converged
        full['eligible'] = eligible
        allplans.append(full)
        f = full.loc[eligible].reset_index(drop=True)
        y = f.kind.eq('true').to_numpy(float)
        train, tune = f.fold.eq(0).to_numpy(), f.fold.eq(1).to_numpy()
        w = f.global_balanced_weight.to_numpy(float)
        probabilities = {}
        for fold in (0, 1):
            row = next(r for r in audit_rows if r['fold'] == fold)
            g = f[f.fold == fold]
            pl = float(g.kind.eq('true').sum() / row['true_systems'])
            pn = float(g.loc[g.kind != 'true', 'global_HT_weight'].sum() / row['null_sourcepairs'])
            if not 0 < pl < 1 or not 0 < pn < 1:
                raise RuntimeError('Selection probability unsupported')
            probabilities[fold] = pl, pn
            rates.append({'deployment': dep, 'fold': fold, 'p_E_given_L_census': pl,
                'p_E_given_N_HT_estimate': pn, 'log_selection_ratio': float(np.log(pl / pn)),
                'eligible_true': int(g.kind.eq('true').sum()), 'eligible_null_sample': int(g.kind.ne('true').sum()),
                'null_HT_ESS': float(g.loc[g.kind != 'true', 'global_HT_weight'].sum()**2 /
                    np.square(g.loc[g.kind != 'true', 'global_HT_weight']).sum()),
                'not_physical_lensing_prior': True})
        bweights = bootstrap_weights(f.loc[tune].reset_index(drop=True), *structures[1],
                                    np.random.default_rng(2026091064 + dep_index))
        for seed in SEEDS:
            g = predictions[(predictions.deployment == dep) & (predictions.seed == seed) & (predictions.arm == base_arm)]
            if set(g.pair_id) != set(f.pair_id) or g.pair_id.duplicated().any():
                raise RuntimeError('R61 eligibility mismatch')
            g = g.set_index('pair_id').loc[f.pair_id]
            if not np.array_equal(g.deficit.to_numpy(), f.deficit.to_numpy()):
                raise RuntimeError('Frozen physical deficit changed')
            z = g.waveform_score.to_numpy(float)
            base = float(np.dot(w[tune], binary_loss(y[tune], z[tune])))
            identity = {'intercept': 0., 'slope': 1., 'no_update': True, 'ridge': None}
            choices = [(base, True, float('inf'), identity, z)]
            grids.append({'deployment': dep, 'seed': seed, 'ridge': None, 'no_update': True,
                          'eligible_contribution_to_global_balanced_risk': base})
            for ridge in RIDGES:
                spec = fit_affine(z[train], y[train], w[train], ridge)
                value = apply(z, f.deficit, spec, ref['deficit_support'])
                risk = float(np.dot(w[tune], binary_loss(y[tune], value[tune])))
                choices.append((risk, False, ridge, spec, value))
                grids.append({'deployment': dep, 'seed': seed, 'ridge': ridge, 'no_update': False,
                              'eligible_contribution_to_global_balanced_risk': risk, **spec})
            best = min(x[0] for x in choices)
            winner = min([x for x in choices if x[0] <= best + 1e-12], key=lambda x: (not x[1], -x[2]))
            offset = {'intercept': float(np.log(probabilities[0][0] / probabilities[0][1])),
                      'slope': 1., 'no_update': False, 'ridge': None}
            scores = {ARMS[0]: (identity, z), ARMS[1]: (offset, apply(z, f.deficit, offset, ref['deficit_support'])),
                      ARMS[2]: (winner[3], winner[4])}
            for arm, (spec, value) in scores.items():
                configs[f'{dep}/{seed}/{arm}'] = {**spec, 'base_arm': base_arm,
                    'deficit_support': ref['deficit_support'], 'minimum_power_support': ref['minimum_power_support'],
                    'cap': 16., 'eligibility_unchanged': True, 'fit_selection_offset': offset['intercept']}
                for fold in (0, 1):
                    use = f.fold.eq(fold).to_numpy()
                    loss = float(np.dot(w[use], binary_loss(y[use], value[use])))
                    diff = loss - float(np.dot(w[use], binary_loss(y[use], z[use])))
                    metrics.append({'deployment': dep, 'seed': seed, 'arm': arm, 'fold': fold,
                        'eligible_contribution_to_global_balanced_risk': loss, 'global_risk_delta': diff,
                        'eligible_pairs': int(use.sum()), 'no_update': spec['no_update']})
                delta = binary_loss(y[tune], value[tune]) - binary_loss(y[tune], z[tune])
                draw = bweights @ delta
                run_boot[dep, seed, arm] = draw
                bootstrap.append({'deployment': dep, 'seed': str(seed), 'arm': arm,
                    'delta_mean': float(np.dot(w[tune], delta)),
                    'ci_lower': float(np.quantile(draw, .025)), 'ci_upper': float(np.quantile(draw, .975)),
                    'one_sided95_upper': float(np.quantile(draw, .95)), 'draws': BOOTSTRAPS})
                out = f[['pair_id', 'deployment', 'fold', 'kind', 'source_i', 'source_j', 'group_i', 'group_j',
                         'global_balanced_weight', 'global_HT_weight', 'full_sourcepair_multiplicity',
                         'inclusion_probability', 'deficit', 'minimum_independent_power']].copy()
                out['seed'] = seed; out['arm'] = arm; out['waveform_score'] = value; out['reference_score'] = z
                outpred.append(out)
            print('GLOBAL_BRANCH_FIT', dep, seed, 'affine', winner[3], flush=True)
    metric = pd.DataFrame(metrics)
    gates = []
    for arm in ARMS[1:]:
        good = True
        for dep in DEPS:
            points = metric[(metric.deployment == dep) & (metric.arm == arm) & (metric.fold == 1)].global_risk_delta
            draws = np.mean([run_boot[dep, seed, arm] for seed in SEEDS], axis=0)
            upper = float(np.quantile(draws, .95))
            passed = bool((points <= 1e-12).all() and points.mean() < -1e-12 and upper <= 0.)
            good &= passed
            gates.append({'deployment': dep, 'arm': arm, 'all_model_point_nonworse': bool((points <= 1e-12).all()),
                          'mean_delta': float(points.mean()), 'one_sided95_upper': upper, 'run_pass': passed})
            bootstrap.append({'deployment': dep, 'seed': 'MODEL_MEAN', 'arm': arm,
                'delta_mean': float(points.mean()), 'ci_lower': float(np.quantile(draws, .025)),
                'ci_upper': float(np.quantile(draws, .975)), 'one_sided95_upper': upper, 'draws': BOOTSTRAPS})
    passes = [arm for arm in ARMS[1:] if all(g['run_pass'] for g in gates if g['arm'] == arm)]
    chosen = ARMS[2] if ARMS[2] in passes else ARMS[1] if ARMS[1] in passes else None
    pd.concat(allplans).to_parquet(root / 'tables/POPULATION_WEIGHTED_PAIR_PLAN.parquet', index=False)
    pd.concat(outpred).to_parquet(root / 'tables/PREDICTIONS.parquet', index=False)
    for name, value in [('POPULATION_AUDIT', audits), ('ELIGIBILITY_RATES', rates), ('AFFINE_GRID', grids),
                        ('GLOBAL_RISK_METRICS', metrics), ('GLOBAL_RISK_BOOTSTRAP', bootstrap), ('PILOT_GATES', gates)]:
        csv(root / f'tables/{name}.csv', value)
    write(root / 'configs/ALL_CALIBRATIONS.json', configs)
    verified = check(root)
    write(root / 'audit/INPUT_HASH_RECHECK.json', {'UTC': utc(), 'count': verified, 'failures': 0, 'initial_count': count})
    gate = {'UTC': utc(), 'gate': 'PASS' if chosen else 'FAIL', 'selected_common_arm': chosen,
            'passing_arms': passes, 'status': STATUS, 'goal_achieved': False,
            'catalog_or_real_scored': False, 'full_goal_not_implied': True, 'seconds': time.monotonic() - start}
    write(root / 'contracts/PILOT_GATE.json', gate)
    print(json.dumps(gate, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--stage', choices=('freeze', 'fit'), required=True)
    for name in ('data', 'expansion', 'reference', 'catalog'):
        parser.add_argument('--' + name, type=Path)
    args = parser.parse_args()
    if args.stage == 'freeze':
        freeze(args.root, args.data, args.expansion, args.reference, args.catalog)
    else:
        fit(args.root)
