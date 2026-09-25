#!/usr/bin/env python3
"""Joint intrinsic compatibility rather than conditional-only bonuses."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.isotonic import IsotonicRegression

P = Path('/root/autodl-tmp/gw-catalog')
sys.path.insert(0, str(P / 'scripts/experiments'))
import mcwf_conditional_intrinsics_20260908 as model
import mcwf_temporal_response_evaluate_20260908 as ev
t, dev, cf = model.t, model.dev, model.cf
UPSTREAM = P / 'results/mcwf_conditional_intrinsics_exploratory_20260908T154330Z'
GAMMA = (0., .125, .25, .5, 1., 2., 4.)
BETA = (0., .0625, .125, .25, .5, 1., 2.)


def initialize(root):
    if root.exists():
        raise RuntimeError('New independent directory required')
    for name in ('contracts', 'tables', 'calibration', 'evaluation', 'reports', 'figures', 'scripts', 'logs', 'manifest', 'cache'):
        (root / name).mkdir(parents=True)
    dev.json_write(root / 'contracts/ANALYSIS_CONTRACT.json', {
        'id': 'MCWF-JOINT-INTRINSIC-EVIDENCE-12', 'UTC': datetime.now(timezone.utc).isoformat(),
        'status': t.STATUS, 'goal_achieved': False, 'same_both_runs': True, 'upstream': str(UPSTREAM),
        'hypothesis': 'Conditional eta/spin support alone may reward pairs whose joint Mc+intrinsic predictions are incompatible. Calibrate the complete joint predictive density once,not add independent parameter overlaps.',
        'models': 'Freeze CONDITIONAL-CHI and CONDITIONAL-ETA-CHI models;no retraining,unchanged mass marginals and source population priors.',
        'statistics': {'PRIOR': 'log integral p_i(M,theta)p_j(M,theta)/prior(M,theta)', 'BC': 'log sum sqrt(p_i(M,theta)p_j(M,theta));bounded similarity,not Bayes factor'},
        'calibration': 'Same source/noise-disjoint fit/audit halves. Class-balanced isotonic positive/null LR,finite independent-source joint-BC lower-tail penalty. Increment clip[-4,4],positiveOOD orjointBCtail<.05 ->0 positive reward,massendpointOOD ->neutral.',
        'integration': 'ADD to retainedOMC or REPLACE oldOMC mass penalty+increment anchoredatFRT. Unchanged OMC remains an explicit candidate;no use of official/PE labels in grid selection.',
        'grid': {'gamma': list(GAMMA), 'beta': list(BETA)},
        'priorities': ['candidate:F50,F90,-AUPRC,-R10,-R1', 'retrieval:-R10,-R1,-AUPRC,F50,F90'],
        'tie': 'UnchangedOMC first,then min gamma^2+beta^2,then gamma,beta.',
        'guards': 'Same per-seed waveform+fusion R10/AP/F50/F90 validation guardrails as retained experiment.',
        'frozen': ['all neural parameters', 'time_score', 'sky_raw_log_bf', 'outer weights', 'scope', 'history', 'paper'],
        'external': 'Adaptive development audit after selection;no real PE,official labels,event IDs as scoreinputs;notblindconfirmation.',
        'limitations': 'Neural predictive densities are not PE posteriors. Prior-overlap and BC are different proxies;neither calibrated ranking sum is a physical proper Bayes factor. Model-development reuse is disclosed.',
        'references': ['https://arxiv.org/abs/gr-qc/9402014', 'https://arxiv.org/abs/2104.09339'],
        'reference_scope': 'Motivation for correlated intrinsic consistency and prior correction;not a Hanabi implementation.'})
    protected = t.protected() + [{'path': str(p), 'sha256': dev.sha(p), 'bytes': p.stat().st_size}
                               for p in (UPSTREAM / 'models').glob('*/*/*/selected.pt')]
    dev.csv_write(root / 'manifest/INPUT_SHA256.csv', pd.DataFrame(protected))
    shutil.copy2(__file__, root / 'scripts/joint_intrinsics_evidence.py')
    dev.json_write(root / 'contracts/FREEZE.json', {'contract_sha256': dev.sha(root / 'contracts/ANALYSIS_CONTRACT.json'), 'code_sha256': dev.sha(Path(__file__))})


def features(root, dep, slot, seed, split, kind):
    path = root / f'cache/{kind}/{dep}/{slot}_{seed}_{split}.npz'
    if path.exists():
        return dict(np.load(path))
    model.KIND = kind
    # Read-only upstream predictions are complete;do not create a new upstream file.
    upstream = UPSTREAM / f'predictions/{kind}/{dep}/{slot}_{seed}_{split}.npz'
    if not upstream.exists():
        raise RuntimeError('Missing frozen upstream predictive density')
    a = dict(np.load(upstream))
    ck = torch.load(UPSTREAM / f'models/{kind}/{dep}/seed_{slot}/selected.pt', map_location='cpu', weights_only=False)
    conditional = model.original.interpolate_rows(ck['conditional_prior'], model.old.CENTERS)
    prior = ck['prior'][:, None] * conditional
    p = torch.as_tensor(a['joint'], dtype=torch.float64, device='cuda').flatten(1)
    z = p / torch.as_tensor(np.sqrt(prior).reshape(1, -1), device='cuda')
    overlap = (z @ z.T).cpu().numpy().clip(1e-300)
    bc = (torch.sqrt(p) @ torch.sqrt(p).T).cpu().numpy().clip(1e-15, 1.)
    result = {'joint_logbf': np.log(overlap), 'joint_logbc': np.log(bc), 'joint_bc': bc,
              'outside': a['outside']}
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **result)
    return result


def calibration(root, dep, slot, kind):
    meta = pd.read_parquet(t.PREVIOUS / f'expanded_data/{dep}/validation/event_metadata.parquet')
    groups = meta.source_uid.to_numpy(str)
    banks = sorted(meta.noise_bank_index.unique(), key=lambda x: hashlib.sha256(f'202609850:{dep}:{x}'.encode()).hexdigest())
    fold = meta.noise_bank_index.map({x: k % 2 for k, x in enumerate(banks)}).to_numpy(int)
    mixed = {g for g in np.unique(groups) if len(np.unique(fold[groups == g])) > 1}
    fold[np.isin(groups, list(mixed))] = -1
    x = features(root, dep, slot, 0, 'development', kind)
    cohorts = {}
    for side in (0, 1):
        ids = np.flatnonzero(fold == side)
        i, j = np.triu_indices(len(ids), 1)
        i, j = ids[i], ids[j]
        y = groups[i] == groups[j]
        if y.sum() < 30:
            raise RuntimeError('Insufficient source/noise-disjoint fit support')
        cohorts[side] = (i, j, y)
    i, j, y = cohorts[0]
    weights = np.where(y, .5 / y.sum(), .5 / (~y).sum())
    specs, audits = {}, []
    for statistic, column in (('PRIOR', 'joint_logbf'), ('BC', 'joint_logbc')):
        value = x[column][i, j]
        iso = IsotonicRegression(increasing=True, out_of_bounds='clip').fit(value, y, sample_weight=weights)
        prob = iso.y_thresholds_.clip(1 / (y.sum() + 2), 1 - 1 / (y.sum() + 2))
        spec = {'statistic': statistic, 'column': column, 'knots': iso.X_thresholds_.tolist(),
                'loglr': (np.log(prob) - np.log1p(-prob)).tolist(), 'minimum': float(value.min()), 'maximum': float(value.max()),
                'joint_reference': np.sort(-np.log(x['joint_bc'][i[y], j[y]])).tolist(),
                'independent_positive_sources': int(y.sum()), 'discarded_cross_noise_sources': len(mixed),
                'fit_noise_banks': [int(v) for v in np.unique(meta.noise_bank_index[fold == 0])],
                'audit_noise_banks': [int(v) for v in np.unique(meta.noise_bank_index[fold == 1])],
                'model_development_reuse': True}
        for side, (ii, jj, yy) in cohorts.items():
            pp = ev.tail.tail_probability(-np.log(x['joint_bc'][ii, jj]), spec['joint_reference'])
            audits.append({'deployment': dep, 'slot': slot, 'kind': kind, 'statistic': statistic,
                           'cohort': 'fit' if side == 0 else 'noise_disjoint_audit', 'true_sources': int(yy.sum()),
                           'true_joint_tail_below_0p05': float((pp[yy] < .05).mean()), 'null_pairs': int((~yy).sum())})
        specs[statistic] = spec
        dev.json_write(root / f'calibration/{kind}/{dep}/{slot}_{statistic}.json', spec)
    return specs, audits


def values(root, dep, slot, seed, split, kind, spec):
    frame = cf.read(dep, seed, split)
    x = features(root, dep, slot, seed, split, kind)
    i, j = frame.idx_i.to_numpy(int), frame.idx_j.to_numpy(int)
    value = x[spec['column']][i, j]
    bc = x['joint_bc'][i, j]
    endpoint = (x['outside'][i] > .25) | (x['outside'][j] > .25)
    if not np.isfinite(value).all() or not np.isfinite(bc).all():
        raise RuntimeError('Invalid strict-scope joint prediction')
    pp = ev.tail.tail_probability(-np.log(bc), spec['joint_reference'])
    outside = endpoint | (value < spec['minimum']) | (value > spec['maximum'])
    penalty = np.minimum(np.log(pp / .05), 0.)
    increment = np.interp(value, spec['knots'], spec['loglr'])
    increment = np.where(outside | (pp < .05), np.minimum(increment, 0.), increment).clip(-4, 4)
    penalty, increment = np.where(endpoint, 0., penalty), np.where(endpoint, 0., increment)
    return frame, penalty, increment, {'joint_statistic': value, 'joint_BC': bc, 'joint_tail_probability': pp, 'joint_ood': outside}


def select(root):
    if (root / 'contracts/INTEGRATION_FROZEN.json').exists():
        raise RuntimeError('Already frozen')
    configs, grids, audits = [], [], []
    for kind in model.KINDS:
        for dep in t.DEPS:
            for slot, seed in zip(t.MODEL_SLOTS, t.SEEDS):
                specs, audit = calibration(root, dep, slot, kind)
                audits += audit
                for statistic, spec in specs.items():
                    frame, penalty, increment, _ = values(root, dep, slot, seed, 'validation', kind, spec)
                    baseline = frame.waveform_score.to_numpy(float)
                    w = cf.frozen_weights(dep, seed)
                    bm, bwm = cf.fast_metrics(frame, cf.channels(frame, baseline) @ w), cf.fast_metrics(frame, baseline)
                    for integration in ('ADD', 'REPLACE'):
                        anchor = baseline if integration == 'ADD' else frame.FRT_baseline_waveform_score.to_numpy(float)
                        rows = [{'gamma': 0., 'beta': 0., 'unchanged_OMC': True, 'pass': True, **bm}]
                        for gamma in GAMMA:
                            for beta in BETA:
                                wf = anchor + gamma * penalty + beta * increment
                                mm, wm = cf.fast_metrics(frame, cf.channels(frame, wf) @ w), cf.fast_metrics(frame, wf)
                                row = {'gamma': gamma, 'beta': beta, 'unchanged_OMC': False, 'pass': cf.guard(mm, bm) and cf.guard(wm, bwm), **mm}
                                rows.append(row)
                                grids.append({'kind': kind, 'statistic': statistic, 'integration': integration, 'deployment': dep, 'seed': seed, **row, **{'waveform_' + k: v for k, v in wm.items()}})
                        for policy in ('CANDIDATE', 'RETRIEVAL'):
                            def key(r):
                                primary = (r['false_at_recall_0p5'], r['false_at_recall_0p9'], -r['average_precision'], -r['macro_r_at_10'], -r['macro_r_at_1']) if policy == 'CANDIDATE' else (-r['macro_r_at_10'], -r['macro_r_at_1'], -r['average_precision'], r['false_at_recall_0p5'], r['false_at_recall_0p9'])
                                return (*primary, not r['unchanged_OMC'], r['gamma']**2 + r['beta']**2, r['gamma'], r['beta'])
                            chosen = min((r for r in rows if r['pass']), key=key)
                            configs.append({'method': kind.replace('CONDITIONAL-', 'JOINT-') + '-' + statistic + '-' + integration + '-' + policy,
                                            'kind': kind, 'statistic': statistic, 'integration': integration, 'deployment': dep, 'seed': seed, 'slot': slot,
                                            'gamma': chosen['gamma'], 'beta': chosen['beta'], 'unchanged_OMC': chosen['unchanged_OMC'], 'weights': w.tolist(), 'calibration': spec})
    dev.csv_write(root / 'tables/VALIDATION_GRID.csv', pd.DataFrame(grids))
    dev.csv_write(root / 'tables/CALIBRATION_AUDIT.csv', pd.DataFrame(audits))
    dev.csv_write(root / 'tables/SELECTED_COEFFICIENTS.csv', pd.DataFrame([{k: v for k, v in c.items() if k not in ('weights', 'calibration')} for c in configs]))
    dev.json_write(root / 'calibration/SELECTED.json', configs)
    dev.json_write(root / 'contracts/INTEGRATION_FROZEN.json', {'file': 'calibration/SELECTED.json', 'sha256': dev.sha(root / 'calibration/SELECTED.json'),
                   'UTC': datetime.now(timezone.utc).isoformat(), 'code_sha256': dev.sha(Path(__file__)), 'real_or_test_selection': False})


def scored(root, c, split):
    if c['method'] == 'OMC' or c.get('unchanged_OMC', False):
        frame = cf.read(c['deployment'], c['seed'], split)
        return frame, frame.waveform_score.to_numpy(float), {}
    frame, penalty, increment, audit = values(root, c['deployment'], c['slot'], c['seed'], split, c['kind'], c['calibration'])
    anchor = frame.waveform_score.to_numpy(float) if c['integration'] == 'ADD' else frame.FRT_baseline_waveform_score.to_numpy(float)
    return frame, anchor + c['gamma'] * penalty + c['beta'] * increment, {**audit, 'penalty': penalty, 'increment': increment}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--stage', choices=['initialize', 'select', 'evaluate', 'real', 'assess'], required=True)
    args = p.parse_args()
    ev.scored = scored
    if args.stage in ('initialize', 'select'):
        globals()[args.stage](args.root)
    else:
        getattr(ev, args.stage)(args.root)
