#!/usr/bin/env python3
"""Paired, validation-selected conditional-intrinsic waveform controls."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
import argparse
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import shutil
import sys
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

P = Path('/root/autodl-tmp/gw-catalog')
sys.path.insert(0, str(P / 'scripts/experiments'))
import mcwf_conditional_intrinsics_20260908 as model
import mcwf_conditional_eta_evaluate_20260908 as eta_eval
import mcwf_temporal_response_evaluate_20260908 as ev
t, dev, cf = model.t, model.dev, model.cf
BETAS = eta_eval.BETAS


def calibration(root, dep, slot, kind):
    model.KIND = kind
    meta = pd.read_parquet(t.PREVIOUS / f'expanded_data/{dep}/validation/event_metadata.parquet')
    groups = meta.source_uid.to_numpy(str)
    banks = sorted(meta.noise_bank_index.unique(), key=lambda x: hashlib.sha256(f'202609850:{dep}:{x}'.encode()).hexdigest())
    fold = meta.noise_bank_index.map({x: k % 2 for k, x in enumerate(banks)}).to_numpy(int)
    mixed = {g for g in np.unique(groups) if len(np.unique(fold[groups == g])) > 1}
    fold[np.isin(groups, list(mixed))] = -1
    a = dict(model.prediction(root, dep, slot, 0, 'development'))
    cohorts = {}
    for side in (0, 1):
        ids = np.flatnonzero(fold == side)
        i, j = np.triu_indices(len(ids), 1)
        i, j = ids[i], ids[j]
        y = groups[i] == groups[j]
        if y.sum() < 30:
            raise RuntimeError('Insufficient independent fit sources')
        cohorts[side] = y, model.pair_features(root, dep, slot, a, i, j)
    y, x = cohorts[0]
    weights = np.where(y, .5 / y.sum(), .5 / (~y).sum())
    delta = x['conditional_eta_overlap']
    iso = IsotonicRegression(increasing=True, out_of_bounds='clip').fit(delta, y, sample_weight=weights)
    prob = iso.y_thresholds_.clip(1 / (y.sum() + 2), 1 - 1 / (y.sum() + 2))
    lr = np.log(prob) - np.log1p(-prob)
    spec = {'kind': kind, 'knots': iso.X_thresholds_.tolist(), 'loglr': lr.tolist(),
            'zero_reference': float(np.interp(0., iso.X_thresholds_, lr)), 'minimum': float(delta.min()), 'maximum': float(delta.max()),
            'mass_reference': np.sort(-np.log(x['mass_BC'][y])).tolist(), 'independent_positive_sources': int(y.sum()),
            'fit_noise_banks': [int(v) for v in np.unique(meta.noise_bank_index[fold == 0])],
            'audit_noise_banks': [int(v) for v in np.unique(meta.noise_bank_index[fold == 1])],
            'dropped_cross_noise_sources': len(mixed), 'model_development_reuse': True,
            'checkpoint_sha256': dev.sha(root / f'models/{kind}/{dep}/seed_{slot}/selected.pt')}
    audit = []
    for side, (yy, xx) in cohorts.items():
        increment, _ = eta_eval.increment(xx, spec)
        audit.append({'kind': kind, 'deployment': dep, 'slot': slot, 'cohort': 'fit' if side == 0 else 'noise_disjoint_audit',
                      'true_sources': int(yy.sum()), 'dependent_null_pairs': int((~yy).sum()),
                      'true_median_delta': float(np.median(xx['conditional_eta_overlap'][yy])),
                      'null_median_delta': float(np.median(xx['conditional_eta_overlap'][~yy])),
                      'true_negative_increment_fraction': float((increment[yy] < 0).mean()),
                      'null_positive_increment_fraction': float((increment[~yy] > 0).mean()), 'model_development_reuse': True})
    dev.json_write(root / f'calibration/{kind}/{dep}/{slot}.json', spec)
    return spec, audit


def values(root, dep, slot, seed, split, kind, spec):
    model.KIND = kind
    frame = cf.read(dep, seed, split)
    a = dict(model.prediction(root, dep, slot, seed, split))
    x = model.pair_features(root, dep, slot, a, frame.idx_i.to_numpy(int), frame.idx_j.to_numpy(int))
    if not all(np.isfinite(v).all() for v in x.values()):
        raise RuntimeError('Invalid strict-scope intrinsic features')
    increment, audit = eta_eval.increment(x, spec)
    x['conditional_intrinsic_overlap'] = x.pop('conditional_eta_overlap')
    return frame, increment, {**x, **audit, 'increment': increment}


def select(root):
    if (root / 'contracts/INTEGRATION_FROZEN.json').exists():
        raise RuntimeError('Selection already frozen')
    shutil.copy2(__file__, root / 'scripts/conditional_intrinsics_evaluate.py')
    configs, grids, audits = [], [], []
    for kind in model.KINDS:
        for dep in t.DEPS:
            for slot, seed in zip(t.MODEL_SLOTS, t.SEEDS):
                spec, audit = calibration(root, dep, slot, kind)
                audits += audit
                frame, increment, _ = values(root, dep, slot, seed, 'validation', kind, spec)
                baseline = frame.waveform_score.to_numpy(float)
                w = cf.frozen_weights(dep, seed)
                bm = cf.fast_metrics(frame, cf.channels(frame, baseline) @ w)
                bwm = cf.fast_metrics(frame, baseline)
                rows = []
                for beta in BETAS:
                    wf = baseline + beta * increment
                    mm = cf.fast_metrics(frame, cf.channels(frame, wf) @ w)
                    wm = cf.fast_metrics(frame, wf)
                    row = {'beta': beta, 'pass': cf.guard(mm, bm) and cf.guard(wm, bwm), **mm}
                    rows.append(row)
                    grids.append({'kind': kind, 'deployment': dep, 'seed': seed, **row, **{'waveform_' + k: v for k, v in wm.items()}})
                for policy in ('CANDIDATE', 'RETRIEVAL'):
                    def key(r):
                        primary = (r['false_at_recall_0p5'], r['false_at_recall_0p9'], -r['average_precision'], -r['macro_r_at_10'], -r['macro_r_at_1']) if policy == 'CANDIDATE' else (-r['macro_r_at_10'], -r['macro_r_at_1'], -r['average_precision'], r['false_at_recall_0p5'], r['false_at_recall_0p9'])
                        return (*primary, r['beta'])
                    winner = min((r for r in rows if r['pass']), key=key)
                    configs.append({'method': kind + '-' + policy, 'kind': kind, 'deployment': dep, 'seed': seed,
                                    'slot': slot, 'beta': winner['beta'], 'gamma': 0., 'weights': w.tolist(), 'calibration': spec})
    dev.csv_write(root / 'tables/VALIDATION_GRID.csv', pd.DataFrame(grids))
    dev.csv_write(root / 'tables/CALIBRATION_AUDIT.csv', pd.DataFrame(audits))
    dev.csv_write(root / 'tables/SELECTED_COEFFICIENTS.csv', pd.DataFrame([{k: v for k, v in c.items() if k not in ('weights', 'calibration')} for c in configs]))
    dev.json_write(root / 'calibration/SELECTED.json', configs)
    dev.json_write(root / 'contracts/INTEGRATION_FROZEN.json', {'UTC': datetime.now(timezone.utc).isoformat(),
                   'sha256': dev.sha(root / 'calibration/SELECTED.json'), 'code_sha256': dev.sha(Path(__file__)), 'real_or_test_selection': False})


def scored(root, config, split):
    if config['method'] == 'OMC':
        frame = cf.read(config['deployment'], config['seed'], split)
        return frame, frame.waveform_score.to_numpy(float), {}
    frame, increment, audit = values(root, config['deployment'], config['slot'], config['seed'], split, config['kind'], config['calibration'])
    return frame, frame.waveform_score.to_numpy(float) + config['beta'] * increment, audit


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--stage', choices=['select', 'evaluate', 'real', 'assess'], required=True)
    args = parser.parse_args()
    ev.scored = scored
    if args.stage == 'select':
        select(args.root)
    else:
        getattr(ev, args.stage)(args.root)
