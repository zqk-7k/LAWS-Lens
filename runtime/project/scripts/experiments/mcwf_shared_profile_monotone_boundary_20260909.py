#!/usr/bin/env python3
"""Separate experiment: bounded monotone low-deficit handling, frozen fits."""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
from pathlib import Path
import json
import shutil
import sys
import numpy as np
import pandas as pd

P = Path('/root/autodl-tmp/gw-catalog')
sys.path.insert(0, str(P/'scripts/experiments'))
import mcwf_shared_profile_catalog_20260909 as parent
import mcwf_shared_profile_catalog_audit_20260909 as audit
n = parent.n
NEW = 'SHARED-PROFILE-MONOTONE-BOUNDARY'
METHODS = ('NODUP-DIRECT-REPLAY', 'SHARED-PROFILE-SINGLE-WF', NEW)
ROOT = SOURCE = PILOT = None
RAW_EXPORT = parent.export


def config_map():
    return {(row['deployment'], row['seed']): row for row in json.loads(
        (SOURCE/'configs/SELECTED_CONFIGURATIONS.json').read_text())
        if row['method'] == 'SHARED-PROFILE-SINGLE-WF'}


def predict_policy(x, config, revised):
    d = np.expm1(-x[:, 2])
    lo, hi = config['deficit_support']
    features = x.copy()
    if revised:
        features[:, 2] = -np.log1p(np.maximum(d, lo))
    logits = parent.classifier.predict(features, config['shared_classifier'])
    cap = config['shared_cap']
    bounded = logits if cap is None else logits.clip(-cap, cap)
    outside = d > hi if revised else (d < lo) | (d > hi)
    return np.where(outside, np.minimum(bounded, 0.), bounded)


def freeze():
    if ROOT.exists():
        raise RuntimeError('Independent monotone-boundary output required')
    for name in ('contracts', 'configs', 'tables', 'audit', 'reports', 'scripts', 'manifest', 'results', 'logs'):
        (ROOT/name).mkdir(parents=True)
    n.write_json(ROOT/'contracts/ANALYSIS_CONTRACT.json', {
        'UTC': n.utc(), 'id': 'MCWF-SHARED-PROFILE-MONOTONE-BOUNDARY-55',
        'status': n.STATUS, 'goal_achieved': False, 'adaptive_development': True,
        'source': str(SOURCE), 'pilot': str(PILOT), 'methods': METHODS, 'primary': NEW,
        'reason': 'R51 follows its stated OOD rule, but lowering nonnegative shared deficit below the smallest fit value zeros positive support. This is a policy nonmonotonicity, not an arithmetic bug.',
        'new_policy': 'For eligible pairs only, evaluate the frozen classifier at max(D,D_fit_min). No reward above the observed lower-D boundary. Keep upper-tail rule, power support, optimization, cap and OOD flags frozen.',
        'no_selected_hyperparameter': True,
        'validation_gate': 'R50 tune true/random and true/hard subsets inside existing power support; same source-balanced weights, capped-policy logloss must be nonworse separately in both runs, averaged over3encoders. Equality is allowed for an unexercised boundary.',
        'unit': 'At fixed cosine/power, score must be nonincreasing as D increases through 0 and the fit boundary. Smaller-than-fit D receives exactly boundary score, not larger extrapolation.',
        'real_expectation_not_selection': 'Previously measured real pairs all have D above the affected boundary. Real scores/ranks must replay exactly; this correction cannot solve the outstanding per-seed official budget losses by itself.',
        'frozen': ['all classifiers', 'all encoder checkpoints', 'outer weights', 'time', 'sky', 'event scope', 'D measurements'],
        'no_old_Mc_q_or_total_blend': True, 'not_new_blind_confirmation': True})
    selections = []
    for (dep, seed), row in config_map().items():
        for method in METHODS:
            selections.append({**row, 'method': method, 'boundary_policy': 'floor_D_at_frozen_fit_min' if method == NEW else 'R51_unchanged'})
    n.write_json(ROOT/'configs/SELECTED_CONFIGURATIONS.json', selections)
    shutil.copy2(__file__, ROOT/'scripts/shared_profile_monotone_boundary.py')
    n.write_json(ROOT/'contracts/START_FREEZE.json', {'UTC': n.utc(),
        'runtime_sha256': n.sha(Path(__file__)),
        'contract_sha256': n.sha(ROOT/'contracts/ANALYSIS_CONTRACT.json'),
        'config_sha256': n.sha(ROOT/'configs/SELECTED_CONFIGURATIONS.json')})
    pilot_contract = PILOT/'contracts/ANALYSIS_CONTRACT.json'
    development = Path(json.loads(pilot_contract.read_text())['data'])
    inputs = [SOURCE/'configs/SELECTED_CONFIGURATIONS.json', PILOT/'tables/PAIR_RESULTS.parquet',
              PILOT/'tables/CLASSIFIER_PREDICTIONS.parquet', Path(parent.__file__), Path(audit.__file__)]
    inputs.append(pilot_contract)
    inputs.extend(development/f'predictions/{dep}/{seed}_embedding.npy'
                  for dep in n.DEPS for seed in n.SEEDS)
    inputs.extend((SOURCE/'results/SHARED-PROFILE-SINGLE-WF').glob('gwtc*/seed_*/*/pairs.parquet'))
    inputs.extend((SOURCE/'results/SHARED-PROFILE-SINGLE-WF').glob('gwtc*/seed_*/real/fusion_all_pairs.parquet'))
    n.write_csv(ROOT/'manifest/INPUT_SHA256.csv', [{'path': str(path), 'sha256': n.sha(path)} for path in inputs])
    rows, units = [], []
    pairs = pd.read_parquet(PILOT/'tables/PAIR_RESULTS.parquet')
    for (dep, seed), config in config_map().items():
        lo, hi = config['deficit_support']
        powers = config['minimum_power_support']
        d = np.unique(np.r_[0., lo/4, lo/2, lo, np.nextafter(lo, np.inf),
                            np.geomspace(max(lo, 1e-8), hi, 100), hi+1])
        x = np.column_stack([np.full(len(d), .8), np.full(len(d), np.log(np.sqrt(powers[0]*powers[1]))), -np.log1p(d)])
        scores = predict_policy(x, config, True)
        if np.any(np.diff(scores) > 1e-10) or not np.allclose(scores[d <= lo], scores[d <= lo][0], atol=1e-10, rtol=0):
            raise RuntimeError('Bounded monotonicity failed')
        units.append({'deployment': dep, 'seed': seed, 'monotone': True,
            'low_boundary_plateau': True, 'tested_deficits': len(d)})
        f = pairs[(pairs.deployment == dep) & (pairs.fold == 1)].copy()
        f = f[f.minimum_independent_power.between(*powers) & f.any_shared_converged]
        path = development/f'predictions/{dep}/{seed}_embedding.npy'
        embedding = np.load(path).astype(float)
        embedding /= np.linalg.norm(embedding, axis=1, keepdims=True)
        cosine = np.einsum('ij,ij->i', embedding[f.idx_i], embedding[f.idx_j])
        x = np.column_stack([cosine, np.log(f.minimum_independent_power), -np.log1p(f.deficit)])
        before, after = predict_policy(x, config, False), predict_policy(x, config, True)
        for kind in ('random_null', 'hard_null'):
            take = f.kind.isin(['true', kind]).to_numpy()
            y, raw = parent.classifier.base_weights(f[take])
            w = parent.classifier.balanced(y, raw)
            a = parent.classifier.loss(y, before[take], w)
            b = parent.classifier.loss(y, after[take], w)
            rows.append({'deployment': dep, 'seed': seed, 'null_kind': kind,
                'R51_policy_logloss': a, 'monotone_policy_logloss': b,
                'delta': b-a, 'changed_pairs': int(np.sum(before[take] != after[take]))})
    table = pd.DataFrame(rows)
    check = table.groupby(['deployment', 'null_kind']).delta.mean()
    passed = bool((check <= 1e-12).all())
    n.write_csv(ROOT/'tables/BOUNDARY_VALIDATION_METRICS.csv', rows)
    n.write_csv(ROOT/'audit/BOUNDARY_MONOTONICITY_UNITS.csv', units)
    n.write_json(ROOT/'contracts/PILOT_GATE.json', {'UTC': n.utc(), 'gate': 'PASS' if passed else 'FAIL',
        'goal_achieved': False, 'no_scoring_on_fail': True, 'same_rule_both_runs': True})
    print(table.to_string(index=False), flush=True)
    print('BOUNDARY_VALIDATION_GATE', 'PASS' if passed else 'FAIL', flush=True)


def load_panel(dep, seed, split, catalog=None):
    parent.ROOT = SOURCE
    return parent.load_panel(dep, seed, split, catalog)


def infer(frame, config):
    old_config = {**config, 'method': 'SHARED-PROFILE-SINGLE-WF' if config['method'] == NEW else config['method']}
    z, ood, clipped = parent.infer(frame, old_config)
    frame['shared_low_boundary_saturated'] = False
    if config['method'] != NEW:
        return z, ood, clipped
    eligible = frame.shared_profile_eligible.to_numpy(bool)
    change = eligible & (frame.shared_profile_deficit.to_numpy() < config['deficit_support'][0])
    if change.any():
        x = np.column_stack([frame.embedding_only.to_numpy()[change],
            np.log(frame.shared_profile_minimum_power.to_numpy()[change]),
            -np.log1p(frame.shared_profile_deficit.to_numpy()[change])])
        z[change] = predict_policy(x, config, True)
        frame.loc[change, 'shared_profile_candidate_waveform'] = z[change]
        frame.loc[change, 'shared_low_boundary_saturated'] = True
        clipped[change] = True
    return z, ood, clipped


def export(frame, z, weights, method):
    result = RAW_EXPORT(frame, z, weights, method)
    for name in ('sky_j50', 'sky_j90', 'shared_low_boundary_saturated'):
        if name in frame:
            result[name] = frame[name].to_numpy(copy=True)
    return result


def save_csv(path, rows):
    if Path(path).name == 'PER_SEED_PE_OFFICIAL_BUDGETS.csv':
        rows = []
        for dep in n.DEPS:
            for seed in n.SEEDS:
                for method in (n.BASELINE, *METHODS):
                    frame = pd.read_parquet(ROOT/f'results/{method}/{dep}/seed_{seed}/real/fusion_all_pairs.parquet')
                    rows.extend({**n.dev.budget_row(frame, method, dep, 'fusion', b), 'seed': seed} for b in (10, 20, 50, 100))
    return parent.RAW_CSV(path, rows)


def run(stage):
    frozen = json.loads((ROOT/'contracts/START_FREEZE.json').read_text())
    for name, path in (('runtime_sha256', Path(__file__)),
                       ('contract_sha256', ROOT/'contracts/ANALYSIS_CONTRACT.json'),
                       ('config_sha256', ROOT/'configs/SELECTED_CONFIGURATIONS.json')):
        if frozen[name] != n.sha(path):
            raise RuntimeError('Boundary runtime or contract changed after freeze')
    if json.loads((ROOT/'contracts/PILOT_GATE.json').read_text())['gate'] != 'PASS':
        raise RuntimeError('Boundary validation failed')
    receipt = ROOT/'contracts'/('EVALUATION_COMPLETE.json' if stage == 'evaluate' else 'REAL_COMPLETE.json')
    if receipt.exists():
        raise RuntimeError('Completed stage is immutable')
    if stage == 'real' and not (ROOT/'contracts/EVALUATION_COMPLETE.json').exists():
        raise RuntimeError('Injection evaluation must finish before real replay')
    n.METHODS = METHODS
    n.load_panel, n.infer, n.public_frame, n.write_csv = load_panel, infer, export, save_csv
    n.run(ROOT, stage)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--pilot-root', type=Path, required=True)
    parser.add_argument('--stage', choices=('freeze', 'evaluate', 'real'), required=True)
    args = parser.parse_args()
    ROOT, SOURCE, PILOT = args.root, args.source_root, args.pilot_root
    if args.stage == 'freeze':
        freeze()
    else:
        run(args.stage)
