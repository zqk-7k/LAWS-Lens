#!/usr/bin/env python3
"""Calibrate existing shared-waveform features with audited survey weights."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import json
from pathlib import Path
import shutil
import sys
import numpy as np
import pandas as pd

P = Path('/root/autodl-tmp/gw-catalog')
sys.path.insert(0, str(P/'scripts/experiments'))
import mcwf_shared_joint_feature_pilot_20260909 as joint
base, n = joint.base, joint.n
ARMS = ('R51-FROZEN', 'HT-SHARED-PROFILE', 'HT-JOINT-FEATURES')


def main(root, pilot, joint_root, sampling):
    if root.exists():
        raise RuntimeError('Independent weighted-fit pilot required')
    if not json.loads((sampling/'contracts/SAMPLING_AUDIT_COMPLETE.json').read_text())['passed']:
        raise RuntimeError('Sampling audit must pass')
    for name in ('contracts', 'tables', 'scripts', 'manifest', 'reports', 'audit'):
        (root/name).mkdir(parents=True)
    n.write_json(root/'contracts/ANALYSIS_CONTRACT.json', {
        'UTC': n.utc(), 'id': 'MCWF-HT-SHARED-PROFILE-54-PILOT', 'status': n.STATUS,
        'goal_achieved': False, 'adaptive_development': True, 'prior_failed_R52_retained': True,
        'motivation': 'Include physically similar hard nulls in fit without pretending they have their oversampled frequency in the ordinary background.',
        'source': str(sampling), 'arms': list(ARMS),
        'fit': 'R50 fold0, true/random/hard rows; class-balanced Horvitz-Thompson source-pair weights reconstructed by exact selection audit.',
        'selection': 'R50 fold1, same HT target; proper balanced logloss chooses ridge separately per run/encoder, ties stronger ridge.',
        'grid': list(base.RIDGES), 'features_and_constraints': 'Exactly R52 3D or 4D features and monotone slopes. No new waveform fit or learned encoder.',
        'diagnostics': ['full HT-weighted population', 'random draw, source-pair weights',
                        'neighbor-null population including any random-draw neighbors, source-pair weights'],
        'pilot_gate': 'For an arm: mean loss across encoders nonworse than frozen R51 in all 3 diagnostics separately in both runs; at least one strict improvement.',
        'common_primary_selection_if_multiple_pass': 'Largest minimum full-population mean-loss gain across runs, then largest mean gain, then simpler HT-SHARED-PROFILE.',
        'real_or_catalog_outcome_selection': False, 'scoring': 'Not done unless pilot passes; one classifier logit, not a blend of old/new waveform or total scores.',
        'frozen': ['encoder', 'time', 'sky', 'outer weights', 'scope', 'historical results'],
        'uncertainty_limit': 'Self-normalized class weights give a ratio estimator. R50 development has been examined previously; this is not an independent confirmation.',
        'reference': 'https://www.tandfonline.com/doi/abs/10.1080/01621459.1952.10483446'})
    shutil.copy2(__file__, root/'scripts/shared_profile_weighted_fit.py')
    n.write_json(root/'contracts/START_FREEZE.json', {'UTC': n.utc(),
        'script_sha256': n.sha(Path(__file__)),
        'contract_sha256': n.sha(root/'contracts/ANALYSIS_CONTRACT.json')})
    pairs = pd.read_parquet(pilot/'tables/PAIR_RESULTS.parquet')
    weights = pd.read_parquet(sampling/'tables/SAMPLING_WEIGHTS.parquet')
    keep = ['pair_id', 'neighbor_population', 'source_pair_weight', 'HT_weight', 'inclusion_probability']
    pairs = pairs.merge(weights[keep], on='pair_id', how='left', validate='one_to_one')
    if len(pairs) != 480 or pairs.HT_weight.isna().any():
        raise RuntimeError('Incomplete pair weights')
    bc_source = pd.read_parquet(joint_root/'tables/CLASSIFIER_PREDICTIONS.parquet')
    old = json.loads((pilot/'contracts/SELECTED_PILOT_CLASSIFIERS.json').read_text())
    grids, metrics, predictions, specs = [], [], [], {}
    inputs = [pilot/'tables/PAIR_RESULTS.parquet', sampling/'tables/SAMPLING_WEIGHTS.parquet',
              joint_root/'tables/CLASSIFIER_PREDICTIONS.parquet',
              pilot/'contracts/SELECTED_PILOT_CLASSIFIERS.json', Path(joint.__file__), Path(base.__file__)]
    for dep in n.DEPS:
        frame = pairs[pairs.deployment == dep].reset_index(drop=True)
        fit = frame.fold.eq(0).to_numpy(); tune = ~fit
        y = frame.kind.eq('true').to_numpy(float)
        fw = base.balanced(y[fit], frame.HT_weight.to_numpy()[fit])
        tw = base.balanced(y[tune], frame.HT_weight.to_numpy()[tune])
        for seed in n.SEEDS:
            path = joint.DATA/f'predictions/{dep}/{seed}_embedding.npy'
            inputs.append(path)
            embedding = np.load(path).astype(float)
            embedding /= np.linalg.norm(embedding, axis=1, keepdims=True)
            cosine = np.einsum('ij,ij->i', embedding[frame.idx_i], embedding[frame.idx_j])
            bc_frame = bc_source[(bc_source.deployment == dep) & (bc_source.seed == seed)
                                & (bc_source.arm == joint.ARMS[0])].set_index('pair_id')
            bc = bc_frame.loc[frame.pair_id, 'learned_joint_BC'].to_numpy(float)
            x0 = np.column_stack([cosine, np.log(frame.minimum_independent_power), -np.log1p(frame.deficit)])
            for arm in ARMS:
                x = np.column_stack([x0, np.log(bc)]) if arm == ARMS[2] else x0
                if arm == ARMS[0]:
                    spec = old[f'{dep}/{seed}/SHARED-PROFILE-DEFICIT']
                else:
                    options = []
                    for ridge in base.RIDGES:
                        candidate = joint.fit(x[fit], y[fit], fw, ridge)
                        loss = base.loss(y[tune], base.predict(x[tune], candidate), tw)
                        grids.append({'deployment': dep, 'seed': seed, 'arm': arm,
                            'ridge': ridge, 'tune_HT_logloss': loss})
                        options.append((loss, -ridge, candidate))
                    spec = min(options, key=lambda a: (a[0], a[1]))[2]
                specs[f'{dep}/{seed}/{arm}'] = spec
                z = base.predict(x, spec)
                masks = {'full_population': tune,
                         'random_draw': tune & frame.kind.ne('hard_null').to_numpy(),
                         'neighbor_population': tune & (frame.neighbor_population.to_numpy() | (y == 1))}
                for diagnostic, mask in masks.items():
                    column = 'HT_weight' if diagnostic == 'full_population' else 'source_pair_weight'
                    w = base.balanced(y[mask], frame[column].to_numpy()[mask])
                    metrics.append({'deployment': dep, 'seed': seed, 'arm': arm,
                        'diagnostic': diagnostic, 'balanced_logloss': base.loss(y[mask], z[mask], w),
                        'pairs': int(mask.sum()), 'null_rows': int((y[mask] == 0).sum())})
                out = frame[['pair_id', 'deployment', 'fold', 'kind', 'source_i', 'source_j',
                             'HT_weight', 'source_pair_weight', 'neighbor_population']].copy()
                out['seed'] = seed; out['arm'] = arm; out['logit'] = z
                predictions.extend(out.to_dict('records'))
            print('HT_PROFILE_FIT', dep, seed, flush=True)
    metric = pd.DataFrame(metrics)
    summary = metric.groupby(['deployment', 'arm', 'diagnostic']).balanced_logloss.agg(['mean', 'std']).reset_index()
    comparison = summary.pivot(index=['deployment', 'diagnostic'], columns='arm', values='mean')
    passing = []
    for arm in ARMS[1:]:
        difference = comparison[arm]-comparison[ARMS[0]]
        passed = bool((difference <= 1e-12).all() and (difference < -1e-12).any())
        comparison[arm+'_delta'] = difference
        if passed:
            gains = -difference.xs('full_population', level='diagnostic')
            passing.append({'arm': arm, 'minimum_gain': float(gains.min()), 'mean_gain': float(gains.mean())})
    selected = min(passing, key=lambda a: (-a['minimum_gain'], -a['mean_gain'], ARMS.index(a['arm']))) if passing else None
    n.write_csv(root/'tables/WEIGHTED_METRICS_PER_SEED.csv', metrics)
    n.write_csv(root/'tables/WEIGHTED_METRICS_SUMMARY.csv', summary)
    n.write_csv(root/'tables/PILOT_GATE_COMPARISON.csv', comparison.reset_index())
    n.write_csv(root/'tables/WEIGHTED_RIDGE_GRID.csv', grids)
    pd.DataFrame(predictions).to_parquet(root/'tables/WEIGHTED_CLASSIFIER_PREDICTIONS.parquet', index=False)
    n.write_json(root/'contracts/WEIGHTED_CLASSIFIERS.json', specs)
    n.write_csv(root/'manifest/INPUT_SHA256.csv', [{'path': str(path), 'sha256': n.sha(path)}
        for path in sorted(set(inputs))])
    n.write_json(root/'contracts/PILOT_GATE.json', {'UTC': n.utc(), 'gate': 'PASS' if selected else 'FAIL',
        'passing_arms': passing, 'selected_common_arm': selected, 'goal_achieved': False,
        'no_real_or_catalog_scoring': True, 'status': n.STATUS})
    print(comparison.to_string(), flush=True)
    print('HT_PROFILE_PILOT_GATE', 'PASS' if selected else 'FAIL', selected, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--pilot-root', type=Path, required=True)
    parser.add_argument('--joint-root', type=Path, required=True)
    parser.add_argument('--sampling-root', type=Path, required=True)
    args = parser.parse_args()
    main(args.root, args.pilot_root, args.joint_root, args.sampling_root)
