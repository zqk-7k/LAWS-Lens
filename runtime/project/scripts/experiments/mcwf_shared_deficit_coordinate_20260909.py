#!/usr/bin/env python3
"""R59: finite simulation-only comparison of shared-fit deficit coordinates."""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
import argparse
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import pandas as pd

P = Path('/root/autodl-tmp/gw-catalog')
sys.path.insert(0, str(P / 'scripts/experiments'))
import mcwf_shared_joint_feature_pilot_20260909 as classifier
import mcwf_shared_profile_monotone_boundary_20260909 as boundary

n = classifier.n
RIDGES = (1e-5, 3e-5, 1e-4, 3e-4, .001, .003, .01, .03, .1, .3, 1.)
COORDINATES = ('log1p', 'sqrt', 'raw')
ARMS = tuple(f'{c}-{d}D' for d in (3, 4) for c in COORDINATES)


def write_once(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2)


def features(cosine, power, deficit, coordinate, joint=None):
    deficit = np.asarray(deficit, dtype=float)
    if np.any(deficit < 0) or not np.isfinite(deficit).all():
        raise RuntimeError('Deficit must be finite and nonnegative')
    value = {'log1p': np.log1p, 'sqrt': np.sqrt, 'raw': lambda a: a}[coordinate](deficit)
    x = np.column_stack([cosine, np.log(power), -value])
    return x if joint is None else np.column_stack([x, np.log(joint)])


def policy(cosine, power, deficit, config, joint=None):
    """Apply support in physical D units, before the coordinate transform."""
    low, high = config['deficit_support']
    x = features(cosine, power, np.maximum(deficit, low), config['coordinate'], joint)
    raw = classifier.base.predict(x, config['shared_classifier'])
    cap = config['shared_cap']
    z = raw if cap is None else raw.clip(-cap, cap)
    return np.where(np.asarray(deficit) > high, np.minimum(z, 0.), z)


def freeze(root, expansion, previous):
    if root.exists():
        raise RuntimeError('Independent R59 directory required')
    for name in ('contracts', 'configs', 'tables', 'audit', 'scripts', 'reports', 'manifest', 'logs'):
        (root / name).mkdir(parents=True)
    source = json.loads((expansion / 'contracts/ANALYSIS_CONTRACT.json').read_text())
    old_gate = json.loads((previous / 'contracts/PILOT_GATE.json').read_text())
    if old_gate['gate'] != 'FAIL':
        raise RuntimeError('Expected archived R58 failure; do not reinterpret it')
    contract = {
        'UTC': n.utc(), 'id': 'MCWF-SHARED-DEFICIT-COORDINATE-59',
        'status': n.STATUS, 'goal_achieved': False, 'adaptive_development': True,
        'expansion': str(expansion), 'previous': str(previous), 'data': source['data'],
        'arms': ARMS, 'ridge_grid': RIDGES,
        'scientific_question': 'Does logarithmic compression of a shared-source projection deficit limit the linear calibration? Compare three fixed monotone coordinates, with/without the same learned joint-distribution BC.',
        'physical_statistic': 'D=P_independent-P_shared >=0, frozen measurement; P is a surrogate projection power, NOT normalized log likelihood or optimal SNR.',
        'coordinate_rationale': {
            'log1p': 'Historical concave robust compression; exact R58 control.',
            'sqrt': 'Intermediate concave mismatch magnitude; local quadratic-distance motivation only, no sigma or chi-square interpretation.',
            'raw': 'Preserves the additive shared-versus-independent projection loss scale; no Bayes-factor interpretation.'},
        'fit': 'Same R56 source/noise-isolated fold0, HT/source-pair class-balanced loss and monotone logistic slopes. Standardization uses fold0 only.',
        'selection': 'For each run/model/arm, select ridge by fold1 full HT capped-policy proper logloss; exact tie chooses stronger ridge.',
        'eligibility_and_policy': 'Exactly R55 frozen power support, optimization flags, D lower-bound saturation, upper-tail positive-neutral policy and cap16. Apply D boundaries before coordinate transform.',
        'gate': 'Every full-HT, random-draw and neighbor-population mean tune loss must be nonworse than R55 in BOTH runs, averaged over3frozen encoders; at leastone strict improvement. Tolerance1e-12 is numerical only.',
        'common_arm_selection': 'Among passing arms maximize worst-run relative full-HT gain, then mean gain, then fewer features, then fixed ARMS order.',
        'future_guards': 'Unchanged R51/R55 per-model/per-catalog waveform+fusion injection guards and consensus+per-model PE/official no-loss budgets. Calibration PASS is not full goal success.',
        'on_fail': 'Stop at simulation pilot. No catalog or real rescoring for failed arms; retain all results. Do not extend coordinate/grid in this contract.',
        'frozen': source['unchanged'] + ['physical pair fits', 'R55 boundaries/caps'],
        'no_old_encoder_Mc_q_score': True, 'no_total_score_blend': True,
        'no_real_PE_or_official_selection': True, 'no_new_blind_confirmation': True,
        'references': [
            {'url': 'https://arxiv.org/abs/gr-qc/9402014',
             'supports': 'Mass/spin information and degeneracy in inspiral waveform.',
             'does_not_prove': 'This projection statistic, coordinate or neural representation is optimal.'},
            {'url': 'https://arxiv.org/abs/2104.09339',
             'supports': 'Shared-source versus independent-source comparison.',
             'does_not_prove': 'Our discriminative logit is the Hanabi evidence or physical Bayes factor.'},
            {'url': 'https://www.jmlr.org/papers/v11/cawley10a.html',
             'supports': 'Repeated model selection can overfit validation.',
             'implication': 'This finite comparison is adaptive development, not independent confirmation.'}],
    }
    write_once(root / 'contracts/ANALYSIS_CONTRACT.json', contract)
    data = Path(source['data'])
    paths = [Path(__file__), Path(classifier.__file__), Path(classifier.base.__file__),
             Path(boundary.__file__), expansion / 'contracts/ANALYSIS_CONTRACT.json',
             expansion / 'contracts/PAIR_PLAN.parquet', expansion / 'tables/PAIR_RESULTS.parquet',
             expansion / 'configs/FROZEN_REFERENCE_CONFIGS.json',
             previous / 'contracts/PILOT_GATE.json', previous / 'tables/ALL_CALIBRATION_METRICS.csv',
             previous / 'configs/EXPANDED_JOINT_CLASSIFIERS.json']
    for dep in n.DEPS:
        paths.extend([data / f'data/{dep}/event_metadata.parquet',
                      data / f'data/{dep}/noise/noise_manifest.csv'])
        for seed in n.SEEDS:
            slot = n.recipes()[dep, seed]['slot']
            paths.extend([data / f'predictions/{dep}/{seed}_embedding.npy',
                          data / f'predictions/{dep}/{slot}_parent.npz'])
    n.write_csv(root / 'manifest/INPUT_SHA256.csv',
                [{'path': str(p), 'sha256': n.sha(p)} for p in sorted(set(paths))])
    shutil.copy2(__file__, root / 'scripts/shared_deficit_coordinate.py')
    write_once(root / 'contracts/START_FREEZE.json', {
        'UTC': n.utc(), 'runtime_sha256': n.sha(Path(__file__)),
        'contract_sha256': n.sha(root / 'contracts/ANALYSIS_CONTRACT.json'),
        'input_manifest_sha256': n.sha(root / 'manifest/INPUT_SHA256.csv')})
    print('COORDINATE_STUDY_FROZEN', root, flush=True)


def check(root):
    frozen = json.loads((root / 'contracts/START_FREEZE.json').read_text())
    for name, path in [('runtime_sha256', Path(__file__)),
                       ('contract_sha256', root / 'contracts/ANALYSIS_CONTRACT.json'),
                       ('input_manifest_sha256', root / 'manifest/INPUT_SHA256.csv')]:
        if n.sha(path) != frozen[name]:
            raise RuntimeError('Frozen contract/runtime changed')
    for row in pd.read_csv(root / 'manifest/INPUT_SHA256.csv').itertuples():
        if n.sha(Path(row.path)) != row.sha256:
            raise RuntimeError('Frozen input changed: ' + row.path)


def fit(root):
    check(root)
    if (root / 'contracts/PILOT_GATE.json').exists():
        raise RuntimeError('Completed pilot is immutable')
    start = time.monotonic()
    contract = json.loads((root / 'contracts/ANALYSIS_CONTRACT.json').read_text())
    expansion, data, previous = (Path(contract[k]) for k in ('expansion', 'data', 'previous'))
    pairs = pd.read_parquet(expansion / 'tables/PAIR_RESULTS.parquet')
    refs = {(r['deployment'], r['seed']): r for r in json.loads(
        (expansion / 'configs/FROZEN_REFERENCE_CONFIGS.json').read_text())}
    rows, grids, predictions, units, isolation, norms = [], [], [], [], [], []
    configs = {}
    for dep in n.DEPS:
        meta = pd.read_parquet(data / f'data/{dep}/event_metadata.parquet')
        noise = pd.read_csv(data / f'data/{dep}/noise/noise_manifest.csv').set_index('bank_index')
        chunks = meta.noise_bank_index.map(noise.parent_file_gps)
        ns = len(set(meta.source_uid[meta.fold == 0]) & set(meta.source_uid[meta.fold == 1]))
        nn = len(set(chunks[meta.fold == 0]) & set(chunks[meta.fold == 1]))
        if ns or nn:
            raise RuntimeError('Source/noise-parent leakage')
        isolation.append({'deployment': dep, 'source_overlap': ns, 'noise_parent_overlap': nn})
        f = pairs[pairs.deployment == dep].reset_index(drop=True)
        ref = refs[dep, n.SEEDS[0]]
        f = f[f.minimum_independent_power.between(*ref['minimum_power_support']) &
              f.any_shared_converged].reset_index(drop=True)
        train, tune = f.fold.eq(0).to_numpy(), f.fold.eq(1).to_numpy()
        y = f.kind.eq('true').to_numpy(float)
        fitw = classifier.base.balanced(y[train], f.HT_weight.to_numpy()[train])
        tunew = classifier.base.balanced(y[tune], f.HT_weight.to_numpy()[tune])
        d, power = f.deficit.to_numpy(float), f.minimum_independent_power.to_numpy(float)
        diagnostics = {'full_population': tune,
            'random_draw': tune & f.kind.ne('hard_null').to_numpy(),
            'neighbor_population': tune & (f.neighbor_population.to_numpy() | (y == 1))}
        for seed in n.SEEDS:
            ref = refs[dep, seed]
            embedding = np.load(data / f'predictions/{dep}/{seed}_embedding.npy').astype(float)
            embedding /= np.linalg.norm(embedding, axis=1, keepdims=True)
            cosine = np.einsum('ij,ij->i', embedding[f.idx_i], embedding[f.idx_j])
            slot = n.recipes()[dep, seed]['slot']
            with np.load(data / f'predictions/{dep}/{slot}_parent.npz') as bank:
                joint = bank['joint'].astype(float).reshape(len(embedding), -1)
            mass = joint.sum(1)
            if np.any(joint < 0) or not np.isfinite(joint).all() or np.any(mass <= 0):
                raise RuntimeError('Invalid frozen joint probability')
            norms.append({'deployment': dep, 'seed': seed, 'mass_error': float(abs(mass - 1).max())})
            joint = np.sqrt(joint / mass[:, None])
            bc = np.asarray([np.dot(joint[int(i)], joint[int(j)]) for i, j in zip(f.idx_i, f.idx_j)]).clip(1e-300, 1.)
            controls = {'R55-FROZEN': boundary.predict_policy(
                features(cosine, power, d, 'log1p'), ref, True)}
            for arm in ARMS:
                coordinate, dimension = arm.split('-')
                bj = bc if dimension == '4D' else None
                x = features(cosine, power, d, coordinate, bj)
                options = []
                for ridge in RIDGES:
                    spec = classifier.fit(x[train], y[train], fitw, ridge)
                    config = {**ref, 'shared_classifier': spec, 'coordinate': coordinate,
                              'features': int(dimension[0]), 'arm': arm}
                    z = policy(cosine, power, d, config, bj)
                    loss = classifier.base.loss(y[tune], z[tune], tunew)
                    grids.append({'deployment': dep, 'seed': seed, 'arm': arm, 'ridge': ridge,
                                  'tune_HT_policy_logloss': loss})
                    options.append((loss, -ridge, config))
                config = min(options, key=lambda a: (a[0], a[1]))[2]
                configs[f'{dep}/{seed}/{arm}'] = config
                controls[arm] = policy(cosine, power, d, config, bj)
                lo, hi = ref['deficit_support']
                ds = np.unique(np.r_[0., lo / 2, lo, np.nextafter(lo, np.inf),
                    np.geomspace(lo, hi, 100), np.nextafter(hi, np.inf), hi * 1.1])
                zz = policy(np.full(len(ds), .8), np.full(len(ds), np.sqrt(np.prod(ref['minimum_power_support']))),
                            ds, config, np.full(len(ds), .5) if dimension == '4D' else None)
                monotone = bool((np.diff(zz) <= 1e-10).all())
                plateau = bool(np.max(abs(zz[ds <= lo] - zz[0])) <= 1e-10)
                if not monotone or not plateau:
                    raise RuntimeError('D policy monotonicity/boundary unit failed')
                units.append({'deployment': dep, 'seed': seed, 'arm': arm,
                              'monotone': monotone, 'low_D_plateau': plateau})
            for arm, z in controls.items():
                for diagnostic, take in diagnostics.items():
                    raw = f.HT_weight.to_numpy() if diagnostic == 'full_population' else f.source_pair_weight.to_numpy()
                    weights = classifier.base.balanced(y[take], raw[take])
                    rows.append({'deployment': dep, 'seed': seed, 'arm': arm, 'diagnostic': diagnostic,
                        'logloss': classifier.base.loss(y[take], z[take], weights), 'pairs': int(take.sum())})
                out = f[['pair_id', 'deployment', 'fold', 'kind', 'source_i', 'source_j', 'HT_weight',
                         'source_pair_weight', 'neighbor_population', 'deficit', 'minimum_independent_power']].copy()
                out['seed'], out['arm'], out['waveform_score'], out['learned_joint_BC'] = seed, arm, z, bc
                predictions.extend(out.to_dict('records'))
            print('COORDINATE_FIT', dep, seed, flush=True)
    metrics = pd.DataFrame(rows)
    old = pd.read_csv(previous / 'tables/ALL_CALIBRATION_METRICS.csv')
    replays = metrics[metrics.arm.isin(['R55-FROZEN', 'log1p-4D'])].copy()
    replays['arm'] = replays.arm.replace({'log1p-4D': 'EXPANDED-HT-JOINT'})
    replay = replays.merge(old, on=['deployment', 'seed', 'arm', 'diagnostic'], validate='one_to_one', suffixes=('', '_old'))
    maxdiff = float(abs(replay.logloss - replay.logloss_old).max())
    if len(replay) != 36 or maxdiff > 1e-12:
        raise RuntimeError('Historical log-coordinate replay changed')
    summary = metrics.groupby(['deployment', 'arm', 'diagnostic']).logloss.agg(['mean', 'std']).reset_index()
    compare = summary.pivot(index=['deployment', 'diagnostic'], columns='arm', values='mean')
    passing = []
    for arm in ARMS:
        delta = compare[arm] - compare['R55-FROZEN']
        compare[arm + '_delta'] = delta
        if (delta <= 1e-12).all() and (delta < -1e-12).any():
            gain = (-delta / compare['R55-FROZEN']).xs('full_population', level='diagnostic')
            passing.append({'arm': arm, 'worst_run_relative_gain': float(gain.min()),
                            'mean_relative_gain': float(gain.mean())})
    winner = min(passing, key=lambda a: (-a['worst_run_relative_gain'], -a['mean_relative_gain'],
                 int(a['arm'][-2]), ARMS.index(a['arm']))) if passing else None
    n.write_csv(root / 'tables/CALIBRATION_METRICS_PER_SEED.csv', metrics)
    n.write_csv(root / 'tables/CALIBRATION_SUMMARY.csv', summary)
    n.write_csv(root / 'tables/PILOT_GATE_COMPARISON.csv', compare.reset_index())
    n.write_csv(root / 'tables/RIDGE_GRID.csv', grids)
    n.write_csv(root / 'audit/MONOTONICITY_UNITS.csv', units)
    n.write_csv(root / 'audit/FOLD_ISOLATION.csv', isolation)
    n.write_csv(root / 'audit/JOINT_NORMALIZATION.csv', norms)
    n.write_csv(root / 'audit/R58_REPLAY.csv', replay)
    pd.DataFrame(predictions).to_parquet(root / 'tables/PREDICTIONS.parquet', index=False)
    write_once(root / 'configs/ALL_CLASSIFIERS.json', configs)
    write_once(root / 'contracts/PILOT_GATE.json', {'UTC': n.utc(), 'gate': 'PASS' if winner else 'FAIL',
        'passing_arms': passing, 'selected_common_arm': winner, 'goal_achieved': False,
        'status': n.STATUS, 'real_or_catalog_outcomes_read': False,
        'R58_replay_max_difference': maxdiff, 'seconds': time.monotonic() - start})
    print(compare.to_string(), flush=True)
    print('COORDINATE_PILOT_GATE', 'PASS' if winner else 'FAIL', winner, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--expansion-root', type=Path)
    p.add_argument('--previous-root', type=Path)
    p.add_argument('--stage', choices=('freeze', 'fit'), required=True)
    a = p.parse_args()
    if a.stage == 'freeze':
        freeze(a.root, a.expansion_root, a.previous_root)
    else:
        fit(a.root)
