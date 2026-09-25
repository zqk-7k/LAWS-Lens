#!/usr/bin/env python3
"""R62 catalog replay of the simulation-selected R61 classifier."""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
import argparse
from functools import lru_cache
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd

P = Path('/root/autodl-tmp/gw-catalog')
sys.path.insert(0, str(P / 'scripts/experiments'))
import mcwf_shared_reference_regularization_20260910 as pilot
import mcwf_shared_profile_catalog_20260909 as parent
import mcwf_shared_profile_catalog_audit_20260909 as audit

n = parent.n
METHODS = ('NODUP-DIRECT-REPLAY', 'R55-FROZEN-REPLAY', 'REFERENCE-REGULARIZED-SINGLE-WF')
NEW = METHODS[2]
ROOT = PILOT = REFERENCE = PHYSICAL = None
BASE_EXPORT = parent.export
JOINT = P / 'results/mcwf_conditional_intrinsics_exploratory_20260908T154330Z/predictions/CONDITIONAL-ETA-CHI'


def archive_path(root, method, dep, seed, split, catalog=None):
    folder = split if catalog is None else f'{split}_{catalog}'
    name = 'fusion_all_pairs.parquet' if split == 'real' else 'pairs.parquet'
    return root / f'results/{method}/{dep}/seed_{seed}/{folder}/{name}'


def joint_path(dep, seed, split, catalog=None):
    slot = n.recipes()[dep, seed]['slot']
    if catalog is None:
        return JOINT / f'{dep}/{slot}_{seed}_{split}.npz'
    return n.FRESH / f'confirmation/{dep}/catalog_{catalog}/model_{slot}/joint_predictions.npz'


def freeze():
    if ROOT.exists():
        raise RuntimeError('Independent R62 output required')
    pilot.check(PILOT)
    gate = json.loads((PILOT / 'contracts/PILOT_GATE.json').read_text())
    if gate['gate'] != 'PASS':
        raise RuntimeError('Simulation pilot must pass')
    for name in ('contracts', 'configs', 'tables', 'audit', 'reports', 'scripts', 'manifest', 'results', 'logs', 'figures'):
        (ROOT / name).mkdir(parents=True)
    chosen = gate['selected_common_arm']['arm']
    selected = json.loads((PILOT / 'configs/ALL_SELECTED_CLASSIFIERS.json').read_text())
    refs = {(c['deployment'], c['seed']): c for c in json.loads(
        (REFERENCE / 'configs/SELECTED_CONFIGURATIONS.json').read_text()) if c['method'] == 'SHARED-PROFILE-MONOTONE-BOUNDARY'}
    configs = []
    for (dep, seed), ref in refs.items():
        for method in METHODS:
            c = selected[f'{dep}/{seed}/{chosen}'] if method == NEW else ref
            if c['weights'] != ref['weights']:
                raise RuntimeError('Outer weights changed')
            configs.append({**c, 'method': method, 'reference_method': 'R55-FROZEN',
                            'selected_common_arm': chosen})
    contract = {'UTC': n.utc(), 'id': 'MCWF-REFERENCE-REGULARIZED-CATALOG-62',
        'pilot': str(PILOT), 'reference': str(REFERENCE), 'physical': str(PHYSICAL),
        'status': n.STATUS, 'goal_achieved': False, 'adaptive_development': True,
        'methods': METHODS, 'primary': NEW, 'selected_common_arm': chosen,
        'new_method': 'Use the single R61 selected waveform classifier on the unchanged eligible scope. R61 no-update returns exact R55. Ineligible pairs retain exact NODUP, with explicit flags.',
        'joint_feature': 'For updated4D configurations only: read existing learned event joint arrays, normalize once per event in float64, dot square-root probabilities. No public PE. Do not rely on rounded legacy pair BC or clipped old logBC.',
        'no_new_strain_fit_PE_or_encoder_training': True,
        'frozen': ['encoder/predictor checkpoints', 'time', 'sky', 'outer weights', 'scope', 'physical pair measurements', 'R55 eligibility/support/cap'],
        'no_old_Mc_q_score_or_total_blend': True,
        'evaluation': 'All original validation/test plus3reused catalogs,3models,2runs. Reused data are not a new locked test.',
        'external': 'After complete injection evaluation only; PE/official joined after ranking, never select configurations.',
        'goal_guards': json.loads((REFERENCE / 'audit/REFERENCE_AUDIT_FROZEN.json').read_text())['PE_no_loss'],
        'injection_guard': 'Existing n.cf.guard per model/panel and waveform/fusion against NODUP; output strict pointwise no-loss separately. No tolerance changes.',
        'key_pair_must_leave_top10': 'O3 consensus and all3models',
        'known_limit': 'All O3 R61 selections are exact reference fallback. This evaluation cannot itself resolve the known O3 per-model external budget losses; do not declare overall success.'}
    pilot.common.write_once(ROOT / 'contracts/ANALYSIS_CONTRACT.json', contract)
    n.write_json(ROOT / 'configs/SELECTED_CONFIGURATIONS.json', configs)
    pilot.common.write_once(ROOT / 'contracts/CONFIGURATIONS_FROZEN.json', {'UTC': n.utc(),
        'file': 'configs/SELECTED_CONFIGURATIONS.json', 'sha256': n.sha(ROOT / 'configs/SELECTED_CONFIGURATIONS.json'),
        'real_or_catalog_selection': False})
    files = [Path(__file__), Path(parent.__file__), Path(pilot.__file__), Path(pilot.common.__file__), Path(audit.__file__),
        PILOT / 'contracts/PILOT_GATE.json', PILOT / 'configs/ALL_SELECTED_CLASSIFIERS.json',
        REFERENCE / 'configs/SELECTED_CONFIGURATIONS.json', REFERENCE / 'audit/REFERENCE_AUDIT_FROZEN.json']
    for method in ('NODUP-DIRECT-REPLAY', 'SHARED-PROFILE-MONOTONE-BOUNDARY'):
        files.extend((REFERENCE / f'results/{method}').glob('gwtc*/seed_*/*/pairs.parquet'))
        files.extend((REFERENCE / f'results/{method}').glob('gwtc*/seed_*/real/fusion_all_pairs.parquet'))
    files.extend(PHYSICAL.glob('pairs/*/*/*.json'))
    for c in configs:
        if c['method'] != NEW or c['features'] != 4:
            continue
        for split, catalog in [('validation', None), ('test', None), ('real', None)] + [('sept8_reused', v) for v in n.CATALOGS]:
            files.append(joint_path(c['deployment'], c['seed'], split, catalog))
    n.write_csv(ROOT / 'manifest/INPUT_SHA256.csv', [{'path': str(p), 'sha256': n.sha(p)} for p in sorted(set(files))])
    shutil.copy2(__file__, ROOT / 'scripts/reference_regularized_catalog.py')
    pilot.common.write_once(ROOT / 'contracts/START_FREEZE.json', {'UTC': n.utc(),
        'runtime_sha256': n.sha(Path(__file__)), 'contract_sha256': n.sha(ROOT / 'contracts/ANALYSIS_CONTRACT.json'),
        'config_sha256': n.sha(ROOT / 'configs/SELECTED_CONFIGURATIONS.json'),
        'input_manifest_sha256': n.sha(ROOT / 'manifest/INPUT_SHA256.csv')})
    print('REFERENCE_CATALOG_FROZEN', chosen, len(files), flush=True)


@lru_cache(maxsize=2)
def pair_bc(dep, seed, split, catalog, ids):
    file = joint_path(dep, seed, split, catalog)
    with np.load(file) as bank:
        arr = bank['joint'][np.asarray(ids)].astype(float).reshape(len(ids), -1)
    sums = arr.sum(1)
    if not np.isfinite(arr).all() or np.any(arr < 0) or np.any(sums <= 0):
        raise RuntimeError('Invalid frozen learned probability')
    normalized = np.sqrt(arr / sums[:, None])
    bc = (normalized @ normalized.T).clip(1e-300, 1.)
    if not np.allclose(np.diag(bc), 1., atol=1e-12, rtol=0.):
        raise RuntimeError('Normalized BC self-overlap failed')
    return bc, float(abs(sums - 1).max()), file


def load_panel(dep, seed, split, catalog=None):
    parent.ROOT = PHYSICAL
    f = parent.load_panel(dep, seed, split, catalog)
    ref = pd.read_parquet(archive_path(REFERENCE, 'SHARED-PROFILE-MONOTONE-BOUNDARY', dep, seed, split, catalog))
    ref = ref.set_index(['idx_i', 'idx_j']).loc[pd.MultiIndex.from_frame(f[['idx_i', 'idx_j']])]
    for col in ('time_score', 'sky_raw_log_bf', 'embedding_only'):
        if not np.array_equal(f[col], ref[col]):
            raise RuntimeError('R55 frozen input changed')
    for target, source in [('R55_frozen_waveform', 'waveform_score'), ('R55_frozen_ood', 'score_ood'),
                           ('R55_frozen_clip', 'score_clipped'), ('shared_profile_eligible', 'shared_profile_eligible')]:
        f[target] = ref[source].to_numpy()
    f['shared_normalized_joint_BC'] = np.nan
    f['shared_joint_probability_mass_error'] = np.nan
    c = next(c for c in n.selections(ROOT) if c['deployment'] == dep and c['seed'] == seed and c['method'] == NEW)
    if c['features'] == 4 and not c['no_update']:
        use = f.shared_profile_eligible.to_numpy(bool)
        i, j = f.idx_i.to_numpy(int)[use], f.idx_j.to_numpy(int)[use]
        ids = tuple(sorted(set(i) | set(j)))
        if ids:
            matrix, error, path = pair_bc(dep, seed, split, catalog, ids)
            lookup = {v: k for k, v in enumerate(ids)}
            f.loc[use, 'shared_normalized_joint_BC'] = matrix[[lookup[v] for v in i], [lookup[v] for v in j]]
            f.loc[use, 'shared_joint_probability_mass_error'] = error
            if split != 'real':
                print('JOINT_FEATURE_REPLAY', dep, seed, split, catalog, 'mass_error', error, flush=True)
    return f


def infer(frame, config):
    method = config['method']
    active = frame.shared_profile_eligible.to_numpy(bool)
    original = frame.NODUP_frozen_waveform.to_numpy(float)
    if method == METHODS[0]:
        z = original.copy()
        ood, clipped = frame.NODUP_frozen_ood.to_numpy(bool).copy(), frame.NODUP_frozen_clip.to_numpy(bool).copy()
    else:
        z = frame.R55_frozen_waveform.to_numpy(float).copy()
        ood, clipped = frame.R55_frozen_ood.to_numpy(bool).copy(), frame.R55_frozen_clip.to_numpy(bool).copy()
    frame['shared_reference_no_update'] = bool(method != NEW or config.get('no_update', False))
    frame['shared_profile_used'] = active & (method != METHODS[0])
    frame['shared_profile_candidate_waveform'] = z
    frame['shared_profile_OOD'] = ood
    frame['NN_joint_BC_used_in_waveform'] = ~frame.shared_profile_used
    if method == NEW and not config['no_update']:
        c = frame.embedding_only.to_numpy(float)[active]
        p = frame.shared_profile_minimum_power.to_numpy(float)[active]
        d = frame.shared_profile_deficit.to_numpy(float)[active]
        bc = frame.shared_normalized_joint_BC.to_numpy(float)[active] if config['features'] == 4 else None
        value = pilot.common.policy(c, p, d, config, bc)
        x = pilot.common.features(c, p, np.maximum(d, config['deficit_support'][0]), 'log1p', bc)
        raw = pilot.common.classifier.base.predict(x, config['shared_classifier'])
        z[active] = value
        ood[active] = (d < config['deficit_support'][0]) | (d > config['deficit_support'][1])
        clipped[active] = (raw != value) | (d < config['deficit_support'][0])
        frame.loc[active, 'shared_profile_candidate_waveform'] = value
        frame.loc[active, 'shared_profile_OOD'] = ood[active]
        frame.loc[active, 'NN_joint_BC_used_in_waveform'] = config['features'] == 4
    if not np.isfinite(z).all() or not np.array_equal(z[~active], original[~active]):
        raise RuntimeError('Nonfinite score or changed ineligible fallback')
    return z, ood, clipped


def export(frame, z, weights, method):
    out = BASE_EXPORT(frame, z, weights, method)
    for col in ('sky_j50', 'sky_j90', 'shared_normalized_joint_BC', 'shared_joint_probability_mass_error', 'shared_reference_no_update'):
        if col in frame:
            out[col] = frame[col].to_numpy(copy=True)
    return out


def save_csv(path, rows):
    if Path(path).name == 'PER_SEED_PE_OFFICIAL_BUDGETS.csv':
        rows = []
        for dep in n.DEPS:
            for seed in n.SEEDS:
                for method in (n.BASELINE, *METHODS):
                    f = pd.read_parquet(archive_path(ROOT, method, dep, seed, 'real'))
                    rows.extend({**n.dev.budget_row(f, method, dep, 'fusion', b), 'seed': seed} for b in (10, 20, 50, 100))
    return parent.RAW_CSV(path, rows)


def run(stage):
    record = json.loads((ROOT / 'contracts/START_FREEZE.json').read_text())
    for name, file in [('runtime_sha256', Path(__file__)), ('contract_sha256', ROOT / 'contracts/ANALYSIS_CONTRACT.json'),
                       ('config_sha256', ROOT / 'configs/SELECTED_CONFIGURATIONS.json'),
                       ('input_manifest_sha256', ROOT / 'manifest/INPUT_SHA256.csv')]:
        if n.sha(file) != record[name]:
            raise RuntimeError('Frozen catalog runtime changed')
    if not (ROOT / 'contracts/PIPELINE_UNIT_PASS.json').exists():
        raise RuntimeError('Pre-evaluation pipeline invariance units required')
    if stage == 'real' and not (ROOT / 'contracts/EVALUATION_COMPLETE.json').exists():
        raise RuntimeError('Finish injection evaluation before external audit')
    n.METHODS = METHODS
    n.load_panel, n.infer, n.public_frame, n.write_csv = load_panel, infer, export, save_csv
    n.run(ROOT, stage)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('root', 'pilot-root', 'reference-root', 'physical-root'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--stage', choices=('freeze', 'evaluate', 'real'), required=True)
    a = p.parse_args()
    ROOT, PILOT, REFERENCE, PHYSICAL = a.root, a.pilot_root, a.reference_root, a.physical_root
    if a.stage == 'freeze':
        freeze()
    else:
        run(a.stage)
