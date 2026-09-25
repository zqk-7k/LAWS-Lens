#!/usr/bin/env python3
"""R69: frozen full-catalog evaluation of the R68 compatibility ceiling."""
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
sys.path.insert(0, str(P / 'scripts/experiments'))
import mcwf_nodup_conditional_catalog_20260910 as previous
import mcwf_conditional_tail_ceiling_20260910 as pilot

n, io, audit, archived = previous.n, previous.io, previous.audit, previous.archived
METHODS = ('NODUP-DIRECT-REPLAY', 'R67-FROZEN-REPLAY', pilot.METHOD)
NEW = METHODS[-1]
ROOT = PILOT = REFERENCE = None
BASE_CSV = n.write_csv


def freeze():
    if ROOT.exists():
        raise RuntimeError('Independent R69 output required')
    pilot.check(PILOT)
    if json.loads((PILOT / 'contracts/PILOT_GATE.json').read_text())['gate'] != 'PASS':
        raise RuntimeError('R68 two-run pilot must pass')
    for name in ('contracts', 'configs', 'tables', 'audit', 'reports', 'scripts', 'manifest', 'logs', 'figures', 'results'):
        (ROOT / name).mkdir(parents=True)
    candidates = json.loads((PILOT / 'configs/SELECTED_CONFIGURATIONS.json').read_text())
    old = {(c['deployment'], c['seed']): c for c in n.selections(REFERENCE) if c['method'] == previous.NEW}
    configs = []
    for (dep, seed), ref in old.items():
        spec = candidates[f'{dep}/{seed}']
        if ref['conditional_model'] != spec['conditional_model'] or ref['weights'] != spec['weights']:
            raise RuntimeError('R66 waveform model or outer weights changed')
        configs += [{**ref, 'method': method, 'ceiling_model': spec} for method in METHODS]
    contract = {'UTC': io.utc(), 'id': 'MCWF-CONDITIONAL-CEILING-CATALOG-69', 'status': n.STATUS,
        'goal_achieved': False, 'adaptive_development': True, 'pilot': str(PILOT), 'reference': str(REFERENCE),
        'methods': METHODS, 'primary': NEW, 'formula': 'R66 supported waveform score with the fixed R68/R65 tail95 nonpositive ceiling; exact NODUP fallback elsewhere.',
        'only_change': 'Within the frozen supported empirical incompatibility tail, positive waveform reward becomes zero. No added independent evidence.',
        'frozen': ['encoder and predictor checkpoints', 'R66 coefficients', 'R65 quantile threshold', 'physical measurements',
                   'time', 'sky', 'outer weights', 'splits/scope', 'legacy-head removal', 'total-blend removal', 'all previous results'],
        'full_goal': 'Keypair outside Top10 consensus and everymodel; bothrun consensus and everymodel Top10/20 PE+official nonloss versus NODUP; everymodel/panel waveform+fusion existing injection guard. No criterion relaxation.',
        'guard': {'R10_drop': .02, 'AP_drop': .005, 'F50_F90_max_ratio': 1.1},
        'scope': 'All30 frozen injection panels, then six real model tables and two consensuses. Real PE/official fields joined downstream only.',
        'claims': 'Adaptive exploration, not locked confirmation; positive-reward ceiling is a ranking constraint, not a calibrated Bayes factor. Official overlap is not truth. No automatic adoption.'}
    io.write(ROOT / 'contracts/ANALYSIS_CONTRACT.json', contract)
    io.write(ROOT / 'configs/SELECTED_CONFIGURATIONS.json', configs)
    io.write(ROOT / 'contracts/CONFIGURATIONS_FROZEN.json', {'UTC': io.utc(),
        'file': 'configs/SELECTED_CONFIGURATIONS.json', 'sha256': io.sha(ROOT / 'configs/SELECTED_CONFIGURATIONS.json'),
        'real_and_test_not_used_for_selection': True})
    files = [Path(__file__), Path(pilot.__file__), Path(previous.__file__),
             PILOT / 'contracts/PILOT_GATE.json', PILOT / 'configs/SELECTED_CONFIGURATIONS.json',
             PILOT / 'tables/PREDICTIONS.parquet', REFERENCE / 'configs/SELECTED_CONFIGURATIONS.json',
             REFERENCE / 'contracts/ANALYSIS_CONTRACT.json']
    for root in (PILOT, REFERENCE):
        files += [Path(r.path) for r in pd.read_csv(root / 'manifest/INPUT_SHA256.csv').itertuples()]
    for method in (METHODS[0], previous.NEW, n.BASELINE):
        files += list((REFERENCE / f'results/{method}').glob('gwtc*/seed_*/*/pairs.parquet'))
        files += list((REFERENCE / f'results/{method}').glob('gwtc*/seed_*/real/fusion_all_pairs.parquet'))
    for module in list(sys.modules.values()):
        path = getattr(module, '__file__', None)
        if path and str(P / 'scripts/experiments') in path and Path(path).suffix == '.py':
            files.append(Path(path))
    io.snapshot(ROOT, files)
    shutil.copy2(__file__, ROOT / 'scripts/conditional_ceiling_catalog.py')
    io.write(ROOT / 'contracts/START_FREEZE.json', {'UTC': io.utc(), 'runtime_sha256': io.sha(Path(__file__)),
        'contract_sha256': io.sha(ROOT / 'contracts/ANALYSIS_CONTRACT.json'),
        'manifest_sha256': io.sha(ROOT / 'manifest/INPUT_SHA256.csv'),
        'config_sha256': io.sha(ROOT / 'configs/SELECTED_CONFIGURATIONS.json')})
    print('CONDITIONAL_CEILING_CATALOG_FROZEN', ROOT, flush=True)


def check():
    record = json.loads((ROOT / 'contracts/START_FREEZE.json').read_text())
    for key, file in [('runtime_sha256', Path(__file__)), ('contract_sha256', ROOT / 'contracts/ANALYSIS_CONTRACT.json'),
                      ('manifest_sha256', ROOT / 'manifest/INPUT_SHA256.csv'), ('config_sha256', ROOT / 'configs/SELECTED_CONFIGURATIONS.json')]:
        if io.sha(file) != record[key]:
            raise RuntimeError('Frozen runtime/config changed')
    files = pd.read_csv(ROOT / 'manifest/INPUT_SHA256.csv')
    for r in files.itertuples():
        if io.sha(r.path) != r.sha256:
            raise RuntimeError('Protected input changed: ' + r.path)
    return len(files)


def load_panel(dep, seed, split, catalog=None):
    parent = json.loads((REFERENCE / 'contracts/ANALYSIS_CONTRACT.json').read_text())
    previous.REFERENCE = Path(parent['reference'])
    f = previous.load_panel(dep, seed, split, catalog)
    old = pd.read_parquet(archived.path(REFERENCE, previous.NEW, dep, seed, split, catalog))
    old = old.sort_values(['idx_i', 'idx_j'], kind='stable').reset_index(drop=True)
    for col in ('idx_i', 'idx_j', 'time_score', 'sky_raw_log_bf', 'embedding_only'):
        if not f[col].equals(old[col]):
            raise RuntimeError('R67 source panel mismatch: ' + col)
    for target, col in [('waveform', 'waveform_score'), ('ood', 'score_ood'), ('clip', 'score_clipped')]:
        f['R67_frozen_' + target] = old[col].to_numpy(copy=True)
    return f


def infer(frame, config):
    method = config['method']
    prefix = 'R67' if method == METHODS[1] else 'NODUP'
    z, ood, clipped = [frame[prefix + '_frozen_' + k].to_numpy(copy=True) for k in ('waveform', 'ood', 'clip')]
    active = frame.shared_profile_eligible.to_numpy(bool)
    tail, changed, used = [np.zeros(len(frame), bool) for _ in range(3)]
    if method == NEW:
        z[active], oo, cc, tt, hh = pilot.apply(z[active], frame.shared_profile_minimum_power.to_numpy(float)[active],
            frame.shared_profile_deficit.to_numpy(float)[active], config['ceiling_model'])
        ood[active], clipped[active], tail[active], changed[active], used[active] = oo, cc, tt, hh, ~oo
        before = frame.R67_frozen_waveform.to_numpy()
        if np.any(z > before) or not np.array_equal(z[~tail], before[~tail]):
            raise RuntimeError('Ceiling affected an ineligible score or increased it')
    if not np.isfinite(z).all():
        raise RuntimeError('Nonfinite score')
    frame['ceiling_tail_screen'], frame['ceiling_positive_withheld'] = tail, changed
    frame['ceiling_R67_reference'] = frame.R67_frozen_waveform.to_numpy(copy=True)
    frame['ceiling_cutoff_D'] = config['ceiling_model']['tail_calibration']['cutoff_D']
    frame['shared_profile_used'] = used if method == NEW else active & (method == METHODS[1])
    frame['shared_profile_candidate_waveform'] = z
    frame['NN_joint_BC_used_in_waveform'] = True
    return z, ood, clipped


def export(frame, z, weights, method):
    out = archived.export(frame, z, weights, method)
    for col in frame:
        if col.startswith('ceiling_'):
            out[col] = frame[col].to_numpy(copy=True)
    return out


def save_csv(file, rows):
    if Path(file).name == 'PER_SEED_PE_OFFICIAL_BUDGETS.csv':
        rows = []
        for dep in n.DEPS:
            for seed in n.SEEDS:
                for method in (n.BASELINE, *METHODS):
                    f = pd.read_parquet(archived.path(ROOT, method, dep, seed, 'real'))
                    rows += [{**n.dev.budget_row(f, method, dep, 'fusion', b), 'seed': seed} for b in (10, 20, 50, 100)]
    BASE_CSV(file, rows)


def units():
    check()
    rows = []
    for c in n.selections(ROOT):
        dep, seed, method = c['deployment'], c['seed'], c['method']
        f = load_panel(dep, seed, 'validation')
        z = infer(f.copy(), c)[0]
        poisoned = f.copy()
        for col in ('time_score', 'sky_raw_log_bf', 'waveform_score', 'final_score', 'PATH875_waveform',
                    'PATH875_final_score', 'old_Mc_score', 'old_q_score', 'pe_mc_bhattacharyya_coefficient', 'official_frontend'):
            poisoned[col] = -99999.
        if not np.array_equal(z, infer(poisoned, {**c, 'alpha': .875, 'legacy_mc_weight': 999.})[0]):
            raise RuntimeError('Forbidden score dependency')
        if not np.array_equal(z, infer(f.iloc[::-1].copy(), c)[0][::-1]):
            raise RuntimeError('Row ordering affects score')
        if method != NEW:
            prefix = 'NODUP' if method == METHODS[0] else 'R67'
            if not np.array_equal(z, f[prefix + '_frozen_waveform'].to_numpy()):
                raise RuntimeError('Historical replay failed')
        if method == NEW:
            g = pd.read_parquet(PILOT / 'tables/PREDICTIONS.parquet')
            g = g[(g.deployment == dep) & (g.seed == seed)]
            predicted = pilot.apply(g.NODUP_waveform.to_numpy(), g.minimum_independent_power.to_numpy(),
                                    g.deficit.to_numpy(), c['ceiling_model'])[0]
            if not np.array_equal(predicted, g.waveform_score.to_numpy()):
                raise RuntimeError('Pilot active-row serialization replay failed')
        rows.append({'deployment': dep, 'seed': seed, 'method': method, 'forbidden_delta': 0., 'ordering_delta': 0.})
    io.csv(ROOT / 'audit/PIPELINE_UNITS.csv', rows)
    io.write(ROOT / 'contracts/PIPELINE_UNIT_PASS.json', {'UTC': io.utc(), 'tests': len(rows), 'active_pilot_replayed': True})


def run(stage):
    check()
    if not (ROOT / 'contracts/PIPELINE_UNIT_PASS.json').exists():
        raise RuntimeError('Units first')
    if stage == 'real' and not (ROOT / 'contracts/EVALUATION_COMPLETE.json').exists():
        raise RuntimeError('Injection evaluation first')
    n.METHODS = METHODS
    n.load_panel, n.infer, n.public_frame, n.write_csv = load_panel, infer, export, save_csv
    n.run(ROOT, stage)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for arg in ('root', 'pilot', 'reference'):
        parser.add_argument('--' + arg, required=True, type=Path)
    parser.add_argument('--stage', choices=('freeze', 'units', 'evaluate', 'real'), required=True)
    a = parser.parse_args()
    ROOT, PILOT, REFERENCE = a.root, a.pilot, a.reference
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=1)
    if a.stage in ('freeze', 'units'):
        globals()[a.stage]()
    else:
        run(a.stage)
