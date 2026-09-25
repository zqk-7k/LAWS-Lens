#!/usr/bin/env python3
"""R67: frozen catalog evaluation of a passing R66 waveform calibrator."""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd

P = Path('/root/autodl-tmp/gw-catalog')
sys.path.insert(0, str(P / 'scripts/experiments'))
import mcwf_nodup_conditional_deficit_20260910 as pilot
import mcwf_shared_deficit_tail_catalog_20260910 as archived
import mcwf_shared_profile_catalog_audit_20260909 as audit

n, io = pilot.n, pilot.population
METHODS = ('NODUP-DIRECT-REPLAY', 'R62-FROZEN-REPLAY', 'NODUP-CONDITIONAL-DEFICIT')
NEW = METHODS[-1]
ROOT = PILOT = REFERENCE = None
BASE_CSV = n.write_csv


def freeze():
    if ROOT.exists():
        raise RuntimeError('Independent R67 directory required')
    pilot.check(PILOT)
    gate = json.loads((PILOT / 'contracts/PILOT_GATE.json').read_text())
    if gate['gate'] != 'PASS':
        raise RuntimeError('R66 both-run simulation gate required')
    for name in ('contracts', 'configs', 'tables', 'audit', 'reports', 'scripts', 'manifest', 'logs', 'figures', 'results'):
        (ROOT / name).mkdir(parents=True)
    choices = json.loads((PILOT / 'configs/ALL_CONDITIONAL_MODELS.json').read_text())
    original = {(c['deployment'], c['seed']): c for c in n.selections(REFERENCE)
                if c['method'] == 'REFERENCE-REGULARIZED-SINGLE-WF'}
    configs = []
    for (dep, seed), ref in original.items():
        model = choices[f'{dep}/{seed}/{gate["selected_common_arm"]}']
        if model['weights'] != ref['weights']:
            raise RuntimeError('Outer weights changed')
        for method in METHODS:
            configs.append({**ref, 'method': method, 'conditional_model': model})
    contract = {'UTC': io.utc(), 'id': 'MCWF-NODUP-CONDITIONAL-CATALOG-67',
        'status': n.STATUS, 'goal_achieved': False, 'adaptive_development': True,
        'pilot': str(PILOT), 'reference': str(REFERENCE), 'methods': METHODS, 'primary': NEW,
        'selected_common_arm': gate['selected_common_arm'],
        'waveform': 'Within frozen eligible scope apply the SINGLE selected R66 model to NODUP waveform, minimum shared-profile power and deficit. Outside support exact NODUP. R62 scores are comparison only.',
        'no_old_Mc_q_no_total_blend': True,
        'unchanged': ['encoder/predictor checkpoints', 'time', 'sky', 'outer weights', 'scope', 'physical measurements', 'source/noise splits', 'historical results'],
        'evaluation': 'All30 original validation/test/reused injection panels, waveform AND fusion. Six model real tables and two consensus tables after injection calculations. No real outcome selection.',
        'required_goal': 'Keypair outside Top10 consensus and all3models. Both runs consensus AND eachmodel Top10/20 PE/official nonloss versus NODUP, plus every injection panel/mode existing guard. No threshold relaxation.',
        'guards': {'R10_drop': .02, 'AP_drop': .005, 'F50_F90_ratio': 1.1,
                   'PE_higher': audit.HIGHER, 'PE_lower': 'catastrophic_mc'},
        'claims': 'Adaptive development, not a new locked test or physical Bayes factor. Official membership is not a lensing label. No automatic adoption.'}
    io.write(ROOT / 'contracts/ANALYSIS_CONTRACT.json', contract)
    io.write(ROOT / 'configs/SELECTED_CONFIGURATIONS.json', configs)
    io.write(ROOT / 'contracts/CONFIGURATIONS_FROZEN.json', {'UTC': io.utc(),
        'file': 'configs/SELECTED_CONFIGURATIONS.json', 'sha256': io.sha(ROOT / 'configs/SELECTED_CONFIGURATIONS.json'),
        'real_and_test_not_used_for_selection': True})
    files = [Path(__file__), Path(pilot.__file__), Path(archived.__file__), Path(audit.__file__),
             PILOT / 'contracts/PILOT_GATE.json', PILOT / 'configs/ALL_CONDITIONAL_MODELS.json',
             REFERENCE / 'configs/SELECTED_CONFIGURATIONS.json']
    for method in (METHODS[0], 'REFERENCE-REGULARIZED-SINGLE-WF', n.BASELINE):
        files += list((REFERENCE / f'results/{method}').glob('gwtc*/seed_*/*/pairs.parquet'))
        files += list((REFERENCE / f'results/{method}').glob('gwtc*/seed_*/real/fusion_all_pairs.parquet'))
    files += [n.t.EXTERNAL / f'{dep}_external_reference.parquet' for dep in n.DEPS]
    for module in list(sys.modules.values()):
        path = getattr(module, '__file__', None)
        if path and str(P / 'scripts/experiments') in path and Path(path).suffix == '.py':
            files.append(Path(path))
    io.snapshot(ROOT, files)
    shutil.copy2(__file__, ROOT / 'scripts/nodup_conditional_catalog.py')
    io.write(ROOT / 'contracts/START_FREEZE.json', {'UTC': io.utc(), 'runtime_sha256': io.sha(Path(__file__)),
        'contract_sha256': io.sha(ROOT / 'contracts/ANALYSIS_CONTRACT.json'),
        'manifest_sha256': io.sha(ROOT / 'manifest/INPUT_SHA256.csv'),
        'config_sha256': io.sha(ROOT / 'configs/SELECTED_CONFIGURATIONS.json')})
    print('CONDITIONAL_CATALOG_FROZEN', ROOT, flush=True)


def check():
    record = json.loads((ROOT / 'contracts/START_FREEZE.json').read_text())
    for key, file in [('runtime_sha256', Path(__file__)), ('contract_sha256', ROOT / 'contracts/ANALYSIS_CONTRACT.json'),
                      ('manifest_sha256', ROOT / 'manifest/INPUT_SHA256.csv'), ('config_sha256', ROOT / 'configs/SELECTED_CONFIGURATIONS.json')]:
        if io.sha(file) != record[key]:
            raise RuntimeError('Frozen runtime/config changed')
    files = pd.read_csv(ROOT / 'manifest/INPUT_SHA256.csv')
    for row in files.itertuples():
        if io.sha(row.path) != row.sha256:
            raise RuntimeError('Protected input changed: ' + row.path)
    return len(files)


def load_panel(dep, seed, split, catalog=None):
    archived.REFERENCE = REFERENCE
    return archived.load_panel(dep, seed, split, catalog)


def infer(frame, config):
    method = config['method']
    prefix = 'R62' if method == METHODS[1] else 'NODUP'
    z, ood, clipped = [frame[prefix + '_frozen_' + k].to_numpy(copy=True) for k in ('waveform', 'ood', 'clip')]
    active = frame.shared_profile_eligible.to_numpy(bool)
    used = np.zeros(len(frame), bool)
    outside = np.zeros(len(frame), bool)
    if method == NEW:
        c = config['conditional_model']
        zz, oo, cc = pilot.apply(z[active], frame.shared_profile_minimum_power.to_numpy(float)[active],
                                 frame.shared_profile_deficit.to_numpy(float)[active], c)
        z[active] = zz
        outside[active] = oo
        if not c['no_update']:
            ood[active], clipped[active] = oo, cc
            used[active] = ~oo
        if not np.array_equal(z[~active], frame.NODUP_frozen_waveform.to_numpy()[~active]):
            raise RuntimeError('Inactive NODUP fallback changed')
    if not np.isfinite(z).all():
        raise RuntimeError('Nonfinite waveform')
    frame['conditional_waveform_used'] = used
    frame['conditional_waveform_ood'] = outside
    frame['conditional_waveform_no_update'] = bool(config['conditional_model']['no_update'])
    frame['conditional_NODUP_reference'] = frame.NODUP_frozen_waveform.to_numpy(copy=True)
    frame['shared_profile_used'] = used if method == NEW else active & (method == METHODS[1])
    frame['shared_profile_candidate_waveform'] = z
    frame['NN_joint_BC_used_in_waveform'] = True
    return z, ood, clipped


def export(frame, z, weights, method):
    out = archived.export(frame, z, weights, method)
    for name in frame:
        if name.startswith('conditional_'):
            out[name] = frame[name].to_numpy(copy=True)
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
    configs = {(c['deployment'], c['seed'], c['method']): c for c in n.selections(ROOT)}
    rows = []
    for dep in n.DEPS:
        for seed in n.SEEDS:
            f = load_panel(dep, seed, 'validation')
            for method in METHODS:
                c = configs[dep, seed, method]
                z = infer(f.copy(), c)[0]
                poison = f.copy()
                for field in ('time_score', 'sky_raw_log_bf', 'waveform_score', 'final_score', 'PATH875_waveform',
                              'PATH875_final_score', 'old_Mc_score', 'old_q_score', 'pe_mc_bhattacharyya_coefficient', 'official_frontend'):
                    poison[field] = -99999.
                if not np.array_equal(z, infer(poison, {**c, 'alpha': 999., 'old_Mc_weight': 999.})[0]):
                    raise RuntimeError('Forbidden score input')
                if not np.array_equal(z, infer(f.iloc[::-1].copy(), c)[0][::-1]):
                    raise RuntimeError('Pair ordering changes score')
                if method != NEW:
                    col = 'NODUP_frozen_waveform' if method == METHODS[0] else 'R62_frozen_waveform'
                    if not np.array_equal(z, f[col]):
                        raise RuntimeError('Historical replay changed')
                rows.append({'deployment': dep, 'seed': seed, 'method': method, 'forbidden_delta': 0., 'order_delta': 0.})
            # Original validation rarely activates the branch. Exercise it using simulation pilot rows.
            g = pd.read_parquet(PILOT / 'tables/PREDICTIONS.parquet')
            c = configs[dep, seed, NEW]
            arm = c['conditional_model']['arm']
            g = g[(g.deployment == dep) & (g.seed == seed) & (g.arm == arm)].copy()
            measured = pilot.apply(g.NODUP_waveform.to_numpy(), g.minimum_independent_power.to_numpy(),
                                  g.deficit.to_numpy(), c['conditional_model'])[0]
            if not np.array_equal(measured, g.waveform_score.to_numpy()):
                raise RuntimeError('R66 serialization replay changed')
    io.csv(ROOT / 'audit/PIPELINE_UNITS.csv', rows)
    io.write(ROOT / 'contracts/PIPELINE_UNIT_PASS.json', {'UTC': io.utc(), 'tests': len(rows),
        'active_simulation_rows_replayed': True, 'forbidden_inputs_no_effect': True})


def run(stage):
    check()
    if not (ROOT / 'contracts/PIPELINE_UNIT_PASS.json').exists():
        raise RuntimeError('Pre-evaluation units required')
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
