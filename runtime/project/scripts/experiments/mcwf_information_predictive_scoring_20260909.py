#!/usr/bin/env python3
"""Integrate simulation-validated information-dependent mass predictions."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import sys
import numpy as np
import pandas as pd

P = Path('/root/autodl-tmp/gw-catalog')
sys.path.insert(0, str(P / 'scripts/experiments'))
import mcwf_profile_expected_information_20260909 as information
import mcwf_profile_single_waveform_20260909 as s
h, co, r, n = s.h, s.co, s.r, s.n
d, score = co.d, co.score
PREDICTIVE = P / 'results/mcwf_nodup_expected_information_18_20260909T110338Z'
PARENT = information.PARENT
ROOT = None
WORKERS = 20
ORIGINAL_MATRICES = score.matrices
METHODS = ('NODUP-DIRECT-REPLAY', 'INFORMATION-JOINT-FIXED', 'INFORMATION-LINEAR',
           'INFORMATION-TREE', 'INFORMATION-ACTIVE-LINEAR', 'INFORMATION-ACTIVE-TREE')
STATE = {}


def worker_init(folder):
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)
    co.app.worker_init(folder)
    STATE.update(co.app.STATE)


def information_event(job):
    k, row = job
    model = information.physical.LowBand(STATE['full20'][k], STATE['frequency'], STATE['psd'][k], 16)
    point = np.array([row['logmc'], row['q'], row['chieff_equal']])
    result = information.information(model, point, STATE['full20'][k])
    result['row_index'] = row['row_index']
    return result


def panel_information(dep, seed, split, catalog, active):
    tag = co.app.tag_for(seed, split, catalog)
    if split == 'development':
        return [json.loads(f.read_text()) for f in sorted((PREDICTIVE / 'events').glob(dep+'_*.json'))]
    if split == 'real' and not (ROOT / 'contracts/EVALUATION_COMPLETE.json').exists():
        raise RuntimeError('Simulation evaluation must precede real waveform processing')
    folder = PARENT / f'cache/profile_inputs/{dep}/{tag}'
    out = ROOT / f'information/{dep}/{tag}'
    out.mkdir(parents=True, exist_ok=True)
    if not active.any():
        return []
    ids = np.load(folder / 'event_ids.npy')
    jobs = []
    for k, idx in enumerate(ids):
        if active[idx] and not (out / f'{idx}.json').exists():
            row = json.loads((PARENT / f'profile_events/{dep}/{tag}/{idx}.json').read_text())
            jobs.append((k, row))
    if jobs:
        with ProcessPoolExecutor(max_workers=min(WORKERS, len(jobs)), mp_context=mp.get_context('spawn'),
                                 initializer=worker_init, initargs=(folder,)) as pool:
            for row in pool.map(information_event, jobs):
                n.write_json(out / f'{row["row_index"]}.json', row)
        print('INFORMATION_PANEL', dep, tag, len(jobs), flush=True)
    return [json.loads((out / f'{idx}.json').read_text()) for idx in ids if active[idx]]


def predict(dep, seed, split, catalog=None):
    slot = n.recipes()[dep, seed]['slot']
    tag = co.app.tag_for(seed, split, catalog)
    dest = ROOT / f'predictions/{dep}/{slot}_{tag}.npz'
    if dest.exists():
        return dict(np.load(dest))
    receipt = json.loads((PREDICTIVE / 'contracts/PREDICTIVE_FROZEN.json').read_text())
    config = PREDICTIVE / 'calibration/PROFILE_EXPECTED_INFORMATION.json'
    if not receipt['both_runs_pass'] or n.sha(config) != receipt['sha256']:
        raise RuntimeError('Information predictive gate/hash failed')
    spec = json.loads(config.read_text())[dep]['spec']
    original = dict(np.load(PARENT / f'predictions/{dep}/{slot}_{tag}.npz'))
    widths = np.full(len(original['p']), np.nan)
    for row in panel_information(dep, seed, split, catalog, original['active']):
        if row['information_valid']:
            widths[int(row['row_index'])] = row['information_logmc_width']
    active = original['active'] & np.isfinite(widths) & (widths >= spec['minimum_h']) & (widths <= spec['maximum_h'])
    p = original['p'].copy()
    if active.any():
        p[active] = information.h.density(original['profile_centers'][active], widths[active], spec)
    if not np.array_equal(p[~active], original['p'][~active], equal_nan=True):
        raise RuntimeError('Inactive predictive density changed')
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest, p=p, outside=original['outside'], active=original['active'],
        profile_centers=original['profile_centers'], information_active=active, information_width=widths)
    n.write_json(dest.with_suffix('.json'), {'UTC': n.utc(), 'information_active': int(active.sum()),
        'profile_active': int(original['active'].sum()), 'parent_fallback_exact': True,
        'no_PE_parameter_input': True, 'not_full_PE': True, 'predictive_spec_sha256': receipt['sha256']})
    return dict(np.load(dest))


def load_panel(dep, seed, split, catalog=None):
    frame = h.load_panel(dep, seed, split, catalog)
    mass = predict(dep, seed, split, catalog)
    ii, jj = frame.idx_i.to_numpy(int), frame.idx_j.to_numpy(int)
    frame['information_active_i'] = mass['information_active'][ii]
    frame['information_active_j'] = mass['information_active'][jj]
    frame['information_width_i'] = mass['information_width'][ii]
    frame['information_width_j'] = mass['information_width'][jj]
    return frame


def infer(frame, config):
    if config['method'] == METHODS[1]:
        return co.BASE_INFER(frame, config)
    return s.infer(frame, config)


def freeze():
    if ROOT.exists():
        raise RuntimeError('Independent output required')
    check = json.loads((PREDICTIVE / 'contracts/PREDICTIVE_FROZEN.json').read_text())
    if not check['both_runs_pass']:
        raise RuntimeError('Both predictive gates must pass')
    for folder in ('contracts', 'configs', 'calibration', 'predictions', 'information', 'cache', 'tables',
                   'audit', 'reports', 'scripts', 'logs', 'manifest', 'results', 'figures'):
        (ROOT / folder).mkdir(parents=True)
    n.write_json(ROOT / 'contracts/ANALYSIS_CONTRACT.json', {
        'id': 'MCWF-NODUP-INFORMATION-INTEGRATED-19', 'UTC': n.utc(), 'status': n.STATUS,
        'goal_achieved': False, 'adaptive_development': True, 'same_both_runs': True,
        'predictive_stage': str(PREDICTIVE), 'predictive_sha256': check['sha256'],
        'density': 'Replace only R10 mass marginal where expected-information covariate is numerically valid and in fit support. Keep conditional eta/chi|Mc unchanged.',
        'interpretation': 'Simulation-calibrated predictive density;not Fisher covariance plugged in as a PE posterior.',
        'methods': list(METHODS), 'fixed_control': 'One new jointBC fitted on source/noise fold0;retain immutable NODUP gamma,beta and outer weights. No old Mc/q regression.',
        'single_waveform_arms': 'Same R14 global/active LINEAR/TREE ONE classifier;embedding cosine and mass/conditionalBC features. No separate Zcos or intrinsic LR addition in these four arms.',
        'fit_tune': 'Source/noise fold0 fit,fold1 proper balanced logloss;ridge .001,.01,.1,1;TREE3/7leaves100iters;all arms retained.',
        'inactive_classifier': 'Immutable original NODUP fallback if no event passes R10 profile quality;expected-information invalid events still retain R10 predictive density.',
        'OOD_and_cap': 'Same frozen fit-feature box;positiveOOD0;caplog(nfittrue+1). No candidate-specific adjustment.',
        'frozen': ['encoder', 'profile point estimates', 'time', 'sky', 'outer weights', 'scope', 'oldresults', 'paper'],
        'forbidden': ['oldencoderMc/q', 'total_score_blend', 'realPE_or_official_scoring', 'eventIDveto'],
        'evaluation': 'Both NODUP and historicalPATH comparisons;all three modelseeds and reusedcatalogs. New real scores only after simulation evaluation.',
        'limitations': 'Finite reused development data and locally approximated information. Not independent confirmation.'})
    rows = pd.read_csv(PREDICTIVE / 'manifest/INPUT_SHA256.csv').to_dict('records')
    for path in sorted((PREDICTIVE / 'calibration').glob('*.json')):
        rows.append({'path': str(path), 'sha256': n.sha(path), 'bytes': path.stat().st_size})
    n.write_csv(ROOT / 'manifest/INPUT_SHA256.csv', rows)
    shutil.copy2(__file__, ROOT / 'scripts/information_predictive_scoring.py')
    n.write_json(ROOT / 'contracts/START_FREEZE.json', {'UTC': n.utc(),
        'script_sha256': n.sha(Path(__file__)), 'contract_sha256': n.sha(ROOT / 'contracts/ANALYSIS_CONTRACT.json')})


def calibrate():
    configs, rows = [], []
    for dep in n.DEPS:
        for seed in n.SEEDS:
            old = h.isolated.ORIGINALS[dep, seed]
            base = {**old, 'method': METHODS[0]}
            configs.append(base)
            frame = load_panel(dep, seed, 'validation')
            reference = infer(frame, base)[0]
            archived = pd.read_parquet(r.PRIOR / f'results/NODUP-DIRECT/{dep}/seed_{seed}/validation/pairs.parquet')
            if not np.array_equal(frame[['idx_i','idx_j']].to_numpy(), archived[['idx_i','idx_j']].to_numpy()):
                raise RuntimeError('Baseline row mismatch')
            difference = float(abs(reference-archived.waveform_score.to_numpy()).max())
            if difference > 1e-12:
                raise RuntimeError('Baseline waveform replay changed')
            fm = n.cf.fast_metrics(frame, n.cf.channels(frame, reference) @ np.asarray(old['weights']))
            wm = n.cf.fast_metrics(frame, reference)
            spec = score.calibrate_one(dep, seed)
            configs.append({**old, 'method': METHODS[1], 'joint_calibration_subgrid': spec})
            for active in (False, True):
                s.ACTIVE_FIT = active
                for kind in ('LINEAR', 'TREE'):
                    spec = s.fit_classifier(dep, seed, kind)
                    method = ('INFORMATION-ACTIVE-' if active else 'INFORMATION-')+kind
                    c = {**old, 'method': method, 'waveform_calibrator': spec, 'active_only': active}
                    z, oo, cl = infer(frame, c)
                    m = n.cf.fast_metrics(frame, n.cf.channels(frame, z) @ np.asarray(old['weights']))
                    ww = n.cf.fast_metrics(frame, z)
                    c.update(tune_guard=n.cf.guard(m, fm) and n.cf.guard(ww, wm), tune_metrics=m)
                    poisoned = frame.copy()
                    poisoned['pair_key'] = 'ignored'
                    poisoned['pe_mc_bhattacharyya_coefficient'] = -99.
                    poisoned['official_po_fpp'] = 1.
                    assert np.array_equal(z, infer(poisoned, c)[0])
                    configs.append(c)
                    rows.append({'deployment': dep, 'seed': seed, 'method': method,
                        'baseline_max_difference': difference, 'forbidden_input_delta': 0.,
                        'tune_guard': c['tune_guard'], 'calibration_logloss': spec['selected']['logloss'], **m})
                    print('INFORMATION_SCORE_SELECTION', dep, seed, method, c['tune_guard'], flush=True)
    n.write_csv(ROOT / 'tables/INFORMATION_SCORING_SELECTION.csv', rows)
    n.write_json(ROOT / 'configs/SELECTED_CONFIGURATIONS.json', configs)
    n.write_json(ROOT / 'contracts/CONFIGURATIONS_FROZEN.json', {'UTC': n.utc(),
        'file': 'configs/SELECTED_CONFIGURATIONS.json', 'sha256': n.sha(ROOT / 'configs/SELECTED_CONFIGURATIONS.json'),
        'real_or_test_selection': False, 'old_total_blend': False})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--stage', choices=('freeze', 'calibrate', 'evaluate', 'real'), required=True)
    parser.add_argument('--workers', type=int, default=20)
    args = parser.parse_args()
    ROOT = s.ROOT = h.ROOT = co.ROOT = r.ROOT = score.ROOT = d.ROOT = args.root
    WORKERS = args.workers
    r.install()
    co.mass = d.predict = predict
    co.matrices = score.matrices = ORIGINAL_MATRICES
    s.METHODS = score.METHODS = METHODS
    n.METHODS, n.load_panel, n.infer = METHODS, load_panel, infer
    old_export, old_consensus = n.public_frame, n.dev.BASE.consensus_real
    fields = h.FEATURES + ('information_active_i', 'information_active_j', 'information_width_i', 'information_width_j')
    def export(frame, z, weights, method):
        out = old_export(frame, z, weights, method)
        for name in fields:
            out[name] = frame[name].to_numpy()
        return out
    def consensus(items, method):
        out = old_consensus(items, method)
        extra = pd.concat(items).groupby('pair_key')[list(fields)].mean().add_suffix('_mean').reset_index()
        return out.merge(extra, on='pair_key', validate='one_to_one')
    n.public_frame, n.dev.BASE.consensus_real = export, consensus
    if args.stage in ('freeze', 'calibrate'):
        globals()[args.stage]()
    else:
        n.run(ROOT, args.stage)
