#!/usr/bin/env python3
"""Validate an expected waveform-information covariate, not full PE."""
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
import time
import numpy as np
import pandas as pd
from scipy.ndimage import maximum_filter1d

P = Path('/root/autodl-tmp/gw-catalog')
sys.path.insert(0, str(P / 'scripts/experiments'))
import mcwf_profile_heteroscedastic_20260909 as h
import mcwf_lowband_profile_20260908 as physical
import mcwf_multirate_features_20260908 as replay
n, r, d, cal = h.n, h.r, h.d, h.cal
PARENT, PROFILE = h.PARENT, h.PROFILE
ROOT = None
STATE = {}
STEPS = np.array([.0004, .002, .002])


def freeze():
    if ROOT.exists():
        raise RuntimeError('Independent output required')
    for folder in ('contracts', 'events', 'calibration', 'predictions', 'tables', 'audit',
                   'reports', 'scripts', 'logs', 'manifest'):
        (ROOT / folder).mkdir(parents=True)
    n.write_json(ROOT / 'contracts/ANALYSIS_CONTRACT.json', {
        'id': 'MCWF-NODUP-EXPECTED-INFORMATION-18', 'UTC': n.utc(), 'status': n.STATUS,
        'goal_achieved': False, 'adaptive_development': True, 'same_both_runs': True,
        'parent': str(PARENT), 'profiles': str(PROFILE),
        'motivation': 'Observed profile-power curvature is noisy. Test expected derivative Gram information with amplitude/phase/time nuisances projected out.',
        'data': 'Only R10-quality-active development events, exact original injection and noise; old40Hz2s must replay bitexact. Reuse profile optima; do not refit centers.',
        'operator': 'Same LowBand20-580Hz16s conditioned IMRPhenomD quadratures. Fit detector-specific quadrature amplitudes and allowed peak lag.',
        'information': 'For each detector, D_a = a*dU/dtheta_a+b*dV/dtheta_a. Project D onto orthogonal complement of U,V,time derivative. Sum D_residual.T@D_residual/noise_variance.',
        'noise_variance': 'Robust MAD variance in first8s of the full24s conditioned window; empirical covariate only. It is not an exact stationary Gaussian likelihood.',
        'derivatives': {'coordinates': ['logMc', 'q', 'equal_chi'], 'central_steps': STEPS.tolist(),
            'halfstep_check': True, 'max_width_relative_difference': .10,
            'positive_definite_required': True, 'max_information_condition': 1e12,
            'interior_required': True},
        'important': 'Expected local information is an approximation, not a PE posterior. Its width only conditions simulation-error calibration; never insert inverse Fisher as an exact posterior covariance.',
        'predictive': 'Truncated Student-t around unchanged profile center. Fitfold0 one totalweight/source; df3,5,10,30; ridge.001,.01,.1,1; logscale affine in loginformationwidth.',
        'selection_gate': 'Same R16 proper NLL, MAE, catastrophic-error and global/active source-bootstrap90percent coverage gates, separately O3/O4a; no real/test scoring unless both pass.',
        'fallback': 'Exact R10 density if inactive, invalid, derivative unstable or outside frozen fit support.',
        'frozen': ['encoder', 'time', 'sky', 'outerweights', 'scope', 'historicaloutputs', 'paper'],
        'old_Mc_q_head': False, 'total_score_blend': False,
        'references': ['https://arxiv.org/abs/gr-qc/9402014', 'https://arxiv.org/abs/gr-qc/0703086',
                       'https://arxiv.org/abs/1804.06788']})
    rows = pd.read_csv(PARENT / 'manifest/INPUT_SHA256.csv').to_dict('records')
    for dep in n.DEPS:
        meta = pd.read_parquet(n.t.PREVIOUS / f'expanded_data/{dep}/validation/event_metadata.parquet')
        p = dict(np.load(PARENT / f'predictions/{dep}_development_ENSEMBLE.npz'))
        kept = meta[p['active']].copy()
        kept.to_parquet(ROOT / f'contracts/{dep}_EVENT_PLAN.parquet', index=False)
        for path in (PARENT / f'predictions/{dep}_development_ENSEMBLE.npz',
                     n.t.PREVIOUS / f'expanded_data/{dep}/noise/psd.npy'):
            rows.append({'path': str(path), 'sha256': n.sha(path), 'bytes': path.stat().st_size})
    n.write_csv(ROOT / 'manifest/INPUT_SHA256.csv', rows)
    shutil.copy2(__file__, ROOT / 'scripts/profile_expected_information.py')
    n.write_json(ROOT / 'contracts/START_FREEZE.json', {'UTC': n.utc(),
        'script_sha256': n.sha(Path(__file__)), 'contract_sha256': n.sha(ROOT / 'contracts/ANALYSIS_CONTRACT.json')})


def project_information(derivatives, nuisance, variance):
    u, singular, _ = np.linalg.svd(nuisance.T, full_matrices=False)
    q = u[:, singular > singular[0] * 1e-10]
    residual = derivatives - (derivatives @ q) @ q.T
    return residual @ residual.T / variance, residual


def algebra_unit():
    rng = np.random.default_rng(2026091018)
    z = rng.normal(size=(3, 100))
    nuisance = rng.normal(size=(3, 100))
    g, residual = project_information(z, nuisance, 2.)
    g2, _ = project_information(z + np.eye(3) @ nuisance, nuisance, 2.)
    assert np.max(abs(g - g2)) < 1e-10
    assert np.max(abs(residual @ nuisance.T)) < 1e-10
    assert np.linalg.eigvalsh(g).min() > 0
    return {'passed': True, 'nuisance_invariance': float(np.max(abs(g-g2))),
            'orthogonality': float(np.max(abs(residual @ nuisance.T)))}


def information(model, point, raw24):
    low = np.array([np.log(5.), .25, -.8])
    high = np.array([np.log(200.), 1., .8])
    result = {'information_valid': False, 'interior': bool(((point-STEPS > low)&(point+STEPS < high)).all())}
    if not result['interior']:
        return result
    bank = model.template(point)
    kernels = np.fft.rfft(bank, n=model.nfft, axis=-1)
    c = np.fft.irfft(model.data[:, None] * kernels.conj(), n=model.nfft, axis=-1)
    lag = model.lags
    power = (c[..., lag % model.nfft]**2).sum(1)
    k = int(np.argmax(power[0] + maximum_filter1d(power[1], size=43, mode='constant', cval=-np.inf)))
    near = np.arange(max(k-21, 0), min(k+22, len(lag)))
    kl = int(near[np.argmax(power[1, near])])
    coefficients = np.stack([c[0, :, lag[k] % model.nfft], c[1, :, lag[kl] % model.nfft]])
    first = np.asarray(raw24)[:, :8*2048]
    variance = (np.median(abs(first-np.median(first, axis=-1, keepdims=True)), axis=-1)/.6744897501960817)**2
    if not np.isfinite(variance).all() or (variance <= 0).any():
        raise RuntimeError('Invalid off-source local variance')
    widths, matrices, conditions = [], [], []
    for factor in (1., .5):
        derivative = []
        for axis in range(3):
            delta = np.zeros(3)
            delta[axis] = factor * STEPS[axis]
            derivative.append((model.template(point+delta)-model.template(point-delta))/(2*delta[axis]))
        derivative = np.stack(derivative)
        matrix = np.zeros((3, 3))
        for detector in range(2):
            fitted = np.sum(coefficients[detector, :, None] * bank[detector], axis=0)
            dt = np.fft.irfft(2j*np.pi*np.fft.rfftfreq(len(fitted), 1/2048)*np.fft.rfft(fitted), n=len(fitted))
            nuisance = np.vstack([bank[detector], dt])
            deriv = np.einsum('p,apl->al', coefficients[detector], derivative[:, detector])
            g, _ = project_information(deriv, nuisance, variance[detector])
            matrix += g
        eigen = np.linalg.eigvalsh(matrix)
        condition = float(eigen[-1]/max(eigen[0], 1e-300))
        matrices.append(matrix.tolist())
        conditions.append(condition)
        widths.append(float(np.sqrt(np.linalg.inv(matrix)[0, 0])) if eigen[0] > 0 and condition < 1e12 else None)
    result.update(matrices=matrices, conditions=conditions, widths=widths,
                  projection_power=float(.5*np.sum(coefficients**2)), coefficients=coefficients.tolist(),
                  detector_lags=[int(lag[k]), int(lag[kl])], offsource_variance=variance.tolist())
    if all(x is not None and np.isfinite(x) and x > 0 for x in widths):
        relative = abs(widths[0]/widths[1]-1.)
        result.update(width_relative_difference=float(relative), information_valid=bool(relative <= .10),
                      information_logmc_width=widths[1])
    return result


def initialize_worker(dep):
    replay.init_development(dep, 'validation')
    physical.base.STATE['v3'] = replay.CTX['v3']
    STATE['dep'] = dep


def event(row):
    start = time.perf_counter()
    ctx = replay.CTX
    src, v3 = ctx['src'], ctx['v3']
    dep, index = STATE['dep'], int(row['row_index'])
    old = json.loads((PROFILE / f'events/{dep}_{index}.json').read_text())
    number = 1 if row['image'] == 'a' else 2
    clean, _ = src.detector_response(ctx['generator'], ctx['ifos'],
        src.source_parameters(pd.Series(row), row['gps_'+row['image']]),
        src.lens_factor(row[f'mu_image{number}'], row[f'morse_image{number}']))
    b, off = int(row['noise_bank_index']), int(row['noise_offset_samples'])
    noise = np.asarray(ctx['refs'][b, :, off:off+v3.RAW_PADDED_SAMPLES], np.float32)
    scaled, _, _ = v3.scale_to_network_snr(np.asarray(clean, np.float32), row['target_network_snr'], ctx['freq'], ctx['psds'][b])
    raw = noise + v3.embed_signal_in_padded_window(scaled)
    original = v3.preprocess_24s(raw, ctx['freq'], ctx['psds'][b])
    view = n.dev.TRAIN.make_window_view(original[None].astype(np.float32), 2)[0].astype(np.float16)
    if not np.array_equal(view, ctx['oldraw'][index]):
        raise RuntimeError('Oldpeak2s replay mismatch')
    full = v3.preprocess_24s(raw, ctx['freq'], ctx['psds'][b], band_low_hz=20.)
    model = physical.LowBand(full, ctx['freq'], ctx['psds'][b], 16)
    result = information(model, np.array([old['logmc'], old['q'], old['chieff_equal']]), full)
    result.update(deployment=dep, row_index=index, source_uid=row['source_uid'], old_peak2s_bitexact=True,
                  wall_seconds=time.perf_counter()-start)
    return result


def compute(workers):
    n.write_json(ROOT / 'audit/INFORMATION_ALGEBRA_UNIT.json', algebra_unit())
    for dep in n.DEPS:
        meta = pd.read_parquet(ROOT / f'contracts/{dep}_EVENT_PLAN.parquet')
        pending = [row for row in meta.to_dict('records') if not (ROOT / f'events/{dep}_{int(row["row_index"])}.json').exists()]
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn'),
                                 initializer=initialize_worker, initargs=(dep,)) as pool:
            for k, result in enumerate(pool.map(event, pending), 1):
                n.write_json(ROOT / f'events/{dep}_{result["row_index"]}.json', result)
                if k % 20 == 0 or k == len(pending):
                    print('EXPECTED_INFORMATION', dep, k, len(pending), flush=True)
    n.write_json(ROOT / 'contracts/INFORMATION_COMPLETE.json', {'UTC': n.utc(), 'no_real_or_test_read': True})


def calibrate():
    selected, grid, outputs = {}, [], []
    previous = json.loads((PARENT / 'calibration/PROFILE_PREDICTIVE.json').read_text())
    for dep in n.DEPS:
        meta = pd.read_parquet(n.t.PREVIOUS / f'expanded_data/{dep}/validation/event_metadata.parquet')
        meta['deployment'] = dep
        fold, truth = r.source_fold(meta), np.log(meta.mc_det.to_numpy())
        groups = meta.source_uid.to_numpy(str)
        original = np.mean(cal.parent_predictions(dep), axis=0)
        parent_data = dict(np.load(PARENT / f'predictions/{dep}_development_ENSEMBLE.npz'))
        parent = parent_data['p']
        widths = np.full(len(meta), np.nan)
        centers = parent_data['profile_centers']
        rows = [json.loads(p.read_text()) for p in sorted((ROOT/'events').glob(dep+'_*.json'))]
        for row in rows:
            if row['information_valid']:
                widths[int(row['row_index'])] = row['information_logmc_width']
        eligible = parent_data['active'] & np.isfinite(widths) & (widths > 0)
        fit_mask = eligible & (fold == 0)
        if len(np.unique(groups[fit_mask])) < 20:
            selected[dep] = {'pass': False, 'reason': 'Less than20independentfit sources'}
            continue
        bins = np.searchsorted(d.EDGES, truth, side='right').clip(1, 512)-1
        parent_log = np.log(parent[np.arange(len(parent)), bins].clip(1e-300))-np.log(np.diff(d.EDGES)[bins])
        pa = parent_data['active']
        parent_log[pa] = cal.normalized_logpdf(truth[pa], centers[pa], previous[dep]['spec'])
        tune, wt = fold == 1, h.weights(groups[fold == 1])
        ref_nll = -float(wt @ parent_log[tune])
        olderr = abs(r.mass_summary(original)[:, 0]-truth)
        options = []
        for df in (3, 5, 10, 30):
            for ridge in (.001, .01, .1, 1.):
                spec = h.fit(truth[fit_mask], centers[fit_mask], widths[fit_mask], groups[fit_mask], df, ridge)
                spec['covariate'] = 'expected_information_width_NOT_PE'
                active = eligible & (widths >= spec['minimum_h']) & (widths <= spec['maximum_h'])
                pp, lp = parent.copy(), parent_log.copy()
                pp[active] = h.density(centers[active], widths[active], spec)
                lp[active] = h.logpdf(truth[active], centers[active], widths[active], spec)
                error = abs(r.mass_summary(pp)[:, 0]-truth)
                cov, lo, hi, nt = h.coverage_interval(pp, truth, groups, tune)
                acov, alo, ahi, na = h.coverage_interval(pp, truth, groups, tune & active)
                row = {'deployment': dep, 'df': df, 'ridge': ridge, 'slope': spec['slope'],
                    'fit_sources': spec['fit_sources'], 'tune_sources': nt, 'active_tune_sources': na,
                    'NLL': -float(wt @ lp[tune]), 'R10_NLL': ref_nll,
                    'MAE': float(wt @ error[tune]), 'NN_MAE': float(wt @ olderr[tune]),
                    'catastrophic10percent': int((error[tune] > np.log(1.1)).sum()),
                    'NN_catastrophic10percent': int((olderr[tune] > np.log(1.1)).sum()),
                    'coverage90': cov, 'coverage_low': lo, 'coverage_high': hi,
                    'active_coverage90': acov, 'active_coverage_low': alo, 'active_coverage_high': ahi}
                row['PASS'] = bool(row['NLL'] <= ref_nll and row['MAE'] <= row['NN_MAE'] and
                    row['catastrophic10percent'] <= row['NN_catastrophic10percent'] and na >= 20 and
                    lo <= .9 <= hi and alo <= .9 <= ahi)
                options.append((row, spec, pp, active))
                grid.append(row)
        passed = [x for x in options if x[0]['PASS']]
        row, spec, pp, active = min(passed or options, key=lambda x: (x[0]['NLL'], x[0]['slope'], -x[0]['ridge']))
        selected[dep] = {'pass': bool(passed), 'selection': row, 'spec': spec}
        np.savez_compressed(ROOT / f'predictions/{dep}_development.npz', p=pp, parent_p=parent,
            active=active, profile_active=pa, information_width=widths, profile_centers=centers, fold=fold, truth=truth)
        outputs.append(row)
        print('EXPECTED_INFORMATION_PREDICTIVE', dep, selected[dep], flush=True)
    n.write_csv(ROOT / 'tables/PREDICTIVE_GRID.csv', grid)
    n.write_csv(ROOT / 'tables/PREDICTIVE_SELECTION.csv', outputs)
    n.write_json(ROOT / 'calibration/PROFILE_EXPECTED_INFORMATION.json', selected)
    n.write_json(ROOT / 'contracts/PREDICTIVE_FROZEN.json', {'UTC': n.utc(),
        'both_runs_pass': all(x['pass'] for x in selected.values()),
        'sha256': n.sha(ROOT / 'calibration/PROFILE_EXPECTED_INFORMATION.json'), 'no_real_test_scored': True})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--stage', choices=('freeze', 'compute', 'calibrate'), required=True)
    parser.add_argument('--workers', type=int, default=20)
    args = parser.parse_args()
    ROOT = args.root
    if args.stage == 'compute':
        compute(args.workers)
    else:
        globals()[args.stage]()
