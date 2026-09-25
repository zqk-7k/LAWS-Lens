#!/usr/bin/env python3
"""Simulation-only continuous intrinsic-template pilot, not Bayesian PE."""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import logging
import multiprocessing as mp
from pathlib import Path
import shutil
import sys
import time
import numpy as np
import pandas as pd
from scipy import signal
from scipy.ndimage import maximum_filter1d
from scipy.optimize import minimize
from pycbc.waveform import get_fd_waveform

PROJECT = Path('/root/autodl-tmp/gw-catalog')
sys.path.insert(0, str(PROJECT / 'scripts/experiments'))
import mcwf_temporal_response_20260908 as parent
import mcwf_expanded_data_20260907 as expanded

dev, old = parent.dev, parent.old
ROOT0 = parent.PREVIOUS
FS, RAW_FS, RAW_N = 2048, 4096, 26 * 4096
LAG = 304
COUNTS = 20
STATE = {}


def initialize(root):
    if root.exists():
        raise RuntimeError('Refuse an existing output directory')
    for name in ('contracts', 'scripts', 'logs', 'events', 'tables', 'reports', 'manifest'):
        (root / name).mkdir(parents=True)
    contract = {
        'id': 'MCWF-CONTINUOUS-PROFILE-03-PILOT',
        'utc': datetime.now(timezone.utc).isoformat(),
        'status': parent.STATUS, 'goal_achieved': False,
        'purpose': 'Test continuous mass/q/aligned-spin refinement and a longer waveform context on simulations before any real ranking',
        'difference_from_previous_8s': 'Earlier8s used576 fixed templates and an MLP; here optimize continuous intrinsic parameters with a direct signal projection and the actual4096Hz conditioning operator',
        'preserved': ['all historical files', 'encoder checkpoints', 'time', 'sky', 'C-fixed outer weights'],
        'scope': 'existing expanded-development simulations only,no BAYESTAR test or real event read',
        'selection': 'four source parents per existing mass_bin,lowest SHA256(source_uid|202609860),both images',
        'source_count_per_run': COUNTS, 'noise': 'replay exact recorded bank,offset,targetPSD-optimalSNR;last2s float16 must match archived input exactly',
        'variants': ['continuous2s', 'continuous8s'],
        'model': 'IMRPhenomD;Mc5-200,q0.25-1,equal aligned chi[-0.8,0.8];m1,m2>=3;not a precessing or higher-mode PE model',
        'operator': 'get_fd_waveform26s4096Hz30-2048Hz;align unwhitened envelope peak;call unchanged physical_common.preprocess_24s for each phase quadrature;orthonormalize within the requested crop',
        'statistic': 'max over arrival time and two detector-specific quadrature amplitudes of sum(projected power)/2,H1L1 relative lag<=21samples;not a normalized likelihood or evidence',
        'noise_units': 'retain archived full24s robust-MAD amplitude units,do not additionally divide by each2s/8s window SD',
        'starts': 'four separated peaks from existing fine253x9 physical network response grid;not learned predictions or injection truth',
        'optimizer': 'Nelder-Mead,maxfev300 per start,xatol1e-4,fatol1e-4;retain best including initialpoints;mass within30percent of each start',
        'finite_difference_hessian': 'diagnostic only,never relabel as full PE covariance',
        'truth_use': 'post-fit mass error only;noise-free algebra test uses an explicitly synthetic unit waveform',
        'expansion_rule': 'no real ranking;first inspect reconstruction,algebra,walltime and mass-error comparison;failure retained,not hidden',
        'sources': ['https://pycbc.org/pycbc/latest/html/pycbc.filter.html',
                    'https://pycbc.org/pycbc/latest/html/filter.html'],
        'not_relative_binning': 'Full Fourier transforms are evaluated;does not claim implementing relative-binning acceleration',
    }
    dev.json_write(root / 'contracts/ANALYSIS_CONTRACT.json', contract)
    rows = []
    for dep in parent.DEPS:
        base = ROOT0 / f'expanded_data/{dep}'
        for rel in ('validation/source_plan.parquet', 'validation/event_metadata.parquet',
                    'validation/raw2s.npy', 'noise/frequency.npy', 'noise/psd.npy'):
            path = base / rel
            rows.append({'path': str(path), 'sha256': dev.sha(path)})
        plans = pd.read_parquet(base / 'validation/source_plan.parquet')
        plans['hash_order'] = plans.source_uid.map(lambda s: hashlib.sha256(f'{s}|202609860'.encode()).hexdigest())
        keep = plans.sort_values('hash_order').groupby('mass_bin', sort=True).head(4).sort_values('source_index')
        if len(keep) != COUNTS:
            raise RuntimeError(f'Unexpected stratified source count:{dep}:{len(keep)}')
        keep.to_parquet(root / f'contracts/{dep}_PILOT_SOURCES.parquet', index=False)
    dev.csv_write(root / 'manifest/INPUT_SHA256.csv', pd.DataFrame(rows))
    shutil.copy2(__file__, root / 'scripts/continuous_profile.py')
    dev.json_write(root / 'contracts/CONTRACT_HASH.json', {'sha256': dev.sha(root / 'contracts/ANALYSIS_CONTRACT.json')})


def init_worker(dep):
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=1)
    src, data, v3 = expanded.modules()
    logging.getLogger('bilby').setLevel(logging.ERROR)
    noise = ROOT0 / f'expanded_data/{dep}/noise'
    STATE.update(dep=dep, src=src, data=data, v3=v3,
                 generator=src.build_waveform_generator(),
                 ifos=[src.bilby.gw.detector.get_empty_interferometer(d) for d in ('H1', 'L1')],
                 refs=np.load(noise / 'reference.npy', mmap_mode='r'),
                 frequency=np.load(noise / 'frequency.npy'), psds=np.load(noise / 'psd.npy', mmap_mode='r'),
                 oldraw=np.load(ROOT0 / f'expanded_data/{dep}/validation/raw2s.npy', mmap_mode='r'))


class Profile:
    def __init__(self, raw, frequency, psd, seconds):
        self.seconds = seconds
        self.length = seconds * FS
        self.nfft = 2 * self.length
        self.frequency, self.psd = frequency, psd
        self.raw = np.array(raw[..., -self.length:], dtype=float, copy=True)
        self.raw -= self.raw.mean(-1, keepdims=True)
        self.data = np.fft.rfft(self.raw, self.nfft)
        self.lags = np.arange(-LAG, LAG + 1)
        self.trace = []
        self.count = 0

    def template(self, x):
        logmc, q, chi = x
        mc = np.exp(logmc)
        m1 = mc * (1 + q)**.2 / q**.6
        hp, _ = get_fd_waveform(approximant='IMRPhenomD', mass1=m1, mass2=m1*q,
                               spin1z=chi, spin2z=chi, delta_f=1/26,
                               f_lower=30., f_final=2048., distance=1000., inclination=0.)
        hp.resize(RAW_N // 2 + 1)
        h = np.fft.irfft(np.asarray(hp), n=RAW_N)
        analytic = signal.hilbert(h)
        shift = int(24.75 * RAW_FS) - int(np.argmax(abs(analytic)))
        analytic = np.roll(analytic, shift)
        phases = []
        for a in (analytic.real, analytic.imag):
            z = STATE['v3'].preprocess_24s(np.stack([a, a]), self.frequency, self.psd)
            phases.append(z[..., -self.length:])
        bank = np.stack(phases, 1).astype(float)
        bank -= bank.mean(-1, keepdims=True)
        u, v = bank[:, 0], bank[:, 1]
        u /= np.linalg.norm(u, axis=-1, keepdims=True)
        v -= np.sum(u*v, -1, keepdims=True)*u
        v /= np.linalg.norm(v, axis=-1, keepdims=True)
        if not np.isfinite(bank).all():
            raise RuntimeError('Nonfinite quadrature template')
        return bank

    def power(self, bank, maximize=True):
        kernels = np.fft.rfft(bank, n=self.nfft, axis=-1)
        c = np.fft.irfft(self.data[:, None]*kernels.conj(), n=self.nfft, axis=-1)
        lags = self.lags if maximize else np.array([0])
        power = (c[..., lags % self.nfft]**2).sum(1)
        net = power[0] + maximum_filter1d(power[1], size=43 if maximize else 1, mode='constant', cval=-np.inf)
        k = int(np.argmax(net))
        return float(net[k] / 2), int(lags[k])

    def evaluate(self, x):
        if not (np.log(5) <= x[0] <= np.log(200) and .25 <= x[1] <= 1 and -.8 <= x[2] <= .8):
            return -1e10
        z, lag = self.power(self.template(x))
        self.count += 1
        self.trace.append([float(v) for v in (*x, z, lag)])
        return z

    def fit(self, starts):
        before = time.perf_counter()
        runs = []
        for x in starts:
            bounds = [(max(np.log(5), x[0]-np.log(1.3)), min(np.log(200), x[0]+np.log(1.3))), (.25, 1.), (-.8, .8)]
            simplex = np.tile(x, (4, 1))
            for k, step in enumerate((.004, .035, .035)):
                simplex[k+1, k] += step if x[k]+step <= bounds[k][1] else -step
            res = minimize(lambda theta: -self.evaluate(theta), x, method='Nelder-Mead', bounds=bounds,
                           options={'maxfev': 300, 'xatol': 1e-4, 'fatol': 1e-4, 'initial_simplex': simplex})
            runs.append({'parameters': res.x.tolist(), 'value': float(-res.fun), 'success': bool(res.success),
                         'nfev': int(res.nfev), 'message': str(res.message)})
        best = max(self.trace, key=lambda row: row[3])
        return {'logmc': best[0], 'q': best[1], 'chieff_equal': best[2], 'projection_statistic': best[3],
                'best_H1_lag_samples': best[4], 'function_calls': self.count, 'wall_seconds': time.perf_counter()-before,
                'optimizer_runs': runs, 'any_converged': any(v['success'] for v in runs)}


def initial_peaks(features):
    # Existing feature layout is detector,q,spin,ordered mass.
    network = np.asarray(features).reshape(3, 3, 3, 253)[2].transpose(2, 0, 1)
    order = np.argsort(network.ravel())[::-1]
    starts = []
    for idx in order:
        m, q, s = np.unravel_index(idx, network.shape)
        x = np.array([old.LOG_CENTERS[m], (.25, .5, 1.)[q], (-.5, 0., .5)[s]])
        if all(abs(x[0]-other[0]) > .035 or abs(x[2]-other[2]) > .25 for other in starts):
            starts.append(x)
        if len(starts) == 4:
            break
    return starts


def work(item):
    row, feature = item
    src, v3 = STATE['src'], STATE['v3']
    image = row['image']
    number = 1 if image == 'a' else 2
    clean, _ = src.detector_response(STATE['generator'], STATE['ifos'],
        src.source_parameters(pd.Series(row), row[f'gps_{image}']),
        src.lens_factor(row[f'mu_image{number}'], row[f'morse_image{number}']))
    bi, offset = int(row['noise_bank_index']), int(row['noise_offset_samples'])
    noise = np.asarray(STATE['refs'][bi, :, offset:offset+v3.RAW_PADDED_SAMPLES], np.float64)
    pure, mixed, audit = v3._preprocess_injection(clean, noise, STATE['frequency'], STATE['psds'][bi], row['target_network_snr'])
    idx = int(row['row_index'])
    reencoded = mixed[..., -4096:].astype(np.float16)
    difference = float(abs(reencoded.astype(float)-STATE['oldraw'][idx].astype(float)).max())
    if difference != 0:
        raise RuntimeError(f'Archived2s replay differs:{idx}:{difference}')
    starts = initial_peaks(feature)
    output = {'deployment': STATE['dep'], 'source_uid': row['source_uid'], 'row_index': idx,
              'image': image, 'noise_bank_index': bi, 'snr': row['target_network_snr'],
              'true_logmc': float(np.log(row['mc_det'])), 'mc_det': row['mc_det'],
              'true_q': row['m2_det']/row['m1_det'], 'mass_bin': row['mass_bin'],
              'archived2s_max_abs_difference': difference, 'starts': [x.tolist() for x in starts]}
    for seconds in (2, 8):
        p = Profile(mixed, STATE['frequency'], STATE['psds'][bi], seconds)
        result = p.fit(starts)
        result['absolute_logmc_error'] = abs(result['logmc']-output['true_logmc'])
        output[str(seconds)] = result
    return output


def unit(root):
    init_worker('gwtc3')
    x = np.array([np.log(10.), .7, .2])
    p = Profile(np.zeros((2, 49152)), STATE['frequency'], STATE['psds'][96], 2)
    bank = p.template(x)
    p.raw = 2*bank[:, 0] + bank[:, 1]
    p.data = np.fft.rfft(p.raw, p.nfft)
    power, lag = p.power(bank, maximize=False)
    gram = np.einsum('dpl,dql->dpq', bank, bank)
    error = float(abs(gram-np.eye(2)).max())
    if error > 1e-10 or abs(power-5) > 1e-10:
        raise RuntimeError('Quadrature projection algebra failed')
    started = time.perf_counter()
    for k in range(5):
        p.evaluate(x + [.0001*k, 0, 0])
    timing = (time.perf_counter()-started)/5
    dev.json_write(root / 'contracts/UNIT_TEST.json', {'pass': True, 'gram_error': error,
        'pure_quadrature_projection': power, 'expected': 5., 'seconds_per_evaluation': timing,
        'estimated_worst_pilot_cpu_seconds': 2*COUNTS*2*2*4*300*timing})
    print(json.dumps({'unit_pass': True, 'seconds_per_evaluation': timing}), flush=True)


def pilot(root, workers):
    for dep in parent.DEPS:
        selected = pd.read_parquet(root / f'contracts/{dep}_PILOT_SOURCES.parquet')
        x, meta = old.data(ROOT0, dep, 'validation')
        records = meta[meta.source_uid.isin(selected.source_uid)].to_dict('records')
        pending = [(r, x[int(r['row_index'])]) for r in records if not (root / f'events/{dep}_{r["row_index"]}.json').exists()]
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn'),
                                 initializer=init_worker, initargs=(dep,)) as pool:
            for output in pool.map(work, pending, chunksize=1):
                dev.json_write(root / f'events/{dep}_{output["row_index"]}.json', output)
                print(json.dumps({'pilot': dep, 'event': output['row_index'],
                    'error2s': output['2']['absolute_logmc_error'], 'error8s': output['8']['absolute_logmc_error'],
                    'seconds': output['2']['wall_seconds']+output['8']['wall_seconds']}), flush=True)
    summarize(root)


def summarize(root):
    rows = []
    for dep in parent.DEPS:
        predictions = []
        for slot in parent.MODEL_SLOTS:
            path = ROOT0 / f'ordered_mass_predictor/models/{dep}/seed_{slot}/validation_predictions.npz'
            a = np.load(path)
            predictions.append(a['p'] @ old.CENTERS)
        for file in sorted((root / 'events').glob(f'{dep}_*.json')):
            record = json.loads(file.read_text())
            idx = record['row_index']
            base = np.array([p[idx] for p in predictions])
            shared = {k: record[k] for k in ('deployment', 'source_uid', 'row_index', 'image', 'noise_bank_index', 'snr', 'mc_det', 'mass_bin')}
            rows.append({**shared, 'method': 'OMC-mean-predicted-logMc', 'logmc': float(base.mean()),
                         'absolute_logmc_error': float(abs(base.mean()-record['true_logmc'])), 'wall_seconds': 0.})
            for seconds in (2, 8):
                r = record[str(seconds)]
                rows.append({**shared, 'method': f'continuous{seconds}s',
                             **{k: r[k] for k in ('logmc', 'absolute_logmc_error', 'wall_seconds', 'function_calls', 'any_converged')}})
    table = pd.DataFrame(rows)
    dev.csv_write(root / 'tables/PILOT_EVENT_RESULTS.csv', table)
    summary = table.groupby(['deployment', 'method']).agg(events=('row_index', 'size'),
        sources=('source_uid', 'nunique'), mean_absolute_logmc_error=('absolute_logmc_error', 'mean'),
        median_absolute_logmc_error=('absolute_logmc_error', 'median'),
        p90_absolute_logmc_error=('absolute_logmc_error', lambda x: x.quantile(.9)),
        cpu_seconds=('wall_seconds', 'sum')).reset_index()
    dev.csv_write(root / 'tables/PILOT_SUMMARY.csv', summary)
    checks = pd.read_csv(root / 'manifest/INPUT_SHA256.csv')
    checks['after'] = [dev.sha(Path(p)) for p in checks.path]
    checks['unchanged'] = checks.sha256 == checks.after
    dev.csv_write(root / 'manifest/PROTECTED_RECHECK.csv', checks)
    if not checks.unchanged.all():
        raise RuntimeError('Protected inputs changed')
    dev.json_write(root / 'contracts/PILOT_COMPLETE.json', {'goal_achieved': False, 'status': parent.STATUS,
        'no_real_catalog_used': True, 'no_density_fitted': True, 'no_fusion_changed': True})
    print(summary.to_string(index=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--stage', choices=['initialize', 'unit', 'pilot', 'summarize'], required=True)
    parser.add_argument('--workers', type=int, default=12)
    args = parser.parse_args()
    if args.stage == 'pilot':
        pilot(args.root, args.workers)
    else:
        globals()[args.stage](args.root)
