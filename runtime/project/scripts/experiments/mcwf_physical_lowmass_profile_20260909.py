#!/usr/bin/env python3
"""Simulation-only enlargement of the continuous low-mass waveform audit."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import multiprocessing as mp
import shutil
import sys
import time
import numpy as np
import pandas as pd

P = Path('/root/autodl-tmp/gw-catalog')
sys.path.insert(0, str(P / 'scripts/experiments'))
import mcwf_lowband_profile_20260908 as profile
import mcwf_multirate_features_20260908 as replay
import mcwf_subgrid_density_20260909 as d
n, r = d.n, d.reliability
ROOT = None


def freeze():
    if ROOT.exists():
        raise RuntimeError('Independent output required')
    for folder in ('contracts', 'events', 'tables', 'audit', 'reports', 'scripts', 'logs', 'manifest'):
        (ROOT / folder).mkdir(parents=True)
    contract = {'id': 'MCWF-NODUP-LOWMASS-PROFILE-07', 'UTC': n.utc(), 'status': n.STATUS,
        'goal_achieved': False, 'stage': 'simulationdevelopmentonly', 'same_both_runs': True,
        'hypothesis': 'Existing20Hz16sprofilepilot improvedlowmass medianerrors;testallsimulateddevelopmentparents selectedbywaveformprediction,notrealPE.',
        'selection': '512existingdevelopmentparents;include BOTHimages ifeitherimage ensemblepredictedmedianMc<=15Msun. Threshold15 isexistingfirstphysicalmassbin boundary.',
        'predictor': 'FrozenMULTIRATE massdensityensemble;notpublicPE or oldencoderMc/q',
        'fit': 'InheritedLowBand continuouslogMc/q/equalchi IMRPhenomD20-580Hz16s,maxquadraturepower,H1L1relative<=21samples;4feature-onlyinitialpeaks,Nelder-Mead300eval/start.',
        'reconstruction': 'Regenerateexactrecordedsourceandnoise;old40Hzpeak2sfloat16mustreplaybitexact beforeloweringcutoff20Hz.No newSNRscale.',
        'output': 'Pointprofile,optimizationquality,anddiagnosticfiniteHessian;NOT normalizedGWlikelihood,evidence orfullPEposterior',
        'statistics': 'source/noisehashfold0and1;pointbias/errorsstratifiedbyrun,existingmassbins,SNR;sourcelevelbootstrap',
        'next_gate': 'Only ifbothruns lowmasspoint errorimproveswithout newcatastrophicrate;anyprobabilitycurve needsindependent empiricalerrorcalibration andcoveragebefore realranking.',
        'frozen': ['oldencoder', 'time_score', 'sky_raw_log_bf', 'allrankings', 'scope', 'paper'],
        'adaptive_development': True, 'test_or_real_read_for_this_stage': False,
        'refs': ['https://arxiv.org/abs/gr-qc/9402014', 'https://pycbc.org/pycbc/latest/html/filter.html'],
        'no_physicalPEclaim': True}
    n.write_json(ROOT / 'contracts/ANALYSIS_CONTRACT.json', contract)
    rows = []
    for dep in n.DEPS:
        x, meta = d.mult.training_data(d.LOW, dep, 'validation', 'MULTIRATE')
        ps = []
        for slot in n.t.MODEL_SLOTS:
            file = r.INTR / f'predictions/CONDITIONAL-ETA-CHI/{dep}/{slot}_0_development.npz'
            with np.load(file) as a:
                ps.append(a['p'])
            rows.append({'path': str(file), 'sha256': n.sha(file), 'bytes': file.stat().st_size})
        p = np.mean(ps, axis=0)
        sm = r.mass_summary(p)
        mass = np.exp(sm[:, 0])
        keep = set(meta.source_uid[mass <= 15.])
        meta['deployment'] = dep
        meta['fit_tune_fold'] = r.source_fold(meta)
        meta['parent_predicted_logmc'] = sm[:, 0]
        meta['parent_predicted_halfwidth'] = sm[:, 1]
        selected = meta[meta.source_uid.isin(keep)].copy()
        selected.to_parquet(ROOT / f'contracts/{dep}_SELECTED_EVENTS.parquet', index=False)
        np.save(ROOT / f'contracts/{dep}_INITIAL_FEATURES.npy', x[:, :27])
        n.write_json(ROOT / f'audit/{dep}_SELECTION.json', {'events': len(selected), 'sources': selected.source_uid.nunique(),
            'selection_uses_waveform_only': True, 'mass_threshold': 15., 'both_images_selected': True})
        print('PROFILE_PLANNED', dep, len(selected), selected.source_uid.nunique(), flush=True)
    n.write_csv(ROOT / 'manifest/INPUT_SHA256.csv', rows)
    shutil.copy2(__file__, ROOT / 'scripts/physical_lowmass_profile.py')
    n.write_json(ROOT / 'contracts/START_FREEZE.json', {'UTC': n.utc(),
        'contract_sha256': n.sha(ROOT / 'contracts/ANALYSIS_CONTRACT.json'), 'script_sha256': n.sha(Path(__file__))})


def init_worker(dep):
    replay.init_development(dep, 'validation')
    profile.base.STATE['v3'] = replay.CTX['v3']


def fit_one(job):
    row, features = job
    ctx = replay.CTX
    src, v3 = ctx['src'], ctx['v3']
    number = 1 if row['image'] == 'a' else 2
    clean, _ = src.detector_response(ctx['generator'], ctx['ifos'],
        src.source_parameters(pd.Series(row), row['gps_' + row['image']]),
        src.lens_factor(row[f'mu_image{number}'], row[f'morse_image{number}']))
    bank, offset = int(row['noise_bank_index']), int(row['noise_offset_samples'])
    noise = np.asarray(ctx['refs'][bank, :, offset:offset + v3.RAW_PADDED_SAMPLES], np.float32)
    scaled, scale, _ = v3.scale_to_network_snr(np.asarray(clean, np.float32), row['target_network_snr'], ctx['freq'], ctx['psds'][bank])
    raw = noise + v3.embed_signal_in_padded_window(scaled)
    original = v3.preprocess_24s(raw, ctx['freq'], ctx['psds'][bank])
    old = n.dev.TRAIN.make_window_view(original[None].astype(np.float32), 2)[0].astype(np.float16)
    if not np.array_equal(old, ctx['oldraw'][int(row['row_index'])]):
        raise RuntimeError('Originalpeak2s replayfailed')
    full = v3.preprocess_24s(raw, ctx['freq'], ctx['psds'][bank], band_low_hz=20.)
    model = profile.LowBand(full, ctx['freq'], ctx['psds'][bank], 16)
    result = model.fit(profile.base.initial_peaks(features))
    point = np.array([result['logmc'], result['q'], result['chieff_equal']])
    steps = np.array([.0008, .004, .004])
    lower, upper = np.array([np.log(5), .25, -.8]), np.array([np.log(200), 1., .8])
    interior = bool(((point - steps > lower) & (point + steps < upper)).all())
    result['hessian_interior'] = interior
    result['hessian_positive_definite'] = False
    if interior:
        h = np.empty((3, 3))
        e = np.diag(steps)
        center = model.evaluate(point)
        for k in range(3):
            h[k, k] = -(model.evaluate(point + e[k]) - 2 * center + model.evaluate(point - e[k])) / steps[k]**2
            for ell in range(k):
                h[k, ell] = h[ell, k] = -(model.evaluate(point + e[k] + e[ell]) - model.evaluate(point + e[k] - e[ell])
                    - model.evaluate(point - e[k] + e[ell]) + model.evaluate(point - e[k] - e[ell])) / (4 * steps[k] * steps[ell])
        eig = np.linalg.eigvalsh(h)
        result['diagnostic_hessian'] = h.tolist()
        result['diagnostic_hessian_eigenvalues'] = eig.tolist()
        result['hessian_positive_definite'] = bool((eig > 0).all())
        if result['hessian_positive_definite']:
            result['diagnostic_logmc_curvature_width_NOT_PE'] = float(np.sqrt(np.linalg.inv(h)[0, 0]))
    best_converged = any(a['success'] and abs(a['value'] - result['projection_statistic']) < 1e-3 for a in result['optimizer_runs'])
    result.update(deployment=row['deployment'], source_uid=row['source_uid'], row_index=int(row['row_index']),
        image=row['image'], snr=float(row['target_network_snr']), mc_det=float(row['mc_det']),
        fit_tune_fold=int(row['fit_tune_fold']), parent_predicted_logmc=float(row['parent_predicted_logmc']),
        source_noise_record_frozen=True, old_peak2s_bitexact=True, best_run_converged=bool(best_converged))
    result['profile_error'] = result['logmc'] - np.log(row['mc_det'])
    result['parent_error'] = row['parent_predicted_logmc'] - np.log(row['mc_det'])
    return result


def run(workers):
    runtime = ROOT / 'contracts/RUNTIME_IMPLEMENTATION.json'
    if not runtime.exists():
        n.write_json(runtime, {'UTC': n.utc(), 'script_sha256': n.sha(Path(__file__)),
            'change_after_plan_freeze': 'ImplementthealreadydeclaredHessiandiagnostic beforeanynewfits;selectionandoptimizerunchanged',
            'no_new_data_results_before_change': True, 'workers': workers})
        shutil.copy2(__file__, ROOT / 'scripts/physical_lowmass_profile_runtime.py')
    for dep in n.DEPS:
        meta = pd.read_parquet(ROOT / f'contracts/{dep}_SELECTED_EVENTS.parquet')
        features = np.load(ROOT / f'contracts/{dep}_INITIAL_FEATURES.npy')
        pending = [(row, features[int(row['row_index'])]) for row in meta.to_dict('records')
                   if not (ROOT / f'events/{dep}_{int(row["row_index"])}.json').exists()]
        start = time.perf_counter()
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn'), initializer=init_worker, initargs=(dep,)) as pool:
            for count, result in enumerate(pool.map(fit_one, pending, chunksize=1), 1):
                n.write_json(ROOT / f'events/{dep}_{result["row_index"]}.json', result)
                if count % 16 == 0 or count == len(pending):
                    print('PROFILE_PROGRESS', dep, count, len(pending), round(time.perf_counter() - start, 1), flush=True)
    summarize()


def summarize():
    import json
    rows = [json.loads(p.read_text()) for p in sorted((ROOT / 'events').glob('*.json'))]
    f = pd.DataFrame([{k: v for k, v in row.items() if not isinstance(v, list)} for row in rows])
    summaries = []
    for (dep, fold), a in f.groupby(['deployment', 'fit_tune_fold']):
        for method in ('parent', 'profile'):
            e = abs(a[method + '_error'])
            summaries.append({'deployment': dep, 'fold': fold, 'method': method, 'events': len(a),
                'sources': a.source_uid.nunique(), 'MAE_logMc': float(e.mean()), 'median_error': float(e.median()),
                'P90_error': float(e.quantile(.9)), 'error_above10percent': float((e > np.log(1.1)).mean()),
                'best_converged_fraction': float(a.best_run_converged.mean())})
    n.write_csv(ROOT / 'tables/PROFILE_EVENT_RESULTS.csv', f)
    n.write_csv(ROOT / 'tables/PROFILE_SUMMARY.csv', summaries)
    n.write_json(ROOT / 'contracts/PROFILE_DEVELOPMENT_COMPLETE.json', {'UTC': n.utc(), 'status': n.STATUS,
        'events': len(f), 'real_or_test_scored': False, 'goal_achieved': False})
    print(pd.DataFrame(summaries).to_string(index=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--stage', choices=('freeze', 'run', 'summarize'), required=True)
    parser.add_argument('--workers', type=int, default=32)
    args = parser.parse_args()
    ROOT = args.root
    if args.stage == 'run':
        run(args.workers)
    else:
        globals()[args.stage]()
