#!/usr/bin/env python3
"""Source-disjoint, paired frequency-band audit of a waveform mass profile."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import sys
import time
import numpy as np
import pandas as pd
from scipy import signal
from pycbc.waveform import get_fd_waveform

P = Path('/root/autodl-tmp/gw-catalog')
sys.path.insert(0, str(P/'scripts/experiments'))
import mcwf_independent_profile_features_20260909 as features
prof, generation, n = features.prof, features.generation, features.n
base = prof.profile.base
ROOT = DATA = None
CUTOFFS = (80., 160.)
PARENTS_PER_FOLD = 24
SEED = 2026090940


class BandProfile(prof.profile.LowBand):
    def __init__(self, raw, frequency, psd, high):
        self.high = float(high)
        super().__init__(raw, frequency, psd, 16)

    def template(self, x):
        logmc, q, chi = x
        m1 = np.exp(logmc)*(1+q)**.2/q**.6
        hp, _ = get_fd_waveform(approximant='IMRPhenomD', mass1=m1, mass2=m1*q,
            spin1z=chi, spin2z=chi, delta_f=1/26, f_lower=20., f_final=2048.,
            distance=1000., inclination=0.)
        hp.resize(base.RAW_N//2+1)
        a = signal.hilbert(np.fft.irfft(np.asarray(hp), n=base.RAW_N))
        a = np.roll(a, int(24.75*4096)-int(np.argmax(abs(a))))
        phases = []
        for phase in (a.real, a.imag):
            z = base.STATE['v3'].preprocess_24s(np.stack([phase, phase]), self.frequency,
                self.psd, band_low_hz=20., band_high_hz=self.high)
            phases.append(z[..., -self.length:])
        bank = np.stack(phases, 1).astype(float)
        bank -= bank.mean(-1, keepdims=True)
        u, v = bank[:, 0], bank[:, 1]
        u /= np.linalg.norm(u, axis=-1, keepdims=True)
        v -= np.sum(u*v, -1, keepdims=True)*u
        v /= np.linalg.norm(v, axis=-1, keepdims=True)
        if not np.isfinite(bank).all():
            raise RuntimeError('Nonfinite inspiral template')
        return bank


def freeze():
    if ROOT.exists():
        raise RuntimeError('Independent result directory required')
    for folder in ('contracts', 'events', 'tables', 'audit', 'reports', 'scripts', 'logs', 'manifest'):
        (ROOT/folder).mkdir(parents=True)
    records = []
    for dep in n.DEPS:
        path = DATA/f'contracts/{dep}_PROFILE_EVENT_PLAN.parquet'
        frame = pd.read_parquet(path)
        sources = frame[['source_uid', 'fit_tune_fold']].drop_duplicates()
        if sources.source_uid.duplicated().any():
            raise RuntimeError('A source appears in both folds')
        sources['hash_order'] = sources.source_uid.map(lambda uid: hashlib.sha256(
            f'{SEED}/{dep}/{uid}'.encode()).hexdigest())
        selected = sources.sort_values('hash_order').groupby('fit_tune_fold', sort=True).head(PARENTS_PER_FOLD)
        if selected.groupby('fit_tune_fold').size().to_dict() != {0: PARENTS_PER_FOLD, 1: PARENTS_PER_FOLD}:
            raise RuntimeError('Insufficient source support')
        plan = frame[frame.source_uid.isin(selected.source_uid)].copy()
        if not (plan.groupby('source_uid').size() == 2).all():
            raise RuntimeError('Both images must remain paired')
        plan.to_parquet(ROOT/f'contracts/{dep}_PILOT_PLAN.parquet', index=False)
        for file in (path, DATA/f'data/{dep}/event_metadata.parquet',
                     DATA/f'features/{dep}/peak.npy', DATA/f'data/{dep}/noise/noise_manifest.csv'):
            records.append({'path': str(file), 'sha256': n.sha(file), 'bytes': file.stat().st_size})
        for idx in plan.row_index:
            file = DATA/f'profile_events/{dep}/{int(idx)}.json'
            if not file.exists():
                raise RuntimeError('Missing frozen 580Hz comparison')
            records.append({'path': str(file), 'sha256': n.sha(file), 'bytes': file.stat().st_size})
    n.write_json(ROOT/'contracts/ANALYSIS_CONTRACT.json', {
        'UTC': n.utc(), 'id': 'MCWF-NODUP-INSPIRAL-PROFILE-PILOT-40', 'status': n.STATUS,
        'data': str(DATA), 'seed': SEED, 'parents_per_run_fold': PARENTS_PER_FOLD,
        'source_selection': 'SHA256 order within each explicit new fit/tune fold, among previously frozen predictedMc<=15 source selection; both images retained. No public PE, candidate IDs, or real ranks.',
        'bands_Hz': [[20, 80], [20, 160], [20, 580]], 'reference_high_Hz': 580,
        'changes': 'Only upper band edge, applied identically to reconstructed strain and templates. Model and optimizer unchanged.',
        'unchanged': ['IMRPhenomD equal aligned spin template', '16s profile window',
                      'four original waveform-feature starts', '300maxfev per start',
                      'maximized quadrature projection and detector delays', 'all encoder checkpoints',
                      'raw signal and noise and per-image SNR scale', 'time', 'sky', 'outer weights'],
        'hypothesis': 'Removing merger-dominated frequencies may reduce aligned-template model bias for low-mass inspiral; information loss may instead worsen recovery. This is tested, not presumed.',
        'not_full_PE': True, 'no_evidence_or_Fisher_precision_claim': True,
        'gate': 'On each run tune sources, source-mean absolute logMc error and source-mean90percentile must not exceed same-source580Hz reference; above10percent error fraction must not increase. At least one run MAE strictly improves. No relaxed tolerances.',
        'selection': 'Common passing cutoff minimizing worst-run MAE/reference ratio, then mean ratio, then lower high frequency. No winner if anyrunfails.',
        'uncertainty': '5000 source-paired bootstrap draws, conditional on shared fixed noise; report intervals, not independent-event significance.',
        'expansion': 'Only a passing pilot can be expanded to all new fit/tune events. Any probability density needs empirical error calibration before use in ranking.',
        'real_or_test': 'Not read, fitted or ranked in this pilot.',
        'references': ['https://arxiv.org/abs/gr-qc/9402014', 'https://arxiv.org/abs/gr-qc/0703086',
                       'https://pycbc.org/pycbc/latest/html/filter.html'],
        'reference_limit': 'Motivation for inspiral mass information and approximation checks only, not proof that80or160Hz is optimal.',
        'adaptive_development': True, 'no_blind_confirmation_claim': True})
    n.write_csv(ROOT/'manifest/INPUT_SHA256.csv', records)
    shutil.copy2(__file__, ROOT/'scripts/inspiral_profile_pilot.py')
    n.write_json(ROOT/'contracts/START_FREEZE.json', {'UTC': n.utc(),
        'contract_sha256': n.sha(ROOT/'contracts/ANALYSIS_CONTRACT.json'),
        'script_sha256': n.sha(Path(__file__))})


def init_worker(data, dep):
    features.init_worker(data, dep)


def raw_for(row):
    ctx = generation.STATE
    src, v3 = ctx['src'], ctx['v3']
    number = 1 if row['image'] == 'a' else 2
    clean, _ = src.detector_response(ctx['generator'], ctx['ifos'],
        src.source_parameters(pd.Series(row), row['gps_'+row['image']]),
        src.lens_factor(row[f'mu_image{number}'], row[f'morse_image{number}']))
    bi, offset = int(row['noise_bank_index']), int(row['noise_offset_samples'])
    noise = np.asarray(ctx['refs'][bi, :, offset:offset+v3.RAW_PADDED_SAMPLES], np.float32)
    scaled, scale, _ = v3.scale_to_network_snr(np.asarray(clean, np.float32),
        row['target_network_snr'], ctx['freq'], ctx['psds'][bi])
    raw = noise+v3.embed_signal_in_padded_window(scaled)
    old = n.dev.TRAIN.make_window_view(v3.preprocess_24s(raw, ctx['freq'],
        ctx['psds'][bi])[None].astype(np.float32), 2)[0].astype(np.float16)
    if not np.array_equal(old, ctx['oldraw'][int(row['row_index'])]):
        raise RuntimeError('Frozen2s reconstruction changed')
    return raw, ctx['freq'], ctx['psds'][bi], scale


def fit_event(job):
    row, feature, high = job
    start = time.perf_counter()
    raw, frequency, psd, scale = raw_for(row)
    v3 = generation.STATE['v3']
    prepared = v3.preprocess_24s(raw, frequency, psd, band_low_hz=20., band_high_hz=high)
    model = BandProfile(prepared, frequency, psd, high)
    result = model.fit(base.initial_peaks(feature))
    result.update(deployment=row['deployment'], row_index=int(row['row_index']),
        source_uid=row['source_uid'], fit_tune_fold=int(row['fit_tune_fold']),
        image=row['image'], high_Hz=high, snr=float(row['target_network_snr']),
        mc_det=float(row['mc_det']), profile_error=float(result['logmc']-np.log(row['mc_det'])),
        old_peak2s_bitexact=True, SNR_scale_unchanged=scale,
        total_wall_seconds=time.perf_counter()-start)
    return result


def unit():
    rows = []
    for dep in n.DEPS:
        features.init_worker(str(DATA), dep)
        row = pd.read_parquet(ROOT/f'contracts/{dep}_PILOT_PLAN.parquet').iloc[0].to_dict()
        raw, frequency, psd, _ = raw_for(row)
        full = generation.STATE['v3'].preprocess_24s(raw, frequency, psd, band_low_hz=20.)
        old, new = prof.profile.LowBand(full, frequency, psd, 16), BandProfile(full, frequency, psd, 580.)
        x = np.array([row['parent_predicted_logmc'], .5, 0.])
        a, b = old.template(x), new.template(x)
        if not np.array_equal(a, b) or old.evaluate(x) != new.evaluate(x):
            raise RuntimeError('580Hz operator replay failed')
        rows.append({'deployment': dep, 'row_index': row['row_index'],
                     'template_max_delta': float(abs(a-b).max()), 'power_delta': 0.})
    n.write_csv(ROOT/'audit/OPERATOR_REPLAY.csv', rows)
    n.write_json(ROOT/'contracts/OPERATOR_UNIT_PASS.json', {'UTC': n.utc(), 'passed': True})


def run(workers):
    if not (ROOT/'contracts/OPERATOR_UNIT_PASS.json').exists():
        raise RuntimeError('Operator tests required')
    for dep in n.DEPS:
        plan = pd.read_parquet(ROOT/f'contracts/{dep}_PILOT_PLAN.parquet')
        x = np.load(DATA/f'features/{dep}/peak.npy', mmap_mode='r')
        jobs = [(row, x[int(row['row_index'])], high) for row in plan.to_dict('records') for high in CUTOFFS
                if not (ROOT/f'events/{dep}_{int(row["row_index"])}_{int(high)}.json').exists()]
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn'),
                                 initializer=init_worker, initargs=(str(DATA), dep)) as pool:
            for count, result in enumerate(pool.map(fit_event, jobs), 1):
                n.write_json(ROOT/f'events/{dep}_{result["row_index"]}_{int(result["high_Hz"])}.json', result)
                if count % 16 == 0 or count == len(jobs):
                    print('INSPIRAL_PROFILE', dep, count, len(jobs), flush=True)
    summarize()


def summarize():
    records = [json.loads(p.read_text()) for p in sorted((ROOT/'events').glob('*.json'))]
    for dep in n.DEPS:
        plan = pd.read_parquet(ROOT/f'contracts/{dep}_PILOT_PLAN.parquet')
        for idx in plan.row_index:
            row = json.loads((DATA/f'profile_events/{dep}/{int(idx)}.json').read_text())
            row['high_Hz'] = 580.
            records.append(row)
    frame = pd.DataFrame([{k:v for k,v in row.items() if not isinstance(v, (dict,list))} for row in records])
    expected = len(n.DEPS)*PARENTS_PER_FOLD*2*2*3
    if len(frame) != expected:
        raise RuntimeError(f'Incomplete profile pilot:{len(frame)}/{expected}')
    frame['absolute_error'] = abs(frame.profile_error)
    frame['catastrophic'] = frame.absolute_error > np.log(1.1)
    source = frame.groupby(['deployment','fit_tune_fold','high_Hz','source_uid'], as_index=False).agg(
        absolute_error=('absolute_error','mean'), signed_error=('profile_error','mean'),
        catastrophic=('catastrophic','mean'))
    summary, comparisons = [], []
    rng = np.random.default_rng(SEED)
    for (dep, fold, high), a in source.groupby(['deployment','fit_tune_fold','high_Hz']):
        summary.append({'deployment':dep,'fold':fold,'high_Hz':high,'sources':len(a),
            'MAE':a.absolute_error.mean(),'median_absolute_error':a.absolute_error.median(),
            'q90_absolute_error':a.absolute_error.quantile(.9),'mean_bias':a.signed_error.mean(),
            'catastrophic_fraction':a.catastrophic.mean()})
        if high == 580.:
            continue
        ref = source[(source.deployment == dep)&(source.fit_tune_fold == fold)&(source.high_Hz == 580.)]
        joined = a.merge(ref,on='source_uid',suffixes=('','_ref'),validate='one_to_one')
        delta = joined.absolute_error.to_numpy()-joined.absolute_error_ref.to_numpy()
        draw = rng.integers(0,len(joined),(5000,len(joined)))
        low, upper = np.quantile(delta[draw].mean(1),[.025,.975])
        passing = bool(a.absolute_error.mean() <= ref.absolute_error.mean() and
            a.absolute_error.quantile(.9) <= ref.absolute_error.quantile(.9) and
            a.catastrophic.mean() <= ref.catastrophic.mean())
        comparisons.append({'deployment':dep,'fold':fold,'high_Hz':high,'point_guard':passing,
            'MAE_ratio':a.absolute_error.mean()/ref.absolute_error.mean(),
            'delta_MAE':delta.mean(),'delta_MAE_CI_low':low,'delta_MAE_CI_high':upper,
            'delta_catastrophic':a.catastrophic.mean()-ref.catastrophic.mean(),
            'independent_noise_significance_claim':False})
    comp = pd.DataFrame(comparisons)
    eligible = []
    for high, a in comp[comp.fold == 1].groupby('high_Hz'):
        if len(a) == 2 and a.point_guard.all() and (a.MAE_ratio < 1).any():
            eligible.append((a.MAE_ratio.max(),a.MAE_ratio.mean(),high))
    selected = min(eligible)[2] if eligible else None
    n.write_csv(ROOT/'tables/PROFILE_EVENT_RESULTS.csv',frame)
    n.write_csv(ROOT/'tables/PROFILE_SOURCE_RESULTS.csv',source)
    n.write_csv(ROOT/'tables/PROFILE_SUMMARY.csv',summary)
    n.write_csv(ROOT/'tables/PAIRED_COMPARISONS.csv',comp)
    n.write_json(ROOT/'contracts/PILOT_COMPLETE.json',{'UTC':n.utc(),'status':n.STATUS,
        'passed':bool(eligible),'selected_upper_Hz':selected,'real_or_test_read':False,
        'no_rank_change':True,'goal_achieved':False})
    print(pd.DataFrame(summary).to_string(index=False),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--stage',choices=('freeze','unit','run','summarize'),required=True)
    parser.add_argument('--workers',type=int,default=24)
    args=parser.parse_args();ROOT,DATA=args.root,args.data_root
    if args.stage == 'run':
        run(args.workers)
    else:
        globals()[args.stage]()
