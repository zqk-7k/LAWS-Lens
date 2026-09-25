#!/usr/bin/env python3
"""Paired20Hz16s waveform-information pilot; historical input stays unchanged."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
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
sys.path.insert(0, str(P / 'scripts/experiments'))
import mcwf_continuous_profile_recovery_v2_20260908 as base

dev, old = base.dev, base.old
REFERENCE = P / 'results/mcwf_continuous_profile_pilot_20260908T143256Z'


class LowBand(base.Profile):
    def template(self, x):
        logmc, q, chi = x
        m1 = np.exp(logmc)*(1+q)**.2/q**.6
        hp, _ = get_fd_waveform(approximant='IMRPhenomD', mass1=m1, mass2=m1*q,
            spin1z=chi, spin2z=chi, delta_f=1/26, f_lower=20, f_final=2048,
            distance=1000, inclination=0)
        hp.resize(base.RAW_N//2+1)
        a = signal.hilbert(np.fft.irfft(np.asarray(hp), n=base.RAW_N))
        a = np.roll(a, int(24.75*4096)-int(np.argmax(abs(a))))
        phases = []
        for phase in (a.real, a.imag):
            z = base.STATE['v3'].preprocess_24s(np.stack([phase, phase]), self.frequency, self.psd, band_low_hz=20.)
            phases.append(z[..., -self.length:])
        bank = np.stack(phases, 1).astype(float)
        bank -= bank.mean(-1, keepdims=True)
        u, v = bank[:, 0], bank[:, 1]
        u /= np.linalg.norm(u, axis=-1, keepdims=True)
        v -= np.sum(u*v, -1, keepdims=True)*u
        v /= np.linalg.norm(v, axis=-1, keepdims=True)
        if not np.isfinite(bank).all():
            raise RuntimeError('Nonfinite low-frequency template')
        return bank


def initialize(root):
    if root.exists():
        raise RuntimeError('Independent directory required')
    for name in ('contracts', 'scripts', 'logs', 'events', 'tables', 'reports', 'manifest'):
        (root/name).mkdir(parents=True)
    for dep in base.parent.DEPS:
        path = REFERENCE / f'contracts/{dep}_PILOT_SOURCES.parquet'
        shutil.copy2(path, root / 'contracts'/path.name)
    shutil.copy2(REFERENCE / 'manifest/INPUT_SHA256.csv', root / 'manifest/INPUT_SHA256.csv')
    shutil.copy2(__file__, root / 'scripts/lowband_profile.py')
    dev.json_write(root / 'contracts/ANALYSIS_CONTRACT.json', {
        'id': 'MCWF-LOWFREQUENCY-PROFILE-05-PILOT', 'utc': datetime.now(timezone.utc).isoformat(),
        'status': base.parent.STATUS, 'goal_achieved': False,
        'purpose': 'Paired test of mass information excluded by40Hz cutoff;not a change to historical time/sky/rankings',
        'same_both_runs': True, 'source_selection': str(REFERENCE / 'contracts'),
        'sources_per_run': 20, 'images_per_source': 2, 'new_source_selection': False,
        'changed_physics': 'new auxiliary waveform profile uses20-580Hz16s,not8s of the old40Hz-filtered signal',
        'reconstruction': 'raw26s regenerated from unchanged stored physical source,noise bank,offset and targetSNR;original40Hz peak2s must still replay exactly',
        'do_not_invert_old_filter': True, 'no_second_SNR_rescale': True,
        'new_operator': 'identical4096Hz PSD whitening,Butter6zero-phasebandpass,antialiasresampling,robust24sMAD;only band_low_hz20 vs40',
        'template': 'same alignedIMRPhenomD continuouslogMc,q,equalchi;lowerf20 instead30;16s crop withmerger0.25s beforeend',
        'optimization': 'same four separated initial grid peaks,maxfev300 perstart,Nelder-Mead,identical bounds andstopping',
        'statistic': 'same two-detector maximizedquadrature projection,not fullPE,not calibratedoptimalSNR,evidence or posterior',
        'comparison': 'originalOMC prediction and PROFILE03 continuous2s/8s onexactly the same sources',
        'selection_data': 'simulated development only,no realPE/official catalog,no heldout ranking',
        'limitations': 'aligned-spin pointoptimizer can fail on precession/highermodes;noiseglitches mayworsen atlowfrequency;optimizerlocalmodes and convergence mustbe reported',
        'motivation': 'previous nominal-input-band audit found roughly10-15percent lowbandrho2 forlowMc sources;this doesnot itself prove parameter information or improved rankings',
        'references': ['https://pycbc.org/pycbc/latest/html/filter.html',
                       'https://arxiv.org/abs/gr-qc/9402014'],
        'freeze': ['all originalencoders', 'time', 'sky', 'outerweights', 'officialscope', 'historicalresults', 'paper'],
        'storage': 'eventfitJSON only;rawfull26s andtemporarytemplatesnotpersisted',
        'expansion': 'only after simulationpilot audit,not automaticreal-rank tuning'})


def work(item):
    row, feature = item
    ctx = base.STATE
    src, v3 = ctx['src'], ctx['v3']
    number = 1 if row['image'] == 'a' else 2
    clean, _ = src.detector_response(ctx['generator'], ctx['ifos'],
        src.source_parameters(pd.Series(row), row['gps_'+row['image']]),
        src.lens_factor(row[f'mu_image{number}'], row[f'morse_image{number}']))
    bi, offset = int(row['noise_bank_index']), int(row['noise_offset_samples'])
    noise = np.asarray(ctx['refs'][bi, :, offset:offset+v3.RAW_PADDED_SAMPLES], np.float64)
    _, mixed, audit = v3._preprocess_injection(clean, noise, ctx['frequency'], ctx['psds'][bi], row['target_network_snr'])
    encoded = dev.TRAIN.make_window_view(mixed[None].astype(np.float32), 2)[0].astype(np.float16)
    idx = int(row['row_index'])
    if not np.array_equal(encoded, ctx['oldraw'][idx]):
        raise RuntimeError('Archived40Hz2s replay failed')
    scaled, scale, recovered = v3.scale_to_network_snr(np.asarray(clean, np.float32), row['target_network_snr'], ctx['frequency'], ctx['psds'][bi])
    raw = np.asarray(noise, np.float32) + v3.embed_signal_in_padded_window(scaled)
    low = v3.preprocess_24s(raw, ctx['frequency'], ctx['psds'][bi], band_low_hz=20.)
    if scale != audit['physical_strain_scale_factor']:
        raise RuntimeError('Unexpected additional SNR scaling')
    profile = LowBand(low, ctx['frequency'], ctx['psds'][bi], 16)
    result = profile.fit(base.initial_peaks(feature))
    result.update(deployment=ctx['dep'], source_uid=row['source_uid'], row_index=idx,
                  image=row['image'], snr=row['target_network_snr'], mc_det=row['mc_det'],
                  mass_bin=row['mass_bin'], true_logmc=float(np.log(row['mc_det'])),
                  archived_peak2s_bit_exact=True, noise_bank_index=bi)
    result['absolute_logmc_error'] = abs(result['logmc']-result['true_logmc'])
    return result


def pilot(root, workers):
    for dep in base.parent.DEPS:
        selected = pd.read_parquet(root / f'contracts/{dep}_PILOT_SOURCES.parquet')
        x, meta = old.data(base.ROOT0, dep, 'validation')
        records = meta[meta.source_uid.isin(selected.source_uid)].to_dict('records')
        jobs = [(r, x[int(r['row_index'])]) for r in records if not (root / f'events/{dep}_{r["row_index"]}.json').exists()]
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn'),
                                 initializer=base.init_worker, initargs=(dep,)) as pool:
            for result in pool.map(work, jobs, chunksize=1):
                dev.json_write(root / f'events/{dep}_{result["row_index"]}.json', result)
                print(json.dumps({'lowband': dep, 'event': result['row_index'],
                    'logMc_error': result['absolute_logmc_error'], 'seconds': result['wall_seconds']}), flush=True)
    summarize(root)


def summarize(root):
    reference = pd.read_csv(REFERENCE / 'tables/PILOT_EVENT_RESULTS.csv')
    rows = []
    for path in sorted((root / 'events').glob('*.json')):
        r = json.loads(path.read_text())
        rows.append({k: v for k, v in r.items() if k != 'optimizer_runs'})
    table = pd.DataFrame(rows)
    table['method'] = 'continuous20Hz16s'
    combined = pd.concat([reference, table], ignore_index=True)
    dev.csv_write(root / 'tables/PAIRED_PILOT_EVENT_RESULTS.csv', combined)
    summary = combined.groupby(['deployment', 'method']).agg(events=('row_index', 'size'),
        sources=('source_uid', 'nunique'), MAE_logMc=('absolute_logmc_error', 'mean'),
        median_absolute_logMc=('absolute_logmc_error', 'median'),
        p90_absolute_logMc=('absolute_logmc_error', lambda x: x.quantile(.9))).reset_index()
    dev.csv_write(root / 'tables/PILOT_SUMMARY.csv', summary)
    dev.json_write(root / 'contracts/PILOT_COMPLETE.json', {'status': base.parent.STATUS,
        'goal_achieved': False, 'no_real_data': True, 'no_rank_change': True})
    print(summary.to_string(index=False), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True)
    p.add_argument('--stage', choices=['initialize', 'pilot', 'summarize'], required=True)
    p.add_argument('--workers', type=int, default=24)
    a = p.parse_args()
    if a.stage == 'pilot':
        pilot(a.root, a.workers)
    else:
        globals()[a.stage](a.root)
