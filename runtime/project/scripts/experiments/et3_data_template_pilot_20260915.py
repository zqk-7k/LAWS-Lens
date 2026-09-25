#!/usr/bin/env python3
"""Development-only data-driven template selection; PyCBC verifies GPU screening."""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import time
import numpy as np
import pandas as pd


def main(source, root):
    spec = importlib.util.spec_from_file_location('base_bank', source/'scripts/et3_physical_trigger_repair_20260915_v2.py')
    b = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(b)
    import torch
    import lal
    from pycbc.waveform import get_fd_waveform
    from pycbc.types import FrequencySeries
    from pycbc.filter import matched_filter, sigma
    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError('GPU unavailable; no silent compute-path change')
    root.mkdir(parents=True, exist_ok=False)
    for part in ('contracts', 'scripts', 'triggers', 'tables', 'logs', 'reports', 'manifest', 'cache'):
        (root/part).mkdir()
    shutil.copy2(__file__, root/'scripts'/Path(__file__).name)
    rows = json.loads((source/'contracts/EVENTS.json').read_text())
    contract = {'created_utc': b.utc(), 'source': str(source), 'scope': 'development template-search pilot only; no test, no encoder training',
        'waveform': 'IMRPhenomD aligned-spin search templates for XPHM injected strain; approximation audited, not hidden',
        'intrinsic_truth_used_in_selection': False, 'sky_truth_used_in_selection': False,
        'bank': {'Mc_grid': [10., 512., 253, 'geomspace'], 'mass_ratio': [.25, .5, 1.], 'chi_eff': [-.8, -.4, 0., .4, .8]},
        'range_basis': 'already inspected training-only Mc range13.56..413.36; no validation/test opened',
        'duration_seconds': 24, 'sample_rate': 4096, 'f_lower': 20., 'f_upper': 1024.,
        'statistic': 'sum over ET detectors of max |complex matched SNR|^2 in trigger_time +/-0.125s',
        'GPU_screening': 'Torch complex64 FFT, algebraically same matched filter; selected output recomputed by PyCBC complex128',
        'numerical_screen_gate_relative': 2e-5, 'deterministic_tie': 'first bank index in lexicographic construction order',
        'SNR_sampling': 'retain exact bandlimited subset at next_power_of_two(8*fmax), capped at4096Hz',
        'not_BBH_PE_or_posterior': True, 'not_adopted_without_sky_and_coverage_gates': True,
        'bulk_started': False, 'historical_outputs_unchanged': True}
    b.write(root/'contracts/ANALYSIS_CONTRACT.json', contract)
    b.write(root/'contracts/CONTRACT_SHA256.json', {'sha256': b.sha(root/'contracts/ANALYSIS_CONTRACT.json')})
    b.write(root/'contracts/EVENTS.json', rows)
    _, _, geom, _ = b.deps(source)
    f = np.arange(b.N//2+1)*b.DF
    keep = (f >= b.LOW)&(f < b.HIGH)
    ifos, _, _ = geom.geometry()
    psds = np.asarray([ifo.power_spectral_density.get_power_spectral_density_array(f) for ifo in ifos])
    psds = np.where(np.isfinite(psds)&(psds > 0), psds, np.inf)
    bank, params, invalid = [], [], []
    began = time.perf_counter()
    for mc in np.geomspace(10., 512., 253):
        for q in (.25, .5, 1.):
            eta = q/(1+q)**2
            total = mc/eta**.6
            for chi in (-.8, -.4, 0., .4, .8):
                args = {'mass1': float(total/(1+q)), 'mass2': float(total*q/(1+q)),
                    'spin1z': chi, 'spin2z': chi, 'spin1x': 0., 'spin1y': 0., 'spin2x': 0., 'spin2y': 0.,
                    'f_ref': 20., 'f_final': b.HIGH}
                try:
                    hp, _ = get_fd_waveform(approximant='IMRPhenomD', delta_f=b.DF, f_lower=b.LOW,
                        distance=1., inclination=0., coa_phase=0., **args)
                    hp.resize(len(f))
                    h = np.asarray(hp, np.complex128).copy()
                    h[~keep] = 0
                    if not np.any(h) or not np.isfinite(h).all():
                        raise ValueError('invalid supported template')
                except (ValueError, RuntimeError) as exc:
                    invalid.append({'Mc': float(mc), 'q': q, 'chi': chi, 'error': str(exc)})
                    continue
                bank.append(h.astype(np.complex64))
                params.append({'Mc': float(mc), 'q': q, 'chi': chi, 'args': args})
    bank = np.stack(bank)
    np.save(root/'cache/TEMPLATE_BANK.npy', bank)
    b.write(root/'contracts/BANK_PARAMETERS.json', params)
    b.write(root/'contracts/BANK_BUILD.json', {'valid': len(bank), 'invalid': invalid, 'seconds': time.perf_counter()-began,
        'path': str(root/'cache/TEMPLATE_BANK.npy'), 'sha256': b.sha(root/'cache/TEMPLATE_BANK.npy'),
        'GPU': torch.cuda.get_device_name(), 'torch': torch.__version__})
    print(json.dumps({'stage': 'bank_built', 'templates': len(bank), 'seconds': time.perf_counter()-began}), flush=True)
    done = []
    for idx, row in enumerate(rows):
        b.guard(root)
        start = time.perf_counter()
        data_path = source/'strain'/f'{row["event_uid"]}.npz'
        with np.load(data_path) as a:
            data = np.asarray(a['noisy_padded'][:, b.FS:b.FS+b.N], np.float64)
        fd = np.fft.rfft(data)/b.FS
        lo, hi = int(23.75*b.FS)-512, int(23.75*b.FS)+513
        scores = np.zeros(len(bank), float)
        gpu_fd = torch.as_tensor(fd.astype(np.complex64), device='cuda')
        for first in range(0, len(bank), 32):
            h = bank[first:first+32].astype(np.complex128)
            norms = np.sqrt(4*b.DF*np.sum(np.abs(h[:, None, :])**2/psds[None, :, :], axis=-1))
            weights = (h[:, None, :].conj()/psds[None, :, :]).astype(np.complex64)
            corr = torch.zeros((len(h), 3, b.N), dtype=torch.complex64, device='cuda')
            corr[:, :, :len(f)] = torch.as_tensor(weights, device='cuda')*gpu_fd[None, :, :]
            snr = torch.fft.ifft(corr, dim=-1)*(4*b.DF*b.N)
            maxima = torch.amax(torch.abs(snr[:, :, lo:hi]), dim=-1).cpu().numpy()/norms
            scores[first:first+len(h)] = np.sum(maxima**2, axis=1)
            del corr, snr
        best = int(np.argmax(scores))
        np.save(root/'cache'/f'event{idx:02d}_bank_scores.npy', scores)
        h = bank[best].astype(np.complex128)
        template = FrequencySeries(h, delta_f=b.DF)
        check_indices = sorted({best, 0, len(bank)//2}) if idx == 0 else [best]
        discrepancies = []
        selected = []
        horizons = []
        for k in check_indices:
            exact_score = 0.
            for channel in range(3):
                ht = FrequencySeries(bank[k].astype(np.complex128), delta_f=b.DF)
                psd = FrequencySeries(psds[channel], delta_f=b.DF)
                snr = np.asarray(matched_filter(ht, FrequencySeries(fd[channel], delta_f=b.DF), psd=psd,
                    low_frequency_cutoff=b.LOW, high_frequency_cutoff=b.HIGH))
                peak = lo+int(np.argmax(abs(snr[lo:hi])))
                exact_score += float(abs(snr[peak])**2)
                if k == best:
                    selected.append((snr, peak))
                    horizons.append(float(sigma(ht, psd=psd, low_frequency_cutoff=b.LOW, high_frequency_cutoff=b.HIGH)))
            discrepancies.append(abs(exact_score-scores[k])/max(exact_score, 1.))
        error = max(discrepancies)
        if error > contract['numerical_screen_gate_relative']:
            b.write(root/'contracts/NUMERICAL_FAIL.json', {'index': idx, 'max_relative_error': error})
            raise RuntimeError('GPU screening does not reproduce PyCBC')
        fmax = float(f[abs(h) > 0].max())
        rate = int(min(b.FS, 2**np.ceil(np.log2(8*fmax))))
        step = b.FS//rate
        if np.any(h[f >= rate/2] != 0):
            raise RuntimeError('Nonzero aliased template support')
        snippets, epochs, channels = [], [], []
        for channel, (snr, peak) in enumerate(selected):
            snippets.append(snr[peak-512:peak+513:step].astype(np.complex64))
            epochs.append(int((lal.LIGOTimeGPS(row['gps'])-23.75+(peak-512)/b.FS).ns()))
            channels.append({'detector': ifos[channel].name, 'matched_snr': float(abs(snr[peak])),
                'phase': float(np.angle(snr[peak])), 'peak_sample': peak})
        dest = root/'triggers'/f'event{idx:02d}'
        dest.mkdir()
        np.savez_compressed(dest/'TRIGGER.npz', snr_series=snippets, epochs_ns=np.asarray(epochs, np.int64),
            psds=psds, horizons=horizons, template_power=abs(h)**2)
        b.write(dest/'TRIGGER.json', {'event_uid': row['event_uid'], 'source_uid': row['source_uid'],
            'template_args': params[best]['args'], 'template': 'IMRPhenomD', 'sample_rate': rate,
            'channels': channels, 'source_strain': str(data_path), 'source_strain_sha256': b.sha(data_path),
            'trigger_sha256': b.sha(dest/'TRIGGER.npz'), 'data_derived_intrinsic_template': True,
            'oracle_template': False, 'sky_truth_used_in_selection': False, 'bank_index': best})
        # Truth is read only after data-driven selection and is a diagnostic,
        # never an input to the bank or to candidate-template scoring.
        mc_true = (row['m1_det']*row['m2_det'])**.6/(row['m1_det']+row['m2_det'])**.2
        done.append({'event_uid': row['event_uid'], 'source_id': row['source_id'], 'bank_index': best,
            'selected_Mc': params[best]['Mc'], 'selected_q': params[best]['q'], 'selected_chi_eff': params[best]['chi'],
            'truth_Mc_diagnostic_only': mc_true, 'relative_Mc_error': params[best]['Mc']/mc_true-1,
            'network_matched_snr': float(np.sqrt(scores[best])), 'PyCBC_relative_error': error,
            'sample_rate': rate, 'seconds': time.perf_counter()-start})
        pd.DataFrame(done).to_csv(root/'tables/DATA_DERIVED_TEMPLATE_RECOVERY.csv', index=False)
        b.write(root/'STATUS.json', {'state': 'TEMPLATE_RECOVERY_RUNNING', 'events_complete': len(done), 'total': len(rows), 'utc': b.utc()})
        print(json.dumps({'stage': 'data_template', **done[-1]}), flush=True)
    b.write(root/'contracts/RESULT.json', {'state': 'DATA_DERIVED_TRIGGER_PILOT_COMPLETE_NOT_LOCALIZATION_VALIDATED',
        'events': len(done), 'max_GPU_PyCBC_relative_error': max(x['PyCBC_relative_error'] for x in done),
        'encoder_training_started': False, 'full_PE': False, 'maps_generated': 0,
        'GPU_peak_memory_bytes': torch.cuda.max_memory_allocated(), 'locked_test_opened': False})
    b.write(root/'STATUS.json', json.loads((root/'contracts/RESULT.json').read_text()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    a = parser.parse_args()
    main(a.source, a.root)
