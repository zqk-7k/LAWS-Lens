#!/usr/bin/env python3
"""Frequency-partition residuals for the existing peak-two-second filter bank.

This is an Allen-style projection diagnostic, not a calibrated chi-square
probability for real, tapered, band-limited detector noise.
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time
import numpy as np
import pandas as pd
import torch
import mcwf_pe_frontend_extension_v2_20260907 as e
import mcwf_adaptive_psd_encoder_20260906 as adaptive
import mcwf_finelag_eventpsd_20260907 as contexts

dev = e.dev
NFFT, BINS, LAGS = 8192, 8, 304


def geometry(bank):
    kernels = torch.fft.rfft(torch.as_tensor(bank, dtype=torch.float32, device='cuda'), n=NFFT)
    weight = torch.full((NFFT // 2 + 1,), 2. / NFFT, device='cuda')
    weight[0] = weight[-1] = 1. / NFFT
    gram = (kernels[..., :, None, :] * kernels[..., None, :, :].conj()).real * weight
    cumulative = torch.nn.functional.pad(gram.cumsum(-1), (1, 0))
    trace = cumulative[..., 0, 0, :] + cumulative[..., 1, 1, :]
    boundaries = torch.searchsorted(trace.contiguous(), (trace[..., -1, None] * torch.arange(1, BINS, device='cuda') / BINS).contiguous())
    edges = torch.cat([torch.zeros_like(boundaries[..., :1]), boundaries,
                       torch.full_like(boundaries[..., :1], len(weight))], -1)
    if not torch.all(edges[..., 1:] > edges[..., :-1]):
        raise RuntimeError('Empty frequency partition')
    selected = cumulative.gather(-1, edges[..., None, None, :].expand(*edges.shape[:-1], 2, 2, BINS + 1))
    blocks = (selected[..., 1:] - selected[..., :-1]).permute(0, 1, 4, 2, 3)
    eig = torch.linalg.eigvalsh(blocks)
    if eig.min() < 1e-7:
        raise RuntimeError('Ill-conditioned subband Gram matrix')
    inverse = torch.linalg.inv(blocks)
    identity_error = (blocks.sum(-3) - torch.eye(2, device='cuda')).abs().max().item()
    if identity_error > 2e-5:
        raise RuntimeError('Quadrature basis is not orthonormal')
    return kernels, weight, edges, blocks, inverse, identity_error


@torch.no_grad()
def extract(values, bank, batch=12, normalize=True, maximize_lag=True):
    kernels, weight, edges, blocks, inverse, identity_error = geometry(bank)
    lag = torch.arange(-LAGS, LAGS + 1, device='cuda') if maximize_lag else torch.zeros(1, dtype=torch.long, device='cuda')
    frequency = torch.arange(NFFT // 2 + 1, device='cuda')
    powers, residuals = [], []
    max_reconstruction_error = 0.
    for start in range(0, len(values), batch):
        x = torch.as_tensor(np.array(values[start:start+batch], dtype=np.float32), device='cuda')
        if x.shape[1:] != (2, 4096) or not torch.isfinite(x).all():
            raise RuntimeError('Need complete H1/L1 peak-two-second inputs')
        if normalize:
            x = (x - x.mean(-1, keepdim=True)) / x.std(-1, keepdim=True).clamp_min(1e-6)
        spectrum = torch.fft.rfft(x, n=NFFT)
        detector_powers, detector_residuals = [], []
        for d in range(2):
            pp, rr = [], []
            for k in range(0, len(bank), 64):
                sl = slice(k, k + 64)
                product = spectrum[:, d, None, None, :] * kernels[None, sl, d].conj()
                corr = torch.fft.irfft(product, n=NFFT)[..., lag.remainder(NFFT)]
                power = corr.square().sum(-2)
                best = power.argmax(-1)
                a = corr.gather(-1, best[..., None, None].expand(-1, -1, 2, 1)).squeeze(-1)
                phase = torch.exp(2j * np.pi / NFFT * lag[best][..., None] * frequency)
                cumulative = torch.nn.functional.pad(((product * phase[..., None, :]).real * weight).cumsum(-1), (1, 0))
                limits = edges[sl, d][None, :, None, :].expand(len(x), -1, 2, -1)
                summed = cumulative.gather(-1, limits)
                measured = (summed[..., 1:] - summed[..., :-1]).permute(0, 1, 3, 2)
                predicted = torch.einsum('kpab,nkb->nkpa', blocks[sl, d], a)
                residual = measured - predicted
                statistic = torch.einsum('nkpa,kpab,nkpb->nk', residual, inverse[sl, d], residual)
                error = (measured.sum(-2) - a).abs().max().item()
                max_reconstruction_error = max(max_reconstruction_error, error)
                if not torch.isfinite(statistic).all() or statistic.min() < -1e-5:
                    raise RuntimeError('Invalid frequency-partition residual')
                pp.append(power.gather(-1, best[..., None]).squeeze(-1).cpu().numpy())
                rr.append(statistic.clamp_min(0).cpu().numpy())
            detector_powers.append(np.concatenate(pp, 1))
            detector_residuals.append(np.concatenate(rr, 1))
        powers.append(np.stack(detector_powers, 1))
        residuals.append(np.stack(detector_residuals, 1))
    if max_reconstruction_error > .01:
        raise RuntimeError('Subband correlations do not reconstruct full correlation')
    return np.concatenate(powers), np.concatenate(residuals), {
        'gram_identity_error': identity_error,
        'subband_sum_vs_full_correlation_max_abs_error': max_reconstruction_error}


def tests(bank):
    rng = np.random.default_rng(202609331)
    small = bank[:8]
    pure = (2. * small[:, :, 0] + small[:, :, 1]).astype(np.float32)
    p, r, audit = extract(pure, small, normalize=False, maximize_lag=False)
    diagonal = np.arange(len(small))
    exact = r[diagonal, :, diagonal]
    power_error = abs(p[diagonal, :, diagonal] - 5.).max()
    if exact.max() > 1e-5 or power_error > 1e-4:
        raise RuntimeError('Exact template projection test failed')
    noise = rng.normal(size=(256, 2, 4096)).astype(np.float32)
    _, statistic, _ = extract(noise, small, normalize=False, maximize_lag=False)
    independent_means = statistic.mean((1, 2))
    mean = float(independent_means.mean())
    se = float(independent_means.std(ddof=1) / np.sqrt(len(noise)))
    expected = 2 * (BINS - 1)
    if abs(mean - expected) > 6 * se:
        raise RuntimeError('White-noise projection degrees-of-freedom test failed')
    return {'pass': True, 'exact_signal_max_residual': float(exact.max()),
        'exact_signal_power_error': float(power_error), 'white_noise_draws': len(noise),
        'fixed_lag_white_noise_mean': mean, 'white_noise_expected_mean': expected,
        'independent_draw_standard_error': se, **audit,
        'limit': 'This test validates algebra only. Production maximizes lag and uses normalized non-Gaussian real-noise inputs; no chi-square p-value interpretation.'}


def initialize(root):
    out = root / 'band_consistency'
    contract = out / 'contracts/FEATURES.json'
    if contract.exists():
        return
    dev.json_write(contract, {'created_utc': datetime.now(timezone.utc).isoformat(),
        'input': 'unchanged H1L1 peak2s4096,40-580Hz;unchanged576template event-PSD quadrature bank',
        'bins': BINS, 'partitions': 'eight contiguous bands with equal total quadrature template power, separately pertemplate/detector',
        'formula': 'a=Q*x at best full-power lag; G_b=Q F_b Q^T; r_b=Q F_b x-G_b a; residual=sum r_b^T G_b^-1 r_b',
        'features': 'original1728 powers plus1152 log1p(residual/14) features; no new frequency band, window, time or sky information',
        'interpretation': 'Allen-style waveform consistency feature; NOT a calibrated chi-square test, false-alarm probability or fullPE',
        'production_caveats': ['lag maximization', 'per-window normalization', 'nonGaussian noise', 'finite crop/taper'],
        'training_scope': 'same4096train/512validation sourceparents and96/32noiseblocks',
        'no_PE_official_ID_inputs': True,
        'references': ['https://arxiv.org/abs/gr-qc/0405045', 'https://pycbc.org/pycbc/latest/html/pycbc.vetoes.html']})
    (out / 'scripts').mkdir(exist_ok=True)
    shutil.copy2(__file__, out / 'scripts' / Path(__file__).name)


def grouped(root, raw, freq, psds, indices, out, original=None):
    initialize(root)
    if out.exists():
        result = np.load(out)
        if len(result) != len(raw):
            raise RuntimeError('Cached row count mismatch')
        return result
    h = np.load(e.TRAINED / 'cache/adaptive_psd/unwhitened_aligned_template_spectra.npy', mmap_mode='r')
    result = np.empty((len(raw), 2, 576), np.float32)
    rows = []
    for pid in np.unique(indices):
        take = np.flatnonzero(indices == pid)
        bank = adaptive.whitened_bank(h, freq, psds[int(pid)])
        test_path = root / 'band_consistency/contracts/NUMERICAL_TESTS.json'
        if not test_path.exists():
            dev.json_write(test_path, tests(bank))
        started = time.perf_counter()
        power, residual, audit = extract(raw[take], bank)
        discrepancy = float(np.max(abs(np.log1p(power) - original[take, :2].reshape(len(take), 2, 576)))) if original is not None else None
        if discrepancy is not None and discrepancy > .002:
            raise RuntimeError(f'Existing template power changed: {discrepancy}')
        result[take] = np.log1p(residual / (2 * BINS - 2))
        rows.append({'psd_index': int(pid), 'events': len(take), 'seconds': time.perf_counter()-started,
                     'existing_logpower_max_abs_difference': discrepancy, **audit})
    if not np.isfinite(result).all():
        raise RuntimeError('Nonfinite band features')
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, result)
    dev.csv_write(out.with_suffix('.audit.csv'), pd.DataFrame(rows))
    dev.json_write(out.with_suffix('.sha256.json'), {'sha256': dev.sha(out), 'events': len(raw)})
    print(json.dumps({'band_features': str(out), 'events': len(raw)}), flush=True)
    return result


def development(root, dep, split):
    folder = root / f'expanded_data/{dep}'
    meta = pd.read_parquet(folder / f'{split}/event_metadata.parquet')
    raw = np.load(folder / f'{split}/raw2s.npy', mmap_mode='r')
    original = np.load(root / f'expanded_encoder/features/{dep}/{split}.npy', mmap_mode='r')
    new = grouped(root, raw, np.load(folder / 'noise/frequency.npy'), np.load(folder / 'noise/psd.npy'),
                  meta.noise_bank_index.to_numpy(int), root / f'band_consistency/features/{dep}/{split}.npy', original)
    return np.concatenate([original.reshape(len(meta), -1), new.reshape(len(meta), -1)], 1), meta


def deployment(root, dep, es, split):
    if split == 'real':
        full, events = dev.real_inputs(dep)
        valid = events.strict_h1l1_preprocessing_pass.to_numpy(bool)
        raw = dev.TRAIN.make_window_view(np.asarray(full[valid], np.float32), 2)
    else:
        plan = dev.BASE.retained_event_plan(dep, es, split)
        full = dev.ORCH.event_array_for_plan(dep, es, split, plan)
        raw = dev.TRAIN.make_window_view(np.asarray(full, np.float32), 2)
    freq, psds, indices, _ = contexts.psd_context(dep, es, split)
    prefix = e.TRAINED / f'cache/deployment_event_psd/{dep}'
    original = np.load(prefix / ('real_features.npy' if split == 'real' else f'{es}_{split}_features.npy'))
    out = root / f'band_consistency/features/{dep}' / ('real.npy' if split == 'real' else f'{es}_{split}.npy')
    new = grouped(root, raw, freq, psds, indices, out, original)
    return np.concatenate([original.reshape(len(raw), -1), new.reshape(len(raw), -1)], 1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--deployment', choices=e.DEPS, required=True)
    args = parser.parse_args()
    for split in ('validation', 'train'):
        development(args.root, args.deployment, split)
