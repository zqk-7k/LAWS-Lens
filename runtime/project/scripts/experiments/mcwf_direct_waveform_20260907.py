#!/usr/bin/env python3
"""Simulation-calibrated direct phase-invariant waveform comparison."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
import argparse
from datetime import datetime, timezone
from pathlib import Path
import json
import shutil
import numpy as np
import pandas as pd
import torch
from scipy.signal.windows import tukey
from sklearn.isotonic import IsotonicRegression
import mcwf_pe_frontend_extension_v2_20260907 as e

dev, body, ev = e.dev, e.body, e.ev
NFFT = 8192
FS = 2048
KINDS = ('network', 'best_detector', 'spectrum')
BETAS = (0., .0625, .125, .25, .5, 1., 2.)
torch.set_num_threads(2)


@torch.no_grad()
def spectral(raw):
    raw = np.asarray(raw, dtype=np.float32)
    if raw.shape[1:] != (2, 4096) or not np.isfinite(raw).all():
        raise RuntimeError('Require finite two-detector peak2s input')
    x = torch.as_tensor(raw, device='cuda')
    x = (x-x.mean(-1, keepdim=True))*torch.as_tensor(tukey(4096, .1), dtype=torch.float32, device='cuda')
    z = torch.fft.rfft(x, n=NFFT)
    freq = torch.fft.rfftfreq(NFFT, 1/FS, device='cuda')
    band = (freq >= 40) & (freq <= 580)
    z *= band
    norms = z.abs().square().sum(-1, keepdim=True).sqrt()
    if torch.any(norms < 1e-10):
        raise RuntimeError('No waveform energy in physical frequency band')
    z /= norms
    bins = torch.as_tensor(np.geomspace(40, 580.001, 65), device='cuda')
    power = torch.stack([z[..., (freq >= lo) & (freq < hi)].abs().square().mean(-1)
                         for lo, hi in zip(bins[:-1], bins[1:])], -1).mean(1)
    power = torch.nn.functional.normalize(power, dim=-1)
    return z, power


@torch.no_grad()
def compare(raw, i, j):
    z, power = spectral(raw)
    lag = torch.arange(-205, 206, device='cuda').remainder(NFFT)
    chunks = []
    for start in range(0, len(i), 96):
        a = torch.as_tensor(i[start:start+96], device='cuda')
        b = torch.as_tensor(j[start:start+96], device='cuda')
        cross = z[a, :, None, :]*z[b, None, :, :].conj()
        c = (torch.fft.ifft(cross, n=NFFT, dim=-1)*NFFT)[..., lag].abs().amax(-1)
        if not torch.isfinite(c).all() or c.max() > 1.00001:
            raise RuntimeError('Normalized direct match violates Cauchy-Schwarz')
        net = ((c[:, 0, 0].square()+c[:, 1, 1].square())/2).sqrt()
        chunks.append(torch.stack([net, c.amax((1, 2)), (power[a]*power[b]).sum(-1)], 1).cpu().numpy())
    return np.concatenate(chunks)


def unit_test():
    rng = np.random.default_rng(202609079)
    raw = rng.normal(size=(2, 2, 4096)).astype(np.float32)
    out = compare(raw, np.array([0, 1, 0, 1]), np.array([0, 1, 1, 0]))
    if not np.allclose(out[:2], 1., atol=2e-6) or not np.allclose(out[2], out[3], atol=2e-6):
        raise RuntimeError('Direct comparison identity/symmetry failure')
    from pycbc.types import TimeSeries
    from pycbc.filter import match
    z, _ = spectral(raw)
    x = raw[0, 0].astype(float)
    x = (x-x.mean())*tukey(4096, .1)
    y = raw[1, 0].astype(float)
    y = (y-y.mean())*tukey(4096, .1)
    x = np.pad(x, (0, 4096)); y = np.pad(y, (0, 4096))
    reference, _ = match(TimeSeries(x, delta_t=1/FS), TimeSeries(y, delta_t=1/FS),
                         low_frequency_cutoff=40., high_frequency_cutoff=580.25)
    actual = float((torch.fft.ifft(z[0, 0]*z[1, 0].conj(), n=NFFT)*NFFT).abs().max())
    if abs(actual-reference) > 1e-5:
        raise RuntimeError(f'PyCBC normalized-match discrepancy {actual} {reference}')
    return {'identity_and_symmetry_pass': True, 'pycbc_match_abs_error': abs(actual-reference)}


def cached(root, dep, es, split):
    path = root/f'direct_waveform/{dep}/{es}_{split}.npz'
    if split == 'development':
        path = root/f'direct_waveform/{dep}/development.npz'
        meta = pd.read_parquet(e.TRAINED/f'cache/{dep}/validation_metadata.parquet')
        i, j = np.triu_indices(len(meta), 1)
        f = pd.DataFrame({'idx_i': i, 'idx_j': j, 'is_true_pair':
                          meta.waveform_parent_uid.to_numpy()[i] == meta.waveform_parent_uid.to_numpy()[j]})
        source = e.TRAINED/f'cache/{dep}/validation_raw2s.npy'
        raw = np.load(source)
    else:
        f = pd.read_parquet(e.BASE/f'evaluation/{dep}/seed_{es}/{split}_pairs.parquet')
        if split == 'real':
            path = root/f'direct_waveform/{dep}/real.npz'
            full, events = dev.real_inputs(dep)
            valid = events.strict_h1l1_preprocessing_pass.to_numpy(bool)
            remap = np.full(len(events), -1, int); remap[valid] = np.arange(valid.sum())
            i, j = remap[f.idx_i.to_numpy(int)], remap[f.idx_j.to_numpy(int)]
            if (i < 0).any() or (j < 0).any():
                raise RuntimeError('Invalid detector scope')
            raw = np.asarray(full[valid, :, -4096:])
        else:
            plan = dev.BASE.retained_event_plan(dep, es, split)
            full = dev.ORCH.event_array_for_plan(dep, es, split, plan)
            raw = np.asarray(full[..., -4096:])
            i, j = f.idx_i.to_numpy(int), f.idx_j.to_numpy(int)
    if path.exists():
        a = np.load(path)
        if not np.array_equal(a['idx_i'], f.idx_i) or not np.array_equal(a['idx_j'], f.idx_j):
            raise RuntimeError('Pair order changed')
        return f, a['features']
    x = compare(raw, i, j)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, features=x, idx_i=f.idx_i, idx_j=f.idx_j)
    dev.json_write(path.with_suffix('.sha256.json'), {'sha256': dev.sha(path), 'pairs': len(f), 'events': len(raw)})
    print(json.dumps({'direct_features': dep, 'split': split, 'seed': es, 'pairs': len(f)}), flush=True)
    return f, x


def fit(f, x, kind):
    column = KINDS.index(kind); raw = x[:, column]
    y = f.is_true_pair.to_numpy(bool)
    weight = np.where(y, .5/y.sum(), .5/(~y).sum())
    iso = IsotonicRegression(increasing=True, out_of_bounds='clip').fit(raw, y.astype(float), sample_weight=weight)
    floor = 1/(y.sum()+2); prob = np.clip(iso.y_thresholds_, floor, 1-floor)
    return {'kind': kind, 'column': column, 'knots': iso.X_thresholds_.tolist(),
            'loglr': (np.log(prob)-np.log1p(-prob)).tolist(), 'fit_min': float(raw.min()),
            'fit_max': float(raw.max()), 'fit_positive_systems': int(y.sum()), 'floor': float(floor)}


def score(f, x, spec):
    raw = x[:, spec['column']]
    value = np.interp(raw, spec['knots'], spec['loglr'])
    ood = (raw < spec['fit_min']) | (raw > spec['fit_max'])
    value = np.clip(np.where(ood, np.minimum(value, 0), value), -4, 4)
    return f.waveform_score.to_numpy(float)+spec['beta']*value, value, ood


def run(root):
    trial = root/'trials/DIRECT-PHASE-MATCH'
    if trial.exists():
        raise RuntimeError('Independent output required')
    for name in ('contracts', 'tables', 'calibration', 'results', 'evaluation'):
        (trial/name).mkdir(parents=True)
    shutil.copy2(__file__, trial/'contracts'/Path(__file__).name)
    dev.json_write(trial/'contracts/RECIPE.json', {
        'created_utc': datetime.now(timezone.utc).isoformat(), 'same_O3_O4a': True,
        'input': 'existing independently PSD-whitened and filtered H1L1 peak2s,4096points,2048Hz',
        'window': 'Tukey0.1,physical40-580Hz,8192linear FFT',
        'comparison': 'positive-frequency normalized analytic cross-correlation, maximize phase and +-0.1001s time shift; same-detector RMS or best detector pair',
        'alternative': 'cosine of64 log-frequency binned normalized power shapes',
        'calibration': '240 encoder-development sources with equal positive/negative totalweight; isotonic; no real PE or official labels',
        'selection': 'BAYESTAR validation deterministic objective, current FRT guardrails; same kinds and beta grid for both runs',
        'kinds': KINDS, 'betas': BETAS, 'increment_cap': [-4, 4], 'positive_OOD': 'zero new reward',
        'limits': 'Noisy data-to-data matches are ranking features, not optimal coherent likelihoods or PE overlap; PSD differences and glitches may confound this test.',
        'frozen': ['time', 'sky', 'outerweights', 'scope', 'historical models/results'],
        'reference': 'https://pycbc.org/pycbc/latest/html/pycbc.filter.html', 'unit_test': unit_test()})
    chosen = {}; statuses = []
    for dep in e.DEPS:
        ffit, xfit = cached(root, dep, 0, 'development')
        calibrators = {kind: fit(ffit, xfit, kind) for kind in KINDS}
        for es in dev.SEEDS:
            f, x = cached(root, dep, es, 'validation'); bm = ev.metrics(f, f.waveform_score.to_numpy(float), dep, es)
            choices = []; rows = []
            for kind in KINDS:
                for beta in BETAS:
                    cfg = {**calibrators[kind], 'beta': beta}
                    z, _, ood = score(f, x, cfg); m = ev.metrics(f, z, dep, es); ok = ev.guard(m, bm)
                    rows.append({'kind': kind, 'beta': beta, 'pass': ok, 'ood': float(ood.mean()), **{a+'_'+k:v for a,b in m.items() for k,v in b.items()}})
                    if ok:
                        choices.append((e.old_selection.objective(m, beta, KINDS.index(kind)), cfg))
            out = trial/f'calibration/{dep}/seed_{es}'; out.mkdir(parents=True)
            dev.csv_write(out/'validation_grid.csv', pd.DataFrame(rows))
            cfg = min(choices, key=lambda c:c[0])[1]; chosen[dep, es] = cfg
            dev.json_write(out/'SELECTED_CONFIG.json', cfg)
            statuses.append({'deployment': dep, 'seed': es, 'kind': cfg['kind'], 'beta': cfg['beta']})
    dev.json_write(trial/'contracts/SELECTED_FREEZE.json', {'configs': statuses, 'real_used_for_selection': False})
    print(json.dumps({'direct_selected': statuses}), flush=True)
    rows = []; guards = []
    for dep in e.DEPS:
        real = {}
        for es in dev.SEEDS:
            for split in ('validation', 'test', 'real'):
                f, x = cached(root, dep, es, split); z, value, ood = score(f, x, chosen[dep, es])
                n = f.copy(); n['FRT_baseline_waveform_score'] = f.waveform_score; n['waveform_score'] = z
                n['direct_match_increment'] = value; n['direct_match_ood'] = ood
                for k, kind in enumerate(KINDS): n['direct_'+kind] = x[:, k]
                out = trial/f'evaluation/{dep}/seed_{es}'; out.mkdir(parents=True, exist_ok=True)
                n.to_parquet(out/f'{split}_pairs.parquet', index=False)
                if split == 'real': real[es] = n
                else:
                    bm = ev.metrics(f, f.waveform_score.to_numpy(float), dep, es); m = ev.metrics(f, z, dep, es)
                    guards.append({'deployment': dep, 'seed': es, 'split': split, 'pass': ev.guard(m, bm)})
                    for method in bm:
                        for config, metrics in [('FRT_BASELINE', bm[method]), ('CANDIDATE', m[method])]:
                            rows.append({'deployment': dep, 'seed': es, 'split': split, 'method': method, 'config': config, **metrics})
        dev.save_evaluation(trial, 'CANDIDATE', dep, real, pd.read_parquet(root/f'audit/{dep}_frozen_external_reference.parquet'))
    dev.csv_write(trial/'tables/RETRIEVAL_PER_SEED.csv', pd.DataFrame(rows))
    dev.csv_write(trial/'tables/REUSED_GUARDRAILS.csv', pd.DataFrame(guards)); e.assess(root, trial)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True); a = p.parse_args(); run(a.root)
