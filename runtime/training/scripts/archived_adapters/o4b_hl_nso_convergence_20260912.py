#!/usr/bin/env python3
"""Memory-bounded HEALPix convergence using exact NESTED interval integrals.

Native MOC cells are piecewise constant. Their fixed-resolution probability
masses can be represented by intervals of NESTED pixel indices without keeping
the full dense sky. Values are checked against ligo.skymap's rasterizer.
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
import argparse
import hashlib
import json
from pathlib import Path
import time

import healpy as hp
import numpy as np
import pandas as pd
from astropy_healpix import uniq_to_level_ipix
from ligo.skymap.io.fits import read_sky_map
from scipy.stats import spearmanr

import o4b_hl_nso_data_training_20260912 as s
import o4b_hl_nso_bayestar_20260912 as sky
import o4b_hl_nso_inference_20260912 as inference


def intervals(native, nside, temperature):
    target = int(round(np.log2(nside))); npix = hp.nside2npix(nside)
    order, pixel = uniq_to_level_ipix(np.asarray(native['UNIQ'], np.int64))
    order, pixel = np.asarray(order), np.asarray(pixel)
    density = np.asarray(native['PROBDENSITY'], float)
    if not np.isfinite(density).all() or (density < 0).any():
        raise RuntimeError('Invalid native MOC density; no silent filling')
    coarse = order <= target
    width = 4**(target-order[coarse])
    start = pixel[coarse]*width
    end = start+width
    value = density[coarse]*(4*np.pi/npix)
    if (~coarse).any():
        parent = pixel[~coarse]//(4**(order[~coarse]-target))
        unique, inverse = np.unique(parent, return_inverse=True)
        mass = density[~coarse]*(4*np.pi/(12.*4.**order[~coarse]))
        total = np.bincount(inverse, weights=mass)
        start, end, value = np.r_[start, unique], np.r_[end, unique+1], np.r_[value, total]
    permutation = np.argsort(start)
    start, end, value = start[permutation], end[permutation], value[permutation]
    if np.any(start[1:] < end[:-1]) or start[0] < 0 or end[-1] > npix:
        raise RuntimeError('Overlapping or invalid NESTED MOC intervals')
    # Missing cells have zero probability, as in the reference rasterizer.
    gaps_start = np.r_[0, end]; gaps_end = np.r_[start, npix]
    keep = gaps_end > gaps_start
    start, end, value = np.r_[start, gaps_start[keep]], np.r_[end, gaps_end[keep]], np.r_[value, np.zeros(keep.sum())]
    permutation = np.argsort(start)
    start, end, value = start[permutation], end[permutation], value[permutation]
    count = end-start
    value /= np.dot(value, count)
    if not np.isclose(temperature, 1.):
        lp = np.log(value.clip(1e-300))/temperature
        value = np.exp(lp-lp.max()); value /= np.dot(value, count)
    # Same float32 storage and float64 renormalization as the formal sky scorer.
    value = value.astype(np.float32).astype(np.float64)
    value /= np.dot(value, count)
    if abs(np.dot(value, count)-1) > 1e-12 or not np.array_equal(end[:-1], start[1:]):
        raise RuntimeError('Sky interval coverage/normalization failed')
    return start.astype(np.int64), end.astype(np.int64), value


def dot(a, b):
    sa, ea, va = a; sb, eb, vb = b
    boundaries = np.union1d(np.r_[sa, ea[-1]], np.r_[sb, eb[-1]])
    width = np.diff(boundaries)
    ia = np.searchsorted(sa, boundaries[:-1], side='right')-1
    ib = np.searchsorted(sb, boundaries[:-1], side='right')-1
    return float(np.sum(width*va[ia]*vb[ib], dtype=np.float64))


def register(root):
    path = root/'contracts/CONVERGENCE_AUDIT_PROTOCOL.json'
    if path.exists():
        if s.sha(path) != json.loads((root/'contracts/CONVERGENCE_AUDIT_FREEZE.json').read_text())['sha256']:
            raise RuntimeError('Convergence audit selection changed')
        return
    s.write(path, {'utc': s.now(), 'analysis_nside': 512, 'audit_nsides': [256, 512, 1024],
                   'selected_before_test': True,
                   'injection_selection': 'all true companions; deterministic hash1000null; top1%null by frozen512sky; any256/512signflip orabsdifference>.05',
                   'real_selection': 'all3655strict-scopepairs, not only attractivehead',
                   'hash_seed': 'O4B-NSO-CONVERGENCE-20260912',
                   'same_map_temperature_all_resolutions': True,
                   'reference': 'original nativeMOC -> exact fixed-resolution piecewise-constant NESTED intervals; validated against ligo.skymap rasterizer',
                   'no_dense_1024_persistence': True,
                   'reported': 'median,p90,p95,p99,maxlogBFdifference,signflipfraction,Spearman,true/null-tailstrata',
                   'decision': 'diagnostic, no post-test changes to weights orresolution; instability isreported, not hidden orcalledsuccess',
                   'formal_scope': 'This tests discretization of the same MOC, not correctness or coverage of BAYESTAR or public PE'})
    s.write(root/'contracts/CONVERGENCE_AUDIT_FREEZE.json', {'sha256': s.sha(path), 'code_sha256': s.sha(Path(__file__))})


def run_case(root, seed, split):
    out = root/f'results/sky_convergence/seed_{seed}/{split}'
    if (out/'COMPLETE.json').exists():
        return pd.read_csv(out/'summary.csv')
    out.mkdir(parents=True, exist_ok=True)
    folder = root/f'sky_pair_scores/seed_{seed}/{split}'
    original = pd.read_parquet(folder/'pairs.parquet')
    if split == 'real':
        events = pd.read_parquet(folder/'event_plan.parquet')
        paths = events.sky_map_path.tolist(); temp = 1.
        selected = np.ones(len(original), bool)
    else:
        events = pd.read_parquet(root/f'event_maps/seed_{seed}/{split}/event_metrics.parquet').sort_values('idx')
        paths = events.moc_path.tolist()
        temp = json.loads((root/f'event_maps/seed_{seed}/validation/TEMPERATURE_SELECTED.json').read_text())['temperature']
        null = original[~original.is_true_pair]
        top = null.nlargest(max(1, int(np.ceil(.01*len(null)))), 'sky_log_bf_nside512').index
        random = sorted(null.index, key=lambda i: hashlib.sha256(f'O4B-NSO-CONVERGENCE-20260912:{seed}:{split}:{original.loc[i,"event_i"]}:{original.loc[i,"event_j"]}'.encode()).hexdigest())[:1000]
        selected = (original.is_true_pair | original.index.isin(top) | original.index.isin(random) |
                    original.sign_flip_256_512 | (original.abs_delta_256_512 > .05)).to_numpy()
        original['random_null'] = original.index.isin(random)
        original['positive_null_tail'] = original.index.isin(top)
    pairs = original.loc[selected].copy()
    b = s.load(sky.REF, 'o4b_convergence_reference')
    rows, numerical = [], []
    used = sorted(set(pairs.idx_i) | set(pairs.idx_j))
    for nside in (256, 512, 1024):
        start = time.monotonic(); maps = {}
        for idx in used:
            native = read_sky_map(paths[idx], moc=True)
            maps[idx] = intervals(native, nside, temp)
            if idx in used[:3]:
                reference = b.apply_temperature(b.raster_probability(native, nside), temp).astype(np.float32).astype(float)
                reference /= reference.sum()
                sa, ea, va = maps[idx]
                error = max(float(np.max(abs(reference[a:z]-v))) for a, z, v in zip(sa, ea, va))
                if error > 1e-10:
                    raise RuntimeError(f'Interval/rasterizer mismatch:{seed}:{split}:{nside}:{idx}:{error}')
                numerical.append({'nside': nside, 'idx': idx, 'raster_mass_max_abs_error': error})
        values = np.array([np.log(max(hp.nside2npix(nside)*dot(maps[a], maps[b]), 1e-300))
                           for a, b in zip(pairs.idx_i, pairs.idx_j)])
        pairs[f'audit_log_bf_{nside}'] = values
        if nside in (256, 512):
            err = abs(values-pairs[f'sky_log_bf_nside{nside}'].to_numpy())
            if err.max() > 1e-5:
                raise RuntimeError(f'Audit/formal score mismatch:{nside}:{err.max()}')
        rows.append({'nside': nside, 'seconds': time.monotonic()-start, 'events': len(used), 'pairs': len(pairs)})
    pairs['abs_delta_512_1024'] = abs(pairs.audit_log_bf_512-pairs.audit_log_bf_1024)
    pairs['sign_flip_512_1024'] = np.sign(pairs.audit_log_bf_512) != np.sign(pairs.audit_log_bf_1024)
    populations = {'all_audited': np.ones(len(pairs), bool)}
    if split != 'real':
        populations.update(companion=pairs.is_true_pair, random_null=pairs.random_null, high_null_tail=pairs.positive_null_tail)
    summary = []
    for population, mask in populations.items():
        f = pairs.loc[mask]; x = f.abs_delta_512_1024.to_numpy()
        summary.append({'seed': seed, 'split': split, 'population': population, 'n_pairs': len(f),
                        **{f'delta_q{q:g}': float(np.quantile(x, q)) for q in (.5, .9, .95, .99, 1)},
                        'sign_flips': int(f.sign_flip_512_1024.sum()), 'sign_flip_fraction': float(f.sign_flip_512_1024.mean()),
                        'spearman': float(spearmanr(f.audit_log_bf_512, f.audit_log_bf_1024).statistic),
                        'independent_pair_binomial_CI_not_claimed': True})
    pairs.to_parquet(out/'audited_pairs.parquet', index=False)
    pd.DataFrame(summary).to_csv(out/'summary.csv', index=False, encoding='utf-8-sig')
    s.write(out/'COMPLETE.json', {'utc': s.now(), 'numerical_tests': numerical, 'timing': rows,
                                'pair_sha256': s.sha(out/'audited_pairs.parquet'), 'not_native_PE_resolution_improvement': True})
    return pd.DataFrame(summary)


def run(root):
    register(root); inference.guard(root, 'test')
    summaries = []
    for seed in (2026091221, 2026091222, 2026091223):
        for split in ('validation', 'test'):
            summaries.append(run_case(root, seed, split))
    summaries.append(run_case(root, 2026091221, 'real'))
    pd.concat(summaries).to_csv(root/'tables/sky_resolution_convergence.csv', index=False, encoding='utf-8-sig')
    s.write(root/'contracts/SKY_CONVERGENCE_AUDIT_COMPLETE.json', {'utc': s.now(), 'analysis_nside_unchanged': 512,
                                                               'no_post_test_reranking': True})


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True)
    p.add_argument('--register-only', action='store_true'); a = p.parse_args()
    register(a.root) if a.register_only else run(a.root)
