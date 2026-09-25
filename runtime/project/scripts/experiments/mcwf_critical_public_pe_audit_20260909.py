#!/usr/bin/env python3
"""Frozen-public-group audit, never an input to waveform model selection."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import json
from pathlib import Path
import shutil
import sys
import h5py
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, wasserstein_distance
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

P = Path('/root/autodl-tmp/gw-catalog')
sys.path.insert(0, str(P/'scripts/experiments'))
import mcwf_nodup_reliability_20260909 as r
n = r.n
MAIN = P/'results/main_o3official_cfixed_v1_20260904_20260904T072435Z'
EVENTS = ('GW191103_012549', 'GW191105_143521')
FIELDS = ('chirp_mass', 'mass_ratio', 'chi_eff', 'luminosity_distance')
PREFIXES = ('mc', 'q', 'chieff', 'dl')


def values(posterior, field, indices):
    available = set(posterior.dtype.names or ()) if isinstance(posterior, h5py.Dataset) else set(posterior.keys())
    def read(name):
        return np.asarray(posterior[name][indices], float)
    if field in available:
        return read(field)
    if field in ('chirp_mass', 'mass_ratio') and {'mass_1', 'mass_2'} <= available:
        a, b = read('mass_1'), read('mass_2')
        return (a*b)**.6/(a+b)**.2 if field == 'chirp_mass' else np.minimum(a, b)/np.maximum(a, b)
    raise RuntimeError('Missing public posterior parameter:'+field)


def stats(a, b, bins=256):
    mi, mj = np.median(a), np.median(b)
    si, sj = [(np.quantile(v, .84)-np.quantile(v, .16))/2 for v in (a, b)]
    lo = min(np.quantile(a, .001), np.quantile(b, .001))
    hi = max(np.quantile(a, .999), np.quantile(b, .999))
    edges = np.linspace(lo, hi, bins+1)
    x, y = [np.histogram(v, bins=edges)[0].astype(float) for v in (a, b)]
    x /= x.sum(); y /= y.sum()
    return {'median_i': float(mi), 'median_j': float(mj), 'sigma_i': float(si), 'sigma_j': float(sj),
        'standardized_distance': float(abs(mi-mj)/np.hypot(si, sj)),
        'bhattacharyya_coefficient': float(np.sqrt(x*y).sum()),
        'overlap_coefficient': float(np.minimum(x, y).sum()),
        'wasserstein_distance': float(wasserstein_distance(a, b))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args(); root = args.root
    if root.exists():
        raise RuntimeError('Independent output required')
    for folder in ('contracts', 'tables', 'data', 'figures', 'scripts', 'manifest', 'reports'):
        (root/folder).mkdir(parents=True)
    manifest_path = MAIN/'cache/source_run/data/event_manifest.csv'
    external_path = n.t.EXTERNAL/'gwtc3_external_reference.parquet'
    manifest = pd.read_csv(manifest_path).set_index('event_name')
    external = pd.read_parquet(external_path).set_index('pair_key').loc['--'.join(EVENTS)]
    hashes = [{'path': str(p), 'sha256': n.sha(p), 'bytes': p.stat().st_size}
              for p in (manifest_path, external_path)]
    samples, records = {}, []
    for event in EVENTS:
        item = manifest.loc[event]
        path = Path(item.sky_map_path)
        group = item.sky_map_internal_group
        with h5py.File(path, 'r') as file:
            posterior = file[group+'/posterior_samples']
            count = len(posterior) if isinstance(posterior, h5py.Dataset) else min(len(v) for v in posterior.values())
            indices = np.linspace(0, count-1, min(count, 30000), dtype=np.int64)
            for field in FIELDS:
                x = values(posterior, field, indices)
                samples[event+'_'+field] = x[np.isfinite(x)]
            m1, m2 = values(posterior, 'mass_1', indices), values(posterior, 'mass_2', indices)
            expected = (m1*m2)**.6/(m1+m2)**.2
            actual = values(posterior, 'chirp_mass', indices)
            relative = float(np.max(abs(expected-actual)/np.maximum(expected, 1e-12)))
        if relative > 1e-8:
            raise RuntimeError('Detector-frame mass conversion mismatch')
        hashes.append({'path': str(path), 'sha256': n.sha(path), 'bytes': path.stat().st_size})
        records.append({'event': event, 'file': str(path), 'group': group,
            'posterior_rows': count, 'audit_rows': len(indices), 'mass_conversion_relative_error': relative,
            'dynamic_group_selection': False})
    np.savez_compressed(root/'data/FROZEN_PUBLIC_POSTERIOR_SAMPLES.npz', **samples)
    metrics, checks, sensitivity = [], [], []
    for field, prefix in zip(FIELDS, PREFIXES):
        a, b = [samples[event+'_'+field] for event in EVENTS]
        row = stats(a, b)
        metrics.append({'parameter': field, **row})
        for key in ('median_i', 'median_j', 'sigma_i', 'sigma_j', 'standardized_distance', 'bhattacharyya_coefficient'):
            column = 'pe_'+prefix+'_'+key
            if column in external:
                delta = abs(row[key]-float(external[column]))
                checks.append({'parameter': field, 'metric': key, 'frozen': external[column],
                               'recomputed': row[key], 'absolute_difference': delta})
                if delta > 1e-10*max(abs(float(external[column])), 1):
                    raise RuntimeError('Frozen public audit mismatch:'+column)
        for bins in (100, 256, 512, 1024):
            sensitivity.append({'parameter': field, 'bins': bins, **stats(a, b, bins)})
    n.write_csv(root/'tables/PUBLIC_PE_RECOMPUTATION.csv', metrics)
    n.write_csv(root/'tables/FROZEN_METRIC_INVARIANCE.csv', checks)
    n.write_csv(root/'tables/HISTOGRAM_SENSITIVITY.csv', sensitivity)
    n.write_csv(root/'manifest/PUBLIC_PE_GROUPS.csv', records)
    n.write_csv(root/'manifest/INPUT_SHA256.csv', hashes)
    colors = ('#156b98', '#c34e42')
    labels = ('Detector-frame chirp mass [solar masses]', 'Mass ratio', 'Effective spin',
              'Apparent luminosity distance [Mpc]')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'pdf.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.4), layout='constrained')
    kde_tables = []
    for ax, field, label, metric in zip(axes.flat, FIELDS, labels, metrics):
        arrays = [samples[event+'_'+field] for event in EVENTS]
        lo = min(np.quantile(a, .001) for a in arrays)
        hi = max(np.quantile(a, .999) for a in arrays)
        bins = np.linspace(lo, hi, 100)
        for event, a, color in zip(EVENTS, arrays, colors):
            ax.hist(a, bins=bins, density=True, histtype='step', linewidth=1.4, color=color, label=event)
            ax.axvline(np.median(a), color=color, linestyle=':', linewidth=.9)
        ax.set(xlabel=label, ylabel='Posterior density')
        ax.set_title(f"BC = {metric['bhattacharyya_coefficient']:.3f}, D = {metric['standardized_distance']:.2f}")
        grid = np.linspace(min(a.min() for a in arrays), max(a.max() for a in arrays), 1024)
        densities = [gaussian_kde(a)(grid) for a in arrays]
        for density in densities:
            density /= np.trapezoid(density, grid)
        kde_tables.append({'parameter': field, 'BC_KDE_grid': float(np.trapezoid(np.sqrt(densities[0]*densities[1]), grid)),
            'overlap_KDE_grid': float(np.trapezoid(np.minimum(densities[0], densities[1]), grid)),
            'replaces_frozen_metric': False})
    axes[0, 0].legend(fontsize=8)
    fig.savefig(root/'figures/CRITICAL_PAIR_PUBLIC_PE.pdf')
    fig.savefig(root/'figures/CRITICAL_PAIR_PUBLIC_PE.png', dpi=180)
    plt.close(fig)
    n.write_csv(root/'tables/KDE_SENSITIVITY.csv', kde_tables)
    n.write_json(root/'contracts/PUBLIC_PE_AUDIT_COMPLETE.json', {'UTC': n.utc(),
        'read_only': True, 'event_pair': '--'.join(EVENTS), 'frozen_groups': True,
        'frozen_metric_matches': True, 'ranking_or_selection_input': False,
        'display': 'Actual public posterior samples;histogram100edgesforplot,frozenBC256binsforlabel. KDE only independent sensitivity.',
        'distance': 'Apparent luminosity distance is affected by magnification;not an intrinsic consistency veto.',
        'no_new_PE_or_Hanabi': True, 'status': n.STATUS})
    shutil.copy2(__file__, root/'scripts/critical_public_pe_audit.py')
    print(pd.DataFrame(metrics).to_string(index=False), flush=True)
    print(pd.DataFrame(kde_tables).to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
