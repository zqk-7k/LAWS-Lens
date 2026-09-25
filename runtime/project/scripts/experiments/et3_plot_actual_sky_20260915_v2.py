#!/usr/bin/env python3
"""Fixed development examples, not candidates selected for favorable recovery."""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MPLBACKEND'] = 'Agg'
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np


def main(root, q):
    import healpy as hp
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    s = importlib.util.spec_from_file_location('plot_base', root/'scripts/et3_physical_trigger_repair_20260915_v2.py')
    base = importlib.util.module_from_spec(s)
    s.loader.exec_module(base)
    events = json.loads((root/'contracts/EVENTS.json').read_text())
    indices = [0, 6, 7, 11, 18, 22]
    arrays, records = [], []
    for idx in indices:
        r = json.loads((root/'maps'/f'event{idx:02d}_q{q}'/'RESULT.json').read_text())
        arrays.append(base.raster(r['map_path'], 512)/hp.nside2pixarea(512, degrees=True))
        records.append(r)
    vmax = max(float(a.max()) for a in arrays)
    fig, axes = plt.subplots(3, 2, figsize=(12, 10),
                             subplot_kw={'projection': 'mollweide'}, layout='constrained')
    longitude = np.linspace(-np.pi, np.pi, 721)
    latitude = np.linspace(-np.pi/2, np.pi/2, 361)
    xx, yy = np.meshgrid(longitude, latitude)
    pixels = hp.ang2pix(512, np.pi/2-yy, (-xx) % (2*np.pi), nest=True)
    norm = LogNorm(vmin=1e-7, vmax=vmax)
    for ax, idx, a, r in zip(axes.flat, indices, arrays, records):
        area = next(x['A90_deg2'] for x in r['raster_audit'] if x['nside'] == 512)
        title = f"{events[idx]['event_uid']} | A90={area:.0f} deg2"
        artist = ax.pcolormesh(xx, yy, np.maximum(a[pixels], 1e-7),
                               norm=norm, cmap='viridis', shading='auto', rasterized=True)
        ax.set_title(title, fontsize=10, pad=12)
        ax.set_xticks(np.radians([-120, -60, 0, 60, 120]))
        ax.set_xticklabels(['120', '60', '0', '300', '240'], fontsize=8)
        ax.set_yticks(np.radians([-60, -30, 0, 30, 60]))
        ax.tick_params(axis='y', labelsize=8)
        ax.grid(alpha=.25, linewidth=.5)
        ra = (-events[idx]['ra'] + np.pi) % (2*np.pi) - np.pi
        ax.scatter(ra, events[idx]['dec'], marker='*', s=90,
                   facecolors='white', edgecolors='black', linewidths=.7)
    fig.suptitle(f'ET development: data-recovered templates, quadrature {q}\n'
                 'Star: injected direction; RA increases leftward; not full BBH PE', fontsize=12)
    fig.colorbar(artist, ax=axes, orientation='horizontal', shrink=.8, pad=.04,
                 label='Probability density (deg^-2); common logarithmic color scale')
    folder = root/'figures'
    folder.mkdir(exist_ok=True)
    fig.savefig(folder/f'actual_strain_sky_examples_q{q}_layout2.png', dpi=150)
    fig.savefig(folder/f'actual_strain_sky_examples_q{q}_layout2.pdf')
    plt.close(fig)
    base.write(folder/f'actual_strain_sky_examples_q{q}_layout2.json', {
        'indices': indices, 'q': q, 'color_scale_per_deg2': [1e-7, vmax],
        'stars_are_truth_for_diagnostic_only': True,
        'examples_not_a_population_coverage_test': True})


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--q', type=int, required=True)
    args = p.parse_args()
    main(args.root, args.q)
