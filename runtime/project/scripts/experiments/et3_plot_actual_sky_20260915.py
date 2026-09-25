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
    fig = plt.figure(figsize=(12, 10))
    for panel, (idx, a, r) in enumerate(zip(indices, arrays, records), 1):
        area = next(x['A90_deg2'] for x in r['raster_audit'] if x['nside'] == 512)
        title = f"{events[idx]['event_uid']} | A90={area:.0f} deg2"
        hp.mollview(a, nest=True, fig=fig.number, sub=(3, 2, panel), title=title,
                    min=1e-7, max=vmax, norm='log', cmap='viridis',
                    unit='Probability density (deg^-2)', margins=(.03, .06, .03, .04),
                    cbar=True, notext=False)
        hp.projscatter(np.pi/2-events[idx]['dec'], events[idx]['ra'], marker='*',
                       s=90, facecolors='white', edgecolors='black', linewidths=.7)
    fig.suptitle(f'ET development: data-recovered templates, quadrature {q}\n'
                 'Star: injected direction; common color scale; not full BBH PE', y=.995, fontsize=12)
    folder = root/'figures'
    folder.mkdir(exist_ok=True)
    fig.savefig(folder/f'actual_strain_sky_examples_q{q}.png', dpi=150)
    fig.savefig(folder/f'actual_strain_sky_examples_q{q}.pdf')
    plt.close(fig)
    base.write(folder/f'actual_strain_sky_examples_q{q}.json', {
        'indices': indices, 'q': q, 'color_scale_per_deg2': [1e-7, vmax],
        'stars_are_truth_for_diagnostic_only': True,
        'examples_not_a_population_coverage_test': True})


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--q', type=int, required=True)
    args = p.parse_args()
    main(args.root, args.q)
