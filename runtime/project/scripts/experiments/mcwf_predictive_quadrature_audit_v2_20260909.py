#!/usr/bin/env python3
"""Mass-overlap quadrature audit; no score fitting, real pair or test access."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import json
from pathlib import Path
import shutil
import sys
import time
import numpy as np
import pandas as pd
from numpy.polynomial.legendre import leggauss

P = Path('/root/autodl-tmp/gw-catalog')
sys.path.insert(0, str(P/'scripts/experiments'))
import mcwf_profile_heteroscedastic_20260909 as h
n, d, cal = h.n, h.d, h.cal


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args(); root = args.root
    if root.exists():
        raise RuntimeError('Independent audit directory required')
    for folder in ('contracts', 'tables', 'data', 'scripts', 'manifest', 'reports', 'logs'):
        (root/folder).mkdir(parents=True)
    contract = {'UTC': n.utc(), 'id': 'MCWF-NODUP-PREDICTIVE-QUADRATURE-30',
        'goal_achieved': False, 'status': n.STATUS, 'same_algorithm_both_runs': True,
        'inputs': 'Only archived development R10 mass predictions; no real candidates or injection test.',
        'question': 'Does bin-integrated512mass BC overestimate overlap of narrow continuous profile predictions?',
        'density_definition': 'NN fallback stays piecewise uniform on original512 logMc bins. Active profile uses exact normalized R10 Student-t. No interpolation that claims new NN information.',
        'audit': 'Gauss-Legendre2,4,8,16nodes within every original bin;integrate sqrt(p_i p_j). All valid source/noise pairs evaluated;true versus null separately.',
        'threshold': {'maximum_norm_error': 1e-6, 'q99_BC_difference': 1e-4, 'max_BC_difference': 1e-3},
        'decision': 'Report smallest quadrature converged against next rule. Do not choose on recall/PE. No ranking or calibration changed by this script.',
        'statistic_not_lensing_BF': True, 'frozen_time_sky_weights': True}
    n.write_json(root/'contracts/ANALYSIS_CONTRACT.json', contract)
    shutil.copy2(__file__, root/'scripts/predictive_quadrature_audit.py')
    rows, sources = [], []
    for dep in n.DEPS:
        path = h.PARENT/f'predictions/{dep}_development_ENSEMBLE.npz'
        a = dict(np.load(path)); parent, active = a['p'], a['active'].astype(bool)
        center = a['profile_centers']
        spec = json.loads((h.PARENT/'calibration/PROFILE_PREDICTIVE.json').read_text())[dep]['spec']
        meta = pd.read_parquet(n.t.PREVIOUS/f'expanded_data/{dep}/validation/event_metadata.parquet')
        i, j = np.triu_indices(len(meta), 1)
        truth = meta.source_uid.to_numpy()[i] == meta.source_uid.to_numpy()[j]
        bc = {'bin512': np.sqrt(parent)@np.sqrt(parent).T}
        norms = {}
        widths = np.diff(d.EDGES)
        for count in (2, 4, 8, 16):
            start = time.perf_counter()
            x, w = leggauss(count)
            nodes = (d.EDGES[:-1, None]+widths[:, None]*(x[None, :]+1)/2).ravel()
            weights = (widths[:, None]*w[None, :]/2).ravel()
            mass = np.repeat(parent/widths, count, axis=1)
            ids = np.flatnonzero(active)
            for index in ids:
                mass[index] = np.exp(cal.normalized_logpdf(nodes, np.full(len(nodes), center[index]), spec))
            norm = mass@weights
            norms[str(count)] = float(abs(norm-1).max())
            # No renormalization conceals quadrature error. Report norms explicitly.
            feature = np.sqrt(mass*weights)
            bc[str(count)] = feature@feature.T
            print('PREDICTIVE_QUADRATURE', dep, count, norms[str(count)], time.perf_counter()-start, flush=True)
        for left, right in (('bin512', '16'), ('2', '4'), ('4', '8'), ('8', '16')):
            delta = abs(bc[left][i, j]-bc[right][i, j])
            signed = bc[left][i, j]-bc[right][i, j]
            for name, mask in (('all', np.ones(len(i), bool)), ('true_companion', truth),
                               ('non_companion', ~truth), ('profile_involved', active[i] | active[j])):
                values = delta[mask]
                rows.append({'deployment': dep, 'left': left, 'right': right, 'stratum': name,
                    'pairs': int(mask.sum()), 'median_abs': float(np.median(values)),
                    'q90_abs': float(np.quantile(values, .9)), 'q99_abs': float(np.quantile(values, .99)),
                    'max_abs': float(values.max()), 'mean_signed': float(signed[mask].mean()),
                    'max_norm_error_left': norms.get(left, 0.), 'max_norm_error_right': norms[right],
                    'converged': bool(np.quantile(values, .99)<=1e-4 and values.max()<=1e-3
                                      and max(norms.get(left, 0.), norms[right])<=1e-6)})
        np.savez_compressed(root/f'data/{dep}_mass_overlap_quadrature.npz', **bc, active=active,
                            idx_i=i, idx_j=j, true_companion=truth)
        sources.append({'path': str(path), 'sha256': n.sha(path), 'bytes': path.stat().st_size})
    n.write_csv(root/'tables/MASS_BC_QUADRATURE_AUDIT.csv', rows)
    n.write_csv(root/'manifest/INPUT_SHA256.csv', sources)
    n.write_json(root/'contracts/QUADRATURE_AUDIT_COMPLETE.json', {'UTC': n.utc(),
        'real_or_test_read': False, 'no_score_or_width_modified': True})


if __name__ == '__main__':
    main()
