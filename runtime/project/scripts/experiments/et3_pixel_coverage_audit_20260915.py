#!/usr/bin/env python3
"""HEALPix ordering, pixel representation, and descriptive source-block audit."""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import argparse
import importlib.util
import json
from pathlib import Path
import numpy as np
import pandas as pd


def main(root, q):
    import healpy as hp
    from astropy.io import fits
    from scipy.stats import spearmanr
    spec = importlib.util.spec_from_file_location('pixel_base', root/'scripts/et3_physical_trigger_repair_20260915_v2.py')
    b = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(b)
    rows = json.loads((root/'contracts/EVENTS.json').read_text())
    results = [json.loads((root/'maps'/f'event{i:02d}_q{q}'/'RESULT.json').read_text()) for i in range(len(rows))]
    matrices, header_rows, coverage = {}, [], []
    for nside in (256, 512, 1024):
        maps = []
        for idx, r in enumerate(results):
            mass = b.raster(r['map_path'], nside)
            if nside == 512:
                with fits.open(r['map_path']) as hdus:
                    h = hdus[1].header
                    header_rows.append({'event_uid': r['event_uid'], 'q': q,
                        'FITS_ORDERING': h.get('ORDERING'), 'COORDSYS': h.get('COORDSYS'),
                        'MOCORDER': h.get('MOCORDER'), 'raster_ordering': 'NESTED',
                        'analysis_nside': nside, 'normalization_error': abs(float(mass.sum())-1),
                        'sha256': b.sha(r['map_path']), 'UNIQ_aware_reader': True})
                true_pixel = hp.ang2pix(nside, np.pi/2-rows[idx]['dec'], rows[idx]['ra']%(2*np.pi), nest=True)
                level_min = float(mass[mass > mass[true_pixel]].sum())
                level_max = float(mass[mass >= mass[true_pixel]].sum())
                coverage.append({'event_uid': r['event_uid'], 'source_id': rows[idx]['source_id'],
                    'family': rows[idx]['family'], 'HPD_lower': level_min, 'HPD_upper': level_max,
                    'in_90_definite': level_max <= .9, 'in_90_possible': level_min <= .9})
            maps.append(mass)
        matrix = np.stack(maps)
        del maps
        scores = np.log(np.maximum(len(matrix[0])*(matrix@matrix.T), np.finfo(float).tiny))
        matrices[nside] = scores
        if nside == 512:
            p, r = matrix[:2]
            invariant = abs(float(np.dot(p, r))-float(np.dot(hp.reorder(p, n2r=True), hp.reorder(r, n2r=True))))
        del matrix
    pairs = []
    for i in range(len(rows)):
        for j in range(i+1, len(rows)):
            a, c, d = [float(matrices[n][i, j]) for n in (256, 512, 1024)]
            pairs.append({'i': i, 'j': j, 'event_i': rows[i]['event_uid'], 'event_j': rows[j]['event_uid'],
                'source_i': rows[i]['source_id'], 'source_j': rows[j]['source_id'],
                'companion': rows[i]['source_id'] == rows[j]['source_id'], 'q': q,
                'Z_256': a, 'Z_512': c, 'Z_1024': d, 'abs_delta_256_512': abs(c-a),
                'abs_delta_512_1024': abs(d-c), 'sign_flip_512_1024': (c > 0) != (d > 0)})
    frame = pd.DataFrame(pairs)
    frame.to_csv(root/'tables'/f'PIXEL_PAIR_AUDIT_Q{q}.csv', index=False)
    pd.DataFrame(header_rows).to_csv(root/'tables'/f'NATIVE_MOC_ORDERING_Q{q}.csv', index=False)
    pd.DataFrame(coverage).to_csv(root/'tables'/f'TRUTH_HPD_Q{q}.csv', index=False)
    summary = {'q': q, 'events': len(rows), 'pairs': len(frame),
        'max_abs_delta_512_1024': float(frame.abs_delta_512_1024.max()),
        'q99_abs_delta_512_1024': float(frame.abs_delta_512_1024.quantile(.99)),
        'max_abs_delta_256_512': float(frame.abs_delta_256_512.max()),
        'sign_flips_512_1024': int(frame.sign_flip_512_1024.sum()),
        'spearman_512_1024': float(spearmanr(frame.Z_512, frame.Z_1024).statistic),
        'NESTED_RING_permutation_dot_absolute_error': invariant,
        'not_an_increase_in_native_angular_information': True}
    summary['pass'] = summary['max_abs_delta_512_1024'] <= .2 and summary['q99_abs_delta_512_1024'] <= .05 and invariant <= 1e-14
    b.write(root/'contracts'/f'PIXEL_GATE_Q{q}.json', summary)
    cov = pd.DataFrame(coverage).groupby(['family', 'source_id'])[['in_90_definite', 'in_90_possible']].mean().reset_index()
    rng = np.random.default_rng(2026091501)
    draws = np.zeros((10000, 2))
    for _, group in cov.groupby('family'):
        values = group[['in_90_definite', 'in_90_possible']].to_numpy()
        draws += values[rng.integers(0, len(group), size=(10000, len(group)))].sum(axis=1)/len(cov)
    b.write(root/'contracts'/f'DESCRIPTIVE_COVERAGE_Q{q}.json', {
        'sources': len(cov), 'events': len(coverage),
        'source_weighted_90_definite': float(cov.in_90_definite.mean()),
        'source_weighted_90_possible': float(cov.in_90_possible.mean()),
        'source_stratified_bootstrap_CI_definite': np.quantile(draws[:, 0], [.025, .975]).tolist(),
        'source_stratified_bootstrap_CI_possible': np.quantile(draws[:, 1], [.025, .975]).tolist(),
        'bootstrap_repeats': 10000, 'bootstrap_seed': 2026091501,
        'descriptive_development_only': True,
        'not_a_pass_for_population_calibration': 'small, preselected source/SNR strata; not independent locked validation drawn from the inference prior'})
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--q', type=int, required=True)
    a = p.parse_args()
    main(a.root, a.q)
