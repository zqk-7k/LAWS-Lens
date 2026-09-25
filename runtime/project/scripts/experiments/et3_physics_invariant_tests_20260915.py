#!/usr/bin/env python3
"""Read-only-input forward-model and catalog invariant tests."""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import argparse
import importlib.util
import json
from pathlib import Path
import numpy as np
import pandas as pd


def main(root, output):
    import lal
    import bilby
    spec = importlib.util.spec_from_file_location('physics_invariants', root/'scripts/et3_physical_trigger_repair_20260915_v2.py')
    b = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(b)
    _, src, geom, _ = b.deps(root)
    rows = json.loads((root/'contracts/EVENTS.json').read_text())
    checks = []
    for row in rows:
        ifos, detectors, _ = geom.geometry()
        gps = lal.LIGOTimeGPS(row['gps'])
        gmst = lal.GreenwichMeanSiderealTime(gps)
        for ifo in ifos:
            det = detectors[ifo.name]
            response = lal.ComputeDetAMResponse(det.response, row['ra'], row['dec'], row['psi'], gmst)
            bf = [ifo.antenna_response(row['ra'], row['dec'], row['gps'], row['psi'], mode) for mode in ('plus', 'cross')]
            antenna_error = float(np.max(np.abs(np.asarray(response)-bf)))
            ld = lal.TimeDelayFromEarthCenter(det.location, row['ra'], row['dec'], gps)
            bd = ifo.time_delay_from_geocenter(row['ra'], row['dec'], row['gps'])
            checks.append({'test': 'antenna_and_geocentric_delay', 'event': row['event_uid'],
                'detector': ifo.name, 'antenna_absolute_error': antenna_error, 'delay_absolute_seconds': abs(ld-bd),
                'pass': antenna_error <= 2e-7 and abs(ld-bd) <= 1e-8})
        inputs = [row[k] for k in ('theta_jn', 'phijl', 'tilt1', 'tilt2', 'phi12', 'a1', 'a2')]
        spin = bilby.gw.conversion.bilby_to_lalsimulation_spins(*inputs,
            row['m1_det']*lal.MSUN_SI, row['m2_det']*lal.MSUN_SI, 20., row['phase'])
        error = float(max(abs(np.linalg.norm(spin[1:4])-row['a1']), abs(np.linalg.norm(spin[4:7])-row['a2'])))
        checks.append({'test': 'spin_rotation_norm', 'event': row['event_uid'], 'absolute_error': error, 'pass': error <= 1e-12})
    for sid in sorted({x['source_id'] for x in rows}):
        group = [x for x in rows if x['source_id'] == sid]
        if len(group) != 2:
            continue
        group = sorted(group, key=lambda x: x['image'])
        info = src.choose_images(pd.Series(group[0]))
        expected = info['delay_days']*86400
        actual = group[1]['gps']-group[0]['gps']
        error = abs(actual-expected)
        checks.append({'test': 'catalog_delay_seconds', 'source_id': sid, 'expected': expected,
            'actual': actual, 'absolute_error': error, 'pass': error <= 1e-5})
    row = rows[0]
    gen = src.build_waveform_generator()
    params = src.source_parameters(pd.Series(row), row['gps'])
    reference, _ = src.detector_response(gen, geom.geometry()[0], params, 1.+0j)
    fd_ref = np.fft.rfft(reference.astype(float))/b.FS
    f = np.arange(fd_ref.shape[-1])*b.DF
    keep = (f >= 20)&(f < 1024)
    for mu in (1., -1.44, 4.):
        for morse in (0., .5, 1.):
            factor = src.lens_factor(mu, morse)
            td, _ = src.detector_response(gen, geom.geometry()[0], params, factor)
            fd = np.fft.rfft(td.astype(float))/b.FS
            expected = fd_ref[:, keep]*factor
            error = float(np.linalg.norm(fd[:, keep]-expected)/np.linalg.norm(expected))
            checks.append({'test': 'geometric_amplitude_Morse_operator', 'mu': mu, 'morse': morse,
                'relative_FD_error': error, 'pass': error <= 1e-6})
    output.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(checks).to_csv(output/'PHYSICS_INVARIANT_TESTS.csv', index=False)
    result = {'tests': len(checks), 'passed': sum(x['pass'] for x in checks),
        'failed': [x for x in checks if not x['pass']], 'waveform_generated_for_test_only': True,
        'no_historical_outputs_modified': True}
    b.write(output/'RESULT.json', result)
    print(json.dumps(result), flush=True)
    if result['failed']:
        raise RuntimeError('Forward physical invariant test failed')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    main(a.root, a.output)
