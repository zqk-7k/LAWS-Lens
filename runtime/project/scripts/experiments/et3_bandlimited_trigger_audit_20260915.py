#!/usr/bin/env python3
"""Separate exact-bandlimit SNR sampling audit; leaves reference runs intact."""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['OMP_STACKSIZE'] = '512M'
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


def load_base(path):
    spec = importlib.util.spec_from_file_location('et_base', path)
    base = importlib.util.module_from_spec(spec)
    sys.modules['et_base'] = base
    spec.loader.exec_module(base)
    # The subprocess entry must remain this adapter, which sets the event's
    # sample rate before invoking the unchanged localization implementation.
    base.__file__ = str(Path(__file__).resolve())
    return base


def prepare(base, source, root):
    root.mkdir(parents=True, exist_ok=False)
    for part in ('contracts', 'scripts', 'triggers', 'maps', 'tables', 'logs', 'reports', 'build', 'manifest', 'package'):
        (root/part).mkdir()
    for path in (source/'scripts').glob('*.py'):
        shutil.copy2(path, root/'scripts'/path.name)
    shutil.copy2(__file__, root/'scripts'/Path(__file__).name)
    for name in ('EVENTS.json', 'GEOMETRY.json', 'RESERVED_SYSTEM_SPLITS.json', 'PHYSICAL_GATE.json', 'TRIGGER_GATE.json'):
        shutil.copy2(source/'contracts'/name, root/'contracts'/name)
    contract = json.loads((source/'contracts/ANALYSIS_CONTRACT.json').read_text())
    contract.update({'id': root.name, 'created_utc': base.utc(), 'sampling_reference': str(source),
        'map_workers': 4, 'SNR_sampling_rule': 'fs=min(4096, next_power_of_two(8*f_max_exact_nonzero_template_power))',
        'sampling_change_only': True, 'raw_strain_and_waveform_windows_changed': False,
        'retained_samples': 'exact subset of the original complex SNR series; original peak and GPS epoch retained',
        'antialias_proof': 'actual matched-filter template power is identically zero above new Nyquist',
        'no_low_power_truncation': True, 'extra_sampling_gate': {'map_TV_max': .02, 'A90_relative_max': .05,
            'pair_delta_max_nats': .2, 'same_q_reference': 32},
        'reference_compute_boundary': 'unaccelerated q32 sufficient for interpolation audit; unaccelerated q64/q128 may be stopped for resource conservation with all partial outputs retained'})
    base.write(root/'contracts/ANALYSIS_CONTRACT.json', contract)
    base.write(root/'contracts/CONTRACT_HASH.json', {'sha256': base.sha(root/'contracts/ANALYSIS_CONTRACT.json')})
    protected = []
    for p in sorted((source/'strain').glob('*.npz')):
        protected.append({'path': str(p), 'sha256': base.sha(p), 'bytes': p.stat().st_size})
    protected.extend(json.loads((source/'manifest/PROTECTED_BEFORE.json').read_text()))
    protected.extend(json.loads((source/'manifest/HISTORICAL_MANIFEST_AS_RECEIVED.json').read_text()))
    base.write(root/'manifest/PROTECTED_BEFORE.json', protected)
    audits = []
    for tr in sorted((source/'triggers').glob('event*')):
        record = json.loads((tr/'TRIGGER.json').read_text())
        if base.sha(tr/'TRIGGER.npz') != record['trigger_sha256']:
            raise RuntimeError('Reference trigger changed')
        with np.load(tr/'TRIGGER.npz') as a:
            cache = {k: a[k] for k in a.files}
        power = cache['template_power']
        f = np.arange(len(power))*base.DF
        fmax = float(f[power > 0].max())
        rate = int(min(4096, 2**np.ceil(np.log2(8*fmax))))
        step = 4096//rate
        if np.any(power[f >= rate/2] != 0) or 512 % step != 0:
            raise RuntimeError('Decimation has nonzero aliased support or loses original peak')
        old_snr = cache['snr_series']
        cache['snr_series'] = old_snr[:, ::step]
        if not np.array_equal(cache['snr_series'][:, cache['snr_series'].shape[1]//2], old_snr[:, 512]):
            raise RuntimeError('Original complex peak changed')
        dest = root/'triggers'/tr.name
        dest.mkdir()
        np.savez_compressed(dest/'TRIGGER.npz', **cache)
        record.update({'sample_rate': rate, 'sample_stride': step, 'fmax_exact_Hz': fmax,
            'alias_power_above_new_Nyquist': 0., 'trigger_sha256': base.sha(dest/'TRIGGER.npz'),
            'original_trigger_sha256': base.sha(tr/'TRIGGER.npz'),
            'complex_peak_exact': True, 'no_array_interpolation': True})
        base.write(dest/'TRIGGER.json', record)
        audits.append({'event_uid': record['event_uid'], 'sample_rate': rate, 'stride': step,
            'template_exact_support_Hz': fmax, 'complex_peak_exact': True, 'aliased_power': 0.,
            'SNR_series_length': cache['snr_series'].shape[1]})
    pd.DataFrame(audits).to_csv(root/'tables/BANDLIMIT_AUDIT.csv', index=False)
    shutil.copy2(source/'tables/PHYSICAL_STRAIN.csv', root/'tables/PHYSICAL_STRAIN.csv')
    shutil.copy2(source/'tables/STRAIN_TRIGGER_AUDIT.csv', root/'tables/STRAIN_TRIGGER_AUDIT.csv')
    base.write(root/'STATUS.json', {'state': 'BANDLIMIT_CONTRACT_FROZEN', 'utc': base.utc()})


def sampling_audit(base, source, root):
    records, vectors = [], []
    for idx in range(24):
        a = source/'maps'/f'event{idx:02d}_q32/RESULT.json'
        b = root/'maps'/f'event{idx:02d}_q32/RESULT.json'
        if not a.exists() or not b.exists():
            continue
        aa, bb = json.loads(a.read_text()), json.loads(b.read_text())
        p, q = base.raster(aa['map_path'], 512), base.raster(bb['map_path'], 512)
        a90, b90 = [next(y['A90_deg2'] for y in x['raster_audit'] if y['nside'] == 512) for x in (aa, bb)]
        records.append({'index': idx, 'event_uid': aa['event_uid'], 'TV': float(.5*np.abs(p-q).sum()),
            'A90_relative': abs(b90-a90)/a90, 'reference_wall': aa['wall_seconds'],
            'bandlimited_wall': bb['wall_seconds'], 'timing_not_controlled_for_concurrent_load': True})
        vectors.append((idx, p, q))
    differences = []
    for i, (idx, p, q) in enumerate(vectors):
        for jdx, r, s in vectors[i+1:]:
            za = float(np.log(max(len(p)*np.dot(p, r), np.finfo(float).tiny)))
            zb = float(np.log(max(len(q)*np.dot(q, s), np.finfo(float).tiny)))
            differences.append({'i': idx, 'j': jdx, 'reference_Z': za, 'bandlimited_Z': zb, 'abs_delta': abs(zb-za)})
    frame, pairs = pd.DataFrame(records), pd.DataFrame(differences)
    frame.to_csv(root/'tables/SAMPLING_MAP_AUDIT.csv', index=False)
    pairs.to_csv(root/'tables/SAMPLING_PAIR_AUDIT.csv', index=False)
    gate = {'complete_events': len(frame), 'max_TV': None if frame.empty else float(frame.TV.max()),
        'max_A90_relative': None if frame.empty else float(frame.A90_relative.max()),
        'max_pair_delta': None if pairs.empty else float(pairs.abs_delta.max())}
    gate['pass'] = bool(len(frame) == 24 and gate['max_TV'] <= .02 and gate['max_A90_relative'] <= .05 and gate['max_pair_delta'] <= .2)
    base.write(root/'contracts/SAMPLING_GATE.json', gate)
    print(json.dumps({'stage': 'sampling_audit', **gate}), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--source', type=Path)
    p.add_argument('--case', type=Path)
    p.add_argument('--core-test', type=Path)
    p.add_argument('--all', action='store_true')
    p.add_argument('--audit', action='store_true')
    a = p.parse_args()
    base_path = a.root/'scripts/et3_physical_trigger_repair_20260915_v2.py'
    if not base_path.exists():
        if a.source is None:
            raise RuntimeError('Source required for initialization')
        base_path = a.source/'scripts/et3_physical_trigger_repair_20260915_v2.py'
    base = load_base(base_path)
    if a.core_test:
        if base.load_core(a.core_test).test() != 0:
            raise RuntimeError('Core tests failed')
        return
    if a.case:
        case = json.loads(a.case.read_text())
        tr = a.root/'triggers'/f'event{case["idx"]:02d}'/'TRIGGER.json'
        base.FS = json.loads(tr.read_text())['sample_rate']
        base.worker(a.root, case)
        return
    if a.all:
        prepare(base, a.source, a.root)
        base.maps(a.root, 32)
        base.maps(a.root, 64)
        convergence = base.audit(a.root, 32, 64)
        if not convergence['pass']:
            base.build128(a.root)
            base.maps(a.root, 128)
            base.audit(a.root, 64, 128)
        sampling_audit(base, a.source, a.root)
        base.finish(a.root)
    if a.audit:
        sampling_audit(base, a.source, a.root)
        base.finish(a.root)


if __name__ == '__main__':
    main()
