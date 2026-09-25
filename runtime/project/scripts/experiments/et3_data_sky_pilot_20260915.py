#!/usr/bin/env python3
"""Localize actual-data bank triggers; no oracle intrinsics or production claims."""
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
import numpy as np
import pandas as pd


def load_base(path):
    spec = importlib.util.spec_from_file_location('data_sky_base', path)
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)
    base.__file__ = str(Path(__file__).resolve())
    return base


def prepare(base, bank, reference, root):
    import lal
    from ligo.skymap.bayestar import filter as sky_filter
    if json.loads((bank/'contracts/RESULT.json').read_text())['events'] != 24:
        raise RuntimeError('Bank trigger development incomplete')
    root.mkdir(parents=True, exist_ok=False)
    for part in ('contracts', 'scripts', 'triggers', 'maps', 'tables', 'logs', 'reports', 'build', 'manifest', 'package'):
        (root/part).mkdir()
    for p in (reference/'scripts').glob('*.py'):
        shutil.copy2(p, root/'scripts'/p.name)
    shutil.copy2(__file__, root/'scripts'/Path(__file__).name)
    contract = json.loads((reference/'contracts/ANALYSIS_CONTRACT.json').read_text())
    contract.update({'id': root.name, 'created_utc': base.utc(), 'reference': str(reference),
        'bank_source': str(bank), 'map_workers': 2, 'template_control': 'Data-selected PhenomD bank, not injected parameters',
        'conditional_intrinsics_oracle_control': False, 'intrinsics_fixed_to_data_recovered_template': True,
        'not_full_BBH_PE': True, 'current_stage': 'q32 actual-data development maps, not a converged formal release',
        'data_derived_intrinsics_gate': 'passed GPU/PyCBC search agreement; no posterior accuracy claim',
        'distance_prior_power': 2, 'cosmology': False,
        'distance_prior_scope': 'frozen BAYESTAR Euclidean default; ET population coverage not yet validated',
        'likelihood_rescale': .83, 'temperature': 1.,
        'rate_for_each_event': 'exact-bandlimit SNR sampling, same rule as separate acceleration audit',
        'no_production_expansion': True})
    base.write(root/'contracts/ANALYSIS_CONTRACT.json', contract)
    base.write(root/'contracts/CONTRACT_HASH.json', {'sha256': base.sha(root/'contracts/ANALYSIS_CONTRACT.json')})
    for name in ('EVENTS.json', 'GEOMETRY.json', 'PHYSICAL_GATE.json', 'RESERVED_SYSTEM_SPLITS.json'):
        shutil.copy2(reference/'contracts'/name, root/'contracts'/name)
    shutil.copy2(reference/'manifest/PROTECTED_BEFORE.json', root/'manifest/PROTECTED_BEFORE.json')
    checks = []
    for tr in sorted((bank/'triggers').glob('event*')):
        dest = root/'triggers'/tr.name
        shutil.copytree(tr, dest)
        r = json.loads((tr/'TRIGGER.json').read_text())
        if base.sha(tr/'TRIGGER.npz') != r['trigger_sha256'] or r['oracle_template']:
            raise RuntimeError('Input trigger identity/origin mismatch')
        with np.load(tr/'TRIGGER.npz') as a:
            f = np.arange(len(a['template_power']))*base.DF
            h = lal.CreateREAL8FrequencySeries('actual bank template power', 0, 0, base.DF, lal.StrainUnit**2, len(f)-1)
            h.data.data[:] = a['template_power'][:-1]
            for i in range(3):
                psd = sky_filter.InterpolatedPSD(f, a['psds'][i], f_high_truncate=1.)
                model = sky_filter.SignalModel(sky_filter.signal_psd_series(h, psd))
                error = abs(model.get_horizon_distance()/a['horizons'][i]-1)
                checks.append({'event_uid': r['event_uid'], 'channel': i, 'horizon_relative': float(error), 'pass': error <= 2e-3})
    pd.DataFrame(checks).to_csv(root/'tables/BANK_TRIGGER_HORIZON_AUDIT.csv', index=False)
    passed = all(c['pass'] for c in checks)
    base.write(root/'contracts/TRIGGER_GATE.json', {'pass': passed, 'oracle_parameters': False,
        'checks': len(checks), 'max_horizon_relative': max(c['horizon_relative'] for c in checks)})
    if not passed:
        raise RuntimeError('Actual bank trigger/horizon normalization mismatch')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--bank', type=Path)
    p.add_argument('--reference', type=Path)
    p.add_argument('--case', type=Path)
    p.add_argument('--prepare', action='store_true')
    p.add_argument('--maps', type=int, choices=[32, 64, 128])
    a = p.parse_args()
    base_path = a.root/'scripts/et3_physical_trigger_repair_20260915_v2.py'
    if not base_path.exists():
        base_path = a.reference/'scripts/et3_physical_trigger_repair_20260915_v2.py'
    base = load_base(base_path)
    if a.case:
        case = json.loads(a.case.read_text())
        t = json.loads((a.root/'triggers'/f'event{case["idx"]:02d}'/'TRIGGER.json').read_text())
        base.FS = t['sample_rate']
        base.worker(a.root, case)
        path = a.root/'maps'/case['id']/'RESULT.json'
        r = json.loads(path.read_text())
        r['conditional_intrinsics_oracle_control'] = False
        r['intrinsics_fixed_to_data_recovered_template'] = True
        r['data_derived_template'] = t['template']
        base.write(path, r)
        return
    if a.prepare:
        prepare(base, a.bank, a.reference, a.root)
    if a.maps:
        base.maps(a.root, a.maps)
        results = [json.loads(p.read_text()) for p in (a.root/'maps').glob(f'*q{a.maps}/RESULT.json')]
        base.write(a.root/'contracts'/f'DATA_SKY_Q{a.maps}_RESULT.json', {
            'state': 'DEVELOPMENT_MAPS_ONLY_NO_PRODUCTION_ADOPTION', 'maps_complete': len(results),
            'q': a.maps, 'oracle_template': False, 'full_BBH_PE': False, 'encoder_training_started': False,
            'bulk_started': False, 'heldout_test_opened': False})
        base.write(a.root/'STATUS.json', json.loads((a.root/'contracts'/f'DATA_SKY_Q{a.maps}_RESULT.json').read_text()))


if __name__ == '__main__':
    main()
