"""Regenerate three frozen doublets, search the full bank, and compare sky maps."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import ast
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import runpy
import shutil
import sys
import time
import types

RUNS = ('O3', 'O4a', 'O4b')

def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def prepare(root, core, baseline):
    import pandas as pd
    target = root/'fixtures/injections'
    shutil.copytree(core/'fixtures/injections', target)
    (root/'scripts').mkdir(exist_ok=True)
    for name in ('verify_injections.py', 'portable_run.py'):
        shutil.copy2(core/'scripts'/name, root/'scripts'/name)
    freeze = root/'fixtures/recovery'
    freeze.mkdir()
    shutil.copy2(core/'runtime/training/bank/COMPLETE.json', freeze/'EXPECTED_BANK.json')
    shutil.copy2(core/'runtime/training/bank/templates.csv', freeze/'EXPECTED_BANK_PARAMS.csv')
    for run in RUNS:
        events = pd.read_parquet(target/run/'events.parquet')
        for uid in sorted(events.event_uid):
            dest = freeze/run/uid
            dest.mkdir(parents=True)
            old = baseline/f'timings/{run}/C_PHYSICAL/validation/{uid}'
            for name in ('RESULT.json', 'TRIGGERS.json', 'TRIGGERS.npz'):
                shutil.copy2(old/name, dest/name)
            result = json.loads((old/'RESULT.json').read_text())
            shutil.copy2(result['moc_path'], dest/'EXPECTED_MAP.fits.gz')

def init_worker(root, release):
    historical = Path('/root/autodl-tmp/gw-catalog')
    release = Path(release).resolve()
    environments = tuple({Path(sys.prefix).resolve(), Path(sys.base_prefix).resolve()})
    def guard(event, args):
        if event == 'open' and args and isinstance(args[0], (str, bytes)):
            path = Path(os.fsdecode(args[0]))
            if path.is_absolute() and path.is_relative_to(historical) and not path.is_relative_to(release):
                if not any(path.is_relative_to(env) for env in environments):
                    raise PermissionError('Recovery worker attempted historical-project read')
    sys.addaudithook(guard)
    sys.path.insert(0, str(Path(root)/'scripts'))
    import c_trigger
    original_loader = c_trigger.module
    def diagnostics_only(path, name):
        if Path(path).name != 'bayestar_si_fixed.py':
            return original_loader(path, name)
        # These two frozen pure functions do not need the legacy script's
        # import-time source catalog and experiment configuration side effects.
        tree = ast.parse(Path(path).read_text())
        selected = []
        functions = {'raster_probability', 'posterior_metrics'}
        constants = {'ANALYSIS_NSIDE', 'TEMPERATURE_GRID'}
        found = set()
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name in functions:
                selected.append(node)
                found.add(node.name)
            elif isinstance(node, ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0], ast.Name):
                if node.targets[0].id in constants:
                    selected.append(node)
                    found.add(node.targets[0].id)
        if found != functions | constants:
            raise RuntimeError('Frozen diagnostic definitions incomplete')
        imports=ast.parse('from __future__ import annotations\nimport math\nfrom typing import Any\nimport numpy as np\nimport healpy as hp\n').body
        obj=types.ModuleType(name)
        exec(compile(ast.Module(body=imports+selected,type_ignores=[]),str(path),'exec'),obj.__dict__)
        return obj
    c_trigger.module = diagnostics_only
    c_trigger.worker_init(root)

def recover(task):
    import c_trigger
    return c_trigger.run_event(task)

def execute(root, tag):
    import numpy as np
    import pandas as pd
    from astropy.table import Table
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)
    out = root/'verification'/tag
    out.mkdir(parents=True, exist_ok=False)
    saved_argv = sys.argv
    sys.argv = ['verify_injections.py', '--release', str(root), '--output-tag', tag+'_strain']
    try:
        runpy.run_path(str(root/'scripts/verify_injections.py'), run_name='__main__')
    finally:
        sys.argv = saved_argv
    strain_root = root/'verification'/(tag+'_strain')
    (out/'scripts').mkdir()
    for name in ('c_trigger.py', 'bayestar_si_fixed.py'):
        shutil.copy2(root/'runtime/training/scripts'/name, out/'scripts'/name)
    sys.path.insert(0, str(out/'scripts'))
    import c_trigger as engine
    (out/'bank').mkdir()
    print('BUILDING_FROZEN_8192_BANK', flush=True)
    engine.build_bank(out)
    expected = json.loads((root/'fixtures/recovery/EXPECTED_BANK.json').read_text())
    bank_hash = sha(out/'bank/templates.npy')
    if bank_hash != expected['sha256']:
        raise RuntimeError('Regenerated template bank differs')
    before = pd.read_csv(root/'fixtures/recovery/EXPECTED_BANK_PARAMS.csv')
    after = pd.read_csv(out/'bank/templates.csv')
    cols = [x for x in before if x != 'template_seconds']
    pd.testing.assert_frame_equal(before[cols], after[cols], check_exact=True)
    tasks = []
    for run in RUNS:
        source_uid = json.loads((root/f'fixtures/injections/{run}/EXPECTED.json').read_text())['source_uid']
        folder = strain_root/f'data/{run}/main/validation/{source_uid}'
        events = pd.read_parquet(folder/'metadata.parquet')
        for record in events.sort_values('event_uid').to_dict('records'):
            record.update(strain_path=record['raw_strain_path'],
                strain_sha256=record['raw_strain_sha256'], start_gps=record['raw_start_gps'],
                optimal_snr=record['optimal_network_snr'])
            tasks.append((record, run, True))
    started = time.monotonic()
    results = []
    with ProcessPoolExecutor(max_workers=3, mp_context=mp.get_context('spawn'),
                             initializer=init_worker, initargs=(str(out),str(root))) as pool:
        for row in pool.map(recover, tasks):
            if row['status'] != 'PASS':
                (out/'FAILURE.json').write_text(json.dumps(row, indent=2))
                raise RuntimeError(row.get('traceback', 'Recovery failed'))
            run, uid = row['run'], row['event_uid']
            ref = root/f'fixtures/recovery/{run}/{uid}'
            old = json.loads((ref/'RESULT.json').read_text())
            checks = []
            for name in ('selected_template', 'bank_templates'):
                checks.append({'field': name, 'exact': row[name] == old[name]})
            for name in ('network_matched_snr', 'reweighted_network_snr', 'sky_normalization',
                         'area50_deg2','area90_deg2','entropy_nats','kl_from_isotropic_nats',
                         'truth_credible_level_raw'):
                checks.append({'field': name, 'close': bool(np.isclose(row[name], old[name], rtol=1e-9, atol=1e-10)),
                               'max_abs': float(abs(row[name]-old[name]))})
            with np.load(ref/'TRIGGERS.npz') as expected_triggers, np.load(out/f'timings/{run}/{uid}/TRIGGERS.npz') as actual:
                for key in expected_triggers.files:
                    diff = float(np.max(np.abs(actual[key]-expected_triggers[key])))
                    checks.append({'field': 'trigger/'+key, 'exact': bool(np.array_equal(actual[key], expected_triggers[key])),
                                   'max_abs': diff})
            original = Table.read(ref/'EXPECTED_MAP.fits.gz')
            current = Table.read(row['moc_path'])
            for name in original.colnames:
                exact = bool(np.array_equal(original[name], current[name]))
                close = exact if name=='UNIQ' else bool(np.allclose(original[name], current[name], rtol=1e-7, atol=1e-12, equal_nan=True))
                checks.append({'field': 'map/'+name, 'exact': exact, 'close': close})
            passed = all(c.get('close', c.get('exact', False)) for c in checks)
            results.append({'run': run, 'event_uid': uid, 'passed': passed, 'checks': checks,
                            'search_seconds': row['search_seconds'], 'sky_seconds': row['sky_seconds']})
            (out/'PROGRESS.json').write_text(json.dumps(results, indent=2))
            print(run, uid, 'PASS' if passed else 'MISMATCH', flush=True)
    report = {'status': 'PASS' if all(r['passed'] for r in results) else 'FAIL',
              'events': len(results), 'bank_templates': len(after), 'bank_sha256_exact': True,
              'bank_sha256': bank_hash, 'wall_seconds': time.monotonic()-started, 'results': results,
              'scope': 'Three fixed validation doublets, regenerated noisy strain, full 8192-template search and native BAYESTAR maps.',
              'noise_source': 'Frozen real-noise bank slices, not fresh downloaded strain.',
              'recovery_worker_historical_project_reads_blocked': True,
              'diagnostic_loader': 'Exact AST functions/constants from frozen bayestar_si_fixed.py; unused legacy top-level initialization not executed.',
              'not_validated': ['Whole-population retraining', 'Fresh noise-bank generation', 'Cross-host floating-point identity']}
    (out/'REPORT.json').write_text(json.dumps(report, indent=2))
    if report['status'] != 'PASS':
        raise RuntimeError('Numerical reproduction mismatch; historical results unchanged')

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--release', required=True, type=Path)
    p.add_argument('--core', type=Path)
    p.add_argument('--baseline', type=Path)
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--output-tag', default='end_to_end_recovery')
    a = p.parse_args()
    if a.prepare_only:
        prepare(a.release, a.core, a.baseline)
    else:
        execute(a.release, a.output_tag)
