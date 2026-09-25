"""Bounded C pipeline; any failed stage records HOLD and does not unlock test."""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import argparse
from concurrent.futures import ProcessPoolExecutor
import fcntl
import json
import multiprocessing as mp
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np
import pandas as pd

import unified_ab as u
import c_data
import c_trigger


def status(root, stage, **more):
    u.write(root/'RUN_STATUS.json', dict(state=stage, utc=u.now(), controller_pid=os.getpid(),
        complete_results=False, **more))


def task(root, label, script, *args):
    receipt = root/'contracts/tasks'/f'{label}.json'
    if receipt.exists() and json.loads(receipt.read_text())['exit_code'] == 0:
        return
    u.guard(root)
    path = root/'logs'/f'{label}_{time.time_ns()}.log'
    command = [sys.executable, '-B', '-u', str(root/'scripts'/script), '--root', str(root), *map(str,args)]
    start = time.monotonic()
    with path.open('x') as stream:
        child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
            env={**os.environ, 'PYTHONPATH': str(u.P)+os.pathsep+os.environ.get('PYTHONPATH','')})
        status(root, label, child_pid=child.pid, log=str(path))
        code = child.wait()
    u.write(receipt, dict(state=label, command=command, exit_code=code, log=str(path),
        seconds=time.monotonic()-start, utc=u.now()))
    if code:
        raise RuntimeError(f'{label} failed; {path}')


def pilot(root):
    u.verify(root)
    # Deterministic algebra tests do not inspect any validation or test score.
    for ratio in (1., 2., 10., 100.):
        for draw in (.0001, .5, .9999):
            base = np.array([3., 3.*ratio])
            factor = c_data.common_factor(base, draw)
            result = base*factor
            np.testing.assert_allclose(result[1]/result[0], ratio, rtol=1e-12)
            assert 8 <= result.min() <= 40
    u.write(root/'contracts/COMMON_AMPLITUDE_UNIT_TESTS.json', dict(state='PASS',
        cases=12, image_ratio_preserved=True, uses_one_factor=True))
    plan = pd.read_parquet(root/'plans/C_PILOT_SOURCES.parquet')
    rows = []
    for run in u.RUNS:
        status(root, 'C_PILOT_GENERATION', run=run)
        sources = pd.read_parquet(root/'plans'/run/'sources.parquet')
        sources = sources[sources.source_uid.isin(plan[plan.run == run].source_uid)]
        assert sources.split.eq('train').all() and len(sources) == 18
        with ProcessPoolExecutor(max_workers=8, mp_context=mp.get_context('spawn'),
                initializer=c_data.worker_init, initargs=(str(root),run,'main','train')) as pool:
            receipts = list(pool.map(c_data.generate_source, sources.to_dict('records')))
        for source in sources.source_uid:
            folder = root/'data'/run/'main/train'/source
            events = pd.read_parquet(folder/'metadata.parquet')
            if events.physical_strain_scale_factor.nunique() != 1:
                raise RuntimeError('More than one amplitude scale per source')
            # Reconstruct both waveform inputs from the persisted shared noisy strain.
            c_data.worker_init(root, run, 'main', 'train')
            physical, views = c_data.STATE['physical'], c_data.STATE['views']
            short = np.load(folder/'C_PHYSICAL_short.npy')
            low = np.load(folder/'C_PHYSICAL_long.npy')
            from scipy.signal import resample_poly
            for i, event in events[events.variant == 0].iterrows():
                with np.load(event.raw_strain_path) as raw:
                    cut = raw['noisy_raw'][:, int(event.raw_waveform_crop_start):int(event.raw_waveform_crop_stop)]
                    p = physical.preprocess_24s(cut, raw['psd_frequency'], raw['psd'])
                    np.testing.assert_array_equal(short[i], views.make_window_view(p[None],2)[0])
                    p = physical.preprocess_24s(cut, raw['psd_frequency'], raw['psd'],band_low_hz=20,band_high_hz=80)
                    np.testing.assert_array_equal(low[i], resample_poly(p,1,8,axis=-1,window=('kaiser',8.6))[:,-4096:])
                rec = event.to_dict()
                rec.update(run=run,strain_path=event.raw_strain_path,strain_sha256=event.raw_strain_sha256,
                    start_gps=event.raw_start_gps,optimal_snr=event.optimal_network_snr)
                rows.append(rec)
    u.write(root/'contracts/DATA_UNIT_TESTS_PASS.json', dict(state='PASS', pilot_events=len(rows),
        common_amplitude_all_views=True, waveform_inputs_rebuilt_bitwise_from_sky_raw_strain=True))
    pd.DataFrame(rows).to_parquet(root/'plans/PILOT_EVENT_MANIFEST.parquet',index=False)
    status(root, 'C_PILOT_INDEPENDENT_TEMPLATE_BANK', templates=c_trigger.BANK_COUNT)
    c_trigger.build_bank(root)
    tasks = [(row, 'pilot/'+row['run'], True) for row in rows]
    results = []
    status(root, 'C_PILOT_DATA_DERIVED_BAYESTAR', events=len(rows), workers=12)
    start = time.monotonic()
    with ProcessPoolExecutor(max_workers=12, mp_context=mp.get_context('spawn'),
            initializer=c_trigger.worker_init,initargs=(str(root),)) as pool:
        for result in pool.map(c_trigger.run_event,tasks,chunksize=1):
            results.append(result)
            u.write(root/'contracts/PILOT_PROGRESS.json', dict(completed=len(results),total=len(rows),
                failed=sum(r['status']!='PASS' for r in results),utc=u.now()))
            print(json.dumps({k:result.get(k) for k in ('run','event_uid','status','search_seconds','sky_seconds','traceback')}),flush=True)
    # Record all outcomes, including failures, before deciding continuation.
    u.write(root/'contracts/PILOT_OUTCOMES.json', results)
    failure = [r for r in results if r['status'] != 'PASS']
    if failure:
        raise RuntimeError(f'{len(failure)} C trigger/map jobs failed')
    frame = pd.DataFrame(results)
    frame.to_csv(root/'reports/C_PILOT_METRICS.csv',index=False,encoding='utf-8-sig')
    checks = []
    for run, group in frame.groupby('run'):
        coverage = float((group.truth_credible_T1 <= .9).mean())
        checks.append(dict(run=run,events=len(group),raw_HPD90_coverage=coverage,
            raw_HPD90_gross_failure_screen_pass=coverage>=.5,
            filter_seconds_p50=float(group.search_seconds.median()),
            filter_seconds_p90=float(group.search_seconds.quantile(.9)),
            reweighted_snr_median=float(group.reweighted_network_snr.median()),
            bank_coverage_certified=False, posterior_calibration_certified=False))
    passed = all(row['raw_HPD90_gross_failure_screen_pass'] for row in checks)
    u.write(root/'contracts/C_PILOT_GATE.json',dict(state='PASS' if passed else 'FAIL',
        checks=checks,seconds=time.monotonic()-start,limited_engineering_gate=True,
        not_scientific_acceptance_of_the_full_experiment=True))
    if not passed:
        raise RuntimeError('C development maps fail the preregistered gross coverage check')
    u.write(root/'contracts/BAYESTAR_PILOT_PASS.json',dict(state='PASS', events=len(frame),
        same_data_inputs=True, calibrated_posterior_claim=False, bank_coverage_claim=False))


def train(root):
    if not (root/'contracts/BAYESTAR_PILOT_PASS.json').exists():
        raise RuntimeError('C pilot not passed')
    for run in u.RUNS:
        for split in ('validation','train'):
            task(root,f'MAIN_{run}_{split}','unified_ab.py','--stage','generate','--run',run,
                 '--role','main','--split',split,'--workers',8)
        for seed in u.SEEDS:
            task(root,f'SHORT_{run}_C_PHYSICAL_{seed}','unified_ab.py','--stage','short','--run',run,
                 '--arm','C_PHYSICAL','--seed',seed)
        for split in ('validation','train'):
            task(root,f'AUX_{run}_{split}','unified_ab.py','--stage','generate','--run',run,
                 '--role','aux','--split',split,'--workers',8)
            task(root,f'FEATURES_{run}_{split}','train_components.py','--stage','features',
                 '--run',run,'--arm','C_PHYSICAL','--split',split)
        task(root,f'RNC_PREPARE_{run}','train_components.py','--stage','rnc-prepare','--run',run,'--arm','C_PHYSICAL')
        for seed in u.SEEDS:
            for stage in ('rnc','ordered','multirate','conditional'):
                task(root,f'{stage}_{run}_C_PHYSICAL_{seed}','train_components.py','--stage',stage,
                     '--run',run,'--arm','C_PHYSICAL','--seed',seed)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True)
    p.add_argument('--stage',choices=['run','pilot','train','complete'],default='run')
    args=p.parse_args(); root=args.root
    with (root/'contracts/C_CONTROLLER.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            if args.stage in ('run','pilot'):
                if not (root/'contracts/BAYESTAR_PILOT_PASS.json').exists():
                    pilot(root)
            if args.stage in ('run','train'):
                train(root)
            if args.stage in ('run','complete'):
                import c_finish
                c_finish.complete(root)
        except Exception:
            record=dict(state='HOLD_C_ERROR',utc=u.now(),complete_results=False,error=traceback.format_exc())
            u.write(root/'contracts'/f'FAILURE_{time.time_ns()}.json',record)
            u.write(root/'RUN_STATUS.json',record)
            raise


if __name__ == '__main__':
    main()
