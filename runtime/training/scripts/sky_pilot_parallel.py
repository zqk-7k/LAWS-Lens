#!/usr/bin/env python3
"""Parallelize the frozen pilot without changing its inputs or algorithms."""
import os
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[k] = '1'
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import unified_ab as u

CTX = {}


def initialize(root):
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)
    root = Path(root)
    b = u.module(root/'scripts/bayestar_si_fixed.py', 'uab_parallel_sky')
    CTX.update(root=root, b=b)


def one(item):
    from ligo.skymap.io.fits import write_sky_map
    run, row = item
    root, b = CTX['root'], CTX['b']
    b._WORKER_PSD_FREQ = np.load(root/'plans'/run/'noise_psd_frequency.npy')
    b._WORKER_PSD_BANK = np.load(root/'plans'/run/'noise_psd_bank.npy', mmap_mode='r')
    path = root/'pilot'/run/row['arm']/f'{row["event_uid"]}.fits.gz'
    receipt = path.with_suffix('.json')
    if receipt.exists():
        result = json.loads(receipt.read_text())
        if u.sha(path) != result['sha256']:
            raise RuntimeError('Completed pilot map hash mismatch')
        return result
    u.guard(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    sky, audit = b._simulate_with_waveform(row, pd.Series(row), b.PRIMARY_WAVEFORM)
    probability = b.raster_probability(sky, 512)
    if not np.isfinite(probability).all() or abs(probability.sum()-1) > 1e-10:
        raise RuntimeError('Invalid pilot probability map')
    temporary = path.with_name(path.name+'.temporary.fits.gz')
    write_sky_map(temporary, sky)
    if path.exists():
        raise RuntimeError('Uncommitted map exists; preserve it for inspection')
    temporary.replace(path)
    result = {**audit, 'run': run, 'arm': row['arm'], 'event_uid': row['event_uid'],
              'source_uid': row['source_uid'], 'mass_unit': 'kg for spin conversion',
              'analysis_nside': 512, 'ordering': 'NESTED', 'native_format': 'MOC density per steradian',
              'seconds': time.monotonic()-start, 'sha256': u.sha(path), 'not_full_BBH_PE': True}
    u.write(receipt, result)
    return result


def main(root, workers):
    u.verify(root)
    items = []
    for run in u.RUNS:
        frame = pd.concat([pd.read_parquet(p) for p in sorted((root/'data'/run/'main/validation').glob('*/metadata.parquet'))])
        items.extend((run, row) for row in frame.to_dict('records'))
    if len(items) != 60:
        raise RuntimeError(f'Frozen pilot expects 60 maps, found {len(items)}')
    rows = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn'),
                             initializer=initialize, initargs=(str(root),)) as pool:
        futures = [pool.submit(one, item) for item in items]
        for future in as_completed(futures):
            result = future.result(); rows.append(result)
            print(json.dumps({'complete': len(rows), 'total': len(items), 'run': result['run'], 'arm': result['arm'],
                              'event': result['event_uid'], 'seconds': result['seconds']}), flush=True)
    pd.DataFrame(rows).sort_values(['run', 'arm', 'event_uid']).to_csv(root/'pilot/BAYESTAR_PILOT.csv', index=False)
    u.write(root/'contracts/BAYESTAR_PILOT_PASS.json', {'utc': u.now(), 'events': len(rows),
             'full_experiment_complete': False, 'parallel_workers': workers,
             'meaning': 'Numerical end-to-end success only; not population coverage validation',
             'serial_scheduler_replaced_without_physical_changes': True})


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True); p.add_argument('--workers', type=int, default=8)
    args = p.parse_args(); main(args.root, args.workers)
