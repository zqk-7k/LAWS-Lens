#!/usr/bin/env python3
"""Prefetch at most the next two frozen proposals; no change to sample selection."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import shutil
import sys
import threading

P = Path('/root/autodl-tmp/gw-catalog')
sys.path.insert(0, str(P/'scripts/experiments'))
import mcwf_bulk_noise_relay_runtime_20260909 as relay
b = relay.b
LOCK = threading.Lock()
LOCKS = {}
QUEUED = set()
PLAN = None
POOL = None
ORIGINAL_URLS = b.get_urls


def locked_download(url):
    with LOCK:
        lock = LOCKS.setdefault(url, threading.Lock())
    with lock:
        return relay.download(url)


def acquire(row, detector):
    try:
        urls = ORIGINAL_URLS(detector, row.query_gps, row.query_gps+1, dataset=row.run, sample_rate=4096)
        if len(urls) == 1:
            locked_download(urls[0])
    except Exception as error:
        # Selection and error decisions stay with the original serial consumer.
        print('PREFETCH_DEFERRED', row.proposal, detector, type(error).__name__, str(error), flush=True)


def urls(detector, start, end, **kwargs):
    matches = PLAN.index[PLAN.query_gps == start]
    if len(matches) == 1:
        index = int(matches[0])
        for row in PLAN.iloc[index+1:index+3].itertuples():
            for det in ('H1', 'L1'):
                key = row.proposal, det
                if key not in QUEUED:
                    QUEUED.add(key)
                    POOL.submit(acquire, row, det)
    return ORIGINAL_URLS(detector, start, end, **kwargs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--workers', type=int, default=20)
    args = parser.parse_args()
    relay.ROOT = b.ROOT = b.g.ROOT = args.root
    relay.PORT = args.port
    b.g.SEED = b.SEED
    receipt = args.root/'contracts/PREFETCH_TRANSPORT_ADDENDUM.json'
    if not receipt.exists():
        b.n.write_json(receipt, {'UTC': b.n.utc(), 'transport_only': True,
            'rule': 'At most nexttwo frozen proposalrows,4backgroundworkers,perURLlock,originalserialqualityselectionunchanged.',
            'unused_prefetched_files': 'Retainasattemptedpublicsources;nevercountasacceptednoise.',
            'scientific_thresholds_unchanged': True, 'script_sha256': b.n.sha(Path(__file__))})
        shutil.copy2(__file__, args.root/'scripts/bulk_noise_prefetch_runtime.py')
    b.download, b.get_urls, b.g.noise = locked_download, urls, b.noise
    for dep in b.n.DEPS:
        PLAN = b.plan(dep)
        QUEUED.clear()
        with ThreadPoolExecutor(max_workers=4) as POOL:
            b.g.generate(dep, args.workers)
