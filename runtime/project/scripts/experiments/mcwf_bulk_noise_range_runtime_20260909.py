#!/usr/bin/env python3
"""Bounded parallel byte-range transfers, with immutable source validation."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import re
import shutil
import sys
import threading
import time
from urllib.parse import urlparse
import h5py
import requests

P = Path('/root/autodl-tmp/gw-catalog')
sys.path.insert(0, str(P/'scripts/experiments'))
import mcwf_independent_bulk_noise_development_20260909 as b
ROOT = None
CHUNK = 512*1024
LOCAL = threading.local()


def fetch(url, lo, hi, total=None, etag=None):
    if not hasattr(LOCAL, 'session'):
        LOCAL.session = requests.Session()
    error = None
    for attempt in range(4):
        try:
            headers = {'Range': f'bytes={lo}-{hi}', 'Accept-Encoding': 'identity'}
            if etag and not etag.startswith('W/'):
                headers['If-Range'] = etag
            response = LOCAL.session.get(url, headers=headers, timeout=(15, 30))
            response.raise_for_status()
            match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', response.headers.get('Content-Range', ''))
            if response.status_code != 206 or match is None:
                raise RuntimeError('Server must honor byte ranges')
            aa, bb, size = map(int, match.groups())
            if aa != lo or bb != hi or len(response.content) != hi-lo+1 or (total is not None and size != total):
                raise RuntimeError('Invalid byte-range payload')
            if etag is not None and response.headers.get('ETag') != etag:
                raise RuntimeError('Remote ETag changed during transfer')
            return response.content, size, response.headers.get('ETag'), response.headers.get('Last-Modified')
        except requests.RequestException as exc:
            error = str(exc)
            time.sleep(attempt+1)
    raise RuntimeError('Byte range transfer failed: '+str(error))


def download(url):
    dest = ROOT/'raw_sources'/Path(urlparse(url).path).name
    dest.parent.mkdir(parents=True, exist_ok=True)
    receipt = dest.with_suffix('.source.json')
    if dest.exists() and receipt.exists():
        if b.n.sha(dest) != json.loads(receipt.read_text())['sha256']:
            raise RuntimeError('Downloaded source changed')
        return dest
    tick = time.perf_counter()
    _, total, etag, modified = fetch(url, 0, 0)
    if total > 256*2**20 or shutil.disk_usage(ROOT).free < 25*2**30:
        raise RuntimeError('HOLD_DOWNLOAD_DISK_LIMIT')
    progress = dest.with_suffix('.ranges.json')
    state = json.loads(progress.read_text()) if progress.exists() else {
        'url': url, 'total': total, 'ETag': etag, 'Last-Modified': modified, 'chunk_bytes': CHUNK, 'done': []}
    if (state['url'], state['total'], state['ETag'], state['chunk_bytes']) != (url, total, etag, CHUNK):
        raise RuntimeError('Source changed from transfer checkpoint')
    part = dest.with_suffix('.ranges.partial')
    fd = os.open(part, os.O_RDWR | os.O_CREAT, 0o600)
    os.ftruncate(fd, total)
    jobs = [(k, lo, min(lo+CHUNK, total)-1) for k, lo in enumerate(range(0, total, CHUNK)) if k not in state['done']]
    try:
        with ThreadPoolExecutor(max_workers=8) as pool:
            pending = {pool.submit(fetch, url, lo, hi, total, etag): (k, lo) for k, lo, hi in jobs}
            for future in as_completed(pending):
                k, lo = pending[future]
                data = future.result()[0]
                count = os.pwrite(fd, data, lo)
                if count != len(data):
                    raise RuntimeError('Short local file write')
                os.fsync(fd)
                state['done'].append(k)
                b.n.write_json(progress, state)
    finally:
        os.close(fd)
    with h5py.File(part) as file:
        if 'strain/Strain' not in file:
            raise RuntimeError('Downloaded HDF5 missing strain')
    part.rename(dest)
    b.n.write_json(receipt, {**state, 'UTC': b.n.utc(), 'sha256': b.n.sha(dest),
        'bytes': total, 'seconds': time.perf_counter()-tick, 'transport': '8workers512KiBRange206verified'})
    print('PUBLIC_NOISE_RANGE_COMPLETE', dest.name, total, round(time.perf_counter()-tick, 2), flush=True)
    return dest


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=20)
    args = parser.parse_args()
    ROOT = b.ROOT = b.g.ROOT = args.root
    b.g.SEED = b.SEED
    record = ROOT/'contracts/RANGE_TRANSFER_ADDENDUM.json'
    if not record.exists():
        b.n.write_json(record, {'UTC': b.n.utc(), 'change': 'Only data transport:validated512KiBbyte ranges,8concurrentrequests,range-levelcheckpoint.',
            'reason': 'Remote singleHTTPtransfers stall/truncate around1MiB;publicalleventsJSONdownloaded locallyandtransferred unchanged.',
            'event_catalog_url': 'https://gwosc.org/eventapi/json/allevents/',
            'event_catalog_sha256': b.n.sha(ROOT/'contracts/GWOSC_ALL_EVENTS.json'),
            'no_source_selection_or_scientific_threshold_change': True, 'code_sha256': b.n.sha(Path(__file__))})
        shutil.copy2(__file__, ROOT/'scripts/bulk_noise_range_runtime.py')
    b.download = download
    b.g.noise = b.noise
    for dep in b.n.DEPS:
        b.g.generate(dep, args.workers)
