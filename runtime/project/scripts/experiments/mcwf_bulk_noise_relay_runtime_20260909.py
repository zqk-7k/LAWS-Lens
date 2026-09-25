#!/usr/bin/env python3
"""Use a verified loopback SSH relay; science and public source URLs unchanged."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import json
from pathlib import Path
import shutil
import sys
import time
from urllib.parse import urlencode, urlparse
import h5py
import requests

P = Path('/root/autodl-tmp/gw-catalog')
sys.path.insert(0, str(P/'scripts/experiments'))
import mcwf_independent_bulk_noise_development_20260909 as b
ROOT = None
PORT = None


def valid(path):
    with h5py.File(path) as file:
        if 'strain/Strain' not in file or len(file['strain/Strain']) != 4096*4096:
            raise RuntimeError('Not a complete public 4096s/4096Hz strain product')


def download(url):
    dest = ROOT/'raw_sources'/Path(urlparse(url).path).name
    receipt = dest.with_suffix('.source.json')
    if dest.exists() and receipt.exists():
        if b.n.sha(dest) != json.loads(receipt.read_text())['sha256']:
            raise RuntimeError('Input hash changed')
        return dest
    if shutil.disk_usage(ROOT).free < 25*2**30 or sum(p.stat().st_size for p in dest.parent.glob('*.hdf5')) > 20*2**30:
        raise RuntimeError('HOLD_DISK_LIMIT')
    tick = time.perf_counter()
    relay = dest.with_suffix('.local_relay.hdf5')
    if relay.exists():
        # This first archive file was already downloaded by curl and verified at both ends.
        expected = 'f668aef63e659d837bc05f36a6c4b993954a71751539f93eaae26f2f93cce6d7'
        if dest.name != 'H-H1_GWOSC_O3a_4KHZ_R1-1250869248-4096.hdf5' or b.n.sha(relay) != expected:
            raise RuntimeError('Unrecognized local relay artifact')
        valid(relay)
        if dest.exists():
            raise RuntimeError('Unreceipted destination already exists')
        shutil.copy2(relay, dest)
        transport = 'local-public-curl-SCP-matching-SHA256'
        attempts = 0
    else:
        part = dest.with_suffix('.ssh_relay.partial')
        for attempts in range(1, 4):
            try:
                u = f'http://127.0.0.1:{PORT}/?'+urlencode({'url': url})
                with requests.get(u, stream=True, timeout=(15, 90)) as response:
                    response.raise_for_status()
                    size = response.headers.get('Content-Length')
                    with part.open('wb') as file:
                        for chunk in response.iter_content(1024*1024):
                            file.write(chunk)
                        file.flush()
                        os.fsync(file.fileno())
                if size is not None and part.stat().st_size != int(size):
                    raise RuntimeError('Truncated relay transfer')
                valid(part)
                if dest.exists():
                    raise RuntimeError('Unreceipted destination already exists')
                part.rename(dest)
                break
            except (requests.RequestException, OSError) as error:
                print('PUBLIC_RELAY_RETRY', dest.name, attempts, str(error), flush=True)
                if attempts == 3:
                    raise
                time.sleep(attempts)
        transport = 'public-GWOSC-HTTPS-to-loopback-SSH-stream'
    b.n.write_json(receipt, {'UTC': b.n.utc(), 'url': url, 'sha256': b.n.sha(dest),
        'bytes': dest.stat().st_size, 'seconds': time.perf_counter()-tick, 'attempts': attempts, 'transport': transport})
    print('PUBLIC_NOISE_RELAY_COMPLETE', dest.name, dest.stat().st_size, round(time.perf_counter()-tick, 2), flush=True)
    return dest


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--workers', type=int, default=20)
    args = parser.parse_args()
    ROOT = b.ROOT = b.g.ROOT = args.root
    PORT = args.port
    b.g.SEED = b.SEED
    path = ROOT/'contracts/PUBLIC_RELAY_TRANSPORT_ADDENDUM.json'
    if not path.exists():
        b.n.write_json(path, {'UTC': b.n.utc(), 'change': 'Transport only: loopback-only public GWOSC HTTPS relay across authorized SSH tunnel; no local raw archive accumulation.',
            'prior_transport_attempts_preserved': True, 'scientific_selection_unchanged': True,
            'script_sha256': b.n.sha(Path(__file__))})
        shutil.copy2(__file__, ROOT/'scripts/bulk_noise_relay_runtime.py')
    b.download = download
    b.g.noise = b.noise
    for dep in b.n.DEPS:
        b.g.generate(dep, args.workers)
