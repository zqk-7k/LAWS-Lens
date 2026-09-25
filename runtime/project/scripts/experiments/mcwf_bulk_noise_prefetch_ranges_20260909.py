#!/usr/bin/env python3
"""Prefetch registered O4a public files without altering the generation order."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import shutil
import sys
from urllib.parse import urlencode

P = Path('/root/autodl-tmp/gw-catalog')
sys.path.insert(0, str(P/'scripts/experiments'))
import mcwf_bulk_noise_range_runtime_20260909 as ranges
b = ranges.b
FETCH = ranges.fetch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--port', type=int, default=29742)
    parser.add_argument('--proposals', type=int, default=20)
    args = parser.parse_args()
    root = args.root
    b.ROOT = b.g.ROOT = root
    ranges.ROOT = root/'cache/registered_prefetch'
    ranges.CHUNK = 4*1024*1024
    ranges.fetch = lambda url, *a, **k: FETCH(
        'http://127.0.0.1:'+str(args.port)+'/?'+urlencode({'url': url}), *a, **k)
    plan = b.plan('gwtc4').head(args.proposals)
    receipt = root/'contracts/REGISTERED_PREFETCH_TRANSPORT.json'
    if not receipt.exists():
        b.n.write_json(receipt, {'UTC': b.n.utc(), 'transport_only': True,
            'deployment': 'gwtc4', 'registered_proposals': plan.proposal.tolist(),
            'maximum_file_workers': 4, 'byte_ranges_per_file': 8,
            'publication': 'Verified immutable HDF5 and SHA; no scientific selection; original generator retains registered proposal order.',
            'code_sha256': b.n.sha(Path(__file__))})
        shutil.copy2(__file__, root/'scripts/bulk_noise_prefetch_ranges.py')
    urls = []
    for row in plan.itertuples():
        for det in ('H1', 'L1'):
            values = b.get_urls(det, row.query_gps, row.query_gps+1, dataset=row.run, sample_rate=4096)
            if len(values) == 1:
                urls.append(values[0])
    urls = list(dict.fromkeys(urls))
    def download(url):
        name = Path(url.split('?')[0]).name
        target = root/'raw_sources'/name
        target_receipt = target.with_suffix('.source.json')
        if target.exists() and target_receipt.exists():
            if b.n.sha(target) != json.loads(target_receipt.read_text())['sha256']:
                raise RuntimeError('Existing source SHA mismatch')
            return name
        if shutil.disk_usage(root).free < 25*2**30:
            raise RuntimeError('HOLD_DISK_LIMIT')
        source = ranges.download(url)
        data = json.loads(source.with_suffix('.source.json').read_text())
        if b.n.sha(source) != data['sha256']:
            raise RuntimeError('Prefetch SHA mismatch')
        # Hard links publish verified immutable data without a second dense copy.
        for a, z in ((source.with_suffix('.source.json'), target_receipt), (source, target)):
            try:
                os.link(a, z)
            except FileExistsError:
                if b.n.sha(a) != b.n.sha(z):
                    if a.suffix == '.json':
                        if json.loads(z.read_text()).get('sha256') == data['sha256']:
                            continue
                    raise RuntimeError('Refusing to replace a different canonical source')
        print('REGISTERED_PREFETCH_READY', name, flush=True)
        return name
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(download, url) for url in urls]
        completed = [f.result() for f in as_completed(futures)]
    b.n.write_json(root/'contracts/REGISTERED_PREFETCH_COMPLETE.json',
                   {'UTC': b.n.utc(), 'files': completed, 'count': len(completed)})


if __name__ == '__main__':
    main()
