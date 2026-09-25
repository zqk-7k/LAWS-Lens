"""Retrieve immutable public inputs with bounded curl transport and SHA checks."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

p = argparse.ArgumentParser()
p.add_argument('--manifest', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=False)
checks = []
for row in json.loads(a.manifest.read_text()):
    target = a.output / row['filename']
    partial = target.with_suffix('.part')
    start = time.monotonic()
    proc = subprocess.run(['curl', '--fail', '--location', '--connect-timeout', '10',
                           '--max-time', '45', '--retry', '1', '--output', str(partial),
                           row['acquisition_urls'][0]], capture_output=True, text=True)
    item = {'input_id': row['input_id'], 'file': row['filename'], 'url': row['acquisition_urls'][0],
            'returncode': proc.returncode, 'wall_seconds': time.monotonic() - start}
    if proc.returncode == 0:
        with partial.open('rb') as f:
            digest = hashlib.sha256()
            for block in iter(lambda: f.read(2**20), b''):
                digest.update(block)
            item['sha256'] = digest.hexdigest()
        item['matches_manifest'] = item['sha256'] == row['sha256']
        if item['matches_manifest']:
            partial.rename(target)
    else:
        item['error'] = proc.stderr[-2000:]
    checks.append(item)
    (a.output / 'DOWNLOAD_REPORT.json').write_text(json.dumps({'checks': checks,
        'complete': len(checks) == 3 and all(r.get('matches_manifest') for r in checks)}, indent=2))
    print(json.dumps(item), flush=True)
raise SystemExit(0 if all(r.get('matches_manifest') for r in checks) else 1)
