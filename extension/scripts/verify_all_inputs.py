"""Read-only verification of every frozen original input, not a download test."""
import argparse
import hashlib
import json
from pathlib import Path
import time

p = argparse.ArgumentParser()
p.add_argument('--manifest', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=False)
rows = json.loads(a.manifest.read_text())
checks = []
start = time.monotonic()
with (a.output / 'INPUT_CHECKS.jsonl').open('x') as output:
    for row in sorted(rows, key=lambda r: (r['bytes'], r['input_id'])):
        path = Path(row['original_path'])
        item = {'input_id': row['input_id'], 'path': str(path), 'expected': row['sha256'], 'bytes': row['bytes']}
        try:
            with path.open('rb') as f:
                item['actual'] = hashlib.file_digest(f, 'sha256').hexdigest()
            item['passed'] = item['actual'] == row['sha256'] and path.stat().st_size == row['bytes']
        except OSError as exc:
            item.update(passed=False, error=str(exc))
        checks.append(item)
        output.write(json.dumps(item) + '\n')
        output.flush()
        if len(checks) % 50 == 0:
            print(json.dumps({'checked': len(checks), 'failed': sum(not x['passed'] for x in checks),
                              'elapsed_seconds': time.monotonic()-start}), flush=True)
report = {'status': 'PASS' if all(x['passed'] for x in checks) else 'FAIL',
          'files': len(checks), 'bytes': sum(x['bytes'] for x in checks),
          'wall_seconds': time.monotonic()-start,
          'failed': [x for x in checks if not x['passed']], 'fresh_download': False}
(a.output / 'REPORT.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report), flush=True)
raise SystemExit(0 if report['status']=='PASS' else 1)
