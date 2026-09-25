import csv
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
data = root/'extracted/gwlr_tail02_20260922T113500Z_r2'
manifest = data/'manifests/FINAL_SHA256.csv'
rows = list(csv.DictReader(manifest.open(encoding='utf-8-sig', newline='')))
seen = set()
for row in rows:
    p = (data/row['path']).resolve()
    assert p.is_relative_to(data.resolve())
    assert row['path'] not in seen
    seen.add(row['path'])
    assert p.stat().st_size == int(row['bytes']), row['path']
    with p.open('rb') as f:
        assert hashlib.file_digest(f, 'sha256').hexdigest() == row['sha256'], row['path']
actual = {p.relative_to(data).as_posix() for p in data.rglob('*') if p.is_file()}
assert actual - seen == {'manifests/FINAL_SHA256.csv'}
report = {'verified_files': len(rows), 'hash_failures': 0, 'unlisted_files': sorted(actual-seen), 'note': 'Final manifest excludes its own hash. Archived scripts were not executed.'}
with (root/'audit/INTERNAL_MANIFEST_CHECK.json').open('x', encoding='utf-8') as f:
    json.dump(report, f, indent=2)
print(json.dumps(report))
for name in ['gpd_fits.csv', 'HELDOUT_ENDPOINT_DIAGNOSTIC.csv', 'REAL_TOP10_EXTRAPOLATION_SUPPORT.csv', 'gates.csv']:
    print(name)
    rr = list(csv.DictReader((data/'results'/name).open(encoding='utf-8-sig', newline='')))
    print(json.dumps(rr[:3], ensure_ascii=False))
