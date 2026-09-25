"""Record the bounded public-download pilot without implying corpus coverage."""
import argparse
import hashlib
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--release', type=Path, required=True)
p.add_argument('--retry-exit-code', type=int)
args = p.parse_args()
root = args.release
manifest = root / 'supplement/GWLR_UC01_REPRODUCIBILITY_SUPPLEMENT/inputs/RAW_INPUT_ACQUISITION.json'
rows = json.loads(manifest.read_text())
checks = []
for row in rows:
    if row['input_id'] not in ('input_0000', 'input_0001', 'input_0002'):
        continue
    target = root / 'verification/public_download_pilot' / row['target_relative_path']
    check = {'input_id': row['input_id'], 'file': row['filename'], 'present': target.is_file()}
    if target.is_file():
        with target.open('rb') as stream:
            check['sha256'] = hashlib.file_digest(stream, 'sha256').hexdigest()
        check['matches_manifest'] = check['sha256'] == row['sha256']
    checks.append(check)
passed = all(c.get('matches_manifest') for c in checks) and len(checks) == 3
result = {
    'status': 'PASS' if passed else 'PARTIAL_NOT_FULL_CORPUS_VALIDATION',
    'checks': checks,
    'scope': 'Three pinned GW-LMC source tables only; no complete raw strain or PE re-download.',
    'retry_exit_code': args.retry_exit_code,
    'incomplete_reason': None if passed else 'Selected files remain absent; the bounded retry exit code is recorded when available. No success claimed for absent files.'
}
(root / 'audit/PUBLIC_DOWNLOAD_PILOT.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
