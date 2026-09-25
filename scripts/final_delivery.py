"""Freeze final audit bundle after testing the actual delivered payloads."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import shutil
import tarfile

from package_release import sha, verify


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--release', type=Path, required=True)
    root = parser.parse_args().release.resolve()
    packages = root / 'packages'
    smoke = json.loads((packages / 'BUNDLE_SMOKE_REPORT.json').read_text())
    if smoke['status'] != 'PASS':
        raise RuntimeError('Actual bundle replay did not pass; cannot finish successful delivery')
    guards = []
    replay = root / 'verification/delivered_bundle_replay'
    for path in sorted((replay / 'verification').glob('PORTABILITY_*.json')):
        item = json.loads(path.read_text())
        guards.append({'file': path.name, 'status': item['status'], 'violations': item['violations']})
    if len(guards) != 4 or any(row['status'] != 'PASS' or row['violations'] for row in guards):
        raise RuntimeError('Historical-data isolation checks incomplete or failed')
    (root / 'reports/validation_evidence/DELIVERED_PATH_ISOLATION.json').write_text(
        json.dumps({'status': 'PASS', 'checks': guards}, indent=2) + '\n')
    evidence = root / 'reports/validation_evidence/DELIVERED_BUNDLE_REPLAY.json'
    if evidence.exists():
        raise FileExistsError(evidence)
    shutil.copy2(packages / 'BUNDLE_SMOKE_REPORT.json', evidence)
    note = """# Final Packaging and Replay Verification

All six recommended archives have file-level manifests and SHA-256 checks.
The four modular payloads were extracted together into a new directory.
Using the isolated environment created in this task, the actual packaged
metrics/sky, features, injection examples, models and paper-analysis entry
points all passed. The historical-project read guard remains enabled for
features, models, injection and metrics/sky replay, permitting only the
current interpreter environment and the extracted release data.

This is a same-server extraction test, not a new-host full reproduction.
The limits in MISSING_ITEMS.json remain: eleven current data-figure entry
points unverified, no full raw-corpus download or full retraining, and
author/license review pending. The paper stage reproduces the fixed-GPD
tables and two supported plots, not all manuscript figures.

Packaging implementation repairs: the wheel archive contains two metadata
files in addition to wheels; the verifier now checks these against the
independent supplement copies, including the renamed requirements.lock.
No wheel, model, score, original input or historical result was changed.
Replay log names are unique, so bundled audit logs do not block a new run.

Use RELEASE_PACKAGE_INDEX_FINAL.json and SHA256SUMS_FINAL.txt. The earlier
audit tarball is retained as a superseded packaging attempt. The final
audit-v2 tarball replaces it in the recommended six-file set; do not unpack
both audit versions into the same directory.

Status: HOLD_FOR_AUTHOR_REVIEW_NO_PUBLICATION.
"""
    (root / 'reports/FINAL_DELIVERY_NOTE.md').write_text(note)
    index = json.loads((packages / 'RELEASE_PACKAGE_INDEX.json').read_text())
    quarantine = set(json.loads((root / 'contracts/PUBLICATION_QUARANTINE.json').read_text())['paths'])
    paths = []
    for folder in ('audit', 'contracts', 'reports', 'scripts', 'supplement', 'logs', 'README.md'):
        source = root / folder
        for path in ([source] if source.is_file() else source.rglob('*')):
            rel = path.relative_to(root)
            if not path.is_file() or path.is_symlink() or str(rel) in quarantine:
                continue
            if any(x in rel.parts for x in ('.git', '.secrets', '__pycache__')) or path.suffix == '.pyc':
                continue
            paths.append(path)
    manifest = [{'path': str(path.relative_to(root)), 'bytes': path.stat().st_size, 'sha256': sha(path)}
                for path in sorted(set(paths))]
    content = json.dumps(manifest, ensure_ascii=False, indent=2).encode()
    manifest_name = 'package_manifests/release_audit.json'
    expected = {row['path']: row['sha256'] for row in manifest}
    expected[manifest_name] = hashlib.sha256(content).hexdigest()
    target = packages / 'LAWS_Lens_v1_DRAFT_release_audit_v2_20260924.tar.gz'
    with tarfile.open(target, 'x:gz', compresslevel=1) as archive:
        for path in sorted(set(paths)):
            archive.add(path, arcname=str(path.relative_to(root)), recursive=False)
        member = tarfile.TarInfo(manifest_name)
        member.size = len(content)
        member.mode = 0o644
        archive.addfile(member, io.BytesIO(content))
    check = verify(target, expected)
    replacement = {'role': 'release_audit', 'file': target.name, 'bytes': target.stat().st_size,
                   'sha256': sha(target), **check}
    previous = next(row for row in index['packages'] if row['role'] == 'release_audit')
    index['packages'] = [replacement if row['role'] == 'release_audit' else row for row in index['packages']]
    index['superseded_not_recommended'] = [previous['file']]
    index['total_bytes'] = sum(row['bytes'] for row in index['packages'])
    index['status'] = 'HOLD_FOR_AUTHOR_REVIEW_NO_PUBLICATION'
    index['delivery_replay'] = 'BUNDLE_SMOKE_REPORT.json'
    index['audit_v2_change_scope'] = 'Packaging verifier repair and replay evidence only; scientific payloads unchanged.'
    (packages / 'RELEASE_PACKAGE_INDEX_FINAL.json').write_text(json.dumps(index, indent=2) + '\n')
    (packages / (target.name + '.sha256')).write_text(replacement['sha256'] + '  ' + target.name + '\n')
    checksums = ''.join(row['sha256'] + '  ' + row['file'] + '\n' for row in index['packages'])
    for name in ('RELEASE_PACKAGE_INDEX_FINAL.json', 'BUNDLE_SMOKE_REPORT.json'):
        checksums += sha(packages / name) + '  ' + name + '\n'
    (packages / 'SHA256SUMS_FINAL.txt').write_text(checksums)
    print(json.dumps(index, indent=2))


if __name__ == '__main__':
    main()
