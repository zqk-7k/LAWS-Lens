#!/usr/bin/env python3
"""Package completed, including failed, simulation pilots and sampling audits."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import tarfile


def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def main(roots, report, output):
    stage = output.with_suffix('').with_suffix('')
    if output.exists() or stage.exists():
        raise RuntimeError('Independent staging and archive required')
    for root in roots:
        if not any((root/'contracts'/name).exists() for name in
                   ('PILOT_GATE.json', 'SAMPLING_AUDIT_COMPLETE.json')):
            raise RuntimeError('Source has no completed gate or audit')
    stage.mkdir(parents=True)
    source_files = []
    for root in roots:
        for path in sorted(root.rglob('*')):
            if not path.is_file():
                continue
            relative = path.relative_to(root)
            if path.is_symlink():
                raise RuntimeError('No symlinks')
            if any(v in ('cache', '__pycache__', '.ssh', '.env') for v in relative.parts):
                continue
            if path.suffix in ('.npy', '.npz', '.h5', '.hdf5', '.key', '.pem', '.pyc'):
                continue
            if path.stat().st_size < 16 * 1024**2 and re.search(
                rb'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----[\r\n]', path.read_bytes()):
                raise RuntimeError('Private key content detected')
            dest = stage/root.name/relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            digest = sha(path)
            shutil.copy2(path, dest)
            if digest != sha(dest) or digest != sha(path):
                raise RuntimeError('Source changed during packaging')
            source_files.append({'path': str(path), 'sha256': digest})
    shutil.copy2(report, stage/report.name)
    shutil.copy2(__file__, stage/Path(__file__).name)
    (stage/'PACKAGE_SCOPE.json').write_text(json.dumps({
        'UTC': datetime.now(timezone.utc).isoformat(), 'source_roots': list(map(str, roots)),
        'source_files': source_files, 'goal_achieved': False,
        'status': 'HOLD_FOR_AUTHOR_REVIEW_NO_ADOPTION_NO_OVERWRITE',
        'scope': 'R52 and R54 failed simulation-only pilots; R53 sampling-probability audit. No new real ranking.'},
        indent=2), encoding='utf-8')
    hashes = {str(path.relative_to(stage)): sha(path) for path in sorted(stage.rglob('*')) if path.is_file()}
    (stage/'SHA256SUMS.txt').write_text(''.join(digest+'  '+name+'\n' for name, digest in hashes.items()))
    with tarfile.open(output, 'w:gz') as archive:
        archive.add(stage, arcname=stage.name)
    found = set()
    with tarfile.open(output, 'r|gz') as archive:
        for member in archive:
            if not member.isfile():
                continue
            name = str(Path(member.name).relative_to(stage.name))
            if name == 'SHA256SUMS.txt':
                continue
            if name not in hashes:
                raise RuntimeError('Unexpected member')
            digest = hashlib.sha256(archive.extractfile(member).read()).hexdigest()
            if digest != hashes[name]:
                raise RuntimeError('Archive hash mismatch')
            found.add(name)
    if found != set(hashes):
        raise RuntimeError('Archive missing files')
    result = {'archive': str(output), 'sha256': sha(output), 'bytes': output.stat().st_size,
        'hashed_files': len(found), 'hash_failures': 0, 'goal_achieved': False}
    output.with_suffix(output.suffix+'.sha256').write_text(result['sha256']+'  '+output.name+'\n')
    output.with_suffix(output.suffix+'.verification.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, nargs='+', required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    main(args.source, args.report, args.output)
