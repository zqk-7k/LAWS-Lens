#!/usr/bin/env python3
"""Build and stream-verify a compact, immutable research-result package."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import tarfile

FORBIDDEN_NAMES = {'.ssh', 'id_rsa', 'id_ed25519', 'authorized_keys', '.env'}
OMIT_DIRS = {'cache', '__pycache__'}
OMIT_SUFFIXES = {'.npy', '.npz', '.h5', '.hdf5', '.pyc', '.pem', '.key'}
PRIVATE_KEY = re.compile(rb'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----\s+[A-Za-z0-9+/=\r\n]+-----END (?:RSA |OPENSSH |EC )?PRIVATE KEY-----')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda: stream.read(1024*1024), b''):
            h.update(part)
    return h.hexdigest()


def write(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, nargs='+', required=True)
    parser.add_argument('--output-directory', type=Path, required=True)
    parser.add_argument('--name', required=True)
    args = parser.parse_args()
    out = args.output_directory
    out.mkdir(parents=True, exist_ok=True)
    stage = out/(args.name+'_deliverables')
    archive = out/(args.name+'_deliverables.tar.gz')
    if stage.exists() or archive.exists():
        raise RuntimeError('Independent staging/package name required')
    for root in args.source:
        pilot = root/'contracts/PILOT_GATE.json'
        final = root/'contracts/FROZEN_COMPUTATIONS_COMPLETE.json'
        if not (pilot.exists() or final.exists()):
            raise RuntimeError('Source computations not complete: '+str(root))
        if final.exists() and not (root/'reports/FINAL_SHARED_PROFILE_CATALOG_REPORT_CN.md').exists():
            raise RuntimeError('Final catalog report is required before packaging')
    stage.mkdir()
    included, omitted = [], []
    for root in args.source:
        for path in sorted(root.rglob('*')):
            if not path.is_file():
                continue
            rel = path.relative_to(root)
            if path.is_symlink():
                raise RuntimeError('Symlink not permitted: '+str(path))
            if FORBIDDEN_NAMES.intersection(rel.parts):
                raise RuntimeError('Credential filename in source tree')
            if OMIT_DIRS.intersection(rel.parts) or path.suffix.lower() in OMIT_SUFFIXES:
                omitted.append({'root': str(root), 'relative_path': str(rel), 'bytes': path.stat().st_size})
                continue
            if path.stat().st_size < 16*1024**2 and PRIVATE_KEY.search(path.read_bytes()):
                raise RuntimeError('Complete private-key block in source file')
            dest = stage/root.name/rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            expected = sha(path)
            shutil.copy2(path, dest)
            if sha(dest) != expected or sha(path) != expected:
                raise RuntimeError('Source changed while copying')
            included.append({'source_path': str(path), 'package_path': str(dest.relative_to(stage)),
                'sha256': expected, 'bytes': dest.stat().st_size})
    shutil.copy2(__file__, stage/'package_reproduction.py')
    write(stage/'PACKAGE_PROVENANCE.json', {
        'UTC': datetime.now(timezone.utc).isoformat(), 'roots': [str(p) for p in args.source],
        'status': 'HOLD_FOR_AUTHOR_REVIEW_NO_ADOPTION_NO_OVERWRITE',
        'package_is_not_a_goal_success_declaration': True,
        'included_source_files': included, 'omitted_reconstructable_or_raw_files': omitted,
        'scope': 'Read each experiment contract; no raw strain, PE HDF5, private keys, or dense reconstruction cache.'})
    files = sorted(p for p in stage.rglob('*') if p.is_file())
    hashes = {str(p.relative_to(stage)): sha(p) for p in files}
    write(stage/'OUTPUT_SHA256.json', hashes)
    with (stage/'SHA256SUMS.txt').open('x') as stream:
        for name, value in hashes.items():
            stream.write(value+'  '+name+'\n')
    with tarfile.open(archive, 'w:gz', compresslevel=6) as tar:
        tar.add(stage, arcname=stage.name)
    failures, seen, members = [], set(), 0
    with tarfile.open(archive, 'r|gz') as tar:
        for member in tar:
            if not member.isfile():
                continue
            members += 1
            name = str(Path(member.name).relative_to(stage.name))
            if name not in hashes:
                if name not in {'OUTPUT_SHA256.json', 'SHA256SUMS.txt'}:
                    failures.append('Unexpected member: '+name)
                continue
            h = hashlib.sha256()
            stream = tar.extractfile(member)
            for part in iter(lambda: stream.read(1024*1024), b''):
                h.update(part)
            if h.hexdigest() != hashes[name]:
                failures.append('Hash mismatch: '+name)
            seen.add(name)
    missing = sorted(set(hashes)-seen)
    if failures or missing:
        raise RuntimeError('Archive failed integrity verification: '+repr((failures, missing)))
    digest = sha(archive)
    receipt = {'UTC': datetime.now(timezone.utc).isoformat(), 'archive': str(archive),
        'sha256': digest, 'bytes': archive.stat().st_size, 'regular_file_members': members,
        'internally_hashed_files': len(hashes), 'internal_hash_failures': len(failures),
        'missing_files': len(missing), 'unhashed_self_manifest_files': ['OUTPUT_SHA256.json', 'SHA256SUMS.txt']}
    write(out/(args.name+'_ARCHIVE_VERIFICATION.json'), receipt)
    with (out/(args.name+'_deliverables.sha256')).open('x') as stream:
        stream.write(digest+'  '+archive.name+'\n')
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == '__main__':
    main()
