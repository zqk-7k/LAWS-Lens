"""Validate actual shipped archives in a separate, new extraction directory."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import time


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--release', type=Path, required=True)
    args = parser.parse_args()
    root = args.release.resolve()
    packages = root / 'packages'
    index = json.loads((packages / 'RELEASE_PACKAGE_INDEX.json').read_text())
    destination = root / 'verification/delivered_bundle_replay'
    destination.mkdir()
    check = {'scope': 'Fresh extraction on the same physical server, using this task\'s isolated environment.',
             'status': 'RUNNING', 'archives': [], 'stages': []}
    report = packages / 'BUNDLE_SMOKE_REPORT.json'

    def save():
        report.write_text(json.dumps(check, indent=2) + '\n')

    for row in index['packages']:
        bundle = packages / row['file']
        actual = sha(bundle)
        if actual != row['sha256']:
            raise ValueError('Archive checksum mismatch: ' + row['file'])
        check['archives'].append({'file': row['file'], 'sha256': actual, 'matches_index': True})
        if row['role'] not in ('code_models_scores', 'paper_fpp_speed', 'validation_fixtures', 'release_audit'):
            continue
        with tarfile.open(bundle, 'r|*') as archive:
            for member in archive:
                path = Path(member.name)
                if not member.isfile() or path.is_absolute() or '..' in path.parts:
                    raise ValueError('Unsafe or unexpected member: ' + member.name)
                if (destination / path).exists():
                    raise ValueError('Overlapping package member: ' + member.name)
                archive.extract(member, destination, filter='data')
    save()
    for stage in ('metrics-sky', 'features', 'injections', 'models', 'paper'):
        start = time.monotonic()
        process = subprocess.run([sys.executable, '-B', str(destination / 'scripts/reproduce.py'),
                                  stage, '--release', str(destination)], capture_output=True, text=True)
        entry = {'stage': stage, 'returncode': process.returncode,
                 'wall_seconds': time.monotonic() - start, 'stdout': process.stdout,
                 'stderr': process.stderr, 'status': 'PASS' if process.returncode == 0 else 'FAIL'}
        check['stages'].append(entry)
        print(json.dumps(entry), flush=True)
        save()
    check['status'] = 'PASS' if all(x['returncode'] == 0 for x in check['stages']) else 'FAIL'
    check['does_not_certify'] = ['Full population retraining or regeneration', 'Cross-host reproducibility',
                                  'All paper figures rebuilt', 'Scientific significance of real candidates']
    save()
    raise SystemExit(0 if check['status'] == 'PASS' else 1)


if __name__ == '__main__':
    main()
