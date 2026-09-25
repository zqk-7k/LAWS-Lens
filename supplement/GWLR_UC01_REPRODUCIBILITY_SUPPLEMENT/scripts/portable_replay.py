"""Verify/extract immutable published artifacts into a new directory for replay."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile

def sha(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reviewed', type=Path, required=True)
    parser.add_argument('--reviewed-sha', default='bd5e31e619b0de31a7d605351fc06ff6be5cb828b59d5d0d667269d58e1865a5')
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--assets-manifest', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--verify-script', type=Path, required=True)
    args = parser.parse_args()
    if args.destination.exists():
        raise RuntimeError('Use a new destination; no historical directory is modified')
    if sha(args.reviewed) != args.reviewed_sha:
        raise RuntimeError('Wrong C reviewed artifact')
    assets = {r['archive_path']: r['sha256'] for r in json.loads(args.assets_manifest.read_text())}
    args.destination.mkdir(parents=True)
    for bundle in (args.reviewed, args.assets):
        with tarfile.open(bundle) as tar:
            for member in tar:
                relative = Path(member.name)
                if relative.is_absolute() or '..' in relative.parts or not member.isfile():
                    raise RuntimeError('Unexpected archive member: '+member.name)
                target = args.destination/relative
                with tar.extractfile(member) as stream:
                    content = stream.read()
                digest = hashlib.sha256(content).hexdigest()
                if bundle == args.assets and assets.get(member.name) != digest:
                    raise RuntimeError('Reconstruction asset mismatch')
                if target.exists():
                    if sha(target) != digest:
                        raise RuntimeError('Conflicting archives: '+str(target))
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open('xb') as stream:
                    stream.write(content)
    cmd = [sys.executable, '-B', str(args.verify_script),
           '--source-root', str(args.destination/'training'),
           '--project-root', str(args.destination/'project'),
           '--completion-root', str(args.destination/'completion'),
           '--output', str(args.destination/'verification')]
    subprocess.run(cmd, check=True)

if __name__ == '__main__':
    main()
