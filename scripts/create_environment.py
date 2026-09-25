"""Install a new isolated environment from the frozen, hashed wheelhouse."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import venv

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--target', type=Path, required=True)
    ap.add_argument('--wheelhouse', type=Path, required=True)
    ap.add_argument('--lock', type=Path, required=True)
    ap.add_argument('--report-dir', type=Path, required=True)
    a = ap.parse_args()
    if a.target.exists():
        raise SystemExit('Use a new environment; existing environments are never modified.')
    if sys.version_info[:2] != (3,12):
        raise SystemExit('This archived lock requires CPython 3.12, Linux x86_64.')
    a.report_dir.mkdir(parents=True, exist_ok=True)
    tick = time.monotonic()
    venv.EnvBuilder(with_pip=True, system_site_packages=False).create(a.target)
    python = a.target/'bin/python'
    commands = [
        [str(python),'-m','pip','install','--no-index','--find-links',str(a.wheelhouse),
         '--require-hashes','-r',str(a.lock)],
        [str(python),'-m','pip','check'],
    ]
    results = []
    for i, command in enumerate(commands):
        with (a.report_dir/f'step_{i}.log').open('w') as stream:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
        results.append(dict(command=command, returncode=result.returncode))
        if result.returncode:
            break
    (a.report_dir/'REPORT.json').write_text(json.dumps(dict(
        status='PASS' if len(results)==2 and all(x['returncode']==0 for x in results) else 'FAIL',
        results=results, seconds=time.monotonic()-tick,
        system_site_packages=False, offline_hash_locked=True),indent=2))
    if any(x['returncode'] for x in results):
        raise SystemExit('Installation failed; see logs.')
    print('FRESH_ENVIRONMENT_PASS', flush=True)

if __name__ == '__main__':
    main()
