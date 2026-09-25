#!/usr/bin/env python3
"""Run only preregistered score contrasts; stop on any execution error."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def preserve_failed_cache_attempt(root, trial, kind, arm, log):
    record = log.with_suffix('.runtime.json')
    state = json.loads(record.read_text())
    message = log.read_text()
    if state.get('exit_code') != 1 or 'FileNotFoundError' not in message or '.json.tmp' not in message:
        raise RuntimeError('Not the diagnosed shared-cache publication failure')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    destination = root / 'failed_attempts' / f'{kind}_{arm}_{stamp}'
    destination.mkdir(parents=True, exist_ok=False)
    manifest = [{'path': str(p.relative_to(trial)), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
                for p in sorted(trial.rglob('*')) if p.is_file()]
    trial.rename(destination / 'trial')
    (destination / 'RECOVERY.json').write_text(json.dumps({
        'reason': 'Concurrent publication of identical frozen warm logits; no science/config change',
        'original_path': str(trial), 'original_log': str(log), 'runtime': state,
        'preserved_files': manifest, 'retrained': False, 'UTC': stamp}, indent=2))
    return root / 'logs' / f'evaluate_{kind}_{arm}_recovery_{stamp}.log'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--kind', required=True)
    p.add_argument('--skip-existing', action='store_true')
    p.add_argument('--recover-cache-collision', action='store_true')
    args = p.parse_args()
    scripts = Path(__file__).parent
    logger = scripts / 'mcwf_omc_extension_run_logged_20260907.py'
    program = scripts / ('mcwf_spectral_residual_20260908.py' if args.kind.startswith('TF-')
                         else 'mcwf_noise_context_20260908.py')
    for arm in ('ADD-PRIOR', 'ADD-BC', 'REPLACE-PRIOR', 'REPLACE-BC'):
        trial = args.root / 'variants' / args.kind / 'trials' / arm
        marker = trial / 'contracts/SEARCH_COMPLETE.json'
        if marker.exists() and args.skip_existing:
            continue
        log = args.root / 'logs' / f'evaluate_{args.kind}_{arm}.log'
        if trial.exists() and args.recover_cache_collision:
            log = preserve_failed_cache_attempt(args.root, trial, args.kind, arm, log)
        elif trial.exists():
            raise RuntimeError('Incomplete/existing trial must not be overwritten: ' + str(trial))
        command = [sys.executable, '-B', str(logger), '--log', str(log), sys.executable, '-B', str(program),
                   '--root', str(args.root), '--phase', 'evaluate', '--kind', args.kind, '--arm', arm]
        subprocess.run(command, check=True)
        if not marker.exists():
            raise RuntimeError('Child exited without completion marker')
        state = json.loads(marker.read_text())
        print(json.dumps({'completed': str(trial), 'both_runs_qualify': state['both_runs_qualify']}), flush=True)


if __name__ == '__main__':
    main()
