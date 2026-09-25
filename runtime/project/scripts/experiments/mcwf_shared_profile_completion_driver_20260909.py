#!/usr/bin/env python3
"""Execute only frozen R51 stages, stopping on any missing input or error."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import psutil

P = Path('/root/autodl-tmp/gw-catalog')
SCRIPTS = P/'scripts/experiments'
PYTHON = '/root/miniconda3/bin/python'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--pilot-root', type=Path, required=True)
    parser.add_argument('--measurement-pid', type=int, required=True)
    args = parser.parse_args()
    root = args.root
    required = ['PIPELINE_UNIT_PASS.json', 'CONFIGURATIONS_FROZEN.json', 'START_FREEZE.json']
    for name in required:
        if not (root/'contracts'/name).exists():
            raise RuntimeError('Missing prerequisite '+name)
    if not (root/'audit/REFERENCE_AUDIT_FROZEN.json').exists():
        raise RuntimeError('Freeze reference audit first')
    catalog = SCRIPTS/'mcwf_shared_profile_catalog_20260909.py'
    audit = SCRIPTS/'mcwf_shared_profile_catalog_audit_20260909.py'
    bootstrap = SCRIPTS/'mcwf_shared_profile_system_bootstrap_20260909.py'
    summary = SCRIPTS/'mcwf_shared_profile_bootstrap_summary_20260909.py'
    expected = json.loads((root/'contracts/START_FREEZE.json').read_text())['runtime_sha256']
    if sha(catalog) != expected:
        raise RuntimeError('Production runtime changed')
    audit_expected = json.loads((root/'audit/REFERENCE_AUDIT_FROZEN.json').read_text())['runtime_sha256']
    if sha(audit) != audit_expected:
        raise RuntimeError('Reference audit runtime changed')
    files = [catalog, audit, bootstrap, summary, Path(__file__)]
    frozen = {str(path): sha(path) for path in files}
    for path in files:
        target = root/'scripts'/path.name
        if target.exists():
            if sha(target) != sha(path):
                raise RuntimeError('Snapshot conflict')
        else:
            shutil.copy2(path, target)
    write(root/'contracts/COMPLETION_DRIVER_FROZEN.json', {'UTC': utc(), 'scripts': frozen,
        'measurement_pid': args.measurement_pid, 'workers': 24, 'bootstrap_workers': 6,
        'real_stage_only_after_complete_injection_evaluation': True,
        'no_calibration_or_reselection': True, 'no_automatic_adoption': True,
        'failures_stop_pipeline': True, 'goal_achieved': False})
    env = os.environ.copy()
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        env[key] = '1'

    def check_sources():
        for path, expected_hash in frozen.items():
            if sha(Path(path)) != expected_hash:
                raise RuntimeError('Frozen driver dependency changed: '+path)

    def stage(label, command, receipt):
        check_sources()
        if receipt.exists():
            raise RuntimeError('Stage already completed; no implicit reuse: '+label)
        start = time.monotonic()
        print('BEGIN_STAGE', label, utc(), flush=True)
        with (root/f'logs/DRIVER_{label}.stdout.log').open('x') as log:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, env=env)
            with process:
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    print(label, line.rstrip(), flush=True)
            code = process.returncode
        record = {'UTC': utc(), 'command': command, 'returncode': code,
            'wall_seconds': time.monotonic()-start, 'required_receipt': str(receipt),
            'receipt_exists': receipt.exists()}
        write(root/f'logs/DRIVER_{label}_RECEIPT.json', record)
        if code or not receipt.exists():
            raise RuntimeError('Stage did not complete: '+label)
        print('END_STAGE', label, utc(), flush=True)

    try:
        marker = root/'contracts/INJECTION_MEASUREMENT_COMPLETE.json'
        while not marker.exists():
            if (root/'contracts/INJECTION_MEASUREMENT_FAIL.json').exists():
                raise RuntimeError('Injection measurement failed')
            if not psutil.pid_exists(args.measurement_pid):
                raise RuntimeError('Measurement process ended without complete receipt')
            time.sleep(30)
        while psutil.pid_exists(args.measurement_pid):
            try:
                if psutil.Process(args.measurement_pid).status() == psutil.STATUS_ZOMBIE:
                    break
            except psutil.NoSuchProcess:
                break
            time.sleep(1)
        common = [PYTHON, '-B', str(catalog), '--root', str(root), '--pilot-root', str(args.pilot_root)]
        stage('INJECTION_EVALUATION', common+['--stage', 'evaluate'], root/'contracts/EVALUATION_COMPLETE.json')
        stage('REAL_MEASUREMENT', common+['--stage', 'measure', '--scope', 'real', '--workers', '24'],
              root/'contracts/REAL_MEASUREMENT_COMPLETE.json')
        stage('REAL_AUDIT', common+['--stage', 'real'], root/'contracts/REAL_COMPLETE.json')
        stage('REFERENCE_COMPARISON', [PYTHON, '-B', str(audit), '--root', str(root), '--stage', 'compare'],
              root/'audit/REFERENCE_COMPARISON_COMPLETE.json')
        boot = root/'uncertainty'
        stage('SYSTEM_BOOTSTRAP', [PYTHON, '-B', str(bootstrap), '--experiment-root', str(root), '--root', str(boot),
              '--methods', 'NODUP-DIRECT-REPLAY', 'PATH875-ARCHIVED', 'SHARED-PROFILE-SINGLE-WF',
              'SHARED-PROFILE-REJECT-ONLY', '--pair-draws', '2000', '--query-draws', '10000', '--workers', '6'],
              boot/'contracts/COMPLETE.json')
        out = root/'uncertainty_summary'
        stage('BOOTSTRAP_SUMMARY', [PYTHON, '-B', str(summary), '--bootstrap-root', str(boot), '--root', str(out)],
              out/'contracts/SUMMARY_CONTRACT.json')
        write(root/'contracts/FROZEN_COMPUTATIONS_COMPLETE.json', {'UTC': utc(),
            'status': 'HOLD_FOR_AUTHOR_REVIEW_NO_ADOPTION_NO_OVERWRITE', 'goal_achieved': False,
            'next': 'Review complete guard tables, produce report/figures and verify compact archive; no automatic upgrade.'})
    except Exception as exc:
        write(root/'logs/COMPLETION_DRIVER_FAIL.json', {'UTC': utc(), 'exception': repr(exc),
            'no_reselection_or_downstream_continuation': True})
        raise


if __name__ == '__main__':
    main()
