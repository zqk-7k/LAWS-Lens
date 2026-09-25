#!/usr/bin/env python3
"""Resume-safe finite R74 -> R75 evaluation driver. Never alters science."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

P = Path('/root/autodl-tmp/gw-catalog')
PY = '/root/miniconda3/bin/python'
SCRIPTS = P / 'scripts/experiments'
STATUS = 'HOLD_FOR_AUTHOR_REVIEW_NO_ADOPTION_NO_OVERWRITE'


def sha(file):
    h = hashlib.sha256()
    with Path(file).open('rb') as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def write(path, obj):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(obj, stream, indent=2, ensure_ascii=False)


def now():
    return datetime.now(timezone.utc).isoformat()


def main(root, physical, pid, output):
    scorer = SCRIPTS / 'mcwf_shared_mass_catalog_scoring_20260910.py'
    delivery = SCRIPTS / 'mcwf_shared_mass_catalog_delivery_20260910.py'
    bootstrap = SCRIPTS / 'mcwf_shared_profile_system_bootstrap_20260909.py'
    contract = root / 'contracts/COMPLETION_DRIVER_FROZEN.json'
    expected = {str(p): sha(p) for p in (scorer, delivery, bootstrap, Path(__file__))}
    if contract.exists():
        if json.loads(contract.read_text())['scripts'] != expected:
            raise RuntimeError('Frozen automation code changed')
    else:
        write(contract, {'UTC': now(), 'scripts': expected, 'physical_root': str(physical),
            'process_pid': pid, 'status': STATUS, 'no_threshold_or_model_changes': True,
            'order': ['R74 measurement PASS', 'R75 injection', 'paired system bootstrap',
                      'if ALL injection guards PASS: real measurement and real audit', 'report all outcomes', 'verified archive'],
            'on_failure': 'Preserve logs, do not auto-retry numerical failures, do not rank real if injection guard FAIL.'})
    log = root / 'logs/COMPLETION_DRIVER_EVENTS.jsonl'
    def event(stage, **kwargs):
        with log.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps({'UTC': now(), 'stage': stage, **kwargs}) + '\n')
    def execute(name, command, marker):
        if marker.exists():
            event(name, state='ALREADY_COMPLETE')
            return
        for file, digest in expected.items():
            if sha(file) != digest:
                raise RuntimeError('Automation runtime changed')
        event(name, state='START', argv=command)
        with (root / f'logs/{name}.stdout.log').open('x') as out, (root / f'logs/{name}.stderr.log').open('x') as err:
            result = subprocess.run(command, cwd=P, stdout=out, stderr=err)
        event(name, state='EXIT', returncode=result.returncode)
        if result.returncode or not marker.exists():
            raise RuntimeError(name + ' failed; preserve logs and stop')
    gate = physical / 'contracts/MEASUREMENT_GATE.json'
    event('WAIT_R74', state='START')
    last = None
    while not gate.exists():
        count = sum(1 for _ in physical.glob('pairs/*/*/*.json'))
        if count != last:
            event('WAIT_R74', measured=count, expected=5241)
            last = count
        proc = Path(f'/proc/{pid}/stat')
        if not proc.exists() or proc.read_text().split(') ', 1)[1].split()[0] == 'Z':
            time.sleep(10)
            if not gate.exists():
                raise RuntimeError('R74 process ended without a measurement gate')
        time.sleep(120)
    if json.loads(gate.read_text())['gate'] != 'PASS':
        raise RuntimeError('R74 numerical gate FAIL')
    execute('r75_evaluate', [PY, '-B', str(scorer), '--root', str(root), '--stage', 'evaluate'], root / 'contracts/INJECTION_GUARD.json')
    bootroot = root / 'uncertainty'
    execute('r75_bootstrap', [PY, '-B', str(bootstrap), '--experiment-root', str(root), '--root', str(bootroot),
        '--methods', 'NODUP-DIRECT-REPLAY', 'R73-FROZEN-CONDITIONAL-WF', '--pair-draws', '2000', '--query-draws', '10000', '--workers', '6'],
        bootroot / 'contracts/COMPLETE.json')
    inj = json.loads((root / 'contracts/INJECTION_GUARD.json').read_text())
    if inj['gate'] == 'PASS':
        execute('r75_measure_real', [PY, '-B', str(scorer), '--root', str(root), '--stage', 'measure-real', '--workers', '24'], root / 'contracts/REAL_MEASUREMENT_COMPLETE.json')
        execute('r75_real', [PY, '-B', str(scorer), '--root', str(root), '--stage', 'real'], root / 'contracts/REAL_COMPLETE.json')
    else:
        event('r75_real', state='BLOCKED_BY_INJECTION_GATE', failures=inj['failures'])
    execute('r75_final_audit', [PY, '-B', str(delivery), '--root', str(root), '--stage', 'audit'], root / 'contracts/FINAL_AUDIT.json')
    execute('r75_report', [PY, '-B', str(delivery), '--root', str(root), '--stage', 'report'], root / 'reports/R72_R75_FULL_REPORT_CN.md')
    # Record completion of computational stages before checksumming stable outputs.
    write(root / 'contracts/FROZEN_COMPUTATIONS_COMPLETE.json', {'UTC': now(), 'status': STATUS,
        'goal_achieved': json.loads((root / 'contracts/FINAL_AUDIT.json').read_text())['goal_achieved'], 'scientific_rules_unchanged': True})
    execute('r75_package', [PY, '-B', str(delivery), '--root', str(root), '--stage', 'package', '--output', str(output)], root / 'contracts/DELIVERY_COMPLETE.json')
    event('END', state=STATUS)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--physical', type=Path, required=True)
    p.add_argument('--pid', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    try:
        main(args.root, args.physical, args.pid, args.output)
    except Exception as exc:
        path = args.root / ('logs/COMPLETION_FAILURE_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '.json')
        write(path, {'UTC': now(), 'error': repr(exc), 'status': STATUS, 'goal_achieved': False})
        raise
