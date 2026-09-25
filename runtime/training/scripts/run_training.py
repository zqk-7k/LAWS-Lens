#!/usr/bin/env python3
"""Durable, bounded two-arm training; no test opening or real-rank tuning."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import unified_ab as u


def task(root, label, script, args):
    u.guard(root)
    log = root/'logs'/f'{label}_{time.time_ns()}.log'
    cmd = [sys.executable, '-B', '-u', str(root/'scripts'/script), '--root', str(root), *map(str, args)]
    record = {'utc': u.now(), 'state': label, 'log': str(log), 'command': cmd, 'complete_results': False}
    with log.open('x') as f:
        env = {**os.environ, 'PYTHONPATH': str(u.P)+os.pathsep+os.environ.get('PYTHONPATH', '')}
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
        record['pid'] = proc.pid
        u.write(root/'RUN_STATUS.json', record)
        rc = proc.wait()
    u.write(root/'contracts/tasks'/f'{label}_{time.time_ns()}.json', {**record, 'exit_code': rc, 'finished_utc': u.now()})
    if rc:
        raise RuntimeError(f'{label}: exit {rc}; {log}')


def run(root):
    u.verify(root)
    if not (root/'contracts/DATA_UNIT_TESTS_PASS.json').exists():
        raise RuntimeError('Data unit tests missing')
    if not (root/'contracts/BAYESTAR_PILOT_PASS.json').exists():
        raise RuntimeError('End-to-end BAYESTAR pilot missing')
    for run in u.RUNS:
        for split in ('validation', 'train'):
            task(root, f'MAIN_{run}_{split}', 'unified_ab.py',
                 ['--stage', 'generate', '--run', run, '--role', 'main', '--split', split, '--workers', 8])
        for arm in u.ARMS:
            for seed in u.SEEDS:
                task(root, f'SHORT_{run}_{arm}_{seed}', 'unified_ab.py',
                     ['--stage', 'short', '--run', run, '--arm', arm, '--seed', seed])
        for split in ('validation', 'train'):
            task(root, f'AUX_{run}_{split}', 'unified_ab.py',
                 ['--stage', 'generate', '--run', run, '--role', 'aux', '--split', split, '--workers', 8])
        for arm in u.ARMS:
            base = ['--run', run, '--arm', arm]
            for split in ('validation', 'train'):
                task(root, f'FEATURES_{run}_{arm}_{split}', 'train_components.py',
                     base+['--stage', 'features', '--split', split])
            task(root, f'RNC_PREPARE_{run}_{arm}', 'train_components.py', base+['--stage', 'rnc-prepare'])
            for seed in u.SEEDS:
                for stage in ('rnc', 'ordered', 'multirate', 'conditional'):
                    task(root, f'{stage}_{run}_{arm}_{seed}', 'train_components.py',
                         base+['--stage', stage, '--seed', seed])
    u.write(root/'RUN_STATUS.json', {'utc': u.now(), 'state': 'ALL_WAVEFORM_COMPONENTS_TRAINED_SCORE_AUDIT_PENDING',
                                    'complete_results': False, 'locked_test_opened': False,
                                    'next': 'Audit common calibration/inference adapters, freeze scoring, then evaluate both experiments',
                                    'not_a_full_NEW_SCORE_ONLY_delivery': True})


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True); args = p.parse_args()
    with (args.root/'contracts/TRAINING_CONTROLLER.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            run(args.root)
        except Exception:
            u.write(args.root/'RUN_STATUS.json', {'utc': u.now(), 'state': 'HOLD_TRAINING_FAILURE',
                    'complete_results': False, 'traceback': traceback.format_exc()})
            raise
