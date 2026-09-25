"""Overlap CPU-only validation localization with later-run GPU training.

The existing map lock and completion hashes are authoritative. No test data or
real candidates are opened here, and no scoring/training settings are changed.
"""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

import unified_ab as u


def ready(root, run):
    for seed in u.SEEDS:
        for stage in ('SHORT', 'rnc', 'ordered', 'multirate', 'conditional'):
            path = root/'contracts/tasks'/f'{stage}_{run}_C_PHYSICAL_{seed}.json'
            if not path.exists() or json.loads(path.read_text())['exit_code'] != 0:
                return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    root = parser.parse_args().root
    status = root/'contracts/VALIDATION_PREFETCH_STATUS.json'
    with (root/'contracts/VALIDATION_PREFETCH.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            u.verify(root)
            for run in u.RUNS:
                while not ready(root, run):
                    state = json.loads((root/'RUN_STATUS.json').read_text())
                    if state['state'].startswith('HOLD'):
                        raise RuntimeError('Main controller is paused: '+state['state'])
                    u.write(status, dict(state='WAIT_TRAINING', run=run, utc=u.now()))
                    time.sleep(30)
                log = root/'logs'/f'PREFETCH_MAPS_{run}_validation_{time.time_ns()}.log'
                command = [sys.executable, '-B', '-u', str(root/'scripts/c_finish.py'),
                           '--root', str(root), '--action', 'maps', '--run', run,
                           '--split', 'validation']
                with log.open('x') as stream:
                    child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                             cwd=u.P, env=os.environ.copy())
                    u.write(status, dict(state='VALIDATION_MAPS', run=run, child_pid=child.pid,
                                         log=str(log), utc=u.now()))
                    code = child.wait()
                if code:
                    raise RuntimeError('Validation localization failed: '+str(log))
            u.write(status, dict(state='COMPLETE', utc=u.now(), validation_only=True))
        except Exception:
            u.write(status, dict(state='HOLD_PREFETCH_ERROR', utc=u.now(), error=traceback.format_exc()))
            raise


if __name__ == '__main__':
    main()
