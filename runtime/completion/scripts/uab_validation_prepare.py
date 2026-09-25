"""Materialize validation-only sky inputs while scoring integration is audited."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    script = a.out/'scripts/uab_completion.py'
    def task(label, args):
        log = a.out/'logs'/f'{label}_{time.time_ns()}.log'
        command = [sys.executable, '-B', '-u', str(script), '--root', str(a.root), '--out', str(a.out), *args]
        with log.open('x') as f:
            proc = subprocess.Popen(command, stdout=f, stderr=subprocess.STDOUT)
            record = dict(stage=label, pid=proc.pid, log=str(log), utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                          complete_results=False, test_opened=False, continuation_directory=str(a.out))
            (a.out/'contracts/VALIDATION_PREPARATION_STATUS.json').write_text(json.dumps(record, indent=2))
            code = proc.wait()
        (a.out/'contracts'/f'{label}_EXIT.json').write_text(json.dumps(dict(**record, exit_code=code), indent=2))
        if code:
            raise RuntimeError(f'{label} failed; {log}')
    task('INPUT_AUDIT', ['--stage', 'audit'])
    for run in ('O3', 'O4a', 'O4b'):
        for arm in ('A_NEUTRAL', 'B_CUE'):
            base = ['--run', run, '--arm', arm, '--split', 'validation']
            task(f'MAPS_{run}_{arm}', ['--stage', 'maps', '--workers', '6', *base])
    (a.out/'contracts/VALIDATION_PREPARATION_STATUS.json').write_text(json.dumps(dict(
        stage='VALIDATION_NATIVE_MAPS_COMPLETE', complete_results=False, test_opened=False), indent=2))


if __name__ == '__main__':
    main()
