"""Prepare all validation waveform scores, without opening held-out data."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def run(root, out):
    root, out = Path(root), Path(out)
    sys.path.insert(0, str(out/'scripts'))
    import uab_completion as c
    c.initialize(root, out)
    status = out/'contracts/INFERENCE_PREPARATION_STATUS.json'
    try:
        for deployment in c.U.RUNS:
            for arm in c.U.ARMS:
                for seed in c.U.SEEDS:
                    for split in ('development', 'validation'):
                        name = f'INFER_{deployment}_{arm}_{seed}_{split}'
                        receipt = out/'contracts'/f'{name}_EXIT.json'
                        if receipt.exists() and json.loads(receipt.read_text())['exit_code'] == 0:
                            continue
                        log = out/'logs'/f'{name}_{time.time_ns()}.log'
                        command = [sys.executable, '-B', str(out/'scripts/uab_completion.py'),
                                   '--root', str(root), '--out', str(out), '--stage', 'infer',
                                   '--run', deployment, '--arm', arm, '--seed', str(seed), '--split', split]
                        with log.open('w') as stream:
                            child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
                            c.write(status, dict(stage=name, pid=child.pid, log=str(log), utc=c.U.now(), test_opened=False))
                            code = child.wait()
                        c.write(receipt, dict(exit_code=code, command=command, log=str(log), utc=c.U.now()))
                        if code:
                            raise RuntimeError(name+' failed: '+str(log))
        c.write(status, dict(stage='ALL_VALIDATION_WAVEFORM_INFERENCE_COMPLETE', test_opened=False, utc=c.U.now()))
    except Exception as error:
        c.write(status, dict(stage='HOLD_INFERENCE_ERROR', error=repr(error), utc=c.U.now(), test_opened=False))
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    run(a.root, a.out)
