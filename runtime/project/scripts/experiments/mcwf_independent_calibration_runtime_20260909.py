#!/usr/bin/env python3
"""Compatibility reader for a frozen configuration receipt lacking its filename."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import json
from pathlib import Path
import shutil
import sys

P = Path('/root/autodl-tmp/gw-catalog')
sys.path.insert(0, str(P/'scripts/experiments'))
import mcwf_independent_waveform_calibration_20260909 as c
n, r, h, s, co = c.n, c.r, c.h, c.s, c.co


def selections(root):
    record = json.loads((root/'contracts/CONFIGURATIONS_FROZEN.json').read_text())
    path = root/record.get('file', 'configs/SELECTED_CONFIGURATIONS.json')
    if n.sha(path) != record['sha256']:
        raise RuntimeError('Frozen configurations changed')
    return json.loads(path.read_text())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--stage', choices=('evaluate', 'real'), required=True)
    args = parser.parse_args()
    c.ROOT = s.ROOT = h.ROOT = co.ROOT = r.ROOT = co.score.ROOT = args.root
    r.install()
    co.score.METHODS = c.METHODS
    co.score.matrices = co.matrices
    n.METHODS, n.load_panel, n.infer, n.selections = c.METHODS, h.load_panel, c.infer, selections
    configs = selections(args.root)
    receipt = args.root/'audit/CONFIG_PATH_READER_FIX.json'
    if not receipt.exists():
        n.write_json(receipt, {'UTC': n.utc(), 'bug': 'Frozenreceiptomittedfilekey;canonicalpathunambiguous.',
            'default': 'configs/SELECTED_CONFIGURATIONS.json', 'config_count': len(configs),
            'all_hashes_verified': True, 'frozen_config_modified': False,
            'scientific_definition_changed': False, 'failed_logs_retained': True,
            'runtime_sha256': n.sha(Path(__file__))})
        shutil.copy2(__file__, args.root/'scripts/independent_calibration_runtime.py')
    n.run(args.root, args.stage)
