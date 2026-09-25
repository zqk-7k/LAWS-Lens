#!/usr/bin/env python3
"""Read the frozen round14 configuration with its omitted fixed relative path."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import json
from pathlib import Path
import shutil
import sys
import pandas as pd

P = Path('/root/autodl-tmp/gw-catalog')
sys.path.insert(0, str(P / 'scripts/experiments'))
import mcwf_profile_single_waveform_20260909 as s
h, co, r, n = s.h, s.co, s.r, s.n


def selections(root):
    freeze = json.loads((root / 'contracts/CONFIGURATIONS_FROZEN.json').read_text())
    path = root / freeze.get('file', 'configs/SELECTED_CONFIGURATIONS.json')
    if n.sha(path) != freeze['sha256']:
        raise RuntimeError('Frozen configurations changed')
    return json.loads(path.read_text())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--stage', choices=('evaluate', 'real'), required=True)
    args = parser.parse_args()
    s.ROOT = h.ROOT = co.ROOT = r.ROOT = co.score.ROOT = args.root
    r.install()
    co.score.METHODS = s.METHODS
    co.score.matrices = co.matrices
    n.METHODS, n.load_panel, n.infer, n.selections = s.METHODS, h.load_panel, s.infer, selections
    original_export, original_consensus = n.public_frame, n.dev.BASE.consensus_real
    def export(frame, z, weights, method):
        out = original_export(frame, z, weights, method)
        for name in h.FEATURES:
            out[name] = frame[name].to_numpy()
        return out
    def consensus(items, method):
        out = original_consensus(items, method)
        extra = pd.concat(items).groupby('pair_key')[list(h.FEATURES)].mean().add_suffix('_mean').reset_index()
        return out.merge(extra, on='pair_key', validate='one_to_one')
    n.public_frame, n.dev.BASE.consensus_real = export, consensus
    configs = selections(args.root)
    receipt = args.root / 'audit/CONFIG_PATH_READER_FIX.json'
    if not receipt.exists():
        n.write_json(receipt, {'UTC': n.utc(), 'bug': 'freeze receipt omitted file key; fixed relative filename is unambiguous',
            'compatibility_default': 'configs/SELECTED_CONFIGURATIONS.json', 'config_count': len(configs),
            'config_sha256_verified': True, 'frozen_file_modified': False, 'scientific_definition_changed': False,
            'failed_logs_retained': True, 'runtime_sha256': n.sha(Path(__file__))})
        shutil.copy2(__file__, args.root / 'scripts/single_waveform_runtime.py')
    n.run(args.root, args.stage)
