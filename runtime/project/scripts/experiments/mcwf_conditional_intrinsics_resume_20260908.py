#!/usr/bin/env python3
"""Resume after a missing receipt field, without editing frozen selections."""
import argparse
import json
from pathlib import Path
import shutil
import sys
sys.path.insert(0, '/root/autodl-tmp/gw-catalog/scripts/experiments')
import mcwf_conditional_intrinsics_evaluate_20260908 as implementation
ev, dev, t, cf = implementation.ev, implementation.dev, implementation.t, implementation.cf


def load_selections(root):
    receipt = json.loads((root / 'contracts/INTEGRATION_FROZEN.json').read_text())
    path = root / receipt.get('file', 'calibration/SELECTED.json')
    if dev.sha(path) != receipt['sha256']:
        raise RuntimeError('Frozen selection checksum mismatch')
    out = json.loads(path.read_text())
    for dep in t.DEPS:
        for seed in t.SEEDS:
            out.append({'deployment': dep, 'seed': seed, 'method': 'OMC',
                        'weights': cf.frozen_weights(dep, seed).tolist(), 'gamma': 0., 'beta': 0.})
    return out


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--stage', choices=['evaluate', 'real', 'assess'], required=True)
    a = p.parse_args()
    evidence = a.root / 'contracts/RECEIPT_FIELD_COMPATIBILITY_FIX.json'
    if not evidence.exists():
        dev.json_write(evidence, {'error': 'Missing file key in integration receipt,raised before held-out scoring.',
                       'fix': 'Use fixed calibration/SELECTED.json path and verify original frozen SHA-256.',
                       'selection_changed': False, 'training_changed': False, 'score_formula_changed': False,
                       'original_receipt_sha256': dev.sha(a.root / 'contracts/INTEGRATION_FROZEN.json'),
                       'code_sha256': dev.sha(Path(__file__))})
        shutil.copy2(__file__, a.root / 'scripts/conditional_intrinsics_resume.py')
    ev.load_selections = load_selections
    ev.scored = implementation.scored
    getattr(ev, a.stage)(a.root)
