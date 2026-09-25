#!/usr/bin/env python3
"""Record-only compatibility wrapper; leave the failed R61 preflight intact."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import numpy as np

P = Path('/root/autodl-tmp/gw-catalog')
sys.path.insert(0, str(P / 'scripts/experiments'))
import mcwf_shared_reference_regularization_20260910 as app
ORIGINAL_UNIT = app.unit


def json_unit(reference):
    return [{k: v.item() if isinstance(v, np.generic) else v for k, v in r.items()}
            for r in ORIGINAL_UNIT(reference)]


def main(args):
    app.unit = json_unit
    receipt = args.root / 'contracts/SERIALIZATION_COMPATIBILITY.json'
    if args.stage == 'freeze':
        if (args.failed_root / 'contracts/START_FREEZE.json').exists():
            raise RuntimeError('Expected serialization failure before complete protocol freeze')
        if list((args.failed_root / 'tables').glob('*')):
            raise RuntimeError('Expected no calibration outcomes before serialization repair')
        inputs = [{'path': str(p), 'sha256': app.n.sha(p)} for p in args.failed_root.rglob('*') if p.is_file()]
        app.freeze(args.root, args.expansion_root, args.reference_root)
        app.common.write_once(receipt, {'UTC': app.n.utc(), 'wrapper_sha256': app.n.sha(Path(__file__)),
            'parent_runtime_sha256': app.n.sha(Path(app.__file__)),
            'failed_preflight_root': str(args.failed_root), 'preserved_failed_files': inputs,
            'observed_error': 'TypeError: Object of type bool is not JSON serializable (numpy.bool_)',
            'change': 'Convert NumPy scalar types to Python scalars when serializing unit-test outputs only.',
            'score_fit_selection_or_physics_changed': False, 'old_directory_unchanged': True})
        shutil.copy2(__file__, args.root / 'scripts/shared_reference_serialization.py')
    else:
        record = json.loads(receipt.read_text())
        if record['wrapper_sha256'] != app.n.sha(Path(__file__)):
            raise RuntimeError('Serialization wrapper changed')
        for row in record['preserved_failed_files']:
            if app.n.sha(Path(row['path'])) != row['sha256']:
                raise RuntimeError('Original failed preflight changed')
        app.fit(args.root)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('root', 'failed-root', 'expansion-root', 'reference-root'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--stage', choices=('freeze', 'fit'), required=True)
    main(p.parse_args())
