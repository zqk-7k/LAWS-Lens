#!/usr/bin/env python3
"""Verify all frozen scientific inputs again after R51 computations."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
import pandas as pd


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main(root, pilot):
    if not (root / 'contracts/FROZEN_COMPUTATIONS_COMPLETE.json').exists():
        raise RuntimeError('Only audit completed calculations')
    output = root / 'audit/FINAL_FROZEN_INPUT_INTEGRITY.json'
    if output.exists():
        raise RuntimeError('Do not overwrite final input audit')
    manifests = [root / 'manifest/INPUT_SHA256.csv',
                 root / 'manifest/REFERENCE_INPUT_SHA256.csv',
                 pilot / 'manifest/INPUT_SHA256.csv']
    expected, origins = {}, {}
    for manifest in manifests:
        for row in pd.read_csv(manifest, encoding='utf-8-sig').itertuples():
            path = str(row.path)
            if path in expected and expected[path] != row.sha256:
                raise RuntimeError('Conflicting frozen input hash: ' + path)
            expected[path] = row.sha256
            origins.setdefault(path, []).append(str(manifest))
    actual = []
    for name, required in expected.items():
        path = Path(name)
        found = sha(path) if path.exists() else None
        actual.append({'path': name, 'expected_sha256': required,
            'actual_sha256': found, 'unchanged': found == required,
            'manifests': '|'.join(origins[name])})
    for directory in (root, pilot):
        freeze = json.loads((directory / 'contracts/START_FREEZE.json').read_text())
        for field, relative in [('contract_sha256', 'contracts/ANALYSIS_CONTRACT.json'),
                                ('pair_plan_sha256', 'contracts/PAIR_PLAN.parquet')]:
            path = directory / relative
            found = sha(path)
            actual.append({'path': str(path), 'expected_sha256': freeze[field],
                'actual_sha256': found, 'unchanged': found == freeze[field],
                'manifests': str(directory / 'contracts/START_FREEZE.json')})
    config = json.loads((root / 'contracts/CONFIGURATIONS_FROZEN.json').read_text())
    path = root / config['file']
    actual.append({'path': str(path), 'expected_sha256': config['sha256'],
        'actual_sha256': sha(path), 'unchanged': sha(path) == config['sha256'],
        'manifests': str(root / 'contracts/CONFIGURATIONS_FROZEN.json')})
    pd.DataFrame(actual).to_csv(root / 'audit/FINAL_FROZEN_INPUT_HASHES.csv',
        index=False, encoding='utf-8-sig')
    pair_plan = pd.read_parquet(root / 'contracts/PAIR_PLAN.parquet')
    pair_checks = []
    for scope in ('injection', 'real'):
        data = pd.read_parquet(root / f'tables/{scope}_shared_profile_pairs.parquet')
        required = pair_plan[pair_plan.scope == scope]
        match = (len(data) == len(required) and not data.pair_id.duplicated().any() and
                 set(data.pair_id) == set(required.pair_id))
        finite = bool(np.isfinite(data.deficit).all() and (data.deficit >= 0).all())
        replay = float(data.profile_replay_max_difference.max())
        pair_checks.append({'scope': scope, 'pairs': len(data),
            'expected_pairs': len(required), 'identity_complete': bool(match),
            'all_measurements_complete': bool((data.status == 'COMPLETE').all()),
            'finite_nonnegative_deficit': finite,
            'single_profile_replay_max_difference': replay,
            'profile_replay_pass': replay <= 1e-6})
    passed = (all(row['unchanged'] for row in actual) and
        all(row['identity_complete'] and row['all_measurements_complete'] and
            row['finite_nonnegative_deficit'] and row['profile_replay_pass']
            for row in pair_checks))
    result = {'UTC': datetime.now(timezone.utc).isoformat(), 'passed': bool(passed),
        'unique_input_paths': len(expected), 'hash_checks': len(actual),
        'changed_hashes': sum(not row['unchanged'] for row in actual),
        'pair_checks': pair_checks, 'goal_achieved': False,
        'not_a_scientific_performance_gate': True}
    with output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    shutil.copy2(__file__, root / 'scripts/shared_profile_final_integrity.py')
    print(json.dumps(result), flush=True)
    if not passed:
        raise RuntimeError('Final frozen-input integrity audit failed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--pilot-root', type=Path, required=True)
    args = parser.parse_args()
    main(args.root, args.pilot_root)
