#!/usr/bin/env python3
"""Complete R51 in a new directory; retain two existing diagnostic columns."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np
import pandas as pd

P = Path('/root/autodl-tmp/gw-catalog')
S = P / 'scripts/experiments'
sys.path.insert(0, str(S))
import mcwf_shared_profile_catalog_20260909 as app


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)


def snapshot(source, root, pilot):
    if root.exists():
        raise RuntimeError('Independent recovery output required')
    if not (source / 'contracts/REAL_MEASUREMENT_COMPLETE.json').exists():
        raise RuntimeError('Reuse only fully completed physical measurements')
    failure = json.loads((source / 'logs/COMPLETION_DRIVER_FAIL.json').read_text())
    if 'REAL_AUDIT' not in failure['exception']:
        raise RuntimeError('Unexpected original failure; audit before recovery')
    rows = []
    for path in sorted(source.rglob('*')):
        if not path.is_file():
            continue
        relative = path.relative_to(source)
        if relative.parts[0] == 'results' and any(v in ('real', 'consensus') for v in relative.parts):
            continue
        if relative.parts[0] == 'logs' or relative.as_posix() == 'audit/TRANSITIVE_RUNTIME_FINAL_CHECK.json':
            target = root / 'source_failure_snapshot' / relative
        else:
            target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        before = sha(path)
        shutil.copy2(path, target)
        if before != sha(target):
            raise RuntimeError('Recovery snapshot mismatch')
        rows.append({'source': str(path), 'destination': str(target), 'sha256': before})
    (root / 'logs').mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(root / 'manifest/RECOVERY_SOURCE_SHA256.csv',
        index=False, encoding='utf-8-sig')
    write(root / 'contracts/DIAGNOSTIC_EXPORT_RECOVERY.json', {
        'UTC': utc(), 'id': 'MCWF-SHARED-PROFILE-51B-DIAGNOSTIC-COMPATIBILITY',
        'source_result': str(source), 'pilot_root': str(pilot),
        'original_failure_preserved': True,
        'cause': 'The archived numerical export dropped sky_j50/sky_j90; unchanged consensus reducer requires these descriptive columns.',
        'only_repair': 'Carry the existing finite sky_j50/sky_j90 values from frozen NODUP input into output. No map regeneration, no invented values.',
        'scientific_scores_or_selection_changed': False,
        'no_retraining_or_remeasurement': True,
        'partial_real_outputs_not_overwritten': True,
        'partial_real_outputs_will_be_replayed_exactly': True,
        'goal_achieved': False,
        'status': 'HOLD_FOR_AUTHOR_REVIEW_NO_ADOPTION_NO_OVERWRITE'})
    shutil.copy2(__file__, root / 'scripts/shared_profile_completion_recovery.py')
    print('RECOVERY_SNAPSHOT_COMPLETE', len(rows), flush=True)


def repair_real(root, pilot, source):
    app.ROOT, app.PILOT = root, pilot
    original_export = app.export

    def export(frame, z, weights, method):
        result = original_export(frame, z, weights, method)
        frozen_scores = result[['waveform_score', 'time_score', 'sky_raw_log_bf', 'final_score']].copy()
        for name in ('sky_j50', 'sky_j90'):
            if name not in frame or not np.isfinite(frame[name]).all():
                raise RuntimeError('Existing sky diagnostic missing or invalid: ' + name)
            result[name] = frame[name].to_numpy(copy=True)
        if not result[frozen_scores.columns].equals(frozen_scores):
            raise RuntimeError('Diagnostic export changed numerical scores')
        return result

    app.export = export
    app.evaluate('real')
    checks = []
    for path in sorted((source / 'results').glob('*/gwtc*/seed_*/real/*_all_pairs.parquet')):
        relative = path.relative_to(source)
        before = pd.read_parquet(path).sort_values(['idx_i', 'idx_j']).reset_index(drop=True)
        after = pd.read_parquet(root / relative).sort_values(['idx_i', 'idx_j']).reset_index(drop=True)
        same = before.equals(after[before.columns])
        checks.append({'source': str(path), 'recovered': str(root / relative),
            'pairs': len(before), 'all_preexisting_columns_exact': bool(same)})
        if not same:
            raise RuntimeError('Recovery changed an existing real numerical result')
    if not checks:
        raise RuntimeError('Expected partial real scores for exact replay verification')
    pd.DataFrame(checks).to_csv(root / 'audit/PARTIAL_REAL_EXACT_RECOVERY.csv',
        index=False, encoding='utf-8-sig')
    write(root / 'audit/DIAGNOSTIC_EXPORT_RECOVERY_PASS.json', {
        'UTC': utc(), 'passed': True, 'partial_files_exact': len(checks),
        'score_rank_scope_and_existing_PE_fields_exact': True,
        'no_reselection': True})
    print('DIAGNOSTIC_RECOVERY_REAL_PASS', len(checks), flush=True)


def complete(source, root, pilot):
    snapshot(source, root, pilot)
    files = {r['source']: r['sha256'] for r in json.loads(
        (source / 'manifest/TRANSITIVE_RUNTIME_SHA256.json').read_text())['files']}
    files[str(Path(__file__).resolve())] = sha(Path(__file__))

    def verify():
        changed = [name for name, digest in files.items() if sha(Path(name)) != digest]
        if changed:
            raise RuntimeError('Recovery runtime hash changed: ' + repr(changed))

    def stage(label, command, receipt):
        verify()
        start = time.monotonic()
        print('BEGIN_RECOVERY_STAGE', label, utc(), flush=True)
        with (root / f'logs/RECOVERY_{label}.stdout.log').open('x') as stream:
            process = subprocess.Popen(command, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, bufsize=1)
            with process:
                for line in process.stdout:
                    stream.write(line)
                    stream.flush()
                    print(label, line.rstrip(), flush=True)
        write(root / f'logs/RECOVERY_{label}_RECEIPT.json', {
            'UTC': utc(), 'command': command, 'returncode': process.returncode,
            'wall_seconds': time.monotonic() - start, 'receipt_exists': receipt.exists()})
        verify()
        if process.returncode or not receipt.exists():
            raise RuntimeError('Recovery stage failed: ' + label)

    py = ['/root/miniconda3/bin/python', '-B']
    try:
        stage('REAL_AUDIT', py + [str(Path(__file__)), '--source-root', str(source),
            '--root', str(root), '--pilot-root', str(pilot), '--stage', 'real'],
            root / 'audit/DIAGNOSTIC_EXPORT_RECOVERY_PASS.json')
        stage('REFERENCE_COMPARISON', py + [str(S / 'mcwf_shared_profile_catalog_audit_20260909.py'),
            '--root', str(root), '--stage', 'compare'], root / 'audit/REFERENCE_COMPARISON_COMPLETE.json')
        boot = root / 'uncertainty'
        stage('SYSTEM_BOOTSTRAP', py + [str(S / 'mcwf_shared_profile_system_bootstrap_20260909.py'),
            '--experiment-root', str(root), '--root', str(boot), '--methods',
            'NODUP-DIRECT-REPLAY', 'PATH875-ARCHIVED', 'SHARED-PROFILE-SINGLE-WF',
            'SHARED-PROFILE-REJECT-ONLY', '--pair-draws', '2000', '--query-draws', '10000', '--workers', '6'],
            boot / 'contracts/COMPLETE.json')
        out = root / 'uncertainty_summary'
        stage('BOOTSTRAP_SUMMARY', py + [str(S / 'mcwf_shared_profile_bootstrap_summary_20260909.py'),
            '--bootstrap-root', str(boot), '--root', str(out)], out / 'contracts/SUMMARY_CONTRACT.json')
        verify()
        write(root / 'audit/TRANSITIVE_RUNTIME_FINAL_CHECK.json', {
            'UTC': utc(), 'files': len(files), 'hash_failures': 0,
            'computations_complete': True, 'source': 'exact runtime snapshots checked before and after every recovery stage'})
        write(root / 'contracts/FROZEN_COMPUTATIONS_COMPLETE.json', {
            'UTC': utc(), 'source_result': str(source), 'diagnostic_export_repaired': True,
            'scientific_scores_or_selection_changed': False, 'goal_achieved': False,
            'status': 'HOLD_FOR_AUTHOR_REVIEW_NO_ADOPTION_NO_OVERWRITE'})
    except Exception as exc:
        write(root / 'logs/RECOVERY_DRIVER_FAIL.json', {'UTC': utc(), 'exception': repr(exc)})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--pilot-root', type=Path, required=True)
    parser.add_argument('--stage', choices=('complete', 'real'), default='complete')
    args = parser.parse_args()
    if args.stage == 'real':
        repair_real(args.root, args.pilot_root, args.source_root)
    else:
        complete(args.source_root, args.root, args.pilot_root)
