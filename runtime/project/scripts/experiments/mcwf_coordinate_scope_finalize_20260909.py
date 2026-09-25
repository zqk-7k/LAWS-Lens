#!/usr/bin/env python3
"""Complete a duplicate-module snapshot failure without rewriting results."""
import argparse
import json
from pathlib import Path
import re
import shutil
import sys

import pandas as pd

P = Path('/root/autodl-tmp/gw-catalog')
sys.path.insert(0, str(P / 'scripts/experiments'))
import mcwf_shared_deficit_coordinate_20260909 as c
n = c.n


def main(root, scope, reference):
    c.check(root)
    if (root / 'contracts/FINAL_EXPORT_COMPLETE.json').exists():
        raise RuntimeError('Already finalized')
    required = ['reports/R59_R60_COMPLETE_REPORT_CN.md', 'tables/CONDITIONAL_SOURCE_BOOTSTRAP_CI.csv',
        'tables/SELECTED_COEFFICIENTS_PER_SEED.csv', 'audit/FINAL_INPUT_HASH_CHECKS.csv',
        'figures/R59_COORDINATE_COMPARISON.png', 'figures/R59_COORDINATE_COMPARISON.pdf']
    for name in required:
        if not (root / name).exists():
            raise RuntimeError('Missing completed scientific/report artifact: ' + name)
    existing = [{'path': str(p), 'sha256': n.sha(p)} for r in (root, scope)
                for p in r.rglob('*') if p.is_file()]
    failure = {'UTC_recorded': n.utc(), 'stage': 'report runtime-dependency snapshot',
        'failed_command_script': 'mcwf_coordinate_scope_report_20260909.py',
        'observed_exit_code': 1, 'observed_exception': 'RuntimeError: Dependency snapshot already exists',
        'scientific_products_complete_before_failure': required,
        'cause': 'Multiple imported module names resolve to the same physical project file; the snapshot loop required a new destination each time.',
        'recovery': 'Deduplicate resolved source paths and verify any existing snapshot is identical. Do not rerun calibration, rewrite reports or change scores.'}
    c.write_once(root / 'logs/REPORT_EXPORT_FAILURE_RETAINED.json', failure)
    checks = []
    for r in (root, scope):
        for row in pd.read_csv(r / 'manifest/INPUT_SHA256.csv').itertuples():
            after = n.sha(Path(row.path))
            if after != row.sha256:
                raise RuntimeError('Historical input changed')
            checks.append({'path': row.path, 'before': row.sha256, 'after': after, 'unchanged': True})
    aliases = {}
    for name, module in list(sys.modules.items()):
        file = getattr(module, '__file__', None)
        if file is None:
            continue
        path = Path(file).resolve()
        if path.suffix == '.py' and path.is_relative_to(P / 'scripts'):
            aliases.setdefault(path, []).append(name)
    aliases.setdefault(P / 'scripts/experiments/mcwf_coordinate_scope_report_20260909.py', []).append('original_failed_reporter')
    aliases.setdefault(P / 'scripts/experiments/mcwf_waveform_scope_feasibility_20260909.py', []).append('scope_audit_runtime')
    records = []
    for path, names in sorted(aliases.items()):
        content = path.read_bytes()
        if re.search(rb'-----BEGIN (RSA |OPENSSH |EC )?PRIVATE KEY-----\s+[A-Za-z0-9+/=\r\n]{64,}', content):
            raise RuntimeError('Cannot package credentials')
        dest = root / 'scripts/runtime_dependencies' / path.relative_to(P / 'scripts')
        dest.parent.mkdir(parents=True, exist_ok=True)
        before = dest.exists()
        if before:
            if n.sha(dest) != n.sha(path):
                raise RuntimeError('Existing dependency snapshot is different')
        else:
            shutil.copy2(path, dest)
        records.append({'source': str(path), 'aliases': '|'.join(names), 'sha256': n.sha(path),
                        'destination': str(dest.relative_to(root)), 'already_present_identical': before})
    for row in existing:
        if n.sha(Path(row['path'])) != row['sha256']:
            raise RuntimeError('Pre-recovery artifact modified')
    n.write_csv(root / 'audit/EXPORT_RECOVERY_PREEXISTING_HASHES.csv', existing)
    n.write_csv(root / 'audit/FINAL_RECOVERY_INPUT_HASHES.csv', checks)
    n.write_csv(root / 'manifest/RUNTIME_DEPENDENCIES.csv', records)
    shutil.copy2(__file__, root / 'scripts/coordinate_scope_finalize.py')
    shutil.copy2(P / 'scripts/experiments/mcwf_coordinate_scope_report_20260909.py',
                 root / 'scripts/coordinate_scope_report.py')
    with (root / 'reports/EXPORT_RECOVERY_NOTICE_CN.md').open('x', encoding='utf-8') as f:
        f.write('# 归档恢复说明\n\n报告正文、图、分数表和 bootstrap 已在归档错误前完成，全部保持原 hash。'
                '重复模块别名导致依赖快照中断，本次仅按实际文件路径去重并核对已有副本。'
                '错误保存在 `logs/REPORT_EXPORT_FAILURE_RETAINED.json`。没有重新计算分数、修改门槛或重排候选。\n')
    gate = json.loads((root / 'contracts/PILOT_GATE.json').read_text())
    scope_gate = json.loads((scope / 'contracts/PILOT_GATE.json').read_text())
    c.write_once(root / 'audit/FINAL_DIAGNOSTIC_AUDIT.json', {'UTC': n.utc(),
        'R59': gate, 'R60': scope_gate,
        'latest_full_catalog': json.loads((reference / 'audit/FINAL_GOAL_READOUT.json').read_text()),
        'goal_achieved': False, 'new_catalog_metrics_generated': False,
        'historical_input_hash_checks': len(checks), 'historical_input_changes': 0,
        'preexisting_artifact_hash_checks': len(existing), 'preexisting_artifact_changes': 0,
        'runtime_dependencies': len(records), 'report_export_failure_preserved': True, 'status': n.STATUS})
    for r in (root, scope):
        c.write_once(r / 'contracts/FINAL_EXPORT_COMPLETE.json', {'UTC': n.utc(), 'goal_achieved': False,
            'status': n.STATUS, 'report': str(root / 'reports/R59_R60_COMPLETE_REPORT_CN.md'),
            'no_historical_overwrite': True, 'report_export_recovery': 'hash-verified duplicate-module deduplication'})
    print('FINAL_EXPORT_COMPLETE', len(existing), len(records), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--scope-root', type=Path, required=True)
    p.add_argument('--reference-root', type=Path, required=True)
    a = p.parse_args()
    main(a.root, a.scope_root, a.reference_root)
