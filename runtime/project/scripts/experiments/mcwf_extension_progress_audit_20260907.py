#!/usr/bin/env python3
"""Append-only inventory of completed and incomplete waveform ablations."""
from datetime import datetime, timezone
import argparse
import json
from pathlib import Path
import pandas as pd
import mcwf_pe_frontend_extension_v2_20260907 as e

dev = e.dev


def audit(root):
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out = root / f'audit/progress_{timestamp}'
    out.mkdir()
    protected = pd.read_csv(root / 'manifest/PROTECTED_INPUT_SHA256.csv')
    checks = []
    for item in protected.to_dict('records'):
        path = Path(item['path'])
        actual = dev.sha(path) if path.is_file() else None
        checks.append({'path': str(path), 'expected_sha256': item['sha256'],
                       'actual_sha256': actual, 'unchanged': actual == item['sha256']})
    dev.csv_write(out / 'PROTECTED_INPUT_RECHECK.csv', pd.DataFrame(checks))
    dev.json_write(out / 'PROTECTED_INPUT_RECHECK.json', {
        'checked_files': len(checks), 'unchanged_files': sum(r['unchanged'] for r in checks),
        'pass': all(r['unchanged'] for r in checks),
        'baseline_archive_not_repacked': True})
    if not all(r['unchanged'] for r in checks):
        raise RuntimeError('Protected input hash mismatch; see append-only audit')
    rows, budgets = [], []
    for trial in sorted((root / 'trials').iterdir()):
        if not trial.is_dir():
            continue
        target = trial / 'contracts/TARGET_AUDIT.json'
        row = {'trial': trial.name, 'status': 'INCOMPLETE_OR_SELECTION_REJECTED',
            'minimum_target_pass': False, 'fresh_confirmation': False}
        if target.exists():
            result = json.loads(target.read_text())
            row.update(status='COMPLETE_TARGET_PASS' if result['minimum_target_pass'] else 'COMPLETE_TARGET_FAIL',
                minimum_target_pass=result['minimum_target_pass'],
                fresh_confirmation=result['fresh_confirmation_done'],
                reused_injection_pass=result['reused_injection_pass'],
                waveform_PE_pass=all(r['pass'] for r in result['waveform_PE']),
                PE_budget_cells_pass=sum(r['PE_pass'] for r in result['real_development']),
                official_budget_cells_pass=sum(r['official_pass'] for r in result['real_development']),
                target_sha256=dev.sha(target))
        elif (trial / 'contracts/SEARCH_COMPLETE.json').exists():
            search = json.loads((trial / 'contracts/SEARCH_COMPLETE.json').read_text())
            row['status'] = 'COMPLETE_GRID_REJECTED' if not search.get('both_runs_qualify', False) else 'GRID_PASS_EVALUATION_PENDING'
        table = trial / 'tables/PE_OFFICIAL_ALL.csv'
        if table.exists():
            frame = pd.read_csv(table)
            frame = frame[(frame.seed.astype(str) == 'consensus') & frame.budget.isin([10, 20])].copy()
            frame['trial'] = trial.name
            budgets.append(frame)
        rows.append(row)
    dev.csv_write(out / 'TRIAL_LEDGER.csv', pd.DataFrame(rows))
    if budgets:
        dev.csv_write(out / 'ALL_CONSENSUS_BUDGETS.csv', pd.concat(budgets, ignore_index=True))
    for dep in e.DEPS:
        folder = root / f'expanded_data/{dep}'
        if not (folder / 'validation/source_plan.parquet').exists():
            continue
        plans = {s: pd.read_parquet(folder / f'{s}/source_plan.parquet') for s in ('train', 'validation')
                 if (folder / f'{s}/source_plan.parquet').exists()}
        result = {'deployment': dep, 'splits': {}, 'statistical_limit': 'new sources and noise blocks, reused lens environments'}
        for split, frame in plans.items():
            result['splits'][split] = {'sources': len(frame), 'unique_source_ids': frame.source_uid.nunique(),
                'gps_a_min': frame.gps_a.min(), 'gps_a_max': frame.gps_a.max(),
                'gps_b_min': frame.gps_b.min(), 'gps_b_max': frame.gps_b.max(),
                'unique_lens_environment_rows': frame.gwlmc_environment_row.nunique()}
        if len(plans) == 2:
            result['source_id_overlap'] = len(set(plans['train'].source_uid) & set(plans['validation'].source_uid))
            cols = ['m1_det', 'm2_det', 'a1', 'a2', 'tilt1', 'tilt2', 'theta_jn', 'phi12', 'phijl', 'phase', 'ra', 'dec']
            hashes = {s: set(pd.util.hash_pandas_object(f[cols], index=False).to_numpy()) for s, f in plans.items()}
            result['physical_source_parameter_hash_overlap'] = len(hashes['train'] & hashes['validation'])
            if result['source_id_overlap'] or result['physical_source_parameter_hash_overlap']:
                raise RuntimeError('Expanded source split overlap')
        if dep == 'gwtc3':
            result['calendar_caveat'] = 'Noise is O3-only, but source arrival/antenna-response epochs deliberately retain the frozen cumulative O1-O3 baseline calendar. Do not claim the whole generation is strict O3-exposure matched. No time channel was changed.'
        dev.json_write(out / f'{dep}_EXPANDED_DATA_AUDIT.json', result)
    dev.json_write(out / 'STATUS.json', {'utc': timestamp,
        'completed_trial_targets_pass': sum(r['minimum_target_pass'] for r in rows),
        'number_of_trial_directories': len(rows), 'finished_user_goal': False,
        'reporting_limit': 'This is an append-only progress audit, not final success or a selected new method.'})
    print(json.dumps({'audit': str(out), 'trials': len(rows), 'targets_pass': sum(r['minimum_target_pass'] for r in rows)}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    a = p.parse_args()
    audit(a.root)
