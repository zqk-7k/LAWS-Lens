#!/usr/bin/env python3
"""Independent, read-only reference and scope audits for the R51 catalog run."""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys
import time
import numpy as np
import pandas as pd
import psutil

P = Path('/root/autodl-tmp/gw-catalog')
sys.path.insert(0, str(P/'scripts/experiments'))
import mcwf_shared_profile_catalog_20260909 as app
n = app.n
ROOT = None
KEY = 'GW191103_012549--GW191105_143521'
HIGHER = ('BC_mc_ge_0p5', 'median_BC_mc', 'Dmax_le_3', 'official_frontend', 'official_hanabi')
KEY_METRICS = ('macro_r_at_1', 'macro_r_at_10', 'average_precision',
               'false_at_recall_0p5', 'false_at_recall_0p9')


def aligned(path):
    frame = pd.read_parquet(path).sort_values(['idx_i', 'idx_j']).reset_index(drop=True)
    if frame.duplicated(['idx_i', 'idx_j']).any():
        raise RuntimeError('Duplicate unordered-pair identity: '+str(path))
    return frame


def preflight():
    dest = ROOT/'audit/REFERENCE_AUDIT_FROZEN.json'
    if dest.exists():
        raise RuntimeError('Reference audit already frozen')
    if not json.loads((ROOT/'contracts/PIPELINE_UNIT_PASS.json').read_text())['passed']:
        raise RuntimeError('Pipeline unit tests required')
    if (ROOT/'contracts/EVALUATION_COMPLETE.json').exists():
        raise RuntimeError('Freeze reference audit before full catalog outcomes')
    inputs = []
    for method in ('NODUP-DIRECT', n.BASELINE):
        files = sorted((app.BASE/f'results/{method}').glob('gwtc*/seed_*/*/pairs.parquet'))
        files += sorted((app.BASE/f'results/{method}').glob('gwtc*/seed_*/real/fusion_all_pairs.parquet'))
        if len(files) != 36:
            raise RuntimeError('Expected 36 frozen panels per archived method')
        for path in files:
            inputs.append({'path': str(path), 'sha256': n.sha(path), 'bytes': path.stat().st_size})
    for dep in n.DEPS:
        path = n.t.EXTERNAL/f'{dep}_external_reference.parquet'
        inputs.append({'path': str(path), 'sha256': n.sha(path), 'bytes': path.stat().st_size})
    manifest = ROOT/'manifest/REFERENCE_INPUT_SHA256.csv'
    n.write_csv(manifest, inputs)
    shutil.copy2(__file__, ROOT/'scripts/shared_profile_catalog_audit.py')
    versions = {}
    for name in ('numpy', 'scipy', 'pandas', 'pycbc', 'lalsuite', 'psutil'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = 'not found as an installed distribution'
    n.write_json(dest, {'UTC': n.utc(), 'runtime_sha256': n.sha(Path(__file__)),
        'reference_manifest_sha256': n.sha(manifest), 'reference_files': len(inputs),
        'python': sys.version, 'versions': versions,
        'primary_reference': 'NODUP-DIRECT-REPLAY', 'secondary_reference': n.BASELINE,
        'primary_arm': 'SHARED-PROFILE-SINGLE-WF',
        'sensitivity_arm': 'SHARED-PROFILE-REJECT-ONLY; not selected by real outcomes',
        'PE_no_loss': {'budgets': [10, 20], 'higher_is_better': list(HIGHER),
            'lower_is_better': ['catastrophic_mc'], 'units': ['consensus', 'each model seed'],
            'numerical_tolerance': 1e-12, 'runs_must_pass_separately': True},
        'key_pair': {'identity': KEY, 'report_all_ranks': True,
            'must_leave_top10': ['O3 consensus', 'each O3 model seed']},
        'injection_guard': 'Reuse existing n.cf.guard without changed tolerances, against NODUP; report strict raw no-loss separately.',
        'injection_guard_units': 'Each frozen model and each catalog, waveform and fusion separately; no averaging away a failed panel.',
        'bootstrap': 'Conditional system-level paired bootstrap; not independent confirmation after adaptive development.',
        'no_arm_or_parameter_choice_by_real_results': True,
        'scope': 'Post-freeze external descriptive audit; neither official membership nor marginal PE is a lensing label.',
        'goal_achieved': False, 'status': n.STATUS})
    print('REFERENCE_AUDIT_FROZEN', len(inputs), flush=True)


def verify_inputs():
    manifest = pd.read_csv(ROOT/'manifest/REFERENCE_INPUT_SHA256.csv')
    changed = [r.path for r in manifest.itertuples() if n.sha(Path(r.path)) != r.sha256]
    if changed:
        n.write_json(ROOT/'audit/REFERENCE_INPUT_FAILURE.json', {'UTC': n.utc(), 'changed': changed})
        raise RuntimeError('Historical input changed')
    return len(manifest)


def compare():
    if not (ROOT/'contracts/REAL_COMPLETE.json').exists():
        raise RuntimeError('Complete injection and post-freeze real audit first')
    if (ROOT/'audit/REFERENCE_COMPARISON_COMPLETE.json').exists():
        raise RuntimeError('Read-only comparison already saved')
    checked = verify_inputs()
    configs = {(v['deployment'], int(v['seed']), v['method']): v for v in n.selections(ROOT)}
    invariance, guards, keyrows = [], [], []
    for method in app.METHODS:
        files = sorted((ROOT/f'results/{method}').glob('gwtc*/seed_*/*/pairs.parquet'))
        files += sorted((ROOT/f'results/{method}').glob('gwtc*/seed_*/real/fusion_all_pairs.parquet'))
        for path in files:
            rel = path.relative_to(ROOT/f'results/{method}')
            dep, seedstr, panel = rel.parts[:3]
            seed = int(seedstr.split('_')[1])
            frame = aligned(path)
            baseline = aligned(ROOT/'results/NODUP-DIRECT-REPLAY'/rel)
            old = aligned(app.BASE/'results/NODUP-DIRECT'/rel)
            columns = ['idx_i', 'idx_j', 'time_score', 'sky_raw_log_bf', 'embedding_only']
            columns += [c for c in ('time_contribution', 'sky_contribution') if c in frame]
            for column in columns:
                if not frame[column].equals(baseline[column]):
                    raise RuntimeError('Frozen channel/scope changed: '+column)
            for column in ('waveform_score', 'final_score'):
                if not np.array_equal(baseline[column], old[column]):
                    raise RuntimeError('Current NODUP replay is not exact')
            if panel == 'real':
                external = [c for c in baseline if c.startswith(('pe_', 'official_'))]
                if not frame[external].equals(baseline[external]):
                    raise RuntimeError('External audit values changed')
            cfg = configs[dep, seed, method]
            if cfg['weights'] != configs[dep, seed, app.METHODS[0]]['weights']:
                raise RuntimeError('Outer weights changed')
            expected = n.cf.channels(frame, frame.waveform_score.to_numpy(float)) @ np.asarray(cfg['weights'])
            error = float(np.max(np.abs(expected-frame.final_score.to_numpy(float))))
            if error > 1e-12:
                raise RuntimeError('Final three-channel reconstruction failed')
            active = frame.shared_profile_eligible.to_numpy(bool)
            if not np.array_equal(frame.waveform_score.to_numpy()[~active], baseline.waveform_score.to_numpy()[~active]):
                raise RuntimeError('Exact inactive NODUP fallback failed')
            if method == app.METHODS[2] and np.any(frame.waveform_score.to_numpy() > baseline.waveform_score.to_numpy()):
                raise RuntimeError('Rejection-only arm increased waveform score')
            invariance.append({'method': method, 'deployment': dep, 'seed': seed, 'panel': panel,
                'pairs': len(frame), 'eligible_pairs': int(active.sum()),
                'changed_waveform_pairs': int(np.sum(frame.waveform_score.to_numpy()!=baseline.waveform_score.to_numpy())),
                'time_sky_max_difference': 0., 'final_reconstruction_max_difference': error,
                'PE_official_unchanged': True if panel == 'real' else None})
            if panel == 'real':
                for row in frame[frame.pair_key.eq(KEY)].to_dict('records'):
                    keyrows.append({**row, 'method': method, 'deployment': dep, 'seed': seed, 'unit': 'model'})
                continue
            for reference in (app.METHODS[0], n.BASELINE):
                other = aligned(ROOT/'results'/reference/rel)
                for mode, column in [('waveform', 'waveform_score'), ('fusion', 'final_score')]:
                    a = n.cf.fast_metrics(frame, frame[column].to_numpy(float))
                    b = n.cf.fast_metrics(other, other[column].to_numpy(float))
                    strict = (a['macro_r_at_10'] >= b['macro_r_at_10']-1e-12 and
                        a['average_precision'] >= b['average_precision']-1e-12 and
                        a['false_at_recall_0p5'] <= b['false_at_recall_0p5'] and
                        a['false_at_recall_0p9'] <= b['false_at_recall_0p9'])
                    guards.append({'method': method, 'deployment': dep, 'seed': seed, 'panel': panel,
                        'mode': mode, 'reference': reference, 'existing_guard': bool(n.cf.guard(a, b)),
                        'strict_pointwise_no_loss': bool(strict), **{k+'_delta': a[k]-b[k] for k in KEY_METRICS}})
        for dep in n.DEPS:
            frame = pd.read_parquet(ROOT/f'results/{method}/{dep}/consensus/fusion_all_pairs.parquet')
            for row in frame[frame.pair_key.eq(KEY)].to_dict('records'):
                keyrows.append({**row, 'method': method, 'deployment': dep, 'seed': 'consensus', 'unit': 'consensus'})
    budget_frames = []
    for filename, unit in [('PE_OFFICIAL_BUDGETS.csv', 'consensus'), ('PER_SEED_PE_OFFICIAL_BUDGETS.csv', 'model')]:
        frame = pd.read_csv(ROOT/'tables'/filename)
        frame['unit'] = unit
        if unit == 'consensus':
            frame['seed'] = 'consensus'
        budget_frames.append(frame)
    budgets = pd.concat(budget_frames, ignore_index=True)
    pe = []
    for row in budgets[budgets.budget.isin([10, 20])].to_dict('records'):
        for reference in (app.METHODS[0], n.BASELINE):
            matches = budgets[(budgets.config == reference) & (budgets.deployment == row['deployment']) &
                (budgets.budget == row['budget']) & (budgets.unit == row['unit']) &
                (budgets.seed.astype(str) == str(row['seed']))]
            if len(matches) != 1:
                raise RuntimeError('Nonunique seed/budget reference')
            other = matches.iloc[0]
            checks = {k: row[k] >= other[k]-1e-12 for k in HIGHER}
            checks['catastrophic_mc'] = row['catastrophic_mc'] <= other['catastrophic_mc']
            pe.append({'method': row['config'], 'deployment': row['deployment'], 'unit': row['unit'],
                'seed': row['seed'], 'budget': row['budget'], 'reference': reference,
                'all_no_loss': all(checks.values()), **{k+'_pass': bool(v) for k, v in checks.items()},
                **{k+'_delta': float(row[k]-other[k]) for k in (*HIGHER, 'catastrophic_mc')}})
    n.write_csv(ROOT/'audit/PAIR_ALIGNED_INVARIANCE.csv', invariance)
    n.write_csv(ROOT/'tables/REFERENCE_SPECIFIC_INJECTION_GUARDS.csv', guards)
    n.write_csv(ROOT/'tables/REFERENCE_SPECIFIC_PE_OFFICIAL_GUARDS.csv', pe)
    n.write_csv(ROOT/'tables/CRITICAL_PAIR_ALL_RANKS.csv', keyrows)
    n.write_json(ROOT/'audit/REFERENCE_COMPARISON_COMPLETE.json', {'UTC': n.utc(),
        'historical_files_unchanged': checked, 'invariance_panels': len(invariance),
        'status': n.STATUS, 'goal_achieved': False,
        'note': 'Full guard table, not an automatic declaration of upgrade or goal completion.'})
    print('REFERENCE_COMPARISON_COMPLETE', len(invariance), len(pe), flush=True)


def monitor(pid, label):
    output = ROOT/f'logs/RESOURCE_MONITOR_{label}.jsonl'
    with output.open('x') as stream:
        while True:
            try:
                parent = psutil.Process(pid)
                processes = [parent]+parent.children(recursive=True)
                active = parent.is_running() and parent.status()!=psutil.STATUS_ZOMBIE
            except psutil.NoSuchProcess:
                processes, active = [], False
            rss = cpu = 0.
            for process in processes:
                try:
                    rss += process.memory_info().rss
                    t = process.cpu_times()
                    cpu += t.user+t.system
                except psutil.NoSuchProcess:
                    pass
            records = []
            for path in (ROOT/'pairs').rglob('*.json'):
                try:
                    records.append(json.loads(path.read_text()))
                except json.JSONDecodeError:
                    pass
            free = psutil.disk_usage(str(ROOT)).free/1024**3
            row = {'UTC': datetime.now(timezone.utc).isoformat(), 'pid': pid, 'parent_alive': active,
                'processes': len(processes), 'summed_RSS_GiB': rss/1024**3, 'live_process_CPU_seconds': cpu,
                'disk_free_GiB': free, 'receipts': len(records),
                'failed_receipts': sum(r.get('status')!='COMPLETE' for r in records)}
            stream.write(json.dumps(row)+'\n')
            stream.flush()
            if not active:
                break
            time.sleep(30)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--stage', choices=('preflight', 'compare', 'monitor'), required=True)
    parser.add_argument('--pid', type=int)
    parser.add_argument('--label', default='injection')
    args = parser.parse_args()
    ROOT = args.root
    if args.stage == 'monitor':
        monitor(args.pid, args.label)
    else:
        globals()[args.stage]()
