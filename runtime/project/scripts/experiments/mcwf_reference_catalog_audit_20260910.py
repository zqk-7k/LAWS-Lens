#!/usr/bin/env python3
"""R62 post-freeze audit; thresholds are inherited, not selected here."""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd

P = Path('/root/autodl-tmp/gw-catalog')
sys.path.insert(0, str(P / 'scripts/experiments'))
import mcwf_reference_regularized_catalog_20260910 as app
import mcwf_shared_profile_catalog_audit_20260909 as old
n = app.n


def exact(a, b, columns, label):
    for col in columns:
        if col not in a or col not in b or not a[col].equals(b[col]):
            raise RuntimeError(label + ': ' + col)


def main(root):
    target = root / 'audit/REFERENCE_COMPARISON_COMPLETE.json'
    if target.exists():
        raise RuntimeError('Do not overwrite a completed audit')
    for name in ('PIPELINE_UNIT_PASS', 'EVALUATION_COMPLETE', 'REAL_COMPLETE'):
        if not (root / f'contracts/{name}.json').exists():
            raise RuntimeError('Finish calculation and unit tests first: ' + name)
    contract = json.loads((root / 'contracts/ANALYSIS_CONTRACT.json').read_text())
    reference, pilot = Path(contract['reference']), Path(contract['pilot'])
    configs = {(c['deployment'], c['seed'], c['method']): c for c in n.selections(root)}
    refs = (app.METHODS[0], app.METHODS[1], n.BASELINE)
    checks, guards, changes, critical, metric_rows = [], [], [], [], []
    files = sorted((root / f'results/{app.NEW}').glob('gwtc*/seed_*/*/pairs.parquet'))
    files += sorted((root / f'results/{app.NEW}').glob('gwtc*/seed_*/real/fusion_all_pairs.parquet'))
    if len(files) != 36:
        raise RuntimeError('Expected 30 injection and 6 real panels')
    for path in files:
        rel = path.relative_to(root / f'results/{app.NEW}')
        dep, seedname, panel = rel.parts[:3]
        seed = int(seedname.split('_')[1])
        frames = {m: old.aligned(root / 'results' / m / rel) for m in (*app.METHODS, n.BASELINE)}
        nd, previous = frames[app.METHODS[0]], frames[app.METHODS[1]]
        for method, historical in ((app.METHODS[0], 'NODUP-DIRECT-REPLAY'),
                                   (app.METHODS[1], 'SHARED-PROFILE-MONOTONE-BOUNDARY')):
            archived = old.aligned(reference / 'results' / historical / rel)
            cols = ['idx_i', 'idx_j', 'waveform_score', 'final_score']
            if panel == 'real':
                cols.append('rank')
            exact(frames[method], archived, cols, 'Historical replay changed')
        metrics = {}
        for method in app.METHODS:
            frame, cfg = frames[method], configs[dep, seed, method]
            fixed = ['idx_i', 'idx_j', 'time_score', 'sky_raw_log_bf', 'embedding_only']
            fixed += [c for c in ('time_contribution', 'sky_contribution') if c in frame]
            exact(frame, nd, fixed, 'Frozen data/channel changed')
            if cfg['weights'] != configs[dep, seed, app.METHODS[0]]['weights']:
                raise RuntimeError('Frozen outer weights changed')
            total = n.cf.channels(frame, frame.waveform_score.to_numpy(float)) @ np.asarray(cfg['weights'])
            error = float(abs(total - frame.final_score.to_numpy()).max())
            if error > 1e-12:
                raise RuntimeError('Total-score reconstruction failed')
            active = frame.shared_profile_eligible.to_numpy(bool)
            exact(frame.loc[~active], nd.loc[~active], ['waveform_score'], 'Inactive fallback changed')
            delta = frame.waveform_score.to_numpy() - previous.waveform_score.to_numpy()
            changed = delta != 0
            if method == app.METHODS[1] or (method == app.NEW and cfg['no_update']):
                exact(frame, previous, ['waveform_score', 'final_score'], 'No-update replay changed')
                if panel == 'real':
                    exact(frame, previous, ['rank'], 'No-update ranking changed')
            if method == app.NEW:
                if np.any(changed & ~active):
                    raise RuntimeError('New score changed an ineligible pair')
                ids = [c for c in ('idx_i', 'idx_j', 'pair_key', 'is_true_pair', 'true_pair_family',
                    'embedding_only', 'shared_profile_deficit', 'shared_profile_minimum_power',
                    'shared_normalized_joint_BC', 'score_ood', 'score_clipped') if c in frame]
                for pos in np.flatnonzero(changed):
                    changes.append({**frame.iloc[pos][ids].to_dict(), 'deployment': dep,
                        'seed': seed, 'panel': panel, 'R55_waveform': float(previous.waveform_score.iloc[pos]),
                        'new_waveform': float(frame.waveform_score.iloc[pos]), 'delta': float(delta[pos])})
            if panel == 'real':
                cols = [c for c in nd if c.startswith(('pe_', 'official_'))]
                exact(frame, nd, cols, 'External audit values changed')
            checks.append({'method': method, 'deployment': dep, 'seed': seed, 'panel': panel,
                'pairs': len(frame), 'eligible_pairs': int(active.sum()),
                'changed_from_R55': int(changed.sum()), 'frozen_channels_exact': True,
                'inactive_fallback_exact': True, 'outer_weights_exact': True,
                'total_reconstruction_max_error': error,
                'no_update': bool(cfg.get('no_update', method != app.NEW)),
                'PE_official_unchanged': True if panel == 'real' else None})
        if panel == 'real':
            for method, frame in frames.items():
                for row in frame[frame.pair_key.eq(old.KEY)].to_dict('records'):
                    critical.append({**row, 'method': method, 'deployment': dep,
                                     'seed': seed, 'unit': 'model'})
            continue
        for method, frame in frames.items():
            for mode, col in (('waveform', 'waveform_score'), ('fusion', 'final_score')):
                result = n.cf.fast_metrics(frame, frame[col].to_numpy(float))
                metrics[method, mode] = result
                metric_rows.append({'method': method, 'deployment': dep, 'seed': seed,
                                    'panel': panel, 'mode': mode, **result})
        for method in app.METHODS:
            for ref in refs:
                for mode in ('waveform', 'fusion'):
                    a, b = metrics[method, mode], metrics[ref, mode]
                    strict = all(a[k] >= b[k] - 1e-12 for k in ('macro_r_at_10', 'average_precision'))
                    strict &= all(a[k] <= b[k] for k in ('false_at_recall_0p5', 'false_at_recall_0p9'))
                    guards.append({'method': method, 'deployment': dep, 'seed': seed, 'panel': panel,
                        'mode': mode, 'reference': ref, 'existing_guard': bool(n.cf.guard(a, b)),
                        'strict_pointwise_no_loss': bool(strict),
                        **{k + '_delta': a[k] - b[k] for k in old.KEY_METRICS}})
    for method in (*app.METHODS, n.BASELINE):
        for dep in n.DEPS:
            frame = pd.read_parquet(root / f'results/{method}/{dep}/consensus/fusion_all_pairs.parquet')
            if method == app.NEW and dep == 'gwtc3':
                previous = pd.read_parquet(root / f'results/{app.METHODS[1]}/{dep}/consensus/fusion_all_pairs.parquet')
                exact(frame, previous, ['pair_key', 'consensus_rank', 'rank_mean', 'final_score_mean'],
                      'O3 no-update consensus changed')
            critical.extend({**row, 'method': method, 'deployment': dep, 'seed': 'consensus', 'unit': 'consensus'}
                            for row in frame[frame.pair_key.eq(old.KEY)].to_dict('records'))
    budget_frames = []
    for filename, unit in (('PE_OFFICIAL_BUDGETS.csv', 'consensus'), ('PER_SEED_PE_OFFICIAL_BUDGETS.csv', 'model')):
        frame = pd.read_csv(root / 'tables' / filename)
        frame['unit'] = unit
        if unit == 'consensus':
            frame['seed'] = 'consensus'
        budget_frames.append(frame)
    budgets = pd.concat(budget_frames, ignore_index=True)
    if budgets.duplicated(['config', 'deployment', 'unit', 'seed', 'budget']).any():
        raise RuntimeError('Nonunique budget row')
    pe = []
    for row in budgets[budgets.budget.isin([10, 20])].to_dict('records'):
        for ref in refs:
            match = budgets[(budgets.config == ref) & (budgets.deployment == row['deployment']) &
                (budgets.unit == row['unit']) & (budgets.budget == row['budget']) &
                (budgets.seed.astype(str) == str(row['seed']))]
            if len(match) != 1:
                raise RuntimeError('Ambiguous budget reference')
            b = match.iloc[0]
            fields = {k: row[k] >= b[k] - 1e-12 for k in old.HIGHER}
            fields['catastrophic_mc'] = row['catastrophic_mc'] <= b['catastrophic_mc']
            pe.append({'method': row['config'], 'deployment': row['deployment'],
                'unit': row['unit'], 'seed': row['seed'], 'budget': row['budget'], 'reference': ref,
                'all_no_loss': all(fields.values()), **{k + '_pass': bool(v) for k, v in fields.items()},
                **{k + '_delta': float(row[k] - b[k]) for k in (*old.HIGHER, 'catastrophic_mc')}})
    hashes = {}
    for directory in (root, pilot):
        for row in pd.read_csv(directory / 'manifest/INPUT_SHA256.csv').itertuples():
            if row.path in hashes and hashes[row.path] != row.sha256:
                raise RuntimeError('Conflicting historical hash')
            hashes[row.path] = row.sha256
        frozen = json.loads((directory / 'contracts/START_FREEZE.json').read_text())
        for field, name in (('contract_sha256', 'contracts/ANALYSIS_CONTRACT.json'),
                            ('config_sha256', 'configs/SELECTED_CONFIGURATIONS.json'),
                            ('input_manifest_sha256', 'manifest/INPUT_SHA256.csv')):
            if field in frozen:
                hashes[str(directory / name)] = frozen[field]
    verified = [{'path': path, 'expected': expected, 'actual': n.sha(Path(path))}
                for path, expected in hashes.items()]
    if any(r['expected'] != r['actual'] for r in verified):
        raise RuntimeError('Frozen input hash changed')
    tables = [('REFERENCE_SPECIFIC_INJECTION_GUARDS', guards), ('REFERENCE_SPECIFIC_PE_OFFICIAL_GUARDS', pe),
              ('RECOMPUTED_METRICS_ALL_PANELS', metric_rows), ('REFERENCE_CHANGED_PAIRS', changes),
              ('CRITICAL_PAIR_ALL_RANKS', critical)]
    for name, rows in tables:
        n.write_csv(root / f'tables/{name}.csv', rows)
    n.write_csv(root / 'audit/PAIR_ALIGNED_INVARIANCE.csv', checks)
    n.write_csv(root / 'audit/FINAL_INPUT_HASH_CHECK.csv', verified)
    gf, pf, kf = pd.DataFrame(guards), pd.DataFrame(pe), pd.DataFrame(critical)
    primary = gf[(gf.method == app.NEW) & (gf.reference == app.METHODS[0])]
    external = pf[(pf.method == app.NEW) & (pf.reference == app.METHODS[0])]
    key = kf[kf.method.eq(app.NEW)]
    key_ranks = key['rank'].where(key.unit.eq('model'), key.consensus_rank)
    key_pass = len(key) == 4 and bool((key_ranks > 10).all())
    goal = key_pass and bool(primary.existing_guard.all()) and bool(external.all_no_loss.all())
    result = {'UTC': n.utc(), 'status': n.STATUS, 'goal_achieved': bool(goal),
        'candidate': app.NEW, 'primary_reference': app.METHODS[0],
        'key_pair_outside_top10_all_models_and_consensus': key_pass,
        'injection_failed_panels': primary[~primary.existing_guard].to_dict('records'),
        'external_failed_budgets': external[~external.all_no_loss].to_dict('records'),
        'strict_injection_no_loss': bool(primary.strict_pointwise_no_loss.all()),
        'consensus_PE_official_no_loss': bool(external[external.unit.eq('consensus')].all_no_loss.all()),
        'per_model_PE_official_no_loss': bool(external[external.unit.eq('model')].all_no_loss.all()),
        'changed_inputs': 0, 'hash_checks': len(verified), 'invariance_panels': len(checks),
        'audit_code_sha256': n.sha(Path(__file__)),
        'postfreeze_readout_not_parameter_selection': True,
        'no_automatic_adoption': True}
    n.write_json(root / 'audit/FINAL_GOAL_READOUT.json', result)
    shutil.copy2(__file__, root / 'scripts/reference_catalog_audit.py')
    n.write_json(target, {'UTC': n.utc(), 'completed': True, 'hash_checks': len(verified),
                         'panels': len(checks), 'goal_achieved': bool(goal), 'status': n.STATUS})
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    main(parser.parse_args().root)
