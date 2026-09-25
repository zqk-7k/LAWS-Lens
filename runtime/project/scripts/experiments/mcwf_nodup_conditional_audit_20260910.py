#!/usr/bin/env python3
"""Independent requirement-by-requirement audit of frozen R67 outputs."""
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
import mcwf_nodup_conditional_catalog_20260910 as app

n, io, old = app.n, app.io, app.audit


def exact(a, b, columns):
    for col in columns:
        if col not in a or col not in b or not a[col].equals(b[col]):
            raise RuntimeError('Frozen column differs: ' + col)


def main(root):
    if (root / 'audit/FINAL_GOAL_READOUT.json').exists():
        raise RuntimeError('Do not overwrite an audit')
    for name in ('PIPELINE_UNIT_PASS', 'EVALUATION_COMPLETE', 'REAL_COMPLETE'):
        if not (root / f'contracts/{name}.json').exists():
            raise RuntimeError('Incomplete prerequisite ' + name)
    contract = json.loads((root / 'contracts/ANALYSIS_CONTRACT.json').read_text())
    reference, pilot = Path(contract['reference']), Path(contract['pilot'])
    app.ROOT, app.PILOT, app.REFERENCE = root, pilot, reference
    app.check()
    app.pilot.check(pilot)
    configs = {(c['deployment'], c['seed'], c['method']): c for c in n.selections(root)}
    methods, refs = (*app.METHODS, n.BASELINE), (app.METHODS[0], app.METHODS[1], n.BASELINE)
    invariant, guards, scores, critical, deltas = [], [], [], [], []
    files = list((root / f'results/{app.NEW}').glob('gwtc*/seed_*/*/pairs.parquet'))
    files += list((root / f'results/{app.NEW}').glob('gwtc*/seed_*/real/fusion_all_pairs.parquet'))
    if len(files) != 36:
        raise RuntimeError('Need exactly30 injection and6real model panels')
    for file in sorted(files):
        rel = file.relative_to(root / f'results/{app.NEW}')
        dep, tag, panel = rel.parts[:3]
        seed = int(tag.split('_')[1])
        frames = {method: old.aligned(root / 'results' / method / rel) for method in methods}
        nd = frames[app.METHODS[0]]
        for method, historical in [(app.METHODS[0], 'NODUP-DIRECT-REPLAY'),
                                   (app.METHODS[1], 'REFERENCE-REGULARIZED-SINGLE-WF')]:
            original = old.aligned(reference / 'results' / historical / rel)
            exact(frames[method], original, ['idx_i', 'idx_j', 'waveform_score', 'final_score'] + (['rank'] if panel == 'real' else []))
        for method in app.METHODS:
            f, cfg = frames[method], configs[dep, seed, method]
            exact(f, nd, ['idx_i', 'idx_j', 'time_score', 'sky_raw_log_bf', 'embedding_only', 'time_contribution', 'sky_contribution'])
            if cfg['weights'] != configs[dep, seed, app.METHODS[0]]['weights']:
                raise RuntimeError('Outer weights differ')
            total = n.cf.channels(f, f.waveform_score.to_numpy(float)) @ np.asarray(cfg['weights'])
            err = float(abs(total - f.final_score.to_numpy()).max())
            if err > 1e-12:
                raise RuntimeError('Score reconstruction failed')
            active = f.shared_profile_eligible.to_numpy(bool)
            exact(f.loc[~active], nd.loc[~active], ['waveform_score'])
            if method == app.NEW:
                expected = nd.waveform_score.to_numpy(float).copy()
                expected[active], outside, clipped = app.pilot.apply(expected[active],
                    f.shared_profile_minimum_power.to_numpy(float)[active],
                    f.shared_profile_deficit.to_numpy(float)[active], cfg['conditional_model'])
                if not np.array_equal(expected, f.waveform_score.to_numpy(float)):
                    raise RuntimeError('Candidate score differs from frozen R66 formula')
                changed = expected != nd.waveform_score.to_numpy()
                columns = [c for c in ('idx_i', 'idx_j', 'pair_key', 'is_true_pair', 'true_pair_family',
                    'shared_profile_deficit', 'shared_profile_minimum_power', 'conditional_waveform_ood',
                    'conditional_waveform_used', 'embedding_only') if c in f]
                for position in np.flatnonzero(changed):
                    deltas.append({**f.iloc[position][columns].to_dict(), 'deployment': dep,
                        'seed': seed, 'panel': panel, 'NODUP_waveform': float(nd.waveform_score.iloc[position]),
                        'new_waveform': float(expected[position]),
                        'R62_waveform': float(frames[app.METHODS[1]].waveform_score.iloc[position])})
            if panel == 'real':
                exact(f, nd, [c for c in nd if c.startswith(('pe_', 'official_'))])
            invariant.append({'deployment': dep, 'seed': seed, 'panel': panel, 'method': method,
                'pairs': len(f), 'eligible': int(active.sum()), 'time_sky_weights_exact': True,
                'inactive_exact': True, 'total_max_error': err, 'external_values_exact': panel == 'real'})
        if panel == 'real':
            for method, f in frames.items():
                critical += [{**r, 'method': method, 'deployment': dep, 'seed': seed, 'unit': 'model'}
                    for r in f[f.pair_key.eq(old.KEY)].to_dict('records')]
            continue
        metrics = {}
        for method, f in frames.items():
            for mode, column in [('waveform', 'waveform_score'), ('fusion', 'final_score')]:
                value = n.cf.fast_metrics(f, f[column].to_numpy(float))
                metrics[method, mode] = value
                scores.append({'deployment': dep, 'seed': seed, 'panel': panel, 'method': method, 'mode': mode, **value})
        for method in app.METHODS:
            for ref in refs:
                for mode in ('waveform', 'fusion'):
                    a, b = metrics[method, mode], metrics[ref, mode]
                    strict = all(a[k] >= b[k] - 1e-12 for k in ('macro_r_at_10', 'average_precision'))
                    strict &= all(a[k] <= b[k] for k in ('false_at_recall_0p5', 'false_at_recall_0p9'))
                    guards.append({'method': method, 'deployment': dep, 'seed': seed, 'panel': panel,
                        'mode': mode, 'reference': ref, 'existing_guard': bool(n.cf.guard(a, b)),
                        'strict_pointwise_no_loss': bool(strict), **{k + '_delta': a[k] - b[k] for k in old.KEY_METRICS}})
    for method in methods:
        for dep in n.DEPS:
            f = pd.read_parquet(root / f'results/{method}/{dep}/consensus/fusion_all_pairs.parquet')
            critical += [{**r, 'method': method, 'deployment': dep, 'seed': 'consensus', 'unit': 'consensus'}
                for r in f[f.pair_key.eq(old.KEY)].to_dict('records')]
    budgets = []
    for name, unit in [('PE_OFFICIAL_BUDGETS.csv', 'consensus'), ('PER_SEED_PE_OFFICIAL_BUDGETS.csv', 'model')]:
        f = pd.read_csv(root / 'tables' / name)
        f['unit'] = unit
        if unit == 'consensus':
            f['seed'] = 'consensus'
        budgets.append(f)
    budgets = pd.concat(budgets, ignore_index=True)
    if budgets.duplicated(['config', 'deployment', 'unit', 'seed', 'budget']).any():
        raise RuntimeError('Duplicated PE budget')
    external = []
    for a in budgets[budgets.budget.isin([10, 20])].to_dict('records'):
        for ref in refs:
            match = budgets[(budgets.config == ref) & (budgets.deployment == a['deployment']) &
                (budgets.unit == a['unit']) & (budgets.budget == a['budget']) &
                (budgets.seed.astype(str) == str(a['seed']))]
            if len(match) != 1:
                raise RuntimeError('Budget reference mismatch')
            b = match.iloc[0]
            passes = {k: a[k] >= b[k] - 1e-12 for k in old.HIGHER}
            passes['catastrophic_mc'] = a['catastrophic_mc'] <= b['catastrophic_mc']
            external.append({'method': a['config'], 'deployment': a['deployment'], 'unit': a['unit'],
                'seed': a['seed'], 'budget': a['budget'], 'reference': ref,
                'all_no_loss': all(passes.values()), **{k + '_pass': bool(v) for k, v in passes.items()},
                **{k + '_delta': float(a[k] - b[k]) for k in (*old.HIGHER, 'catastrophic_mc')}})
    hashes = {}
    for directory in (root, pilot):
        for r in pd.read_csv(directory / 'manifest/INPUT_SHA256.csv').itertuples():
            if r.path in hashes and hashes[r.path] != r.sha256:
                raise RuntimeError('Conflicting frozen hash')
            hashes[r.path] = r.sha256
    verified = [{'path': file, 'expected': expected, 'actual': io.sha(file)} for file, expected in hashes.items()]
    if any(r['expected'] != r['actual'] for r in verified):
        raise RuntimeError('Changed protected input')
    for name, data in [('REFERENCE_SPECIFIC_INJECTION_GUARDS', guards), ('REFERENCE_SPECIFIC_PE_OFFICIAL_GUARDS', external),
                       ('RECOMPUTED_METRICS_ALL_PANELS', scores), ('CRITICAL_PAIR_ALL_RANKS', critical),
                       ('REFERENCE_CHANGED_PAIRS', deltas)]:
        io.csv(root / f'tables/{name}.csv', data)
    io.csv(root / 'audit/PAIR_ALIGNED_INVARIANCE.csv', invariant)
    io.csv(root / 'audit/FINAL_INPUT_HASH_CHECK.csv', verified)
    gf, ef, kf = pd.DataFrame(guards), pd.DataFrame(external), pd.DataFrame(critical)
    g = gf[(gf.method == app.NEW) & (gf.reference == app.METHODS[0])]
    e = ef[(ef.method == app.NEW) & (ef.reference == app.METHODS[0])]
    key = kf[kf.method == app.NEW]
    keyrank = key['rank'].where(key.unit.eq('model'), key.consensus_rank)
    keypass = len(key) == 4 and bool((keyrank > 10).all())
    goal = keypass and bool(g.existing_guard.all()) and bool(e.all_no_loss.all())
    result = {'UTC': io.utc(), 'status': n.STATUS, 'goal_achieved': goal,
        'candidate': app.NEW, 'primary_reference': app.METHODS[0],
        'key_pair_outside_top10_all_models_and_consensus': keypass,
        'injection_failed_panels': g[~g.existing_guard].to_dict('records'),
        'external_failed_budgets': e[~e.all_no_loss].to_dict('records'),
        'consensus_PE_official_no_loss': bool(e[e.unit.eq('consensus')].all_no_loss.all()),
        'per_model_PE_official_no_loss': bool(e[e.unit.eq('model')].all_no_loss.all()),
        'strict_injection_no_loss': bool(g.strict_pointwise_no_loss.all()),
        'hash_checks': len(verified), 'changed_inputs': 0, 'invariance_panels': len(invariant),
        'postfreeze_audit_not_parameter_selection': True, 'no_automatic_adoption': True}
    io.write(root / 'audit/FINAL_GOAL_READOUT.json', result)
    io.write(root / 'audit/REFERENCE_COMPARISON_COMPLETE.json', {'UTC': io.utc(), 'completed': True,
        'goal_achieved': goal, 'status': n.STATUS, 'audit_sha256': io.sha(Path(__file__))})
    shutil.copy2(__file__, root / 'scripts/nodup_conditional_audit.py')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    main(parser.parse_args().root)
