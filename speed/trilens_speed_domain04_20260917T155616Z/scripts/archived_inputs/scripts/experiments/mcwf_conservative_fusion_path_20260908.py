#!/usr/bin/env python3
"""A common continuous replacement path between frozen ranking methods."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import numpy as np
import pandas as pd
P = Path('/root/autodl-tmp/gw-catalog')
sys.path.insert(0, str(P/'scripts/experiments'))
import mcwf_temporal_response_evaluate_20260908 as ev
import mcwf_joint_global_development_v2_20260908 as globaldev
t, dev, cf = ev.t, ev.dev, ev.cf
ROOTS = [P/'results'/name for name in (
    'mcwf_joint_fusion_retune_exploratory_20260908T160928Z',
    'mcwf_joint_predictive_ensemble_exploratory_20260908T161608Z',
    'mcwf_joint_2d_calibration_exploratory_20260908T162756Z')]
ALPHAS = (0., .0625, .125, .25, .375, .5, .625, .75, .875, 1.)


def specs():
    out = []
    for upstream in ROOTS:
        configs = json.loads((upstream/'calibration/SELECTED.json').read_text())
        for name in sorted({c['method'] for c in configs}):
            if 'REFIT' not in name or 'NONNEGATIVE' in name:
                continue
            group = [c for c in configs if c['method'] == name]
            if len(group) != 6 or any(min(c['weights']) <= 0 for c in group):
                raise RuntimeError('Incomplete positive-weight upstream')
            out.append({'upstream': str(upstream), 'method': name, 'configs': group})
    return out


def initialize(root):
    if root.exists():
        raise RuntimeError('New output required')
    for name in ('contracts', 'scripts', 'logs', 'tables', 'reports', 'figures', 'manifest', 'calibration', 'evaluation'):
        (root/name).mkdir(parents=True)
    contracts = specs()
    dev.json_write(root/'contracts/ANALYSIS_CONTRACT.json', {
        'id': 'MCWF-CONSERVATIVE-FUSION-PATH-25', 'UTC': datetime.now(timezone.utc).isoformat(),
        'status': t.STATUS, 'goal_achieved': False, 'same_algorithm_both_runs': True,
        'hypothesis': 'A full replacement can discard useful retained waveform evidence or move too far along validation-fitted fusion weights. Test a continuous common replacement amplitude,not onlythebinarychoicesof18.',
        'upstreams': [{'root': s['upstream'], 'method': s['method']} for s in contracts],
        'alpha': ALPHAS, 'grid_size': len(contracts)*len(ALPHAS),
        'formula': 'Normalize both positive W/T/S vectors tosum1. Snew=(1-alpha)*Sold_norm+alpha*Supstream_norm. Effective weights=(1-alpha)*wold+alpha*wupstream; effective Zwf is weightedaverage ofonlyold/newwaveform terms divided byeffective wW.',
        'waveform_decomposition': 'The newscore remainsonewaveform+oneunchangedtime+oneunchangedsky matrix. Neither rawtime norrawsky is recalibrated. Effective outerweights changeandmustbereported.',
        'selection': 'EXPLICIT ADAPTIVE REAL DEVELOPMENT,not validation-only. All6run/seed validation and REUSED test waveform/fusion guardrails first;thenjointTop10/20PE+officialtarget. One upstreammethod andonealpha acrossbothruns/allseeds.',
        'tie': 'maxminimumrun frontendgain;then summedMcpassgain;then summedmedianBCgain;then totalfrontendgain;then smalleralpha;thenfixedmethodlexicographic.',
        'real_data_use': 'PE/official labels joinedonlyafterrank,butoutcome metrics mayselect GLOBAL alpha/method. No eventwise weights,IDs,PE orofficialvalues inthescoringformula. Thisisinherentlyadaptive andnot independent realvalidation.',
        'frozen': ['allnetworks', 'allpredictiondensitycalibrators', 'time_score', 'sky_raw_log_bf', 'scope', 'historicalweightsandranks', 'paper'],
        'changed_channels': ['waveform'], 'outer_weights_changed': True,
        'fresh_confirmation_required': True, 'official_overlap_is_not_lensing_truth': True,
        'negative_results_retained': True})
    inputs = t.protected()
    for s in contracts:
        up = Path(s['upstream'])
        for c in s['configs']:
            folder = up/f"evaluation/{s['method']}/{c['deployment']}/seed_{c['seed']}"
            for name in ('validation_pairs.parquet', 'test_pairs.parquet', 'real_fusion_pairs.parquet'):
                path = folder/name
                inputs.append({'path': str(path), 'sha256': dev.sha(path), 'bytes': path.stat().st_size})
    dev.csv_write(root/'manifest/INPUT_SHA256.csv', pd.DataFrame(inputs).drop_duplicates('path'))
    shutil.copy2(__file__, root/'scripts/conservative_fusion_path.py')
    rng = np.random.default_rng(202610030)
    errors = []
    for alpha in ALPHAS:
        a, b = rng.uniform(.1, 1, (2, 3));a /= a.sum();b /= b.sum()
        x = rng.normal(size=(100, 4));w = (1-alpha)*a+alpha*b
        waveform = ((1-alpha)*a[0]*x[:, 0]+alpha*b[0]*x[:, 1])/w[0]
        direct = (1-alpha)*np.c_[x[:, 0], x[:, 2:]]@a+alpha*np.c_[x[:, 1], x[:, 2:]]@b
        reconstructed = np.c_[waveform, x[:, 2:]]@w
        error = float(abs(direct-reconstructed).max());errors.append(error)
        if error > 2e-14:
            raise RuntimeError('Fusion algebra test failed')
    dev.json_write(root/'contracts/ALGEBRA_TEST.json', {'pass': True, 'maxerror': max(errors)})
    dev.json_write(root/'contracts/START_FREEZE.json', {
        'contract_sha256': dev.sha(root/'contracts/ANALYSIS_CONTRACT.json'), 'code_sha256': dev.sha(Path(__file__))})


def inputs(spec):
    frames, new, weights = {}, {}, {}
    audits = []
    for c in spec['configs']:
        dep, seed = c['deployment'], c['seed']
        w = np.asarray(c['weights'], float);weights[dep, seed] = w/w.sum()
        for split in ('validation', 'test', 'real'):
            baseline = cf.read(dep, seed, split)
            name = 'real_fusion_pairs.parquet' if split == 'real' else split+'_pairs.parquet'
            f = pd.read_parquet(Path(spec['upstream'])/f"evaluation/{spec['method']}/{dep}/seed_{seed}/{name}")
            keys = ['pair_key'] if split == 'real' else ['idx_i', 'idx_j']
            if f.duplicated(keys).any() or baseline.duplicated(keys).any():
                raise RuntimeError('Duplicate pair identity')
            f = f.set_index(keys).reindex(baseline.set_index(keys).index).reset_index()
            if len(f) != len(baseline):
                raise RuntimeError('Pair scope mismatch')
            for col in ('time_score', 'sky_raw_log_bf'):
                if not np.array_equal(f[col].to_numpy(), baseline[col].to_numpy()):
                    raise RuntimeError('Frozen physical channel changed:'+col)
            if not np.isfinite(f.waveform_score).all():
                raise RuntimeError('Invalid waveform')
            key = dep, seed, split
            frames[key], new[key] = baseline, f.waveform_score.to_numpy(float)
            audits.append({'upstream': spec['method'], 'deployment': dep, 'seed': seed, 'split': split,
                           'time_exact': True, 'sky_exact': True, 'pair_identity_exact': True})
    return frames, new, weights, audits


def mix(frame, new, w1, dep, seed, alpha):
    w0 = cf.frozen_weights(dep, seed);w0 /= w0.sum()
    w = (1-alpha)*w0+alpha*w1
    z = ((1-alpha)*w0[0]*frame.waveform_score.to_numpy(float)+alpha*w1[0]*new)/w[0]
    return z, w


def task(spec):
    frames, new, weights, audits = inputs(spec)
    metrics, budgets, rows, success = [], [], [], []
    baselines, external = {}, {}
    for dep in t.DEPS:
        ext = pd.read_parquet(t.EXTERNAL/f'{dep}_external_reference.parquet');external[dep] = ext
        fs = {seed: frames[dep, seed, 'real'] for seed in t.SEEDS}
        consensus = globaldev.consensus(fs, {seed: f.waveform_score.to_numpy(float) for seed, f in fs.items()}, ext, dep)
        for b in (10, 20):
            baselines[dep, b] = dev.budget_row(consensus, 'OMC', dep, 'fusion', b)
    for alpha in ALPHAS:
        code = spec['method']+f'__PATH{alpha:g}'
        valid, testpass = True, True
        for split in ('validation', 'test'):
            if split == 'test' and not valid:
                break
            for dep in t.DEPS:
                for seed in t.SEEDS:
                    key = dep, seed, split;f = frames[key]
                    z, w = mix(f, new[key], weights[dep, seed], dep, seed, alpha)
                    oldz = f.waveform_score.to_numpy(float);oldw = cf.frozen_weights(dep, seed)
                    for mode, value, reference in (
                        ('waveform', z, oldz), ('fusion', cf.channels(f, z)@w, cf.channels(f, oldz)@oldw)):
                        m = cf.fast_metrics(f, value);bm = cf.fast_metrics(f, reference)
                        passed = cf.guard(m, bm)
                        metrics.append({'configuration': code, 'deployment': dep, 'seed': seed, 'split': split,
                                        'mode': mode, 'guard_pass': passed, **m})
                        if split == 'validation':valid &= passed
                        else:testpass &= passed
        row = {'configuration': code, 'upstream': spec['upstream'], 'upstream_method': spec['method'],
               'alpha': alpha, 'validation_guard': bool(valid), 'reused_test_guard': bool(testpass) if valid else None,
               'both_real_target': False}
        if valid:
            br = []
            for dep in t.DEPS:
                ranked = []
                for seed in t.SEEDS:
                    key = dep, seed, 'real';f = frames[key].copy()
                    z, w = mix(f, new[key], weights[dep, seed], dep, seed, alpha)
                    f['waveform_score'] = z
                    ranked.append(dev.BASE.rank_real(f, cf.weights_dict(w), 'fusion', seed))
                result = dev.BASE.consensus_real(ranked, 'fusion')
                ext = external[dep]
                result = result.merge(ext[[c for c in ext if c not in result or c == 'pair_key']], on='pair_key', validate='one_to_one')
                br += [dev.budget_row(result, code, dep, 'fusion', b) for b in (10, 20, 50, 100)]
            passed, details = globaldev.target(br, baselines)
            row['both_real_target'] = passed
            for d in details:
                for key, value in d.items():
                    if key != 'deployment':row[d['deployment']+'_'+key] = value
            budgets += br
            if passed and testpass:
                objective = (-min(d['frontend_gain'] for d in details), -sum(d['Mc_count_gain'] for d in details),
                             -sum(d['Mc_median_gain'] for d in details), -sum(d['frontend_gain'] for d in details), alpha, code)
                success.append({**row, 'objective': objective})
        rows.append(row)
    return rows, budgets, metrics, success, audits


def select(root):
    if (root/'contracts/SEARCH_COMPLETE.json').exists():
        raise RuntimeError('Already completed')
    rows, budgets, metrics, successes, audits = [], [], [], [], []
    with ProcessPoolExecutor(max_workers=8) as pool:
        for a, b, c, d, e in pool.map(task, specs()):
            rows += a;budgets += b;metrics += c;successes += d;audits += e
            dev.csv_write(root/'tables/ALL_GLOBAL_CONFIGURATIONS.csv', pd.DataFrame(rows))
            dev.csv_write(root/'tables/ALL_GLOBAL_PE_OFFICIAL_BUDGETS.csv', pd.DataFrame(budgets))
            dev.csv_write(root/'tables/ALL_REUSED_SIMULATION_METRICS.csv', pd.DataFrame(metrics))
            print(json.dumps({'configurations': len(rows), 'qualifying': len(successes)}), flush=True)
    dev.csv_write(root/'tables/FROZEN_CHANNEL_INPUT_AUDIT.csv', pd.DataFrame(audits))
    dev.json_write(root/'contracts/SEARCH_COMPLETE.json', {'configurations': len(rows), 'qualifying': len(successes),
        'real_used_for_development_selection': True, 'fresh_confirmation': False})
    dev.json_write(root/'contracts/ALL_DEVELOPMENT_QUALIFIERS.json', successes)
    if successes:
        chosen = min(successes, key=lambda s: s['objective'])
        dev.json_write(root/'contracts/SELECTED_GLOBAL_DEVELOPMENT.json', chosen)
    else:
        chosen = None
    dev.json_write(root/'contracts/ROUND_ASSESSMENT.json', {'goal_achieved': False, 'status': t.STATUS,
        'development_candidates_requiring_fresh_confirmation': [chosen['configuration']] if chosen else []})


if __name__ == '__main__':
    p = argparse.ArgumentParser();p.add_argument('--root', type=Path, required=True)
    p.add_argument('--stage', choices=['initialize', 'select'], required=True)
    a = p.parse_args();globals()[a.stage](a.root)
