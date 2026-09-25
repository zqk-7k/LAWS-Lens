#!/usr/bin/env python3
"""Finite, explicitly adaptive audit of every simulation-eligible operating point."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
import argparse
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import mcwf_pe_frontend_extension_v2_20260907 as e
import mcwf_fine_mass_only_evaluate_20260907 as single
import mcwf_predictive_mass_pool_evaluate_20260907 as pooled
import mcwf_seed_plateau_20260907 as shared
import mcwf_runwise_development_20260907 as rw

dev, body, ev = e.dev, e.body, e.ev


def batch_audit(combos, pools, pe, baseline, dep):
    values = {'catastrophic_mc': ((pe.pe_mc_bhattacharyya_coefficient < .1) | (pe.pe_mc_standardized_distance > 5)).to_numpy(),
              'BC_mc_ge_0p5': (pe.pe_mc_bhattacharyya_coefficient >= .5).to_numpy(),
              'Dmax_le_3': (pe.pe_dmax_intrinsic <= 3).to_numpy(),
              'official_frontend': pe['official_po_or_ml_fpp_below_0p01' if dep == 'gwtc3' else 'official_po_or_phazap_fpp_below_0p01'].fillna(False).to_numpy(bool),
              'official_hanabi': pe.official_any_pair_resolved_hanabi_overlap.fillna(False).to_numpy(bool)}
    out = {'PE_cells': np.zeros(len(combos), int), 'official_cells': np.zeros(len(combos), int),
           'BC_gain': np.zeros(len(combos), int), 'Dmax_gain': np.zeros(len(combos), int),
           'front_gain': np.zeros(len(combos), int), 'hanabi_gain': np.zeros(len(combos), int)}
    for method in ('C_fixed', 'waveform_only'):
        ranks = [np.stack([c['rank_arrays'][method][0] for c in pool])[combos[:, k]] for k, pool in enumerate(pools)]
        scores = [np.stack([c['rank_arrays'][method][1] for c in pool])[combos[:, k]] for k, pool in enumerate(pools)]
        rankmean = np.mean(ranks, axis=0)
        rankmax = np.max(ranks, axis=0)
        scoremean = np.mean(scores, axis=0)
        index = np.lexsort((np.broadcast_to(np.arange(len(pe)), rankmean.shape), -scoremean, rankmax, rankmean), axis=1)
        for budget in ((10, 20) if method == 'C_fixed' else (10,)):
            top = index[:, :budget]
            old = baseline[(baseline.deployment == dep) & (baseline.method == method) & (baseline.budget == budget)].iloc[0]
            count = {k: v[top].sum(1) for k, v in values.items()}
            median = np.median(pe.pe_mc_bhattacharyya_coefficient.to_numpy(float)[top], axis=1)
            if method == 'waveform_only':
                out['waveform_PE_pass'] = (count['catastrophic_mc'] == 0) & (count['BC_mc_ge_0p5'] >= old.BC_mc_ge_0p5) & (count['Dmax_le_3'] >= old.Dmax_le_3)
                continue
            pe_ok = (count['catastrophic_mc'] <= old.catastrophic_mc) & (count['BC_mc_ge_0p5'] >= old.BC_mc_ge_0p5) & (count['Dmax_le_3'] >= old.Dmax_le_3) & (median >= old.median_BC_mc - .02)
            off_ok = (count['official_frontend'] >= old.official_frontend + 1) & (count['official_hanabi'] >= old.official_hanabi + 1)
            out['PE_cells'] += pe_ok
            out['official_cells'] += off_ok
            for metric, key in [('BC_mc_ge_0p5', 'BC_gain'), ('Dmax_le_3', 'Dmax_gain'), ('official_frontend', 'front_gain'), ('official_hanabi', 'hanabi_gain')]:
                out[key] += count[metric] - old[metric]
            for key, value in count.items():
                out[f'Top{budget}_{key}'] = value
            out[f'Top{budget}_median_BC_mc'] = median
    out['target_pass'] = (out['PE_cells'] == 2) & (out['official_cells'] == 2) & out['waveform_PE_pass']
    return out


def run(root, model, arm):
    module = single if model == 'fine' else pooled
    if model != 'fine':
        module.mass_model.MODE = model
    code = f'MASS-COMPLETE-GRID-{model.upper()}-{arm}'
    trial = root / 'trials' / code
    if trial.exists():
        raise RuntimeError('Independent trial required')
    for name in ('contracts', 'configs', 'tables', 'evaluation', 'results'):
        (trial / name).mkdir(parents=True)
    shutil.copy2(__file__, trial / 'contracts' / Path(__file__).name)
    dev.json_write(trial / 'contracts/RECIPE.json', {
        'created_utc': datetime.now(timezone.utc).isoformat(), 'same_O3_O4a_global_method': [model, arm],
        'formula': 'identical preceding mass-only evidence formula; no new event features',
        'grid': {'gamma': module.GAMMAS, 'beta': module.BETAS},
        'retain': 'ALL operating points passing original validation and reused-injection guards; no top4 truncation',
        'real_feedback': 'EXPLICIT adaptive-development hyperparameter selection using frozen PE/official budget outcomes, NOT blind validation',
        'no_PE_official_ID_scoring_inputs': True,
        'ranking': 'unchanged seed rank-mean consensus with max-rank, negative mean-score and lexical ties',
        'selection': 'unchanged frozen target, then PE gains, minimum official gain, total official gain, weakest coefficient change, lexical ties',
        'fresh_independent_confirmation_required': True,
        'frozen': ['time', 'sky', 'outer weights', 'scope', 'historical outputs', 'target thresholds']})
    baseline = pd.read_csv(root / 'contracts/BASELINE_BUDGETS.csv')
    baseline = baseline[baseline.seed.astype(str) == 'consensus']
    data, selected, grid_rows, combo_rows = {}, {}, [], []
    for dep in e.DEPS:
        pe = pd.read_parquet(root / f'audit/{dep}_frozen_external_reference.parquet').sort_values('pair_key').reset_index(drop=True)
        keys, pools = pe.pair_key.tolist(), []
        for ms, es in zip(body.MODEL_SEEDS, dev.SEEDS):
            fs, xs, bm = {}, {}, {}
            for split in ('validation', 'test', 'real'):
                fs[split], xs[split] = module.values(root, dep, ms, es, split)
                if split != 'real':
                    bm[split] = ev.metrics(fs[split], fs[split].waveform_score.to_numpy(float), dep, es)
            data[dep, es] = fs, xs, bm
            cal = module.calibrate(root, dep, ms)
            pool = []
            for gamma in module.GAMMAS:
                for beta in module.BETAS:
                    spec = {**cal, 'gamma': gamma, 'beta': beta}
                    z, _, _ = module.score(fs['validation'], xs['validation'], spec, arm)
                    vm = ev.metrics(fs['validation'], z, dep, es)
                    vp, tp = ev.guard(vm, bm['validation']), False
                    if vp:
                        z, _, _ = module.score(fs['test'], xs['test'], spec, arm)
                        tp = ev.guard(ev.metrics(fs['test'], z, dep, es), bm['test'])
                    ident = f'G{gamma:g}_B{beta:g}'
                    grid_rows.append({'deployment': dep, 'seed': es, 'id': ident, 'gamma': gamma, 'beta': beta,
                                      'validation_pass': vp, 'reused_test_pass': tp,
                                      **{a + '_' + k: v for a, b in vm.items() for k, v in b.items()}})
                    if vp and tp:
                        pool.append({'id': ident, 'spec': spec})
            dev.json_write(trial / f'configs/{dep}_{es}_ELIGIBLE.json', {'eligible': pool})
            for c in pool:
                z, _, _ = module.score(fs['real'], xs['real'], c['spec'], arm)
                c['rank_arrays'] = shared.rank_arrays(fs['real'], z, dep, es, keys)
            pools.append(pool)
        dev.csv_write(trial / 'tables/VALIDATION_GRID.csv', pd.DataFrame(grid_rows))
        combos = np.array(list(itertools.product(*(range(len(p)) for p in pools))), int)
        qualified = []
        for start in range(0, len(combos), 128):
            block = combos[start:start + 128]
            result = batch_audit(block, pools, pe, baseline, dep)
            if start == 0:
                slow = rw.target(shared.fast_budgets([pool[c] for pool, c in zip(pools, block[0])], pe, dep), baseline, dep)
                for key in ('target_pass', 'PE_cells', 'official_cells', 'waveform_PE_pass', 'BC_gain', 'Dmax_gain', 'front_gain', 'hanabi_gain'):
                    if slow[key] != result[key][0]:
                        raise RuntimeError('Vectorized consensus audit mismatch: ' + key)
                dev.json_write(trial / f'contracts/{dep}_VECTOR_AUDIT_TEST.json', {'passed': True, 'exact_reference_agreement': True})
            for k, choice in enumerate(block):
                ids = [pool[c]['id'] for pool, c in zip(pools, choice)]
                row = {'deployment': dep, 'combination': start + k, 'seed1': ids[0], 'seed2': ids[1], 'seed3': ids[2],
                       **{key: value[k].item() for key, value in result.items()}}
                combo_rows.append(row)
                if row['target_pass']:
                    records = [pool[c] for pool, c in zip(pools, choice)]
                    balanced = min(row[f'Top{b}_{m}'] - baseline[(baseline.deployment == dep) & (baseline.method == 'C_fixed') & (baseline.budget == b)].iloc[0][m]
                                   for b in (10, 20) for m in ('official_frontend', 'official_hanabi'))
                    key = (-row['BC_gain'] - row['Dmax_gain'], -balanced, -row['front_gain'] - row['hanabi_gain'],
                           sum(c['spec']['gamma'] + c['spec']['beta'] for c in records), ids)
                    qualified.append((key, records, row))
        dev.csv_write(trial / 'tables/ALL_COMBINATIONS.csv', pd.DataFrame(combo_rows))
        if qualified:
            _, records, row = min(qualified, key=lambda a: a[0])
            selected[dep] = {es: record['spec'] for es, record in zip(dev.SEEDS, records)}
            dev.json_write(trial / f'contracts/{dep}_SELECTED.json', {'configs': selected[dep], 'development_outcome': row})
        print(json.dumps({'complete_mass_grid': code, 'deployment': dep, 'pools': [len(p) for p in pools],
                          'combinations': len(combos), 'qualifying': len(qualified)}), flush=True)
    dev.json_write(trial / 'contracts/SEARCH_COMPLETE.json', {'both_runs_qualify': len(selected) == 2,
                   'selected': selected, 'fresh_confirmation_done': False})
    if len(selected) != 2:
        return
    metrics, guards = [], []
    for dep in e.DEPS:
        real = {}
        for es in dev.SEEDS:
            fs, xs, bm = data[dep, es]
            for split, f in fs.items():
                z, penalty, inc = module.score(f, xs[split], selected[dep][es], arm)
                n = f.rename(columns={c: 'baseline_' + c for c in ('final_score', 'rank', 'method', 'waveform_contribution', 'time_contribution', 'sky_contribution') if c in f}).copy()
                n['FRT_baseline_waveform_score'], n['waveform_score'] = f.waveform_score, z
                n['mass_penalty'], n['mass_increment'] = penalty, inc
                out = trial / f'evaluation/{dep}/seed_{es}'
                out.mkdir(parents=True, exist_ok=True)
                n.to_parquet(out / f'{split}_pairs.parquet', index=False)
                if split == 'real':
                    real[es] = n
                else:
                    mm = ev.metrics(f, z, dep, es)
                    guards.append({'deployment': dep, 'seed': es, 'split': split, 'pass': ev.guard(mm, bm[split])})
                    for method in mm:
                        for config, m in [('FRT_BASELINE', bm[split][method]), ('CANDIDATE', mm[method])]:
                            metrics.append({'deployment': dep, 'seed': es, 'split': split, 'method': method, 'config': config, **m})
        dev.save_evaluation(trial, 'CANDIDATE', dep, real, pe=pd.read_parquet(root / f'audit/{dep}_frozen_external_reference.parquet'))
    dev.csv_write(trial / 'tables/RETRIEVAL_PER_SEED.csv', pd.DataFrame(metrics))
    dev.csv_write(trial / 'tables/REUSED_GUARDRAILS.csv', pd.DataFrame(guards))
    e.assess(root, trial)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    for model in ('fine', 'mixture', 'geometric'):
        for arm in ('BC', 'PRIOR'):
            run(args.root, model, arm)
