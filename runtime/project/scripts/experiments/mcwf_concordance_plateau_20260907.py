#!/usr/bin/env python3
"""Finite, explicitly adaptive development audit of concordant waveform scores."""
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
import mcwf_concordant_dense_20260907 as dense
import mcwf_finite_reference_tail_20260907 as tail
import mcwf_seed_plateau_20260907 as shared
import mcwf_runwise_development_20260907 as rw

dev, body, ev = e.dev, e.body, e.ev


def score(f, x, spec):
    if spec['gate_mode'] == 'positive_only':
        return dense.score(f, x, spec)
    raw = x['profile']
    value = np.interp(raw, spec['knots'], spec['loglr'])
    ood = (raw < spec['fit_min']) | (raw > spec['fit_max'])
    value = np.clip(np.where(ood, np.minimum(value, 0.), value), -4, 4)
    old = tail.tail_probability(x['old_mass'], spec['old_mass_reference'])
    new = tail.tail_probability(x['new_mass'], spec['new_mass_reference'])
    gate = np.minimum(old, new)**spec['power']
    return f.waveform_score.to_numpy(float)+spec['beta']*value*gate, value*gate, gate, ood


def run(root, kind, gate_mode):
    code = f'CONCORDANCE-PLATEAU-{kind.upper()}-{gate_mode.upper()}'
    trial = root / 'trials' / code
    if trial.exists():
        raise RuntimeError('Independent trial required')
    for name in ('contracts', 'configs', 'tables', 'evaluation', 'results'):
        (trial / name).mkdir(parents=True)
    shutil.copy2(__file__, trial / 'contracts' / Path(__file__).name)
    dev.json_write(trial / 'contracts/ADAPTIVE_DEVELOPMENT_ADDENDUM.json', {
        'created_utc': datetime.now(timezone.utc).isoformat(), 'same_O3_O4a_kind_and_gate_mode': [kind, gate_mode],
        'formula': 'FRT plus dense intrinsic-profile increment gated by min(old,new waveform mass-concordance tail)^power',
        'gate_semantics': 'positive_only attenuates only reward; both_signs attenuates reward and penalty. These are separate global recipe arms, not run-specific methods.',
        'rationale': 'conditional q/spin profile evidence is unreliable when the waveform mass hypotheses are poorly shared; retain the existing FRT mass penalty instead of counting unsupported profile evidence again',
        'beta_grid': dense.BETAS, 'power_grid': dense.POWERS,
        'plateau': 'unchanged FRT plus up to4 best validation points per power passing all reused-injection guards,maximum13choices/seed',
        'maximum_combinations_per_run': 13**3,
        'real_usage': 'EXPLICIT ADAPTIVE DEVELOPMENT selection by the frozen real PE/official budget goals; NOT validation-only,NOT blindtest',
        'tie_order': ['PE_count_gains', 'minimum_official_gain', 'total_official_gain', 'minimum_coefficient_change', 'fixed_lexical_order'],
        'forbidden': 'PE values,official membership,eventIDs or handmade pair masks as scoring inputs',
        'frozen': ['time', 'sky', 'outerweights', 'scope', 'historical outputs', 'target criteria'],
        'fresh_confirmation_required': True,
        'multiplicity': 'all grids and combinations retained; no real-catalog significance claim'})
    baseline = pd.read_csv(root / 'contracts/BASELINE_BUDGETS.csv')
    baseline = baseline[baseline.seed.astype(str) == 'consensus']
    selected, data, allrows, combinations = {}, {}, [], []
    for dep in e.DEPS:
        pe = pd.read_parquet(root / f'audit/{dep}_frozen_external_reference.parquet').sort_values('pair_key').reset_index(drop=True)
        keys = pe.pair_key.to_list()
        cal = json.loads((root / f'dense_profile/{dep}/FIT.json').read_text())[kind]
        pools = []
        for ms, es in zip(body.MODEL_SEEDS, dev.SEEDS):
            fs, xs, bm = {}, {}, {}
            for split in ('validation', 'test', 'real'):
                fs[split], xs[split] = dense.get(root, dep, ms, es, split, kind)
                if split != 'real':
                    bm[split] = ev.metrics(fs[split], fs[split].waveform_score.to_numpy(float), dep, es)
            data[dep, es] = fs, xs, bm
            y = fs['validation'].is_true_pair.to_numpy(bool)
            ref = {**cal, 'gate_mode': gate_mode,
                'old_mass_reference': np.sort(xs['validation']['old_mass'][y]).tolist(),
                'new_mass_reference': np.sort(xs['validation']['new_mass'][y]).tolist()}
            eligible = []
            for power in dense.POWERS:
                for beta in dense.BETAS:
                    if beta == 0 and power != dense.POWERS[0]:
                        continue
                    spec = {**ref, 'power': power, 'beta': beta}
                    z, *_ = score(fs['validation'], xs['validation'], spec)
                    vm = ev.metrics(fs['validation'], z, dep, es)
                    vp, tp = ev.guard(vm, bm['validation']), False
                    if vp:
                        z, *_ = score(fs['test'], xs['test'], spec)
                        tp = ev.guard(ev.metrics(fs['test'], z, dep, es), bm['test'])
                    ident = f'P{power:g}_B{beta:g}'
                    allrows.append({'deployment': dep, 'seed': es, 'id': ident, 'power': power, 'beta': beta,
                        'validation_pass': vp, 'reused_test_pass': tp,
                        **{a+'_'+k: v for a, b in vm.items() for k, v in b.items()}})
                    if vp and tp:
                        eligible.append({'id': ident, 'spec': spec, 'objective': e.old_selection.objective(vm, beta, power)})
            eligible.sort(key=lambda c: (c['objective'], c['id']))
            base = next(c for c in eligible if c['spec']['beta'] == 0)
            pool = [base]
            for power in dense.POWERS:
                pool.extend([c for c in eligible if c['spec']['power'] == power and c['spec']['beta'] > 0][:4])
            dev.json_write(trial / f'configs/{dep}_{es}_PLATEAU.json', {'eligible': len(eligible), 'retained': pool,
                'real_used_to_construct_plateau': False})
            for c in pool:
                z, *_ = score(fs['real'], xs['real'], c['spec'])
                c['rank_arrays'] = shared.rank_arrays(fs['real'], z, dep, es, keys)
            pools.append(pool)
        dev.csv_write(trial / 'tables/VALIDATION_GRID.csv', pd.DataFrame(allrows))
        qualified = []
        for ci, combo in enumerate(itertools.product(*pools)):
            budgets = shared.fast_budgets(combo, pe, dep)
            result = rw.target(budgets, baseline, dep)
            ids = [c['id'] for c in combo]
            combinations.append({'deployment': dep, 'combination': ci,
                'seed1': ids[0], 'seed2': ids[1], 'seed3': ids[2], **{k: v for k, v in result.items() if k != 'details'}})
            if result['target_pass']:
                balanced = min(min(c['front_delta'], c['hanabi_delta']) for c in result['details'])
                key = (-result['BC_gain']-result['Dmax_gain'], -balanced,
                    -result['front_gain']-result['hanabi_gain'], sum(c['spec']['beta'] for c in combo), ids)
                qualified.append((key, combo, result))
        dev.csv_write(trial / 'tables/ALL_COMBINATIONS.csv', pd.DataFrame(combinations))
        if qualified:
            _, combo, result = min(qualified, key=lambda x: x[0])
            selected[dep] = {es: c['spec'] for es, c in zip(dev.SEEDS, combo)}
            dev.json_write(trial / f'contracts/{dep}_SELECTED.json', {'configs': selected[dep],
                'development_target': result, 'qualifying': len(qualified)})
        print(json.dumps({'concordance_plateau': code, 'deployment': dep, 'pools': [len(p) for p in pools],
            'qualifying': len(qualified), 'selected': selected.get(dep)}), flush=True)
    dev.json_write(trial / 'contracts/SEARCH_COMPLETE.json', {'both_runs_qualify': len(selected) == 2,
        'selected': selected, 'fresh_confirmation_done': False})
    if len(selected) != 2:
        return
    rows, guards = [], []
    for dep in e.DEPS:
        real = {}
        for es in dev.SEEDS:
            fs, xs, bm = data[dep, es]
            for split, f in fs.items():
                z, inc, gate, ood = score(f, xs[split], selected[dep][es])
                n = f.copy()
                n['FRT_baseline_waveform_score'] = f.waveform_score
                n['waveform_score'], n['concordant_increment'], n['mass_concordance_gate'], n['dense_profile_ood'] = z, inc, gate, ood
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
                            rows.append({'deployment': dep, 'seed': es, 'split': split, 'method': method, 'config': config, **m})
        dev.save_evaluation(trial, 'CANDIDATE', dep, real, pd.read_parquet(root / f'audit/{dep}_frozen_external_reference.parquet'))
    dev.csv_write(trial / 'tables/RETRIEVAL_PER_SEED.csv', pd.DataFrame(rows))
    dev.csv_write(trial / 'tables/REUSED_GUARDRAILS.csv', pd.DataFrame(guards))
    e.assess(root, trial)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--kind', choices=['full', 'conditional_q_spin'], required=True)
    p.add_argument('--gate-mode', choices=['positive_only', 'both_signs'], required=True)
    a = p.parse_args()
    run(a.root, a.kind, a.gate_mode)
