#!/usr/bin/env python3
"""R55 read-only comparison and exact policy-change localization."""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import json
from pathlib import Path
import shutil
import sys
import numpy as np
import pandas as pd

P = Path('/root/autodl-tmp/gw-catalog')
sys.path.insert(0, str(P/'scripts/experiments'))
import mcwf_shared_profile_monotone_boundary_20260909 as app
import mcwf_shared_profile_catalog_audit_20260909 as old
n = app.n
NEW = app.NEW


def preflight(root, source):
    target = root/'audit/REFERENCE_AUDIT_FROZEN.json'
    if target.exists() or (root/'contracts/EVALUATION_COMPLETE.json').exists():
        raise RuntimeError('Freeze checks once, before catalog evaluation')
    inherited = source/'audit/REFERENCE_AUDIT_FROZEN.json'
    rules = json.loads(inherited.read_text())
    rules.update(UTC=n.utc(), primary_arm=NEW, sensitivity_arm='R51 frozen replay',
                 inherited_from=str(inherited), inherited_sha256=n.sha(inherited),
                 extra_required=['R51 replay exact', 'changes confined to eligible low-D pairs',
                                 'real waveform/final/rank exactly unchanged relative to R51'],
                 runtime_sha256=n.sha(Path(__file__)), goal_achieved=False)
    n.write_json(target, rules)
    rows = pd.read_csv(source/'manifest/REFERENCE_INPUT_SHA256.csv').to_dict('records')
    for path in (source/'results/SHARED-PROFILE-SINGLE-WF').rglob('*.parquet'):
        rows.append({'path': str(path), 'sha256': n.sha(path), 'bytes': path.stat().st_size})
    n.write_csv(root/'manifest/REFERENCE_INPUT_SHA256.csv', rows)
    shutil.copy2(__file__, root/'scripts/shared_boundary_audit.py')
    print('REFERENCE_AUDIT_FROZEN', len(rows), flush=True)


def compare(root, source):
    if not (root/'contracts/REAL_COMPLETE.json').exists():
        raise RuntimeError('Finish injection and real replay first')
    if (root/'audit/REFERENCE_COMPARISON_COMPLETE.json').exists():
        raise RuntimeError('Do not overwrite completed audit')
    rules = json.loads((root/'audit/REFERENCE_AUDIT_FROZEN.json').read_text())
    if rules['runtime_sha256'] != n.sha(Path(__file__)):
        raise RuntimeError('Frozen audit changed')
    checks, guards, changed, keyrows = [], [], [], []
    configs = {(c['deployment'], c['seed'], c['method']): c for c in n.selections(root)}
    for method in app.METHODS:
        files = list((root/f'results/{method}').glob('gwtc*/seed_*/*/pairs.parquet'))
        files += list((root/f'results/{method}').glob('gwtc*/seed_*/real/fusion_all_pairs.parquet'))
        if len(files) != 36:
            raise RuntimeError('Incomplete panel coverage')
        for path in sorted(files):
            rel = path.relative_to(root/f'results/{method}')
            dep, seedname, panel = rel.parts[:3]
            seed = int(seedname.split('_')[1])
            f = old.aligned(path)
            b = old.aligned(root/'results/NODUP-DIRECT-REPLAY'/rel)
            prior = old.aligned(source/'results/SHARED-PROFILE-SINGLE-WF'/rel)
            historical = old.aligned(app.parent.BASE/'results/NODUP-DIRECT'/rel)
            if not b[['waveform_score','final_score']].equals(historical[['waveform_score','final_score']]):
                raise RuntimeError('NODUP replay changed')
            fixed = ['idx_i','idx_j','time_score','sky_raw_log_bf','embedding_only']
            fixed += [c for c in ('time_contribution','sky_contribution') if c in f]
            if not f[fixed].equals(b[fixed]):
                raise RuntimeError('Frozen channel or identity changed')
            cfg = configs[dep, seed, method]
            if cfg['weights'] != configs[dep, seed, app.METHODS[0]]['weights']:
                raise RuntimeError('Outer weights changed')
            score = n.cf.channels(f, f.waveform_score.to_numpy(float))@np.asarray(cfg['weights'])
            if np.max(np.abs(score-f.final_score)) > 1e-12:
                raise RuntimeError('Three-channel total does not reconstruct')
            eligible = f.shared_profile_eligible.to_numpy(bool)
            if not np.array_equal(f.waveform_score[~eligible], b.waveform_score[~eligible]):
                raise RuntimeError('Inactive fallback changed')
            delta = f.waveform_score.to_numpy()-prior.waveform_score.to_numpy()
            mask = delta != 0
            if method == 'SHARED-PROFILE-SINGLE-WF' and mask.any():
                raise RuntimeError('R51 waveform replay changed')
            if method == NEW:
                permitted = eligible & (f.shared_profile_deficit.to_numpy() < cfg['deficit_support'][0])
                if np.any(mask & ~permitted):
                    raise RuntimeError('Change outside preregistered low-D boundary')
                if np.any(delta < -1e-12):
                    raise RuntimeError('Low-D saturation lowered support')
                for pos in np.flatnonzero(mask):
                    row = f.iloc[pos].to_dict()
                    changed.append({**row, 'deployment':dep, 'seed':seed, 'panel':panel,
                        'previous_waveform':float(prior.waveform_score.iloc[pos]),
                        'new_waveform':float(f.waveform_score.iloc[pos]),
                        'fit_D_min':cfg['deficit_support'][0]})
            if panel == 'real':
                external = [c for c in b if c.startswith(('pe_','official_'))]
                if not f[external].equals(b[external]):
                    raise RuntimeError('External audit values changed')
                if method != app.METHODS[0]:
                    columns = ['waveform_score','final_score','rank']
                    if not f[columns].equals(prior[columns]):
                        raise RuntimeError('Unexpected real ranking change')
                keyrows.extend({**row,'method':method,'seed':seed,'unit':'model'}
                    for row in f[f.pair_key.eq(old.KEY)].to_dict('records'))
            else:
                for reference in (app.METHODS[0], 'SHARED-PROFILE-SINGLE-WF', n.BASELINE):
                    other = old.aligned(root/'results'/reference/rel)
                    for mode, column in [('waveform','waveform_score'),('fusion','final_score')]:
                        a = n.cf.fast_metrics(f, f[column].to_numpy(float))
                        baseline = n.cf.fast_metrics(other, other[column].to_numpy(float))
                        strict = all(a[k] >= baseline[k]-1e-12 for k in ('macro_r_at_10','average_precision'))
                        strict &= all(a[k] <= baseline[k] for k in ('false_at_recall_0p5','false_at_recall_0p9'))
                        guards.append({'method':method,'deployment':dep,'seed':seed,'panel':panel,
                            'mode':mode,'reference':reference,'existing_guard':bool(n.cf.guard(a,baseline)),
                            'strict_pointwise_no_loss':bool(strict),
                            **{k+'_delta':a[k]-baseline[k] for k in old.KEY_METRICS}})
            checks.append({'method':method,'deployment':dep,'seed':seed,'panel':panel,
                'pairs':len(f),'changed_from_R51':int(mask.sum()),'frozen_channels_exact':True})
        for dep in n.DEPS:
            f = pd.read_parquet(root/f'results/{method}/{dep}/consensus/fusion_all_pairs.parquet')
            if method != app.METHODS[0]:
                prior = pd.read_parquet(source/f'results/SHARED-PROFILE-SINGLE-WF/{dep}/consensus/fusion_all_pairs.parquet')
                cols = ['pair_key','consensus_rank','rank_mean','final_score_mean']
                if not f[cols].equals(prior[cols]):
                    raise RuntimeError('Real consensus changed')
            keyrows.extend({**row,'method':method,'seed':'consensus','unit':'consensus'}
                for row in f[f.pair_key.eq(old.KEY)].to_dict('records'))
    budgets = []
    for name, unit in [('PE_OFFICIAL_BUDGETS.csv','consensus'),('PER_SEED_PE_OFFICIAL_BUDGETS.csv','model')]:
        f = pd.read_csv(root/'tables'/name)
        f['unit'] = unit
        if unit == 'consensus':
            f['seed'] = 'consensus'
        budgets.append(f)
    budgets = pd.concat(budgets, ignore_index=True)
    pe = []
    for row in budgets[budgets.budget.isin([10,20])].to_dict('records'):
        for reference in (app.METHODS[0], 'SHARED-PROFILE-SINGLE-WF', n.BASELINE):
            match = budgets[(budgets.config == reference)&(budgets.deployment == row['deployment'])&
                (budgets.budget == row['budget'])&(budgets.unit == row['unit'])&
                (budgets.seed.astype(str) == str(row['seed']))]
            if len(match) != 1:
                raise RuntimeError('Ambiguous budget reference')
            b = match.iloc[0]
            passfields = {k:row[k]>=b[k]-1e-12 for k in old.HIGHER}
            passfields['catastrophic_mc'] = row['catastrophic_mc']<=b['catastrophic_mc']
            pe.append({'method':row['config'],'deployment':row['deployment'],'unit':row['unit'],
                'seed':row['seed'],'budget':row['budget'],'reference':reference,
                'all_no_loss':all(passfields.values()),
                **{k+'_pass':bool(v) for k,v in passfields.items()},
                **{k+'_delta':float(row[k]-b[k]) for k in (*old.HIGHER,'catastrophic_mc')}})
    hashes = []
    for filename in ('REFERENCE_INPUT_SHA256.csv','INPUT_SHA256.csv'):
        for row in pd.read_csv(root/'manifest'/filename).itertuples():
            hashes.append({'path':row.path,'before':row.sha256,'after':n.sha(Path(row.path))})
    if any(r['before'] != r['after'] for r in hashes):
        raise RuntimeError('Frozen input hash changed')
    for name, records in [('PAIR_ALIGNED_INVARIANCE',checks),('FINAL_INPUT_HASH_CHECK',hashes)]:
        n.write_csv(root/f'audit/{name}.csv',records)
    for name, records in [('REFERENCE_SPECIFIC_INJECTION_GUARDS',guards),
                         ('REFERENCE_SPECIFIC_PE_OFFICIAL_GUARDS',pe),
                         ('BOUNDARY_CHANGED_PAIRS',changed),('CRITICAL_PAIR_ALL_RANKS',keyrows)]:
        n.write_csv(root/f'tables/{name}.csv',records)
    n.write_json(root/'audit/REFERENCE_COMPARISON_COMPLETE.json',{'UTC':n.utc(),
        'panels':len(checks),'changed_model_panel_rows':len(changed),'hash_checks':len(hashes),
        'real_ranks_exact_R51':True,'goal_achieved':False,'status':n.STATUS})
    print('BOUNDARY_COMPARISON_COMPLETE',len(checks),len(changed),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--source-root',type=Path,required=True)
    parser.add_argument('--stage',choices=('preflight','compare'),required=True)
    args=parser.parse_args()
    globals()[args.stage](args.root,args.source_root)
