#!/usr/bin/env python3
"""Export the frozen common PATH25 method without any further selection."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

P = Path('/root/autodl-tmp/gw-catalog')
sys.path.insert(0, str(P/'scripts/experiments'))
import mcwf_conservative_fusion_path_20260908 as pathmodel
import mcwf_temporal_response_evaluate_20260908 as ev
t, dev, cf = ev.t, ev.dev, ev.cf
CODE = 'MCWF-UNIFIED-PATH875-DEVCONF'
BUDGETS = (10, 20, 50, 100)
KNOWN = (
    'GW190924_021846--GW191105_143521',
    'GW190412--GW191204_171526',
    'GW190924_021846--GW190930_133541',
    'GW190728_064510--GW191204_171526',
)


def join_external(frame, external):
    result = frame.merge(external[[c for c in external if c not in frame or c == 'pair_key']],
                         on='pair_key', validate='one_to_one', how='left')
    if result.pe_mc_bhattacharyya_coefficient.isna().any():
        raise RuntimeError('Incomplete frozen PE audit')
    return result


def export(root):
    if (root/'contracts/SELECTED_EXPORT_COMPLETE.json').exists():
        raise RuntimeError('Export already complete; do not overwrite')
    selected = json.loads((root/'contracts/SELECTED_GLOBAL_DEVELOPMENT.json').read_text())
    if selected['alpha'] != .875:
        raise RuntimeError('Unexpected frozen configuration')
    spec = next(s for s in pathmodel.specs() if s['method'] == selected['upstream_method']
                and s['upstream'] == selected['upstream'])
    frames, incoming, weights, audits = pathmodel.inputs(spec)
    records, budgets, weightrows, distributions, correlations = [], [], [], [], []
    consensus = {}
    (root/'scripts').mkdir(exist_ok=True)
    shutil.copy2(__file__, root/'scripts/path_release_v3.py')
    dev.json_write(root/'contracts/EXPORT_IMPLEMENTATION_FREEZE_v3.json', {
        'UTC': datetime.now(timezone.utc).isoformat(), 'code_sha256': dev.sha(Path(__file__)),
        'selected_sha256': dev.sha(root/'contracts/SELECTED_GLOBAL_DEVELOPMENT.json'),
        'code': CODE, 'no_new_selection': True, 'real_outcomes_are_adaptive_development': True})
    for dep in t.DEPS:
        external = pd.read_parquet(t.EXTERNAL/f'{dep}_external_reference.parquet')
        real = {(name, mode): [] for name in ('OMC', CODE) for mode in ('waveform', 'fusion')}
        for seed in t.SEEDS:
            for split in ('validation', 'test', 'real'):
                key = dep, seed, split
                b = frames[key].copy()
                z, w = pathmodel.mix(b, incoming[key], weights[dep, seed], dep, seed, selected['alpha'])
                c = b.copy()
                c['retained_OMC_waveform'] = b.waveform_score
                c['upstream_joint_waveform'] = incoming[key]
                c['waveform_score'] = z
                w0 = cf.frozen_weights(dep, seed)
                if split == 'validation':
                    weightrows.append({'deployment': dep, 'seed': seed, 'alpha': selected['alpha'],
                        **{f'old_weight_{k}': float(v) for k, v in zip(('waveform','time','sky'), w0)},
                        **{f'old_normalized_weight_{k}': float(v) for k, v in zip(('waveform','time','sky'), w0/w0.sum())},
                        **{f'upstream_weight_{k}': float(v) for k, v in zip(('waveform','time','sky'), weights[dep,seed])},
                        **{f'effective_weight_{k}': float(v) for k, v in zip(('waveform','time','sky'), w)}})
                for name, f, ww in (('OMC', b, w0), (CODE, c, w)):
                    folder = root/f'evaluation/{name}/{dep}/seed_{seed}'
                    folder.mkdir(parents=True, exist_ok=True)
                    f['waveform_contribution'] = f.waveform_score.to_numpy(float)*ww[0]
                    f['time_contribution'] = f.time_score.to_numpy(float)*ww[1]
                    f['sky_contribution'] = f.sky_raw_log_bf.to_numpy(float)*ww[2]
                    f['final_score'] = cf.channels(f, f.waveform_score.to_numpy(float))@ww
                    if not np.allclose(f[['waveform_contribution','time_contribution','sky_contribution']].sum(axis=1), f.final_score, atol=1e-12, rtol=1e-12):
                        raise RuntimeError('Contribution reconstruction failed')
                    if split != 'real':
                        f.to_parquet(folder/f'{split}_pairs.parquet', index=False)
                        for mode, scores in (('waveform', f.waveform_score.to_numpy(float)), ('fusion', f.final_score.to_numpy(float))):
                            m = dev.BASE.full_metrics(f, scores)
                            records.append({'configuration': name, 'deployment': dep, 'seed': seed,
                                            'split': split, 'mode': mode, **m})
                    else:
                        for mode, ranking_weights in (('waveform', np.array([1.,0.,0.])), ('fusion', ww)):
                            ranked = dev.BASE.rank_real(f, cf.weights_dict(ranking_weights), mode, seed)
                            real[name,mode].append(ranked)
                            audited = join_external(ranked, external)
                            audited.to_parquet(folder/f'real_{mode}_pairs.parquet', index=False)
                            dev.csv_write(folder/f'real_{mode}_top100.csv', audited.head(100))
                            for budget in BUDGETS:
                                budgets.append(dev.budget_row(audited, name, dep, mode, budget, seed))
                    for channel in ('waveform_score', 'time_score', 'sky_raw_log_bf'):
                        labels = f.is_true_pair.to_numpy(bool) if split != 'real' else None
                        groups = [('all_pairs', f)] if split == 'real' else [('true', f[labels]), ('null', f[~labels])]
                        for label, group in groups:
                            value = group[channel].to_numpy(float)
                            distributions.append({'configuration': name, 'deployment': dep, 'seed': seed,
                                'split': split, 'population': label, 'channel': channel, 'n': len(value),
                                'mean': float(value.mean()), 'sd': float(value.std(ddof=1)),
                                **{f'q{q:g}': float(np.quantile(value,q)) for q in (0,.01,.1,.25,.5,.75,.9,.99,1)}})
                for column in ('time_score','sky_raw_log_bf'):
                    if not np.array_equal(b[column], c[column]):
                        raise RuntimeError('Physical channel changed')
        for (name, mode), items in real.items():
            out = join_external(dev.BASE.consensus_real(items, mode), external)
            folder = root/f'evaluation/{name}/{dep}'
            out.to_parquet(folder/f'consensus_{mode}_all_pairs.parquet', index=False)
            dev.csv_write(folder/f'consensus_{mode}_all_pairs.csv', out)
            for budget in BUDGETS:
                dev.csv_write(folder/f'consensus_{mode}_top{budget}.csv', out.head(budget))
                budgets.append(dev.budget_row(out, name, dep, mode, budget))
            consensus[name,dep,mode] = out
            for score in ('waveform_score_mean','final_score_mean'):
                for pe in ('pe_mc_bhattacharyya_coefficient','pe_mc_standardized_distance'):
                    result = spearmanr(out[score], out[pe])
                    correlations.append({'configuration':name,'deployment':dep,'mode':mode,
                        'score':score,'PE':pe,'spearman':float(result.statistic),
                        'ordinary_pair_pvalue_not_reported':True,'reason':'Pairs share events; adaptive real development.'})
    metrics, budgetframe = pd.DataFrame(records), pd.DataFrame(budgets)
    dev.csv_write(root/'tables/SELECTED_RETRIEVAL_PER_SEED.csv', metrics)
    dev.csv_write(root/'tables/SELECTED_PE_OFFICIAL_BUDGETS.csv', budgetframe)
    dev.csv_write(root/'tables/SELECTED_WEIGHTS.csv', pd.DataFrame(weightrows))
    dev.csv_write(root/'tables/SELECTED_CHANNEL_DISTRIBUTIONS.csv', pd.DataFrame(distributions))
    dev.csv_write(root/'tables/SELECTED_REAL_SCORE_PE_CORRELATIONS.csv', pd.DataFrame(correlations))
    dev.csv_write(root/'tables/SELECTED_FROZEN_CHANNEL_AUDIT.csv', pd.DataFrame(audits))
    numeric = ['macro_r_at_1','macro_r_at_10','average_precision','false_at_recall_0p5','false_at_recall_0p9']
    summary = metrics.groupby(['configuration','deployment','split','mode'])[numeric].agg(['mean','std']).reset_index()
    summary.columns = ['_'.join(filter(None,c)) if isinstance(c,tuple) else c for c in summary.columns]
    dev.csv_write(root/'tables/SELECTED_RETRIEVAL_SUMMARY.csv', summary)
    oldsearch = pd.read_csv(root/'tables/ALL_GLOBAL_PE_OFFICIAL_BUDGETS.csv')
    oldsearch = oldsearch[oldsearch.config == selected['configuration']]
    checkcols = ['BC_mc_ge_0p5','median_BC_mc','catastrophic_mc','Dmax_le_3','official_frontend','official_hanabi']
    for dep in t.DEPS:
        for budget in BUDGETS:
            a = budgetframe[(budgetframe.config==CODE)&(budgetframe.deployment==dep)&(budgetframe.method=='fusion')&(budgetframe.seed=='consensus')&(budgetframe.budget==budget)].iloc[0]
            b = oldsearch[(oldsearch.deployment==dep)&(oldsearch.budget==budget)].iloc[0]
            if not np.allclose(a[checkcols].to_numpy(float),b[checkcols].to_numpy(float),atol=1e-12,rtol=0):
                raise RuntimeError('Selected ranking changed since frozen development search')
    changes, known = [], []
    for dep in t.DEPS:
        for mode in ('waveform','fusion'):
            a = consensus['OMC',dep,mode];b = consensus[CODE,dep,mode]
            merged = a.merge(b,on='pair_key',suffixes=('_old','_new'),validate='one_to_one')
            merged['rank_change'] = merged.consensus_rank_new-merged.consensus_rank_old
            dev.csv_write(root/f'tables/{dep}_{mode}_RANK_CHANGES.csv', merged)
            for budget in BUDGETS:
                aa,bb = set(a.head(budget).pair_key),set(b.head(budget).pair_key)
                changes.append({'deployment':dep,'mode':mode,'budget':budget,'overlap':len(aa&bb),
                                'jaccard':len(aa&bb)/len(aa|bb),'entered':'|'.join(sorted(bb-aa)),'left':'|'.join(sorted(aa-bb))})
            if dep == 'gwtc3':
                for pair in KNOWN:
                    first,second = pair.split('--')
                    match = merged[((merged.event_i_new==first)&(merged.event_j_new==second))|
                                   ((merged.event_i_new==second)&(merged.event_j_new==first))]
                    for row in match.to_dict('records'):
                        known.append({'requested_pair':pair,'mode':mode,**row})
                    if match.empty:
                        known.append({'requested_pair':pair,'mode':mode,'not_in_frozen_scope':True})
    dev.csv_write(root/'tables/TOP_B_SET_CHANGES.csv',pd.DataFrame(changes))
    dev.csv_write(root/'tables/KNOWN_PAIR_AUDIT.csv',pd.DataFrame(known))
    dev.json_write(root/'contracts/SELECTED_EXPORT_COMPLETE.json', {
        'UTC':datetime.now(timezone.utc).isoformat(),'code':CODE,'search_replay_exact':True,
        'time_sky_exact':True,'outer_weights_changed':True,'goal_achieved':False,
        'fresh_confirmation_and_delivery_pending':True,'status':t.STATUS})
    print(json.dumps({'export_complete':True,'config':CODE,'metrics':len(metrics),'budget_rows':len(budgets)}),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();export(args.root)
