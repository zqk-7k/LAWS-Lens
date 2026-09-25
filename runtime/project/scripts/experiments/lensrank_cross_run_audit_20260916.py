"""Frozen-score size control and read-only cross-run difficulty diagnostics."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

FIX = Path('/root/autodl-tmp/gw-catalog/results/lensrank_sky_spin_si_fixed_20260916T023040Z_r2')
SEED = 2026091601
DRAWS = 500
INPUTS = {}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def read(path):
    path = Path(path)
    INPUTS[str(path)] = sha(path)
    return json.loads(path.read_text()) if path.suffix == '.json' else pd.read_parquet(path)


def write(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def csv(root, name, rows):
    f = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    f.to_csv(root / name, index=False, encoding='utf-8-sig')
    return f


def ranks(values, y, n):
    ii, jj = np.triu_indices(n, 1)
    m = np.full((n, n), -np.inf)
    m[ii, jj] = m[jj, ii] = values
    queries = np.r_[ii[y], jj[y]]
    thresholds = np.tile(values[y], 2)
    best = 1 + (m[queries] > thresholds[:, None]).sum(1)
    worst = (m[queries] >= thresholds[:, None]).sum(1)
    return best, worst, queries


def metric(values, y, n):
    best, worst, _ = ranks(values, y, n)
    out = {'R1': float((best <= 1).mean()), 'R10': float((best <= 10).mean()),
           'R10_pessimistic': float((worst <= 10).mean()),
           'AP': float(average_precision_score(y, values)), 'ROC_AUC': float(roc_auc_score(y, values))}
    order = np.argsort(-values, kind='stable')
    tp = np.cumsum(y[order]); fp = np.cumsum(~y[order])
    for target, name in [(0.5, 'F50'), (0.9, 'F90')]:
        out[name] = int(fp[np.searchsorted(tp / y.sum(), target)])
    return out


def snr_rule(snr, y, n):
    bins = np.digitize(snr, [8., 10., 12., 20., 40.]) - 1
    ii, jj = np.triu_indices(n, 1)
    lo, hi = np.minimum(bins[ii], bins[jj]), np.maximum(bins[ii], bins[jj])
    values = (((lo == 0) & (hi == 1)) | ((lo == 2) & (hi == 3))).astype(float)
    a, b, _ = ranks(values, y, n)
    expected = np.clip((11 - a) / (b - a + 1), 0, 1)
    return {'SNR_rule_AUC': float(roc_auc_score(y, values)), 'SNR_rule_expected_R10': float(expected.mean())}


def arrays(group, recipe):
    f = read(recipe['pair_path'])
    p = read(FIX / 'results' / group['id'] / f"model_{recipe['slot']}" / 'paired_scores.parquet')
    for col in ('idx_i', 'idx_j', 'is_true_pair'):
        if not np.array_equal(f[col], p[col]):
            raise RuntimeError('Pair indexing mismatch')
    if group['deployment'] == 'o4b':
        columns = ['Z_wf_short', 'Z_wf_FRT', 'Z_wf_OMC']
    else:
        columns = ['previous_waveform_score', 'FRT_baseline_waveform_score', 'retained_OMC_waveform']
    v = {name: f[col].to_numpy(float) for name, col in zip(['short', 'R_branch', 'M_branch'], columns)}
    v.update({
        'waveform': p.waveform_score_frozen.to_numpy(float), 'time': p.time_score.to_numpy(float),
        'sky_old': p.sky_raw_log_bf_archived.to_numpy(float), 'sky_fixed': p.sky_raw_log_bf_SI_fixed.to_numpy(float),
        'fusion_old': p.final_score_archived.to_numpy(float), 'fusion_fixed': p.final_score_SI_fixed.to_numpy(float),
    })
    return f, p, v


def event_frame(g):
    rows = []
    for item in g['records']:
        row = dict(item['record']); s = item['source']
        row['mc'] = (s['m1_det'] * s['m2_det']) ** .6 / (s['m1_det'] + s['m2_det']) ** .2
        row['source_uid'] = str(row['source_uid'])
        rows.append(row)
    e = pd.DataFrame(rows).sort_values('idx').reset_index(drop=True)
    if not np.array_equal(e.idx, np.arange(len(e))):
        raise RuntimeError('Event indices not contiguous')
    return e


def main(root):
    root.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, root / Path(__file__).name)
    contract = {
        'version': 'GWLR-CROSSRUN-AUDIT-01', 'analysis': 'read-only diagnostic and independent size-control resampling',
        'seed': SEED, 'rng': 'numpy default_rng PCG64', 'draws': DRAWS,
        'selection': 'Uniform without replacement:35 SIS-slot and35 PM-slot doublets plus50 singleton sources; no performance selection',
        'shared_source_draws_across_three_models': True,
        'events_per_catalog': 190, 'pairs': 17955, 'true_pairs': 70,
        'old_500_draw_manifest_reproduction': False, 'old_500_draw_manifest_missing': True,
        'summary': 'Average500 catalogs within each model, then mean and sampleSD across3 models; overlap means not500 independent trials',
        'calibration_training_real_ranking_changes': False,
        'mass_close_rule': 'max(Mc_i,Mc_j)/min(Mc_i,Mc_j)<=1.1; also report symmetric and max-relative definitions',
        'O3_intervals_GPS': [[1238166018,1253977218],[1256655618,1269363618]],
        'O3_intervals_source': ['https://gwosc.org/O3/O3a/','https://gwosc.org/O3/O3b/'],
        'close_competitor_window_days': 7,
        'unit_fix_baseline': str(FIX), 'script_sha256': sha(__file__),
        'status': 'HOLD_FOR_AUTHOR_REVIEW_NO_ADOPTION_NO_OVERWRITE',
    }
    write(root / 'ANALYSIS_CONTRACT.json', contract)
    c = read(FIX / 'contracts/ANALYSIS_CONTRACT.json')
    groups = c['groups']
    eframes = {g['id']: event_frame(g) for g in groups}
    o4groups = [g for g in groups if g['deployment'] == 'o4b']
    source = eframes[o4groups[0]['id']].groupby('source_uid').agg(family=('family_slot','first'), count=('idx','size'))
    for g in o4groups[1:]:
        other = eframes[g['id']].groupby('source_uid').agg(family=('family_slot','first'), count=('idx','size'))
        pd.testing.assert_frame_equal(source, other)
    pools = {tag: sorted(source.index[(source.family == tag) & (source['count'] == 2)]) for tag in ['SIS', 'PM']}
    pools['singleton'] = sorted(source.index[source['count'] == 1])
    rng = np.random.default_rng(SEED)
    samples = []
    for draw in range(DRAWS):
        selected = {tag: rng.choice(pool, 50 if tag == 'singleton' else 35, replace=False).tolist() for tag, pool in pools.items()}
        samples.append({'draw': draw, 'sources': selected})
    write(root / 'O4B_190_SOURCE_DRAW_MANIFEST.json', {'seed': SEED, 'samples': samples})
    write(root / 'SAMPLING_FREEZE.json', {'source_manifest_sha256': sha(root/'O4B_190_SOURCE_DRAW_MANIFEST.json'),
                                        'contract_sha256':sha(root/'ANALYSIS_CONTRACT.json'), 'before_subset_score_calculation':True})
    diagnostic, stages, changes, subsets, source_stats, snr_bins, mass_stats, joint_stats = [], [], [], [], [], [], [], []
    tstart = time.time()
    for g in groups:
        e = eframes[g['id']]; n = len(e)
        ii,jj = np.triu_indices(n,1)
        ids = e.source_uid.to_numpy(); y = ids[ii] == ids[jj]
        gps=e.gps_obs.to_numpy(); snr=e.target_network_snr.to_numpy(); mc=e.mc.to_numpy()
        dt=abs(gps[ii]-gps[jj])/86400
        both_o3=(((gps>=1238166018)&(gps<1253977218))|((gps>=1256655618)&(gps<1269363618)))
        near=(abs(gps[:,None]-gps[None,:])<=7*86400)&(ids[:,None]!=ids[None,:])
        a,b=np.minimum(mc[ii],mc[jj]),np.maximum(mc[ii],mc[jj])
        close=(b/a<=1.1)&~y
        meta={'deployment':g['deployment'],'catalog':g['catalog'],'group':g['id'],'n_events':n}
        counts=e.groupby('source_uid').idx.transform('size').to_numpy()
        source_stats.append({**meta, 'source_units':len(set(ids)), 'time_span_days':float((gps.max()-gps.min())/86400),
                             'outside_O3_intervals':int((~both_o3).sum()) if g['deployment']=='gwtc3' else None,
                             'SNR_q10':float(np.quantile(snr,.1)),'SNR_median':float(np.median(snr)), 'SNR_q90':float(np.quantile(snr,.9)),
                             'Mc_q10':float(np.quantile(mc,.1)),'Mc_median':float(np.median(mc)),'Mc_q90':float(np.quantile(mc,.9)),
                             'near_null_competitors_7d_all':float(near.sum(1).mean()),
                             'near_null_competitors_7d_queries':float(near.sum(1)[counts==2].mean())})
        bins=np.digitize(snr,[8.,10.,12.,20.,40.])-1
        for kind,mask in [('paired_image',counts==2),('singleton',counts==1)]:
            for k in range(-1,5):snr_bins.append({**meta,'type':kind,'bin':k,'count':int((mask&(bins==k)).sum())})
        mass_stats.append({**meta,'null_pairs':int((~y).sum()),'ratio_le_1p1':float(close.sum()/(~y).sum()),
                           'relative_to_max_le_10pct':float(((b-a)/b<=.1)[~y].mean()),
                           'relative_to_mean_le_10pct':float((2*(b-a)/(a+b)<=.1)[~y].mean())})
        for number,recipe in enumerate(g['recipes']):
            f,p,vs=arrays(g,recipe)
            if not np.array_equal(f.is_true_pair,y):raise RuntimeError('Source truth mismatch')
            key={**meta,'slot':recipe['slot']}
            for name,v in vs.items():stages.append({**key,'method':name,**metric(v,y,n)})
            if number==0:
                diag={**meta, **snr_rule(snr,y,n), 'nearest_time_R10':metric(-dt,y,n)['R10'], 'time_R10':metric(vs['time'],y,n)['R10']}
                for name,v in [('time',vs['time']),('nearest',-dt)]:
                    rr,_,queries=ranks(v,y,n)
                    inpair=np.tile(both_o3[ii[y]]&both_o3[jj[y]],2)
                    for label,mask in [('both_in_O3',inpair),('some_outside_O3',~inpair)]:
                        diag[name+'_'+label+'_queries']=int(mask.sum())
                        diag[name+'_'+label+'_R10']=float((rr[mask]<=10).mean()) if mask.any() else None
                diagnostic.append(diag)
            old,_,_=ranks(vs['fusion_old'],y,n);new,_,_=ranks(vs['fusion_fixed'],y,n)
            delta=vs['sky_fixed']-vs['sky_old']
            changes.append({**key,'query_count':len(old),'R10_enters':int(((old>10)&(new<=10)).sum()),
                            'R10_exits':int(((old<=10)&(new>10)).sum()),'query_rank_changed_fraction':float((old!=new).mean()),
                            'fusion_spearman':float(spearmanr(vs['fusion_old'],vs['fusion_fixed']).statistic),
                            'sky_spearman':float(spearmanr(vs['sky_old'],vs['sky_fixed']).statistic),
                            'sky_median_abs_delta':float(np.median(abs(delta))),'sky_p99_abs_delta':float(np.quantile(abs(delta),.99)),
                            'sky_positive_sign_flips':int(((vs['sky_old']>0)!=(vs['sky_fixed']>0)).sum())})
            bc=f.joint_BC.to_numpy(float)
            for label,mask in [('true',y),('null_close_Mc',close),('null_other',(~y)&~close)]:
                joint_stats.append({**key,'population':label,'pairs':int(mask.sum()),'BC_median':float(np.median(bc[mask])),
                                    'BC_q90':float(np.quantile(bc[mask],.9)),'BC_ge05_fraction':float((bc[mask]>=.5).mean()),
                                    'joint_increment_median':float(np.median(f.joint_increment.to_numpy(float)[mask]))})
            if g['deployment']=='o4b':
                index=np.full((n,n),-1,int);index[ii,jj]=index[jj,ii]=np.arange(len(ii))
                aa,bb=np.triu_indices(190,1)
                for sample in samples:
                    chosen={s for values in sample['sources'].values() for s in values}
                    nodes=np.flatnonzero(np.isin(ids,list(chosen)))
                    if len(nodes)!=190:raise RuntimeError('Broken source closure')
                    edge=index[nodes[aa],nodes[bb]];yy=y[edge]
                    if yy.sum()!=70:raise RuntimeError('Wrong true-pair count')
                    diag=snr_rule(snr[nodes],yy,190)
                    for name,v in vs.items():
                        subsets.append({**key,'n_events':190,'draw':sample['draw'],'method':name,**metric(v[edge],yy,190)})
                    subsets.append({**key,'n_events':190,'draw':sample['draw'],'method':'diagnostic',**diag,
                                    'nearest_time_R10':metric(-dt[edge],yy,190)['R10'],
                                    'close_Mc_null_fraction':float(close[edge].sum()/(~yy).sum()),
                                    'near_null_competitors_7d_queries':float(near[np.ix_(nodes,nodes)].sum(1)[counts[nodes]==2].mean())})
        print(json.dumps({'finished_group':g['id'],'seconds':time.time()-tstart}),flush=True)
    for name,rows in [('calendar_SNR_diagnostic',diagnostic),('full_catalog_stage_metrics',stages),('unit_fix_rank_changes',changes),
                      ('source_population_summary',source_stats),('SNR_class_bins',snr_bins),('null_mass_difficulty',mass_stats),('joint_branch_difficulty',joint_stats)]:
        csv(root,name+'.csv',rows)
    sub=csv(root,'O4B_190_per_draw_model.csv',subsets)
    cols=['R1','R10','AP','ROC_AUC','F50','F90','SNR_rule_AUC','SNR_rule_expected_R10','nearest_time_R10','close_Mc_null_fraction','near_null_competitors_7d_queries']
    means=sub.groupby(['method','slot'])[cols].mean().reset_index()
    csv(root,'O4B_190_per_model_mean.csv',means)
    summary=means.groupby('method')[cols].agg(['mean','std']);summary.columns=['_'.join(v) for v in summary.columns]
    csv(root,'O4B_190_summary.csv',summary.reset_index())
    quant=sub.groupby(['method','draw'])[cols].mean().groupby('method').quantile([.025,.5,.975]).reset_index()
    csv(root,'O4B_190_conditional_draw_ranges.csv',quant)
    stage=pd.DataFrame(stages);means=stage.groupby(['deployment','slot','method'])[cols[:6]].mean().reset_index()
    summary_all=means.groupby(['deployment','method'])[cols[:6]].agg(['mean','std']);summary_all.columns=['_'.join(v) for v in summary_all.columns]
    csv(root,'full_catalog_stage_summary.csv',summary_all.reset_index())
    for p,h in INPUTS.items():
        if sha(p)!=h:raise RuntimeError('Input changed during audit')
    write(root/'INPUT_SHA256.json',INPUTS)
    write(root/'STATUS.json',{'state':'HOLD_FOR_AUTHOR_REVIEW_NO_ADOPTION_NO_OVERWRITE','input_hashes_unchanged':True,
                            'new_O4b_subsamples':500,'same_subsamples_for_old_fixed_and_all_models':True,
                            'units_corrected_input':True,'models_weights_real_rankings_changed':False})
    print(summary.reset_index().to_string(index=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    main(parser.parse_args().root)
