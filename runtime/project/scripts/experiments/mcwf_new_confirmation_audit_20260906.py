#!/usr/bin/env python3
"""Audit both changed deployments, never fabricate zero O4 intervals."""
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
import mcwf_development_20260905 as dev
import mcwf_encoder_body_20260906 as body
import mcwf_summarize_20260905 as old
from mcwf_confirmation_audit_20260906 import noise_cluster_ci
import mcwf_new_confirmation_20260906 as fresh


def run(root):
    frozen=fresh.verify(root)
    directory=root/'confirmation'
    independence=json.loads((directory/'GLOBAL_SOURCE_NOISE_INDEPENDENCE.json').read_text())
    if not independence['source_noise_independence_pass']:
        raise RuntimeError('Independent source/GPS audit must pass before confirmation is summarized')
    rows=[]
    invariance=[]
    sources=[]
    for dep in fresh.u.DEPS:
        for cs in fresh.CATALOGS:
            cat=directory/dep/f'catalog_{cs}'
            events=pd.read_parquet(cat/'event_manifest.parquet')
            src=pd.read_parquet(cat/'source_systems.parquet')
            sources.append(src.assign(deployment=dep,catalog_seed=cs))
            plan=pd.DataFrame({'system_id':events.source_uid,'family':events.family_slot})
            for ms,es in zip(body.MODEL_SEEDS,dev.SEEDS):
                out=cat/f'model_{ms}'
                bframe=pd.read_parquet(out/'BASELINE_pairs.parquet')
                cframe=pd.read_parquet(out/'CANDIDATE_pairs.parquet')
                for col in ('idx_i','idx_j','is_true_pair','true_pair_family','time_score','sky_raw_log_bf'):
                    if not np.array_equal(bframe[col],cframe[col]):
                        raise RuntimeError('Frozen column changed: '+col)
                changed=float((bframe.waveform_score!=cframe.waveform_score).mean())
                if changed==0:
                    raise RuntimeError('Old-only fallback or inactive new model')
                invariance.append({'deployment':dep,'catalog_seed':cs,'model_seed':ms,
                    'frozen_channels_exact':True,'changed_waveform_pair_fraction':changed})
                for method in ('waveform_only','C_fixed'):
                    ck=out/f'{method}_uncertainty.json'
                    if ck.exists():
                        rows.append(json.loads(ck.read_text()))
                        continue
                    w={'waveform':1.,'time':0.,'sky':0.} if method=='waveform_only' else dev.BASE.FROZEN_V93_WEIGHTS[dep][es]
                    b=dev.BASE.score_vector(bframe,w)
                    s=dev.BASE.score_vector(cframe,w)
                    bm=dev.BASE.full_metrics(bframe,b)
                    cm=dev.BASE.full_metrics(cframe,s)
                    ci,ranks=old.ranks_bootstrap(cframe,s,b,cs+ms,repeats=10000)
                    ranks.to_parquet(out/f'{method}_query_ranks.parquet',index=False)
                    row={'deployment':dep,'catalog_seed':cs,'model_seed':ms,'eval_seed':es,
                         'method':method,**cm,**ci}
                    for key in ('macro_r_at_1','macro_r_at_10','average_precision','false_at_recall_0p5','false_at_recall_0p9'):
                        row['baseline_'+key]=bm[key]
                        row['delta_'+key]=cm[key]-bm[key]
                    row['point_guardrail_pass']=bool(cm['macro_r_at_10']>=bm['macro_r_at_10']-.02
                        and cm['average_precision']>=bm['average_precision']-.005
                        and cm['false_at_recall_0p5']<=1.1*bm['false_at_recall_0p5']
                        and cm['false_at_recall_0p9']<=1.1*bm['false_at_recall_0p9'])
                    row.update(old.pair_bootstrap(cframe,s,b,plan,cs+ms,repeats=2000))
                    row.update(noise_cluster_ci(cframe,s,b,events,cs+ms,repeats=2000))
                    nfalse=int((~cframe.is_true_pair.to_numpy(dtype=bool)).sum())
                    row['legacy_minimum_one_FP_budget_actual_rate']=max(1,int(np.floor(1e-5*nfalse)))/nfalse
                    dev.json_write(ck,row)
                    rows.append(row)
                print(json.dumps({'audited':dep,'catalog':cs,'model':ms}),flush=True)
    f=pd.DataFrame(rows)
    src=pd.concat(sources,ignore_index=True)
    cols=['m1_det','m2_det','a1','a2','tilt1','tilt2','theta_jn','phi12','phijl','psi','phase','ra','dec']
    if src.source_uid.duplicated().any() or src.duplicated(cols).any():
        raise RuntimeError('Repeated new source parameters')
    dev.csv_write(directory/'PAIRED_METRICS_AND_CI.csv',f)
    metrics=[c for c in f if c.startswith(('macro_','average_','false_','baseline_','delta_')) and 'ci_' not in c]
    means=f.groupby(['deployment','model_seed','method'])[metrics].mean().reset_index()
    means['point_guardrail_pass']=(means.delta_macro_r_at_10.ge(-.02)&means.delta_average_precision.ge(-.005)
        &means.false_at_recall_0p5.le(1.1*means.baseline_false_at_recall_0p5)
        &means.false_at_recall_0p9.le(1.1*means.baseline_false_at_recall_0p9))
    dev.csv_write(directory/'MODEL_MEAN_OVER_CATALOGS.csv',means)
    summary=means.groupby(['deployment','method'])[metrics].agg(['mean','std']).reset_index()
    summary.columns=['_'.join(filter(None,c)) if isinstance(c,tuple) else c for c in summary.columns]
    dev.csv_write(directory/'SUMMARY.csv',summary)
    dev.csv_write(directory/'FROZEN_CHANNEL_INVARIANCE.csv',pd.DataFrame(invariance))
    dev.csv_write(directory/'NEW_SOURCE_PARAMETERS.csv',src)
    noise=[]
    for dep in fresh.u.DEPS:
        blocks=pd.read_csv(directory/dep/'noise/noise_manifest.csv')
        excluded=pd.read_csv(directory/dep/'noise/EXCLUDED_BLOCKS.csv')
        for r in blocks.itertuples():
            if ((excluded.start<r.end_gps)&(excluded.end>r.start_gps)).any():
                raise RuntimeError('New noise overlaps historical/development data')
        starts=blocks.sort_values('start_gps')
        if np.any(starts.start_gps.to_numpy()[1:]<starts.end_gps.to_numpy()[:-1]):
            raise RuntimeError('Global GPS overlap within new noise blocks')
        noise.append({'deployment':dep,'blocks':len(blocks),'historical_overlap':0,'new_block_overlap':0})
    dev.csv_write(directory/'GLOBAL_NOISE_AUDIT.csv',pd.DataFrame(noise))
    passed=bool(means.point_guardrail_pass.all())
    dev.json_write(directory/'FINAL_GUARDRAIL_AUDIT.json',{
        'recipe':frozen['recipe'],'both_deployments_new_model_active':True,
        'fresh_BBH_sources':len(src),'unique_source_parameters':True,
        'global_independence_audit_sha256':dev.sha(directory/'GLOBAL_SOURCE_NOISE_INDEPENDENCE.json'),
        'time_sky_labels_exact':True,'independent_noise_blocks':sum(v['blocks'] for v in noise),
        'catalog_model_method_passes':int(f.point_guardrail_pass.sum()),'catalog_model_method_total':len(f),
        'per_model_mean_across3catalog_guardrails_pass':passed,
        'model_checks':means[['deployment','model_seed','method','point_guardrail_pass']].to_dict('records'),
        'status':'PASS_CONDITIONAL_NEW_SOURCE_NOISE_TEST' if passed else 'FAIL_CONDITIONAL_NEW_SOURCE_NOISE_TEST',
        'limitations':['source-conditional and noise-conditional bootstrap are separate diagnostics, not a proof of population coverage',
            'same lens environments and balanced mass/SNR proposal, not a fresh astrophysical population',
            'conditional Gaussian-measurement BAYESTAR, not PE of exact non-Gaussian strain',
            'F50/F90 retain the frozen rank-based crossing convention; AP groups score ties',
            'legacy 1e-5 field enforces at least one false pair; report actual rate instead of claiming 1e-5 resolution'],
        'final_author_review_required':True})
    print(summary.to_json(orient='records'),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    run(p.parse_args().root)
