#!/usr/bin/env python3
"""Frozen multi-seed aggregation diagnostics, not an ensemble injection claim."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import sys
import numpy as np
import pandas as pd

P=Path('/root/autodl-tmp/gw-catalog')
sys.path.insert(0,str(P/'scripts/experiments'))
import mcwf_temporal_response_evaluate_20260908 as ev
t,dev,cf=ev.t,ev.dev,ev.cf
JOINT=P/'results/mcwf_joint_intrinsic_evidence_exploratory_20260908T155830Z'
REFIT=P/'results/mcwf_joint_fusion_retune_exploratory_20260908T160928Z'
SPECS=[('OMC',JOINT,'OMC')]
for inner in ('CANDIDATE','RETRIEVAL'):
    for head in ('CHI','ETA-CHI'):
        name=f'JOINT-{head}-BC-ADD-{inner}'
        SPECS.append((name,JOINT,name))
        SPECS.append(('REFIT-'+name,REFIT,'REFIT-'+name+'-POSITIVE-CANDIDATE'))
RULES=('MEAN-RANK','MEAN-SCORE','MEDIAN-RANK','MEDIAN-SCORE')


def initialize(root):
    if root.exists():raise RuntimeError('New independent directory required')
    for name in ('contracts','tables','evaluation','reports','scripts','logs','manifest'):(root/name).mkdir(parents=True)
    dev.json_write(root/'contracts/ANALYSIS_CONTRACT.json',{
        'id':'MCWF-CONSENSUS-DIAGNOSTIC-16','UTC':datetime.now(timezone.utc).isoformat(),'status':t.STATUS,
        'goal_achieved':False,'same_both_runs':True,'rules':RULES,
        'upstreams':[{'code':a,'root':str(b),'method':c} for a,b,c in SPECS],
        'question':'Mean ranks discard score magnitudes and can be sensitive to a single seed outlier. Compare equal-weight normalized score pooling and medians without any new fit or per-run manual choices.',
        'normalization':'Divide each finalscore bysumofthatseed W/T/S weights; no use of real catalog rowmeans/stds or real PE. The original meanrank output is exactly reproduced.',
        'ties':'Meanrankusesoriginaltieorder;alternativesusetheirprimary,thenoriginalrankmean,rankmax,finalscoremean;noeventIDsorofficiallabels.',
        'frozen':['eachmodel','all perseedrawchannels','all perseedweights','perseedpairrank','scope','history','paper'],
        'not_new_model':'Changes only consensus across existing model deployments. Do not call this a waveform encoder improvement.',
        'injection':'Existing perseed test guardrails copied with inputhash;NO ensemble injectionperformance is inferred fromthese numbers. A common fresh catalog scored by allmodels is required ifa consensus targetcandidate emerges.',
        'external':'Adaptive development audit,not blind confirmation. PE andofficial joinedafteraggregation. Official overlapnotlens truth.',
        'final_gate':'PE+official target plus original perseed guard gives only a development candidate,never goalcompletion.'})
    rows=t.protected()
    for code,up,name in SPECS:
        for dep in t.DEPS:
            for seed in t.SEEDS:
                p=up/f'evaluation/{name}/{dep}/seed_{seed}/real_fusion_pairs.parquet'
                rows.append({'path':str(p),'sha256':dev.sha(p),'bytes':p.stat().st_size})
    dev.csv_write(root/'manifest/INPUT_SHA256.csv',pd.DataFrame(rows).drop_duplicates('path'))
    shutil.copy2(__file__,root/'scripts/consensus_diagnostic.py')
    dev.json_write(root/'contracts/FREEZE.json',{'contract_sha256':dev.sha(root/'contracts/ANALYSIS_CONTRACT.json'),'code_sha256':dev.sha(Path(__file__))})


def run(root):
    budgets=[];replays=[];guards=[]
    for code,up,name in SPECS:
        selection=ev.load_selections(up)
        summary=pd.read_csv(up/'tables/RETRIEVAL_PER_SEED.csv')
        for dep in t.DEPS:
            frames=[];metadata=[]
            for seed in t.SEEDS:
                f=pd.read_parquet(up/f'evaluation/{name}/{dep}/seed_{seed}/real_fusion_pairs.parquet')
                c=next(c for c in selection if c['method']==name and c['deployment']==dep and c['seed']==seed)
                raw=f.final_score.to_numpy(float)
                if not np.isfinite(raw).all():raise RuntimeError('Nonfinite realscore')
                f['normalized_final_score']=raw/np.sum(c['weights']);frames.append(f)
                metadata.append(f[['pair_key','normalized_final_score','rank']])
            old=dev.BASE.consensus_real(frames,name)
            block=pd.concat(metadata,ignore_index=True)
            extra=block.groupby('pair_key').agg(normalized_score_mean=('normalized_final_score','mean'),
                normalized_score_median=('normalized_final_score','median'),rank_median=('rank','median'))
            base=old.merge(extra,on='pair_key',validate='one_to_one')
            external=pd.read_parquet(t.EXTERNAL/f'{dep}_external_reference.parquet')
            for rule in RULES:
                code_rule=code+'__'+rule
                if rule=='MEAN-RANK':out=base.copy()
                else:
                    primary={'MEAN-SCORE':'normalized_score_mean','MEDIAN-SCORE':'normalized_score_median','MEDIAN-RANK':'rank_median'}[rule]
                    out=base.sort_values([primary,'rank_mean','rank_max','final_score_mean'],ascending=[rule=='MEDIAN-RANK',True,True,False],kind='stable').reset_index(drop=True)
                    out['consensus_rank']=np.arange(1,len(out)+1)
                if rule=='MEAN-RANK' and not np.array_equal(out.pair_key,old.pair_key):raise RuntimeError('Baseline consensus didnotreplay')
                out['method']=code_rule
                newcols=[c for c in external if c=='pair_key' or c not in out]
                out=out.merge(external[newcols],on='pair_key',how='left',validate='one_to_one').sort_values('consensus_rank')
                dest=root/f'evaluation/{code_rule}/{dep}';dest.mkdir(parents=True,exist_ok=True)
                out.to_parquet(dest/'consensus_fusion_pairs.parquet',index=False)
                dev.csv_write(dest/'TOP100.csv',out.head(100))
                for b in (10,20,50,100):
                    budgets.append(dev.budget_row(out,code_rule,dep,'fusion',b,seed='consensus'))
                a=summary[(summary.method==name)&(summary.deployment==dep)&(summary.split=='test')]
                guards.append({'configuration':code_rule,'deployment':dep,'original_perseed_injection_guard':bool(a.guard_pass.all()),
                    'samecatalog_ensemble_test_completed':False,'original_metrics_path':str(up/'tables/RETRIEVAL_PER_SEED.csv')})
                replays.append({'configuration':code_rule,'deployment':dep,'perseed_scores_ranks_unchanged':True,'meanrank_exactreplay':True})
    a=pd.DataFrame(budgets);dev.csv_write(root/'tables/PE_OFFICIAL_BUDGETS.csv',a)
    dev.csv_write(root/'tables/PERSEED_GUARDS_NOT_ENSEMBLE_TEST.csv',pd.DataFrame(guards))
    dev.csv_write(root/'tables/FROZEN_SCORE_REPLAY.csv',pd.DataFrame(replays))
    checks=[]
    for config in sorted(set(a.config)-{'OMC__MEAN-RANK'}):
        for dep in t.DEPS:
            base=a[(a.config=='OMC__MEAN-RANK')&(a.deployment==dep)].set_index('budget')
            new=a[(a.config==config)&(a.deployment==dep)].set_index('budget')
            guard=True;pe=False;official=False
            for b in (10,20):
                for col in ('BC_mc_ge_0p5','median_BC_mc','Dmax_le_3','official_frontend','official_hanabi'):
                    guard &= new.loc[b,col]>=base.loc[b,col]-1e-12
                guard &= new.loc[b,'catastrophic_mc']<=base.loc[b,'catastrophic_mc']
                pe |= new.loc[b,'BC_mc_ge_0p5']>base.loc[b,'BC_mc_ge_0p5'] or new.loc[b,'median_BC_mc']>base.loc[b,'median_BC_mc']+1e-12
                official |= new.loc[b,'official_frontend']>base.loc[b,'official_frontend']
            perseed=next(g['original_perseed_injection_guard'] for g in guards if g['configuration']==config and g['deployment']==dep)
            checks.append({'configuration':config,'deployment':dep,'no_loss_top10_top20':guard,'PE_gain':pe,'official_gain':official,
                'original_perseed_injection_guard':perseed,'run_development_candidate':bool(guard and pe and official and perseed),
                'common_catalog_ensemble_guard_pass':False})
    check=pd.DataFrame(checks);dev.csv_write(root/'tables/DEVELOPMENT_TARGET_CHECK.csv',check)
    win=check.groupby('configuration').run_development_candidate.all();winners=list(win[win].index)
    dev.json_write(root/'contracts/ROUND_ASSESSMENT.json',{'development_candidates_requiring_fresh_confirmation':winners,
        'goal_achieved':False,'status':t.STATUS,'fresh_common_catalog_ensemble_test_required':True,'samecatalog_ensemble_test_completed':False})
    print(json.dumps({'development_candidates':winners,'not_goal_completion':True}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--stage',choices=['initialize','run'],required=True)
    a=p.parse_args();globals()[a.stage](a.root)
