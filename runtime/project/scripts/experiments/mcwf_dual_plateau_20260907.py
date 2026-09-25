#!/usr/bin/env python3
"""Declared adaptive development selection on a finite dual-mass plateau."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='2'
import argparse
from datetime import datetime,timezone
from pathlib import Path
import itertools
import json
import shutil
import numpy as np
import pandas as pd
import mcwf_pe_frontend_extension_v2_20260907 as e
import mcwf_dual_tail_20260907 as dual
import mcwf_seed_plateau_20260907 as shared
import mcwf_runwise_development_20260907 as rw

dev,body,ev=e.dev,e.body,e.ev


def run(root):
    trial=root/'trials/DUAL-MASS-PLATEAU-DEVELOPMENT'
    if trial.exists():raise RuntimeError('Independent trial required')
    for name in ('contracts','configs','tables','results','evaluation'):(trial/name).mkdir(parents=True)
    shutil.copy2(__file__,trial/'contracts'/Path(__file__).name)
    dev.json_write(trial/'contracts/ADAPTIVE_DEVELOPMENT_ADDENDUM.json',{
        'created_utc':datetime.now(timezone.utc).isoformat(),'same_O3_O4_method':True,
        'formula':'same DUAL-MASS-TAIL formula and grids in both runs, separate numerical operating point per model seed',
        'plateau':'unchanged FRT plus at most4best validation-objective points per p0 that also pass all reused-injection guards; deduplicate identical gamma=beta=0',
        'maximum_combinations_per_run':17**3,'PE_used_to_build_plateau':False,
        'final_selection':'explicit adaptive real-development goals on all plateau combinations; do not call this validation-only or blind real test',
        'tie_order':['PE count gains','minimum official gain across Top10/20','total official gains','minimum waveform coefficient change','fixed lexical order'],
        'forbidden':'no real PE values,event identities,official membership or per-pair manually defined masks in score',
        'unchanged_target_contract':str(root/'contracts/EXPERIMENT_CONTRACT.json'),
        'requires_new_injection_confirmation_after_selection':True,
        'time_sky_outerweights_historical_unchanged':True,
        'scientific_limit':'high-multiplicity exploratory tuning to real development outcomes; no independent claim from the same real catalog'})
    baseline=pd.read_csv(root/'contracts/BASELINE_BUDGETS.csv');baseline=baseline[baseline.seed.astype(str)=='consensus']
    selected={};allrows=[];allcombos=[];data={}
    for dep in e.DEPS:
        pe=pd.read_parquet(root/f'audit/{dep}_frozen_external_reference.parquet').sort_values('pair_key').reset_index(drop=True);keys=pe.pair_key.to_list()
        pools=[];prior=e.reference_prior(dep)
        for ms,es in zip(body.MODEL_SEEDS,dev.SEEDS):
            original=root/f'trials/DUAL-MASS-TAIL/calibration/{dep}/seed_{es}'
            basecfg=json.loads((original/'SELECTED_CONFIG.json').read_text());grid=pd.read_csv(original/'validation_grid.csv')
            frames={};xx={};bm={}
            for split in ('validation','test','real'):
                f,x=e.frame_features(dep,ms,es,split,prior);frames[split]=f;xx[split]=dual.components(f,x,basecfg)
                if split!='real':bm[split]=ev.metrics(f,f.waveform_score.to_numpy(float),dep,es)
            data[dep,es]=(frames,xx,bm);eligible=[]
            for row in grid[grid['pass']].to_dict('records'):
                cfg={**basecfg,**{c:float(row[c]) for c in ('p0','gamma','beta')}}
                z,*_=dual.score(frames['test'],cfg,xx['test']);tm=ev.metrics(frames['test'],z,dep,es);ok=ev.guard(tm,bm['test'])
                ident=f"P{cfg['p0']:g}_G{cfg['gamma']:g}_B{cfg['beta']:g}"
                allrows.append({'deployment':dep,'seed':es,'id':ident,'reused_test_pass':ok,**row})
                if ok:
                    vm={method:{k[len(method)+1:]:v for k,v in row.items() if k.startswith(method+'_')} for method in ('waveform_only','C_fixed')}
                    eligible.append({'id':ident,'spec':cfg,'objective':e.old_selection.objective(vm,cfg['gamma']+cfg['beta'],cfg['p0'])+(cfg['gamma'],cfg['beta'])})
            eligible.sort(key=lambda c:(c['objective'],c['id']))
            base=next(c for c in eligible if c['spec']['gamma']==0 and c['spec']['beta']==0)
            pool=[base]
            for p0 in dual.P0:
                pool.extend([c for c in eligible if c['spec']['p0']==p0 and c['id']!=base['id']][:4])
            dev.json_write(trial/f'configs/{dep}_{es}_PLATEAU.json',{'eligible':len(eligible),'retained':pool,'real_used_to_build_plateau':False})
            for c in pool:
                z,*_=dual.score(frames['real'],c['spec'],xx['real']);c['rank_arrays']=shared.rank_arrays(frames['real'],z,dep,es,keys)
            pools.append(pool)
        dev.csv_write(trial/'tables/ELIGIBLE_VALIDATION_GRID.csv',pd.DataFrame(allrows));qualified=[]
        for ci,combo in enumerate(itertools.product(*pools)):
            budgets=shared.fast_budgets(combo,pe,dep);result=rw.target(budgets,baseline,dep);ids=[c['id'] for c in combo]
            allcombos.append({'deployment':dep,'combination':ci,'seed1':ids[0],'seed2':ids[1],'seed3':ids[2],**{k:v for k,v in result.items() if k!='details'}})
            if result['target_pass']:
                balanced=min(min(c['front_delta'],c['hanabi_delta']) for c in result['details'])
                deviation=sum(c['spec']['gamma']+c['spec']['beta'] for c in combo)
                key=(-result['BC_gain']-result['Dmax_gain'],-balanced,-result['front_gain']-result['hanabi_gain'],deviation,ids)
                qualified.append((key,combo,result))
        dev.csv_write(trial/'tables/ALL_COMBINATIONS.csv',pd.DataFrame(allcombos))
        if qualified:
            _,combo,result=min(qualified,key=lambda c:c[0]);selected[dep]={es:c['spec'] for es,c in zip(dev.SEEDS,combo)}
            dev.json_write(trial/f'contracts/{dep}_SELECTED.json',{'configs':selected[dep],'development_target':result,'qualifying':len(qualified)})
        print(json.dumps({'dual_plateau':dep,'pools':[len(p) for p in pools],'qualifying':len(qualified),'selected':selected.get(dep)}),flush=True)
    dev.json_write(trial/'contracts/SEARCH_COMPLETE.json',{'both_runs_qualify':len(selected)==2,'selected':selected,'fresh_confirmation_done':False})
    if len(selected)!=2:return
    rows=[];guards=[]
    for dep in e.DEPS:
        real={}
        for es in dev.SEEDS:
            frames,xx,bm=data[dep,es]
            for split,f in frames.items():
                z,penalty,reward,ood=dual.score(f,selected[dep][es],xx[split]);n=f.copy();n['FRT_baseline_waveform_score']=f.waveform_score;n['waveform_score']=z
                n['old_mass_tail_penalty']=penalty;n['concordant_reward']=reward;n['new_positive_ood']=ood
                out=trial/f'evaluation/{dep}/seed_{es}';out.mkdir(parents=True,exist_ok=True);n.to_parquet(out/f'{split}_pairs.parquet',index=False)
                if split=='real':real[es]=n
                else:
                    m=ev.metrics(f,z,dep,es);guards.append({'deployment':dep,'seed':es,'split':split,'pass':ev.guard(m,bm[split])})
                    for method in m:
                        for config,metrics in [('FRT_BASELINE',bm[split][method]),('CANDIDATE',m[method])]:
                            rows.append({'deployment':dep,'seed':es,'split':split,'method':method,'config':config,**metrics})
        dev.save_evaluation(trial,'CANDIDATE',dep,real,pd.read_parquet(root/f'audit/{dep}_frozen_external_reference.parquet'))
    dev.csv_write(trial/'tables/RETRIEVAL_PER_SEED.csv',pd.DataFrame(rows));dev.csv_write(trial/'tables/REUSED_GUARDRAILS.csv',pd.DataFrame(guards));e.assess(root,trial)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();run(a.root)
