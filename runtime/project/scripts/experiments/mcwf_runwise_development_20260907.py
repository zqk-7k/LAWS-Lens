#!/usr/bin/env python3
"""Same bounded waveform rule, separately calibrated operating points by run."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='2'
import argparse
from datetime import datetime,timezone
from pathlib import Path
import json
import shutil
import numpy as np
import pandas as pd
import mcwf_pe_frontend_extension_v2_20260907 as e
import mcwf_global_development_20260907 as g

dev,body,ev=e.dev,e.body,e.ev


def target(budgets,baseline,dep):
    checks=[]
    for b in (10,20):
        n=budgets[(budgets.method=='C_fixed')&(budgets.budget==b)].iloc[0]
        old=baseline[(baseline.deployment==dep)&(baseline.method=='C_fixed')&(baseline.budget==b)].iloc[0]
        pe=bool(n.catastrophic_mc<=old.catastrophic_mc and n.BC_mc_ge_0p5>=old.BC_mc_ge_0p5 and n.Dmax_le_3>=old.Dmax_le_3 and n.median_BC_mc>=old.median_BC_mc-.02)
        off=bool(n.official_frontend>=old.official_frontend+1 and n.official_hanabi>=old.official_hanabi+1)
        checks.append({'budget':b,'PE_pass':pe,'official_pass':off,'BC_delta':int(n.BC_mc_ge_0p5-old.BC_mc_ge_0p5),
            'Dmax_delta':int(n.Dmax_le_3-old.Dmax_le_3),'front_delta':int(n.official_frontend-old.official_frontend),
            'hanabi_delta':int(n.official_hanabi-old.official_hanabi)})
    n=budgets[(budgets.method=='waveform_only')&(budgets.budget==10)].iloc[0]
    old=baseline[(baseline.deployment==dep)&(baseline.method=='waveform_only')&(baseline.budget==10)].iloc[0]
    wave=bool(n.catastrophic_mc==0 and n.BC_mc_ge_0p5>=old.BC_mc_ge_0p5 and n.Dmax_le_3>=old.Dmax_le_3)
    return {'target_pass':all(c['PE_pass'] and c['official_pass'] for c in checks) and wave,
            'PE_cells':sum(c['PE_pass'] for c in checks),'official_cells':sum(c['official_pass'] for c in checks),'waveform_PE_pass':wave,
            'BC_gain':sum(c['BC_delta'] for c in checks),'Dmax_gain':sum(c['Dmax_delta'] for c in checks),
            'front_gain':sum(c['front_delta'] for c in checks),'hanabi_gain':sum(c['hanabi_delta'] for c in checks),'details':checks}


def run(root):
    trial=root/'trials/RUNWISE-DEVELOPMENT-PARETO'
    if trial.exists():raise RuntimeError('Independent output required')
    for name in ('contracts','tables','configs','results'):(trial/name).mkdir(parents=True)
    shutil.copy2(__file__,trial/'contracts'/Path(__file__).name)
    dev.json_write(trial/'contracts/ADAPTIVE_DEVELOPMENT_ADDENDUM.json',{
        'created_utc':datetime.now(timezone.utc).isoformat(),'version':'runwise-real-development-v1',
        'reason':'previous common-coefficient search had no qualifying solution; allow the same scientifically named waveform rule to have a different operating point by observing run',
        'unchanged':'same architecture, loss, inputs, feature and score formula, 288-point grid, validation guardrails, target criteria and deterministic selection algorithm for BOTH runs',
        'numeric_parameters':'one triplet per run, shared across its three seeds; original per-seed FRT and C-fixed outer weights frozen',
        'formula':'Zold+a*(Z_FRT-Zold)+b*bounded_predictive_mass_reference_overlap+c*bounded_simulation_calibrated_RNC_cosine',
        'selection_used_real_development_PE_and_official_budgets':True,
        'not_claimed':'not validation-only final choice; not blind real-catalog validation; not lensing confirmation',
        'calibrators':'simulation-development data only, public PE and FPP never score inputs',
        'grid':{'a':g.SCALES,'b':g.REF,'c':g.COS},
        'selection_order':'all original targets and reused injection guards; most PE count improvement, then balanced and total official gains, weakest coefficient change, lexical ties',
        'future_test':'must freeze before NEW independent source/noise injections; real goals remain development-only even if new injections pass',
        'no_candidate_specific_rules':True,'time_sky_scope_outer_weights_unchanged':True})
    baseline=pd.read_csv(root/'contracts/BASELINE_BUDGETS.csv');baseline=baseline[baseline.seed.astype(str)=='consensus']
    chosen={};allrows=[];allbudgets=[];frames={};features={};base_metrics={}
    for dep in e.DEPS:
        pe=pd.read_parquet(root/f'audit/{dep}_frozen_external_reference.parquet')
        prior=e.reference_prior(dep)
        for ms,es in zip(body.MODEL_SEEDS,dev.SEEDS):
            specs={recipe.split('-')[0]:e.fit_reference(dep,ms,recipe,prior) for recipe in ('REF-RAW','COS-ISO')}
            dev.json_write(trial/f'configs/{dep}_{es}_simulation_calibration.json',specs)
            for split in ('validation','test','real'):
                f,x=e.frame_features(dep,ms,es,split,prior);frames[dep,es,split]=f
                features[dep,es,split]={k:e.increment(x,s)[0] for k,s in specs.items()}
                if split!='real':base_metrics[dep,es,split]=ev.metrics(f,f.waveform_score.to_numpy(float),dep,es)
        options=[]
        for a in g.SCALES:
            for b in g.REF:
                for c in g.COS:
                    spec={'tail_scale':a,'ref_weight':b,'cos_weight':c};code=f'T{a:g}_R{b:g}_C{c:g}'
                    valpass=True
                    for es in dev.SEEDS:
                        f=frames[dep,es,'validation'];z=g.score(f,features[dep,es,'validation'],spec)
                        if not ev.guard(ev.metrics(f,z,dep,es),base_metrics[dep,es,'validation']):valpass=False;break
                    if not valpass:
                        allrows.append({'deployment':dep,'id':code,**spec,'validation_pass':False,'target_pass':False});continue
                    testpass=True
                    for es in dev.SEEDS:
                        f=frames[dep,es,'test'];z=g.score(f,features[dep,es,'test'],spec)
                        if not ev.guard(ev.metrics(f,z,dep,es),base_metrics[dep,es,'test']):testpass=False
                    fs={es:frames[dep,es,'real'] for es in dev.SEEDS}
                    zs={es:g.score(fs[es],features[dep,es,'real'],spec) for es in dev.SEEDS}
                    budgets=[]
                    for method in ('C_fixed','waveform_only'):
                        cons=g.quick_consensus(fs,zs,pe,dep,method);budgets.extend(g.budget_summary(cons,dep,method))
                    t=target(pd.DataFrame(budgets),baseline,dep)
                    allrows.append({'deployment':dep,'id':code,**spec,'validation_pass':True,'test_pass':testpass,**{k:v for k,v in t.items() if k!='details'}})
                    allbudgets.extend([{**bb,'id':code} for bb in budgets])
                    dev.json_write(trial/f'configs/{dep}_{code}.json',{**spec,**t,'test_pass':testpass})
                    if t['target_pass'] and testpass:
                        balanced=min(min(cc['front_delta'],cc['hanabi_delta']) for cc in t['details'])
                        objective=(-t['BC_gain']-t['Dmax_gain'],-balanced,-t['front_gain']-t['hanabi_gain'],abs(a-1)+b+c,a,b,c)
                        options.append((objective,code,spec))
                dev.csv_write(trial/'tables/ALL_CONFIGURATIONS.csv',pd.DataFrame(allrows))
                dev.csv_write(trial/'tables/ALL_PE_OFFICIAL_BUDGETS.csv',pd.DataFrame(allbudgets))
        if options:
            _,code,spec=min(options,key=lambda v:v[0]);chosen[dep]=spec
            dev.json_write(trial/f'contracts/{dep}_SELECTED.json',{'id':code,**spec,'real_used_as_development':True})
        print(json.dumps({'runwise_search':dep,'qualifying':len(options),'selected':chosen.get(dep)}),flush=True)
    dev.json_write(trial/'contracts/SEARCH_COMPLETE.json',{'both_runs_qualify':len(chosen)==2,'choices':chosen,'fresh_confirmation_done':False})
    if len(chosen)!=2:return
    metrics=[];guards=[]
    for dep in e.DEPS:
        real={}
        for es in dev.SEEDS:
            for split in ('validation','test','real'):
                f=frames[dep,es,split];z=g.score(f,features[dep,es,split],chosen[dep])
                changed=f.copy();changed['FRT_baseline_waveform_score']=f.waveform_score;changed['waveform_score']=z
                out=trial/f'evaluation/{dep}/seed_{es}';out.mkdir(parents=True,exist_ok=True)
                changed.to_parquet(out/f'{split}_pairs.parquet',index=False)
                if split=='real':real[es]=changed
                else:
                    old=base_metrics[dep,es,split];new=ev.metrics(f,z,dep,es)
                    guards.append({'deployment':dep,'seed':es,'split':split,'pass':ev.guard(new,old)})
                    for method in old:
                        for conf,mm in [('FRT_BASELINE',old[method]),('CANDIDATE',new[method])]:
                            metrics.append({'deployment':dep,'seed':es,'split':split,'method':method,'config':conf,**mm})
        pe=pd.read_parquet(root/f'audit/{dep}_frozen_external_reference.parquet')
        dev.save_evaluation(trial,'CANDIDATE',dep,real,pe)
    dev.csv_write(trial/'tables/RETRIEVAL_PER_SEED.csv',pd.DataFrame(metrics));dev.csv_write(trial/'tables/REUSED_GUARDRAILS.csv',pd.DataFrame(guards))
    e.assess(root,trial)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();run(a.root)
