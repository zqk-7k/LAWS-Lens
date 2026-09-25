#!/usr/bin/env python3
"""Explicitly adaptive real-catalog development; never a blind real test."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key]='2'
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import mcwf_pe_frontend_extension_v2_20260907 as e

dev, body, ev = e.dev, e.body, e.ev
SCALES=(.5,.75,1.,1.25,1.5,2.,3.,4.)
REF=(0.,.0625,.125,.25,.5,.75,1.,1.5,2.)
COS=(0.,.125,.25,.5)


def score(f, x, spec):
    old=f.previous_waveform_score.to_numpy(float)
    delta=f.waveform_score.to_numpy(float)-old
    return old+spec['tail_scale']*delta+spec['ref_weight']*x['REF']+spec['cos_weight']*x['COS']


def budget_summary(frame, dep, method):
    return [dev.budget_row(frame,'CANDIDATE',dep,method,b) for b in (10,20,50,100)]


def quick_consensus(frames, scores, pe, dep, method):
    ranks=[]
    for es,f in frames.items():
        changed=f.copy();changed['waveform_score']=scores[es]
        w={'waveform':1.,'time':0.,'sky':0.} if method=='waveform_only' else dev.BASE.FROZEN_V93_WEIGHTS[dep][es]
        ranks.append(dev.BASE.rank_real(changed,w,method,es))
    out=dev.BASE.consensus_real(ranks,method)
    cols=[c for c in pe if c not in out or c=='pair_key']
    return out.merge(pe[cols],on='pair_key',validate='one_to_one')


def target_summary(budgets, baseline):
    realchecks=[];wavechecks=[]
    for dep in e.DEPS:
        for b in (10,20):
            n=budgets[(budgets.deployment==dep)&(budgets.method=='C_fixed')&(budgets.budget==b)].iloc[0]
            old=baseline[(baseline.deployment==dep)&(baseline.method=='C_fixed')&(baseline.budget==b)].iloc[0]
            pe=bool(n.catastrophic_mc<=old.catastrophic_mc and n.BC_mc_ge_0p5>=old.BC_mc_ge_0p5 and n.Dmax_le_3>=old.Dmax_le_3 and n.median_BC_mc>=old.median_BC_mc-.02)
            official=bool(n.official_frontend>=old.official_frontend+1 and n.official_hanabi>=old.official_hanabi+1)
            realchecks.append({'dep':dep,'budget':b,'PE_pass':pe,'official_pass':official,
                'catastrophic_delta':int(n.catastrophic_mc-old.catastrophic_mc),
                'BC_delta':int(n.BC_mc_ge_0p5-old.BC_mc_ge_0p5),'Dmax_delta':int(n.Dmax_le_3-old.Dmax_le_3),
                'front_delta':int(n.official_frontend-old.official_frontend),'hanabi_delta':int(n.official_hanabi-old.official_hanabi)})
        n=budgets[(budgets.deployment==dep)&(budgets.method=='waveform_only')&(budgets.budget==10)].iloc[0]
        old=baseline[(baseline.deployment==dep)&(baseline.method=='waveform_only')&(baseline.budget==10)].iloc[0]
        wavechecks.append(bool(n.catastrophic_mc==0 and n.BC_mc_ge_0p5>=old.BC_mc_ge_0p5 and n.Dmax_le_3>=old.Dmax_le_3))
    passed=all(c['PE_pass'] and c['official_pass'] for c in realchecks) and all(wavechecks)
    return {'target_pass':passed,'PE_cells':sum(c['PE_pass'] for c in realchecks),
            'official_cells':sum(c['official_pass'] for c in realchecks),'waveform_PE_pass':all(wavechecks),
            'front_gain':sum(c['front_delta'] for c in realchecks),'hanabi_gain':sum(c['hanabi_delta'] for c in realchecks),
            'BC_gain':sum(c['BC_delta'] for c in realchecks),'Dmax_gain':sum(c['Dmax_delta'] for c in realchecks),'details':realchecks}


def run(root):
    trial=root/'trials/GLOBAL-DEVELOPMENT-PARETO'
    if trial.exists():raise RuntimeError('No overwrite; new protocol version required')
    for name in ('contracts','tables','configs','results'):(trial/name).mkdir(parents=True)
    shutil.copy2(__file__,trial/'contracts'/Path(__file__).name)
    contract={'created_utc':datetime.now(timezone.utc).isoformat(),
        'version':'explicit-real-development-v1',
        'change_from_prior_trials':'after validation-only mechanisms failed, explicitly allow the already-examined real PE/official budget goals to select three GLOBAL waveform combination coefficients',
        'not_claimed':'not validation-only coefficient selection, not a pristine real test, not a population Bayes factor, not lens detection',
        'no_real_labels_as_inputs':'PE values, FPP values, official labels and event names never enter event features or pair score; they are development OUTCOME metrics for global hyperparameter selection',
        'same_parameter_values_all_runs_and_seeds':True,
        'formula':'Zold+tail_scale*(Z_FRT-Zold)+ref_weight*bounded_predictive_mass_reference_overlap+cos_weight*bounded_simulation_calibrated_RNC_cosine',
        'tail_scale_grid':SCALES,'ref_weight_grid':REF,'cos_weight_grid':COS,
        'finite_search_size':len(SCALES)*len(REF)*len(COS),
        'parameters_pruned_by':'all six simulation validation waveform and C-fixed guardrails relative to current FRT',
        'development_success_criteria':'exactly the already frozen root EXPERIMENT_CONTRACT, unchanged',
        'selection_order':'target PASS; all reused simulation guards PASS; most PE count improvement, then largest balanced minimum official gain, then summed official gains; weakest total deviation from baseline and lexicographic tie-break',
        'fresh_confirmation':'mandatory after selection with NEW source/noise realizations; old tests explicitly treated as reused development',
        'prohibited':'no candidate-specific exceptions, no time/sky/outerweight updates, no event-id scoring, no historical overwrites',
        'final_status':'HOLD_FOR_AUTHOR_REVIEW_NO_ADOPTION_NO_OVERWRITE'}
    dev.json_write(trial/'contracts/ADAPTIVE_DEVELOPMENT_ADDENDUM.json',contract)
    baseline=pd.read_csv(root/'contracts/BASELINE_BUDGETS.csv')
    baseline=baseline[baseline.seed.astype(str)=='consensus']
    frames={};features={};base_metrics={};pe={d:pd.read_parquet(root/f'audit/{d}_frozen_external_reference.parquet') for d in e.DEPS}
    for dep in e.DEPS:
        prior=e.reference_prior(dep)
        for ms,es in zip(body.MODEL_SEEDS,dev.SEEDS):
            specs={recipe.split('-')[0]:e.fit_reference(dep,ms,recipe,prior) for recipe in ('REF-RAW','COS-ISO')}
            dev.json_write(trial/f'configs/{dep}_{es}_simulation_calibration.json',specs)
            for split in ('validation','test','real'):
                f,x=e.frame_features(dep,ms,es,split,prior)
                frames[dep,es,split]=f
                features[dep,es,split]={k:e.increment(x,spec)[0] for k,spec in specs.items()}
                if split!='real':base_metrics[dep,es,split]=ev.metrics(f,f.waveform_score.to_numpy(float),dep,es)
    dev.json_write(trial/'contracts/INPUTS_FROZEN.json',{'baseline_identity_check':'grid(1,0,0) must reproduce current FRT exactly',
        'calibration_hashes':{str(p.relative_to(trial)):dev.sha(p) for p in sorted((trial/'configs').glob('*.json'))}})
    rows=[];allbudgets=[];success=[]
    for scale in SCALES:
        for ref in REF:
            for cos in COS:
                spec={'tail_scale':scale,'ref_weight':ref,'cos_weight':cos}
                code=f'T{scale:g}_R{ref:g}_C{cos:g}'
                valpass=True;testpass=True
                for dep in e.DEPS:
                    for es in dev.SEEDS:
                        f=frames[dep,es,'validation'];z=score(f,features[dep,es,'validation'],spec)
                        if not ev.guard(ev.metrics(f,z,dep,es),base_metrics[dep,es,'validation']):valpass=False;break
                    if not valpass:break
                if not valpass:
                    rows.append({'id':code,**spec,'validation_pass':False,'test_pass':None,'target_pass':False})
                    continue
                for dep in e.DEPS:
                    for es in dev.SEEDS:
                        f=frames[dep,es,'test'];z=score(f,features[dep,es,'test'],spec)
                        if not ev.guard(ev.metrics(f,z,dep,es),base_metrics[dep,es,'test']):testpass=False
                budgets=[]
                for dep in e.DEPS:
                    fs={es:frames[dep,es,'real'] for es in dev.SEEDS}
                    zs={es:score(fs[es],features[dep,es,'real'],spec) for es in dev.SEEDS}
                    for method in ('C_fixed','waveform_only'):
                        cons=quick_consensus(fs,zs,pe[dep],dep,method)
                        budgets.extend(budget_summary(cons,dep,method))
                bd=pd.DataFrame(budgets)
                target=target_summary(bd,baseline)
                row={'id':code,**spec,'validation_pass':True,'test_pass':testpass,**{k:v for k,v in target.items() if k!='details'}}
                rows.append(row)
                allbudgets.extend([{**b,'id':code} for b in budgets])
                dev.json_write(trial/f'configs/{code}.json',{**spec,**target,'test_pass':testpass})
                if scale==1 and ref==0 and cos==0:
                    for b in budgets:
                        match=baseline[(baseline.deployment==b['deployment'])&(baseline.method==b['method'])&(baseline.budget==b['budget'])].iloc[0]
                        for col in ('catastrophic_mc','BC_mc_ge_0p5','Dmax_le_3','official_frontend','official_hanabi'):
                            assert b[col]==match[col],(col,b,match.to_dict())
                if target['target_pass'] and testpass:
                    balanced=min(min(c['front_delta'],c['hanabi_delta']) for c in target['details'])
                    objective=(-target['BC_gain']-target['Dmax_gain'],-balanced,-target['front_gain']-target['hanabi_gain'],abs(scale-1)+ref+cos,scale,ref,cos)
                    success.append((objective,code,spec))
                if len(rows)%12==0:
                    print(json.dumps({'global_grid_done':len(rows),'success':len(success),'last':row}),flush=True)
            dev.csv_write(trial/'tables/ALL_GLOBAL_CONFIGURATIONS.csv',pd.DataFrame(rows))
            dev.csv_write(trial/'tables/ALL_GLOBAL_PE_OFFICIAL_BUDGETS.csv',pd.DataFrame(allbudgets))
    dev.json_write(trial/'contracts/SEARCH_COMPLETE.json',{'configurations':len(rows),'qualifying':len(success),'real_is_development':True,'fresh_confirmation_done':False})
    if success:
        _,code,spec=min(success,key=lambda x:x[0])
        dev.json_write(trial/'contracts/SELECTED_GLOBAL_CONFIG.json',{'id':code,**spec,'selection_used_real_development_outcomes':True})
        metrics=[];guards=[]
        for dep in e.DEPS:
            real={}
            for es in dev.SEEDS:
                for split in ('validation','test','real'):
                    f=frames[dep,es,split];z=score(f,features[dep,es,split],spec)
                    changed=f.copy();changed['FRT_baseline_waveform_score']=f.waveform_score;changed['waveform_score']=z
                    out=trial/f'evaluation/{dep}/seed_{es}';out.mkdir(parents=True,exist_ok=True)
                    changed.to_parquet(out/f'{split}_pairs.parquet',index=False)
                    if split=='real':real[es]=changed
                    else:
                        b=base_metrics[dep,es,split];n=ev.metrics(f,z,dep,es)
                        guards.append({'deployment':dep,'seed':es,'split':split,'pass':ev.guard(n,b)})
                        for method in b:
                            for c,mm in [('FRT_BASELINE',b[method]),('CANDIDATE',n[method])]:
                                metrics.append({'deployment':dep,'seed':es,'split':split,'method':method,'config':c,**mm})
            dev.save_evaluation(trial,'CANDIDATE',dep,real,pe[dep])
        dev.csv_write(trial/'tables/RETRIEVAL_PER_SEED.csv',pd.DataFrame(metrics));dev.csv_write(trial/'tables/REUSED_GUARDRAILS.csv',pd.DataFrame(guards))
        e.assess(root,trial)
    print(json.dumps({'global_complete':len(rows),'qualifying':len(success)}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();run(a.root)
