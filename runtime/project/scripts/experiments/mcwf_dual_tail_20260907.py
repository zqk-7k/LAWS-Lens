#!/usr/bin/env python3
"""Agreement and contradiction from two waveform-only mass estimators."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='2'
import argparse
from datetime import datetime,timezone
from pathlib import Path
import shutil
import json
import numpy as np
import pandas as pd
import mcwf_pe_frontend_extension_v2_20260907 as e
import mcwf_finite_reference_tail_20260907 as tail

dev,body,ev=e.dev,e.body,e.ev
P0=(.05,.25,.5,1.)
GAMMA=(0.,.125,.25,.5,1.,2.,4.)
BETA=(0.,.125,.25,.5,1.,2.,4.)


def components(f,x,spec):
    compatibility=tail.tail_probability(f.waveform_abs_delta_logmc_std.to_numpy(float),spec['old_mass_reference'])
    ref,_,ood=e.increment(x,spec['new_reference'])
    return compatibility,np.maximum(ref,0.)*compatibility,ood


def score(f,cfg,data):
    comp,reward,ood=data
    penalty=np.minimum(np.log(comp/cfg['p0']),0.)
    return f.waveform_score.to_numpy(float)+cfg['gamma']*penalty+cfg['beta']*reward,penalty,reward,ood


def run(root):
    trial=root/'trials/DUAL-MASS-TAIL'
    if trial.exists():raise RuntimeError('Independent trial required')
    for n in ('contracts','calibration','tables','results','evaluation'):(trial/n).mkdir(parents=True)
    shutil.copy2(__file__,trial/'contracts'/Path(__file__).name)
    dev.json_write(trial/'contracts/RECIPE.json',{'created_utc':datetime.now(timezone.utc).isoformat(),
        'same_both_runs':True,'formula':'Z_FRT+gamma*min(log(p_old/p0),0)+beta*max(REF_new,0)*p_old',
        'p_old':'finite fraction of validation companions with old waveform Mc regression difference at least as large; NOT a PE overlap or FPP',
        'motivation':'new predictive agreement can be wrong; retain both the original estimator disagreement and the new estimator specificity, rather than only attenuating a positive reward',
        'p0_grid':P0,'gamma_grid':GAMMA,'beta_grid':BETA,
        'all_parameters':'common grid and deterministic selection rule; run/seed numerical choices may differ',
        'selection':'simulation validation only; F50/F90/AP/R10 objective and FRT-relative guardrails',
        'original_FRT_unchanged':True,'time_sky_outer_scope_unchanged':True,
        'no_PE_or_official_score_inputs':True,'real_is_adaptive_development':True,
        'calibrated_ranking_not_independent_evidence':True,'fresh_confirmation_required':True})
    selected={};priors={d:e.reference_prior(d) for d in e.DEPS};states=[];cache={}
    for dep in e.DEPS:
        for ms,es in zip(body.MODEL_SEEDS,dev.SEEDS):
            f,x=e.frame_features(dep,ms,es,'validation',priors[dep]);y=f.is_true_pair.to_numpy(bool)
            spec={'new_reference':e.fit_reference(dep,ms,'REF-RAW',priors[dep]),
                  'old_mass_reference':np.sort(f.waveform_abs_delta_logmc_std.to_numpy(float)[y]).tolist(),
                  'deployment':dep,'model_seed':ms,'eval_seed':es}
            data=components(f,x,spec);cache[dep,es]=(f,data)
            baseline=ev.metrics(f,f.waveform_score.to_numpy(float),dep,es);rows=[];options=[]
            for p0 in P0:
                for gamma in GAMMA:
                    if gamma==0 and p0!=.05:continue
                    for beta in BETA:
                        cfg={**spec,'p0':p0,'gamma':gamma,'beta':beta};z,*_=score(f,cfg,data);m=ev.metrics(f,z,dep,es);passed=ev.guard(m,baseline)
                        rows.append({'p0':p0,'gamma':gamma,'beta':beta,'pass':passed,**{method+'_'+k:v for method,mm in m.items() for k,v in mm.items()}})
                        if passed:options.append((e.old_selection.objective(m,gamma+beta,p0)+(gamma,beta),cfg))
            out=trial/f'calibration/{dep}/seed_{es}';out.mkdir(parents=True);dev.csv_write(out/'validation_grid.csv',pd.DataFrame(rows))
            cfg=min(options,key=lambda c:c[0])[1];selected[dep,es]=cfg;dev.json_write(out/'SELECTED_CONFIG.json',cfg)
            states.append({'deployment':dep,'eval_seed':es,'p0':cfg['p0'],'gamma':cfg['gamma'],'beta':cfg['beta']})
    dev.json_write(trial/'contracts/SELECTED_FREEZE.json',{'status':states,'selection_used_real':False});print(json.dumps({'dual_selected':states}),flush=True)
    rows=[];guards=[]
    for dep in e.DEPS:
        real={}
        for ms,es in zip(body.MODEL_SEEDS,dev.SEEDS):
            for split in ('validation','test','real'):
                if split=='validation':f,data=cache[dep,es]
                else:
                    f,x=e.frame_features(dep,ms,es,split,priors[dep]);data=components(f,x,selected[dep,es])
                z,penalty,reward,ood=score(f,selected[dep,es],data)
                new=f.copy();new['FRT_baseline_waveform_score']=f.waveform_score;new['waveform_score']=z
                new['old_mass_tail_penalty']=penalty;new['concordant_reward']=reward;new['new_positive_ood']=ood
                out=trial/f'evaluation/{dep}/seed_{es}';out.mkdir(parents=True,exist_ok=True);new.to_parquet(out/f'{split}_pairs.parquet',index=False)
                if split=='real':real[es]=new
                else:
                    bm=ev.metrics(f,f.waveform_score.to_numpy(float),dep,es);mm=ev.metrics(f,z,dep,es)
                    guards.append({'deployment':dep,'seed':es,'split':split,'pass':ev.guard(mm,bm)})
                    for method in bm:
                        for name,m in [('FRT_BASELINE',bm[method]),('CANDIDATE',mm[method])]:
                            rows.append({'deployment':dep,'seed':es,'split':split,'method':method,'config':name,**m})
        dev.save_evaluation(trial,'CANDIDATE',dep,real,pd.read_parquet(root/f'audit/{dep}_frozen_external_reference.parquet'))
    dev.csv_write(trial/'tables/RETRIEVAL_PER_SEED.csv',pd.DataFrame(rows));dev.csv_write(trial/'tables/REUSED_GUARDRAILS.csv',pd.DataFrame(guards))
    e.assess(root,trial)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();run(a.root)
