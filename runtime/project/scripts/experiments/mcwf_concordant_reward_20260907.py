#!/usr/bin/env python3
"""Reward new predictive specificity only when the old mass estimate agrees."""
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


def score(f,x,spec):
    compatibility=tail.tail_probability(f.waveform_abs_delta_logmc_std.to_numpy(float),spec['old_mass_reference'])
    ref,_,ood=e.increment(x,spec['new_reference'])
    inc=np.maximum(ref,0.)*compatibility**spec['power']
    return f.waveform_score.to_numpy(float)+spec['beta']*inc,inc,compatibility,ood


def run(root):
    trial=root/'trials/CONCORDANT-PREDICTIVE-REWARD'
    if trial.exists():raise RuntimeError('Independent trial required')
    for n in ('contracts','calibration','tables','results','evaluation'):(trial/n).mkdir(parents=True)
    shutil.copy2(__file__,trial/'contracts'/Path(__file__).name)
    dev.json_write(trial/'contracts/RECIPE.json',{'created_utc':datetime.now(timezone.utc).isoformat(),
        'same_both_runs':True,'formula':'Z_FRT+beta*max(bounded predictive mass reference overlap,0)*p_old_Mc_compatibility**power',
        'p_old_Mc_compatibility':'finite reference fraction of validation companions with old regression discrepancy at least as large; NOT a lens posterior or FPP',
        'motivation':'a positive claim from the new mass predictor should not ignore a discrepant old waveform mass estimate; require concordance without removing the current FRT negative tail',
        'power_grid':[1.,2.,4.],'beta_grid':[0.,.25,.5,1.,2.,4.,8.],
        'selection':'simulation validation only, unchanged deterministic objective and FRT-relative guards',
        'time_sky_outer_scope_unchanged':True,'new_training':False,'no_PE_or_official_score_inputs':True,
        'real_is_adaptive_development':True,'fresh_confirmation_required':True})
    selected={};priors={d:e.reference_prior(d) for d in e.DEPS};states=[]
    for dep in e.DEPS:
        for ms,es in zip(body.MODEL_SEEDS,dev.SEEDS):
            f,x=e.frame_features(dep,ms,es,'validation',priors[dep]);y=f.is_true_pair.to_numpy(bool)
            spec={'new_reference':e.fit_reference(dep,ms,'REF-RAW',priors[dep]),
                  'old_mass_reference':np.sort(f.waveform_abs_delta_logmc_std.to_numpy(float)[y]).tolist(),
                  'deployment':dep,'model_seed':ms,'eval_seed':es}
            baseline=ev.metrics(f,f.waveform_score.to_numpy(float),dep,es);rows=[];options=[]
            for power in (1.,2.,4.):
                for beta in (0.,.25,.5,1.,2.,4.,8.):
                    cfg={**spec,'power':power,'beta':beta};z,*_=score(f,x,cfg);m=ev.metrics(f,z,dep,es);passed=ev.guard(m,baseline)
                    rows.append({'power':power,'beta':beta,'pass':passed,**{method+'_'+k:v for method,mm in m.items() for k,v in mm.items()}})
                    if passed:options.append((e.old_selection.objective(m,beta,power),cfg))
            out=trial/f'calibration/{dep}/seed_{es}';out.mkdir(parents=True);dev.csv_write(out/'validation_grid.csv',pd.DataFrame(rows))
            cfg=min(options,key=lambda c:c[0])[1];selected[dep,es]=cfg;dev.json_write(out/'SELECTED_CONFIG.json',cfg)
            states.append({'deployment':dep,'eval_seed':es,'power':cfg['power'],'beta':cfg['beta']})
    dev.json_write(trial/'contracts/SELECTED_FREEZE.json',{'status':states,'selection_used_real':False});print(json.dumps({'concordant_selected':states}),flush=True)
    rows=[];guards=[]
    for dep in e.DEPS:
        real={}
        for ms,es in zip(body.MODEL_SEEDS,dev.SEEDS):
            for split in ('validation','test','real'):
                f,x=e.frame_features(dep,ms,es,split,priors[dep]);z,inc,comp,ood=score(f,x,selected[dep,es])
                new=f.copy();new['FRT_baseline_waveform_score']=f.waveform_score;new['waveform_score']=z
                new['concordant_increment']=inc;new['old_mass_compatibility']=comp;new['concordant_ood']=ood
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
