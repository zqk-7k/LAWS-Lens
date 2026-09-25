#!/usr/bin/env python3
"""Monotone calibration of the entire frozen composite waveform score."""
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
from scipy.optimize import minimize
from scipy.special import expit
import mcwf_pe_frontend_extension_v2_20260907 as e

dev,body,ev=e.dev,e.body,e.ev


def fit(f):
    x=f.waveform_score.to_numpy(float);y=f.is_true_pair.to_numpy(bool);weight=np.where(y,.5/y.sum(),.5/(~y).sum())
    def loss(theta):
        a,b=theta;v=a*x+b;r=weight*(expit(v)-y)
        return float(np.sum(weight*(np.logaddexp(0,v)-y*v))),np.array([np.dot(r,x),r.sum()])
    result=minimize(loss,[1.,0.],jac=True,bounds=[(.001,100.),(None,None)],method='L-BFGS-B',options={'ftol':1e-12,'gtol':1e-9,'maxiter':1000})
    if not result.success:raise RuntimeError(result.message)
    return {'slope':float(result.x[0]),'intercept':float(result.x[1]),'fit_min':float(x.min()),'fit_max':float(x.max()),
            'balanced_logloss':float(result.fun),'uncalibrated_balanced_logloss':loss([1.,0.])[0],
            'fit_positive_systems':int(y.sum()),'fit_negative_pairs':int((~y).sum()),'optimizer_success':True}


def score(f,spec):
    x=f.waveform_score.to_numpy(float);ood=(x<spec['fit_min'])|(x>spec['fit_max'])
    fitted=spec['slope']*x+spec['intercept'];delta=fitted-x
    delta=np.where(ood & (delta>0),0,delta)
    return x+spec['fraction']*delta,delta,ood


def run(root):
    trial=root/'trials/COMPOSITE-WAVEFORM-CALIBRATION'
    if trial.exists():raise RuntimeError('Independent trial required')
    for n in ('contracts','calibration','tables','results','evaluation'):(trial/n).mkdir(parents=True)
    shutil.copy2(__file__,trial/'contracts'/Path(__file__).name)
    dev.json_write(trial/'contracts/RECIPE.json',{'created_utc':datetime.now(timezone.utc).isoformat(),
        'same_both_runs':True,'input':'entire frozen Z_FRT, not real PE or official candidates',
        'fit':'positive-slope sigmoid calibration, equal total classweights, BAYESTAR validation true/nontrue pairs',
        'new_waveform_formula':'Z_FRT + fraction*((a*Z_FRT+b)-Z_FRT); positive increments disabled outside fit support',
        'fraction_grid':[0.,.25,.5,1.],'selection':'existing simulation-validation priority and current FRT guardrails',
        'scope_of_change':'waveform composite score calibration ONLY, no new encoder information',
        'relative_contribution_disclosure':'changing waveform scale changes its relative strength in fusion even though outerC-fixedweights and time/sky stay fixed; this is explicitly a calibration ablation, not an unchanged full scoring function',
        'classification_prior':'class-balanced log odds estimate a conditional score likelihood ratio, NOT lens posterior odds or full strainBF',
        'frozen':['all encoders','all time/sky scores','outerweights','scope','historical results'],
        'same_real_data_used_adaptively':True,'fresh_confirmation_required':True,
        'references':['https://scikit-learn.org/stable/modules/calibration.html','https://arxiv.org/abs/2104.08846']})
    selected={};states=[]
    for dep in e.DEPS:
        for es in dev.SEEDS:
            f=pd.read_parquet(e.BASE/f'evaluation/{dep}/seed_{es}/validation_pairs.parquet');spec=fit(f);baseline=ev.metrics(f,f.waveform_score.to_numpy(float),dep,es)
            rows=[];choices=[]
            for fraction in (0.,.25,.5,1.):
                cfg={**spec,'fraction':fraction};z,_,ood=score(f,cfg);m=ev.metrics(f,z,dep,es);ok=ev.guard(m,baseline)
                rows.append({'fraction':fraction,'pass':ok,'ood':float(ood.mean()),**{a+'_'+k:v for a,b in m.items() for k,v in b.items()}})
                if ok:choices.append((e.old_selection.objective(m,fraction,1),cfg))
            out=trial/f'calibration/{dep}/seed_{es}';out.mkdir(parents=True);dev.csv_write(out/'validation_grid.csv',pd.DataFrame(rows))
            cfg=min(choices,key=lambda c:c[0])[1];selected[dep,es]=cfg;dev.json_write(out/'SELECTED_CONFIG.json',cfg)
            states.append({'deployment':dep,'seed':es,**cfg})
    dev.json_write(trial/'contracts/SELECTED_FREEZE.json',{'configs':states,'real_used_to_choose_calibration':False});print(json.dumps({'score_calibration_selected':states}),flush=True)
    rows=[];guards=[]
    for dep in e.DEPS:
        real={}
        for es in dev.SEEDS:
            for split in ('validation','test','real'):
                f=pd.read_parquet(e.BASE/f'evaluation/{dep}/seed_{es}/{split}_pairs.parquet');z,delta,ood=score(f,selected[dep,es])
                n=f.copy();n['FRT_baseline_waveform_score']=f.waveform_score;n['waveform_score']=z;n['composite_calibration_delta']=delta;n['composite_calibration_ood']=ood
                out=trial/f'evaluation/{dep}/seed_{es}';out.mkdir(parents=True,exist_ok=True);n.to_parquet(out/f'{split}_pairs.parquet',index=False)
                if split=='real':real[es]=n
                else:
                    bm=ev.metrics(f,f.waveform_score.to_numpy(float),dep,es);m=ev.metrics(f,z,dep,es);guards.append({'deployment':dep,'seed':es,'split':split,'pass':ev.guard(m,bm)})
                    for method in m:
                        for config,metrics in [('FRT_BASELINE',bm[method]),('CANDIDATE',m[method])]:rows.append({'deployment':dep,'seed':es,'split':split,'method':method,'config':config,**metrics})
        dev.save_evaluation(trial,'CANDIDATE',dep,real,pd.read_parquet(root/f'audit/{dep}_frozen_external_reference.parquet'))
    dev.csv_write(trial/'tables/RETRIEVAL_PER_SEED.csv',pd.DataFrame(rows));dev.csv_write(trial/'tables/REUSED_GUARDRAILS.csv',pd.DataFrame(guards));e.assess(root,trial)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();run(a.root)
