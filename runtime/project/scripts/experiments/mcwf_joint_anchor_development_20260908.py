#!/usr/bin/env python3
"""Finite shared adaptive-development study of the old waveform anchor."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import sys
import numpy as np
import pandas as pd

P=Path('/root/autodl-tmp/gw-catalog');sys.path.insert(0,str(P/'scripts/experiments'))
import mcwf_joint_global_development_v2_20260908 as g
ev,t,dev,cf=g.ev,g.t,g.dev,g.cf
READ=g.read_values
ALPHAS=(0.,.25,.5,.75,1.,1.25,1.5,2.,4.)
COEFFICIENTS=(0.,.0625,.125,.25,.5,1.,2.,4.,8.)


def initialize(root):
    if root.exists():raise RuntimeError('Independent directory required')
    for name in ('contracts','scripts','logs','tables','reports','figures','manifest','calibration','evaluation'):(root/name).mkdir(parents=True)
    dev.json_write(root/'contracts/ANALYSIS_CONTRACT.json',{
      'id':'MCWF-JOINT-ANCHOR-DEVELOPMENT-22','UTC':datetime.now(timezone.utc).isoformat(),'status':t.STATUS,'goal_achieved':False,
      'change':'Previous joint REPLACE arms still fixed the pre-OMC FRT waveform anchor at1. Explicitly vary its contribution,including0,without changing time/sky or individual-pair exceptions.',
      'formula':'Zwf=alpha*Z_FRT+gamma*jointtail+beta*boundedjointLR',
      'same_numerical_coefficients_all_runs_and_seeds':True,
      'grid':{'alpha':ALPHAS,'gamma':COEFFICIENTS,'beta':COEFFICIENTS,'joint':['CHI','ETA-CHI'],'calibration':['BC','PRIOR']},
      'configurations':2916,'pure_joint_control':'alpha0 removesFRTanchor;gamma0 removesoptionalconflictpenalty;beta0 removesoptionalpositivereward. Reportactive components.',
      'calibration':'Frozen12simulation-only densitycalibrations;no neuraltraining orPE-based densityfit.',
      'selection':'Explicitly adaptive realdevelopment outcomes AFTER allsimulationvalidationguards. Reusedtest guardalso required;notblind. Choosejointtargetthenminimumrunfrontendgain,Mcpassgain,medianBCgain,totalfrontendgain,smallest(alpha-1)^2+gamma^2+beta^2,lexicographic.',
      'guardrails':'Unchanged perseed waveform+fusion R10-.02;AP-.005;F50/F90<=1.1baseline;bothruns Top10/20 PE/official/Hanabi protection,strictPE+frontendgain perrun.',
      'ranking_inputs':'Only frozenwaveformoutputs;no PE,FPP,officiallabels,Hanabiresults,eventnames orcandidate-specific rules.',
      'external_selection_disclosure':'PE/official labels ARE outcome metrics for globalhyperparameter selection. This is development optimization,not independentrealvalidation,not validation-only coefficient selection.',
      'changed_channels':['waveform'],'outer_weights_changed':False,
      'frozen':['allencoders','time_score','sky_raw_log_bf','outerC-fixedweights','scope','history','paper'],
      'required':'One selectedconfiguration frozenbeforeNEWsource/noise confirmation;no declarationofgoaluntilindependent simulationguardrails pass;no claimofconfirmedlensing.'})
    rows=t.protected()
    for p in (g.UPSTREAM/'calibration').rglob('*.json'):rows.append({'path':str(p),'sha256':dev.sha(p),'bytes':p.stat().st_size})
    dev.csv_write(root/'manifest/INPUT_SHA256.csv',pd.DataFrame(rows));shutil.copy2(__file__,root/'scripts/joint_anchor_development.py')
    dev.json_write(root/'contracts/START_FREEZE.json',{'contract_sha256':dev.sha(root/'contracts/ANALYSIS_CONTRACT.json'),'code_sha256':dev.sha(Path(__file__))})


def task(spec):
    kind,statistic,alpha=spec
    def scaled(*args):
        frame,p,i,a=READ(*args);frame=frame.copy()
        frame['FRT_baseline_waveform_score']=alpha*frame.FRT_baseline_waveform_score
        return frame,p,i,a
    g.read_values=scaled;g.GRID=COEFFICIENTS
    rows,budgets,metrics,candidates=g.task((kind,statistic,'REPLACE'))
    prefix=f'A{alpha:g}-'
    for row in rows:
        row['configuration']=prefix+row['configuration'];row['alpha']=alpha
    for row in budgets:row['config']=prefix+row['config']
    for row in metrics:row['configuration']=prefix+row['configuration']
    for row in candidates:
        row['configuration']=prefix+row['configuration'];row['alpha']=alpha
        row['objective']=(-min(row[d+'_frontend_gain'] for d in t.DEPS),-sum(row[d+'_Mc_count_gain'] for d in t.DEPS),
            -sum(row[d+'_Mc_median_gain'] for d in t.DEPS),-sum(row[d+'_frontend_gain'] for d in t.DEPS),
            (alpha-1)**2+row['gamma']**2+row['beta']**2,row['configuration'])
    return rows,budgets,metrics,candidates


def select(root):
    if (root/'contracts/SEARCH_COMPLETE.json').exists():raise RuntimeError('Alreadycomplete')
    specs=[(kind,stat,alpha) for kind in ('CONDITIONAL-CHI','CONDITIONAL-ETA-CHI') for stat in ('BC','PRIOR') for alpha in ALPHAS]
    rows=[];budgets=[];metrics=[];success=[]
    with ProcessPoolExecutor(max_workers=8) as pool:
        for a,b,c,d in pool.map(task,specs):
            rows+=a;budgets+=b;metrics+=c;success+=d
            dev.csv_write(root/'tables/ALL_GLOBAL_CONFIGURATIONS.csv',pd.DataFrame(rows))
            dev.csv_write(root/'tables/ALL_GLOBAL_PE_OFFICIAL_BUDGETS.csv',pd.DataFrame(budgets))
            dev.csv_write(root/'tables/ALL_REUSED_SIMULATION_METRICS.csv',pd.DataFrame(metrics))
            print(json.dumps({'configurations':len(rows),'qualifying':len(success)}),flush=True)
    dev.json_write(root/'contracts/SEARCH_COMPLETE.json',{'configurations':len(rows),'qualifying':len(success),'real_used_for_development_selection':True,'fresh_confirmation':False})
    dev.json_write(root/'contracts/ALL_DEVELOPMENT_QUALIFIERS.json',[{k:v for k,v in s.items() if k!='calibrations'} for s in success])
    configs=[]
    if success:
        chosen=min(success,key=lambda a:a['objective'])
        dev.json_write(root/'contracts/SELECTED_GLOBAL_DEVELOPMENT.json',{k:v for k,v in chosen.items() if k!='calibrations'})
        for dep in t.DEPS:
            for slot,seed in zip(t.MODEL_SLOTS,t.SEEDS):
                configs.append({'method':'GLOBAL-ANCHOR-'+chosen['configuration'],'deployment':dep,'seed':seed,'slot':slot,
                    'kind':chosen['kind'],'statistic':chosen['statistic'],'integration':'REPLACE','alpha':chosen['alpha'],
                    'gamma':chosen['gamma'],'beta':chosen['beta'],'weights':cf.frozen_weights(dep,seed).tolist(),
                    'calibration':chosen['calibrations'][f'{dep}_{seed}'],'real_used_for_development_selection':True})
    dev.json_write(root/'calibration/SELECTED.json',configs)
    dev.csv_write(root/'tables/SELECTED_COEFFICIENTS.csv',pd.DataFrame([{k:v for k,v in c.items() if k not in ('weights','calibration')} for c in configs]))
    dev.json_write(root/'contracts/INTEGRATION_FROZEN.json',{'file':'calibration/SELECTED.json','sha256':dev.sha(root/'calibration/SELECTED.json'),
        'UTC':datetime.now(timezone.utc).isoformat(),'real_or_reused_test_selection':True,'not_blind_confirmation':True})


def scored(root,c,split):
    if c['method']=='OMC':
        f=cf.read(c['deployment'],c['seed'],split);return f,f.waveform_score.to_numpy(float),{}
    f,p,i,a=READ(c['deployment'],c['slot'],c['seed'],split,c['kind'],c['calibration'])
    return f,c['alpha']*f.FRT_baseline_waveform_score.to_numpy(float)+c['gamma']*p+c['beta']*i,{**a,'penalty':p,'increment':i}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--stage',required=True,choices=['initialize','select','evaluate','real','assess']);a=p.parse_args();ev.scored=scored
    if a.stage in ('initialize','select'):globals()[a.stage](a.root)
    else:getattr(ev,a.stage)(a.root)
