#!/usr/bin/env python3
"""Explicit finite adaptive-development combinations of frozen waveform evidence."""
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
import mcwf_temporal_response_evaluate_20260908 as ev
import mcwf_joint_intrinsics_evidence_20260908 as joint
t,dev,cf=ev.t,ev.dev,ev.cf
UPSTREAM=P/'results/mcwf_joint_intrinsic_evidence_exploratory_20260908T155830Z'
GRID=(0.,.03125,.0625,.125,.25,.5,1.,2.,4.)


def initialize(root):
    if root.exists():raise RuntimeError('Independent directory required')
    for name in ('contracts','scripts','logs','tables','reports','figures','manifest','calibration','evaluation'):(root/name).mkdir(parents=True)
    dev.json_write(root/'contracts/ANALYSIS_CONTRACT.json',{
      'id':'MCWF-JOINT-GLOBAL-DEVELOPMENT-21','UTC':datetime.now(timezone.utc).isoformat(),'status':t.STATUS,'goal_achieved':False,
      'change_from_validation_only_rounds':'Explicitly allow already-examined realdevelopment PE/official budgets to select one prelisted GLOBAL waveform integration configuration. Not validation-only coefficient selection.',
      'same_parameters_all_runs_and_seeds':True,'neural_training':'No newtraining;only frozen10neuraldensity and12simulationisotoniccalibration.',
      'formula':'Zwf=anchor+gamma*joint_mass_intrinsic_tailpenalty+beta*bounded_jointcompatibility_LR;anchor retainedOMC orFRT beforeoldmass term.',
      'grid':{'kind':['CONDITIONAL-CHI','CONDITIONAL-ETA-CHI'],'statistic':['BC','PRIOR'],'integration':['ADD','REPLACE'],'gamma':GRID,'beta':GRID},
      'grid_size':648,'unchanged_OMC_control':True,
      'ranking_inputs':'waveform-derived predictions only;no eventnames,PE values,officiallabels,FPP orHanabiresults areinputfeatures or candidate-specific score adjustments.',
      'validation_filter':'All6run/seed simulation waveform+fusion guardrails R10>=base-.02;AP>=base-.005;F50/F90<=1.1base.',
      'reused_test':'Old injectiontests already repeatedly inspected,explicitly DEVELOPMENT guardrails,not independent confirmation.',
      'real_goal':'Bothruns Top10/20 Mc count+median,Dmax,official1pct,Hanabi counts nondecreasing,catastrophes nonincreasing. Eachrun strictPE gain andstrictfrontendgain.',
      'selection':'Jointtarget plus all12reusedsimulationguards;then maxminimumfrontendcountgain acrossruns;then summedMcpassgain;then summedmedianBCgain;then totalfrontendgain;then smallestgamma2+beta2;lexicographic.',
      'frozen':['allencoders','densitycalibrators','time_score','sky_raw_log_bf','outer C-fixed weights','scope','history','paper'],
      'changed_channels':['waveform'],'outer_weights_changed':False,
      'fresh_confirmation':'Afteronewinnerfixed,NEW independent source+noise confirmation mandatory. Realoutcomes remainadaptive development eveniffreshinjectionpasses;no independent realvalidation claim.',
      'no_detection_claim':'Official/Hanabi tableoverlapisnotlensingtruth;noHanabi run;no proper lensBayesfactor.',
      'full_negative_record':True})
    rows=t.protected()
    for p in (UPSTREAM/'calibration').rglob('*.json'):rows.append({'path':str(p),'sha256':dev.sha(p),'bytes':p.stat().st_size})
    dev.csv_write(root/'manifest/INPUT_SHA256.csv',pd.DataFrame(rows));shutil.copy2(__file__,root/'scripts/joint_global_development.py')
    dev.json_write(root/'contracts/START_FREEZE.json',{'contract_sha256':dev.sha(root/'contracts/ANALYSIS_CONTRACT.json'),'code_sha256':dev.sha(Path(__file__))})


def read_values(dep,slot,seed,split,kind,spec):
    path=UPSTREAM/f'cache/{kind}/{dep}/{slot}_{seed}_{split}.npz'
    if not path.exists():raise RuntimeError('Read-only upstream features missing:'+str(path))
    return joint.values(UPSTREAM,dep,slot,seed,split,kind,spec)


def consensus(frames,scores,external,dep,method='fusion'):
    ranked=[]
    for seed,frame in frames.items():
        x=frame.copy();x['waveform_score']=scores[seed]
        w=cf.frozen_weights(dep,seed) if method=='fusion' else np.asarray([1.,0.,0.])
        ranked.append(dev.BASE.rank_real(x,cf.weights_dict(w),method,seed))
    result=dev.BASE.consensus_real(ranked,method)
    return result.merge(external[[c for c in external if c not in result or c=='pair_key']],on='pair_key',validate='one_to_one')


def target(rows,baseline):
    checks=[];details=[]
    for dep in t.DEPS:
        keep=True;pe_gain=0.;mc_gain=0;median_gain=0.;official_gain=0
        for b in (10,20):
            new=next(r for r in rows if r['deployment']==dep and r['budget']==b)
            old=baseline[dep,b]
            for col in ('BC_mc_ge_0p5','median_BC_mc','Dmax_le_3','official_frontend','official_hanabi'):
                keep &= new[col]>=old[col]-1e-12
            keep &= new['catastrophic_mc']<=old['catastrophic_mc']
            mc_gain+=new['BC_mc_ge_0p5']-old['BC_mc_ge_0p5']
            median_gain+=new['median_BC_mc']-old['median_BC_mc']
            pe_gain+=int(new['BC_mc_ge_0p5']>old['BC_mc_ge_0p5'] or new['median_BC_mc']>old['median_BC_mc']+1e-10)
            official_gain+=new['official_frontend']-old['official_frontend']
        checks.append(keep and pe_gain>0 and official_gain>0)
        details.append({'deployment':dep,'no_loss':bool(keep),'PE_gain':bool(pe_gain>0),'frontend_gain':official_gain,
            'Mc_count_gain':mc_gain,'Mc_median_gain':median_gain,'run_target':bool(checks[-1])})
    return bool(all(checks)),details


def task(spec):
    kind,statistic,integration=spec
    frames={};penalties={};increments={};bm={};external={};baseline={};calibrations={}
    for dep in t.DEPS:
        external[dep]=pd.read_parquet(t.EXTERNAL/f'{dep}_external_reference.parquet')
        for slot,seed in zip(t.MODEL_SLOTS,t.SEEDS):
            cal=json.loads((UPSTREAM/f'calibration/{kind}/{dep}/{slot}_{statistic}.json').read_text());calibrations[dep,seed]=cal
            for split in ('validation','test','real'):
                f,pen,inc,_=read_values(dep,slot,seed,split,kind,cal);key=dep,seed,split
                frames[key],penalties[key],increments[key]=f,pen,inc
                if split!='real':
                    z=f.waveform_score.to_numpy(float);w=cf.frozen_weights(dep,seed)
                    bm[key]=[cf.fast_metrics(f,z),cf.fast_metrics(f,cf.channels(f,z)@w)]
        fs={seed:frames[dep,seed,'real'] for seed in t.SEEDS}
        a=consensus(fs,{seed:f.waveform_score.to_numpy(float) for seed,f in fs.items()},external[dep],dep)
        for b in (10,20):baseline[dep,b]=dev.budget_row(a,'OMC',dep,'fusion',b)
    rows=[];budgets=[];metrics=[];candidates=[]
    for gamma in GRID:
        for beta in GRID:
            code=f'{kind.replace("CONDITIONAL-","")}-{statistic}-{integration}-G{gamma:g}-B{beta:g}'
            valid=True;testpass=True;scores={};configmetrics=[]
            for split in ('validation','test'):
                if split=='test' and not valid:break
                for dep in t.DEPS:
                    for seed in t.SEEDS:
                        key=dep,seed,split;f=frames[key]
                        anchor=f.waveform_score.to_numpy(float) if integration=='ADD' else f.FRT_baseline_waveform_score.to_numpy(float)
                        z=anchor+gamma*penalties[key]+beta*increments[key]
                        w=cf.frozen_weights(dep,seed)
                        for j,(mode,score) in enumerate((('waveform',z),('fusion',cf.channels(f,z)@w))):
                            m=cf.fast_metrics(f,score);okay=cf.guard(m,bm[key][j])
                            configmetrics.append({'configuration':code,'deployment':dep,'seed':seed,'split':split,'mode':mode,'guard_pass':okay,**m})
                            if split=='validation':valid &= okay
                            else:testpass &= okay
            row={'configuration':code,'kind':kind,'statistic':statistic,'integration':integration,'gamma':gamma,'beta':beta,
                'validation_guard':bool(valid),'reused_test_guard':bool(testpass) if valid else None,'both_real_target':False}
            metrics+=configmetrics
            if valid:
                br=[]
                for dep in t.DEPS:
                    fs={seed:frames[dep,seed,'real'] for seed in t.SEEDS};zs={}
                    for seed,f in fs.items():
                        key=dep,seed,'real';anchor=f.waveform_score.to_numpy(float) if integration=='ADD' else f.FRT_baseline_waveform_score.to_numpy(float)
                        zs[seed]=anchor+gamma*penalties[key]+beta*increments[key]
                    result=consensus(fs,zs,external[dep],dep)
                    br += [dev.budget_row(result,code,dep,'fusion',b) for b in (10,20,50,100)]
                passed,details=target(br,baseline);row['both_real_target']=passed
                for d in details:
                    for key,value in d.items():
                        if key!='deployment':row[d['deployment']+'_'+key]=value
                budgets+=br
                if passed and testpass:
                    objective=(-min(d['frontend_gain'] for d in details),-sum(d['Mc_count_gain'] for d in details),
                        -sum(d['Mc_median_gain'] for d in details),-sum(d['frontend_gain'] for d in details),gamma*gamma+beta*beta,code)
                    candidates.append({'objective':objective,**row,'calibrations':{f'{dep}_{seed}':c for (dep,seed),c in calibrations.items()}})
            rows.append(row)
    return rows,budgets,metrics,candidates


def select(root):
    if (root/'contracts/SEARCH_COMPLETE.json').exists():raise RuntimeError('Alreadycomplete')
    specs=[(kind,stat,integration) for kind in ('CONDITIONAL-CHI','CONDITIONAL-ETA-CHI') for stat in ('BC','PRIOR') for integration in ('ADD','REPLACE')]
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
                configs.append({'method':'GLOBAL-'+chosen['configuration'],'deployment':dep,'seed':seed,'slot':slot,
                    'kind':chosen['kind'],'statistic':chosen['statistic'],'integration':chosen['integration'],
                    'gamma':chosen['gamma'],'beta':chosen['beta'],'weights':cf.frozen_weights(dep,seed).tolist(),
                    'calibration':chosen['calibrations'][f'{dep}_{seed}'],'real_used_for_development_selection':True})
    dev.json_write(root/'calibration/SELECTED.json',configs)
    dev.csv_write(root/'tables/SELECTED_COEFFICIENTS.csv',pd.DataFrame([{k:v for k,v in c.items() if k not in ('weights','calibration')} for c in configs]))
    dev.json_write(root/'contracts/INTEGRATION_FROZEN.json',{'file':'calibration/SELECTED.json','sha256':dev.sha(root/'calibration/SELECTED.json'),
        'UTC':datetime.now(timezone.utc).isoformat(),'real_or_reused_test_selection':True,'not_blind_confirmation':True})


def scored(root,c,split):
    if c['method']=='OMC':
        f=cf.read(c['deployment'],c['seed'],split);return f,f.waveform_score.to_numpy(float),{}
    f,pen,inc,a=read_values(c['deployment'],c['slot'],c['seed'],split,c['kind'],c['calibration'])
    anchor=f.waveform_score.to_numpy(float) if c['integration']=='ADD' else f.FRT_baseline_waveform_score.to_numpy(float)
    return f,anchor+c['gamma']*pen+c['beta']*inc,{**a,'penalty':pen,'increment':inc}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--stage',required=True,choices=['initialize','select','evaluate','real','assess']);a=p.parse_args();ev.scored=scored
    if a.stage in ('initialize','select'):globals()[a.stage](a.root)
    else:getattr(ev,a.stage)(a.root)
