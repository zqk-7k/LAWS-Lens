#!/usr/bin/env python3
"""Validation-source stability gate for optional fusion refitting."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import numpy as np
import pandas as pd

P=Path('/root/autodl-tmp/gw-catalog');sys.path.insert(0,str(P/'scripts/experiments'))
import mcwf_temporal_response_evaluate_20260908 as ev
import mcwf_joint_predictive_ensemble_v2_20260908 as evaluator
t,dev,cf=ev.t,ev.dev,ev.cf
J=P/'results/mcwf_joint_intrinsic_evidence_exploratory_20260908T155830Z'
R=P/'results/mcwf_joint_fusion_retune_exploratory_20260908T160928Z'
E=P/'results/mcwf_joint_predictive_ensemble_exploratory_20260908T161608Z'
SPECS=[]
for head in ('CHI','ETA-CHI'):
    for policy in ('CANDIDATE','RETRIEVAL'):
        name=f'JOINT-{head}-BC-ADD-{policy}'
        SPECS.append((name,J,name,R,'REFIT-'+name+'-POSITIVE-CANDIDATE'))
        name=f'ENSEMBLE-{head}'
        SPECS.append((name+'-'+policy,E,name+'-FIXED-'+policy,E,name+'-REFIT-'+policy+'-CANDIDATE'))


def initialize(root):
    if root.exists():raise RuntimeError('Fresh output directory required')
    for name in ('contracts','tables','calibration','evaluation','reports','figures','scripts','logs','manifest'):(root/name).mkdir(parents=True)
    dev.json_write(root/'contracts/ANALYSIS_CONTRACT.json',{
        'id':'MCWF-VALIDATION-FUSION-STABILITY-18','UTC':datetime.now(timezone.utc).isoformat(),'status':t.STATUS,
        'goal_achieved':False,'same_both_runs':True,
        'rationale':'Refitting a few weights on a small validation catalog can overfit discrete false-pair counts. Keep fixed fusion unless its paired improvement exceeds validation source-sampling uncertainty.',
        'scope':'Same frozen waveform percomparison;only decide between original C-fixed weights and previously simulation-selected positive-simplex weights. No new numeric weight search.',
        'bootstrap':{'replicates':2000,'unit':'family-stratified source systems;bothimages together;shared noise not separately resampled',
            'pair_weight':'n_g*n_h for unrelated source pairs;n_g for truecompanions','same_draws_for_all_models':True},
        'rules':{'ONE-SE':'Use refit iff point deltaF50 < -bootstrapSD(deltaF50);SD is samplinguncertainty,NOT dividedbysqrt2000.',
            'CI95':'Use refit iff95%percentileupper(deltaF50)<0;reportF90/AP/R10diagnostics. Both choices already passed originalvalidationguards.'},
        'interpretation':'These are development stability heuristics,not valid post-selection confirmatory hypothesis tests. Bootstrap doesnotundo choosingbestweights onthe samevalidationcatalog. Freshconfirmationrequired.',
        'fallback':'Otherwise retain original fixed fusion;apply identical rule independently perseed/run,no manualchoosing.',
        'frozen':['all neuralmodels','waveformperpair','raw Z_time','raw Z_sky','scope','history','paper'],
        'changed_channels':[],'outer_weights_changed':True,
        'external':'No PE/official labels or eventIDs in selection. Method familymotivatedbyadaptive development;notblindconfirmation.',
        'upstream_models':[{'code':c,'fixed_root':str(a),'fixed_method':b,'refit_root':str(d),'refit_method':e} for c,a,b,d,e in SPECS]})
    rows=t.protected()
    for _,a,b,d,e in SPECS:
        for r,name in ((a,b),(d,e)):
            for dep in t.DEPS:
                for seed in t.SEEDS:
                    p=r/f'evaluation/{name}/{dep}/seed_{seed}/validation_pairs.parquet'
                    rows.append({'path':str(p),'sha256':dev.sha(p),'bytes':p.stat().st_size})
    dev.csv_write(root/'manifest/INPUT_SHA256.csv',pd.DataFrame(rows).drop_duplicates('path'))
    shutil.copy2(__file__,root/'scripts/fusion_stability.py')
    dev.json_write(root/'contracts/START_FREEZE.json',{'contract_sha256':dev.sha(root/'contracts/ANALYSIS_CONTRACT.json'),'code_sha256':dev.sha(Path(__file__))})


def ordered_evaluator(scores,truth):
    order=np.argsort(-scores,kind='stable');y=truth[order]
    ends=np.r_[np.flatnonzero(np.diff(scores[order])!=0),len(order)-1]
    def calculate(weights):
        ww=weights[order];tp=np.cumsum(ww*y);fp=np.cumsum(ww*(~y))
        tt,ff=tp[ends],fp[ends];delta=np.diff(np.r_[0.,tt])
        ap=np.sum(delta*np.divide(tt,tt+ff,out=np.zeros_like(tt),where=tt+ff>0))/tt[-1]
        f50=fp[min(np.searchsorted(tp,.5*tp[-1]),len(tp)-1)]
        f90=fp[min(np.searchsorted(tp,.9*tp[-1]),len(tp)-1)]
        return np.asarray([f50,f90,ap])
    return calculate


def task(args):
    spec,dep,seed=args;code,fr,fm,rr,rm=spec
    fixed=pd.read_parquet(fr/f'evaluation/{fm}/{dep}/seed_{seed}/validation_pairs.parquet')
    refit=pd.read_parquet(rr/f'evaluation/{rm}/{dep}/seed_{seed}/validation_pairs.parquet')
    refit=refit.set_index(['idx_i','idx_j']).loc[fixed.set_index(['idx_i','idx_j']).index].reset_index()
    for col in ('waveform_score','time_score','sky_raw_log_bf','is_true_pair'):
        if not np.array_equal(fixed[col],refit[col]):raise RuntimeError('Comparison changed frozenchannel '+col)
    truth=fixed.is_true_pair.to_numpy(bool)
    score0,score1=fixed.final_score.to_numpy(float),refit.final_score.to_numpy(float)
    fn0,fn1=ordered_evaluator(score0,truth),ordered_evaluator(score1,truth)
    point=fn1(np.ones(len(truth)))-fn0(np.ones(len(truth)))
    for score,fn in ((score0,fn0),(score1,fn1)):
        reference=cf.fast_metrics(fixed,score);a=fn(np.ones(len(truth)))
        if not np.allclose(a,[reference[k] for k in ('false_at_recall_0p5','false_at_recall_0p9','average_precision')],atol=1e-10):
            raise RuntimeError('Bootstrap metricdoesnotreplay')
    plan=dev.BASE.retained_event_plan(dep,seed,'validation');ids=plan.set_index('idx').system_id
    names=sorted(ids.unique());mapping={g:i for i,g in enumerate(names)}
    a=ids.loc[fixed.idx_i].map(mapping).to_numpy(int);b=ids.loc[fixed.idx_j].map(mapping).to_numpy(int)
    strata={g:'background' for g in range(len(names))}
    for g,fam in zip(a[truth],fixed.loc[truth,'true_pair_family']):strata[g]=str(fam)
    groups=[np.asarray([g for g in strata if strata[g]==f]) for f in sorted(set(strata.values()))]
    rng=np.random.default_rng(int(hashlib.sha256(f'202609972:{dep}:{seed}'.encode()).hexdigest()[:12],16))
    samples=[]
    for _ in range(2000):
        counts=np.zeros(len(names),int)
        for group in groups:counts+=np.bincount(rng.choice(group,len(group),replace=True),minlength=len(names))
        weights=counts[a]*counts[b];weights[truth]=counts[a[truth]]
        samples.append(fn1(weights)-fn0(weights))
    samples=np.asarray(samples);sd=samples.std(0,ddof=1);lo,hi=np.quantile(samples,[.025,.975],axis=0)
    result={'code':code,'deployment':dep,'seed':seed,'sources':len(names),'bootstrap_replicates':2000,
        'point_delta_F50':point[0],'SD_delta_F50':sd[0],'CI_low_delta_F50':lo[0],'CI_high_delta_F50':hi[0],
        'point_delta_F90':point[1],'CI_low_delta_F90':lo[1],'CI_high_delta_F90':hi[1],
        'point_delta_AP':point[2],'CI_low_delta_AP':lo[2],'CI_high_delta_AP':hi[2],
        'bootstrap_P_F50_improves':float((samples[:,0]<0).mean()),'ONE-SE':bool(point[0]<-sd[0]),'CI95':bool(hi[0]<0)}
    configs=[]
    for rule in ('ONE-SE','CI95'):
        up,name=(rr,rm) if result[rule] else (fr,fm)
        orig=next(c for c in ev.load_selections(up) if c['method']==name and c['deployment']==dep and c['seed']==seed)
        configs.append({'method':code+'-STABLE-'+rule,'deployment':dep,'seed':seed,'slot':orig.get('slot',t.MODEL_SLOTS[t.SEEDS.index(seed)]),
            'weights':orig['weights'],'gamma':0.,'beta':0.,'origin_root':str(up),'origin_method':name,
            'stability_rule':rule,'refit_selected':result[rule],'original_configuration':orig})
    return result,configs


def select(root):
    if (root/'contracts/INTEGRATION_FROZEN.json').exists():raise RuntimeError('Alreadyfrozen')
    rows,configs=[],[];tasks=[(s,dep,seed) for s in SPECS for dep in t.DEPS for seed in t.SEEDS]
    with ProcessPoolExecutor(max_workers=8) as pool:
        for row,cs in pool.map(task,tasks):
            rows.append(row);configs+=cs;print(json.dumps(row),flush=True)
    dev.csv_write(root/'tables/VALIDATION_FUSION_STABILITY.csv',pd.DataFrame(rows))
    dev.csv_write(root/'tables/SELECTED_COEFFICIENTS.csv',pd.DataFrame([{**{k:v for k,v in c.items() if k not in ('original_configuration','weights')},
        **dict(zip(('lambda_waveform','lambda_time','lambda_sky'),c['weights']))} for c in configs]))
    dev.json_write(root/'calibration/SELECTED.json',configs)
    dev.json_write(root/'contracts/INTEGRATION_FROZEN.json',{'file':'calibration/SELECTED.json','sha256':dev.sha(root/'calibration/SELECTED.json'),
        'UTC':datetime.now(timezone.utc).isoformat(),'real_or_test_selection':False})


def scored(root,c,split):
    f=cf.read(c['deployment'],c['seed'],split)
    if c['method']=='OMC':return f,f.waveform_score.to_numpy(float),{}
    filename='real_fusion_pairs.parquet' if split=='real' else f'{split}_pairs.parquet'
    p=Path(c['origin_root'])/f'evaluation/{c["origin_method"]}/{c["deployment"]}/seed_{c["seed"]}'/filename
    a=pd.read_parquet(p);key=['pair_key'] if split=='real' else ['idx_i','idx_j']
    a=a.set_index(key).loc[f.set_index(key).index]
    for col in ('time_score','sky_raw_log_bf'):
        if not np.array_equal(a[col],f[col]):raise RuntimeError('Frozen physicalscorechanged')
    return f,a.waveform_score.to_numpy(float),{}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--stage',required=True,
        choices=['initialize','select','evaluate','real','assess']);a=p.parse_args();ev.scored=scored;evaluator.scored=scored
    if a.stage in ('initialize','select'):globals()[a.stage](a.root)
    elif a.stage=='evaluate':evaluator.evaluate(a.root)
    else:getattr(ev,a.stage)(a.root)
