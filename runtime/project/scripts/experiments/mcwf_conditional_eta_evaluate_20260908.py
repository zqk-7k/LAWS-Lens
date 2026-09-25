#!/usr/bin/env python3
"""Freeze conditional-eta calibration and evaluate without real-label fitting."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='2'
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

P=Path('/root/autodl-tmp/gw-catalog')
sys.path.insert(0,str(P/'scripts/experiments'))
import mcwf_conditional_eta_20260908 as model
import mcwf_temporal_response_evaluate_20260908 as ev
t,dev,cf=model.t,model.dev,model.cf
BETAS=(0.,.0625,.125,.25,.5,1.,2.,4.)


def calibration(root,dep,slot):
    meta=pd.read_parquet(t.PREVIOUS/f'expanded_data/{dep}/validation/event_metadata.parquet')
    groups=meta.source_uid.to_numpy(str)
    banks=sorted(meta.noise_bank_index.unique(),key=lambda x:hashlib.sha256(f'202609850:{dep}:{x}'.encode()).hexdigest())
    fold=meta.noise_bank_index.map({x:i%2 for i,x in enumerate(banks)}).to_numpy(int)
    mixed={g for g in np.unique(groups) if len(np.unique(fold[groups==g]))>1}
    fold[np.isin(groups,list(mixed))]=-1
    a=dict(model.prediction(root,dep,slot,0,'development'))
    cohorts={}
    for side in (0,1):
        ids=np.flatnonzero(fold==side);i,j=np.triu_indices(len(ids),1);i,j=ids[i],ids[j]
        y=groups[i]==groups[j]
        if y.sum()<30:raise RuntimeError('Insufficient independent conditional-eta fit sources')
        cohorts[side]=(y,model.pair_features(root,dep,slot,a,i,j))
    y,x=cohorts[0];weights=np.where(y,.5/y.sum(),.5/(~y).sum())
    values=x['conditional_eta_overlap']
    iso=IsotonicRegression(increasing=True,out_of_bounds='clip').fit(values,y,sample_weight=weights)
    p=iso.y_thresholds_.clip(1/(y.sum()+2),1-1/(y.sum()+2));lr=np.log(p)-np.log1p(-p)
    spec={'knots':iso.X_thresholds_.tolist(),'loglr':lr.tolist(),'zero_reference':float(np.interp(0.,iso.X_thresholds_,lr)),
        'minimum':float(values.min()),'maximum':float(values.max()),'mass_reference':np.sort(-np.log(x['mass_BC'][y])).tolist(),
        'independent_positive_sources':int(y.sum()),'fit_noise_banks':[int(v) for v in np.unique(meta.noise_bank_index[fold==0])],
        'audit_noise_banks':[int(v) for v in np.unique(meta.noise_bank_index[fold==1])],'dropped_cross_noise_sources':len(mixed),
        'eta_checkpoint_sha256':dev.sha(root/f'models/{model.KIND}/{dep}/seed_{slot}/selected.pt')}
    audit=[]
    for side,(yy,xx) in cohorts.items():
        delta=xx['conditional_eta_overlap'];inc,_=increment(xx,spec)
        audit.append({'deployment':dep,'slot':slot,'cohort':'fit' if side==0 else 'noise_disjoint_audit',
            'true_sources':int(yy.sum()),'dependent_null_pairs':int((~yy).sum()),
            'positive_median_delta':float(np.median(delta[yy])),'null_median_delta':float(np.median(delta[~yy])),
            'true_eta_negative_fraction':float((inc[yy]<0).mean()),'null_eta_positive_fraction':float((inc[~yy]>0).mean()),
            'model_development_reuse':True})
    dev.json_write(root/f'calibration/{dep}/{slot}.json',spec)
    return spec,audit


def increment(x,spec):
    delta=x['conditional_eta_overlap']
    outside=(delta<spec['minimum'])|(delta>spec['maximum'])|x['ood']
    mass_tail=ev.tail.tail_probability(-np.log(x['mass_BC']),spec['mass_reference'])
    raw=np.interp(delta,spec['knots'],spec['loglr'])-spec['zero_reference']
    raw=np.where(outside|(mass_tail<.05),np.minimum(raw,0.),raw)
    raw=np.where(x['ood'],0.,raw).clip(-4,4)
    return raw,{'conditional_eta_calibration_ood':outside,'mass_compatibility_tail':mass_tail}


def values(root,dep,slot,es,split,spec):
    f=cf.read(dep,es,split)
    a=dict(model.prediction(root,dep,slot,es,split))
    x=model.pair_features(root,dep,slot,a,f.idx_i.to_numpy(int),f.idx_j.to_numpy(int))
    if not all(np.isfinite(v).all() for v in x.values()):raise RuntimeError('Invalid conditional-eta strict-scope features')
    inc,audit=increment(x,spec)
    return f,inc,{**x,**audit,'increment':inc}


def select(root):
    if (root/'contracts/INTEGRATION_FROZEN.json').exists():raise RuntimeError('Already frozen')
    shutil.copy2(__file__,root/'scripts/conditional_eta_evaluate.py')
    configs=[];grids=[];audits=[]
    for dep in t.DEPS:
        for slot,es in zip(t.MODEL_SLOTS,t.SEEDS):
            spec,audit=calibration(root,dep,slot);audits+=audit
            f,inc,_=values(root,dep,slot,es,'validation',spec)
            old=f.waveform_score.to_numpy(float);w=cf.frozen_weights(dep,es)
            bm,bwm=cf.fast_metrics(f,cf.channels(f,old)@w),cf.fast_metrics(f,old)
            rows=[]
            for beta in BETAS:
                z=old+beta*inc;mm,wm=cf.fast_metrics(f,cf.channels(f,z)@w),cf.fast_metrics(f,z)
                row={'beta':beta,'pass':cf.guard(mm,bm) and cf.guard(wm,bwm),**mm}
                rows.append(row);grids.append({'deployment':dep,'seed':es,**row,**{'waveform_'+k:v for k,v in wm.items()}})
            for policy in ('CANDIDATE','RETRIEVAL'):
                def key(r):
                    primary=(r['false_at_recall_0p5'],r['false_at_recall_0p9'],-r['average_precision'],-r['macro_r_at_10'],-r['macro_r_at_1']) if policy=='CANDIDATE' else (-r['macro_r_at_10'],-r['macro_r_at_1'],-r['average_precision'],r['false_at_recall_0p5'],r['false_at_recall_0p9'])
                    return (*primary,r['beta'])
                win=min((r for r in rows if r['pass']),key=key)
                configs.append({'method':model.KIND+'-'+policy,'deployment':dep,'seed':es,'slot':slot,
                    'beta':win['beta'],'gamma':0.,'weights':w.tolist(),'calibration':spec})
    dev.csv_write(root/'tables/VALIDATION_GRID.csv',pd.DataFrame(grids));dev.csv_write(root/'tables/CALIBRATION_AUDIT.csv',pd.DataFrame(audits))
    dev.csv_write(root/'tables/SELECTED_COEFFICIENTS.csv',pd.DataFrame([{k:v for k,v in c.items() if k not in ('weights','calibration')} for c in configs]))
    dev.json_write(root/'calibration/SELECTED.json',configs)
    dev.json_write(root/'contracts/INTEGRATION_FROZEN.json',{'UTC':datetime.now(timezone.utc).isoformat(),'file':'calibration/SELECTED.json',
        'sha256':dev.sha(root/'calibration/SELECTED.json'),'real_or_test_selection':False,'code_sha256':dev.sha(Path(__file__))})


def scored(root,c,split):
    if c['method']=='OMC':
        f=cf.read(c['deployment'],c['seed'],split);return f,f.waveform_score.to_numpy(float),{}
    f,inc,x=values(root,c['deployment'],c['slot'],c['seed'],split,c['calibration'])
    return f,f.waveform_score.to_numpy(float)+c['beta']*inc,x


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--stage',choices=['select','evaluate','real','assess'],required=True);a=p.parse_args()
    ev.scored=scored
    if a.stage=='select':select(a.root)
    else:getattr(ev,a.stage)(a.root)
