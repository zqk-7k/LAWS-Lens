#!/usr/bin/env python3
"""Coarse intrinsic-profile agreement as a calibrated waveform diagnostic."""
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
from scipy.special import logsumexp
from sklearn.isotonic import IsotonicRegression
import mcwf_pe_frontend_extension_v2_20260907 as e

dev,body,ev=e.dev,e.body,e.ev
TEMPERATURES=(.5,1.,2.,4.,8.,16.,32.,64.,128.)
BETAS=(0.,.0625,.125,.25,.5,1.,2.,4.)
KINDS=('full','conditional_q_spin')


def distribution(features,temperature):
    power=np.expm1(np.asarray(features[:,2],float)).reshape(len(features),-1)
    logits=(power-power.max(1,keepdims=True))/(2*temperature)
    lp=logits-logsumexp(logits,axis=1,keepdims=True)
    return lp,logsumexp(lp.reshape(len(features),64,9),axis=2)


def raw_pair(features,i,j,temperature):
    lp,lm=distribution(features,temperature);out=[]
    for start in range(0,len(i),2048):
        a,b=i[start:start+2048],j[start:start+2048]
        full=np.log(576)+logsumexp(lp[a]+lp[b],axis=1)
        mass=np.log(64)+logsumexp(lm[a]+lm[b],axis=1)
        out.append(np.stack([full,full-mass],1))
    return np.concatenate(out)


def fit(root,dep):
    out=root/f'physical_profile/{dep}';out.mkdir(parents=True,exist_ok=True)
    path=out/'FIT.json'
    if path.exists():return json.loads(path.read_text())
    features=np.load(e.TRAINED/f'cache/finelag/{dep}/validation_features.npy')
    target=np.load(body.PREVIOUS/f'cache/masstf/{dep}/validation_targets.npy')
    rows=[]
    for t in TEMPERATURES:
        _,lm=distribution(features,t);rows.append({'temperature':t,'Mc_CE':float(-(target*lm).sum(1).mean())})
    temperature=min(rows,key=lambda r:(r['Mc_CE'],r['temperature']))['temperature']
    meta=pd.read_parquet(e.TRAINED/f'cache/{dep}/validation_metadata.parquet');i,j=np.triu_indices(len(meta),1)
    groups=meta.waveform_parent_uid.to_numpy();y=groups[i]==groups[j]
    x=raw_pair(features,i,j,temperature);specs={}
    for column,kind in enumerate(KINDS):
        raw=x[:,column];weights=np.where(y,.5/y.sum(),.5/(~y).sum())
        iso=IsotonicRegression(increasing=True,out_of_bounds='clip').fit(raw,y.astype(float),sample_weight=weights)
        floor=1/(y.sum()+2);p=np.clip(iso.y_thresholds_,floor,1-floor)
        specs[kind]={'temperature':temperature,'kind':kind,'column':column,'knots':iso.X_thresholds_.tolist(),
                     'loglr':(np.log(p)-np.log1p(-p)).tolist(),'fit_min':float(raw.min()),'fit_max':float(raw.max()),'floor':float(floor)}
    dev.csv_write(out/'TEMPERATURE_GRID.csv',pd.DataFrame(rows));dev.json_write(path,specs)
    print(json.dumps({'physical_profile_temperature':dep,'temperature':temperature,'best_CE':min(r['Mc_CE'] for r in rows)}),flush=True)
    return specs


def get(dep,es,split,temperature):
    f=pd.read_parquet(e.BASE/f'evaluation/{dep}/seed_{es}/{split}_pairs.parquet')
    i,j=f.idx_i.to_numpy(int),f.idx_j.to_numpy(int)
    if split=='real':
        feature=np.load(e.TRAINED/f'cache/deployment_event_psd/{dep}/real_features.npy');_,events=dev.real_inputs(dep)
        valid=events.strict_h1l1_preprocessing_pass.to_numpy(bool);remap=np.full(len(events),-1,int);remap[valid]=np.arange(valid.sum());i,j=remap[i],remap[j]
        if (i<0).any() or (j<0).any():raise RuntimeError('Invalid detector mapping')
    else:feature=np.load(e.TRAINED/f'cache/deployment_event_psd/{dep}/{es}_{split}_features.npy')
    return f,raw_pair(feature,i,j,temperature)


def score(f,x,spec):
    raw=x[:,spec['column']];v=np.interp(raw,spec['knots'],spec['loglr']);ood=(raw<spec['fit_min'])|(raw>spec['fit_max'])
    v=np.clip(np.where(ood,np.minimum(v,0),v),-4,4)
    return f.waveform_score.to_numpy(float)+spec['beta']*v,v,ood


def run(root):
    trial=root/'trials/PHYSICAL-INTRINSIC-PROFILE'
    if trial.exists():raise RuntimeError('Independent trial required')
    for name in ('contracts','calibration','tables','evaluation','results'):(trial/name).mkdir(parents=True)
    shutil.copy2(__file__,trial/'contracts'/Path(__file__).name)
    dev.json_write(trial/'contracts/RECIPE.json',{'created_utc':datetime.now(timezone.utc).isoformat(),
        'same_both_runs':True,'features':'existing event-PSD quadrature matched-template powers from2s waveforms,576templates64Mc*3q*3alignedspin',
        'event_profile':'normalize exp[(network_profile_power-max_power)/(2T)] over discrete uniform576grid',
        'temperature':'encoder-development validation soft-label Mc cross entropy only, shared across3 modelseeds/run',
        'temperature_grid':TEMPERATURES,'raw_features':['log576+logsum(P_i*P_j)','full_overlap_minus_marginal_Mc_overlap'],
        'conditional_feature':'q/spin agreement in coarse grid conditional on mass overlap; attempts to retain information discarded by Mc-only prediction',
        'calibration':'240 independent-source encoder validation pairs, isotonic equal-classweights; finitefloor; cap+-4 and positiveOOD0',
        'beta_grid':BETAS,'selection':'same BAYESTAR validation-only objective and FRT-relative guards',
        'limits':['coarse profiles are NOT fullPE posteriors','maximized time and phase rather than marginalized','isotropic discrete prior is an engineering reference','temperature calibration ofMc does NOT establish q/spin coverage','no proper Bayes-factor claim'],
        'frozen':['time','sky','outerweights','all encoders','scope','historical results'],
        'references':['https://pycbc.org/pycbc/latest/html/pycbc.filter.html','https://arxiv.org/abs/1807.07062']})
    selected={};statuses=[];cache={}
    for dep in e.DEPS:
        cal=fit(root,dep)
        for es in dev.SEEDS:
            f,x=get(dep,es,'validation',cal[KINDS[0]]['temperature']);cache[dep,es]=(f,x);bm=ev.metrics(f,f.waveform_score.to_numpy(float),dep,es)
            rows=[];choices=[]
            for kind in KINDS:
                for beta in BETAS:
                    cfg={**cal[kind],'beta':beta};z,_,ood=score(f,x,cfg);m=ev.metrics(f,z,dep,es);ok=ev.guard(m,bm)
                    rows.append({'kind':kind,'beta':beta,'pass':ok,'ood':float(ood.mean()),**{a+'_'+k:v for a,b in m.items() for k,v in b.items()}})
                    if ok:choices.append((e.old_selection.objective(m,beta,KINDS.index(kind)),cfg))
            out=trial/f'calibration/{dep}/seed_{es}';out.mkdir(parents=True);dev.csv_write(out/'validation_grid.csv',pd.DataFrame(rows))
            cfg=min(choices,key=lambda c:c[0])[1];selected[dep,es]=cfg;dev.json_write(out/'SELECTED_CONFIG.json',cfg)
            statuses.append({'deployment':dep,'seed':es,'kind':cfg['kind'],'beta':cfg['beta']})
    dev.json_write(trial/'contracts/SELECTED_FREEZE.json',{'configs':statuses,'real_used_to_select':False});print(json.dumps({'profile_selected':statuses}),flush=True)
    rows=[];guards=[]
    for dep in e.DEPS:
        real={}
        for es in dev.SEEDS:
            cfg=selected[dep,es]
            for split in ('validation','test','real'):
                f,x=cache[dep,es] if split=='validation' else get(dep,es,split,cfg['temperature']);z,val,ood=score(f,x,cfg)
                n=f.copy();n['FRT_baseline_waveform_score']=f.waveform_score;n['waveform_score']=z;n['intrinsic_profile_increment']=val;n['intrinsic_profile_ood']=ood
                for k,kind in enumerate(KINDS):n['intrinsic_profile_'+kind]=x[:,k]
                out=trial/f'evaluation/{dep}/seed_{es}';out.mkdir(parents=True,exist_ok=True);n.to_parquet(out/f'{split}_pairs.parquet',index=False)
                if split=='real':real[es]=n
                else:
                    bm=ev.metrics(f,f.waveform_score.to_numpy(float),dep,es);m=ev.metrics(f,z,dep,es)
                    guards.append({'deployment':dep,'seed':es,'split':split,'pass':ev.guard(m,bm)})
                    for method in bm:
                        for config,metrics in [('FRT_BASELINE',bm[method]),('CANDIDATE',m[method])]:rows.append({'deployment':dep,'seed':es,'split':split,'method':method,'config':config,**metrics})
        dev.save_evaluation(trial,'CANDIDATE',dep,real,pd.read_parquet(root/f'audit/{dep}_frozen_external_reference.parquet'))
    dev.csv_write(trial/'tables/RETRIEVAL_PER_SEED.csv',pd.DataFrame(rows));dev.csv_write(trial/'tables/REUSED_GUARDRAILS.csv',pd.DataFrame(guards));e.assess(root,trial)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();run(a.root)
