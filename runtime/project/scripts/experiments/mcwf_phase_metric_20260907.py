#!/usr/bin/env python3
"""Regularized source-discriminant metric on physical waveform-match profiles."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='2'
import argparse
from datetime import datetime,timezone
from pathlib import Path
import shutil
import json
import time
import numpy as np
import pandas as pd
from scipy.linalg import eigh
from sklearn.covariance import OAS
from sklearn.isotonic import IsotonicRegression
import mcwf_pe_frontend_extension_v2_20260907 as e

dev,body,ev=e.dev,e.body,e.ev
DIMS=(16,32,64,128)
BETAS=(0.,.0625,.125,.25,.5,1.,2.)


def profile(x):
    x=np.asarray(x,float).reshape(len(x),3,-1)
    x=x-x.mean(2,keepdims=True)
    return x.reshape(len(x),-1)


def fit(root,dep):
    out=root/f'phase_metric/{dep}';out.mkdir(parents=True,exist_ok=True)
    path=out/'metric.npz'
    if path.exists():return np.load(path)
    start=time.perf_counter()
    trainpath=e.TRAINED/f'cache/finelag/{dep}/train_features.npy'
    x=profile(np.load(trainpath));meta=pd.read_parquet(e.TRAINED/f'cache/{dep}/train_metadata.parquet')
    group,names=pd.factorize(meta.waveform_parent_uid,sort=True)
    means=np.stack([x[group==k].mean(0) for k in range(len(names))]);center=means.mean(0)
    sd=x.std(0).clip(.05);xx=(x-center)/sd;means=(means-center)/sd
    residual=xx-means[group]
    cov=OAS(assume_centered=True).fit(residual)
    sw=cov.covariance_;sb=means.T@means/len(means)
    vals,vecs=eigh(sb,sw,subset_by_index=[x.shape[1]-128,x.shape[1]-1],driver='gvx')
    vals,vecs=vals[::-1],vecs[:,::-1]
    gram=vecs.T@sw@vecs
    err=float(np.max(abs(gram-np.eye(len(vals)))))
    if err>1e-5 or not np.isfinite(vecs).all():raise RuntimeError('Discriminant metric numerical failure')
    np.savez_compressed(path,center=center,sd=sd,projection=vecs,eigenvalues=vals)
    dev.json_write(out/'FIT.json',{'training_sources':len(names),'training_events':len(x),'OAS_shrinkage':cov.shrinkage_,
        'within_metric_orthogonality_error':err,'seconds':time.perf_counter()-start,'training_input_sha256':dev.sha(trainpath),
        'model_sha256':dev.sha(path),'real_or_validation_used_to_fit_projection':False})
    print(json.dumps({'phase_metric_fitted':dep,'sources':len(names),'seconds':time.perf_counter()-start,'shrinkage':cov.shrinkage_}),flush=True)
    return np.load(path)


def embed(x,metric,dim):
    y=((profile(x)-metric['center'])/metric['sd'])@metric['projection'][:,:dim]
    return y/np.maximum(np.linalg.norm(y,axis=1,keepdims=True),1e-12)


def calibration(dep,metric,dim):
    x=np.load(e.TRAINED/f'cache/finelag/{dep}/validation_features.npy');z=embed(x,metric,dim)
    meta=pd.read_parquet(e.TRAINED/f'cache/{dep}/validation_metadata.parquet');groups=meta.waveform_parent_uid.to_numpy()
    i,j=np.triu_indices(len(z),1);y=groups[i]==groups[j];raw=(z[i]*z[j]).sum(1)
    w=np.where(y,.5/y.sum(),.5/(~y).sum())
    iso=IsotonicRegression(increasing=True,out_of_bounds='clip').fit(raw,y.astype(float),sample_weight=w)
    floor=1/(y.sum()+2);p=np.clip(iso.y_thresholds_,floor,1-floor)
    return {'dim':dim,'knots':iso.X_thresholds_.tolist(),'loglr':(np.log(p)-np.log1p(-p)).tolist(),
            'min':float(raw.min()),'max':float(raw.max()),'source_pairs':int(y.sum()),'floor':float(floor)}


def pair_raw(dep,es,split,metric,dim):
    f=pd.read_parquet(e.BASE/f'evaluation/{dep}/seed_{es}/{split}_pairs.parquet')
    if split=='real':
        x=np.load(e.TRAINED/f'cache/deployment_event_psd/{dep}/real_features.npy');_,events=dev.real_inputs(dep)
        valid=events.strict_h1l1_preprocessing_pass.to_numpy(bool);zz=embed(x,metric,dim);z=np.full((len(events),dim),np.nan);z[valid]=zz
    else:
        x=np.load(e.TRAINED/f'cache/deployment_event_psd/{dep}/{es}_{split}_features.npy');z=embed(x,metric,dim)
    i,j=f.idx_i.to_numpy(int),f.idx_j.to_numpy(int)
    return f,(z[i]*z[j]).sum(1)


def score(f,raw,spec):
    value=np.interp(raw,spec['knots'],spec['loglr']);ood=(raw<spec['min'])|(raw>spec['max'])
    value=np.where(ood,np.minimum(value,0),value);value=np.clip(value,-4,4)
    return f.waveform_score.to_numpy(float)+spec['beta']*value,value,ood


def run(root):
    trial=root/'trials/PHASE-SOURCE-METRIC'
    if trial.exists():raise RuntimeError('Independent trial required')
    for n in ('contracts','calibration','tables','results','evaluation'):(trial/n).mkdir(parents=True)
    shutil.copy2(__file__,trial/'contracts'/Path(__file__).name)
    dev.json_write(trial/'contracts/RECIPE.json',{'created_utc':datetime.now(timezone.utc).isoformat(),
        'same_O3_O4a':True,'new_features':'mean-centered detector template-match profiles from existing2s waveforms',
        'training':'pooled within-source residual covariance with OAS regularization; maximize between-source/within-source variance',
        'independent_source_weights':'each source has8views, each class mean weighted equally',
        'dimensions':DIMS,'beta_grid':BETAS,'score':'Z_FRT+beta*clip(simulation-calibrated cosine log ratio,-4,4)',
        'fit':'960 training sources per run; class identity from simulation, no public labels',
        'calibration':'240 disjoint encoder-validation sources, positive and negative totalweights0.5each',
        'selection':'BAYESTAR simulation validation only, same FRT-relative injection guardrails and deterministic objective',
        'new_metric_shared_across_three_old_model_seeds':True,
        'not_claimed':'not a physical Bayes factor or a newly calibrated real-PE posterior',
        'references':['https://scikit-learn.org/stable/modules/lda_qda.html','https://scikit-learn.org/stable/modules/generated/sklearn.covariance.OAS.html']})
    selected={};metrics_by_dep={};states=[]
    for dep in e.DEPS:
        metric=fit(root,dep);metrics_by_dep[dep]=metric;cal={d:calibration(dep,metric,d) for d in DIMS}
        for es in dev.SEEDS:
            rows=[];choices=[]
            for dim in DIMS:
                f,raw=pair_raw(dep,es,'validation',metric,dim);base=ev.metrics(f,f.waveform_score.to_numpy(float),dep,es)
                for beta in BETAS:
                    spec={**cal[dim],'beta':beta,'deployment':dep,'eval_seed':es}
                    z,_,ood=score(f,raw,spec);mm=ev.metrics(f,z,dep,es);passed=ev.guard(mm,base)
                    rows.append({'dim':dim,'beta':beta,'pass':passed,'ood':float(ood.mean()),**{method+'_'+k:v for method,m in mm.items() for k,v in m.items()}})
                    if passed:choices.append((e.old_selection.objective(mm,beta,dim),spec))
            out=trial/f'calibration/{dep}/seed_{es}';out.mkdir(parents=True);dev.csv_write(out/'validation_grid.csv',pd.DataFrame(rows))
            cfg=min(choices,key=lambda c:c[0])[1];selected[dep,es]=cfg;dev.json_write(out/'SELECTED_CONFIG.json',cfg)
            states.append({'deployment':dep,'seed':es,'dim':cfg['dim'],'beta':cfg['beta']})
    dev.json_write(trial/'contracts/SELECTED_FREEZE.json',{'status':states,'real_used_to_select':False});print(json.dumps({'phase_metric_selected':states}),flush=True)
    rows=[];guards=[]
    for dep in e.DEPS:
        real={}
        for es in dev.SEEDS:
            spec=selected[dep,es]
            for split in ('validation','test','real'):
                f,raw=pair_raw(dep,es,split,metrics_by_dep[dep],spec['dim']);z,inc,ood=score(f,raw,spec)
                new=f.copy();new['FRT_baseline_waveform_score']=f.waveform_score;new['waveform_score']=z;new['phase_metric_raw']=raw;new['phase_metric_increment']=inc;new['phase_metric_ood']=ood
                out=trial/f'evaluation/{dep}/seed_{es}';out.mkdir(parents=True,exist_ok=True);new.to_parquet(out/f'{split}_pairs.parquet',index=False)
                if split=='real':real[es]=new
                else:
                    bm=ev.metrics(f,f.waveform_score.to_numpy(float),dep,es);mm=ev.metrics(f,z,dep,es);guards.append({'deployment':dep,'seed':es,'split':split,'pass':ev.guard(mm,bm)})
                    for method in bm:
                        for name,m in [('FRT_BASELINE',bm[method]),('CANDIDATE',mm[method])]:rows.append({'deployment':dep,'seed':es,'split':split,'method':method,'config':name,**m})
        dev.save_evaluation(trial,'CANDIDATE',dep,real,pd.read_parquet(root/f'audit/{dep}_frozen_external_reference.parquet'))
    dev.csv_write(trial/'tables/RETRIEVAL_PER_SEED.csv',pd.DataFrame(rows));dev.csv_write(trial/'tables/REUSED_GUARDRAILS.csv',pd.DataFrame(guards));e.assess(root,trial)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();run(a.root)
