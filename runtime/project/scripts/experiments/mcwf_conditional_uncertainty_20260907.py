#!/usr/bin/env python3
"""Source-disjoint calibration conditional on waveform prediction uncertainty."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='2'
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss
import mcwf_pe_frontend_extension_v2_20260907 as e
import mcwf_mixture_density_20260907 as mdn
import mcwf_mixture_evaluate_20260907 as mev
import mcwf_expanded_calibration_20260907 as calibration

dev,body,ev=e.dev,e.body,e.ev
BETAS=(0.,.0625,.125,.25,.5,1.,2.)


def features(f,a):
    i,j=f.idx_i.to_numpy(int),f.idx_j.to_numpy(int)
    w,m,c=np.asarray(a['w'],float),np.asarray(a['m'],float),np.asarray(a['cov'],float)
    mean=np.sum(w[:,:,None]*m,axis=1)
    variance=(np.sum(w[:,:,None]*(np.diagonal(c,axis1=-2,axis2=-1)+m*m),axis=1)-mean*mean).clip(1e-8)
    pooled=variance[i]+variance[j]
    distance=np.abs(mean[i]-mean[j])/np.sqrt(pooled)
    v=mev.features(a,i,j)
    return np.column_stack([f.waveform_score.to_numpy(float),v['mass'],v['conditional'],
        distance,np.log(pooled[:,0]),(mean[i,0]+mean[j,0])/2,
        np.abs(np.log(variance[i,0]/variance[j,0]))])


def model(seed):
    return HistGradientBoostingClassifier(learning_rate=.05,max_iter=150,max_leaf_nodes=4,
        max_depth=2,min_samples_leaf=40,l2_regularization=5.,early_stopping=False,random_state=seed)


def weights(y):
    return np.where(y,1.,y.sum()/max(1,(~y).sum()))


def fit(root,trial,dep,ms,es):
    out=trial/f'calibration/{dep}/seed_{es}';out.mkdir(parents=True)
    f=calibration.development(root,dep,ms,es)
    a=np.load(root/f'mixture_density/models/{dep}/seed_{ms}/validation_predictions.npz')
    x=features(f,a);y=f.is_true_pair.to_numpy(bool)
    source=a['group'];fold=np.random.default_rng(202609201).permutation(len(np.unique(source)))%5
    fi,fj=fold[source[f.idx_i.to_numpy(int)]],fold[source[f.idx_j.to_numpy(int)]]
    rows=[]
    for k in range(5):
        train=(fi!=k)&(fj!=k);test=(fi==k)&(fj==k)
        if not (train.any() and test.any()):raise RuntimeError('Invalid source-fold split')
        for kind,xx in [('base',x[:,:1]),('full',x)]:
            clf=model(ms);clf.fit(xx[train],y[train],sample_weight=weights(y[train]))
            rows.append({'fold':k,'kind':kind,'train_positive_sources':int(y[train].sum()),
                'test_positive_sources':int(y[test].sum()),
                'balanced_log_loss':float(log_loss(y[test],clf.predict_proba(xx[test])[:,1],sample_weight=weights(y[test]))),
                'no_source_crossing':True})
    fitted={}
    for kind,xx in [('base',x[:,:1]),('full',x)]:
        clf=model(ms);clf.fit(xx,y,sample_weight=weights(y));fitted[kind]=clf
        joblib.dump(clf,out/f'{kind}_classifier.joblib')
    spec={'min':x.min(0).tolist(),'max':x.max(0).tolist(),'independent_sources':512,'noise_blocks':32}
    dev.csv_write(out/'SOURCE_FOLD_AUDIT.csv',pd.DataFrame(rows));dev.json_write(out/'FIT.json',spec)
    return fitted,spec


def get(root,dep,ms,es,split):
    f=pd.read_parquet(e.BASE/f'evaluation/{dep}/seed_{es}/{split}_pairs.parquet')
    return f,features(f,mdn.prediction(root,dep,ms,es,split))


def score(f,x,cal,spec):
    full=cal['full'].decision_function(x);base=cal['base'].decision_function(x[:,:1])
    delta=np.clip(full-base,-3.,3.)
    ood=((x<np.asarray(spec['min']))|(x>np.asarray(spec['max']))).any(1)
    delta=np.where(ood,np.minimum(delta,0.),delta)
    return f.waveform_score.to_numpy(float)+spec['beta']*delta,delta,ood


def run(root):
    trial=root/'trials/UNCERTAINTY-CONDITIONAL-WAVEFORM-EVIDENCE'
    if trial.exists():raise RuntimeError('Independent trial required')
    for name in ('contracts','tables','calibration','evaluation','results'):(trial/name).mkdir(parents=True)
    shutil.copy2(__file__,trial/'contracts'/Path(__file__).name)
    dev.json_write(trial/'contracts/RECIPE.json',{'created_utc':datetime.now(timezone.utc).isoformat(),
        'same_O3_O4a':True,'features':['Z_FRT','MDNlogMcproduct','MDNjoint-minus-Mcproduct','DpredMc','Dpredq','Dpredchi',
            'log_sum_mass_variance','mean_mass_center','abs_log_mass_variance_ratio'],
        'Dpred_definition':'differences of learned mixture means over pooled predictive widths,not public PE D',
        'fit':'512 source-disjoint expanded validation parents,one companion pair/source,class-balanced null;32 shared noise blocks',
        'classifier':'HGB,150 trees,maxdepth2,max4leaves,lr0.05,L2=5,minleaf40,no row-level early-stopping split',
        'source_folds':'fixed5source folds for diagnostic only;crossfold pairs excluded,no model choice by fold outcome',
        'formula':'Z_FRT+beta*clip(logLR(full)-logLR(Z_FRT),-3,3)',
        'support':'no positive increment outside development feature ranges',
        'beta_grid':BETAS,'selection':'original BAYESTAR validation objective/guards',
        'frozen':['time','sky','outer weights','event scope','old checkpoints','historical files'],
        'interpretation':'correlated waveform predictive evidence,not physical posterior/Bayesfactor',
        'no_PE_official_or_ID_inputs':True,'adaptive_real_development':True,'fresh_confirmation_required':True,
        'reference':'https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html'})
    selected,cache,calibrators,states={},{},{},[]
    for dep in e.DEPS:
        for ms,es in zip(body.MODEL_SEEDS,dev.SEEDS):
            cal,spec=fit(root,trial,dep,ms,es);calibrators[dep,es]=cal
            f,x=get(root,dep,ms,es,'validation');cache[dep,es]=f,x
            bm=ev.metrics(f,f.waveform_score.to_numpy(float),dep,es);rows,options=[],[]
            for beta in BETAS:
                candidate={**spec,'beta':beta};z,*_=score(f,x,cal,candidate)
                mm=ev.metrics(f,z,dep,es);ok=ev.guard(mm,bm)
                rows.append({'beta':beta,'pass':ok,**{a+'_'+k:v for a,b in mm.items() for k,v in b.items()}})
                if ok:options.append((e.old_selection.objective(mm,0,beta),candidate))
            out=trial/f'calibration/{dep}/seed_{es}'
            dev.csv_write(out/'validation_grid.csv',pd.DataFrame(rows));spec=min(options,key=lambda a:a[0])[1]
            selected[dep,es]=spec;dev.json_write(out/'SELECTED_CONFIG.json',spec)
            states.append({'deployment':dep,'seed':es,'beta':spec['beta']})
    dev.json_write(trial/'contracts/SELECTED_FREEZE.json',{'configs':states,'real_used_to_select':False})
    print(json.dumps({'uncertainty_selected':states}),flush=True)
    rows,guards=[],[]
    for dep in e.DEPS:
        real={}
        for ms,es in zip(body.MODEL_SEEDS,dev.SEEDS):
            for split in ('validation','test','real'):
                f,x=cache[dep,es] if split=='validation' else get(root,dep,ms,es,split)
                z,inc,ood=score(f,x,calibrators[dep,es],selected[dep,es]);n=f.copy()
                for col in ('final_score','waveform_contribution','time_contribution','sky_contribution','rank','method'):
                    if col in n:n=n.rename(columns={col:'baseline_'+col})
                n['FRT_baseline_waveform_score']=f.waveform_score;n['waveform_score']=z
                n['uncertainty_increment']=inc;n['uncertainty_ood']=ood
                out=trial/f'evaluation/{dep}/seed_{es}';out.mkdir(parents=True,exist_ok=True)
                n.to_parquet(out/f'{split}_pairs.parquet',index=False)
                if split=='real':real[es]=n
                else:
                    bm,mm=ev.metrics(f,f.waveform_score.to_numpy(float),dep,es),ev.metrics(f,z,dep,es)
                    guards.append({'deployment':dep,'seed':es,'split':split,'pass':ev.guard(mm,bm)})
                    for method in mm:
                        for config,m in [('FRT_BASELINE',bm[method]),('CANDIDATE',mm[method])]:
                            rows.append({'deployment':dep,'seed':es,'split':split,'method':method,'config':config,**m})
        dev.save_evaluation(trial,'CANDIDATE',dep,real,pd.read_parquet(root/f'audit/{dep}_frozen_external_reference.parquet'))
    dev.csv_write(trial/'tables/RETRIEVAL_PER_SEED.csv',pd.DataFrame(rows))
    dev.csv_write(trial/'tables/REUSED_GUARDRAILS.csv',pd.DataFrame(guards));e.assess(root,trial)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    run(parser.parse_args().root)
