#!/usr/bin/env python3
"""Development-only conditional temperature/tilt of waveform mass densities."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize,check_grad
from scipy.special import softmax,logsumexp
import torch

P=Path('/root/autodl-tmp/gw-catalog');sys.path.insert(0,str(P/'scripts/experiments'))
import mcwf_temporal_response_evaluate_20260908 as ev
import mcwf_multirate_train_20260908 as mult
t,old,dev,cf=ev.t,ev.t.old,ev.dev,ev.cf
MULTI=P/'results/mcwf_multirate_lowband_exploratory_20260908T145551Z'
KINDS=('OMC-CONDITIONAL-CAL','MULTIRATE-CONDITIONAL-CAL')
RIDGES=(.001,.01,.1,1.,10.)


def features(p):
    logp=np.log(p.clip(1e-300));mean=p@old.CENTERS
    sd=np.sqrt((p*(old.CENTERS[None]-mean[:,None])**2).sum(1)).clip(.007)
    entropy=-(p*logp).sum(1)/np.log(p.shape[1])
    return np.c_[mean,np.log(sd),entropy],logp,((old.CENTERS[None]-mean[:,None])/sd[:,None]).clip(-8,8)


def target(truth):
    v=np.clip(truth,old.CENTERS[0],old.CENTERS[-1]);h=np.searchsorted(old.CENTERS,v,side='right').clip(1,511);l=h-1
    f=(v-old.CENTERS[l])/(old.CENTERS[h]-old.CENTERS[l]);y=np.zeros((len(v),512));y[np.arange(len(v)),l]=1-f;y[np.arange(len(v)),h]=f
    return y


def objective(theta,x,lp,z,y,w,ridge):
    a,b=x@theta[:4],x@theta[4:]
    temp=np.exp(a.clip(-np.log(2),np.log(2)));tilt=b.clip(-1,1)
    logits=lp/temp[:,None]+tilt[:,None]*z
    logprob=logits-logsumexp(logits,axis=1,keepdims=True);res=(np.exp(logprob)-y)*w[:,None]
    loss=float(-(w[:,None]*y*logprob).sum()+.5*ridge*(theta@theta))
    da=(res*(-lp/temp[:,None])).sum(1)*(abs(a)<np.log(2))
    db=(res*z).sum(1)*(abs(b)<1)
    grad=np.r_[x.T@da,x.T@db]+ridge*theta
    return loss,grad


def transform(p,spec):
    f,lp,z=features(p);x=np.c_[np.ones(len(p)),((f-spec['mu'])/spec['sd']).clip(-4,4)]
    theta=np.asarray(spec['theta']);temp=np.exp((x@theta[:4]).clip(-np.log(2),np.log(2)));tilt=(x@theta[4:]).clip(-1,1)
    out=softmax(lp/temp[:,None]+tilt[:,None]*z,1)
    outside=((f<np.asarray(spec['minimum'])-1e-10)|(f>np.asarray(spec['maximum'])+1e-10)).any(1)
    out[outside]=p[outside];temp[outside]=1;tilt[outside]=0
    return out,{'conditional_temperature':temp,'conditional_tilt':tilt,'mass_calibration_OOD_identity':outside}


def numerical_tests():
    rng=np.random.default_rng(202609985);p=softmax(rng.normal(size=(11,512)),1);f,lp,z=features(p)
    mu,sd=f.mean(0),f.std(0).clip(.01);x=np.c_[np.ones(len(p)),(f-mu)/sd]
    y=target(rng.uniform(np.log(6),np.log(150),len(p)));w=np.ones(len(p))/len(p);theta=rng.normal(0,.025,8)
    error=check_grad(lambda a:objective(a,x,lp,z,y,w,.1)[0],lambda a:objective(a,x,lp,z,y,w,.1)[1],theta)
    spec={'mu':mu,'sd':sd,'theta':np.zeros(8),'minimum':f.min(0),'maximum':f.max(0)}
    result,_=transform(p,spec);identity=float(abs(result-p).max())
    if error>2e-5 or identity>1e-12 or not np.allclose(result.sum(1),1):raise RuntimeError('Conditional calibration numericaltest')
    return {'pass':True,'gradient_norm_error':error,'identity_max_abs_difference':identity,'probability_normalization':True}


def initialize(root):
    if root.exists():raise RuntimeError('Fresh output only')
    for name in ('contracts','scripts','logs','models','predictions','calibration','tables','reports','manifest','evaluation','figures'):(root/name).mkdir(parents=True)
    contract={'id':'MCWF-CONDITIONAL-MASS-CALIBRATION-20','UTC':datetime.now(timezone.utc).isoformat(),'status':t.STATUS,
      'goal_achieved':False,'changed_channels':['waveform'],'outer_weights_changed':False,'same_both_runs':True,
      'hypothesis':'One global predictive-density temperature may miss mass/quality-dependent bias and uncertainty. Test a bounded8-parameter correction using waveform-predicted quantities only.',
      'parents':['unchanged original OMC mass predictor','frozen06 MULTIRATE predictor'],
      'features':'predictive meanlogMc,logpredictiveSD,normalizedentropy;standardizeonfitnoisehalf;never PE,actualsource mass,network SNR field,time,sky or ID at inference.',
      'formula':'logp_cal(k)=logp(k)/T(x)+b(x)*clip((logMc_k-mean)/SD,-8,8)-logsumexp;T=exp(clip(Xa,-ln2,ln2)),b=clip(Xb,-1,1)',
      'fit':'evenhashdevelopmentnoisehalf;eachsource totalweight1;discardcross-half sources;truthfromsimulation only;regularizedproperlogscore',
      'tune':'other source/noise-disjointdevelopmenthalf chooses ridgefrom.001,.01,.1,1,10 or identity;no validation/testpair metrics in densityfit',
      'support':'outside eventfeature min/max ofdevelopment fit+tune ->identity;temperature.5..2,tilt-1..1;not ad hoc narrowing to match realPE.',
      'pair_calibration':'Same frozen split conventions forfiniteMc-tail andisotonicprioroverlap;reuseofdevelopment noted. Paircoefficients selectedonlyBAYESTAR validation.',
      'frozen':['allencoders','peak2s data','one-dimensionaltime','skyrawlogBF','outerweights','scope','history'],
      'limits':'Additionalcalibrator,notfullPE;developmentusedforpriorcheckpointselection so not anindependent coverage test. Post-calibration marginal prior is approximate,score empirical not a proper lensBayesfactor.',
      'no_external_inputs':True,'real_feedback':'adaptive development,notblindconfirmation','required':'fresh independent source/noise confirmation afteranyjointwinner'}
    dev.json_write(root/'contracts/ANALYSIS_CONTRACT.json',contract)
    rows=t.protected()
    for p in (MULTI/'models/MULTIRATE').glob('*/*/selected.pt'):rows.append({'path':str(p),'sha256':dev.sha(p),'bytes':p.stat().st_size})
    dev.csv_write(root/'manifest/INPUT_SHA256.csv',pd.DataFrame(rows));dev.json_write(root/'contracts/NUMERICAL_TESTS.json',numerical_tests())
    shutil.copy2(__file__,root/'scripts/conditional_mass_calibration.py')
    dev.json_write(root/'contracts/START_FREEZE.json',{'contract_sha256':dev.sha(root/'contracts/ANALYSIS_CONTRACT.json'),'code_sha256':dev.sha(Path(__file__))})


def parent(kind,dep,slot):
    if kind.startswith('OMC'):
        folder=t.PREVIOUS/f'ordered_mass_predictor/models/{dep}/seed_{slot}'
        return folder/'validation_selected_model.pt',folder/'validation_predictions.npz'
    folder=MULTI/f'models/MULTIRATE/{dep}/seed_{slot}';return folder/'selected.pt',folder/'development_predictions.npz'


def fit_one(root,dep,kind,slot):
    out=root/f'models/{kind}/{dep}/seed_{slot}';out.mkdir(parents=True,exist_ok=True)
    if (out/'COMPLETE.json').exists():return
    cp,path=parent(kind,dep,slot);ck=torch.load(cp,map_location='cpu',weights_only=False);original=np.load(path)
    p=original['p'];truth=original['truth'];meta=pd.read_parquet(t.PREVIOUS/f'expanded_data/{dep}/validation/event_metadata.parquet')
    if len(p)!=len(meta) or not np.allclose(truth,np.log(meta.mc_det),atol=1e-12):raise RuntimeError('Developmenttruthindex')
    group=meta.source_uid.to_numpy(str)
    banks=sorted(meta.noise_bank_index.unique(),key=lambda x:hashlib.sha256(f'202609986:{dep}:{x}'.encode()).hexdigest())
    fold=meta.noise_bank_index.map({b:i%2 for i,b in enumerate(banks)}).to_numpy(int)
    mixed={g for g in np.unique(group) if len(set(fold[group==g]))>1};fold[np.isin(group,list(mixed))]=-1
    fit,tune=fold==0,fold==1
    f,lp,z=features(p);mu,sd=f[fit].mean(0),f[fit].std(0).clip(.01);x=np.c_[np.ones(len(p)),((f-mu)/sd).clip(-4,4)];y=target(truth)
    w=np.zeros(len(p))
    for mask in (fit,tune):
        counts=pd.Series(group[mask]).value_counts();w[mask]=pd.Series(group[mask]).map(1/counts).to_numpy()/len(counts)
    rows=[];configs=[]
    for ridge in (None,*RIDGES):
        if ridge is None:theta=np.zeros(8);success=True;iterations=0
        else:
            result=minimize(lambda th:objective(th,x[fit],lp[fit],z[fit],y[fit],w[fit],ridge),np.zeros(8),jac=True,
                method='L-BFGS-B',bounds=[(-2,2)]*8,options={'maxiter':400,'ftol':1e-12,'gtol':1e-7})
            theta=result.x;success=bool(result.success);iterations=result.nit
        spec={'mu':mu.tolist(),'sd':sd.tolist(),'theta':theta.tolist(),'minimum':f[fold>=0].min(0).tolist(),'maximum':f[fold>=0].max(0).tolist()}
        calibrated,_=transform(p,spec)
        nll=float(-(w[tune,None]*y[tune]*np.log(calibrated[tune].clip(1e-300))).sum())
        rows.append({'ridge':'identity' if ridge is None else ridge,'tuneNLL':nll,'success':success,'iterations':iterations,'theta_norm':float(np.linalg.norm(theta))})
        configs.append(spec)
    eligible=[i for i,row in enumerate(rows) if row['success']]
    win=min(eligible,key=lambda i:(rows[i]['tuneNLL'],rows[i]['theta_norm']))
    spec=configs[win];spec.update(parent_sha256=dev.sha(cp),parent_path=str(cp),selection=rows[win],fit_sources=len(set(group[fit])),
        tune_sources=len(set(group[tune])),discard_crossnoise_sources=len(mixed),no_test_or_real_used=True)
    calibrated,diag=transform(p,spec)
    torch.save({'prior':ck['prior'],'density_calibration':spec,'temperature':'conditionalnotglobal','kind':kind,'slot':slot,'epoch':'posthoc'},out/'selected.pt')
    dev.json_write(out/'EVENT_DENSITY_CALIBRATION.json',spec);dev.csv_write(out/'RIDGE_GRID.csv',pd.DataFrame(rows))
    np.savez_compressed(out/'development_predictions.npz',p=calibrated,outside=original['outside'],group=original['group'],truth=truth,**diag)
    audit=[]
    for side,mask in (('fit',fit),('noise_disjoint_tune',tune)):
        for name,prob in (('original',p),('conditional_calibrated',calibrated)):
            cdf=np.c_[np.zeros(len(prob)),prob.cumsum(1)];pit=np.array([np.interp(a,old.EDGES,b) for a,b in zip(truth,cdf)])
            audit.append({'cohort':side,'model':name,'sources':len(set(group[mask])),'logMc_MAE':float((w[mask]*abs(prob[mask]@old.CENTERS-truth[mask])).sum()),
                'central90coverage':float((w[mask]*((pit[mask]>=.05)&(pit[mask]<=.95))).sum()),'central50coverage':float((w[mask]*((pit[mask]>=.25)&(pit[mask]<=.75))).sum())})
    dev.csv_write(out/'COVERAGE_AUDIT.csv',pd.DataFrame(audit))
    dev.json_write(out/'COMPLETE.json',{'kind':kind,'deployment':dep,'slot':slot,'identity_selected':win==0,
        'sha256':dev.sha(out/'selected.pt'),'selection':rows[win],'density_only_no_encoder_training':True})
    print(json.dumps({'fit':[dep,kind,slot],**rows[win]}),flush=True)


def predict(root,dep,kind,slot,es,split):
    dest=root/f'predictions/{kind}/{dep}/model_{slot}_eval_{es}/{split}.npz'
    if dest.exists():return np.load(dest)
    ck=torch.load(root/f'models/{kind}/{dep}/seed_{slot}/selected.pt',map_location='cpu',weights_only=False)
    path=(t.PREVIOUS/f'ordered_mass_predictor/predictions/{dep}/model_{slot}_eval_{es}/{split}.npz'
          if kind.startswith('OMC') else MULTI/f'predictions/MULTIRATE/{dep}/model_{slot}_eval_{es}/{split}.npz')
    if not path.exists():raise RuntimeError('Required frozen prediction missing: '+str(path))
    a=np.load(path)
    p=a['p'];valid=np.isfinite(p).all(1);out=p.copy()
    transformed,diag=transform(p[valid],ck['density_calibration']);out[valid]=transformed
    columns={}
    for key,value in diag.items():
        extended=np.full(len(p),np.nan);extended[valid]=value;columns[key]=extended
    dest.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest,p=out,outside=a['outside'],**columns)
    return np.load(dest)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--stage',required=True,choices=['initialize','fit','select','evaluate','real','assess']);a=p.parse_args()
    t.predict=predict;mult.KINDS=KINDS
    if a.stage=='initialize':initialize(a.root)
    elif a.stage=='fit':
        for dep in t.DEPS:
            for kind in KINDS:
                for slot in t.MODEL_SLOTS:fit_one(a.root,dep,kind,slot)
    elif a.stage=='select':mult.select(a.root)
    else:getattr(ev,a.stage)(a.root)
