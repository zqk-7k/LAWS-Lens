#!/usr/bin/env python3
"""A simulation-calibrated joint physical-profile distance expert."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse
from copy import deepcopy
import json
from pathlib import Path
import pickle
import shutil
import sys
import numpy as np
import pandas as pd
from scipy.special import gammaln
from sklearn.ensemble import HistGradientBoostingClassifier

P=Path('/root/autodl-tmp/gw-catalog')
sys.path.insert(0,str(P/'scripts/experiments'))
import mcwf_confidence_level_validation_20260909 as cv
import mcwf_minimal_joint_waveform_classifier_20260909 as minimal
st,ctrl,engine,base,n,r,co,h=(getattr(cv,k)for k in ('st','ctrl','engine','base','n','r','co','h'))
ROOT=PARENT=DATA=PREDICTIVE=None
METHODS=('NODUP-DIRECT-REPLAY','JOINTSTATE-FROZEN95','PROFILEPAIR-LINEAR',
         'PROFILEPAIR-TREE','PROFILEPAIR-VALIDATION-SELECTED')
FIELDS=st.FIELDS+('physical_profile_pair_used','physical_profile_distance2',
                 'physical_profile_variance_scale','physical_profile_OOD')
MEMO={}


def freeze():
    if ROOT.exists():
        raise RuntimeError('Independent output required')
    cv.frozen_parent()
    for folder in ('contracts','configs','calibration','tables','audit','reports','scripts',
                   'logs','manifest','results','figures'):
        (ROOT/folder).mkdir(parents=True)
    n.write_json(ROOT/'contracts/ANALYSIS_CONTRACT.json',{'UTC':n.utc(),
        'id':'MCWF-NODUP-PHYSICAL-PAIR-METRIC-41','status':n.STATUS,
        'data':str(DATA),'parent':str(PARENT),
        'metric_coordinates':['profile logMc','profile eta=q/(1+q)^2','profile equal-aligned chi'],
        'motivation':'Retain correlated intrinsic-profile disagreement; the previous updated marginal kept broad NN conditional eta/spin. This is a new waveform-only metric, not public PE.',
        'source_fit':'One difference vector per true source pair; sign is irrelevant and mean fixed zero for exchange symmetry. Repeated signs are NOT counted as independent samples.',
        'covariance':'Mean delta*delta.T / v. Shrink toward its diagonal; do not treat this as inverseFisher or PE covariance.',
        'variance_scaling':'v=(power_i/p0)^(-a)/2+(power_j/p0)^(-a)/2, a=0or1; projection power is a fitted-waveform quality proxy,not catalogPE SNR.',
        'metric_grid':{'power_exponent':[0,1],'diagonal_shrinkage':[.01,.1,.5,1.],
                       'Student_df':[3,5,10,30]},
        'metric_choice':'Fit fold0 true systems; tune fold1 true-pair Student-t proper logloss, with covariance/scale conversion included. Per-run calibration, same method.',
        'classifier_features':['embedding cosine','-log(1+joint_profile_distance_squared)'],
        'classifier_grid':{'ridge':[.001,.01,.1,1.],'leaves':[3,7],'iterations':100},
        'classifier_choice':'Source/noise-disjoint new tune balanced proper logloss. Linear and monotone tree both reported; selected type by logloss,linear tie.',
        'only_both_valid_profiles':'For pair with two R10-quality-valid profiles and power in fit support, replace WHOLE waveform score by ONE calibrated cosine+joint-distance expert. No additional cosine, Mc, q, spin or jointBC increment.',
        'fallback':'All other pairs exactly retain frozen R35 JOINTSTATE-GLOBAL. This is feature-availability gating,not numeric total-score blending.',
        'state_offset':'Conditional expert LR plus logP(both-valid|L)/P(both-valid|N), from source-weighted fit pairs.',
        'support':'At least20fitand20tune true systems. FeatureboxOOD positive reward0; fixed cap log(totalfittrue+1).',
        'probability_limit':'Empirical correlated profile-metric classifier is a ranking proxy. It is not a lensing Bayes factor or full parameter estimation.',
        'fixed':['allencoders','profile optima','time','sky','outerweights','eventscope','historicalresults'],
        'no_old_encoder_Mc_q':True,'no_total_blend':True,'same_framework_both_runs':True,
        'adaptive_real_feedback':True,'blind_confirmation':False,
        'no_real_PE_official_input_or_hyperparameter_selection':True,
        'refs':['https://arxiv.org/abs/gr-qc/9402014','https://arxiv.org/abs/1506.02169',
                'https://www.jmlr.org/papers/v11/cawley10a.html'],
        'references_do_not_prove_this_metric_optimal':True})
    paths=[PARENT/'configs/SELECTED_CONFIGURATIONS.json',Path(__file__),Path(cv.__file__),Path(minimal.__file__)]
    for dep in n.DEPS:
        paths.extend([DATA/f'data/{dep}/event_metadata.parquet',
            PREDICTIVE/f'predictions/{dep}_BOUNDED_development.npz'])
        paths.extend(sorted((DATA/f'profile_events/{dep}').glob('*.json')))
    n.write_csv(ROOT/'manifest/INPUT_SHA256.csv',[{'path':str(p),'sha256':n.sha(p),'bytes':p.stat().st_size}for p in paths])
    shutil.copy2(__file__,ROOT/'scripts/physical_pair_metric.py')
    n.write_json(ROOT/'contracts/START_FREEZE.json',{'UTC':n.utc(),
        'contract_sha256':n.sha(ROOT/'contracts/ANALYSIS_CONTRACT.json')})


def point_arrays(paths, count):
    point=np.full((count,3),np.nan);power=np.full(count,np.nan)
    for path in paths:
        row=json.loads(path.read_text());idx=int(row['row_index']);q=float(row['q'])
        point[idx]=[row['logmc'],q/(1+q)**2,row['chieff_equal']]
        power[idx]=row['projection_statistic']
    return point,power


def development(dep):
    key=('dev',dep)
    if key not in MEMO:
        meta=pd.read_parquet(DATA/f'data/{dep}/event_metadata.parquet')
        active=np.load(PREDICTIVE/f'predictions/{dep}_BOUNDED_development.npz')['profile_active']
        point,power=point_arrays(sorted((DATA/f'profile_events/{dep}').glob('*.json')),len(meta))
        if not np.isfinite(point[active]).all()or not np.isfinite(power[active]).all()or (power[active]<=0).any():
            raise RuntimeError('Missing valid physical-profile input')
        MEMO[key]=(meta,active,point,power)
    return MEMO[key]


def source_pairs(meta,active,point,power,side):
    subset=meta[(meta.fold==side)&active].groupby('source_uid',sort=True)
    i,j,groups=[],[],[]
    for group,rows in subset:
        if len(rows)==2:
            a,b=rows.row_index.to_numpy(int);i.append(a);j.append(b);groups.append(group)
    i,j=np.array(i),np.array(j)
    if len(i)<20:
        raise RuntimeError('HOLD_INSUFFICIENT_TRUE_SOURCE_SUPPORT')
    return point[i]-point[j],power[i],power[j],groups


def variance(a,b,spec):
    return .5*((a/spec['power_reference'])**(-spec['power_exponent'])+
               (b/spec['power_reference'])**(-spec['power_exponent']))


def distance(delta,a,b,spec):
    scale=variance(a,b,spec)
    d2=np.einsum('ni,ij,nj->n',delta,np.linalg.inv(np.asarray(spec['covariance'])),delta)/scale
    return np.maximum(d2,0.),scale


def metric(dep):
    meta,active,point,power=development(dep)
    x,a,b,groups=source_pairs(meta,active,point,power,0)
    tx,ta,tb,tg=source_pairs(meta,active,point,power,1)
    reference=float(np.median(np.r_[a,b]));trials=[]
    for exponent in (0,1):
        spec={'power_reference':reference,'power_exponent':exponent}
        v=variance(a,b,spec)
        empirical=np.einsum('ni,nj,n->ij',x,x,1/v)/len(x)
        for shrink in (.01,.1,.5,1.):
            cov=(1-shrink)*empirical+shrink*np.diag(np.diag(empirical))
            if np.linalg.eigvalsh(cov).min()<=0:
                raise RuntimeError('Nonpositive covariance')
            spec.update(covariance=cov.tolist(),shrinkage=shrink)
            d2,tv=distance(tx,ta,tb,spec)
            if not np.allclose(distance(-tx,ta,tb,spec)[0],d2,rtol=0,atol=1e-10):
                raise RuntimeError('Pair-swap invariant failed')
            for df in (3,5,10,30):
                factor=(df-2)/df
                logdet=np.linalg.slogdet(cov)[1]+3*np.log(tv*factor)
                lp=gammaln((df+3)/2)-gammaln(df/2)-1.5*np.log(df*np.pi)-.5*logdet-.5*(df+3)*np.log1p(d2/(factor*df))
                trials.append({**deepcopy(spec),'df':df,'NLL':float(-lp.mean())})
    best=min(trials,key=lambda x:(x['NLL'],x['power_exponent'],-x['shrinkage'],-x['df']))
    best.update(fit_sources=len(groups),tune_sources=len(tg),power_min=float(min(a.min(),b.min())),
                power_max=float(max(a.max(),b.max())))
    n.write_csv(ROOT/f'tables/{dep}_METRIC_GRID.csv',[{k:v for k,v in row.items()if k!='covariance'}for row in trials])
    n.write_json(ROOT/f'calibration/{dep}_METRIC.json',best)
    print('PROFILE_METRIC_SELECTED',dep,{k:v for k,v in best.items()if k!='covariance'},flush=True)
    return best


def panels(dep,seed,spec):
    meta,active,point,power=development(dep)
    fold=meta.fold.to_numpy();groups=meta.source_uid.to_numpy(str);noise=meta.noise_bank_index.to_numpy(int)
    if set(groups[fold==0])&set(groups[fold==1])or set(noise[fold==0])&set(noise[fold==1]):
        raise RuntimeError('Source/noise leakage')
    z=np.load(DATA/f'predictions/{dep}/{seed}_embedding.npy').astype(float)
    z/=np.linalg.norm(z,axis=1,keepdims=True)
    out={}
    for side in (0,1):
        ids=np.flatnonzero(fold==side);i,j=np.triu_indices(len(ids),1);i,j=ids[i],ids[j]
        keep=noise[i]!=noise[j];i,j=i[keep],j[keep]
        y=groups[i]==groups[j]
        valid=(active[i]&active[j]&(power[i]>=spec['power_min'])&(power[i]<=spec['power_max'])&
               (power[j]>=spec['power_min'])&(power[j]<=spec['power_max']))
        codes=np.minimum(i//2,j//2)*len(meta)+np.maximum(i//2,j//2)
        _,inv,counts=np.unique(codes,return_inverse=True,return_counts=True)
        w=1/counts[inv];w[y]*=.5/w[y].sum();w[~y]*=.5/w[~y].sum()
        offset=float(np.log(w[valid&y].sum()/w[y].sum())-np.log(w[valid&~y].sum()/w[~y].sum()))
        ii,jj=i[valid],j[valid]
        ds,_=distance(point[ii]-point[jj],power[ii],power[jj],spec)
        x=np.column_stack([np.sum(z[ii]*z[jj],1),-np.log1p(ds)])
        yy,ww=y[valid],w[valid].copy();ww[yy]*=.5/ww[yy].sum();ww[~yy]*=.5/ww[~yy].sum()
        out[side]=(x,yy,ww,offset,int(y.sum()))
    return out


def fit_classifier(dep,seed,kind,panels):
    x,y,w,offset,total=panels[0];tx,ty,tw,_,_=panels[1]
    if min(y.sum(),ty.sum())<20:
        raise RuntimeError('HOLD_INSUFFICIENT_TRUE_CLASSIFIER_SUPPORT')
    models=[];trials=[]
    for ridge in (.001,.01,.1,1.):
        for leaves in ((0,)if kind=='LINEAR'else(3,7)):
            if kind=='LINEAR':
                model=minimal.linear(x,y,w,ridge)
            else:
                estimator=HistGradientBoostingClassifier(max_iter=100,learning_rate=.05,max_leaf_nodes=leaves,
                    min_samples_leaf=20,l2_regularization=ridge,early_stopping=False,monotonic_cst=[1,1],random_state=2026090941)
                estimator.fit(x,y,sample_weight=w*len(x));model={'kind':'TREE','estimator':estimator,'leaves':leaves}
            logits=r.raw_predict(model,tx)
            loss=float(tw@(np.logaddexp(0.,logits)-ty*logits))
            models.append(model);trials.append({'ridge':ridge,'leaves':leaves,'logloss':loss})
    win=min(range(len(trials)),key=lambda k:(trials[k]['logloss'],trials[k]['leaves'],-trials[k]['ridge']))
    path=ROOT/f'calibration/{dep}_{seed}_{kind}.pkl'
    with path.open('xb')as file:
        pickle.dump(models[win],file)
    n.write_csv(path.with_suffix('.GRID.csv'),trials)
    return {'file':str(path.relative_to(ROOT)),'sha256':n.sha(path),'state':2,'minimum':x.min(0).tolist(),
            'maximum':x.max(0).tolist(),'partition_log_ratio':offset,'cap':float(np.log(total+1)),
            'fit_sources':int(y.sum()),'tune_sources':int(ty.sum()),'selected':trials[win],'kind':kind}


def panel(dep,seed,split,catalog=None):
    frame=cv.panel(dep,seed,split,catalog)
    tag=co.app.tag_for(seed,split,catalog)
    key=('panel',dep,tag)
    count=int(max(frame.idx_i.max(),frame.idx_j.max()))+1
    if key not in MEMO:
        paths=sorted((co.PARENT/f'profile_events/{dep}/{tag}').glob('*.json'))
        MEMO[key]=point_arrays(paths,count)
    point,power=MEMO[key];i,j=frame.idx_i.to_numpy(int),frame.idx_j.to_numpy(int)
    for k,name in enumerate(('logmc','eta','chi')):
        frame['physical_delta_'+name]=point[i,k]-point[j,k]
    frame['physical_power_i'],frame['physical_power_j']=power[i],power[j]
    return frame


def infer(frame,config):
    if config['method']==METHODS[0]:
        return st.infer(frame,{**config,'method':METHODS[0]})
    parent=deepcopy(config['fallback'])
    parent['method']='PARENT_GLOBAL'
    result=st.infer(frame,parent)
    if config['method']==METHODS[1]:
        return result
    spec=config['metric']
    delta=frame[['physical_delta_logmc','physical_delta_eta','physical_delta_chi']].to_numpy(float)
    a,b=frame.physical_power_i.to_numpy(float),frame.physical_power_j.to_numpy(float)
    valid=frame.profile_active_i.to_numpy(bool)&frame.profile_active_j.to_numpy(bool)
    if not np.isfinite(delta[valid]).all()or not np.isfinite(a[valid]).all()or not np.isfinite(b[valid]).all():
        raise RuntimeError('Missing valid profile')
    support=(a>=spec['power_min'])&(a<=spec['power_max'])&(b>=spec['power_min'])&(b<=spec['power_max'])
    use=valid&support;d2=np.full(len(frame),np.nan);scale=d2.copy()
    d2[use],scale[use]=distance(delta[use],a[use],b[use],spec)
    out=tuple(v.copy()for v in result)
    if use.any():
        x=np.column_stack([frame.embedding_only.to_numpy(float)[use],-np.log1p(d2[use])])
        update=minimal.apply(x,config['expert'])
        for dest,src in zip(out,update):
            dest[use]=src
    if any(not np.array_equal(a[~use],b[~use])for a,b in zip(out,result)):
        raise RuntimeError('Exact fallback changed')
    frame['actual_waveform_joint_BC']=np.where(use,np.nan,frame.actual_waveform_joint_BC)
    frame['physical_profile_pair_used']=use
    frame['physical_profile_distance2']=d2
    frame['physical_profile_variance_scale']=scale
    frame['physical_profile_OOD']=valid&~support
    return out


def calibrate():
    parents={(c['deployment'],c['seed']):c for c in cv.frozen_parent()if c['method']=='JOINTSTATE-GLOBAL'}
    configs=[];rows=[];units=[]
    for dep in n.DEPS:
        spec=metric(dep)
        for seed in n.SEEDS:
            old=h.isolated.ORIGINALS[dep,seed]
            frame=panel(dep,seed,'validation')
            cbase={**old,'method':METHODS[0]};ref=infer(frame,cbase)[0]
            fixed={**old,'method':METHODS[1],'fallback':parents[dep,seed]}
            archived=pd.read_parquet(PARENT/f'results/JOINTSTATE-GLOBAL/{dep}/seed_{seed}/validation/pairs.parquet')
            if not np.array_equal(infer(frame,fixed)[0],archived.waveform_score.to_numpy()):
                raise RuntimeError('Frozen fallback replay failed')
            configs.extend([cbase,fixed]);p=panels(dep,seed,spec)
            refs=((n.cf.fast_metrics(frame,n.cf.channels(frame,ref)@np.asarray(old['weights'])),n.cf.fast_metrics(frame,ref)),
                  (n.cf.fast_metrics(frame,frame.PATH875_final_score.to_numpy()),n.cf.fast_metrics(frame,frame.PATH875_waveform.to_numpy())))
            arms=[]
            for method,kind in zip(METHODS[2:4],('LINEAR','TREE')):
                expert=fit_classifier(dep,seed,kind,p)
                c={**deepcopy(fixed),'method':method,'metric':spec,'expert':expert}
                z=infer(frame,c)[0]
                fm=n.cf.fast_metrics(frame,n.cf.channels(frame,z)@np.asarray(old['weights']));wm=n.cf.fast_metrics(frame,z)
                c['tune_guard']=all(n.cf.guard(fm,fref)and n.cf.guard(wm,wref)for fref,wref in refs)
                c['tune_metrics']=fm
                poison=frame.copy();poison['pair_key']='unused';poison['pe_mc_bhattacharyya_coefficient']=-999.;poison['official_po_fpp']=1.
                if not np.array_equal(z,infer(poison,{**c,'alpha':-999.,'old_Mc_weight':999.})[0]):
                    raise RuntimeError('Forbidden field effect')
                configs.append(c);arms.append(c)
                rows.append({'deployment':dep,'seed':seed,'method':method,'guard':c['tune_guard'],
                    'physical_expert_fraction':float(frame.physical_profile_pair_used.mean()),**fm})
                units.append({'deployment':dep,'seed':seed,'method':method,'fallback_delta':0.,'PE_official_delta':0.})
            chosen=min(arms,key=lambda c:(c['expert']['selected']['logloss'],c['expert']['kind']!='LINEAR'))
            configs.append({**deepcopy(chosen),'method':METHODS[4]})
            print('PROFILE_PAIR_SELECTED',dep,seed,chosen['method'],chosen['tune_guard'],flush=True)
    n.write_csv(ROOT/'tables/VALIDATION_GUARDS.csv',rows)
    n.write_csv(ROOT/'audit/INVARIANCE.csv',units)
    n.write_json(ROOT/'configs/SELECTED_CONFIGURATIONS.json',configs)
    n.write_json(ROOT/'contracts/CONFIGURATIONS_FROZEN.json',{'UTC':n.utc(),
        'file':'configs/SELECTED_CONFIGURATIONS.json','sha256':n.sha(ROOT/'configs/SELECTED_CONFIGURATIONS.json'),
        'no_real_or_test_selection':True})


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--parent',type=Path,required=True)
    parser.add_argument('--data-root',type=Path,required=True);parser.add_argument('--predictive-root',type=Path,required=True)
    parser.add_argument('--stage',choices=('freeze','calibrate','evaluate','real'),required=True)
    args=parser.parse_args();ROOT,PARENT,DATA,PREDICTIVE=args.root,args.parent,args.data_root,args.predictive_root
    cv.ROOT,cv.PARENT=ROOT,PARENT
    base.s.ROOT=h.ROOT=co.ROOT=r.ROOT=co.score.ROOT=ROOT
    engine.ROOT=PARENT;minimal.ROOT=ROOT
    r.install();st.FIELDS=FIELDS;st.install_export()
    st.METHODS=(METHODS[0],'PARENT_GLOBAL','UNUSED_REJECT')
    co.score.matrices=co.matrices
    n.METHODS,n.load_panel,n.infer=METHODS,panel,infer
    if args.stage in ('freeze','calibrate'):
        globals()[args.stage]()
    else:
        n.run(ROOT,args.stage)
