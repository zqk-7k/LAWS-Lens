#!/usr/bin/env python3
"""A single three-feature expert where both physical profiles are valid."""
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
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.ensemble import HistGradientBoostingClassifier

P=Path('/root/autodl-tmp/gw-catalog')
sys.path.insert(0,str(P/'scripts/experiments'))
import mcwf_confidence_level_validation_20260909 as cv
import mcwf_minimal_joint_waveform_classifier_20260909 as minimal
st,ctrl,engine,base,n,r,co,h=(getattr(cv,k)for k in ('st','ctrl','engine','base','n','r','co','h'))
ROOT=PARENT=DATA=None
METHODS=('NODUP-DIRECT-REPLAY','JOINTSTATE-FROZEN95','PAIRED3-LINEAR','PAIRED3-TREE','PAIRED3-VALIDATION-SELECTED')
FEATURES=('embedding_only','profile_log_mass_BC','profile_log_conditional_BC')
FIELDS=st.FIELDS+('paired3_expert_used','paired3_log_mass_BC','paired3_log_conditional_BC')


def freeze():
    if ROOT.exists():
        raise RuntimeError('Independent output required')
    cv.frozen_parent()
    for folder in ('contracts','configs','calibration','tables','audit','reports','scripts','logs','manifest','results','figures'):
        (ROOT/folder).mkdir(parents=True)
    n.write_json(ROOT/'contracts/ANALYSIS_CONTRACT.json',{'UTC':n.utc(),
        'id':'MCWF-NODUP-PAIRED-INTRINSIC-EXPERT-46','status':n.STATUS,
        'parent':str(PARENT),'data':str(DATA),'features':list(FEATURES),
        'reason':'JointBC alone compresses well-resolved mass mismatch and weak conditional-spin agreement. Preserve marginal/conditional decomposition in ONEclassifier,only when BOTH profiles are quality-valid.',
        'identity':'jointBC=massBC*mass-overlap-weighted conditionalBC;these are correlated features,not independent multiplied evidences.',
        'score':'On state2 replace WHOLE waveform score by one classbalanced logit+fitstateoffset. No addedcosLR,massLR,oldMc/q heads or total-score mixture.',
        'fallback':'States0/1 exactly retain frozenR35 JOINTSTATE-GLOBAL,including its own conditional state calibration. Same availability rule inO3/O4a.',
        'fit':'Read-onlyR35 updated predictions on newsource/noise-disjointfit/tunefolds;eachsourcepairtotalweight1,balancedclasses,exclude identical-noisebanknulls;>=20truessystemsinfiteachtunestate.',
        'grid':{'ridge':[.001,.01,.1,1.],'leaves':[3,7],'iterations':100,'learning_rate':.05},
        'monotonic':[1,1,1],'classification':'Linear and monotone tree both reported;selecttypebytuneproperbalancedlogloss,linear tie. No ranking/PE/officialtypechoice.',
        'support':'Exact frozen fitfeaturebox;positiveOOD0;caplog(allfittrue+1). State2 validprofileflag is fixedR10,not chosenbyrealPE.',
        'fixed':['allpredictors','profileinputs','R29predictivedensities','time','sky','outerweights','scope','historicalresults'],
        'no_old_encoder_Mc_q':True,'no_total_blend':True,'same_framework_both_runs':True,
        'adaptive_development':True,'not_blind_confirmation':True,'goal_achieved':False,
        'references':['https://arxiv.org/abs/gr-qc/9402014','https://arxiv.org/abs/1506.02169'],
        'interpretation':'Empirical waveformranking proxy,not a physicalPEposterior or lensingBayesfactor.'})
    paths=[Path(__file__),Path(cv.__file__),Path(minimal.__file__),
           PARENT/'configs/SELECTED_CONFIGURATIONS.json',PARENT/'configs/SELECTED_PREDICTIVE_KIND.json']
    for dep in n.DEPS:
        paths.extend([DATA/f'data/{dep}/event_metadata.parquet',DATA/f'data/{dep}/noise/noise_manifest.csv'])
        paths.extend(PARENT/f'cache/{dep}_{seed}_development.npz'for seed in n.SEEDS)
    n.write_csv(ROOT/'manifest/INPUT_SHA256.csv',[{'path':str(p),'sha256':n.sha(p),'bytes':p.stat().st_size}for p in paths])
    shutil.copy2(__file__,ROOT/'scripts/paired_intrinsic_expert.py')
    n.write_json(ROOT/'contracts/START_FREEZE.json',{'UTC':n.utc(),
        'contract_sha256':n.sha(ROOT/'contracts/ANALYSIS_CONTRACT.json'),'script_sha256':n.sha(Path(__file__))})


def population(dep,seed):
    key=dep,seed
    if key not in base.MEMO:
        base.MEMO[key]=dict(np.load(PARENT/f'cache/{dep}_{seed}_development.npz'))
    return base.MEMO[key]


def panels(dep,seed):
    pop=population(dep,seed)
    meta=pd.read_parquet(DATA/f'data/{dep}/event_metadata.parquet')
    noise=pd.read_csv(DATA/f'data/{dep}/noise/noise_manifest.csv').set_index('bank_index')
    chunk=meta.noise_bank_index.map(noise.parent_file_gps)
    if set(chunk[meta.fold==0])&set(chunk[meta.fold==1]):
        raise RuntimeError('Noise-parent leakage')
    return {side:(x[:,:3],y,w,pop['active'][i].astype(int)+pop['active'][j].astype(int))
            for side,(x,y,w,_,i,j)in base.panels(dep,seed).items()}


def linear(x,y,w,ridge):
    mu=w@x;sd=np.sqrt(w@((x-mu)**2)).clip(1e-6);z=(x-mu)/sd
    def objective(theta):
        logits=theta[0]+z@theta[1:];err=w*(expit(logits)-y)
        return float(w@(np.logaddexp(0.,logits)-y*logits)+.5*ridge*(theta[1:]@theta[1:])),np.r_[err.sum(),z.T@err+ridge*theta[1:]]
    result=minimize(objective,np.zeros(4),jac=True,method='L-BFGS-B',
        bounds=[(None,None)]+[(0.,None)]*3,options={'maxiter':2000,'ftol':1e-12,'gtol':1e-8})
    if not result.success:
        raise RuntimeError('Classifier optimizer failure:'+str(result.message))
    return {'kind':'LINEAR','mu':mu,'sd':sd,'theta':result.x}


def fit_one(dep,seed,kind,p):
    xx,yy,ww,ss=p[0];vx,vy,vw,vs=p[1]
    mask=ss==2;vm=vs==2
    x,y,w=xx[mask],yy[mask],ww[mask].copy();test,label,tw=vx[vm],vy[vm],vw[vm].copy()
    if min(y.sum(),label.sum())<20:
        raise RuntimeError('HOLD_INSUFFICIENT_TRUE_SOURCE_SUPPORT')
    offset=float(np.log(ww[mask&yy].sum()/ww[yy].sum())-np.log(ww[mask&~yy].sum()/ww[~yy].sum()))
    w[y]*=.5/w[y].sum();w[~y]*=.5/w[~y].sum();tw[label]*=.5/tw[label].sum();tw[~label]*=.5/tw[~label].sum()
    trials=[];models=[]
    for ridge in (.001,.01,.1,1.):
        for leaves in ((0,)if kind=='LINEAR'else(3,7)):
            if kind=='LINEAR':
                model=linear(x,y,w,ridge)
            else:
                estimator=HistGradientBoostingClassifier(max_iter=100,learning_rate=.05,max_leaf_nodes=leaves,
                    min_samples_leaf=20,l2_regularization=ridge,early_stopping=False,
                    monotonic_cst=[1,1,1],random_state=2026090946)
                estimator.fit(x,y,sample_weight=w*len(x));model={'kind':'TREE','estimator':estimator,'leaves':leaves}
            z=r.raw_predict(model,test)
            loss=float(tw@(np.logaddexp(0.,z)-label*z))
            trials.append({'ridge':ridge,'leaves':leaves,'logloss':loss});models.append(model)
    win=min(range(len(trials)),key=lambda k:(trials[k]['logloss'],trials[k]['leaves'],-trials[k]['ridge']))
    path=ROOT/f'calibration/{dep}_{seed}_{kind}.pkl'
    with path.open('xb')as file:
        pickle.dump(models[win],file)
    n.write_csv(path.with_suffix('.GRID.csv'),trials)
    return {'file':str(path.relative_to(ROOT)),'sha256':n.sha(path),'kind':kind,
        'minimum':x.min(0).tolist(),'maximum':x.max(0).tolist(),'partition_log_ratio':offset,
        'cap':float(np.log(yy.sum()+1)),'fit_sources':int(y.sum()),'tune_sources':int(label.sum()),
        'selected':trials[win]}


def infer(frame,config):
    if config['method']==METHODS[0]:
        return st.infer(frame,config)
    result=st.infer(frame,{**config['fallback'],'method':'PARENT_GLOBAL'})
    if config['method']==METHODS[1]:
        frame['paired3_expert_used']=False
        return result
    used=st.states(frame)==2
    out=tuple(a.copy()for a in result)
    if used.any():
        x=frame.loc[used,list(FEATURES)].to_numpy(float)
        for a,b in zip(out,minimal.apply(x,config['expert'])):
            a[used]=b
    if any(not np.array_equal(a[~used],b[~used])for a,b in zip(out,result)):
        raise RuntimeError('Exact fallback changed')
    frame['paired3_expert_used']=used
    frame['paired3_log_mass_BC']=frame.profile_log_mass_BC
    frame['paired3_log_conditional_BC']=frame.profile_log_conditional_BC
    return out


def calibrate():
    parents={(c['deployment'],c['seed']):c for c in cv.frozen_parent()if c['method']=='JOINTSTATE-GLOBAL'}
    configs=[];rows=[];units=[]
    for dep in n.DEPS:
        for seed in n.SEEDS:
            old=h.isolated.ORIGINALS[dep,seed];frame=cv.panel(dep,seed,'validation')
            cbase={**old,'method':METHODS[0]};ref=infer(frame,cbase)[0]
            fixed={**old,'method':METHODS[1],'fallback':parents[dep,seed]}
            expected=pd.read_parquet(PARENT/f'results/JOINTSTATE-GLOBAL/{dep}/seed_{seed}/validation/pairs.parquet')
            if not np.array_equal(infer(frame,fixed)[0],expected.waveform_score.to_numpy()):
                raise RuntimeError('R35 replay changed')
            configs.extend([cbase,fixed]);p=panels(dep,seed)
            refs=((n.cf.fast_metrics(frame,n.cf.channels(frame,ref)@np.asarray(old['weights'])),n.cf.fast_metrics(frame,ref)),
                  (n.cf.fast_metrics(frame,frame.PATH875_final_score.to_numpy()),n.cf.fast_metrics(frame,frame.PATH875_waveform.to_numpy())))
            arms=[]
            for method,kind in zip(METHODS[2:4],('LINEAR','TREE')):
                expert=fit_one(dep,seed,kind,p)
                c={**deepcopy(fixed),'method':method,'expert':expert}
                z=infer(frame,c)[0];fm=n.cf.fast_metrics(frame,n.cf.channels(frame,z)@np.asarray(old['weights']));wm=n.cf.fast_metrics(frame,z)
                c['tune_guard']=all(n.cf.guard(fm,fr)and n.cf.guard(wm,wr)for fr,wr in refs);c['tune_metrics']=fm
                poison=frame.copy();poison['pair_key']='unused';poison['pe_mc_bhattacharyya_coefficient']=-999.;poison['official_po_fpp']=1.
                if not np.array_equal(z,infer(poison,{**c,'alpha':-999.,'old_Mc_weight':999.})[0]):
                    raise RuntimeError('Forbidden input effect')
                if not np.array_equal(z,infer(frame.iloc[::-1].copy(),c)[0][::-1]):
                    raise RuntimeError('Row-order dependence')
                arms.append(c);configs.append(c)
                rows.append({'deployment':dep,'seed':seed,'method':method,'guard':c['tune_guard'],**fm})
                units.append({'deployment':dep,'seed':seed,'method':method,'fallback_delta':0.,'PE_official_delta':0.,'row_order_delta':0.})
            chosen=min(arms,key=lambda c:(c['expert']['selected']['logloss'],c['expert']['kind']!='LINEAR'))
            configs.append({**deepcopy(chosen),'method':METHODS[4]})
            base.MEMO.clear()
            print('PAIRED3_SELECTED',dep,seed,chosen['method'],chosen['tune_guard'],flush=True)
    n.write_csv(ROOT/'tables/VALIDATION_GUARDS.csv',rows);n.write_csv(ROOT/'audit/INVARIANCE.csv',units)
    n.write_json(ROOT/'configs/SELECTED_CONFIGURATIONS.json',configs)
    n.write_json(ROOT/'contracts/CONFIGURATIONS_FROZEN.json',{'UTC':n.utc(),
        'file':'configs/SELECTED_CONFIGURATIONS.json','sha256':n.sha(ROOT/'configs/SELECTED_CONFIGURATIONS.json'),
        'no_real_or_test_selection':True})


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for arg in ('root','parent','data-root'):
        parser.add_argument('--'+arg,type=Path,required=True)
    parser.add_argument('--stage',choices=('freeze','calibrate','evaluate','real'),required=True)
    args=parser.parse_args();ROOT,PARENT,DATA=args.root,args.parent,args.data_root
    cv.ROOT,cv.PARENT=ROOT,PARENT
    base.s.ROOT=h.ROOT=co.ROOT=r.ROOT=co.score.ROOT=ROOT
    engine.ROOT=PARENT;minimal.ROOT=ROOT;base.ROOT=ROOT;base.DATA=DATA;base.population=population
    r.install();st.FIELDS=FIELDS;st.install_export()
    st.METHODS=(METHODS[0],'PARENT_GLOBAL','UNUSED_REJECT')
    co.score.matrices=co.matrices
    n.METHODS,n.load_panel,n.infer=METHODS,cv.panel,infer
    if args.stage in ('freeze','calibrate'):
        globals()[args.stage]()
    else:
        n.run(ROOT,args.stage)
