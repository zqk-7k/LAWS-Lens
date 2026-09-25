#!/usr/bin/env python3
"""Two-dimensional simulated calibration of joint intrinsic compatibility."""
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
from scipy.optimize import minimize
from scipy.special import expit
from scipy.spatial import cKDTree

P=Path('/root/autodl-tmp/gw-catalog');sys.path.insert(0,str(P/'scripts/experiments'))
import mcwf_joint_intrinsics_evidence_20260908 as joint
import mcwf_joint_predictive_ensemble_v2_20260908 as ensemble
ev,t,dev,cf=joint.ev,joint.t,joint.dev,joint.cf
SINGLE=P/'results/mcwf_joint_intrinsic_evidence_exploratory_20260908T155830Z'
ENSEMBLE=P/'results/mcwf_joint_predictive_ensemble_exploratory_20260908T161608Z'
KINDS=('SINGLE-CHI','SINGLE-ETA-CHI','ENSEMBLE-CHI','ENSEMBLE-ETA-CHI')
RIDGE=(.001,.01,.1,1.,10.)


def initialize(root):
    if root.exists():raise RuntimeError('Fresh output required')
    for name in ('contracts','tables','calibration','evaluation','reports','figures','scripts','logs','manifest'):(root/name).mkdir(parents=True)
    dev.json_write(root/'contracts/ANALYSIS_CONTRACT.json',{
        'id':'MCWF-JOINT-2D-CALIBRATION-17','UTC':datetime.now(timezone.utc).isoformat(),'status':t.STATUS,
        'goal_achieved':False,'same_both_runs':True,'kinds':KINDS,
        'hypothesis':'Normalized similarity can reward broad predictions;prior-weighted overlap alone can overemphasize population density. Calibrate their correlated information together,not add two independent evidences.',
        'features':['log integral joint_p_i*joint_p_j/joint_training_prior','log joint_Bhattacharyya'],
        'classifier':'Class-balanced logistic regression with nonnegative slopes,L2 ridge,unpenalized intercept. Logit is a model-based approximate simulated density ratio,not a guaranteed physical lens Bayes factor.',
        'ridge_grid':RIDGE,'hyperparameter_choice':'Source/noise-disjoint development tuning half balanced logloss;tiespreferlargerridge. No injection-test or realcatalog metrics.',
        'source_units':'Same512development source systems as12;fit/tuninghalvesdefinedbyhashednoisebanks,mixedsourcesdiscarded;modeldevelopmentreuseexplicit,notfullyindependentcalibration.',
        'support':'10th-nearestfitpoint distance in fit-standardized2D space;capatmax(class-specificq99tuningdistance),positiveOODrewardzero. No realdatausedtosetsupport.',
        'tail':'Same independent-source jointBC finite-tail penalty as12;positive LR disallowed iftail<.05;massendpointOODneutral. Incrementclip[-4,4] inherited,notretuned.',
        'integration':'ADD toOMC. Innergamma/betasame12;FIXEDorPOSITIVE simplexouterweightsselectedonsimvalidation.',
        'priorities':['candidate:F50,F90,-AP,-R10,-R1','retrieval:-R10,-R1,-AP,F50,F90'],
        'guards':'Same perseed waveform+fusionretainedOMCguardrails;zeroadditionalincrementfallbackalwayspresent.',
        'frozen':['all neuralmodels','raw Z_time','raw Z_sky','scope','historicalresults','paper'],
        'changed_channels':[],'outer_weights_changed':True,
        'external':'Adaptive mechanismdevelopment;PE/official notinputs. If successfulrequiresfreshsource/noise confirmation. EnsemblecatalogseedSDnotindependentensembletrainingvariance.',
        'references':['https://arxiv.org/abs/1506.02169'],
        'reference_scope':'Discriminative simulation-based ratio-estimation principle only;not proof this low-dimensional logistic model is calibrated.'})
    rows=t.protected()
    for up in (SINGLE,ENSEMBLE):
        for p in (up/'cache').rglob('*.npz'):
            rows.append({'path':str(p),'sha256':dev.sha(p),'bytes':p.stat().st_size})
    dev.csv_write(root/'manifest/INPUT_SHA256.csv',pd.DataFrame(rows))
    shutil.copy2(__file__,root/'scripts/joint_2d_calibration.py')
    dev.json_write(root/'contracts/START_FREEZE.json',{'contract_sha256':dev.sha(root/'contracts/ANALYSIS_CONTRACT.json'),
        'code_sha256':dev.sha(Path(__file__))})


def features(dep,slot,seed,split,kind):
    origin,head=kind.split('-',1);up=SINGLE if origin=='SINGLE' else ENSEMBLE
    if origin=='SINGLE':name=f'{slot}_{seed}_{split}.npz'
    else:name=f'ensemble_{t.SEEDS[0] if split=="real" else seed}_{split}.npz'
    p=up/f'cache/CONDITIONAL-{head}/{dep}/{name}'
    if not p.exists():raise RuntimeError('Missing frozen joint-feature cache: '+str(p))
    return dict(np.load(p))


def calibration(root,dep,slot,kind):
    cache_slot=slot if kind.startswith('SINGLE') else t.MODEL_SLOTS[0]
    path=root/f'calibration/{kind}/{dep}/{cache_slot}.json'
    if path.exists():return json.loads(path.read_text())
    meta=pd.read_parquet(t.PREVIOUS/f'expanded_data/{dep}/validation/event_metadata.parquet')
    groups=meta.source_uid.to_numpy(str)
    banks=sorted(meta.noise_bank_index.unique(),key=lambda x:hashlib.sha256(f'202609850:{dep}:{x}'.encode()).hexdigest())
    fold=meta.noise_bank_index.map({x:k%2 for k,x in enumerate(banks)}).to_numpy(int)
    mixed={g for g in np.unique(groups) if len(np.unique(fold[groups==g]))>1};fold[np.isin(groups,list(mixed))]=-1
    matrices=features(dep,slot,0,'development',kind);parts={}
    for side in (0,1):
        ids=np.flatnonzero(fold==side);i,j=np.triu_indices(len(ids),1);i,j=ids[i],ids[j]
        y=groups[i]==groups[j]
        if y.sum()<30:raise RuntimeError('Insufficient independent source support')
        x=np.column_stack([matrices['joint_logbf'][i,j],matrices['joint_logbc'][i,j]])
        weights=np.where(y,.5/y.sum(),.5/(~y).sum())
        parts[side]=(x,y,weights,matrices['joint_bc'][i,j])
    x,y,w,bc=parts[0];mu=np.average(x,axis=0,weights=w);sd=np.sqrt(np.average((x-mu)**2,axis=0,weights=w)).clip(1e-8)
    fit=(x-mu)/sd;xx,yy,ww,_=parts[1];tune=(xx-mu)/sd
    rows=[]
    for ridge in RIDGE:
        def loss(theta):
            z=theta[0]+fit@theta[1:]
            residual=w*(expit(z)-y)
            value=np.dot(w,np.logaddexp(0,z)-y*z)+.5*ridge*np.dot(theta[1:],theta[1:])
            gradient=np.r_[residual.sum(),fit.T@residual+ridge*theta[1:]]
            return value,gradient
        opt=minimize(loss,np.zeros(3),jac=True,method='L-BFGS-B',bounds=[(None,None),(0,None),(0,None)],
            options={'maxiter':2000,'ftol':1e-12,'gtol':1e-8})
        if not opt.success:raise RuntimeError('Calibration optimizer failed: '+str(opt.message))
        logits=opt.x[0]+tune@opt.x[1:]
        nll=float(np.dot(ww,np.logaddexp(0,logits)-yy*logits))
        rows.append({'ridge':ridge,'theta':opt.x.tolist(),'tuning_balanced_NLL':nll,'fit_loss':float(opt.fun),'converged':bool(opt.success)})
    chosen=min(rows,key=lambda r:(r['tuning_balanced_NLL'],-r['ridge']))
    distance=cKDTree(fit).query(tune,k=10)[0][:,-1]
    radius=max(float(np.quantile(distance[yy],.99)),float(np.quantile(distance[~yy],.99)))
    fit_path=root/f'calibration/{kind}/{dep}/{cache_slot}_fit_points.npz';fit_path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(fit_path,points=fit.astype(np.float64))
    spec={'deployment':dep,'slot':cache_slot,'kind':kind,'mu':mu.tolist(),'sd':sd.tolist(),**chosen,
        'ridge_grid_results':rows,'fit_points':str(fit_path.relative_to(root)),'fit_points_sha256':dev.sha(fit_path),
        'support_radius':radius,'joint_reference':np.sort(-np.log(bc[y])).tolist(),'independent_positive_sources':int(y.sum()),
        'tuning_positive_sources':int(yy.sum()),'discarded_cross_noise_sources':len(mixed),'model_development_reuse':True,
        'fit_noise_banks':[int(z) for z in np.unique(meta.noise_bank_index[fold==0])],
        'tuning_noise_banks':[int(z) for z in np.unique(meta.noise_bank_index[fold==1])],
        'tuning_positive_OOD':float((distance[yy]>radius).mean()),'tuning_null_OOD':float((distance[~yy]>radius).mean())}
    dev.json_write(path,spec);return spec


def values(root,c,split):
    f=cf.read(c['deployment'],c['seed'],split);a=features(c['deployment'],c['slot'],c['seed'],split,c['kind']);spec=c['calibration']
    i,j=f.idx_i.to_numpy(int),f.idx_j.to_numpy(int)
    x=np.column_stack([a['joint_logbf'][i,j],a['joint_logbc'][i,j]])
    if not np.isfinite(x).all():raise RuntimeError('Invalid strict feature')
    x=(x-np.asarray(spec['mu']))/np.asarray(spec['sd'])
    path=root/spec['fit_points']
    if dev.sha(path)!=spec['fit_points_sha256']:raise RuntimeError('Support points changed')
    distance=cKDTree(np.load(path)['points']).query(x,k=10)[0][:,-1]
    endpoint=(a['outside'][i]>.25)|(a['outside'][j]>.25)
    outside=endpoint|(distance>spec['support_radius'])
    tail=ev.tail.tail_probability(-a['joint_logbc'][i,j],spec['joint_reference'])
    theta=np.asarray(spec['theta']);increment=theta[0]+x@theta[1:]
    penalty=np.minimum(np.log(tail/.05),0.)
    increment=np.where(outside|(tail<.05),np.minimum(increment,0.),increment).clip(-4,4)
    penalty,increment=np.where(endpoint,0.,penalty),np.where(endpoint,0.,increment)
    return f,penalty,increment,{'joint_logbf':a['joint_logbf'][i,j],'joint_BC':a['joint_bc'][i,j],
        'joint_support_distance':distance,'joint_ood':outside,'joint_tail_probability':tail,'penalty':penalty,'increment':increment}


def select(root):
    if (root/'contracts/INTEGRATION_FROZEN.json').exists():raise RuntimeError('Already frozen')
    configs,innergrid,outergrid=[],[],[]
    grid=np.asarray([(i/20,j/20,(20-i-j)/20) for i in range(1,20) for j in range(1,20-i)])
    def key(r,policy):
        return ((r['false_at_recall_0p5'],r['false_at_recall_0p9'],-r['average_precision'],-r['macro_r_at_10'],-r['macro_r_at_1'])
            if policy=='CANDIDATE' else (-r['macro_r_at_10'],-r['macro_r_at_1'],-r['average_precision'],r['false_at_recall_0p5'],r['false_at_recall_0p9']))
    for kind in KINDS:
        for dep in t.DEPS:
            for slot,seed in zip(t.MODEL_SLOTS,t.SEEDS):
                spec=calibration(root,dep,slot,kind);common={'kind':kind,'deployment':dep,'slot':slot,'seed':seed,'calibration':spec}
                f,pen,inc,_=values(root,common,'validation');old=f.waveform_score.to_numpy(float);weights=cf.frozen_weights(dep,seed);norm=weights/weights.sum()
                bm,bwm=cf.fast_metrics(f,cf.channels(f,old)@weights),cf.fast_metrics(f,old)
                rows=[{'gamma':0.,'beta':0.,'unchanged_OMC':True,'pass':True,**bm}]
                for gamma in joint.GAMMA:
                    for beta in joint.BETA:
                        z=old+gamma*pen+beta*inc;m,wm=cf.fast_metrics(f,cf.channels(f,z)@weights),cf.fast_metrics(f,z)
                        row={'gamma':gamma,'beta':beta,'unchanged_OMC':False,'pass':cf.guard(m,bm) and cf.guard(wm,bwm),**m}
                        rows.append(row);innergrid.append({**{k:common[k] for k in ('kind','deployment','seed')},**row})
                for policy in ('CANDIDATE','RETRIEVAL'):
                    chosen=min((r for r in rows if r['pass']),key=lambda r:(*key(r,policy),not r['unchanged_OMC'],r['gamma']**2+r['beta']**2,r['gamma'],r['beta']))
                    c={**common,'method':'LR2D-'+kind+'-FIXED-'+policy,'gamma':chosen['gamma'],'beta':chosen['beta'],
                        'unchanged_OMC':chosen['unchanged_OMC'],'weights':weights.tolist()};configs.append(c)
                    z=old if c['unchanged_OMC'] else old+c['gamma']*pen+c['beta']*inc;wm=cf.fast_metrics(f,z)
                    points=np.vstack([grid,norm]) if not np.any(np.all(np.isclose(grid,norm,atol=1e-12),axis=1)) else grid
                    choices=[]
                    for w in points:
                        m=cf.fast_metrics(f,cf.channels(f,z)@w);row={'weights':w.tolist(),'pass':cf.guard(m,bm) and cf.guard(wm,bwm),**m}
                        choices.append(row);outergrid.append({'kind':kind,'deployment':dep,'seed':seed,'inner_policy':policy,**row})
                    for outer in ('CANDIDATE','RETRIEVAL'):
                        chosen2=min((r for r in choices if r['pass']),key=lambda r:(*key(r,outer),float(np.square(np.asarray(r['weights'])-norm).sum()),*r['weights']))
                        configs.append({**c,'method':'LR2D-'+kind+'-REFIT-'+policy+'-'+outer,'weights':chosen2['weights']})
    dev.csv_write(root/'tables/VALIDATION_INNER_GRID.csv',pd.DataFrame(innergrid));dev.csv_write(root/'tables/VALIDATION_OUTER_GRID.csv',pd.DataFrame(outergrid))
    dev.csv_write(root/'tables/SELECTED_COEFFICIENTS.csv',pd.DataFrame([{**{k:v for k,v in c.items() if k not in ('weights','calibration')},
        **dict(zip(('lambda_waveform','lambda_time','lambda_sky'),c['weights']))} for c in configs]))
    dev.json_write(root/'calibration/SELECTED.json',configs)
    dev.json_write(root/'contracts/INTEGRATION_FROZEN.json',{'file':'calibration/SELECTED.json','sha256':dev.sha(root/'calibration/SELECTED.json'),
        'UTC':datetime.now(timezone.utc).isoformat(),'real_or_test_selection':False})


def scored(root,c,split):
    if c['method']=='OMC' or c.get('unchanged_OMC',False):
        f=cf.read(c['deployment'],c['seed'],split);return f,f.waveform_score.to_numpy(float),{}
    f,pen,inc,a=values(root,c,split);return f,f.waveform_score.to_numpy(float)+c['gamma']*pen+c['beta']*inc,a


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--stage',required=True,
        choices=['initialize','select','evaluate','real','assess']);a=p.parse_args();ev.scored=scored;ensemble.scored=scored
    if a.stage in ('initialize','select'):globals()[a.stage](a.root)
    elif a.stage=='evaluate':ensemble.evaluate(a.root)
    else:getattr(ev,a.stage)(a.root)
