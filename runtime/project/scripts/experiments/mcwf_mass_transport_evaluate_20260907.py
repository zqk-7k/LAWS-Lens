#!/usr/bin/env python3
"""Simulation-calibrated log-mass transport distance, never real PE inputs."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='2'
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from sklearn.isotonic import IsotonicRegression
import mcwf_pe_frontend_extension_v2_20260907 as e
import mcwf_mass_tf_20260905 as tf
import mcwf_finite_reference_tail_20260907 as tail
import mcwf_large_population_encoder_20260907 as large
import mcwf_training_recipe_ensemble_20260907 as ensemble

dev,body,ev=e.dev,e.body,e.ev
MODE='large'
GAMMAS=(0.,.125,.25,.5,1.,2.,4.)
BETAS=(0.,.0625,.125,.25,.5,1.,2.)


def pair_features(p,i,j):
    p=np.asarray(p,float)
    cdf=p.cumsum(-1)
    mean=p@tf.LOG_CENTERS
    w=np.empty(len(i));gap=np.empty(len(i))
    for start in range(0,len(i),4096):
        sl=slice(start,start+4096)
        w[sl]=(abs(cdf[i[sl],:-1]-cdf[j[sl],:-1])*np.diff(tf.LOG_CENTERS)).sum(1)
        gap[sl]=abs(mean[i[sl]]-mean[j[sl]])
    return {'W1':w,'MEAN':gap}


def tests():
    rng=np.random.default_rng(202609351)
    p=rng.dirichlet(np.ones(64),size=20)
    i,j=np.triu_indices(len(p),1)
    x=pair_features(p,i,j)
    expected=np.array([wasserstein_distance(tf.LOG_CENTERS,tf.LOG_CENTERS,p[a],p[b]) for a,b in zip(i,j)])
    error=float(abs(x['W1']-expected).max())
    if error>1e-12:raise RuntimeError('W1 disagrees with SciPy')
    same=pair_features(p,np.arange(len(p)),np.arange(len(p)))['W1']
    swapped=pair_features(p,j,i)['W1']
    if np.max(same)!=0 or not np.array_equal(swapped,x['W1']):raise RuntimeError('Transport symmetry/self-distance failure')
    if (x['W1']+1e-12<x['MEAN']).any():raise RuntimeError('W1 must bound the difference of means')
    return {'pass':True,'scipy_max_absolute_error':error,'pairs':len(i),
            'symmetry_exact':True,'self_distance_zero':True,'W1_ge_mean_gap':True}


def prediction(root,dep,ms,es,split):
    if split=='development':
        if MODE=='frozen':
            p=np.load(root/f'expanded_calibration/development/{dep}/seed_{es}/frozen_RNC_predictions.npz')['p']
            meta=pd.read_parquet(root/f'expanded_data/{dep}/validation/event_metadata.parquet')
            group=meta.source_uid.to_numpy()
        else:
            name='large_population_encoder' if MODE=='large' else 'training_recipe_ensemble'
            if MODE=='ensemble':ensemble.prepare(root,dep,ms)
            a=np.load(root/f'{name}/models/{dep}/seed_{ms}/validation_predictions.npz')
            p,group=a['p'],a['group']
        return p,group
    if MODE=='frozen':a=e.predictions(dep,ms,es,split)
    else:a=(large if MODE=='large' else ensemble).prediction(root,dep,ms,es,split)
    return a['p'],None


def values(root,dep,ms,es,split):
    p,group=prediction(root,dep,ms,es,split)
    if split=='development':
        i,j=np.triu_indices(len(p),1)
        f=pd.DataFrame({'idx_i':i,'idx_j':j,'is_true_pair':group[i]==group[j]})
    else:
        f=pd.read_parquet(e.BASE/f'evaluation/{dep}/seed_{es}/{split}_pairs.parquet')
        i,j=f.idx_i.to_numpy(int),f.idx_j.to_numpy(int)
    x=pair_features(p,i,j)
    if not all(np.isfinite(v).all() for v in x.values()):raise RuntimeError('Invalid mass-transport features')
    return f,x


def calibrate(root,dep,ms,es):
    f,x=values(root,dep,ms,es,'development')
    y=f.is_true_pair.to_numpy(bool);weight=np.where(y,.5/y.sum(),.5/(~y).sum())
    result={}
    for arm,distance in x.items():
        iso=IsotonicRegression(increasing=True,out_of_bounds='clip').fit(-distance,y,sample_weight=weight)
        p=iso.y_thresholds_.clip(1/(y.sum()+2),1-1/(y.sum()+2))
        result[arm]={'reference':np.sort(distance[y]).tolist(),'knots':iso.X_thresholds_.tolist(),
            'loglr':(np.log(p)-np.log1p(-p)).tolist(),'minimum':float((-distance).min()),
            'maximum':float((-distance).max()),'independent_source_pairs':int(y.sum()),'noise_blocks':32}
    return result


def score(f,x,spec,arm):
    distance=x[arm]
    p=tail.tail_probability(distance,spec['reference'])
    penalty=np.minimum(np.log(p/.05),0.)
    increment=np.interp(-distance,spec['knots'],spec['loglr']).clip(-4,4)
    ood=(-distance<spec['minimum'])|(-distance>spec['maximum'])
    increment=np.where(ood|(p<.05),np.minimum(increment,0),increment)
    return f.waveform_score.to_numpy(float)+spec['gamma']*penalty+spec['beta']*increment,penalty,increment


def run(root,arm):
    trial=root/f'trials/MASS-TRANSPORT-{MODE.upper()}-{arm}'
    if trial.exists():raise RuntimeError('Independent trial required')
    for name in ('contracts','calibration','tables','evaluation','results'):(trial/name).mkdir(parents=True)
    shutil.copy2(__file__,trial/'contracts'/Path(__file__).name)
    dev.json_write(trial/'contracts/RECIPE.json',{
        'created_utc':datetime.now(timezone.utc).isoformat(),'same_O3_O4a':True,'predictor':MODE,'arm':arm,
        'input':'unchanged2s40-580Hz waveform-derived logMc predictive mass distribution,not public PE',
        'W1_formula':'sum_k |CDF_i[k]-CDF_j[k]|*(logMc_center[k+1]-logMc_center[k]); exact for64point discrete predictive measures',
        'MEAN_control':'absolute difference of predicted logMc means; no division by predictive uncertainty',
        'rationale':'test location/shape discrepancy on an absolute logmass scale instead of only normalized overlap; broad uncertain agreement must be learned from simulation,not presumed',
        'formula':'Z_FRT+gamma*finite_companion_distance_tail+beta*bounded_isotonic_distance_LR',
        'gamma_grid':GAMMAS,'beta_grid':BETAS,'calibration':'512 simulation source pairs/32noiseblocks;dependent nullpairs,balanced classes',
        'selection':'original simulation validation guards and F50/F90/AP/R10 objective,three paired seeds',
        'real_PE_official_ID_inputs':False,'real_used_to_select':False,
        'real_outcomes':'adaptive-development audit,not blind confirmation',
        'frozen':['time','sky','outerweights','scope','historicalresults'],'fresh_confirmation_required':True,
        'limits':'transport distance is not a posterior-overlap Bayes factor; learned distributions are not fullPE; systematic prediction bias remains possible',
        'references':['https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wasserstein_distance.html'],
        'numerical_tests':tests()})
    choices,cache,states={},{},[]
    for dep in e.DEPS:
        for ms,es in zip(body.MODEL_SEEDS,dev.SEEDS):
            cal=calibrate(root,dep,ms,es)[arm]
            f,x=values(root,dep,ms,es,'validation');cache[dep,es]=f,x
            baseline=ev.metrics(f,f.waveform_score.to_numpy(float),dep,es)
            candidates=[];rows=[]
            for gamma in GAMMAS:
                for beta in BETAS:
                    spec={**cal,'gamma':gamma,'beta':beta}
                    z,_,_=score(f,x,spec,arm);m=ev.metrics(f,z,dep,es);ok=ev.guard(m,baseline)
                    rows.append({'gamma':gamma,'beta':beta,'pass':ok,**{a+'_'+k:v for a,b in m.items() for k,v in b.items()}})
                    if ok:candidates.append((e.old_selection.objective(m,gamma,beta),spec))
            selected=min(candidates,key=lambda a:a[0])[1];choices[dep,es]=selected
            out=trial/f'calibration/{dep}/seed_{es}';out.mkdir(parents=True)
            dev.csv_write(out/'validation_grid.csv',pd.DataFrame(rows));dev.json_write(out/'SELECTED_CONFIG.json',selected)
            states.append({'deployment':dep,'seed':es,'gamma':selected['gamma'],'beta':selected['beta']})
    dev.json_write(trial/'contracts/SELECTED_FREEZE.json',{'configs':states,'real_used_to_select':False})
    print(json.dumps({'mass_transport_selected':trial.name,'configs':states}),flush=True)
    metrics,guards=[],[]
    for dep in e.DEPS:
        real={}
        for ms,es in zip(body.MODEL_SEEDS,dev.SEEDS):
            for split in ('validation','test','real'):
                f,x=cache[dep,es] if split=='validation' else values(root,dep,ms,es,split)
                z,penalty,increment=score(f,x,choices[dep,es],arm)
                n=f.rename(columns={c:'baseline_'+c for c in ('final_score','rank','method','waveform_contribution','time_contribution','sky_contribution') if c in f}).copy()
                n['FRT_baseline_waveform_score'],n['waveform_score']=f.waveform_score,z
                n['logmass_W1'],n['logmass_mean_gap']=x['W1'],x['MEAN']
                n['transport_tail_penalty'],n['transport_LR']=penalty,increment
                out=trial/f'evaluation/{dep}/seed_{es}';out.mkdir(parents=True,exist_ok=True);n.to_parquet(out/f'{split}_pairs.parquet',index=False)
                if split=='real':real[es]=n
                else:
                    bm=ev.metrics(f,f.waveform_score.to_numpy(float),dep,es);m=ev.metrics(f,z,dep,es)
                    guards.append({'deployment':dep,'seed':es,'split':split,'pass':ev.guard(m,bm)})
                    for method in m:
                        for config,v in [('FRT_BASELINE',bm[method]),('CANDIDATE',m[method])]:metrics.append({'deployment':dep,'seed':es,'split':split,'method':method,'config':config,**v})
        dev.save_evaluation(trial,'CANDIDATE',dep,real,pd.read_parquet(root/f'audit/{dep}_frozen_external_reference.parquet'))
    dev.csv_write(trial/'tables/RETRIEVAL_PER_SEED.csv',pd.DataFrame(metrics));dev.csv_write(trial/'tables/REUSED_GUARDRAILS.csv',pd.DataFrame(guards))
    e.assess(root,trial)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    for MODE in ('frozen','large','ensemble'):
        for arm in ('W1','MEAN'):run(a.root,arm)
