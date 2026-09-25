#!/usr/bin/env python3
"""Negative-only mass consistency from learned continuous predictive densities."""
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
import torch
from scipy.special import ndtr
import mcwf_pe_frontend_extension_v2_20260907 as e
import mcwf_mixture_density_20260907 as mdn
import mcwf_joint_source_density_20260907 as source
import mcwf_finite_reference_tail_20260907 as tail

dev,body,ev=e.dev,e.body,e.ev
GAMMAS=(0.,.125,.25,.5,1.,2.,4.)


def marginal(a,ck):
    w=np.array(a['w'],float);w/=w.sum(1,keepdims=True)
    m=np.asarray(a['m'],float)[...,0]*ck['ys'][0]+ck['ym'][0]
    var=np.asarray(a['cov'],float)[...,0,0]*ck['ys'][0]**2
    mean=(w*m).sum(1);variance=(w*(var+(m-mean[:,None])**2)).sum(1)
    edges=np.r_[-np.inf,np.linspace(np.log(3),np.log(300),513),np.inf]
    cdf=(w[...,None]*ndtr((edges[None,None,:]-m[...,None])/np.sqrt(var[...,None]))).sum(1)
    p=np.diff(cdf,axis=1).clip(0);p/=p.sum(1,keepdims=True)
    good=np.isfinite(w).all(1)
    if not np.allclose(p[good].sum(1),1,atol=1e-12):raise RuntimeError('Massprobabilitynormalization')
    return p,mean,variance


def values(root,dep,ms,es,split,family):
    namespace='mixture_density' if family=='phase' else 'joint_source_density'
    cp=root/f'{namespace}/models/{dep}/seed_{ms}/validation_selected_model.pt'
    ck=torch.load(cp,weights_only=False,map_location='cpu')
    if split=='development':
        a=np.load(cp.parent/'validation_predictions.npz');i,j=np.triu_indices(len(a['w']),1)
        f=pd.DataFrame({'idx_i':i,'idx_j':j,'is_true_pair':a['group'][i]==a['group'][j]})
    else:
        f=pd.read_parquet(e.BASE/f'evaluation/{dep}/seed_{es}/{split}_pairs.parquet')
        a=(mdn.prediction if family=='phase' else source.prediction)(root,dep,ms,es,split)
        i,j=f.idx_i.to_numpy(int),f.idx_j.to_numpy(int)
    p,m,v=marginal(a,ck)
    bc=np.sqrt(p[i]*p[j]).sum(1).clip(1e-15,1)
    return f,{'BC':-np.log(bc),'D':abs(m[i]-m[j])/np.sqrt(v[i]+v[j]).clip(1e-12)}


def score(f,x,spec):
    probability=tail.tail_probability(x[spec['statistic']],spec['reference'])
    penalty=np.minimum(np.log(probability/.05),0.)
    return f.waveform_score.to_numpy(float)+spec['gamma']*penalty,penalty,probability


def run(root,family,statistic):
    code=f'CONTINUOUS-MC-GUARD-{family.upper()}-{statistic}'
    trial=root/'trials'/code
    if trial.exists():raise RuntimeError('Independenttrialrequired')
    for name in ('contracts','calibration','tables','evaluation','results'):(trial/name).mkdir(parents=True)
    shutil.copy2(__file__,trial/'contracts'/Path(__file__).name)
    dev.json_write(trial/'contracts/RECIPE.json',{'created_utc':datetime.now(timezone.utc).isoformat(),
        'same_global_recipe_O3_O4a':[family,statistic],
        'mass_density':'analyticmarginalof4componentlearned(logMc,logitq,atanhchieff)Gaussianmixture;notpublicPE',
        'mass_probability_grid':'512intervalsinlogMc3-300plusbothunboundedtails;GaussianCDFintegration,notupsamplingold64binhead',
        'BC':'exactprobabilitymassesafterfinitebinintegration;sum_sqrt_pi_pj',
        'D':'absolutemixturemeanlogMcgapdividedbytotalmixturepredictiveSDincludingbetweencomponentvariance;notGaussianPEsignificance',
        'formula':'Z_FRT+gamma*min(log(p_new_mass_conflict_tail/0.05),0)',
        'reference':'512source-disjointnewdevelopmentparents,1truepair/source;32sharedvalidationnoiseblocks',
        'gamma_grid':GAMMAS,'selection':'originalBAYESTARvalidationpriorityandguards','no_positive_reward':True,
        'frozen':['time','sky','outerweights','scope','historicaloutputs'],'no_PE_official_ID_inputs':True,
        'fresh_confirmation_required':True})
    selected,cache,states={},{},[]
    for dep in e.DEPS:
        for ms,es in zip(body.MODEL_SEEDS,dev.SEEDS):
            df,dx=values(root,dep,ms,es,'development',family);y=df.is_true_pair.to_numpy(bool)
            cal={'family':family,'statistic':statistic,'reference':np.sort(dx[statistic][y]).tolist()}
            f,x=values(root,dep,ms,es,'validation',family);cache[dep,es]=f,x;bm=ev.metrics(f,f.waveform_score.to_numpy(float),dep,es)
            rows,choices=[],[]
            for gamma in GAMMAS:
                spec={**cal,'gamma':gamma};z,*_=score(f,x,spec);m=ev.metrics(f,z,dep,es);ok=ev.guard(m,bm)
                rows.append({'gamma':gamma,'pass':ok,**{a+'_'+k:v for a,b in m.items() for k,v in b.items()}})
                if ok:choices.append((e.old_selection.objective(m,gamma,0),spec))
            out=trial/f'calibration/{dep}/seed_{es}';out.mkdir(parents=True);dev.csv_write(out/'validation_grid.csv',pd.DataFrame(rows))
            spec=min(choices,key=lambda c:c[0])[1];selected[dep,es]=spec;dev.json_write(out/'SELECTED_CONFIG.json',spec)
            states.append({'deployment':dep,'seed':es,'gamma':spec['gamma']})
    dev.json_write(trial/'contracts/SELECTED_FREEZE.json',{'configs':states,'real_used_to_select':False})
    print(json.dumps({'continuous_mass_selection':code,'states':states}),flush=True)
    rows,guards=[],[]
    for dep in e.DEPS:
        real={}
        for ms,es in zip(body.MODEL_SEEDS,dev.SEEDS):
            for split in ('validation','test','real'):
                f,x=cache[dep,es] if split=='validation' else values(root,dep,ms,es,split,family)
                z,penalty,p=score(f,x,selected[dep,es]);n=f.copy();n['FRT_baseline_waveform_score']=f.waveform_score
                n['waveform_score']=z;n['continuous_mass_penalty']=penalty;n['continuous_mass_tail']=p
                for key,v in x.items():n['continuous_mass_'+key]=v
                out=trial/f'evaluation/{dep}/seed_{es}';out.mkdir(parents=True,exist_ok=True);n.to_parquet(out/f'{split}_pairs.parquet',index=False)
                if split=='real':real[es]=n
                else:
                    bm,mm=ev.metrics(f,f.waveform_score.to_numpy(float),dep,es),ev.metrics(f,z,dep,es)
                    guards.append({'deployment':dep,'seed':es,'split':split,'pass':ev.guard(mm,bm)})
                    for method in mm:
                        for config,m in [('FRT_BASELINE',bm[method]),('CANDIDATE',mm[method])]:rows.append({'deployment':dep,'seed':es,'split':split,'method':method,'config':config,**m})
        dev.save_evaluation(trial,'CANDIDATE',dep,real,pd.read_parquet(root/f'audit/{dep}_frozen_external_reference.parquet'))
    dev.csv_write(trial/'tables/RETRIEVAL_PER_SEED.csv',pd.DataFrame(rows));dev.csv_write(trial/'tables/REUSED_GUARDRAILS.csv',pd.DataFrame(guards));e.assess(root,trial)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    for family in ('phase','raw'):
        for statistic in ('BC','D'):run(a.root,family,statistic)
