#!/usr/bin/env python3
"""Conditional waveform information from the retrained raw/source density model."""
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
from scipy.optimize import minimize
from scipy.special import expit
import mcwf_pe_frontend_extension_v2_20260907 as e
import mcwf_intrinsic_rank_encoder_20260907 as source
import mcwf_mixture_density_20260907 as mixture
import mcwf_finite_reference_tail_20260907 as tail

dev,body,ev=e.dev,e.body,e.ev
GAMMAS=(0.,.125,.25,.5,1.,2.)
BETAS=(0.,.0625,.125,.25,.5,1.,2.)


def features(a,i,j):
    z=np.array(a['z'],float);z/=np.linalg.norm(z,axis=1,keepdims=True).clip(1e-12)
    full=mixture.pair_overlap(a,i,j);mass=mixture.pair_overlap(a,i,j,(0,))
    return {'cosine':np.sum(z[i]*z[j],1),'intrinsic':full-mass,
        'mass':-np.log(np.sqrt(a['p'][i]*a['p'][j]).sum(1).clip(1e-12))}


def data(root,dep,ms,es,split):
    if split=='development':
        path=root/f'intrinsic_rank_encoder/pair_features/{dep}/seed_{es}/development.npz'
        f=pd.read_parquet(root/f'expanded_calibration/development/{dep}/seed_{es}/pairs.parquet')
        a=np.load(root/f'intrinsic_rank_encoder/models/{dep}/seed_{ms}/validation_predictions.npz')
    else:
        path=root/f'intrinsic_rank_encoder/pair_features/{dep}/seed_{es}/{split}.npz'
        f=pd.read_parquet(e.BASE/f'evaluation/{dep}/seed_{es}/{split}_pairs.parquet')
        a=None
    if path.exists():
        b=np.load(path)
        if not np.array_equal(b['i'],f.idx_i) or not np.array_equal(b['j'],f.idx_j):raise RuntimeError('Pairalignmentchanged')
        return f,{k:b[k] for k in ('cosine','intrinsic','mass')}
    if a is None:a=source.prediction(root,dep,ms,es,split)
    x=features(a,f.idx_i.to_numpy(int),f.idx_j.to_numpy(int))
    path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,**x,i=f.idx_i,j=f.idx_j)
    return f,x


def logistic(xx,y):
    w=np.where(y,.5/y.sum(),.5/(~y).sum())
    mean=np.average(xx,axis=0,weights=w);sd=np.sqrt(np.average((xx-mean)**2,axis=0,weights=w)).clip(1e-6)
    x=(xx-mean)/sd
    def loss(theta):
        a,b=theta[:-1],theta[-1];v=x@a+b;r=w*(expit(v)-y)
        return float(np.dot(w,np.logaddexp(0,v)-y*v)+1e-4*np.dot(a,a)),np.r_[x.T@r+2e-4*a,r.sum()]
    fit=minimize(loss,np.r_[np.ones(x.shape[1]),0.],jac=True,method='L-BFGS-B',
        bounds=[(.0001,100.)]+[(None,None)]*x.shape[1],options={'ftol':1e-12,'gtol':1e-9,'maxiter':1000})
    if not fit.success:raise RuntimeError(fit.message)
    return {'mean':mean.tolist(),'sd':sd.tolist(),'coef':fit.x[:-1].tolist(),'intercept':float(fit.x[-1]),
        'minimum':xx.min(0).tolist(),'maximum':xx.max(0).tolist(),'balanced_loss':float(fit.fun)}


def matrix(f,x,kind):
    v=[f.waveform_score.to_numpy(float),x['cosine']]
    if kind=='source_and_intrinsic':v.append(x['intrinsic'])
    return np.column_stack(v)


def fit(root,dep,ms,es,kind):
    path=root/f'intrinsic_rank_encoder/calibration/{kind}/{dep}/seed_{es}/fit.json'
    if path.exists():return json.loads(path.read_text())
    f,x=data(root,dep,ms,es,'development');y=f.is_true_pair.to_numpy(bool)
    spec={'kind':kind,'old':logistic(f.waveform_score.to_numpy(float)[:,None],y),'full':logistic(matrix(f,x,kind),y),
        'mass_reference':np.sort(x['mass'][y]).tolist(),'source_count':int(y.sum()),'shared_noise_blocks':32}
    path.parent.mkdir(parents=True,exist_ok=True);dev.json_write(path,spec);return spec


def decision(x,cal):return ((x-np.array(cal['mean']))/np.array(cal['sd']))@np.array(cal['coef'])+cal['intercept']


def score(f,x,spec):
    old=f.waveform_score.to_numpy(float);xx=matrix(f,x,spec['kind'])
    increment=decision(xx,spec['full'])-decision(old[:,None],spec['old'])
    ood=((xx<np.array(spec['full']['minimum']))|(xx>np.array(spec['full']['maximum']))).any(1)
    p=tail.tail_probability(x['mass'],spec['mass_reference'])
    penalty=np.minimum(np.log(p/.05),0.)
    increment=np.clip(np.where(ood|(p<.05),np.minimum(increment,0.),increment),-4,4)
    return old+spec['gamma']*penalty+spec['beta']*increment,penalty,increment,ood


def run(root,kind):
    trial=root/'trials'/f'INTRINSIC-RNC-CONDITIONAL-{kind.upper()}'
    if trial.exists():raise RuntimeError('Independenttrialrequired')
    for n in ('contracts','calibration','tables','evaluation','results'):(trial/n).mkdir(parents=True)
    shutil.copy2(__file__,trial/'contracts'/Path(__file__).name)
    dev.json_write(trial/'contracts/RECIPE.json',{'created_utc':datetime.now(timezone.utc).isoformat(),
        'same_kind_O3_O4a':kind,'new_model':'retrainedraw2s+phase sourceencoderwithuncertainty-scaledmultivariateRNC(logMc,q,chieff)andjointpredictivedensityhead',
        'calibration_features':'full=oldZ_FRT,newcosine,optionalMDNconditionalq/spin;old=Z_FRTalone',
        'increment':'differenceofclass-balancedfullandoldlogisticlogodds;approximatesconditionalnewwaveforminformation,notproductofindependentBFs',
        'formula':'Z_FRT+gamma*newfinite-referenceMccontradiction+beta*boundedconditionalwaveformincrement',
        'mass_reference':'512independentnewdevelopment-validationparents,onecompanionpair/source',
        'positive_screen':'positiveincrementsdisabledoutsidefitsupportornewmass5percentcontradictionregion',
        'gamma_grid':GAMMAS,'beta_grid':BETAS,'cap':4,'logistic_ridge':1e-4,
        'selection':'BAYESTARvalidationonly,originalpriorityandguards','real_selection':False,
        'frozen':['time','sky','outerweights','scope','historicaloutputs'],'no_PE_official_ID_features':True,
        'fresh_confirmation_required':True,'not_PE_or_normalized_lensing_evidence':True})
    selected,cache,states={},{},[]
    for dep in e.DEPS:
        for ms,es in zip(body.MODEL_SEEDS,dev.SEEDS):
            cal=fit(root,dep,ms,es,kind);f,x=data(root,dep,ms,es,'validation');cache[dep,es]=f,x
            bm=ev.metrics(f,f.waveform_score.to_numpy(float),dep,es);rows,choices=[],[]
            for gamma in GAMMAS:
                for beta in BETAS:
                    spec={**cal,'gamma':gamma,'beta':beta};z,*_=score(f,x,spec);m=ev.metrics(f,z,dep,es);ok=ev.guard(m,bm)
                    rows.append({'gamma':gamma,'beta':beta,'pass':ok,**{a+'_'+k:v for a,b in m.items() for k,v in b.items()}})
                    if ok:choices.append((e.old_selection.objective(m,gamma,beta),spec))
            out=trial/f'calibration/{dep}/seed_{es}';out.mkdir(parents=True);dev.csv_write(out/'validation_grid.csv',pd.DataFrame(rows))
            spec=min(choices,key=lambda c:c[0])[1];selected[dep,es]=spec;dev.json_write(out/'SELECTED_CONFIG.json',spec)
            states.append({'deployment':dep,'seed':es,'gamma':spec['gamma'],'beta':spec['beta']})
    dev.json_write(trial/'contracts/SELECTED_FREEZE.json',{'configs':states,'real_used_to_select':False})
    print(json.dumps({'intrinsic_rank_selection':kind,'states':states}),flush=True)
    rows,guards=[],[]
    for dep in e.DEPS:
        real={}
        for ms,es in zip(body.MODEL_SEEDS,dev.SEEDS):
            for split in ('validation','test','real'):
                f,x=cache[dep,es] if split=='validation' else data(root,dep,ms,es,split)
                z,penalty,inc,ood=score(f,x,selected[dep,es]);n=f.copy();n['FRT_baseline_waveform_score']=f.waveform_score
                n['waveform_score']=z;n['new_mass_penalty']=penalty;n['conditional_waveform_increment']=inc;n['waveform_new_ood']=ood
                out=trial/f'evaluation/{dep}/seed_{es}';out.mkdir(parents=True,exist_ok=True);n.to_parquet(out/f'{split}_pairs.parquet',index=False)
                if split=='real':real[es]=n
                else:
                    bm,mm=ev.metrics(f,f.waveform_score.to_numpy(float),dep,es),ev.metrics(f,z,dep,es)
                    guards.append({'deployment':dep,'seed':es,'split':split,'pass':ev.guard(mm,bm)})
                    for method in bm:
                        for config,m in [('FRT_BASELINE',bm[method]),('CANDIDATE',mm[method])]:rows.append({'deployment':dep,'seed':es,'split':split,'method':method,'config':config,**m})
        dev.save_evaluation(trial,'CANDIDATE',dep,real,pd.read_parquet(root/f'audit/{dep}_frozen_external_reference.parquet'))
    dev.csv_write(trial/'tables/RETRIEVAL_PER_SEED.csv',pd.DataFrame(rows));dev.csv_write(trial/'tables/REUSED_GUARDRAILS.csv',pd.DataFrame(guards));e.assess(root,trial)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    for kind in ('source_only','source_and_intrinsic'):run(a.root,kind)
