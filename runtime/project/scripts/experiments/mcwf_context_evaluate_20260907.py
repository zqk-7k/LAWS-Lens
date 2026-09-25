#!/usr/bin/env python3
"""Shared FRT evaluation for additional waveform-context encoder ablations."""
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
import torch
from scipy.special import softmax
import mcwf_pe_frontend_extension_v2_20260907 as e
import mcwf_finite_reference_tail_20260907 as tail
import mcwf_dense_features_20260907 as df
import mcwf_dense_encoder_20260907 as dn
import mcwf_long_features_20260907 as lf
import mcwf_long_encoder_20260907 as ln

dev,body,ev=e.dev,e.body,e.ev
torch.set_num_threads(2)
GAMMAS=(.125,.25,.5,1.,2.,4.)
BETAS=(0.,.0625,.125,.25,.5,1.,2.)


def prediction(root,kind,dep,ms,es,split):
    folder=root/f'{kind}_context/models/{dep}/seed_{ms}'
    if not (folder/'COMPLETE.json').exists():raise RuntimeError('All training must complete before model evaluation')
    cp=folder/'validation_selected_model.pt';ck=torch.load(cp,weights_only=False,map_location='cpu')
    if ck['epoch']==0:
        # Zero-weight branches cannot be presented as new information. Retain
        # exact archived predictions, avoiding GEMM-rounding rank changes.
        return e.predictions(dep,ms,es,split)
    path=root/f'{kind}_context/predictions/{dep}/model_{ms}_eval_{es}/{split}.npz'
    if path.exists():
        a=np.load(path)
        if str(a['checkpoint_sha256'])!=dev.sha(cp):raise RuntimeError('Changed model')
        return a
    fm,nm=(df,dn) if kind=='dense' else (lf,ln)
    extra=fm.deployment(root,dep,es,split)
    if split=='real':
        full,events=dev.real_inputs(dep);valid=events.strict_h1l1_preprocessing_pass.to_numpy(bool);raw=dev.TRAIN.make_window_view(np.asarray(full[valid],np.float32),2)
        short=np.load(e.TRAINED/f'cache/deployment_event_psd/{dep}/real_features.npy')
    else:
        plan=dev.BASE.retained_event_plan(dep,es,split);full=dev.ORCH.event_array_for_plan(dep,es,split,plan);raw=dev.TRAIN.make_window_view(np.asarray(full,np.float32),2)
        short=np.load(e.TRAINED/f'cache/deployment_event_psd/{dep}/{es}_{split}_features.npy')
    prefix='extra' if kind=='dense' else 'long'
    x=np.concatenate([((short-ck['mu'])/ck['sd']).reshape(len(short),-1),((extra-ck[prefix+'_mu'])/ck[prefix+'_sd']).reshape(len(short),-1)],1)
    model=nm.model().cuda().eval();model.load_state_dict(ck['model']);logits,z=body.infer(model,x,raw);p=softmax(logits.astype(float)/ck['temperature'],1)
    if split=='real':
        pp=np.full((len(full),64),np.nan);zz=np.full((len(full),128),np.nan);pp[valid],zz[valid]=p,z;p,z=pp,zz
    path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,p=p,z=z,checkpoint_sha256=dev.sha(cp))
    return np.load(path)


def get(root,kind,dep,ms,es,split):
    f=pd.read_parquet(e.BASE/f'evaluation/{dep}/seed_{es}/{split}_pairs.parquet');a=prediction(root,kind,dep,ms,es,split)
    i,j=f.idx_i.to_numpy(int),f.idx_j.to_numpy(int);p,z=a['p'],a['z']
    return f,{'mass':-np.log(np.sum(np.sqrt(p[i]*p[j]),1).clip(1e-12)),
              'reference':np.log(np.sum(p[i]*p[j]/e.reference_prior(dep),1).clip(1e-30)),
              'cosine':np.sum(z[i]*z[j],1)}


def support(root,kind,dep,ms):
    folder=root/f'{kind}_context/models/{dep}/seed_{ms}';ck=torch.load(folder/'validation_selected_model.pt',weights_only=False,map_location='cpu')
    if ck['epoch']==0:
        ref=e.fit_reference(dep,ms,'REF-RAW',e.reference_prior(dep));return ref['fit_min'],ref['fit_max']
    a=np.load(folder/'validation_predictions.npz');p=a['p'];i,j=np.triu_indices(len(p),1)
    raw=np.log(np.sum(p[i]*p[j]/e.reference_prior(dep),1).clip(1e-30));return float(raw.min()),float(raw.max())


def score(f,x,spec):
    p=tail.tail_probability(x['mass'],spec['reference_mass']);penalty=np.minimum(np.log(p/.05),0.)
    ref=np.clip(x['reference'],-4,4);ood=(x['reference']<spec['fit_min'])|(x['reference']>spec['fit_max']);ref=np.where(ood,np.minimum(ref,0),ref)
    return f.previous_waveform_score.to_numpy(float)+spec['gamma']*penalty+spec['beta']*ref,penalty,ref,ood


def run(root,kind):
    trial=root/'trials'/('DENSE-INTRINSIC-ENCODER' if kind=='dense' else 'PEAK2PLUS8-ENCODER')
    if trial.exists():raise RuntimeError('Independent trial required')
    for n in ('contracts','calibration','tables','results','evaluation'):(trial/n).mkdir(parents=True)
    shutil.copy2(__file__,trial/'contracts'/Path(__file__).name)
    dev.json_write(trial/'contracts/RECIPE.json',{'created_utc':datetime.now(timezone.utc).isoformat(),
        'same_both_runs':True,'kind':kind,'training_contract_sha256':dev.sha(root/f'{kind}_context/contracts/TRAINING.json'),
        'formula':'Zold+gamma*finite_reference_Mc_tail(new_p)+beta*bounded_predictive_reference_overlap(new_p)',
        'gammas':GAMMAS,'betas':BETAS,'selection':'BAYESTAR simulationvalidation only, current FRT-relative guardrails and deterministic F50/F90/AP/R10 priority',
        'epoch_zero_policy':'use exact current archived RNC predictions if validation selects epoch0; no claimed information gain or floating-point rank gain from zero additional weights',
        'reference_overlap':'ranking feature, NOT calibrated physicalPE Bayes factor; positive extrapolation disabled',
        'frozen':['time','sky','outerweights','scope','all old files'],'real_role':'adaptive development only','fresh_confirmation_required':True})
    selected={};states=[];cache={}
    for dep in e.DEPS:
        for ms,es in zip(body.MODEL_SEEDS,dev.SEEDS):
            f,x=get(root,kind,dep,ms,es,'validation');cache[dep,es]=(f,x);y=f.is_true_pair.to_numpy(bool)
            lo,hi=support(root,kind,dep,ms);spec={'reference_mass':np.sort(x['mass'][y]).tolist(),'fit_min':lo,'fit_max':hi,'deployment':dep,'model_seed':ms,'eval_seed':es,'kind':kind}
            bm=ev.metrics(f,f.waveform_score.to_numpy(float),dep,es);rows=[];choices=[]
            for gamma in GAMMAS:
                for beta in BETAS:
                    cfg={**spec,'gamma':gamma,'beta':beta};z,*_=score(f,x,cfg);m=ev.metrics(f,z,dep,es);ok=ev.guard(m,bm)
                    rows.append({'gamma':gamma,'beta':beta,'pass':ok,**{a+'_'+k:v for a,b in m.items() for k,v in b.items()}})
                    if ok:choices.append((e.old_selection.objective(m,gamma,beta),cfg))
            out=trial/f'calibration/{dep}/seed_{es}';out.mkdir(parents=True);dev.csv_write(out/'validation_grid.csv',pd.DataFrame(rows))
            st={'deployment':dep,'seed':es,'eligible':len(choices)}
            if choices:
                cfg=min(choices,key=lambda c:c[0])[1];selected[dep,es]=cfg;dev.json_write(out/'SELECTED_CONFIG.json',cfg);st.update(gamma=cfg['gamma'],beta=cfg['beta'])
            states.append(st)
    dev.json_write(trial/'contracts/SELECTED_FREEZE.json',{'configs':states,'pass':len(selected)==6,'real_used_to_select':False});print(json.dumps({'context_selected':kind,'configs':states}),flush=True)
    if len(selected)!=6:return
    rows=[];guards=[]
    for dep in e.DEPS:
        real={}
        for ms,es in zip(body.MODEL_SEEDS,dev.SEEDS):
            for split in ('validation','test','real'):
                f,x=cache[dep,es] if split=='validation' else get(root,kind,dep,ms,es,split);z,penalty,ref,ood=score(f,x,selected[dep,es])
                n=f.copy();n['FRT_baseline_waveform_score']=f.waveform_score;n['waveform_score']=z;n['context_mass_penalty']=penalty;n['context_reference']=ref;n['context_ood']=ood
                out=trial/f'evaluation/{dep}/seed_{es}';out.mkdir(parents=True,exist_ok=True);n.to_parquet(out/f'{split}_pairs.parquet',index=False)
                if split=='real':real[es]=n
                else:
                    bm=ev.metrics(f,f.waveform_score.to_numpy(float),dep,es);m=ev.metrics(f,z,dep,es);guards.append({'deployment':dep,'seed':es,'split':split,'pass':ev.guard(m,bm)})
                    for method in bm:
                        for config,metrics in [('FRT_BASELINE',bm[method]),('CANDIDATE',m[method])]:rows.append({'deployment':dep,'seed':es,'split':split,'method':method,'config':config,**metrics})
        dev.save_evaluation(trial,'CANDIDATE',dep,real,pd.read_parquet(root/f'audit/{dep}_frozen_external_reference.parquet'))
    dev.csv_write(trial/'tables/RETRIEVAL_PER_SEED.csv',pd.DataFrame(rows));dev.csv_write(trial/'tables/REUSED_GUARDRAILS.csv',pd.DataFrame(guards));e.assess(root,trial)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--kind',choices=['long','dense'],required=True);a=p.parse_args();run(a.root,a.kind)
