#!/usr/bin/env python3
"""Evaluate shared Mc/q encoders with frozen time/sky and FRT controls."""
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
from scipy.special import softmax
import mcwf_pe_frontend_extension_v2_20260907 as e
import mcwf_pe_frontend_joint_intrinsic_20260907 as net
import mcwf_finite_reference_tail_20260907 as tail

dev,body,ev=e.dev,e.body,e.ev
torch.set_num_threads(2)


def prediction(root,dep,ms,es,split):
    folder=root/f'joint_intrinsic/predictions/{dep}/model_{ms}_eval_{es}';folder.mkdir(parents=True,exist_ok=True)
    path=folder/f'{split}.npz'
    if path.exists():return np.load(path)
    if split=='real':
        full,events=dev.real_inputs(dep);valid=events.strict_h1l1_preprocessing_pass.to_numpy(bool)
        values=np.asarray(full[valid]);feature=e.TRAINED/f'cache/deployment_event_psd/{dep}/real_features.npy'
    else:
        events=dev.BASE.retained_event_plan(dep,es,split);values=dev.ORCH.event_array_for_plan(dep,es,split,events)
        feature=e.TRAINED/f'cache/deployment_event_psd/{dep}/{es}_{split}_features.npy'
    cp=root/f'joint_intrinsic/models/{dep}/seed_{ms}/validation_selected_model.pt'
    ck=torch.load(cp,weights_only=False,map_location='cpu');model=net.Encoder().cuda().eval();model.load_state_dict(ck['model'])
    raw=dev.TRAIN.make_window_view(np.asarray(values,dtype=np.float32),2)
    m,q,z=net.infer(model,(np.load(feature)-ck['mu'])/ck['sd'],raw)
    p=softmax(m.astype(float)/ck['temperature'],1);pq=softmax(q.astype(float)/ck['q_temperature'],1)
    if split=='real':
        pp=np.full((len(full),64),np.nan);qq=np.full((len(full),len(net.probe.QGRID)),np.nan);zz=np.full((len(full),128),np.nan)
        pp[valid],qq[valid],zz[valid]=p,pq,z;p,pq,z=pp,qq,zz
    np.savez_compressed(path,p=p,q=pq,z=z,checkpoint_sha256=dev.sha(cp))
    return np.load(path)


def get(root,dep,ms,es,split):
    f=pd.read_parquet(e.BASE/f'evaluation/{dep}/seed_{es}/{split}_pairs.parquet')
    saved=prediction(root,dep,ms,es,split)
    i,j=f.idx_i.to_numpy(int),f.idx_j.to_numpy(int)
    pm,pq,z=saved['p'],saved['q'],saved['z']
    return f,{'mass':-np.log(np.sum(np.sqrt(pm[i]*pm[j]),1).clip(1e-12)),
              'q':-np.log(np.sum(np.sqrt(pq[i]*pq[j]),1).clip(1e-12)),
              'reference':np.log(np.sum(pm[i]*pm[j]/e.reference_prior(dep),1).clip(1e-30)),
              'cosine':np.sum(z[i]*z[j],1)}


def score(f,x,spec):
    a=np.minimum(np.log(tail.tail_probability(x['mass'],spec['reference_mass'])/.05),0.)
    q=np.minimum(np.log(tail.tail_probability(x['q'],spec['reference_q'])/.05),0.)
    return f.previous_waveform_score.to_numpy(float)+spec['gamma']*a+spec['q_gamma']*q


def run(root,useq):
    name='JOINT-MCQ-FRT' if useq else 'JOINT-MC-FRT'
    trial=root/'trials'/name
    if trial.exists():raise RuntimeError('No overwrite')
    for folder in ('contracts','calibration','tables','evaluation','results'):(trial/folder).mkdir(parents=True)
    shutil.copy2(__file__,trial/'contracts'/Path(__file__).name)
    dev.json_write(trial/'contracts/RECIPE.json',{'created_utc':datetime.now(timezone.utc).isoformat(),
        'same_both_runs':True,'formula':'Zold + gamma*finite_reference_tail(newMcBC) + q_gamma*finite_reference_tail(newqBC)',
        'q_gamma_grid':[0.,.25,.5,1.,2.] if useq else [0.],
        'gamma_grid':[.125,.25,.5,1.,2.,4.],
        'new_model':'joint Mc/q source-SupCon and RNC2D training, six50epoch runs',
        'selection':'simulation validation-only original F50/F90/AP/R10 priority, guardrails relative to current FRT',
        'q_regression_limit':'q predictions and predictive BC are NOT public PE; weak validation q accuracy must be reported',
        'no_real_PE_or_official_training_labels':True,'fresh_confirmation_required':True})
    configs={};status=[];data={}
    for dep in e.DEPS:
        for ms,es in zip(body.MODEL_SEEDS,dev.SEEDS):
            cp=root/f'joint_intrinsic/models/{dep}/seed_{ms}/COMPLETE.json'
            if not cp.exists():raise RuntimeError('Training incomplete '+str(cp))
            f,x=get(root,dep,ms,es,'validation');data[dep,es,'validation']=(f,x)
            y=f.is_true_pair.to_numpy(bool);base=ev.metrics(f,f.waveform_score.to_numpy(float),dep,es)
            candidates=[];rows=[]
            for gamma in (.125,.25,.5,1.,2.,4.):
                for qgamma in ((0.,.25,.5,1.,2.) if useq else (0.,)):
                    spec={'gamma':gamma,'q_gamma':qgamma,'reference_mass':np.sort(x['mass'][y]).tolist(),
                          'reference_q':np.sort(x['q'][y]).tolist(),'deployment':dep,'model_seed':ms,'eval_seed':es}
                    z=score(f,x,spec);m=ev.metrics(f,z,dep,es);passed=ev.guard(m,base)
                    rows.append({'gamma':gamma,'q_gamma':qgamma,'pass':passed,**{method+'_'+k:v for method,mm in m.items() for k,v in mm.items()}})
                    if passed:candidates.append((e.old_selection.objective(m,gamma,qgamma),spec))
            out=trial/f'calibration/{dep}/seed_{es}';out.mkdir(parents=True)
            dev.csv_write(out/'validation_grid.csv',pd.DataFrame(rows))
            st={'deployment':dep,'eval_seed':es,'model_seed':ms,'eligible':len(candidates)}
            if candidates:
                configs[dep,es]=min(candidates,key=lambda c:c[0])[1];dev.json_write(out/'SELECTED_CONFIG.json',configs[dep,es])
                st.update(gamma=configs[dep,es]['gamma'],q_gamma=configs[dep,es]['q_gamma'])
            status.append(st)
    dev.json_write(trial/'contracts/SELECTED_FREEZE.json',{'pass':len(configs)==6,'status':status,'selection_used_real':False})
    print(json.dumps({'joint_selected':name,'status':status}),flush=True)
    if len(configs)!=6:return
    metrics=[];guards=[]
    for dep in e.DEPS:
        real={}
        for ms,es in zip(body.MODEL_SEEDS,dev.SEEDS):
            for split in ('validation','test','real'):
                f,x=data[dep,es,split] if (dep,es,split) in data else get(root,dep,ms,es,split)
                z=score(f,x,configs[dep,es]);changed=f.copy();changed['FRT_baseline_waveform_score']=f.waveform_score;changed['waveform_score']=z
                for k,v in x.items():changed['joint_'+k]=v
                out=trial/f'evaluation/{dep}/seed_{es}';out.mkdir(parents=True,exist_ok=True);changed.to_parquet(out/f'{split}_pairs.parquet',index=False)
                if split=='real':real[es]=changed
                else:
                    b=ev.metrics(f,f.waveform_score.to_numpy(float),dep,es);n=ev.metrics(f,z,dep,es)
                    guards.append({'deployment':dep,'seed':es,'split':split,'pass':ev.guard(n,b)})
                    for method in b:
                        for conf,mm in [('FRT_BASELINE',b[method]),('CANDIDATE',n[method])]:
                            metrics.append({'deployment':dep,'seed':es,'split':split,'method':method,'config':conf,**mm})
        dev.save_evaluation(trial,'CANDIDATE',dep,real,pd.read_parquet(root/f'audit/{dep}_frozen_external_reference.parquet'))
    dev.csv_write(trial/'tables/RETRIEVAL_PER_SEED.csv',pd.DataFrame(metrics));dev.csv_write(trial/'tables/REUSED_GUARDRAILS.csv',pd.DataFrame(guards))
    e.assess(root,trial)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--with-q',action='store_true');a=p.parse_args();run(a.root,a.with_q)
