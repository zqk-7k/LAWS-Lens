#!/usr/bin/env python3
"""Independent-population calibration of frozen composite waveform evidence."""
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
from scipy.optimize import minimize
from scipy.special import softmax,expit
import mcwf_pe_frontend_extension_v2_20260907 as e
import mcwf_fresh_confirmation_20260906 as fresh
import mcwf_finite_reference_tail_20260907 as tail
import mcwf_mixture_evaluate_20260907 as mdn

dev,body,ev=e.dev,e.body,e.ev
FRACTIONS=(0.,.125,.25,.5,.75,1.)


def development(root,dep,ms,es):
    path=root/f'expanded_calibration/development/{dep}/seed_{es}/pairs.parquet'
    if path.exists():return pd.read_parquet(path)
    src=root/f'expanded_data/{dep}/validation'
    raw=np.asarray(np.load(src/'raw2s.npy',mmap_mode='r'),np.float32)
    meta=pd.read_parquet(src/'event_metadata.parquet')
    i,j=np.triu_indices(len(meta),1)
    f=pd.DataFrame({'idx_i':i,'idx_j':j,'is_true_pair':meta.source_uid.to_numpy()[i]==meta.source_uid.to_numpy()[j]})
    f=fresh.old_waveform(dep,es,raw,f);f['previous_waveform_score']=f.waveform_score
    cp=e.TRAINED/f'models/RAW-PHASE-SOURCE/{dep}/seed_{ms}/validation_selected_model.pt'
    ck=torch.load(cp,weights_only=False,map_location='cpu')
    x=np.load(root/f'expanded_encoder/features/{dep}/validation.npy')
    model=body.Encoder('RAW-PHASE-SOURCE').cuda().eval();model.load_state_dict(ck['model'])
    logits,z=body.infer(model,(x-ck['mu'])/ck['sd'],raw)
    if 'temperature' in ck:temperature=ck['temperature']
    else:
        grid=pd.read_csv(cp.parent/'temperature_grid.csv')
        temperature=min(grid.to_dict('records'),key=lambda r:(r['ce'],abs(r['temperature']-1)))['temperature']
    p=softmax(logits.astype(float)/temperature,1)
    f['new_mass_predictive_BC']=np.sqrt(p[i]*p[j]).sum(1)
    spec=json.loads((e.BASE/f'calibration/{dep}/seed_{es}/SELECTED_CONFIG.json').read_text())
    f['waveform_score'],penalty,_,ood=tail.score(f,spec)
    f['RNC_mass_penalty']=penalty
    mc=np.load(root/f'mixture_density/calibration/{dep}/seed_{ms}/fit.npz')
    if not np.array_equal(mc['i'],i) or not np.array_equal(mc['j'],j):raise RuntimeError('Mixture development pair mismatch')
    f['MDN_mass']=mc['mass'];f['MDN_conditional']=mc['conditional']
    path.parent.mkdir(parents=True,exist_ok=True);f.to_parquet(path,index=False)
    np.savez_compressed(path.parent/'frozen_RNC_predictions.npz',p=p,z=z)
    dev.json_write(path.with_suffix('.json'),{'checkpoint_sha256':dev.sha(cp),'temperature':temperature,
        'waveforms_sha256':dev.sha(src/'raw2s.npy'),'independent_sources':meta.source_uid.nunique(),
        'noise_blocks':meta.noise_bank_index.nunique(),'sources_not_in_model_training':True,
        'data_profile':'coveragebalancedMc/SNR physicalwaveforms,sharednoiseblocks;notastrophysicalpopulation'})
    return f


def matrix(f,x,kind):
    if kind=='affine':return f.waveform_score.to_numpy(float)[:,None]
    return np.column_stack([f.waveform_score,x['mass'],x['conditional']])


def fit(f,kind):
    x=matrix(f,{'mass':f.MDN_mass.to_numpy(float),'conditional':f.MDN_conditional.to_numpy(float)},kind)
    y=f.is_true_pair.to_numpy(bool);w=np.where(y,.5/y.sum(),.5/(~y).sum())
    mean=np.average(x,axis=0,weights=w);sd=np.sqrt(np.average((x-mean)**2,axis=0,weights=w)).clip(1e-6)
    xx=(x-mean)/sd
    def loss(theta):
        a,b=theta[:-1],theta[-1];value=xx@a+b;res=w*(expit(value)-y)
        reg=1e-4*np.dot(a,a)
        return float(np.dot(w,np.logaddexp(0,value)-y*value)+reg),np.r_[xx.T@res+2e-4*a,res.sum()]
    result=minimize(loss,np.r_[np.ones(x.shape[1]),0.],jac=True,method='L-BFGS-B',
        bounds=[(.0001,100.)]+[(0.,100.)]*(x.shape[1]-1)+[(None,None)],options={'ftol':1e-12,'gtol':1e-9,'maxiter':1000})
    if not result.success:raise RuntimeError(result.message)
    return {'kind':kind,'mean':mean.tolist(),'sd':sd.tolist(),'coef':result.x[:-1].tolist(),
        'intercept':float(result.x[-1]),'min':x.min(0).tolist(),'max':x.max(0).tolist(),
        'balanced_loss':float(result.fun),'positive_sources':int(y.sum()),'global_regularization':1e-4}


def get(root,dep,ms,es,split):return mdn.get(root,dep,ms,es,split)


def score(f,x,spec):
    xx=matrix(f,x,spec['kind']);z=f.waveform_score.to_numpy(float)
    fitted=((xx-np.array(spec['mean']))/np.array(spec['sd']))@np.array(spec['coef'])+spec['intercept']
    ood=((xx<np.array(spec['min']))|(xx>np.array(spec['max']))).any(1)
    delta=fitted-z
    delta=np.where(ood&(delta>0),0.,delta)
    return z+spec['fraction']*delta,delta,ood


def run(root,kind):
    trial=root/'trials'/f'EXPANDED-COMPOSITE-CALIBRATION-{kind.upper()}'
    if trial.exists():raise RuntimeError('Independent trial required')
    for name in ('contracts','calibration','tables','evaluation','results'):(trial/name).mkdir(parents=True)
    shutil.copy2(__file__,trial/'contracts'/Path(__file__).name)
    dev.json_write(trial/'contracts/RECIPE.json',{'created_utc':datetime.now(timezone.utc).isoformat(),
        'same_kind_both_runs':kind,'fit':'512newindependentdevelopment-sources,notBAYESTARtuningcatalog;class-balancedconstrainedlogisticconditionaldensityratio',
        'features':'affine:Z_FRT only;joint:Z_FRT,MDNlogMcproductintegral,MDNjoint-minus-Mcproductintegral',
        'new_encoder_information':kind=='joint','no_real_PE_or_official_inputs':True,
        'coefficient_constraints':'increasing ineachwaveformcompatibilityfeature;ridge1e-4innormalizedunits',
        'blend_grid':FRACTIONS,'selection':'unchangedBAYESTARvalidationobjectiveandguards',
        'formula':'Z_FRT+fraction*(new_joint_calibrated_LR-Z_FRT);noaddedpositiveincrementoutsidefitdomain',
        'interpretation':'correlatedstatisticalwaveformevidence,notindependentphysicalBayesfactors',
        'relative_weight_disclosure':'recalibrationchangeswaveformstrengthrelative totime/sky,despitefrozenouterC-fixedweights',
        'frozen':['time','sky','outerweights','scope','historicalresults'],'fresh_confirmation_required':True})
    selected,cache,states={},{},[]
    for dep in e.DEPS:
        for ms,es in zip(body.MODEL_SEEDS,dev.SEEDS):
            cal=fit(development(root,dep,ms,es),kind);f,x=get(root,dep,ms,es,'validation');cache[dep,es]=f,x
            bm=ev.metrics(f,f.waveform_score.to_numpy(float),dep,es);rows,choices=[],[]
            for fraction in FRACTIONS:
                spec={**cal,'fraction':fraction};z,*_=score(f,x,spec);mm=ev.metrics(f,z,dep,es);ok=ev.guard(mm,bm)
                rows.append({'fraction':fraction,'pass':ok,**{a+'_'+k:v for a,b in mm.items() for k,v in b.items()}})
                if ok:choices.append((e.old_selection.objective(mm,fraction,1),spec))
            out=trial/f'calibration/{dep}/seed_{es}';out.mkdir(parents=True);dev.csv_write(out/'validation_grid.csv',pd.DataFrame(rows))
            spec=min(choices,key=lambda c:c[0])[1];selected[dep,es]=spec;dev.json_write(out/'SELECTED_CONFIG.json',spec)
            states.append({'deployment':dep,'seed':es,**spec})
    dev.json_write(trial/'contracts/SELECTED_FREEZE.json',{'configs':states,'real_used_to_select':False})
    print(json.dumps({'expanded_calibration_selected':kind,'states':states}),flush=True)
    rows,guards=[],[]
    for dep in e.DEPS:
        real={}
        for ms,es in zip(body.MODEL_SEEDS,dev.SEEDS):
            for split in ('validation','test','real'):
                f,x=cache[dep,es] if split=='validation' else get(root,dep,ms,es,split)
                z,delta,ood=score(f,x,selected[dep,es]);n=f.copy();n['FRT_baseline_waveform_score']=f.waveform_score
                n['waveform_score']=z;n['expanded_recalibration_delta']=delta;n['expanded_recalibration_ood']=ood
                for col in ('time_score','sky_raw_log_bf'):assert np.array_equal(n[col],f[col])
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
    for kind in ('affine','joint'):run(a.root,kind)
