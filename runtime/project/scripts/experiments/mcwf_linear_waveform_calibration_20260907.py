#!/usr/bin/env python3
"""Low-capacity monotone residual calibration, fit on simulations only."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[k]='2'
import argparse
import json
from pathlib import Path
import shutil
import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
import mcwf_development_20260905 as dev
import mcwf_encoder_body_20260906 as body
import mcwf_encoder_evaluate_20260906 as ev
import mcwf_unified_waveform_20260906 as u
import mcwf_conditional_waveform_calibration_20260906 as c
import mcwf_cross_domain_selection_20260906 as cross


def fit_linear(x,y,ridge):
    center=np.median(x,axis=0)
    scale=np.maximum(x.std(axis=0),1e-6)
    a=(x-center)/scale
    weights=np.where(y,.5/y.sum(),.5/(~y).sum())
    def objective(theta):
        z=a@theta[1:]+theta[0]
        loss=(weights*(np.logaddexp(0,z)-y*z)).sum()+ridge*(theta[1:]**2).sum()/2
        residual=weights*(expit(z)-y)
        grad=np.r_[residual.sum(),a.T@residual+ridge*theta[1:]]
        return loss,grad
    result=minimize(objective,np.zeros(a.shape[1]+1),jac=True,method='L-BFGS-B',
                    bounds=[(None,None)]+[(0,None)]*a.shape[1],options={'maxiter':1000,'ftol':1e-12,'gtol':1e-8})
    if not result.success:
        raise RuntimeError(result.message)
    return {'kind':'nonnegative_logistic','center':center,'scale':scale,'coef':result.x[1:],'intercept':result.x[0],
            'ridge':ridge,'optimizer_success':result.success,'objective':result.fun}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    a=p.parse_args()
    if a.root.exists():
        raise RuntimeError('Independent root required')
    u.initialize(a.root)
    contract=json.loads((a.root/'contracts/UNIFIED_METHOD.json').read_text())
    contract.update(code='MCWF-MONOTONE-LINEAR-WAVEFORM',features=c.EXTENDED_COLUMNS,
        rationale='Regularized linear calibration avoids leafwise saturated rejection from a small number of independent positive sources. Expose old morphology and intrinsic estimates separately; all predictors derive solely from waveform.',
        fitting='weighted logistic loss with equal total classes plus ridge; nonnegative coefficients; fit only auxiliary source/noise-disjoint calibration partition',
        ridge_grid=[.01,.1,1.],gamma_grid=[.0625,.125,.25,.5],
        selection='same BAY validation plus auxiliary-noise tune guardrails and deterministic false-burden objective; no real PE/official inputs',
        interpretation='regularized discriminative waveform ranking, not an independent Bayes factor or calibrated PE posterior',
        real_role='previously examined development audit, not blinded validation',
        formula='oldZ+gamma*min(clip(b+w*x,-6,6)-oldZ,0)')
    dev.json_write(a.root/'contracts/UNIFIED_METHOD.json',contract)
    shutil.copy2(__file__,a.root/'scripts'/Path(__file__).name)
    statuses=[]
    for dep in u.DEPS:
        for ms,es in zip(body.MODEL_SEEDS,dev.SEEDS):
            src=c.SOURCE/dep/f'seed_{es}'
            f=pd.read_parquet(src/'pairs.parquet')
            meta=pd.read_parquet(src/'event_metadata.parquet')
            fit,tune,noise=c.split_metadata(meta,dep)
            ff,tf=c.subset(f,fit),c.subset(f,tune)
            x=c.design(ff,c.EXTENDED_COLUMNS)
            y=ff.is_true_pair.to_numpy(bool)
            if y.sum()<30 or tf.is_true_pair.to_numpy(bool).sum()<10:
                raise RuntimeError('Insufficient independent calibration support')
            bay=u.frame(dep,ms,es,'validation')
            bm=ev.metrics(bay,bay.previous_waveform_score.to_numpy(),dep,es)
            tm=dev.BASE.full_metrics(tf,tf.previous_waveform_score.to_numpy())
            out=a.root/f'calibration/{dep}/seed_{es}'
            out.mkdir(parents=True)
            dev.json_write(out/'SPLIT_AUDIT.json',{'fit_sources':int(y.sum()),'tune_sources':int(tf.is_true_pair.to_numpy(bool).sum()),
                'source_overlap':0,'noise_overlap':0,'input_sha256':dev.sha(src/'pairs.parquet')})
            grid=[]
            options=[]
            for ridge in (.01,.1,1.):
                model=fit_linear(x,y,ridge)
                mp=out/f'calibrator_ridge{ridge:g}.joblib'
                joblib.dump(model,mp)
                for gamma in (.0625,.125,.25,.5):
                    spec={'recipe':'conditional_monotone_waveform','features':c.EXTENDED_COLUMNS,'calibrator_path':str(mp),
                        'calibrator_sha256':dev.sha(mp),'bounds':[x.min(0).tolist(),x.max(0).tolist()],
                        'deployment':dep,'model_seed':ms,'eval_seed':es,'gamma':gamma,'mass_weight':1.,'ridge':ridge}
                    bs,*_=c.score(bay,spec)
                    ts,*_=c.score(tf,spec)
                    bmet=ev.metrics(bay,bs,dep,es)
                    tmet=dev.BASE.full_metrics(tf,ts)
                    passed=ev.guard(bmet,bm) and cross.wf_guard(tmet,tm)
                    grid.append({'ridge':ridge,'gamma':gamma,'pass':passed,**{m+'_'+k:v for m,d in bmet.items() for k,v in d.items()}})
                    if passed:
                        options.append((u.objective(bmet,gamma,1.)+(-ridge,),spec))
            dev.csv_write(out/'validation_grid.csv',pd.DataFrame(grid))
            st={'deployment':dep,'model_seed':ms,'eval_seed':es,'pass':bool(options)}
            if options:
                selected=min(options,key=lambda r:r[0])[1]
                dev.json_write(out/'SELECTED_CONFIG.json',selected)
                st.update(gamma=selected['gamma'],ridge=selected['ridge'])
            statuses.append(st)
            print(json.dumps(st),flush=True)
    dev.json_write(a.root/'contracts/VALIDATION_GATE.json',{'pass':all(s['pass'] for s in statuses),'seeds':statuses})
    if all(s['pass'] for s in statuses):
        u.evaluate(a.root,score_function=c.score)


if __name__=='__main__':
    main()
