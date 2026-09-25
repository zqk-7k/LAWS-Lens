#!/usr/bin/env python3
"""Uncertainty-normalized simulated-mass contradiction, shared across runs."""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import mcwf_development_20260905 as dev
import mcwf_encoder_body_20260906 as body
import mcwf_encoder_evaluate_20260906 as ev
import mcwf_mass_tf_20260905 as tf
import mcwf_unified_waveform_20260906 as u

ORIGINAL_FRAME=u.frame
ALPHA=.05
GAMMAS=(.25,.5,1.,2.)


def add_uncertainty(frame,p):
    mean=p@tf.LOG_CENTERS
    var=np.maximum(p@(tf.LOG_CENTERS**2)-mean**2,0.)
    f=frame.copy()
    i,j=f.idx_i.to_numpy(int),f.idx_j.to_numpy(int)
    f['new_mass_log_variance_i']=var[i]
    f['new_mass_log_variance_j']=var[j]
    f['new_mass_standardized_distance']=np.abs(mean[i]-mean[j])/np.sqrt(np.maximum(var[i]+var[j],1e-6))
    return f


def score(f,spec):
    x=f.new_mass_standardized_distance.to_numpy(float)
    penalty=-np.clip((x-spec['threshold'])/spec['scale'],0.,6.)
    s=f.previous_waveform_score.to_numpy(float)+spec['gamma']*penalty
    return s,penalty,x,x>spec['validation_max']


def features(root,trained,split):
    for dep in u.DEPS:
        for ms,es in zip(body.MODEL_SEEDS,dev.SEEDS):
            path=root/f'features/{dep}/seed_{es}/{split}.parquet'
            if path.exists():
                continue
            p,z=ev.input_prediction(trained,dep,ms,es,'RAW-PHASE-SOURCE',split)
            f=ORIGINAL_FRAME(dep,ms,es,split)
            f=add_uncertainty(f,p)
            path.parent.mkdir(parents=True,exist_ok=True)
            f.to_parquet(path,index=False)


def fit(root,trained):
    u.initialize(root)
    definition=json.loads((root/'contracts/UNIFIED_METHOD.json').read_text())
    definition.update(code='MCWF-UNIFIED-PREDICTIVE-UNCERTAINTY',
        waveform_recipe='Zwf=Zold-gamma*clip((Dpred-q95_true)/IQR_true,0,6)',
        new_score='Dpred=abs(ElogMc_i-ElogMc_j)/sqrt(VarlogMc_i+VarlogMc_j)',
        variance_floor=1e-6,scale_floor=.05,alpha=ALPHA,gamma_grid=GAMMAS,
        rationale='Mass-distribution BC also responds to different uncertainty widths. Test a mean-disagreement statistic scaled by both simulated predictive uncertainties, preserving broad low-SNR uncertainty.',
        interpretation='Neural predictive uncertainty is NOT a public PE posterior; Dpred is an empirically calibrated ranking diagnostic, not a Gaussian significance or physical Bayes factor.',
        selection='same validation-only paired guardrails/priority in both runs; fixed alpha5%, no PE-based threshold',
        real_audit='previously inspected development catalog, never a blind confirmation')
    dev.json_write(root/'contracts/UNIFIED_METHOD.json',definition)
    shutil.copy2(__file__,root/'scripts'/Path(__file__).name)
    statuses=[]
    for dep in u.DEPS:
        for ms,es in zip(body.MODEL_SEEDS,dev.SEEDS):
            f=pd.read_parquet(root/f'features/{dep}/seed_{es}/validation.parquet')
            x=f.new_mass_standardized_distance.to_numpy(float)
            true=x[f.is_true_pair.to_numpy(bool)]
            spec={'recipe':'uncertainty_companion_tail','threshold':float(np.quantile(true,1-ALPHA)),
                'scale':max(float(np.subtract(*np.quantile(true,[.75,.25]))),.05),
                'validation_max':float(x.max()),'mass_weight':1.,'alpha':ALPHA,
                'model_seed':ms,'eval_seed':es,'deployment':dep}
            base=ev.metrics(f,f.previous_waveform_score.to_numpy(),dep,es)
            grid=[]
            options=[]
            for gamma in GAMMAS:
                s={**spec,'gamma':gamma}
                candidate,*_=score(f,s)
                m=ev.metrics(f,candidate,dep,es)
                good=ev.guard(m,base)
                grid.append({'gamma':gamma,'pass':good,**{method+'_'+k:v for method,mm in m.items() for k,v in mm.items()}})
                if good:
                    options.append((u.objective(m,gamma,1.),s))
            out=root/f'calibration/{dep}/seed_{es}'
            out.mkdir(parents=True,exist_ok=True)
            dev.csv_write(out/'validation_joint_grid.csv',pd.DataFrame(grid))
            st={'deployment':dep,'model_seed':ms,'eval_seed':es,'status':'PASS' if options else 'FAIL_NO_NONZERO_SHARED_RECIPE','eligible':len(options)}
            if options:
                selected=min(options,key=lambda q:q[0])[1]
                dev.json_write(out/'SELECTED_CONFIG.json',selected)
                st.update(gamma=selected['gamma'],mass_weight=1.,selected_sha256=dev.sha(out/'SELECTED_CONFIG.json'))
            dev.json_write(out/'FIT_STATUS.json',st)
            statuses.append(st)
            print(json.dumps(st),flush=True)
    dev.csv_write(root/'tables/VALIDATION_SELECTION.csv',pd.DataFrame(statuses))
    dev.json_write(root/'contracts/VALIDATION_GATE.json',{'pass':all(r['status']=='PASS' for r in statuses),'seeds':statuses})


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    a=p.parse_args()
    if a.root.exists():
        raise RuntimeError('Independent new output required')
    trained=u.PREV
    u.initialize(a.root)
    features(a.root,trained,'validation')
    fit(a.root,trained)
    if not json.loads((a.root/'contracts/VALIDATION_GATE.json').read_text())['pass']:
        return
    for split in ('test','real'):
        features(a.root,trained,split)
    u.frame=lambda dep,ms,es,split:pd.read_parquet(a.root/f'features/{dep}/seed_{es}/{split}.parquet')
    u.evaluate(a.root,score_function=score)


if __name__=='__main__':
    main()
