#!/usr/bin/env python3
"""Finite-reference predictive-mass tail correction, not a physical Bayes factor."""
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
import mcwf_development_20260905 as dev
import mcwf_encoder_body_20260906 as body
import mcwf_encoder_evaluate_20260906 as ev
import mcwf_unified_waveform_20260906 as u
import mcwf_evaluate_finelag_20260907 as evaluate
import mcwf_finelag_eventpsd_20260907 as psd


def tail_probability(x,reference):
    reference=np.sort(np.asarray(reference,float))
    if len(reference)==0 or not np.isfinite(reference).all():
        raise ValueError('Finite nonempty reference required')
    count=len(reference)-np.searchsorted(reference,np.asarray(x,float),side='left')
    return (1.+count)/(len(reference)+1.)


def score(f,spec):
    x=-np.log(np.maximum(f.new_mass_predictive_BC.to_numpy(float),1e-12))
    p=tail_probability(x,spec['reference'])
    penalty=np.minimum(np.log(p/spec['alpha']),0.)
    old=f.previous_waveform_score.to_numpy(float)
    return old+spec['gamma']*penalty,penalty,x,x>max(spec['reference'])


def tests():
    r=np.array([0.,1.,1.,2.])
    x=np.array([-1.,0.,1.,1.5,2.,3.,100.])
    p=tail_probability(x,r)
    expected=np.array([1.,1.,.8,.4,.4,.2,.2])
    assert np.allclose(p,expected,atol=0,rtol=1e-15)
    assert np.all(np.diff(p)<=0)
    assert np.array_equal(p,tail_probability(x,r[::-1]))
    return {'pass':True,'ties_conservative':True,'permutation_invariant':True,
        'tail_probability_lower_bound':'1/(n_reference+1)','extreme_magnitude_does_not_extrapolate':True}


def run(root,trained):
    if root.exists():
        raise RuntimeError('Independent result directory required')
    u.initialize(root)
    cfg=json.loads((root/'contracts/UNIFIED_METHOD.json').read_text())
    tc=json.loads((trained/'contracts/FINE_LAG_TRAINING.json').read_text())
    cfg.update(code='MCWF-FINITE-REFERENCE-MASS-TAIL',created_utc=datetime.now(timezone.utc).isoformat(),
        waveform_recipe='Zold + gamma*min(log(p_tail/0.05),0)',
        new_score='p_tail=(1+number of reference companion disagreements>=observed)/(n_reference+1)',
        reference='one predictive-BC disagreement per validation companion system; no real PE',
        alpha=.05,gamma_grid=[2.**k for k in range(-6,2)],
        changed_mechanism='replace IQR-scaled extreme-tail extrapolation with bounded finite-reference tail rank; no new classifier or event-specific hand edit',
        rationale='A finite companion calibration sample cannot justify arbitrarily increasing contradiction strength far beyond its observed tail.',
        statistical_limit='Conformal-style smoothed tail rank, NOT a valid reported FPP or guaranteed coverage test: validation also selects gamma and exchangeability with real events is unproven.',
        additional_reference='https://arxiv.org/abs/2107.07511',
        training_objective=tc['objective'],waveform_feature_implementation=psd.IMPLEMENTATION,
        numerical_tests=tests())
    dev.json_write(root/'contracts/UNIFIED_METHOD.json',cfg)
    dev.json_write(root/'contracts/DISTILLED_MODEL_PROVENANCE.json',{
        'training_root':str(trained),'feature_implementation':psd.IMPLEMENTATION,
        'training_contract_sha256':dev.sha(trained/'contracts/FINE_LAG_TRAINING.json'),
        'new_training_in_calibration':False,'six_new_encoders_previously_trained':True,
        'training_objective':tc['objective'],'not_distillation':'legacy provenance filename only'})
    evaluate.ev.input_prediction=psd.input_prediction
    evaluate.IMPLEMENTATION=psd.IMPLEMENTATION
    evaluate.features(trained,'validation')
    u.PREV=trained
    statuses=[]
    for dep in u.DEPS:
        for ms,es in zip(body.MODEL_SEEDS,dev.SEEDS):
            out=root/f'calibration/{dep}/seed_{es}'
            out.mkdir(parents=True)
            f=u.frame(dep,ms,es,'validation')
            y=f.is_true_pair.to_numpy(bool)
            reference=-np.log(f.new_mass_predictive_BC.to_numpy(float)[y].clip(1e-12))
            base=ev.metrics(f,f.previous_waveform_score.to_numpy(float),dep,es)
            rows,options=[],[]
            for gamma in cfg['gamma_grid']:
                spec={'recipe':'finite_reference_companion_tail','reference':np.sort(reference).tolist(),
                    'alpha':.05,'gamma':gamma,'mass_weight':1.,'model_seed':ms,'eval_seed':es,'deployment':dep,
                    'validation_min':0.,'validation_max':float(reference.max()),
                    'reference_frame_sha256':dev.sha(trained/f'evaluation/RAW-PHASE-SOURCE/{dep}/model_{ms}_eval_{es}/validation_NEW-PHYSICAL_pairs.parquet')}
                s,penalty,_,_=score(f,spec)
                met=ev.metrics(f,s,dep,es)
                passed=ev.guard(met,base)
                rows.append({'gamma':gamma,'pass':passed,'reference_systems':len(reference),
                    'minimum_possible_penalty':float(min(np.log(1/(len(reference)+1)/.05),0)),
                    'true_penalty_fraction':float((penalty[y]<0).mean()),
                    **{method+'_'+k:v for method,m in met.items() for k,v in m.items()}})
                if passed:
                    options.append((u.objective(met,gamma,.05),spec))
            dev.csv_write(out/'validation_joint_grid.csv',pd.DataFrame(rows))
            st={'deployment':dep,'model_seed':ms,'eval_seed':es,'status':'PASS' if options else 'FAIL_NO_NONZERO_SHARED_RECIPE','eligible':len(options)}
            if options:
                selected=min(options,key=lambda x:x[0])[1]
                dev.json_write(out/'SELECTED_CONFIG.json',selected)
                st.update(selected_sha256=dev.sha(out/'SELECTED_CONFIG.json'),gamma=selected['gamma'],mass_weight=1.)
            statuses.append(st)
            dev.json_write(out/'FIT_STATUS.json',st)
            print(json.dumps(st),flush=True)
    dev.csv_write(root/'tables/VALIDATION_SELECTION.csv',pd.DataFrame(statuses))
    passed=all(s['status']=='PASS' for s in statuses)
    dev.json_write(root/'contracts/VALIDATION_GATE.json',{'pass':passed,'seeds':statuses})
    shutil.copy2(__file__,root/'scripts'/Path(__file__).name)
    if passed:
        for split in ('test','real'):
            evaluate.features(trained,split)
        u.evaluate(root,score_function=score)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--trained-root',type=Path,required=True)
    a=p.parse_args();run(a.root,a.trained_root)
