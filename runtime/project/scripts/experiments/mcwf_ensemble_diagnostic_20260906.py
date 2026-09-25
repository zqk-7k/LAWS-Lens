#!/usr/bin/env python3
"""Shared predictive-distribution ensemble; explicitly a dependent pilot."""
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
import mcwf_unified_tail_20260906 as tail

ORIGINAL_FRAME=u.frame
GAMMAS=tuple(2.**k for k in range(-6,2))


def features(root,split):
    predictor=root/'predictor_cache'
    predictor.mkdir(exist_ok=True)
    if not (predictor/'models').exists():
        (predictor/'models').symlink_to(u.PREV/'models',target_is_directory=True)
    for dep in u.DEPS:
        for ms,es in zip(body.MODEL_SEEDS,dev.SEEDS):
            path=root/f'features/{dep}/seed_{es}/{split}.parquet'
            if path.exists():
                continue
            f=ORIGINAL_FRAME(dep,ms,es,split)
            probabilities=[]
            cosine=[]
            means=[]
            for member in body.MODEL_SEEDS:
                p,z=ev.input_prediction(predictor,dep,member,es,'RAW-PHASE-SOURCE',split)
                x=ev.features(f,p,z)
                probabilities.append(p)
                cosine.append(x['cosine'])
                means.append(p@tf.LOG_CENTERS)
            p=np.mean(probabilities,axis=0)
            mean=p@tf.LOG_CENTERS
            i,j=f.idx_i.to_numpy(int),f.idx_j.to_numpy(int)
            f=f.copy()
            f['new_encoder_cosine']=np.mean(cosine,axis=0)
            f['new_mass_similarity']=-np.abs(mean[i]-mean[j])
            f['new_mass_pred_logmc_i']=mean[i]
            f['new_mass_pred_logmc_j']=mean[j]
            f['new_mass_predictive_BC']=np.sqrt(np.maximum(p[i]*p[j],0)).sum(-1)
            epi=np.var(means,axis=0)
            f['new_mass_epistemic_variance_i']=epi[i]
            f['new_mass_epistemic_variance_j']=epi[j]
            path.parent.mkdir(parents=True,exist_ok=True)
            f.to_parquet(path,index=False)
            np.savez_compressed(path.with_suffix('.npz'),p=p,member_mean=np.asarray(means))
            print(json.dumps({'ensemble_features':dep,'split':split,'eval_seed':es}),flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    a=p.parse_args()
    if a.root.exists():
        raise RuntimeError('Independent output directory required')
    u.initialize(a.root)
    info={'code':'MCWF-ENSEMBLE-DIAGNOSTIC','members_per_run':body.MODEL_SEEDS,
        'combination':'arithmetic mean of aligned 64-bin predictive probability masses; never average non-aligned embedding vectors',
        'same_three_members_reused_for_each_baseline_deployment':True,
        'not_three_independent_ensemble_training_repeats':True,
        'cannot_satisfy_final_goal_by_itself':True,
        'rationale':'Separate between-model uncertainty from a single-model mass contradiction; no real PE model input',
        'reference':'https://arxiv.org/abs/1612.01474',
        'reference_limit':'deep ensembles motivate uncertainty aggregation, not proof of this GW application or threshold',
        'required_next_if_promising':'disjoint member groups for three independent training repetitions before new confirmation'}
    dev.json_write(a.root/'contracts/ENSEMBLE_DIAGNOSTIC.json',info)
    c=json.loads((a.root/'contracts/UNIFIED_METHOD.json').read_text())
    c.update(code=info['code'],shared_model='three RAW-PHASE-SOURCE predictors per run',
        waveform_recipe='Zwf=Zold-gamma*clip((-logBC_of_mean_predictive_p-q95_true)/IQR_true,0,6)',
        alpha=.05,gamma_grid=GAMMAS,
        independent_training_repeats=False,not_final_candidate=True)
    dev.json_write(a.root/'contracts/UNIFIED_METHOD.json',c)
    for path in (Path(__file__),Path(tail.__file__)):
        shutil.copy2(path,a.root/'scripts'/path.name)
    features(a.root,'validation')
    u.frame=lambda dep,ms,es,split:pd.read_parquet(a.root/f'features/{dep}/seed_{es}/{split}.parquet')
    statuses=[]
    for dep in u.DEPS:
        for ms,es in zip(body.MODEL_SEEDS,dev.SEEDS):
            f=u.frame(dep,ms,es,'validation')
            x=-np.log(np.maximum(f.new_mass_predictive_BC.to_numpy(float),1e-12))
            true=x[f.is_true_pair.to_numpy(bool)]
            spec={'recipe':'companion_tail_negative_only','threshold':float(np.quantile(true,.95)),
                'scale':max(float(np.subtract(*np.quantile(true,[.75,.25]))),.05),
                'validation_max':float(x.max()),'alpha':.05,'mass_weight':1.,'model_seed':ms,'eval_seed':es,
                'deployment':dep,'neutral_floor':False,'ensemble_members':body.MODEL_SEEDS}
            base=ev.metrics(f,f.previous_waveform_score.to_numpy(),dep,es)
            options=[]
            grid=[]
            for gamma in GAMMAS:
                option={**spec,'gamma':gamma}
                s,*_=tail.score(f,option)
                m=ev.metrics(f,s,dep,es)
                good=ev.guard(m,base)
                grid.append({'gamma':gamma,'pass':good,**{method+'_'+k:v for method,mm in m.items() for k,v in mm.items()}})
                if good:
                    options.append((u.objective(m,gamma,1.),option))
            out=a.root/f'calibration/{dep}/seed_{es}'
            out.mkdir(parents=True,exist_ok=True)
            dev.csv_write(out/'validation_joint_grid.csv',pd.DataFrame(grid))
            st={'deployment':dep,'model_seed':ms,'eval_seed':es,'status':'PASS' if options else 'FAIL_NO_ENSEMBLE_CORRECTION','eligible':len(options)}
            if options:
                selected=min(options,key=lambda q:q[0])[1]
                dev.json_write(out/'SELECTED_CONFIG.json',selected)
                st.update(gamma=selected['gamma'],mass_weight=1.,selected_sha256=dev.sha(out/'SELECTED_CONFIG.json'))
            dev.json_write(out/'FIT_STATUS.json',st)
            statuses.append(st)
            print(json.dumps(st),flush=True)
    passed=all(r['status']=='PASS' for r in statuses)
    dev.json_write(a.root/'contracts/VALIDATION_GATE.json',{'pass':passed,'seeds':statuses})
    dev.csv_write(a.root/'tables/VALIDATION_SELECTION.csv',pd.DataFrame(statuses))
    if passed:
        for split in ('test','real'):
            features(a.root,split)
        u.evaluate(a.root,score_function=tail.score)
        dev.json_write(a.root/'contracts/FINAL_ELIGIBILITY.json',{'eligible_for_final_delivery':False,'reason':'dependent ensemble diagnostic requires independent ensemble replication'})


if __name__=='__main__':
    main()
