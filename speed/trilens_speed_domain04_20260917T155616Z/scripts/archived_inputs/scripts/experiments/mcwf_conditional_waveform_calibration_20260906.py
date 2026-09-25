#!/usr/bin/env python3
"""Small monotone waveform-only calibration with source/noise separation."""
import argparse
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='4'
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import shutil
import joblib
import numpy as np
import pandas as pd
import sklearn
from threadpoolctl import threadpool_limits
from sklearn.ensemble import HistGradientBoostingClassifier
import mcwf_development_20260905 as dev
import mcwf_encoder_body_20260906 as body
import mcwf_encoder_evaluate_20260906 as ev
import mcwf_unified_waveform_20260906 as u
import mcwf_cross_domain_selection_20260906 as cross

SOURCE=dev.PROJECT/'results/mcwf_unified_cross_domain_tail_20260906/auxiliary_validation'
GAMMAS=(.0625,.125,.25,.5,1.)
COLUMNS=['previous_waveform_score','new_encoder_cosine','new_mass_similarity','new_mass_predictive_BC']


EXTENDED_COLUMNS=COLUMNS+['waveform_embedding_cosine','negative_old_mass_gap','negative_old_q_gap']


def design(frame,columns=None):
    if columns is None:
        columns=COLUMNS
    values=frame.assign(negative_old_mass_gap=-frame.waveform_abs_delta_logmc_std,
                        negative_old_q_gap=-frame.waveform_abs_delta_logitq_std)
    return values[columns].to_numpy(float)


@lru_cache(maxsize=12)
def load_model(path,digest):
    if dev.sha(Path(path))!=digest:
        raise RuntimeError('Frozen calibration changed')
    return joblib.load(path)


def score(f,spec):
    x=design(f,spec.get('features',COLUMNS))
    bounds=np.asarray(spec['bounds'])
    ood=((x<bounds[0])|(x>bounds[1])).any(1)
    model=load_model(spec['calibrator_path'],spec['calibrator_sha256'])
    clipped=np.clip(x,bounds[0],bounds[1])
    if isinstance(model,dict) and model.get('kind')=='nonnegative_logistic':
        unbounded=(clipped-model['center'])/model['scale']@model['coef']+model['intercept']
    else:
        unbounded=model.decision_function(clipped)
    pred=np.clip(unbounded,-6.,6.)
    old=f.previous_waveform_score.to_numpy(float)
    delta=pred-old
    if spec.get('residual_mode')=='two_sided_support':
        delta=np.where(ood & (delta>0),0.,delta)
        return old+spec['gamma']*delta,pred,delta,ood
    delta=np.minimum(delta,0.)
    if spec.get('residual_mode')=='neutral_mixture':
        residual=np.minimum(np.logaddexp(np.log1p(-spec['gamma']),np.log(spec['gamma'])+delta),0.)
        return old+residual,pred,residual,ood
    return old+spec['gamma']*delta,pred,delta,ood


def subset(frame,mask):
    idx=np.flatnonzero(mask)
    remap=np.full(len(mask),-1,dtype=int)
    remap[idx]=np.arange(len(idx))
    keep=mask[frame.idx_i.to_numpy(int)]&mask[frame.idx_j.to_numpy(int)]
    sub=frame.loc[keep].copy().reset_index(drop=True)
    sub['idx_i']=remap[sub.idx_i.to_numpy(int)]
    sub['idx_j']=remap[sub.idx_j.to_numpy(int)]
    sub['event_count']=len(idx)
    return sub


def split_metadata(meta,dep):
    noise=np.where(meta.image.eq('a'),meta.a_noise_bank_index,meta.b_noise_bank_index).astype(int)
    group=meta.waveform_parent_uid.astype(str)
    fit_noise=np.array([int(hashlib.sha256(f'{dep}:cal-noise:{k}:202609070'.encode()).hexdigest()[:8],16)%3!=0 for k in noise])
    fit=np.zeros(len(meta),dtype=bool)
    tune=np.zeros(len(meta),dtype=bool)
    for uid in group.unique():
        indices=np.flatnonzero(group.eq(uid))
        if len(indices)!=2:
            raise RuntimeError('Expected one two-view source in auxiliary validation')
        if fit_noise[indices].all():
            fit[indices]=True
        elif (~fit_noise[indices]).all():
            tune[indices]=True
    if set(noise[fit])&set(noise[tune]) or set(group[fit])&set(group[tune]):
        raise RuntimeError('Calibration/selection source or noise overlap')
    return fit,tune,noise


def main():
    global SOURCE
    threading_controller=threadpool_limits(limits=4)
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--max-gamma',type=float,choices=(.25,.5,1.),default=1.)
    p.add_argument('--old-components',action='store_true')
    p.add_argument('--source-root',type=Path)
    p.add_argument('--trained-root',type=Path)
    p.add_argument('--two-sided',action='store_true')
    a=p.parse_args()
    if a.root.exists():
        raise RuntimeError('Independent output required')
    u.initialize(a.root)
    if a.source_root:
        SOURCE=a.source_root
    if a.trained_root:
        if not (a.trained_root/'contracts/ALL_SIX_TRAINED.json').exists():
            raise RuntimeError('All six fine-lag training runs must finish')
        u.PREV=a.trained_root
        implementation=json.loads((a.trained_root/'contracts/FINE_LAG_TRAINING.json').read_text()).get('feature_implementation','sample_resolved_zero_padded_fft_v1')
        dev.json_write(a.root/'contracts/DISTILLED_MODEL_PROVENANCE.json',{
            'training_root':str(a.trained_root),'training_contract_sha256':dev.sha(a.trained_root/'contracts/FINE_LAG_TRAINING.json'),
            'feature_implementation':implementation,'six_models_newly_trained':True,'new_epochs_each':50,
            'not_distillation':'legacy metadata name; CE+SupCon training'})
    contract=json.loads((a.root/'contracts/UNIFIED_METHOD.json').read_text())
    columns=EXTENDED_COLUMNS if a.old_components else COLUMNS
    contract.update(code='MCWF-CONDITIONAL-MONOTONE-CALIBRATION',
        waveform_recipe='Zwf=Zold+gamma*min(clip(f_cal(oldZ,newcos,-abs(delta_pred_logMc),BCpred),-6,6)-Zold,0)',
        features=columns,monotonic_directions=[1]*len(columns),
        interpretation='balanced-class discriminative ranking calibration; neither posterior lens probability nor independent proper Bayes factor; only waveform-derived inputs',
        calibrator={'class':'HistGradientBoostingClassifier','max_iter':100,'learning_rate':.05,'max_leaf_nodes':7,'min_samples_leaf':20,'l2_regularization':10.,'early_stopping':False},
        sklearn_version=sklearn.__version__,gamma_grid=[g for g in GAMMAS if g<=a.max_gamma],
        source_noise_split='hash auxiliary-validation noise blocks into2/3 fit and1/3 tune; retain a source only if both of its views use blocks in the same partition; no cross-partition sources/noise',
        fit='auxiliary fit partition only, equal total class weights; each two-image source contributes one positive pair',
        selection='original BAY validation guardrails plus auxiliary tune waveform guardrails; same deterministic objective for both runs',
        additional_limitation='auxiliary sources previously used to select encoder epoch; calibration remains development, not independent confirmation',
        OOD='clip features to fit range; no positive residual rewards; report OOD explicitly, never interpret clipped extrapolation as stronger physical evidence',
        reference='https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html')
    if a.max_gamma==.5:
        contract.update(code='MCWF-CONDITIONAL-MONOTONE-HALF',
            rationale='The unrestricted gamma1 option can wholly replace old waveform support based on a small calibration sample. Retain at least half of old evidence in both runs; this is a declared shrinkage control, not a physical mixing probability.',
            max_new_calibration_fraction=.5)
    if a.max_gamma==.25:
        contract.update(code='MCWF-CONDITIONAL-MONOTONE-QUARTER',
            rationale='Half-strength calibration still fails one reused O3 high-recall false-burden check. Test a common quarter-maximum residual in a separately archived development ablation; no claim of independent validation from the reused catalog.',
            max_new_calibration_fraction=.25)
    if a.old_components:
        contract.update(code='MCWF-CONDITIONAL-MONOTONE-COMPONENTS',
            rationale='The scalar old waveform score discards whether old support came from morphology or intrinsic regression. Preserve its separate waveform-only components so calibration can learn when a new mass-head contradiction is unsupported by the old waveform estimates; no PE features or labels enter calibration.')
    if a.trained_root:
        contract.update(code='MCWF-FINELAG-CONDITIONAL',new_training_this_round=True,
            waveform_feature_implementation=implementation,training_root=str(a.trained_root),
            auxiliary_source_root=str(SOURCE))
    if a.two_sided:
        contract.update(code='MCWF-CONDITIONAL-TWO-SIDED-SUPPORTED',
            waveform_recipe='Zwf=Zold+gamma*(clip(f_cal,-6,6)-Zold); positive residual=0 outside frozen fit-feature support',
            rationale='One-sided corrections cannot recover independently supported true companions whose old waveform evidence is weak. Test calibrated support and contradiction together, retaining the same training fit/tune isolation, half-maximum blend and unchanged guardrails.',
            positive_OOD_rule='no new positive support outside fit bounds',
            new_training_this_round=False,trained_waveform_models_reused=True)
    dev.json_write(a.root/'contracts/UNIFIED_METHOD.json',contract)
    shutil.copy2(__file__,a.root/'scripts'/Path(__file__).name)
    statuses=[]
    for dep in u.DEPS:
        for ms,es in zip(body.MODEL_SEEDS,dev.SEEDS):
            original=SOURCE/dep/f'seed_{es}'
            extra=pd.read_parquet(original/'pairs.parquet')
            meta=pd.read_parquet(original/'event_metadata.parquet')
            fitmask,tunemask,noise=split_metadata(meta,dep)
            fitframe=subset(extra,fitmask)
            tuneframe=subset(extra,tunemask)
            out=a.root/f'calibration/{dep}/seed_{es}'
            out.mkdir(parents=True,exist_ok=True)
            y=fitframe.is_true_pair.to_numpy(bool)
            nt=int(tuneframe.is_true_pair.to_numpy(bool).sum())
            splitinfo={'fit_sources':int(y.sum()),'tune_sources':nt,'fit_events':int(fitmask.sum()),'tune_events':int(tunemask.sum()),
                'source_overlap':0,'noise_block_overlap':0,'fit_blocks':sorted(set(noise[fitmask].tolist())),'tune_blocks':sorted(set(noise[tunemask].tolist())),
                'source_input_sha256':dev.sha(original/'pairs.parquet')}
            dev.json_write(out/'SPLIT_AUDIT.json',splitinfo)
            meta.assign(calibration_fit=fitmask,calibration_tune=tunemask,calibration_noise_index=noise).to_parquet(out/'source_noise_partition.parquet',index=False)
            if y.sum()<30 or nt<10:
                st={'deployment':dep,'model_seed':ms,'eval_seed':es,'status':'FAIL_INSUFFICIENT_SOURCE_NOISE_SUPPORT','eligible':0}
                statuses.append(st)
                dev.json_write(out/'FIT_STATUS.json',st)
                continue
            model=HistGradientBoostingClassifier(max_iter=100,learning_rate=.05,max_leaf_nodes=7,min_samples_leaf=20,
                l2_regularization=10.,early_stopping=False,monotonic_cst=[1]*len(columns),random_state=ms)
            x=design(fitframe,columns)
            weights=np.where(y,len(y)/(2*y.sum()),len(y)/(2*(~y).sum()))
            model.fit(x,y,sample_weight=weights)
            modelpath=out/'calibrator.joblib'
            joblib.dump(model,modelpath)
            spec={'recipe':'conditional_monotone_waveform','calibrator_path':str(modelpath),'calibrator_sha256':dev.sha(modelpath),
                'features':columns,
                'bounds':[x.min(0).tolist(),x.max(0).tolist()],'model_seed':ms,'eval_seed':es,'deployment':dep,'mass_weight':1.}
            if a.two_sided:
                spec['residual_mode']='two_sided_support'
            bay=u.frame(dep,ms,es,'validation')
            base=ev.metrics(bay,bay.previous_waveform_score.to_numpy(),dep,es)
            tb=dev.BASE.full_metrics(tuneframe,tuneframe.previous_waveform_score.to_numpy())
            options=[]
            grid=[]
            for gamma in GAMMAS:
                if gamma>a.max_gamma:
                    continue
                option={**spec,'gamma':gamma}
                s,*_=score(bay,option)
                ts,*_=score(tuneframe,option)
                m=ev.metrics(bay,s,dep,es)
                tm=dev.BASE.full_metrics(tuneframe,ts)
                good=ev.guard(m,base) and cross.wf_guard(tm,tb)
                grid.append({'gamma':gamma,'pass':good,'bay_pass':ev.guard(m,base),'auxiliary_tune_pass':cross.wf_guard(tm,tb),
                    **{method+'_'+k:v for method,mm in m.items() for k,v in mm.items()},**{'tune_'+k:v for k,v in tm.items()}})
                if good:
                    options.append((u.objective(m,gamma,1.),option))
            dev.csv_write(out/'validation_joint_grid.csv',pd.DataFrame(grid))
            st={'deployment':dep,'model_seed':ms,'eval_seed':es,'status':'PASS' if options else 'FAIL_CALIBRATOR_GUARDRAILS','eligible':len(options)}
            if options:
                selected=min(options,key=lambda q:q[0])[1]
                dev.json_write(out/'SELECTED_CONFIG.json',selected)
                st.update(gamma=selected['gamma'],mass_weight=1.,selected_sha256=dev.sha(out/'SELECTED_CONFIG.json'))
            dev.json_write(out/'FIT_STATUS.json',st)
            statuses.append(st)
            print(json.dumps(st),flush=True)
    passed=all(r['status']=='PASS' for r in statuses)
    dev.csv_write(a.root/'tables/VALIDATION_SELECTION.csv',pd.DataFrame(statuses))
    dev.json_write(a.root/'contracts/VALIDATION_GATE.json',{'pass':passed,'seeds':statuses})
    if passed:
        u.evaluate(a.root,score_function=score)


if __name__=='__main__':
    main()
