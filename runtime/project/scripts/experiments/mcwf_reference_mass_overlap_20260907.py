#!/usr/bin/env python3
"""Training-reference-adjusted predictive overlap, not a PE Bayes factor."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[k]='2'
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import mcwf_development_20260905 as dev
import mcwf_encoder_body_20260906 as body
import mcwf_encoder_evaluate_20260906 as ev
import mcwf_unified_waveform_20260906 as u

TRAINED=dev.PROJECT/'results/mcwf_unified_physical_mass_anchor_encoder_20260907'
GAMMAS=tuple(2.**k for k in range(-6,2))


def add_feature(f,p,prior):
    a=p[f.idx_i.to_numpy(int)]
    b=p[f.idx_j.to_numpy(int)]
    overlap=np.sum(a*b/np.asarray(prior),axis=-1)
    if not np.isfinite(overlap).all():
        raise RuntimeError('Invalid predictive overlap')
    f=f.copy()
    f['predictive_mass_log_reference_overlap']=np.log(np.maximum(overlap,1e-30))
    return f


def score(f,spec):
    raw=f.predictive_mass_log_reference_overlap.to_numpy(float)
    calibrated=dev.ORCH.PHYS.apply_score_likelihood_ratio(raw,spec['lookup']) if 'lookup' in spec else raw
    new=np.clip(calibrated,-6.,0.)
    if spec.get('neutral_mixture'):
        new=np.minimum(np.logaddexp(np.log(.5),np.log(.5)+new),0.)
    return f.previous_waveform_score.to_numpy(float)+spec['gamma']*new,new,raw,(raw<spec['validation_min'])|(raw>spec['validation_max'])


def prior_for(dep):
    meta=pd.read_parquet(TRAINED/f'cache/{dep}/train_metadata.parquet')
    target=np.load(body.PREVIOUS/f'cache/masstf/{dep}/train_targets.npy')
    group=meta.waveform_parent_uid.to_numpy()
    prior=np.stack([target[group==uid].mean(0) for uid in np.unique(group)]).mean(0)
    prior=prior.clip(1e-6);prior/=prior.sum()
    return prior


def main(root,calibrated=False,neutral_mixture=False):
    if root.exists():
        raise RuntimeError('Independent overlap output required')
    u.initialize(root)
    cfg=json.loads((root/'contracts/UNIFIED_METHOD.json').read_text())
    cfg.update(code='MCWF-PREDICTIVE-REFERENCE-OVERLAP',
        waveform_recipe='Zwf=Zold+gamma*clip(log sum_b p_i[b]*p_j[b]/pi_train[b],-6,0)',
        predictive_distribution='frozen equal neural/physical mass distribution, derived only from the2s waveform',
        prior='mean soft target over independent training source parents, floor1e-6 then normalize',
        gamma_grid=GAMMAS,
        neutral='If either predictive distribution equals the reference, overlap=1 and log overlap=0',
        rationale='Shape similarity alone does not distinguish informative conflict from two broad predictions. Account for the reference mass density before damping contradicted old support.',
        limitation='Predictive density is not a full PE posterior, physical component uses a finite grid and approximate profile, prior coherence is approximate. This is a ranking statistic inspired by posterior overlap, NOT a proper Bayes factor.',
        reference='https://arxiv.org/abs/1807.07062',
        selection='original validation-only point guardrails and deterministic objective, unchanged',
        new_training_this_round=False,training_root=str(TRAINED),feature_implementation='event_psd_sample_resolved_fft_v1')
    dev.json_write(root/'contracts/UNIFIED_METHOD.json',cfg)
    if calibrated:
        cfg.update(code='MCWF-CALIBRATED-REFERENCE-OVERLAP',
            waveform_recipe='Zwf=Zold+gamma*clip(KDE_log_density_ratio(predictive_log_reference_overlap),-6,0)',
            calibration='existing one-dimensional score KDE, bandwidth_scale1, true/noncompanion simulation validation only; fixed cap and gamma grid unchanged',
            rationale='A predictive reference overlap is not exact evidence. Estimate its actual companion/background distributions before combination instead of treating raw zero as a fully calibrated boundary.')
        dev.json_write(root/'contracts/UNIFIED_METHOD.json',cfg)
    if neutral_mixture:
        if not calibrated:
            raise ValueError('The uncertainty mixture is defined for the calibrated predictive density ratio')
        cfg.update(code='MCWF-ROBUST-CALIBRATED-REFERENCE-OVERLAP',
            waveform_recipe='Zwf=Zold+gamma*log(0.5+0.5*exp(clip(KDE_logLR(reference_overlap),-6,0)))',
            neutral_component=.5,
            rationale='Finite2s mass estimates and incomplete waveform template grids can be misspecified. A fixed half-neutral component bounds the damage from an overconfident mass conflict; retain full validation-only gamma selection and all previous guardrails.',
            interpretation='declared robust ranking ablation; half-neutral coefficient is an engineering assumption, not an estimated physical lensing probability',
            maximum_negative_increment_per_unit_gamma=float(np.log(2)))
        dev.json_write(root/'contracts/UNIFIED_METHOD.json',cfg)
    dev.json_write(root/'contracts/DISTILLED_MODEL_PROVENANCE.json',{'training_root':str(TRAINED),
        'feature_implementation':'event_psd_sample_resolved_fft_v1','not_distillation':True,
        'training_contract_sha256':dev.sha(TRAINED/'contracts/FINE_LAG_TRAINING.json')})
    shutil.copy2(__file__,root/'scripts'/Path(__file__).name)
    states=[]
    priors={dep:prior_for(dep) for dep in u.DEPS}
    for dep in u.DEPS:
        for ms,es in zip(body.MODEL_SEEDS,dev.SEEDS):
            old=TRAINED/f'evaluation/RAW-PHASE-SOURCE/{dep}/model_{ms}_eval_{es}'
            out=root/f'evaluation/RAW-PHASE-SOURCE/{dep}/model_{ms}_eval_{es}'
            out.mkdir(parents=True,exist_ok=True)
            f=pd.read_parquet(old/'validation_NEW-PHYSICAL_pairs.parquet')
            p=np.load(TRAINED/f'cache/predictions/RAW-PHASE-SOURCE/{dep}/model_{ms}_eval_{es}/validation.npz')['p']
            f=add_feature(f,p,priors[dep])
            f.to_parquet(out/'validation_NEW-PHYSICAL_pairs.parquet',index=False)
            values=f.predictive_mass_log_reference_overlap.to_numpy()
            spec={'recipe':'predictive_reference_overlap','prior':priors[dep].tolist(),'deployment':dep,
                'eval_seed':es,'model_seed':ms,'mass_weight':1.,'validation_min':float(values.min()),'validation_max':float(values.max())}
            if calibrated:
                positive=f.is_true_pair.to_numpy(bool)
                spec['lookup']=dev.ORCH.PHYS.fit_score_likelihood_ratio(values[positive],values[~positive],bandwidth_scale=1.)
            if neutral_mixture:
                spec['neutral_mixture']=True
            baseline=ev.metrics(f,f.previous_waveform_score.to_numpy(),dep,es)
            grid=[];options=[]
            for gamma in GAMMAS:
                option={**spec,'gamma':gamma}
                z,*_=score(f,option)
                m=ev.metrics(f,z,dep,es)
                good=ev.guard(m,baseline)
                grid.append({'gamma':gamma,'pass':good,**{method+'_'+k:v for method,mm in m.items() for k,v in mm.items()}})
                if good:
                    options.append((u.objective(m,gamma,1.),option))
            folder=root/f'calibration/{dep}/seed_{es}'
            folder.mkdir(parents=True)
            dev.csv_write(folder/'validation_joint_grid.csv',pd.DataFrame(grid))
            st={'deployment':dep,'model_seed':ms,'eval_seed':es,'eligible':len(options),'status':'PASS' if options else 'FAIL_REFERENCE_OVERLAP_GUARDS'}
            if options:
                selected=min(options,key=lambda q:q[0])[1]
                dev.json_write(folder/'SELECTED_CONFIG.json',selected)
                st.update(gamma=selected['gamma'],mass_weight=1.,selected_sha256=dev.sha(folder/'SELECTED_CONFIG.json'))
            dev.json_write(folder/'FIT_STATUS.json',st)
            states.append(st)
            print(json.dumps(st),flush=True)
    passed=all(s['status']=='PASS' for s in states)
    dev.json_write(root/'contracts/VALIDATION_GATE.json',{'pass':passed,'seeds':states})
    dev.csv_write(root/'tables/VALIDATION_SELECTION.csv',pd.DataFrame(states))
    if passed:
        for dep in u.DEPS:
            for ms,es in zip(body.MODEL_SEEDS,dev.SEEDS):
                for split in ('test','real'):
                    f=pd.read_parquet(TRAINED/f'evaluation/RAW-PHASE-SOURCE/{dep}/model_{ms}_eval_{es}/{split}_NEW-PHYSICAL_pairs.parquet')
                    p=np.load(TRAINED/f'cache/predictions/RAW-PHASE-SOURCE/{dep}/model_{ms}_eval_{es}/{split}.npz')['p']
                    f=add_feature(f,p,priors[dep])
                    f.to_parquet(root/f'evaluation/RAW-PHASE-SOURCE/{dep}/model_{ms}_eval_{es}/{split}_NEW-PHYSICAL_pairs.parquet',index=False)
        u.PREV=root
        u.evaluate(root,score_function=score)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--calibrated',action='store_true');p.add_argument('--neutral-mixture',action='store_true');a=p.parse_args();main(a.root,a.calibrated,a.neutral_mixture)
