#!/usr/bin/env python3
"""Two simulation validation domains; shared formula, no PE selection."""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import torch
import mcwf_development_20260905 as dev
import mcwf_encoder_body_20260906 as body
import mcwf_encoder_evaluate_20260906 as ev
import mcwf_fresh_confirmation_20260906 as fresh_old
import mcwf_unified_waveform_20260906 as u
import mcwf_unified_tail_20260906 as tail

GAMMAS=tuple(2.**k for k in range(-6,2))


def auxiliary(root,dep,ms,es):
    out=root/f'auxiliary_validation/{dep}/seed_{es}'
    path=out/'pairs.parquet'
    if path.exists():
        return pd.read_parquet(path)
    out.mkdir(parents=True,exist_ok=True)
    cache=u.PREV/f'cache/{dep}'
    meta=pd.read_parquet(cache/'validation_metadata.parquet')
    train=pd.read_parquet(cache/'train_metadata.parquet')
    if set(meta.waveform_parent_uid)&set(train.waveform_parent_uid):
        raise RuntimeError('Auxiliary validation source leakage')
    raw=np.load(cache/'validation_raw2s.npy',mmap_mode='r')
    checkpoint=u.PREV/f'models/RAW-PHASE-SOURCE/{dep}/seed_{ms}/validation_selected_model.pt'
    payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
    x=np.load(body.PREVIOUS/f'cache/phasebank/{dep}/validation_features.npy')
    x=(x-payload['mu'])/payload['sd']
    model=body.Encoder('RAW-PHASE-SOURCE').cuda()
    model.load_state_dict(payload['model'])
    logits,z=body.infer(model,x,raw)
    del model
    lp=logits.astype(float)/payload['temperature']
    lp-=np.logaddexp.reduce(lp,axis=-1,keepdims=True)
    p=np.exp(lp)
    i,j=np.triu_indices(len(meta),1)
    y=meta.waveform_parent_uid.to_numpy()[i]==meta.waveform_parent_uid.to_numpy()[j]
    f=pd.DataFrame({'idx_i':i,'idx_j':j,'is_true_pair':y,'true_pair_family':np.where(y,meta.family_slot.str.upper().to_numpy()[i],'NONE'),'event_count':len(meta)})
    f=fresh_old.old_waveform(dep,es,raw,f)
    f['previous_waveform_score']=f.waveform_score
    features=ev.features(f,p,z)
    for name,key in (('new_encoder_cosine','cosine'),('new_mass_similarity','mass'),('new_mass_predictive_BC','predictive_BC'),('new_mass_pred_logmc_i','mean_i'),('new_mass_pred_logmc_j','mean_j')):
        f[name]=features[key]
    f.to_parquet(path,index=False)
    meta.to_parquet(out/'event_metadata.parquet',index=False)
    np.savez_compressed(out/'predictions.npz',p=p,z=z)
    dev.json_write(out/'PROVENANCE.json',{'source_groups':meta.waveform_parent_uid.nunique(),
        'train_source_overlap':0,'events':len(meta),'checkpoint_sha256':dev.sha(checkpoint),
        'previous_use':'simulation validation used for encoder epoch/temperature, now explicitly also for score robustness, never called a new blind test',
        'legacy_O3_noise':'mixed O1/O2/O3; does not replace run-matched fresh confirmation',
        'same_mass_labels_as_training':False,'same_generation_distribution_as_training':True})
    return f


def wf_guard(m,b):
    return bool(m['macro_r_at_10']>=b['macro_r_at_10']-.02 and m['average_precision']>=b['average_precision']-.005
        and m['false_at_recall_0p5']<=1.1*b['false_at_recall_0p5'] and m['false_at_recall_0p9']<=1.1*b['false_at_recall_0p9'])


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--neutral-floor',action='store_true')
    p.add_argument('--trained-root',type=Path)
    p.add_argument('--source-root',type=Path)
    a=p.parse_args()
    if a.root.exists():
        raise RuntimeError('Use a new independent output directory')
    u.initialize(a.root)
    if bool(a.trained_root)!=bool(a.source_root):
        raise ValueError('Both trained-root and independently documented auxiliary source-root are required')
    if a.trained_root:
        u.PREV=a.trained_root
    contract=json.loads((a.root/'contracts/UNIFIED_METHOD.json').read_text())
    contract.update(code='MCWF-UNIFIED-CROSS-DOMAIN-TAIL',
        waveform_recipe='Zwf=Zold-gamma*clip((-logBCpred-q95_BAY_true)/IQR_BAY_true,0,6)',
        alpha=.05,gamma_grid=GAMMAS,
        validation_domains=['existing BAYESTAR retained validation','240 source-group/480 event representation-development validation'],
        new_guard='In addition to original BAY waveform/C-fixed guards, require identical waveform-only guards on auxiliary simulation validation. Keep old deterministic objective among eligible candidates.',
        purpose='Counter small-catalog-specific correction selection with a second source-disjoint simulation domain, not real PE tuning.',
        auxiliary_reuse='already used for epoch/temperature; development only, no claim of nested or independent confirmation',
        calibration='threshold/scaling still frozen on BAY validation, extra domain is a robustness constraint only')
    if a.neutral_floor:
        contract.update(code='MCWF-UNIFIED-CROSS-DOMAIN-WITHDRAWAL',
            waveform_recipe='Zwf=min(Zold,max(0,Zold-gamma*clip((-logBCpred-q95_BAY_true)/IQR_BAY_true,0,6)))',
            rationale='Only withdraw positive old support; do not add unvalidated negative evidence or change old-negative pairs. This directly controls damage to the low-support recall tail, using the natural neutral log-evidence boundary.',
            neutral_floor=True)
    dev.json_write(a.root/'contracts/UNIFIED_METHOD.json',contract)
    if a.trained_root:
        provenance=json.loads((a.trained_root/'contracts/FINE_LAG_TRAINING.json').read_text())
        contract.update(trained_root=str(a.trained_root),auxiliary_source_root=str(a.source_root),
            feature_implementation=provenance.get('feature_implementation','sample_resolved_zero_padded_fft_v1'),
            same_scoring_rule_both_runs=True)
        dev.json_write(a.root/'contracts/UNIFIED_METHOD.json',contract)
        dev.json_write(a.root/'contracts/DISTILLED_MODEL_PROVENANCE.json',{'trained_root':str(a.trained_root),
            'distillation':False,'feature_implementation':contract['feature_implementation'],
            'compatibility_filename_only':True})
    for path in (Path(__file__),Path(tail.__file__)):
        shutil.copy2(path,a.root/'scripts'/path.name)
    statuses=[]
    for dep in u.DEPS:
        for ms,es in zip(body.MODEL_SEEDS,dev.SEEDS):
            f=u.frame(dep,ms,es,'validation')
            extra=(pd.read_parquet(a.source_root/dep/f'seed_{es}/pairs.parquet')
                   if a.source_root else auxiliary(a.root,dep,ms,es))
            eb=dev.BASE.full_metrics(extra,extra.previous_waveform_score.to_numpy())
            x=-np.log(np.maximum(f.new_mass_predictive_BC.to_numpy(float),1e-12))
            true=x[f.is_true_pair.to_numpy(bool)]
            spec={'recipe':'companion_tail_negative_only','alpha':.05,'threshold':float(np.quantile(true,.95)),
                'scale':max(float(np.subtract(*np.quantile(true,[.75,.25]))),.05),'validation_max':float(x.max()),
                'mass_weight':1.,'deployment':dep,'eval_seed':es,'model_seed':ms,'neutral_floor':a.neutral_floor}
            if a.neutral_floor:
                spec['recipe']='companion_tail_withdraw_positive_support'
            base=ev.metrics(f,f.previous_waveform_score.to_numpy(),dep,es)
            options=[]
            grid=[]
            for gamma in GAMMAS:
                option={**spec,'gamma':gamma}
                s,*_=tail.score(f,option)
                ex,*_=tail.score(extra,option)
                met=ev.metrics(f,s,dep,es)
                em=dev.BASE.full_metrics(extra,ex)
                passed=ev.guard(met,base) and wf_guard(em,eb)
                grid.append({'gamma':gamma,'pass':passed,'bay_guard_pass':ev.guard(met,base),'auxiliary_guard_pass':wf_guard(em,eb),
                    **{m+'_'+k:v for m,mm in met.items() for k,v in mm.items()},**{'auxiliary_'+k:v for k,v in em.items()}})
                if passed:
                    options.append((u.objective(met,gamma,1.),option))
            out=a.root/f'calibration/{dep}/seed_{es}'
            out.mkdir(parents=True,exist_ok=True)
            dev.csv_write(out/'validation_joint_grid.csv',pd.DataFrame(grid))
            dev.json_write(out/'auxiliary_baseline_metrics.json',eb)
            st={'deployment':dep,'model_seed':ms,'eval_seed':es,'status':'PASS' if options else 'FAIL_TWO_VALIDATION_DOMAINS','eligible':len(options)}
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
        u.evaluate(a.root,score_function=tail.score)


if __name__=='__main__':
    main()
