#!/usr/bin/env python3
"""Validation source-bootstrap stability for one common residual recipe."""
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
import mcwf_unified_tail_20260906 as tail
import mcwf_summarize_20260905 as summarizer

REPEATS=500
GAMMAS=tuple(2.**k for k in range(-6,2))


def blocks(f,plan,seed):
    uid,index=np.unique(plan.system_id.astype(str),return_inverse=True)
    family=np.array([str(plan.iloc[np.flatnonzero(index==k)[0]].family) for k in range(len(uid))])
    counts=np.zeros((REPEATS,len(uid)),dtype=np.int16)
    rng=np.random.default_rng(seed)
    for fam in np.unique(family):
        ids=np.flatnonzero(family==fam)
        counts[:,ids]=rng.multinomial(len(ids),np.full(len(ids),1/len(ids)),size=REPEATS)
    return counts,index[f.idx_i.to_numpy(int)],index[f.idx_j.to_numpy(int)]


def metrics_boot(f,s,state):
    counts,i,j=state
    order=np.argsort(-s,kind='stable')
    y=f.is_true_pair.to_numpy(bool)[order]
    ends=np.r_[np.flatnonzero(np.diff(s[order])!=0),len(order)-1]
    output=[]
    for k in range(0,REPEATS,25):
        c=counts[k:k+25]
        w=c[:,i].astype(float)*c[:,j]
        w[:,i==j]=c[:,i[i==j]]
        w=w[:,order]
        tp=np.cumsum(w*y,1)
        fp=np.cumsum(w*~y,1)
        total=np.maximum(tp[:,-1],1.)
        te,fe=tp[:,ends],fp[:,ends]
        precision=np.divide(te,te+fe,out=np.zeros_like(te),where=te+fe>0)
        ap=(precision*np.diff(np.pad(te/total[:,None],((0,0),(1,0))),axis=1)).sum(1)
        f50=fp[np.arange(len(w)),np.argmax(tp>=.5*total[:,None],axis=1)]
        f90=fp[np.arange(len(w)),np.argmax(tp>=.9*total[:,None],axis=1)]
        output.append(np.stack((ap,f50,f90),-1))
    return np.concatenate(output)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--trained-root',type=Path)
    a=p.parse_args()
    if a.root.exists():
        raise RuntimeError('Do not overwrite a previous trial')
    u.initialize(a.root)
    if a.trained_root:
        u.PREV=a.trained_root
    contract=json.loads((a.root/'contracts/UNIFIED_METHOD.json').read_text())
    contract.update(code='MCWF-UNIFIED-STABLE-TAIL',
        waveform_recipe='Zwf=Zold-gamma*clip((-logBCpred-q95_true)/IQR_true,0,6)',
        alpha=.05,gamma_grid=GAMMAS,
        why='Earlier point-selected corrections do not preserve all reused-catalog guardrails. Test validation-only stability and allow finer shrinkage without changing formula or using an old-only O4 model.',
        selection_stability={'source_bootstrap_repeats':REPEATS,'joint_pair_guard_fraction_min':.9,'paired_R10_CI_lower_min':-.02},
        interpretation='90% resample rule is an exploratory engineering robustness criterion, not a probability the method is correct or a physical lensing rate.',
        bootstrap='stratified source block; same-system pair multiplicity m, cross-system m_i*m_j; fixed-model/query rank bootstrap; not iid pair uncertainty',
        priority='among stable options preserve the earlier deterministic F50/F90/AP/R10 priority, then least gamma',
        new_data_opened_before_selection=False,real_role='development only')
    dev.json_write(a.root/'contracts/UNIFIED_METHOD.json',contract)
    if a.trained_root:
        model_contract=json.loads((a.trained_root/'contracts/FINE_LAG_TRAINING.json').read_text())
        contract.update(trained_root=str(a.trained_root),
            feature_implementation=model_contract.get('feature_implementation','sample_resolved_zero_padded_fft_v1'))
        dev.json_write(a.root/'contracts/UNIFIED_METHOD.json',contract)
        dev.json_write(a.root/'contracts/DISTILLED_MODEL_PROVENANCE.json',{'trained_root':str(a.trained_root),
            'feature_implementation':contract['feature_implementation'],'distillation':False,'compatibility_filename_only':True})
    for path in (Path(__file__),Path(tail.__file__)):
        shutil.copy2(path,a.root/'scripts'/path.name)
    statuses=[]
    for dep in u.DEPS:
        for ms,es in zip(body.MODEL_SEEDS,dev.SEEDS):
            f=u.frame(dep,ms,es,'validation')
            x=-np.log(np.maximum(f.new_mass_predictive_BC.to_numpy(float),1e-12))
            true=x[f.is_true_pair.to_numpy(bool)]
            spec={'recipe':'companion_tail_negative_only','alpha':.05,'threshold':float(np.quantile(true,.95)),
                'scale':max(float(np.subtract(*np.quantile(true,[.75,.25]))),.05),'validation_max':float(x.max()),
                'mass_weight':1.,'deployment':dep,'eval_seed':es,'model_seed':ms,'neutral_floor':False}
            old=f.previous_waveform_score.to_numpy(float)
            base=ev.metrics(f,old,dep,es)
            weights=dev.BASE.FROZEN_V93_WEIGHTS[dep][es]
            other=weights['time']*f.time_score.to_numpy()+weights['sky']*f.sky_raw_log_bf.to_numpy()
            bscores={'waveform_only':old,'C_fixed':weights['waveform']*old+other}
            plan=dev.BASE.retained_event_plan(dep,es,'validation')
            state=blocks(f,plan,es+730000)
            bboot={m:metrics_boot(f,s,state) for m,s in bscores.items()}
            grid=[]
            options=[]
            for gamma in GAMMAS:
                option={**spec,'gamma':gamma}
                s,*_=tail.score(f,option)
                met=ev.metrics(f,s,dep,es)
                point=ev.guard(met,base)
                row={'gamma':gamma,'point_guard_pass':point}
                stable=False
                if point:
                    scores={'waveform_only':s,'C_fixed':weights['waveform']*s+other}
                    both=np.ones(REPEATS,dtype=bool)
                    r10=True
                    for method,score in scores.items():
                        boot=metrics_boot(f,score,state)
                        bb=bboot[method]
                        ok=(boot[:,0]>=bb[:,0]-.005)&(boot[:,1]<=1.1*bb[:,1])&(boot[:,2]<=1.1*bb[:,2])
                        both &= ok
                        ci,_=summarizer.ranks_bootstrap(f,score,bscores[method],es+730000,repeats=2000)
                        r10 &= ci['delta_r10_system_ci_low']>=-.02
                        row[method+'_pair_bootstrap_guard_fraction']=float(ok.mean())
                        row[method+'_delta_r10_lower']=ci['delta_r10_system_ci_low']
                    stable=both.mean()>=.9 and r10
                    row['joint_pair_guard_fraction']=float(both.mean())
                row.update(pass_stability=bool(stable),**{m+'_'+k:v for m,mm in met.items() for k,v in mm.items()})
                grid.append(row)
                if stable:
                    options.append((u.objective(met,gamma,1.),option))
            out=a.root/f'calibration/{dep}/seed_{es}'
            out.mkdir(parents=True,exist_ok=True)
            dev.csv_write(out/'validation_joint_grid.csv',pd.DataFrame(grid))
            st={'deployment':dep,'model_seed':ms,'eval_seed':es,'status':'PASS' if options else 'FAIL_NO_STABLE_NONZERO_RECIPE','eligible':len(options)}
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
