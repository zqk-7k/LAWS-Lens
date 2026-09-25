#!/usr/bin/env python3
"""Bounded negative waveform evidence with an explicit neutral mixture."""
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
import mcwf_conditional_waveform_calibration_20260906 as c
import mcwf_cross_domain_selection_20260906 as cross
import mcwf_stable_tail_selection_20260906 as stability
import mcwf_summarize_20260905 as summarizer


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--bootstrap-stability',action='store_true')
    a=p.parse_args()
    if a.root.exists():
        raise RuntimeError('Independent root required')
    u.initialize(a.root)
    base=dev.PROJECT/'results/mcwf_unified_conditional_components_20260907'
    grid=(.05,.1,.25,.5,.75,.9,.95)
    contract=json.loads((a.root/'contracts/UNIFIED_METHOD.json').read_text())
    contract.update(code='MCWF-ROBUST-NEUTRAL-MIXTURE',
        formula='oldZ+log((1-gamma)+gamma*exp(min(calibratedZ-oldZ,0)))',
        rationale='Allow the additional waveform verifier to be uninformative with nonzero probability. Its worst possible negative contribution is log(1-gamma), not an arbitrarily negative calibrated tail. This is a bounded ranking mixture, not a measured probability of encoder correctness or an independent physical Bayes factor.',
        gamma_grid=grid,calibration='Frozen source/noise-disjoint components HGB from prior archived development; no refitting on test/real',
        selection='BAY validation plus independent auxiliary-noise tune guards; same deterministic false-burden objective in both runs',
        real_role='Repeatedly inspected development, never blind confirmation')
    if a.bootstrap_stability:
        contract.update(code='MCWF-ROBUST-NEUTRAL-MIXTURE-STABILITY',
            source_bootstrap_repeats=500,joint_guard_fraction_min=.9,R10_delta_CI_lower_min=-.02,
            rationale_extension='Reuse the earlier predefined source-bootstrap stability rule with the new bounded mixture; reject point-selected strengths whose validation gains disappear under source resampling. Neither the gate tolerances nor the physical inputs are relaxed.')
    dev.json_write(a.root/'contracts/UNIFIED_METHOD.json',contract)
    shutil.copy2(__file__,a.root/'scripts'/Path(__file__).name)
    statuses=[]
    for dep in u.DEPS:
        for ms,es in zip(body.MODEL_SEEDS,dev.SEEDS):
            spec=json.loads((base/f'calibration/{dep}/seed_{es}/SELECTED_CONFIG.json').read_text())
            src=c.SOURCE/dep/f'seed_{es}'
            f=pd.read_parquet(src/'pairs.parquet')
            meta=pd.read_parquet(src/'event_metadata.parquet')
            fit,tune,noise=c.split_metadata(meta,dep)
            tf=c.subset(f,tune)
            bay=u.frame(dep,ms,es,'validation')
            bm=ev.metrics(bay,bay.previous_waveform_score.to_numpy(),dep,es)
            tm=dev.BASE.full_metrics(tf,tf.previous_waveform_score.to_numpy())
            weights=dev.BASE.FROZEN_V93_WEIGHTS[dep][es]
            other=weights['time']*bay.time_score.to_numpy()+weights['sky']*bay.sky_raw_log_bf.to_numpy()
            bscores={'waveform_only':bay.previous_waveform_score.to_numpy(),
                     'C_fixed':weights['waveform']*bay.previous_waveform_score.to_numpy()+other}
            if a.bootstrap_stability:
                state=stability.blocks(bay,dev.BASE.retained_event_plan(dep,es,'validation'),es+730000)
                bboot={m:stability.metrics_boot(bay,s,state) for m,s in bscores.items()}
            out=a.root/f'calibration/{dep}/seed_{es}'
            out.mkdir(parents=True)
            rows=[]
            options=[]
            for gamma in grid:
                option={**spec,'gamma':gamma,'residual_mode':'neutral_mixture'}
                bs,*_=c.score(bay,option)
                ts,*_=c.score(tf,option)
                bmet=ev.metrics(bay,bs,dep,es)
                tmet=dev.BASE.full_metrics(tf,ts)
                passed=ev.guard(bmet,bm) and cross.wf_guard(tmet,tm)
                row={'gamma':gamma,'point_guard_pass':passed,**{m+'_'+k:v for m,d in bmet.items() for k,v in d.items()}}
                if passed and a.bootstrap_stability:
                    both=np.ones(stability.REPEATS,dtype=bool)
                    r10_ok=True
                    for method,s in {'waveform_only':bs,'C_fixed':weights['waveform']*bs+other}.items():
                        boot=stability.metrics_boot(bay,s,state)
                        bb=bboot[method]
                        ok=(boot[:,0]>=bb[:,0]-.005)&(boot[:,1]<=1.1*bb[:,1])&(boot[:,2]<=1.1*bb[:,2])
                        both &= ok
                        ci,_=summarizer.ranks_bootstrap(bay,s,bscores[method],es+730000,repeats=2000)
                        r10_ok &= ci['delta_r10_system_ci_low']>=-.02
                        row[method+'_bootstrap_guard_fraction']=float(ok.mean())
                    row['joint_bootstrap_guard_fraction']=float(both.mean())
                    passed=both.mean()>=.9 and r10_ok
                row['pass']=bool(passed)
                rows.append(row)
                if passed:
                    options.append((u.objective(bmet,gamma,1.),option))
            dev.csv_write(out/'validation_grid.csv',pd.DataFrame(rows))
            st={'deployment':dep,'model_seed':ms,'eval_seed':es,'pass':bool(options)}
            if options:
                selected=min(options,key=lambda r:r[0])[1]
                dev.json_write(out/'SELECTED_CONFIG.json',selected)
                st.update(gamma=selected['gamma'],selected_sha256=dev.sha(out/'SELECTED_CONFIG.json'))
            statuses.append(st)
            print(json.dumps(st),flush=True)
    dev.json_write(a.root/'contracts/VALIDATION_GATE.json',{'pass':all(s['pass'] for s in statuses),'seeds':statuses})
    if all(s['pass'] for s in statuses):
        u.evaluate(a.root,score_function=c.score)


if __name__=='__main__':
    main()
