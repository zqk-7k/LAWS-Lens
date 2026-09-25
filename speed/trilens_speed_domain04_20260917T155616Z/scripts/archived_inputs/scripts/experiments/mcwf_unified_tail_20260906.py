#!/usr/bin/env python3
"""Source-compatibility tail calibration, shared by O3 and O4a."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import mcwf_development_20260905 as dev
import mcwf_encoder_body_20260906 as body
import mcwf_encoder_evaluate_20260906 as ev
import mcwf_unified_waveform_20260906 as u

FIRST = dev.PROJECT/'results/mcwf_unified_v3_20260906T234500Z'
ALPHA = (.05, .1, .2)
GAMMAS = (.25, .5, 1., 2.)


def score(f,spec):
    x=-np.log(np.maximum(f.new_mass_predictive_BC.to_numpy(float),1e-12))
    raw=-np.maximum((x-spec['threshold'])/spec['scale'],0.)
    penalty=np.clip(raw,-6.,0.)
    old=f.previous_waveform_score.to_numpy(float)
    candidate=old+spec['gamma']*penalty
    if spec.get('neutral_floor',False):
        candidate=np.minimum(old,np.maximum(0.,candidate))
    return candidate,penalty,x,x>spec['validation_max']


def fit(root,neutral_floor=False):
    if root.exists():
        raise RuntimeError('Do not overwrite a prior round')
    for d in ('contracts','scripts','calibration','evaluation','results','tables','audit','logs','manifest','reports','figures'):
        (root/d).mkdir(parents=True,exist_ok=True)
    contract=json.loads((FIRST/'contracts/UNIFIED_METHOD.json').read_text())
    contract.update(code='MCWF-UNIFIED-v3-R3',created_utc=datetime.now(timezone.utc).isoformat(),
        prior_failures=[str(FIRST),'mcwf_unified_v3r2_20260907T000500Z'],
        rationale='Balanced LLR compares population rarity, not a false-dismissal controlled contradiction test. Preserve the bulk of true-pair predictive disagreement and penalize only its upper tail.',
        waveform_recipe='Zwf=Zold-gamma*min(6,max(0,(-logBC_pred-q_lensed)/IQR_lensed))',
        new_score='full new-encoder predictive mass distribution BC; no public PE input; zero reward inside validation companion tolerance',
        alpha_grid=ALPHA,gamma_grid=GAMMAS,mass_weight_grid=None,
        threshold='quantile(1-alpha) of VALIDATION true-pair -logBC',
        scale='max(validation true-pair IQR,0.05); fixed numerical regularizer',
        cap=6.,selection='same per-seed validation guardrails and priority; then gamma,alpha; no old-only fallback',
        interpretation='bounded waveform compatibility ranking correction, not posterior probability or independent Bayes factor',
        data_caveat='same six simulation-trained models, with legacy mixed-run O3 development noise disclosed; no training in this calibration round')
    if neutral_floor:
        contract.update(code='MCWF-UNIFIED-v3-R4',
            prior_failure='mcwf_unified_v3r3_20260907T001500Z',
            rationale='Uncertain new-model conflicts must not manufacture strong negative waveform evidence. Withdraw old positive support, but never drive it below neutral or alter already-negative old waveform evidence.',
            waveform_recipe='Zwf=min(Zold,max(0,Zold-gamma*min(6,max(0,(-logBC_pred-q_lensed)/IQR_lensed))))',
            neutral_floor=0.,new_reward_allowed=False)
    dev.json_write(root/'contracts/UNIFIED_METHOD.json',contract)
    shutil.copy2(FIRST/'manifest/PROTECTED_INPUT_SHA256.csv',root/'manifest/PROTECTED_INPUT_SHA256.csv')
    for p in (FIRST/'audit').glob('*.parquet'):
        shutil.copy2(p,root/'audit'/p.name)
    for p in (Path(__file__),Path(u.__file__)):
        shutil.copy2(p,root/'scripts'/p.name)
    statuses=[]
    for dep in u.DEPS:
        for ms,es in zip(body.MODEL_SEEDS,dev.SEEDS):
            out=root/f'calibration/{dep}/seed_{es}'
            out.mkdir(parents=True)
            f=u.frame(dep,ms,es,'validation')
            x=-np.log(np.maximum(f.new_mass_predictive_BC.to_numpy(float),1e-12))
            true=x[f.is_true_pair.to_numpy(bool)]
            scale=max(float(np.subtract(*np.quantile(true,[.75,.25]))),.05)
            base=ev.metrics(f,f.previous_waveform_score.to_numpy(),dep,es)
            rows=[]
            options=[]
            for alpha in ALPHA:
                for gamma in GAMMAS:
                    spec={'recipe':'companion_tail_negative_only','alpha':alpha,'gamma':gamma,'threshold':float(np.quantile(true,1-alpha)),
                        'scale':scale,'validation_max':float(x.max()),'mass_weight':1.,'model_seed':ms,'eval_seed':es,'deployment':dep,
                        'n_companion_systems':len(true)}
                    spec['neutral_floor']=neutral_floor
                    if neutral_floor:
                        spec['recipe']='companion_tail_withdraw_positive_support'
                    s,penalty,_,_=score(f,spec)
                    m=ev.metrics(f,s,dep,es)
                    good=ev.guard(m,base)
                    row={'alpha':alpha,'gamma':gamma,'threshold':spec['threshold'],'scale':scale,'pass':good,
                        'true_penalty_fraction':float((penalty[f.is_true_pair.to_numpy(bool)]<0).mean()),
                        **{method+'_'+k:v for method,mm in m.items() for k,v in mm.items()}}
                    rows.append(row)
                    if good:
                        options.append((u.objective(m,gamma,alpha),spec))
            dev.csv_write(out/'validation_joint_grid.csv',pd.DataFrame(rows))
            st={'deployment':dep,'model_seed':ms,'eval_seed':es,'status':'PASS' if options else 'FAIL_NO_NONZERO_SHARED_RECIPE','eligible':len(options)}
            if options:
                selected=min(options,key=lambda v:v[0])[1]
                dev.json_write(out/'SELECTED_CONFIG.json',selected)
                st.update(selected_sha256=dev.sha(out/'SELECTED_CONFIG.json'),gamma=selected['gamma'],alpha=selected['alpha'],mass_weight=1.)
            dev.json_write(out/'FIT_STATUS.json',st)
            statuses.append(st)
            print(json.dumps(st),flush=True)
    dev.csv_write(root/'tables/VALIDATION_SELECTION.csv',pd.DataFrame(statuses))
    dev.json_write(root/'contracts/VALIDATION_GATE.json',{'pass':all(s['status']=='PASS' for s in statuses),'seeds':statuses})


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--phase',choices=('fit','evaluate'),required=True)
    p.add_argument('--neutral-floor',action='store_true')
    a=p.parse_args()
    if a.phase=='fit':
        fit(a.root,neutral_floor=a.neutral_floor)
    else:
        u.evaluate(a.root,score_function=score)
