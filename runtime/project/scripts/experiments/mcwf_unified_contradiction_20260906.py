#!/usr/bin/env python3
"""Same uncertainty-aware, negative-only waveform update in both runs."""
from datetime import datetime, timezone
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
import mcwf_development_20260905 as dev
import mcwf_encoder_body_20260906 as body
import mcwf_encoder_evaluate_20260906 as ev
import mcwf_unified_waveform_20260906 as u

FIRST = dev.PROJECT / 'results/mcwf_unified_v3_20260906T234500Z'
GAMMAS = (.25, .5, 1., 2.)
CAP = 6.


def score(f, spec):
    x = np.log(np.maximum(f.new_mass_predictive_BC.to_numpy(float), 1e-12))
    p = np.interp(x, spec['x_thresholds'], spec['p_thresholds'])
    llr = np.log(p) - np.log1p(-p)
    penalty = np.clip(llr, -CAP, 0.)
    return f.previous_waveform_score.to_numpy(float) + spec['gamma']*penalty, llr, x, (x < spec['validation_min']) | (x > spec['validation_max'])


def fit(root):
    if root.exists():
        raise RuntimeError('Round already exists; do not overwrite')
    for d in ('contracts','scripts','calibration','evaluation','results','tables','audit','logs','manifest','reports','figures'):
        (root/d).mkdir(parents=True, exist_ok=True)
    contract = json.loads((FIRST/'contracts/UNIFIED_METHOD.json').read_text())
    contract.update(code='MCWF-UNIFIED-v3-R2', created_utc=datetime.now(timezone.utc).isoformat(),
        prior_failure=str(FIRST),
        rationale='New mass point predictions can coincide even for physically different signals; positive reinforcement magnifies those mistakes. Use the full predicted mass distribution and only reject calibrated conflicts.',
        new_score='x=log(sum sqrt(p_Mc_i*p_Mc_j)); balanced-label monotone isotonic calibration yields log odds; these neural distributions are NOT public PE or full strain PE.',
        waveform_recipe='Zwf=Zold+gamma*clip(logit(Pbalanced(companion|logBC_pred)),-6,0)',
        gamma_grid=GAMMAS, mass_weight_grid=None,
        calibration='isotonic increasing; class-balanced total weights; eps=1e-4 probability bounds; preserve uncertainty via all64 predicted mass bins; same for all runs/seeds',
        cap=CAP,
        limitations='Conservative new-waveform contradiction gate, not independent Bayes evidence; no claim that remaining high scores imply true shared source. Reused validation pairs are dependent, bootstrap by source.',
        selection='validation guardrails then identical objective to round1; no old-only fallback; all gamma>0',
        no_data_or_network_training_change_this_round=True)
    dev.json_write(root/'contracts/UNIFIED_METHOD.json',contract)
    shutil.copy2(FIRST/'manifest/PROTECTED_INPUT_SHA256.csv',root/'manifest/PROTECTED_INPUT_SHA256.csv')
    for p in (FIRST/'audit').glob('*.parquet'):
        shutil.copy2(p,root/'audit'/p.name)
    for p in (Path(__file__),Path(u.__file__)):
        shutil.copy2(p,root/'scripts'/p.name)
    allstatus=[]
    for dep in u.DEPS:
        for ms,es in zip(body.MODEL_SEEDS,dev.SEEDS):
            out=root/f'calibration/{dep}/seed_{es}'
            out.mkdir(parents=True)
            f=u.frame(dep,ms,es,'validation')
            y=f.is_true_pair.to_numpy(bool)
            x=np.log(np.maximum(f.new_mass_predictive_BC.to_numpy(float),1e-12))
            iso=IsotonicRegression(y_min=1e-4,y_max=1-1e-4,out_of_bounds='clip')
            weights=np.where(y,.5/y.sum(),.5/(~y).sum())
            iso.fit(x,y.astype(float),sample_weight=weights)
            rawspec={'recipe':'predictive_BC_negative_only','x_thresholds':iso.X_thresholds_.tolist(),'p_thresholds':iso.y_thresholds_.tolist(),
                'validation_min':float(x.min()),'validation_max':float(x.max()),'mass_weight':1.,'model_seed':ms,'eval_seed':es,'deployment':dep,
                'positive_count':int(y.sum()),'negative_count':int((~y).sum()),'n_isotonic_steps':len(iso.X_thresholds_)}
            base=ev.metrics(f,f.previous_waveform_score.to_numpy(),dep,es)
            choices=[]
            rows=[]
            for gamma in GAMMAS:
                spec={**rawspec,'gamma':gamma}
                s,_,_,_=score(f,spec)
                m=ev.metrics(f,s,dep,es)
                good=ev.guard(m,base)
                rows.append({'gamma':gamma,'pass':good,**{method+'_'+k:v for method,mm in m.items() for k,v in mm.items()}})
                if good:
                    choices.append((u.objective(m,gamma,1.),spec))
            dev.csv_write(out/'validation_joint_grid.csv',pd.DataFrame(rows))
            st={'deployment':dep,'model_seed':ms,'eval_seed':es,'status':'PASS' if choices else 'FAIL_NO_NONZERO_SHARED_RECIPE','eligible':len(choices)}
            if choices:
                selected=min(choices,key=lambda c:c[0])[1]
                dev.json_write(out/'SELECTED_CONFIG.json',selected)
                st.update(selected_sha256=dev.sha(out/'SELECTED_CONFIG.json'),gamma=selected['gamma'],mass_weight=1.)
            dev.json_write(out/'FIT_STATUS.json',st)
            allstatus.append(st)
            print(json.dumps(st),flush=True)
    dev.csv_write(root/'tables/VALIDATION_SELECTION.csv',pd.DataFrame(allstatus))
    dev.json_write(root/'contracts/VALIDATION_GATE.json',{'pass':all(s['status']=='PASS' for s in allstatus),'seeds':allstatus})


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--phase',choices=('fit','evaluate'),required=True)
    a=p.parse_args()
    if a.phase=='fit':
        fit(a.root)
    else:
        u.evaluate(a.root,score_function=score)


if __name__=='__main__':
    main()
