#!/usr/bin/env python3
"""Disclosed score-scale control; not a new encoder or physical Bayes factor."""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
import mcwf_intrinsic_grid_20260907 as g

e,dev=g.e,g.dev
BLEND=(0.,.1,.25,.5,.75,.9,1.)
SCALE=(.25,.5,1.,2.,4.)


def initialize(root):
    g.initialize(root,'CONDITIONAL')
    dev.json_write(root/'contracts/SCORE_CALIBRATION_CONTROL.json',{
        'arms':['ISOTONIC','LOGISTIC'],'prior_contract_training_not_applicable':True,
        'no_training':True,'base':'latest OMC waveform_score',
        'fit':'class-balanced labeled simulation validation pairs, independently per run/seed',
        'formula':'(1-r)*Z_OMC+r*a*logit(p_balanced(L|Z_OMC))',
        'blend_grid':BLEND,'scale_grid':SCALE,
        'physical_scope':'Empirical monotone recalibration of a composite ranking score, not a full Bayes factor.',
        'scale_disclosure':'Changing waveform scale changes effective balance with fixed time/sky; not new information.',
        'logistic_nonpositive_slope':'stop; do not flip labels or reorder by PE',
        'tail':'Clip probability by finite true-pair count; no linear tail extrapolation',
        'selection':'Real PE/official outcomes used for adaptive development. Not blind validation.',
        'guards':'Inherit original both-run target and O4 GLOBAL-PRIOR noninferiority without changes'})
    shutil.copy2(__file__,root/'scripts'/Path(__file__).name)


def values(root,dep,es,split):
    f=pd.read_parquet(e.TRIAL/f'evaluation/{dep}/seed_{es}/{split}_pairs.parquet')
    return f,{}


def calibrate(root,dep,es,arm):
    path=root/f'contracts/{dep}_{es}_{arm}_CALIBRATION.json'
    if path.exists():return json.loads(path.read_text())
    f,_=values(root,dep,es,'validation')
    # The frozen pair table supplies its own labels, never the real candidate audit.
    label=next(c for c in ('is_true_pair','is_companion','is_lensed_pair','label') if c in f)
    y=f[label].to_numpy(bool);x=f.waveform_score.to_numpy(float)
    w=np.where(y,.5/y.sum(),.5/(~y).sum())
    floor=1/(int(y.sum())+2)
    spec={'arm':arm,'true_pairs':int(y.sum()),'dependent_null_pairs':int((~y).sum()),
          'probability_floor':floor,'minimum':float(x.min()),'maximum':float(x.max()),
          'fit_pair_table_sha256':dev.sha(e.TRIAL/f'evaluation/{dep}/seed_{es}/validation_pairs.parquet')}
    if arm=='ISOTONIC':
        fit=IsotonicRegression(increasing=True,out_of_bounds='clip').fit(x,y,sample_weight=w)
        p=fit.y_thresholds_.clip(floor,1-floor)
        spec.update(knots=fit.X_thresholds_.tolist(),loglr=(np.log(p)-np.log1p(-p)).tolist())
    else:
        mu=float(np.sum(x*w));sd=float(np.sqrt(np.sum((x-mu)**2*w)))
        fit=LogisticRegression(C=1e4,max_iter=1000).fit(((x-mu)/sd)[:,None],y,sample_weight=w*len(y))
        slope=float(fit.coef_[0,0]);assert slope>0
        spec.update(mu=mu,sd=sd,slope=slope,intercept=float(fit.intercept_[0]))
    dev.json_write(path,spec);return spec


def score(f,x,spec,arm):
    z=f.waveform_score.to_numpy(float)
    if spec.get('unchanged_baseline'):return z,np.zeros(len(z)),np.zeros(len(z))
    if spec['arm']=='ISOTONIC':cal=np.interp(z,spec['knots'],spec['loglr'])
    else:
        cal=(z.clip(spec['minimum'],spec['maximum'])-spec['mu'])/spec['sd']*spec['slope']+spec['intercept']
        bound=np.log((1-spec['probability_floor'])/spec['probability_floor']);cal=cal.clip(-bound,bound)
    return (1-spec['gamma'])*z+spec['gamma']*spec['beta']*cal,np.zeros(len(z)),cal


def run(root,arm):
    e.values=values;e.calibrate=lambda r,d,s:calibrate(r,d,s,arm);e.score=score
    e.GAMMAS=BLEND;e.BETAS=SCALE
    original=e.batch_target
    comparator=pd.read_csv(root/'comparators/O4a_GLOBAL_PRIOR/pe_official_budget.csv')
    def target(combos,pools,pe,base,dep):
        out=original(combos,pools,pe,base,dep)
        if dep=='gwtc4':
            for b in (10,20):
                row=comparator[(comparator.seed.astype(str)=='consensus')&(comparator.method=='C_fixed')&(comparator.budget==b)].iloc[0]
                for key in ('BC_mc_ge_0p5','Dmax_le_3','official_frontend','official_hanabi','median_BC_mc'):
                    out['target_pass']&=out[f'Top{b}_{key}']>=row[key]-1e-12
                out['target_pass']&=out[f'Top{b}_catastrophic_mc']<=row.catastrophic_mc
        return out
    e.batch_target=target;e.run(root,arm)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--phase',choices=('init','run'),required=True);p.add_argument('--arm',choices=('ISOTONIC','LOGISTIC'))
    a=p.parse_args();initialize(a.root) if a.phase=='init' else run(a.root,a.arm)
