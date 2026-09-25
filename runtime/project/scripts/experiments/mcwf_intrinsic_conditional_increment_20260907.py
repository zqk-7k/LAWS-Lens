#!/usr/bin/env python3
"""Predictive intrinsic consistency conditional on existing mass consistency."""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
from sklearn.isotonic import IsotonicRegression
import mcwf_intrinsic_grid_20260907 as g

e,dev=g.e,g.dev


def basework(root):return root.parent


def setup(root):
    if root.parent.name=='dense':
        import mcwf_intrinsic_dense_20260907 as d
        g.prediction=d.prediction


def initialize(root):
    g.initialize(root,'CONDITIONAL')
    dev.json_write(root/'contracts/CONDITIONAL_INCREMENT.json',{
        'model_root':str(basework(root)),'no_new_training':True,
        'overrides_parent_training_contract':True,
        'arms':['JOINT-GIVEN-MASS','Q-GIVEN-MASS'],
        'feature':'log predictive joint overlap/prior minus log predictive mass overlap/prior',
        'purpose':'Test q/spin consistency beyond Mc; do not add a second full mass reward',
        'gamma_grid':[0.],'beta_grid':[0.,.0625,.125,.25,.5,1.,2.,4.],
        'calibration':'Balanced isotonic on development systems; same finite cap and mass OOD controls',
        'interpretation':'Empirical conditional-predictive descriptor, not a physical Bayes factor',
        'real_audit_selection':'Adaptive development, not blind validation',
        'frozen':'Time/sky/outer weights/scopes/old results; target thresholds unchanged'})
    shutil.copy2(__file__,root/'scripts'/Path(__file__).name)


def pair(a,i,j,prior,arm):
    joint=g.pair_features(a,i,j,prior,'JOINT-PRIOR' if arm=='JOINT-GIVEN-MASS' else 'MASS-Q-PRIOR')
    mass=g.pair_features(a,i,j,prior,'MASS-PRIOR')
    return {**mass,'prior_overlap':joint['prior_overlap']-mass['prior_overlap']}


def values(root,dep,es,split,arm):
    w=basework(root)
    cache=root/f'cache/pair_features/{arm}/{dep}_{es}_{split}.npz'
    f,x=read_values(w,dep,es,split,'JOINT-PRIOR' if arm=='JOINT-GIVEN-MASS' else 'MASS-Q-PRIOR')
    if cache.exists():return f,dict(np.load(cache))
    _,m=read_values(w,dep,es,split,'MASS-PRIOR')
    v={**m,'prior_overlap':x['prior_overlap']-m['prior_overlap']}
    cache.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(cache,**v);return f,v


def calibrate(root,dep,es,arm):
    path=root/f'contracts/{dep}_{arm}_CALIBRATION.json'
    if path.exists():return json.loads(path.read_text())
    a=g.ensemble(basework(root),dep,0,'development');i,j=np.triu_indices(len(a['p']),1)
    y=a['group'][i]==a['group'][j]
    x=pair(a,i,j,g.get_prior(basework(root),dep),arm)
    w=np.where(y,.5/y.sum(),.5/(~y).sum())
    iso=IsotonicRegression(increasing=True,out_of_bounds='clip').fit(x['prior_overlap'],y,sample_weight=w)
    p=iso.y_thresholds_.clip(1/(y.sum()+2),1-1/(y.sum()+2))
    spec={'mass_reference':np.sort(-np.log(x['bc'][y])).tolist(),'source_pairs':int(y.sum()),
          'prior_overlap':{'knots':iso.X_thresholds_.tolist(),'loglr':(np.log(p)-np.log1p(-p)).tolist(),
                           'minimum':float(x['prior_overlap'].min()),'maximum':float(x['prior_overlap'].max())}}
    dev.json_write(path,spec);return spec


def run(root,arm):
    setup(root)
    # g.run installs the shared target; intercept only its value/calibration/grid setup.
    original_values,original_calibrate=g.values,g.calibrate
    g.values=lambda r,d,s,sp,a:values(r,d,s,sp,a)
    g.calibrate=lambda r,d,s,a:calibrate(r,d,s,a)
    original_run=e.run
    def restricted(r,a):
        e.GAMMAS=(0.,);return original_run(r,a)
    e.run=restricted
    # values needs the original reader rather than the temporary dispatcher.
    global read_values
    read_values=original_values
    g.run(root,arm)


read_values=g.values

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--phase',choices=('init','run'),required=True)
    p.add_argument('--arm',choices=('JOINT-GIVEN-MASS','Q-GIVEN-MASS'));a=p.parse_args()
    initialize(a.root) if a.phase=='init' else run(a.root,a.arm)
