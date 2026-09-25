#!/usr/bin/env python3
"""Replace old OMC auxiliary evidence instead of stacking another mass term."""
import argparse
from pathlib import Path
import shutil
import mcwf_intrinsic_grid_20260907 as g

e,dev=g.e,g.dev
READ_VALUES=g.values;READ_CALIBRATE=g.calibrate


def initialize(root):
    g.initialize(root,'CONDITIONAL')
    dev.json_write(root/'contracts/REPLACEMENT_CONTROL.json',{
        'no_new_training':True,'overrides_parent_training_contract':True,'models':str(root.parent/'models'),
        'arms':['JOINT-PRIOR','JOINT-BC'],
        'formula':'Z_FRT + gamma*new_predictive_mass_tail_penalty + beta*bounded_joint_predictive_increment',
        'compared_against':'latest OMC baseline,not older FRT; retain GLOBAL O4 comparison',
        'purpose':'Remove old OMC auxiliary term,avoiding double-counted learned mass evidence',
        'gamma':[0,.125,.25,.5,1,2,4],'beta':[0,.0625,.125,.25,.5,1,2,4],
        'time_sky_outer_weights_scope_frozen':True,'real_selection':'adaptive development,not blind',
        'all_previous_guards_unchanged':True})
    shutil.copy2(__file__,root/'scripts'/Path(__file__).name)


def values(root,dep,es,split,arm):return READ_VALUES(root.parent,dep,es,split,arm)


def calibrate(root,dep,es,arm):return {**READ_CALIBRATE(root.parent,dep,es,arm),'replace_OMC':True}


def score(f,x,spec,arm):
    if spec.get('unchanged_baseline'):return g.score(f,x,spec,arm)
    f=f.copy();f['waveform_score']=f.FRT_baseline_waveform_score
    return g.score(f,x,spec,arm)


def run(root,arm):
    if root.parent.name=='dense':
        import mcwf_intrinsic_dense_20260907 as d
        g.prediction=d.prediction
    g.values=values;g.calibrate=calibrate
    original=e.run
    def configured(r,a):
        e.GAMMAS=(0.,.125,.25,.5,1.,2.,4.);e.score=score
        return original(r,a)
    e.run=configured;g.run(root,arm)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--phase',choices=('init','run'),required=True)
    p.add_argument('--arm',choices=('JOINT-PRIOR','JOINT-BC'));a=p.parse_args()
    initialize(a.root) if a.phase=='init' else run(a.root,a.arm)
