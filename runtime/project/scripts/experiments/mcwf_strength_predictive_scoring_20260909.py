#!/usr/bin/env python3
"""Integrate the frozen waveform-strength predictive covariate, not Fisher PE."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse
from pathlib import Path
import shutil
import sys
import pandas as pd

P=Path('/root/autodl-tmp/gw-catalog')
sys.path.insert(0,str(P/'scripts/experiments'))
import mcwf_profile_strength_calibration_20260909 as strength
import mcwf_information_predictive_scoring_20260909 as app
app.information.information=strength.strength
s,h,co,r,n,score,d=app.s,app.h,app.co,app.r,app.n,app.score,app.d


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--predictive-root',type=Path,required=True)
    p.add_argument('--stage',choices=('freeze','calibrate','evaluate','real'),required=True)
    p.add_argument('--workers',type=int,default=20)
    args=p.parse_args()
    app.PREDICTIVE=args.predictive_root
    app.ROOT=s.ROOT=h.ROOT=co.ROOT=r.ROOT=score.ROOT=d.ROOT=args.root
    app.WORKERS=args.workers
    r.install()
    co.mass=d.predict=app.predict
    co.matrices=score.matrices=app.ORIGINAL_MATRICES
    s.METHODS=score.METHODS=app.METHODS
    n.METHODS,n.load_panel,n.infer=app.METHODS,app.load_panel,app.infer
    base_export=n.public_frame
    def export(frame,z,weights,method):
        out=base_export(frame,z,weights,method)
        for field in h.FEATURES+('information_active_i','information_active_j','information_width_i','information_width_j'):
            out[field]=frame[field].to_numpy()
        return out
    n.public_frame=export
    if args.stage=='freeze':
        app.freeze()
        path=args.root/'contracts/STRENGTH_SCORING_ADDENDUM.json'
        n.write_json(path,{'UTC':n.utc(),'id':'MCWF-NODUP-STRENGTH-INTEGRATED-28',
            'overrides':'ALLreferencesinparentcontracttoFisherorexpected-derivativeinformationarereplacedbyfrozenR27inversewaveformprojectionstrength.',
            'internal_method_names':'INFORMATIONmeansstrength-conditionedpredictivemasshere;notFishercovarianceorfullPE.',
            'predictive_root':str(args.predictive_root),'predictive_sha256':n.sha(args.predictive_root/'calibration/PROFILE_EXPECTED_INFORMATION.json'),
            'score_options':'Samefixed-joint andglobal/active ONEclassifierLINEAR/TREE;fit/tunerulesunchanged;allarmsretained.',
            'same_both_runs':True,'frozen':['encoder','time','sky','outerweights','scope','historicaloutputs'],
            'no_old_Mc_q_or_totalblend':True,'no_PE_official_scoring_inputs':True,
            'premerger_MAD_context_not_guaranteed_signal_free':True,'not_optimal_network_SNR':True,
            'adaptive_development':True,'goal_achieved':False,'code_sha256':n.sha(Path(__file__))})
        shutil.copy2(__file__,args.root/'scripts/strength_predictive_scoring.py')
        shutil.copy2(strength.__file__,args.root/'scripts/profile_strength_calibration.py')
        n.write_json(args.root/'contracts/STRENGTH_SCORE_FROZEN.json',{'UTC':n.utc(),'sha256':n.sha(path),
            'before_new_validation_test_real_scoring':True})
    elif args.stage=='calibrate':
        app.calibrate()
    else:
        n.run(args.root,args.stage)
