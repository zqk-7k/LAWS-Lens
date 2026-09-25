#!/usr/bin/env python3
"""Require comparable trusted profile estimates at both pair endpoints."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse
from pathlib import Path
import shutil
import sys
import numpy as np

P=Path('/root/autodl-tmp/gw-catalog')
sys.path.insert(0,str(P/'scripts/experiments'))
import mcwf_conservative_profile_refinement_20260909 as parent
co,n,r,score=parent.co,parent.n,parent.r,parent.score
BASE=parent.infer
METHODS=('NODUP-DIRECT-REPLAY','PAIRED-PROFILE-GLOBAL','PAIRED-PROFILE-CONDITIONAL')


def infer(frame,config):
    local=frame.copy()
    local['profile_pair_active']=frame.profile_active_i.to_numpy(bool)&frame.profile_active_j.to_numpy(bool)
    result=BASE(local,config)
    original=parent.isolated.ORIGINALS[config['deployment'],config['seed']]
    ref=co.BASE_INFER(frame,{**original,'method':METHODS[0]})[0]
    mask=local.profile_pair_active.to_numpy(bool)
    if not np.array_equal(result[0][~mask],ref[~mask]):
        raise RuntimeError('Unpaired profile changed evidence')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--stage',choices=('freeze','evaluate','real'),required=True)
    args=parser.parse_args()
    parent.ROOT=co.ROOT=r.ROOT=score.ROOT=args.root
    parent.METHODS=score.METHODS=METHODS
    parent.infer=infer
    r.install()
    score.matrices=co.matrices
    n.METHODS,n.load_panel,n.infer=METHODS,co.load_panel,infer
    if args.stage=='freeze':
        parent.freeze()
        path=args.root/'contracts/BOTH_ENDPOINT_PROTOCOL_ADDENDUM.json'
        n.write_json(path,{'UTC':n.utc(),'id':'MCWF-NODUP-PAIRED-PROFILE-26',
            'overrides':'PairactivationrequiresBOTHendpointsR10profilequality,notOR.',
            'reason':'Comparetwoquality-approvedestimatesfromthesameprofileprocedure;avoidmixedNN/profileprecisionbeingtreatedasdirectlycomparable.',
            'no_new_threshold':True,'no_new_parameter_search':True,'same_both_runs':True,
            'no_extra_score_or_total_mixture':True,'both_arms_reported':True,
            'not_Bayesian_confirmation':True,'adaptive_development':True,
            'freeze_before_evaluation':True,'runtime_sha256':n.sha(Path(__file__))})
        shutil.copy2(__file__,args.root/'scripts/paired_profile_refinement.py')
        n.write_json(args.root/'contracts/PAIRED_RULE_FROZEN.json',{'UTC':n.utc(),'sha256':n.sha(path),
            'configuration_sha256':n.sha(args.root/'configs/SELECTED_CONFIGURATIONS.json')})
    else:
        n.run(args.root,args.stage)
