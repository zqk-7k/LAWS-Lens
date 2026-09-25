#!/usr/bin/env python3
"""Use a lower envelope only for a frozen 95% profile-inconsistency flag."""
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
import mcwf_quality_profile_envelope_20260909 as parent
st,cv,engine,base,n,r,co,h=(getattr(parent,k)for k in ('st','cv','engine','base','n','r','co','h'))
ORIGINAL_INFER=parent.infer
METHODS=('NODUP-DIRECT-REPLAY','JOINTSTATE-FROZEN95','QUALITY-PROFILE-GLOBAL-REJECT95','QUALITY-PROFILE-CONDITIONAL-REJECT95')
FIELDS=parent.FIELDS+('profile_reference_true_tail','profile_reference_reject95')


def infer(frame,config):
    candidate=ORIGINAL_INFER(frame,config)
    if config['method']in METHODS[:2]:
        frame['profile_reference_true_tail']=np.nan;frame['profile_reference_reject95']=False
        return candidate
    reference=st.infer(frame.copy(),{**config['fallback'],'method':'PARENT_GLOBAL'})
    spec=parent.profile_specs(config['deployment'],config['seed'])[config['profile_reference']]
    value=-np.log(frame.profile_reference_joint_BC.to_numpy(float).clip(1e-300,1.))
    tail=n.ev.tail.tail_probability(value,spec['reference'])
    flag=tail<.05
    take=frame.profile_envelope_used.to_numpy(bool)&flag
    result=tuple(np.where(take,a,b)for a,b in zip(candidate,reference))
    if np.any(result[0]>reference[0])or any(not np.array_equal(a[~take],b[~take])for a,b in zip(result,reference)):
        raise RuntimeError('Confidence-only conservative fallback failed')
    frame['profile_reference_true_tail']=tail;frame['profile_reference_reject95']=flag
    frame['profile_envelope_used']=take
    frame['actual_waveform_joint_BC']=np.where(take,frame.profile_reference_joint_BC,frame.joint_BC)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--parent',type=Path,required=True)
    parser.add_argument('--stage',choices=('freeze','evaluate','real'),required=True)
    args=parser.parse_args();parent.ROOT,parent.PARENT=args.root,args.parent
    cv.ROOT,cv.PARENT=args.root,args.parent
    base.s.ROOT=h.ROOT=co.ROOT=r.ROOT=co.score.ROOT=args.root;engine.ROOT=args.parent
    parent.METHODS=METHODS;parent.infer=infer
    r.install();st.FIELDS=FIELDS;st.install_export()
    st.METHODS=(METHODS[0],'PARENT_GLOBAL','UNUSED_REJECT')
    co.score.matrices=co.matrices;n.METHODS,n.load_panel,n.infer=METHODS,parent.panel,infer
    if args.stage=='freeze':
        parent.freeze()
        path=args.root/'contracts/REJECTION_ONLY_ADDENDUM.json'
        n.write_json(path,{'UTC':n.utc(),'id':'MCWF-NODUP-QUALITY-PROFILE-REJECT95-48',
            'overrides_parent':'Only apply R47 lower-envelope reduction if the SAME referenceprofile jointBC has empirical true-source tail<.05. OtherwiseexactR35.',
            'reason':'A lower positive score is not necessarily evidence of incompatibility. Restrict attenuation to the existing95percent inconsistency flag to reduce unnecessary changes to plausible truepairs.',
            'threshold':.05,'threshold_source':'Unchanged originalprofile true-tail threshold,not newlysearched.',
            'no_hyperparameter_search':True,'same_both_runs':True,'no_total_mixture':True,
            'no_old_encoder_Mc_q':True,'time_sky_outerweights_frozen':True,
            'reference':'Fixed R10/R12 simulated jointBC true-source references,not publicPE.',
            'statistical_limit':'Empirical finite-source confidence flag in a reuseddata exploratory study,not a PEpvalue,lensingFPP or formalconformalguarantee.',
            'all_global_conditional_arms_reported':True,'adaptive_development':True,
            'real_or_test_not_hyperparameter_input':True,'goal_achieved':False})
        shutil.copy2(__file__,args.root/'scripts/quality_profile_rejection.py')
        n.write_json(args.root/'contracts/REJECTION_RULE_FROZEN.json',{'UTC':n.utc(),
            'addendum_sha256':n.sha(path),'configuration_sha256':n.sha(args.root/'configs/SELECTED_CONFIGURATIONS.json'),
            'runtime_sha256':n.sha(Path(__file__))})
    else:
        n.run(args.root,args.stage)
