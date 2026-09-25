#!/usr/bin/env python3
"""Controlled mass-marginal confidence check of one new joint prediction."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse
from copy import deepcopy
from pathlib import Path
import shutil
import sys
import numpy as np

P=Path('/root/autodl-tmp/gw-catalog')
sys.path.insert(0,str(P/'scripts/experiments'))
import mcwf_quality_state_joint_calibration_20260909 as st
ctrl,engine,base,n,r,co,h=(getattr(st,k) for k in ('ctrl','engine','base','n','r','co','h'))
ROOT=DATA=PREDICTIVE=SELECTION=None
METHODS=('NODUP-DIRECT-REPLAY','JOINTSTATE-MASS-GUARD','MASS-REJECT-ONLY')
REFERENCES={}


def freeze():
    st.freeze()
    n.write_json(ROOT/'contracts/MASS_CONFIDENCE_ADDENDUM.json',{
        'UTC':n.utc(),'id':'MCWF-NODUP-MARGINAL-MASS-CONFIDENCE-37',
        'overrides_quality_state35':'Only the true-companion confidence tail changes:use mass-marginal BC ratherthan jointBC.The jointBC LR,threequalitystates,andstateoffsetremainunchanged.',
        'reason':'Broad conditional spin/q distributions can make joint true-tail tolerance a weak check of a mass conflict.Test a mass-specific predictive-confidence guard.',
        'not_an_extra_LR':'OnejointBC LR plus confidence attenuation,asbefore.No independent mass LR is added.No oldencoderMc/q calibration.',
        'same_new_joint_density':True,'confidence_statistic':'negative logBC of newmassmarginal',
        'fit':'Same newsourcefold0,one companionpair persource,state0/1/2;minimum20fitand20tune true sources.',
        'fixed_threshold':.05,'no_threshold_grid':True,'gamma_beta_outerweights_frozen':True,
        'caps_OOD':'As35:joint LR [-4,4];no positiveOOD;endpointOODneutral',
        'methods':list(METHODS),'REJECT_ONLY':'Both endpoints profilevalid and mass-tail<.05,thenlowerwaveformevidenceonly;otherwise exactNODUP.',
        'assumption_limit':'Empiricalsource-tail requirespopulationexchangeability;not posteriorlensingprobability,not independent evidence,sharednoise limitscoverageclaims.',
        'real_feedback_adaptive':True,'real_test_not_for_fitting':True,
        'no_oldnew_totalblend':True,'status':n.STATUS,'script_sha256':n.sha(Path(__file__))})
    shutil.copy2(__file__,ROOT/'scripts/marginal_mass_confidence.py')


def mass_specs(dep,seed):
    key=dep,seed
    if key in REFERENCES:
        return REFERENCES[key]
    pop=base.population(dep,seed)
    panels=base.panels(dep,seed)
    x,y,_,_,i,j=panels[0]
    state=pop['active'][i].astype(int)+pop['active'][j].astype(int)
    specs={}
    vx,vy,_,_,vi,vj=panels[1]
    vs=pop['active'][vi].astype(int)+pop['active'][vj].astype(int)
    rows=[]
    for k in range(3):
        use=y&(state==k)
        tune=vy&(vs==k)
        if min(use.sum(),tune.sum())<20:
            raise RuntimeError('HOLD_INSUFFICIENT_MASS_TAIL_SOURCE_SUPPORT')
        ref=np.sort(-x[use,1])
        specs[str(k)]={'reference':ref.tolist(),'fit_sources':int(use.sum()),
            'minimum':float(x[state==k,1].min()),'maximum':float(x[state==k,1].max())}
        p=n.ev.tail.tail_probability(-vx[tune,1],ref)
        rows.append({'deployment':dep,'seed':seed,'state':k,'fit_sources':int(use.sum()),
            'tune_sources':int(tune.sum()),'tune_mass_tail_below_0p05':float((p<.05).mean())})
    n.write_json(ROOT/f'calibration/{dep}_{seed}_MASS_REFERENCE.json',specs)
    n.write_csv(ROOT/f'audit/{dep}_{seed}_MASS_TAIL_TUNE.csv',rows)
    REFERENCES[key]=specs
    return specs


def infer(frame,config):
    old=h.isolated.ORIGINALS[config['deployment'],config['seed']]
    reference=co.BASE_INFER(frame,{**old,'method':METHODS[0]})
    state=st.states(frame)
    frame['profile_state']=state
    if config['method']==METHODS[0]:
        frame['actual_waveform_joint_BC']=frame.parent_joint_BC
        frame['new_prediction_used_in_score']=False
        frame['profile_state_true_tail']=np.nan
        frame['profile_state_loglr']=np.nan
        frame['profile_state_OOD']=False
        return reference
    if 'mass_specs' not in config:
        config['mass_specs']=deepcopy(mass_specs(config['deployment'],config['seed']))
    value=frame.joint_logbc.to_numpy(float)
    endpoint=frame.joint_ood.to_numpy(bool)
    mass_value=frame.profile_log_mass_BC.to_numpy(float)
    tail=np.empty(len(frame))
    increment=np.empty(len(frame))
    outside=np.empty(len(frame),bool)
    for k in range(3):
        mask=state==k
        spec=config['state_specs'][str(k)]
        ms=config['mass_specs'][str(k)]
        tail[mask]=n.ev.tail.tail_probability(-mass_value[mask],ms['reference'])
        increment[mask]=(np.interp(value[mask],spec['knots'],spec['loglr'])+spec['state_log_ratio']).clip(-4.,4.)
        outside[mask]=(value[mask]<spec['minimum'])|(value[mask]>spec['maximum'])|endpoint[mask]
        outside[mask]|=(mass_value[mask]<ms['minimum'])|(mass_value[mask]>ms['maximum'])
    increment[outside|(tail<.05)]=np.minimum(increment[outside|(tail<.05)],0.)
    penalty=np.minimum(np.log(tail/.05),0.)
    increment[endpoint],penalty[endpoint]=0.,0.
    cosine,oo,cl=n.apply_model(frame,old['cosine_calibration'])
    candidate=(cosine+old['gamma']*penalty+old['beta']*increment,outside|oo,cl|(abs(increment)>=4.))
    use=np.ones(len(frame),bool)
    if config['method']==METHODS[2]:
        use=(state==2)&(tail<.05)&(candidate[0]<reference[0])
        result=tuple(np.where(use,a,b)for a,b in zip(candidate,reference))
        if np.any(result[0]>reference[0])or not np.array_equal(result[0][~use],reference[0][~use]):
            raise RuntimeError('Exact fallback changed')
    else:
        result=candidate
    frame['actual_waveform_joint_BC']=np.where(use,frame.joint_BC,frame.parent_joint_BC)
    frame['new_prediction_used_in_score']=use
    frame['profile_state_true_tail']=tail
    frame['profile_state_loglr']=increment
    frame['profile_state_OOD']=outside
    return result


def main():
    global ROOT,DATA,PREDICTIVE,SELECTION
    parser=argparse.ArgumentParser()
    for name in ('root','data-root','predictive-root','selection-root'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--stage',choices=('freeze','prepare','calibrate','evaluate','real'),required=True)
    args=parser.parse_args()
    ROOT,DATA,PREDICTIVE,SELECTION=args.root,args.data_root,args.predictive_root,args.selection_root
    st.ROOT,st.DATA,st.PREDICTIVE,st.SELECTION=ROOT,DATA,PREDICTIVE,SELECTION
    st.METHODS,st.infer=METHODS,infer
    ctrl.ROOT,ctrl.SELECTION=ROOT,SELECTION
    engine.ROOT,engine.DATA,engine.PREDICTIVE=ROOT,DATA,PREDICTIVE
    base.ROOT,base.DATA=ROOT,DATA
    base.s.ROOT=h.ROOT=co.ROOT=r.ROOT=co.score.ROOT=ROOT
    r.install()
    st.install_export()
    co.score.matrices=co.matrices
    base.population=engine.population
    n.METHODS,n.load_panel,n.infer=METHODS,ctrl.panel,infer
    if args.stage=='freeze':
        freeze()
    elif args.stage=='prepare':
        ctrl.prepare()
    elif args.stage=='calibrate':
        st.calibrate()
    else:
        n.run(ROOT,args.stage)


if __name__=='__main__':
    main()
