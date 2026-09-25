#!/usr/bin/env python3
"""Fixed lower envelope of two waveform-only profile calibrations."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys
import numpy as np
import pandas as pd

P=Path('/root/autodl-tmp/gw-catalog')
sys.path.insert(0,str(P/'scripts/experiments'))
import mcwf_confidence_level_validation_20260909 as cv
st,ctrl,engine,base,n,r,co,h=(getattr(cv,k)for k in ('st','ctrl','engine','base','n','r','co','h'))
ROOT=PARENT=None
METHODS=('NODUP-DIRECT-REPLAY','JOINTSTATE-FROZEN95','QUALITY-PROFILE-GLOBAL-ENVELOPE','QUALITY-PROFILE-CONDITIONAL-ENVELOPE')
REFINERS={'GLOBAL':co.PARENT,'CONDITIONAL':P/'results/mcwf_nodup_isolated_profile_12_20260909T092250Z'}
FIELDS=st.FIELDS+('profile_envelope_used','profile_reference_joint_BC','profile_reference_waveform',
                 'quality_reference_waveform')
SPECS={}


def profile_specs(dep,seed):
    key=dep,seed
    if key not in SPECS:
        SPECS[key]={name:json.loads((root/f'calibration/{dep}_{seed}_JOINT.json').read_text())
                    for name,root in REFINERS.items()}
    return SPECS[key]


def panel(dep,seed,split,catalog=None):
    frame=cv.panel(dep,seed,split,catalog)
    original=engine.ORIGINAL_PANEL(dep,seed,split,catalog)
    if not np.array_equal(frame[['idx_i','idx_j']],original[['idx_i','idx_j']]):
        raise RuntimeError('Panel identity differs')
    old=h.isolated.ORIGINALS[dep,seed]
    for kind,spec in profile_specs(dep,seed).items():
        z,oo,clip=co.BASE_INFER(original,{**old,'method':'FROZEN_PROFILE_REF',
            'joint_calibration_subgrid':spec})
        frame['profile_ref_'+kind]=z;frame['profile_ref_OOD_'+kind]=oo;frame['profile_ref_clip_'+kind]=clip
    frame['profile_reference_joint_BC']=original.joint_BC.to_numpy()
    return frame


def infer(frame,config):
    if config['method']==METHODS[0]:
        return st.infer(frame,config)
    reference=st.infer(frame,{**config['fallback'],'method':'PARENT_GLOBAL'})
    frame['quality_reference_waveform']=reference[0]
    frame['profile_envelope_used']=False
    frame['profile_reference_waveform']=np.nan
    if config['method']==METHODS[1]:
        return reference
    kind=config['profile_reference']
    candidate=(frame['profile_ref_'+kind].to_numpy(float),frame['profile_ref_OOD_'+kind].to_numpy(bool),
               frame['profile_ref_clip_'+kind].to_numpy(bool))
    take=(st.states(frame)==2)&(candidate[0]<reference[0])
    result=tuple(np.where(take,new,old)for new,old in zip(candidate,reference))
    if np.any(result[0]>reference[0])or any(not np.array_equal(a[~take],b[~take])for a,b in zip(result,reference)):
        raise RuntimeError('Noncompensating exact fallback failed')
    frame['profile_envelope_used']=take
    frame['profile_reference_waveform']=candidate[0]
    frame['actual_waveform_joint_BC']=np.where(take,frame.profile_reference_joint_BC,frame.actual_waveform_joint_BC)
    return result


def freeze():
    if ROOT.exists():
        raise RuntimeError('Independent output required')
    parents={(c['deployment'],c['seed']):c for c in cv.frozen_parent()if c['method']=='JOINTSTATE-GLOBAL'}
    for folder in ('contracts','configs','calibration','tables','audit','reports','scripts','logs','manifest','results','figures'):
        (ROOT/folder).mkdir(parents=True)
    n.write_json(ROOT/'contracts/ANALYSIS_CONTRACT.json',{'UTC':n.utc(),
        'id':'MCWF-NODUP-QUALITY-PROFILE-ENVELOPE-47','status':n.STATUS,
        'parent':str(PARENT),'profile_references':{k:str(v)for k,v in REFINERS.items()},
        'hypothesis':'Quality-state calibration improves cross-availability score comparability,while two reliable profiles can supply conservative incompatibility evidence. Combine by a fixed lower envelope,not positive compensation.',
        'rule':'If BOTH endpoint R10 profiles valid: min(R35waveform,frozenR10orR12profilewaveform);otherwise exactR35. Same rule bothruns.',
        'one_component':'Cosine calibration,gamma,beta are identical across the two waveform expressions. The minimum chooses one joint-consistency contribution;it does not sum two mass evidence terms.',
        'not_total_blend':True,'no_free_mixture_weight':True,'no_old_encoder_Mc_q':True,
        'statistical_limit':'Lower envelope of correlated empirical ranking scores is a conservative triage heuristic,NOT a normalized Bayes factor or proof of calibrated posterior odds.',
        'parent_data_limit':'R10 error calibration had only38-39fit sources. R35 used independent expanded fit/tune. Both references remain explicit;smaller calibration is not silently called the newlarger population.',
        'methods':list(METHODS),'no_new_hyperparameter_selection':True,
        'fixed':['allencoders','allmassdistributions','alljointcalibrations','profilequality',
                 'time','sky','gamma','beta','outerweights','scope','oldresults'],
        'checks':'NODUP andR35 replay;R26 paired-envelope replay against saved outputs;forbidden field poisoning;row-order invariance;no positive scoreincrease;exactinactivefallback.',
        'evaluation':'Allarms,bothruns,3seeds,reusedinjectionandrealPE/officialaudits;comparebothNDandPATH. No winner selected by real labels.',
        'adaptive_development':True,'new_blind_confirmation':False,'goal_achieved':False,
        'physical_motivation':'Waveform intrinsic consistency and model approximation uncertainty. The minimum itself is an engineering rule to be tested,not prescribed by a cited physics paper.'})
    configs=[];rows=[];units=[];paths=[Path(__file__),PARENT/'configs/SELECTED_CONFIGURATIONS.json']
    for dep in n.DEPS:
        for seed in n.SEEDS:
            old=h.isolated.ORIGINALS[dep,seed];frame=panel(dep,seed,'validation')
            cb={**old,'method':METHODS[0]};nd=infer(frame,cb)[0]
            fixed={**old,'method':METHODS[1],'fallback':parents[dep,seed]}
            ref=infer(frame,fixed)[0]
            oldframe=pd.read_parquet(PARENT/f'results/JOINTSTATE-GLOBAL/{dep}/seed_{seed}/validation/pairs.parquet')
            if not np.array_equal(ref,oldframe.waveform_score.to_numpy()):
                raise RuntimeError('R35 replay failed')
            configs.extend([cb,fixed])
            refs=((n.cf.fast_metrics(frame,n.cf.channels(frame,nd)@np.asarray(old['weights'])),n.cf.fast_metrics(frame,nd)),
                  (n.cf.fast_metrics(frame,frame.PATH875_final_score.to_numpy()),n.cf.fast_metrics(frame,frame.PATH875_waveform.to_numpy())))
            for method,kind in zip(METHODS[2:],('GLOBAL','CONDITIONAL')):
                config={**deepcopy(fixed),'method':method,'profile_reference':kind}
                z=infer(frame,config)[0]
                fm=n.cf.fast_metrics(frame,n.cf.channels(frame,z)@np.asarray(old['weights']));wm=n.cf.fast_metrics(frame,z)
                config['tune_guard']=all(n.cf.guard(fm,f)and n.cf.guard(wm,w)for f,w in refs)
                config['tune_metrics']=fm;configs.append(config)
                legacy=P/f'results/mcwf_nodup_paired_profile_26_20260909T120700Z/results/PAIRED-PROFILE-{kind}/{dep}/seed_{seed}/validation/pairs.parquet'
                replay=np.where(st.states(frame)==2,np.minimum(nd,frame['profile_ref_'+kind]),nd)
                expected=pd.read_parquet(legacy).waveform_score.to_numpy()
                delta=float(abs(replay-expected).max())
                if delta>1e-12:
                    raise RuntimeError('R26 refiner replay failed:'+str(delta))
                poison=frame.copy();poison['pair_key']='unused';poison['pe_mc_bhattacharyya_coefficient']=-999.;poison['official_po_fpp']=1.
                if not np.array_equal(z,infer(poison,{**config,'alpha':-999.,'old_Mc_weight':999.})[0]):
                    raise RuntimeError('Forbidden field effect')
                if not np.array_equal(z,infer(frame.iloc[::-1].copy(),config)[0][::-1]):
                    raise RuntimeError('Row order effect')
                rows.append({'deployment':dep,'seed':seed,'method':method,'guard':config['tune_guard'],**fm})
                units.append({'deployment':dep,'seed':seed,'method':method,'R26_replay_max_delta':delta,
                    'inactive_exact':True,'max_score_increase':float((z-ref).max()),'PE_official_delta':0.,'row_order_delta':0.})
                paths.extend([REFINERS[kind]/f'calibration/{dep}_{seed}_JOINT.json',legacy])
            print('QUALITY_ENVELOPE_FROZEN',dep,seed,flush=True)
    n.write_csv(ROOT/'tables/VALIDATION_GUARDS.csv',rows);n.write_csv(ROOT/'audit/INVARIANCE.csv',units)
    n.write_csv(ROOT/'manifest/INPUT_SHA256.csv',[{'path':str(p),'sha256':n.sha(p),'bytes':p.stat().st_size}for p in paths])
    n.write_json(ROOT/'configs/SELECTED_CONFIGURATIONS.json',configs)
    n.write_json(ROOT/'contracts/CONFIGURATIONS_FROZEN.json',{'UTC':n.utc(),
        'file':'configs/SELECTED_CONFIGURATIONS.json','sha256':n.sha(ROOT/'configs/SELECTED_CONFIGURATIONS.json'),
        'no_new_hyperparameter_selection':True,'no_real_or_test_selection':True})
    shutil.copy2(__file__,ROOT/'scripts/quality_profile_envelope.py')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--parent',type=Path,required=True)
    parser.add_argument('--stage',choices=('freeze','evaluate','real'),required=True)
    args=parser.parse_args();ROOT,PARENT=args.root,args.parent
    cv.ROOT,cv.PARENT=ROOT,PARENT
    base.s.ROOT=h.ROOT=co.ROOT=r.ROOT=co.score.ROOT=ROOT;engine.ROOT=PARENT
    r.install();st.FIELDS=FIELDS;st.install_export()
    st.METHODS=(METHODS[0],'PARENT_GLOBAL','UNUSED_REJECT')
    co.score.matrices=co.matrices;n.METHODS,n.load_panel,n.infer=METHODS,panel,infer
    if args.stage=='freeze':
        freeze()
    else:
        n.run(ROOT,args.stage)
