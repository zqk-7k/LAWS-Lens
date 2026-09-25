#!/usr/bin/env python3
"""Separate nuisance-matched null reference for waveform consistency."""
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
import mcwf_waveform_internal_validation_20260909 as iv
cv,st,engine,base,n,r,co,h=(getattr(iv,k)for k in ('cv','st','engine','base','n','r','co','h'))
ROOT=PARENT=None
METHODS=('NODUP-DIRECT-REPLAY','JOINTSTATE-UNCONDITIONAL-FROZEN',
         'QUALITY-MATCHED-REFERENCE-FIXED','QUALITY-MATCHED-REFERENCE-VALIDATED')


def freeze():
    if ROOT.exists():
        raise RuntimeError('Independent output required')
    cv.frozen_parent()
    for folder in ('contracts','configs','tables','audit','reports','scripts','logs','manifest','results','figures'):
        (ROOT/folder).mkdir(parents=True)
    n.write_json(ROOT/'contracts/ANALYSIS_CONTRACT.json',{'UTC':n.utc(),
        'id':'MCWF-NODUP-QUALITY-MATCHED-NULL-42','status':n.STATUS,
        'scientific_difference':'Change waveform null reference, not silently omit a normalization term. All historical joint-LR offsets were correct for their unconditional null.',
        'old_reference':'pN(x,A)=pN(x|A)pN(A)',
        'matched_reference':'qN(x,A)=pN(x|A)pL(A), qL(x,A)=pL(x|A)pL(A); their ratio is pL(x|A)/pN(x|A).',
        'motivation':'Test whether predictive-quality state occurrence provides a simulator-specific shortcut. Conditional discrimination compares true and false pairs with the same profile-availability state.',
        'implementation':'Use existing class-balanced within-state jointBC calibration. Explicitly set ONLY state occurrence logratios to0; do not remove the conditional LR or reverse its sign.',
        'A':'Zero,one,two R10-quality-valid waveform profiles, determined without truth/PE/official metadata.',
        'not_a_global_Bayes_factor':'Nuisance-matched null is NOT the unconditional catalog null. Scores cannot be presented as exact catalog odds/FPP.',
        'arms':list(METHODS),'fixed_arm':'Original NODUP gamma,beta and alpha=.05.',
        'validated_arm':{'gamma':list(iv.GAMMAS),'beta':list(iv.BETAS),'confidence_alpha':list(cv.LEVELS)},
        'selection':'Existing BOTH-NODUP/PATH waveform/fusion validation guards,thenF50,F90,-AP,-R10,-R1,nearestfrozencoefficients,lexicographic. Perseed/run. All grid points reported.',
        'no_passing_point':'Keep failure; no automatic upgrade.',
        'same_framework_both_runs':True,'no_old_encoder_Mc_q':True,'no_total_blend':True,
        'frozen':['encoder','eventpredictivedensities','withinstateconditionalLR','time','sky','outerweights','scope','historicalresults'],
        'adaptive_real_feedback':True,'not_blind_confirmation':True,
        'real_PE_official_not_inputs_or_selection':True,
        'reference':'https://arxiv.org/abs/1506.02169',
        'reference_limit':'Classifier likelihood-ratio interpretation, not proof that this matched reference improves GW retrieval.'})
    paths=[Path(__file__),Path(iv.__file__),PARENT/'configs/SELECTED_CONFIGURATIONS.json']
    n.write_csv(ROOT/'manifest/INPUT_SHA256.csv',[{'path':str(p),'sha256':n.sha(p),'bytes':p.stat().st_size}for p in paths])
    shutil.copy2(__file__,ROOT/'scripts/quality_matched_reference.py')
    n.write_json(ROOT/'contracts/START_FREEZE.json',{'UTC':n.utc(),'contract_sha256':n.sha(ROOT/'contracts/ANALYSIS_CONTRACT.json')})


def calibrate():
    parents={(c['deployment'],c['seed']):c for c in cv.frozen_parent()if c['method']=='JOINTSTATE-GLOBAL'}
    configs=[];grid=[];selected=[];units=[]
    for dep in n.DEPS:
        for seed in n.SEEDS:
            old=h.isolated.ORIGINALS[dep,seed];frame=cv.panel(dep,seed,'validation')
            cbase={**old,'method':METHODS[0]};ref=iv.infer(frame,cbase)[0]
            refs=((n.cf.fast_metrics(frame,n.cf.channels(frame,ref)@np.asarray(old['weights'])),n.cf.fast_metrics(frame,ref)),
                  (n.cf.fast_metrics(frame,frame.PATH875_final_score.to_numpy()),n.cf.fast_metrics(frame,frame.PATH875_waveform.to_numpy())))
            control={**deepcopy(parents[dep,seed]),'method':METHODS[1],'confidence_alpha':.05}
            archived=pd.read_parquet(PARENT/f'results/JOINTSTATE-GLOBAL/{dep}/seed_{seed}/validation/pairs.parquet')
            if not np.array_equal(iv.infer(frame,control)[0],archived.waveform_score.to_numpy()):
                raise RuntimeError('Unconditional reference changed')
            fixed={**deepcopy(control),'method':METHODS[2]}
            for spec in fixed['state_specs'].values():
                spec['original_state_log_ratio']=spec['state_log_ratio'];spec['state_log_ratio']=0.
            fz=iv.infer(frame,fixed)[0]
            ff=n.cf.fast_metrics(frame,n.cf.channels(frame,fz)@np.asarray(old['weights']))
            fw=n.cf.fast_metrics(frame,fz)
            fixed.update(tune_guard=all(n.cf.guard(ff,a)and n.cf.guard(fw,b)for a,b in refs),tune_metrics=ff)
            configs.extend([cbase,control,fixed]);options=[]
            for alpha in cv.LEVELS:
                for gamma in iv.GAMMAS:
                    for beta in iv.BETAS:
                        c={**deepcopy(fixed),'confidence_alpha':alpha,'gamma':gamma,'beta':beta}
                        z=iv.infer(frame,c)[0]
                        fm=n.cf.fast_metrics(frame,n.cf.channels(frame,z)@np.asarray(old['weights']));wm=n.cf.fast_metrics(frame,z)
                        guard=all(n.cf.guard(fm,a)and n.cf.guard(wm,b)for a,b in refs)
                        row={'deployment':dep,'seed':seed,'confidence_alpha':alpha,'gamma':gamma,'beta':beta,
                            'both_guard':guard,'distance':abs(gamma-old['gamma'])/4+abs(np.log2(beta/old['beta']))+abs(alpha-.05)/.05,
                            **fm,**{'waveform_'+k:v for k,v in wm.items()}}
                        grid.append(row);options.append((row,c))
            passing=[v for v in options if v[0]['both_guard']]
            row,c=min(passing or options,key=lambda item:iv.key(item[0]))
            c.update(method=METHODS[3],tune_guard=bool(passing),tune_metrics=row)
            configs.append(c);selected.append(row)
            for config in (fixed,c):
                z=iv.infer(frame,config)[0];poison=frame.copy()
                poison['pair_key']='unused';poison['pe_mc_bhattacharyya_coefficient']=-999.;poison['official_po_fpp']=1.
                if not np.array_equal(z,iv.infer(poison,{**config,'alpha':-999.,'old_Mc_weight':999.})[0]):
                    raise RuntimeError('Forbidden input changed waveform')
                units.append({'deployment':dep,'seed':seed,'method':config['method'],
                    'original_reference_delta':0.,'forbidden_input_delta':0.,'outer_weights_unchanged':True})
            print('QUALITY_MATCHED_SELECTED',dep,seed,row['confidence_alpha'],row['gamma'],row['beta'],bool(passing),flush=True)
    n.write_csv(ROOT/'tables/MATCHED_REFERENCE_GRID.csv',grid)
    n.write_csv(ROOT/'tables/MATCHED_REFERENCE_SELECTION.csv',selected)
    n.write_csv(ROOT/'audit/INVARIANCE.csv',units)
    n.write_json(ROOT/'configs/SELECTED_CONFIGURATIONS.json',configs)
    n.write_json(ROOT/'contracts/CONFIGURATIONS_FROZEN.json',{'UTC':n.utc(),
        'file':'configs/SELECTED_CONFIGURATIONS.json','sha256':n.sha(ROOT/'configs/SELECTED_CONFIGURATIONS.json'),
        'real_or_test_selection':False})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--parent',type=Path,required=True)
    parser.add_argument('--stage',choices=('freeze','calibrate','evaluate','real'),required=True)
    args=parser.parse_args();ROOT,PARENT=args.root,args.parent
    iv.ROOT,iv.PARENT=ROOT,PARENT;cv.ROOT,cv.PARENT=ROOT,PARENT
    base.s.ROOT=h.ROOT=co.ROOT=r.ROOT=co.score.ROOT=ROOT;engine.ROOT=PARENT
    r.install();st.FIELDS=iv.FIELDS;st.install_export();st.METHODS=(METHODS[0],'PARENT','UNUSED_REJECT')
    co.score.matrices=co.matrices
    n.METHODS,n.load_panel,n.infer=METHODS,cv.panel,iv.infer
    if args.stage in ('freeze','calibrate'):
        globals()[args.stage]()
    else:
        n.run(ROOT,args.stage)
