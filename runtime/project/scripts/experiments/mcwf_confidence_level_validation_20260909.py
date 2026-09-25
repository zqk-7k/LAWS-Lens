#!/usr/bin/env python3
"""Validation-only selection of an empirical source-confidence attenuation."""
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
import mcwf_quality_state_joint_calibration_20260909 as st
ctrl,engine,base,n,r,co,h=(getattr(st,k)for k in ('ctrl','engine','base','n','r','co','h'))
ROOT=PARENT=None
LEVELS=(.01,.025,.05,.10)
FIXED={f'CONFIDENCE-ALPHA-{level:g}':level for level in LEVELS}
METHODS=('NODUP-DIRECT-REPLAY',*FIXED,'CONFIDENCE-VALIDATION-SELECTED')
ALPHA=.05
ORIGINAL_TERMS=st.terms
ORIGINAL_INFER=st.infer


def frozen_parent():
    receipt=json.loads((PARENT/'contracts/CONFIGURATIONS_FROZEN.json').read_text())
    path=PARENT/receipt['file']
    if n.sha(path)!=receipt['sha256']:
        raise RuntimeError('Frozen state-calibration changed')
    return json.loads(path.read_text())


def freeze():
    if ROOT.exists():
        raise RuntimeError('Independent result directory required')
    frozen_parent()
    for folder in ('contracts','configs','calibration','tables','audit','reports','scripts','logs','manifest','results','figures'):
        (ROOT/folder).mkdir(parents=True)
    n.write_json(ROOT/'contracts/ANALYSIS_CONTRACT.json',{'UTC':n.utc(),
        'id':'MCWF-NODUP-CONFIDENCE-VALIDATION-38','status':n.STATUS,
        'parent':str(PARENT),'change':'Only joint true-tail confidence alpha;predictive distributions,LRcalibration,gamma,beta,outerweights unchanged.',
        'levels':list(LEVELS),'motivation':'Compare conventional99/97.5/95/90percent empirical consistency tolerances and false-rejection cost,not chooseoneusingrealPE.',
        'selection_per_run_seed':['current ND waveformANDfusion guard','F50','F90','-AUPRC','-R10','-R1','distance from .05','alpha'],
        'no_passing_level':'Keep failedselection forcompleteexploratory reporting,marknoteligible;do not declareupgrade.',
        'report_all_levels':True,'same_framework_both_runs':True,'real_not_select_alpha':True,
        'not_a_distribution_free_guarantee':'Predictorandreferenceusefitdata;methoddirectionsareadaptive.Reusedtune/test andnoise correlations preclude a blindconformalcoverageclaim.',
        'finite_resolution':'Analpha below1/(truefitcount+1)cannottriggerthatstate;report this,do not fabricate smallerp.',
        'not_total_score_blend':True,'legacy_alpha_unused':True,'new_field':'confidence_alpha',
        'no_old_encoder_Mc_q':True,'time_sky_frozen':True,'adaptive_real_feedback':True,
        'script_sha256':n.sha(Path(__file__))})
    n.write_json(ROOT/'contracts/START_FREEZE.json',{'UTC':n.utc(),
        'contract_sha256':n.sha(ROOT/'contracts/ANALYSIS_CONTRACT.json'),
        'parent_configuration_sha256':n.sha(PARENT/'configs/SELECTED_CONFIGURATIONS.json')})
    shutil.copy2(__file__,ROOT/'scripts/confidence_level_validation.py')


def panel(dep,seed,split,catalog=None):
    slot=n.recipes()[dep,seed]['slot']
    tag=co.app.tag_for(seed,split,catalog)
    paneltag=split if catalog is None else f'{split}_{catalog}'
    paths=[PARENT/f'predictions/{dep}/{slot}_{tag}.npz',
           PARENT/f'cache/panels/{dep}_{seed}_{paneltag}.npz']
    if not all(path.exists()for path in paths):
        raise RuntimeError('Parent features must alreadyexist;no writing toparent')
    return ctrl.panel(dep,seed,split,catalog)


def terms(value,state,endpoint,specs):
    tail,_,_,outside=ORIGINAL_TERMS(value,state,endpoint,specs)
    inc=np.empty(len(value))
    for k in range(3):
        mask=state==k;spec=specs[str(k)]
        inc[mask]=(np.interp(value[mask],spec['knots'],spec['loglr'])+spec['state_log_ratio']).clip(-4.,4.)
    inc[outside|(tail<ALPHA)]=np.minimum(inc[outside|(tail<ALPHA)],0.)
    penalty=np.minimum(np.log(tail/ALPHA),0.)
    inc[endpoint],penalty[endpoint]=0.,0.
    return tail,inc,penalty,outside


def infer(frame,config):
    global ALPHA
    ALPHA=float(config.get('confidence_alpha',.05))
    frame['waveform_confidence_alpha']=ALPHA
    return ORIGINAL_INFER(frame,config)


def calibrate():
    parent={(c['deployment'],c['seed']):c for c in frozen_parent()if c['method']=='JOINTSTATE-GLOBAL'}
    configs,grid,units=[],[],[]
    for dep in n.DEPS:
        for seed in n.SEEDS:
            old=h.isolated.ORIGINALS[dep,seed]
            frame=panel(dep,seed,'validation')
            cref={**old,'method':METHODS[0]}
            ref=infer(frame,cref)[0]
            fm=n.cf.fast_metrics(frame,n.cf.channels(frame,ref)@np.asarray(old['weights']))
            wm=n.cf.fast_metrics(frame,ref)
            configs.append(cref)
            options=[]
            for method,alpha in FIXED.items():
                config={**deepcopy(parent[dep,seed]),'method':method,'confidence_alpha':alpha}
                z=infer(frame,config)[0]
                metric=n.cf.fast_metrics(frame,n.cf.channels(frame,z)@np.asarray(old['weights']))
                waveform=n.cf.fast_metrics(frame,z)
                guard=n.cf.guard(metric,fm)and n.cf.guard(waveform,wm)
                config.update(tune_guard=guard,tune_metrics=metric)
                configs.append(config)
                row={'deployment':dep,'seed':seed,'method':method,'confidence_alpha':alpha,'guard':guard,**metric}
                grid.append({**row,**{'waveform_'+k:v for k,v in waveform.items()}})
                options.append((row,config))
                poison=frame.copy()
                poison['pair_key'],poison['pe_mc_bhattacharyya_coefficient'],poison['official_po_fpp']='unused',-999.,1.
                if not np.array_equal(z,infer(poison,{**config,'alpha':-999.,'old_Mc_weight':999.})[0]):
                    raise RuntimeError('Forbidden scoreinputeffect')
                if alpha==.05:
                    path=PARENT/f'results/JOINTSTATE-GLOBAL/{dep}/seed_{seed}/validation/pairs.parquet'
                    archived=pd.read_parquet(path)
                    if not np.array_equal(z,archived.waveform_score.to_numpy()):
                        raise RuntimeError('Alpha.05doesnotreplayfrozenparent')
                units.append({'deployment':dep,'seed':seed,'method':method,'forbidden_delta':0.,
                    'alpha005_parent_delta':0. if alpha==.05 else None,'weights_gamma_beta_unchanged':True})
            passing=[item for item in options if item[0]['guard']]
            row,selected=min(passing or options,key=lambda item:(item[0]['false_at_recall_0p5'],
                item[0]['false_at_recall_0p9'],-item[0]['average_precision'],-item[0]['macro_r_at_10'],
                -item[0]['macro_r_at_1'],abs(item[0]['confidence_alpha']-.05),item[0]['confidence_alpha']))
            selected={**deepcopy(selected),'method':METHODS[-1],'tune_guard':bool(passing)}
            configs.append(selected)
            print('CONFIDENCE_SELECTED',dep,seed,row['confidence_alpha'],bool(passing),flush=True)
    n.write_csv(ROOT/'tables/CONFIDENCE_VALIDATION_GRID.csv',grid)
    n.write_csv(ROOT/'audit/INVARIANCE.csv',units)
    n.write_json(ROOT/'configs/SELECTED_CONFIGURATIONS.json',configs)
    n.write_json(ROOT/'contracts/CONFIGURATIONS_FROZEN.json',{'UTC':n.utc(),
        'file':'configs/SELECTED_CONFIGURATIONS.json','sha256':n.sha(ROOT/'configs/SELECTED_CONFIGURATIONS.json'),
        'no_real_or_test_selection':True})


def main():
    global ROOT,PARENT
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--parent',type=Path,required=True)
    parser.add_argument('--stage',choices=('freeze','calibrate','evaluate','real'),required=True)
    args=parser.parse_args();ROOT,PARENT=args.root,args.parent
    base.s.ROOT=h.ROOT=co.ROOT=r.ROOT=co.score.ROOT=ROOT
    engine.ROOT=PARENT
    r.install()
    st.FIELDS=st.FIELDS+('waveform_confidence_alpha',)
    st.install_export()
    st.terms=terms
    st.METHODS=(METHODS[0],'GENERIC-GLOBAL','UNUSED-REJECT-METHOD')
    co.score.matrices=co.matrices
    n.METHODS,n.load_panel,n.infer=METHODS,panel,infer
    if args.stage in ('freeze','calibrate'):
        globals()[args.stage]()
    else:
        n.run(ROOT,args.stage)


if __name__=='__main__':
    main()
