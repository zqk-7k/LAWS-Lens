#!/usr/bin/env python3
"""Pipeline frozen predictions as independent run data become complete."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import sys
import time
import numpy as np
import pandas as pd

P=Path('/root/autodl-tmp/gw-catalog')
sys.path.insert(0,str(P/'scripts/experiments'))
import mcwf_independent_profile_features_20260909 as f
n=f.n
STATE={}


def initialize(root,dep):
    f.init_worker(root,dep)
    constructor=f.prof.profile.LowBand
    def captured(*args,**kwargs):
        model=constructor(*args,**kwargs)
        STATE['model']=model
        STATE['raw24']=args[0]
        return model
    f.prof.profile.LowBand=captured


def event(job):
    tick=time.perf_counter()
    row,feature=job
    fit=f.prof.fit_one((row,feature))
    fit['wall_seconds']=time.perf_counter()-tick
    point=np.asarray([fit['logmc'],fit['q'],fit['chieff_equal']])
    info=f.info.information(STATE['model'],point,STATE['raw24'])
    info.update(row_index=fit['row_index'],deployment=fit['deployment'],
                source_uid=fit['source_uid'],old_peak2s_bitexact=True)
    prior={'row_index':fit['row_index'],'valid':False,'reference_not_physicalPE':True}
    if info['information_valid']:
        try:
            m64,s64,ess64=f.bounded.bounded_moments(info,fit,64)
            m128,s128,ess128=f.bounded.bounded_moments(info,fit,128)
            delta=abs(s64/s128-1)
            valid=bool(np.isfinite([m128,s128]).all() and s128>0 and delta<=.05 and
                abs(m64-m128)<=.001 and ess128>=10 and m128-5*s128>f.d.EDGES[0] and
                m128+5*s128<f.d.EDGES[-1])
            prior.update(valid=valid,mean64=m64,mean128=m128,width64=s64,width128=s128,
                width_relative_difference=delta,effective_nodes64=ess64,effective_nodes128=ess128,
                mean_shift_from_profile=m128-fit['logmc'])
        except (np.linalg.LinAlgError,FloatingPointError,RuntimeError) as error:
            prior['reason']=type(error).__name__+':'+str(error)
    info['total_wall_seconds']=time.perf_counter()-tick
    return fit,info,prior


def profiles(root,dep,workers):
    plan=root/f'contracts/{dep}_PROFILE_EVENT_PLAN.parquet'
    frame=pd.read_parquet(plan)
    folder=root/f'profile_events/{dep}'
    infofolder=root/f'profile_information/{dep}'
    priorfolder=root/f'profile_bounded_information/{dep}'
    for p in (folder,infofolder,priorfolder):
        p.mkdir(parents=True,exist_ok=True)
    feature=np.load(root/f'features/{dep}/peak.npy',mmap_mode='r')
    jobs=[(row,feature[int(row['row_index'])]) for row in frame.to_dict('records')
          if not all((p/f'{int(row["row_index"])}.json').exists() for p in (folder,infofolder,priorfolder))]
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn'),
            initializer=initialize,initargs=(str(root),dep)) as pool:
        for count,(fit,info,prior) in enumerate(pool.map(event,jobs),1):
            for p,row in ((folder,fit),(infofolder,info),(priorfolder,prior)):
                path=p/f'{fit["row_index"]}.json'
                if path.exists():
                    old=json.loads(path.read_text())
                    for key in ('logmc','q','chieff_equal','projection_statistic'):
                        if key in row and abs(row[key]-old[key])>1e-12:
                            raise RuntimeError('Restart changed frozen point fit')
                else:
                    n.write_json(path,row)
            if count%32==0 or count==len(jobs):
                print('INDEPENDENT_PROFILE_AND_INFORMATION',dep,count,len(jobs),flush=True)
    n.write_json(root/f'contracts/{dep}_PROFILES_COMPLETE.json',{'UTC':n.utc(),'events':len(frame),
        'plan_sha256':n.sha(plan),'expected_information':True,'bounded_information':True,
        'point_fit_unchanged':True,'fit_tune_only':True,'real_or_test_read':False})


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--producer-pid',type=int,required=True)
    parser.add_argument('--workers',type=int,default=12)
    args=parser.parse_args();root=args.root
    f.ROOT=root
    path=root/'contracts/READY_PROFILE_PIPELINE_ADDENDUM.json'
    if not path.exists():
        n.write_json(path,{'UTC':n.utc(),'execution_only':True,'independent_run_pipelining':True,
            'feature_profile_contract_sha256':n.sha(root/'contracts/FEATURE_PROFILE_CONTRACT.json'),
            'point_profile':'Exact frozen R7/R10 four-start optimizer, initializer, likelihood surrogate and quality; no additional refit.',
            'information':'Complete the already listed R18/R21 expected information and bounded-reference-prior diagnostics at the unchanged optimum. Reuse the waveform array and fitted object, not truth or PE.',
            'numerics':'Same derivative, condition,64/128quadrature,5percentwidth,.001logmean,ESS10,massbound safeguards.',
            'not_a_new_PE_posterior':True,'for_scientific_support_audit_not_real_selection':True,
            'original_feature_code_sha256':n.sha(Path(f.__file__)),'runtime_sha256':n.sha(Path(__file__))})
        shutil.copy2(__file__,root/'scripts/ready_profile_pipeline.py')
    start=time.monotonic()
    for dep in n.DEPS:
        while not (root/f'data/{dep}/COMPLETE.json').exists():
            if not Path(f'/proc/{args.producer_pid}').exists():
                raise RuntimeError('HOLD_DATA_PRODUCER_STOPPED:'+dep)
            if time.monotonic()-start>6*3600:
                raise RuntimeError('HOLD_DATA_WAIT_LIMIT')
            if shutil.disk_usage(root).free<25*2**30:
                raise RuntimeError('HOLD_DISK_LIMIT')
            time.sleep(15)
        print('INDEPENDENT_RUN_READY',dep,n.utc(),flush=True)
        f.numerical_test()
        if not (root/f'contracts/{dep}_PREDICTIONS_COMPLETE.json').exists():
            f.features(dep)
            f.predict(dep)
        if not (root/f'contracts/{dep}_PROFILES_COMPLETE.json').exists():
            profiles(root,dep,args.workers)
    n.write_json(root/'contracts/FROZEN_FEATURE_PROFILE_ALL_COMPLETE.json',{'UTC':n.utc(),
        'same_models_and_physical_fits':True,'no_new_encoder_training':True})


if __name__=='__main__':
    main()
