#!/usr/bin/env python3
"""Paired simulated-only D versus XAS equal-spin profile comparison."""
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
from scipy import signal
from pycbc.waveform import get_fd_waveform

P=Path('/root/autodl-tmp/gw-catalog')
sys.path.insert(0,str(P/'scripts/experiments'))
import mcwf_inspiral_profile_pilot_20260909 as band
base,prof,n=band.base,band.prof,band.n
ROOT=DATA=PLAN=None
SEED=2026090945


class ApproximantProfile(prof.profile.LowBand):
    def __init__(self,raw,frequency,psd,approximant):
        self.approximant=approximant
        super().__init__(raw,frequency,psd,16)

    def template(self,x):
        logmc,q,chi=x
        m1=np.exp(logmc)*(1+q)**.2/q**.6
        hp,_=get_fd_waveform(approximant=self.approximant,mass1=m1,mass2=m1*q,
            spin1z=chi,spin2z=chi,delta_f=1/26,f_lower=20.,f_final=2048.,
            distance=1000.,inclination=0.)
        hp.resize(base.RAW_N//2+1)
        a=signal.hilbert(np.fft.irfft(np.asarray(hp),n=base.RAW_N))
        a=np.roll(a,int(24.75*4096)-int(np.argmax(abs(a))))
        phases=[]
        for phase in (a.real,a.imag):
            z=base.STATE['v3'].preprocess_24s(np.stack([phase,phase]),self.frequency,self.psd,band_low_hz=20.)
            phases.append(z[...,-self.length:])
        bank=np.stack(phases,1).astype(float)
        bank-=bank.mean(-1,keepdims=True)
        u,v=bank[:,0],bank[:,1]
        u/=np.linalg.norm(u,axis=-1,keepdims=True)
        v-=np.sum(u*v,-1,keepdims=True)*u
        v/=np.linalg.norm(v,axis=-1,keepdims=True)
        if not np.isfinite(bank).all():
            raise RuntimeError('Nonfinite approximant template')
        return bank


def freeze():
    if ROOT.exists():
        raise RuntimeError('Independent output required')
    for folder in ('contracts','events','tables','audit','reports','scripts','logs','manifest'):
        (ROOT/folder).mkdir(parents=True)
    paths=[Path(__file__),Path(band.__file__)]
    for dep in n.DEPS:
        path=PLAN/f'contracts/{dep}_PILOT_PLAN.parquet'
        frame=pd.read_parquet(path)
        source=frame[['source_uid','fit_tune_fold']].drop_duplicates()
        if source.source_uid.duplicated().any()or source.groupby('fit_tune_fold').size().to_dict()!={0:24,1:24}:
            raise RuntimeError('Source support/split changed')
        noise=pd.read_csv(DATA/f'data/{dep}/noise/noise_manifest.csv').set_index('bank_index')
        chunk=frame.noise_bank_index.map(noise.parent_file_gps)
        if set(chunk[frame.fit_tune_fold==0])&set(chunk[frame.fit_tune_fold==1]):
            raise RuntimeError('Parent noise overlaps folds')
        shutil.copy2(path,ROOT/f'contracts/{dep}_PILOT_PLAN.parquet')
        paths.extend([path,DATA/f'data/{dep}/noise/noise_manifest.csv'])
        paths.extend(DATA/f'profile_events/{dep}/{int(i)}.json'for i in frame.row_index)
    n.write_json(ROOT/'contracts/ANALYSIS_CONTRACT.json',{
        'UTC':n.utc(),'id':'MCWF-NODUP-APPROXIMANT-PILOT-45','status':n.STATUS,
        'data':str(DATA),'pilot_plan_source':str(PLAN),'source_count_per_run_fold':24,
        'selection':'Same frozen hash-selected waveform-only lowpredictedmass sources as R40,both images retained. No selection by true error,public PE or candidate identity.',
        'one_changed_factor':'IMRPhenomD -> IMRPhenomXAS template approximant. Both use equal aligned spins,dominantmode;not a precession/highmode upgrade.',
        'fixed':['20-580Hz','16s','four waveform-feature starts','maxfev300 perstart',
                 'quadrature and time maximization','raw signal/noise/SNR scaling','original2s bitexact',
                 'NN checkpoints','time','sky','outerweights','real scope','oldresults'],
        'why':'Different frequency-domain phase approximations may alter profile bias. Literature motivates testing,not assuming an improvement.',
        'gate':'Per run tune source-mean logMc absolute error,sourceq90 error,and >10percent errorfraction must not exceed pairedD reference;atleastone runMAE strictly improves. Bothruns required.',
        'bootstrap':{'draws':5000,'seed':SEED,'unit':'source with bothimages,conditional on reusednoise'},
        'optimizer':'Identical original Nelder-Mead,bounds,tolerance,maxfev and starts. Report allconvergenceflags;no selecting bestseed.',
        'expansion':'Only both-run pilotPASS can lead to new independent error calibration and scoring. No real/test data read here.',
        'not_full_PE':True,'not_Bayes_evidence':True,'adaptive_development':True,
        'reference':'https://arxiv.org/abs/2001.11412',
        'reference_limit':'XAS improves the underlying aligned waveform modeling;it does not prove equalspin profile,existing preprocessing or this pilot optimal.',
        'goal_achieved':False})
    n.write_csv(ROOT/'manifest/INPUT_SHA256.csv',[{'path':str(p),'sha256':n.sha(p),'bytes':p.stat().st_size}for p in paths])
    shutil.copy2(__file__,ROOT/'scripts/profile_approximant_pilot.py')
    n.write_json(ROOT/'contracts/START_FREEZE.json',{'UTC':n.utc(),
        'contract_sha256':n.sha(ROOT/'contracts/ANALYSIS_CONTRACT.json'),'script_sha256':n.sha(Path(__file__))})


def unit():
    rows=[]
    for dep in n.DEPS:
        band.features.init_worker(str(DATA),dep)
        row=pd.read_parquet(ROOT/f'contracts/{dep}_PILOT_PLAN.parquet').iloc[0].to_dict()
        raw,freq,psd,_=band.raw_for(row)
        full=band.generation.STATE['v3'].preprocess_24s(raw,freq,psd,band_low_hz=20.)
        a=prof.profile.LowBand(full,freq,psd,16)
        b=ApproximantProfile(full,freq,psd,'IMRPhenomD')
        c=ApproximantProfile(full,freq,psd,'IMRPhenomXAS')
        point=np.array([row['parent_predicted_logmc'],.5,0.])
        if not np.array_equal(a.template(point),b.template(point))or a.evaluate(point)!=b.evaluate(point):
            raise RuntimeError('D operator replay failed')
        value=c.evaluate(point)
        if not np.isfinite(value):
            raise RuntimeError('XAS operator invalid')
        rows.append({'deployment':dep,'D_template_delta':0.,'D_power_delta':0.,
                     'XAS_power':value,'old2s_bitexact':True})
    n.write_csv(ROOT/'audit/OPERATOR_REPLAY.csv',rows)
    n.write_json(ROOT/'contracts/UNIT_PASS.json',{'UTC':n.utc(),'passed':True})


def event(job):
    row,feature=job;start=time.perf_counter()
    raw,freq,psd,scale=band.raw_for(row)
    full=band.generation.STATE['v3'].preprocess_24s(raw,freq,psd,band_low_hz=20.)
    model=ApproximantProfile(full,freq,psd,'IMRPhenomXAS')
    out=model.fit(base.initial_peaks(feature))
    out.update(deployment=row['deployment'],source_uid=row['source_uid'],row_index=int(row['row_index']),
        fit_tune_fold=int(row['fit_tune_fold']),image=row['image'],mc_det=float(row['mc_det']),
        profile_error=float(out['logmc']-np.log(row['mc_det'])),approximant='IMRPhenomXAS',
        old2s_bitexact=True,original_SNR_scale=scale,total_wall_seconds=time.perf_counter()-start)
    return out


def run(workers):
    if not (ROOT/'contracts/UNIT_PASS.json').exists():
        raise RuntimeError('Unit pass required')
    for dep in n.DEPS:
        plan=pd.read_parquet(ROOT/f'contracts/{dep}_PILOT_PLAN.parquet')
        x=np.load(DATA/f'features/{dep}/peak.npy',mmap_mode='r')
        jobs=[(row,x[int(row['row_index'])])for row in plan.to_dict('records')
              if not (ROOT/f'events/{dep}_{int(row["row_index"])}.json').exists()]
        with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn'),
            initializer=band.init_worker,initargs=(str(DATA),dep))as pool:
            for count,row in enumerate(pool.map(event,jobs),1):
                n.write_json(ROOT/f'events/{dep}_{row["row_index"]}.json',row)
                if count%16==0 or count==len(jobs):
                    print('APPROXIMANT_PILOT',dep,count,len(jobs),flush=True)
    summarize()


def summarize():
    records=[json.loads(p.read_text())for p in sorted((ROOT/'events').glob('*.json'))]
    for dep in n.DEPS:
        plan=pd.read_parquet(ROOT/f'contracts/{dep}_PILOT_PLAN.parquet')
        for idx in plan.row_index:
            row=json.loads((DATA/f'profile_events/{dep}/{int(idx)}.json').read_text())
            row['approximant']='IMRPhenomD';records.append(row)
    if len(records)!=384:
        raise RuntimeError('Incomplete paired pilot')
    frame=pd.DataFrame([{k:v for k,v in row.items()if not isinstance(v,(dict,list))}for row in records])
    frame['absolute_error']=abs(frame.profile_error);frame['catastrophic']=frame.absolute_error>np.log(1.1)
    source=frame.groupby(['deployment','fit_tune_fold','approximant','source_uid'],as_index=False).agg(
        absolute_error=('absolute_error','mean'),signed_error=('profile_error','mean'),
        catastrophic=('catastrophic','mean'))
    rows=[];comparisons=[];rng=np.random.default_rng(SEED)
    for (dep,fold,model),a in source.groupby(['deployment','fit_tune_fold','approximant']):
        rows.append({'deployment':dep,'fold':fold,'approximant':model,'sources':len(a),
            'MAE':a.absolute_error.mean(),'median_error':a.absolute_error.median(),
            'q90_error':a.absolute_error.quantile(.9),'bias':a.signed_error.mean(),
            'catastrophic':a.catastrophic.mean()})
        if model=='IMRPhenomD':
            continue
        ref=source[(source.deployment==dep)&(source.fit_tune_fold==fold)&(source.approximant=='IMRPhenomD')]
        joined=a.merge(ref,on='source_uid',suffixes=('','_ref'),validate='one_to_one')
        delta=joined.absolute_error.to_numpy()-joined.absolute_error_ref.to_numpy()
        draw=rng.integers(0,len(joined),(5000,len(joined)))
        ci=np.quantile(delta[draw].mean(1),[.025,.975])
        comparisons.append({'deployment':dep,'fold':fold,'delta_MAE':delta.mean(),
            'CI_low':ci[0],'CI_high':ci[1],'delta_q90':a.absolute_error.quantile(.9)-ref.absolute_error.quantile(.9),
            'delta_catastrophic':a.catastrophic.mean()-ref.catastrophic.mean(),
            'PASS':bool(a.absolute_error.mean()<=ref.absolute_error.mean()and
                a.absolute_error.quantile(.9)<=ref.absolute_error.quantile(.9)and a.catastrophic.mean()<=ref.catastrophic.mean())})
    compare=pd.DataFrame(comparisons);tune=compare[compare.fold==1]
    passed=bool(len(tune)==2 and tune.PASS.all()and(tune.delta_MAE<0).any())
    n.write_csv(ROOT/'tables/PROFILE_EVENT_RESULTS.csv',frame)
    n.write_csv(ROOT/'tables/PROFILE_SOURCE_RESULTS.csv',source)
    n.write_csv(ROOT/'tables/PROFILE_SUMMARY.csv',rows)
    n.write_csv(ROOT/'tables/PAIRED_COMPARISONS.csv',compare)
    n.write_json(ROOT/'contracts/PILOT_COMPLETE.json',{'UTC':n.utc(),'status':n.STATUS,
        'passed':passed,'selected_approximant':'IMRPhenomXAS'if passed else None,
        'real_or_test_read':False,'no_rank_change':True,'goal_achieved':False})
    print(pd.DataFrame(rows).to_string(index=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for arg in ('root','data-root','plan-root'):
        parser.add_argument('--'+arg,type=Path,required=True)
    parser.add_argument('--stage',choices=('freeze','unit','run','summarize'),required=True)
    parser.add_argument('--workers',type=int,default=24)
    args=parser.parse_args();ROOT,DATA,PLAN=args.root,args.data_root,args.plan_root
    if args.stage=='run':
        run(args.workers)
    else:
        globals()[args.stage]()
