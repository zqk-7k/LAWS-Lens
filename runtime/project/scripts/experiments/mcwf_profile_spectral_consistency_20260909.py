#!/usr/bin/env python3
"""PyCBC spectral-consistency covariate for empirical mass-error calibration."""
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
from scipy.ndimage import maximum_filter1d
from pycbc.filter import matched_filter
from pycbc.types import TimeSeries, FrequencySeries
from pycbc.vetoes import power_chisq
import pycbc

P=Path('/root/autodl-tmp/gw-catalog')
sys.path.insert(0,str(P/'scripts/experiments'))
import mcwf_inspiral_profile_pilot_20260909 as replay
import mcwf_independent_predictive_calibration_20260909 as pred
f,n,r,h,d= replay.features,pred.n,pred.r,pred.h,pred.d
ROOT=DATA=PREDICTIVE=None
EXPONENTS=(0.,.25,.5,1.)


def statistic(model,point):
    bank=model.template(point)
    filters=[];templates=[];data=[];psds=[]
    for detector in range(2):
        template=TimeSeries(np.pad(bank[detector,0],(0,model.length)),delta_t=1/2048).to_frequencyseries()
        strain=TimeSeries(np.pad(model.raw[detector],(0,model.length)),delta_t=1/2048).to_frequencyseries()
        psd=FrequencySeries(np.ones(len(template)),delta_f=template.delta_f)
        snr=matched_filter(template,strain,psd=psd,low_frequency_cutoff=20.,high_frequency_cutoff=580.)
        templates.append(template);data.append(strain);psds.append(psd);filters.append(np.asarray(snr))
    filters=np.stack(filters)
    lag=model.lags;power=abs(filters[:,lag%model.nfft])**2
    k=int(np.argmax(power[0]+maximum_filter1d(power[1],size=43,mode='constant',cval=-np.inf)))
    near=np.arange(max(k-21,0),min(k+22,len(lag)))
    kl=int(near[np.argmax(power[1,near])]);indices=[int(lag[k]%model.nfft),int(lag[kl]%model.nfft)]
    snr2=np.array([abs(filters[det,indices[det]])**2 for det in range(2)])
    result={'spectral_valid':False,'empirical_flat_PSD':True,'formal_chisquare_probability':False,
            'matched_power_conditioned_units':snr2.tolist(),'lag_samples':[int(lag[k]),int(lag[kl])]}
    if not np.isfinite(snr2).all()or snr2.sum()<=0:
        return result
    for bins in (8,16):
        chisq=[]
        for det in range(2):
            value=power_chisq(templates[det],data[det],bins,psds[det],
                low_frequency_cutoff=20.,high_frequency_cutoff=580.)
            chisq.append(float(value[indices[det]]))
        chisq=np.asarray(chisq)
        if not np.isfinite(chisq).all()or chisq.min() < -1e-8:
            raise RuntimeError('Invalid PyCBC spectral statistic')
        result[f'chisq_{bins}_conditioned_units']=np.maximum(chisq,0).tolist()
        result[f'fraction_{bins}']=float(max(chisq.sum()/snr2.sum(),1e-12))
    result['spectral_valid']=True
    return result


def freeze():
    if ROOT.exists():
        raise RuntimeError('Independent output required')
    for folder in ('contracts','events','calibration','predictions','tables','audit','scripts','logs','manifest','reports','figures'):
        (ROOT/folder).mkdir(parents=True)
    paths=[Path(__file__),Path(replay.__file__),PREDICTIVE/'calibration/INDEPENDENT_PREDICTIVE.json']
    for dep in n.DEPS:
        meta=pd.read_parquet(DATA/f'data/{dep}/event_metadata.parquet')
        active=np.load(PREDICTIVE/f'predictions/{dep}_BOUNDED_development.npz')['profile_active']
        plan=meta[active].copy();plan['deployment']=dep
        plan.to_parquet(ROOT/f'contracts/{dep}_EVENT_PLAN.parquet',index=False)
        paths.extend([DATA/f'data/{dep}/event_metadata.parquet',
            DATA/f'data/{dep}/noise/noise_manifest.csv',
            PREDICTIVE/f'predictions/{dep}_BOUNDED_development.npz'])
    n.write_json(ROOT/'contracts/ANALYSIS_CONTRACT.json',{'UTC':n.utc(),
        'id':'MCWF-NODUP-SPECTRAL-CONSISTENCY-43','status':n.STATUS,
        'data':str(DATA),'parent_predictive':str(PREDICTIVE),
        'hypothesis':'Large projection need not mean consistent spectral shape. Test an Allen-type subband mismatch covariate to explain physical-profile errors.',
        'library':'pycbc.vetoes.power_chisq and pycbc.filter.matched_filter','pycbc_version':pycbc.__version__,
        'input':'Frozen PSD-conditioned20-580Hz16s profile data and SAME fixed template optimum;no refit. Original40Hz2s window must replaybitexact.',
        'not_standard_GW_search_chisq':'Conditioned,cropped and robust-normalized series use a flat reference PSD. Fraction chisq/snr2 is an empirical spectral-shape descriptor,NOT the search pipeline chi-square pvalue,physical networkSNR or GWlikelihood.',
        'scale_invariance':'chisq/snr2 invariant to a common data amplitude factor; verify numerically.',
        'bins':{'main':16,'sensitivity':8},'floor':1e-12,
        'feature':'h_eff=bounded_information_width * spectral_fraction**exponent',
        'exponents':list(EXPONENTS),
        'predictive':'Same normalized truncatedStudent-t calibrated on logh_eff. Each source totalweight1;fitfold0,tunefold1.',
        'grid':{'df':[3,5,10,30],'ridge':[.001,.01,.1,1.]},
        'gate':'Both runs: tune NLL<=R29BOUNDED; MAE and >10percent errors<=NN; source-bootstrap95percent intervals contain.9 globally and active;atleast20fitand20active-tunesources.',
        'e0_control':'Reproduce R29BOUNDED predictions/logloss when spectral feature availability does not remove rows. Record exact support changes.',
        'common_selection':'Among nonzero16binexponents passing bothruns, maximize smaller run NLL gain,then averagegain,then lowerexponent. No selection using real data.',
        'fallback':'Invalid or feature-support OOD retains exact R29 BOUNDED prediction, which already falls back to R10/NN.',
        'no_old_Mc_q':'No encoderregression difference terms','no_total_blend':True,
        'frozen':['encoders','profile optima','time','sky','outerweights','scope','historicalresults'],
        'real_or_test':'Not read in this development/calibration stage.',
        'adaptive_development':True,'full_PE':False,
        'references':['https://arxiv.org/abs/gr-qc/0405045','https://pycbc.org/pycbc/latest/html/pycbc.vetoes.html',
                      'https://arxiv.org/abs/1602.02828'],
        'reference_limit':'These justify testing spectral consistency and waveform mismatch,not the accuracy of this empiricalcovariate or currentNNposterior.'})
    n.write_csv(ROOT/'manifest/INPUT_SHA256.csv',[{'path':str(p),'sha256':n.sha(p),'bytes':p.stat().st_size}for p in paths])
    shutil.copy2(__file__,ROOT/'scripts/profile_spectral_consistency.py')
    n.write_json(ROOT/'contracts/START_FREEZE.json',{'UTC':n.utc(),
        'contract_sha256':n.sha(ROOT/'contracts/ANALYSIS_CONTRACT.json')})


def init_worker(data,dep):
    f.init_worker(data,dep)


def event(job):
    row,profile=job;start=time.perf_counter()
    raw,frequency,psd,_=replay.raw_for(row)
    full=replay.generation.STATE['v3'].preprocess_24s(raw,frequency,psd,band_low_hz=20.)
    model=replay.prof.profile.LowBand(full,frequency,psd,16)
    point=np.array([profile['logmc'],profile['q'],profile['chieff_equal']])
    result=statistic(model,point)
    result.update(deployment=row['deployment'],row_index=int(row['row_index']),source_uid=row['source_uid'],
                  old_peak2s_bitexact=True,wall_seconds=time.perf_counter()-start)
    return result


def unit():
    dep='gwtc3';f.init_worker(str(DATA),dep)
    row=pd.read_parquet(ROOT/f'contracts/{dep}_EVENT_PLAN.parquet').iloc[0].to_dict()
    raw,frequency,psd,_=replay.raw_for(row)
    full=replay.generation.STATE['v3'].preprocess_24s(raw,frequency,psd,band_low_hz=20.)
    profile=json.loads((DATA/f'profile_events/{dep}/{int(row["row_index"])}.json').read_text())
    point=np.array([profile['logmc'],profile['q'],profile['chieff_equal']])
    model=replay.prof.profile.LowBand(full,frequency,psd,16)
    a=statistic(model,point)
    b=statistic(replay.prof.profile.LowBand(3*full,frequency,psd,16),point)
    delta=max(abs(a[f'fraction_{k}']-b[f'fraction_{k}'])for k in (8,16))
    if not a['spectral_valid']or not b['spectral_valid']or delta>1e-10:
        raise RuntimeError('Spectral amplitude-invariance failed')
    n.write_json(ROOT/'contracts/SPECTRAL_UNIT_PASS.json',{'UTC':n.utc(),'passed':True,
        'amplitude_ratio_delta':delta,'library':pycbc.__version__})


def compute(workers):
    if not (ROOT/'contracts/SPECTRAL_UNIT_PASS.json').exists():
        raise RuntimeError('Unit test required')
    for dep in n.DEPS:
        plan=pd.read_parquet(ROOT/f'contracts/{dep}_EVENT_PLAN.parquet')
        jobs=[(row,json.loads((DATA/f'profile_events/{dep}/{int(row["row_index"])}.json').read_text()))
              for row in plan.to_dict('records')if not (ROOT/f'events/{dep}_{int(row["row_index"])}.json').exists()]
        with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn'),
                                 initializer=init_worker,initargs=(str(DATA),dep))as pool:
            for count,result in enumerate(pool.map(event,jobs),1):
                n.write_json(ROOT/f'events/{dep}_{result["row_index"]}.json',result)
                if count%32==0 or count==len(jobs):
                    print('SPECTRAL_COVARIATE',dep,count,len(jobs),flush=True)
    n.write_json(ROOT/'contracts/COVARIATES_COMPLETE.json',{'UTC':n.utc(),'real_or_test_read':False})


def calibrate():
    if not (ROOT/'contracts/COVARIATES_COMPLETE.json').exists():
        raise RuntimeError('Covariates incomplete')
    oldspec=json.loads((PREDICTIVE/'calibration/INDEPENDENT_PREDICTIVE.json').read_text())
    manifest=pd.read_csv(ROOT/'manifest/INPUT_SHA256.csv')
    if any(n.sha(Path(row.path))!=row.sha256 for row in manifest.itertuples()):
        raise RuntimeError('Frozen input changed')
    rows=[];selected={};grids=[];controls=[]
    for dep in n.DEPS:
        meta=pd.read_parquet(DATA/f'data/{dep}/event_metadata.parquet')
        groups=meta.source_uid.to_numpy(str);fold=meta.fold.to_numpy(int);truth=np.log(meta.mc_det.to_numpy(float))
        noise=pd.read_csv(DATA/f'data/{dep}/noise/noise_manifest.csv').set_index('bank_index')
        chunk=meta.noise_bank_index.map(noise.parent_file_gps).to_numpy()
        if set(groups[fold==0])&set(groups[fold==1])or set(chunk[fold==0])&set(chunk[fold==1]):
            raise RuntimeError('Source or parent-noise chunk leakage')
        parent=dict(np.load(PREDICTIVE/f'predictions/{dep}_BOUNDED_development.npz'))
        pred.DATA=DATA
        _,original,_,lp,_,_,_,_=pred.inputs(dep)
        spec0=oldspec[dep+'_BOUNDED']['spec'];active0=parent['active']
        replay_p=parent['p'].copy()
        replay_p[active0]=h.density(parent['center'][active0],parent['width'][active0],spec0)
        replay_delta=float(abs(replay_p-parent['p']).max())
        if replay_delta>1e-12:
            raise RuntimeError('R29 density cannot replay')
        lp[active0]=h.logpdf(truth[active0],parent['center'][active0],parent['width'][active0],spec0)
        tune=fold==1;weight=h.weights(groups[tune]);ref_nll=-float(weight@lp[tune])
        olderr=abs(r.mass_summary(original)[:,0]-truth)
        quality={k:np.full(len(meta),np.nan)for k in (8,16)}
        for path in (ROOT/'events').glob(dep+'_*.json'):
            row=json.loads(path.read_text())
            if row['spectral_valid']:
                for bins in (8,16):
                    quality[bins][row['row_index']]=row[f'fraction_{bins}']
        for bins in (8,16):
            for exponent in EXPONENTS:
                value=parent['width']*quality[bins]**exponent
                eligible=active0&np.isfinite(value)&(value>0);fit=eligible&(fold==0)
                if len(np.unique(groups[fit]))<20:
                    raise RuntimeError('HOLD_INSUFFICIENT_PREDICTIVE_FIT')
                options=[]
                for df in (3,5,10,30):
                    for ridge in (.001,.01,.1,1.):
                        spec=pred.analytic.fit(truth[fit],parent['center'][fit],value[fit],groups[fit],df,ridge)
                        use=eligible&(value>=spec['minimum_h'])&(value<=spec['maximum_h'])
                        pp,parentlog=parent['p'].copy(),lp.copy()
                        pp[use]=h.density(parent['center'][use],value[use],spec)
                        if not np.isfinite(pp).all()or abs(pp.sum(1)-1).max()>1e-10:
                            raise RuntimeError('Invalid normalized predictive mass')
                        parentlog[use]=h.logpdf(truth[use],parent['center'][use],value[use],spec)
                        error=abs(r.mass_summary(pp)[:,0]-truth)
                        cov,low,high,nt=h.coverage_interval(pp,truth,groups,tune)
                        acov,alow,ahigh,na=h.coverage_interval(pp,truth,groups,tune&use)
                        row={'deployment':dep,'bins':bins,'exponent':exponent,'df':df,'ridge':ridge,
                            'fit_sources':len(np.unique(groups[fit])),'active_tune_sources':na,
                            'NLL':-float(weight@parentlog[tune]),'R29_NLL':ref_nll,
                            'MAE':float(weight@error[tune]),'NN_MAE':float(weight@olderr[tune]),
                            'catastrophic10percent':int((error[tune]>np.log(1.1)).sum()),
                            'NN_catastrophic10percent':int((olderr[tune]>np.log(1.1)).sum()),
                            'coverage90':cov,'coverage_low':low,'coverage_high':high,
                            'active_coverage90':acov,'active_coverage_low':alow,'active_coverage_high':ahigh}
                        row['PASS']=bool(row['NLL']<=ref_nll and row['MAE']<=row['NN_MAE']and
                            row['catastrophic10percent']<=row['NN_catastrophic10percent']and na>=20 and
                            low<=.9<=high and alow<=.9<=ahigh)
                        options.append((row,spec,pp,use));grids.append(row)
                passing=[v for v in options if v[0]['PASS']]
                row,spec,pp,use=min(passing or options,key=lambda v:(v[0]['NLL'],v[1]['slope'],-v[0]['ridge']))
                key=f'{dep}_B{bins}_E{exponent:g}'
                selected[key]={'pass':bool(passing),'spec':spec,'selection':row}
                if exponent==0:
                    delta=float(abs(pp-parent['p']).max())
                    nll_delta=float(row['NLL']-ref_nll)
                    controls.append({'deployment':dep,'bins':bins,
                        'direct_R29_replay_delta':replay_delta,'refit_e0_density_delta':delta,
                        'refit_e0_NLL_delta':nll_delta,
                        'active_support_changes':int((use!=active0).sum()),
                        'source_cross_count':0,'noise_parent_cross_count':0})
                    if delta>1e-10 or abs(nll_delta)>1e-10 or not np.array_equal(use,active0):
                        raise RuntimeError('Zero-exponent refit does not reproduce R29')
                np.savez_compressed(ROOT/f'predictions/{key}.npz',p=pp,active=use,
                    profile_active=parent['profile_active'],center=parent['center'],width=parent['width'],
                    quality=quality[bins],effective_width=value,fold=fold,truth=truth)
                rows.append(row)
                print('SPECTRAL_PREDICTIVE',key,row['PASS'],row['NLL']-ref_nll,flush=True)
    choices=[]
    for exponent in EXPONENTS[1:]:
        models=[selected[f'{dep}_B16_E{exponent:g}']for dep in n.DEPS]
        if all(m['pass']for m in models):
            gains=[m['selection']['R29_NLL']-m['selection']['NLL']for m in models]
            choices.append((min(gains),np.mean(gains),-exponent,exponent))
    chosen=max(choices)[-1]if choices else None
    n.write_csv(ROOT/'tables/PREDICTIVE_GRID.csv',grids)
    n.write_csv(ROOT/'audit/R29_ZERO_EXPONENT_REPLAY.csv',controls)
    n.write_csv(ROOT/'tables/PREDICTIVE_SELECTION.csv',rows)
    n.write_json(ROOT/'calibration/SPECTRAL_PREDICTIVE.json',selected)
    n.write_json(ROOT/'contracts/PREDICTIVE_FROZEN.json',{'UTC':n.utc(),'both_runs_pass':bool(choices),
        'selected_bins':16,'selected_exponent':chosen,'sha256':n.sha(ROOT/'calibration/SPECTRAL_PREDICTIVE.json'),
        'real_or_test_scored':False,'goal_achieved':False})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--data-root',type=Path,required=True);parser.add_argument('--predictive-root',type=Path,required=True)
    parser.add_argument('--stage',choices=('freeze','unit','compute','calibrate'),required=True)
    parser.add_argument('--workers',type=int,default=24)
    args=parser.parse_args();ROOT,DATA,PREDICTIVE=args.root,args.data_root,args.predictive_root
    if args.stage=='compute':
        compute(args.workers)
    else:
        globals()[args.stage]()
