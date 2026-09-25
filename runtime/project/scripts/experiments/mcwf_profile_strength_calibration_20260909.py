#!/usr/bin/env python3
"""Simulation-calibrated mass uncertainty conditioned on waveform fit strength."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse
import json
from pathlib import Path
import shutil
import sys
import numpy as np
import pandas as pd
from scipy.ndimage import maximum_filter1d

P=Path('/root/autodl-tmp/gw-catalog')
sys.path.insert(0,str(P/'scripts/experiments'))
import mcwf_profile_expected_information_20260909 as info
n,r,d,cal,h=info.n,info.r,info.d,info.cal,info.h
ROOT=None


def strength(model,point,raw24):
    bank=model.template(point)
    kernels=np.fft.rfft(bank,n=model.nfft,axis=-1)
    values=np.fft.irfft(model.data[:,None]*kernels.conj(),n=model.nfft,axis=-1)
    lag=model.lags
    power=(values[...,lag%model.nfft]**2).sum(1)
    first=np.asarray(raw24)[:,:8*2048]
    variance=(np.median(abs(first-np.median(first,axis=-1,keepdims=True)),axis=-1)/.6744897501960817)**2
    if not np.isfinite(variance).all() or (variance<=0).any():
        return {'information_valid':False,'reason':'invalid_premerger_robust_variance'}
    # Match the fitted lag convention; variance only scales the subsequent covariate.
    k=int(np.argmax(power[0]+maximum_filter1d(power[1],size=43,mode='constant',cval=-np.inf)))
    near=np.arange(max(k-21,0),min(k+22,len(lag)))
    kl=int(near[np.argmax(power[1,near])])
    coefficients=np.stack([values[0,:,lag[k]%model.nfft],values[1,:,lag[kl]%model.nfft]])
    rho2=float(np.sum(coefficients**2/variance[:,None]))
    valid=np.isfinite(rho2) and rho2>0
    return {'information_valid':bool(valid),'information_logmc_width':float(1/np.sqrt(rho2)) if valid else None,
        'waveform_projection_strength':float(np.sqrt(rho2)) if valid else None,
        'premerger_robust_variance':variance.tolist(),'detector_lags':[int(lag[k]),int(lag[kl])],
        'not_catalog_or_optimal_network_SNR':True,'not_a_Fisher_width':True,
        'premerger_context_may_contain_inspiral':True}


info.information=strength


def freeze():
    if ROOT.exists():
        raise RuntimeError('Independent directory required')
    for folder in ('contracts','events','calibration','predictions','tables','audit','reports','scripts','logs','manifest'):
        (ROOT/folder).mkdir(parents=True)
    n.write_json(ROOT/'contracts/ANALYSIS_CONTRACT.json',{'UTC':n.utc(),'id':'MCWF-NODUP-WAVEFORM-STRENGTH-27',
        'goal_achieved':False,'status':n.STATUS,'adaptive_development':True,'same_both_runs':True,
        'motivation':'Parameterprecisionusuallydependsonsignalstrength;testanempiricalstrengthcovariateinstead ofcommonwidthforallqualityactiveevents.',
        'observable':'sqrt(sum_detector sum_quadrature coefficient^2 /premergerMADvariance). AllvaluesfromwaveformandlocalPSD;noPE,publicSNRortruthasinput.',
        'variance_window':'First8sofconditioned24sPREMERGERcontext,NOTguaranteedsignal-free. Maycontaininspiral. Its effectisempiricallyvalidated,notusedasexactnoisePSD.',
        'predictive':'FrozenR10profilecenter andquality;Student-tlogscaleaffineinlog(1/strength),slope[0,2]. Fitfold0sourceweights;df3,5,10,30 andridge.001,.01,.1,1selectedbyfold1properNLL.',
        'minimum_support':20,'gates':'SameR18NLL,MAE,10percentpoint-error,globalANDactive source-bootstrap90percentcoverage criteria;bothrunsmustpass.',
        'fallback':'UnchangedR10densityoutsidefitstrengthsupportorinvalidobservable.',
        'old_internal_field_names':'information_logmc_widthstoresinverseprojectionstrength,notFisherwidth. PROFILE_EXPECTED_INFORMATION.jsonfilenameiscompatibilityonly.',
        'frozen':['allneuralmodels','allprofilecenters','time','sky','outerweights','scope','historicaloutputs'],
        'forbidden':['oldMc_q_heads','totalblend','PE_or_officialscoreinput'],
        'references':['https://arxiv.org/abs/gr-qc/9402014','https://arxiv.org/abs/gr-qc/0703086'],
        'citation_limit':'Motivatesstrengthdependenceandlimitations;doesnotproveour16sorMADcovariateoptimalorPE-equivalent.'})
    previous=P/'results/mcwf_nodup_expected_information_18_20260909T110338Z'
    for dep in n.DEPS:
        shutil.copy2(previous/f'contracts/{dep}_EVENT_PLAN.parquet',ROOT/f'contracts/{dep}_EVENT_PLAN.parquet')
    n.write_csv(ROOT/'manifest/INPUT_SHA256.csv',pd.read_csv(previous/'manifest/INPUT_SHA256.csv'))
    shutil.copy2(__file__,ROOT/'scripts/profile_strength_calibration.py')
    n.write_json(ROOT/'contracts/START_FREEZE.json',{'UTC':n.utc(),'script_sha256':n.sha(Path(__file__)),
        'contract_sha256':n.sha(ROOT/'contracts/ANALYSIS_CONTRACT.json')})


def calibrate():
    chosen,grid,outputs={},[],[]
    oldspec=json.loads((info.PARENT/'calibration/PROFILE_PREDICTIVE.json').read_text())
    for dep in n.DEPS:
        meta=pd.read_parquet(n.t.PREVIOUS/f'expanded_data/{dep}/validation/event_metadata.parquet')
        meta['deployment']=dep
        fold,truth=r.source_fold(meta),np.log(meta.mc_det.to_numpy())
        groups=meta.source_uid.to_numpy(str)
        original=np.mean(cal.parent_predictions(dep),axis=0)
        pa=dict(np.load(info.PARENT/f'predictions/{dep}_development_ENSEMBLE.npz'))
        parent,centers=pa['p'],pa['profile_centers']
        covariate=np.full(len(meta),np.nan)
        for path in (ROOT/'events').glob(dep+'_*.json'):
            row=json.loads(path.read_text())
            if row['information_valid']:
                covariate[int(row['row_index'])]=row['information_logmc_width']
        eligible=pa['active']&np.isfinite(covariate)&(covariate>0)
        fit=eligible&(fold==0)
        if len(np.unique(groups[fit]))<20:
            chosen[dep]={'pass':False,'reason':'FIT_SUPPORT'}
            continue
        bins=np.searchsorted(d.EDGES,truth,side='right').clip(1,512)-1
        parent_log=np.log(parent[np.arange(len(parent)),bins].clip(1e-300))-np.log(np.diff(d.EDGES)[bins])
        act=pa['active']
        parent_log[act]=cal.normalized_logpdf(truth[act],centers[act],oldspec[dep]['spec'])
        tune=fold==1
        weights=h.weights(groups[tune])
        reference=-float(weights@parent_log[tune])
        olderror=abs(r.mass_summary(original)[:,0]-truth)
        options=[]
        for df in (3,5,10,30):
            for ridge in (.001,.01,.1,1.):
                spec=h.fit(truth[fit],centers[fit],covariate[fit],groups[fit],df,ridge)
                spec['covariate']='inverse_waveform_projection_strength_NOT_optimalSNR_or_Fisher'
                active=eligible&(covariate>=spec['minimum_h'])&(covariate<=spec['maximum_h'])
                pp,lp=parent.copy(),parent_log.copy()
                pp[active]=h.density(centers[active],covariate[active],spec)
                lp[active]=h.logpdf(truth[active],centers[active],covariate[active],spec)
                err=abs(r.mass_summary(pp)[:,0]-truth)
                cv,lo,hi,nt=h.coverage_interval(pp,truth,groups,tune)
                ac,al,ah,na=h.coverage_interval(pp,truth,groups,tune&active)
                row={'deployment':dep,'df':df,'ridge':ridge,'slope':spec['slope'],'fit_sources':spec['fit_sources'],
                    'tune_sources':nt,'active_tune_sources':na,'NLL':-float(weights@lp[tune]),'R10_NLL':reference,
                    'MAE':float(weights@err[tune]),'NN_MAE':float(weights@olderror[tune]),
                    'catastrophic10percent':int((err[tune]>np.log(1.1)).sum()),
                    'NN_catastrophic10percent':int((olderror[tune]>np.log(1.1)).sum()),
                    'coverage90':cv,'coverage_low':lo,'coverage_high':hi,
                    'active_coverage90':ac,'active_coverage_low':al,'active_coverage_high':ah}
                row['PASS']=bool(row['NLL']<=reference and row['MAE']<=row['NN_MAE'] and
                    row['catastrophic10percent']<=row['NN_catastrophic10percent'] and na>=20 and lo<=.9<=hi and al<=.9<=ah)
                options.append((row,spec,pp,active));grid.append(row)
        passed=[v for v in options if v[0]['PASS']]
        row,spec,pp,active=min(passed or options,key=lambda v:(v[0]['NLL'],v[0]['slope'],-v[0]['ridge']))
        chosen[dep]={'pass':bool(passed),'selection':row,'spec':spec}
        np.savez_compressed(ROOT/f'predictions/{dep}_development.npz',p=pp,parent_p=parent,active=active,
            profile_active=pa['active'],information_width=covariate,profile_centers=centers,fold=fold,truth=truth)
        outputs.append(row)
        print('STRENGTH_PREDICTIVE',dep,chosen[dep],flush=True)
    n.write_csv(ROOT/'tables/PREDICTIVE_GRID.csv',grid)
    n.write_csv(ROOT/'tables/PREDICTIVE_SELECTION.csv',outputs)
    path=ROOT/'calibration/PROFILE_EXPECTED_INFORMATION.json'
    n.write_json(path,chosen)
    n.write_json(ROOT/'contracts/PREDICTIVE_FROZEN.json',{'UTC':n.utc(),'both_runs_pass':all(v['pass'] for v in chosen.values()),
        'sha256':n.sha(path),'no_real_test_scored':True,'not_expected_Fisher_information':True})


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--stage',choices=('freeze','compute','calibrate'),required=True)
    p.add_argument('--workers',type=int,default=20)
    args=p.parse_args()
    ROOT=info.ROOT=args.root
    if args.stage=='compute':
        info.compute(args.workers)
    else:
        globals()[args.stage]()
