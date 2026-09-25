#!/usr/bin/env python3
"""Simulation-only bounded-parameter local-information diagnostic."""
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
from scipy.special import logsumexp
from numpy.polynomial.legendre import leggauss

P=Path('/root/autodl-tmp/gw-catalog')
sys.path.insert(0,str(P/'scripts/experiments'))
import mcwf_profile_expected_information_20260909 as e
n,r,d,cal,h=e.n,e.r,e.d,e.cal,e.h
PARENT=e.PARENT
INFO=P/'results/mcwf_nodup_expected_information_18_20260909T110338Z'
ROOT=None


def bounded_moments(row,profile,count):
    point=np.array([profile['logmc'],profile['q'],profile['chieff_equal']])
    covariance=np.linalg.inv(np.asarray(row['matrices'][-1]))
    q=point[1]
    jac=np.diag([1.,(1-q)/(1+q)**3,1.])
    covariance=jac@covariance@jac.T
    eta=q/(1+q)**2
    v=covariance[1:,1:]
    gain=np.linalg.solve(v,covariance[1:,0])
    residual=covariance[0,0]-covariance[0,1:]@gain
    if residual<=0:
        raise RuntimeError('Nonpositive conditional information variance')
    x,w=leggauss(count)
    qs=.25+(x+1)*.75/2
    chis=x*.8
    qgrid,cgrid=np.meshgrid(qs,chis,indexing='ij')
    weights=np.outer(w,w)
    delta=np.stack([qgrid/(1+qgrid)**2-eta,cgrid-point[2]],axis=-1).reshape(-1,2)
    logits=-.5*np.einsum('ni,ij,nj->n',delta,np.linalg.inv(v),delta)+np.log(weights.ravel())
    weights=np.exp(logits-logsumexp(logits))
    conditional_mean=point[0]+delta@gain
    mean=float(weights@conditional_mean)
    variance=float(residual+weights@((conditional_mean-mean)**2))
    return mean,float(np.sqrt(variance)),float(1/np.sum(weights**2))


def freeze():
    if ROOT.exists():
        raise RuntimeError('Independent output required')
    for folder in ('contracts','calibration','predictions','events','tables','audit','scripts','logs','manifest','reports'):
        (ROOT/folder).mkdir(parents=True)
    n.write_json(ROOT/'contracts/ANALYSIS_CONTRACT.json',{'id':'MCWF-NODUP-PRIOR-CONSTRAINED-INFORMATION-21',
        'UTC':n.utc(),'status':n.STATUS,'goal_achieved':False,'adaptive_development':True,'same_both_runs':True,
        'motivation':'Near equal mass, an unconstrained local Gaussian can spread over unphysical nuisance parameters. Test bounded nuisance integration before using the local width as a predictive covariate.',
        'inputs':'Only existing source/noise-disjoint development profiles and expected derivative information;no new waveform fit,realPE,officialrank or test input.',
        'coordinates':'Transform local covariance(logMc,q,chi) into(logMc,eta=q/(1+q)^2,chi) with Jacobian;then use local Gaussian in eta/chi.',
        'reference_prior':'Uniform q in[.25,1],equal chi in[-.8,.8],same bounds as point optimizer. This is a reference approximation,NOTthe publishedPE or astrophysical population prior.',
        'quadrature':'Gauss-Legendre64and128 inq/chi;integrate conditionalGaussian logMc moments. Width convergence<=5percent,mean difference<=.001logmass,128-grid effective nodes>=10;otherwise fallback.',
        'boundary_safeguard':'Require mean+-5width containedwithin log(5)..log(200);otherwise no untruncated-mass approximation.',
        'prediction':'Use reference-prior local mean andwidth ONLYas inputs to simulation-error Student-t calibration;do not label the localGaussian a PE posterior.',
        'fit_selection':'Same fitfold0/ridge/df andsourceweightedproperloglossfold1 as18;mustpass bothrunsNLLvsR10,MAE/catvsNN,andglobal/active90percentcoverage checks.',
        'limitations':'Finite local expansion,aligned spin,reference priors andsmall sample size;not fullPE. Failure retained;no automaticrealranking.',
        'frozen':['encoder','time','sky','outerweights','historicalrankings','scope','paper'],
        'references':['https://arxiv.org/abs/gr-qc/0703086','https://arxiv.org/abs/gr-qc/9402014']})
    n.write_csv(ROOT/'manifest/INPUT_SHA256.csv',pd.read_csv(INFO/'manifest/INPUT_SHA256.csv'))
    shutil.copy2(__file__,ROOT/'scripts/prior_constrained_information.py')
    n.write_json(ROOT/'contracts/START_FREEZE.json',{'UTC':n.utc(),'script_sha256':n.sha(Path(__file__)),
        'contract_sha256':n.sha(ROOT/'contracts/ANALYSIS_CONTRACT.json')})


def calibrate():
    selected,grids,outputs=[],[],[]
    specs={}
    previous=json.loads((PARENT/'calibration/PROFILE_PREDICTIVE.json').read_text())
    for dep in n.DEPS:
        meta=pd.read_parquet(n.t.PREVIOUS/f'expanded_data/{dep}/validation/event_metadata.parquet')
        meta['deployment']=dep
        fold,truth=r.source_fold(meta),np.log(meta.mc_det.to_numpy())
        groups=meta.source_uid.to_numpy(str)
        original=np.mean(cal.parent_predictions(dep),axis=0)
        parent_data=dict(np.load(PARENT/f'predictions/{dep}_development_ENSEMBLE.npz'))
        parent=parent_data['p']
        center=parent_data['profile_centers'].copy()
        widths=np.full(len(meta),np.nan)
        for path in sorted((INFO/'events').glob(dep+'_*.json')):
            row=json.loads(path.read_text())
            if not row['information_valid']:
                continue
            idx=int(row['row_index'])
            profile=json.loads((e.PROFILE/f'events/{dep}_{idx}.json').read_text())
            m64,s64,ess64=bounded_moments(row,profile,64)
            m128,s128,ess128=bounded_moments(row,profile,128)
            relative=abs(s64/s128-1)
            valid=bool(np.isfinite([m128,s128]).all() and s128>0 and relative<=.05 and abs(m64-m128)<=.001 and ess128>=10 and
                       m128-5*s128>d.EDGES[0] and m128+5*s128<d.EDGES[-1])
            record={'deployment':dep,'row_index':idx,'valid':valid,'mean64':m64,'mean128':m128,
                'width64':s64,'width128':s128,'width_relative_difference':relative,'effective_nodes64':ess64,'effective_nodes128':ess128,
                'mean_shift_from_profile':m128-profile['logmc'],'reference_not_physicalPE':True}
            n.write_json(ROOT/f'events/{dep}_{idx}.json',record)
            if valid:
                center[idx],widths[idx]=m128,s128
        eligible=parent_data['active']&np.isfinite(widths)&(widths>0)
        fit=eligible&(fold==0)
        if len(np.unique(groups[fit]))<20:
            specs[dep]={'pass':False,'reason':'Less than20independentfit sources'}
            continue
        bins=np.searchsorted(d.EDGES,truth,side='right').clip(1,512)-1
        parent_log=np.log(parent[np.arange(len(parent)),bins].clip(1e-300))-np.log(np.diff(d.EDGES)[bins])
        pa=parent_data['active']
        parent_log[pa]=cal.normalized_logpdf(truth[pa],parent_data['profile_centers'][pa],previous[dep]['spec'])
        tune=fold==1
        wt=h.weights(groups[tune])
        ref_nll=-float(wt@parent_log[tune])
        olderr=abs(r.mass_summary(original)[:,0]-truth)
        options=[]
        for df in (3,5,10,30):
            for ridge in (.001,.01,.1,1.):
                spec=h.fit(truth[fit],center[fit],widths[fit],groups[fit],df,ridge)
                spec['covariate']='bounded_reference_prior_information_width_NOT_PE'
                active=eligible&(widths>=spec['minimum_h'])&(widths<=spec['maximum_h'])
                pp,lp=parent.copy(),parent_log.copy()
                pp[active]=h.density(center[active],widths[active],spec)
                lp[active]=h.logpdf(truth[active],center[active],widths[active],spec)
                error=abs(r.mass_summary(pp)[:,0]-truth)
                cov,lo,hi,nt=h.coverage_interval(pp,truth,groups,tune)
                acov,alo,ahi,na=h.coverage_interval(pp,truth,groups,tune&active)
                row={'deployment':dep,'df':df,'ridge':ridge,'slope':spec['slope'],'fit_sources':spec['fit_sources'],
                    'tune_sources':nt,'active_tune_sources':na,'NLL':-float(wt@lp[tune]),'R10_NLL':ref_nll,
                    'MAE':float(wt@error[tune]),'NN_MAE':float(wt@olderr[tune]),
                    'catastrophic10percent':int((error[tune]>np.log(1.1)).sum()),
                    'NN_catastrophic10percent':int((olderr[tune]>np.log(1.1)).sum()),
                    'coverage90':cov,'coverage_low':lo,'coverage_high':hi,
                    'active_coverage90':acov,'active_coverage_low':alo,'active_coverage_high':ahi}
                row['PASS']=bool(row['NLL']<=ref_nll and row['MAE']<=row['NN_MAE'] and
                    row['catastrophic10percent']<=row['NN_catastrophic10percent'] and na>=20 and lo<=.9<=hi and alo<=.9<=ahi)
                options.append((row,spec,pp,active));grids.append(row)
        passing=[x for x in options if x[0]['PASS']]
        row,spec,pp,active=min(passing or options,key=lambda x:(x[0]['NLL'],x[0]['slope'],-x[0]['ridge']))
        specs[dep]={'pass':bool(passing),'selection':row,'spec':spec}
        np.savez_compressed(ROOT/f'predictions/{dep}_development.npz',p=pp,active=active,
            reference_center=center,information_width=widths,profile_active=pa)
        outputs.append(row)
        print('BOUNDED_INFORMATION_PREDICTIVE',dep,specs[dep],flush=True)
    n.write_csv(ROOT/'tables/PREDICTIVE_GRID.csv',grids)
    n.write_csv(ROOT/'tables/PREDICTIVE_SELECTION.csv',outputs)
    n.write_json(ROOT/'calibration/PROFILE_BOUNDED_INFORMATION.json',specs)
    n.write_json(ROOT/'contracts/PREDICTIVE_FROZEN.json',{'UTC':n.utc(),'both_runs_pass':all(s['pass'] for s in specs.values()),
        'sha256':n.sha(ROOT/'calibration/PROFILE_BOUNDED_INFORMATION.json'),'no_real_test_scored':True})


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--stage',choices=('freeze','calibrate'),required=True)
    args=parser.parse_args();ROOT=args.root
    globals()[args.stage]()
