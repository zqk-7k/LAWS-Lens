#!/usr/bin/env python3
"""Analytic-gradient implementation repair; predictive criteria unchanged."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse
from pathlib import Path
import shutil
import sys
import numpy as np
from scipy.optimize import minimize
from scipy.optimize._numdiff import approx_derivative
from scipy.stats import t

P=Path('/root/autodl-tmp/gw-catalog')
sys.path.insert(0,str(P/'scripts/experiments'))
import mcwf_prior_constrained_information_20260909 as b
h,n,d=b.h,b.n,b.d


def objective(theta,truth,centers,widths,weight,df,ridge,mu):
    predictor=np.log(widths)-mu
    scale=np.exp(theta[1]+theta[2]*predictor)
    loc=centers+theta[0]
    z=(truth-loc)/scale
    lo=(d.EDGES[0]-loc)/scale
    hi=(d.EDGES[-1]-loc)/scale
    norm=(t.cdf(hi,df)-t.cdf(lo,df)).clip(1e-300)
    logp=t.logpdf(z,df)-np.log(scale)-np.log(norm)
    dlo=(df+1)*z/((df+z*z)*scale)-(t.pdf(lo,df)-t.pdf(hi,df))/(scale*norm)
    ds=(df+1)*z*z/(df+z*z)-1-(lo*t.pdf(lo,df)-hi*t.pdf(hi,df))/norm
    loss=-float(weight@logp)+.5*ridge*theta[2]**2
    grad=np.array([-weight@dlo,-weight@ds,-weight@(ds*predictor)+ridge*theta[2]])
    return loss,grad


def fit(truth,centers,widths,groups,df,ridge):
    weight=h.weights(groups)
    mu=float(weight@np.log(widths))
    start=np.array([0.,np.log(.005),.5])
    fun=lambda theta:objective(theta,truth,centers,widths,weight,df,ridge,mu)
    numeric=approx_derivative(lambda theta:np.array([fun(theta)[0]]),start,method='3-point').ravel()
    error=float(np.max(abs(numeric-fun(start)[1])/np.maximum(1.,abs(numeric))))
    if error>1e-4:
        raise RuntimeError('Analytic gradient validation failed:'+str(error))
    result=minimize(fun,start,jac=True,method='L-BFGS-B',
        bounds=[(-.02,.02),(np.log(.0002),np.log(.2)),(0.,2.)],
        options={'maxiter':2000,'ftol':1e-12,'gtol':1e-7,'maxls':50})
    if not result.success:
        raise RuntimeError('Analytic gradient fit failed:'+str(result.message))
    return {'location':float(result.x[0]),'intercept':float(result.x[1]),'slope':float(result.x[2]),
        'logh_center':mu,'df':df,'ridge':ridge,'minimum_h':float(widths.min()),'maximum_h':float(widths.max()),
        'fit_sources':len(np.unique(groups)),'fit_events':len(widths),'fit_objective':float(result.fun),
        'hessian_is_PE_covariance':False,'analytic_gradient_relative_error':error,'optimizer_success':True}


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--stage',choices=('freeze','calibrate'),required=True)
    args=parser.parse_args();b.ROOT=args.root
    h.fit=fit
    if args.stage=='freeze':
        b.freeze()
        n.write_json(b.ROOT/'contracts/OPTIMIZER_IMPLEMENTATION_ADDENDUM.json',{
            'UTC':n.utc(),'prior_attempt':str(P/'results/mcwf_nodup_bounded_information_21_20260909T111754Z'),
            'prior_failure':'Finite-difference L-BFGS-B ABNORMAL line search on an O4 fitgrid point;retained original failure,not a scientific PASS/FAIL.',
            'repair':'Same normalizedStudent-tobjective andbounds;analyticgradient checked by3pointfinite differences;maxls50.',
            'all_grid_points_refitted':True,'selection_rules_changed':False,'thresholds_changed':False,
            'real_or_test_used':False,'runtime_sha256':n.sha(Path(__file__))})
        shutil.copy2(__file__,b.ROOT/'scripts/bounded_information_runtime.py')
    else:
        b.calibrate()
