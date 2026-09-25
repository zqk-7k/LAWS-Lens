"""Finite-endpoint GPD fits using background-only maximum product spacings.

No candidate-dependent endpoint constraints or threshold selection.
All ranges are sensitivity diagnostics, not confidence intervals.
"""
from pathlib import Path
import hashlib,json,time,platform,sys
import numpy as np
import pandas as pd
import scipy
from scipy.optimize import minimize
from scipy.stats import genpareto

BASE=Path(__file__).resolve().parent
ROOT=next((BASE.parent/'extracted').iterdir())
OUT=BASE/'tables';OUT.mkdir(exist_ok=True)
KEY='final_score_POSITIVE';QS=[.9,.925,.95,.975,.98]
inputs=[];fits=[];pred=[];deletions=[];checks=[];moments=[]

def track(p):
    inputs.append(dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))

def mps(x,q):
    x=np.asarray(x,float);u=float(np.quantile(x,q));y=np.sort(x[x>u]-u)
    assert len(y)>=30 and len(np.unique(y))==len(y)
    ymax=y[-1];ys=y/ymax
    def calc(theta,return_par=False):
        # Endpoint excess E is larger than max(y), but not tied to any candidate.
        E=1+np.exp(theta[0]);a=np.exp(theta[1])
        ls=a*np.log1p(-ys/E)
        # D0=F(y1), Di=F(yi+1)-F(yi), Dlast=1-F(ymax).
        lsp=np.r_[np.log(-np.expm1(ls[0])),ls[:-1]+np.log(-np.expm1(np.diff(ls))),ls[-1]]
        if return_par:return E*ymax,a,lsp
        return float(-lsp.sum()) if np.isfinite(lsp).all() else 1e100
    opts=[]
    for gap,a in [(.01,1),(.1,2),(.5,4),(2,8),(10,40)]:
        opt=minimize(calc,np.log([gap,a]),method='Nelder-Mead',
                     bounds=[(-20,16),(-8,20)],options={'maxiter':1800,'xatol':1e-8,'fatol':1e-8})
        if np.isfinite(opt.fun):opts.append(opt)
    best=min(opts,key=lambda v:v.fun)
    E,a,lsp=calc(best.x,True);xi=-1/a;scale=E/a
    # Independent evaluation of the objective with SciPy's spacing implementation.
    scipy_obj=float(genpareto._penalized_nlpsf((xi,0,scale),y))
    assert abs(scipy_obj-best.fun)<1e-5,(scipy_obj,best.fun)
    cdf=genpareto.cdf(y,xi,scale=scale);n=len(y)
    ks=max(np.max(np.arange(1,n+1)/n-cdf),np.max(cdf-np.arange(n)/n))
    bound_hit=any(abs(v-b)<.1 for v,bs in zip(best.x,[(-20,16),(-8,20)]) for b in bs)
    return dict(method='MPS',q=q,u=u,n=len(x),m=n,tail_weight=n/len(x),xi=xi,scale=scale,
                endpoint=u+E,descriptive_KS=ks,negative_log_spacing=float(best.fun),
                optimizer_success=bool(best.success),boundary_hit=bound_hit,
                multistart_objective_spread=float(max(z.fun for z in opts)-best.fun),background_max=float(x.max()))

def predict(f,s):
    s=np.asarray(s,float);assert np.all(s>=f['u'])
    p=f['tail_weight']*genpareto.sf(s-f['u'],f['xi'],scale=f['scale'])
    return p,s>=f['endpoint']

start=time.time()
for run in ['O3','O4a','O4b']:
    bp=ROOT/f'background/{run}/mean_S_calibration_pairs.csv';track(bp)
    bg=pd.read_csv(bp);x=bg[KEY].to_numpy()
    # Fit all declared thresholds BEFORE opening candidate/test tables in this run.
    fitted=[]
    for q in QS:
        f=mps(x,q);fitted.append(f);fits.append(dict(run=run,**f))
        u=np.quantile(x,q);y=np.sort(x[x>u]-u);mu=y.mean()
        l2=2*np.mean(y*np.arange(len(y))/(len(y)-1))-mu
        for method,xi in [('PWM',2-mu/l2),('MOM',.5*(1-mu*mu/y.var()))]:
            sc=mu*(1-xi);endpoint=u-sc/xi if xi<0 else np.inf
            moments.append(dict(run=run,q=q,method=method,xi=xi,scale=sc,endpoint=endpoint,
                background_max=x.max(),contains_all_fitting_data=bool(endpoint>x.max())))
    rp=ROOT/f'tables/{run}/real_GWTC_Top20_FPP.csv';tp=ROOT/f'tables/{run}/injection_mean_S_FPP.parquet'
    track(rp);track(tp);real=pd.read_csv(rp).sort_values('consensus_rank');test=pd.read_parquet(tp)
    single=test[test.event_i.str.contains('unlensed') & test.event_j.str.contains('unlensed')]
    assert len(single)==4005 and not single.is_true_pair.any()
    for f in fitted:
        ps,outside=predict(f,real[KEY])
        for (_,r),p,o in zip(real.iterrows(),ps,outside):
            pred.append(dict(run=run,rank=int(r.consensus_rank),pair=r.pair_key,S=r[KEY],q=f['q'],
                endpoint=f['endpoint'],xi=f['xi'],scale=f['scale'],estimated_tail_probability=float(p),
                outside_fitted_support=bool(o),empirical_count=int(r.background_exceedances),
                boundary_hit=f['boundary_hit']))
        for target in [.01,.005,.001,.0005,.00025]:
            threshold=f['u']+genpareto.isf(target/f['tail_weight'],f['xi'],scale=f['scale'])
            k=int((single[KEY]>=threshold).sum())
            checks.append(dict(run=run,q=f['q'],target_probability=target,threshold_S=threshold,
                n_pairs=4005,observed_count=k,expected_count=4005*target,
                test_pairs_outside_endpoint=int((single[KEY]>=f['endpoint']).sum())))
    for dq in ([.95,.975,.98] if '--extended' in sys.argv else [.95]):
        for kind,ci,cj in [('source','source_i','source_j'),('noise_parent','noise_i','noise_j')]:
            for key in sorted(set(bg[ci])|set(bg[cj])):
                sub=bg[(bg[ci]!=key)&(bg[cj]!=key)]
                f=mps(sub[KEY],dq);ps,outs=predict(f,real[KEY])
                for (_,r),p,o in zip(real.iterrows(),ps,outs):
                    deletions.append(dict(run=run,kind=kind,deleted=key,q=dq,rank=int(r.consensus_rank),
                        endpoint=f['endpoint'],p=float(p),outside_fitted_support=bool(o),
                        boundary_hit=f['boundary_hit']))
        f=mps(np.delete(x,np.argmax(x)),dq);ps,outs=predict(f,real[KEY])
        for (_,r),p,o in zip(real.iterrows(),ps,outs):
            deletions.append(dict(run=run,kind='maximum_pair',deleted=bg.iloc[np.argmax(x)].pair_key,q=dq,
                rank=int(r.consensus_rank),endpoint=f['endpoint'],p=float(p),outside_fitted_support=bool(o),boundary_hit=f['boundary_hit']))
    print(run,'done',round(time.time()-start,1),flush=True)

for name,rows in [('finite_mps_fits',fits),('finite_mps_Top20_all_thresholds',pred),('finite_mps_deletions',deletions),('finite_mps_test_checks',checks),('moment_comparators',moments)]:
    pd.DataFrame(rows).to_csv(OUT/(name+'.csv'),index=False)
for inp in inputs:assert hashlib.sha256(Path(inp['path']).read_bytes()).hexdigest()==inp['sha256']
audit=dict(status='EXPLORATORY_NOT_ADOPTED',seconds=time.time()-start,inputs=inputs,python=platform.python_version(),scipy=scipy.__version__,numpy=np.__version__,
           thresholds=QS,central_threshold=.95,candidate_endpoint_constraint=False,new_simulations=False,
           paper_changed=False,independent_pair_assumption_for_uncertainty=False,
           fits=len(fits),deletion_fits=len(deletions)//20,
           objective_crosscheck='SciPy _penalized_nlpsf for every fit, tolerance 1e-5',
           optimization_bound_hits=int(sum(f['boundary_hit'] for f in fits)))
(BASE/'AUDIT.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
print(pd.DataFrame(fits)[['run','q','xi','endpoint','boundary_hit']].to_string(index=False))
print(pd.DataFrame(pred).query('run=="O4b" and rank==1').to_string(index=False))
