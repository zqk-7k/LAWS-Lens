#!/usr/bin/env python3
"""Read-only source-level prediction-error dependence diagnostic."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse
from pathlib import Path
import shutil
import sys
import numpy as np
import pandas as pd
from scipy.stats import pearsonr,spearmanr

P=Path('/root/autodl-tmp/gw-catalog')
sys.path.insert(0,str(P/'scripts/experiments'))
import mcwf_independent_predictive_scoring_20260909 as engine
n,r=engine.n,engine.r


def bootstrap(x,y,rng,draws=5000):
    ix=rng.integers(0,len(x),(draws,len(x)))
    a,b=x[ix],y[ix]
    a-=a.mean(1,keepdims=True)
    b-=b.mean(1,keepdims=True)
    va,vb=(a*a).mean(1),(b*b).mean(1)
    cov=(a*b).mean(1)
    rho=cov/np.sqrt(va*vb).clip(1e-30)
    ratio=(va+vb-2*cov)/(va+vb).clip(1e-30)
    return np.quantile(rho,[.025,.975]),np.quantile(ratio,[.025,.975])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--predictive-root',type=Path,required=True)
    parser.add_argument('--data-root',type=Path,required=True)
    args=parser.parse_args()
    if args.root.exists():
        raise RuntimeError('New independent diagnostic directory required')
    for name in ('contracts','tables','reports','manifest','scripts'):
        (args.root/name).mkdir(parents=True)
    rows,event_rows,manifest=[],[],[]
    rng=np.random.default_rng(2026090937)
    for dep in n.DEPS:
        path=args.predictive_root/f'predictions/{dep}_BOUNDED_development.npz'
        meta_path=args.data_root/f'data/{dep}/event_metadata.parquet'
        a=dict(np.load(path));meta=pd.read_parquet(meta_path)
        neural_path=args.data_root/f'predictions/{dep}/ensemble_mass.npy'
        neural=np.load(neural_path)
        for p in (path,meta_path,neural_path):
            manifest.append({'path':str(p),'sha256':n.sha(p),'bytes':p.stat().st_size})
        for kind in ('NN','R10HYBRID','R10PROFILE','BOUNDED'):
            if kind=='R10PROFILE':
                center=a['profile_center']
            else:
                distribution=neural if kind=='NN' else a['parent_p' if kind=='R10HYBRID' else 'p']
                center=r.mass_summary(distribution)[:,0]
            active=(np.ones(len(center),bool)if kind in ('NN','R10HYBRID') else
                    a['profile_active' if kind=='R10PROFILE' else 'active'])
            for fold in (0,1):
                i=np.arange(0,len(center),2)
                i=i[(a['fold'][i]==fold)&active[i]&active[i+1]];j=i+1
                if not np.array_equal(meta.source_uid.to_numpy()[i],meta.source_uid.to_numpy()[j]):
                    raise RuntimeError('Companion grouping mismatch')
                x,y=center[i]-a['truth'][i],center[j]-a['truth'][j]
                ci,vr=bootstrap(x,y,rng)
                rows.append({'deployment':dep,'kind':kind,'fold':fold,'sources':len(i),
                    'pearson':float(pearsonr(x,y).statistic),'pearson_CI_low':ci[0],'pearson_CI_high':ci[1],
                    'spearman_descriptive':float(spearmanr(x,y).statistic),
                    'difference_variance_over_independent_sum':float(np.var(x-y)/(np.var(x)+np.var(y))),
                    'variance_ratio_CI_low':vr[0],'variance_ratio_CI_high':vr[1],
                    'median_abs_difference':float(np.median(abs(x-y))),
                    'enough_for_previous_minimum20_rule':len(i)>=20})
                for ii,xx,yy in zip(i,x,y):
                    event_rows.append({'deployment':dep,'kind':kind,'fold':fold,
                        'source_uid':meta.source_uid.iloc[ii],'logmc_error_a':xx,'logmc_error_b':yy})
    n.write_csv(args.root/'tables/SHARED_ERROR_DEPENDENCE.csv',rows)
    n.write_csv(args.root/'tables/SOURCE_ERRORS.csv',event_rows)
    n.write_csv(args.root/'manifest/INPUT_SHA256.csv',manifest)
    n.write_json(args.root/'contracts/DIAGNOSTIC_COMPLETE.json',{'UTC':n.utc(),
        'read_only':True,'no_PE_or_real_data':True,'not_a_predictive_model_upgrade':True,
        'reporting_version':2,'confirmed_reporting_fix':'R29parent_p is R10hybrid,not originalNN. Load original neural ensemble separately and retain both labels.No score function changed.',
        'bootstrap':'5000sourceblocks;conditionalonreusednoise;not fully crossedsource/noise uncertainty',
        'interpretation':'Source-dependent estimator bias can correlate errors although strain noise is independent.Do not substitute pooledfit correlation into all predictive covariances.',
        'minimum_count_failed_for_some_bounded_groups':True,'status':n.STATUS,
        'script_sha256':n.sha(Path(__file__))})
    shutil.copy2(__file__,args.root/'scripts/shared_error_diagnostic.py')
    print(pd.DataFrame(rows).to_string(index=False),flush=True)


if __name__=='__main__':
    main()
