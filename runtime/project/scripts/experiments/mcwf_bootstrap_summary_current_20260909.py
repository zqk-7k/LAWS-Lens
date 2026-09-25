#!/usr/bin/env python3
"""Summarize paired bootstrap draws without treating model reuse as new data."""
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

P=Path('/root/autodl-tmp/gw-catalog')
sys.path.insert(0,str(P/'scripts/experiments'))
import mcwf_nodup_reliability_20260909 as r
n=r.n


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--bootstrap-root',type=Path,required=True)
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args()
    if args.root.exists():
        raise RuntimeError('Independent summary directory required')
    for d in ('contracts','tables','scripts','manifest'):
        (args.root/d).mkdir(parents=True)
    contract=json.loads((args.bootstrap_root/'contracts/BOOTSTRAP_CONTRACT.json').read_text())
    rows=pd.read_csv(args.bootstrap_root/'tables/CONDITIONAL_SYSTEM_CI_PER_PANEL.csv')
    values={}; inputs=[]
    for path in args.bootstrap_root.glob('draws/gwtc*/seed_*/*/*/*.parquet'):
        dep,seed,panel,method,name=path.relative_to(args.bootstrap_root/'draws').parts
        mode=name.split('_')[0]
        frame=pd.read_parquet(path)
        for quantity in frame:
            values[dep,seed,panel,method,mode,quantity]=frame[quantity].to_numpy(float)
        inputs.append({'path':str(path),'sha256':n.sha(path),'bytes':path.stat().st_size})
    summary,differences=[],[]
    def summarize(group,keys):
        columns=['deployment','method','mode','quantity']
        arrays={}
        for (dep,method,mode,quantity), frame in group.groupby(columns):
            a=np.mean([values[row.deployment,row.seed,row.panel,row.method,row.mode,row.quantity]
                       for row in frame.itertuples()],axis=0)
            arrays[dep,method,mode,quantity]=(a,float(frame.nominal.mean()))
            summary.append({**keys,'deployment':dep,'method':method,'mode':mode,'quantity':quantity,
                'nominal':float(frame.nominal.mean()),'q025':float(np.quantile(a,.025)),
                'q975':float(np.quantile(a,.975)),'bootstrap_repetitions':len(a),'panel_count':len(frame)})
        for (dep,method,mode,quantity),(a,nominal) in arrays.items():
            for baseline in ('NODUP-DIRECT-REPLAY','PATH875-ARCHIVED'):
                if baseline==method or (dep,baseline,mode,quantity) not in arrays:
                    continue
                b,bnom=arrays[dep,baseline,mode,quantity]
                delta=a-b
                sign=-1 if quantity.startswith('F') else 1
                differences.append({**keys,'deployment':dep,'method':method,'baseline':baseline,
                    'mode':mode,'quantity':quantity,'nominal_delta':nominal-bnom,
                    'delta_q025':float(np.quantile(delta,.025)),'delta_q975':float(np.quantile(delta,.975)),
                    'bootstrap_fraction_nondegradation':float(np.mean(sign*delta>=0)),
                    'descriptive_bootstrap_frequency_not_posterior_probability':True})
    for (dep,seed),frame in rows.groupby(['deployment','seed']):
        for split in ('test','sept8_reused'):
            f=frame[frame.panel.eq('test')] if split=='test' else frame[frame.panel.str.startswith('sept8_reused_')]
            if len(f):
                summarize(f,{'seed':seed,'split':split,'aggregation':'per_model_conditional'})
    reused=rows[rows.panel.str.startswith('sept8_reused_')]
    summarize(reused,{'seed':'three_fixed_models','split':'sept8_reused',
        'aggregation':'same_catalog_system_draws_shared_across_models'})
    n.write_csv(args.root/'tables/CONDITIONAL_SYSTEM_CI_SUMMARY.csv',summary)
    n.write_csv(args.root/'tables/PAIRED_METHOD_DELTA_CI.csv',differences)
    n.write_csv(args.root/'manifest/INPUT_SHA256.csv',inputs)
    n.write_json(args.root/'contracts/SUMMARY_CONTRACT.json',{'UTC':n.utc(),'bootstrap_root':str(args.bootstrap_root),
        'prior_contract':contract,'legacy_test_cross_model_CI_not_combined':True,
        'reason':'Historical model-specific test catalogs overlap, but a verified global-source map across those catalogs is not part of this conditional audit. Report per-model intervals rather than assume independence.',
        'reused_catalog_model_combination':'For each shared catalog use identical system draws for all models; average catalogs within each fixed model, then average models. Model-seed SD remains separate.',
        'not_model_sampling_or_independent_confirmation':True,'does_not_change_point_estimate_guardrails':True})
    shutil.copy2(__file__,args.root/'scripts/bootstrap_summary.py')
    print('BOOTSTRAP_SUMMARY_COMPLETE',len(summary),len(differences),flush=True)


if __name__=='__main__':
    main()
