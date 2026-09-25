#!/usr/bin/env python3
"""Frozen post-selection attribution; never used to reselect the method."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
import shutil
import numpy as np
import pandas as pd
P=Path('/root/autodl-tmp/gw-catalog');sys.path.insert(0,str(P/'scripts/experiments'))
import mcwf_path_release_v3_20260908 as release
t,dev,cf=release.t,release.dev,release.cf


def run(root):
    done=root/'contracts/ATTRIBUTION_COMPLETE.json'
    if done.exists():raise RuntimeError('Already complete; do not overwrite')
    selected=json.loads((root/'contracts/SELECTED_GLOBAL_DEVELOPMENT.json').read_text())
    methods=('OLD-WAVEFORM-NEW-WEIGHTS','NEW-WAVEFORM-OLD-WEIGHTS')
    dev.json_write(root/'contracts/ATTRIBUTION_FREEZE.json',{'UTC':datetime.now(timezone.utc).isoformat(),
        'selected_sha256':dev.sha(root/'contracts/SELECTED_GLOBAL_DEVELOPMENT.json'),'code_sha256':dev.sha(Path(__file__)),
        'methods':methods,'purpose':'Frozen post-selection causal-component diagnostic. No new selection or adoption.'})
    shutil.copy2(__file__,root/'scripts/path_attribution.py')
    ws=pd.read_csv(root/'tables/SELECTED_WEIGHTS.csv').set_index(['deployment','seed'])
    metrics,budgets=[],[]
    for dep in t.DEPS:
        external=pd.read_parquet(t.EXTERNAL/f'{dep}_external_reference.parquet')
        ranked={m:[] for m in methods}
        for seed in t.SEEDS:
            neww=ws.loc[(dep,seed),['effective_weight_waveform','effective_weight_time','effective_weight_sky']].to_numpy(float)
            oldw=cf.frozen_weights(dep,seed)
            for split in ('validation','test','real'):
                name='real_fusion_pairs.parquet' if split=='real' else split+'_pairs.parquet'
                base=root/f'evaluation/OMC/{dep}/seed_{seed}/{name}'
                candidate=root/f'evaluation/{release.CODE}/{dep}/seed_{seed}/{name}'
                b,c=pd.read_parquet(base),pd.read_parquet(candidate)
                keys=['pair_key'] if split=='real' else ['idx_i','idx_j']
                c=c.set_index(keys).reindex(b.set_index(keys).index).reset_index()
                for method,frame,w in ((methods[0],b,neww),(methods[1],c,oldw)):
                    dest=root/f'evaluation/{method}/{dep}/seed_{seed}';dest.mkdir(parents=True,exist_ok=True)
                    if split=='real':
                        f=dev.BASE.rank_real(frame,cf.weights_dict(w),'fusion',seed)
                        ranked[method].append(f)
                        release.join_external(f,external).to_parquet(dest/'real_fusion_pairs.parquet',index=False)
                    else:
                        score=cf.channels(frame,frame.waveform_score.to_numpy(float))@w
                        metrics.append({'configuration':method,'deployment':dep,'seed':seed,'split':split,
                                        **dev.BASE.full_metrics(frame,score)})
        for method,items in ranked.items():
            f=release.join_external(dev.BASE.consensus_real(items,'fusion'),external)
            dest=root/f'evaluation/{method}/{dep}'
            f.to_parquet(dest/'consensus_fusion_all_pairs.parquet',index=False)
            dev.csv_write(dest/'consensus_fusion_top100.csv',f.head(100))
            for budget in release.BUDGETS:budgets.append(dev.budget_row(f,method,dep,'fusion',budget))
    dev.csv_write(root/'tables/ATTRIBUTION_INJECTION_METRICS.csv',pd.DataFrame(metrics))
    dev.csv_write(root/'tables/ATTRIBUTION_PE_OFFICIAL_BUDGETS.csv',pd.DataFrame(budgets))
    if dev.sha(root/'contracts/SELECTED_GLOBAL_DEVELOPMENT.json')!=json.loads((root/'contracts/ATTRIBUTION_FREEZE.json').read_text())['selected_sha256']:
        raise RuntimeError('Selected method changed')
    dev.json_write(done,{'complete':True,'no_reselection':True,'real_diagnostics_adaptive':True})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();run(a.root)
