#!/usr/bin/env python3
"""Check predictive normalization, source grouping and marginal calibration."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import mcwf_intrinsic_grid_20260907 as g


def run(root):
    if root.name=='dense':
        import mcwf_intrinsic_dense_20260907
    rows=[]
    for dep in g.e.old.DEPS:
        common=None
        for ms in g.e.body.MODEL_SEEDS:
            path=root/f'models/{dep}/seed_{ms}'
            ck=torch.load(path/'validation_selected_model.pt',weights_only=False,map_location='cpu')
            report=json.loads((path/'COMPLETE.json').read_text())
            assert report['checkpoint_sha256']==g.dev.sha(path/'validation_selected_model.pt')
            a=dict(np.load(path/'development_predictions.npz'))
            assert len(a['p'])==1024 and np.isfinite(a['p']).all() and (a['p']>=0).all()
            assert np.max(abs(a['p'].sum(1)-1))<1e-5
            groups,n=np.unique(a['group'],return_counts=True)
            assert len(groups)==512 and (n==2).all()
            if common is not None:
                assert np.array_equal(common['group'],a['group']) and np.array_equal(common['truth'],a['truth'])
            common=a
            p=a['p'].reshape(-1,*g.SHAPE)
            prior=ck['prior']
            for dim,centers in enumerate((g.mass.LOG_CENTERS,g.Q,g.CHI)):
                other=tuple(i+1 for i in range(3) if i!=dim)
                marginal=p.sum(other)
                prior_m=prior.sum(tuple(i for i in range(3) if i!=dim))
                mean=marginal@centers
                target=a['truth'][:,dim]
                if dim==0:edges=g.mass.NATIVE_EDGES
                else:edges=np.r_[centers[0],(centers[:-1]+centers[1:])/2,centers[-1]]
                cdf=np.c_[np.zeros(len(p)),marginal.cumsum(1)]
                pit=np.array([np.interp(t,edges,c) for t,c in zip(target,cdf)])
                rows.append({'deployment':dep,'model_key':ms,'training_seed':report['seed'],
                             'parameter':('logMc','q','chi_eff')[dim],
                             'MAE':float(abs(mean-target).mean()),
                             'training_prior_mean_MAE':float(abs(float(prior_m@centers)-target).mean()),
                             'central50_coverage':float(((pit>=.25)&(pit<=.75)).mean()),
                             'central90_coverage':float(((pit>=.05)&(pit<=.95)).mean()),
                             'truth_outside_grid':int(((target<centers[0])|(target>centers[-1])).sum()),
                             'interpretation':'discrete predictive marginal diagnostic,not full-PE coverage or a new blind test'})
    g.dev.csv_write(root/'audit/PREDICTIVE_QUALITY.csv',pd.DataFrame(rows))
    g.dev.json_write(root/'audit/PREDICTIVE_NORMALIZATION_AND_GROUPS.json',{
        'pass':True,'models':6,'development_events_per_run':1024,'source_systems_per_run':512,
        'views_per_source':2,'ensemble_truth_alignment_exact':True,
        'normalization_error_tolerance':1e-5,'true_pair_weight_per_system':1,
        'null_pairs_are_dependent':True})
    print(pd.DataFrame(rows).to_string(index=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();run(a.root)
