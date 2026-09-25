#!/usr/bin/env python3
"""Signal-only algebra tests independent of the pilot outcomes and real PE."""
import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd

P=Path('/root/autodl-tmp/gw-catalog')
sys.path.insert(0,str(P/'scripts/experiments'))
import mcwf_shared_profile_coherence_pilot_20260909 as p


def inject(model, theta, amplitude):
    bank=model.template(theta)
    model.raw=amplitude*bank[:,0]+.37*amplitude*bank[:,1]
    model.data=np.fft.rfft(model.raw,model.nfft)


def main(root):
    p.init_compute()
    plans=pd.read_parquet(root/'contracts/PAIR_PLAN.parquet')
    theta=[np.array([np.log(10.),.6,.1]),np.array([np.log(14.),.7,-.1])]
    outputs=[]
    for dep in p.n.DEPS:
        row=plans[plans.deployment.eq(dep)&plans.kind.eq('true')].iloc[0]
        models=[p.model(root,dep,int(k))for k in (row.idx_i,row.idx_j)]
        for model,point in zip(models,theta):
            inject(model,point,10.)
        result=p.coherent(models,theta,budget=180,refine_budget=120)
        swapped=p.coherent(models[::-1],theta[::-1],budget=180,refine_budget=120)
        bound=sum(.5*np.sum(m.raw**2)for m in models)
        if not result['deficit']>0 or result['shared_power']>bound*(1+1e-9):
            raise RuntimeError('Nonidentical sources or projection-energy bound failed')
        if abs(result['deficit']-swapped['deficit'])>1e-7:
            raise RuntimeError('Nonidentical-source exchange symmetry failed')
        outputs.append({'deployment':dep,'different_source_deficit':result['deficit'],
            'swap_absolute_difference':abs(result['deficit']-swapped['deficit']),
            'shared_power':result['shared_power'],'data_energy_upper_bound':bound})
    p.n.write_csv(root/'audit/SYNTHETIC_DIFFERENT_SOURCE_OPERATOR_TESTS.csv',outputs)
    p.n.write_json(root/'contracts/ADDITIONAL_OPERATOR_TESTS_PASS.json',{
        'UTC':p.n.utc(),'PASS':True,'uses_pilot_outcomes':False,'uses_real_PE':False,
        'limit':'Algebra and units only;does not establish global optimization or population coverage.'})
    print('ADDITIONAL_OPERATOR_TESTS_PASS',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    main(parser.parse_args().root)
