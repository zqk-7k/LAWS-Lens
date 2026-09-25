#!/usr/bin/env python3
"""Freeze the O4b one-dimensional delay LR using training systems only.

The likelihood-ratio estimator is unchanged. The O4b calibration population is
new and source-disjoint, not the historical O3/O4a numerical lookup. Public
joint-live windows constitute an exposure proxy, not a full observing calendar.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import o4b_hl_nso_data_training_20260912 as s


def run(root):
    ws,shared,v3,phys,source=s.environment(root);s.get_split(root)
    out=root/'calibration/time';out.mkdir(parents=True,exist_ok=True)
    contract=out/'TIME_CONTRACT.json';lookup=out/'time_delay_likelihood_ratio.json'
    if (out/'FREEZE.json').exists():
        f=json.loads((out/'FREEZE.json').read_text())
        if s.sha(lookup)!=f['lookup_sha256'] or s.sha(contract)!=f['contract_sha256']:
            raise RuntimeError('Frozen time calibration changed')
        print(json.dumps(f));return
    frame=pd.read_parquet(root/'tables/SOURCE_SPLIT_v2.parquet')
    lens=frame[(frame.split=='train') & frame.family.isin(['SIS','PM'])].copy()
    held=frame[frame.split!='train']
    if set(lens.global_source_id)&set(held.global_source_id):raise RuntimeError('Time-prior source leakage')
    if lens.global_source_id.duplicated().any():raise RuntimeError('One equal-weight row per independent source required')
    schedule_path=ws/'runs/real_gwtc5_o4b_v93_extension_20260830/shared/h1l1_live_schedule.csv'
    table=pd.read_csv(schedule_path)
    columns={'run','start_gps','end_gps','weight_per_second'}
    if not columns.issubset(table):raise RuntimeError('Unknown live-schedule schema')
    schedule=[phys.LiveSegment(start=float(r.start_gps),end=float(r.end_gps),run=str(r.run),
                weight_per_second=float(r.weight_per_second)) for r in table.itertuples()]
    for col in ('gps_image1','gps_image2'):
        if not phys.in_live_segments(lens[col].to_numpy(float),schedule).all():
            raise RuntimeError('Training image lies outside contracted exposure')
    delays=lens.delay_days.to_numpy(float)
    if not np.isfinite(delays).all() or (delays<=0).any():raise RuntimeError('Invalid positive delays')
    observed=(lens.gps_image2.to_numpy()-lens.gps_image1.to_numpy())/86400.
    if not np.allclose(delays,observed,rtol=0,atol=1e-10):raise RuntimeError('Delay/GPS mismatch')
    cfg={'code':'O4B-HL-NSO-01-TIME','utc':s.now(),
        'formula':'log p(log10Delta_days|L,O4b)-log p(log10Delta_days|N,O4b)',
        'jacobian':'identical transformation cancels in the density ratio',
        'estimator':'archived physical_common.fit_time_likelihood_ratio;GaussianScottKDE;grid2048;bandwidth_scale1',
        'lensed_source':'main O4b training sources only;equal weight per global source;both images already placed in contracted exposure',
        'mixture':'420 GW-LMC smooth/non-subhalo plus420 subhalo-present;experiment mixture,not a measured population fraction',
        'independent_lens_systems':len(lens),'unique_numeric_delay_values':int(np.unique(delays).size),
        'no_positive_bootstrap_augmentation':True,
        'null_source':'250000 independent MonteCarlo time pairs under frozen exposure weights;not250000 empirical independent catalog pairs',
        'MonteCarlo_seed':2026091250,'schedule':str(schedule_path),'schedule_sha256':s.sha(schedule_path),
        'schedule_limitation':'public available H1L1 live windows near catalog events;not demonstrated complete O4b duty cycle',
        'source_split_sha256':s.sha(root/'tables/SOURCE_SPLIT_v2.parquet'),
        'validation_test_source_intersection':0,'same_lookup_all_model_seeds':True,
        'SNR_ratio_or_two_dimensional_evidence':False,'real_PE_or_pair_ranks_used':False,
        'out_of_grid':'archived constant endpoint lookup;reportOOD fraction,not linear extrapolation',
        'new_O4b_calibration_not_frozen_O3_numbers':True}
    s.write(contract,cfg)
    null=phys.draw_null_delays(schedule,250000,np.random.default_rng(cfg['MonteCarlo_seed']))
    model=phys.fit_time_likelihood_ratio(delays,null,grid_size=2048,bandwidth_scale=1.)
    for k,v in model.items():
        if isinstance(v,np.ndarray) and not np.isfinite(v).all():raise RuntimeError('Nonfinite time lookup')
    s.write(lookup,{k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in model.items()})
    lens[['global_source_id','family','delay_days','gps_image1','gps_image2']].to_parquet(out/'density_source_systems.parquet',index=False)
    np.save(out/'null_delay_montecarlo_days.npy',null)
    test=np.array([.01,.1,1.,10.,100.]);values=phys.apply_time_likelihood_ratio(test,model)
    expected=np.interp(np.log10(test),model['log10_delay_grid'],model['log_likelihood_ratio']).astype(np.float32)
    if not np.array_equal(values,expected):raise RuntimeError('Time lookup implementation mismatch')
    f={'state':'PASS','utc':s.now(),'lookup_sha256':s.sha(lookup),'contract_sha256':s.sha(contract),
       'source_sha256':s.sha(out/'density_source_systems.parquet'),'null_sha256':s.sha(out/'null_delay_montecarlo_days.npy'),
       'independent_lensed_sources':len(lens),'heldout_scored':False}
    s.write(out/'FREEZE.json',f);print(json.dumps(f),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();run(a.root)
