#!/usr/bin/env python3
"""Conditional event-specific O4b BAYESTAR, matching the archived sky method.

Uses known intrinsic parameters and Gaussian matched-filter trigger errors with
each empirical PSD. This is explicitly not PE of the identical noisy strain.
"""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import argparse
import fcntl
from concurrent.futures import ProcessPoolExecutor
import json
import logging
import multiprocessing as mp
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import pandas as pd
import o4b_hl_nso_data_training_20260912 as s

P=Path('/root/autodl-tmp/gw-catalog')
REF=P/'results/bayestar_injection_sky_pe_20260901_20260901T102000Z/scripts/bayestar_injection_sky_full_experiment.py'
CTX={}


def initialize(root):
    contract=root/'contracts/BAYESTAR_O4B_CONTRACT.json'
    if not contract.exists():
        s.write(contract,{'utc':s.now(),'source':str(REF),'source_sha256':s.sha(REF),
            'detectors':['H1','L1'],'intrinsic_template':'known injected masses/spins;IMRPhenomPv2 primary;IMRPhenomD fallback logged',
            'measurement_noise':'official ligo.skymap simulate_snr Gaussian errors, independently seeded for each image',
            'empirical_PSD':'same assigned O4b noise-bank PSD as waveform injection',
            'not_same_noise_strain_PE':True,'not_full_BBH_PE':True,'not_rotated_PE_templates':True,
            'analysis_nside':512,'dense_persistence':False,'native_output':'NUNIQ multi-order probability density;ICRS equatorial',
            'transient_raster_ordering':'NESTED; common final pair scorer must explicitly convert or require same ordering for every operand',
            'temperature_grid':[1,1.25,1.5,2,2.5,3,4],'temperature_selection':'validation only;one lensed-system weight1,one unlensed-event weight1',
            'test_requires_final_score_freeze':True,'real_maps':'public PE;75HLV+11HL, not homogeneous network with HL injection',
            'measurement_seeds':'stable hash of O4b,catalog_seed,split,event index;independent from waveform-noise sampling'})
        s.write(root/'contracts/BAYESTAR_O4B_FREEZE.json',{'sha256':s.sha(contract)})


def records(root,seed,split):
    initialize(root);s.get_split(root)
    if split=='test' and not (root/'contracts/FINAL_SCORE_CONFIG_FREEZE.json').exists():
        raise RuntimeError('Locked test must remain sealed before final waveform/time/sky calibration')
    shared=root/'workspace/results/real_noise_injection_v5_physical_source_20260721/gwtc5/shared'
    inj=shared.parent/f'seed_{seed}/data/real_noise_injections'
    if not (inj/'catalog_complete.json').exists():raise RuntimeError('Injection catalog incomplete')
    frame=pd.read_parquet(inj/'compact_injection_metadata.parquet')
    frame=frame[frame.split==('val' if split=='validation' else 'test')]
    result=[]
    for r in frame.sort_values(['family','sample_index']).to_dict('records'):
        for image in range(1,2 if r['family']=='unlensed' else 3):
            idx=len(result);morse=r.get(f'morse_image{image}',0)
            record={'idx':idx,'family':r['family'],'source_index':r['sample_index'],'global_source_id':r['global_source_id'],
                'tag':'U' if r['family']=='unlensed' else f'L{image}','image_number':image,'model_seed':seed,'catalog_seed':int(r['catalog_seed']),
                'split':split,'gwlmc_row':int(r['gwlmc_row']),'gps_obs':r[f'gps_image{image}'],
                'ra_true':r['ra_true'],'dec_true':r['dec_true'],'noise_bank_index':int(r[f'image{image}_noise_bank_index']),
                'target_network_snr':r[f'target_snr_image{image}'],'morse_index':float(morse) if np.isfinite(morse) else 0.,
                'source_noise_parent':r[f'image{image}_noise_parent']}
            import hashlib
            record['measurement_seed']=int.from_bytes(hashlib.sha256(f'O4bBAYESTAR:{r["catalog_seed"]}:{split}:{idx}'.encode()).digest()[:4],'little')
            record['event_uid']=f'O4B_{r["catalog_seed"]}_{r["family"]}_{r["sample_index"]}_{image}'
            result.append(record)
    if len(result)!=450:raise RuntimeError('Unexpected held-out catalog size')
    out=root/f'event_maps/seed_{seed}/{split}';out.mkdir(parents=True,exist_ok=True)
    path=out/'event_plan.parquet'
    if path.exists():
        if not pd.read_parquet(path).equals(pd.DataFrame(result)):raise RuntimeError('BAYESTAR event plan changed')
    else:pd.DataFrame(result).to_parquet(path,index=False)
    return result


def worker_init(root,seed,split):
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=1)
    root=Path(root);b=s.load(REF,'o4b_archived_bayestar')
    shared=root/'workspace/results/real_noise_injection_v5_physical_source_20260721/gwtc5/shared'
    b._WORKER_PSD_FREQ=np.load(shared/'noise_psd_frequency.npy')
    b._WORKER_PSD_BANK=np.load(shared/'noise_psd_bank.npy',mmap_mode='r')
    CTX.update(root=root,b=b,source=pd.read_csv(b.SOURCE_CSV),out=root/f'event_maps/seed_{seed}/{split}')
    logging.getLogger('ligo.skymap').setLevel(logging.ERROR)


def localize(record):
    from ligo.skymap.io.fits import write_sky_map
    from ligo.skymap import moc
    b=CTX['b'];root=CTX['root'];out=CTX['out'];stem=f'event_{record["idx"]:04d}'
    path=out/f'{stem}.fits.gz';marker=out/f'{stem}.json';started=time.monotonic()
    if marker.exists():
        previous=json.loads(marker.read_text())
        if not path.exists() or s.sha(path)!=previous['moc_sha256']:raise RuntimeError('Completed MOC changed')
        return previous
    s.disk_guard(root)
    try:
        source=CTX['source'].iloc[record['gwlmc_row']];fallback=False;error=None
        try:sky,audit=b._simulate_with_waveform(record,source,b.PRIMARY_WAVEFORM)
        except Exception as exc:
            fallback=True;error=repr(exc);sky,audit=b._simulate_with_waveform(record,source,b.FALLBACK_WAVEFORM)
        density=np.asarray(sky['PROBDENSITY'])
        if not np.isfinite(density).all() or (density<0).any():raise RuntimeError('Invalid native MOC')
        prob=b.raster_probability(sky,512)
        metrics=b.posterior_metrics(prob,record['ra_true'],record['dec_true'])
        orders=moc.uniq2order(np.asarray(sky['UNIQ'],np.int64))
        temporary=out/f'{stem}.tmp.fits.gz';write_sky_map(temporary,sky);temporary.replace(path)
        row={**record,**audit,**metrics,'fallback_used':fallback,'primary_error':error,
            'analysis_nside':512,'transient_ordering':'NESTED','native_moc_order_min':int(orders.min()),'native_moc_order_max':int(orders.max()),
            'probability_sum':float(prob.sum(dtype=np.float64)),'moc_path':str(path),'moc_bytes':path.stat().st_size,'moc_sha256':s.sha(path),
            'seconds':time.monotonic()-started,'dense_persisted':False}
        s.write(marker,row);return row
    except Exception:
        s.write(out/f'{stem}.failure.{int(time.time())}.json',{'record':record,'traceback':traceback.format_exc()});raise


def run(root,seed,split,workers,pilot):
    rows=records(root,seed,split)
    if pilot:rows=rows[:pilot]
    progress=[]
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn'),initializer=worker_init,initargs=(str(root),seed,split)) as pool:
        for i,row in enumerate(pool.map(localize,rows,chunksize=1),1):
            progress.append(row)
            if i%10==0 or i==len(rows):print(json.dumps({'BAYESTAR':split,'seed':seed,'complete':i,'total':len(rows),'seconds_last':row['seconds']}),flush=True)
    out=root/f'event_maps/seed_{seed}/{split}';name='pilot_metrics' if pilot else 'event_metrics'
    frame=pd.DataFrame(progress);frame.to_parquet(out/f'{name}.parquet',index=False)
    s.write(out/f'{name}.COMPLETE.json',{'events':len(progress),'failures':0,'native_MOC_only':True,'test_scored':False})
    if not pilot and split=='validation':
        b=s.load(REF,'o4b_bayestar_temperature')
        temperature,grid=b.select_temperature(frame)
        grid.to_csv(out/'temperature_grid.csv',index=False)
        config={'temperature':temperature,'selection':'source-weighted validation coverage/CVM only',
                'events':len(frame),'event_metrics_sha256':s.sha(out/f'{name}.parquet'),'no_test_or_real':True}
        s.write(out/'TEMPERATURE_SELECTED.json',config)
        s.write(out/'TEMPERATURE_FREEZE.json',{'sha256':s.sha(out/'TEMPERATURE_SELECTED.json')})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--split',choices=['validation','test'],default='validation');p.add_argument('--workers',type=int,default=4)
    p.add_argument('--pilot',type=int,default=0);a=p.parse_args()
    with (a.root/'contracts'/f'BAYESTAR_{a.seed}_{a.split}.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        run(a.root,a.seed,a.split,a.workers,a.pilot)
