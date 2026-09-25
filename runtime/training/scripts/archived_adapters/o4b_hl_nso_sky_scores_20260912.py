#!/usr/bin/env python3
"""Map-only sky evidence in a common NESTED probability-mass grid.

No rankings, PE masses, waveform scores or test inference are read. Temporary
dense banks are removed only after pair/manifest checksums have been written.
"""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='4'
import argparse
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import healpy as hp
from astropy.io import fits
from ligo.skymap.io.fits import read_sky_map
from threadpoolctl import threadpool_limits
import o4b_hl_nso_data_training_20260912 as s
import o4b_hl_nso_bayestar_20260912 as sky


def tests():
    rng=np.random.default_rng(2026091251);nside=8;npix=hp.nside2npix(nside)
    p=rng.random(npix);p/=p.sum();q=rng.random(npix);q/=q.sum()
    r1=hp.reorder(p,n2r=True);r2=hp.reorder(q,n2r=True)
    error=abs(p@q-r1@r2)
    if error>1e-14:raise RuntimeError('Common ordering dot-product invariant failed')
    uniform=np.full(npix,1/npix)
    if abs(npix*(p@uniform)-1)>1e-12:raise RuntimeError('Uniform-map neutrality failed')
    k=13;hot=np.zeros(npix);hot[k]=1
    if np.argmax(hp.reorder(hot,n2r=True))!=hp.nest2ring(nside,k):raise RuntimeError('NESTED to RING pixel mismatch')
    return {'state':'PASS','ordering_dot_difference':error,'uniform_neutrality':True,
            'hot_pixel_nest_to_ring':True,'float64_accumulation':True}


def run(root,seed,split):
    out=root/f'sky_pair_scores/seed_{seed}/{split}';out.mkdir(parents=True,exist_ok=True)
    if (out/'COMPLETE.json').exists():
        record=json.loads((out/'COMPLETE.json').read_text())
        if s.sha(out/'pairs.parquet')!=record['pair_sha256']:raise RuntimeError('Completed sky score table changed')
        return
    b=s.load(sky.REF,'o4b_sky_pair_reference')
    if split=='real':
        if not (root/'contracts/FINAL_SCORE_CONFIG_FREEZE.json').exists():
            raise RuntimeError('Real score extraction deferred until final simulation freeze')
        path=root/'workspace/runs/real_gwtc5_o4b_v93_extension_20260830/data/event_manifest.csv'
        events=pd.read_csv(path);events['idx']=np.arange(len(events));events['event_uid']=events.event_name
        paths=events.sky_map_path.map(Path).tolist();temperature=1.
    else:
        folder=root/f'event_maps/seed_{seed}/{split}'
        if not (folder/'event_metrics.COMPLETE.json').exists():raise RuntimeError('BAYESTAR maps incomplete')
        if split=='test' and not (root/'contracts/FINAL_SCORE_CONFIG_FREEZE.json').exists():raise RuntimeError('Test locked')
        cal=root/f'event_maps/seed_{seed}/validation'
        expected=json.loads((cal/'TEMPERATURE_FREEZE.json').read_text())['sha256']
        if s.sha(cal/'TEMPERATURE_SELECTED.json')!=expected:raise RuntimeError('Sky temperature changed')
        temperature=json.loads((cal/'TEMPERATURE_SELECTED.json').read_text())['temperature']
        events=pd.read_parquet(folder/'event_metrics.parquet').sort_values('idx').reset_index(drop=True)
        paths=events.moc_path.map(Path).tolist()
        for p,checksum in zip(paths,events.moc_sha256):
            if s.sha(p)!=checksum:raise RuntimeError('Native MOC checksum mismatch')
    if not np.array_equal(events.idx,np.arange(len(events))):raise RuntimeError('Event index not contiguous')
    ii,jj=np.triu_indices(len(events),1)
    pairs=pd.DataFrame({'idx_i':ii,'idx_j':jj,'event_i':events.event_uid.to_numpy()[ii],
                        'event_j':events.event_uid.to_numpy()[jj]})
    if split!='real':
        groups=events.global_source_id.to_numpy()
        pairs['is_true_pair']=groups[ii]==groups[jj]
        if int(pairs.is_true_pair.sum())!=180:raise RuntimeError('Wrong companion-pair count')
    records=[];temporary=root/'tmp/sky';temporary.mkdir(parents=True,exist_ok=True)
    for nside in (256,512):
        start=time.monotonic();s.disk_guard(root);npix=hp.nside2npix(nside)
        path=temporary/f'{seed}_{split}_{nside}.npy'
        if path.exists():raise RuntimeError('Stale dense temporary; verify previous interruption before resuming')
        bank=np.lib.format.open_memmap(path,mode='w+',dtype=np.float32,shape=(len(events),npix))
        for idx,moc_path in enumerate(paths):
            native=read_sky_map(moc_path,moc=True)
            with fits.open(moc_path) as hdus:frame=str(hdus[1].header.get('COORDSYS','UNKNOWN')).upper()
            if frame not in ('C','ICRS','EQUATORIAL'):raise RuntimeError(f'Unexpected map coordinates:{frame}')
            p=b.raster_probability(native,nside)
            p=b.apply_temperature(p,temperature)
            if not np.isfinite(p).all() or (p<0).any() or abs(p.sum()-1)>1e-10:
                raise RuntimeError('Invalid normalized probability-mass map')
            bank[idx]=p
        bank.flush()
        normalizer=np.array([np.asarray(row,dtype=np.float64).sum() for row in bank])
        overlap=np.zeros((len(events),len(events)),dtype=np.float64)
        bc=np.zeros_like(overlap)
        # Pixel blocking preserves float64 summation with bounded working RAM.
        with threadpool_limits(limits=4):
            for first in range(0,npix,16384):
                block=np.asarray(bank[:,first:first+16384],dtype=np.float64)/normalizer[:,None]
                overlap+=block@block.T
                rootblock=np.sqrt(block);bc+=rootblock@rootblock.T
        if not np.isfinite(overlap).all() or not np.isfinite(bc).all():raise RuntimeError('Invalid sky Gram matrix')
        pairs[f'sky_log_bf_nside{nside}']=np.log(np.maximum(npix*overlap[ii,jj],1e-300))
        pairs[f'sky_BC_nside{nside}']=bc[ii,jj].clip(0,1)
        records.append({'nside':nside,'events':len(events),'pairs':len(pairs),'seconds':time.monotonic()-start,
            'ordering':'NESTED','coordinates':'equatorial','probability':'pixel mass;renormalized after float32 storage',
            'temperature':temperature,'accumulation':'float64','native_map_sha256':[s.sha(p) for p in paths],
            'temporary_dense_path':str(path),'temporary_bytes':path.stat().st_size})
        del bank
    pairs['sky_raw_log_bf']=pairs.sky_log_bf_nside512
    pairs['sky_BC']=pairs.sky_BC_nside512
    pairs['sign_flip_256_512']=np.sign(pairs.sky_log_bf_nside256)!=np.sign(pairs.sky_log_bf_nside512)
    pairs['abs_delta_256_512']=abs(pairs.sky_log_bf_nside256-pairs.sky_log_bf_nside512)
    pairs.to_parquet(out/'pairs.parquet',index=False)
    events.to_parquet(out/'event_plan.parquet',index=False)
    s.write(out/'MAP_MANIFEST.json',records)
    s.write(out/'NUMERICAL_TESTS.json',tests())
    record={'state':'PASS','events':len(events),'pairs':len(pairs),'utc':s.now(),
        'pair_sha256':s.sha(out/'pairs.parquet'),'events_sha256':s.sha(out/'event_plan.parquet'),
        'map_manifest_sha256':s.sha(out/'MAP_MANIFEST.json'),'rankings_computed':False,
        'analysis_nside':512,'audit1024':'targeted audit still required before final claims',
        'formal_statistic':'log(Npix*sum(pixel_mass_i*pixel_mass_j));BC diagnostic only'}
    s.write(out/'COMPLETE.json',record)
    for row in records:Path(row['temporary_dense_path']).unlink()
    print(json.dumps(record),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--split',choices=('validation','test','real'),default='validation')
    p.add_argument('--test-functions',action='store_true');a=p.parse_args()
    if a.test_functions:print(json.dumps(tests()))
    else:run(a.root,a.seed,a.split)
