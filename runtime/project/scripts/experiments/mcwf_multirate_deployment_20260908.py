#!/usr/bin/env python3
"""Replay frozen raw injection/real inputs for an independent lowband branch."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd

P=Path('/root/autodl-tmp/gw-catalog')
sys.path.insert(0,str(P/'scripts/experiments'))
import mcwf_multirate_features_20260908 as m
t,dev=m.t,m.dev


def injection_raw(root,dep,es,split):
    folder=root/f'cache/deployment/{dep}/{es}_{split}'
    if (folder/'COMPLETE.json').exists():
        return np.load(folder/'low16s.npy',mmap_mode='r')
    folder.mkdir(parents=True,exist_ok=True)
    _,_,v3=m.expanded.modules()
    plan=dev.BASE.retained_event_plan(dep,es,split)
    old=dev.ORCH.event_array_for_plan(dep,es,split,plan)
    meta=pd.read_parquet(dev.ORCH.INPUT_ROOT/dep/f'seed_{es}/data/real_noise_injections/compact_injection_metadata.parquet').set_index(['family','sample_index'])
    base=dev.ORCH.SOURCE_ROOT/dep/'shared'
    refs=np.load(base/'noise_reference_bank.npy',mmap_mode='r')
    freq=np.load(base/'noise_psd_frequency.npy');psds=np.load(base/'noise_psd_bank.npy',mmap_mode='r')
    bank=base/'physical_h1l1_source_bank'
    stores={('SIS',1):np.load(bank/'SIS_data_0222/SIS_h_strain_1.npy',mmap_mode='r'),
            ('SIS',2):np.load(bank/'SIS_data_0222/SIS_h_strain_2.npy',mmap_mode='r'),
            ('PM',1):np.load(bank/'PM_data_0222/PM_h_strain_1.npy',mmap_mode='r'),
            ('PM',2):np.load(bank/'PM_data_0222/PM_h_strain_2.npy',mmap_mode='r'),
            ('unlensed',1):np.load(bank/'Unlensed_data_0222/unlensed_h_strain.npy',mmap_mode='r')}
    rows=[];values=[];started=time.perf_counter()
    for i,event in enumerate(plan.itertuples(index=False)):
        family='unlensed' if event.tag=='U' else event.family
        image=2 if event.tag=='L2' else 1
        record=meta.loc[(family,int(event.source_index))]
        b,offset=int(record[f'image{image}_noise_bank_index']),int(record[f'image{image}_noise_offset_samples'])
        if b!=int(event.parent_noise_bank):raise RuntimeError('Noise PSD provenance mismatch')
        target=float(record[f'image{image}_target_network_snr'])
        clean=np.asarray(stores[(family,image)][int(event.source_index)],np.float32)
        scaled,_,_=v3.scale_to_network_snr(clean,target,freq,psds[b])
        noise=np.asarray(refs[b,:,offset:offset+v3.RAW_PADDED_SAMPLES],np.float32)
        raw=noise+v3.embed_signal_in_padded_window(scaled)
        full=v3.preprocess_24s(raw,freq,psds[b])
        difference=float(abs(full-old[i]).max())
        if not np.array_equal(full,old[i]):raise RuntimeError(f'Frozen injection replay mismatch:{dep}:{es}:{split}:{i}:{difference}')
        values.append(m.low_view(raw,freq,psds[b],v3))
        rows.append({'row':i,'idx':int(event.idx),'source_index':int(event.source_index),
            'system_id':event.system_id,'noise_bank':b,'offset':offset,'target_network_SNR_frozen':target,
            'full24_replay_max_difference':difference,'lowband_std_H1':values[-1][0].std(),'lowband_std_L1':values[-1][1].std()})
    values=np.stack(values);np.save(folder/'low16s.npy',values)
    dev.csv_write(folder/'RAW_REPLAY.csv',pd.DataFrame(rows))
    dev.json_write(folder/'COMPLETE.json',{'events':len(values),'full24_exact':True,'sha256':dev.sha(folder/'low16s.npy'),
        'seconds':time.perf_counter()-started,'new_noise_or_SNR_scaling':False})
    return values


def real_raw(root,dep):
    if not (root/'contracts/INTEGRATION_FROZEN.json').exists():raise RuntimeError('No real inputs before simulation selection freeze')
    folder=root/f'cache/deployment/{dep}/real'
    if (folder/'COMPLETE.json').exists():return np.load(folder/'low16s.npy',mmap_mode='r')
    folder.mkdir(parents=True,exist_ok=True)
    _,_,v3=m.expanded.modules()
    full,events=dev.real_inputs(dep)
    source=dev.MAIN/'cache/source_run' if dep=='gwtc3' else v3.SOURCES['GWTC4']
    cache=v3.HdfCache(source,max_files=6)
    freq,psds,ids,_=t.contexts.psd_context(dep,0,'real')
    valid=events[events.strict_h1l1_preprocessing_pass.astype(bool)]
    rows=[];values=[];started=time.perf_counter()
    for i,event in enumerate(valid.itertuples(index=False)):
        details=json.loads(event.detector_audit)
        bydet={d['detector']:d for d in details}
        channels=[]
        for detector in ('H1','L1'):
            detail=bydet[detector]
            data,gps_start,duration=cache.get(detail['path'])
            if abs(len(data)/duration-4096)>1e-3:raise RuntimeError('Unexpected GWOSC sample rate')
            onsource=v3._extract_channel_window(data,gps_start,float(event.gps_time),v3.RAW_PADDED_SAMPLES,-24.75)
            if onsource is None or not np.isfinite(onsource).all():raise RuntimeError('Incomplete strict H1L1 event')
            channels.append(onsource)
        raw=np.stack(channels);psd=psds[ids[i]]
        original=v3.preprocess_24s(raw,freq,psd)
        difference=float(abs(original-full[int(event.idx)]).max())
        if not np.array_equal(original,full[int(event.idx)]):raise RuntimeError(f'Frozen real replay mismatch:{dep}:{event.event_name}:{difference}')
        values.append(m.low_view(raw,freq,psd,v3))
        rows.append({'idx':int(event.idx),'event':event.event_name,'PSD_index':int(ids[i]),
                     'full24_replay_max_difference':difference,'H1_path':bydet['H1']['path'],'L1_path':bydet['L1']['path']})
    values=np.stack(values);np.save(folder/'low16s.npy',values)
    dev.csv_write(folder/'RAW_REPLAY.csv',pd.DataFrame(rows))
    dev.json_write(folder/'COMPLETE.json',{'events':len(values),'sha256':dev.sha(folder/'low16s.npy'),
        'full24_exact':True,'seconds':time.perf_counter()-started,'PE_or_official_used':False})
    return values


def deployment_features(root,dep,es,split):
    path=root/f'features/{dep}'/('real.npy' if split=='real' else f'{es}_{split}.npy')
    if path.with_suffix('.COMPLETE.json').exists():return path
    freq,psds,ids,_=t.contexts.psd_context(dep,es,split)
    raw=real_raw(root,dep) if split=='real' else injection_raw(root,dep,es,split)
    if len(raw)!=len(ids):raise RuntimeError('PSD/input mapping length')
    m.grouped(root,raw,freq,psds,ids,path)
    return path


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--split',choices=['validation','test','real'],default='validation')
    a=p.parse_args()
    for dep in t.DEPS:
        for es in t.SEEDS[:1] if a.split=='real' else t.SEEDS:
            deployment_features(a.root,dep,es,a.split)
