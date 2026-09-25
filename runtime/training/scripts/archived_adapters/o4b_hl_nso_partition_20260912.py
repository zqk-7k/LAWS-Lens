#!/usr/bin/env python3
"""Freeze source and empirical-noise partitions without inspecting PE values."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda:f.read(8*1024**2),b''):h.update(data)
    return h.hexdigest()


def write(path,value):
    if path.exists():raise FileExistsError(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def key(value):return hashlib.sha256(('O4B-HL-NSO-01:partition:'+str(value)).encode()).hexdigest()


def build(root):
    target=root/'contracts/DATA_SPLIT_CONTRACT.json'
    if target.exists():
        c=json.loads(target.read_text())
        for x in c['outputs']:
            if sha(x['path'])!=x['sha256']:raise RuntimeError('Frozen partition hash changed')
        print(json.dumps({'state':'VERIFIED_EXISTING_PARTITION','contract':str(target)}));return
    m=pd.read_parquet(root/'tables/PHYSICAL_SOURCE_METADATA.parquet')
    if len(m)!=1800 or m.gwlmc_event_id.nunique()!=1800:raise RuntimeError('Source IDs not unique')
    m['global_source_id']=m.gwlmc_event_id.astype(int).astype(str)
    m['source_index']=m.pair_id.fillna(m.sample_index).astype(int)
    # A common index split keeps the legacy validation control layout exact.
    # The global-ID audit above ensures that this does not split duplicate sources.
    order=sorted(range(600),key=key)
    index_split={i:('train' if j<420 else 'val' if j<510 else 'test') for j,i in enumerate(order)}
    m['split']=m.source_index.map(index_split)
    source_path=root/'tables/SOURCE_SPLIT.parquet';m.to_parquet(source_path,index=False)
    source_csv=root/'tables/SOURCE_SPLIT.csv';m.to_csv(source_csv,index=False,encoding='utf-8-sig')
    detailed=root/'contracts/historical/stage0/gwtc5_o4b_stage0_event_audit_detailed.json'
    entries=json.loads(detailed.read_text())
    known=np.array([x['catalog_entry']['gps'] for x in entries],dtype=float)
    raw=root/'workspace/runs/real_gwtc5_o4b_v93_extension_20260830/data/real_strain'
    by_parent={}
    for item in entries:
        details=item.get('detector_details',{})
        if not all(d in details and details[d].get('data_segments') for d in ('H1','L1')):continue
        h,l=details['H1'],details['L1']
        lo=max(h['gps_start'],l['gps_start']);hi=min(h['gps_start']+h['duration'],l['gps_start']+l['duration'])
        if hi-lo<256:continue
        parent=f"{int(h['gps_start'])}:{int(l['gps_start'])}"
        for start in range(int(np.ceil(lo/256)*256),int(hi)-255,256):
            end=start+256
            if np.any((known+128>start)&(known-128<end)):continue
            if not all(any(a<=start and end<=b for a,b in details[d]['data_segments']) for d in ('H1','L1')):continue
            by_parent.setdefault(parent,{})[start]={
                'parent_raw_file_group':parent,'noise_block_uid':f'HL:{start}:256',
                'start_gps':start,'duration_s':256,'sample_rate':4096,
                'h1_path':str(raw/Path(h['hdf5_url']).name),'l1_path':str(raw/Path(l['hdf5_url']).name),
                'h1_file_start':h['gps_start'],'l1_file_start':l['gps_start']}
    parents=sorted(by_parent,key=key)
    if len(parents)<20:raise RuntimeError('Too few independent raw-file noise groups')
    a=int(.70*len(parents));b=int(.85*len(parents));rows=[]
    for index,parent in enumerate(parents):
        split='train' if index<a else 'val' if index<b else 'test'
        rows.extend({**row,'split':split} for row in by_parent[parent].values())
    noise=pd.DataFrame(rows).sort_values(['split','noise_block_uid']).reset_index(drop=True)
    if noise.noise_block_uid.duplicated().any():raise RuntimeError('Repeated physical noise interval')
    # Select by hash, not by measured PSD or retrieval performance. Invalid blocks
    # later fail the gate rather than silently being replaced after evaluation.
    selected=[]
    for split,n in [('train',128),('val',32),('test',32)]:
        sub=noise[noise.split==split].copy()
        sub['hash_order']=sub.noise_block_uid.map(key)
        if len(sub)<n:raise RuntimeError(f'Insufficient {split} noise blocks: {len(sub)} < {n}')
        selected.append(sub.sort_values('hash_order').head(n).drop(columns='hash_order'))
    bank=pd.concat(selected,ignore_index=True);bank['noise_bank_index']=range(len(bank))
    for detector in ('h1_path','l1_path'):
        if bank.groupby(detector).split.nunique().max()!=1:raise RuntimeError('Raw file crosses noise splits')
    noise_path=root/'tables/NOISE_BLOCK_CANDIDATES.parquet';noise.to_parquet(noise_path,index=False)
    bank_path=root/'tables/NOISE_BANK_SELECTION.parquet';bank.to_parquet(bank_path,index=False)
    cfg={'code':'O4B-HL-NSO-01','created_utc':datetime.now(timezone.utc).isoformat(),
         'partition_rule':'deterministic SHA256, frozen before learning or test scoring',
         'model_seeds':[2026091221,2026091222,2026091223],
         'source_split_fixed_across_model_seeds':True,
         'source_counts':m.groupby(['family','split']).size().unstack().to_dict('index'),
         'source_ID_cross_split_intersections':0,'unique_global_sources':1800,
         'index_split':{s:[i for i in order if index_split[i]==s] for s in ('train','val','test')},
         'noise_counts':bank.groupby('split').size().to_dict(),
         'noise_parent_groups':noise.groupby('split').parent_raw_file_group.nunique().to_dict(),
         'known_event_exclusion_count':len(known),'known_event_exclusion_halfwidth_s':128,
         'noise_parent_raw_file_cross_split_intersections':0,
         'noise_reference_s':256,'input_noise_window_s':26,
         'noise_reuse_within_split_allowed':True,
         'bootstrap_note':'source/noise reuse induces dependence; retain parent IDs for cluster audit',
         'real_catalog_PE_parameter_values_used':False,'locked_test_scored':False,
         'source_timing_caveat':'physical source arrival times and empirical noise GPS are run-matched, not necessarily the same GPS',
         'invalid_selected_noise_policy':'fail gate; do not silently replace after scoring',
         'outputs':[{'path':str(p),'sha256':sha(p)} for p in [source_path,source_csv,noise_path,bank_path]],
         'inputs':[{'path':str(p),'sha256':sha(p)} for p in [detailed,root/'tables/PHYSICAL_SOURCE_METADATA.parquet']]}
    write(target,cfg)
    write(root/'contracts/DATA_SPLIT_FREEZE.json',{'path':str(target),'sha256':sha(target)})
    print(json.dumps(cfg,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    build(p.parse_args().root)
