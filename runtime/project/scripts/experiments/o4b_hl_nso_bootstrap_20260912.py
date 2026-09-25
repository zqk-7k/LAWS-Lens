#!/usr/bin/env python3
"""Read-only upstream snapshot and bounded, checksummed O4b input restoration.

This entry point never claims that input restoration is a completed experiment.
It does not start training, read PE parameter values, or rank real candidates.
"""
from __future__ import annotations
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import traceback

import h5py
import numpy as np
import pandas as pd
import requests

PROJECT = Path('/root/autodl-tmp/gw-catalog')
OLD = PROJECT / 'results/gwtc5_o4b_scheme_c_exact_20260831_20260831T093631Z'
BASE = OLD / 'incoming/baseline/gwtc5_o4b_v93_extension_20260830'
PATH875 = PROJECT / 'results/mcwf_unified_path875_devconf_20260908T181500Z'
BAYESTAR = PROJECT / 'results/bayestar_injection_sky_pe_20260901_20260901T102000Z'
HOLD = 'HOLD_FOR_AUTHOR_REVIEW_NO_ADOPTION_NO_OVERWRITE'
LOCK = threading.Lock()


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest(path, algorithm='sha256'):
    h = hashlib.new(algorithm)
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(data)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    temp.replace(path)


def disk_guard(root):
    free = shutil.disk_usage(root).free
    if free < 40*1024**3:
        raise RuntimeError('HOLD_DISK_LIMIT: less than 40 GiB free')
    return free


def initialize():
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    root = PROJECT / 'results' / f'o4b_hl_bayestar_new_score_only_{stamp}'
    root.mkdir(exist_ok=False)
    for name in ('contracts','manifests','scripts','logs','reports','tables','workspace',
                 'models','event_maps','results/validation','results/locked_test','results/real'):
        (root/name).mkdir(parents=True,exist_ok=True)
    workspace = root/'workspace'
    for sub in ('scripts','matchgw'):
        source = BASE/'scripts'/sub if sub=='matchgw' else BASE/sub
        shutil.copytree(source,workspace/sub,ignore=shutil.ignore_patterns('__pycache__','*.orig'))
    run = workspace/'runs/real_gwtc5_o4b_v93_extension_20260830'
    (run/'data/real_strain').mkdir(parents=True)
    (run/'shared').mkdir(parents=True)
    (workspace/'external/pe_posteriors').mkdir(parents=True)
    for name in ('event_manifest.csv','strain_gwosc_download_manifest.csv'):
        shutil.copy2(BASE/'input_contract'/name,run/'data'/name)
    shutil.copy2(OLD/'contracts/h1l1_live_schedule_reconstructed.csv',run/'shared/h1l1_live_schedule.csv')
    for sub in ('input_contract','stage0'):
        shutil.copytree(BASE/sub,root/'contracts'/'historical'/sub)
    maps = pd.read_csv(OLD/'contracts/public_pe_template_library_nside512.csv')
    manifest = pd.read_csv(run/'data/event_manifest.csv')
    map_by_event = maps.set_index('event_name').map_path.to_dict()
    manifest['sky_map_path'] = manifest.event_name.map(map_by_event)
    if len(manifest)!=86 or manifest.sky_map_path.isna().any():
        raise RuntimeError('Frozen real scope/map lookup mismatch')
    for path in manifest.sky_map_path:
        if not Path(path).is_file():raise FileNotFoundError(path)
    manifest.to_csv(run/'data/event_manifest.csv',index=False)
    protected = []
    for directory in (OLD,PATH875,BAYESTAR):
        for name in ('contracts','reports','results','tables'):
            path = directory/name
            if path.exists():
                for f in path.rglob('*'):
                    if f.is_file() and f.suffix in ('.json','.csv','.parquet','.md'):
                        protected.append({'path':str(f),'sha256':digest(f)})
    write_json(root/'manifests/PROTECTED_UPSTREAM.json',protected)
    contract = {
        'code':'O4B-HL-NSO-01','created_utc':utc(),'root':str(root),'workspace':str(workspace),
        'protocol':'O4b run-matched H1/L1 training and NEW-SCORE-ONLY extension',
        'final_status_required':HOLD,'historical_adoption_authorized':False,
        'real_scope':{'events':86,'unordered_pairs':3655,'strain_detectors':['H1','L1'],
                      'public_PE_networks':{'H1+L1+V1':75,'H1+L1':11}},
        'model_seeds':[2026091221,2026091222,2026091223],
        'simulation_catalog_seeds':[2026091231,2026091232,2026091233],
        'waveform':{'framework_reference':str(PATH875),'input_short_seconds':2,
                    'short_points':4096,'input_long_seconds':16,'long_band_hz':[20,80],
                    'long_sample_rate_hz':256,'long_points':4096,
                    'train_on':'O4b off-source empirical H1/L1 noise',
                    'retain_internal_OMC_components':True,
                    'new_score':'Z_wf_OMC + gamma*T + beta*I',
                    'outer_old_new_total_score_mixture':False,'alpha_equivalent':1.0,
                    'not_NODUP_DIRECT':True,'V1_strain_in_encoder':False},
        'sky':{'reference':str(BAYESTAR),'analysis_nside':512,
               'injection':'conditional BAYESTAR; known intrinsic template, empirical PSD, Gaussian matched-filter trigger realization',
               'not_rotated_PE_templates':True,'not_full_strain_PE':True,
               'real':'frozen public PE MOC/FITS at Nside512, correct ordering',
               'network_caveat':'HL-only injection localization versus75 HLV real PE maps is not network-homogeneous; report separately, do not claim otherwise',
               'formula':'log(Npix * sum(P_i*P_j))','dense_persistence':False},
        'time':{'model':'one-dimensional run-exposure-conditioned delay likelihood ratio',
                'run':'O4b','schedule':str(run/'shared/h1l1_live_schedule.csv'),
                'no_250_day_synthetic_calendar':True,'no_SNR_ratio_score':True},
        'selection':{'real_PE_used':False,'official_overlap_used':False,
                     'per_seed_validation_only':True,'simplex_step':0.05,
                     'weight_sum':1,'zero_weights_allowed':True,
                     'ties':'normalized frozen baseline distance then deterministic lexicographic order',
                     'locked_test_once_after_final_config_hash':True},
        'isolation':{'source_group':'global physical source IDs, not legacy family-local row',
                     'noise':'nonoverlapping GPS intervals and parent256s PSD blocks',
                     'train_validation_test_disjoint':True,'no_silent_zero_fill':True},
        'metrics':['R@1','R@5','R@10','R@50','pair AUPRC','F50','F90',
                   'Top10/20/50/100/200/500 precision and false pairs',
                   'per-seed mean SD and source-system bootstrap95CI',
                   'realTop10/20/50 PE BC/Dmax and available official-stage crossmatch'],
        'no_Hanabi_run':True,'official_overlap_not_truth':True,
        'full_experiment_complete':False,'input_restore_only_at_launch':True,
        'stages':['INPUT_RESTORE','INPUT_WINDOW_AND_SCOPE_AUDIT','SOURCE_NOISE_SPLIT',
                  'BASE_ENCODER_TRAIN','RAW_PHASE_ORDERED_MASS_TRAIN','MULTISCALE_CONDITIONAL_TRAIN',
                  'BAYESTAR_VALIDATION_AND_FROZEN_CALIBRATION','LOCKED_TEST','REAL_AUDIT','PACKAGE_HOLD'],
        'resources':{'min_free_GiB':40,'restore_workers':4,'retain_original_inputs':True},
    }
    write_json(root/'contracts/ANALYSIS_CONTRACT.json',contract)
    write_json(root/'contracts/FREEZE.json',{'sha256':digest(root/'contracts/ANALYSIS_CONTRACT.json')})
    shutil.copy2(__file__,root/'scripts'/Path(__file__).name)
    jobs = []
    strain = pd.read_csv(BASE/'input_contract/strain_download_byte_audit.csv')
    for row in strain.itertuples(index=False):
        jobs.append({'kind':'strain','url':row.url,'path':str(run/'data/real_strain'/Path(row.path).name),
                     'bytes':int(row.size_bytes),'checksum_algorithm':'sha256','checksum':row.sha256})
    pe = pd.read_csv(BASE/'stage0/gwtc5_o4b_frozen_manifest.csv')
    for row in pe.itertuples(index=False):
        algorithm,value=row.pe_checksum.split(':',1)
        jobs.append({'kind':'PE_HDF5','event_name':row.event_name,'url':row.pe_download_url,
                     'path':str(workspace/'external/pe_posteriors'/row.pe_filename),
                     'bytes':int(row.pe_size_bytes),'checksum_algorithm':algorithm,'checksum':value})
    if len(jobs)!=259:raise RuntimeError(f'Unexpected manifest size:{len(jobs)}')
    write_json(root/'manifests/DOWNLOAD_JOBS.json',jobs)
    write_json(root/'STATUS.json',{'state':'INPUT_RESTORE_READY','utc':utc(),'total_files':len(jobs),
                                 'expected_bytes':sum(x['bytes'] for x in jobs),'training_started':False})
    print(json.dumps({'root':str(root),'files':len(jobs),'bytes':sum(x['bytes'] for x in jobs)}),flush=True)
    return root


def transfer(root,job):
    target=Path(job['path']);partial=target.with_suffix(target.suffix+'.part')
    audit_path=root/'manifests/downloads'/f'{target.name}.json'
    if target.exists():
        if target.stat().st_size==job['bytes'] and digest(target,job['checksum_algorithm'])==job['checksum']:
            return {'kind':job['kind'],'path':str(target),'state':'VERIFIED_EXISTING','bytes':job['bytes']}
        raise RuntimeError(f'Existing completed file fails expected checksum: {target}')
    begin=time.monotonic();errors=[]
    for attempt in range(1,4):
        try:
            disk_guard(root)
            offset=partial.stat().st_size if partial.exists() else 0
            if offset>job['bytes']:raise RuntimeError('Partial file larger than contract')
            if offset<job['bytes']:
                headers={'User-Agent':'O4B-HL-NSO-01-input-restore','Accept-Encoding':'identity'}
                if offset:headers['Range']=f'bytes={offset}-'
                with requests.get(job['url'],headers=headers,stream=True,timeout=(20,120)) as r:
                    r.raise_for_status()
                    if offset and (r.status_code!=206 or not r.headers.get('Content-Range','').startswith(f'bytes {offset}-')):
                        raise RuntimeError('Server did not honor exact resume offset')
                    mode='ab' if offset else 'wb'
                    last_guard=time.monotonic()
                    with partial.open(mode) as f:
                        for chunk in r.iter_content(4*1024*1024):
                            if chunk:f.write(chunk)
                            if time.monotonic()-last_guard>10:
                                disk_guard(root);last_guard=time.monotonic()
                        f.flush();os.fsync(f.fileno())
            if partial.stat().st_size!=job['bytes']:raise RuntimeError('Incomplete or oversized download')
            got=digest(partial,job['checksum_algorithm'])
            if got!=job['checksum']:raise RuntimeError('Downloaded checksum differs from frozen upstream')
            with h5py.File(partial,'r') as f:
                if job['kind']=='strain':
                    d=f['strain/Strain']
                    if d.shape!=(4096*4096,) or float(d.attrs['Xspacing'])!=1/4096:
                        raise RuntimeError('Unexpected strain size/rate')
                    structure={'samples':d.shape[0],'sample_rate':4096}
                else:structure={'top_level_groups':list(f.keys())}
            partial.replace(target)
            result={**job,'state':'VERIFIED','sha256':digest(target),'wall_seconds':time.monotonic()-begin,
                    'attempts':attempt,'structure':structure,'utc':utc()}
            write_json(audit_path,result)
            return result
        except Exception as e:
            errors.append(str(e))
            if 'checksum' in str(e).lower() or 'DISK_LIMIT' in str(e):break
            if attempt<3:time.sleep(5*attempt)
    result={**job,'state':'FAILED','errors':errors,'wall_seconds':time.monotonic()-begin,'utc':utc()}
    write_json(audit_path,result)
    return result


def restore(root,kind='all',workers=4):
    jobs=json.loads((root/'manifests/DOWNLOAD_JOBS.json').read_text())
    if kind!='all':jobs=[j for j in jobs if j['kind']==kind]
    write_json(root/'STATUS.json',{'state':'RESTORING_INPUTS','utc':utc(),'kind':kind,'files':len(jobs),
                                 'pid':os.getpid(),'training_started':False})
    results=[];started=time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending={pool.submit(transfer,root,j):j for j in jobs}
        for future in as_completed(pending):
            row=future.result();results.append(row)
            failed=sum(x['state']=='FAILED' for x in results)
            status={'state':'RESTORING_INPUTS','utc':utc(),'kind':kind,'completed':len(results),
                    'total':len(jobs),'failed':failed,'elapsed_seconds':time.monotonic()-started,
                    'free_GiB':shutil.disk_usage(root).free/2**30,'training_started':False}
            write_json(root/'STATUS.json',status)
            print(json.dumps({**status,'last':Path(row['path']).name,'last_state':row['state']}),flush=True)
    write_json(root/f'manifests/RESTORE_{kind}.json',results)
    failed=[r for r in results if r['state']=='FAILED']
    write_json(root/'STATUS.json',{'state':'HOLD_INPUT_RESTORE_FAILED' if failed else 'INPUT_RESTORE_COMPLETE',
                                 'utc':utc(),'kind':kind,'verified':len(results)-len(failed),'failed':len(failed),
                                 'training_started':False,'full_experiment_complete':False})
    if failed:raise RuntimeError(f'{len(failed)} input transfers failed; no training authorized')


def verify_windows(root):
    run=root/'workspace/runs/real_gwtc5_o4b_v93_extension_20260830'
    rows=[]
    manifest=pd.read_csv(run/'data/strain_gwosc_download_manifest.csv')
    for row in manifest.itertuples(index=False):
        for kind,path,start,duration in (
            ('onsource',row.local_path,row.onsource_start_gps,row.onsource_end_gps-row.onsource_start_gps),
            ('PSD',row.psd_local_path,row.psd_reference_start_gps,row.psd_reference_duration_s)):
            path=run/path
            with h5py.File(path,'r') as f:
                d=f['strain/Strain'];t0=float(d.attrs['Xstart']);dt=float(d.attrs['Xspacing'])
                a=int(round((start-t0)/dt));n=int(round(duration/dt))
                v=np.asarray(d[a:a+n]) if a>=0 else np.array([])
            valid=len(v)==n and bool(np.isfinite(v).all())
            rows.append({'event':row.event_name,'detector':row.detector,'window':kind,
                         'path':str(path),'start':start,'samples':len(v),'expected':n,'passed':valid})
    pd.DataFrame(rows).to_csv(root/'tables/INPUT_WINDOW_AUDIT.csv',index=False,encoding='utf-8-sig')
    passed=all(r['passed'] for r in rows) and len(rows)==344
    changes=[r['path'] for r in json.loads((root/'manifests/PROTECTED_UPSTREAM.json').read_text())
             if digest(r['path'])!=r['sha256']]
    write_json(root/'contracts/INPUT_GATE.json',{'passed':passed and not changes,'windows':len(rows),
               'failed_windows':sum(not r['passed'] for r in rows),'historical_changes':changes,'utc':utc()})
    if not passed or changes:raise RuntimeError('Input window/history gate failed')
    print(json.dumps({'input_gate_pass':True,'windows':len(rows),'historical_changes':changes}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path)
    parser.add_argument('--stage',choices=['initialize','restore','windows'],required=True)
    parser.add_argument('--kind',choices=['all','strain','PE_HDF5'],default='all')
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    if args.stage=='initialize':initialize()
    elif args.stage=='restore':restore(args.root,args.kind,args.workers)
    else:verify_windows(args.root)
