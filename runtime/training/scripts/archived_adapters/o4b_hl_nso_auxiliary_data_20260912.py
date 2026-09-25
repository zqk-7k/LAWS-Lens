#!/usr/bin/env python3
"""Coverage-balanced O4b waveform-only development population.

Reuses the archived source generator and sampling helpers, not real PE labels.
Both 2s and 16s arrays come from the same physical injection and same noise.
"""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import logging
import multiprocessing as mp
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from scipy.signal import resample_poly

import o4b_hl_nso_data_training_20260912 as support

CTX={}
DATA_SEEDS={'train':2026091241,'validation':2026091242}
COUNTS={'train':4096,'validation':512}


def modules(root):
    ws=root/'workspace';sys.path.insert(0,str(ws))
    src=support.load(ws/'scripts/real_search/34_generate_physical_h1l1_source_bank.py','o4b_aux_source')
    helpers=Path('/root/autodl-tmp/gw-catalog/results/waveform_domain_multiscale_exploratory_20260903_20260903T063500Z/scripts/waveform_multiscale_data.py')
    data=support.load(helpers,'o4b_aux_sampling')
    from scripts.real_search import physical_common as phys
    return src,data,phys


def initialize(root):
    p=root/'contracts/AUXILIARY_POPULATION_CONTRACT.json'
    if p.exists():return
    src,data,phys=modules(root)
    support.write(p,{'UTC':support.now(),'purpose':'waveform-only coverage-balanced development for ordered Mc and multiscale conditional models',
        'data_seeds':DATA_SEEDS,'source_counts':COUNTS,'views_per_image':{'train':4,'validation':1},
        'mass_bins':data.MASS_BINS,'SNR_bins':data.SNR_BINS,
        'mass_prior':'same archived balanced_mass_draws: log-uniform within equal-count Mc strata, q uniform .25-1, component masses3-300Msun',
        'spins':'a1,a2 uniform0-.8; isotropic spin tilts, orientation and sky; phases uniform',
        'not_astrophysical_population_inference':True,
        'generation':'physical IMRPhenomXPHM24s4096Hz H1/L1; lens image amplitudes and Morse phases, run-matched times',
        'network_SNR':'same archived coverage-balanced per-image PSD-optimal scaling, not response-derived ratio validation',
        'source_isolation':'new waveform source UIDs; lens environments exclude all1800 main bank global sourceIDs; auxiliary train/validation environment IDs are also disjoint',
        'lens_environment_reuse':'within development partition permitted and explicitly counted;4096 waveform parents are not4096 independent lens environments',
        'noise':'first96 of frozen train blocks, all32 val blocks; no test noise; same selected parent for all views is allowed only within a partition',
        'short_input':[2,4096],'long_input':[2,4096],'long_rate_hz':256,'long_band_hz':[20,80],
        'no_rescaling_after_long_band_change':True,'no_full_strain_archive_required':'reconstruct from source plan and noise bank',
        'model_topology_reference':'same ordered-mass and multirate/conditional modules as PATH875, without outer score mixture',
        'training_population_size_caveat':'4096 development-training sources matches archived multirate branch; does not reproduce the earlier12288-source short ordered-mass training set',
        'real_PE_used':False,'locked_test_scored':False,
        'source_code':[{'path':str(p),'sha256':support.sha(p)} for p in [Path(src.__file__),Path(data.__file__),Path(phys.__file__)]]})
    support.write(root/'contracts/AUXILIARY_POPULATION_FREEZE.json',{'sha256':support.sha(p)})


def plan(root,split):
    initialize(root);out=root/f'auxiliary_data/{split}';out.mkdir(parents=True,exist_ok=True)
    path=out/'source_plan.parquet'
    if path.exists():
        if support.sha(path)!=json.loads((out/'PLAN_FREEZE.json').read_text())['sha256']:raise RuntimeError('Auxiliary plan changed')
        return pd.read_parquet(path)
    src,data,_=modules(root)
    table=src.load_tables(src.GW_LMC_ROOT)
    banned=set(pd.read_parquet(root/'tables/SOURCE_SPLIT_v2.parquet').gwlmc_event_id.astype(int))
    valid=table[~table.event_id.astype(int).isin(banned)].copy()
    ids=sorted(valid.event_id.astype(int).unique(),key=lambda i:hashlib.sha256(f'O4bAuxEnv:{i}'.encode()).hexdigest())
    selected=set(ids[:int(.75*len(ids))] if split=='train' else ids[int(.75*len(ids)):])
    environments={False:[],True:[]}
    for _,row in valid[valid.event_id.astype(int).isin(selected)].iterrows():
        images=src.choose_images(row)
        if images is not None and images['proposal_snr_ratio']<=4:environments[bool(row.lens_is_subhalo)].append((row,images))
    if min(map(len,environments.values()))<30:raise RuntimeError('Too few independent lens environments')
    shared=root/'workspace/results/real_noise_injection_v5_physical_source_20260721/gwtc5/shared'
    schedule=src.load_schedule(shared/'h1l1_live_schedule.csv')
    rng=np.random.default_rng(DATA_SEEDS[split]);rows=[]
    for index,(m1,m2,massbin) in enumerate(data.balanced_mass_draws(COUNTS[split],rng)):
        sub=bool(index%2);choices=environments[sub]
        for j in rng.permutation(len(choices)):
            row,images=choices[j];times=src.place_pair(images['delay_days'],schedule,rng)
            if times is not None:break
        else:raise RuntimeError('Unable to place lens environment in available O4b exposure')
        rows.append({'source_index':index,'source_uid':f'O4B_AUX_{DATA_SEEDS[split]}_{index:05d}','split':split,
            'family':'gwlmc_subhalo_present' if sub else 'gwlmc_smooth_non_subhalo',
            'gwlmc_environment_row':int(row.gwlmc_row),'gwlmc_environment_id':int(row.event_id),
            'm1_det':m1,'m2_det':m2,'mc_det':data.chirp_mass(m1,m2),'mass_bin':massbin,
            'a1':rng.uniform(0,.8),'a2':rng.uniform(0,.8),'tilt1':np.arccos(rng.uniform(-1,1)),
            'tilt2':np.arccos(rng.uniform(-1,1)),'theta_jn':np.arccos(rng.uniform(-1,1)),
            'phi12':rng.uniform(0,2*np.pi),'phijl':rng.uniform(0,2*np.pi),'psi':rng.uniform(0,np.pi),
            'phase':rng.uniform(0,2*np.pi),'ra':rng.uniform(0,2*np.pi),'dec':np.arcsin(rng.uniform(-1,1)),
            'dl_source':1000.,'gps_a':times[0],'gps_b':times[1],**images})
    frame=pd.DataFrame(rows);frame.to_parquet(path,index=False)
    support.write(out/'PLAN_FREEZE.json',{'sha256':support.sha(path),'waveform_source_parents':len(frame),
        'independent_lens_environments':frame.gwlmc_environment_id.nunique(),'main_bank_ID_intersection':0,
        'frozen_before_generation':True})
    return frame


def init_worker(root,split):
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=1)
    root=Path(root);src,data,phys=modules(root);logging.getLogger('bilby').setLevel(logging.ERROR)
    shared=root/'workspace/results/real_noise_injection_v5_physical_source_20260721/gwtc5/shared'
    noise=pd.read_parquet(shared/'noise_reference_metadata.parquet')
    banks=noise[noise.split==('val' if split=='validation' else 'train')].noise_bank_index.to_numpy(int)
    if split=='train':banks=banks[:96]
    CTX.update(root=root,split=split,src=src,data=data,phys=phys,generator=src.build_waveform_generator(),
        ifos=[src.bilby.gw.detector.get_empty_interferometer(d) for d in ['H1','L1']],
        refs=np.load(shared/'noise_reference_bank.npy',mmap_mode='r'),psds=np.load(shared/'noise_psd_bank.npy',mmap_mode='r'),
        freq=np.load(shared/'noise_psd_frequency.npy'),banks=banks)


def system(row):
    src,data,p=CTX['src'],CTX['data'],CTX['phys'];split=CTX['split']
    rng=np.random.default_rng(data.stable_seed(row['source_uid'],'physical-noise-views'))
    waves=[];longs=[];records=[];views=4 if split=='train' else 1
    for image,number in [('a',1),('b',2)]:
        clean,peaks=src.detector_response(CTX['generator'],CTX['ifos'],
            src.source_parameters(pd.Series(row),row[f'gps_{image}']),src.lens_factor(row[f'mu_image{number}'],row[f'morse_image{number}']))
        if not np.isfinite(clean).all():raise RuntimeError('Nonfinite auxiliary physical strain')
        for variant in range(views):
            b=int(rng.choice(CTX['banks']));ref=CTX['refs'][b];offset=int(rng.integers(ref.shape[-1]-p.RAW_PADDED_SAMPLES+1))
            nw=np.asarray(ref[:,offset:offset+p.RAW_PADDED_SAMPLES],np.float32)
            target=data.snr_draw((row['source_index']+number+variant)%4,rng)
            scaled,factor,recovered=p.scale_to_network_snr(np.asarray(clean,np.float32),target,CTX['freq'],CTX['psds'][b])
            mixed=nw+p.embed_signal_in_padded_window(scaled)
            full=p.preprocess_24s(mixed,CTX['freq'],CTX['psds'][b])
            short=full[:,-4096:]
            low=resample_poly(p.preprocess_24s(mixed,CTX['freq'],CTX['psds'][b],band_low_hz=20.,band_high_hz=80.),1,8,
                              axis=-1,window=('kaiser',8.6))[:,-4096:]
            if not np.isfinite(short).all() or not np.isfinite(low).all() or abs(recovered-target)>1e-4*target:
                raise RuntimeError('Auxiliary preprocessing/SNR check failed')
            waves.append(short.astype(np.float16));longs.append(low.astype(np.float32))
            records.append({**row,**peaks,'image':image,'noise_variant':variant,'noise_bank_index':b,'noise_offset_samples':offset,
                            'target_network_snr':target,'recovered_optimal_network_snr':recovered,'physical_strain_scale_factor':factor})
    return row['source_index'],np.stack(waves),np.stack(longs),records


def generate(root,split,workers):
    frame=plan(root,split);shared=root/'workspace/results/real_noise_injection_v5_physical_source_20260721/gwtc5/shared'
    if not (shared/'noise_bank_complete.json').exists():raise RuntimeError('Noise bank incomplete')
    out=root/f'auxiliary_data/{split}';marker=out/'COMPLETE.json'
    if marker.exists():print(marker.read_text(),flush=True);return
    nview=8 if split=='train' else 2;shape=(len(frame)*nview,2,4096)
    path=out/'raw2s.npy';longpath=out/'low16s.npy'
    raw=np.lib.format.open_memmap(path,mode='r+' if path.exists() else 'w+',dtype=np.float16,shape=shape)
    low=np.lib.format.open_memmap(longpath,mode='r+' if longpath.exists() else 'w+',dtype=np.float32,shape=shape)
    chunks=out/'chunks';chunks.mkdir(exist_ok=True);done=set()
    for f in chunks.glob('*.parquet'):done.update(pd.read_parquet(f,columns=['source_index']).source_index.unique())
    pending=frame[~frame.source_index.isin(done)].to_dict('records');tick=time.monotonic()
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn'),initializer=init_worker,initargs=(str(root),split)) as pool:
        for n,(idx,a,b,records) in enumerate(pool.map(system,pending,chunksize=1),1):
            support.disk_guard(root);raw[idx*nview:(idx+1)*nview]=a;low[idx*nview:(idx+1)*nview]=b
            raw.flush();low.flush()
            for k,row in enumerate(records):row['row_index']=idx*nview+k
            pd.DataFrame(records).to_parquet(chunks/f'{idx:05d}.parquet',index=False)
            if n%64==0:print(json.dumps({'auxiliary_data':split,'sources_complete':len(done)+n,'total':len(frame),'seconds':time.monotonic()-tick}),flush=True)
    meta=pd.concat([pd.read_parquet(f) for f in chunks.glob('*.parquet')]).sort_values('row_index')
    if not np.array_equal(meta.row_index,np.arange(len(raw))):raise RuntimeError('Incomplete auxiliary event table')
    meta.to_parquet(out/'event_metadata.parquet',index=False)
    support.write(marker,{'state':'PASS','waveform_parents':len(frame),'events':len(raw),'seconds':time.monotonic()-tick,
        'raw2s_sha256':support.sha(path),'low16s_sha256':support.sha(longpath),
        'independent_lens_environments':frame.gwlmc_environment_id.nunique(),'no_PE_or_locked_test':True})
    print(marker.read_text(),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--stage',choices=['plan','generate'],required=True)
    p.add_argument('--split',choices=['train','validation'],required=True);p.add_argument('--workers',type=int,default=8);a=p.parse_args()
    if a.stage=='plan':
        f=plan(a.root,a.split);print(json.dumps({'split':a.split,'sources':len(f),'lens_environments':f.gwlmc_environment_id.nunique()}))
    else:generate(a.root,a.split,a.workers)
