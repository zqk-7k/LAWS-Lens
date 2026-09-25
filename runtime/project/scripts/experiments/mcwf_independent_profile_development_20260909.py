#!/usr/bin/env python3
"""New waveform-only calibration systems and independently reserved noise."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import logging
import multiprocessing as mp
from pathlib import Path
import shutil
import sys
import time
import numpy as np
import pandas as pd

P=Path('/root/autodl-tmp/gw-catalog')
sys.path.insert(0,str(P/'scripts/experiments'))
import mcwf_expanded_data_20260907 as expanded
import mcwf_multirate_features_20260908 as mult
import mcwf_nodup_reliability_20260909 as r
n=r.n
ROOT=None
COUNT=1024
BLOCKS=64
SEED=2026090922
STATE={}


def freeze():
    if ROOT.exists():
        raise RuntimeError('Independent output required')
    for folder in ('contracts','data','scripts','logs','tables','audit','manifest','reports','cache','features','predictions'):
        (ROOT/folder).mkdir(parents=True)
    n.write_json(ROOT/'contracts/ANALYSIS_CONTRACT.json',{'id':'MCWF-NODUP-INDEPENDENT-PROFILE-DEVELOPMENT-22',
        'UTC':n.utc(),'status':n.STATUS,'goal_achieved':False,'adaptive_development':True,'same_both_runs':True,
        'reason':'Round21b O4a had only17 accepted tune sources. Expand physical calibration support instead of relaxing20-source minimum or duplicating existing systems.',
        'sources_per_run':COUNT,'images_per_source':2,'noise_blocks_per_run':BLOCKS,'noise_block_seconds':256,
        'source_split':'512fit and512tune,assigned beforegeneration. Both images ofeachsource sharefold;different noise blocks withinfold.',
        'noise_split':'32new256sblocksfit and32new256sblockstune. ExcludeallrecordedpreviousMCWFtrain/development/confirmationGPS intervals with16sguards.',
        'source_model':'Same coverage-balanced Mc5-200,q.25-1,componentmasses3-300,spin/orientationdraws and IMRPhenomXPHM H1/L1 simulator asexistingdevelopment. Newsourceparameterdraws andUIDs.',
        'lens_model_limitation':'GW-LMC lens environments mayrepeat;new independentwaveformparents areNOTnew independentlenspopulations.',
        'SNR':'Samebalanced8-10,10-12,12-20,20-40bins;per-imagePSDoptimal scaling. No claimofnewresponse-derivedtime/SNR-ratio test.',
        'stored':'peak2s4096float16 and20-80Hz16s256Hzfloat32;metadata,offsourcePSD/reference andCWTfeatures. No newsky maps/fullPE/Hanabi.',
        'waveform_conditioning':'Samepublishedphysicalpipeline,20Hzauxiliaryviews generatedfromraw26s,not inversionof40Hz filteredwaveform.',
        'neural_networks':'Allencoder/newdensitycheckpointsfrozen;thisstageonlygeneratesadditionalcalibrationdata.',
        'downstream':'Re-evaluate frozen R10/R18/R21 predictionrules onnew fit/tune. No real/eventIDinput,no test-driven thresholds. Any newselectedmethod stillrequiresbothrunPE/officialandfullrankingchecks.',
        'statistics':'Reportbothsourceandnoise-blockcounts andbootstrap. Repeatednoiseoffsets arenotindependent256sblocks.',
        'frozen':['historicaltime','historicalsky','outerweights','oldrankings','scope','paper','ET3'],
        'no_total_blend':True,'no_oldencoderMc_q':True,
        'seed':SEED,'disk_minimum_free_GiB':25})
    n.write_csv(ROOT/'manifest/INPUT_SHA256.csv',pd.read_csv(r.PRIOR/'manifest/INPUT_SHA256.csv'))
    shutil.copy2(__file__,ROOT/'scripts/independent_profile_development.py')
    n.write_json(ROOT/'contracts/START_FREEZE.json',{'UTC':n.utc(),'script_sha256':n.sha(Path(__file__)),
        'contract_sha256':n.sha(ROOT/'contracts/ANALYSIS_CONTRACT.json')})


def exclusions(dep):
    rows=expanded.previous.exclusions(dep).to_dict('records')
    files=[]
    for root in sorted((P/'results').glob('mcwf_*')):
        for f in root.rglob('noise_manifest.csv'):
            if dep not in f.parts or ROOT==root:
                continue
            a=pd.read_csv(f)
            if not {'start_gps','end_gps'}<=set(a.columns):
                raise RuntimeError('Unrecognized historical noise manifest:'+str(f))
            files.append({'path':str(f),'sha256':n.sha(f)})
            rows.extend({'start':float(row.start_gps),'end':float(row.end_gps),'origin':str(f)} for row in a.itertuples())
    out=pd.DataFrame(rows).drop_duplicates(['start','end'])
    n.write_csv(ROOT/f'audit/{dep}_EXCLUDED_NOISE.csv',out)
    n.write_csv(ROOT/f'manifest/{dep}_NOISE_SOURCE_MANIFESTS.csv',files)
    return out


def noise(dep):
    folder=ROOT/f'data/{dep}/noise'
    folder.mkdir(parents=True,exist_ok=True)
    if (folder/'COMPLETE.json').exists():
        return
    _,data,v3=expanded.modules()
    historic=exclusions(dep)
    excluded=[(float(row.start)-16,float(row.end)+16) for row in historic.itertuples()]
    if dep=='gwtc3':
        source_run=n.dev.MAIN/'cache/source_run'
        pool=pd.read_csv(expanded.previous.PREVIOUS/'data/o3_only_noise/official_O3_full_strain_offsource_pool.csv')
    else:
        source_run=data.SOURCE_RUNS[dep]
        pool=pd.read_parquet(source_run/'data/real_noise_injections/offsource_noise_segments.parquet')
    path=folder/'reference.npy'
    progress=folder/'noise_manifest.csv'
    rows=pd.read_csv(progress).to_dict('records') if progress.exists() else []
    refs=np.lib.format.open_memmap(path,mode='r+' if path.exists() else 'w+',dtype=np.float32,
                                  shape=(BLOCKS,2,v3.NOISE_REFERENCE_SAMPLES))
    psds=[]
    for row in rows:
        excluded.append((row['start_gps']-16,row['end_gps']+16))
        psds.append(np.load(folder/f'psd_{row["bank_index"]:03d}.npy'))
    cache=v3.HdfCache(source_run,max_files=8)
    for k in range(len(rows),BLOCKS):
        parts=[]
        for row in pool.to_dict('records'):
            for aa,bb in expanded.rm.subtract_intervals(float(row['segment_start']),float(row['segment_end']),excluded):
                if bb-aa>=v3.PSD_SECONDS+2:
                    parts.append({**row,'segment_start':aa,'segment_end':bb})
        if not parts:
            raise RuntimeError('HOLD_NO_INDEPENDENT_NOISE_REMAINING')
        chosen,rejected=data.select_valid_noise_references(pd.DataFrame(parts),1,
            np.random.default_rng(data.stable_seed(dep,'nodup-profile-newnoise',SEED,k)),v3.PSD_SECONDS,cache,v3)
        row,start,ref,freq,psd=chosen[0]
        if any(start<bb and start+v3.PSD_SECONDS>aa for aa,bb in excluded):
            raise RuntimeError('Global noise overlap')
        refs[k]=ref;refs.flush()
        np.save(folder/f'psd_{k:03d}.npy',psd)
        np.save(folder/'frequency.npy',freq)
        psds.append(psd)
        excluded.append((start-16,start+v3.PSD_SECONDS+16))
        rows.append({'bank_index':k,'fold':int(k>=BLOCKS//2),'parent_event':str(row.event_name),
            'start_gps':start,'end_gps':start+v3.PSD_SECONDS,'H1_path':row.H1_path,'L1_path':row.L1_path,'rejections':len(rejected)})
        n.write_csv(progress,rows)
        if k%8==0 or k==BLOCKS-1:
            print('NEW_DEVELOPMENT_NOISE',dep,k+1,BLOCKS,flush=True)
    np.save(folder/'psd.npy',np.stack(psds))
    n.write_json(folder/'COMPLETE.json',{'UTC':n.utc(),'independent_noise_blocks':len(rows),
        'global_GPS_overlap':0,'fit_tune_noise_overlap':0,'excluded_historical_blocks':len(historic),
        'reference_sha256':n.sha(path),'psd_sha256':n.sha(folder/'psd.npy')})


def plan(dep):
    path=ROOT/f'data/{dep}/source_plan.parquet'
    if path.exists():
        return pd.read_parquet(path)
    src,data,_=expanded.modules()
    table=src.load_tables(src.GW_LMC_ROOT)
    schedule=src.load_schedule(n.dev.ORCH.SOURCE_ROOT/dep/'shared/h1l1_live_schedule.csv')
    rng=np.random.default_rng(data.stable_seed(SEED,dep,'independent-profile-sources'))
    environments={False:[],True:[]}
    for _,row in table.iterrows():
        images=src.choose_images(row)
        if images is not None and images['proposal_snr_ratio']<=4:
            environments[bool(row.lens_is_subhalo)].append((row,images))
    rows=[]
    for k,(m1,m2,massbin) in enumerate(data.balanced_mass_draws(COUNT,rng)):
        sub=bool((k//2)%2)
        for j in rng.permutation(len(environments[sub])):
            row,images=environments[sub][j]
            times=src.place_pair(images['delay_days'],schedule,rng)
            if times is not None:
                break
        else:
            raise RuntimeError('No valid lens environment')
        # Alternation balances each mass/SNR stratum across the frozen folds.
        fold=k%2
        rows.append({'source_index':k,'source_uid':f'NODUP22_{dep}_{SEED}_{k:05d}',
            'fold':fold,'split':'fit' if fold==0 else 'tune','family':'gwlmc_subhalo_present' if bool(row.lens_is_subhalo) else 'gwlmc_smooth_non_subhalo',
            'gwlmc_environment_row':int(row.gwlmc_row),'gwlmc_environment_id':int(row.event_id),
            'm1_det':m1,'m2_det':m2,'mc_det':data.chirp_mass(m1,m2),'mass_bin':massbin,
            'a1':rng.uniform(0,.8),'a2':rng.uniform(0,.8),'tilt1':np.arccos(rng.uniform(-1,1)),
            'tilt2':np.arccos(rng.uniform(-1,1)),'theta_jn':np.arccos(rng.uniform(-1,1)),
            'phi12':rng.uniform(0,2*np.pi),'phijl':rng.uniform(0,2*np.pi),'psi':rng.uniform(0,np.pi),
            'phase':rng.uniform(0,2*np.pi),'ra':rng.uniform(0,2*np.pi),'dec':np.arcsin(rng.uniform(-1,1)),
            'dl_source':1000.,'gps_a':times[0],'gps_b':times[1],**images})
    frame=pd.DataFrame(rows)
    # Source IDs and source parameters must be new, not duplicated records.
    if frame.source_uid.duplicated().any() or frame[['m1_det','m2_det','a1','a2']].duplicated().any():
        raise RuntimeError('Duplicate newly generated source')
    frame.to_parquet(path,index=False)
    n.write_json(path.with_suffix('.json'),{'UTC':n.utc(),'sha256':n.sha(path),'sources':len(frame),
        'fold_counts':frame.groupby('fold').size().to_dict(),'lens_environment_count':frame.gwlmc_environment_row.nunique(),
        'not_independent_lens_population':True})
    return frame


def worker_init(dep):
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)
    src,data,v3=expanded.modules()
    logging.getLogger('bilby').setLevel(logging.ERROR)
    folder=STATE['root']/f'data/{dep}/noise'
    STATE.update(src=src,data=data,v3=v3,dep=dep,generator=src.build_waveform_generator(),
        ifos=[src.bilby.gw.detector.get_empty_interferometer(x) for x in ('H1','L1')],
        refs=np.load(folder/'reference.npy',mmap_mode='r'),freq=np.load(folder/'frequency.npy'),
        psds=np.load(folder/'psd.npy',mmap_mode='r'))


def init(root,dep):
    STATE['root']=Path(root)
    worker_init(dep)


def generate_system(row):
    start=time.perf_counter()
    src,data,v3=STATE['src'],STATE['data'],STATE['v3']
    rng=np.random.default_rng(data.stable_seed(row['source_uid'],'independent-noise-view'))
    banks=row['fold']*(BLOCKS//2)+rng.choice(BLOCKS//2,size=2,replace=False)
    peak,low,records=[],[],[]
    for offset_image,(image,number) in enumerate((('a',1),('b',2))):
        clean,peaks=src.detector_response(STATE['generator'],STATE['ifos'],
            src.source_parameters(pd.Series(row),row[f'gps_{image}']),src.lens_factor(row[f'mu_image{number}'],row[f'morse_image{number}']))
        bi=int(banks[offset_image])
        offset=int(rng.integers(STATE['refs'].shape[-1]-v3.RAW_PADDED_SAMPLES+1))
        noise_window=np.asarray(STATE['refs'][bi,:,offset:offset+v3.RAW_PADDED_SAMPLES],np.float32)
        snr=data.snr_draw((row['source_index']//2+number)%4,rng)
        scaled,scale,recovered=v3.scale_to_network_snr(np.asarray(clean,np.float32),snr,STATE['freq'],STATE['psds'][bi])
        raw=noise_window+v3.embed_signal_in_padded_window(scaled)
        mixed=v3.preprocess_24s(raw,STATE['freq'],STATE['psds'][bi])
        view=n.dev.TRAIN.make_window_view(mixed[None].astype(np.float32),2)[0]
        auxiliary=mult.low_view(raw,STATE['freq'],STATE['psds'][bi],v3)
        if not np.isfinite(view).all() or not np.isfinite(auxiliary).all():
            raise RuntimeError('Invalid new waveform')
        if abs(recovered-snr)>1e-4*snr:
            raise RuntimeError('New PSD-optimal SNR mismatch')
        peak.append(view.astype(np.float16));low.append(auxiliary)
        records.append({**row,'image':image,'noise_variant':0,'noise_bank_index':bi,'noise_offset_samples':offset,
            'target_network_snr':snr,'physical_strain_scale_factor':scale,'recovered_optimal_network_snr':recovered,
            'row_index':2*row['source_index']+offset_image,**peaks})
    return row['source_index'],np.stack(peak),np.stack(low),records,time.perf_counter()-start


def generate(dep,workers):
    noise(dep)
    sources=plan(dep)
    folder=ROOT/f'data/{dep}'
    if (folder/'COMPLETE.json').exists():
        return
    chunks=folder/'chunks';chunks.mkdir(exist_ok=True)
    peak_path,low_path=folder/'raw2s.npy',folder/'low16s.npy'
    peak=np.lib.format.open_memmap(peak_path,mode='r+' if peak_path.exists() else 'w+',dtype=np.float16,shape=(COUNT*2,2,4096))
    low=np.lib.format.open_memmap(low_path,mode='r+' if low_path.exists() else 'w+',dtype=np.float32,shape=(COUNT*2,2,4096))
    done={int(p.stem) for p in chunks.glob('*.parquet')}
    jobs=sources[~sources.source_index.isin(done)].to_dict('records')
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn'),initializer=init,initargs=(str(ROOT),dep)) as pool:
        for k,(idx,a,b,records,seconds) in enumerate(pool.map(generate_system,jobs),1):
            peak[idx*2:idx*2+2]=a;low[idx*2:idx*2+2]=b
            peak.flush();low.flush()
            for row in records:
                row['generation_seconds']=seconds
            pd.DataFrame(records).to_parquet(chunks/f'{idx:05d}.parquet',index=False)
            if k%64==0 or k==len(jobs):
                print('NEW_PROFILE_SOURCES',dep,len(done)+k,COUNT,flush=True)
                if shutil.disk_usage(ROOT).free<25*2**30:
                    raise RuntimeError('HOLD_DISK_LIMIT')
    meta=pd.concat([pd.read_parquet(p) for p in sorted(chunks.glob('*.parquet'))]).sort_values('row_index')
    if not np.array_equal(meta.row_index.to_numpy(),np.arange(COUNT*2)):
        raise RuntimeError('Metadata completeness failure')
    if set(meta.noise_bank_index[meta.fold==0])&set(meta.noise_bank_index[meta.fold==1]):
        raise RuntimeError('Fit-tune noise leakage')
    if set(meta.source_uid[meta.fold==0])&set(meta.source_uid[meta.fold==1]):
        raise RuntimeError('Fit-tune source leakage')
    meta.to_parquet(folder/'event_metadata.parquet',index=False)
    n.write_csv(ROOT/f'tables/{dep}_POPULATION_STRATA.csv',meta.groupby(['fold','family','mass_bin']).agg(events=('row_index','size'),sources=('source_uid','nunique'),noise_blocks=('noise_bank_index','nunique')).reset_index())
    n.write_json(folder/'COMPLETE.json',{'UTC':n.utc(),'sources':COUNT,'events':len(meta),'noise_blocks':BLOCKS,
        'source_fold_intersection':0,'noise_fold_intersection':0,'peak_shape':list(peak.shape),'low_shape':list(low.shape),
        'peak_sha256':n.sha(peak_path),'low_sha256':n.sha(low_path),'metadata_sha256':n.sha(folder/'event_metadata.parquet'),
        'encoder_trained':False,'sky_or_time_score_changed':False})


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--stage',choices=('freeze','generate'),required=True)
    parser.add_argument('--workers',type=int,default=20)
    args=parser.parse_args();ROOT=args.root
    if args.stage=='freeze':
        freeze()
    else:
        for dep in n.DEPS:
            generate(dep,args.workers)
