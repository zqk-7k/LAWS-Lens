"""Frozen O3 scaling benchmark; public products ready, not equal-recall proof."""
import argparse
import contextlib
import gc
import hashlib
import itertools
import json
import multiprocessing as mp
import os
from pathlib import Path
import resource
import shutil
import signal
import sys
import time
import traceback

P = Path('/root/autodl-tmp/gw-catalog')
PREVIOUS = P/'results/trilens_frontend_quick_20260917T130410Z_r3'
PILOT = P/'results/lensrank_speed_scientific_pilot_20260917T073200Z'
O3 = P/'results/main_o3official_cfixed_v1_20260904_20260904T072435Z'
SIZES = (5, 15, 46, 62)
WORKERS = 6
REPEATS = 2
FINAL = 'HOLD_FOR_AUTHOR_REVIEW_NO_ADOPTION_NO_OVERWRITE'
ROOT = None
B = F = CAT = NP = PD = TORCH = None
TIMINGS = []
PHASES = {}


def js(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str)+'\n')
    temporary.replace(path)


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def status(stage, **kwargs):
    record = dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), stage=stage, **kwargs)
    js(ROOT/'contracts/RUN_STATUS.json', record)
    print(json.dumps(record), flush=True)


def guard():
    if shutil.disk_usage(ROOT).free < 30*2**30:
        raise RuntimeError('HOLD_DISK_LIMIT')


def sync():
    if TORCH is not None and TORCH.cuda.is_initialized():
        TORCH.cuda.synchronize()


def timed(method, n, boundary, rep, callback):
    sync()
    start, cpu = time.perf_counter(), time.process_time()
    answer = callback()
    sync()
    record = dict(method=method, n_events=n, n_pairs=n*(n-1)//2,
                  boundary=boundary, repetition=rep, wall_s=time.perf_counter()-start,
                  parent_cpu_s=time.process_time()-cpu,
                  cumulative_parent_peak_rss_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
    TIMINGS.append(record)
    PD.DataFrame(TIMINGS).to_csv(ROOT/'tables/scaling_timings.csv', index=False)
    return answer, record


def paths():
    sys.path[:0] = [str(PILOT/'vendor/site'), str(PILOT), str(P/'scripts/experiments'), str(P)]


def phase_initializer(logdir):
    paths()
    os.nice(5)
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = '1'
    logfile = open(Path(logdir)/f'phase_worker_{os.getpid()}.log', 'a', buffering=1)
    os.dup2(logfile.fileno(), 1)
    os.dup2(logfile.fileno(), 2)
    import lensrank_speed_pilot_20260917 as old
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)
    old.track = lambda path: Path(path)


def phase_job(item):
    import pandas as pd
    import lensrank_speed_pilot_20260917 as old
    from phazap.pe_input import ParameterEstimationInput
    from phazap.postprocess_phase import postprocess_phase
    root, row = item
    root = Path(root)
    old.ROOT = root
    begin, cpu = time.perf_counter(), time.process_time()
    pe_row = pd.Series(row).copy()
    pe_row['sky_map_internal_group'] = row['phazap_posterior_group']
    samples, metadata = old.read_pe(pe_row)
    if metadata['waveform_approximant'] != 'IMRPhenomXPHM':
        raise RuntimeError('Phazap common-waveform input contract mismatch')
    loaded = time.perf_counter()
    output = root/'phases'/f'{row["event_name"]}.hdf5'
    if output.exists():
        raise RuntimeError('No phase overwrite')
    phase = postprocess_phase(ParameterEstimationInput(samples, **metadata), flow=20, fbest=40,
        fhigh=100, superevent_name=row['event_name'], label='scaling_full',
        output_dir=str(output.parent), output_filename=output.name)
    import numpy as np
    if not all(np.isfinite(value).all() for value in phase.dataset.values()):
        raise RuntimeError('Nonfinite phase posterior')
    return dict(event_name=row['event_name'], samples=len(samples), path=str(output),
        wall_s=time.perf_counter()-begin, cpu_s=time.process_time()-cpu,
        posterior_read_s=loaded-begin, peak_rss_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)


def pair_initializer(files, ready, logdir):
    global PHASES
    paths()
    os.nice(5)
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = '1'
    logfile = open(Path(logdir)/f'pair_worker_{os.getpid()}.log', 'a', buffering=1)
    os.dup2(logfile.fileno(), 1)
    os.dup2(logfile.fileno(), 2)
    from phazap.postprocess_phase import PostprocessedPhase
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)
    PHASES = {name:PostprocessedPhase.from_file(path) for name,path in files.items()}
    ready.put(dict(pid=os.getpid(), events=len(PHASES)))


def pair_job(pair):
    from phazap import phazap
    import numpy as np
    a, b = pair
    begin, cpu = time.perf_counter(), time.process_time()
    value = phazap(PHASES[a], PHASES[b], plot=False)
    values = [float(value[k]) for k in (0, 1, 2, 4)]
    if not np.isfinite(values).all():
        raise RuntimeError('Nonfinite Phazap statistic: '+a+'--'+b)
    return dict(event_i=a, event_j=b, pair_key='--'.join(sorted(pair)), DJ=values[0],
        volume=values[1], phase_shift=values[2], upstream_p_value=values[3],
        worker_wall_s=time.perf_counter()-begin, worker_cpu_s=time.process_time()-cpu,
        worker_pid=os.getpid(), worker_peak_rss_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)


def init():
    global B, F, CAT, NP, PD, TORCH
    ROOT.mkdir(exist_ok=False)
    for name in ('contracts', 'scripts', 'tables', 'results', 'logs', 'reports', 'figures', 'manifest'):
        (ROOT/name).mkdir()
    shutil.copy2(__file__, ROOT/'scripts'/Path(__file__).name)
    start = time.perf_counter()
    sys.path.insert(0, str(PREVIOUS/'scripts'))
    import trilens_frontend_pilot_20260917 as base
    import numpy as np
    import pandas as pd
    import torch
    from threadpoolctl import threadpool_limits
    B, NP, PD, TORCH = base, np, pd, torch
    B.ROOT = ROOT
    B.TIMES, B.CHECKS, B.INPUTS = [], [], {}
    B.CONTEXT.update(method='SETUP', n_events=62, repetition=-1)
    F, CAT, old = B.load_science()
    threadpool_limits(2)
    torch.set_num_threads(2)
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = '1'
    js(ROOT/'contracts/IMPORT_TIMING.json', dict(science_imports_s=time.perf_counter()-start,
        not_complete_process_cold_start=True))
    manifest = pd.read_csv(B.track(O3/'data/official70_event_manifest.csv'))
    archive = pd.read_parquet(B.ARCHIVE/'development/evaluation/MCWF-UNIFIED-PATH875-DEVCONF/gwtc3/seed_202607241/real_fusion_pairs.parquet', columns=['event_i','event_j'])
    names = set(archive.event_i)|set(archive.event_j)
    chosen = manifest[manifest.event_name.isin(names)].copy()
    chosen['selection_hash'] = chosen.event_name.map(lambda name:hashlib.sha256(('GWLR-SPEED-PILOT-01:'+name).encode()).hexdigest())
    chosen = chosen.sort_values(['selection_hash','event_name']).reset_index(drop=True)
    if len(chosen)!=62 or chosen.event_name.nunique()!=62:
        raise RuntimeError('Frozen 62-event scope mismatch')
    previous = pd.read_csv(PILOT/'contracts/selected_events.csv')
    if chosen.event_name.head(6).tolist()!=previous.event_name.tolist():
        raise RuntimeError('Name-hash prefix differs')
    chosen['phazap_posterior_group'] = chosen.sky_map_internal_group.map(
        lambda group:'C01:IMRPhenomXPHM' if group=='C01:Mixed' else group)
    chosen['posterior_group_differs_from_frozen_sky'] = chosen.phazap_posterior_group.ne(chosen.sky_map_internal_group)
    chosen.to_csv(ROOT/'contracts/selected_events.csv', index=False)
    contract = dict(code='TRILENS-SCALE-03', algorithm='NEW-SCORE-ONLY / PATH1, alpha=1',
        scopes=[dict(events=n,pairs=n*(n-1)//2) for n in SIZES], seeds=[202607241,202607242,202607243],
        selection='Nested name-hash prefixes of all 62 frozen strict O3 events, all unordered pairs; no replacement by results',
        boundaries=['PREPARE_EVENTS: fresh strain/PE processing including checkpoint and sky IO',
                    'EVENT_CACHE_PAIR_RANK: recompute all pair scores and ranks from prepared event features, not cached pair scores',
                    'PRODUCTS_READY_TOTAL: preparation plus first pair evaluation, directly timed'],
        preparation_repeats=1, event_cache_pair_repeats=REPEATS, phazap_workers=WORKERS, threads_per_worker=1,
        trilens_cpu_threads=2, gpu='native CUDA implementation', all_available_posterior_samples=True,
        phazap_version='0.3.3', phase_frequencies_Hz=[20,40,100], official_FPP_and_PO_not_implemented=True,
        phazap_posterior_policy='All 62 use explicit public IMRPhenomXPHM posterior/config. GW200220_124850 uses C01:IMRPhenomXPHM in the same HDF5, not configuration-less C01:Mixed. TriLens frozen Mixed sky for this event is unchanged. Product-group mismatch disclosed; no equal-recall claim.',
        previous_failed_attempt='trilens_scaling_20260917_10to1891: one Mixed posterior lacks a single waveform config; retained. No timing/quality result used to select XPHM.',
        numerical_replay=dict(atol=2e-4,rtol=1e-5,rank='exact',Phazap_atol=1e-8,Phazap_rtol=1e-8),
        neural_batch_policy='62 native slots with zero padding for absent events, extra compute charged; not optimized small-batch scaling',
        diagnostic_sky_BC='excluded from timed scoring because not used by PATH1 ranking',
        both_allow_event_caches=True, no_cached_pair_scores_used=True,
        no_OS_page_cache_flush=True, cold_process_start_not_claimed=True, worker_startup_and_loading_in_PREPARE_EVENTS=True,
        offline_model_training_calibration_PE_generation_network_download_excluded_both=True,
        joint_prediction_checkpoint_integrity_check='archived routine retains a checksum check inside inference; charged conservatively',
        concurrency='UAB continues; shared load. No equal-resource or equal-recall speedup claim',
        gate='Stop on missing input, nonfinite output, replay mismatch, disk guard or timeout. No replacement events.',
        disk_floor_GiB=30, maximum_wall_seconds=14400,
        scientific_status=FINAL, genuine_lensing_labels_absent=True, posterior_audits_not_selection_inputs=True)
    js(ROOT/'contracts/CONTRACT.json', contract)
    js(ROOT/'contracts/FREEZE.json', dict(code_sha256=digest(Path(__file__)),
        contract_sha256=digest(ROOT/'contracts/CONTRACT.json'), event_sha256=digest(ROOT/'contracts/selected_events.csv')))
    status('INPUT_INVENTORY')
    rows, recipes = B.inventory(F,CAT,chosen)
    js(ROOT/'contracts/HARDWARE_BEFORE.json',B.hardware())
    B.track(PREVIOUS/'scripts/trilens_frontend_pilot_20260917.py')
    preflight = ROOT/'results/preflight'
    (preflight/'contracts').mkdir(parents=True)
    old.ROOT, old.track = preflight, B.track
    pe_rows, errors = [], []
    for row in chosen.itertuples(index=False):
        try:
            pe_row=pd.Series(row._asdict())
            pe_row['sky_map_internal_group']=row.phazap_posterior_group
            samples, metadata = old.read_pe(pe_row)
            if metadata['waveform_approximant']!='IMRPhenomXPHM':
                raise RuntimeError('Expected public IMRPhenomXPHM metadata')
            pe_rows.append(dict(event_name=row.event_name,samples=len(samples),
                phazap_posterior_group=row.phazap_posterior_group,
                frozen_sky_group=row.sky_map_internal_group,**metadata))
        except Exception as error:
            errors.append(dict(event_name=row.event_name,error=repr(error)))
    pd.DataFrame(pe_rows).to_csv(ROOT/'tables/PE_INPUT_INVENTORY.csv',index=False)
    js(ROOT/'contracts/PREFLIGHT.json',dict(events=len(chosen),valid=len(pe_rows),errors=errors))
    if errors:
        raise RuntimeError('PE preflight failed, no replacement: '+str(errors))
    js(ROOT/'manifest/INPUTS_BEFORE.json',B.INPUTS)
    return chosen, rows, recipes


def all_features(full, lowviews, frequency, psds):
    feature = F.archived.feature
    adaptive = feature.adaptive
    banks = B.measured('spectral_bank_IO', lambda: (
        NP.load(F.e.TRAINED/'cache/adaptive_psd/unwhitened_aligned_template_spectra.npy'),
        NP.load(F.t.PREVIOUS/'fine_mass_context/cache/fine_mass_spectra.npy'),
        NP.load(F.LOW/'cache/lowband_aligned_spectra.npy')))
    short = F.dev.TRAIN.make_window_view(full, 2)
    coarse, refined, low = [], [], []
    for index in range(len(full)):
        def coarse_fn():
            bank = adaptive.whitened_bank(banks[0], frequency, psds[index])
            return feature.fine.features(short[index:index+1], bank)
        coarse.append(B.measured('coarse_phase_feature', coarse_fn, event_index=index))
        def refined_fn():
            bank = adaptive.whitened_bank(banks[1], frequency, psds[index])
            # The archived fine branch normalizes internally; do not normalize twice.
            return F.archived.fine.features(full[index:index+1, :, -4096:], bank)
        refined.append(B.measured('fine_mass_feature', refined_fn, event_index=index))
        def low_fn():
            bank = F.low.bank_for(banks[2], frequency, psds[index])
            return F.low.features(lowviews[index:index+1], bank)
        low.append(B.measured('low16s_feature', low_fn, event_index=index))
    return NP.concatenate(coarse), NP.concatenate(refined), NP.concatenate(low), short


def prepare_trilens(chosen, rows, recipes, n, dest):
    rows = rows.iloc[:n]
    raw, refs = B.measured('strain_reference_IO',lambda:B.read_raw(rows))
    def psd():
        values=[F.dev.BASE.v7.v3.estimate_psd(item) for item in refs]
        return values[0][0], NP.stack([item[1] for item in values])
    frequency, psds = B.measured('PSD_estimation',psd)
    full = B.measured('short_preprocessing',lambda:NP.stack([F.dev.BASE.v7.v3.preprocess_24s(w,frequency,psds[k]).astype(NP.float32) for k,w in enumerate(raw)]))
    lowviews = B.measured('low16s_preprocessing',lambda:NP.stack([F.low.low_view(w,frequency,psds[k],F.dev.BASE.v7.v3) for k,w in enumerate(raw)]))
    coarse, refined, low, short = all_features(full,lowviews,frequency,psds)
    x = F.ordered.arrange(coarse,refined)
    slots = rows.native_slot.to_numpy(int)
    def pad(array):
        result=NP.zeros((62,)+array.shape[1:],array.dtype)
        result[slots]=array
        return result
    cache = dict(full=full, rows=rows, models={})
    padded_full,padded_short,padded_coarse,padded_x=pad(full),pad(short),pad(coarse),pad(x)
    for recipe in recipes:
        seed,slot=recipe['seed'],recipe['slot']
        def short_infer():
            v7=F.dev.BASE.v7
            model,_=v7.load_unified_model(CAT.checkpoint(seed))
            z,pred=v7.embed_catalog(model,v7.ArrayCatalog(padded_full),batch_size=16)
            calibration=json.loads((F.dev.V7/f'gwtc3/seed_{seed}/results/waveform_channel_calibration_v7.json').read_text())
            return z[slots],pred[slots],calibration
        z,pred,calibration=B.measured('short_checkpoint_inference',short_infer,seed=seed)
        p,embedding=B.measured('RNC_checkpoint_inference',lambda:B.rnc(F,padded_short,padded_coarse,slot),seed=seed)
        spec=json.loads((F.e.BASE/f'calibration/gwtc3/seed_{seed}/SELECTED_CONFIG.json').read_text())
        def ordered_infer():
            ck=TORCH.load(F.t.PREVIOUS/f'ordered_mass_predictor/models/gwtc3/seed_{slot}/validation_selected_model.pt',map_location='cpu',weights_only=False)
            model=F.ordered.Predictor().cuda().eval();model.load_state_dict(ck['model'])
            pp,outside=F.ordered.probability(F.ordered.infer(model,(padded_x-ck['mu'])/ck['sd']),ck['temperature'])
            return pp[slots],outside[slots],ck['prior']
        pp,outside,prior=B.measured('ordered_mass_checkpoint_inference',ordered_infer,seed=seed)
        a=B.measured('multirate_joint_checkpoint_inference',lambda:F.joint_prediction(pad(NP.concatenate([x,low],1)),'gwtc3',slot),seed=seed)
        cache['models'][seed]=dict(z=z,pred=pred,calibration=calibration,p=p[slots],embedding=embedding[slots],
            FRT=spec,pp=pp,outside=outside,prior=prior,joint={key:value[slots] for key,value in a.items()})
    def skies():
        maps,audits=[],[]
        for _,row in chosen.iloc[:n].iterrows():
            probability,audit=CAT.cbase.corrected_read_probability_map(row,512)
            maps.append(probability);audits.append(audit)
        js(dest/'sky_ordering.json',audits)
        return TORCH.as_tensor(NP.stack(maps),device='cuda',dtype=TORCH.float64)
    cache['maps']=B.measured('sky_IO_ordering_Nside512',skies)
    cache['time']=json.loads((F.dev.V7/'gwtc3/shared/time_delay_likelihood_ratio.json').read_text())
    return cache


def score_trilens(cache,recipes,dest,rep):
    rows=cache['rows'];n=len(rows);i,j=NP.triu_indices(n,1)
    names=rows.event_name.to_numpy()
    frame=PD.DataFrame(dict(idx_i=i,idx_j=j,event_i=names[i],event_j=names[j],pair_key=['--'.join(sorted([a,b])) for a,b in zip(names[i],names[j])]))
    delta=abs(rows.gps_time.to_numpy()[i]-rows.gps_time.to_numpy()[j])/86400
    frame['time_score']=F.dev.BASE.v7.apply_time_likelihood_ratio(delta,cache['time'])
    maps=cache['maps'];gram=(maps@maps.T).cpu().numpy()
    frame['sky_raw_log_bf']=NP.log(NP.maximum(maps.shape[1]*gram[i,j],CAT.SKY_BF_FLOOR))
    outputs=[]
    for recipe in recipes:
        seed=recipe['seed'];c=cache['models'][seed];f=frame.copy()
        f['waveform_embedding_cosine']=NP.sum(c['z'][i]*c['z'][j],axis=1)
        f['waveform_abs_delta_logmc_std']=abs(c['pred'][i,0]-c['pred'][j,0])
        f['waveform_abs_delta_logitq_std']=abs(c['pred'][i,1]-c['pred'][j,1])
        f=F.dev.BASE.v7.apply_waveform_channel(f,c['calibration'])
        f['previous_waveform_score']=f.waveform_score
        values=F.archived.ev.features(f,c['p'],c['embedding'])
        for name,key in (('new_encoder_cosine','cosine'),('new_mass_similarity','mass'),('new_mass_predictive_BC','predictive_BC'),('new_mass_pred_logmc_i','mean_i'),('new_mass_pred_logmc_j','mean_j')):
            f[name]=values[key]
        f['waveform_score']=F.archived.finite.score(f,c['FRT'])[0]
        values=F.masscal.pair_features(dict(p=c['pp'],outside=c['outside']),i,j,c['prior'])
        baseline=F.masscal.score(f,values,recipe['OMC_config'],'PRIOR')[0]
        pen,inc,_=F.joint_increment(c['joint'],f,recipe['joint_config'])
        spec=recipe['joint_config']
        f['waveform_score']=baseline if spec.get('unchanged_OMC',False) else baseline+spec['gamma']*pen+spec['beta']*inc
        f['seed']=seed
        f['final_score']=f[['waveform_score','time_score','sky_raw_log_bf']].to_numpy()@NP.asarray(recipe['upstream_weights'])
        f.to_csv(dest/f'seed_{seed}_rep{rep}.csv',index=False)
        outputs.append(f)
    consensus=B.rank_consensus(outputs)
    consensus.to_csv(dest/f'consensus_rep{rep}.csv',index=False)
    return dict(frames=outputs,consensus=consensus,full=cache['full'])


def run_trilens(chosen,rows,recipes,n):
    dest=ROOT/f'results/TriLens_n{n}';dest.mkdir()
    B.CONTEXT.update(method='TriLens',n_events=n,repetition=0)
    status('TRILENS_PREPARE',n_events=n)
    begin=time.perf_counter();cpu=time.process_time()
    cache,_=timed('TriLens',n,'PREPARE_EVENTS',0,lambda:prepare_trilens(chosen,rows,recipes,n,dest))
    result,first=timed('TriLens',n,'EVENT_CACHE_PAIR_RANK',0,lambda:score_trilens(cache,recipes,dest,0))
    TIMINGS.append(dict(method='TriLens',n_events=n,n_pairs=n*(n-1)//2,boundary='PRODUCTS_READY_TOTAL',repetition=0,
        wall_s=time.perf_counter()-begin,parent_cpu_s=time.process_time()-cpu))
    PD.DataFrame(TIMINGS).to_csv(ROOT/'tables/scaling_timings.csv',index=False)
    B.verify_trilens(F,result,rows,recipes,n,0)
    for rep in range(1,REPEATS):
        result,_=timed('TriLens',n,'EVENT_CACHE_PAIR_RANK',rep,lambda:score_trilens(cache,recipes,dest,rep))
        B.verify_trilens(F,result,rows,recipes,n,rep)
    del cache,result
    gc.collect();TORCH.cuda.empty_cache()
    js(dest/'COMPLETE.json',dict(pass_replay=True,phase='SCORING_ONLY_NOT_LENSING_VALIDATION'))


def run_phazap(chosen,n):
    dest=ROOT/f'results/Phazap_n{n}';dest.mkdir()
    for folder in ('phases','contracts','logs'):(dest/folder).mkdir()
    status('PHAZAP_PREPARE',n_events=n,workers=WORKERS)
    begin=time.perf_counter();cpu=time.process_time()
    ctx=mp.get_context('spawn')
    def prepare():
        records=[]
        with ctx.Pool(min(n,WORKERS),initializer=phase_initializer,initargs=(str(dest/'logs'),)) as pool:
            jobs=[(str(dest),row) for row in chosen.iloc[:n].to_dict('records')]
            for value in pool.imap_unordered(phase_job,jobs,chunksize=1):
                records.append(value)
                PD.DataFrame(records).to_csv(dest/'event_preparation.csv',index=False)
                status('PHAZAP_PREPARE',n_events=n,completed_events=len(records),workers=WORKERS)
                guard()
        files={row['event_name']:row['path'] for row in records}
        ready=ctx.Queue()
        pool=ctx.Pool(WORKERS,initializer=pair_initializer,initargs=(files,ready,str(dest/'logs')))
        try:
            initialized=[ready.get(timeout=180) for _ in range(WORKERS)]
        except BaseException:
            pool.terminate();pool.join();raise
        js(dest/'contracts/PAIR_WORKERS.json',initialized)
        return pool
    pool,prep=timed('Phazap',n,'PREPARE_EVENTS',0,prepare)
    pairs=list(itertools.combinations(chosen.event_name.iloc[:n],2))
    reference=None
    try:
        for rep in range(REPEATS):
            def score():
                values=[]
                for result in pool.imap_unordered(pair_job,pairs,chunksize=1):
                    values.append(result)
                    if len(values)%25==0 or len(values)==len(pairs):
                        PD.DataFrame(values).to_csv(dest/f'pair_checkpoint_rep{rep}.csv',index=False)
                        status('PHAZAP_PAIRS',n_events=n,repetition=rep,completed_pairs=len(values),total_pairs=len(pairs),workers=WORKERS)
                        guard()
                frame=PD.DataFrame(values).sort_values('pair_key').reset_index(drop=True)
                frame.sort_values(['DJ','pair_key']).to_csv(dest/f'all_pairs_rep{rep}.csv',index=False)
                return frame
            frame,record=timed('Phazap',n,'EVENT_CACHE_PAIR_RANK',rep,score)
            record['worker_cpu_s_sum']=float(frame.worker_cpu_s.sum())
            record['worker_job_wall_s_sum']=float(frame.worker_wall_s.sum())
            if rep==0:
                TIMINGS.append(dict(method='Phazap',n_events=n,n_pairs=len(pairs),boundary='PRODUCTS_READY_TOTAL',repetition=0,
                    wall_s=time.perf_counter()-begin,parent_cpu_s=time.process_time()-cpu))
            PD.DataFrame(TIMINGS).to_csv(ROOT/'tables/scaling_timings.csv',index=False)
            B.record_check(f'Phazap_count_n{n}_r{rep}',len(frame)==len(pairs) and frame.pair_key.nunique()==len(pairs))
            old=PD.read_csv(PILOT/'fullposterior_diagnostic/tables/phazap_full_pairs.csv')
            old=old[old.samples.eq('full')].copy()
            old['pair_key']=['--'.join(sorted([a,b])) for a,b in zip(old.event_i,old.event_j)]
            overlap=frame.merge(old,on='pair_key',suffixes=('_new','_old'))
            expected=10 if n==5 else 15
            B.record_check(f'Phazap_prior_replay_n{n}_r{rep}',len(overlap)==expected and NP.allclose(overlap.DJ_new,overlap.DJ_old,atol=1e-8,rtol=1e-8),
                pairs=len(overlap),max_abs_delta_DJ=float(abs(overlap.DJ_new-overlap.DJ_old).max()))
            if reference is not None:
                B.record_check(f'Phazap_repeat_n{n}_r{rep}',NP.allclose(reference.DJ,frame.DJ,atol=1e-8,rtol=1e-8),
                    max_abs_delta_DJ=float(abs(reference.DJ-frame.DJ).max()))
            reference=frame
    finally:
        pool.close();pool.join()
    js(dest/'COMPLETE.json',dict(pass_replay=True,official_FPP_not_computed=True))


def finish(success,error=None):
    protected=[]
    if B is not None:
        for path,before in B.INPUTS.items():
            after=digest(path)
            protected.append(dict(path=path,**before,sha256_after=after,unchanged=after==before['sha256_before']))
        PD.DataFrame(protected).to_csv(ROOT/'manifest/PROTECTED_INPUTS.csv',index=False)
        js(ROOT/'contracts/HARDWARE_AFTER.json',B.hardware())
    unchanged=bool(protected) and all(row['unchanged'] for row in protected)
    checks=[] if B is None else B.CHECKS
    js(ROOT/'contracts/FINAL_STATUS.json',dict(compute_complete=success,protected_inputs_unchanged=unchanged,
        checks_total=len(checks),checks_passed=sum(row['passed'] for row in checks),error=error,
        measurements_only=True,matched_recall_not_tested=True,official_PO_not_run=True,scientific_status=FINAL))
    status('SCALING_COMPLETE' if success and unchanged else 'HOLD_FAILURE',error=error)


def main():
    global ROOT
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True,type=Path)
    args=parser.parse_args();ROOT=args.root
    def timeout(signum,frame):
        raise TimeoutError('Four-hour benchmark limit reached')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(14400)
    error=None;success=False
    try:
        chosen,rows,recipes=init()
        for n in SIZES:
            guard()
            run_trilens(chosen,rows,recipes,n)
            run_phazap(chosen,n)
        success=True
    except BaseException as exc:
        error=repr(exc)
        if ROOT.exists():
            (ROOT/'logs/FAIL.txt').write_text(traceback.format_exc())
        raise
    finally:
        if ROOT.exists():finish(success,error)
        signal.alarm(0)


if __name__=='__main__':
    main()
