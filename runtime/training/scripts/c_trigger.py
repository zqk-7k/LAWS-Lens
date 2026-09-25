#!/usr/bin/env python3
"""C known-event HL follow-up, using the actual noisy strain and an independent bank.

Not a blind observing-run search or full intrinsic-parameter PE. Truth is used
only in the downstream localization audit, never in choosing a template.
"""
import os
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

import argparse
from collections import namedtuple
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import heapq
import importlib.util
import json
import logging
import multiprocessing as mp
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np
import pandas as pd
from scipy.ndimage import maximum_filter1d
from scipy.signal import resample_poly
from scipy.signal.windows import tukey

PROJECT = Path('/root/autodl-tmp/gw-catalog')
BASE = PROJECT/'results/gwlr_unified_snr_ab_20260917T113500Z_r4'
RUNS = ('O3', 'O4a', 'O4b')
FS_RAW, FS, DURATION, F_LOW, F_HIGH = 4096, 2048, 64, 20., 900.
N, DF = FS*DURATION, 1/DURATION
BANK_SEED = 2026091801
BANK_COUNT = 8192
STATE = {}


def now():
    return datetime.now(timezone.utc).isoformat()


def stable(*parts):
    return int.from_bytes(hashlib.sha256(':'.join(map(str, parts)).encode()).digest()[:8], 'little')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8*2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    tmp.replace(path)


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    sys.modules[name] = obj
    spec.loader.exec_module(obj)
    return obj


def guard(root):
    if shutil.disk_usage(root).free < 25*2**30:
        raise RuntimeError('HOLD_DISK_LIMIT_25_GIB')


def protect_list():
    paths = [BASE/'contracts/PLAN_FREEZE.json', BASE/'contracts/FINAL_SCORE_FREEZE.json',
             BASE/'contracts/SCORING_SPEC.json', BASE/'scripts/source_generator.py']
    for run in RUNS:
        paths.extend(BASE/'plans'/run/name for name in ('sources.parquet', 'event_plan.parquet', 'noise_plan.parquet'))
    complete = BASE/'completion_20260918T014623Z'
    paths.extend(p for p in complete.rglob('*.csv') if p.stat().st_size < 2**20)
    return sorted(set(paths))


def prepare(root):
    if root.exists():
        raise RuntimeError('New output root required')
    for part in ('contracts', 'scripts', 'bank', 'strain', 'timings', 'logs', 'maps', 'reports', 'manifest'):
        (root/part).mkdir(parents=True, exist_ok=True)
    shutil.copy2(__file__, root/'scripts/c_timing.py')
    guard(root)
    cfg = dict(code='GWLR-C-PHYS-E2E-TIMING-01', utc=now(), timing_only=True,
        no_training=True, no_real_reranking=True, no_paper_changes=True,
        no_holdout_read=True, baseline=str(BASE), raw_fs=FS_RAW, filter_fs=FS,
        seconds=DURATION, coalescence_offset_seconds=48., filter_band=[F_LOW, F_HIGH],
        bank_sizes=[10, 100, 1000], bank_seed=BANK_SEED,
        bank_model='IMRPhenomD, independent log-Mc/q/aligned-spin random proposal',
        bank_is_not_coverage_validated=True, filter_window='fixed catalogue-time +/-0.25 s',
        not_full_run_blind_search=True, truth_intrinsics_not_used_for_filter_templates=True,
        common_distance_proposal_mpc=[500., 1500., 4000.],
        no_faint_image_target_scaling=True, no_detection_rejection_in_timing_pilot=True,
        templates_do_not_use_true_mass_spin=True, sky_temperature=1., sky_nside=512,
        no_claim_of_calibrated_sky=True, production_detection_efficiency_not_measured=True,
        production_source_population_not_frozen=True,
        disk_safety_gib=25, max_output_gib=5)
    write(root/'contracts/TIMING_CONTRACT.json', cfg)
    write(root/'contracts/PROTECTED_INPUTS.json', {str(p):sha(p) for p in protect_list()})
    rows = []
    for run in RUNS:
        frame = pd.read_parquet(BASE/'plans'/run/'sources.parquet')
        events = pd.read_parquet(BASE/'plans'/run/'event_plan.parquet')
        pool = frame[(frame.role == 'main') & (frame.split == 'train') & (frame.family != 'unlensed')]
        for slot, mc in enumerate((7., 15., 30., 60., 120.)):
            sub = pool[pool.family == ('SIS' if slot % 2 == 0 else 'PM')]
            row = sub.iloc[np.argmin(abs(np.log(sub.mc_det.to_numpy()/mc)))].to_dict()
            row['pilot_slot'] = slot
            row['run'] = run
            row['distance_mpc'] = cfg['common_distance_proposal_mpc'][slot % 3]
            row['common_amplitude_factor'] = row['dl_source']/row['distance_mpc']
            ev = events[(events.source_uid == row['source_uid']) & (events.variant == 0)]
            row['noise_records'] = ev.sort_values('image_number').to_dict('records')
            rows.append(row)
    write(root/'contracts/SOURCE_PLAN.json', rows)
    hw = {'utc':now(), 'cpu_count_host':os.cpu_count(), 'affinity_count':len(os.sched_getaffinity(0)),
          'disk':shutil.disk_usage(root)._asdict()}
    for name in ('cpu.max', 'memory.max', 'memory.current', 'cpuset.cpus.effective'):
        hw[name] = (Path('/sys/fs/cgroup')/name).read_text().strip()
    for label, command in [('gpu', ['nvidia-smi']), ('memory', ['free', '-h']), ('cpu',['lscpu'])]:
        hw[label] = subprocess.run(command, capture_output=True, text=True).stdout
    write(root/'contracts/HARDWARE.json', hw)
    print(json.dumps({'stage':'prepared', 'root':str(root), 'sources':len(rows), 'events':2*len(rows)}), flush=True)


def build_bank(root):
    from pycbc.waveform import get_fd_waveform
    guard(root)
    rng = np.random.default_rng(BANK_SEED)
    rows = []
    start = time.perf_counter()
    if (root/'bank/COMPLETE.json').exists():
        receipt = json.loads((root/'bank/COMPLETE.json').read_text())
        if sha(root/'bank/templates.npy') != receipt['sha256']:
            raise RuntimeError('Bank hash changed')
        return
    bank = np.lib.format.open_memmap(root/'bank/templates.npy', mode='w+', dtype=np.complex64, shape=(BANK_COUNT, N//2+1))
    for i in range(BANK_COUNT):
        while True:
            mc = float(np.exp(rng.uniform(np.log(5), np.log(200))))
            q = float(rng.uniform(.25, 1))
            eta = q/(1+q)**2
            total = mc/eta**.6
            m1, m2 = total/(1+q), total*q/(1+q)
            if 3 <= m2 <= m1 <= 300:
                break
        spin = float(rng.uniform(-.8, .8))
        tick = time.perf_counter()
        hp, _ = get_fd_waveform(approximant='IMRPhenomD', mass1=m1, mass2=m2,
            spin1z=spin, spin2z=spin, distance=1., inclination=0., coa_phase=0.,
            delta_f=DF, f_lower=F_LOW, f_final=F_HIGH, f_ref=20.)
        hp.resize(N//2+1)
        bank[i] = np.asarray(hp)
        rows.append(dict(index=i, mass1=m1, mass2=m2, spin1z=spin, spin2z=spin,
                         mc=mc, q=q, template_seconds=time.perf_counter()-tick))
    bank.flush()
    pd.DataFrame(rows).to_csv(root/'bank/templates.csv', index=False)
    write(root/'bank/COMPLETE.json', {'seconds':time.perf_counter()-start,
        'templates':BANK_COUNT, 'bytes':(root/'bank/templates.npy').stat().st_size,
        'sha256':sha(root/'bank/templates.npy'), 'parameter_sha256':sha(root/'bank/templates.csv')})
    print(json.dumps({'stage':'bank', 'seconds':time.perf_counter()-start}), flush=True)


def generate(root):
    sys.path.insert(0, str(PROJECT))
    src = module(BASE/'scripts/source_generator.py', 'c_source')
    from scripts.real_search import physical_common as phys
    logging.getLogger('bilby').setLevel(logging.ERROR)
    src.DURATION, src.N_SAMPLES, src.END_AFTER_GEOCENTER_SECONDS = DURATION, DURATION*FS_RAW, 16.
    gen = src.build_waveform_generator()
    ifos = list(src.bilby.gw.detector.InterferometerList(['H1','L1']))
    event_rows, resources = [], []
    for source in json.loads((root/'contracts/SOURCE_PLAN.json').read_text()):
        run = source['run']
        tick = time.perf_counter()
        rawbank = np.load(BASE/'plans'/run/'noise_reference_bank.npy', mmap_mode='r')
        psdbank = np.load(BASE/'plans'/run/'noise_psd_bank.npy', mmap_mode='r')
        pf = np.load(BASE/'plans'/run/'noise_psd_frequency.npy')
        for im in (1,2):
            event_start = time.perf_counter()
            t = time.perf_counter()
            gps = source['gps_a' if im == 1 else 'gps_b']
            clean, _ = src.detector_response(gen, ifos, src.source_parameters(pd.Series(source),gps),
                src.lens_factor(source[f'mu_image{im}'],source[f'morse_image{im}']))
            wave_seconds = time.perf_counter()-t
            ev = source['noise_records'][im-1]
            bank = int(ev['noise_bank_index'])
            offset = stable('C-timing-offset',run,source['source_uid'],im) % ((256-DURATION)*FS_RAW+1)
            noise = np.asarray(rawbank[bank,:,offset:offset+DURATION*FS_RAW],np.float64)
            if noise.shape != clean.shape:
                raise RuntimeError('Noise slice mismatch')
            psd = np.asarray(psdbank[bank],np.float64)
            original_snr = phys.optimal_network_snr(clean,pf,psd)
            a = float(source['common_amplitude_factor'])
            scaled = clean.astype(np.float64)*a
            rho = phys.optimal_network_snr(scaled,pf,psd)
            if abs(rho/original_snr-a) > 1e-8*max(a,1):
                raise RuntimeError('Common-scale check failed')
            mixed = noise+scaled
            tickprep = time.perf_counter()
            filtered = resample_poly(mixed,1,2,axis=-1,window=('kaiser',8.6))
            resample_seconds = time.perf_counter()-tickprep
            f = np.arange(N//2+1)*DF
            psds = np.array([np.interp(f,pf,p) for p in psd])
            psds = np.maximum(psds,1e-60)
            uid = f'{run}-slot{source["pilot_slot"]}-image{im}'
            path = root/'strain'/f'{uid}.npz'
            t = time.perf_counter()
            np.savez(path, noisy_raw=mixed.astype(np.float32), noisy_filter=filtered,
                     psd=psds, clean_raw=scaled.astype(np.float32))
            row = {k:v for k,v in source.items() if k != 'noise_records'}
            row.update(event_uid=uid,image=im,gps=gps,start_gps=gps-48.,
                noise_bank_index=bank,noise_offset=offset,unscaled_optimal_snr=original_snr,
                optimal_snr=rho,strain_path=str(path),strain_sha256=sha(path),
                waveform_seconds=wave_seconds,resample_seconds=resample_seconds,
                write_seconds=time.perf_counter()-t,total_generation_seconds=time.perf_counter()-event_start,
                strain_bytes=path.stat().st_size)
            event_rows.append(row)
        resources.append({'run':run,'source':source['source_uid'],'wall_seconds':time.perf_counter()-tick})
        print(json.dumps({'stage':'generation','run':run,'slot':source['pilot_slot'],'seconds':time.perf_counter()-tick}),flush=True)
    write(root/'contracts/EVENT_PLAN.json',event_rows)
    pd.DataFrame(event_rows).to_csv(root/'timings/generation.csv',index=False)
    pd.DataFrame(resources).to_csv(root/'timings/generation_systems.csv',index=False)


def worker_init(root):
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)
    STATE['root'] = Path(root)
    STATE['bank'] = np.load(Path(root)/'bank/templates.npy',mmap_mode='r')
    STATE['params'] = pd.read_csv(Path(root)/'bank/templates.csv')
    import ligo.skymap
    ligo.skymap.omp.num_threads = 1


def run_event(task):
    row, batch, localize = task
    root = STATE['root']
    out = root/'timings'/batch/row['event_uid']
    out.mkdir(parents=True,exist_ok=False)
    before_cpu = time.process_time()
    before = time.perf_counter()
    try:
        return run_event_inner(row,batch,localize,out,before,before_cpu)
    except Exception:
        result = {'event_uid':row['event_uid'],'run':row['run'],'batch':batch,'status':'FAIL',
                  'traceback':traceback.format_exc(),'total_seconds':time.perf_counter()-before,
                  'cpu_seconds':time.process_time()-before_cpu}
        write(out/'RESULT.json',result)
        return result


def run_event_inner(row,batch,do_localize,out,before,before_cpu):
    from pycbc.filter import matched_filter, sigma
    from pycbc.types import FrequencySeries
    from ligo.skymap.bayestar import filter as skyfilter
    from pycbc.vetoes import power_chisq
    from pycbc.events.ranking import newsnr
    import lal
    root, bank, params = STATE['root'],STATE['bank'],STATE['params']
    startread=time.perf_counter()
    if sha(row['strain_path']) != row['strain_sha256']:
        raise RuntimeError('Strain hash changed')
    with np.load(row['strain_path']) as data:
        strain=resample_poly(np.array(data['noisy_raw'], dtype=float), 1, 2,
                             axis=-1, window=('kaiser', 8.6))
        freq = np.arange(N//2+1)*DF
        psds=np.maximum(np.array([np.interp(freq, data['psd_frequency'], p)
                                  for p in data['psd']]), 1e-60)
    # The central trigger window is outside the tapered 3.2 s end regions.
    strain *= tukey(N, alpha=.1)[None, :]
    read_seconds=time.perf_counter()-startread
    freq=np.arange(N//2+1)*DF
    fds=[FrequencySeries(np.fft.rfft(x)/FS,delta_f=DF) for x in strain]
    pseries=[FrequencySeries(p,delta_f=DF) for p in psds]
    lo,hi=int(47.75*FS),int(48.25*FS)
    lag=int(np.ceil(.0106*FS))
    best=None; bestscore=-np.inf; snapshots=[]; shortlist=[]
    search_start=time.perf_counter()
    for i in range(len(bank)):
        template=FrequencySeries(np.array(bank[i], dtype=np.complex128),delta_f=DF)
        rho=[np.asarray(matched_filter(template,fd,psd=p,low_frequency_cutoff=F_LOW,
            high_frequency_cutoff=F_HIGH)) for fd,p in zip(fds,pseries)]
        lpower=maximum_filter1d(abs(rho[1][lo-lag:hi+lag])**2,size=2*lag+1,mode='constant')[lag:-lag]
        scores=abs(rho[0][lo:hi])**2+lpower
        ph=lo+int(np.argmax(scores)); score=float(np.max(scores))
        if score>bestscore:
            pl=ph-lag+int(np.argmax(abs(rho[1][ph-lag:ph+lag+1])))
            bestscore=score; best=(i,[rho[0].copy(),rho[1].copy()],[ph,pl])
        candidate=(score, i, ph)
        if len(shortlist)<8:
            heapq.heappush(shortlist, candidate)
        elif candidate>shortlist[0]:
            heapq.heapreplace(shortlist, candidate)
        if i+1 in (10,100,1000,BANK_COUNT):
            snapshots.append({'templates':i+1,'seconds':time.perf_counter()-search_start,
                              'best_template':best[0],'network_matched_snr':float(np.sqrt(bestscore))})
    raw_bestscore=bestscore
    # A fixed top-eight consistency check reduces domination by glitches. Raw
    # complex SNR (not reweighted SNR) remains the BAYESTAR likelihood input.
    veto_records=[]; best_reweighted=-np.inf
    for rawscore, i, ph in sorted(shortlist, reverse=True):
        h=FrequencySeries(np.array(bank[i], dtype=np.complex128), delta_f=DF)
        rho=[np.asarray(matched_filter(h,fd,psd=p,low_frequency_cutoff=F_LOW,
             high_frequency_cutoff=F_HIGH)) for fd,p in zip(fds,pseries)]
        pl=ph-lag+int(np.argmax(abs(rho[1][ph-lag:ph+lag+1])))
        chi=[float(power_chisq(h,fd,16,p,low_frequency_cutoff=F_LOW,
                   high_frequency_cutoff=F_HIGH)[peak])/30
             for fd,p,peak in zip(fds,pseries,(ph,pl))]
        if not np.isfinite(chi).all():
            raise RuntimeError('Invalid chi-square statistic')
        reweighted=float(sum(float(newsnr(abs(r[peak]), x))**2
                         for r,peak,x in zip(rho,(ph,pl),chi)))
        veto_records.append(dict(template=i, raw_rho=float(np.sqrt(rawscore)),
            reduced_chisq=chi, reweighted_rho=float(np.sqrt(reweighted))))
        if reweighted>best_reweighted:
            best_reweighted=reweighted; bestscore=rawscore
            best=(i,[r.copy() for r in rho],[ph,pl])
    search_seconds=time.perf_counter()-search_start
    idx,rhos,peaks=best
    template=np.array(bank[idx],dtype=np.complex128); template_fs=FrequencySeries(template,delta_f=DF)
    pars=params.iloc[idx].to_dict()
    args={k:float(pars[k]) for k in ('mass1','mass2','spin1z','spin2z')}
    args.update(spin1x=0.,spin1y=0.,spin2x=0.,spin2y=0.,f_final=F_HIGH,f_ref=20.)
    keep=(freq>=F_LOW)&(freq<F_HIGH)
    checks=[]; series=[]; half=int(.125*FS)
    start=lal.LIGOTimeGPS(row['start_gps'])
    for d in range(2):
        norm=float(sigma(template_fs,psd=pseries[d],low_frequency_cutoff=F_LOW,high_frequency_cutoff=F_HIGH))
        direct=4*DF/norm*np.sum(np.asarray(fds[d])[keep]*template[keep].conj()/psds[d][keep]
                               *np.exp(2j*np.pi*freq[keep]*peaks[d]/FS))
        self_data=FrequencySeries(template*(13./norm)*np.exp(.37j-2j*np.pi*freq*789/FS),delta_f=DF)
        self_snr=np.asarray(matched_filter(template_fs,self_data,psd=pseries[d],
                                          low_frequency_cutoff=F_LOW,high_frequency_cutoff=F_HIGH))
        power=lal.CreateREAL8FrequencySeries('selected filter power',0,0,DF,lal.StrainUnit**2,len(freq)-1)
        power.data.data[:]=abs(template[:-1])**2*keep[:-1]
        model=skyfilter.SignalModel(skyfilter.signal_psd_series(power,
            skyfilter.InterpolatedPSD(freq,psds[d],f_high_truncate=1.)))
        weighted=power.data.data/psds[d][:-1]
        nonzero=np.flatnonzero(weighted)
        expected_horizon=np.sqrt(4*DF*np.trapezoid(weighted[nonzero[0]:nonzero[-1]+1]))
        check={'detector':('H1','L1')[d],'matched_filter_relative':float(abs(direct-rhos[d][peaks[d]])/max(abs(direct),1)),
               'self_injection_relative':float(abs(self_snr[789]-13*np.exp(.37j))/13),
               'horizon_relative':float(abs(model.get_horizon_distance()/norm-1)),
               'horizon_quadrature_relative':float(abs(model.get_horizon_distance()/expected_horizon-1)),
               'matched_snr':float(abs(rhos[d][peaks[d]]))}
        checks.append(check)
        epoch=start+(peaks[d]-half)/FS
        snrseries=lal.CreateCOMPLEX8TimeSeries('data-derived SNR',epoch,0,1/FS,lal.DimensionlessUnit,2*half+1)
        snrseries.data.data[:]=rhos[d][peaks[d]-half:peaks[d]+half+1]
        series.append(snrseries)
    if max(c[k] for c in checks for k in ('matched_filter_relative','self_injection_relative','horizon_quadrature_relative'))>1e-6:
        raise RuntimeError('Trigger normalization failed: '+json.dumps(checks))
    np.savez(out/'TRIGGERS.npz',snr_series=[np.asarray(s.data.data) for s in series],
             epochs_ns=[int(s.epoch.ns()) for s in series],psds=psds,template_power=abs(template)**2)
    write(out/'TRIGGERS.json',{'template_index':int(idx),'template_args':args,'checks':checks,
                             'strain_sha256':row['strain_sha256'],'true_intrinsics_used':False,
                             'chi_square_shortlist':veto_records,
                             'no_Gaussian_trigger_draw':True})
    sky_seconds=0.; raster_seconds=0.; normsky=None; mapsize=0; native_pixels=0
    if do_localize:
        from ligo.skymap import bayestar, moc
        from ligo.skymap.io.events.base import Event,SingleEvent
        from ligo.skymap.io.fits import write_sky_map
        ST=namedtuple('SingleTuple','detector snr phase time zerolag_time psd snr_series')
        class Single(ST,SingleEvent): pass
        ET=namedtuple('EventTuple','singles template_args')
        class HLEvent(ET,Event): pass
        singles=[]
        for d,name in enumerate(('H1','L1')):
            ps=lal.CreateREAL8FrequencySeries('noise-block PSD',0,0,DF,lal.StrainUnit**2,len(freq))
            ps.data.data[:]=psds[d]
            arrival=start+peaks[d]/FS
            z=rhos[d][peaks[d]]
            singles.append(Single(name,float(abs(z)),float(np.angle(z)),arrival,arrival,ps,series[d]))
        actual=lal.CreateREAL8FrequencySeries('actual selected template power',0,0,DF,lal.StrainUnit**2,len(freq)-1)
        actual.data.data[:]=abs(template[:-1])**2*keep[:-1]
        original=bayestar.filter.sngl_inspiral_psd
        bayestar.filter.sngl_inspiral_psd=lambda *a,**k:actual
        t=time.perf_counter()
        try:
            sky=bayestar.localize(HLEvent(singles,args),waveform='IMRPhenomD',f_low=F_LOW,
                enable_snr_series=True,f_high_truncate=1.,rescale_loglikelihood=.83)
        finally: bayestar.filter.sngl_inspiral_psd=original
        sky_seconds=time.perf_counter()-t
        t=time.perf_counter()
        normsky=float(np.sum(np.asarray(sky['PROBDENSITY'])*moc.uniq2pixarea(sky['UNIQ'])))
        if abs(normsky-1)>1e-5: raise RuntimeError('Map normalization failed')
        path=root/'maps'/batch/f'{row["event_uid"]}.fits.gz';path.parent.mkdir(exist_ok=True,parents=True)
        write_sky_map(path,sky,nest=True)
        import healpy as hp
        raster_times={}
        for nside in (256,512,1024):
            ti=time.perf_counter()
            raster=moc.rasterize(sky,order=int(np.log2(nside)))
            prob=np.asarray(raster['PROBDENSITY'],np.float64)*hp.nside2pixarea(nside)
            if abs(prob.sum()-1)>1e-5: raise RuntimeError('Raster mass failed')
            raster_times[str(nside)]=time.perf_counter()-ti
            del raster,prob
        raster_seconds=time.perf_counter()-t
        mapsize=path.stat().st_size;native_pixels=len(sky)
        write(out/'MAP.json',{'sha256':sha(path),'path':str(path),'normalization':normsky,
            'native_pixels':native_pixels,'raster_seconds':raster_times,'dense_persisted':False})
        metrics_module = module(root/'scripts/bayestar_si_fixed.py', 'c_posterior_metrics')
        metrics = metrics_module.posterior_metrics(metrics_module.raster_probability(sky,512),
                                                  row['ra_true'],row['dec_true'])
        orders=moc.uniq2order(np.asarray(sky['UNIQ'],np.int64))
    result=dict(status='PASS',run=row['run'],event_uid=row['event_uid'],batch=batch,
        optimal_network_snr=row['optimal_snr'],network_matched_snr=float(np.sqrt(bestscore)),
        selected_template=int(idx),selected_mc=float(pars['mc']),true_mc_audit_only=row['mc_det'],
        read_seconds=read_seconds,search_seconds=search_seconds,search_snapshots=snapshots,
        sky_seconds=sky_seconds,raster_and_write_seconds=raster_seconds,
        total_seconds=time.perf_counter()-before,cpu_seconds=time.process_time()-before_cpu,
        peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        map_bytes=mapsize,native_pixels=native_pixels,sky_normalization=normsky,checks=checks)
    result.update(reweighted_network_snr=float(np.sqrt(best_reweighted)),
        maximum_raw_network_snr=float(np.sqrt(raw_bestscore)),bank_templates=len(bank),
        selected_reduced_chisq=next(r['reduced_chisq'] for r in veto_records if r['template']==idx),
        same_noisy_strain=True,truth_used_for_template_selection=False,
        catalogue_time_conditioned_not_blind_search=True)
    if do_localize:
        result.update(**metrics, moc_path=str(path),moc_sha256=sha(path),
            native_order_min=int(orders.min()),native_order_max=int(orders.max()),
            ordering='NESTED',coordinate_frame='ICRS',probability_convention='density per steradian',
            fallback_used=False,primary_error=None,uses_controlled_source_parameters=False,
            map_shared_across_model_seeds=True,seconds=time.perf_counter()-before)
    result['conditioning']='64 s strain; Tukey alpha=0.1; 3.2 s end tapers'
    write(out/'RESULT.json',result)
    return result


def benchmark(root,batch,workers,count,localize):
    guard(root)
    rows=json.loads((root/'contracts/EVENT_PLAN.json').read_text())
    # Interleave runs, fixed before timing or detection outcomes.
    rows=sorted(rows,key=lambda r:(r['pilot_slot'],r['image'],r['run']))[:count]
    start=time.perf_counter()
    results=[]
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn'),initializer=worker_init,
                             initargs=(str(root),)) as pool:
        for r in pool.map(run_event,[(r,batch,localize) for r in rows]):
            results.append(r)
            print(json.dumps({k:r.get(k) for k in ('event_uid','status','search_seconds','sky_seconds','total_seconds','traceback')}),flush=True)
    write(root/'timings'/f'{batch}.json',dict(batch=batch,workers=workers,events=len(rows),
        wall_seconds=time.perf_counter()-start,cpu_seconds=sum(r['cpu_seconds'] for r in results),
        pass_count=sum(r['status']=='PASS' for r in results),results=results))


def finish(root):
    old=json.loads((root/'contracts/PROTECTED_INPUTS.json').read_text())
    changes=[p for p,h in old.items() if sha(p)!=h]
    write(root/'contracts/HISTORICAL_HASH_AUDIT.json',{'checked':len(old),'changes':changes,'pass':not changes})
    if changes: raise RuntimeError('Protected input changed')
    write(root/'manifest/SHA256.json',{str(p.relative_to(root)):sha(p) for p in root.rglob('*')
        if p.is_file() and 'manifest' not in p.parts and p.suffix not in ('.npy','.npz')})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','bank','generate','benchmark','finish'])
    p.add_argument('--root',type=Path,required=True);p.add_argument('--batch',default='serial')
    p.add_argument('--workers',type=int,default=1);p.add_argument('--count',type=int,default=6)
    p.add_argument('--no-localize',action='store_true');a=p.parse_args()
    if a.stage=='prepare':prepare(a.root)
    elif a.stage=='bank':build_bank(a.root)
    elif a.stage=='generate':generate(a.root)
    elif a.stage=='benchmark':benchmark(a.root,a.batch,a.workers,a.count,not a.no_localize)
    else:finish(a.root)
