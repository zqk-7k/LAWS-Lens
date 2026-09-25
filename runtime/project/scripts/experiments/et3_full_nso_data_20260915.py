#!/usr/bin/env python3
"""ET GW-LMC full-run data/sky adapter; immutable historical dependencies."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['OMP_STACKSIZE'] = '512M'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import logging
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import numpy as np
import pandas as pd

PROJECT = Path('/root/autodl-tmp/gw-catalog')
PILOT = PROJECT/'results/et3_bandlimited_trigger_repair_20260915T033604Z'
DATA_PILOT = PROJECT/'results/et3_data_driven_sky_pilot_20260915T035036Z'
PHYSICAL = PROJECT/'results/et3_physical_trigger_repair_20260915T041500Z_r2'
BANK_PILOT = PROJECT/'results/et3_data_template_pilot_20260915T034329Z'
PLAN = PROJECT/'results/et3_gwlmc3000_unified_nso_20260914T051500Z_r2'
SEEDS = (2026091411, 2026091412, 2026091413)
FINAL = 'HOLD_FOR_AUTHOR_REVIEW_NO_ADOPTION_NO_OVERWRITE'
FS, N = 4096, 24*4096


def now():
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False,
        default=lambda x: x.item() if isinstance(x, np.generic) else x.tolist() if isinstance(x, np.ndarray) else str(x))+'\n')
    temp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''): h.update(block)
    return h.hexdigest()


def stable(*parts):
    return int.from_bytes(hashlib.sha256('|'.join(map(str, parts)).encode()).digest()[:4], 'little')


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec); sys.modules[name] = obj
    spec.loader.exec_module(obj)
    return obj


def guard(root):
    if shutil.disk_usage(root).free < 30*2**30:
        raise RuntimeError('HOLD_DISK_LIMIT: minimum free30GiB; no automatic historical deletion')


def base(root):
    return module(root/'scripts/et3_physical_trigger_repair_20260915_v2.py', 'full_physical_base')


def verify(root):
    frozen = json.loads((root/'contracts/PROTOCOL_FREEZE.json').read_text())
    if sha(root/'contracts/ANALYSIS_CONTRACT.json') != frozen['sha256']:
        raise RuntimeError('Analysis contract changed')
    for x in json.loads((root/'manifest/DEPENDENCIES.json').read_text()):
        if sha(x['path']) != x['sha256']:
            raise RuntimeError('Frozen dependency changed: '+x['path'])


def state(root, **values):
    value = {'utc': now(), **values}
    write(root/'STATUS.json', value); print(json.dumps(value), flush=True)


def child(root, name, events):
    p = root/'population'/name
    p.mkdir(parents=True, exist_ok=False)
    for d in ('contracts', 'strain', 'triggers', 'maps', 'tables', 'logs', 'build', 'manifest', 'package'):
        (p/d).mkdir()
    (p/'scripts').symlink_to(root/'scripts', target_is_directory=True)
    (p/'build/q128').symlink_to(root/'build/q128', target_is_directory=True)
    cfg = json.loads((DATA_PILOT/'contracts/ANALYSIS_CONTRACT.json').read_text())
    cfg.update(id=root.name+':'+name, development_event_count=len(events), map_workers=5,
        no_production_expansion=False, scope=name, parent_contract=str(root/'contracts/ANALYSIS_CONTRACT.json'))
    write(p/'contracts/ANALYSIS_CONTRACT.json', cfg)
    write(p/'contracts/EVENTS.json', events)
    return p


def prepare(root):
    gate = json.loads((DATA_PILOT/'contracts/DATA_DRIVEN_FINAL_RESULT.json').read_text())
    if not gate['numerical_ladder_passed'] or not gate['pixel_gate']['pass']:
        raise RuntimeError('Prior actual-data numerical pilot did not pass')
    root.mkdir(parents=True, exist_ok=False)
    for d in ('contracts', 'scripts', 'population', 'build', 'manifest', 'logs', 'tables',
              'reports', 'figures', 'models', 'features', 'predictions', 'calibration', 'results', 'package'):
        (root/d).mkdir()
    guard(root)
    shutil.copy2(__file__, root/'scripts'/Path(__file__).name)
    for p in PHYSICAL.joinpath('scripts').glob('*.py'):
        shutil.copy2(p, root/'scripts'/p.name)
    shutil.copytree(PILOT/'build/q128', root/'build/q128')
    core = json.loads((root/'build/q128/COMPLETE.json').read_text())
    if core['upstream_tests'] != 'PASS' or sha(root/'build/q128/core.abi3.so') != core['sha256']:
        raise RuntimeError('q128 compiled core identity failure')
    frozen_bank = json.loads((BANK_PILOT/'contracts/BANK_BUILD.json').read_text())
    if sha(frozen_bank['path']) != frozen_bank['sha256']:
        raise RuntimeError('Data-derived search bank changed')
    for name in ('BANK_PARAMETERS.json', 'BANK_BUILD.json'):
        shutil.copy2(BANK_PILOT/'contracts'/name, root/'contracts'/name)
    contract = {
        'id': 'ET3-GWLMC3000-NSO-PHYS-01', 'utc': now(), 'author_authorized_full_ET_run': True,
        'authorization_text': '2026-09-15: 开始做ET3的整体实验内容',
        'independent_output': str(root), 'historical_overwrite': False, 'paper_modified': False,
        'comparison_status': 'same NEW-SCORE-ONLY scoring topology; new ET physical/trigger fixes are not yet applied to historical HL, not bitwise matched HL controls',
        'event_total': 3000, 'source_total': 1800,
        'counts': {'train': {'sources': 1260, 'events': 2100}, 'validation': {'sources': 270, 'events': 450}, 'test': {'sources': 270, 'events': 450}},
        'training_internal_development': '42 sources/family from training only; never optimize on outer validation/test',
        'lens_families': ['GW-LMC smooth/non-subhalo', 'GW-LMC subhalo-present'], 'not_analytic_SIS_PM': True,
        'source_distribution': 'same reserved global GW-LMC IDs as3000plan;2.5PLUS BBH Any_Detected_SNR1;proposal ratio<=4 selection inherited and disclosed',
        'sky_coverage_gate': {'independent_sources': 60, 'per_family': 20,
            'excluded': 'all1800reserved sources and all24pilot source IDs',
            'one_random_image_per_lens_system': True, 'snr_anchor': [9., 11., 16., 24.],
            'HPD_ties':'source-hash randomized fractional boundary;report definite/possible bounds too',
            'temperature': 1., 'likelihood_scale': .83, 'quadrature': 128,
            'test': 'one-sided exact binomial undercoverage test at nominal0.9; alpha0.01; no threshold change after outcomes',
            'scope': 'gross undercoverage screen under this controlled generation distribution, not proof of exact Bayesian population calibration',
            'on_fail': 'HOLD, preserve all outcomes; no bulk expansion or new posterior temperature tuning'},
        'physical': {'waveform': 'IMRPhenomXPHM', 'sample_rate':4096, 'duration':24, 'noise_padding_seconds':1,
            'detectors':['ET1','ET2','ET3'], 'noise':'independent ET design Gaussian draws;not observed ET noise',
            'system_amplitude':'one C per system;minimum unscaled image networkSNR anchored to9,11,16,24 chosen by source hash',
            'no_per_image_ratio_reset':True, 'no_individual_SNR_cap':True, 'spin_conversion':'SIkg',
            'distance':'controlled amplitude rescaling, not new cosmological redshift assignment',
            'short':{'window':[-1.75,.25],'sample_rate':2048,'shape':[3,4096],'band':[40,580]},
            'long':{'window':[-15.75,.25],'sample_rate':256,'shape':[3,4096],'band':[20,80]},
            'long_feature_no_extra_SD_normalization':True},
        'sky': {'actual_saved_noisy_strain':True, 'templates':str(BANK_PILOT/'cache/TEMPLATE_BANK.npy'),
            'template_sha256':frozen_bank['sha256'], 'template_search_truth_inputs':False,
            'template_model':'data-selected PhenomD point parameters, not full BBH PE',
            'ordering_native':'NUNIQ ICRS probability density/sr', 'ordering_raster':'NESTED probability mass',
            'Nside':512,'audit_nsides':[256,512,1024],'quadrature':128,'temperature':1.,'likelihood_scale':.83,
            'score':'log(Npix sum(Pi Pj))', 'permanent_dense_maps':False,
            'tail_numerical_audit':'all true pairs,top1percent null byZ512,hash1percent null;256/512/1024 sameMOC representation audit;not new angular information',
            'maps_shared_between_model_seeds':True},
        'time': {'one_dimensional':True,'calendar':'GPS1126259462 +10Julian years,full duty,controlled ET catalog',
            'generation':'fixed GW-LMC delay then base time uniform over[0,T-delay];no extra exposure reweight after conditioned draw',
            'calibration':'only training global source systems,one weight per system;null pairs from same training catalog excluding companions',
            'density':'existing physical_common Gaussian KDE log10delay LR,bandwidth1;no2D SNR extension',
            'not_old_SIS_PM_lookup_reuse':True,'historical_lookup_unchanged':True},
        'models': {'seeds':SEEDS,'all_new_ET_training':True,'historical_checkpoint_used':False,
            'short_encoder':'same attention-Inception d_model192 depth6 embedding96, in_channels3 instead ofHL2;old Mc/q auxiliary terms retained',
            'short_pretrain_epochs':16,'short_adapt_epochs':60,'short_batch_sources':48,'aux_weight':.25,
            'RNC':'same RAW-PHASE/sourceSupCon/RNC,64logMc bins,50epochs,4passes,64sources*2views;ET phase features4*64*3*3',
            'ordered_mass':'same mass-axisCNN,36+coordinate channels;30epochs',
            'multiscale_mass':'short+long72+coordinate;zero-initialized long weights;15epochs;epoch0 eligible',
            'joint':'p(Mc)p(eta,chieff|Mc),16x17 conditional bins,frozen mass marginal,25epochs;epoch0 eligible',
            'mass_support':[10.,512.],'mass_support_basis':'prior TRAIN-only13.56..413.36 range, margin; no test selected bounds',
            'training_label_clipping':False,'test_outside_support':'report OOD, no deletion; no positive corrective evidence',
            'source_weight':'one per source;independent noise views not additional sources',
            'training_noise_views':4,'noise_views_scope':'training only;canonical plus3 independent ET-design noise draws per image;shared source amplitude;no validation/test augmentation',
            'same_data_across_seeds':True,'seed_variance_scope':'initialization and training order, not new source/noise catalogs'},
        'score':{'alpha':1.,'ancestry':'short cosine+old Mc/q ->global calibration ->RNC finite tail FRT ->ordered-Mc OMC ->joint penalty/increment ->3channel weighted sum',
            'no_outer_old_new_total_mix':True,'not_NODUP_DIRECT':True,
            'calibration':'source-disjoint train-internal development fits ordered/joint;outer validation selects weights/coefficient;testlocked',
            'fusion_grid':'simplex0.05,231points;positive andnonnegative both reported;ties closest prior-stage normalized weight then lexicographic',
            'priority':'F50,F90,-AUPRC,-R10,-R1', 'gamma_grid':[0,.125,.25,.5,1,2,4], 'beta_grid':[0,.0625,.125,.25,.5,1,2],
            'tail':'min(log(p_tail/0.05),0),finite conservative ranks', 'increment':'monotone calibrated logodds clip[-4,4];no positive OOD',
            'unordered_pair':'max directed;here global calibration yields symmetric score, verify equality'},
        'test_lock':'no test waveform inference,sky maps,weights,or ranks before all3seed finalconfigfreeze',
        'metrics':['R1,R5,R10,R50','medianrank','AUPRC','F50','F90','Top10,50,100,200,500','perfamily','seedmeanSD','10000systembootstrap'],
        'ET_PE_audit':'injection-truth predicted parameter distributions;no realET PE or officialpairlabels exist',
        'resource':{'CPU_quota':25,'RAM_GiB':92,'GPU':'RTX5090','map_threads':4,'coverage_workers':6,'bulk_map_workers':5,
            'GPU_training_threads':2,'minimum_free_GiB':30,'estimated_new_storage_GiB':45,'dense_persistence':False},
        'failure_policy':'no outcome-based sample dropping,threshold relaxation,old result replacement or earlytestselection',
        'final_state':FINAL,
        'references':{'BAYESTAR':'https://arxiv.org/abs/1508.03634','GW_LMC':'https://github.com/LensedGW/GW-LMC',
            'inspiral_mass':'https://arxiv.org/abs/gr-qc/9402014','temperature':'https://proceedings.mlr.press/v70/guo17a.html',
            'RNC':'https://arxiv.org/abs/2210.01189'},
    }
    write(root/'contracts/ANALYSIS_CONTRACT.json',contract)
    write(root/'contracts/PROTOCOL_FREEZE.json',{'sha256':sha(root/'contracts/ANALYSIS_CONTRACT.json')})
    b = base(root); old, src, geom, phys = b.deps(root)
    table = src.load_tables(old.CAT)
    reserved = json.loads((PLAN/'contracts/RESERVED_SYSTEM_SPLITS.json').read_text())
    write(root/'contracts/RESERVED_SYSTEM_SPLITS.json',reserved)
    seen = {x['source_id'] for x in reserved}
    pilot_ids = {x['source_id'] for x in json.loads((PHYSICAL/'contracts/EVENTS.json').read_text())}
    qa=[]
    for family,subhalo in [('smooth',False),('subhalo',True),('unlensed',None)]:
        candidates=table if subhalo is None else table[table.lens_is_subhalo==subhalo]
        ordered=sorted(candidates.to_dict('records'),key=lambda x:stable('ETNSO-coverage-20260915',int(x['event_id'])))
        chosen=0
        for r in ordered:
            uid=int(r['event_id'])
            if uid in seen or uid in pilot_ids:continue
            images=src.choose_images(pd.Series(r)) if subhalo is not None else None
            if subhalo is not None and (images is None or images['proposal_snr_ratio']>4):continue
            qa.append({'family':family,'source_id':uid,'gwlmc_row':int(r['gwlmc_row']),'split':'coverage','images':images})
            seen.add(uid);chosen+=1
            if chosen==20:break
        if chosen!=20:raise RuntimeError('Insufficient independent coverage sources')
    write(root/'contracts/COVERAGE_SOURCES.json',qa)
    def records(items,role):
        result=[]
        for rec in items:
            r=table.iloc[rec['gwlmc_row']].to_dict(); sid=rec['source_id'];im=rec['images']
            if int(r['event_id'])!=sid:raise RuntimeError('GW-LMC global source/row mismatch')
            delay=im['delay_days']*86400 if im else 0.
            total=10*365.25*86400
            if not 0<=delay<total:raise RuntimeError('Delay outside frozen ET calendar;no silent resampling')
            gps=1126259462.+np.random.default_rng(stable('ETNSO-gps',sid)).uniform(0,total-delay)
            anchor=[9.,11.,16.,24.][stable('ETNSO-anchor',sid)%4]
            keep=1+stable('ETNSO-coverage-image',sid)%2 if im else 0
            for image in ([1,2] if im else [0]):
                result.append({**r,'source_id':sid,'source_uid':'GW-LMC:'+str(sid),'family':rec['family'],'split':role,
                    'event_uid':f'ETNSO_{sid}_{image}','image':image,'gps':gps+(delay if image==2 else 0.),
                    'morse':im[f'morse_image{image}'] if im else 0.,'magnification':im[f'mu_image{image}'] if im else 1.,
                    'old_target_snr_not_enforced':anchor,'system_snr_anchor':anchor,
                    'noise_seed':stable('ETNSO-physical-noise',sid,image),'coverage_evaluate':role=='coverage' and image==keep})
        return result
    child(root,'coverage',records(qa,'coverage'))
    for part in ('train','validation','test'):
        child(root,part,records([r for r in reserved if r['split']==part],part))
    ids={p:{r['source_id'] for r in json.loads((root/f'population/{p}/contracts/EVENTS.json').read_text())} for p in ('coverage','train','validation','test')}
    if any(ids[a]&ids[c] for a in ids for c in ids if a<c):raise RuntimeError('Global source overlap')
    write(root/'contracts/SOURCE_ISOLATION.json',{'pass':True,'counts':{k:len(v) for k,v in ids.items()},'pilot_sources_not_in_coverage':True})
    dependencies=[{'path':str(p),'sha256':sha(p)} for p in sorted((root/'scripts').glob('*.py'))]
    dependencies += [{'path':str(root/'build/q128/core.abi3.so'),'sha256':core['sha256']}]
    for f in ['scripts/real_search/physical_common.py','matchgw/models.py','matchgw/data.py']:
        p=PROJECT/f
        if p.is_file():dependencies.append({'path':str(p),'sha256':sha(p)})
    write(root/'manifest/DEPENDENCIES.json',dependencies)
    shutil.copy2(PILOT/'manifest/PROTECTED_BEFORE.json',root/'manifest/HISTORICAL_PROTECTED.json')
    state(root,state='FULL_PROTOCOL_FROZEN',next='independent60source_sky_coverage_then_bulk')


def generate_system(root,part,sid):
    import bilby
    from scipy.signal import resample_poly
    logging.getLogger('bilby').setLevel(logging.ERROR)
    p=root/'population'/part; b=base(root); old,src,geom,phys=b.deps(root)
    rows=[r for r in json.loads((p/'contracts/EVENTS.json').read_text()) if r['source_id']==sid]
    marker=p/'tables'/f'source_{sid}.json'
    if marker.exists():
        rec=json.loads(marker.read_text())
        if any(sha(x['path'])!=x['sha256'] for x in rec):raise RuntimeError('Completed strain changed')
        return rec
    guard(root); generator=src.build_waveform_generator(); frequency=np.arange(8193)*.25
    raw=[]
    for row in rows:
        ifos,_,_=geom.geometry(); params=src.source_parameters(pd.Series(row),row['gps'])
        h,_=src.detector_response(generator,ifos,params,src.lens_factor(row['magnification'],row['morse']))
        psds=np.asarray([x.power_spectral_density.get_power_spectral_density_array(frequency) for x in ifos])
        psds=np.where((frequency[None]>=20)&np.isfinite(psds)&(psds>0),psds,np.inf)
        h=np.asarray(h,np.float64); rho=float(phys.optimal_network_snr(h,frequency,psds))
        if not np.isfinite(rho) or rho<=0:raise RuntimeError('Invalid physical optimal SNR')
        raw.append((row,h,psds,rho))
    C=rows[0]['system_snr_anchor']/min(r[3] for r in raw);out=[]
    for row,h,psds,rho in raw:
        start=time.perf_counter();clean=C*h
        np.random.seed(row['noise_seed']);bilby.core.utils.random.seed(row['noise_seed'])
        ifos,_,_=geom.geometry();noise=[]
        for ifo in ifos:
            ifo.minimum_frequency=20
            ifo.set_strain_data_from_power_spectral_density(FS,26.,start_time=row['gps']-24.75)
            noise.append(np.asarray(ifo.strain_data.time_domain_strain))
        noisy=np.asarray(noise);noisy[:,FS:FS+N]+=clean
        pure=np.pad(clean,((0,0),(FS,FS)))
        short=old.preprocess_generic(phys,noisy,frequency,psds)[:,-4096:]
        clean_short=old.preprocess_generic(phys,pure,frequency,psds)[:,-4096:]
        long=resample_poly(old.preprocess_generic(phys,noisy,frequency,psds,20.,80.),1,8,axis=-1,window=('kaiser',8.6))[:,-4096:]
        if short.shape!=(3,4096) or long.shape!=(3,4096) or not np.isfinite(noisy).all():raise RuntimeError('Invalid waveform input')
        path=p/'strain'/f'{row["event_uid"]}.npz';temp=path.with_suffix('.partial.npz')
        if path.exists():raise RuntimeError('Uncommitted existing strain;preserve and investigate')
        np.savez_compressed(temp,clean=clean,noisy_padded=noisy,short=short,long=long,clean_short=clean_short,psd_frequency=frequency,psd=psds)
        temp.rename(path)
        out.append({'event_uid':row['event_uid'],'source_id':sid,'family':row['family'],'split':part,
            'amplitude_scale':C,'unscaled_network_optimal_snr':rho,'network_optimal_snr':C*rho,
            'above60':C*rho>60,'noise_seed':row['noise_seed'],'path':str(path),'sha256':sha(path),'bytes':path.stat().st_size,
            'seconds':time.perf_counter()-start,'prior_float32_source_rounding':'retained tested upstream conversion;noise/filteringfloat64'})
    write(marker,out);return out


def generate(root,part,workers=4):
    verify(root);p=root/'population'/part
    if part!='coverage' and not json.loads((root/'contracts/COVERAGE_GATE.json').read_text())['pass']:
        raise RuntimeError('Independent localization coverage gate required before bulk')
    rows=json.loads((p/'contracts/EVENTS.json').read_text());ids=sorted({r['source_id'] for r in rows})
    started=time.monotonic()
    def task(sid):
        command=[sys.executable,'-B',str(root/'scripts'/Path(__file__).name),'--root',str(root),'--stage','one-source','--part',part,'--source',str(sid)]
        with (p/'logs'/f'generate_{sid}.log').open('ab') as f:subprocess.run(command,stdout=f,stderr=f,check=True)
        return json.loads((p/'tables'/f'source_{sid}.json').read_text())
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for k,f in enumerate(as_completed([pool.submit(task,sid) for sid in ids])):
            f.result()
            if k%20==0:state(root,state='GENERATING_PHYSICAL_STRAIN',part=part,sources_done=k+1,total=len(ids))
    frame=pd.DataFrame([x for sid in ids for x in json.loads((p/'tables'/f'source_{sid}.json').read_text())])
    order={r['event_uid']:i for i,r in enumerate(rows)};frame=frame.assign(row=[order[x] for x in frame.event_uid]).sort_values('row')
    if frame.groupby('source_id').amplitude_scale.nunique().max()!=1:raise RuntimeError('Per-image amplitude violation')
    if frame.noise_seed.duplicated().any():raise RuntimeError('Duplicate event noise seed')
    frame.to_csv(p/'tables/PHYSICAL_STRAIN.csv',index=False)
    write(p/'contracts/PHYSICAL_GATE.json',{'pass':True,'events':len(frame),'sources':len(ids),'seconds':time.monotonic()-started,
        'path':str(p/'tables/PHYSICAL_STRAIN.csv'),'sha256':sha(p/'tables/PHYSICAL_STRAIN.csv')})


def triggers(root,part):
    import torch,lal
    from pycbc.types import FrequencySeries
    from pycbc.filter import matched_filter,sigma
    from ligo.skymap.bayestar import filter as sf
    p=root/'population'/part;b=base(root);_,_,geom,_=b.deps(root)
    if part=='test' and not (root/'contracts/FINAL_SCORE_FREEZE.json').exists():raise RuntimeError('Test trigger inference locked')
    bank=np.load(BANK_PILOT/'cache/TEMPLATE_BANK.npy',mmap_mode='r')
    params=json.loads((root/'contracts/BANK_PARAMETERS.json').read_text())
    rows=json.loads((p/'contracts/EVENTS.json').read_text());f=np.arange(N//2+1)/24
    ifos,_,_=geom.geometry();psds=np.asarray([x.power_spectral_density.get_power_spectral_density_array(f) for x in ifos])
    psds=np.where(np.isfinite(psds)&(psds>0),psds,np.inf)
    lo,hi=int(23.75*FS)-512,int(23.75*FS)+513;records=[]
    torch.set_num_threads(1)
    for idx,row in enumerate(rows):
        if part=='coverage' and not row['coverage_evaluate']:continue
        dest=p/'triggers'/f'event{idx:02d}'
        if (dest/'TRIGGER.json').exists():
            t=json.loads((dest/'TRIGGER.json').read_text())
            if sha(dest/'TRIGGER.npz')!=t['trigger_sha256']:raise RuntimeError('Completed trigger changed')
            records.append(t);continue
        guard(root);start=time.monotonic();data_path=p/'strain'/f'{row["event_uid"]}.npz'
        with np.load(data_path) as a:data=np.asarray(a['noisy_padded'][:,FS:FS+N],np.float64)
        fd=np.fft.rfft(data)/FS;gpu=torch.as_tensor(fd.astype(np.complex64),device='cuda');scores=np.zeros(len(bank))
        for first in range(0,len(bank),32):
            h=bank[first:first+32].astype(np.complex128)
            norm=np.sqrt(4/24*np.sum(abs(h[:,None])**2/psds[None],axis=-1))
            product=(h[:,None].conj()/psds[None]).astype(np.complex64)
            corr=torch.zeros((len(h),3,N),dtype=torch.complex64,device='cuda')
            corr[:,:,:len(f)]=torch.as_tensor(product,device='cuda')*gpu[None]
            snr=torch.fft.ifft(corr,dim=-1)*(4/24*N)
            peaks=torch.amax(abs(snr[:,:,lo:hi]),dim=-1).cpu().numpy()/norm
            scores[first:first+len(h)]=(peaks**2).sum(1)
            del corr,snr
        best=int(np.argmax(scores));h=np.asarray(bank[best],np.complex128);ht=FrequencySeries(h,delta_f=1/24)
        snippets=[];epochs=[];horizons=[];channels=[];exact=0.;hor_errors=[]
        fmax=float(f[abs(h)>0].max());rate=int(min(FS,2**np.ceil(np.log2(8*fmax))));step=FS//rate
        if np.any(h[f>=rate/2]!=0):raise RuntimeError('Template power beyond new Nyquist')
        for d in range(3):
            psd=FrequencySeries(psds[d],delta_f=1/24)
            z=np.asarray(matched_filter(ht,FrequencySeries(fd[d],delta_f=1/24),psd=psd,low_frequency_cutoff=20,high_frequency_cutoff=1024))
            peak=lo+int(np.argmax(abs(z[lo:hi])));exact+=float(abs(z[peak])**2)
            horizon=float(sigma(ht,psd=psd,low_frequency_cutoff=20,high_frequency_cutoff=1024));horizons.append(horizon)
            hp=lal.CreateREAL8FrequencySeries('same filter power',0,0,1/24,lal.StrainUnit**2,len(h)-1);hp.data.data[:]=abs(h[:-1])**2
            sm=sf.SignalModel(sf.signal_psd_series(hp,sf.InterpolatedPSD(f,psds[d],f_high_truncate=1.)))
            hor_errors.append(abs(sm.get_horizon_distance()/horizon-1))
            snippets.append(z[peak-512:peak+513:step].astype(np.complex64))
            epochs.append(int((lal.LIGOTimeGPS(row['gps'])-23.75+(peak-512)/FS).ns()))
            channels.append({'detector':ifos[d].name,'matched_snr':float(abs(z[peak])),'peak_sample':peak})
        error=abs(exact-scores[best])/max(exact,1.)
        if error>2e-5 or max(hor_errors)>2e-3:raise RuntimeError('Actual-data trigger numerical gate failed')
        dest.mkdir(exist_ok=False)
        np.savez_compressed(dest/'TRIGGER.npz',snr_series=snippets,epochs_ns=np.asarray(epochs,np.int64),psds=psds,horizons=horizons,template_power=abs(h)**2)
        record={'event_uid':row['event_uid'],'source_uid':row['source_uid'],'template_args':params[best]['args'],
            'template':'IMRPhenomD','sample_rate':rate,'channels':channels,'source_strain':str(data_path),
            'source_strain_sha256':sha(data_path),'trigger_sha256':sha(dest/'TRIGGER.npz'),'data_derived_intrinsic_template':True,
            'oracle_template':False,'sky_truth_used_in_selection':False,'bank_index':best,'GPU_PyCBC_relative_error':error,
            'horizon_relative_error':max(hor_errors),'seconds':time.monotonic()-start}
        write(dest/'TRIGGER.json',record);records.append(record)
        if len(records)%10==0:state(root,state='DATA_DRIVEN_TEMPLATE_SEARCH',part=part,complete=len(records),total=sum(x['coverage_evaluate'] for x in rows) if part=='coverage' else len(rows))
    write(p/'contracts/TRIGGERS_COMPLETE.json',{'events':len(records),'max_error':max(x['GPU_PyCBC_relative_error'] for x in records),'truth_used':False})


def map_case(root,part,idx,q=128):
    p=root/'population'/part
    if part=='test' and not (root/'contracts/FINAL_SCORE_FREEZE.json').exists():raise RuntimeError('Test maps locked')
    b=base(root);tr=json.loads((p/'triggers'/f'event{idx:02d}'/'TRIGGER.json').read_text());b.FS=tr['sample_rate']
    folder=p/'maps'/f'event{idx:02d}_q{q}'
    if folder.exists():raise RuntimeError('Existing uncommitted map;preserve failure, no overwrite')
    folder.mkdir();case={'idx':idx,'q':q,'id':folder.name};write(folder/'CASE.json',case)
    b.worker(p,case)
    r=json.loads((folder/'RESULT.json').read_text())
    r.update(conditional_intrinsics_oracle_control=False,intrinsics_fixed_to_data_recovered_template=True,
        ordering_native='NUNIQ',ordering_raster='NESTED',coordinates='ICRS',full_BBH_PE=False)
    write(folder/'RESULT.json',r)


def maps(root,part,workers=5):
    verify(root);p=root/'population'/part;rows=json.loads((p/'contracts/EVENTS.json').read_text())
    ids=[i for i,r in enumerate(rows) if part!='coverage' or r['coverage_evaluate']]
    def task(i):
        target=p/'maps'/f'event{i:02d}_q128';result=target/'RESULT.json'
        if result.exists():
            r=json.loads(result.read_text())
            if sha(r['map_path'])!=r['map_sha256']:raise RuntimeError('Completed map changed')
            return r
        guard(root)
        with (p/'logs'/f'map_{i:04d}.log').open('ab') as f:
            command=[sys.executable,'-B',str(root/'scripts'/Path(__file__).name),'--root',str(root),'--stage','one-map','--part',part,'--index',str(i)]
            subprocess.run(command,stdout=f,stderr=f,timeout=3600,check=True)
        return json.loads(result.read_text())
    completed=[]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(task,i) for i in ids]):
            completed.append(future.result())
            write(p/'contracts/MAP_PROGRESS.json',{'utc':now(),'done':len(completed),'total':len(ids),'failed':0,'workers':workers})
            print(json.dumps({'map_part':part,'done':len(completed),'total':len(ids)}),flush=True)
    write(p/'contracts/MAPS_COMPLETE.json',{'utc':now(),'events':len(completed),'failures':0,'q':128,'workers':workers,
        'P50_s':float(np.median([x['wall_seconds'] for x in completed])),'P90_s':float(np.quantile([x['wall_seconds'] for x in completed],.9))})


def coverage(root):
    import healpy as hp
    from scipy.stats import binomtest
    p=root/'population/coverage';rows=json.loads((p/'contracts/EVENTS.json').read_text());out=[];b=base(root)
    for i,r in enumerate(rows):
        if not r['coverage_evaluate']:continue
        result=json.loads((p/'maps'/f'event{i:02d}_q128'/'RESULT.json').read_text())
        mass=b.raster(result['map_path'],512)
        pixel=hp.ang2pix(512,np.pi/2-r['dec'],r['ra']%(2*np.pi),nest=True)
        lower=float(mass[mass>mass[pixel]].sum());upper=float(mass[mass>=mass[pixel]].sum())
        fraction=np.random.default_rng(stable('ETNSO-HPD-tie',r['source_id'])).uniform()
        hpd=lower+fraction*(upper-lower)
        out.append({'source_id':r['source_id'],'event_uid':r['event_uid'],'family':r['family'],
            'HPD':hpd,'HPD_lower':lower,'HPD_upper':upper,'inside90':hpd<=.9,
            'definite90':upper<=.9,'possible90':lower<=.9})
    frame=pd.DataFrame(out)
    if len(frame)!=60 or frame.source_id.nunique()!=60:raise RuntimeError('Coverage requires60independent sources')
    k=int(frame.inside90.sum());test=binomtest(k,60,.9,alternative='less');ci=binomtest(k,60).proportion_ci(.95,method='exact')
    result={'utc':now(),'pass':test.pvalue>=.01,'sources':60,'successes':k,'coverage':k/60,
        'exact95CI':[ci.low,ci.high],'undercoverage_pvalue':test.pvalue,'alpha_frozen':.01,
        'not_full_population_calibration':True,'temperature_not_retuned':True,'noise_source_and_pilot_disjoint':True}
    frame.to_csv(root/'tables/INDEPENDENT_COVERAGE.csv',index=False)
    write(root/'contracts/COVERAGE_GATE.json',result)
    if not result['pass']:raise RuntimeError('HOLD_INDEPENDENT_SKY_UNDERCOVERAGE')
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--stage',choices=['prepare','generate','one-source','triggers','maps','one-map','coverage','coverage-all'],required=True)
    p.add_argument('--part',choices=['coverage','train','validation','test'],default='coverage')
    p.add_argument('--source',type=int);p.add_argument('--index',type=int);p.add_argument('--workers',type=int,default=4)
    a=p.parse_args()
    try:
        if a.stage=='prepare':prepare(a.root)
        elif a.stage=='one-source':generate_system(a.root,a.part,a.source)
        elif a.stage=='generate':generate(a.root,a.part,a.workers)
        elif a.stage=='triggers':triggers(a.root,a.part)
        elif a.stage=='maps':maps(a.root,a.part,a.workers)
        elif a.stage=='one-map':map_case(a.root,a.part,a.index)
        elif a.stage=='coverage':coverage(a.root)
        elif a.stage=='coverage-all':
            generate(a.root,'coverage',4);triggers(a.root,'coverage');maps(a.root,'coverage',6);coverage(a.root)
    except Exception:
        path=a.root/'logs'/f'FAILURE_{a.stage}_{int(time.time())}_{os.getpid()}.json'
        write(path,{'utc':now(),'stage':a.stage,'traceback':traceback.format_exc(),'no_history_overwrite':True})
        raise


if __name__=='__main__':main()
