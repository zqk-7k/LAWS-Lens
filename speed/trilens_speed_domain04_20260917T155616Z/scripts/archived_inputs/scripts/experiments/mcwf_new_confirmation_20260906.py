#!/usr/bin/env python3
"""New source/noise catalogs for a frozen, shared O3/O4 waveform method."""
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import time
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
import numpy as np
import pandas as pd
import mcwf_development_20260905 as dev
import mcwf_encoder_body_20260906 as body
import mcwf_encoder_evaluate_20260906 as ev
import mcwf_fresh_confirmation_20260906 as old
import mcwf_o3_runmatched_data_20260906 as rm
import mcwf_unified_waveform_20260906 as u
import mcwf_unified_tail_20260906 as tail
import mcwf_conditional_waveform_calibration_20260906 as conditional
import mcwf_finelag_encoder_20260907 as finelag
import mcwf_finelag_eventpsd_20260907 as finelag_psd
import mcwf_finite_reference_tail_20260907 as finite_tail

CATALOGS=(202609101,202609102,202609103)
PREVIOUS=dev.PROJECT/'results/main_o3_mcwf_encoder_v2_20260906T153600Z'
N_NOISE=16


def freeze(root):
    if (root/'contracts/ENSEMBLE_DIAGNOSTIC.json').exists():
        raise RuntimeError('Dependent ensemble pilot is not three independent runs; cannot freeze as final candidate')
    path=root/'contracts/UNIFIED_FRESH_FREEZE.json'
    if path.exists():
        return verify(root)
    gate=json.loads((root/'contracts/DEVELOPMENT_GATE.json').read_text())
    if not gate['development_pass']:
        raise RuntimeError('No qualified shared candidate; do not generate confirmation')
    provenance=root/'contracts/DISTILLED_MODEL_PROVENANCE.json'
    model_provenance=json.loads(provenance.read_text()) if provenance.exists() else {}
    trained=Path(model_provenance.get('training_root',model_provenance.get('trained_root',PREVIOUS)))
    feature_implementation=model_provenance.get('feature_implementation','legacy_coarse_phasebank')
    files=[]
    recipes=set()
    for dep in u.DEPS:
        for ms,es in zip(body.MODEL_SEEDS,dev.SEEDS):
            checkpoint=trained/f'models/RAW-PHASE-SOURCE/{dep}/seed_{ms}/validation_selected_model.pt'
            config=root/f'calibration/{dep}/seed_{es}/SELECTED_CONFIG.json'
            spec=json.loads(config.read_text())
            recipes.add(spec['recipe'])
            if spec['gamma']<=0:
                raise RuntimeError('No old-only fallback permitted')
            files.extend([checkpoint,config])
            if 'calibrator_path' in spec:
                files.append(Path(spec['calibrator_path']))
    if len(recipes)!=1:
        raise RuntimeError('Run/seed-specific waveform recipes are not allowed')
    files += [root/'contracts/UNIFIED_METHOD.json',Path(__file__),Path(u.__file__),Path(tail.__file__),Path(conditional.__file__),Path(old.__file__),Path(body.__file__),Path(ev.__file__)]
    if recipes=={'finite_reference_companion_tail'}:
        files += [Path(finite_tail.__file__),root/'contracts/FINAL_METHOD_CLARIFICATION.json']
    if feature_implementation in ('sample_resolved_zero_padded_fft_v1','event_psd_sample_resolved_fft_v1'):
        files += [Path(finelag.__file__),Path(finelag.fine.__file__)]
    if feature_implementation=='event_psd_sample_resolved_fft_v1':
        files += [Path(finelag_psd.__file__),Path(finelag_psd.adaptive.__file__),trained/'cache/adaptive_psd/unwhitened_aligned_template_spectra.npy']
        if (trained/'contracts/PHYSICAL_MASS_ANCHOR.json').exists():
            import mcwf_physical_mass_anchor_20260907 as anchor
            files += [Path(anchor.__file__),trained/'contracts/PHYSICAL_MASS_ANCHOR.json']
    if (trained/'contracts/RNC_TRAINING.json').exists():
        import mcwf_rankncontrast_20260907 as rnc
        files += [Path(rnc.__file__),trained/'contracts/RNC_TRAINING.json']
    files += [dev.PROJECT/'scripts/experiments/mcwf_fresh_independence_audit_20260907.py',
              dev.PROJECT/'scripts/experiments/mcwf_new_confirmation_audit_20260906.py']
    for dep in u.DEPS:
        files.append(dev.V7/dep/'shared/time_delay_likelihood_ratio.json')
        for es in dev.SEEDS:
            files += [dev.V7/dep/f'seed_{es}/results/waveform_channel_calibration_v7.json',
                dev.V7/dep/f'seed_{es}/waveform_gate/unified_inception_attention_peak2s_4096_aux_0p25_q_1p0_v7/validation_selected_model.pt']
    files.append(dev.BAY/'contracts/selected_config.json')
    config={'utc':datetime.now(timezone.utc).isoformat(),'training_root':str(trained),'recipe':list(recipes)[0],
        'feature_implementation':feature_implementation,
        'catalog_seeds':CATALOGS,'per_catalog':{'sources_smooth':35,'sources_subhalo':35,'background_sources':50,'events':190,'true_pairs':70,'noise256s_blocks':N_NOISE},
        'freshness':'new BBH masses/spins/sky/orientations/phase/GPS and global-GPS-noise-disjoint blocks; previous confirmation blocks excluded',
        'lens_environments':'same GW-LMC environments may recur; conditional source/noise generalization, NOT independent lens-population evidence',
        'sky':'unchanged conditional BAYESTAR with known intrinsic parameters and local PSD, independent Gaussian matched-filter measurements; not localization inferred from the exact non-Gaussian injected strain',
        'mass_SNR_proposal':'unchanged coverage-style balanced Mc5-200/SNR8-40 from prior generation; not astrophysical-rate sampling',
        'amplitudes':'legacy independent per-image PSD-optimal target scaling retained; not response-derived intensity-ratio experiment',
        'both_runs_use_new_model':True,'guardrails':{'R10_drop_max':.02,'AP_drop_max':.005,'F50_F90_ratio_max':1.1},
        'decision':'per-model means over3 new catalogs must pass in both runs; report every catalog-model-method and source/noise bootstrap CI; never redefine failed catalogs as successes',
        'after_opening':'any new tuning makes this set development and requires a new independent confirmation set, keeping all failures',
        'frozen_files':[{'path':str(p),'sha256':dev.sha(p)} for p in sorted(set(files))]}
    dev.json_write(path,config)
    shutil.copy2(__file__,root/'scripts'/Path(__file__).name)
    return config


def verify(root):
    conf=json.loads((root/'contracts/UNIFIED_FRESH_FREEZE.json').read_text())
    for r in conf['frozen_files']:
        if dev.sha(Path(r['path']))!=r['sha256']:
            raise RuntimeError('Frozen input changed: '+r['path'])
    return conf


def exclusions(dep):
    rows=[]
    hist=pd.read_csv(dev.ORCH.SOURCE_ROOT/dep/'shared/noise_bank_manifest.csv')
    rows += [{'start':r.reference_start_gps,'end':r.reference_start_gps+r.reference_duration_s,'origin':'historical'} for r in hist.itertuples()]
    paths=[dev.OLD/f'data/noise_banks/{dep}/noise_manifest_{s}.csv' for s in ('train','validation')]
    if dep=='gwtc3':
        paths += [PREVIOUS/f'data/o3_only_noise/noise_manifest_{s}.csv' for s in ('train','validation')]
    for p in paths:
        f=pd.read_csv(p)
        rows += [{'start':r.reference_start_gps,'end':r.reference_end_gps,'origin':str(p)} for r in f.itertuples()]
    for name in ('confirmation','confirmation_global_noise_fixed'):
        p=PREVIOUS/name/dep/'noise/noise_manifest.csv'
        f=pd.read_csv(p)
        rows += [{'start':r.start_gps,'end':r.end_gps,'origin':str(p)} for r in f.itertuples()]
    return pd.DataFrame(rows).drop_duplicates(['start','end'])


def noise(root,dep):
    out=root/f'confirmation/{dep}/noise'
    if (out/'COMPLETE.json').exists():
        return
    out.mkdir(parents=True,exist_ok=True)
    _,data,v3,_=old.modules()
    historical=exclusions(dep)
    dev.csv_write(out/'EXCLUDED_BLOCKS.csv',historical)
    excluded=[(float(r.start)-16,float(r.end)+16) for r in historical.itertuples()]
    if dep=='gwtc3':
        pool=pd.read_csv(PREVIOUS/'data/o3_only_noise/official_O3_full_strain_offsource_pool.csv')
        source_run=dev.MAIN/'cache/source_run'
    else:
        source_run=data.SOURCE_RUNS[dep]
        pool=pd.read_parquet(source_run/'data/real_noise_injections/offsource_noise_segments.parquet')
    cache=v3.HdfCache(source_run,max_files=8)
    n=N_NOISE*len(CATALOGS)
    reference=np.lib.format.open_memmap(out/'reference.npy',mode='w+',dtype=np.float32,shape=(n,2,v3.NOISE_REFERENCE_SAMPLES))
    rows=[]
    psds=[]
    for k in range(n):
        parts=[]
        for r in pool.to_dict('records'):
            for a,b in rm.subtract_intervals(float(r['segment_start']),float(r['segment_end']),excluded):
                if b-a>=v3.PSD_SECONDS+2:
                    parts.append({**r,'segment_start':a,'segment_end':b})
        if not parts:
            raise RuntimeError('Independent noise exhausted, do not reuse old blocks')
        chosen,rejected=data.select_valid_noise_references(pd.DataFrame(parts),1,np.random.default_rng(data.stable_seed(dep,'unified-new-noise',202609100,k)),v3.PSD_SECONDS,cache,v3)
        r,start,ref,freq,psd=chosen[0]
        if any(start<b and start+v3.PSD_SECONDS>a for a,b in excluded):
            raise RuntimeError('Global-GPS noise conflict')
        excluded.append((start-16,start+v3.PSD_SECONDS+16))
        reference[k]=ref
        psds.append(psd)
        rows.append({'bank_index':k,'catalog_seed':CATALOGS[k//N_NOISE],'parent_event':str(r.event_name),'start_gps':start,'end_gps':start+v3.PSD_SECONDS,'H1_path':r.H1_path,'L1_path':r.L1_path})
        if k%8==0:
            print(json.dumps({'noise':dep,'selected':k+1,'total':n}),flush=True)
    reference.flush()
    np.save(out/'psd.npy',np.stack(psds))
    np.save(out/'frequency.npy',freq)
    dev.csv_write(out/'noise_manifest.csv',pd.DataFrame(rows))
    dev.json_write(out/'COMPLETE.json',{'global_GPS_conflicts':0,'independent_blocks':n,'historical_exclusions':len(historical),'reference_sha256':dev.sha(out/'reference.npy')})


def init_worker(root,dep,cs):
    from threadpoolctl import threadpool_limits
    import torch
    threadpool_limits(limits=1)
    torch.set_num_threads(1)
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
        os.environ[key]='1'
    old.CATALOG_SEEDS=CATALOGS
    old.NOISE_PER_CATALOG=N_NOISE
    old.worker_init(root,dep,cs)


def generate(root,dep,workers):
    freeze(root)
    verify(root)
    old.CATALOG_SEEDS=CATALOGS
    old.NOISE_PER_CATALOG=N_NOISE
    noise(root,dep)
    for cs in CATALOGS:
        out=root/f'confirmation/{dep}/catalog_{cs}'
        if (out/'GENERATION_COMPLETE.json').exists():
            continue
        plan=old.plan_catalog(root,dep,cs)
        events=[]
        with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn'),initializer=init_worker,initargs=(str(root),dep,cs)) as pool:
            for k,result in enumerate(pool.map(old.generate_system,plan.to_dict('records'))):
                events.extend(result)
                if k%20==0:
                    print(json.dumps({'generation':dep,'catalog':cs,'systems':k+1}),flush=True)
        f=pd.DataFrame(events)
        f.insert(0,'idx',np.arange(len(f)))
        f.to_parquet(out/'event_manifest.parquet',index=False)
        dev.json_write(out/'GENERATION_COMPLETE.json',{'events':len(f),'systems':len(plan),'manifest_sha256':dev.sha(out/'event_manifest.parquet')})


def score(root,dep):
    conf=verify(root)
    trained=Path(conf['training_root'])
    _,_,_,sky=old.modules()
    skyconfig=json.loads((dev.BAY/'contracts/selected_config.json').read_text())
    timecal=json.loads((dev.V7/dep/'shared/time_delay_likelihood_ratio.json').read_text())
    rows=[]
    for cs in CATALOGS:
        out=root/f'confirmation/{dep}/catalog_{cs}'
        events=pd.read_parquet(out/'event_manifest.parquet')
        i,j=np.triu_indices(len(events),1)
        y=events.source_uid.to_numpy()[i]==events.source_uid.to_numpy()[j]
        dt=np.abs(events.gps_obs.to_numpy()[i]-events.gps_obs.to_numpy()[j])/86400
        base=pd.DataFrame({'idx_i':i,'idx_j':j,'is_true_pair':y,'true_pair_family':np.where(y,events.family_slot.to_numpy()[i],'NONE'),
            'event_count':len(events),'delta_t_days':dt,'time_score':dev.BASE.v7.apply_time_likelihood_ratio(dt,timecal)})
        full=np.stack([np.load(out/f'events/{uid}_full24.npy') for uid in events.event_uid])
        rawmaps=np.stack([np.load(out/f'events/{uid}_sky512.npy') for uid in events.event_uid])
        skycache={}
        for ms,es in zip(body.MODEL_SEEDS,dev.SEEDS):
            saved=out/f'model_{ms}'
            saved.mkdir(exist_ok=True)
            if (saved/'COMPLETE.json').exists():
                rows.extend(pd.read_csv(saved/'metrics.csv').to_dict('records'))
                continue
            t=float(skyconfig['deployments'][dep][str(es)]['posterior_temperature'])
            if t not in skycache:
                maps=np.stack([sky.apply_temperature(p,t).astype(np.float32) for p in rawmaps])
                zs,bc,audit=sky.gpu_pair_features(maps)
                del maps
                skycache[t]=(zs,bc)
                dev.json_write(out/f'sky_accumulation_audit_T{t:g}.json',audit)
            zs,bc=skycache[t]
            f=base.copy()
            f['sky_raw_log_bf']=zs
            f['sky_bc']=bc
            f=old.old_waveform(dep,es,full,f)
            oldscore=f.waveform_score.to_numpy(float)
            f['previous_waveform_score']=oldscore
            checkpoint=trained/f'models/RAW-PHASE-SOURCE/{dep}/seed_{ms}/validation_selected_model.pt'
            if conf.get('feature_implementation')=='event_psd_sample_resolved_fft_v1':
                noise_root=root/f'confirmation/{dep}/noise'
                p,z,_=finelag_psd.encode_context(trained,checkpoint,full,np.load(noise_root/'frequency.npy'),
                    np.load(noise_root/'psd.npy',mmap_mode='r'),events.noise_bank_index.to_numpy(int),out/'new_event_PSD_features.npy')
            else:
                encoder=finelag.encode if conf.get('feature_implementation')=='sample_resolved_zero_padded_fft_v1' else body.encode
                p,z,_=encoder(checkpoint,full)
            np.savez_compressed(saved/'new_predictions.npz',p=p,z=z,checkpoint_sha256=dev.sha(checkpoint))
            x=ev.features(f,p,z)
            for name,key in (('new_encoder_cosine','cosine'),('new_mass_similarity','mass'),('new_mass_predictive_BC','predictive_BC'),('new_mass_pred_logmc_i','mean_i'),('new_mass_pred_logmc_j','mean_j')):
                f[name]=x[key]
            spec=json.loads((root/f'calibration/{dep}/seed_{es}/SELECTED_CONFIG.json').read_text())
            scorer=(finite_tail.score if spec['recipe']=='finite_reference_companion_tail'
                    else conditional.score if spec['recipe']=='conditional_monotone_waveform'
                    else tail.score if spec['recipe'].startswith('companion_tail') else u.frozen_score)
            candidate,new,raw,ood=scorer(f,spec)
            f['new_calibrated_waveform_score']=new
            f['raw_new_waveform_statistic']=raw
            f['waveform_calibration_ood']=ood
            if np.array_equal(candidate,oldscore):
                raise RuntimeError('New encoder made no contribution across a full catalog')
            per=[]
            for name,s in (('BASELINE',oldscore),('CANDIDATE',candidate)):
                changed=f.copy()
                changed['waveform_score']=s
                changed.to_parquet(saved/f'{name}_pairs.parquet',index=False)
                for method,m in ev.metrics(changed,s,dep,es).items():
                    per.append({'deployment':dep,'catalog_seed':cs,'model_seed':ms,'eval_seed':es,'variant':name,'method':method,**m})
            dev.csv_write(saved/'metrics.csv',pd.DataFrame(per))
            rows.extend(per)
            dev.json_write(saved/'COMPLETE.json',{'new_model_contributes':True,'new_fit':False,'changed_pair_fraction':float((candidate!=oldscore).mean()),'ood_rate':float(ood.mean()),'time_sky_same':True})
            print(json.dumps({'scored_fresh':dep,'catalog':cs,'model':ms}),flush=True)
        del full,rawmaps
    dev.csv_write(root/f'confirmation/{dep}/metrics_all.csv',pd.DataFrame(rows))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--phase',choices=('freeze','generate','score'),required=True)
    p.add_argument('--deployment',choices=u.DEPS,default='gwtc3')
    p.add_argument('--workers',type=int,default=6)
    a=p.parse_args()
    if a.phase=='freeze':freeze(a.root)
    elif a.phase=='generate':generate(a.root,a.deployment,a.workers)
    else:score(a.root,a.deployment)
