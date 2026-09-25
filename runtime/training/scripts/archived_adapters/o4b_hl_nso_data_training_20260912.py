#!/usr/bin/env python3
"""Run-matched input preparation and short-encoder training for O4B-HL-NSO-01.

No real PE samples, real ranking, or held-out inference are accessed here.
Long-window and conditional-intrinsic stages are separate, required stages.
"""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(name,'2')
os.environ['GW_WAVEFORM_INPUT_SAMPLES']='4096'

import argparse
import fcntl
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import time
import traceback

import h5py
import numpy as np
import pandas as pd


def now():return datetime.now(timezone.utc).isoformat()


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False,default=lambda x:x.item() if isinstance(x,np.generic) else str(x))+'\n')
    tmp.replace(path)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda:f.read(8*1024**2),b''):h.update(data)
    return h.hexdigest()


def load(path,name):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s)
    sys.modules[name]=m;s.loader.exec_module(m);return m


def environment(root):
    ws=root/'workspace';sys.path.insert(0,str(ws))
    shared=ws/'results/real_noise_injection_v5_physical_source_20260721/gwtc5/shared'
    v3=load(ws/'scripts/experiments/20_real_noise_injection_v3_physical.py','o4b_v3_data')
    from scripts.real_search import physical_common as phys
    from scripts.real_search import physical_source_v5_common as source
    return ws,shared,v3,phys,source


def disk_guard(root):
    if shutil.disk_usage(root).free<40*1024**3:raise RuntimeError('HOLD_DISK_LIMIT')


def get_split(root):
    p=root/'contracts/DATA_SPLIT_CONTRACT_v2.json';f=json.loads((root/'contracts/DATA_SPLIT_FREEZE_v2.json').read_text())
    if sha(p)!=f['sha256']:raise RuntimeError('Split contract changed')
    c=json.loads(p.read_text())
    for item in c['outputs']:
        if sha(item['path'])!=item['sha256']:raise RuntimeError('Split input hash changed')
    return c


def allocate(path,shape):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        x=np.load(path,mmap_mode='r+')
        if x.shape!=shape or x.dtype!=np.float32:raise RuntimeError('Existing array schema mismatch')
        return x
    x=np.lib.format.open_memmap(path,mode='w+',dtype=np.float32,shape=shape)
    x[:]=np.nan;x.flush();return x


def build_noise(root,wait):
    ws,shared,v3,phys,source=environment(root);get_split(root)
    output=shared/'noise_bank_complete.json'
    if output.exists():print(output.read_text(),flush=True);return
    table=pd.read_parquet(root/'tables/NOISE_BANK_SELECTION_v2.parquet')
    refs=allocate(shared/'noise_reference_bank.npy',(len(table),2,256*4096))
    psds_path=shared/'noise_psd_bank.npy'
    if psds_path.exists():psds=np.load(psds_path,mmap_mode='r+')
    else:
        psds=np.lib.format.open_memmap(psds_path,mode='w+',dtype=np.float64,shape=(len(table),2,16385));psds[:]=np.nan;psds.flush()
    pending=set(range(len(table)));started=time.monotonic()
    while pending:
        disk_guard(root);progress=False
        for idx in sorted(pending):
            row=table.iloc[idx];marker=shared/'noise_block_audits'/f'{idx:04d}.json'
            if marker.exists():
                audit=json.loads(marker.read_text())
                digest=hashlib.sha256(np.asarray(refs[idx]).tobytes()).hexdigest()
                if digest!=audit['reference_array_sha256']:raise RuntimeError('Noise checkpoint changed')
                pending.remove(idx);continue
            paths=[Path(row.h1_path),Path(row.l1_path)]
            if not all(p.exists() and (root/'manifests/downloads'/f'{p.name}.json').exists() for p in paths):continue
            values=[]
            for det,p in enumerate(paths):
                download=json.loads((root/'manifests/downloads'/f'{p.name}.json').read_text())
                if download['state']!='VERIFIED':raise RuntimeError('Unverified original input')
                with h5py.File(p,'r') as f:
                    d=f['strain/Strain'];gps=float(d.attrs['Xstart']);rate=1/float(d.attrs['Xspacing'])
                    if rate!=4096:raise RuntimeError('Raw sampling mismatch')
                    a=int(round((row.start_gps-gps)*rate));value=d[a:a+256*4096]
                if value.shape!=(256*4096,) or not np.isfinite(value).all() or np.std(value)==0:
                    write(root/'contracts/NOISE_INPUT_FAILURE.json',{'index':idx,'path':str(p),'uid':row.noise_block_uid,'finite_fraction':float(np.isfinite(value).mean()),'utc':now()})
                    raise RuntimeError('Frozen noise window failed finite/nonzero gate')
                values.append(value)
            arr=np.stack(values);freq,psd=phys.estimate_psd(arr)
            if not np.isfinite(psd).all() or np.any(psd<=0):raise RuntimeError('Invalid PSD')
            refs[idx]=arr.astype(np.float32);psds[idx]=psd;refs.flush();psds.flush()
            if not (shared/'noise_psd_frequency.npy').exists():np.save(shared/'noise_psd_frequency.npy',freq)
            audit={**row.to_dict(),'utc':now(),'finite_fraction':1.,'reference_array_sha256':hashlib.sha256(np.asarray(refs[idx]).tobytes()).hexdigest(),
                   'PSD':'8s Hann Welch,50% overlap,one-sided density; original physical_common.estimate_psd',
                   'PSD_sha256':hashlib.sha256(psd.tobytes()).hexdigest()}
            write(marker,audit);pending.remove(idx);progress=True
            print(json.dumps({'phase':'noise_bank','complete':len(table)-len(pending),'total':len(table),'split':row.split}),flush=True)
        if pending:
            write(root/'contracts/NOISE_BUILD_STATUS.json',{'state':'WAIT_INPUT_FILES','pending_blocks':len(pending),'complete_blocks':len(table)-len(pending),'utc':now()})
            if not wait:return
            if not progress:time.sleep(30)
    files=[shared/f for f in ['noise_reference_bank.npy','noise_psd_bank.npy','noise_psd_frequency.npy']]
    table.to_parquet(shared/'noise_reference_metadata.parquet',index=False)
    write(output,{'state':'PASS','n_blocks':len(table),'seconds':time.monotonic()-started,
                  'split_counts':table.groupby('split').size().to_dict(),'finite_fraction':1.,
                  'raw_file_cross_split_intersection':0,'files':[{'path':str(p),'sha256':sha(p)} for p in files]})
    print(output.read_text(),flush=True)


def training_contract(root):
    path=root/'contracts/SHORT_ENCODER_TRAINING_CONTRACT.json'
    if path.exists():return
    write(path,{'created_utc':now(),'architecture':'InceptionAttentionEncoder1D(2,192,96,width_scale=1,depth=6)',
                'input_shape':[2,4096],'input_seconds':2,'model_rate_hz':2048,'window_relative_to_tc_s':[-1.75,.25],
                'preprocess':'same frozen physical_common 40-580Hz/offsource Welch PSD/Tukey/anti-alias/robust scaling, then short crop/peak sign/zscore',
                'clean_pretrain_epochs':16,'real_noise_adapt_epochs':60,'batch_size':48,
                'train_noise_variants_per_source':8,'aux_loss_weight':.25,'mass_ratio_loss_weight':1.,
                'target_parameters':['log(Mc_detector)','logit(q)'],'model_seeds':[2026091221,2026091222,2026091223],
                'checkpoint_selection':'validation hybridR10,hybridR1,embeddingR10,embeddingR1,-medianRank,effectiveRank',
                'selection_data':'SOURCE_SPLIT_v2 val only; no locked test and no real PE',
                'original_implementation':'workspace/scripts/real_search/37_unified_intrinsic_multitask_pilot.py',
                'input_array_storage':'compact catalog retains24s; training-noise variants store only exact trailing4096 samples consumed by this encoder',
                'long_window_stage_still_required':True,'complete_NEW_SCORE_ONLY':False})
    write(root/'contracts/SHORT_ENCODER_TRAINING_FREEZE.json',{'path':str(path),'sha256':sha(path)})


def materialize(root,seed,phase):
    ws,shared,v3,phys,source=environment(root);split=get_split(root);training_contract(root)
    catalog_seed={2026091221:2026091231,2026091222:2026091232,2026091223:2026091233}[seed]
    if not (shared/'noise_bank_complete.json').exists():raise RuntimeError('Noise bank incomplete')
    seedroot=shared.parent/f'seed_{seed}';data=seedroot/'data/real_noise_injections';data.mkdir(parents=True,exist_ok=True)
    manifest=ws/'runs/real_gwtc5_o4b_v93_extension_20260830/data/event_manifest.csv'
    if not (seedroot/'data/event_manifest.csv').exists():shutil.copy2(manifest,seedroot/'data/event_manifest.csv')
    empirical=source.empirical_detected_snrs(manifest)
    refs=np.load(shared/'noise_reference_bank.npy',mmap_mode='r');psds=np.load(shared/'noise_psd_bank.npy',mmap_mode='r')
    freq=np.load(shared/'noise_psd_frequency.npy');noise=pd.read_parquet(shared/'noise_reference_metadata.parquet')
    sm=pd.read_parquet(root/'tables/SOURCE_SPLIT_v2.parquet');bank=shared/'physical_h1l1_source_bank'
    marker=data/f'{phase}_complete.json'
    if marker.exists():print(marker.read_text(),flush=True);return
    all_rows=[];started=time.monotonic();base=data/'matchroots/LIGO'
    families=['SIS','PM','unlensed'] if phase=='catalog' else ['SIS','PM']
    for family in families:
        meta=sm[sm.family==family].sort_values('source_index').reset_index(drop=True)
        folder='Unlensed_data_0222' if family=='unlensed' else f'{family}_data_0222'
        src=bank/folder;dest=base/folder;dest.mkdir(parents=True,exist_ok=True)
        image_count=1 if family=='unlensed' else 2
        clean=[np.load(src/(f'{family}_h_strain_{image+1}.npy' if image_count==2 else 'unlensed_h_strain.npy'),mmap_mode='r') for image in range(image_count)]
        ids=list(range(600)) if phase=='catalog' else split['index_split']['train']
        variants=1 if phase=='catalog' else 8
        outputs={}
        if phase=='catalog':
            for image in range(image_count):
                for kind in ('pure','mixed'):
                    name=(f'{family}_{"h" if kind=="pure" else "data"}_strain_{image+1}.npy' if image_count==2 else f'unlensed_{"h" if kind=="pure" else "data"}_strain.npy')
                    outputs[(image,kind)]=allocate(dest/name,(600,2,phys.MODEL_SAMPLES))
        else:
            dest=data/f'multinoise_{family.lower()}_train_v8';dest.mkdir(parents=True,exist_ok=True)
            outputs[(0,'mixed')]=allocate(dest/'noisy_image_a.npy',(len(ids)*variants,2,4096))
            outputs[(1,'mixed')]=allocate(dest/'noisy_image_b.npy',(len(ids)*variants,2,4096))
        rows=[]
        for number,idx in enumerate(ids):
            disk_guard(root);row=meta.iloc[idx]
            if row.source_index!=idx:raise RuntimeError('Metadata/array source index mismatch')
            allowed=noise[noise.split==row.split].noise_bank_index.to_numpy(int)
            for variant in range(variants):
                tag=f'{catalog_seed}:{phase}:{family}:{idx}:{variant}'
                rng=np.random.default_rng(int.from_bytes(hashlib.sha256(tag.encode()).digest()[:8],'little'))
                snrs=source.draw_detected_pair_snrs(row,empirical,rng) if image_count==2 else [source.draw_unlensed_snr(empirical,rng)]
                used_parent=None;record={'model_seed':seed,'catalog_seed':catalog_seed,'family':family,'sample_index':idx,'variant':variant,'split':row.split,'global_source_id':row.global_source_id,
                    'gwlmc_row':int(row.gwlmc_row),'physical_lens_group':row.physical_lens_group if family!='unlensed' else 'unlensed_control',
                    'gps_image1':float(row.gps_image1 if family!='unlensed' else row.gps),'gps_image2':float(row.gps_image2) if image_count==2 else None,
                    'ra_true':float(row.ra),'dec_true':float(row.dec),'delay_days':float(row.delay_days) if image_count==2 else None,
                    'target_snr_image1':float(snrs[0]),'target_snr_image2':float(snrs[1]) if image_count==2 else None,
                    'snr_ratio_prior':float(row.proposal_snr_ratio) if image_count==2 else None}
                for image in range(image_count):
                    eligible=[int(b) for b in allowed if noise.iloc[int(b)].parent_raw_file_group!=used_parent]
                    if not eligible:raise RuntimeError('Cannot draw separate noise parent for companion image')
                    ni=int(rng.choice(eligible));used_parent=noise.iloc[ni].parent_raw_file_group
                    offset=int(rng.integers(0,refs.shape[-1]-phys.RAW_PADDED_SAMPLES+1))
                    nw=np.asarray(refs[ni,:,offset:offset+phys.RAW_PADDED_SAMPLES])
                    if not np.isfinite(clean[image][idx]).all() or not np.isfinite(nw).all():raise RuntimeError('Nonfinite injection input')
                    scaled,factor,recovered=phys.scale_to_network_snr(clean[image][idx],snrs[image],freq,psds[ni])
                    padded=phys.embed_signal_in_padded_window(scaled)
                    mixed=phys.preprocess_24s(nw+padded,freq,psds[ni])
                    if not np.isfinite(mixed).all():raise RuntimeError('Nonfinite preprocessing output')
                    oi=idx if phase=='catalog' else number*variants+variant
                    outputs[(image,'mixed')][oi]=mixed if phase=='catalog' else mixed[:,-4096:]
                    if phase=='catalog':outputs[(image,'pure')][oi]=phys.preprocess_24s(padded,freq,psds[ni])
                    record.update({f'image{image+1}_noise_bank_index':ni,f'image{image+1}_noise_offset_samples':offset,
                        f'image{image+1}_noise_parent':used_parent,f'image{image+1}_noise_start_gps':float(noise.iloc[ni].start_gps+offset/4096),
                        f'image{image+1}_physical_strain_scale_factor':factor,f'image{image+1}_recovered_optimal_network_snr':recovered})
                    if image_count==2:record[f'morse_image{image+1}']=float(row[f'morse_image{image+1}'])
                rows.append(record)
            if (number+1)%25==0:
                for x in outputs.values():x.flush()
                print(json.dumps({'phase':phase,'seed':seed,'family':family,'sources':number+1,'total':len(ids),'seconds':time.monotonic()-started}),flush=True)
        for x in outputs.values():x.flush()
        frame=pd.DataFrame(rows)
        if phase=='catalog':
            for image in range(image_count):
                name=f'{family}_optimal_SNR_network_{image+1}.npy' if image_count==2 else 'unlensed_optimal_SNR_network.npy'
                np.save(dest/name,frame[f'target_snr_image{image+1}'].to_numpy(np.float32))
        else:
            np.save(dest/'source_index.npy',frame.sample_index.to_numpy(np.int32));np.save(dest/'variant.npy',frame.variant.to_numpy(np.int16))
            frame.to_parquet(dest/'multinoise_metadata.parquet',index=False)
            write(dest/'multinoise_summary.json',{'n_sources':len(ids),'variants':8,'split':'train','stored_samples':4096,'input_crop_exact':True})
        all_rows.extend(rows);del outputs
    frame=pd.DataFrame(all_rows)
    frame.to_parquet(data/('compact_injection_metadata.parquet' if phase=='catalog' else 'training_variants_metadata.parquet'),index=False)
    for column in ('image1_noise_bank_index','image2_noise_bank_index'):
        used=frame.dropna(subset=[column])
        if used.groupby(column).split.nunique().max()!=1:raise RuntimeError('Noise bank reused across splits')
    write(marker,{'state':'PASS','seed':seed,'catalog_seed':catalog_seed,'phase':phase,'rows':len(frame),'seconds':time.monotonic()-started,
                  'source_noise_split_verified':True,'locked_test_inference_performed':False,
                  'snr_note':'replicates archived per-image target-SNR population; not response-derived common-amplitude sampling',
                  'SNR_reference':'run-specific catalog network-SNR marginal, including mixed published network provenance',
                  'new_score_only_complete':False})
    print(marker.read_text(),flush=True)


def train(root,seed):
    ws,shared,v3,phys,source=environment(root);contract=get_split(root);training_contract(root)
    seedroot=shared.parent/f'seed_{seed}';out=root/'models/short_encoder'/f'seed_{seed}'
    for p in ['catalog_complete.json','multinoise_complete.json']:
        if not (seedroot/'data/real_noise_injections'/p).exists():raise RuntimeError('Training data not complete')
    if (out/'summary.json').exists():print((out/'summary.json').read_text(),flush=True);return
    import torch
    torch.set_num_threads(4)
    module=load(ws/'scripts/real_search/37_unified_intrinsic_multitask_pilot.py','o4b_unified_train')
    def frozen_split(n_lensed,n_unlensed,cfg):
        if n_lensed!=600 or n_unlensed!=600:raise RuntimeError('Split size mismatch')
        return {tag:{s:np.array(ids,dtype=np.int64) for s,ids in contract['index_split'].items()} for tag in ('lensed','unlensed')}
    module.split_indices=frozen_split
    out.mkdir(parents=True,exist_ok=True)
    write(out/'STARTED.json',{'utc':now(),'seed':seed,'test_scored':False,'real_PE_used':False,
                            'data_split_contract_sha256':sha(root/'contracts/DATA_SPLIT_CONTRACT_v2.json')})
    sys.argv=[str(module.__file__),'--seed-root',str(seedroot),'--source-bank',str(shared/'physical_h1l1_source_bank'),
        '--seed',str(seed),'--samples','600','--pretrain-epochs','16','--adapt-epochs','60','--batch-size','48',
        '--aux-weight','0.25','--q-loss-weight','1','--backbone','inception_attention','--out-dir',str(out)]
    module.main()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--stage',choices=['noise','catalog','multinoise','train'],required=True)
    p.add_argument('--seed',type=int,choices=[2026091221,2026091222,2026091223],default=2026091221)
    p.add_argument('--wait-inputs',action='store_true');a=p.parse_args()
    try:
        if a.stage=='noise':build_noise(a.root,a.wait_inputs)
        elif a.stage=='train':
            with (a.root/'contracts/GPU_TRAINING.lock').open('a') as stream:
                fcntl.flock(stream,fcntl.LOCK_EX)
                train(a.root,a.seed)
        else:materialize(a.root,a.seed,a.stage)
    except Exception:
        write(a.root/'contracts'/f'FAILURE_{a.stage}_{a.seed}_{int(time.time())}.json',{'utc':now(),'stage':a.stage,'seed':a.seed,'traceback':traceback.format_exc()})
        raise
