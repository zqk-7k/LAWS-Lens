#!/usr/bin/env python3
"""Read-only waveform reconstruction and nominal-versus-input-band SNR audit."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import mcwf_expanded_data_20260907 as generation

dev = generation.dev


def run(root):
    out = root / 'audit' / ('input_band_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    out.mkdir()
    shutil.copy2(__file__, out / Path(__file__).name)
    dev.json_write(out / 'CONTRACT.json', {
        'selection': 'six training waveform-source parents per pre-existing mass bin, deterministic SHA256 order; both images, noise view0',
        'source_of_truth': 'frozen source parameters, PSD, raw noise block/offset and target SNR; no PE or official candidates',
        'reconstruction': 'same physical source/detector/Morse generation and original40-580Hz preprocessing; require bit-exact storedfloat16 peak2s',
        'SNR_bands_Hz': [[20,1024],[20,580],[40,580],[20,40],[580,1024]],
        'partition_edges': 'low band [20,40), central band [40,580], high band (580,1024]; no frequency bin counted twice',
        'limits': 'Sharp-band optimal SNR of the complete24s physical signal. This is NOT the exact soft-filtered/cropped/normalized encoder-input SNR, a Fisher bound, or a claim that a new band improves PE.',
        'formal_input_changed': False, 'new_model_or_candidate_ranking': False})
    rows = []
    for dep in ('gwtc3','gwtc4'):
        generation.init_worker(str(root),dep,'train')
        ctx = generation.CTX
        src,v3 = ctx['src'],ctx['v3']
        snr_function = v3.scale_to_network_snr.__globals__['optimal_network_snr']
        folder = root / f'expanded_data/{dep}/train'
        plan = pd.read_parquet(folder / 'source_plan.parquet')
        plan['audit_hash'] = plan.source_uid.map(lambda value: hashlib.sha256(('BAND-AUDIT-20260907|' + value).encode()).hexdigest())
        selected = plan.sort_values('audit_hash').groupby('mass_bin',sort=True).head(6)
        meta = pd.read_parquet(folder / 'event_metadata.parquet')
        meta = meta[meta.source_uid.isin(selected.source_uid) & meta.noise_variant.eq(0)].copy()
        raw = np.load(folder / 'raw2s.npy',mmap_mode='r')
        selected.to_parquet(out / f'{dep}_SELECTED_SOURCES.parquet',index=False)
        for record in meta.to_dict('records'):
            item = pd.Series(record)
            image = item['image']
            number = 1 if image == 'a' else 2
            clean,_ = src.detector_response(ctx['generator'],ctx['ifos'],
                src.source_parameters(item,item[f'gps_{image}']),
                src.lens_factor(item[f'mu_image{number}'],item[f'morse_image{number}']))
            pid,offset = int(item.noise_bank_index),int(item.noise_offset_samples)
            noise = np.asarray(ctx['refs'][pid,:,offset:offset+v3.RAW_PADDED_SAMPLES],dtype=np.float64)
            _,prepared,audit = v3._preprocess_injection(clean,noise,ctx['freq'],ctx['psds'][pid],float(item.target_network_snr))
            reconstructed = dev.TRAIN.make_window_view(prepared[None].astype(np.float32),2)[0].astype(np.float16)
            old = raw[int(item.row_index)]
            scale = audit['physical_strain_scale_factor']
            scaled = np.asarray(clean,dtype=np.float32).astype(np.float64)*scale
            snrs = {(low,high): snr_function(scaled,ctx['freq'],ctx['psds'][pid],
                        f_low=np.nextafter(580.,np.inf) if low==580 else low,
                        f_high=np.nextafter(40.,-np.inf) if high==40 else high)
                    for low,high in ((20,1024),(20,580),(40,580),(20,40),(580,1024))}
            rows.append({'deployment':dep,'source_uid':item.source_uid,'image':image,'mass_bin':item.mass_bin,
                'mc_det':item.mc_det,'noise_bank_index':pid,'row_index':int(item.row_index),
                'target_network_snr_20_1024':item.target_network_snr,
                'stored_peak2s_bit_exact':np.array_equal(reconstructed,old),
                'stored_peak2s_max_abs_difference':float(abs(reconstructed.astype(float)-old.astype(float)).max()),
                'scale_relative_difference':float(abs(scale/item.physical_strain_scale_factor-1)),
                **{f'optimal_snr_{lo}_{hi}':value for (lo,hi),value in snrs.items()},
                'rho_40_580_over_20_1024':snrs[40,580]/snrs[20,1024],
                'rho2_20_40_fraction':(snrs[20,40]/snrs[20,1024])**2,
                'rho2_580_1024_fraction':(snrs[580,1024]/snrs[20,1024])**2})
        print(json.dumps({'band_audit':dep,'events':len(meta),'sources':meta.source_uid.nunique()}),flush=True)
    frame = pd.DataFrame(rows)
    dev.csv_write(out / 'EVENT_INPUT_BAND_AUDIT.csv',frame)
    summaries = []
    for (dep,bin_id),f in frame.groupby(['deployment','mass_bin']):
        summaries.append({'deployment':dep,'mass_bin':bin_id,'source_systems':f.source_uid.nunique(),'images':len(f),
            'median_Mc':f.mc_det.median(),'median_rho_fraction':f.rho_40_580_over_20_1024.median(),
            'p10_rho_fraction':f.rho_40_580_over_20_1024.quantile(.1),'p90_rho_fraction':f.rho_40_580_over_20_1024.quantile(.9),
            'median_low_band_rho2_fraction':f.rho2_20_40_fraction.median(),
            'median_high_band_rho2_fraction':f.rho2_580_1024_fraction.median()})
    dev.csv_write(out / 'BAND_SNR_SUMMARY.csv',pd.DataFrame(summaries))
    report = {'events':len(frame),'source_systems':frame.source_uid.nunique(),
        'reconstructed_bit_exact':int(frame.stored_peak2s_bit_exact.sum()),
        'reconstruction_pass':bool(frame.stored_peak2s_bit_exact.all()),
        'input_band_changed':False,'interpretation':'diagnostic only; all formal40-580Hz models and rankings remain unchanged',
        'sampling_uncertainty':'small source-stratified diagnostic; two images per source are not independent draws'}
    dev.json_write(out / 'SUMMARY.json',report)
    print(json.dumps({'output':str(out),**report}),flush=True)
    if not report['reconstruction_pass']:
        raise RuntimeError('Physical reconstruction did not match stored peak2s; no new input-band experiment may assume equivalence')


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();run(a.root)
