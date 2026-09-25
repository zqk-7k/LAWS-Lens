#!/usr/bin/env python3
"""Read-only reconstruction of every available two-detector real waveform input."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import time
import numpy as np
import pandas as pd
import mcwf_pe_frontend_extension_v2_20260907 as e

dev=e.dev


def run(root):
    out=root/'audit'/('real_input_reconstruction_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    out.mkdir();shutil.copy2(__file__,out/Path(__file__).name)
    v3=dev.BASE.v7.v3
    dev.json_write(out/'CONTRACT.json',{'mode':'read-only audit;no cache rebuilding or candidate score changes',
        'events':'all waveform-valid events returned by frozen real_inputs,including O4a waveform-valid non-BBH if present;strict-scored scope separately flagged',
        'method':'read original H1/L1 strain,extract same26s onsource and256s offsource reference,estimate PSD,run frozen v3preprocess24s and frozen peak2s view',
        'comparisons':'row/GPS/path/PSD-reference agreement;bitwise and numerical full24s/2s errors',
        'negative_result':'any mismatch reported without rewriting historical data;no scientific result is repaired silently',
        'PE_or_official_selection':False})
    rows=[];started=time.perf_counter()
    for dep in e.DEPS:
        full,events=dev.real_inputs(dep)
        pe=pd.read_parquet(root/f'audit/{dep}_frozen_external_reference.parquet')
        scope=set(pe.event_i)|set(pe.event_j)
        cache=v3.HdfCache(dev.MAIN/'cache/source_run' if dep=='gwtc3' else v3.SOURCES['GWTC4'],max_files=2)
        for record in events.to_dict('records'):
            if not record['strict_h1l1_preprocessing_pass']:
                continue
            row={'deployment':dep,'event_name':record['event_name'],'idx':int(record['idx']),
                 'gps_time':record['gps_time'],'in_strict_scored_scope':record['event_name'] in scope}
            try:
                audit=json.loads(record['detector_audit'])
                channels,refs=[],[]
                for det in ('H1','L1'):
                    entry=next(x for x in audit if x['detector']==det)
                    data,start,duration=cache.get(entry['path'])
                    if abs(len(data)/duration-v3.RAW_SAMPLE_RATE)>1e-3:
                        raise RuntimeError('Sample-rate mismatch')
                    wave=v3._extract_channel_window(data,start,float(record['gps_time']),v3.RAW_PADDED_SAMPLES,-24.75)
                    reference,reference_start=v3._extract_psd_reference(data,start,float(record['gps_time']))
                    if wave is None or reference is None:
                        raise RuntimeError('Missing complete onsource or offsource window')
                    row[det+'_psd_reference_GPS_difference']=float(reference_start-entry['psd_reference_start_gps'])
                    row[det+'_strain_path']=entry['path']
                    channels.append(wave);refs.append(reference)
                frequency,psd=v3.estimate_psd(np.stack(refs))
                rebuilt=v3.preprocess_24s(np.stack(channels),frequency,psd).astype(np.float32)
                old=np.asarray(full[row['idx']],np.float32)
                a=dev.TRAIN.make_window_view(rebuilt[None],2)[0]
                b=dev.TRAIN.make_window_view(old[None],2)[0]
                row.update(read_success=True,full24_bit_exact=np.array_equal(rebuilt,old),peak2s_bit_exact=np.array_equal(a,b),
                           full24_max_abs_difference=float(abs(rebuilt-old).max()),peak2s_max_abs_difference=float(abs(a-b).max()),
                           peak2s_relative_l2_difference=float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-12)),
                           peak2s_H1_abs_peak_index=int(np.argmax(abs(b[0]))),peak2s_L1_abs_peak_index=int(np.argmax(abs(b[1]))))
            except Exception as exc:
                row.update(read_success=False,error=type(exc).__name__+': '+str(exc))
            rows.append(row)
            if len(rows)%20==0:
                print(json.dumps({'real_input_audit_events':len(rows),'seconds':time.perf_counter()-started}),flush=True)
        del cache
    frame=pd.DataFrame(rows);dev.csv_write(out/'EVENT_RECONSTRUCTION.csv',frame)
    summary={'events':len(frame),'strict_scored_events':int(frame.in_strict_scored_scope.sum()),'read_success':int(frame.read_success.sum()),
             'full24_bit_exact':int(frame.full24_bit_exact.fillna(False).sum()),'peak2s_bit_exact':int(frame.peak2s_bit_exact.fillna(False).sum()),
             'maximum_peak2s_difference':float(frame.peak2s_max_abs_difference.max()),
             'seconds':time.perf_counter()-started,'historical_arrays_modified':False}
    dev.json_write(out/'SUMMARY.json',summary);print(json.dumps({'output':str(out),**summary}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();run(a.root)
