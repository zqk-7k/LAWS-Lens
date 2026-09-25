#!/usr/bin/env python3
"""Strict H1/L1 waveform extraction from checksum-verified public O4b strain."""
import argparse
from pathlib import Path
import json

import h5py
import numpy as np
import pandas as pd

import o4b_hl_nso_data_training_20260912 as s
import o4b_hl_nso_inference_20260912 as inference
import o4b_hl_nso_multiscale_training_20260912 as models


def read_window(row, root):
    path = Path(row.path)
    checked = root/'manifests/downloads'/f'{path.name}.json'
    if not checked.exists() or json.loads(checked.read_text())['state'] != 'VERIFIED':
        raise RuntimeError('Original strain not checksum-verified')
    with h5py.File(path, 'r') as f:
        d = f['strain/Strain']
        fs = 1/float(d.attrs['Xspacing']); gps = float(d.attrs['Xstart'])
        if fs != 4096:
            raise RuntimeError('Original strain rate mismatch')
        first = int(round((row.start-gps)*fs))
        value = np.asarray(d[first:first+int(row.samples)], dtype=np.float32)
    if len(value) != row.expected or not np.isfinite(value).all() or np.std(value) == 0:
        raise RuntimeError('Strict finite/nonzero original window failed')
    return value


def run(root):
    inference.guard(root, 'real')
    out = root/'inference_inputs/real'; out.mkdir(parents=True, exist_ok=True)
    if (out/'COMPLETE.json').exists():
        for item in json.loads((out/'COMPLETE.json').read_text())['outputs']:
            if s.sha(item['path']) != item['sha256']:
                raise RuntimeError('Frozen real input changed')
        return
    _, _, _, phys, _ = s.environment(root)
    low, *_ = models.initialize(root)
    short = inference.short_module(root)
    run = root/'workspace/runs/real_gwtc5_o4b_v93_extension_20260830'
    manifest = pd.read_csv(run/'data/event_manifest.csv')
    windows = pd.read_csv(root/'tables/INPUT_WINDOW_AUDIT.csv')
    if not windows.passed.all() or len(manifest) != 86 or len(windows) != 344:
        raise RuntimeError('Frozen strict-HL scope changed')
    raw, short_raw, long, psds, events, audits = [], [], [], [], [], []
    for idx, row in enumerate(manifest.itertuples()):
        ws = windows[windows.event == row.event_name]
        values = {}
        for kind in ws.window.unique():
            selected = ws[ws.window == kind].set_index('detector').loc[['H1', 'L1']]
            values[kind] = np.stack([read_window(v, root) for v in selected.itertuples()])
        if 'onsource' not in values or len(values) != 2:
            raise RuntimeError('Ambiguous on/off-source input windows')
        reference = next(v for k, v in values.items() if k != 'onsource')
        frequency, psd = phys.estimate_psd(reference)
        full = phys.preprocess_24s(values['onsource'], frequency, psd)
        raw.append(low.dev.TRAIN.make_window_view(full[None], 2)[0])
        short_raw.append(short.prepare(full, None, False))
        long.append(low.low_view(values['onsource'], frequency, psd, phys))
        psds.append(psd)
        events.append({'idx': idx, 'event_uid': row.event_name, 'event_name': row.event_name,
                       'gps_time': row.gps_time, 'noise_bank_index': idx,
                       'detectors': row.detectors, 'strict_h1l1_preprocessing_pass': True,
                       'sky_map_path': row.sky_map_path})
        audits.append({'event': row.event_name, 'input_channels': 'H1,L1',
                       'silent_fill': False, 'short_samples': 4096, 'long_samples': 4096,
                       'finite': True, 'reference_samples': reference.shape[-1],
                       'same_original_strain_short_and_long': True,
                       'source_files': '|'.join(sorted(ws.path.unique()))})
    for name, data in [('raw2s', raw), ('short_raw2s', short_raw), ('low16s', long), ('psd', psds), ('frequency', frequency)]:
        np.save(out/f'{name}.npy', np.asarray(data))
    pd.DataFrame(events).to_parquet(out/'events.parquet', index=False)
    pd.DataFrame(audits).to_csv(out/'preprocessing_audit.csv', index=False)
    s.write(out/'COMPLETE.json', {'utc': s.now(), 'state': 'PASS', 'strict_events': len(events),
                                'PE_parameters_used': False, 'rankings_computed': False,
                                'outputs': [{'path': str(p), 'sha256': s.sha(p)} for p in out.glob('*') if p.is_file()]})


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True)
    run(p.parse_args().root)
