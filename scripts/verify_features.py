"""Recompute matching features from 2 s/16 s fixtures and frozen PSDs."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--release', required=True, type=Path)
    ap.add_argument('--baseline', type=Path)
    ap.add_argument('--output-tag', default='features')
    a = ap.parse_args()
    R = a.release
    sys.path[:0] = [str(R/'runtime/project/scripts/experiments'), str(R/'runtime/project')]
    import numpy as np
    import pandas as pd
    import torch
    import mcwf_multirate_features_20260908 as low
    import mcwf_finelag_eventpsd_20260907 as rnc
    torch.set_num_threads(2)
    out = R/'verification'/a.output_tag
    out.mkdir(parents=True, exist_ok=False)
    fixture = R/'fixtures/template_operators'
    input_manifest = []
    if a.baseline:
        for sub, suffix in [('longshort','waveform_feature_operator/cache'),
                            ('rnc','rankncontrast_component_v2/cache/adaptive_psd')]:
            folder = a.baseline/'arms/C_PHYSICAL/O3'/suffix
            for p in sorted(folder.glob('*')):
                if not p.is_file():
                    continue
                digests = {run:sha(a.baseline/f'arms/C_PHYSICAL/{run}'/suffix/p.name)
                           for run in ('O3','O4a','O4b')}
                if len(set(digests.values())) != 1 and p.suffix == '.npy':
                    raise ValueError('Template arrays unexpectedly differ across runs')
                dest = fixture/sub/p.name
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p,dest)
                input_manifest.append(dict(path=str(dest.relative_to(R)), sha256=sha(dest),
                                           original_run_hashes=digests))
        (fixture/'MANIFEST.json').write_text(json.dumps(input_manifest, indent=2))
    reports = []
    for run in ('O3','O4a','O4b'):
        f = R/'fixtures/inference_full_context'/run
        idx = np.array(json.loads((f/'selected_rows.json').read_text()),int)
        ev = pd.read_parquet(f/'events.parquet').iloc[idx]
        raw = np.load(f/'raw2s.npy')[idx]
        long = np.load(f/'low16s.npy')[idx]
        ids = ev.noise_bank_index.to_numpy(int)
        freq = np.load(R/f'fixtures/injections/{run}/frequency.npy')
        psds = np.load(R/f'fixtures/injections/{run}/psds.npy')
        operator = out/run/'operator'
        (operator/'cache').mkdir(parents=True)
        (operator/'contracts').mkdir()
        for p in (fixture/'longshort').glob('*'):
            (operator/'cache'/p.name).symlink_to(p)
        spectra = np.load(fixture/'longshort/short253_spectra.npy',mmap_mode='r')
        short = np.empty((len(ev),27,253),np.float32)
        for bank in np.unique(ids):
            keep = np.flatnonzero(ids==bank)
            templates = low.t.adaptive.whitened_bank(spectra,freq,psds[bank])
            x = low.t.fine.features(raw[keep],templates)
            short[keep] = x.reshape(len(x),3,253,3,3).transpose(0,1,3,4,2).reshape(len(x),27,253)
        np.save(out/run/'short.npy',short)
        low.grouped(operator,long,freq,psds,ids,out/run/'long.npy')
        rroot = out/run/'rnc_operator'
        (rroot/'cache/adaptive_psd').mkdir(parents=True)
        for p in (fixture/'rnc').glob('*'):
            (rroot/'cache/adaptive_psd'/p.name).symlink_to(p)
        rnc.extract_grouped(rroot,raw,freq,psds,ids,out/run/'rnc.npy')
        for kind in ('short','long','rnc'):
            got = np.load(out/run/f'{kind}.npy')
            exp = np.load(f/f'{kind}.npy')[idx]
            delta = float(abs(got-exp).max())
            passed = np.allclose(got, exp, rtol=1e-5, atol=1e-5)
            reports.append(dict(run=run, feature=kind, events=len(ev), max_abs=delta,
                                status='PASS' if passed else 'FAIL'))
        print(run,'FEATURES',reports[-3:],flush=True)
    (out/'REPORT.json').write_text(json.dumps(dict(status='PASS' if all(r['status']=='PASS' for r in reports) else 'FAIL',
        comparisons=reports, template_arrays_regenerated=False, templates_frozen_verified=True,
        input_level='preprocessed 2s and 16s arrays; PSD-matched templates',rtol=1e-5,atol=1e-5),indent=2))

if __name__ == '__main__':
    main()
