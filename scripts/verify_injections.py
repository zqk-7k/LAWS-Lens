"""Regenerate one fixed validation doublet per run from source and noise."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

class IndexedNoise:
    def __init__(self, arrays):
        self.arrays = arrays
    def __getitem__(self, key):
        bank, channel, samples = key
        return self.arrays[str(bank)][channel, samples]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--release', required=True, type=Path)
    ap.add_argument('--baseline', type=Path)
    ap.add_argument('--output-tag', default='injection_regeneration')
    a = ap.parse_args()
    R = a.release
    T = R/'runtime/training'
    P = R/'runtime/project'
    sys.path[:0] = [str(T/'scripts'), str(T/'scripts/archived_adapters'),
                   str(P), str(P/'scripts/experiments')]
    import numpy as np
    import pandas as pd
    import unified_ab as u
    import c_data as data
    import legacy_unified_ab
    import logging
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)
    legacy_unified_ab.P = P
    dest = R/'verification'/a.output_tag
    dest.mkdir(parents=True, exist_ok=False)
    (dest/'scripts').symlink_to(T/'scripts', target_is_directory=True)
    (dest/'plans').mkdir()
    shutil.copy2(T/'plans/C_PILOT_SOURCES.parquet', dest/'plans/C_PILOT_SOURCES.parquet')
    reports = []
    for run in ('O3', 'O4a', 'O4b'):
        fixture = R/'fixtures/injections'/run
        if a.baseline:
            sources = pd.read_parquet(T/f'plans/{run}/sources.parquet')
            row = sources[(sources.role == 'main') & (sources.split == 'validation') &
                          (sources.family != 'unlensed')].sort_values('source_uid').iloc[0].to_dict()
            events = pd.read_parquet(T/f'plans/{run}/event_plan.parquet')
            events = events[events.source_uid.eq(row['source_uid'])]
            fixture.mkdir(parents=True, exist_ok=False)
            pd.DataFrame([row]).to_parquet(fixture/'source.parquet', index=False)
            events.to_parquet(fixture/'events.parquet', index=False)
            bank = np.load(a.baseline/f'plans/{run}/noise_reference_bank.npy', mmap_mode='r')
            indices = sorted(set(events.noise_bank_index.astype(int)))
            np.savez_compressed(fixture/'noise.npz', **{str(i): bank[i] for i in indices})
            np.save(fixture/'psds.npy', np.load(a.baseline/f'plans/{run}/noise_psd_bank.npy'))
            np.save(fixture/'frequency.npy', np.load(a.baseline/f'plans/{run}/noise_psd_frequency.npy'))
            old = a.baseline/f'data/{run}/main/validation'/row['source_uid']
            for name in ('C_PHYSICAL_short.npy', 'C_PHYSICAL_long.npy', 'C_PHYSICAL_clean.npy'):
                shutil.copy2(old/name, fixture/name)
            raw_hashes = {}
            for p in old.glob('*_raw.npz'):
                with np.load(p) as z:
                    raw_hashes[p.name] = {k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in z.items()}
            (fixture/'EXPECTED.json').write_text(json.dumps(dict(
                source_uid=row['source_uid'], source_plan_sha256=sha(T/f'plans/{run}/sources.parquet'),
                raw_array_sha256=raw_hashes, selection='lexicographically first main validation lensed source',
                noise_parent_bank_indices=indices, raw_noise_not_redownloaded=True), indent=2))
        row = pd.read_parquet(fixture/'source.parquet').iloc[0].to_dict()
        events = pd.read_parquet(fixture/'events.parquet')
        source, sampling, views, physical = u.modules(dest)
        source.DURATION, source.N_SAMPLES = 64, 64*4096
        source.END_AFTER_GEOCENTER_SECONDS = 16
        logging.getLogger('bilby').setLevel(logging.ERROR)
        with np.load(fixture/'noise.npz') as z:
            noise = IndexedNoise(dict(z))
        data.STATE.clear()
        data.STATE.update(root=dest, run=run, u=u, source=source, views=views,
            physical=physical, generator=source.build_waveform_generator(),
            ifos=list(source.bilby.gw.detector.InterferometerList(['H1', 'L1'])),
            events={row['source_uid']: events}, noise=noise,
            psds=np.load(fixture/'psds.npy'), frequency=np.load(fixture/'frequency.npy'))
        t = time.monotonic()
        receipt = data.generate_source(row)
        folder = dest/f'data/{run}/main/validation'/row['source_uid']
        arrays = []
        for name in ('C_PHYSICAL_short.npy', 'C_PHYSICAL_long.npy', 'C_PHYSICAL_clean.npy'):
            actual, expected = np.load(folder/name), np.load(fixture/name)
            diff = float(np.max(np.abs(actual-expected)))
            np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)
            arrays.append(dict(file=name, max_abs=diff, exact_sha256=sha(folder/name)==sha(fixture/name)))
        spec = json.loads((fixture/'EXPECTED.json').read_text())
        raw_checks = []
        for name, values in spec['raw_array_sha256'].items():
            with np.load(folder/name) as z:
                raw_checks.extend(dict(file=name, array=k,
                    exact=hashlib.sha256(z[k].tobytes()).hexdigest()==h) for k,h in values.items())
        if not all(r['exact'] for r in raw_checks):
            raise ValueError('Regenerated raw strain/PSD differs')
        reports.append(dict(run=run, status='PASS', source_uid=row['source_uid'],
            events=len(events), seconds=time.monotonic()-t, arrays=arrays, raw_checks=raw_checks,
            common_amplitude_factor=receipt['common_amplitude_factor']))
        print(run, 'REGENERATION_PASS', flush=True)
    (dest/'REPORT.json').write_text(json.dumps(dict(status='PASS', results=reports,
        full_population_regenerated=False, noise_inputs='archived real-noise bank slices; not fresh download',
        sky_recovery_rerun=False, all_training_rerun=False), indent=2))

if __name__ == '__main__':
    main()
