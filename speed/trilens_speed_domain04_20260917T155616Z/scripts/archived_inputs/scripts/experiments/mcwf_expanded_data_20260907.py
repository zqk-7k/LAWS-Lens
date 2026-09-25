#!/usr/bin/env python3
"""Additional physical waveform development data, without any sky generation."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
import logging
import multiprocessing as mp
from pathlib import Path
import shutil
import time
import numpy as np
import pandas as pd
import mcwf_pe_frontend_extension_v2_20260907 as e
import mcwf_new_confirmation_20260906 as previous
import mcwf_o3_runmatched_data_20260906 as rm

dev = e.dev
COUNTS = {'train': 4096, 'validation': 512}
VIEWS = {'train': 4, 'validation': 1}
BLOCKS = {'train': 96, 'validation': 32}
DATA_SEEDS = {'train': 202609130, 'validation': 202609140}
CTX = {}


def modules():
    source = dev.module(dev.PROJECT / 'scripts/real_search/34_generate_physical_h1l1_source_bank.py', 'expanded_source')
    data = dev.module(dev.OLD / 'scripts/waveform_multiscale_data.py', 'expanded_data_helpers')
    v3 = data.load_module(data.V3_SCRIPT, 'expanded_physical_injection')
    return source, data, v3


def initialize(root):
    folder = root / 'expanded_data'
    path = folder / 'contracts/DATA_CONTRACT.json'
    if path.exists():
        return
    dev.json_write(path, {
        'created_utc': datetime.now(timezone.utc).isoformat(), 'same_O3_O4a': True,
        'purpose': 'test source and off-source-noise diversity instead of repeatedly optimizing the same 960 source parents',
        'counts_per_run': COUNTS, 'noise_variants_per_image': VIEWS, 'independent_noise256s_blocks': BLOCKS,
        'data_seeds': DATA_SEEDS,
        'source': 'fresh detector masses, spins, orientations, phase, sky and observing timestamps; coverage-balanced logMc5-200,q0.25-1,ordered component masses3-300',
        'not_population_inference': 'GW-LMC lens environments may recur; new source parameters are not new independent lens populations',
        'families': 'equal GW-LMC smooth/non-subhalo and subhalo-present, not analytic SIS/PM',
        'waveform': 'same physical IMRPhenomXPHM H1/L1 strain,24s4096Hz temporary buffer; same PSD whitening,40-580Hz,anti-aliasing,peak2s4096input',
        'SNR': 'same coverage-balanced bins8-10,10-12,12-20,20-40; per-image PSD-optimal scaling, not a response-derived SNR-ratio experiment',
        'noise': 'O3-only and O4a-only public offsource256s blocks; exclude all known historical train/validation/test and three previous confirmation families by absolute GPS with16s guard',
        'statistics': '4096 training source units, not32768 independent sources;96 training noise units, not32768 independent noise realizations',
        'storage': 'only peak2s float16 mixed input and metadata; no new full-strain archive,sky map,PE,or real-candidate data',
        'frozen': ['all historical data', 'time', 'sky', 'C-fixed weights', 'scope'],
        'selection': 'this is development data; no real PE or official labels enter source generation or network training',
    })
    (folder / 'scripts').mkdir(exist_ok=True)
    shutil.copy2(__file__, folder / 'scripts' / Path(__file__).name)


def noise(root, dep):
    out = root / f'expanded_data/{dep}/noise'
    if (out / 'COMPLETE.json').exists():
        return
    out.mkdir(parents=True, exist_ok=True)
    _, data, v3 = modules()
    exclusions = previous.exclusions(dep).to_dict('records')
    for r in pd.read_csv(e.BASE / f'confirmation/{dep}/noise/noise_manifest.csv').itertuples():
        exclusions.append({'start': r.start_gps, 'end': r.end_gps, 'origin': 'current_FRT_confirmation'})
    historical = pd.DataFrame(exclusions).drop_duplicates(['start', 'end'])
    dev.csv_write(out / 'EXCLUDED_BLOCKS.csv', historical)
    excluded = [(float(r.start)-16, float(r.end)+16) for r in historical.itertuples()]
    if dep == 'gwtc3':
        source_run = dev.MAIN / 'cache/source_run'
        pool = pd.read_csv(previous.PREVIOUS / 'data/o3_only_noise/official_O3_full_strain_offsource_pool.csv')
    else:
        source_run = data.SOURCE_RUNS[dep]
        pool = pd.read_parquet(source_run / 'data/real_noise_injections/offsource_noise_segments.parquet')
    cache = v3.HdfCache(source_run, max_files=8)
    total = sum(BLOCKS.values())
    refs = np.lib.format.open_memmap(out / 'reference.npy', mode='w+', dtype=np.float32,
                                   shape=(total, 2, v3.NOISE_REFERENCE_SAMPLES))
    psds, rows = [], []
    for k in range(total):
        parts = []
        for row in pool.to_dict('records'):
            for a, b in rm.subtract_intervals(float(row['segment_start']), float(row['segment_end']), excluded):
                if b-a >= v3.PSD_SECONDS+2:
                    parts.append({**row, 'segment_start': a, 'segment_end': b})
        if not parts:
            raise RuntimeError('Independent noise exhausted; no historical blocks may be reused')
        chosen, rejected = data.select_valid_noise_references(pd.DataFrame(parts), 1,
            np.random.default_rng(data.stable_seed(dep, 'expanded-waveform-noise', 202609130, k)),
            v3.PSD_SECONDS, cache, v3)
        row, start, ref, freq, psd = chosen[0]
        if any(start < b and start+v3.PSD_SECONDS > a for a, b in excluded):
            raise RuntimeError('Global noise overlap')
        excluded.append((start-16, start+v3.PSD_SECONDS+16))
        split = 'train' if k < BLOCKS['train'] else 'validation'
        refs[k] = ref
        psds.append(psd)
        rows.append({'bank_index': k, 'split': split, 'parent_event': str(row.event_name),
                     'start_gps': start, 'end_gps': start+v3.PSD_SECONDS,
                     'H1_path': row.H1_path, 'L1_path': row.L1_path, 'rejections': len(rejected)})
        if k % 16 == 0:
            print(json.dumps({'expanded_noise': dep, 'selected': k+1, 'total': total}), flush=True)
    refs.flush()
    np.save(out / 'psd.npy', np.stack(psds))
    np.save(out / 'frequency.npy', freq)
    dev.csv_write(out / 'noise_manifest.csv', pd.DataFrame(rows))
    dev.json_write(out / 'COMPLETE.json', {'independent_blocks': total,
        'global_GPS_overlap': 0, 'excluded_blocks': len(historical),
        'reference_sha256': dev.sha(out / 'reference.npy'), 'psd_sha256': dev.sha(out / 'psd.npy')})


def plan(root, dep, split):
    out = root / f'expanded_data/{dep}/{split}'
    out.mkdir(parents=True, exist_ok=True)
    path = out / 'source_plan.parquet'
    if path.exists():
        return pd.read_parquet(path)
    src, data, _ = modules()
    table = src.load_tables(src.GW_LMC_ROOT)
    schedule = src.load_schedule(dev.ORCH.SOURCE_ROOT / dep / 'shared/h1l1_live_schedule.csv')
    rng = np.random.default_rng(data.stable_seed(DATA_SEEDS[split], dep, 'expanded-sources'))
    environments = {False: [], True: []}
    for _, row in table.iterrows():
        images = src.choose_images(row)
        if images is not None and images['proposal_snr_ratio'] <= 4:
            environments[bool(row.lens_is_subhalo)].append((row, images))
    rows = []
    for k, (m1, m2, massbin) in enumerate(data.balanced_mass_draws(COUNTS[split], rng)):
        sub = bool(k % 2)
        available = environments[sub]
        for j in rng.permutation(len(available)):
            row, images = available[j]
            times = src.place_pair(images['delay_days'], schedule, rng)
            if times is not None:
                break
        else:
            raise RuntimeError('No valid lens environment')
        rows.append({'source_index': k, 'source_uid': f'EXPANDED_{dep}_{DATA_SEEDS[split]}_{k:05d}',
            'split': split, 'family': 'gwlmc_subhalo_present' if sub else 'gwlmc_smooth_non_subhalo',
            'gwlmc_environment_row': int(row.gwlmc_row), 'gwlmc_environment_id': int(row.event_id),
            'm1_det': m1, 'm2_det': m2, 'mc_det': data.chirp_mass(m1, m2), 'mass_bin': massbin,
            'a1': rng.uniform(0, .8), 'a2': rng.uniform(0, .8),
            'tilt1': np.arccos(rng.uniform(-1, 1)), 'tilt2': np.arccos(rng.uniform(-1, 1)),
            'theta_jn': np.arccos(rng.uniform(-1, 1)), 'phi12': rng.uniform(0, 2*np.pi),
            'phijl': rng.uniform(0, 2*np.pi), 'psi': rng.uniform(0, np.pi), 'phase': rng.uniform(0, 2*np.pi),
            'ra': rng.uniform(0, 2*np.pi), 'dec': np.arcsin(rng.uniform(-1, 1)), 'dl_source': 1000.,
            'gps_a': times[0], 'gps_b': times[1], **images})
    frame = pd.DataFrame(rows)
    frame.to_parquet(path, index=False)
    dev.json_write(out / 'PLAN_HASH.json', {'sha256': dev.sha(path), 'frozen_before_generation': True,
        'source_systems': len(frame), 'unique_waveform_parents': frame.source_uid.nunique(),
        'independent_lens_environments': frame.gwlmc_environment_row.nunique()})
    return frame


def init_worker(root, dep, split):
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=1)
    src, data, v3 = modules()
    logging.getLogger('bilby').setLevel(logging.ERROR)
    noise_dir = Path(root) / f'expanded_data/{dep}/noise'
    CTX.update(src=src, data=data, v3=v3, dep=dep, split=split,
        generator=src.build_waveform_generator(),
        ifos=[src.bilby.gw.detector.get_empty_interferometer(x) for x in ('H1', 'L1')],
        refs=np.load(noise_dir / 'reference.npy', mmap_mode='r'),
        freq=np.load(noise_dir / 'frequency.npy'), psds=np.load(noise_dir / 'psd.npy', mmap_mode='r'))


def system(row):
    start = time.perf_counter()
    src, data, v3 = CTX['src'], CTX['data'], CTX['v3']
    rng = np.random.default_rng(data.stable_seed(row['source_uid'], 'physical-noise-views'))
    split = CTX['split']
    block_start = 0 if split == 'train' else BLOCKS['train']
    waves, records = [], []
    for image, number in (('a', 1), ('b', 2)):
        clean, peaks = src.detector_response(CTX['generator'], CTX['ifos'],
            src.source_parameters(pd.Series(row), row[f'gps_{image}']),
            src.lens_factor(row[f'mu_image{number}'], row[f'morse_image{number}']))
        if not np.isfinite(clean).all():
            raise RuntimeError('Nonfinite clean strain before normalization')
        for v in range(VIEWS[split]):
            bi = block_start + int(rng.integers(BLOCKS[split]))
            reference = CTX['refs'][bi]
            offset = int(rng.integers(reference.shape[-1]-v3.RAW_PADDED_SAMPLES+1))
            noise_window = np.asarray(reference[:, offset:offset+v3.RAW_PADDED_SAMPLES], dtype=np.float64)
            snr = data.snr_draw((row['source_index']+number+v) % 4, rng)
            _, mixed, audit = v3._preprocess_injection(clean, noise_window, CTX['freq'], CTX['psds'][bi], snr)
            raw = dev.TRAIN.make_window_view(mixed[None].astype(np.float32), 2)[0]
            if raw.shape != (2, 4096) or not np.isfinite(raw).all() or (raw.std(-1) <= 0).any():
                raise RuntimeError('Invalid peak2s data')
            if abs(audit['recovered_optimal_network_snr']-snr) > 1e-4*snr:
                raise RuntimeError('PSD SNR scaling mismatch')
            waves.append(raw.astype(np.float16))
            records.append({**row, 'image': image, 'noise_variant': v, 'noise_bank_index': bi,
                            'noise_offset_samples': offset, **peaks, **audit})
    return row['source_index'], np.stack(waves), records, time.perf_counter()-start


def generate(root, dep, split, workers):
    initialize(root)
    noise(root, dep)
    frame = plan(root, dep, split)
    out = root / f'expanded_data/{dep}/{split}'
    if (out / 'COMPLETE.json').exists():
        return
    chunks = out / 'chunks'
    chunks.mkdir(exist_ok=True)
    nview = 2*VIEWS[split]
    path = out / 'raw2s.npy'
    raw = np.lib.format.open_memmap(path, mode='r+' if path.exists() else 'w+',
                                  shape=(len(frame)*nview, 2, 4096), dtype=np.float16)
    done = set()
    for marker in chunks.glob('*.parquet'):
        done.update(pd.read_parquet(marker, columns=['source_index']).source_index.unique())
    pending = frame[~frame.source_index.isin(done)].to_dict('records')
    records, indices = [], []
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn'),
            initializer=init_worker, initargs=(str(root), dep, split)) as pool:
        for n, (idx, arr, rec, sec) in enumerate(pool.map(system, pending, chunksize=1)):
            raw[idx*nview:(idx+1)*nview] = arr
            for j, r in enumerate(rec):
                r['row_index'] = idx*nview+j
                r['generation_system_seconds'] = sec
            records.extend(rec)
            indices.append(idx)
            if len(indices) == 16 or n == len(pending)-1:
                raw.flush()
                file = chunks / f'{indices[0]:05d}_{indices[-1]:05d}.parquet'
                if file.exists():
                    raise RuntimeError('Refuse chunk overwrite')
                pd.DataFrame(records).to_parquet(file, index=False)
                records, indices = [], []
            if n % 64 == 0:
                print(json.dumps({'expanded_generation': dep, 'split': split,
                    'completed': len(done)+n+1, 'total': len(frame), 'seconds': time.perf_counter()-started}), flush=True)
    meta = pd.concat([pd.read_parquet(p) for p in sorted(chunks.glob('*.parquet'))]).sort_values('row_index')
    if len(meta) != len(raw) or not np.array_equal(meta.row_index, np.arange(len(raw))):
        raise RuntimeError('Generation completeness failure')
    if not np.isfinite(raw).all():
        raise RuntimeError('Nonfinite stored waveform')
    meta.to_parquet(out / 'event_metadata.parquet', index=False)
    dev.json_write(out / 'COMPLETE.json', {'sources': len(frame), 'events': len(meta),
        'raw_shape': list(raw.shape), 'raw_sha256': dev.sha(path),
        'metadata_sha256': dev.sha(out / 'event_metadata.parquet'), 'failures': 0,
        'source_noise_unit_counts': [len(frame), BLOCKS[split]], 'seconds': time.perf_counter()-started})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=10)
    args = parser.parse_args()
    for dep in e.DEPS:
        for split in ('validation', 'train'):
            generate(args.root, dep, split, args.workers)
