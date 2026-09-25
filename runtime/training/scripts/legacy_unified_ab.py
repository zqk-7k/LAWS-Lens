#!/usr/bin/env python3
"""Paired SNR-cue experiment: immutable planning, physical data, bounded training.

This file does not claim full NEW-SCORE-ONLY completion at the short-encoder stage.
"""
import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_key] = '1'

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import logging
import multiprocessing as mp
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

import numpy as np
import pandas as pd

P = Path('/root/autodl-tmp/gw-catalog')
B = P/'results/o4b_hl_bayestar_new_score_only_20260912T072746Z'
FIX = P/'results/lensrank_sky_spin_si_fixed_20260916T023040Z_r2/scripts'
ARMS = ('A_NEUTRAL', 'B_CUE')
RUNS = ('O3', 'O4a', 'O4b')
SEEDS = (2026091721, 2026091722, 2026091723)
SNR_BINS = ((8., 10.), (10., 12.), (12., 20.), (20., 40.))
CTX = {}


def now():
    return datetime.now(timezone.utc).isoformat()


def stable(*parts):
    return int.from_bytes(hashlib.sha256(':'.join(map(str, parts)).encode()).digest()[:8], 'little')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    sys.modules[name] = obj
    spec.loader.exec_module(obj)
    return obj


def guard(root):
    free = shutil.disk_usage(root).free
    if free < 25*2**30:
        raise RuntimeError(f'HOLD_DISK_LIMIT: {free/2**30:.2f} GiB free')


def modules(root):
    sys.path.insert(0, str(P))
    sys.path.insert(0, str(P/'scripts/experiments'))
    source = module(root/'scripts/source_generator.py', 'uab_source')
    sampling = module(root/'scripts/waveform_multiscale_data.py', 'uab_sampling')
    views = module(root/'scripts/waveform_multiscale_train.py', 'uab_views')
    from scripts.real_search import physical_common as phys
    return source, sampling, views, phys


def native_shared(run):
    if run == 'O4b':
        return B/'workspace/results/real_noise_injection_v5_physical_source_20260721/gwtc5/shared'
    dep = {'O3': 'gwtc3', 'O4a': 'gwtc4'}[run]
    return P/'results/real_noise_injection_v5_physical_source_20260721'/dep/'shared'


def freeze_inputs(root):
    if root.exists():
        raise RuntimeError('Use a new independent output directory')
    for part in ('contracts', 'scripts', 'plans', 'reports', 'logs', 'manifests', 'pilot', 'arms', 'package'):
        (root/part).mkdir(parents=True)
    helper = P/'results/waveform_domain_multiscale_exploratory_20260903_20260903T063500Z/scripts'
    sources = {
        'source_generator.py': P/'scripts/real_search/34_generate_physical_h1l1_source_bank.py',
        'short_trainer.py': P/'scripts/real_search/37_unified_intrinsic_multitask_pilot.py',
        'waveform_multiscale_data.py': helper/'waveform_multiscale_data.py',
        'waveform_multiscale_train.py': helper/'waveform_multiscale_train.py',
        'bayestar_si_fixed.py': FIX/'bayestar_injection_sky_full_experiment.py',
        'unified_ab.py': Path(__file__),
    }
    rows = []
    for name, source in sources.items():
        destination = root/'scripts'/name
        shutil.copy2(source, destination)
        rows.append({'path': str(source), 'snapshot': str(destination), 'sha256': sha(source)})
    for source in (B/'scripts').glob('*.py'):
        if source.name.startswith('o4b_hl_nso_'):
            dest = root/'scripts/archived_adapters'/source.name
            dest.parent.mkdir(exist_ok=True)
            shutil.copy2(source, dest)
            rows.append({'path': str(source), 'snapshot': str(dest), 'sha256': sha(source)})
    # Protect baseline summaries/configuration independently of the new outputs.
    for base in (B, P/'results/mcwf_unified_path875_devconf_20260908T181500Z'):
        for folder in ('contracts', 'tables', 'calibration'):
            for f in sorted((base/folder).rglob('*')):
                if f.is_file() and f.stat().st_size < 10*2**20:
                    rows.append({'path': str(f), 'snapshot': None, 'sha256': sha(f)})
    pd.DataFrame(rows).to_csv(root/'manifests/PROTECTED_INPUTS.csv', index=False)
    write(root/'RUN_STATUS.json', {'utc': now(), 'state': 'PLANNING', 'complete_results': False})


def prepare_noise_and_calendars(root):
    report = {}
    for run in RUNS:
        original = native_shared(run)
        out = root/'plans'/run
        out.mkdir(exist_ok=True)
        schedule = pd.read_csv(original/'h1l1_live_schedule.csv')
        schedule = schedule[schedule.run.isin({'O3': ['O3a', 'O3b'], 'O4a': ['O4a'], 'O4b': ['O4b']}[run])].copy()
        schedule = schedule.sort_values('start_gps').reset_index(drop=True)
        if (schedule.start_gps.to_numpy()[1:] < schedule.end_gps.to_numpy()[:-1]).any():
            raise RuntimeError('Overlapping exposure segments')
        # Within-run exposure is uniform live time, not historical O1--O3 rates.
        schedule['duration_s'] = schedule.end_gps - schedule.start_gps
        schedule['weight_per_second'] = 1.0/schedule.duration_s.sum()
        schedule.to_csv(out/'live_schedule.csv', index=False)
        if run == 'O4b':
            noise = pd.read_parquet(original/'noise_reference_metadata.parquet')
            if 'parent_raw_file_group' not in noise:
                raise RuntimeError('Cannot establish O4b parent identity')
            noise['parent_uid'] = noise.parent_raw_file_group.astype(str)
            if 'noise_bank_index' not in noise:
                raise RuntimeError('Missing O4b noise bank index')
        else:
            noise = pd.read_csv(original/'noise_bank_manifest.csv')
            noise['noise_bank_index'] = noise.bank_index.astype(int)
            noise['parent_uid'] = noise.source_event.astype(str)
        lo, hi = schedule.start_gps.min(), schedule.end_gps.max()
        start_col = next((c for c in ('reference_start_gps', 'start_gps') if c in noise), None)
        if start_col is None:
            raise RuntimeError('Missing noise GPS')
        noise = noise[(noise[start_col] >= lo) & (noise[start_col]+256 <= hi)].copy()
        parents = sorted(noise.parent_uid.unique(), key=lambda x: stable('UAB-noise-parent', run, x))
        if len(parents) < 32:
            raise RuntimeError(f'{run}: {len(parents)} noise parents, need 32')
        selected = []
        for k, parent in enumerate(parents[:32]):
            group = noise[noise.parent_uid == parent]
            row = group.iloc[min(range(len(group)), key=lambda j: stable('UAB-noise-block', run, int(group.iloc[j].noise_bank_index)))].to_dict()
            row.update(split='train' if k < 20 else ('validation' if k < 26 else 'test'),
                       original_bank_index=int(row['noise_bank_index']), noise_bank_index=k)
            selected.append(row)
        frame = pd.DataFrame(selected)
        # Detect overlapping samples, including distinct nominal parent identifiers.
        starts = frame[start_col].to_numpy(float)
        for i in range(len(frame)):
            for j in range(i):
                if abs(starts[i]-starts[j]) < 256 and frame.iloc[i].split != frame.iloc[j].split:
                    raise RuntimeError('Cross-split physical noise interval overlap')
        frame.to_parquet(out/'noise_plan.parquet', index=False)
        refs = np.load(original/'noise_reference_bank.npy', mmap_mode='r')
        psds = np.load(original/'noise_psd_bank.npy', mmap_mode='r')
        ids = frame.original_bank_index.to_numpy(int)
        np.save(out/'noise_reference_bank.npy', np.asarray(refs[ids], np.float32))
        np.save(out/'noise_psd_bank.npy', np.asarray(psds[ids], np.float64))
        shutil.copy2(original/'noise_psd_frequency.npy', out/'noise_psd_frequency.npy')
        report[run] = {'events_as_noise_parents_available': len(parents), 'blocks_used': len(frame),
                       'partition': frame.groupby('split').size().to_dict(), 'start_gps': float(lo), 'end_gps': float(hi),
                       'live_seconds': float(schedule.duration_s.sum()), 'source_schedule': str(original/'h1l1_live_schedule.csv'),
                       'source_schedule_sha256': sha(original/'h1l1_live_schedule.csv'),
                       'source_noise_sha256': sha(original/'noise_reference_bank.npy'),
                       'calendar_type': 'archived joint live intervals; not a new official exposure determination'}
    write(root/'contracts/NOISE_CALENDAR_AUDIT.json', report)


def intrinsic(m1, m2, massbin, rng):
    return dict(m1_det=float(m1), m2_det=float(m2), mc_det=float((m1*m2)**.6/(m1+m2)**.2), mass_bin=int(massbin),
                a1=float(rng.uniform(0, .8)), a2=float(rng.uniform(0, .8)),
                tilt1=float(np.arccos(rng.uniform(-1, 1))), tilt2=float(np.arccos(rng.uniform(-1, 1))),
                theta_jn=float(np.arccos(rng.uniform(-1, 1))), phi12=float(rng.uniform(0, 2*np.pi)),
                phijl=float(rng.uniform(0, 2*np.pi)), psi=float(rng.uniform(0, np.pi)), phase=float(rng.uniform(0, 2*np.pi)),
                ra=float(rng.uniform(0, 2*np.pi)), dec=float(np.arcsin(rng.uniform(-1, 1))), dl_source=1000.)


def source_plans(root):
    src, sampling, _, _ = modules(root)
    table = src.load_tables(src.GW_LMC_ROOT)
    schedules = {run: src.load_schedule(root/'plans'/run/'live_schedule.csv') for run in RUNS}
    ids = table.event_id.astype(str)
    unique = table.loc[~ids.duplicated()].copy()
    candidates = {False: [], True: []}
    for _, row in unique.iterrows():
        images = src.choose_images(row)
        if images is None or images['proposal_snr_ratio'] > 4:
            continue
        if all(src.place_pair(images['delay_days'], sc, np.random.default_rng(stable('feasible', row.event_id, run))) is not None
               for run, sc in schedules.items()):
            candidates[bool(row.lens_is_subhalo)].append((row, images))
    for family in candidates:
        candidates[family].sort(key=lambda pair: stable('UAB-global-source', str(pair[0].event_id)))
    # Each environment belongs to one split globally, including auxiliary roles.
    groups = {}
    for family, rows in candidates.items():
        # Singles use distinct smooth-environment IDs; reserve independent
        # auxiliary groups explicitly instead of exhausting a 70/15/15 split.
        main_train, main_val, main_test = (420, 90, 90) if family else (840, 180, 180)
        remaining = len(rows)-main_train-main_val-main_test
        if remaining < 30:
            raise RuntimeError('Insufficient independent environment reserve')
        validation_reserve = max(30, int(.375*remaining))
        train_end = main_train+remaining-validation_reserve
        val_end = len(rows)-main_test
        groups[family] = {'train': rows[:train_end],
                          'validation': rows[train_end:val_end],
                          'test': rows[val_end:]}
    result = []
    main_counts = {'train': 420, 'validation': 90, 'test': 90}
    used = set()
    for split, count in main_counts.items():
        for family in ('SIS', 'PM', 'unlensed'):
            pool = groups[family == 'PM'][split] if family != 'unlensed' else groups[False][split]+groups[True][split]
            choices = [x for x in pool if str(x[0].event_id) not in used]
            if len(choices) < count:
                raise RuntimeError(f'Insufficient global source groups: {split} {family}, {len(choices)} < {count}')
            rng = np.random.default_rng(stable('UAB-main-intrinsics', split, family))
            for k, ((row, images), (m1, m2, mb)) in enumerate(zip(choices[:count], sampling.balanced_mass_draws(count, rng))):
                gid = str(row.event_id); used.add(gid)
                index = k+{'train': 0, 'validation': 420, 'test': 510}[split]
                item = dict(source_uid=f'UAB-main-{family}-{index:04d}', role='main', family=family, split=split,
                            source_index=index, global_source_id=gid, gwlmc_environment_id=gid,
                            gwlmc_row=int(row.gwlmc_row), **images, **intrinsic(m1, m2, mb, rng))
                if family == 'unlensed':
                    item.update(mu_image1=1., morse_image1=0.)
                result.append(item)
    for split, count in (('train', 4096), ('validation', 512)):
        rng = np.random.default_rng(stable('UAB-aux-intrinsics', split))
        # Training waveform parents can reuse training-only lens environments.
        # Auxiliary validation environments remain disjoint from main validation.
        pool = {f: [x for x in groups[f][split] if split == 'train' or str(x[0].event_id) not in used]
                for f in (False, True)}
        if min(map(len, pool.values())) < 30:
            raise RuntimeError('Too few source-disjoint auxiliary environments')
        for k, (m1, m2, mb) in enumerate(sampling.balanced_mass_draws(count, rng)):
            family = bool(k % 2)
            row, images = pool[family][int(rng.integers(len(pool[family])))]
            result.append(dict(source_uid=f'UAB-aux-{split}-{k:05d}', role='aux', family='PM' if family else 'SIS', split=split,
                               source_index=k, global_source_id=str(row.event_id), gwlmc_environment_id=str(row.event_id),
                               gwlmc_row=int(row.gwlmc_row), **images, **intrinsic(m1, m2, mb, rng)))
    source = pd.DataFrame(result)
    if source.groupby('global_source_id').split.nunique().max() != 1:
        raise RuntimeError('Global source split leakage')
    source['source_group_weight'] = 1/source.groupby(['role', 'split', 'global_source_id']).source_uid.transform('size')
    source.to_parquet(root/'plans/source_population.parquet', index=False)
    for run, schedule in schedules.items():
        current = source.copy()
        times = []
        for row in current.to_dict('records'):
            rng = np.random.default_rng(stable('UAB-gps', run, row['source_uid']))
            if row['family'] == 'unlensed':
                first = float(src.sample_live_times(schedule, 1, rng)[0]); second = first
            else:
                first, second = src.place_pair(row['delay_days'], schedule, rng)
            times.append((first, second))
        current['gps_a'], current['gps_b'] = np.array(times).T
        current.to_parquet(root/'plans'/run/'sources.parquet', index=False)
    # Same source subsets across runs, arms and all model seeds.
    subsets = []
    for split in ('validation', 'test'):
        for draw in range(500):
            rng = np.random.default_rng(stable('UAB-190', split, draw))
            for family, count in (('SIS', 35), ('PM', 35), ('unlensed', 50)):
                pool = source[(source.role == 'main') & (source.split == split) & (source.family == family)]
                for uid in rng.choice(pool.source_uid.to_numpy(), count, replace=False):
                    subsets.append(dict(split=split, draw=draw, family=family, source_uid=uid))
    pd.DataFrame(subsets).to_parquet(root/'plans/catalog190_subsets.parquet', index=False)
    return source


def snr_assignments(events, group_key):
    """An arm pair shares its exact target-SNR multiset, not just bin edges."""
    b = np.array([np.random.default_rng(stable('UAB-SNR-within-bin', group_key, r['event_uid'], r['variant'])).uniform(
                  *SNR_BINS[(2*int(r['source_index'])+int(r['image_number'])-1) % 4]) for r in events])
    order = np.random.default_rng(stable('UAB-label-neutral-permutation', group_key)).permutation(len(events))
    return b[order], b


def event_plans(root, sources):
    reference = []
    for row in sources.to_dict('records'):
        nviews = (8 if row['role'] == 'main' else 4) if row['split'] == 'train' else 1
        for image in range(1, 2 if row['family'] == 'unlensed' else 3):
            for variant in range(nviews):
                reference.append(dict(source_uid=row['source_uid'], source_index=row['source_index'], family=row['family'],
                                      role=row['role'], split=row['split'], image_number=image, variant=variant,
                                      event_uid=f'{row["source_uid"]}-{image}',
                                      measurement_seed=int(stable('UAB-BAYESTAR', row['source_uid'], image, variant) % 2**32)))
    ref = pd.DataFrame(reference)
    for _, indices in ref.groupby(['role', 'split', 'variant']).groups.items():
        key = tuple(ref.loc[indices[0], ['role', 'split', 'variant']])
        a, b = snr_assignments(ref.loc[indices].to_dict('records'), key)
        ref.loc[indices, 'snr_A_NEUTRAL'] = a
        ref.loc[indices, 'snr_B_CUE'] = b
        assert np.array_equal(np.sort(a), np.sort(b))
    for run in RUNS:
        noise = pd.read_parquet(root/'plans'/run/'noise_plan.parquet')
        f = ref.copy()
        bank, offset = [], []
        for row in f.to_dict('records'):
            allowed = noise[noise.split == row['split']].noise_bank_index.to_numpy(int)
            rng = np.random.default_rng(stable('UAB-noise', run, row['source_uid'], row['variant']))
            chosen = rng.choice(allowed, 2, replace=False)
            bank.append(int(chosen[row['image_number']-1]))
            # Same draw for both arms, different slices for the two images.
            offsets = rng.integers(0, (256-26)*4096+1, size=2)
            offset.append(int(offsets[row['image_number']-1]))
        f['noise_bank_index'], f['noise_offset_samples'] = bank, offset
        f.to_parquet(root/'plans'/run/'event_plan.parquet', index=False)
    return ref


def plan(root):
    freeze_inputs(root)
    prepare_noise_and_calendars(root)
    source = source_plans(root)
    events = event_plans(root, source)
    audit = dict(global_source_cross_split=0, source_counts=source.groupby(['role', 'split']).size().to_dict(),
                 exact_SNR_multiset_preserved=True, historical_outputs_not_modified=True)
    audit['source_counts'] = {':'.join(k): int(v) for k, v in audit['source_counts'].items()}
    write(root/'contracts/PLANNING_AUDIT.json', audit)
    contract = dict(code='GWLR-UAB-01', utc=now(), arms=ARMS, runs=RUNS, model_seeds=SEEDS,
                    same_event_realization_across_model_seeds=True, sole_arm_difference='target SNR assignment permutation',
                    short_epochs=[16, 60], short_batch_size=48, auxiliary_sources=[4096, 512],
                    source_hash=sha(root/'plans/source_population.parquet'),
                    main_catalog_events=450, size_control=dict(events=190, subsamples=500),
                    target_SNR_bins=SNR_BINS, both_arms_per_image_target_scaling=True,
                    not_response_derived_population=True, not_same_noisy_strain_BAYESTAR=True,
                    no_outer_score_mixing=True, analysis_nside=512, mass_units='kg in low-level spin conversion',
                    selection_uses_real_PE_or_official=False, historical_data_exploratory=True,
                    test_generation_requires_FINAL_SCORE_FREEZE=True,
                    status='HOLD_FOR_AUTHOR_REVIEW_NO_ADOPTION_NO_OVERWRITE')
    write(root/'contracts/ANALYSIS_CONTRACT.json', contract)
    files = [p for p in (root/'plans').rglob('*') if p.is_file()] + list((root/'scripts').glob('*.py'))
    files += [root/'contracts/ANALYSIS_CONTRACT.json']
    write(root/'contracts/PLAN_FREEZE.json', {str(p.relative_to(root)): sha(p) for p in files})
    write(root/'RUN_STATUS.json', {'utc': now(), 'state': 'PLAN_FROZEN', 'complete_results': False})


def verify(root):
    for relative, expected in json.loads((root/'contracts/PLAN_FREEZE.json').read_text()).items():
        if sha(root/relative) != expected:
            raise RuntimeError('Frozen input changed: '+relative)


def worker_init(root, run, role, split):
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)
    root = Path(root)
    source, sampling, views, phys = modules(root)
    logging.getLogger('bilby').setLevel(logging.ERROR)
    plan = pd.read_parquet(root/'plans'/run/'event_plan.parquet')
    plan = plan[(plan.role == role) & (plan.split == split)]
    CTX.update(root=root, run=run, role=role, split=split, src=source, views=views, phys=phys,
               generator=source.build_waveform_generator(),
               ifos=list(source.bilby.gw.detector.InterferometerList(['H1', 'L1'])),
               events={uid: f for uid, f in plan.groupby('source_uid')},
               refs=np.load(root/'plans'/run/'noise_reference_bank.npy', mmap_mode='r'),
               psds=np.load(root/'plans'/run/'noise_psd_bank.npy', mmap_mode='r'),
               freq=np.load(root/'plans'/run/'noise_psd_frequency.npy'))


def generate_source(row):
    from scipy.signal import resample_poly
    root, run, src, p = CTX['root'], CTX['run'], CTX['src'], CTX['phys']
    out = root/'data'/run/row['role']/row['split']/row['source_uid']
    marker = out/'COMPLETE.json'
    if marker.exists():
        receipt = json.loads(marker.read_text())
        for name, expected in receipt['sha256'].items():
            if sha(out/name) != expected:
                raise RuntimeError('Materialized input changed: '+str(out/name))
        return receipt
    guard(root)
    out.mkdir(parents=True, exist_ok=True)
    rows = CTX['events'][row['source_uid']].sort_values(['image_number', 'variant'])
    outputs = {arm: {'short': [], 'long': [], 'clean': []} for arm in ARMS}
    metas = []
    start = time.monotonic()
    for image in sorted(rows.image_number.unique()):
        gps = row['gps_a' if image == 1 else 'gps_b']
        clean, peaks = src.detector_response(CTX['generator'], CTX['ifos'], src.source_parameters(pd.Series(row), gps),
                                            src.lens_factor(row[f'mu_image{image}'], row[f'morse_image{image}']))
        for event in rows[rows.image_number == image].to_dict('records'):
            bank = event['noise_bank_index']; offset = event['noise_offset_samples']
            psd = CTX['psds'][bank]
            noise = np.asarray(CTX['refs'][bank, :, offset:offset+p.RAW_PADDED_SAMPLES], np.float32)
            if noise.shape[-1] != p.RAW_PADDED_SAMPLES:
                raise RuntimeError('Truncated noise slice')
            for arm in ARMS:
                target = event[f'snr_{arm}']
                scaled, factor, recovered = p.scale_to_network_snr(clean, target, CTX['freq'], psd)
                signal = p.embed_signal_in_padded_window(scaled)
                mixed = noise+signal
                full = p.preprocess_24s(mixed, CTX['freq'], psd)
                short = CTX['views'].make_window_view(full[None], 2)[0]
                low = resample_poly(p.preprocess_24s(mixed, CTX['freq'], psd, band_low_hz=20, band_high_hz=80),
                                    1, 8, axis=-1, window=('kaiser', 8.6))[:, -4096:]
                clean_short = CTX['views'].make_window_view(p.preprocess_24s(signal, CTX['freq'], psd)[None], 2)[0]
                if short.shape != (2, 4096) or low.shape != (2, 4096) or not np.isfinite(short).all() or not np.isfinite(low).all():
                    raise RuntimeError('Input dimension/nonfinite error')
                if abs(recovered-target) > 1e-4*target:
                    raise RuntimeError('PSD optimal SNR mismatch')
                outputs[arm]['short'].append(short.astype(np.float32))
                outputs[arm]['long'].append(low.astype(np.float32))
                outputs[arm]['clean'].append(clean_short.astype(np.float32))
                metas.append({**row, **event, **peaks, 'arm': arm, 'gps_obs': gps, 'image': 'a' if image == 1 else 'b',
                              'noise_variant': event['variant'], 'target_network_snr': float(target),
                              'recovered_optimal_network_snr': float(recovered), 'physical_strain_scale_factor': float(factor),
                              'ra_true': row['ra'], 'dec_true': row['dec'], 'morse_index': row[f'morse_image{image}']})
    hashes = {}
    for arm, arrays in outputs.items():
        for kind, values in arrays.items():
            path = out/f'{arm}_{kind}.npy'
            np.save(path, np.stack(values)); hashes[path.name] = sha(path)
    pd.DataFrame(metas).to_parquet(out/'metadata.parquet', index=False)
    hashes['metadata.parquet'] = sha(out/'metadata.parquet')
    receipt = {'source_uid': row['source_uid'], 'run': run, 'role': row['role'], 'split': row['split'],
               'sha256': hashes, 'seconds': time.monotonic()-start, 'both_arms': True}
    write(marker, receipt)
    return receipt


def generate(root, run, role, split, workers, pilot=0):
    verify(root)
    if split == 'test' and not (root/'contracts/FINAL_SCORE_FREEZE.json').exists():
        raise RuntimeError('Locked test is sealed')
    sources = pd.read_parquet(root/'plans'/run/'sources.parquet')
    sources = sources[(sources.role == role) & (sources.split == split)]
    if pilot:
        sources = sources.groupby('family', sort=True).head(pilot)
    rows = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn'), initializer=worker_init,
                             initargs=(str(root), run, role, split)) as pool:
        for i, receipt in enumerate(pool.map(generate_source, sources.to_dict('records')), 1):
            rows.append(receipt)
            if i % 16 == 0 or i == len(sources):
                print(json.dumps({'run': run, 'role': role, 'split': split, 'sources': i, 'total': len(sources),
                                  'last_seconds': receipt['seconds']}), flush=True)
    name = f'{run}_{role}_{split}_'+('PILOT' if pilot else 'COMPLETE')+'.json'
    write(root/'contracts'/name, {'utc': now(), 'source_count': len(rows), 'both_arms': True,
                                 'wall_per_source_seconds': {'median': float(np.median([r['seconds'] for r in rows])),
                                                             'p90': float(np.quantile([r['seconds'] for r in rows], .9))}})


def sky_pilot(root):
    verify(root)
    from ligo.skymap.io.fits import write_sky_map
    b = module(root/'scripts/bayestar_si_fixed.py', 'uab_sky')
    summary = []
    for run in RUNS:
        b._WORKER_PSD_FREQ = np.load(root/'plans'/run/'noise_psd_frequency.npy')
        b._WORKER_PSD_BANK = np.load(root/'plans'/run/'noise_psd_bank.npy', mmap_mode='r')
        allmeta = pd.concat([pd.read_parquet(p) for p in sorted((root/'data'/run/'main/validation').glob('*/metadata.parquet'))])
        # Every pilot source/image is fixed before localization, not chosen by map quality.
        for row in allmeta.to_dict('records'):
            path = root/'pilot'/run/row['arm']/f'{row["event_uid"]}.fits.gz'
            receipt = path.with_suffix('.json')
            if receipt.exists():
                record = json.loads(receipt.read_text())
                if sha(path) != record['sha256']:
                    raise RuntimeError('Pilot MOC changed')
                summary.append(record); continue
            path.parent.mkdir(parents=True, exist_ok=True)
            start = time.monotonic()
            sky, audit = b._simulate_with_waveform(row, pd.Series(row), b.PRIMARY_WAVEFORM)
            probability = b.raster_probability(sky, 512)
            if not np.isfinite(probability).all() or abs(probability.sum()-1) > 1e-10:
                raise RuntimeError('BAYESTAR probability invalid')
            write_sky_map(path, sky)
            record = {**audit, 'run': run, 'arm': row['arm'], 'event_uid': row['event_uid'],
                      'source_uid': row['source_uid'], 'mass_unit': 'kg for spin conversion',
                      'analysis_nside': 512, 'ordering': 'NESTED', 'native_format': 'MOC density per steradian',
                      'seconds': time.monotonic()-start, 'sha256': sha(path), 'not_full_BBH_PE': True}
            write(receipt, record); summary.append(record)
            print(json.dumps({'sky_pilot': run, 'arm': row['arm'], 'event': row['event_uid'], 'seconds': record['seconds']}), flush=True)
    pd.DataFrame(summary).to_csv(root/'pilot/BAYESTAR_PILOT.csv', index=False)
    write(root/'contracts/BAYESTAR_PILOT_PASS.json', {'utc': now(), 'events': len(summary), 'full_experiment_complete': False})


def short_inputs(root, run, arm):
    current = root/'arms'/arm/run
    out = current/'short_data'; out.mkdir(parents=True, exist_ok=True)
    src = pd.read_parquet(root/'plans'/run/'sources.parquet')
    src = src[src.role == 'main']
    manifest = []
    for family in ('SIS', 'PM', 'unlensed'):
        ids = src[src.family == family].sort_values('source_index')
        meta = ids.copy(); meta['mass_1_detector'] = meta.m1_det; meta['mass_2_detector'] = meta.m2_det
        dest = current/'source_bank'/f'{family}_data_0222'; dest.mkdir(parents=True, exist_ok=True)
        meta.to_parquet(dest/'physical_source_pair_metadata.parquet', index=False)
        for kind in ('noisy', 'pure'):
            images = (1,) if family == 'unlensed' else (1, 2)
            for image in images:
                file = out/f'{family}_{kind}_{image}.npy'
                array = np.lib.format.open_memmap(file, mode='w+', dtype=np.float32, shape=(600, 2, 4096))
                array[:] = np.nan  # Test rows stay inaccessible and unmaterialized.
                for row in ids[ids.split != 'test'].to_dict('records'):
                    folder = root/'data'/run/'main'/row['split']/row['source_uid']
                    if not (folder/'COMPLETE.json').exists():
                        raise RuntimeError('Source incomplete: '+str(folder))
                    m = pd.read_parquet(folder/'metadata.parquet'); m = m[m.arm == arm].reset_index(drop=True)
                    index = m.index[(m.image_number == image) & (m.variant == 0)][0]
                    array[row['source_index']] = np.load(folder/f'{arm}_{"short" if kind == "noisy" else "clean"}.npy', mmap_mode='r')[index]
                array.flush(); manifest.append({'path': str(file), 'sha256': sha(file)})
        if family != 'unlensed':
            mult = current/'data/real_noise_injections'/f'multinoise_{family.lower()}_train_v8'
            mult.mkdir(parents=True, exist_ok=True)
            records = ids[ids.split == 'train'].to_dict('records')
            for image, tag in ((1, 'a'), (2, 'b')):
                path = mult/f'noisy_image_{tag}.npy'
                array = np.lib.format.open_memmap(path, mode='w+', dtype=np.float32, shape=(420*8, 2, 4096))
                for i, row in enumerate(records):
                    folder = root/'data'/run/'main/train'/row['source_uid']
                    m = pd.read_parquet(folder/'metadata.parquet'); m = m[m.arm == arm].reset_index(drop=True)
                    indices = m.index[m.image_number == image]
                    array[i*8:(i+1)*8] = np.load(folder/f'{arm}_short.npy', mmap_mode='r')[indices]
                array.flush()
            np.save(mult/'source_index.npy', np.repeat([r['source_index'] for r in records], 8))
            np.save(mult/'variant.npy', np.tile(np.arange(8), 420))
    write(current/'SHORT_DATA_COMPLETE.json', {'utc': now(), 'test_rows_nan': True, 'input_manifest': manifest})


def train_short(root, run, arm, seed):
    import torch
    torch.set_num_threads(2)
    verify(root)
    current = root/'arms'/arm/run
    if not (current/'SHORT_DATA_COMPLETE.json').exists():
        short_inputs(root, run, arm)
    out = current/f'models/short_encoder/seed_{seed}'
    if (out/'summary.json').exists():
        return
    m = module(root/'scripts/short_trainer.py', 'uab_short')
    m.REPO = P; m.INPUT_SAMPLES = 4096
    original_loader = m.module_from
    def load_override(path, name):
        if path.name == '20_real_noise_injection_v3_physical.py':
            return SimpleNamespace(training_config=lambda sr, fam, *args: SimpleNamespace(family=fam))
        return original_loader(path, name)
    m.module_from = load_override
    m.split_indices = lambda *args: {'lensed': {'train': np.arange(420), 'val': np.arange(420, 510)}}
    def arrays(cfg):
        d = current/'short_data'
        get = lambda fam, kind, image: np.load(d/f'{fam}_{kind}_{image}.npy', mmap_mode='r')
        return SimpleNamespace(l1=get(cfg.family, 'noisy', 1), l2=get(cfg.family, 'noisy', 2),
                               l1_pure=get(cfg.family, 'pure', 1), l2_pure=get(cfg.family, 'pure', 2),
                               unlensed=get('unlensed', 'noisy', 1), unlensed_pure=get('unlensed', 'pure', 1))
    m.load_match_arrays = arrays
    sys.argv = ['short_trainer.py', '--seed-root', str(current), '--source-bank', str(current/'source_bank'),
                '--seed', str(seed), '--samples', '600', '--pretrain-epochs', '16', '--adapt-epochs', '60',
                '--batch-size', '48', '--aux-weight', '.25', '--q-loss-weight', '1',
                '--backbone', 'inception_attention', '--out-dir', str(out)]
    m.main()


def command(root, label, args):
    guard(root)
    logfile = root/'logs'/f'{label}_{time.time_ns()}.log'
    cmd = [sys.executable, '-B', '-u', str(root/'scripts/unified_ab.py'), '--root', str(root), *args]
    record = {'utc': now(), 'state': label, 'command': cmd, 'log': str(logfile), 'complete_results': False}
    write(root/'RUN_STATUS.json', record)
    with logfile.open('x') as stream:
        proc = subprocess.Popen(cmd, stdout=stream, stderr=subprocess.STDOUT)
        record['pid'] = proc.pid; write(root/'RUN_STATUS.json', record)
        rc = proc.wait()
    write(root/'contracts/tasks'/f'{label}_{logfile.stem}.json', {**record, 'exit_code': rc, 'finished_utc': now()})
    if rc:
        raise RuntimeError(f'Task {label} failed; {logfile}')


def controller(root):
    verify(root)
    for run in RUNS:
        command(root, 'PILOT_'+run, ['--stage', 'generate', '--run', run, '--role', 'main', '--split', 'validation', '--pilot', '2', '--workers', '4'])
    command(root, 'BAYESTAR_PILOT', ['--stage', 'sky-pilot'])
    for run in RUNS:
        for split in ('validation', 'train'):
            command(root, f'DATA_{run}_{split}', ['--stage', 'generate', '--run', run, '--role', 'main', '--split', split, '--workers', '8'])
        for arm in ARMS:
            for seed in SEEDS:
                command(root, f'SHORT_{run}_{arm}_{seed}', ['--stage', 'short', '--run', run, '--arm', arm, '--seed', str(seed)])
        for split in ('validation', 'train'):
            command(root, f'AUX_{run}_{split}', ['--stage', 'generate', '--run', run, '--role', 'aux', '--split', split, '--workers', '8'])
    write(root/'RUN_STATUS.json', {'utc': now(), 'state': 'TRAINING_COMPONENTS_READY_FULL_SCORE_INTEGRATION_PENDING',
                                 'complete_results': False, 'locked_test_opened': False,
                                 'pending': ['RNC/ordered/multirate/conditional training and scoring integration',
                                             'validation time/sky/score calibration', 'locked-test and real audits', 'final package']})


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--stage', choices=['plan', 'generate', 'sky-pilot', 'short', 'controller', 'verify'], required=True)
    p.add_argument('--run', choices=RUNS, default='O3'); p.add_argument('--role', choices=['main', 'aux'], default='main')
    p.add_argument('--split', choices=['train', 'validation', 'test'], default='validation')
    p.add_argument('--arm', choices=ARMS, default=ARMS[0]); p.add_argument('--seed', type=int, choices=SEEDS, default=SEEDS[0])
    p.add_argument('--workers', type=int, default=4); p.add_argument('--pilot', type=int, default=0)
    args = p.parse_args()
    try:
        if args.stage == 'plan': plan(args.root)
        elif args.stage == 'verify': verify(args.root)
        elif args.stage == 'generate': generate(args.root, args.run, args.role, args.split, args.workers, args.pilot)
        elif args.stage == 'sky-pilot': sky_pilot(args.root)
        elif args.stage == 'short': train_short(args.root, args.run, args.arm, args.seed)
        else: controller(args.root)
    except Exception:
        if args.root.exists():
            write(args.root/'FAILURES'/f'{args.stage}_{time.time_ns()}.json', {'utc': now(), 'traceback': traceback.format_exc()})
            write(args.root/'RUN_STATUS.json', {'utc': now(), 'state': 'HOLD_COMPUTATION_ERROR', 'stage': args.stage, 'complete_results': False})
        raise


if __name__ == '__main__':
    main()
