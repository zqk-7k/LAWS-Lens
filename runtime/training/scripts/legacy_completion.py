"""Continue frozen UAB models through inference and event-specific sky maps."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import fcntl
from concurrent.futures import ProcessPoolExecutor
import importlib.util
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import sys
import time
import traceback

import numpy as np
import pandas as pd

ROOT = OUT = U = None
SKY = {}


def initialize(root, out):
    global ROOT, OUT, U
    ROOT, OUT = Path(root), Path(out)
    sys.path[:0] = [str(ROOT/'scripts'), str(ROOT/'scripts/archived_adapters'),
                   '/root/autodl-tmp/gw-catalog/scripts/experiments', '/root/autodl-tmp/gw-catalog']
    import unified_ab as u
    U = u
    from threadpoolctl import threadpool_limits
    threadpool_limits(2)
    return u


def write(path, data):
    U.write(path, data)


def check_complete(path):
    if not path.exists():
        return False
    for item in json.loads(path.read_text()).get('files', []):
        if U.sha(item['path']) != item['sha256']:
            raise RuntimeError('Completed output changed: '+item['path'])
    return True


def seal(path, files, **details):
    write(path, dict(utc=U.now(), files=[dict(path=str(p), sha256=U.sha(p)) for p in files], **details))


def deployment(run, arm):
    p = OUT/'deployments'/run/arm
    p.mkdir(parents=True, exist_ok=True)
    for name in ('contracts', 'scripts', 'manifests', 'logs', 'tables'):
        (p/name).mkdir(exist_ok=True)
    old = ROOT/'arms'/arm/run
    for name in ('models', 'ordered_mass_predictor', 'rankncontrast_component_v2',
                 'waveform_feature_operator', 'auxiliary_data_v2', 'waveform_features', 'workspace'):
        dest, src = p/name, old/name
        if dest.exists():
            if dest.resolve() != src.resolve():
                raise RuntimeError('Wrong deployment link: '+str(dest))
        else:
            if not src.exists():
                raise FileNotFoundError(src)
            dest.symlink_to(src, target_is_directory=True)
    return p


def gate(split):
    if split not in ('test', 'real'):
        return
    freeze = ROOT/'contracts/FINAL_SCORE_FREEZE.json'
    if not freeze.exists():
        raise RuntimeError('Test and real remain sealed')
    for item in json.loads(freeze.read_text())['files']:
        if U.sha(item['path']) != item['sha256']:
            raise RuntimeError('Final score freeze changed: '+item['path'])


def audit():
    if check_complete(OUT/'contracts/INPUT_AUDIT.json'):
        return
    U.verify(ROOT)
    records = [json.loads(p.read_text()) for p in (ROOT/'contracts/tasks').glob('*.json')]
    model_tasks = [d for d in records if d['state'].split('_')[0] in
                   ('SHORT', 'rnc', 'ordered', 'multirate', 'conditional')]
    if len(model_tasks) != 90 or any(d['exit_code'] for d in model_tasks):
        raise RuntimeError('Expected all 90 successful training tasks')
    paths, model_rows = [], []
    for run in U.RUNS:
        for arm in U.ARMS:
            p = deployment(run, arm)
            for seed in U.SEEDS:
                suffixes = [f'models/short_encoder/seed_{seed}/validation_selected_model.pt',
                    f'rankncontrast_component_v2/models/RAW-PHASE-SOURCE/gwtc5/seed_{seed}/validation_selected_model.pt',
                    f'ordered_mass_predictor/models/gwtc5/seed_{seed}/validation_selected_model.pt',
                    f'models/MULTIRATE/gwtc5/seed_{seed}/selected.pt',
                    f'models/CONDITIONAL-ETA-CHI/gwtc5/seed_{seed}/selected.pt']
                for suffix in suffixes:
                    f = (p/suffix).resolve()
                    paths.append(f)
                    model_rows.append(dict(run=run, arm=arm, seed=seed, path=str(f), sha256=U.sha(f)))
    source = pd.read_parquet(ROOT/'plans/source_population.parquet')
    if source.groupby('global_source_id').split.nunique().max() != 1:
        raise RuntimeError('Source/lens-environment split overlap')
    pd.DataFrame(model_rows).to_csv(OUT/'contracts/TRAINED_MODEL_MANIFEST.csv', index=False)
    paths += [ROOT/'contracts/ANALYSIS_CONTRACT.json', ROOT/'contracts/SCORING_SPEC.json',
              ROOT/'contracts/SCORING_SPEC_FREEZE.json', ROOT/'plans/source_population.parquet']
    seal(OUT/'contracts/INPUT_AUDIT.json', paths, model_tasks=90, model_failures=0,
         source_environment_split_overlap=0, old_models_used=False, test_opened=False)


def prepare_inputs(run, arm, split):
    gate(split)
    p = deployment(run, arm)
    if split == 'development':
        return p/'auxiliary_data_v2/validation'
    if split == 'real':
        path = OUT/'real_inputs'/run
        if not check_complete(path/'COMPLETE.json'):
            raise RuntimeError('Real inputs not prepared')
        return path
    path = p/'inference_inputs'/split
    if check_complete(path/'COMPLETE.json'):
        return path
    path.mkdir(parents=True, exist_ok=True)
    source = pd.read_parquet(ROOT/'plans'/run/'sources.parquet')
    source = source[(source.role == 'main') & (source.split == split)].sort_values(['family', 'source_index'])
    chunks, short, long = [], [], []
    for row in source.itertuples(index=False):
        d = ROOT/'data'/run/'main'/split/row.source_uid
        if not check_complete(d/'COMPLETE.json'):
            raise RuntimeError('Source not materialized: '+str(d))
        receipt = json.loads((d/'COMPLETE.json').read_text())
        for name, digest in receipt['sha256'].items():
            if U.sha(d/name) != digest:
                raise RuntimeError('Source hash differs')
        allmeta = pd.read_parquet(d/'metadata.parquet')
        e = allmeta[allmeta.arm.eq(arm)].reset_index(drop=True)
        if not e.variant.eq(0).all():
            raise RuntimeError('Multiple validation/test views')
        short.append(np.load(d/f'{arm}_short.npy'))
        long.append(np.load(d/f'{arm}_long.npy'))
        chunks.append(e)
    events = pd.concat(chunks, ignore_index=True)
    events['idx'] = np.arange(len(events))
    events['tag'] = np.where(events.family.eq('unlensed'), 'U', 'L'+events.image_number.astype(str))
    noise = pd.read_parquet(ROOT/'plans'/run/'noise_plan.parquet').set_index('noise_bank_index')
    events['noise_parent_uid'] = events.noise_bank_index.map(noise.parent_uid)
    if len(events) != 450 or events.event_uid.nunique() != 450 or events.source_uid.nunique() != 270:
        raise RuntimeError('Wrong 450-event catalog')
    events.to_parquet(path/'events.parquet', index=False)
    np.save(path/'raw2s.npy', np.concatenate(short).astype(np.float32))
    np.save(path/'low16s.npy', np.concatenate(long).astype(np.float32))
    seal(path/'COMPLETE.json', [path/n for n in ('events.parquet', 'raw2s.npy', 'low16s.npy')],
         events=450, source_systems=270, raw_views_reused_exactly=True, old_model_used=False)
    return path


def context(run, arm, split):
    p = deployment(run, arm)
    folder = prepare_inputs(run, arm, split)
    if split == 'development':
        events = pd.read_parquet(folder/'event_metadata.parquet').copy()
        events['global_source_id'] = events.source_uid
        events['idx'] = np.arange(len(events))
        features = p/'waveform_features/auxiliary_v2/validation'
    else:
        events = pd.read_parquet(folder/'events.parquet')
        features = p/'inference_features'/split
    if split == 'real':
        freq = np.load(folder/'frequency.npy')
        psds = np.load(folder/'psd.npy', mmap_mode='r')
    else:
        freq = np.load(ROOT/'plans'/run/'noise_psd_frequency.npy')
        psds = np.load(ROOT/'plans'/run/'noise_psd_bank.npy', mmap_mode='r')
    return folder, events, features, freq, psds


def configure_inference(run, arm):
    import torch
    import o4b_hl_nso_inference_20260912 as inf
    import o4b_hl_nso_multiscale_training_20260912 as models
    import o4b_hl_nso_rankncontrast_v2_20260912 as rnc
    import mcwf_finelag_eventpsd_20260907 as psd
    import mcwf_finelag_encoder_20260907 as trainer
    import mcwf_mass_tf_20260905 as tf
    torch.set_num_threads(2)
    p = deployment(run, arm)
    low, mult, cond = models.modules()
    models.initialize = lambda unused: (low, mult, cond, p/'waveform_feature_operator')
    def rnc_setup(unused):
        rout = p/'rankncontrast_component_v2'
        trainer.b.PREVIOUS = rout
        return rout, psd, trainer, tf
    rnc.setup = rnc_setup
    inf.guard = lambda unused, split: gate(split)
    inf.get_context = lambda unused, seed, split: context(run, arm, split)
    return inf, p


def infer(run, arm, split, seed):
    gate(split)
    inf, p = configure_inference(run, arm)
    inf.infer(p, seed, split)
    marker = p/f'predictions_o4b/seed_{seed}/{split}/COMPLETE.json'
    rec = json.loads(marker.read_text())
    for item in rec['inputs']:
        if str(ROOT/'arms'/arm/run) not in str(Path(item['path']).resolve()):
            raise RuntimeError('Inference used another deployment model')
    write(p/f'contracts/INFERENCE_{split}_{seed}_UAB.json', dict(run=run, arm=arm, seed=seed,
        archived_storage_alias='gwtc5', actual_data=run, inherited_algorithm=True,
        old_model_used=False, model_hash_checks=True, joint_mass_marginal_error=rec['joint_mass_marginal_error']))


def sky_worker_init(root, out, run, arm, split):
    initialize(root, out)
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)
    b = U.module(ROOT/'scripts/bayestar_si_fixed.py', 'uab_completion_sky')
    b._WORKER_PSD_FREQ = np.load(ROOT/'plans'/run/'noise_psd_frequency.npy')
    b._WORKER_PSD_BANK = np.load(ROOT/'plans'/run/'noise_psd_bank.npy', mmap_mode='r')
    SKY.update(b=b, run=run, arm=arm, split=split, dest=OUT/'maps'/run/arm/split)


def localize_event(row):
    from ligo.skymap.io.fits import write_sky_map
    from ligo.skymap import moc
    b, dest = SKY['b'], SKY['dest']
    stem = row['event_uid']
    path, marker = dest/(stem+'.fits.gz'), dest/(stem+'.json')
    if marker.exists():
        record = json.loads(marker.read_text())
        if U.sha(path) != record['moc_sha256']:
            raise RuntimeError('Native MOC changed')
        return record
    U.guard(ROOT)
    begin = time.monotonic()
    try:
        fallback, reason = False, None
        try:
            sky, audit = b._simulate_with_waveform(row, pd.Series(row), b.PRIMARY_WAVEFORM)
        except Exception as exc:
            fallback, reason = True, repr(exc)
            sky, audit = b._simulate_with_waveform(row, pd.Series(row), b.FALLBACK_WAVEFORM)
        density = np.asarray(sky['PROBDENSITY'])
        if not np.isfinite(density).all() or (density < 0).any():
            raise RuntimeError('Invalid native sky density')
        probability = b.raster_probability(sky, 512)
        metrics = b.posterior_metrics(probability, row['ra_true'], row['dec_true'])
        orders = moc.uniq2order(np.asarray(sky['UNIQ'], np.int64))
        temp = dest/(stem+'.tmp.fits.gz')
        write_sky_map(temp, sky)
        temp.replace(path)
        record = dict(**row, **{k: v for k, v in audit.items() if k not in row}, **metrics,
            fallback_used=fallback, primary_error=reason, moc_path=str(path), moc_sha256=U.sha(path),
            native_order_min=int(orders.min()), native_order_max=int(orders.max()),
            ordering='NESTED', coordinate_frame='ICRS', probability_convention='density per steradian',
            seconds=time.monotonic()-begin, mass_units='kg in spin conversion',
            uses_controlled_source_parameters=True, map_shared_across_model_seeds=True)
        write(marker, record)
        return record
    except Exception:
        write(dest/(stem+'.failure.'+str(time.time_ns())+'.json'), dict(error=traceback.format_exc()))
        raise


def maps(run, arm, split, workers):
    lock = OUT/'contracts'/f'MAPS_{run}_{arm}_{split}.lock'
    with lock.open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        return maps_locked(run, arm, split, workers)


def maps_locked(run, arm, split, workers):
    gate(split)
    folder = prepare_inputs(run, arm, split)
    events = pd.read_parquet(folder/'events.parquet')
    dest = OUT/'maps'/run/arm/split
    dest.mkdir(parents=True, exist_ok=True)
    if check_complete(dest/'COMPLETE.json'):
        return
    progress = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn'),
            initializer=sky_worker_init, initargs=(str(ROOT), str(OUT), run, arm, split)) as pool:
        for n, value in enumerate(pool.map(localize_event, events.to_dict('records'), chunksize=1), 1):
            progress.append(value)
            if n % 10 == 0 or n == len(events):
                print(json.dumps(dict(stage='BAYESTAR', run=run, arm=arm, split=split, completed=n, total=len(events))), flush=True)
                write(dest/'PROGRESS.json', dict(completed=n, total=len(events), utc=U.now()))
    frame = pd.DataFrame(progress).sort_values('idx').reset_index(drop=True)
    frame.to_parquet(dest/'events.parquet', index=False)
    if split == 'validation':
        b = U.module(ROOT/'scripts/bayestar_si_fixed.py', 'uab_temperature')
        # One source system has total event weight one, including singletons.
        b.event_weights = lambda f: 1/f.groupby('source_uid').event_uid.transform('size').to_numpy(float)
        temp, grid = b.select_temperature(frame)
        grid.to_csv(dest/'temperature_grid.csv', index=False)
        write(dest/'TEMPERATURE_SELECTED.json', dict(temperature=temp, source_weighted=True,
            source_systems=int(frame.source_uid.nunique()), validation_only=True, real_PE_used=False))
        seal(dest/'TEMPERATURE_FREEZE.json', [dest/'TEMPERATURE_SELECTED.json'])
    seal(dest/'COMPLETE.json', [dest/'events.parquet'], events=len(frame), native_maps=len(frame),
         maps=[dict(path=v['moc_path'], sha256=v['moc_sha256']) for v in progress])


def sky_pairs(run, arm, split):
    gate(split)
    import healpy as hp
    import torch
    from astropy.io import fits
    from ligo.skymap.io.fits import read_sky_map
    b = U.module(ROOT/'scripts/bayestar_si_fixed.py', 'uab_sky_pairs')
    p = deployment(run, arm)
    dest = p/'sky_pair_scores'/split
    if check_complete(dest/'COMPLETE.json'):
        return
    dest.mkdir(parents=True, exist_ok=True)
    source = OUT/'maps'/run/arm/split
    if not check_complete(source/'COMPLETE.json'):
        raise RuntimeError('Maps incomplete')
    events = pd.read_parquet(source/'events.parquet').sort_values('idx')
    selected = OUT/'maps'/run/arm/'validation/TEMPERATURE_SELECTED.json'
    temp = json.loads(selected.read_text())['temperature']
    ii, jj = np.triu_indices(len(events), 1)
    pairs = pd.DataFrame(dict(idx_i=ii, idx_j=jj, event_i=events.event_uid.to_numpy()[ii],
        event_j=events.event_uid.to_numpy()[jj], is_true_pair=events.source_uid.to_numpy()[ii] == events.source_uid.to_numpy()[jj]))
    if pairs.is_true_pair.sum() != 180:
        raise RuntimeError('Wrong companion count')
    records = []
    for nside in (256, 512):
        U.guard(ROOT)
        begin = time.monotonic()
        npix = hp.nside2npix(nside)
        bank = np.empty((len(events), npix), np.float32)
        order_error = 0.
        for k, row in enumerate(events.itertuples(index=False)):
            if U.sha(row.moc_path) != row.moc_sha256:
                raise RuntimeError('Map hash mismatch')
            with fits.open(row.moc_path) as f:
                coord = str(f[1].header.get('COORDSYS', 'UNKNOWN')).upper()
            if coord not in ('C', 'ICRS', 'EQUATORIAL'):
                raise RuntimeError('Unexpected sky coordinates')
            native = read_sky_map(row.moc_path, moc=True)
            nested = b.apply_temperature(b.raster_probability(native, nside), temp)
            ring = hp.reorder(nested, n2r=True)
            if k == 0:
                order_error = float(abs(hp.reorder(ring, r2n=True)-nested).max())
                if order_error != 0:
                    raise RuntimeError('Ordering conversion round trip failed')
            bank[k] = ring
        norm = bank.sum(axis=1, dtype=np.float64)
        overlap = torch.zeros((len(events), len(events)), dtype=torch.float64, device='cuda')
        bc = torch.zeros_like(overlap)
        for first in range(0, npix, 65536):
            block = torch.as_tensor(np.asarray(bank[:, first:first+65536], np.float64)/norm[:, None], device='cuda')
            overlap += block@block.T
            rootblock = block.sqrt()
            bc += rootblock@rootblock.T
        v, c = overlap.cpu().numpy(), bc.cpu().numpy()
        pairs[f'sky_log_bf_nside{nside}'] = np.log(np.maximum(npix*v[ii, jj], 1e-300))
        pairs[f'sky_BC_nside{nside}'] = c[ii, jj].clip(0, 1)
        if not np.isfinite(v).all():
            raise RuntimeError('Invalid overlap')
        records.append(dict(nside=nside, seconds=time.monotonic()-begin, temperature=temp,
            output_ordering='RING', input_ordering='NESTED', float64_accumulation=True,
            float32_transient_bank=True, dense_persisted=False, ordering_roundtrip_max_error=order_error))
        del bank, overlap, bc, block, rootblock
        torch.cuda.empty_cache()
    pairs['sky_raw_log_bf'] = pairs.sky_log_bf_nside512
    pairs['sky_BC'] = pairs.sky_BC_nside512
    pairs.to_parquet(dest/'pairs.parquet', index=False)
    events.to_parquet(dest/'events.parquet', index=False)
    write(dest/'NUMERICAL_AUDIT.json', records)
    seal(dest/'COMPLETE.json', [dest/'pairs.parquet', dest/'events.parquet', dest/'NUMERICAL_AUDIT.json'],
         events=len(events), pairs=len(pairs), analysis_nside=512, temperature=temp)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--stage', choices=['audit', 'inputs', 'infer', 'maps', 'sky-pairs'], required=True)
    p.add_argument('--run', choices=['O3', 'O4a', 'O4b'], default='O3')
    p.add_argument('--arm', choices=['A_NEUTRAL', 'B_CUE'], default='A_NEUTRAL')
    p.add_argument('--split', choices=['development', 'validation', 'test', 'real'], default='validation')
    p.add_argument('--seed', type=int, default=2026091721)
    p.add_argument('--workers', type=int, default=6)
    a = p.parse_args()
    initialize(a.root, a.out)
    for d in ['contracts', 'logs', 'scripts', 'tables', 'reports', 'manifests']:
        (OUT/d).mkdir(parents=True, exist_ok=True)
    if a.stage == 'audit':
        audit()
    elif a.stage == 'inputs':
        prepare_inputs(a.run, a.arm, a.split)
    elif a.stage == 'infer':
        infer(a.run, a.arm, a.split, a.seed)
    elif a.stage == 'maps':
        maps(a.run, a.arm, a.split, a.workers)
    elif a.stage == 'sky-pairs':
        sky_pairs(a.run, a.arm, a.split)


if __name__ == '__main__':
    main()
