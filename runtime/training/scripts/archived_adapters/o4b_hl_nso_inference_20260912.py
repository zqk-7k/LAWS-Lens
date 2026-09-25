#!/usr/bin/env python3
"""UID-aligned waveform inference for O4b; no calibration or rank selection.

The 16 s branch is reconstructed from the same physical signal and noise as
the archived 2 s view, never by extending/resampling a two-second input.
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
import argparse
import fcntl
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from scipy.special import softmax

import o4b_hl_nso_data_training_20260912 as s
import o4b_hl_nso_multiscale_training_20260912 as models
import o4b_hl_nso_bayestar_20260912 as sky
import o4b_hl_nso_rankncontrast_v2_20260912 as rnc_adapter


def guard(root, split):
    if split in ('test', 'real'):
        freeze = root/'contracts/FINAL_SCORE_CONFIG_FREEZE.json'
        if not freeze.exists():
            raise RuntimeError('Test/real inference remains locked')
        for item in json.loads(freeze.read_text())['files']:
            if s.sha(item['path']) != item['sha256']:
                raise RuntimeError('Final score configuration hash changed')


def short_module(root):
    ws, *_ = s.environment(root)
    return s.load(ws/'scripts/real_search/37_unified_intrinsic_multitask_pilot.py', 'o4b_infer_short')


def simulation_inputs(root, seed, split):
    guard(root, split)
    out = root/f'inference_inputs/seed_{seed}/{split}'
    if (out/'COMPLETE.json').exists():
        return out
    out.mkdir(parents=True, exist_ok=True)
    low, _, _, _ = models.initialize(root)
    ws, shared, _, phys, _ = s.environment(root)
    plan = pd.DataFrame(sky.records(root, seed, split))
    compact = shared.parent/f'seed_{seed}/data/real_noise_injections'
    meta = pd.read_parquet(compact/'compact_injection_metadata.parquet').set_index(['family', 'sample_index'])
    refs = np.load(shared/'noise_reference_bank.npy', mmap_mode='r')
    psds = np.load(shared/'noise_psd_bank.npy', mmap_mode='r')
    freq = np.load(shared/'noise_psd_frequency.npy')
    short = short_module(root)
    raw = np.empty((len(plan), 2, 4096), np.float32)
    short_raw = np.empty_like(raw)
    long = np.empty_like(raw)
    arrays, audits = {}, []
    for row in plan.itertuples():
        key = row.family, row.image_number
        if key not in arrays:
            folder = 'Unlensed_data_0222' if row.family == 'unlensed' else f'{row.family}_data_0222'
            mixed_name = 'unlensed_data_strain.npy' if row.family == 'unlensed' else f'{row.family}_data_strain_{row.image_number}.npy'
            pure_name = 'unlensed_h_strain.npy' if row.family == 'unlensed' else f'{row.family}_h_strain_{row.image_number}.npy'
            arrays[key] = (np.load(compact/'matchroots/LIGO'/folder/mixed_name, mmap_mode='r'),
                           np.load(shared/'physical_h1l1_source_bank'/folder/pure_name, mmap_mode='r'))
        stored, clean = arrays[key]
        record = meta.loc[(row.family, row.source_index)]
        bank = int(row.noise_bank_index)
        offset = int(record[f'image{row.image_number}_noise_offset_samples'])
        noise = np.asarray(refs[bank, :, offset:offset+phys.RAW_PADDED_SAMPLES])
        scaled, factor, recovered = phys.scale_to_network_snr(clean[row.source_index], row.target_network_snr, freq, psds[bank])
        mixed = noise + phys.embed_signal_in_padded_window(scaled)
        replay = phys.preprocess_24s(mixed, freq, psds[bank])
        reference = np.asarray(stored[row.source_index])
        if not np.array_equal(replay, reference):
            raise RuntimeError(f'Physical replay differs from frozen catalog: {row.event_uid}:{abs(replay-reference).max()}')
        raw[row.idx] = low.dev.TRAIN.make_window_view(reference[None], 2)[0]
        short_raw[row.idx] = short.prepare(reference, None, False)
        long[row.idx] = low.low_view(mixed, freq, psds[bank], phys)
        audits.append({'event_uid': row.event_uid, 'physical_replay_exact': True,
                       'scale_factor': factor, 'optimal_network_snr': recovered})
    if not all(np.isfinite(x).all() for x in (raw, short_raw, long)):
        raise RuntimeError('Nonfinite waveform view')
    plan.to_parquet(out/'events.parquet', index=False)
    pd.DataFrame(audits).to_parquet(out/'replay_audit.parquet', index=False)
    for name, value in [('raw2s', raw), ('short_raw2s', short_raw), ('low16s', long)]:
        np.save(out/f'{name}.npy', value)
    s.write(out/'COMPLETE.json', {'events': len(plan), 'replay_exact': True,
                                'files': [{'path': str(p), 'sha256': s.sha(p)} for p in out.glob('*.npy')]})
    return out


def get_context(root, seed, split):
    if split == 'development':
        folder = root/'auxiliary_data_v2/validation'
        events = pd.read_parquet(folder/'event_metadata.parquet').copy()
        events['event_uid'] = events.source_uid.astype(str)+'_'+events.image.astype(str)
        events['global_source_id'] = events.source_uid
        events['idx'] = np.arange(len(events))
        features = root/'waveform_features/auxiliary_v2/validation'
    elif split == 'real':
        folder = root/'inference_inputs/real'
        if not (folder/'COMPLETE.json').exists():
            raise RuntimeError('Independent real waveform preparation required')
        events = pd.read_parquet(folder/'events.parquet')
        features = folder/'features'
    else:
        folder = simulation_inputs(root, seed, split)
        events = pd.read_parquet(folder/'events.parquet')
        features = folder/'features'
    shared = root/'workspace/results/real_noise_injection_v5_physical_source_20260721/gwtc5/shared'
    if split == 'real':
        freq = np.load(folder/'frequency.npy')
        psds = np.load(folder/'psd.npy', mmap_mode='r')
    else:
        freq = np.load(shared/'noise_psd_frequency.npy')
        psds = np.load(shared/'noise_psd_bank.npy', mmap_mode='r')
    return folder, events, features, freq, psds


def features(root, seed, split):
    guard(root, split)
    low, mult, cond, operator = models.initialize(root)
    folder, events, dest, freq, psds = get_context(root, seed, split)
    dest.mkdir(parents=True, exist_ok=True)
    ids = events.noise_bank_index.to_numpy(int)
    raw = np.load(folder/'raw2s.npy', mmap_mode='r')
    path = dest/'short.npy'
    if not path.with_suffix('.COMPLETE.json').exists():
        spectra = np.load(operator/'cache/short253_spectra.npy', mmap_mode='r')
        value = np.empty((len(events), 27, 253), np.float32)
        for bank in np.unique(ids):
            keep = np.flatnonzero(ids == bank)
            templates = low.t.adaptive.whitened_bank(spectra, freq, psds[bank])
            x = low.t.fine.features(raw[keep], templates)
            value[keep] = x.reshape(len(x), 3, 253, 3, 3).transpose(0, 1, 3, 4, 2).reshape(len(x), 27, 253)
        np.save(path, value)
        s.write(path.with_suffix('.COMPLETE.json'), {'sha256': s.sha(path), 'events': len(events)})
    low.grouped(operator, np.load(folder/'low16s.npy', mmap_mode='r'), freq, psds, ids, dest/'long.npy')
    rout, psd, trainer, tf = rnc_adapter.setup(root)
    if split == 'development':
        rpath = rout/'cache/finelag/gwtc5/validation_features.npy'
    else:
        rpath = dest/'rnc.npy'
        psd.extract_grouped(rout, raw, freq, psds, ids, rpath)
    return folder, events, dest, rpath


@torch.no_grad()
def infer(root, seed, split):
    guard(root, split)
    out = root/f'predictions_o4b/seed_{seed}/{split}'
    if (out/'COMPLETE.json').exists():
        for item in json.loads((out/'COMPLETE.json').read_text())['outputs']:
            if s.sha(item['path']) != item['sha256']:
                raise RuntimeError('Frozen inference output changed')
        return
    out.mkdir(parents=True, exist_ok=True)
    folder, events, feature_root, rpath = features(root, seed, split)
    low, mult, cond, _ = models.initialize(root)
    old = low.old
    rout, psd, trainer, tf = rnc_adapter.setup(root)
    inputs = []

    def checkpoint(path):
        inputs.append({'path': str(path), 'sha256': s.sha(path)})
        return torch.load(path, weights_only=False, map_location='cpu')

    raw = np.asarray(np.load(folder/'raw2s.npy'), np.float32)
    module = short_module(root)
    ck = checkpoint(root/f'models/short_encoder/seed_{seed}/validation_selected_model.pt')
    net = module.PhysicsRegularizedEncoder(module.build_base_encoder('inception_attention', None)).cuda().eval()
    net.load_state_dict(ck['model_state'])
    short_path = folder/'short_raw2s.npy'
    short_raw = np.load(short_path) if short_path.exists() else np.stack([module.prepare(x, None, False) for x in raw])
    emb, means = [], []
    for start in range(0, len(events), 128):
        e, mu = net(torch.as_tensor(short_raw[start:start+128], device='cuda'), return_parameters=True)
        emb.append(e.cpu().numpy()); means.append(mu.cpu().numpy())
    emb, means = np.concatenate(emb), np.concatenate(means)
    np.savez_compressed(out/'short.npz', z=emb, mean_standardized=means,
                        mean=means*np.asarray(ck['target_std'])+np.asarray(ck['target_mean']))
    del net

    ck = checkpoint(rout/f'models/RAW-PHASE-SOURCE/gwtc5/seed_{seed}/validation_selected_model.pt')
    net = trainer.b.Encoder('RAW-PHASE-SOURCE').cuda().eval(); net.load_state_dict(ck['model'])
    x = np.asarray(np.load(rpath), np.float32)
    logits, z = trainer.b.infer(net, (x-ck['mu'])/ck['sd'], raw)
    prob = softmax(logits.astype(float)/ck['temperature'], axis=-1)
    np.savez_compressed(out/'rnc.npz', p=prob, z=z)
    del net

    short_x = np.asarray(np.load(feature_root/'short.npy'), np.float32)
    long_x = np.asarray(np.load(feature_root/'long.npy'), np.float32)
    x = np.concatenate([short_x, long_x], axis=1)
    for kind, values, path in (
        ('ordered', short_x, root/f'ordered_mass_predictor/models/gwtc5/seed_{seed}/validation_selected_model.pt'),
        ('multirate', x, root/f'models/MULTIRATE/gwtc5/seed_{seed}/selected.pt')):
        ck = checkpoint(path)
        net = (old.Predictor() if kind == 'ordered' else mult.Predictor('MULTIRATE')).cuda().eval()
        net.load_state_dict(ck['model'])
        p, outside = old.probability(old.infer(net, (values-ck['mu'])/ck['sd']), ck['temperature'])
        np.savez_compressed(out/f'{kind}.npz', p=p, outside=outside, prior=ck['prior'])
        del net

    ck = checkpoint(root/f'models/CONDITIONAL-ETA-CHI/gwtc5/seed_{seed}/selected.pt')
    mass_ck = checkpoint(Path(ck['mass_checkpoint']))
    if inputs[-1]['sha256'] != ck['mass_checkpoint_sha256']:
        raise RuntimeError('Conditional mass backbone changed')
    rep = cond.original.representations(x, mass_ck)
    head = cond.Head(ck['conditional_dimensions']).cuda().eval(); head.load_state_dict(ck['model'])
    residual = cond.original.infer(head, ((rep-ck['mu'])/ck['sd']).reshape(-1, 103)).reshape(len(rep), 253, -1)
    pc = softmax(residual/ck['temperature']+np.log(ck['conditional_prior'].clip(1e-12))[None], axis=-1)
    hi = np.searchsorted(old.LOG_CENTERS, old.CENTERS, side='right').clip(1, 252); lo = hi-1
    w = ((old.CENTERS-old.LOG_CENTERS[lo])/(old.LOG_CENTERS[hi]-old.LOG_CENTERS[lo])).clip(0, 1)
    pc = (1-w[None, :, None])*pc[:, lo]+w[None, :, None]*pc[:, hi]
    mass = dict(np.load(out/'multirate.npz'))
    joint = (mass['p'][:, :, None]*pc).astype(np.float32)
    error = float(abs(joint.astype(float).sum(-1)-mass['p']).max())
    if error > 1e-6:
        raise RuntimeError('Joint prediction changed frozen Mc marginal')
    # Full joint predictions are retained per event, never replicated per pair.
    np.savez_compressed(out/'joint.npz', p=joint, outside=mass['outside'])
    events.to_parquet(out/'events.parquet', index=False)
    ii, jj = np.triu_indices(len(events), 1)
    pairs = pd.DataFrame({'idx_i': ii, 'idx_j': jj, 'event_i': events.event_uid.to_numpy()[ii],
                          'event_j': events.event_uid.to_numpy()[jj]})
    if split != 'real':
        groups = events.global_source_id.to_numpy(str)
        pairs['is_true_pair'] = groups[ii] == groups[jj]
    short_values = dict(np.load(out/'short.npz'))
    pairs['waveform_embedding_cosine'] = (short_values['z']@short_values['z'].T)[ii, jj]
    gaps = abs(short_values['mean_standardized'][ii]-short_values['mean_standardized'][jj])
    pairs['waveform_abs_delta_logmc_std'], pairs['waveform_abs_delta_logitq_std'] = gaps.T
    rnc = dict(np.load(out/'rnc.npz'))
    pairs['rnc_cosine'] = (rnc['z']@rnc['z'].T)[ii, jj]
    pairs['rnc_BC'] = (np.sqrt(rnc['p'])@np.sqrt(rnc['p']).T)[ii, jj].clip(1e-15, 1)
    ordered = dict(np.load(out/'ordered.npz'))
    pairs['ordered_BC'] = (np.sqrt(ordered['p'])@np.sqrt(ordered['p']).T)[ii, jj].clip(1e-15, 1)
    pairs['ordered_log_prior_overlap'] = np.log(((ordered['p']/np.sqrt(ordered['prior'])) @
                                                (ordered['p']/np.sqrt(ordered['prior'])).T)[ii, jj].clip(1e-300))
    pairs['ordered_ood'] = (ordered['outside'][ii] > .25) | (ordered['outside'][jj] > .25)
    matrix = torch.as_tensor(np.sqrt(joint.astype(np.float64)).reshape(len(events), -1), device='cuda')
    joint_bc = (matrix@matrix.T).cpu().numpy()[ii, jj].clip(1e-15, 1)
    del matrix
    pairs['joint_BC'] = joint_bc
    pairs['joint_log_BC'] = np.log(joint_bc)
    pairs['joint_ood'] = (mass['outside'][ii] > .25) | (mass['outside'][jj] > .25)
    if not np.isfinite(pairs.select_dtypes('number')).all().all():
        raise RuntimeError('Nonfinite waveform pair features')
    pairs.to_parquet(out/'waveform_features.parquet', index=False)
    files = list(out.glob('*.npz'))+list(out.glob('*.parquet'))
    s.write(out/'COMPLETE.json', {'utc': s.now(), 'events': len(events), 'pairs': len(pairs),
                                'joint_mass_marginal_error': error, 'inputs': inputs,
                                'outputs': [{'path': str(p), 'sha256': s.sha(p)} for p in files]})
    print(json.dumps({'inference_complete': [seed, split], 'events': len(events)}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--split', choices=('development', 'validation', 'test', 'real'), required=True)
    parser.add_argument('--stage', choices=('inputs', 'infer'), default='infer')
    args = parser.parse_args()
    if args.stage == 'inputs':
        simulation_inputs(args.root, args.seed, args.split)
    else:
        with (args.root/'contracts/GPU_TRAINING.lock').open('a') as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            infer(args.root, args.seed, args.split)
