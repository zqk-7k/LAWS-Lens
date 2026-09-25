"""Common-system amplitude and one shared raw-strain realization for C."""
import logging
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.signal import resample_poly

STATE = {}
ARM = 'C_PHYSICAL'
RAW_FS = 4096
SECONDS = 64
CENTER = 48


def worker_init(root, run, role, split):
    import unified_ab as u
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)
    root = Path(root)
    source, sampling, views, physical = u.modules(root)
    source.DURATION, source.N_SAMPLES = SECONDS, SECONDS * RAW_FS
    source.END_AFTER_GEOCENTER_SECONDS = SECONDS - CENTER
    logging.getLogger('bilby').setLevel(logging.ERROR)
    events = pd.read_parquet(root/'plans'/run/'event_plan.parquet')
    events = events[(events.role == role) & (events.split == split)]
    STATE.update(root=root, run=run, u=u, source=source, views=views, physical=physical,
        generator=source.build_waveform_generator(),
        ifos=list(source.bilby.gw.detector.InterferometerList(['H1', 'L1'])),
        events={key: f for key, f in events.groupby('source_uid')},
        noise=np.load(root/'plans'/run/'noise_reference_bank.npy', mmap_mode='r'),
        psds=np.load(root/'plans'/run/'noise_psd_bank.npy', mmap_mode='r'),
        frequency=np.load(root/'plans'/run/'noise_psd_frequency.npy'))


def common_factor(unscaled_snr, uniform):
    """Uniform-volume distance conditional on weakest image optimal rho in [8,40].

    This is a controlled detectable-source population, not the GW-LMC rate model.
    The same amplitude is applied to all images and all noise views of a source.
    """
    rho = np.asarray(unscaled_snr, float)
    if not np.isfinite(rho).all() or np.min(rho) <= 0:
        raise ValueError('Invalid source response')
    distance_fraction = (.2**3 + (1 - .2**3) * float(uniform))**(1/3)
    return 8. / (rho.min() * distance_fraction)


def generate_source(row):
    u, root, run = STATE['u'], STATE['root'], STATE['run']
    physical, src = STATE['physical'], STATE['source']
    folder = root/'data'/run/row['role']/row['split']/row['source_uid']
    marker = folder/'COMPLETE.json'
    if marker.exists():
        receipt = __import__('json').loads(marker.read_text())
        for name, digest in receipt['sha256'].items():
            if u.sha(folder/name) != digest:
                raise RuntimeError('Materialized C source changed')
        return receipt
    u.guard(root)
    folder.mkdir(parents=True, exist_ok=True)
    events = STATE['events'][row['source_uid']].sort_values(['image_number', 'variant'])
    started = time.monotonic()
    clean, peaks, rho0 = {}, {}, []
    for image in sorted(events.image_number.unique()):
        gps = row['gps_a' if image == 1 else 'gps_b']
        signal, peak = src.detector_response(STATE['generator'], STATE['ifos'],
            src.source_parameters(pd.Series(row), gps),
            src.lens_factor(row[f'mu_image{image}'], row[f'morse_image{image}']))
        clean[image], peaks[image] = np.asarray(signal, float), peak
        reference = events[(events.image_number == image) & (events.variant == 0)].iloc[0]
        rho0.append(physical.optimal_network_snr(signal, STATE['frequency'], STATE['psds'][int(reference.noise_bank_index)]))
    rng = np.random.default_rng(u.stable('C-common-distance-v1', run, row['source_uid']))
    factor = common_factor(rho0, rng.random())
    arrays = {'short': [], 'long': [], 'clean': []}
    records, hashes = [], {}
    # The legacy preprocessing expects merger 1.25 s before a 26 s padded end.
    first = int((CENTER - 24.75) * RAW_FS)
    last = first + physical.RAW_PADDED_SAMPLES
    for event in events.to_dict('records'):
        image, bank = int(event['image_number']), int(event['noise_bank_index'])
        offset = int(event['noise_offset_samples'])
        psd = np.asarray(STATE['psds'][bank], float)
        noise = np.asarray(STATE['noise'][bank, :, offset:offset+SECONDS*RAW_FS], float)
        if noise.shape != (2, SECONDS*RAW_FS):
            raise RuntimeError('C noise slice has incorrect dimensions')
        signal = clean[image] * factor
        mixed = (noise + signal).astype(np.float32)
        mixed26, signal26 = mixed[:, first:last], signal[:, first:last]
        prepared = physical.preprocess_24s(mixed26, STATE['frequency'], psd)
        short = STATE['views'].make_window_view(prepared[None], 2)[0]
        low = resample_poly(physical.preprocess_24s(mixed26, STATE['frequency'], psd,
            band_low_hz=20, band_high_hz=80), 1, 8, axis=-1, window=('kaiser', 8.6))[:, -4096:]
        clean_short = STATE['views'].make_window_view(
            physical.preprocess_24s(signal26, STATE['frequency'], psd)[None], 2)[0]
        for kind, value in [('short', short), ('long', low), ('clean', clean_short)]:
            if value.shape != (2, 4096) or not np.isfinite(value).all():
                raise RuntimeError('Invalid waveform feature input')
            arrays[kind].append(value.astype(np.float32))
        rho = physical.optimal_network_snr(signal, STATE['frequency'], psd)
        unscaled = physical.optimal_network_snr(clean[image], STATE['frequency'], psd)
        if not np.isclose(rho/unscaled, factor, rtol=1e-10):
            raise RuntimeError('Common-system scaling identity failed')
        gps = row['gps_a' if image == 1 else 'gps_b']
        record = {**row, **event, **peaks[image], 'run': run, 'arm': ARM, 'gps_obs': gps,
            'image': 'a' if image == 1 else 'b', 'noise_variant': event['variant'],
            'optimal_network_snr': rho, 'recovered_optimal_network_snr': rho,
            'target_network_snr': rho, 'target_network_snr_is_legacy_alias_not_an_input': True,
            'physical_strain_scale_factor': factor, 'common_distance_mpc': row['dl_source']/factor,
            'unscaled_optimal_network_snr': unscaled,
            'ra_true': row['ra'], 'dec_true': row['dec'],
            'morse_index': row[f'morse_image{image}'], 'separate_image_rescaling': False,
            'raw_start_gps': gps-CENTER, 'raw_waveform_crop_start': first,
            'raw_waveform_crop_stop': last}
        # Persist raw strain only for maps: validation/test, and fixed train pilot.
        persist = (row['role'] == 'main' and (row['split'] != 'train' or
                    row['source_uid'] in STATE.get('pilot_sources', set())))
        if not STATE.get('pilot_sources'):
            pilot = pd.read_parquet(root/'plans/C_PILOT_SOURCES.parquet')
            STATE['pilot_sources'] = set(pilot[pilot.run == run].source_uid)
            persist = row['role'] == 'main' and (row['split'] != 'train' or row['source_uid'] in STATE['pilot_sources'])
        if persist and event['variant'] == 0:
            path = folder/(event['event_uid']+'_raw.npz')
            np.savez(path, noisy_raw=mixed, psd=psd, psd_frequency=STATE['frequency'],
                clean_raw=signal.astype(np.float32))
            hashes[path.name] = u.sha(path)
            record.update(raw_strain_path=str(path), raw_strain_sha256=hashes[path.name],
                raw_shared_with_waveform=True)
        records.append(record)
    for kind, values in arrays.items():
        path = folder/f'{ARM}_{kind}.npy'
        np.save(path, np.stack(values)); hashes[path.name] = u.sha(path)
    pd.DataFrame(records).to_parquet(folder/'metadata.parquet', index=False)
    hashes['metadata.parquet'] = u.sha(folder/'metadata.parquet')
    receipt = dict(source_uid=row['source_uid'], run=run, role=row['role'], split=row['split'],
        sha256=hashes, seconds=time.monotonic()-started, both_arms=False,
        common_amplitude_factor=factor, same_factor_all_images_and_views=True,
        reference_optimal_snr_before=rho0, reference_optimal_snr_after=(np.array(rho0)*factor).tolist(),
        not_a_cosmological_rate_population=True)
    u.write(marker, receipt)
    return receipt
