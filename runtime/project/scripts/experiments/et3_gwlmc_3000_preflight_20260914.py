#!/usr/bin/env python3
"""Bounded, immutable ET/GW-LMC development audit before any bulk generation.

This executable intentionally cannot authorize production when the frozen HL
reference fails the method-identity audit. Numerical controls are diagnostics,
not a replacement for the requested end-to-end experiment.
"""
import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "4" if _key == "OMP_NUM_THREADS" else "1"
os.environ["OMP_STACKSIZE"] = "512M"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
import argparse
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import logging
import math
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import tarfile
import time
import traceback

import numpy as np
import pandas as pd

P = Path('/root/autodl-tmp/gw-catalog')
REF = P/'results/bayestar_injection_sky_pe_20260901_20260901T102000Z'
PATH875 = P/'results/mcwf_unified_path875_devconf_20260908T181500Z'
OLD_ET = P/'results/et3_new_score_only_bayestar_20260912T014101Z'
CAT = Path('/root/autodl-tmp/GW-LMC/2.5PLUS/BBH/Any_Detected_SNR1')
EXPECTED = '3ea3cf13992a5154e0dd757ab6087b73344cf28eb0ccf5e4a8a903b0e83850af'
FINAL = 'HOLD_FOR_AUTHOR_REVIEW_NO_ADOPTION_NO_OVERWRITE'
SEEDS = [2026091411, 2026091412, 2026091413]


def utc():
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False,
        default=lambda x: x.item() if isinstance(x, np.generic) else str(x))+'\n')
    temporary.replace(path)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def seed(*parts):
    return int.from_bytes(hashlib.sha256('|'.join(map(str, parts)).encode()).digest()[:4], 'little')


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    out = importlib.util.module_from_spec(spec)
    sys.modules[name] = out
    spec.loader.exec_module(out)
    return out


def disk_guard(root):
    if shutil.disk_usage(root).free < 30*2**30:
        raise RuntimeError('HOLD_DISK_LIMIT')


def dependencies(root):
    sys.path.insert(0, str(P))
    source = module(root/'scripts/34_generate_physical_h1l1_source_bank.py', 'frozen_source')
    geom = module(root/'scripts/et3_new_score_only_20260912.py', 'frozen_et_geometry')
    from scripts.real_search import physical_common
    return source, geom, physical_common


def prepare(root):
    root.mkdir(parents=True, exist_ok=False)
    for folder in ['contracts', 'scripts', 'logs', 'manifests', 'tables', 'reports',
                   'pilot/strain', 'pilot/maps', 'pilot/jobs', 'figures', 'package']:
        (root/folder).mkdir(parents=True)
    shutil.copy2(__file__, root/'scripts'/Path(__file__).name)
    sources = [REF/'scripts/bayestar_injection_sky_full_experiment.py',
        OLD_ET/'scripts/et3_new_score_only_20260912.py',
        P/'scripts/real_search/34_generate_physical_h1l1_source_bank.py',
        P/'scripts/real_search/physical_common.py']
    for f in sources:
        shutil.copy2(f, root/'scripts'/f.name)
    protected = sources + list(CAT.glob('*.csv'))
    protected += list(PATH875.glob('contracts/*.json'))
    protected += [PATH875/'independent_confirmation/contracts/FRESH_FREEZE.json']
    protected += list(OLD_ET.glob('diagnostics/quadrature_build/q*/core*.so'))
    package = P/'packages/bayestar_injection_sky_pe_20260901_20260901T102000Z_deliverables.tar.gz'
    protected.append(package)
    manifest = [{'path': str(f), 'sha256': sha(f), 'bytes': f.stat().st_size} for f in protected]
    write(root/'manifests/PROTECTED_INPUTS.json', manifest)
    if sha(package) != EXPECTED:
        raise RuntimeError('BAYESTAR baseline package fingerprint failed')
    contract = {
        'id': 'ET3-GWLMC3000-UNIFIED-NSO-PREFLIGHT', 'utc': utc(),
        'requested_target': '3000 events, pilot before bulk, shared BAYESTAR/NEW-SCORE-ONLY method',
        'target_event_counts': {'smooth_images': 1200, 'subhalo_images': 1200, 'unlensed': 600},
        'target_system_counts': {'smooth': 600, 'subhalo': 600, 'unlensed': 600},
        'split_systems_per_population': {'train': 420, 'validation': 90, 'test': 90},
        'model_seeds_reserved': SEEDS, 'maps_shared_between_model_seeds': True,
        'model_seed_jobs_started': False, 'test_opened': False,
        'source_unit': 'global GW-LMC event_id; deduplicated across all environment rows',
        'pilot': {'events': 24, 'lensed_systems': 8, 'unlensed_systems': 8,
                  'selection': 'hashed training-only IDs, 4 systems per lens environment',
                  'network_snr_anchor': [9., 11., 16., 24.],
                  'purpose': 'development stress test, not population-unbiased coverage estimation'},
        'generation': {'waveform': 'IMRPhenomXPHM, 20Hz reference, frozen physical-source generator',
            'geometry': 'Bilby ET1/2/3 vertices and response tensors in both forward/inverse adapters',
            'noise': 'independent Gaussian ET design PSD channels; not real ET noise',
            'strain': 'physical dimensionless 24s/4096Hz plus 1s noise padding at each end',
            'SNR': 'PSD optimal 20-1024Hz; per-image target-SNR scaling, as frozen HL injection protocol',
            'time': 'controlled 10-Julian-year full-duty ET synthetic exposure; not a real ET calendar'},
        'input': {'short_seconds': [-1.75, .25], 'short_shape': [3, 4096], 'short_hz': 2048,
            'long_seconds': [-15.75, .25], 'long_shape': [3, 4096], 'long_hz': 256,
            'long_band_hz': [20, 80], 'anti_alias': 'polyphase Kaiser 8.6'},
        'NEW_SCORE_ONLY': {'alpha': 1, 'Zwf': 'Zwf_OMC + gamma*T + beta*I',
            'short_encoder_and_existing_mass_terms_retained': True,
            'six_HL_recipes': str(PATH875/'independent_confirmation/contracts/FRESH_FREEZE.json'),
            'ET_training_and_calibration': 'not started until identity and numerical gates pass'},
        'sky': {'formal_nside': 512, 'audit_nsides': [256, 512, 1024],
            'waveform': 'IMRPhenomPv2, 20Hz reference; no automatic fallback',
            'likelihood_scale': .83, 'temperature_in_numerical_audit': 1.,
            'spin_mass_units': 'SI kg required, legacy solar-unit call evaluated only as diagnostic',
            'data_boundary': 'reference uses synthetic Gaussian trigger series conditional on source truth; not extraction from generated noisy strain',
            'numerical_controls': [10, 32, 64],
            'numerical_control_meaning': 'inclination/polarization quadrature, NOT HEALPix Nside',
            'no_formal_ET_only_core_upgrade': True},
        'pilot_gates_frozen_before_maps': {
            'reference_identity': 'no unresolved scientific unit/interface mismatch with HL reference',
            'data': 'all finite; 3 distinct channels; PSD SNR relative error <=1e-5; exact HL preprocessing replay',
            'quadrature_32_64': {'max_map_TV': .02, 'max_relative_A90': .05,
                                'max_absolute_pair_logBF_delta': .2},
            'nside_512_1024': {'pair_abs_delta_q99_max': .05, 'pair_abs_delta_max': .2},
            'runtime': {'q64_P90_seconds_max': 900},
            'coverage': 'descriptive source-block CI only; this tiny stratified pilot cannot establish 90% population coverage'},
        'gate_provenance': 'engineering screening tolerances, not universal physics constants; frozen before new maps',
        'production_condition': 'all preceding gates pass AND consistent HL/ET scientific implementation; no test-based repair',
        'on_failure': 'finish bounded diagnostics, preserve failures, no bulk maps/no model training',
        'references': ['https://bilby-dev.github.io/bilby/api/bilby.gw.conversion.bilby_to_lalsimulation_spins.html',
            'https://lscsoft.docs.ligo.org/ligo.skymap/tool/bayestar_realize_coincs.html',
            'https://arxiv.org/abs/1508.03634'], 'final_status': FINAL}
    write(root/'contracts/ANALYSIS_CONTRACT.json', contract)
    write(root/'contracts/CONTRACT_HASH.json', {'sha256': sha(root/'contracts/ANALYSIS_CONTRACT.json')})
    src, geom, phys = dependencies(root)
    table = src.load_tables(CAT)
    originals = [pd.read_csv(f) for f in CAT.glob('*.csv')]
    if not all(np.array_equal(originals[0].event_id, x.event_id) for x in originals[1:]):
        raise RuntimeError('GW-LMC source/lens/image row alignment failed')
    rows, used = [], set()
    for family, subhalo in [('smooth', False), ('subhalo', True), ('unlensed', None)]:
        candidates = table if subhalo is None else table[table.lens_is_subhalo == subhalo]
        candidates = candidates.assign(_order=[seed('ET3000-plan-v1', int(x)) for x in candidates.event_id])
        chosen = []
        for _, x in candidates.sort_values(['_order', 'gwlmc_row']).iterrows():
            uid = int(x.event_id)
            if uid in used:
                continue
            images = src.choose_images(x) if subhalo is not None else None
            if subhalo is not None and (images is None or images['proposal_snr_ratio'] > 4):
                continue
            used.add(uid)
            i = len(chosen)
            split = 'train' if i < 420 else 'validation' if i < 510 else 'test'
            chosen.append({'family': family, 'source_id': uid, 'gwlmc_row': int(x.gwlmc_row),
                           'split': split, 'images': images})
            if len(chosen) == 600:
                break
        if len(chosen) != 600:
            raise RuntimeError(f'Insufficient independent GW-LMC population for {family}: {len(chosen)}')
        rows.extend(chosen)
    write(root/'contracts/RESERVED_SYSTEM_SPLITS.json', rows)
    split_counts = pd.DataFrame(rows).groupby(['family', 'split']).size().rename('systems').reset_index()
    split_counts.to_csv(root/'tables/SOURCE_SPLIT_COUNTS.csv', index=False)
    development = []
    for family, count in [('smooth', 4), ('subhalo', 4), ('unlensed', 8)]:
        selection = [r for r in rows if r['family'] == family and r['split'] == 'train'][:count]
        for j, r in enumerate(selection):
            source = table.iloc[r['gwlmc_row']]
            rng = np.random.default_rng(seed('ET-pilot-gps', r['source_id']))
            delay = r['images']['delay_days']*86400 if r['images'] else 0
            t0 = 1126259462.+rng.uniform(0, 10*365.25*86400-delay)
            for image in ([1, 2] if r['images'] else [0]):
                images = r['images']
                faint = [9., 11., 16., 24.][j % 4]
                ratio = images['proposal_snr_ratio'] if images else 1.
                faint = min(faint, 60./ratio)
                proposal = images[f'proposal_snr_image{image}'] if images else 1.
                denominator = min(images['proposal_snr_image1'], images['proposal_snr_image2']) if images else 1.
                development.append({**source.to_dict(), 'event_uid': f'ETDEV_{r["source_id"]}_{image}',
                    'family': family, 'source_id': r['source_id'], 'image': image, 'split': 'train-development',
                    'gps': t0+(delay if image == 2 else 0),
                    'morse': images[f'morse_image{image}'] if images else 0.,
                    'magnification': images[f'mu_image{image}'] if images else 1.,
                    'target_snr': faint*proposal/denominator,
                    'measurement_seed': seed('ETDEV-bayestar', r['source_id'], image),
                    'noise_seed': seed('ETDEV-strain-noise', r['source_id'], image)})
    assert len(development) == 24 and len({r['source_id'] for r in development}) == 16
    write(root/'contracts/PILOT_EVENTS.json', development)
    write(root/'contracts/SOURCE_INDEPENDENCE.json', {
        'rows_in_catalog': len(table), 'global_ids_in_catalog': table.event_id.nunique(),
        'reserved_unique_source_ids': len(used), 'split_intersections': 0,
        'pilot_only_training': True, 'pilot_events': 24, 'pilot_unique_sources': 16,
        'claim_scope': 'within this new experiment; not independent of all historical GW-LMC use',
        'bulk_strain_generated': 0, 'test_strain_or_scores_opened': False})
    import bilby, lal
    baseline = module(root/'scripts/bayestar_injection_sky_full_experiment.py', 'frozen_sky_reference')
    unit_rows = []
    for r in development:
        wrong_iota, wrong_spin = baseline.spin_components(pd.Series(r))
        values = [float(r[k]) for k in ['theta_jn', 'phijl', 'tilt1', 'tilt2', 'phi12', 'a1', 'a2']]
        correct = bilby.gw.conversion.bilby_to_lalsimulation_spins(*values,
            float(r['m1_det'])*lal.MSUN_SI, float(r['m2_det'])*lal.MSUN_SI, 20., float(r['phase']))
        unit_rows.append({'event_uid': r['event_uid'], 'legacy_iota': wrong_iota,
            'SI_iota': correct[0], 'abs_iota_delta_rad': abs(wrong_iota-correct[0]),
            'max_cartesian_spin_delta': float(np.max(np.abs(np.array(wrong_spin)-correct[1:])))})
    pd.DataFrame(unit_rows).to_csv(root/'tables/SPIN_MASS_UNIT_AUDIT.csv', index=False)
    write(root/'contracts/METHOD_IDENTITY_AUDIT.json', {
        'baseline_package_sha256_verified': True,
        'spin_interface_requires_SI': 'All parameters are defined at the reference frequency and in SI units.' in
            bilby.gw.conversion.bilby_to_lalsimulation_spins.__doc__,
        'events_with_nonzero_unit_effect': sum(r['abs_iota_delta_rad'] > 1e-10 or r['max_cartesian_spin_delta'] > 1e-10 for r in unit_rows),
        'max_iota_delta_rad': max(r['abs_iota_delta_rad'] for r in unit_rows),
        'reference_trigger_uses_generated_strain': False,
        'status': 'FAIL_UNRESOLVED_HL_REFERENCE_SI_MASS_UNITS',
        'consequence': 'ET SI-corrected numerical controls are NOT a completed unified comparison; historical HL unchanged'})
    write(root/'contracts/GEOMETRY.json', geom.geometry()[2])
    print(json.dumps({'stage': 'prepared', 'root': str(root), 'pilot_events': 24}), flush=True)


def preprocess_generic(phys, padded, frequency, psds, low=40., high=580.):
    from scipy import signal
    white = phys.whiten_with_psd(padded, frequency, psds)
    filtered = signal.sosfiltfilt(signal.butter(6, [low, high], btype='bandpass', fs=4096, output='sos'), white, axis=-1)
    down = signal.resample_poly(filtered, 1, 2, axis=-1, window=('kaiser', 8.6))
    return phys.robust_scale_channels(down[:, 2048:2048+24*2048])


def generate(root):
    import bilby
    from scipy.signal import resample_poly
    logging.getLogger('bilby').setLevel(logging.ERROR)
    src, geom, phys = dependencies(root)
    generator = src.build_waveform_generator()
    records = json.loads((root/'contracts/PILOT_EVENTS.json').read_text())
    results = []
    for r in records:
        disk_guard(root)
        started = time.perf_counter()
        ifos, _, _ = geom.geometry()
        parameters = src.source_parameters(pd.Series(r), r['gps'])
        clean, peaks = src.detector_response(generator, ifos, parameters, src.lens_factor(r['magnification'], r['morse']))
        frequency = np.arange(8193)*.25
        psds = np.array([ifo.power_spectral_density.get_power_spectral_density_array(frequency) for ifo in ifos])
        psds = np.where((frequency[None, :] >= 20)&np.isfinite(psds)&(psds > 0), psds, np.inf)
        scaled, factor, snr = phys.scale_to_network_snr(clean, r['target_snr'], frequency, psds)
        np.random.seed(r['noise_seed'])
        bilby.core.utils.random.seed(r['noise_seed'])
        noise_rows = []
        # Bilby can retain the time-domain cache on an interferometer previously
        # populated with a 24 s injection. Use fresh instances for 26 s noise.
        noise_ifos, _, _ = geom.geometry()
        for ifo in noise_ifos:
            ifo.minimum_frequency = 20.
            ifo.set_strain_data_from_power_spectral_density(4096, 26., start_time=r['gps']-24.75)
            noise_rows.append(np.asarray(ifo.strain_data.time_domain_strain))
        noise = np.asarray(noise_rows)
        if noise.shape != (3, 26*4096):
            raise RuntimeError(f'Noise duration/cache mismatch: {noise.shape}')
        padded = noise.copy()
        padded[:, 4096:4096+24*4096] += scaled
        full = preprocess_generic(phys, padded, frequency, psds)
        reference = phys.preprocess_24s(padded[:2], frequency, psds[:2])
        short = full[:, -4096:]
        long = resample_poly(preprocess_generic(phys, padded, frequency, psds, 20., 80.),
                             1, 8, axis=-1, window=('kaiser', 8.6))[:, -4096:]
        exact = bool(np.array_equal(full[:2], reference))
        check = bool(np.isfinite(padded).all() and np.isfinite(short).all() and np.isfinite(long).all()
                     and exact and abs(snr-r['target_snr']) <= 1e-5*r['target_snr'])
        if short.shape != (3, 4096) or long.shape != (3, 4096):
            raise RuntimeError('Window shape mismatch')
        path = root/'pilot/strain'/f'{r["event_uid"]}.npz'
        np.savez_compressed(path, clean=scaled, noisy_padded=padded.astype(np.float32), short=short,
            long=long.astype(np.float32), psd_frequency=frequency, psd=psds)
        pair_equal = [bool(np.array_equal(padded[a], padded[b])) for a, b in [(0, 1), (0, 2), (1, 2)]]
        results.append({'event_uid': r['event_uid'], 'source_id': r['source_id'], 'family': r['family'],
            'physical_strain_generated': True, 'pass': check and not any(pair_equal), 'short_HL_operator_exact': exact,
            'snr': snr, 'target_snr': r['target_snr'], 'amplitude_scale': factor,
            'channel_exact_equal_count': sum(pair_equal), 'seconds': time.perf_counter()-started,
            'path': str(path), 'sha256': sha(path), 'bytes': path.stat().st_size,
            'raw_peak_indices': np.argmax(np.abs(scaled), axis=-1).tolist()})
        print(json.dumps({'stage': 'physical_pilot', 'complete': len(results), 'total': 24, 'pass': results[-1]['pass']}), flush=True)
    pd.DataFrame(results).to_csv(root/'tables/PHYSICAL_STRAIN_PILOT.csv', index=False)
    write(root/'contracts/DATA_AUDIT.json', {'events': len(results), 'all_pass': all(r['pass'] for r in results),
        'generated_raw_strain_used_for_sky': False,
        'reason': 'Frozen HL BAYESTAR consumes conditional synthetic triggers, not this noisy strain realization'})


def load_core(q):
    import ligo.skymap
    if 'ligo.skymap.core' in sys.modules:
        raise RuntimeError('Core already imported')
    core = module(OLD_ET/f'diagnostics/quadrature_build/q{q}/core.abi3.so', 'ligo.skymap.core')
    ligo.skymap.core = core
    core.set_num_threads(4)
    return core


def sky_worker(root, case):
    started = time.perf_counter()
    soft, hard = resource.getrlimit(resource.RLIMIT_STACK)
    resource.setrlimit(resource.RLIMIT_STACK, (min(512*2**20, hard) if hard > 0 else 512*2**20, hard))
    load_core(case['q'])
    import bilby, lal
    from ligo.skymap.bayestar import filter, localize
    from ligo.skymap.tool.bayestar_realize_coincs import simulate_snr
    from ligo.skymap.io.events.base import Event, SingleEvent
    from ligo.skymap.io.fits import write_sky_map
    from ligo.skymap import moc
    _, geom, _ = dependencies(root)
    logging.getLogger('bilby').setLevel(logging.ERROR)
    r = json.loads((root/'contracts/PILOT_EVENTS.json').read_text())[case['index']]
    ifos, detectors, _ = geom.geometry()
    values = [float(r[k]) for k in ['theta_jn', 'phijl', 'tilt1', 'tilt2', 'phi12', 'a1', 'a2']]
    unit = 1. if case.get('legacy_units') else lal.MSUN_SI
    spins = bilby.gw.conversion.bilby_to_lalsimulation_spins(*values,
        float(r['m1_det'])*unit, float(r['m2_det'])*unit, 20., float(r['phase']))
    args = {'mass1': r['m1_det'], 'mass2': r['m2_det'], 'f_final': 1024., 'f_ref': 20.}
    args.update(dict(zip(['spin1x', 'spin1y', 'spin1z', 'spin2x', 'spin2y', 'spin2z'], spins[1:])))
    template = filter.sngl_inspiral_psd('IMRPhenomPv2', f_min=20., **args)
    frequency = np.arange(8193)*.25
    psds, interpolated = [], []
    for ifo in ifos:
        raw = ifo.power_spectral_density.get_power_spectral_density_array(frequency)
        raw = np.where((frequency >= 20)&np.isfinite(raw)&(raw > 0), raw, np.inf)
        psd = lal.CreateREAL8FrequencySeries('ET design PSD', 0, 0., .25, lal.StrainUnit**2, len(raw))
        psd.data.data[:] = raw
        psds.append(psd)
        interpolated.append(filter.InterpolatedPSD(filter.abscissa(psd), psd.data.data))
    epoch = lal.LIGOTimeGPS(r['gps'])
    common = dict(ra=r['ra'], dec=r['dec'], psi=r['psi'], inc=spins[0], epoch=epoch,
                  gmst=lal.GreenwichMeanSiderealTime(epoch), H=template)
    def realize(distance, noise):
        return [simulate_snr(distance=distance, S=psd, response=detectors[ifo.name].response,
            location=detectors[ifo.name].location, measurement_error=noise, **common)
            for ifo, psd in zip(ifos, interpolated)]
    zero = realize(r['dl_source'], 'zero-noise')
    snr = math.sqrt(sum(x[1]**2 for x in zero))
    distance = r['dl_source']*snr/r['target_snr']
    np.random.seed(r['measurement_seed'])
    noisy = realize(distance, 'gaussian-noise')
    factor = np.exp(-1j*np.pi*r['morse'])
    adjusted = []
    for horizon, rho, phase, toa, series in noisy:
        series.data.data[:] *= factor
        adjusted.append((horizon, rho, float(np.angle(np.exp(1j*phase)*factor)), toa, series))
    single_tuple = namedtuple('ETPilotSingleTuple', 'detector snr phase time zerolag_time psd snr_series')
    class Single(single_tuple, SingleEvent):
        pass
    event_tuple = namedtuple('ETPilotEventTuple', 'singles template_args')
    class EventAdapter(event_tuple, Event):
        pass
    singles = [Single(ifo.name, float(x[1]), float(x[2]), x[3], x[3], p, x[4])
        for ifo, x, p in zip(ifos, adjusted, psds)]
    before = time.perf_counter()
    with geom.detector_lookup(detectors):
        sky = localize(EventAdapter(singles, args), waveform='IMRPhenomPv2', f_low=20.,
            enable_snr_series=True, f_high_truncate=1., rescale_loglikelihood=.83)
    localize_seconds = time.perf_counter()-before
    mass = np.asarray(sky['PROBDENSITY'])*moc.uniq2pixarea(sky['UNIQ'])
    if not np.isfinite(mass).all() or np.any(mass < 0) or abs(mass.sum()-1) > 1e-5:
        raise RuntimeError('Invalid MOC probability mass')
    path = root/'pilot/maps'/f'{case["id"]}.fits.gz'
    write_sky_map(path, sky, nest=True)
    write(root/'pilot/jobs'/case['id']/'RESULT.json', {**case, 'event_uid': r['event_uid'],
        'source_id': r['source_id'], 'family': r['family'], 'target_snr': r['target_snr'],
        'realized_network_snr': math.sqrt(sum(x[1]**2 for x in adjusted)),
        'map': str(path), 'sha256': sha(path), 'map_bytes': path.stat().st_size,
        'normalization': mass.sum(), 'moc_cells': len(sky), 'coordinate_frame': 'ICRS/equatorial',
        'ordering': 'NESTED', 'localize_seconds': localize_seconds, 'wall_seconds': time.perf_counter()-started,
        'peak_RSS_MiB': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        'conditional_trigger_simulation': True, 'full_strain_PE': False})


def execute(root):
    prepare(root)
    generate(root)
    cases = [{'index': i, 'q': q, 'id': f'et_{i:02d}_q{q}'} for q in [10, 32, 64] for i in range(24)]
    cases += [{'index': i, 'q': 32, 'legacy_units': True, 'id': f'et_{i:02d}_q32_legacy'} for i in [0, 8, 16, 20]]
    write(root/'contracts/MAP_CASES.json', cases)
    statuses = []
    def run(case):
        folder = root/'pilot/jobs'/case['id']
        folder.mkdir()
        write(folder/'CASE.json', case)
        try:
            with open(folder/'stdout.log', 'wb') as stdout, open(folder/'stderr.log', 'wb') as stderr:
                proc = subprocess.run([sys.executable, '-B', '-u', __file__, '--root', str(root),
                    '--case', str(folder/'CASE.json')], stdout=stdout, stderr=stderr, timeout=1200)
            result = {**case, 'exit_code': proc.returncode,
                'status': 'COMPLETE' if proc.returncode == 0 else 'FAILED'}
        except subprocess.TimeoutExpired:
            result = {**case, 'status': 'TIMEOUT'}
        write(folder/'STATUS.json', result)
        return result
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = [pool.submit(run, c) for c in cases]
        for future in as_completed(futures):
            statuses.append(future.result())
            write(root/'contracts/RUNNING_STATUS.json', {'utc': utc(), 'finished': len(statuses),
                'total': len(cases), 'failed': sum(x['status'] != 'COMPLETE' for x in statuses), 'bulk_started': False})
            print(json.dumps(statuses[-1]), flush=True)
    pd.DataFrame(statuses).to_csv(root/'tables/JOB_STATUS.csv', index=False)
    summarize(root)


def probabilities(path, nside):
    import healpy as hp
    from astropy.table import Table
    from ligo.skymap import moc
    sky = Table.read(path)
    raster = moc.rasterize(sky, order=int(round(math.log2(nside))))
    p = np.asarray(raster['PROBDENSITY'], dtype=float)*hp.nside2pixarea(nside)
    if not np.isfinite(p).all() or np.any(p < 0) or p.sum() <= 0:
        raise RuntimeError('Invalid raster')
    return p/p.sum()


def summarize(root):
    import healpy as hp
    from scipy.stats import spearmanr
    records = json.loads((root/'contracts/PILOT_EVENTS.json').read_text())
    results = [json.loads(f.read_text()) for f in sorted((root/'pilot/jobs').glob('*/RESULT.json'))]
    pd.DataFrame(results).to_csv(root/'tables/MAP_RESOURCES.csv', index=False)
    map_metrics, comparisons, pair_rows = [], [], []
    by_case = {x['id']: x for x in results}
    matrix = {}
    for q in [10, 32, 64]:
        maps = {}
        for i, row in enumerate(records):
            key = f'et_{i:02d}_q{q}'
            if key not in by_case:
                continue
            p = probabilities(by_case[key]['map'], 512)
            maps[i] = p
            ordered = np.sort(p)[::-1].cumsum()
            truth_pixel = hp.ang2pix(512, np.pi/2-row['dec'], row['ra'] % (2*np.pi), nest=True)
            map_metrics.append({'index': i, 'q': q, 'event_uid': row['event_uid'],
                'source_id': row['source_id'], 'family': row['family'],
                'A90_deg2': (np.searchsorted(ordered, .9)+1)*hp.nside2pixarea(512, degrees=True),
                'truth_HPD': p[p >= p[truth_pixel]].sum()})
            if q > 10:
                previous = 10 if q == 32 else 32
                oldkey = f'et_{i:02d}_q{previous}'
                if oldkey in by_case:
                    old = probabilities(by_case[oldkey]['map'], 512)
                    old_A = next(x['A90_deg2'] for x in map_metrics if x['index'] == i and x['q'] == previous)
                    comparisons.append({'index': i, 'from_q': previous, 'to_q': q,
                        'TV': .5*np.abs(p-old).sum(),
                        'relative_A90': abs(map_metrics[-1]['A90_deg2']/old_A-1)})
        scores = {}
        for i in maps:
            for j in maps:
                if j <= i:
                    continue
                z = float(np.log(max(len(maps[i])*np.dot(maps[i], maps[j]), 1e-300)))
                scores[(i, j)] = z
                pair_rows.append({'i': i, 'j': j, 'q': q,
                    'true_companion': records[i]['source_id'] == records[j]['source_id'], 'Z_sky': z})
        matrix[q] = scores
        del maps
    pd.DataFrame(map_metrics).to_csv(root/'tables/MAP_HPD_A90.csv', index=False)
    pd.DataFrame(comparisons).to_csv(root/'tables/QUADRATURE_MAP_CONVERGENCE.csv', index=False)
    pd.DataFrame(pair_rows).to_csv(root/'tables/PILOT_PAIR_SKY_SCORES.csv', index=False)
    resolution_rows = []
    # Rasterize one resolution at a time; never retain three dense map banks.
    for ns in [256, 512, 1024]:
        maps = {i: probabilities(by_case[f'et_{i:02d}_q64']['map'], ns)
                for i in range(24) if f'et_{i:02d}_q64' in by_case}
        for i in maps:
            for j in maps:
                if j <= i:
                    continue
                resolution_rows.append({'i': i, 'j': j, 'nside': ns,
                    'true_companion': records[i]['source_id'] == records[j]['source_id'],
                    'Z_sky': float(np.log(max(len(maps[i])*np.dot(maps[i], maps[j]), 1e-300)))})
        del maps
    pd.DataFrame(resolution_rows).to_csv(root/'tables/HEALPIX_RESOLUTION_PAIRS.csv', index=False)
    unit_map = []
    for i in [0, 8, 16, 20]:
        a, b = f'et_{i:02d}_q32', f'et_{i:02d}_q32_legacy'
        if a in by_case and b in by_case:
            unit_map.append({'index': i, 'TV': .5*np.abs(probabilities(by_case[a]['map'], 512)-
                                                        probabilities(by_case[b]['map'], 512)).sum()})
    pd.DataFrame(unit_map).to_csv(root/'tables/SPIN_UNIT_MAP_EFFECT.csv', index=False)
    convergence = []
    for qa, qb in [(10, 32), (32, 64)]:
        keys = sorted(set(matrix[qa]) & set(matrix[qb]))
        a = np.array([matrix[qa][k] for k in keys]); b = np.array([matrix[qb][k] for k in keys])
        d = np.abs(a-b)
        cm = [x for x in comparisons if x['from_q'] == qa and x['to_q'] == qb]
        convergence.append({'from_q': qa, 'to_q': qb, 'pairs': len(keys),
            'map_TV_max': max((x['TV'] for x in cm), default=None),
            'map_relative_A90_max': max((x['relative_A90'] for x in cm), default=None),
            'pair_abs_delta_median_q90_q99_max': np.quantile(d, [.5, .9, .99, 1]).tolist() if len(d) else [],
            'sign_flips': int(np.sum((a > 0) != (b > 0))),
            'spearman': float(spearmanr(a, b).statistic) if len(d) else None})
    write(root/'tables/CONVERGENCE_SUMMARY.json', convergence)
    protected = json.loads((root/'manifests/PROTECTED_INPUTS.json').read_text())
    changed = [r['path'] for r in protected if sha(r['path']) != r['sha256']]
    write(root/'manifests/PROTECTED_AFTER.json', {'checked_files': len(protected), 'changed_files': changed})
    jobs = pd.read_csv(root/'tables/JOB_STATUS.csv')
    identity = json.loads((root/'contracts/METHOD_IDENTITY_AUDIT.json').read_text())
    data = json.loads((root/'contracts/DATA_AUDIT.json').read_text())
    q64_times = [r['wall_seconds'] for r in results if r['q'] == 64]
    last = convergence[-1]
    numerical = (len(results) == 76 and last['map_TV_max'] <= .02 and last['map_relative_A90_max'] <= .05
                 and last['pair_abs_delta_median_q90_q99_max'][-1] <= .2)
    summary = {'utc': utc(), 'status': FINAL, 'experiment_complete': False,
        'bounded_preflight_complete': True, 'bulk_go': False, 'bulk_events_generated': 0,
        'pilot_events': 24, 'pilot_independent_sources': 16, 'maps_complete': len(results),
        'map_jobs_failed': int((jobs.status != 'COMPLETE').sum()), 'data_pass': data['all_pass'],
        'reference_method_identity': identity['status'], 'quadrature_screen_pass': numerical,
        'q64_wall_seconds_P50_P90_max': np.quantile(q64_times, [.5, .9, 1]).tolist() if q64_times else [],
        'reference_unit_map_effect_TV': unit_map,
        'historical_hash_changes': changed, 'test_opened': False, 'model_training_started': False,
        'why_no_bulk': ['Unresolved frozen HL sky spin mass-unit mismatch',
            'Conditional synthetic triggers are not PE of the generated noisy waveform',
            *([] if numerical else ['BAYESTAR nuisance quadrature screen did not pass'])],
        'results_not_available': ['3000-event retrieval R@K', 'new three-channel weights', 'new ET trained models'],
        'production_requires': 'resolve shared HL/ET implementation contract; then rerun numerical gates and validate actual strain-to-trigger linkage'}
    write(root/'contracts/FINAL_PREFLIGHT_RESULT.json', summary)
    report(root, summary, convergence, map_metrics)
    package(root)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def report(root, summary, convergence, map_metrics):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    df = pd.DataFrame(map_metrics)
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    for q in [10, 32, 64]:
        group = df[df.q == q]
        axs[0].plot(group['index'], group.A90_deg2, '.-', label=f'quadrature {q}')
        axs[1].plot(group['index'], group.truth_HPD, '.', label=f'quadrature {q}')
    axs[0].set(xlabel='Development event', ylabel='90% sky area (deg2)', yscale='log')
    axs[1].set(xlabel='Development event', ylabel='Truth HPD probability', ylim=(0, 1.02))
    axs[1].axhline(.9, color='black', lw=.7)
    axs[0].legend(fontsize=8)
    fig.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(root/'figures'/f'pilot_sky_numerics.{ext}', dpi=160)
    plt.close(fig)
    text = f'''# ET3-GWLMC3000 统一方案开发预检报告

状态：{FINAL}

## 结论

本次完成的是24事件开发预检，不是3000事件完整实验。没有启动批量天空图或模型训练，没有打开测试波形或分数。历史结果未覆盖，输入哈希变化数：{len(summary['historical_hash_changes'])}。

物理应变检查通过：{summary['data_pass']}。完成天空图：{summary['maps_complete']}/76。32到64阶内部积分检查通过：{summary['quadrature_screen_pass']}。

## 冻结基线发现的问题

1. 冻结O3/O4a天空脚本将太阳质量数值传入要求SI千克的自旋转换接口。新表SPIN_MASS_UNIT_AUDIT.csv独立复算了倾角和自旋变化；4事件地图对照见SPIN_UNIT_MAP_EFFECT.csv。不能只修ET后声称和未修正的HL版本完全一致，也不能为了一致复制已确认的单位错误。
2. 冻结脚本通过bayestar-realize-coincs模拟高斯SNR/相位/时间序列，输入使用注入真值质量、自旋和方向。该流程属于条件触发模拟，不是对保存的含噪strain做匹配滤波后得到的天空PE。本次实际生成了独立物理strain，但没有把这两条并行数据链冒充同一噪声实现的端到端PE。
3. 10/32/64是倾角和偏振的内部积分点数，不是Nside。所有地图数值比较先固定Nside=512；另单独给出256/512/1024的HEALPix审计。换Nside不能自动修复内部积分误差。

## 本次实际做了什么

- 从GW-LMC同一来源CSV按全局event_id去重，预留1800个独立系统，按每类420/90/90划分train/validation/test，共3000个计划事件。只生成训练集合中的24个开发事件：8个透镜系统的16像和8个非透镜事件。未生成验证和测试波形。
- 两类人口分别为smooth/non-subhalo和subhalo-present，不是解析SIS/PM。
- 使用原物理生成器IMRPhenomXPHM和Bilby ET三个响应，生成量纲正确的24秒4096Hz strain与独立设计PSD高斯噪声。噪声不是未来ET实测噪声；到达日历为受控十年满占空假设。
- 保留短窗2秒4096点及长窗16秒20–80Hz4096点。仅扩展探测器轴，前两通道的预处理与冻结HL函数逐点一致；未改频段、抗混叠或归一化运算。
- 条件BAYESTAR数值诊断使用SI质量、20Hz参考频率、原IMRPhenomPv2、0.83 likelihood scale、固定temperature=1，24事件分别计算10/32/64积分，共72图。另4图仅用于旧单位对照。没有旋转公开天空模板，没有永久保存dense地图。
- NEW-SCORE-ONLY定义保持alpha=1、保留短窗及内部质量项，禁止恢复外层0.875混合。尚未训练，不能报告其新Recall。

## 数值结果

```json
{json.dumps(convergence, indent=2, ensure_ascii=False)}
```

24事件来自16个源且刻意分层，不是24个独立总体覆盖率试验。8对伴随像相关，不能逐像bootstrap后宣称精确coverage校准。这里仅报告逐事件HPD，不选temperature，不根据天空真值提高排名。

## 资源与后续边界

q64每图实测wall time P50/P90/max秒：{summary['q64_wall_seconds_P50_P90_max']}。不能将P50乘900当作保证工期；失败、复杂源和资源竞争必须另计。新原生MOC和少量物理strain保留，dense数组仅临时计算。

必须先解决HL/ET共同接口的单位和strain到trigger契约，再以同一科学实现通过ET内部积分验证。当前未满足，因此按“先开发验证、通过后批量”的要求停止扩展，而不是把失败掩盖后生成3000事件。原O3/O4a结果的改变量尚未重跑，不猜测方向或幅度。

## 文件

- contracts/ANALYSIS_CONTRACT.json：开图前冻结的方案和容差。
- contracts/FINAL_PREFLIGHT_RESULT.json：最终go/no-go。
- tables/PHYSICAL_STRAIN_PILOT.csv：应变、SNR和预处理。
- tables/MAP_HPD_A90.csv、QUADRATURE_MAP_CONVERGENCE.csv：地图数值诊断。
- tables/PILOT_PAIR_SKY_SCORES.csv、HEALPIX_RESOLUTION_PAIRS.csv：全部开发pair分数。
- tables/SPIN_MASS_UNIT_AUDIT.csv、SPIN_UNIT_MAP_EFFECT.csv：冻结参考的接口错误。
- manifests/PROTECTED_INPUTS.json、PROTECTED_AFTER.json：历史输入哈希前后检查。
- scripts/：本次脚本和调用的参考源代码；依赖仍需原环境及manifest输入。

## 科学依据

- [Bilby自旋转换接口](https://bilby-dev.github.io/bilby/api/bilby.gw.conversion.bilby_to_lalsimulation_spins.html)：mass_1/mass_2为SI单位。
- [官方BAYESTAR触发模拟说明](https://lscsoft.docs.ligo.org/ligo.skymap/tool/bayestar_realize_coincs.html)：生成合成触发，不等于从指定strain恢复触发。
- [Singer与Price BAYESTAR论文](https://arxiv.org/abs/1508.03634)：快速条件天空定位的来源；并未验证本项目的ET适配或本次工程容差。
'''
    (root/'reports/ET3_GWLMC3000_PREFLIGHT_CN.md').write_text(text)


def package(root):
    entries = []
    for f in sorted(root.rglob('*')):
        if f.is_file() and 'package' not in f.relative_to(root).parts and f.name != 'OUTPUT_SHA256.csv':
            entries.append({'path': str(f.relative_to(root)), 'sha256': sha(f), 'bytes': f.stat().st_size})
    pd.DataFrame(entries).to_csv(root/'manifests/OUTPUT_SHA256.csv', index=False)
    archive = root/'package'/f'{root.name}_preflight_deliverables.tar.gz'
    with tarfile.open(archive, 'w:gz') as tar:
        for r in entries:
            if r['path'].startswith('pilot/strain/'):
                continue
            tar.add(root/r['path'], arcname=f'{root.name}/{r["path"]}')
        tar.add(root/'manifests/OUTPUT_SHA256.csv', arcname=f'{root.name}/manifests/OUTPUT_SHA256.csv')
    digest = sha(archive)
    archive.with_suffix(archive.suffix+'.sha256').write_text(f'{digest}  {archive.name}\n')
    write(root/'package/PACKAGE_CHECK.json', {'sha256': digest, 'bytes': archive.stat().st_size,
        'raw_pilot_strain_in_package': False, 'raw_pilot_strain_retained_remote': True,
        'contains_maps': True, 'no_credentials_included': True})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--case', type=Path)
    parser.add_argument('--summarize', action='store_true')
    args = parser.parse_args()
    if args.case:
        sky_worker(args.root, json.loads(args.case.read_text()))
    elif args.summarize:
        summarize(args.root)
    else:
        try:
            execute(args.root)
        except Exception:
            write(args.root/'contracts/FATAL_ERROR.json', {'utc': utc(), 'traceback': traceback.format_exc(),
                'status': FINAL, 'bulk_started': False})
            raise
