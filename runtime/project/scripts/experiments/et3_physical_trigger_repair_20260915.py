#!/usr/bin/env python3
"""Immutable ET-only development repair; no production or historical writes."""
import os
for _name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['OMP_STACKSIZE'] = '512M'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import argparse
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import logging
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import sysconfig
import tarfile
import time
import traceback
import numpy as np
import pandas as pd

P = Path('/root/autodl-tmp/gw-catalog')
OLD = P/'results/et3_gwlmc3000_unified_nso_20260914T051500Z_r2'
BUILD = P/'results/et3_new_score_only_bayestar_20260912T014101Z/diagnostics/quadrature_build'
FS, N, DF = 4096, 24*4096, 1/24
LOW, HIGH = 20., 1024.
FINAL = 'HOLD_FOR_AUTHOR_REVIEW_NO_ADOPTION_NO_OVERWRITE'


def utc():
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False,
        default=lambda v: v.item() if isinstance(v, np.generic) else str(v))+'\n')
    temp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    sys.modules[name] = obj
    spec.loader.exec_module(obj)
    return obj


def guard(root):
    if shutil.disk_usage(root).free < 30*2**30:
        raise RuntimeError('HOLD_DISK_LIMIT')


def deps(root):
    sys.path.insert(0, str(P))
    old = module(root/'scripts/et3_gwlmc_3000_preflight_20260914.py', 'repair_previous')
    source, geometry, physical = old.dependencies(root)
    return old, source, geometry, physical


def prepare(root):
    root.mkdir(parents=True, exist_ok=False)
    for part in ('contracts', 'scripts', 'strain', 'triggers', 'maps', 'tables', 'logs', 'reports', 'build', 'manifest', 'package'):
        (root/part).mkdir()
    shutil.copy2(__file__, root/'scripts'/Path(__file__).name)
    names = ('et3_gwlmc_3000_preflight_20260914.py', '34_generate_physical_h1l1_source_bank.py',
             'et3_new_score_only_20260912.py', 'bayestar_injection_sky_full_experiment.py')
    for name in names:
        shutil.copy2(OLD/'scripts'/name, root/'scripts'/name)
    records = json.loads((OLD/'contracts/PILOT_EVENTS.json').read_text())
    for row in records:
        row['source_uid'] = 'GW-LMC:'+str(row['source_id'])
        row['old_target_snr_not_enforced'] = row.pop('target_snr')
    write(root/'contracts/EVENTS.json', records)
    shutil.copy2(OLD/'contracts/RESERVED_SYSTEM_SPLITS.json', root/'contracts/RESERVED_SYSTEM_SPLITS.json')
    protected = []
    for path in sorted(OLD.rglob('*')):
        if path.is_file() and 'package' not in path.parts:
            protected.append({'path': str(path), 'sha256': sha(path), 'bytes': path.stat().st_size})
    # Protect the earlier provenance boundary too, without modifying its files.
    previous = json.loads((OLD/'manifests/PROTECTED_INPUTS.json').read_text())
    write(root/'manifest/HISTORICAL_MANIFEST_AS_RECEIVED.json', previous)
    write(root/'manifest/PROTECTED_BEFORE.json', protected)
    contract = {
        'id': root.name, 'created_utc': utc(), 'scope': 'ET-only known development events; no locked test or real candidates',
        'previous_root': str(OLD), 'development_event_count': len(records), 'independent_sources': 16,
        'bulk_event_target_not_started': 3000, 'no_encoder_training_in_diagnostic_stage': True,
        'historical_HL_not_corrected_here': True,
        'source_population': 'GW-LMC 2.5PLUS BBH Any_Detected_SNR1; selection-limited, not an unbiased ET population',
        'lensing': 'tabulated signed img_mags, geometric sqrt(abs(mu)) amplitude and Morse phase',
        'system_amplitude': 'one common factor C=old faint-SNR anchor / min(unscaled image network optimal SNR)',
        'snr_anchors': [9, 11, 16, 24], 'per_image_target_ratio': False,
        'high_SNR_policy': 'retain all development events; no independent cap, deletion or rescaling; report >60 tail',
        'noise': 'independent ET design Gaussian channel noise; regenerate using frozen per-event noise seeds',
        'forward_waveform': 'IMRPhenomXPHM, 4096Hz, 24s, reference/minimum20Hz; Bilby detector response at image GPS',
        'signal_saved_for_waveform_and_sky_identical': True,
        'short_window': {'seconds': [-1.75, .25], 'sample_rate': 2048, 'samples': 4096, 'band': [40, 580]},
        'long_window': {'seconds': [-15.75, .25], 'sample_rate': 256, 'samples': 4096, 'band': [20, 80]},
        'template_control': 'IMRPhenomPv2 with injected intrinsic parameters ONLY as an oracle development control',
        'not_blind_PE': True, 'production_requires_data_recovered_intrinsics': True,
        'spin_units': 'Bilby spin-conversion masses in SI kg; waveform-generator masses in solar masses',
        'trigger_source': 'PyCBC complex matched_filter on saved physical noisy strain; not simulate_snr',
        'trigger_search_seconds': [-.125, .125], 'trigger_series_half_width_seconds': .125,
        'PSD_horizon_and_autocorrelation': 'same actual filter template power and same ET PSD as matched filtering',
        'geometry': 'Bilby ET tensors and vertices explicitly passed to localization lookup',
        'analysis_nside': 512, 'audit_nsides': [256, 512, 1024], 'permanent_dense_maps': False,
        'quadratures': [32, 64, 128], 'q128_only_if_q32_q64_incomplete_or_not_converged': True,
        'threads_per_map': 4, 'map_workers': 6, 'timeout_seconds_by_q': {'32': 1200, '64': 1800, '128': 3600},
        'likelihood_rescale': .83, 'temperature': 1.,
        'gates': {'matched_filter_relative': 1e-9, 'self_injection_relative': 1e-8,
            'linearity_relative': 1e-9, 'horizon_relative': 2e-3, 'geometry_tensor_absolute': 1e-7,
            'common_scale_relative': 1e-12, 'normalization_absolute': 1e-5,
            'map_TV_max': .02, 'A90_relative_max': .05, 'pair_delta_max_nats': .2,
            'pixel_delta_q99_nats': .05, 'pixel_delta_max_nats': .2,
            'P90_wall_seconds': 900, 'all_maps_complete': True},
        'coverage': 'descriptive source-block coverage; 16 sources insufficient for calibrated population claims',
        'expansion_requires': ['physical and trigger checks', 'numerical convergence and runtime',
            'data-derived template selection', 'independent localization coverage validation'],
        'no_template_bank_no_production': True,
        'final_state': FINAL,
        'references': {
            'GW_LMC': 'https://github.com/LensedGW/GW-LMC',
            'BAYESTAR': 'https://arxiv.org/abs/1508.03634',
            'PyCBC': 'https://pycbc.org/pycbc/latest/html/pycbc.filter.html',
            'SI_spin_interface': 'https://bilby-dev.github.io/bilby/api/bilby.gw.conversion.bilby_to_lalsimulation_spins.html'}
    }
    write(root/'contracts/ANALYSIS_CONTRACT.json', contract)
    write(root/'contracts/CONTRACT_HASH.json', {'sha256': sha(root/'contracts/ANALYSIS_CONTRACT.json')})
    write(root/'STATUS.json', {'state': 'CONTRACT_FROZEN', 'utc': utc()})


def generate(root):
    import bilby
    from scipy.signal import resample_poly
    logging.getLogger('bilby').setLevel(logging.ERROR)
    old, src, geom, phys = deps(root)
    rows = json.loads((root/'contracts/EVENTS.json').read_text())
    generator = src.build_waveform_generator()
    frequency = np.arange(8193)*.25
    base = []
    geometry_records = geom.geometry()[2]
    write(root/'contracts/GEOMETRY.json', geometry_records)
    if max(x['adapter_tensor_error'] for x in geometry_records) > 1e-7:
        raise RuntimeError('Forward/inverse geometry mismatch')
    for row in rows:
        guard(root)
        ifos, _, _ = geom.geometry()
        parameters = src.source_parameters(pd.Series(row), row['gps'])
        clean, _ = src.detector_response(generator, ifos, parameters, src.lens_factor(row['magnification'], row['morse']))
        psds = np.array([x.power_spectral_density.get_power_spectral_density_array(frequency) for x in ifos])
        psds = np.where((frequency[None, :] >= LOW)&np.isfinite(psds)&(psds > 0), psds, np.inf)
        _, _, norm = phys.scale_to_network_snr(clean, 1., frequency, psds)
        # scale_to_network_snr returns achieved SNR; measure the unscaled norm
        # from the returned scale instead of mistaking achieved SNR for raw SNR.
        _, to_one, _ = phys.scale_to_network_snr(clean, 1., frequency, psds)
        raw_snr = 1./to_one
        base.append((row, clean, psds, raw_snr))
    factors = {}
    for sid in sorted({x['source_id'] for x in rows}):
        group = [x for x in base if x[0]['source_id'] == sid]
        anchor = min(x[0]['old_target_snr_not_enforced'] for x in group)
        factors[sid] = anchor/min(x[3] for x in group)
    done = []
    for row, clean, psds, raw_snr in base:
        guard(root)
        begin = time.perf_counter()
        factor = factors[row['source_id']]
        scaled = clean*factor
        np.random.seed(row['noise_seed'])
        bilby.core.utils.random.seed(row['noise_seed'])
        noise_ifos, _, _ = geom.geometry()
        noise = []
        for ifo in noise_ifos:
            ifo.minimum_frequency = LOW
            ifo.set_strain_data_from_power_spectral_density(FS, 26., start_time=row['gps']-24.75)
            noise.append(np.asarray(ifo.strain_data.time_domain_strain))
        padded = np.asarray(noise)
        if padded.shape != (3, 26*FS):
            raise RuntimeError('Noise cache shape mismatch')
        padded[:, FS:FS+N] += scaled
        # Float64 preserves physical strain and exact shared waveform/sky input.
        full = old.preprocess_generic(phys, padded, frequency, psds)
        short = full[:, -4096:]
        long = resample_poly(old.preprocess_generic(phys, padded, frequency, psds, 20., 80.),
            1, 8, axis=-1, window=('kaiser', 8.6))[:, -4096:]
        parity = np.array_equal(full[:2], phys.preprocess_24s(padded[:2], frequency, psds[:2]))
        if short.shape != (3, 4096) or long.shape != (3, 4096) or not parity or not np.isfinite(padded).all():
            raise RuntimeError('Preprocessing gate failed')
        path = root/'strain'/f'{row["event_uid"]}.npz'
        np.savez_compressed(path, clean=scaled, noisy_padded=padded, short=short, long=long,
            psd_frequency=frequency, psd=psds)
        done.append({'event_uid': row['event_uid'], 'source_id': row['source_id'], 'family': row['family'],
            'amplitude_scale': factor, 'unscaled_network_optimal_snr': raw_snr,
            'network_optimal_snr': factor*raw_snr, 'above_historical_60': factor*raw_snr > 60,
            'old_per_image_target': row['old_target_snr_not_enforced'], 'noise_seed': row['noise_seed'],
            'HL_preprocessing_exact': parity, 'path': str(path), 'sha256': sha(path),
            'seconds': time.perf_counter()-begin, 'bytes': path.stat().st_size})
        print(json.dumps({'stage': 'physical_strain', 'complete': len(done), 'snr': factor*raw_snr}), flush=True)
    pd.DataFrame(done).to_csv(root/'tables/PHYSICAL_STRAIN.csv', index=False)
    violations = []
    for sid, group in pd.DataFrame(done).groupby('source_id'):
        if group.amplitude_scale.nunique() != 1:
            violations.append(int(sid))
    write(root/'contracts/PHYSICAL_GATE.json', {'pass': not violations, 'scale_violations': violations,
        'same_physical_noisy_array_for_both_branches': True, 'events': len(done),
        'high_SNR_events_retained': sum(x['above_historical_60'] for x in done)})


def extract_one(root, idx):
    import bilby
    import lal
    from pycbc.waveform import get_fd_waveform
    from pycbc.types import FrequencySeries
    from pycbc.filter import matched_filter, sigma
    from ligo.skymap.bayestar import filter as sky_filter
    _, _, geom, _ = deps(root)
    row = json.loads((root/'contracts/EVENTS.json').read_text())[idx]
    dest = root/'triggers'/f'event{idx:02d}'
    dest.mkdir(exist_ok=False)
    begin = time.perf_counter()
    f = np.arange(N//2+1)*DF
    keep = (f >= LOW)&(f < HIGH)
    vals = [row[k] for k in ('theta_jn', 'phijl', 'tilt1', 'tilt2', 'phi12', 'a1', 'a2')]
    spins = bilby.gw.conversion.bilby_to_lalsimulation_spins(*vals,
        row['m1_det']*lal.MSUN_SI, row['m2_det']*lal.MSUN_SI, 20., row['phase'])
    args = {'mass1': row['m1_det'], 'mass2': row['m2_det'], 'f_ref': 20., 'f_final': HIGH}
    args.update(dict(zip(('spin1x', 'spin1y', 'spin1z', 'spin2x', 'spin2y', 'spin2z'), spins[1:])))
    hp, hc = get_fd_waveform(approximant='IMRPhenomPv2', delta_f=DF, f_lower=LOW,
        distance=1., inclination=0., coa_phase=0., **args)
    hp.resize(len(f)); hc.resize(len(f))
    template = .5*(np.asarray(hp)+1j*np.asarray(hc))
    template[~keep] = 0
    ts = FrequencySeries(template, delta_f=DF)
    data_path = root/'strain'/f'{row["event_uid"]}.npz'
    with np.load(data_path) as a:
        strain = a['noisy_padded'][:, FS:FS+N]
        clean = a['clean']
    if strain.shape != (3, N) or not np.isfinite(strain).all():
        raise RuntimeError('Invalid physical strain')
    start = lal.LIGOTimeGPS(row['gps'])-23.75
    lo, hi = int(23.75*FS)-512, int(23.75*FS)+513
    half = 512
    ifos, _, _ = geom.geometry()
    snippets, epochs, psds, horizons, checks = [], [], [], [], []
    for channel, ifo in enumerate(ifos):
        psd = ifo.power_spectral_density.get_power_spectral_density_array(f)
        psd = np.where(np.isfinite(psd)&(psd > 0), psd, np.inf)
        p = FrequencySeries(psd, delta_f=DF)
        fd = np.fft.rfft(strain[channel])/FS
        clean_fd = np.fft.rfft(clean[channel])/FS
        def compute(v):
            return np.asarray(matched_filter(ts, FrequencySeries(v, delta_f=DF), psd=p,
                low_frequency_cutoff=LOW, high_frequency_cutoff=HIGH))
        snr, clean_snr, noise_snr = compute(fd), compute(clean_fd), compute(fd-clean_fd)
        horizon = float(sigma(ts, psd=p, low_frequency_cutoff=LOW, high_frequency_cutoff=HIGH))
        peak = lo+int(np.argmax(abs(snr[lo:hi])))
        cp = lo+int(np.argmax(abs(clean_snr[lo:hi])))
        direct = 4*DF/horizon*np.sum(fd[keep]*template[keep].conj()/psd[keep]*np.exp(2j*np.pi*f[keep]*peak/FS))
        self_snr = compute(13./horizon*template*np.exp(.37j-2j*np.pi*f*789/FS))
        optimal = float(sigma(FrequencySeries(clean_fd, delta_f=DF), psd=p,
            low_frequency_cutoff=LOW, high_frequency_cutoff=HIGH))
        power = lal.CreateREAL8FrequencySeries('actual template power', 0, 0, DF, lal.StrainUnit**2, len(f)-1)
        power.data.data[:] = abs(template[:-1])**2
        interp = sky_filter.InterpolatedPSD(f, psd, f_high_truncate=1.)
        model = sky_filter.SignalModel(sky_filter.signal_psd_series(power, interp))
        checks.append({'event_uid': row['event_uid'], 'detector': ifo.name,
            'matched_snr': float(abs(snr[peak])), 'phase': float(np.angle(snr[peak])),
            'peak_sample': peak, 'arrival_gps': float(start+peak/FS),
            'clean_peak_sample': cp, 'clean_matched_snr': float(abs(clean_snr[cp])),
            'clean_optimal_snr': optimal, 'fitting_factor': float(abs(clean_snr[cp])/optimal),
            'matched_filter_relative': float(abs(direct-snr[peak])/max(abs(snr[peak]), 1.)),
            'self_injection_relative': float(abs(self_snr[789]-13*np.exp(.37j))/13),
            'linearity_relative': float(np.max(abs(snr-clean_snr-noise_snr))/max(np.max(abs(snr)), 1.)),
            'horizon_relative': float(abs(model.get_horizon_distance()/horizon-1)),
            'template_horizon_Mpc': horizon, 'noise_real_variance': float(np.var(noise_snr.real)),
            'noise_imag_variance': float(np.var(noise_snr.imag))})
        if peak-half < 0 or peak+half >= N:
            raise RuntimeError('SNR snippet outside data')
        snippets.append(snr[peak-half:peak+half+1].astype(np.complex64))
        epochs.append(int((start+(peak-half)/FS).ns()))
        psds.append(psd); horizons.append(horizon)
    np.savez_compressed(dest/'TRIGGER.npz', snr_series=snippets, epochs_ns=np.asarray(epochs, np.int64),
        psds=psds, template_power=abs(template)**2, horizons=horizons)
    write(dest/'TRIGGER.json', {'event_uid': row['event_uid'], 'source_uid': row['source_uid'],
        'template_args': args, 'template': 'IMRPhenomPv2', 'channels': checks,
        'source_strain': str(data_path), 'source_strain_sha256': sha(data_path),
        'trigger_sha256': sha(dest/'TRIGGER.npz'), 'intrinsics_from_injected_truth_diagnostic_only': True,
        'sky_not_passed_to_likelihood': True, 'trigger_from_actual_saved_strain': True,
        'seconds': time.perf_counter()-begin})
    print(json.dumps({'stage': 'strain_trigger', 'event': row['event_uid'],
        'network_matched_snr': float(np.linalg.norm([c['matched_snr'] for c in checks])),
        'seconds': time.perf_counter()-begin}), flush=True)


def extract(root):
    if not json.loads((root/'contracts/PHYSICAL_GATE.json').read_text())['pass']:
        raise RuntimeError('Physical gate failed')
    events = json.loads((root/'contracts/EVENTS.json').read_text())
    for idx in range(len(events)):
        extract_one(root, idx)
    checks = [c for path in sorted((root/'triggers').glob('*/TRIGGER.json'))
              for c in json.loads(path.read_text())['channels']]
    limits = json.loads((root/'contracts/ANALYSIS_CONTRACT.json').read_text())['gates']
    failed = [{'event': c['event_uid'], 'detector': c['detector'], 'key': k, 'value': c[k], 'limit': limits[k]}
        for c in checks for k in ('matched_filter_relative', 'self_injection_relative', 'linearity_relative', 'horizon_relative')
        if not np.isfinite(c[k]) or c[k] > limits[k]]
    pd.DataFrame(checks).to_csv(root/'tables/STRAIN_TRIGGER_AUDIT.csv', index=False)
    write(root/'contracts/TRIGGER_GATE.json', {'pass': not failed, 'failed': failed, 'channels': len(checks),
        'tests': len(checks)*4, 'does_not_establish_BAYESTAR_coverage': True})
    if failed:
        raise RuntimeError('Trigger unit/normalization gate failed')


def load_core(path):
    import ligo.skymap
    if 'ligo.skymap.core' in sys.modules:
        raise RuntimeError('Core already imported')
    core = module(path, 'ligo.skymap.core')
    ligo.skymap.core = core
    core.set_num_threads(4)
    return core


def build128(root):
    source = BUILD/'ligo_skymap-2.5.4'
    lib = BUILD/'sysroot/usr/lib/x86_64-linux-gnu'
    dest = root/'build/q128'
    dest.mkdir(exist_ok=False)
    files = ['bayestar_distance.c', 'bayestar_moc.c', 'bayestar_sky_map.c', 'core.c',
             'cubic_interp.c', 'cubic_interp_test.c', 'find_floor.c']
    args = ['gcc', '-O3', '-shared', '-fPIC', '-fopenmp', '-std=gnu11', '-fvisibility=hidden',
        '-DGSL_RANGE_CHECK_OFF', '-DHAVE_INLINE', '-DPy_LIMITED_API=0x030B0000',
        '-DNPY_TARGET_VERSION=NPY_2_0_API_VERSION', '-DNPY_NO_DEPRECATED_API=NPY_2_0_API_VERSION',
        '-DET_BAYESTAR_NU=128', '-DET_BAYESTAR_NPSI=128', '-I'+np.get_include(),
        '-I'+sysconfig.get_path('include'), '-I'+str(BUILD/'sysroot/usr/include'),
        '-I'+str(source/'cextern/chealpix'), *[str(source/'src'/x) for x in files],
        str(source/'cextern/chealpix/chealpix.c'), '-L'+str(lib), '-Wl,-rpath,'+str(lib),
        '-lgsl', '-lgslcblas', '-lm', '-o', str(dest/'core.abi3.so')]
    write(dest/'BUILD_COMMAND.json', args)
    with open(dest/'stdout.log', 'wb') as out, open(dest/'stderr.log', 'wb') as err:
        subprocess.run(args, stdout=out, stderr=err, check=True, timeout=300)
    with open(dest/'tests.stdout.log', 'wb') as out, open(dest/'tests.stderr.log', 'wb') as err:
        subprocess.run([sys.executable, '-B', __file__, '--root', str(root), '--core-test', str(dest/'core.abi3.so')],
            stdout=out, stderr=err, check=True, timeout=300)
    write(dest/'COMPLETE.json', {'sha256': sha(dest/'core.abi3.so'), 'upstream_tests': 'PASS',
        'source_sha256': sha(source/'src/bayestar_sky_map.c'), 'installed_library_changed': False})


def worker(root, case):
    soft, hard = resource.getrlimit(resource.RLIMIT_STACK)
    resource.setrlimit(resource.RLIMIT_STACK, (min(512*2**20, hard) if hard > 0 else 512*2**20, hard))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    core_path = root/'build/q128/core.abi3.so' if case['q'] == 128 else BUILD/f'q{case["q"]}/core.abi3.so'
    core = load_core(core_path)
    import lal
    import healpy as hp
    import ligo.skymap.bayestar as bayestar
    from ligo.skymap.io.events.base import Event, SingleEvent
    from ligo.skymap.io.fits import write_sky_map
    from ligo.skymap import moc
    _, _, geom, _ = deps(root)
    folder = root/'maps'/case['id']
    tr = root/'triggers'/f'event{case["idx"]:02d}'
    record = json.loads((tr/'TRIGGER.json').read_text())
    if sha(tr/'TRIGGER.npz') != record['trigger_sha256']:
        raise RuntimeError('Frozen trigger changed')
    with np.load(tr/'TRIGGER.npz') as archive:
        cache = {k: archive[k] for k in archive.files}
    ifos, detectors, _ = geom.geometry()
    S = namedtuple('ETSingle', 'detector snr phase time zerolag_time psd snr_series')
    class Single(S, SingleEvent):
        pass
    E = namedtuple('ETEvent', 'singles template_args')
    class ETEvent(E, Event):
        pass
    singles = []
    for i, ifo in enumerate(ifos):
        values = cache['snr_series'][i]
        epoch = lal.LIGOTimeGPS(0, int(cache['epochs_ns'][i]))
        series = lal.CreateCOMPLEX8TimeSeries('physical strain recovered SNR', epoch, 0, 1/FS,
            lal.DimensionlessUnit, len(values))
        series.data.data[:] = values
        psd = lal.CreateREAL8FrequencySeries('ET design PSD', 0, 0, DF, lal.StrainUnit**2, len(cache['psds'][i]))
        psd.data.data[:] = cache['psds'][i]
        mid = len(values)//2
        arrival = epoch+mid/FS
        singles.append(Single(ifo.name, float(abs(values[mid])), float(np.angle(values[mid])), arrival, arrival, psd, series))
    event = ETEvent(singles, record['template_args'])
    h = lal.CreateREAL8FrequencySeries('actual filter template power', 0, 0, DF, lal.StrainUnit**2, len(cache['template_power'])-1)
    h.data.data[:] = cache['template_power'][:-1]
    original = bayestar.filter.sngl_inspiral_psd
    bayestar.filter.sngl_inspiral_psd = lambda *a, **k: h
    begin = time.perf_counter()
    try:
        with geom.detector_lookup(detectors):
            condition = bayestar.condition(event, waveform=record['template'], f_low=LOW,
                enable_snr_series=True, f_high_truncate=1.)
            write(folder/'CONDITIONING.json', {'samples': int(condition[3].shape[1]),
                'sample_rate': condition[1], 'core_sha256': sha(core_path), 'threads': core.get_num_threads()})
            sky = bayestar.localize(event, waveform=record['template'], f_low=LOW,
                enable_snr_series=True, f_high_truncate=1., rescale_loglikelihood=.83)
    finally:
        bayestar.filter.sngl_inspiral_psd = original
    wall = time.perf_counter()-begin
    path = folder/'sky.fits.gz'
    write_sky_map(path, sky, nest=True)
    norm = float(np.sum(np.asarray(sky['PROBDENSITY'])*moc.uniq2pixarea(sky['UNIQ'])))
    row = json.loads((root/'contracts/EVENTS.json').read_text())[case['idx']]
    audits = []
    for nside in (256, 512, 1024):
        started = time.perf_counter()
        mass = np.asarray(moc.rasterize(sky, order=int(np.log2(nside)))['PROBDENSITY'], float)*hp.nside2pixarea(nside)
        mass /= mass.sum()
        truth = hp.ang2pix(nside, np.pi/2-row['dec'], row['ra']%(2*np.pi), nest=True)
        csum = np.cumsum(np.sort(mass)[::-1])
        audits.append({'nside': nside, 'A90_deg2': float((np.searchsorted(csum, .9)+1)*hp.nside2pixarea(nside, degrees=True)),
            'truth_HPD_level': float(mass[mass >= mass[truth]].sum()), 'seconds': time.perf_counter()-started})
    usage = resource.getrusage(resource.RUSAGE_SELF)
    write(folder/'RESULT.json', {**case, 'event_uid': row['event_uid'], 'source_uid': row['source_uid'],
        'map_path': str(path), 'map_sha256': sha(path), 'map_bytes': path.stat().st_size,
        'normalization': norm, 'native_MOC_cells': len(sky), 'wall_seconds': wall,
        'CPU_seconds': usage.ru_utime+usage.ru_stime, 'peak_RSS_KiB': usage.ru_maxrss,
        'ordering': 'NESTED', 'coordinates': 'ICRS', 'raster_audit': audits,
        'actual_saved_noisy_strain': True, 'conditional_intrinsics_oracle_control': True,
        'formal_PE_validated': False})


def maps(root, q):
    if not json.loads((root/'contracts/TRIGGER_GATE.json').read_text())['pass']:
        raise RuntimeError('Trigger gate did not pass')
    c = json.loads((root/'contracts/ANALYSIS_CONTRACT.json').read_text())
    cases = [{'idx': i, 'q': q, 'id': f'event{i:02d}_q{q}'} for i in range(c['development_event_count'])]
    def run(case):
        guard(root)
        folder = root/'maps'/case['id']
        folder.mkdir(exist_ok=False)
        write(folder/'CASE.json', case)
        begin = time.perf_counter()
        with open(folder/'stdout.log', 'wb') as out, open(folder/'stderr.log', 'wb') as err:
            try:
                result = subprocess.run([sys.executable, '-B', '-u', __file__, '--root', str(root),
                    '--case', str(folder/'CASE.json')], stdout=out, stderr=err,
                    timeout=c['timeout_seconds_by_q'][str(q)], check=False)
                status = {**case, 'state': 'COMPLETE' if result.returncode == 0 else 'FAILED', 'exit_code': result.returncode}
            except subprocess.TimeoutExpired:
                status = {**case, 'state': 'TIMEOUT'}
        status['wall_with_startup_seconds'] = time.perf_counter()-begin
        write(folder/'STATUS.json', status)
        return status
    finished = []
    with ThreadPoolExecutor(max_workers=c['map_workers']) as pool:
        jobs = [pool.submit(run, case) for case in cases]
        for job in as_completed(jobs):
            finished.append(job.result())
            write(root/'STATUS.json', {'state': 'NUMERICAL_PILOT_RUNNING', 'q': q,
                'finished': len(finished), 'total': len(cases), 'failed': sum(x['state'] != 'COMPLETE' for x in finished), 'utc': utc()})
            print(json.dumps(finished[-1]), flush=True)
    pd.DataFrame(finished).to_csv(root/'tables'/f'JOBS_Q{q}.csv', index=False)


def raster(path, nside):
    import healpy as hp
    from astropy.table import Table
    from ligo.skymap import moc
    sky = Table.read(path)
    mass = np.asarray(moc.rasterize(sky, order=int(np.log2(nside)))['PROBDENSITY'], float)*hp.nside2pixarea(nside)
    mass /= mass.sum()
    return mass


def audit(root, qa, qb):
    from scipy.stats import spearmanr
    import healpy as hp
    rows = json.loads((root/'contracts/EVENTS.json').read_text())
    map_rows, mats, indices = [], {qa: [], qb: []}, []
    for i, row in enumerate(rows):
        paths = [root/'maps'/f'event{i:02d}_q{q}'/'RESULT.json' for q in (qa, qb)]
        if not all(p.exists() for p in paths):
            continue
        a, b = [json.loads(p.read_text()) for p in paths]
        pa, pb = [raster(x['map_path'], 512) for x in (a, b)]
        area_a, area_b = [next(z['A90_deg2'] for z in x['raster_audit'] if z['nside'] == 512) for x in (a, b)]
        map_rows.append({'index': i, 'event_uid': row['event_uid'], 'source_id': row['source_id'],
            'q_from': qa, 'q_to': qb, 'TV': float(.5*np.abs(pa-pb).sum()),
            'A90_from': area_a, 'A90_to': area_b, 'A90_relative': abs(area_b-area_a)/area_b,
            'wall_seconds': b['wall_seconds']})
        mats[qa].append(pa); mats[qb].append(pb); indices.append(i)
    pair_rows = []
    for ia, i in enumerate(indices):
        for ja in range(ia+1, len(indices)):
            j = indices[ja]
            z = [float(np.log(max(len(mats[q][ia])*np.dot(mats[q][ia], mats[q][ja]), np.finfo(float).tiny))) for q in (qa, qb)]
            pair_rows.append({'i': i, 'j': j, 'source_i': rows[i]['source_id'], 'source_j': rows[j]['source_id'],
                'companion': rows[i]['source_id'] == rows[j]['source_id'],
                'Z_from': z[0], 'Z_to': z[1], 'abs_delta': abs(z[1]-z[0]), 'sign_flip': (z[0] > 0) != (z[1] > 0)})
    frame, pairs = pd.DataFrame(map_rows), pd.DataFrame(pair_rows)
    suffix = f'Q{qa}_Q{qb}'
    frame.to_csv(root/'tables'/f'MAP_CONVERGENCE_{suffix}.csv', index=False)
    pairs.to_csv(root/'tables'/f'PAIR_CONVERGENCE_{suffix}.csv', index=False)
    gates = json.loads((root/'contracts/ANALYSIS_CONTRACT.json').read_text())['gates']
    metrics = {'complete_events': len(frame), 'expected_events': len(rows), 'pairs': len(pairs),
        'max_TV': None if frame.empty else float(frame.TV.max()),
        'max_A90_relative': None if frame.empty else float(frame.A90_relative.max()),
        'max_pair_delta': None if pairs.empty else float(pairs.abs_delta.max()),
        'q99_pair_delta': None if pairs.empty else float(pairs.abs_delta.quantile(.99)),
        'sign_flips': 0 if pairs.empty else int(pairs.sign_flip.sum()),
        'true_pair_sign_flips': 0 if pairs.empty else int(pairs.loc[pairs.companion, 'sign_flip'].sum()),
        'spearman': None if len(pairs) < 2 else float(spearmanr(pairs.Z_from, pairs.Z_to).statistic),
        'P90_wall_seconds_completed_only': None if frame.empty else float(frame.wall_seconds.quantile(.9))}
    metrics['pass'] = bool(len(frame) == len(rows) and len(pairs) and
        metrics['max_TV'] <= gates['map_TV_max'] and metrics['max_A90_relative'] <= gates['A90_relative_max']
        and metrics['max_pair_delta'] <= gates['pair_delta_max_nats'] and metrics['P90_wall_seconds_completed_only'] <= gates['P90_wall_seconds'])
    metrics['missing_or_timeout_is_failure_not_omitted'] = len(frame) != len(rows)
    metrics['not_population_coverage_or_blind_PE_gate'] = True
    write(root/'contracts'/f'CONVERGENCE_{suffix}.json', metrics)
    print(json.dumps({'stage': 'quadrature_audit', 'comparison': suffix, **metrics}), flush=True)
    return metrics


def finish(root):
    protected = json.loads((root/'manifest/PROTECTED_BEFORE.json').read_text())
    changes = [x['path'] for x in protected if not Path(x['path']).is_file() or sha(x['path']) != x['sha256']]
    write(root/'manifest/PROTECTED_AFTER.json', {'files': len(protected), 'changed': changes, 'pass': not changes})
    gates = {p.stem: json.loads(p.read_text()) for p in sorted((root/'contracts').glob('*GATE.json'))}
    convergences = {p.stem: json.loads(p.read_text()) for p in sorted((root/'contracts').glob('CONVERGENCE*.json'))}
    summary = {'state': FINAL, 'utc': utc(), 'root': str(root), 'gates': gates, 'convergence': convergences,
        'protected_unchanged': not changes, 'bulk_started': False, 'encoder_training_started': False,
        'test_opened': False, 'remaining': ['data-recovered intrinsic templates, not oracle templates',
        'independent source/noise coverage validation', 'production only after all gates pass']}
    write(root/'contracts/FINAL_RESULT.json', summary)
    write(root/'STATUS.json', summary)
    text = '# ET-3 physical-strain repair development report\n\n'
    text += 'This is an ET-only development experiment, not a trained 3,000-event result. Historical O3/O4a and ET outputs remain unchanged.\n\n'
    text += '## Implemented repairs\n\nOne common amplitude per lens system; SI spin conversion; actual saved noisy strain matched filtering; identical PSD/template power for filtering and localization; explicit forward/inverse ET geometry.\n\n'
    text += '## Limits\n\nThe first localization control conditions on injected intrinsic parameters. It is NOT blind PE. No production maps or encoder training are authorized by this control alone. ET design independent Gaussian noise and the selection-limited GW-LMC population are controlled assumptions.\n\n'
    text += '```json\n'+json.dumps(summary, indent=2, ensure_ascii=False)+'\n```\n'
    (root/'reports/REPAIR_DEVELOPMENT_REPORT.md').write_text(text)
    files = [{'path': str(p.relative_to(root)), 'bytes': p.stat().st_size, 'sha256': sha(p)}
        for p in sorted(root.rglob('*')) if p.is_file() and p.parts[len(root.parts)] not in ('strain', 'package')
        and p.name not in ('OUTPUT_SHA256.csv',)]
    pd.DataFrame(files).to_csv(root/'manifest/OUTPUT_SHA256.csv', index=False)
    archive = root/'package'/f'{root.name}_development.tar.gz'
    with tarfile.open(archive, 'w:gz') as tar:
        for part in ('contracts', 'scripts', 'triggers', 'maps', 'tables', 'reports', 'manifest', 'build', 'STATUS.json'):
            tar.add(root/part, arcname=root.name+'/'+part)
    (archive.with_suffix('.gz.sha256')).write_text(sha(archive)+'  '+archive.name+'\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--generate', action='store_true')
    parser.add_argument('--extract', action='store_true')
    parser.add_argument('--maps', type=int, choices=[32, 64, 128])
    parser.add_argument('--case', type=Path)
    parser.add_argument('--core-test', type=Path)
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--finish', action='store_true')
    args = parser.parse_args()
    if args.core_test:
        if load_core(args.core_test).test() != 0:
            raise RuntimeError('Upstream core self-test failed')
        return
    if args.case:
        worker(args.root, json.loads(args.case.read_text()))
        return
    if args.prepare or args.all:
        prepare(args.root)
    if args.generate or args.all:
        generate(args.root)
    if args.extract or args.all:
        extract(args.root)
    if args.maps:
        maps(args.root, args.maps)
    if args.all:
        maps(args.root, 32)
        maps(args.root, 64)
        result = audit(args.root, 32, 64)
        if not result['pass']:
            build128(args.root)
            maps(args.root, 128)
            audit(args.root, 64, 128)
    if args.finish or args.all:
        finish(args.root)


if __name__ == '__main__':
    main()
