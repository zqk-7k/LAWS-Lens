"""Frozen real scope, explicit sky ordering, and downstream unified PE audit."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import json
from pathlib import Path

import h5py
import healpy as hp
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import uab_completion as c
from uab_evaluate import csv

P = Path('/root/autodl-tmp/gw-catalog')
ARCHIVE = P/'results/mcwf_unified_path875_devconf_20260908T181500Z'
O3 = P/'results/main_o3official_cfixed_v1_20260904_20260904T072435Z'
O4B = P/'results/o4b_hl_bayestar_new_score_only_20260912T072746Z'
OFFICIAL4 = P/'results/mcwf_pe_frontend_extension_20260907T041300Z/audit/gwtc4_frozen_external_reference.parquet'


def science():
    import mcwf_path_fresh_confirmation_v2_20260908 as f
    return f


def inventory():
    dest = c.OUT/'contracts/REAL_INPUT_MANIFEST.json'
    if c.check_complete(c.OUT/'contracts/REAL_INPUT_MANIFEST_FREEZE.json'):
        return
    f = science(); manifests = {}; paths = [OFFICIAL4, O3/'data/official_followup_stage_contract.csv']
    for run in c.U.RUNS:
        if run == 'O4b':
            path = O4B/'workspace/runs/real_gwtc5_o4b_v93_extension_20260830/data/event_manifest.csv'
            frame = pd.read_csv(path)
            restore = O4B/'manifests/RESTORE_PE_ARIA2.json'
            lookup = {r['event_name']: r for r in json.loads(restore.read_text())}
            entries = []
            for row in frame.to_dict('records'):
                r = lookup[row['event_name']]
                if c.U.sha(r['path']) != r['sha256']:
                    raise RuntimeError('Archived publisher-verified PE file changed')
                entries.append(dict(event_name=row['event_name'], gps_time=row['gps_time'],
                    pe_path=r['path'], pe_group=row['pe_result_samples_key'], map_path=row['sky_map_path'],
                    map_format='MOC_FITS', detectors=row['detectors'], network_snr=row['network_snr']))
                paths += [Path(r['path']), Path(row['sky_map_path'])]
            paths += [path, restore, O4B/'inference_inputs/real/COMPLETE.json']
        else:
            dep = {'O3': 'gwtc3', 'O4a': 'gwtc4'}[run]
            pairpath = ARCHIVE/f'development/evaluation/MCWF-UNIFIED-PATH875-DEVCONF/{dep}/seed_202607241/real_fusion_pairs.parquet'
            pairs = pd.read_parquet(pairpath, columns=['event_i', 'event_j'])
            names = sorted(set(pairs.event_i)|set(pairs.event_j))
            if len(names) != {'O3': 62, 'O4a': 74}[run]:
                raise RuntimeError('Frozen strict real scope changed')
            path = O3/'data/official70_event_manifest.csv' if run == 'O3' else f.dev.BASE.source_run(dep)/'data/event_manifest.csv'
            source = pd.read_csv(path).set_index('event_name').loc[names].reset_index()
            _, inputs = f.dev.real_inputs(dep)
            inputs = inputs.set_index('event_name').loc[names]
            if not inputs.strict_h1l1_preprocessing_pass.all():
                raise RuntimeError('Strict-HL real input audit failed')
            entries = []
            for row in source.to_dict('records'):
                pe = f.dev.BASE.morph.resolve_pe_path(row['sky_map_path'], dep)
                group = row.get('sky_map_internal_group', row.get('sky_map_group'))
                if not isinstance(group, str) or not group:
                    raise RuntimeError('Exact PE group missing')
                with h5py.File(pe, 'r') as h:
                    if group+'/posterior_samples' not in h or group+'/skymap/data' not in h:
                        raise RuntimeError('Frozen posterior/map group absent')
                entries.append(dict(event_name=row['event_name'], gps_time=row['gps_time'],
                    pe_path=str(pe), pe_group=group, map_path=str(pe), map_format='HDF5_HEALPIX',
                    detectors=row.get('detectors_available', 'UNKNOWN'), network_snr=row.get('network_snr'),
                    detector_audit=json.loads(inputs.loc[row['event_name'], 'detector_audit'])))
                paths.append(Path(pe))
            paths += [pairpath, path]
        manifests[run] = entries
    c.write(dest, dict(runs=manifests, PE_values_not_read=True, historical_rank_values_not_read=True,
                      scope_source='Frozen event names only, no replacement or rank filtering',
                      real_PE_groups_frozen=True, official4_path=str(OFFICIAL4)))
    c.seal(c.OUT/'contracts/REAL_INPUT_MANIFEST_FREEZE.json', [dest]+sorted(set(paths)),
           exact_event_counts=dict(O3=62, O4a=74, O4b=86), ordering_not_inferred_from_filename=True)


def entries(run):
    marker = c.OUT/'contracts/REAL_INPUT_MANIFEST_FREEZE.json'
    if not marker.exists():
        raise RuntimeError('Freeze real-input identity before scoring')
    return json.loads((c.OUT/'contracts/REAL_INPUT_MANIFEST.json').read_text())['runs'][run]


def prepare(run):
    c.gate('real')
    dest = c.OUT/'real_inputs'/run
    if c.check_complete(dest/'COMPLETE.json'):
        return
    dest.mkdir(parents=True, exist_ok=True)
    data = entries(run)
    if run == 'O4b':
        original = O4B/'inference_inputs/real'
        for item in json.loads((original/'COMPLETE.json').read_text())['outputs']:
            if c.U.sha(item['path']) != item['sha256']:
                raise RuntimeError('Original real strain views changed')
        e = pd.read_parquet(original/'events.parquet')
        if e.event_name.tolist() != [r['event_name'] for r in data]:
            raise RuntimeError('O4b real order differs')
        for name in ('raw2s.npy', 'short_raw2s.npy', 'low16s.npy', 'psd.npy', 'frequency.npy'):
            (dest/name).symlink_to(original/name)
        e.to_parquet(dest/'events.parquet', index=False)
    else:
        f = science(); phys = f.dev.BASE.v7.v3
        dep = 'gwtc3' if run == 'O3' else 'gwtc4'
        cache = phys.HdfCache(f.dev.MAIN/'cache/source_run' if run == 'O3' else phys.SOURCES['GWTC4'], max_files=2)
        raw, low, psds, rows, audits = [], [], [], [], []
        for idx, row in enumerate(data):
            channels, refs = [], []
            bydet = {d['detector']: d for d in row['detector_audit']}
            for detector in ('H1', 'L1'):
                entry = bydet[detector]
                wave, gps, duration = cache.get(entry['path'])
                if abs(len(wave)/duration-4096) > 1e-3:
                    raise RuntimeError('Strain sample rate differs')
                x = phys._extract_channel_window(wave, gps, float(row['gps_time']), phys.RAW_PADDED_SAMPLES, -24.75)
                reference, start = phys._extract_psd_reference(wave, gps, float(row['gps_time']))
                if x is None or reference is None or abs(start-entry['psd_reference_start_gps']) > 1/4096:
                    raise RuntimeError('Original strain/reference window differs')
                channels.append(x); refs.append(reference)
            frequency, psd = phys.estimate_psd(np.stack(refs))
            full = phys.preprocess_24s(np.stack(channels), frequency, psd)
            raw.append(f.dev.TRAIN.make_window_view(full[None], 2)[0])
            low.append(f.low.low_view(np.stack(channels), frequency, psd, phys))
            psds.append(psd)
            rows.append(dict(idx=idx, event_uid=row['event_name'], event_name=row['event_name'],
                gps_time=row['gps_time'], noise_bank_index=idx, detectors=row['detectors'],
                strict_h1l1_preprocessing_pass=True, sky_map_path=row['map_path']))
            audits.append(dict(event=row['event_name'], strict_HL=True, raw_samples=len(channels[0]),
                short_samples=4096, low_samples=4096, zero_fill=False, PSD_reference_verified=True))
        for name, value in [('raw2s', raw), ('low16s', low), ('psd', psds), ('frequency', frequency)]:
            arr = np.asarray(value)
            if not np.isfinite(arr).all():
                raise RuntimeError('Nonfinite real preprocessing')
            np.save(dest/f'{name}.npy', arr)
        pd.DataFrame(rows).to_parquet(dest/'events.parquet', index=False)
        csv(dest/'preprocessing_audit.csv', pd.DataFrame(audits))
    c.seal(dest/'COMPLETE.json', [p for p in dest.iterdir() if p.is_file()], no_PE_waveform_input=True)


def bool_scalar(value):
    x = np.asarray(value)
    if x.size != 1:
        raise ValueError('Ordering must be scalar')
    x = x.reshape(-1)[0]
    v = x.decode('ascii') if isinstance(x, (bytes, np.bytes_)) else str(x)
    v = v.strip().lower()
    if v in ('1', 'true', 't', 'yes', 'y'): return True
    if v in ('0', 'false', 'f', 'no', 'n'): return False
    raise ValueError('Unrecognized HEALPix ordering flag')


def probability(row, nside):
    path = Path(row['map_path'])
    if row['map_format'] == 'HDF5_HEALPIX':
        group = row['pe_group']
        with h5py.File(path, 'r') as h:
            mass = np.asarray(h[group+'/skymap/data'][:], np.float64).reshape(-1)
            raw_nest = h[group+'/skymap/meta_data/nest'][()]
            nest = bool_scalar(raw_nest)
        native = hp.npix2nside(len(mass))
        if not np.isfinite(mass).all() or (mass < 0).any() or mass.sum() <= 0:
            raise RuntimeError('Invalid public HEALPix probability')
        mass /= mass.sum()
        if nest:
            original = mass
            mass = hp.reorder(original, n2r=True)
            if not np.array_equal(original, hp.reorder(mass, r2n=True)):
                raise RuntimeError('Ordering round trip failed')
        mass = hp.ud_grade(mass, nside_out=nside, order_in='RING', order_out='RING', power=-2)
        meta = dict(native_nside=int(native), source_ordering='NESTED' if nest else 'RING',
                    ordering_raw=repr(raw_nest), NESTED_to_RING=bool(nest), exact_group=group)
    else:
        from astropy.io import fits
        from ligo.skymap.io.fits import read_sky_map
        b = c.U.module(c.ROOT/'scripts/bayestar_si_fixed.py', 'uab_public_raster')
        with fits.open(path) as f:
            if f[1].header.get('COORDSYS') not in ('C', 'ICRS', 'EQUATORIAL'):
                raise RuntimeError('Noncelestial public map')
        moc = read_sky_map(path, moc=True)
        if not np.isfinite(moc['PROBDENSITY']).all() or (moc['PROBDENSITY'] < 0).any():
            raise RuntimeError('Invalid public MOC')
        mass = hp.reorder(b.raster_probability(moc, nside), n2r=True)
        meta = dict(native_nside='multi-order', source_ordering='NESTED', NESTED_to_RING=True, exact_group=row['pe_group'])
    mass = np.asarray(mass, np.float64)
    mass /= mass.sum(dtype=np.float64)
    return mass, dict(**meta, analysis_nside=nside, output_ordering='RING', coordinate_frame='ICRS',
                       map_path=str(path), map_sha256=c.U.sha(path), public_PE_temperature=1.)


def sky(run):
    c.gate('real')
    dest = c.OUT/'real_sky'/run
    if c.check_complete(dest/'COMPLETE.json'):
        return
    import torch
    dest.mkdir(parents=True, exist_ok=True)
    data = entries(run); n = len(data); i, j = np.triu_indices(n, 1)
    pairs = pd.DataFrame(dict(idx_i=i, idx_j=j,
        event_i=np.array([r['event_name'] for r in data])[i], event_j=np.array([r['event_name'] for r in data])[j]))
    audit = []
    for nside in (256, 512, 1024):
        bank = np.empty((n, hp.nside2npix(nside)), np.float32)
        for k, row in enumerate(data):
            mass, record = probability(row, nside); bank[k] = mass
            audit.append(dict(event=row['event_name'], **record))
        norm = bank.sum(1, dtype=np.float64)
        overlap = torch.zeros((n, n), dtype=torch.float64, device='cuda'); bc = torch.zeros_like(overlap)
        for start in range(0, bank.shape[1], 65536):
            block = torch.as_tensor(np.asarray(bank[:, start:start+65536], np.float64)/norm[:, None], device='cuda')
            overlap += block@block.T; root = block.sqrt(); bc += root@root.T
        pairs[f'sky_log_bf_nside{nside}'] = np.log(np.maximum(bank.shape[1]*overlap.cpu().numpy()[i, j], 1e-300))
        pairs[f'sky_BC_nside{nside}'] = bc.cpu().numpy()[i, j].clip(0, 1)
        del bank, overlap, bc, block, root
        torch.cuda.empty_cache()
    pairs['sky_raw_log_bf'] = pairs.sky_log_bf_nside512; pairs['sky_BC'] = pairs.sky_BC_nside512
    pairs['flip_512_1024'] = np.sign(pairs.sky_log_bf_nside512) != np.sign(pairs.sky_log_bf_nside1024)
    pairs['abs_delta_512_1024'] = abs(pairs.sky_log_bf_nside512-pairs.sky_log_bf_nside1024)
    pairs.to_parquet(dest/'pairs.parquet', index=False)
    csv(dest/'map_ordering_audit.csv', pd.DataFrame(audit))
    c.seal(dest/'COMPLETE.json', [dest/'pairs.parquet', dest/'map_ordering_audit.csv'],
           all_strict_pairs=True, analysis_nside=512, public_PE_unmodified=True, shared_both_arms=True)


def official_table(run):
    if run == 'O3':
        path = O3/'data/official_followup_stage_contract.csv'
        d = pd.read_csv(path)
        d['official_frontend'] = d.official_po_or_ml_fpp_below_0p01
    elif run == 'O4a':
        path = OFFICIAL4
        source = pd.read_parquet(path)
        d = source[['pair_key']+[k for k in source if k.startswith('official_')]].copy()
        d['official_frontend'] = d.official_po_or_phazap_fpp_below_0p01
    else:
        return None
    d['public_Hanabi_overlap'] = d.official_any_pair_resolved_hanabi_overlap
    d['official_provenance_file'] = str(path)
    if d.pair_key.duplicated().any():
        raise RuntimeError('Duplicate official pair key')
    return d


def pe(run):
    c.gate('real')
    for arm in c.U.ARMS:
        if not c.check_complete(c.deployment(run, arm)/'real_ranking/COMPLETE.json'):
            raise RuntimeError('Freeze both-arm ranks before PE')
    dest = c.OUT/'real_PE'/run
    if c.check_complete(dest/'COMPLETE.json'):
        return
    dest.mkdir(parents=True, exist_ok=True)
    import o4b_hl_nso_pe_audit_20260912 as pe_api
    samples, audit, summaries = {}, [], []
    for row in entries(run):
        values, record = pe_api.posterior(row['pe_path'], row['pe_group'])
        samples[row['event_name']] = values
        audit.append(dict(event=row['event_name'], path=row['pe_path'], sha256=c.U.sha(row['pe_path']), **record))
        for parameter, v in values.items():
            summaries.append(dict(event=row['event_name'], parameter=parameter, median=float(np.median(v)),
                                  sample_SD=float(np.std(v, ddof=1)), samples=len(v)))
    names = sorted(samples); i, j = np.triu_indices(len(names), 1)
    pairs = pd.DataFrame(dict(pair_key=[names[a]+'--'+names[b] for a, b in zip(i, j)]))
    for parameter in ('Mc', 'q', 'chi_eff', 'apparent_distance'):
        values = [samples[name][parameter] for name in names]
        med = np.array([np.median(v) for v in values]); sd = np.array([np.std(v, ddof=1) for v in values])
        transformed = [np.log(v) for v in values] if parameter in ('Mc', 'apparent_distance') else values
        bounds = (0., 1.) if parameter == 'q' else (-1., 1.) if parameter == 'chi_eff' else None
        if bounds:
            lo, hi = bounds
        else:
            lo, hi = min(v.min() for v in transformed), max(v.max() for v in transformed)
            margin = .05*(hi-lo); lo -= margin; hi += margin
        grid = np.linspace(lo, hi, 4096)
        mass = np.stack([pe_api.density(v, grid, bounds) for v in transformed])
        pairs['BC_'+parameter] = (np.sqrt(mass)@np.sqrt(mass).T)[i, j].clip(0, 1)
        pairs['D_'+parameter] = abs(med[i]-med[j])/np.sqrt(sd[i]**2+sd[j]**2)
        np.savez_compressed(dest/f'{parameter}_marginal_density.npz', event_names=names, grid=grid, probability_mass=mass)
    pairs['Dmax'] = pairs[['D_Mc', 'D_q', 'D_chi_eff']].max(1)
    official = official_table(run)
    if official is None:
        pairs['official_frontend'] = pd.NA; pairs['public_Hanabi_overlap'] = pd.NA
        pairs['official_screening_stage'] = 'NO_VERIFIED_O4B_PAIR_TABLE_IN_FROZEN_INPUTS'
        pairs['official_hanabi_conclusion'] = 'NOT_RUN; NO_VERIFIED_PUBLIC_PAIR_CONCLUSION'
    else:
        pairs = pairs.merge(official, on='pair_key', how='left', validate='one_to_one')
        if pairs.official_frontend.isna().any():
            raise RuntimeError('Missing official entries inside frozen O3/O4a scope')
    pairs.to_parquet(dest/'PE_official_all_pairs.parquet', index=False)
    csv(dest/'event_posterior_summary.csv', pd.DataFrame(summaries))
    c.write(dest/'PE_GROUP_MANIFEST.json', audit)
    budgets, correlations = [], []
    for arm in c.U.ARMS:
        original = c.deployment(run, arm)/'real_ranking/consensus_all_pairs.parquet'
        digest = c.U.sha(original)
        ranked = pd.read_parquet(original).merge(pairs, on='pair_key', how='left', validate='many_to_one')
        if ranked.BC_Mc.isna().any():
            raise RuntimeError('Incomplete downstream PE join')
        for method, group in ranked.groupby('method'):
            group = group.sort_values('consensus_rank')
            out = dest/arm/method; out.mkdir(parents=True, exist_ok=True)
            group.to_parquet(out/'all_pairs_with_PE_official.parquet', index=False)
            csv(out/'all_pairs_with_PE_official.csv', group)
            for b in (10, 20, 50, 100):
                top = group.head(b)
                csv(out/f'top{b}_with_PE_official.csv', top)
                budgets.append(dict(run=run, arm=arm, method=method, budget=b,
                    Mc_BC_ge05=int((top.BC_Mc >= .5).sum()), Dmax_le3=int((top.Dmax <= 3).sum()),
                    median_Mc_BC=float(top.BC_Mc.median()), catastrophic_Mc=int(((top.BC_Mc < .1) | (top.D_Mc > 5)).sum()),
                    official_frontend_count=None if run == 'O4b' else int(top.official_frontend.sum()),
                    public_Hanabi_count=None if run == 'O4b' else int(top.public_Hanabi_overlap.sum()),
                    official_available=run != 'O4b', Hanabi_rerun=False))
            correlations.append(dict(run=run, arm=arm, method=method,
                Spearman_waveform_Mc_BC=float(spearmanr(group.waveform_score_mean, group.BC_Mc).statistic),
                Spearman_waveform_negative_DMc=float(spearmanr(group.waveform_score_mean, -group.D_Mc).statistic),
                independent_pair_p_value_claimed=False))
        if c.U.sha(original) != digest:
            raise RuntimeError('PE audit altered rank table')
    csv(dest/'budget_summary.csv', pd.DataFrame(budgets))
    csv(dest/'score_PE_correlations.csv', pd.DataFrame(correlations))
    c.seal(dest/'COMPLETE.json', [dest/'PE_official_all_pairs.parquet', dest/'budget_summary.csv', dest/'PE_GROUP_MANIFEST.json'],
           common_PE_estimator_all_runs=True, true_lensing_labels=False, new_Hanabi=False)


def tests():
    for value in (True, 1, 'TRUE', b'True', np.array([b'True'])):
        assert bool_scalar(value)
    for value in (False, 0, 'false', b'False', np.array([b'False'])):
        assert not bool_scalar(value)
    for value in ('bad', [], [True, False]):
        try: bool_scalar(value)
        except ValueError: pass
        else: raise AssertionError('Malformed ordering accepted')
    x = np.random.default_rng(42).random(12*32**2); x /= x.sum()
    assert np.array_equal(x, hp.reorder(hp.reorder(x, n2r=True), r2n=True))
    uniform = np.ones_like(x)/len(x)
    assert abs(np.log(len(x)*np.dot(x, uniform))) < 1e-14
    c.write(c.OUT/'contracts/ORDERING_UNIT_TESTS.json', dict(state='PASS', byte_array_True=True,
            false_array=True, ambiguous_rejected=True, permutation_roundtrip_exact=True, uniform_BF_one=True))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--stage', choices=['inventory', 'prepare', 'sky', 'pe', 'tests'], required=True)
    p.add_argument('--run', default='O3'); a = p.parse_args(); c.initialize(a.root, a.out)
    if a.stage in ('inventory', 'tests'): globals()[a.stage]()
    else: globals()[a.stage](a.run)
