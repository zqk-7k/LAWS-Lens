#!/usr/bin/env python3
"""Single waveform calibration with a PyCBC-verified physical phase match."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import json
from pathlib import Path
import pickle
import shutil
import sys
import time
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.ensemble import HistGradientBoostingClassifier
import torch
from pycbc.filter import match as pycbc_match
from pycbc.types import FrequencySeries
from pycbc.waveform import get_fd_waveform

P = Path('/root/autodl-tmp/gw-catalog')
sys.path.insert(0, str(P / 'scripts/experiments'))
import mcwf_profile_single_waveform_20260909 as s
h, co, r, n = s.h, s.co, s.r, s.n
ROOT = None
ACTIVE_FIT = False
FEATURES = ('embedding_only', 'log_physical_phase_match') + h.FEATURES
METHODS = ('NODUP-DIRECT-REPLAY', 'PHASE-MATCH-LINEAR', 'PHASE-MATCH-TREE',
           'PHASE-MATCH-ACTIVE-LINEAR', 'PHASE-MATCH-ACTIVE-TREE')
PARENT = P / 'results/mcwf_nodup_single_waveform_14_20260909T095000Z'
DF, FS, NFFT = 1 / 26, 2048, 53248
FLOW, FHIGH = 20., 580.
CACHE = {}


def parameters(dep, seed, split, catalog):
    slot = n.recipes()[dep, seed]['slot']
    tag = co.app.tag_for(seed, split, catalog)
    dest = ROOT / f'parameters/{dep}/{slot}_{tag}.npz'
    if dest.exists():
        return dict(np.load(dest))
    mass = co.mass(dep, seed, split, catalog)
    source = (r.INTR / f'predictions/CONDITIONAL-ETA-CHI/{dep}/{slot}_0_development.npz'
              if split == 'development' else r.prediction_path(dep, seed, split, catalog))
    with np.load(source) as a:
        joint = a['joint']
    valid = np.isfinite(mass['p']).all(1)
    theta = np.full((len(valid), 3), np.nan)
    theta[valid, 0] = r.mass_summary(mass['p'])[valid, 0]
    marginal = joint[valid].sum(1).reshape(valid.sum(), len(r.physical.ETA), len(r.physical.CHI))
    marginal /= marginal.sum((1, 2), keepdims=True)
    eta = (marginal.sum(2) * r.physical.ETA).sum(1)
    spin = (marginal.sum(1) * r.physical.CHI).sum(1)
    q = (1 - np.sqrt(1 - 4 * eta.clip(1e-9, .25))) / (1 + np.sqrt(1 - 4 * eta.clip(1e-9, .25)))
    theta[valid, 1], theta[valid, 2] = q, spin
    if split == 'development':
        rows = [json.loads(p.read_text()) for p in sorted((co.app.cal.PROFILE / 'events').glob(dep + '_*.json'))]
    else:
        rows = [json.loads(p.read_text()) for p in sorted((co.PARENT / f'profile_events/{dep}/{tag}').glob('*.json'))]
    for row in rows:
        i = int(row['row_index'])
        if mass['active'][i]:
            theta[i] = row['logmc'], row['q'], row['chieff_equal']
    if not np.isfinite(theta[valid]).all() or (theta[valid, 1] <= 0).any() or (theta[valid, 1] > 1).any() or (abs(theta[valid, 2]) > 1).any():
        raise RuntimeError('Invalid waveform-only source template parameters')
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest, theta=theta, valid=valid, active=mass['active'])
    n.write_json(dest.with_suffix('.json'), {'UTC': n.utc(), 'new_parameter_head': False,
        'public_PE_read': False, 'old_encoder_Mc_q_read': False, 'point_template_not_posterior': True,
        'profile_active': int(mass['active'].sum()), 'events': int(valid.sum()), 'hash': n.sha(dest)})
    return dict(theta=theta, valid=valid, active=mass['active'])


def template(theta):
    mass, q, spin = np.exp(theta[0]), theta[1], theta[2]
    m1 = mass * (1 + q)**.2 / q**.6
    hp, _ = get_fd_waveform(approximant='IMRPhenomD', mass1=m1, mass2=m1 * q,
        spin1z=spin, spin2z=spin, delta_f=DF, f_lower=FLOW, f_final=FS / 2,
        distance=1000., inclination=0.)
    hp.resize(NFFT // 2 + 1)
    return np.asarray(hp, dtype=np.complex128)


def pair_match(dep, seed, split, catalog=None):
    key = dep, seed, split, catalog
    if key in CACHE:
        return CACHE[key]
    tag = co.app.tag_for(seed, split, catalog)
    slot = n.recipes()[dep, seed]['slot']
    dest = ROOT / f'matches/{dep}/{slot}_{tag}.npz'
    if dest.exists():
        result = dict(np.load(dest))
        CACHE[key] = result
        return result
    start = time.perf_counter()
    p = parameters(dep, seed, split, catalog)
    valid, theta = p['valid'], p['theta']
    ids = np.flatnonzero(valid)
    hp = np.stack([template(x) for x in theta[valid]])
    reference = np.load(ROOT / f'contracts/{dep}_REFERENCE_PSD.npy')
    kmin, kmax = int(FLOW / DF), int(FHIGH / DF)
    weighted = np.zeros_like(hp)
    weighted[:, kmin:kmax] = hp[:, kmin:kmax] / np.sqrt(reference[None, kmin:kmax])
    norms = np.linalg.norm(weighted, axis=1, keepdims=True)
    if not np.isfinite(weighted).all() or (norms <= 0).any():
        raise RuntimeError('Invalid PSD-weighted source waveform')
    z = torch.as_tensor(weighted / norms, dtype=torch.complex64, device='cuda')
    torch.cuda.reset_peak_memory_stats()
    if split == 'development':
        meta = pd.read_parquet(n.t.PREVIOUS / f'expanded_data/{dep}/validation/event_metadata.parquet')
        meta['deployment'] = dep
        folds = r.source_fold(meta)
        pair_parts = []
        for side in (0, 1):
            local = np.flatnonzero(folds[ids] == side)
            i, j = np.triu_indices(len(local), 1)
            pair_parts.append(np.column_stack([local[i], local[j]]))
        pairs = np.concatenate(pair_parts)
    else:
        pairs = np.column_stack(np.triu_indices(len(ids), 1))
    values = np.empty(len(pairs))
    for start_row in range(0, len(pairs), 256):
        block = pairs[start_row:start_row + 256]
        i = torch.as_tensor(block[:, 0], device='cuda')
        j = torch.as_tensor(block[:, 1], device='cuda')
        frequency = torch.zeros((len(block), NFFT), dtype=torch.complex64, device='cuda')
        frequency[:, :z.shape[1]] = z[i].conj() * z[j]
        value = torch.fft.ifft(frequency, dim=-1).abs().amax(1) * NFFT
        values[start_row:start_row + len(block)] = value.double().cpu().numpy()
    if np.any(values > 1 + 2e-5) or np.any(values < 0) or not np.isfinite(values).all():
        raise RuntimeError('Physical match outside[0,1]')
    checks = []
    for index in np.linspace(0, len(pairs) - 1, min(12, len(pairs))).astype(int):
        i, j = pairs[index]
        expected = float(pycbc_match(FrequencySeries(hp[i], delta_f=DF), FrequencySeries(hp[j], delta_f=DF),
            psd=FrequencySeries(reference, delta_f=DF), low_frequency_cutoff=FLOW,
            high_frequency_cutoff=FHIGH, subsample_interpolation=False)[0])
        error = abs(expected - values[index])
        checks.append({'i': int(ids[i]), 'j': int(ids[j]), 'pycbc': expected, 'gpu': float(values[index]), 'absolute_error': error})
        if error > 2e-5:
            raise RuntimeError('GPU physical match disagrees with PyCBC')
    result = np.full((len(valid), len(valid)), np.nan)
    result[ids[pairs[:, 0]], ids[pairs[:, 1]]] = values.clip(0, 1)
    result[ids[pairs[:, 1]], ids[pairs[:, 0]]] = values.clip(0, 1)
    result[ids, ids] = 1.
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest, match=result)
    n.write_csv(dest.with_name(dest.stem + '_PYCB_UNIT.csv'), checks)
    n.write_json(dest.with_suffix('.json'), {'UTC': n.utc(), 'events': len(ids), 'pairs': len(pairs),
        'wall_seconds': time.perf_counter() - start, 'gpu_peak_bytes': torch.cuda.max_memory_allocated(),
        'pycbc_unit_max_error': max(row['absolute_error'] for row in checks), 'shape': list(result.shape),
        'frozen_reference_psd': n.sha(ROOT / f'contracts/{dep}_REFERENCE_PSD.npy'), 'hash': n.sha(dest)})
    CACHE[key] = {'match': result}
    print('PHYSICAL_PHASE_MATCH', dep, tag, seed, len(pairs), round(time.perf_counter() - start, 2), flush=True)
    return CACHE[key]


def load_panel(dep, seed, split, catalog=None):
    f = h.load_panel(dep, seed, split, catalog)
    a = pair_match(dep, seed, split, catalog)['match']
    f['physical_phase_match'] = a[f.idx_i.to_numpy(int), f.idx_j.to_numpy(int)]
    f['log_physical_phase_match'] = np.log(f.physical_phase_match.to_numpy().clip(1e-7, 1.))
    return f


def development(dep, seed):
    h.ACTIVE_FIT = ACTIVE_FIT
    old = h.development(dep, seed)
    z = np.load(s.EMB / f'{dep}/{seed}_development.npy').astype(float)
    z /= np.linalg.norm(z, axis=1, keepdims=True)
    a = pair_match(dep, seed, 'development')['match']
    return {fold: (np.column_stack([np.sum(z[i] * z[j], axis=1), np.log(a[i, j].clip(1e-7, 1.)), x]), y, w, i, j)
            for fold, (x, y, w, i, j) in old.items()}


def linear(x, y, w, ridge):
    mu = np.sum(w[:, None] * x, axis=0)
    sd = np.sqrt(np.sum(w[:, None] * (x - mu)**2, axis=0)).clip(1e-6)
    z = (x - mu) / sd
    def loss(theta):
        logits = theta[0] + z @ theta[1:]
        err = w * (expit(logits) - y)
        return (float(np.sum(w * (np.logaddexp(0., logits) - y * logits)) + .5 * ridge * (theta[1:] @ theta[1:])),
                np.r_[err.sum(), z.T @ err + ridge * theta[1:]])
    fit = minimize(loss, np.zeros(x.shape[1] + 1), jac=True, method='L-BFGS-B',
        bounds=[(None, None)] + [(0., None)] * 5 + [(None, None)] * 3,
        options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-8})
    if not fit.success:
        raise RuntimeError(fit.message)
    return {'kind': 'LINEAR', 'mu': mu, 'sd': sd, 'theta': fit.x}


def fit_classifier(dep, seed, kind):
    parts = development(dep, seed)
    x, y, w, _, _ = parts[0]
    xx, yy, ww, _, _ = parts[1]
    options, trials = [], []
    for ridge in (.001, .01, .1, 1.):
        models = [linear(x, y, w, ridge)] if kind == 'LINEAR' else []
        if kind == 'TREE':
            for leaves in (3, 7):
                model = HistGradientBoostingClassifier(max_iter=100, learning_rate=.05, max_leaf_nodes=leaves,
                    min_samples_leaf=20, l2_regularization=ridge, early_stopping=False,
                    monotonic_cst=[1, 1, 1, 1, 1, 0, 0, 0], random_state=202609092)
                model.fit(x, y, sample_weight=w * len(x))
                models.append({'kind': 'TREE', 'estimator': model, 'leaves': leaves})
        for model in models:
            logits = r.raw_predict(model, xx)
            loss = float(np.sum(ww * (np.logaddexp(0., logits) - yy * logits)))
            options.append(model)
            trials.append({'ridge': ridge, 'leaves': model.get('leaves', 0), 'logloss': loss})
    win = min(range(len(trials)), key=lambda k: (trials[k]['logloss'], trials[k]['leaves'], -trials[k]['ridge']))
    folder = 'active' if ACTIVE_FIT else 'global'
    dest = ROOT / f'calibration/{folder}/{kind}/{dep}/{seed}.pkl'
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open('xb') as file:
        pickle.dump(options[win], file)
    n.write_csv(dest.with_name(dest.stem + '_GRID.csv'), trials)
    return {'file': str(dest.relative_to(ROOT)), 'sha256': n.sha(dest), 'kind': kind,
            'minimum': x.min(0).tolist(), 'maximum': x.max(0).tolist(), 'cap': float(np.log(int(y.sum()) + 1)),
            'fit_sources': int(y.sum()), 'tune_sources': int(yy.sum()), 'selected': trials[win]}


def infer(f, c):
    original = h.isolated.ORIGINALS[c['deployment'], c['seed']]
    base = co.BASE_INFER(f, {**original, 'method': METHODS[0]})
    if c['method'] == METHODS[0]:
        return base
    score = r.apply_classifier(f[list(FEATURES)].to_numpy(float), c['waveform_calibrator'])
    if c['active_only']:
        active = f.profile_pair_active.to_numpy(bool)
        out = tuple(np.where(active, a, b) for a, b in zip(score, base))
        if not np.array_equal(out[0][~active], base[0][~active]):
            raise RuntimeError('Inactive waveform changed')
        return out
    return score


def freeze():
    if ROOT.exists():
        raise RuntimeError('Independent output required')
    for name in ('contracts', 'calibration', 'configs', 'parameters', 'matches', 'tables', 'audit', 'reports',
                 'scripts', 'logs', 'manifest', 'results', 'figures'):
        (ROOT / name).mkdir(parents=True)
    n.write_json(ROOT / 'contracts/ANALYSIS_CONTRACT.json', {
        'id': 'MCWF-NODUP-PHYSICAL-PHASE-MATCH-17', 'UTC': n.utc(), 'status': n.STATUS,
        'goal_achieved': False, 'adaptive_development': True, 'same_both_runs': True,
        'motivation': 'A high neural embedding cosine need not mean consistent inspiral phase. Test a normalized physical template match, not another explicit old Mc regression score.',
        'parameters': 'For R10-quality-active events use the unchanged IMRPhenomD profile optimum(logMc,q,chi). Otherwise use the frozen new mass median and joint posterior mean eta/chi. Never publicPE or oldencoder Mc/q.',
        'reference_psd': 'Median PSD over simulation fitfold0 noise banks, separately H1/L1; harmonic-mean network reference. No real candidate or target posterior used.',
        'match': 'IMRPhenomD source templates26s2048Hz;PSD-weighted20-580Hz normalized overlap,maximized over relative time and phase. GPUFFT checked against PyCBC.match on12pairs per panel to2e-5 absolute tolerance.',
        'limitations': 'Noise-derived point templates,not full strain likelihood or PE. Aligned-spin approximation and uncertain inactive medians require simulation validation.',
        'features': list(FEATURES), 'methods': list(METHODS),
        'calibration': 'ONE joint waveform classifier. No additional Zcos or mass score added. Source/noise fitfold0,proper balanced logloss fold1;ridge.001,.01,.1,1;3/7leaf100iterationmonotonicTREEorLINEAR.',
        'scope_controls': 'Global and actual-profile-active-only models both reported. Inactive control uses immutable NODUP.',
        'OOD': 'Fitfeaturebox;positiveOOD0;caplog(nfittrue+1). No candidate-driven thresholds.',
        'frozen': ['time', 'sky', 'encoder', 'profile optima', 'outer weights', 'historical results', 'scope', 'paper'],
        'no_total_blend': True, 'no_old_Mc_q_head': True,
        'references': ['https://pycbc.org/pycbc/latest/html/filter.html', 'https://arxiv.org/abs/gr-qc/9402014']})
    files = pd.read_csv(PARENT / 'manifest/INPUT_SHA256.csv').to_dict('records')
    for dep in n.DEPS:
        base = n.t.PREVIOUS / f'expanded_data/{dep}'
        meta = pd.read_parquet(base / 'validation/event_metadata.parquet')
        meta['deployment'] = dep
        banks = np.sort(meta.loc[r.source_fold(meta) == 0, 'noise_bank_index'].unique())
        freq = np.load(base / 'noise/frequency.npy')
        psds = np.load(base / 'noise/psd.npy', mmap_mode='r')
        median = np.median(psds[banks], axis=0)
        reference = 2 / (1 / median[0] + 1 / median[1])
        f = np.arange(NFFT // 2 + 1) * DF
        output = np.exp(np.interp(f, freq, np.log(reference.clip(1e-100))))
        if not np.isfinite(output).all() or (output <= 0).any():
            raise RuntimeError('Invalid fit-only reference PSD')
        np.save(ROOT / f'contracts/{dep}_REFERENCE_PSD.npy', output)
        n.write_json(ROOT / f'contracts/{dep}_REFERENCE_PSD.json', {'banks': banks.tolist(), 'fit_only': True,
            'positive': True, 'frequency_step_hz': DF, 'sampling_hz': FS, 'duration_s': 26.,
            'sha256': n.sha(ROOT / f'contracts/{dep}_REFERENCE_PSD.npy')})
        for name in ('frequency.npy', 'psd.npy'):
            p = base / 'noise' / name
            files.append({'path': str(p), 'sha256': n.sha(p), 'bytes': p.stat().st_size})
    n.write_csv(ROOT / 'manifest/INPUT_SHA256.csv', files)
    shutil.copy2(__file__, ROOT / 'scripts/profile_phase_match.py')
    n.write_json(ROOT / 'contracts/START_FREEZE.json', {'UTC': n.utc(), 'script_sha256': n.sha(Path(__file__)),
        'contract_sha256': n.sha(ROOT / 'contracts/ANALYSIS_CONTRACT.json')})


def calibrate():
    global ACTIVE_FIT
    choices, audits = [], []
    for dep in n.DEPS:
        for seed in n.SEEDS:
            original = h.isolated.ORIGINALS[dep, seed]
            base = {**original, 'method': METHODS[0]}
            choices.append(base)
            f = load_panel(dep, seed, 'validation')
            z = infer(f, base)[0]
            fm = n.cf.fast_metrics(f, n.cf.channels(f, z) @ np.asarray(original['weights']))
            wm = n.cf.fast_metrics(f, z)
            for active in (False, True):
                ACTIVE_FIT = active
                for kind in ('LINEAR', 'TREE'):
                    spec = fit_classifier(dep, seed, kind)
                    method = ('PHASE-MATCH-ACTIVE-' if active else 'PHASE-MATCH-') + kind
                    c = {**original, 'method': method, 'waveform_calibrator': spec, 'active_only': active}
                    new = infer(f, c)[0]
                    newfm = n.cf.fast_metrics(f, n.cf.channels(f, new) @ np.asarray(original['weights']))
                    newwm = n.cf.fast_metrics(f, new)
                    c.update(tune_guard=n.cf.guard(newfm, fm) and n.cf.guard(newwm, wm), tune_metrics=newfm)
                    bad = f.copy()
                    bad['pair_key'], bad['pe_mc_bhattacharyya_coefficient'], bad['official_po_fpp'] = 'ignored', -1., 1.
                    if not np.array_equal(new, infer(bad, c)[0]):
                        raise RuntimeError('Forbidden metadata affected waveform score')
                    choices.append(c)
                    audits.append({'deployment': dep, 'seed': seed, 'method': method, 'tune_guard': c['tune_guard'],
                        'calibration_logloss': spec['selected']['logloss'], 'forbidden_input_delta': 0., **newfm})
                    print('PHASE_MATCH_CALIBRATED', dep, seed, method, spec['selected'], c['tune_guard'], flush=True)
    n.write_csv(ROOT / 'tables/PHASE_MATCH_CALIBRATION.csv', audits)
    n.write_json(ROOT / 'configs/SELECTED_CONFIGURATIONS.json', choices)
    n.write_json(ROOT / 'contracts/CONFIGURATIONS_FROZEN.json', {'UTC': n.utc(),
        'file': 'configs/SELECTED_CONFIGURATIONS.json', 'sha256': n.sha(ROOT / 'configs/SELECTED_CONFIGURATIONS.json'),
        'no_test_or_real_selection': True})


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--stage', choices=('freeze', 'calibrate', 'evaluate', 'real'), required=True)
    a = p.parse_args()
    ROOT = s.ROOT = h.ROOT = co.ROOT = r.ROOT = co.score.ROOT = a.root
    r.install()
    co.score.METHODS, co.score.matrices = METHODS, co.matrices
    n.METHODS, n.load_panel, n.infer = METHODS, load_panel, infer
    old_export, old_consensus = n.public_frame, n.dev.BASE.consensus_real
    extra_columns = h.FEATURES + ('physical_phase_match', 'log_physical_phase_match')
    def export(frame, z, weights, method):
        out = old_export(frame, z, weights, method)
        for name in extra_columns:
            out[name] = frame[name].to_numpy()
        return out
    def consensus(items, method):
        out = old_consensus(items, method)
        extra = pd.concat(items).groupby('pair_key')[list(extra_columns)].mean().add_suffix('_mean').reset_index()
        return out.merge(extra, on='pair_key', validate='one_to_one')
    n.public_frame, n.dev.BASE.consensus_real = export, consensus
    if a.stage in ('freeze', 'calibrate'):
        globals()[a.stage]()
    else:
        n.run(ROOT, a.stage)
