#!/usr/bin/env python3
"""Frozen ordered-mass increment versus current FRT on new source/noise catalogs."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import time
import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree
import mcwf_pe_frontend_extension_v2_20260907 as e
import mcwf_ordered_mass_predictor_20260907 as ordered
import mcwf_ordered_mass_evaluate_20260907 as calibration
import mcwf_fine_mass_features_20260907 as fine
import mcwf_new_confirmation_20260906 as fresh
import mcwf_fresh_confirmation_20260906 as old
import mcwf_finite_reference_tail_20260907 as finite
import mcwf_finelag_eventpsd_20260907 as feature

dev, body, ev = e.dev, e.body, e.ev
CATALOGS = (202609401, 202609402, 202609403)
N_NOISE = 16
TRIAL = 'ORDERED-MASS-COMPLETE-GRID-PRIOR'
ORIGINAL_EXCLUSIONS = fresh.exclusions


def folder(root):
    return root / 'ordered_mass_confirmation'


def verify(root):
    path = folder(root) / 'contracts/FRESH_FREEZE.json'
    conf = json.loads(path.read_text())
    for row in conf['frozen_files']:
        if dev.sha(Path(row['path'])) != row['sha256']:
            raise RuntimeError('Frozen input changed: ' + row['path'])
    return conf


def freeze(root):
    out = folder(root)
    if (out / 'contracts/FRESH_FREEZE.json').exists():
        return verify(root)
    if out.exists():
        raise RuntimeError('Unfrozen output exists')
    trial = root / 'trials' / TRIAL
    gate = json.loads((trial / 'contracts/TARGET_AUDIT.json').read_text())
    reused = pd.read_csv(trial / 'tables/REUSED_GUARDRAILS.csv')
    if not reused['pass'].all():
        raise RuntimeError('Reused injection guardrails failed')
    configs, files = {}, []
    for dep in e.DEPS:
        path = trial / f'contracts/{dep}_SELECTED.json'
        selection = json.loads(path.read_text())
        if not selection['development_outcome']['target_pass']:
            raise RuntimeError('Development targets failed')
        configs[dep] = selection['configs']
        files.append(path)
        for ms, es in zip(body.MODEL_SEEDS, dev.SEEDS):
            files += [root / f'ordered_mass_predictor/models/{dep}/seed_{ms}/validation_selected_model.pt',
                      root / f'ordered_mass_predictor/calibration/{dep}/seed_{ms}/CALIBRATION.json',
                      e.TRAINED / f'models/RAW-PHASE-SOURCE/{dep}/seed_{ms}/validation_selected_model.pt',
                      e.BASE / f'calibration/{dep}/seed_{es}/SELECTED_CONFIG.json',
                      dev.V7 / dep / f'seed_{es}/results/waveform_channel_calibration_v7.json',
                      dev.V7 / dep / f'seed_{es}/waveform_gate/unified_inception_attention_peak2s_4096_aux_0p25_q_1p0_v7/validation_selected_model.pt']
        files += [dev.V7 / dep / 'shared/time_delay_likelihood_ratio.json',
                  dev.ORCH.SOURCE_ROOT / dep / 'shared/h1l1_live_schedule.csv',
                  root / f'audit/{dep}_frozen_external_reference.parquet']
    files += [Path(__file__), Path(ordered.__file__), Path(calibration.__file__), Path(fine.__file__),
              Path(fresh.__file__), Path(old.__file__), Path(feature.__file__), Path(finite.__file__),
              Path(ev.__file__), Path(body.__file__), Path(dev.__file__),
              e.TRAINED / 'cache/adaptive_psd/unwhitened_aligned_template_spectra.npy',
              root / 'fine_mass_context/cache/fine_mass_spectra.npy',
              dev.BAY / 'contracts/selected_config.json', root / 'contracts/EXPERIMENT_CONTRACT.json',
              trial / 'contracts/RECIPE.json', trial / 'contracts/TARGET_AUDIT.json']
    for name in ('contracts', 'scripts', 'confirmation'):
        (out / name).mkdir(parents=True, exist_ok=False)
    config = {'created_utc': datetime.now(timezone.utc).isoformat(),
              'code': 'MCWF-UNIFIED-OMC-DEVCONF', 'selected_trial': str(trial),
              'baseline': str(e.BASE), 'same_method_both_runs': True,
              'formula': 'Z_wf_FRT + gamma * ordered_mass_finite_tail + beta * ordered_mass_calibrated_prior_overlap; unchanged time,sky,outerweights',
              'configs': configs, 'arm': 'PRIOR', 'development_target_audit': gate,
              'selection_disclosure': 'Explicitly adaptive real PE/official development plus reused simulation guards. NOT blind validation. No PE,official labels or event IDs are score inputs.',
              'zero_coefficients': 'Optional ADDITIONAL increment can be zero for a seed, exactly as development grid allowed. The previous FRT neural model remains active for all six seeds. Do not claim six active new increments.',
              'catalog_seeds': CATALOGS, 'per_catalog': {'sources_smooth': 35, 'sources_subhalo': 35, 'background_sources': 50, 'events': 190, 'true_pairs': 70, 'noise_blocks_256s': N_NOISE},
              'freshness': 'New BBH masses,spins,orientations,sky,phase,GPS; noise disjoint from all historical and extension training/development/previous confirmation references with16s guard',
              'lens_environments': 'Repeated GW-LMC environments allowed; not an independent lens-population test',
              'proposal': 'Frozen balanced Mc5-200Msun/SNR8-40 coverage proposal; independent per-image PSD-optimal target scaling, NOT response-derived intensity ratios or astrophysical rates',
              'waveform': 'Unchanged physical IMRPhenomXPHM,H1L1,24s4096Hz generation; same40-580Hz whitening,antialias,peak2s4096 input',
              'sky': 'Unchanged conditional BAYESTAR with known intrinsic parameters/localPSD and independent Gaussian matched-filter measurements; NOT PE of exact non-Gaussian injected strain;Nside512',
              'O3_calendar': 'Frozen cumulative O1-O3 response calendar with O3-only noise; not newly claimed fully O3-response-matched sampling',
              'guardrails': {'R10_drop_max': .02, 'AP_drop_max': .005, 'F50_F90_ratio_max': 1.1},
              'decision': 'Each model and method mean over3newcatalogs in each run must meet ALL point guardrails. All catalog results and source/noise confidence intervals reported. No tuning after opening.',
              'bootstrap': {'system_retrieval': 10000, 'source_pair': 2000, 'noise_cluster': 2000},
              'inference_limit': 'Fresh confirmation tests simulation retrieval guardrails, not independent replication of real PE/official overlap improvements',
              'protected': ['old outputs', 'old ranks', 'paper', 'ET3', 'time', 'sky', 'outer weights'],
              'frozen_files': [{'path': str(p), 'sha256': dev.sha(p)} for p in sorted(set(files))]}
    dev.json_write(out / 'contracts/FRESH_FREEZE.json', config)
    shutil.copy2(__file__, out / 'scripts' / Path(__file__).name)
    print(json.dumps({'freeze': str(out), 'sha256': dev.sha(out / 'contracts/FRESH_FREEZE.json')}), flush=True)
    return config


def exclusions(root, dep):
    frames = [ORIGINAL_EXCLUSIONS(dep)]
    paths = [e.BASE / f'confirmation/{dep}/noise/noise_manifest.csv',
             root / f'expanded_data/{dep}/noise/noise_manifest.csv',
             root / f'additional_population/expanded_data/{dep}/noise/noise_manifest.csv']
    for path in paths:
        f = pd.read_csv(path)
        a = 'start_gps' if 'start_gps' in f else 'reference_start_gps'
        b = 'end_gps' if 'end_gps' in f else 'reference_end_gps'
        frames.append(pd.DataFrame({'start': f[a], 'end': f[b], 'origin': str(path)}))
    return pd.concat(frames, ignore_index=True).drop_duplicates(['start', 'end'])


def init_worker(root, dep, cs):
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=1)
    torch.set_num_threads(1)
    old.CATALOG_SEEDS = CATALOGS
    old.NOISE_PER_CATALOG = N_NOISE
    old.worker_init(str(folder(Path(root))), dep, cs)


def generate(root, dep, workers):
    verify(root)
    fresh.CATALOGS = CATALOGS
    fresh.N_NOISE = N_NOISE
    fresh.exclusions = lambda d: exclusions(root, d)
    old.CATALOG_SEEDS = CATALOGS
    old.NOISE_PER_CATALOG = N_NOISE
    fresh.noise(folder(root), dep)
    for cs in CATALOGS:
        out = folder(root) / f'confirmation/{dep}/catalog_{cs}'
        if (out / 'GENERATION_COMPLETE.json').exists():
            continue
        plan = old.plan_catalog(folder(root), dep, cs)
        events = []
        started = time.perf_counter()
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn'),
                                 initializer=init_worker, initargs=(str(root), dep, cs)) as pool:
            for k, result in enumerate(pool.map(old.generate_system, plan.to_dict('records'))):
                events.extend(result)
                if k % 20 == 0:
                    print(json.dumps({'generation': dep, 'catalog': cs, 'sources': k + 1, 'seconds': time.perf_counter() - started}), flush=True)
        f = pd.DataFrame(events)
        f.insert(0, 'idx', np.arange(len(f)))
        if len(f) != 190 or len(plan) != 120:
            raise RuntimeError('Unexpected catalog size')
        f.to_parquet(out / 'event_manifest.parquet', index=False)
        dev.json_write(out / 'GENERATION_COMPLETE.json', {'events': len(f), 'sources': len(plan), 'seconds': time.perf_counter() - started, 'event_sha256': dev.sha(out / 'event_manifest.parquet')})


def independence(root):
    verify(root)
    import mcwf_fresh_independence_audit_20260907 as previous
    fresh.verify = lambda unused: verify(root)
    fresh.CATALOGS = CATALOGS
    fresh.exclusions = lambda dep: exclusions(root, dep)
    previous.audit(folder(root))
    base = folder(root) / 'confirmation'
    prior_report = json.loads((base / 'GLOBAL_SOURCE_NOISE_INDEPENDENCE.json').read_text())
    paths = []
    for dep in e.DEPS:
        paths += list((e.BASE / f'confirmation/{dep}').glob('catalog_*/source_systems.parquet'))
        for bank in (root / f'expanded_data/{dep}', root / f'additional_population/expanded_data/{dep}'):
            paths += list(bank.rglob('source_systems.parquet'))
            paths += list(bank.glob('*source*plan*.parquet'))
    oldm, uid, manifest = [], set(), []
    for path in sorted(set(paths)):
        f = pd.read_parquet(path)
        if 'm1_det' not in f or 'm2_det' not in f:
            raise RuntimeError('Unknown parent mass schema: ' + str(path))
        oldm.append(f[['m1_det', 'm2_det']].to_numpy(float))
        uid.update(f.source_uid.astype(str))
        manifest.append({'path': str(path), 'sha256': dev.sha(path), 'rows': len(f)})
    if sum(len(v) for v in oldm) < 25000:
        raise RuntimeError('Expanded training source inventory incomplete')
    current = pd.concat([pd.read_parquet(base / f'{dep}/catalog_{cs}/source_systems.parquet') for dep in e.DEPS for cs in CATALOGS], ignore_index=True)
    d, _ = cKDTree(np.concatenate(oldm)).query(current[['m1_det', 'm2_det']], p=np.inf)
    matches = int((d < 1e-10).sum())
    uid_matches = len(uid & set(current.source_uid))
    result = {**prior_report, 'extension_parent_rows_checked': sum(len(v) for v in oldm),
              'extension_mass_matches': matches, 'extension_UID_matches': uid_matches,
              'extension_minimum_mass_distance_Msun': float(d.min()),
              'source_noise_independence_pass': bool(prior_report['source_noise_independence_pass'] and not matches and not uid_matches)}
    dev.json_write(base / 'GLOBAL_SOURCE_NOISE_INDEPENDENCE_COMPLETE.json', result)
    dev.csv_write(base / 'EXTENSION_SOURCE_INVENTORY.csv', pd.DataFrame(manifest))
    if not result['source_noise_independence_pass']:
        raise RuntimeError('Fresh independence failed')
    print(json.dumps(result), flush=True)


def score(root, dep):
    conf = verify(root)
    base = folder(root) / 'confirmation'
    if not json.loads((base / 'GLOBAL_SOURCE_NOISE_INDEPENDENCE_COMPLETE.json').read_text())['source_noise_independence_pass']:
        raise RuntimeError('Source/noise audit must pass before scoring')
    _, _, _, sky = old.modules()
    skyconfig = json.loads((dev.BAY / 'contracts/selected_config.json').read_text())
    timecal = json.loads((dev.V7 / dep / 'shared/time_delay_likelihood_ratio.json').read_text())
    noise = base / f'{dep}/noise'
    freq, psds = np.load(noise / 'frequency.npy'), np.load(noise / 'psd.npy', mmap_mode='r')
    rows = []
    for cs in CATALOGS:
        out = base / f'{dep}/catalog_{cs}'
        events = pd.read_parquet(out / 'event_manifest.parquet')
        i, j = np.triu_indices(len(events), 1)
        y = events.source_uid.to_numpy()[i] == events.source_uid.to_numpy()[j]
        dt = abs(events.gps_obs.to_numpy()[i] - events.gps_obs.to_numpy()[j]) / 86400
        frame = pd.DataFrame({'idx_i': i, 'idx_j': j, 'is_true_pair': y,
                             'true_pair_family': np.where(y, events.family_slot.to_numpy()[i], 'NONE'),
                             'event_count': len(events), 'delta_t_days': dt,
                             'time_score': dev.BASE.v7.apply_time_likelihood_ratio(dt, timecal)})
        full = np.stack([np.load(out / f'events/{uid}_full24.npy') for uid in events.event_uid])
        maps = np.stack([np.load(out / f'events/{uid}_sky512.npy') for uid in events.event_uid])
        skycache = {}
        for ms, es in zip(body.MODEL_SEEDS, dev.SEEDS):
            saved = out / f'model_{ms}'
            saved.mkdir(exist_ok=True)
            if (saved / 'COMPLETE.json').exists():
                rows.extend(pd.read_csv(saved / 'metrics.csv').to_dict('records'))
                continue
            t = float(skyconfig['deployments'][dep][str(es)]['posterior_temperature'])
            if t not in skycache:
                tmp = np.stack([sky.apply_temperature(p, t).astype(np.float32) for p in maps])
                zs, bc, audit = sky.gpu_pair_features(tmp)
                del tmp
                skycache[t] = zs, bc
                dev.json_write(out / f'SKY_T{t:g}_AUDIT.json', audit)
            f = frame.copy()
            f['sky_raw_log_bf'], f['sky_bc'] = skycache[t]
            f = old.old_waveform(dep, es, full, f)
            f['previous_waveform_score'] = f.waveform_score
            cp = e.TRAINED / f'models/RAW-PHASE-SOURCE/{dep}/seed_{ms}/validation_selected_model.pt'
            p, z, _ = feature.encode_context(e.TRAINED, cp, full, freq, psds, events.noise_bank_index.to_numpy(int), out / 'coarse_event_PSD_features.npy')
            values = ev.features(f, p, z)
            for name, key in (('new_encoder_cosine', 'cosine'), ('new_mass_similarity', 'mass'), ('new_mass_predictive_BC', 'predictive_BC'), ('new_mass_pred_logmc_i', 'mean_i'), ('new_mass_pred_logmc_j', 'mean_j')):
                f[name] = values[key]
            baselinespec = json.loads((e.BASE / f'calibration/{dep}/seed_{es}/SELECTED_CONFIG.json').read_text())
            f['waveform_score'] = finite.score(f, baselinespec)[0]
            refined = fine.grouped(root, np.asarray(full[..., -4096:]), freq, psds,
                                   events.noise_bank_index.to_numpy(int), out / 'fine_event_PSD_features.npy')
            x = ordered.arrange(np.load(out / 'coarse_event_PSD_features.npy'), refined)
            ocp = root / f'ordered_mass_predictor/models/{dep}/seed_{ms}/validation_selected_model.pt'
            ck = torch.load(ocp, weights_only=False, map_location='cpu')
            model = ordered.Predictor().cuda().eval()
            model.load_state_dict(ck['model'])
            pp, boundary = ordered.probability(ordered.infer(model, (x - ck['mu']) / ck['sd']), ck['temperature'])
            del model
            np.savez_compressed(saved / 'event_predictions.npz', baseline_p=p, baseline_z=z, ordered_p=pp,
                                ordered_boundary_mass=boundary, ordered_checkpoint_sha256=dev.sha(ocp))
            xx = calibration.pair_features({'p': pp, 'outside': boundary}, i, j, ck['prior'])
            spec = conf['configs'][dep][str(es)]
            candidate, penalty, inc = calibration.score(f, xx, spec, 'PRIOR')
            c = f.copy()
            c['waveform_score'] = candidate
            c['ordered_mass_tail_penalty'], c['ordered_mass_calibrated_evidence'] = penalty, inc
            for key, value in xx.items():
                c['ordered_mass_' + key] = value
            c['ordered_gamma'], c['ordered_beta'] = spec['gamma'], spec['beta']
            c['ordered_pred_logmc_i'], c['ordered_pred_logmc_j'] = (pp @ ordered.CENTERS)[i], (pp @ ordered.CENTERS)[j]
            if spec['gamma'] == 0 and spec['beta'] == 0 and not np.array_equal(candidate, f.waveform_score):
                raise RuntimeError('Inactive optional increment changed baseline')
            f.to_parquet(saved / 'BASELINE_pairs.parquet', index=False)
            c.to_parquet(saved / 'CANDIDATE_pairs.parquet', index=False)
            metrics = []
            for name, fr in [('FRT_BASELINE', f), ('ORDERED_CANDIDATE', c)]:
                for method, mm in ev.metrics(fr, fr.waveform_score.to_numpy(float), dep, es).items():
                    metrics.append({'deployment': dep, 'catalog_seed': cs, 'model_seed': ms, 'eval_seed': es,
                                    'config': name, 'method': method, **mm})
            dev.csv_write(saved / 'metrics.csv', pd.DataFrame(metrics))
            dev.json_write(saved / 'COMPLETE.json', {'ordered_gamma': spec['gamma'], 'ordered_beta': spec['beta'],
                           'changed_pair_fraction': float((candidate != f.waveform_score).mean()),
                           'baseline_sha256': dev.sha(saved / 'BASELINE_pairs.parquet'),
                           'candidate_sha256': dev.sha(saved / 'CANDIDATE_pairs.parquet')})
            rows.extend(metrics)
            print(json.dumps({'scored': dep, 'catalog': cs, 'model': ms, 'gamma': spec['gamma'], 'beta': spec['beta']}), flush=True)
        del maps, full
    dev.csv_write(base / f'{dep}/METRICS_PER_CATALOG_MODEL.csv', pd.DataFrame(rows))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--task', choices=['freeze', 'generate', 'independence', 'score'], required=True)
    p.add_argument('--deployment', choices=e.DEPS)
    p.add_argument('--workers', type=int, default=6)
    a = p.parse_args()
    if a.task == 'freeze': freeze(a.root)
    elif a.task == 'generate': generate(a.root, a.deployment, a.workers)
    elif a.task == 'independence': independence(a.root)
    else: score(a.root, a.deployment)
