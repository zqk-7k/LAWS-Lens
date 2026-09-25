#!/usr/bin/env python3
"""Independent source/noise confirmation of the frozen PATH25 candidate."""
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
import sys
import time
import numpy as np
import pandas as pd
from scipy.special import softmax
from scipy.spatial import cKDTree
import torch
P = Path('/root/autodl-tmp/gw-catalog')
sys.path.insert(0, str(P/'scripts/experiments'))
import mcwf_temporal_response_evaluate_20260908 as ev
import mcwf_conservative_fusion_path_20260908 as pathmodel
import mcwf_ordered_fresh_confirmation_20260907 as archived
import mcwf_conditional_intrinsics_20260908 as intrinsic
import mcwf_multirate_features_20260908 as low
import mcwf_multirate_train_20260908 as mult
t, dev, cf = ev.t, ev.dev, ev.cf
old, fresh, e = archived.old, archived.fresh, archived.e
ordered, masscal = archived.ordered, archived.calibration
CATALOGS = (202609941, 202609942, 202609943)
N_NOISE = 16
LOW = P/'results/mcwf_multirate_lowband_exploratory_20260908T145551Z'
INTRINSIC = P/'results/mcwf_conditional_intrinsics_exploratory_20260908T154330Z'
JOINT = P/'results/mcwf_joint_intrinsic_evidence_exploratory_20260908T155830Z'
ORIGINAL_EXCLUSIONS = fresh.exclusions
torch.set_num_threads(2)


def freeze(root, development):
    if root.exists():
        raise RuntimeError('Fresh independent output required')
    chosen = json.loads((development/'contracts/SELECTED_GLOBAL_DEVELOPMENT.json').read_text())
    if not chosen['both_real_target'] or not chosen['reused_test_guard']:
        raise RuntimeError('No qualified development candidate')
    if chosen['upstream_method'] != 'REFIT-JOINT-ETA-CHI-BC-ADD-CANDIDATE-POSITIVE-CANDIDATE':
        raise RuntimeError('This portable adapter is specific to the frozen winning mechanism')
    for seed in CATALOGS:
        if list(P.glob(f'results/**/catalog_{seed}')):
            raise RuntimeError('Confirmation seed already used')
    for name in ('contracts', 'scripts', 'logs', 'cache', 'calibration', 'confirmation', 'tables', 'reports', 'figures', 'manifest'):
        (root/name).mkdir(parents=True)
    upstream = Path(chosen['upstream'])
    configs = [c for c in json.loads((upstream/'calibration/SELECTED.json').read_text()) if c['method'] == chosen['upstream_method']]
    inner = json.loads((JOINT/'calibration/SELECTED.json').read_text())
    recipes, files = [], [development/'contracts/SELECTED_GLOBAL_DEVELOPMENT.json',
                         development/'contracts/START_FREEZE.json', upstream/'calibration/SELECTED.json',
                         JOINT/'calibration/SELECTED.json', Path(__file__)]
    omc = json.loads((t.PREVIOUS/'ordered_mass_confirmation/contracts/FRESH_FREEZE.json').read_text())
    for c in configs:
        dep, seed = c['deployment'], c['seed']
        recipe = next(v for v in inner if v['method'] == c['waveform_model'] and v['deployment'] == dep and v['seed'] == seed)
        slot = recipe['slot']
        w0 = cf.frozen_weights(dep, seed);w0 /= w0.sum()
        w1 = np.asarray(c['weights'], float);w1 /= w1.sum()
        effective = (1-chosen['alpha'])*w0+chosen['alpha']*w1
        recipes.append({'deployment': dep, 'seed': seed, 'slot': slot, 'alpha': chosen['alpha'],
                        'old_normalized_weights': w0.tolist(), 'upstream_weights': w1.tolist(),
                        'effective_weights': effective.tolist(), 'joint_config': recipe,
                        'OMC_config': omc['configs'][dep][str(seed)]})
        files += [LOW/f'models/MULTIRATE/{dep}/seed_{slot}/selected.pt',
                  INTRINSIC/f'models/CONDITIONAL-ETA-CHI/{dep}/seed_{slot}/selected.pt',
                  t.PREVIOUS/f'ordered_mass_predictor/models/{dep}/seed_{slot}/validation_selected_model.pt']
    files += [Path(m.__file__) for m in (old, fresh, archived, intrinsic, low, mult, ordered, masscal, pathmodel)]
    rows = t.protected()+[{'path': str(p), 'sha256': dev.sha(p), 'bytes': p.stat().st_size} for p in set(files)]
    dev.csv_write(root/'manifest/INPUT_SHA256.csv', pd.DataFrame(rows).drop_duplicates('path'))
    dev.json_write(root/'contracts/FRESH_FREEZE.json', {
        'id': 'MCWF-PATH25-INDEPENDENT-CONFIRMATION-26', 'UTC': datetime.now(timezone.utc).isoformat(),
        'status': t.STATUS, 'goal_achieved': False, 'development_root': str(development), 'selected': chosen,
        'recipes': recipes, 'catalog_seeds': CATALOGS,
        'per_catalog': {'sources_smooth': 35, 'sources_subhalo': 35, 'background_sources': 50, 'events': 190, 'true_pairs': 70, 'noise_blocks_256s': 16},
        'generation': 'Same archived physical IMRPhenomXPHM H1L1 source/PSD-optimal per-image SNR scaling,independent noise;new additional20-80Hz16s branch fromsame unfiltered paddedstrain.Originalfull24 bit-exact replay.',
        'source_proposal': 'Frozen coverage-oriented Mc5-200/SNR8-40,not astrophysical rates;GW-LMC lens environments may recur,not independent lens-population validation.',
        'sky': 'Same conditional BAYESTAR with knownintrinsics/localPSD and independent Gaussian matched-filter measurements,Nside512;not PE from exact nonGaussian injectedstrain. OriginalNESTED probabilitymass convention explicitlyrecorded.',
        'time': 'Unchanged oldlookup and archivedresponsecalendar. O3 uses O3 noise with historicalcumulativeO1-O3 responsecalendar;no claim of anewfullyO3-matched population.',
        'noise': '48new256s blocks/run;actualGPS disjoint fromoldtraining,development andallpreviousconfirmation with16s guard.',
        'guards': {'R10_drop_max': .02, 'AP_drop_max': .005, 'F50_F90_ratio_max': 1.1},
        'decision': 'Forboth waveform/fusion,eachmodel meanover3newcatalogs mustpassinbothruns. Allcatalogmetricsandsource/noiseCIsreported. No tuningafteropening.',
        'bootstrap': {'system_retrieval': 10000, 'source_pair': 2000, 'noise_cluster': 2000},
        'baseline': 'ExactretainedOMC recomputedonsamefreshdata. Never comparetwoindependentcatalogrealizations.',
        'real_interpretation': 'Independent injectionguardconfirmationonly. RealPE/officialimprovementremainsadaptive development,notindependent confirmationor detection.',
        'no_new_training': True, 'no_change_time_sky': True, 'outer_weights_changed': True,
        'upstream_selection': 'Originalinnerandouterparameters simulationsselected;globalpathalpha/method realadaptive selected andfullydisclosed.'})
    shutil.copy2(__file__, root/'scripts/path_fresh_confirmation.py')
    for dep in t.DEPS:
        rows = [ORIGINAL_EXCLUSIONS(dep)]
        for p in P.glob(f'results/mcwf*/**/{dep}/noise/noise_manifest.csv'):
            if root in p.parents:
                continue
            f = pd.read_csv(p)
            a = 'start_gps' if 'start_gps' in f else 'reference_start_gps'
            b = 'end_gps' if 'end_gps' in f else 'reference_end_gps'
            rows.append(pd.DataFrame({'start': f[a], 'end': f[b], 'origin': str(p)}))
        excluded = pd.concat(rows, ignore_index=True).drop_duplicates(['start', 'end'])
        dev.csv_write(root/f'contracts/{dep}_EXCLUDED_NOISE_FROZEN.csv', excluded)
    # Copy only reusable spectral operators; their upstream roots remain read-only.
    pairs = [(e.TRAINED/'cache/adaptive_psd/unwhitened_aligned_template_spectra.npy', root/'cache/adaptive_psd/unwhitened_aligned_template_spectra.npy'),
             (t.PREVIOUS/'fine_mass_context/cache/fine_mass_spectra.npy', root/'fine_mass_context/cache/fine_mass_spectra.npy'),
             (t.PREVIOUS/'fine_mass_context/contracts/FEATURES.json', root/'fine_mass_context/contracts/FEATURES.json'),
             (LOW/'cache/lowband_aligned_spectra.npy', root/'cache/lowband_aligned_spectra.npy')]
    for a, b in pairs:
        b.parent.mkdir(parents=True, exist_ok=True);shutil.copy2(a, b)
    dev.json_write(root/'contracts/START_FREEZE.json', {'contract_sha256': dev.sha(root/'contracts/FRESH_FREEZE.json'),
        'code_sha256': dev.sha(Path(__file__)), 'noise_exclusion_sha256': {dep: dev.sha(root/f'contracts/{dep}_EXCLUDED_NOISE_FROZEN.csv') for dep in t.DEPS}})


def verify(root):
    receipt = json.loads((root/'contracts/START_FREEZE.json').read_text())
    if dev.sha(root/'contracts/FRESH_FREEZE.json') != receipt['contract_sha256'] or dev.sha(Path(__file__)) != receipt['code_sha256']:
        raise RuntimeError('Frozen confirmation code/contract changed')
    for dep, sha in receipt['noise_exclusion_sha256'].items():
        if dev.sha(root/f'contracts/{dep}_EXCLUDED_NOISE_FROZEN.csv') != sha:
            raise RuntimeError('Noise exclusions changed')
    for row in pd.read_csv(root/'manifest/INPUT_SHA256.csv').to_dict('records'):
        if dev.sha(Path(row['path'])) != row['sha256']:
            raise RuntimeError('Protected input changed:'+row['path'])
    return json.loads((root/'contracts/FRESH_FREEZE.json').read_text())


def init_worker(root, dep, cs):
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=1);torch.set_num_threads(1)
    old.CATALOG_SEEDS = CATALOGS;old.NOISE_PER_CATALOG = N_NOISE
    old.worker_init(root, dep, cs)


def generate_system(record):
    events = old.generate_system(record)
    ctx = old._CTX;src, v3 = ctx['source'], ctx['v3']
    out = ctx['root']/f"confirmation/{ctx['dep']}/catalog_{ctx['catalog_seed']}/events"
    for event in events:
        path = out/f"{event['event_uid']}_low16s.npy"
        if path.exists():
            if 'low16s_sha256' not in event:
                event['low16s_sha256'] = dev.sha(path)
            continue
        image, bi, offset = event['image'], event['noise_bank_index'], event['noise_offset_samples']
        number = 1 if image == 'a' else 2
        clean, _ = src.detector_response(ctx['generator'], ctx['ifos'], src.source_parameters(pd.Series(record), event['gps_obs']),
                                        src.lens_factor(record[f'mu_image{number}'], record[f'morse_image{number}']))
        scaled, _, _ = v3.scale_to_network_snr(np.asarray(clean, np.float32), event['target_network_snr'], ctx['freq'], ctx['psds'][bi])
        noise = np.asarray(ctx['refs'][bi, :, offset:offset+v3.RAW_PADDED_SAMPLES], np.float32)
        raw = noise+v3.embed_signal_in_padded_window(scaled)
        full = v3.preprocess_24s(raw, ctx['freq'], ctx['psds'][bi]).astype(np.float32)
        if not np.array_equal(full, np.load(out/f"{event['event_uid']}_full24.npy")):
            raise RuntimeError('Newlowbranch failed exactoriginalinputreplay')
        np.save(path, low.low_view(raw, ctx['freq'], ctx['psds'][bi], v3))
        event['low16s_sha256'], event['old_full24_replay_exact'] = dev.sha(path), True
    dev.json_write(out/f"{record['source_uid']}.json", events)
    return events


def generate(root, dep, workers):
    verify(root)
    fresh.CATALOGS = CATALOGS;fresh.N_NOISE = N_NOISE
    fresh.exclusions = lambda d: pd.read_csv(root/f'contracts/{d}_EXCLUDED_NOISE_FROZEN.csv')
    old.CATALOG_SEEDS = CATALOGS;old.NOISE_PER_CATALOG = N_NOISE
    fresh.noise(root, dep)
    for cs in CATALOGS:
        out = root/f'confirmation/{dep}/catalog_{cs}'
        if (out/'GENERATION_COMPLETE.json').exists():
            continue
        plan = old.plan_catalog(root, dep, cs);events = [];start = time.perf_counter()
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn'),
                                 initializer=init_worker, initargs=(str(root), dep, cs)) as pool:
            for k, result in enumerate(pool.map(generate_system, plan.to_dict('records'))):
                events += result
                if k % 20 == 0:
                    print(json.dumps({'generation': dep, 'catalog': cs, 'sources': k+1, 'seconds': time.perf_counter()-start}), flush=True)
        f = pd.DataFrame(events);f.insert(0, 'idx', np.arange(len(f)))
        if len(f) != 190 or len(plan) != 120:
            raise RuntimeError('Unexpectedcatalogsize')
        f.to_parquet(out/'event_manifest.parquet', index=False)
        dev.json_write(out/'GENERATION_COMPLETE.json', {'events': len(f), 'sources': len(plan),
            'lowbranch_replay_all_exact': bool(f.old_full24_replay_exact.all()), 'manifest_sha256': dev.sha(out/'event_manifest.parquet'), 'seconds': time.perf_counter()-start})


def independence(root):
    verify(root)
    import mcwf_fresh_independence_audit_20260907 as previous
    fresh.verify = verify;fresh.CATALOGS = CATALOGS
    fresh.exclusions = lambda dep: pd.read_csv(root/f'contracts/{dep}_EXCLUDED_NOISE_FROZEN.csv')
    previous.audit(root)
    a = json.loads((root/'confirmation/GLOBAL_SOURCE_NOISE_INDEPENDENCE.json').read_text())
    paths = []
    for parent in (e.BASE, t.PREVIOUS):
        paths += list(parent.rglob('source_systems.parquet'))
        paths += list(parent.glob('expanded_data/*/*source*plan*.parquet'))
        paths += list(parent.glob('additional_population/expanded_data/*/*source*plan*.parquet'))
    sources, ids, manifest = [], set(), []
    for p in sorted(set(paths)):
        f = pd.read_parquet(p)
        if not {'m1_det', 'm2_det', 'source_uid'}.issubset(f.columns):
            raise RuntimeError('Unknownsourceidentityschema:'+str(p))
        sources.append(f[['m1_det', 'm2_det']].to_numpy(float));ids.update(f.source_uid)
        manifest.append({'path': str(p), 'sha256': dev.sha(p), 'rows': len(f)})
    if sum(map(len, sources)) < 25000:
        raise RuntimeError('Trainingpopulationinventoryincomplete')
    current = pd.concat([pd.read_parquet(root/f'confirmation/{dep}/catalog_{cs}/source_systems.parquet') for dep in t.DEPS for cs in CATALOGS])
    distance, _ = cKDTree(np.concatenate(sources)).query(current[['m1_det', 'm2_det']], p=np.inf)
    conflicts = int((distance < 1e-10).sum());uid = len(ids & set(current.source_uid))
    a.update(extension_parent_rows_checked=sum(map(len, sources)), extension_mass_matches=conflicts,
             extension_UID_matches=uid, source_noise_independence_pass=bool(a['source_noise_independence_pass'] and not conflicts and not uid))
    dev.csv_write(root/'confirmation/EXTENDED_SOURCE_INPUT_MANIFEST.csv', pd.DataFrame(manifest))
    dev.json_write(root/'confirmation/GLOBAL_SOURCE_NOISE_INDEPENDENCE_COMPLETE.json', a)
    if not a['source_noise_independence_pass']:
        raise RuntimeError('Freshsource/noiseconflict')


def joint_prediction(x, dep, slot):
    ck = torch.load(INTRINSIC/f'models/CONDITIONAL-ETA-CHI/{dep}/seed_{slot}/selected.pt', map_location='cpu', weights_only=False)
    cp = torch.load(ck['mass_checkpoint'], map_location='cpu', weights_only=False)
    if dev.sha(Path(ck['mass_checkpoint'])) != ck['mass_checkpoint_sha256']:
        raise RuntimeError('Changedmassbackbone')
    model = mult.Predictor('MULTIRATE').cuda().eval();model.load_state_dict(cp['model'])
    pm, outside = ordered.probability(ordered.infer(model, (x-cp['mu'])/cp['sd']), cp['temperature']);del model
    rep = intrinsic.original.representations(x, cp)
    head = intrinsic.Head(ck['conditional_dimensions']).cuda().eval();head.load_state_dict(ck['model'])
    residual = intrinsic.original.infer(head, ((rep-ck['mu'])/ck['sd']).reshape(-1, 103)).reshape(len(x), 253, -1)
    conditional = softmax(residual/ck['temperature']+np.log(ck['conditional_prior'].clip(1e-12))[None], -1)
    hi = np.searchsorted(ordered.LOG_CENTERS, ordered.CENTERS, side='right').clip(1, 252);lo = hi-1
    w = ((ordered.CENTERS-ordered.LOG_CENTERS[lo])/(ordered.LOG_CENTERS[hi]-ordered.LOG_CENTERS[lo])).clip(0, 1)
    conditional = (1-w[None, :, None])*conditional[:, lo]+w[None, :, None]*conditional[:, hi]
    joint = (pm[:, :, None]*conditional).astype(np.float32)
    if abs(joint.sum(-1)-pm).max() > 1e-6:
        raise RuntimeError('Jointmarginalnonconservation')
    return {'p': pm, 'joint': joint, 'outside': outside}


def unit(root):
    verify(root);rows = []
    for dep in t.DEPS:
        for slot, seed in zip(t.MODEL_SLOTS, t.SEEDS):
            # Match original batch shapes so this tests the adapter, not GEMM rounding changes.
            x = np.concatenate([t.old_features(dep, seed, 'validation'), np.load(LOW/f'features/{dep}/{seed}_validation.npy')], 1)
            current = joint_prediction(x, dep, slot)
            olda = np.load(INTRINSIC/f'predictions/CONDITIONAL-ETA-CHI/{dep}/{slot}_{seed}_validation.npz')
            errors = {k: float(abs(current[k]-olda[k]).max()) for k in ('p', 'joint', 'outside')}
            if max(errors.values()) > 2e-6:
                raise RuntimeError('Portablefrozenmodelreplayfailed:'+str(errors))
            rows.append({'deployment': dep, 'slot': slot, 'seed': seed, **errors})
    dev.csv_write(root/'tables/PORTABLE_MODEL_REPLAY.csv', pd.DataFrame(rows))
    dev.json_write(root/'contracts/PORTABLE_MODEL_REPLAY_PASS.json', {'pass': True, 'maximum_tolerance': 2e-6})


def joint_increment(a, frame, spec):
    i, j = frame.idx_i.to_numpy(int), frame.idx_j.to_numpy(int)
    z = torch.sqrt(torch.as_tensor(a['joint'], device='cuda', dtype=torch.float64)).flatten(1)
    bc = (z@z.T).cpu().numpy()[i, j].clip(1e-15, 1)
    value = np.log(bc);cal = spec['calibration']
    endpoint = (a['outside'][i] > .25) | (a['outside'][j] > .25)
    pp = ev.tail.tail_probability(-np.log(bc), cal['joint_reference'])
    outside = endpoint | (value < cal['minimum']) | (value > cal['maximum'])
    penalty = np.minimum(np.log(pp/.05), 0.)
    increment = np.interp(value, cal['knots'], cal['loglr'])
    increment = np.where(outside | (pp < .05), np.minimum(increment, 0.), increment).clip(-4, 4)
    penalty, increment = np.where(endpoint, 0., penalty), np.where(endpoint, 0., increment)
    return penalty, increment, {'joint_BC': bc, 'joint_logBC': value, 'joint_tail': pp, 'joint_ood': outside}


def score(root, dep):
    conf = verify(root)
    if not json.loads((root/'confirmation/GLOBAL_SOURCE_NOISE_INDEPENDENCE_COMPLETE.json').read_text())['source_noise_independence_pass']:
        raise RuntimeError('Independenceauditrequired')
    if not (root/'contracts/PORTABLE_MODEL_REPLAY_PASS.json').exists():
        raise RuntimeError('Portableadapterunverified')
    _, _, _, sky = old.modules()
    skyconfig = json.loads((dev.BAY/'contracts/selected_config.json').read_text())
    timecal = json.loads((dev.V7/dep/'shared/time_delay_likelihood_ratio.json').read_text())
    noise = root/f'confirmation/{dep}/noise'
    freq, psds = np.load(noise/'frequency.npy'), np.load(noise/'psd.npy', mmap_mode='r')
    rows = []
    for cs in CATALOGS:
        out = root/f'confirmation/{dep}/catalog_{cs}'
        events = pd.read_parquet(out/'event_manifest.parquet')
        i, j = np.triu_indices(len(events), 1);y = events.source_uid.to_numpy()[i] == events.source_uid.to_numpy()[j]
        dt = abs(events.gps_obs.to_numpy()[i]-events.gps_obs.to_numpy()[j])/86400
        frame = pd.DataFrame({'idx_i': i, 'idx_j': j, 'is_true_pair': y,
            'true_pair_family': np.where(y, events.family_slot.to_numpy()[i], 'NONE'), 'event_count': len(events),
            'delta_t_days': dt, 'time_score': dev.BASE.v7.apply_time_likelihood_ratio(dt, timecal)})
        full = np.stack([np.load(out/f'events/{uid}_full24.npy') for uid in events.event_uid])
        rawlow = np.stack([np.load(out/f'events/{uid}_low16s.npy') for uid in events.event_uid])
        maps = np.stack([np.load(out/f'events/{uid}_sky512.npy') for uid in events.event_uid])
        skys = {};indices = events.noise_bank_index.to_numpy(int)
        lowfeatures = low.grouped(root, rawlow, freq, psds, indices, out/'low_event_features.npy')
        refined = archived.fine.grouped(root, np.asarray(full[..., -4096:]), freq, psds, indices, out/'fine_event_features.npy')
        for recipe in [c for c in conf['recipes'] if c['deployment'] == dep]:
            slot, seed = recipe['slot'], recipe['seed'];dest = out/f'model_{slot}';dest.mkdir(exist_ok=True)
            if (dest/'COMPLETE.json').exists():
                rows += pd.read_csv(dest/'metrics.csv').to_dict('records');continue
            temp = float(skyconfig['deployments'][dep][str(seed)]['posterior_temperature'])
            if temp not in skys:
                tmp = np.stack([sky.apply_temperature(p, temp).astype(np.float32) for p in maps])
                zs, bc, audit = sky.gpu_pair_features(tmp);del tmp
                skys[temp] = zs, bc;dev.json_write(out/f'SKY_T{temp:g}_AUDIT.json', audit)
            f = frame.copy();f['sky_raw_log_bf'], f['sky_bc'] = skys[temp]
            f = old.old_waveform(dep, seed, full, f);f['previous_waveform_score'] = f.waveform_score
            cp = e.TRAINED/f'models/RAW-PHASE-SOURCE/{dep}/seed_{slot}/validation_selected_model.pt'
            p, emb, _ = archived.feature.encode_context(root, cp, full, freq, psds, indices, out/'coarse_event_features.npy')
            values = archived.ev.features(f, p, emb)
            for name, key in (('new_encoder_cosine', 'cosine'), ('new_mass_similarity', 'mass'), ('new_mass_predictive_BC', 'predictive_BC'), ('new_mass_pred_logmc_i', 'mean_i'), ('new_mass_pred_logmc_j', 'mean_j')):
                f[name] = values[key]
            baselinespec = json.loads((e.BASE/f'calibration/{dep}/seed_{seed}/SELECTED_CONFIG.json').read_text())
            f['waveform_score'] = archived.finite.score(f, baselinespec)[0]
            f['FRT_baseline_waveform_score'] = f.waveform_score
            x = ordered.arrange(np.load(out/'coarse_event_features.npy'), refined)
            ck = torch.load(t.PREVIOUS/f'ordered_mass_predictor/models/{dep}/seed_{slot}/validation_selected_model.pt', map_location='cpu', weights_only=False)
            model = ordered.Predictor().cuda().eval();model.load_state_dict(ck['model'])
            pp, boundary = ordered.probability(ordered.infer(model, (x-ck['mu'])/ck['sd']), ck['temperature']);del model
            xx = masscal.pair_features({'p': pp, 'outside': boundary}, i, j, ck['prior'])
            baseline, penalty, increment = masscal.score(f, xx, recipe['OMC_config'], 'PRIOR')
            f['mass_penalty'], f['mass_increment'], f['waveform_score'] = penalty, increment, baseline
            f['retained_OMC_waveform'] = baseline
            a = joint_prediction(np.concatenate([x, lowfeatures], 1), dep, slot)
            np.savez_compressed(dest/'joint_predictions.npz', **a)
            pen, inc, audit = joint_increment(a, f, recipe['joint_config'])
            joint = recipe['joint_config'];new = baseline if joint.get('unchanged_OMC', False) else baseline+joint['gamma']*pen+joint['beta']*inc
            z, weights = pathmodel.mix(f, new, np.asarray(recipe['upstream_weights']), dep, seed, recipe['alpha'])
            c = f.copy();c['waveform_score'] = z;c['joint_penalty'], c['joint_increment'] = pen, inc
            for key, value in audit.items():c[key] = value
            f['final_score'] = cf.channels(f, baseline)@cf.frozen_weights(dep, seed)
            c['final_score'] = cf.channels(c, z)@weights
            for col in ('time_score', 'sky_raw_log_bf'):
                if not np.array_equal(c[col], f[col]):raise RuntimeError('Frozenphysicalchannelchanged')
            f.to_parquet(dest/'OMC_pairs.parquet', index=False);c.to_parquet(dest/'PATH25_pairs.parquet', index=False)
            metrics = []
            for name, pair in (('OMC', f), ('PATH25', c)):
                for mode, scores in (('waveform', pair.waveform_score.to_numpy(float)), ('fusion', pair.final_score.to_numpy(float))):
                    m = dev.BASE.full_metrics(pair, scores)
                    metrics.append({'configuration': name, 'deployment': dep, 'catalog_seed': cs, 'model_seed': slot,
                        'eval_seed': seed, 'mode': mode, **m})
            dev.csv_write(dest/'metrics.csv', pd.DataFrame(metrics));rows += metrics
            dev.json_write(dest/'COMPLETE.json', {'baseline_sha256': dev.sha(dest/'OMC_pairs.parquet'),
                'candidate_sha256': dev.sha(dest/'PATH25_pairs.parquet'), 'time_sky_exact': True, 'effective_weights': weights.tolist()})
            print(json.dumps({'scored': dep, 'catalog': cs, 'model': slot}), flush=True)
        del full, maps, rawlow
    dev.csv_write(root/f'tables/{dep}_FRESH_METRICS.csv', pd.DataFrame(rows))


def assess(root):
    verify(root)
    f = pd.concat([pd.read_csv(root/f'tables/{dep}_FRESH_METRICS.csv') for dep in t.DEPS])
    columns = ['macro_r_at_1', 'macro_r_at_10', 'average_precision', 'false_at_recall_0p5', 'false_at_recall_0p9']
    mean = f.groupby(['configuration', 'deployment', 'model_seed', 'mode'])[columns].mean().reset_index()
    rows = []
    for (dep, slot, mode), block in mean.groupby(['deployment', 'model_seed', 'mode']):
        b = block[block.configuration == 'OMC'].iloc[0];c = block[block.configuration == 'PATH25'].iloc[0]
        rows.append({'deployment': dep, 'model_seed': int(slot), 'mode': mode, 'pass': bool(cf.guard(c, b)),
                     **{'delta_'+k: float(c[k]-b[k]) for k in columns}})
    dev.csv_write(root/'tables/FRESH_METRICS_PER_CATALOG_MODEL.csv', f)
    dev.csv_write(root/'tables/FRESH_MODEL_MEAN_METRICS.csv', mean)
    dev.csv_write(root/'tables/FRESH_GUARDRAILS.csv', pd.DataFrame(rows))
    dev.json_write(root/'contracts/FRESH_ASSESSMENT.json', {'fresh_guard_pass': all(r['pass'] for r in rows),
        'tested_run_model_modes': len(rows), 'goal_achieved': False, 'status': t.STATUS,
        'packaging_and_bootstrap_required': True, 'real_results_remain_adaptive_development': True})
    print(json.dumps({'fresh_guard_pass': all(r['pass'] for r in rows), 'tested_run_model_modes': len(rows)}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser();p.add_argument('--root', type=Path, required=True)
    p.add_argument('--development', type=Path);p.add_argument('--deployment', choices=t.DEPS)
    p.add_argument('--workers', type=int, default=12)
    p.add_argument('--stage', choices=['freeze', 'unit', 'generate', 'independence', 'score', 'assess'], required=True)
    a = p.parse_args()
    if a.stage == 'freeze':freeze(a.root, a.development)
    elif a.stage == 'generate':generate(a.root, a.deployment, a.workers)
    elif a.stage == 'score':score(a.root, a.deployment)
    else:globals()[a.stage](a.root)
