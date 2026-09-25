#!/usr/bin/env python3
"""Recursive identity audit and clustered uncertainty for frozen PATH25."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
P = Path('/root/autodl-tmp/gw-catalog');sys.path.insert(0, str(P/'scripts/experiments'))
import mcwf_path_fresh_confirmation_v2_20260908 as run
import mcwf_fresh_independence_audit_20260907 as previous
import mcwf_summarize_20260905 as bootstrap
from mcwf_confirmation_audit_20260906 import noise_cluster_ci
dev, t = run.dev, run.t


def freeze(root):
    path = root/'contracts/AUDIT_IMPLEMENTATION_FREEZE.json'
    record = {'UTC': datetime.now(timezone.utc).isoformat(), 'code_sha256': dev.sha(Path(__file__)),
        'reason': 'The generator runner nonrecursive source-plan glob omits expanded_data/run/train/source_plan.parquet. Use recursive inventory before any freshscoring. No change todata,models,coefficients,guards orbootstrap counts.',
        'inventory': 'Include source_systems.parquet and all recursive source_plan.parquet fromretainedtraining/development/confirmation roots,plus originalhistoricalaudit.',
        'bootstrap': {'system_retrieval': 10000, 'source_pair': 2000, 'noise_cluster': 2000},
        'source_and_noise_bootstrap_are_separate': True}
    if path.exists():
        if json.loads(path.read_text())['code_sha256'] != record['code_sha256']:
            raise RuntimeError('Frozen audit code changed')
    else:
        dev.json_write(path, record)


def independence(root):
    run.verify(root);freeze(root)
    run.fresh.verify = run.verify;run.fresh.CATALOGS = run.CATALOGS
    run.fresh.exclusions = lambda dep: pd.read_csv(root/f'contracts/{dep}_EXCLUDED_NOISE_FROZEN.csv')
    previous.audit(root)
    a = json.loads((root/'confirmation/GLOBAL_SOURCE_NOISE_INDEPENDENCE.json').read_text())
    paths = []
    for parent in (run.e.BASE, t.PREVIOUS):
        paths += list(parent.rglob('source_systems.parquet'))
        paths += list(parent.rglob('source_plan.parquet'))
    sources, ids, manifest = [], set(), []
    for p in sorted(set(paths)):
        f = pd.read_parquet(p)
        if not {'m1_det', 'm2_det', 'source_uid'}.issubset(f.columns):
            raise RuntimeError('Unknownsourceidentityschema:'+str(p))
        sources.append(f[['m1_det', 'm2_det']].to_numpy(float));ids.update(f.source_uid)
        manifest.append({'path': str(p), 'sha256': dev.sha(p), 'rows': len(f)})
    if sum(map(len, sources)) < 25000:
        raise RuntimeError('Trainingpopulationinventoryincomplete')
    current = pd.concat([pd.read_parquet(root/f'confirmation/{dep}/catalog_{cs}/source_systems.parquet') for dep in t.DEPS for cs in run.CATALOGS])
    distance, _ = cKDTree(np.concatenate(sources)).query(current[['m1_det', 'm2_det']], p=np.inf)
    conflicts = int((distance < 1e-10).sum());uid = len(ids & set(current.source_uid))
    a.update(extension_parent_rows_checked=sum(map(len, sources)), extension_mass_matches=conflicts,
             extension_UID_matches=uid, extension_minimum_mass_distance_Msun=float(distance.min()),
             source_noise_independence_pass=bool(a['source_noise_independence_pass'] and not conflicts and not uid))
    dev.csv_write(root/'confirmation/EXTENDED_SOURCE_INPUT_MANIFEST.csv', pd.DataFrame(manifest))
    dev.json_write(root/'confirmation/GLOBAL_SOURCE_NOISE_INDEPENDENCE_COMPLETE.json', a)
    windows = []
    for dep in t.DEPS:
        noise = pd.read_csv(root/f'confirmation/{dep}/noise/noise_manifest.csv').set_index('bank_index')
        for cs in run.CATALOGS:
            events = pd.read_parquet(root/f'confirmation/{dep}/catalog_{cs}/event_manifest.parquet')
            intervals = pd.DataFrame({'uid': events.source_uid, 'start': noise.loc[events.noise_bank_index, 'start_gps'].to_numpy()+events.noise_offset_samples/4096})
            overlapping = 0;total = 0
            for _, group in intervals.groupby('uid'):
                if len(group) == 2:
                    total += 1;overlapping += abs(float(group.start.iloc[0]-group.start.iloc[1])) < 26
            windows.append({'deployment': dep, 'catalog_seed': cs, 'true_systems': total,
                            'same_system_noise_windows_overlap': int(overlapping),
                            'interpretation': 'Training/testblocksaredisjoint;withinacatalog noisywindowscanreuseblocks,separateblockbootstraprequired.'})
    dev.csv_write(root/'tables/WITHIN_CATALOG_NOISE_WINDOW_AUDIT.csv', pd.DataFrame(windows))
    if not a['source_noise_independence_pass']:
        raise RuntimeError('Freshsource/noiseconflict')
    print(json.dumps(a), flush=True)


def task(spec):
    root, dep, cs, slot, mode = spec;root = Path(root)
    cat = root/f'confirmation/{dep}/catalog_{cs}';out = cat/f'model_{slot}'
    dest = out/f'{mode}_uncertainty.json'
    if dest.exists():return json.loads(dest.read_text())
    events = pd.read_parquet(cat/'event_manifest.parquet')
    plan = pd.DataFrame({'system_id': events.source_uid, 'family': events.family_slot})
    b = pd.read_parquet(out/'OMC_pairs.parquet');c = pd.read_parquet(out/'PATH25_pairs.parquet')
    for col in ('idx_i', 'idx_j', 'is_true_pair', 'true_pair_family', 'time_score', 'sky_raw_log_bf'):
        if not np.array_equal(b[col], c[col]):raise RuntimeError('Changedpairedinputs:'+col)
    bs = b.waveform_score.to_numpy(float) if mode == 'waveform' else b.final_score.to_numpy(float)
    cscores = c.waveform_score.to_numpy(float) if mode == 'waveform' else c.final_score.to_numpy(float)
    bm, cm = dev.BASE.full_metrics(b, bs), dev.BASE.full_metrics(c, cscores)
    ci, ranks = bootstrap.ranks_bootstrap(c, cscores, bs, cs+slot, repeats=10000)
    ranks.to_parquet(out/f'{mode}_query_ranks.parquet', index=False)
    row = {'deployment': dep, 'catalog_seed': cs, 'model_seed': slot, 'mode': mode, **cm, **ci}
    for key in ('macro_r_at_1', 'macro_r_at_10', 'average_precision', 'false_at_recall_0p5', 'false_at_recall_0p9'):
        row['baseline_'+key], row['delta_'+key] = bm[key], cm[key]-bm[key]
    row['point_guardrail_pass'] = bool(run.cf.guard(cm, bm))
    row.update(bootstrap.pair_bootstrap(c, cscores, bs, plan, cs+slot, repeats=2000))
    row.update(noise_cluster_ci(c, cscores, bs, events, cs+slot, repeats=2000))
    nfalse = int((~c.is_true_pair.to_numpy(bool)).sum())
    row['legacy_minimum1_FP_actual_false_rate'] = max(1, int(np.floor(1e-5*nfalse)))/nfalse
    dev.json_write(dest, row);return row


def uncertainty(root):
    run.verify(root);freeze(root)
    if not json.loads((root/'confirmation/GLOBAL_SOURCE_NOISE_INDEPENDENCE_COMPLETE.json').read_text())['source_noise_independence_pass']:
        raise RuntimeError('Independenceauditrequired')
    specs = [(str(root), dep, cs, slot, mode) for dep in t.DEPS for cs in run.CATALOGS for slot in t.MODEL_SLOTS for mode in ('waveform', 'fusion')]
    rows = []
    with ProcessPoolExecutor(max_workers=12) as pool:
        for row in pool.map(task, specs):
            rows.append(row);dev.csv_write(root/'tables/FRESH_PAIRED_METRICS_AND_CI.csv', pd.DataFrame(rows))
            print(json.dumps({'uncertainty_completed': len(rows), 'total': len(specs)}), flush=True)
    dev.json_write(root/'contracts/BOOTSTRAP_COMPLETE.json', {'completed': len(rows),
        'source_retrieval_repeats': 10000, 'source_pair_repeats': 2000, 'noise_cluster_repeats': 2000,
        'all_configurations_unchanged': True, 'real_outcomes_not_independently_validated': True})


if __name__ == '__main__':
    p = argparse.ArgumentParser();p.add_argument('--root', type=Path, required=True)
    p.add_argument('--stage', choices=['freeze', 'independence', 'uncertainty'], required=True)
    a = p.parse_args();globals()[a.stage](a.root)
