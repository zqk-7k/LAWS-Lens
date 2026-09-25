#!/usr/bin/env python3
"""Full parent inventory and unchanged fresh guardrails for optional mass increment."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import mcwf_ordered_fresh_confirmation_20260907 as run
import mcwf_fresh_independence_audit_20260907 as previous
import mcwf_summarize_20260905 as bootstrap
from mcwf_confirmation_audit_20260906 import noise_cluster_ci

dev, e, body = run.dev, run.e, run.body


def freeze_audit(root):
    out = run.folder(root) / 'contracts/AUDIT_IMPLEMENTATION_FREEZE.json'
    record = {'utc': datetime.now(timezone.utc).isoformat(),
              'script': str(Path(__file__)), 'sha256': dev.sha(Path(__file__)),
              'change': 'Read nested train/validation/source_plan.parquet in complete extension parent inventory; original runner glob was nonrecursive and would stop with incomplete inventory. No model/gate/data-generation change.',
              'guardrails': 'Exactly FRESH_FREEZE.json; optional increment zero is allowed and reported, never counted as a nonzero new-model effect',
              'methods': ['waveform_only', 'C_fixed'], 'bootstrap': [10000, 2000, 2000]}
    if out.exists():
        old = json.loads(out.read_text())
        if old['sha256'] != record['sha256']:
            raise RuntimeError('Frozen audit implementation changed')
    else:
        dev.json_write(out, record)


def independence(root):
    run.verify(root)
    freeze_audit(root)
    run.fresh.verify = lambda unused: run.verify(root)
    run.fresh.CATALOGS = run.CATALOGS
    run.fresh.exclusions = lambda dep: run.exclusions(root, dep)
    previous.audit(run.folder(root))
    base = run.folder(root) / 'confirmation'
    old_report = json.loads((base / 'GLOBAL_SOURCE_NOISE_INDEPENDENCE.json').read_text())
    paths = []
    for dep in e.DEPS:
        paths += list((e.BASE / f'confirmation/{dep}').glob('catalog_*/source_systems.parquet'))
        for bank in (root / f'expanded_data/{dep}', root / f'additional_population/expanded_data/{dep}'):
            paths += list(bank.rglob('source_plan.parquet'))
    oldm, uid, manifest = [], set(), []
    for path in sorted(set(paths)):
        f = pd.read_parquet(path)
        oldm.append(f[['m1_det', 'm2_det']].to_numpy(float))
        uid.update(f.source_uid.astype(str))
        manifest.append({'path': str(path), 'sha256': dev.sha(path), 'rows': len(f)})
    if sum(len(v) for v in oldm) < 25000:
        raise RuntimeError('Incomplete expanded parent inventory')
    src = pd.concat([pd.read_parquet(base / f'{dep}/catalog_{cs}/source_systems.parquet') for dep in e.DEPS for cs in run.CATALOGS], ignore_index=True)
    d, _ = cKDTree(np.concatenate(oldm)).query(src[['m1_det', 'm2_det']], p=np.inf)
    matches, uid_matches = int((d < 1e-10).sum()), len(uid & set(src.source_uid))
    result = {**old_report, 'extension_parent_rows_checked': sum(len(v) for v in oldm),
              'extension_mass_matches': matches, 'extension_UID_matches': uid_matches,
              'extension_minimum_mass_distance_Msun': float(d.min()),
              'source_noise_independence_pass': bool(old_report['source_noise_independence_pass'] and not matches and not uid_matches)}
    dev.csv_write(base / 'EXTENSION_SOURCE_INVENTORY.csv', pd.DataFrame(manifest))
    dev.json_write(base / 'GLOBAL_SOURCE_NOISE_INDEPENDENCE_COMPLETE.json', result)
    if not result['source_noise_independence_pass']:
        raise RuntimeError('Independent confirmation input check failed')
    print(json.dumps(result), flush=True)


def audit_deployment(root, dep):
    conf = run.verify(root)
    freeze_audit(root)
    base = run.folder(root) / 'confirmation'
    if not json.loads((base / 'GLOBAL_SOURCE_NOISE_INDEPENDENCE_COMPLETE.json').read_text())['source_noise_independence_pass']:
        raise RuntimeError('Missing independent-source/noise audit')
    rows, invariant = [], []
    for cs in run.CATALOGS:
        cat = base / f'{dep}/catalog_{cs}'
        events = pd.read_parquet(cat / 'event_manifest.parquet')
        plan = pd.DataFrame({'system_id': events.source_uid, 'family': events.family_slot})
        for ms, es in zip(body.MODEL_SEEDS, dev.SEEDS):
            out = cat / f'model_{ms}'
            bframe = pd.read_parquet(out / 'BASELINE_pairs.parquet')
            cframe = pd.read_parquet(out / 'CANDIDATE_pairs.parquet')
            for col in ('idx_i', 'idx_j', 'is_true_pair', 'true_pair_family', 'time_score', 'sky_raw_log_bf'):
                if not np.array_equal(bframe[col], cframe[col]):
                    raise RuntimeError('Frozen pair input changed: ' + col)
            spec = conf['configs'][dep][str(es)]
            inactive = spec['gamma'] == 0 and spec['beta'] == 0
            change = float((bframe.waveform_score != cframe.waveform_score).mean())
            if inactive and change != 0:
                raise RuntimeError('Disabled optional increment changed scores')
            invariant.append({'deployment': dep, 'catalog_seed': cs, 'model_seed': ms,
                              'gamma': spec['gamma'], 'beta': spec['beta'], 'optional_increment_inactive': inactive,
                              'changed_waveform_pair_fraction': change, 'time_sky_labels_exact': True})
            for method in ('waveform_only', 'C_fixed'):
                path = out / f'{method}_uncertainty.json'
                if path.exists():
                    rows.append(json.loads(path.read_text()))
                    continue
                weight = {'waveform': 1., 'time': 0., 'sky': 0.} if method == 'waveform_only' else dev.BASE.FROZEN_V93_WEIGHTS[dep][es]
                b, c = dev.BASE.score_vector(bframe, weight), dev.BASE.score_vector(cframe, weight)
                bm, cm = dev.BASE.full_metrics(bframe, b), dev.BASE.full_metrics(cframe, c)
                ci, ranks = bootstrap.ranks_bootstrap(cframe, c, b, cs + ms, repeats=10000)
                ranks.to_parquet(out / f'{method}_query_ranks.parquet', index=False)
                row = {'deployment': dep, 'catalog_seed': cs, 'model_seed': ms, 'eval_seed': es,
                       'method': method, 'optional_increment_inactive': inactive, **cm, **ci}
                for key in ('macro_r_at_1', 'macro_r_at_10', 'average_precision', 'false_at_recall_0p5', 'false_at_recall_0p9'):
                    row['baseline_' + key], row['delta_' + key] = bm[key], cm[key] - bm[key]
                row['point_guardrail_pass'] = bool(cm['macro_r_at_10'] >= bm['macro_r_at_10'] - .02 and
                    cm['average_precision'] >= bm['average_precision'] - .005 and
                    cm['false_at_recall_0p5'] <= 1.1 * bm['false_at_recall_0p5'] and
                    cm['false_at_recall_0p9'] <= 1.1 * bm['false_at_recall_0p9'])
                row.update(bootstrap.pair_bootstrap(cframe, c, b, plan, cs + ms, repeats=2000))
                row.update(noise_cluster_ci(cframe, c, b, events, cs + ms, repeats=2000))
                nfalse = int((~cframe.is_true_pair.to_numpy(bool)).sum())
                row['legacy_minimum_one_FP_budget_actual_rate'] = max(1, int(np.floor(1e-5 * nfalse))) / nfalse
                dev.json_write(path, row)
                rows.append(row)
            print(json.dumps({'fresh_audited': dep, 'catalog': cs, 'model': ms}), flush=True)
    dev.csv_write(base / f'{dep}/PAIRED_METRICS_AND_CI.csv', pd.DataFrame(rows))
    dev.csv_write(base / f'{dep}/INVARIANCE.csv', pd.DataFrame(invariant))


def summarize(root):
    conf = run.verify(root)
    base = run.folder(root) / 'confirmation'
    f = pd.concat([pd.read_csv(base / f'{dep}/PAIRED_METRICS_AND_CI.csv') for dep in e.DEPS], ignore_index=True)
    cols = [c for c in f if c.startswith(('macro_', 'average_', 'false_', 'baseline_', 'delta_')) and 'ci_' not in c and pd.api.types.is_numeric_dtype(f[c])]
    means = f.groupby(['deployment', 'model_seed', 'method'])[cols].mean().reset_index()
    means['point_guardrail_pass'] = (means.delta_macro_r_at_10.ge(-.02) & means.delta_average_precision.ge(-.005) &
        means.false_at_recall_0p5.le(1.1 * means.baseline_false_at_recall_0p5) &
        means.false_at_recall_0p9.le(1.1 * means.baseline_false_at_recall_0p9))
    summary = means.groupby(['deployment', 'method'])[cols].agg(['mean', 'std']).reset_index()
    summary.columns = ['_'.join(filter(None, c)) if isinstance(c, tuple) else c for c in summary.columns]
    dev.csv_write(base / 'PAIRED_METRICS_AND_CI.csv', f)
    dev.csv_write(base / 'MODEL_MEAN_OVER_CATALOGS.csv', means)
    dev.csv_write(base / 'SUMMARY.csv', summary)
    passed = bool(means.point_guardrail_pass.all())
    report = {'created_utc': datetime.now(timezone.utc).isoformat(), 'code': conf['code'],
              'status': 'PASS_CONDITIONAL_NEW_SOURCE_NOISE_TEST' if passed else 'FAIL_CONDITIONAL_NEW_SOURCE_NOISE_TEST',
              'per_model_mean_across3catalog_guardrails_pass': passed,
              'catalog_model_method_pass_count': int(f.point_guardrail_pass.sum()), 'catalog_model_method_total': len(f),
              'model_checks': means[['deployment', 'model_seed', 'method', 'point_guardrail_pass']].to_dict('records'),
              'fresh_BBH_sources': 720, 'fresh_events': 1140, 'fresh_true_pairs': 420, 'independent_noise_blocks': 96,
              'optional_new_increment_active_seed_count': sum(c['gamma'] != 0 or c['beta'] != 0 for dep in conf['configs'].values() for c in dep.values()),
              'prior_FRT_model_active_all_six_seeds': True, 'real_PE_official_blind_validation': False,
              'limitations': ['Source/noise conditional test, not independent lens population',
                  'Source and noise bootstrap are separate clustered diagnostics, not joint population-coverage proof',
                  'Conditional Gaussian-measurement BAYESTAR, not PE of exact injected non-Gaussian strain',
                  'Per-image SNR scaling and balanced coverage proposal retained',
                  'Legacy1e-5 field enforces minimum1FP; actual false rate reported',
                  'Fresh simulation guardrails do not independently confirm adaptive real candidate gains'],
              'final_status': 'HOLD_FOR_AUTHOR_REVIEW_NO_ADOPTION_NO_OVERWRITE'}
    dev.json_write(base / 'FINAL_GUARDRAIL_AUDIT.json', report)
    print(json.dumps(report), flush=True)
    print(summary.to_json(orient='records'), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--task', choices=['freeze', 'independence', 'audit', 'summarize'], required=True)
    p.add_argument('--deployment', choices=e.DEPS)
    a = p.parse_args()
    if a.task == 'freeze': freeze_audit(a.root)
    elif a.task == 'independence': independence(a.root)
    elif a.task == 'audit': audit_deployment(a.root, a.deployment)
    else: summarize(a.root)
