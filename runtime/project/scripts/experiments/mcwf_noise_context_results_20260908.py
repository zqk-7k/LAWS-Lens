#!/usr/bin/env python3
"""Complete, independently replayed deliverable for noise and TF experiments."""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import numpy as np
import pandas as pd
import mcwf_noise_context_20260908 as n
import mcwf_spectral_residual_20260908 as tf
import mcwf_omc_extension_results_20260907 as writer
import mcwf_summarize_20260905 as bootstrap

e, dev = n.e, n.dev


def setup(work):
    (tf.configure if work.name.startswith('TF-') else n.configure)()
    return work.name, work


def materialize(root):
    writer.setup = setup
    writer.scored = lambda f, x, s, a: n.score(f, x, s, a)[0]
    writer.materialize(root)
    r = pd.read_csv(root / 'tables/ALL_DIAGNOSTIC_RETRIEVAL.csv')
    keys = ['experiment', 'deployment', 'split', 'method', 'config']
    metrics = [c for c in r if c not in keys + ['seed'] and pd.api.types.is_numeric_dtype(r[c])]
    out = r.groupby(keys)[metrics].agg(['mean', 'std']).reset_index()
    out.columns = ['_'.join(c).rstrip('_') for c in out.columns]
    dev.csv_write(root / 'tables/RETRIEVAL_MEAN_SD.csv', out)


def independent(frame, x, spec, arm):
    old = frame.waveform_score.to_numpy(float)
    if spec.get('unchanged_baseline'): return old.copy()
    reference = np.asarray(spec['mass_reference'])
    distance = -np.log(x['bc'])
    p = (1 + len(reference) - np.searchsorted(reference, distance, side='left')) / (len(reference) + 1)
    penalty = np.minimum(np.log(p / .05), 0.)
    name = 'bc' if arm.endswith('-BC') else 'prior_overlap'
    feat, cal = x[name], spec[name]
    inc = np.interp(feat, cal['knots'], cal['loglr']).clip(-4, 4)
    unsupported = (feat < cal['minimum']) | (feat > cal['maximum']) | (p < .05) | x['ood']
    inc = np.where(unsupported & (inc > 0), 0., inc)
    origin = old if arm.startswith('ADD-') else frame.FRT_baseline_waveform_score.to_numpy(float)
    result = origin + spec['gamma'] * penalty + spec['beta'] * inc
    result[x['ood']] = old[x['ood']]
    return result


def cached(work, dep, es, split):
    f = pd.read_parquet(e.TRIAL / f'evaluation/{dep}/seed_{es}/{split}_pairs.parquet')
    p = work / f'cache/pair_features/{dep}_{es}_{split}.npz'
    if not p.exists(): raise RuntimeError('Missing frozen pair features: ' + str(p))
    return f, dict(np.load(p))


def counterexamples(root):
    budgets, metrics, choices = [], [], []
    for marker in sorted(root.glob('variants/*/trials/*/contracts/SEARCH_COMPLETE.json')):
        trial, work = marker.parent.parent, marker.parents[3]
        code = str(trial.relative_to(root)).replace('/trials/', ':')
        dest = trial / 'PE_leading_not_adopted'
        if (dest / 'COMPLETE.json').exists():
            budgets.append(pd.read_csv(dest / 'BUDGETS.csv'))
            metrics.append(pd.read_csv(dest / 'RETRIEVAL.csv'))
            choices += json.loads((dest / 'CHOICES.json').read_text())
            continue
        dest.mkdir(exist_ok=False)
        combinations = pd.read_csv(trial / 'tables/ALL_COMBINATIONS.csv')
        bs, ms, selections = [], [], []
        for dep in e.old.DEPS:
            winner = combinations[combinations.deployment == dep].sort_values(
                ['BC_gain', 'median_gain', 'front_gain', 'hanabi_gain', 'combination'],
                ascending=[False, False, False, False, True]).iloc[0]
            configs, real = {}, {}
            for k, es in enumerate(dev.SEEDS):
                pool = json.loads((trial / f'configs/{dep}_{es}_ELIGIBLE.json').read_text())['eligible']
                spec = next(a['spec'] for a in pool if a['id'] == winner[f'seed{k+1}'])
                configs[str(es)] = spec
                for split in ('validation', 'test', 'real'):
                    f, x = cached(work, dep, es, split)
                    score = independent(f, x, spec, trial.name)
                    assert np.max(abs(score - n.score(f, x, spec, trial.name)[0])) < 1e-10
                    c = f.drop(columns=[name for name in ('rank', 'final_score', 'method', 'waveform_contribution',
                        'time_contribution', 'sky_contribution') if name in f]).copy()
                    c['OMC_baseline_waveform_score'], c['waveform_score'] = f.waveform_score, score
                    if split == 'real': real[es] = c
                    else:
                        cm, bm = e.ev.metrics(f, score, dep, es), e.ev.metrics(f, f.waveform_score.to_numpy(float), dep, es)
                        assert e.ev.guard(cm, bm)
                        for method, values in cm.items():
                            ms.append({'experiment': code, 'deployment': dep, 'seed': es, 'split': split,
                                       'method': method, 'config': 'PE_LEADING_NOT_ADOPTED', **values})
            dev.save_evaluation(dest, 'PE_LEADING_NOT_ADOPTED', dep, real,
                                pd.read_parquet(root / f'audit/{dep}_external_reference.parquet'))
            b = pd.read_csv(dest / f'results/PE_LEADING_NOT_ADOPTED/{dep}/pe_official_budget.csv')
            b.insert(0, 'experiment', code); b['development_target_pass'] = bool(winner.target_pass)
            bs.append(b)
            for budget in (10, 20):
                row = b[(b.seed.astype(str) == 'consensus') & (b.method == 'C_fixed') & (b.budget == budget)].iloc[0]
                for name in ('catastrophic_mc', 'BC_mc_ge_0p5', 'Dmax_le_3', 'median_BC_mc', 'official_frontend', 'official_hanabi'):
                    assert abs(row[name] - winner[f'Top{budget}_{name}']) < 1e-10
            selections.append({'experiment': code, 'deployment': dep, 'configs': configs,
                               'target_pass': bool(winner.target_pass), 'selection': 'Maximum PE improvement in simulation-admissible grid; NOT adopted,not blind'})
        b, m = pd.concat(bs, ignore_index=True), pd.DataFrame(ms)
        dev.csv_write(dest / 'BUDGETS.csv', b); dev.csv_write(dest / 'RETRIEVAL.csv', m)
        dev.json_write(dest / 'CHOICES.json', selections)
        dev.json_write(dest / 'COMPLETE.json', {'all_budgets_reproduced': True, 'independent_score_pass': True})
        budgets.append(b); metrics.append(m); choices += selections
        print(json.dumps({'counterexample_complete': code}), flush=True)
    dev.csv_write(root / 'tables/PE_LEADING_BUDGETS.csv', pd.concat(budgets, ignore_index=True))
    dev.csv_write(root / 'tables/PE_LEADING_RETRIEVAL.csv', pd.concat(metrics, ignore_index=True))
    dev.json_write(root / 'tables/PE_LEADING_CHOICES.json', choices)


def verify(root):
    checks = []
    for marker in root.glob('variants/*/trials/*/diagnostic_export/CHOICES.json'):
        dest, trial, work = marker.parent, marker.parents[1], marker.parents[3]
        for choice in json.loads(marker.read_text()):
            for es, spec in choice['configs'].items():
                for split in ('validation', 'test', 'real'):
                    f, x = cached(work, choice['deployment'], int(es), split)
                    p = dest / f'evaluation/{choice["deployment"]}/seed_{es}/{split}_pairs.parquet'
                    c = pd.read_parquet(p)
                    expected = independent(f, x, spec, trial.name)
                    error = float(abs(expected - c.waveform_score.to_numpy(float)).max())
                    columns = ['idx_i', 'idx_j', 'time_score', 'sky_raw_log_bf']
                    if split != 'real':
                        columns.append('is_true_pair')
                    frozen = all(np.array_equal(f[k], c[k]) for k in columns)
                    assert error < 1e-10 and frozen
                    checks.append({'path': str(p), 'max_absolute_score_error': error, 'time_sky_indices_labels_exact': frozen})
    dev.csv_write(root / 'audit/INDEPENDENT_SCORE_REPLAY.csv', pd.DataFrame(checks))
    historical = []
    for row in pd.read_csv(root / 'manifest/PROTECTED_INPUT_SHA256.csv').itertuples():
        actual = dev.sha(Path(row.path)); assert actual == row.sha256, row.path
        historical.append({'path': row.path, 'sha256': actual, 'unchanged': True})
    dev.csv_write(root / 'manifest/FINAL_HISTORICAL_HASH_RECHECK.csv', pd.DataFrame(historical))
    dev.json_write(root / 'audit/FINAL_SCORE_AND_HISTORY_VERIFICATION.json', {
        'all_pass': True, 'independent_score_replays': len(checks), 'historical_files_unchanged': len(historical),
        'time_sky_unchanged': True, 'O4_GLOBAL_PRIOR_preserved': True})
    warm = []
    for path in sorted((root / 'cache/TF_warm_logits').glob('*/*.npy')):
        record = json.loads(path.with_suffix('.json').read_text())
        match = re.fullmatch(r'(\d+)_(train|development|real|(\d+)_(validation|test))', path.stem)
        assert match, path
        ms, es = int(match[1]), int(match[3]) if match[3] else 0
        split = match[4] or match[2]
        cp = n.GLOBAL / f'models/{path.parent.name}/seed_{ms}/validation_selected_model.pt'
        assert record['warm_sha256'] == dev.sha(cp)
        assert record['logits_sha256'] == dev.sha(path)
        cached_logits = np.load(path)
        replay = tf.compute_warm_logits(path.parent.name, ms, es, split)
        error = float(np.max(np.abs(replay - cached_logits)))
        assert error <= 1e-4 and np.isfinite(cached_logits).all(), (path, error)
        warm.append({'path': str(path), 'sha256': record['logits_sha256'],
                     'max_absolute_recomputed_logit_error': error,
                     'bit_exact_replay': bool(np.array_equal(replay, cached_logits))})
    dev.csv_write(root / 'audit/WARM_CACHE_INDEPENDENT_REPLAY.csv', pd.DataFrame(warm))
    import torch
    dev.json_write(root / 'audit/SOFTWARE_AND_HARDWARE.json', {
        'UTC': datetime.now(timezone.utc).isoformat(), 'python': sys.version,
        'numpy': np.__version__, 'pandas': pd.__version__, 'torch': torch.__version__,
        'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(),
        'nvidia_smi': subprocess.check_output(['nvidia-smi'], text=True),
        'disk': subprocess.check_output(['df', '-h', str(root)], text=True),
        'ram': subprocess.check_output(['free', '-h'], text=True),
        'cpu': subprocess.check_output(['lscpu'], text=True)})


def failure_audit(root):
    baseline = pd.read_csv(root / 'contracts/BASELINE_BUDGETS.csv')
    baseline = baseline[(baseline.seed.astype(str) == 'consensus') & (baseline.method == 'C_fixed')]
    output = []
    for path in sorted(root.glob('variants/*/trials/*/tables/ALL_COMBINATIONS.csv')):
        frame = pd.read_csv(path)
        code = str(path.parents[1].relative_to(root)).replace('/trials/', ':')
        for dep, g in frame.groupby('deployment'):
            b = baseline[baseline.deployment == dep].set_index('budget')
            pe, public = np.ones(len(g), bool), np.ones(len(g), bool)
            for budget in (10, 20):
                pe &= (g[f'Top{budget}_catastrophic_mc'] <= b.loc[budget, 'catastrophic_mc'])
                for name in ('BC_mc_ge_0p5', 'Dmax_le_3', 'median_BC_mc'):
                    pe &= g[f'Top{budget}_{name}'] >= b.loc[budget, name] - 1e-12
                for name in ('official_frontend', 'official_hanabi'):
                    public &= g[f'Top{budget}_{name}'] >= b.loc[budget, name]
            improved_pe = pe & ((g.BC_gain > 0) | (g.median_gain > 1e-10))
            improved_public = public & (g.front_gain > 0) & (g.hanabi_gain > 0)
            output.append({'experiment': code, 'deployment': dep, 'grid_combinations': len(g),
                'C_fixed_PE_noninferior': int(pe.sum()),
                'C_fixed_PE_improved': int(improved_pe.sum()),
                'official_noninferior': int(public.sum()),
                'both_official_counts_improved': int(improved_public.sum()),
                'C_fixed_PE_and_official_improved': int((improved_pe & improved_public).sum()),
                'waveform_only_and_GLOBAL_guarded_target_pass': int(g.target_pass.sum()),
                'note': 'Descriptive decomposition of frozen gates,not new acceptance criteria;grid points are dependent'})
    dev.csv_write(root / 'tables/TARGET_FAILURE_DECOMPOSITION.csv', pd.DataFrame(output))


def event_audit(root):
    from scipy.stats import spearmanr
    rows, summary = [], []
    destination = root / 'audit/predictive_mass'
    destination.mkdir(exist_ok=False)
    for dep in e.old.DEPS:
        f = pd.read_parquet(e.TRIAL / f'evaluation/{dep}/seed_{dev.SEEDS[0]}/real_pairs.parquet')
        events = pd.concat([f[[f'idx_{s}', f'event_{s}']].rename(columns={
            f'idx_{s}': 'index', f'event_{s}': 'event'}) for s in ('i', 'j')]).drop_duplicates()
        assert events['index'].is_unique and events.event.is_unique
        ref = pd.read_parquet(root / f'audit/{dep}_external_reference.parquet')
        pe = pd.concat([ref[[f'event_{s}', f'pe_mc_median_{s}', f'pe_mc_sigma_{s}']].rename(columns={
            f'event_{s}': 'event', f'pe_mc_median_{s}': 'public_PE_Mc_median',
            f'pe_mc_sigma_{s}': 'public_PE_Mc_sigma'}) for s in ('i', 'j')]).drop_duplicates()
        assert pe.event.is_unique
        events = events.merge(pe, on='event', validate='one_to_one').sort_values('index')
        assert len(events) == (62 if dep == 'gwtc3' else 74)
        for work in sorted((root / 'variants').iterdir()):
            if not work.is_dir(): continue
            members = []
            for ms in e.body.MODEL_SEEDS:
                file = work / f'cache/predictions/{dep}/{dev.SEEDS[0]}_real_{ms}.npz'
                members.append(np.load(file)['p'][events['index'].to_numpy(int)])
            stack = np.stack(members)
            for name, p in [(str(ms), pp) for ms, pp in zip(e.body.MODEL_SEEDS, members)] + [('ensemble', stack.mean(0))]:
                assert np.isfinite(p).all() and np.allclose(p.sum(1), 1., atol=1e-5)
                cdf = np.c_[np.zeros(len(p)), p.cumsum(1)]
                quantiles = np.exp(np.array([np.interp([.05, .5, .95], c, n.mass.EDGES) for c in cdf]))
                current = events.copy()
                current['model_family'], current['deployment'], current['model_key'] = work.name, dep, name
                current[['predictive_Mc_q05', 'predictive_Mc_q50', 'predictive_Mc_q95']] = quantiles
                current['absolute_log_median_ratio'] = np.abs(np.log(current.predictive_Mc_q50 / current.public_PE_Mc_median))
                current['public_PE_median_inside_predictive90'] = (
                    (current.public_PE_Mc_median >= current.predictive_Mc_q05) &
                    (current.public_PE_Mc_median <= current.predictive_Mc_q95))
                rows.append(current)
                summary.append({'model_family': work.name, 'deployment': dep, 'model_key': name,
                    'n_events': len(current), 'mean_absolute_log_median_ratio': current.absolute_log_median_ratio.mean(),
                    'median_absolute_log_median_ratio': current.absolute_log_median_ratio.median(),
                    'PE_median_inside_predictive90_fraction': current.public_PE_median_inside_predictive90.mean(),
                    'Mc_median_spearman_descriptive': spearmanr(current.predictive_Mc_q50, current.public_PE_Mc_median).statistic,
                    'interpretation': 'PE median is not truth;this is not posterior coverage or an acceptance gate'})
            np.savez_compressed(destination / f'{work.name}_{dep}.npz',
                                event=events.event.to_numpy(str), mass_log_edges=n.mass.EDGES,
                                model_keys=np.asarray(e.body.MODEL_SEEDS),
                                predictive_probability_mass=stack.astype(np.float32))
    dev.csv_write(root / 'tables/EVENT_MASS_PREDICTIONS_VS_PUBLIC_PE.csv', pd.concat(rows, ignore_index=True))
    dev.csv_write(root / 'tables/EVENT_MASS_PREDICTION_SUMMARY.csv', pd.DataFrame(summary))


def uncertainty(root):
    output = root / 'tables/SOURCE_BOOTSTRAP_DIAGNOSTICS.csv'
    if output.exists(): return
    choices = json.loads((root / 'tables/ALL_DIAGNOSTIC_CHOICES.json').read_text())
    budgets = pd.read_csv(root / 'tables/ALL_DIAGNOSTIC_BUDGETS.csv')
    cis = []
    for dep in e.old.DEPS:
        for kind in [p.name for p in (root / 'variants').iterdir() if p.is_dir()]:
            bs = budgets[(budgets.deployment == dep) & (budgets.seed.astype(str) == 'consensus') &
                         (budgets.method == 'C_fixed') & (budgets.budget.isin([10, 20])) &
                         budgets.experiment.str.startswith('variants/' + kind + ':')]
            if not len(bs): continue
            summary = bs.groupby('experiment')[['BC_mc_ge_0p5', 'official_frontend', 'official_hanabi', 'median_BC_mc']].sum()
            code = summary.sort_values(['BC_mc_ge_0p5', 'median_BC_mc', 'official_frontend', 'official_hanabi'], ascending=False).index[0]
            selection = next(c for c in choices if c['experiment'] == code and c['deployment'] == dep)
            arm = code.split(':')[-1]; work = root / 'variants' / kind
            for es in dev.SEEDS:
                f, x = cached(work, dep, es, 'test'); spec = selection['configs'][str(es)]
                new = independent(f, x, spec, arm)
                plan = dev.BASE.retained_event_plan(dep, es, 'test')
                for method in ('waveform_only', 'C_fixed'):
                    base = f.waveform_score.to_numpy(float)
                    candidate = new.copy()
                    if method == 'C_fixed':
                        w = dev.BASE.FROZEN_V93_WEIGHTS[dep][es]
                        # Use the frozen project's score-vector API, not an inferred weight ordering.
                        frame = f.copy(); frame['waveform_score'] = candidate
                        candidate, base = dev.BASE.score_vector(frame, w), dev.BASE.score_vector(f, w)
                    rci, ranks = bootstrap.ranks_bootstrap(f, candidate, base, es + 900, repeats=10000)
                    pci = bootstrap.pair_bootstrap(f, candidate, base, plan, es + 900, repeats=1000)
                    cis.append({'experiment': code, 'deployment': dep, 'seed': es, 'method': method,
                                'selected_for': 'Descriptive per-family admissible diagnostic; selection not independent', **rci, **pci})
                    target = root / f'audit/query_ranks/{kind}_{arm}_{dep}_{es}_{method}.parquet'
                    target.parent.mkdir(parents=True, exist_ok=True); ranks.to_parquet(target, index=False)
            print(json.dumps({'bootstrap_complete': [dep, kind, code]}), flush=True)
    dev.csv_write(output, pd.DataFrame(cis))
    dev.json_write(root / 'contracts/BOOTSTRAP_INTERPRETATION.json', {
        'query': '10000 source-system stratified draws,two image queries stay together',
        'pairs': '1000 paired source-block weighted draws;not iid pairs,not new catalog simulation',
        'limitations': 'Conditional reused-test sampling uncertainty after adaptive development selection,not selection-corrected significance or independent real validation',
        'seed_SD': 'Three old deployments with same uniform new3model mixture. Not three independently retrained complete ensembles.'})


def collect(root):
    source = dev.PROJECT / 'scripts/experiments'
    dest = root / 'scripts/reproduction_dependencies'; dest.mkdir(exist_ok=True)
    queue = list(source.glob('mcwf_noise*_20260908.py')) + [source/'mcwf_spectral_residual_20260908.py', source/'mcwf_omc_extension_run_logged_20260907.py']
    seen = set()
    while queue:
        p = queue.pop()
        if p in seen or not p.exists(): continue
        seen.add(p); shutil.copy2(p, dest / p.name)
        for node in ast.walk(ast.parse(p.read_text())):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ''] if isinstance(node, ast.ImportFrom) else []
            queue += [source / (name + '.py') for name in names if name.startswith('mcwf_')]
    dev.csv_write(root / 'manifest/SCRIPT_SOURCE_SHA256.csv', pd.DataFrame([
        {'path': str(p), 'sha256': dev.sha(p)} for p in sorted(seen)]))


def report(root):
    ledger = pd.read_csv(root / 'tables/EXPERIMENT_LEDGER.csv')
    assert ledger.experiment.nunique() == 20 and len(ledger) == 40
    if ledger.both_runs_target_pass.any():
        raise RuntimeError('Both-run development candidate found. Freeze and perform fresh confirmation before finalizing.')
    models = [{'family': p.parents[3].name, **json.loads(p.read_text())}
              for p in root.glob('variants/*/models/*/*/COMPLETE.json')]
    assert len(models) == 30
    dev.csv_write(root / 'tables/TRAINING_SUMMARY.csv', pd.DataFrame(models))
    timings = [{'path': str(p.relative_to(root)), **json.loads(p.read_text())} for p in root.glob('logs/*.runtime.json')]
    dev.csv_write(root / 'tables/RUNTIME_SUMMARY.csv', pd.DataFrame(timings))
    b = pd.read_csv(root / 'tables/ALL_DIAGNOSTIC_BUDGETS.csv')
    b = b[(b.seed.astype(str) == 'consensus') & (b.method == 'C_fixed') & b.budget.isin([10, 20])]
    baseline = pd.read_csv(root / 'contracts/BASELINE_BUDGETS.csv')
    baseline = baseline[(baseline.seed.astype(str) == 'consensus') & (baseline.method == 'C_fixed') & baseline.budget.isin([10,20])]
    rec = pd.read_csv(root / 'tables/RETRIEVAL_MEAN_SD.csv')
    lines = []
    for dep in e.old.DEPS:
        for code in ledger.experiment.unique():
            m = rec[(rec.deployment == dep) & (rec.experiment == code) & (rec.split == 'test') &
                    (rec.method == 'C_fixed') & (rec.config == 'DIAGNOSTIC')].iloc[0]
            a = b[(b.deployment == dep) & (b.experiment == code)].set_index('budget')
            lines.append(f'| {dep} | {code} | {m.macro_r_at_1_mean:.4f} +/- {m.macro_r_at_1_std:.4f} | {m.macro_r_at_10_mean:.4f} +/- {m.macro_r_at_10_std:.4f} | {m.average_precision_mean:.4f} | {m.false_at_recall_0p5_mean:.1f}/{m.false_at_recall_0p9_mean:.1f} | {int(a.loc[10].BC_mc_ge_0p5)}/{int(a.loc[20].BC_mc_ge_0p5)} | {int(a.loc[10].official_frontend)}/{int(a.loc[20].official_frontend)} | {int(a.loc[10].official_hanabi)}/{int(a.loc[20].official_hanabi)} |')
    totals = ledger.groupby('deployment')[['combinations', 'target_pass_combinations']].sum()
    differences = []
    for r in b.to_dict('records'):
        old = baseline[(baseline.deployment == r['deployment']) & (baseline.budget == r['budget'])].iloc[0]
        differences.append({**r, **{'baseline_'+k: old[k] for k in ('BC_mc_ge_0p5','Dmax_le_3','catastrophic_mc','median_BC_mc','official_frontend','official_hanabi')}})
    dev.csv_write(root / 'tables/BASELINE_VS_DIAGNOSTICS.csv', pd.DataFrame(differences))
    failed = pd.read_csv(root / 'tables/PE_LEADING_BUDGETS.csv')
    failure_audit(root)
    failures_by_gate = pd.read_csv(root / 'tables/TARGET_FAILURE_DECOMPOSITION.csv')
    o3_failure = failures_by_gate[failures_by_gate.deployment == 'gwtc3']
    changed_retrieval = pd.read_csv(root / 'tables/PE_LEADING_RETRIEVAL.csv')
    changed_retrieval = changed_retrieval[(changed_retrieval.split == 'test') & (changed_retrieval.method == 'C_fixed')]
    changed_lines = []
    for dep in e.old.DEPS:
        for code in ledger.experiment.unique():
            m = changed_retrieval[(changed_retrieval.deployment == dep) & (changed_retrieval.experiment == code)]
            a = failed[(failed.deployment == dep) & (failed.experiment == code) &
                       (failed.seed.astype(str) == 'consensus') & (failed.method == 'C_fixed') &
                       failed.budget.isin([10, 20])].set_index('budget')
            assert len(m) == 3
            changed_lines.append(f'| {dep} | {code} | {m.macro_r_at_1.mean():.4f} +/- {m.macro_r_at_1.std():.4f} | {m.macro_r_at_10.mean():.4f} +/- {m.macro_r_at_10.std():.4f} | {m.average_precision.mean():.4f} | {m.false_at_recall_0p5.mean():.1f}/{m.false_at_recall_0p9.mean():.1f} | {int(a.loc[10].BC_mc_ge_0p5)}/{int(a.loc[20].BC_mc_ge_0p5)} | {int(a.loc[10].official_frontend)}/{int(a.loc[20].official_frontend)} | {int(a.loc[10].official_hanabi)}/{int(a.loc[20].official_hanabi)} |')
    zero_epochs = sum(m['selected_epoch'] == 0 for m in models)
    failures = list(root.glob('failed_attempts/*/RECOVERY.json'))
    dev.json_write(root / 'contracts/FINAL_DECISION.json', {
        'status': 'HOLD_FOR_AUTHOR_REVIEW_NO_ADOPTION_NO_OVERWRITE',
        'joint_objective_achieved': False, 'complete_training_runs': len(models),
        'epoch_zero_selected': zero_epochs, 'score_contrasts': 20,
        'fresh_confirmation_run': False, 'failed_execution_attempts_preserved': len(failures),
        'reason': 'No O3/O4a joint qualifier under frozen development targets',
        'preserve_OMC_and_O4_GLOBAL_PRIOR': True, 'time_sky_unchanged': True})
    text = f'''# 双运行期波形噪声条件化与时频残差探索报告

状态：HOLD_FOR_AUTHOR_REVIEW_NO_ADOPTION_NO_OVERWRITE

## 结论

完成 {len(models)} 个新模型运行、{ledger.experiment.nunique()} 个完整评分对照。本轮未找到同时满足 O3/O4a 的 PE 改善、官方前端和公开 Hanabi 重合增加以及注入检索保护条件的统一方法。目标未达成，不能把 O4a 的局部改善包装为双运行期成功。历史 OMC 主结果和 O4a GLOBAL-PRIOR 保留。

O3 共 {int(totals.loc['gwtc3','combinations'])} 个模拟指标允许的部署系数组合，全部目标通过 {int(totals.loc['gwtc3','target_pass_combinations'])}；O4a 对应 {int(totals.loc['gwtc4','combinations'])} 和 {int(totals.loc['gwtc4','target_pass_combinations'])}。这些共享模型、共享事件的网格点不是独立实验或显著性样本。

关键卡点不是“O3完全不能改善Mc”：O3有{int(o3_failure.C_fixed_PE_improved.sum())}个网格点改善了预定PE预算，但在保持两预算官方重合均不降低的条件下，同时增加官方前端和Hanabi重合的网格点为{int(o3_failure.both_official_counts_improved.sum())}。因此单独把Mc筛得更相容，并不自动满足官方候选重合目标；不能手工补入官方pair，或把官方重合当成透镜真值。

## 本轮实际修改

1. 首先重新检查旧审计：真实输入137事件 full24s/peak2s逐点复现，未发现新增预处理错误；旧40--580Hz频段诊断是全窗口SNR诊断，不等于实际2s输入的信息损失估计。
2. CONTROL：冻结旧GLOBAL初始化来源，继续优化30epochs，作为额外训练对照。
3. PSD：在同一GLOBAL网络中加入64维真实off-source PSD条件，经有界FiLM调制；PSD-ROBUST另加入按训练噪声块损失更新的有界样本权重。未改变频段、原始波形输入或时间天空矩阵。
4. TF-POWER/TF-COMPLEX：冻结原GLOBAL质量网络，添加原始H1/L1峰值2秒的时频残差网络。分别比较功率谱和额外复数STFT相位分量。新分支40epochs，质量logits残差限制在正负2。不是再次对同一个最大匹配标量调权。
5. 每一结构在O3/O4a各运行三个训练种子。checkpoint/temperature只根据模拟development质量预测选取，包含epoch0原初始化，避免强行使用变差checkpoint。

30个训练运行中有{zero_epochs}个选择epoch0。这表示继续训练未改善其development交叉熵，不能称为新权重带来的改善。逐模型选择的epoch、质量误差和90%预测区间覆盖见TRAINING_SUMMARY.csv。

执行中发生{len(failures)}次共享参考logits缓存写入冲突，退出记录和未完成的输出原样保存在failed_attempts/，原日志保留。随后仅增加文件锁和原子发布、使用已有checkpoint重新评估，未重训、未改变科学规则。全部共享缓存另从冻结父模型重建比对，见WARM_CACHE_INDEPENDENT_REPLAY.csv。

## 未修改

峰值2秒、4096点、2048Hz、40--580Hz、H1/L1双通道、一维时间证据、BAYESTAR天空图和分数、逐seed C-fixed外层权重、O3的62事件及O4a的74事件strict scope、全部历史模型与论文均保持不变。O4 PSD覆盖审计含75个可嵌入事件，正式排名只用冻结74事件；二者不是分母错误。

## 数据与选择的科学边界

模型训练复用12288个物理波形source draws/160个noise blocks，每个源8个image/noise views；development为512源/32噪声块，source/noise ID无交集。GW-LMC透镜环境仍可能重复，不能声称12288独立透镜系统。注入仍继承逐像PSD-optimal target-SNR缩放、旧响应时间日历和conditional known-intrinsics BAYESTAR，不是重新生成的完整response-derived/full-PE实验。

新噪声条件化依据[DINGO的噪声条件化研究](https://arxiv.org/abs/2106.12594)与[PSD分布变化研究](https://arxiv.org/abs/2211.08801)；它们支持检查噪声条件化，不保证本项目网络能获得完整PE精度。时频分支参考[基于谱图的引力波参数推断研究](https://arxiv.org/abs/2011.10425)的表示思路，网络宽度、学习率和截断均为明确冻结的探索工程选择，不冒充论文推荐默认值。

质量概率输出是模拟训练下的预测分布，不是完整PE后验。预测重叠经512 development源对的balanced isotonic校准后截断正负4。PRIOR表示用训练质量先验校正的重叠特征，不是proper Bayes factor。ADD在旧OMC内部加新证据；REPLACE从更早FRT起加入新质量项，但验收始终相对于最新OMC，不使用较弱基线抬高改善。PSD距离支持域仅由training/development冻结；质量边界或PSD-OOD时整个新更新回退为旧OMC。

真实PE和官方字段未进入网络训练或分数输入，但本轮有限系数选择明确使用它们作为**适应性开发目标**。不能同时称“validation-only真实独立验证”。以前已反复使用的test只是复用开发比较；没有联合达标候选，因此未启动新的独立确认，更不能引用旧确认冒充本轮新确认。

## 全部对照

下面为每个对照的非劣开发性诊断。若最优诊断全部为UNCHANGED，数值等于旧OMC，应理解为新方法未获资格，而不是完成了改善。实际积极修改但不达标的结果另完整输出到PE_leading_not_adopted。三seed混合模型在各旧部署共享，表内SD不是三个独立重训ensemble的方差。

| 运行期 | 对照 | R@1 | R@10 | Pair AP | F50/F90 | Mc BC>=.5 Top10/20 | 官方1% Top10/20 | Hanabi重合 Top10/20 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(lines)}

## PE优先但未采纳的实际对照

下表不要求真实官方预算非劣，而是在模拟检索保护条件内按预定PE优先次序列出结果，用于展示新评分为什么没有晋级。它不能替代上面的统一Gate；完整系数、所有排名和失败原因均保留。这里的recall仍来自同一复用test，不是新的locked test。

| 运行期 | 对照 | R@1 | R@10 | Pair AP | F50/F90 | Mc BC>=.5 Top10/20 | 官方1% Top10/20 | Hanabi重合 Top10/20 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(changed_lines)}

## PE、官方阶段与置信区间

每个对照包含逐seed和consensus的waveform-only/C-fixed全pair表、Top100以及Top10/20/50/100预算。真实表保留Mc/q/chi_eff/表观距离BC和D、官方PO/ML或PO/Phazap FPP、GOLUM及公开Hanabi阶段。公开表未解析的阶段标为未解析，不填造“通过”。公开Hanabi表重合不表示透镜支持，更不是检测。

`SOURCE_BOOTSTRAP_DIAGNOSTICS.csv`对每类模型预先定义的非劣诊断报告10000次系统级R@K及1000次source-block加权pair区间。这是固定复用目录下的条件抽样波动，不校正适应性选参，也不是新数据确认。真实pair共享事件，不提供普通iid-pair显著性结论。

`TARGET_FAILURE_DECOMPOSITION.csv`分别列出PE非劣、PE改善、官方非劣、前端及Hanabi均增加、以及全部保护条件通过的数量。这只是对原门槛的分解，不事后放宽门槛。训练曲线恶化提示该分支在当前development分布下泛化受限，不证明O3的物理信息不可恢复。

另外做了一个仅用于审核目标约束的整数规划诊断：现有官方/PE表中存在满足外部预算目标的集合，但这不是能由waveform-only方法发现它们的证明。该审计不输出或使用“按官方标签挑选”的排名，不参与训练。

## 交付导航

- `contracts/NOISE_CONTEXT_CONTRACT.json`与`TF_RESIDUAL_ADDENDUM.json`：两轮冻结定义。
- `tables/EXPERIMENT_LEDGER.csv`：全部组合通过/失败。
- `tables/ALL_DIAGNOSTIC_RETRIEVAL.csv`、`RETRIEVAL_MEAN_SD.csv`：逐seed及mean/SD检索结果。
- `tables/ALL_DIAGNOSTIC_BUDGETS.csv`、`PE_LEADING_BUDGETS.csv`：全部PE/官方结果及失败对照。
- `variants/<模型>/trials/<评分>/diagnostic_export/`：完整pair与排名；`PE_leading_not_adopted/`：实际新分数的PE优先失败例。
- `tables/TRAINING_SUMMARY.csv`、`NOISE_DOMAIN_AUDIT.csv`：模型、噪声支持域与训练诊断。
- `tables/EVENT_MASS_PREDICTIONS_VS_PUBLIC_PE.csv`：各事件预测区间与公开PE中位数的描述性核对；公开PE中位数不是真值，不能把区间命中率称为coverage。
- `audit/predictive_mass/`：全部真实事件的新质量预测分布，含事件名和logMc网格；不是完整PE后验。
- `audit/INDEPENDENT_SCORE_REPLAY.csv`：独立NumPy复算，时间天空逐元素不变。
- `manifest/FINAL_HISTORICAL_HASH_RECHECK.csv`：历史保护哈希。

正式复现入口位于scripts/reproduction_dependencies/，含缓存并发修复后的最终脚本；scripts/顶层保留启动时快照。需在原服务器历史数据和冻结checkpoint仍可访问的环境运行。交付包不是含全部原始strain的自包含数据集。执行顺序为NOISE_CONTEXT_CONTRACT中的init/audit/train/evaluate、TF_RESIDUAL_ADDENDUM的train/evaluate，再由mcwf_noise_context_results_20260908.py完成materialize/counterexamples/verify/uncertainty/report/package。已完成目录不得原地重跑。

本报告不宣布替代、不修改论文、不把条件开发增益称为独立验证，也不以官方候选重合作为真实透镜标签。
'''
    (root / 'reports/FINAL_NOISE_TF_EXPLORATION_CN.md').write_text(text, encoding='utf-8')
    (root / 'README_CN.md').write_text('# MCWF-NOISE-TF 探索交付\n\n'+
        '完整结论见 reports/FINAL_NOISE_TF_EXPLORATION_CN.md。目标未达成，不采纳、不覆盖。\n', encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.ticker import MaxNLocator
    for font in Path('/usr/share/fonts').rglob('*.ttf'):
        if 'times' in font.name.lower(): font_manager.fontManager.addfont(str(font))
    available = {f.name for f in font_manager.fontManager.ttflist}
    plt.rcParams.update({'font.family': 'Times New Roman' if 'Times New Roman' in available else 'DejaVu Serif',
                         'font.size': 9, 'pdf.fonttype': 42})
    fig, axes = plt.subplots(2,2,figsize=(8,6))
    for row,budget in enumerate((10,20)):
        for ax,dep in zip(axes[row],e.old.DEPS):
            sub=failed[(failed.deployment==dep)&(failed.seed.astype(str)=='consensus')&(failed.method=='C_fixed')&(failed.budget==budget)]
            orig=baseline[(baseline.deployment==dep)&(baseline.budget==budget)].iloc[0]
            points=sub.groupby(['BC_mc_ge_0p5','official_frontend']).size().reset_index(name='count')
            ax.scatter(points.BC_mc_ge_0p5,points.official_frontend,c='#267a80',s=35+6*points['count'],alpha=.6,label='PE-leading experiments')
            for p in points.itertuples():
                if p.count>1: ax.annotate(f'n={p.count}',(p.BC_mc_ge_0p5,p.official_frontend),xytext=(0,11),textcoords='offset points',ha='center',fontsize=8)
            ax.scatter([orig.BC_mc_ge_0p5],[orig.official_frontend],marker='*',s=95,color='#b13c43',label='OMC baseline',zorder=3)
            xs=np.r_[points.BC_mc_ge_0p5,orig.BC_mc_ge_0p5];ys=np.r_[points.official_frontend,orig.official_frontend]
            ax.set(xlabel=f'Top-{budget}: Mc BC >= 0.5',ylabel=f'Top-{budget}: official frontend overlap',
                   title=('O3' if dep=='gwtc3' else 'O4a')+f', Top-{budget}',
                   xlim=(xs.min()-.7,xs.max()+.7),ylim=(ys.min()-.7,ys.max()+.9))
            ax.xaxis.set_major_locator(MaxNLocator(integer=True));ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            ax.spines[['top','right']].set_visible(False)
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',ncol=2,frameon=False)
    fig.tight_layout(rect=(0,0,1,.94))
    fig.savefig(root/'figures/fig_PE_official_tradeoff.pdf',bbox_inches='tight')
    fig.savefig(root/'figures/fig_PE_official_tradeoff.png',dpi=200,bbox_inches='tight');plt.close(fig)


def package(root):
    required = ['contracts/FINAL_DECISION.json', 'contracts/NOISE_CONTEXT_CONTRACT.json',
                'contracts/TF_RESIDUAL_ADDENDUM.json', 'contracts/BOOTSTRAP_INTERPRETATION.json',
                'audit/FINAL_SCORE_AND_HISTORY_VERIFICATION.json', 'audit/CACHE_CONCURRENCY_UNIT_TEST.json',
                'audit/WARM_CACHE_INDEPENDENT_REPLAY.csv', 'audit/SOFTWARE_AND_HARDWARE.json',
                'tables/RETRIEVAL_MEAN_SD.csv', 'tables/ALL_DIAGNOSTIC_RETRIEVAL.csv',
                'tables/ALL_DIAGNOSTIC_BUDGETS.csv', 'tables/PE_LEADING_BUDGETS.csv',
                'tables/TARGET_FAILURE_DECOMPOSITION.csv', 'tables/TRAINING_SUMMARY.csv',
                'tables/SOURCE_BOOTSTRAP_DIAGNOSTICS.csv', 'tables/EVENT_MASS_PREDICTIONS_VS_PUBLIC_PE.csv',
                'reports/FINAL_NOISE_TF_EXPLORATION_CN.md', 'figures/fig_PE_official_tradeoff.pdf',
                'figures/fig_PE_official_tradeoff.png']
    assert all((root / name).is_file() for name in required)
    assert len(list(root.glob('variants/*/trials/*/contracts/SEARCH_COMPLETE.json'))) == 20
    assert len(list(root.glob('variants/*/models/*/*/COMPLETE.json'))) == 30
    assert len(list(root.glob('variants/*/trials/*/*/results/*/*/consensus_*_all_pairs_pe_official.parquet'))) == 160
    assert len(pd.read_csv(root / 'tables/SOURCE_BOOTSTRAP_DIAGNOSTICS.csv')) == 60
    assert json.loads((root / 'audit/FINAL_SCORE_AND_HISTORY_VERIFICATION.json').read_text())['all_pass']
    dev.json_write(root / 'audit/DELIVERABLE_COMPLETENESS.json', {
        'pass': True, 'required_files': required, 'score_contrasts': 20, 'training_runs': 30,
        'real_consensus_pair_tables': 160, 'bootstrap_rows': 60,
        'official_and_PE_available_for_every_contrast': True,
        'no_joint_target_success_claim': True})
    collect(root)
    files=[]
    for p in sorted(root.rglob('*')):
        if not p.is_file(): continue
        rel=p.relative_to(root)
        if 'cache' in rel.parts or '__pycache__' in rel.parts or p.suffix == '.npy' or p.name.startswith('resume'): continue
        if p.suffix == '.npz' and rel.parts[:2] != ('audit', 'predictive_mass'): continue
        if p.name in ('OUTPUT_SHA256.csv','PACKAGE_CONTENT_VERIFICATION.json'): continue
        if p.suffix in ('.py','.md','.json','.csv','.log','.txt'):
            txt=p.read_text(encoding='utf-8',errors='replace')
            if re.search(r'-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----',txt): raise RuntimeError('Secret in output')
        files.append(p)
    manifest=root/'manifest/OUTPUT_SHA256.csv'
    dev.csv_write(manifest,pd.DataFrame([{'path':str(p.relative_to(root)),'sha256':dev.sha(p),'bytes':p.stat().st_size} for p in files]))
    files.append(manifest)
    dest=dev.PROJECT/'packages'/(root.name+'_deliverables.tar.gz')
    if dest.exists(): raise RuntimeError('Archive exists, refusing overwrite')
    with tarfile.open(dest,'w:gz',compresslevel=3) as tar:
        for p in files: tar.add(p,arcname=root.name+'/'+str(p.relative_to(root)),recursive=False)
    digest=dev.sha(dest)
    dest.with_suffix(dest.suffix+'.sha256').write_text(digest+'  '+dest.name+'\n')
    expected={root.name+'/'+str(p.relative_to(root)):dev.sha(p) for p in files}
    verified=0
    with tarfile.open(dest,'r:gz') as tar:
        for member in tar:
            if not member.isfile(): raise RuntimeError('Unexpected archive entry')
            h=hashlib.sha256()
            with tar.extractfile(member) as handle:
                for chunk in iter(lambda:handle.read(8<<20),b''):h.update(chunk)
            assert expected[member.name]==h.hexdigest(),member.name
            verified+=1
    assert verified==len(files)
    dev.json_write(dest.with_suffix(dest.suffix+'.verification.json'),{
        'archive':str(dest),'sha256':digest,'bytes':dest.stat().st_size,'verified_files':verified,'all_pass':True})
    print(json.dumps({'archive':str(dest),'sha256':digest,'verified_files':verified}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--phase',choices=('materialize','counterexamples','verify','uncertainty','event_audit','report','package'),required=True)
    a=p.parse_args();globals()[a.phase](a.root)
