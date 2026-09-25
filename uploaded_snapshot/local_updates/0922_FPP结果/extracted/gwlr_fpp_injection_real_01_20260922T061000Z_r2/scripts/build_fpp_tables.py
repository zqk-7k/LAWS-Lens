"""Append conditional injection/real FPP tables without changing frozen C scores."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import shutil
import time

import numpy as np
import pandas as pd

from fpp_core import ConditionalNull, assert_disjoint, catalog_queries, conservative_cut, interval_union

RUNS = {'O3': 62, 'O4a': 74, 'O4b': 86}
SEEDS = [2026091721, 2026091722, 2026091723]
MODELS = [str(s) for s in SEEDS]+['mean_S']
LEVELS = [.1, .05, .01, .001]
CHANNELS = ['final_score_POSITIVE', 'waveform_contribution_POSITIVE',
            'time_contribution_POSITIVE', 'sky_contribution_POSITIVE',
            'waveform_score', 'time_score', 'sky_raw_log_bf']
FIELDS = ['event_uid', 'source_uid', 'global_source_id', 'noise_parent_uid']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1024*1024), b''):
            h.update(part)
    return h.hexdigest()


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)


def table(path, data, csv=True, parquet=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    for suffix, enabled in [('.csv', csv), ('.parquet', parquet)]:
        target = path.with_suffix(suffix)
        if enabled:
            if target.exists():
                raise RuntimeError('Refuse overwrite: '+str(target))
            if suffix == '.csv':
                data.to_csv(target, index=False, encoding='utf-8-sig')
            else:
                data.to_parquet(target, index=False)


class Inputs:
    def __init__(self):
        self.hashes = {}

    def track(self, path, expected=None):
        path = Path(path)
        digest = sha(path)
        if expected is not None and digest != expected:
            raise RuntimeError('Historical receipt mismatch: '+str(path))
        if str(path) in self.hashes and self.hashes[str(path)] != digest:
            raise RuntimeError('Input changed while running: '+str(path))
        self.hashes[str(path)] = digest

    def read(self, path, receipt=None):
        if receipt is not None:
            self.track(receipt)
            items = json.loads(Path(receipt).read_text())['files']
            expected = {r['path']: r['sha256'] for r in items}[str(path)]
            self.track(path, expected)
        else:
            self.track(path)
        return pd.read_parquet(path)

    def verify(self):
        for path, digest in self.hashes.items():
            if sha(path) != digest:
                raise RuntimeError('Protected input changed: '+path)


def keyed(frame):
    f = frame.copy()
    keys = ['--'.join(sorted((str(a), str(b)))) for a, b in zip(f.event_i, f.event_j)]
    if 'pair_key' in f and f.pair_key.tolist() != keys:
        raise RuntimeError('Noncanonical archived pair key')
    f['pair_key'] = keys
    if f.pair_key.duplicated().any():
        raise RuntimeError('Duplicate pair table')
    if not np.isfinite(f[CHANNELS].to_numpy(float)).all():
        raise RuntimeError('Nonfinite frozen score')
    if not np.allclose(f[CHANNELS[1:4]].sum(axis=1), f.final_score_POSITIVE, rtol=1e-12, atol=1e-12):
        raise RuntimeError('Score decomposition mismatch')
    return f


def verify_sealed_real(frame, sealed, seed):
    reference = sealed[(sealed.seed == seed) & (sealed.method == 'three-channel')]
    if reference.pair_key.duplicated().any() or set(reference.pair_key) != set(frame.pair_key):
        raise RuntimeError('Sealed real seed scope differs')
    reference = reference.set_index('pair_key').loc[frame.pair_key]
    mapping = dict(final_score_POSITIVE='score', waveform_contribution_POSITIVE='wf_contribution',
                   time_contribution_POSITIVE='time_contribution', sky_contribution_POSITIVE='sky_contribution',
                   waveform_score='waveform_score', time_score='time_score', sky_raw_log_bf='sky_raw_log_bf')
    for source, target in mapping.items():
        if not np.allclose(frame[source], reference[target], rtol=1e-12, atol=1e-12):
            raise RuntimeError('Raw score differs from sealed real ranking: '+source)


def load_scored(baseline, run, split, inputs):
    root = baseline/f'completion/deployments/{run}/C_PHYSICAL'
    output, event_ref, pair_ref = {}, None, None
    if split == 'real':
        folder = root/'real_ranking'
        sealed = inputs.read(folder/'all_seed_rankings.parquet', folder/'COMPLETE.json')
    for seed in SEEDS:
        if split == 'real':
            folder = root/'real_ranking'
            f = keyed(inputs.read(folder/f'seed_{seed}_all_scores.parquet'))
            verify_sealed_real(f, sealed, seed)
        else:
            folder = root/f'evaluation/{split}/seed_{seed}'
            f = keyed(inputs.read(folder/'all_pair_scores.parquet', folder/'COMPLETE.json'))
            events = inputs.read(folder/'events.parquet', folder/'COMPLETE.json')
            if events.event_uid.duplicated().any() or not np.array_equal(events.idx, np.arange(len(events))):
                raise RuntimeError('Invalid event indices')
            for end in ('i', 'j'):
                if not np.array_equal(events.event_uid.to_numpy()[f['idx_'+end]], f['event_'+end]):
                    raise RuntimeError('Pair and event UID mismatch')
            if not np.array_equal(events.source_uid.to_numpy()[f.idx_i] == events.source_uid.to_numpy()[f.idx_j], f.is_true_pair):
                raise RuntimeError('Incorrect truth mapping')
            if event_ref is None:
                event_ref = events
            elif not event_ref[FIELDS].equals(events[FIELDS]):
                raise RuntimeError('Event realization differs across model seeds')
        if pair_ref is None:
            pair_ref = f.pair_key.tolist()
        elif set(f.pair_key) != set(pair_ref):
            raise RuntimeError('Model pair scope mismatch')
        f = f.set_index('pair_key', drop=False).loc[pair_ref].reset_index(drop=True)
        output[str(seed)] = f
    mean = output[str(SEEDS[0])].copy()
    for column in CHANNELS:
        mean[column] = np.mean([output[str(s)][column].to_numpy() for s in SEEDS], axis=0)
    output['mean_S'] = mean
    return event_ref, output


def make_null(events, frame):
    chosen = events.family.eq('unlensed').to_numpy()
    null_events = events[chosen].copy().reset_index(drop=True)
    if len(null_events) != 90 or null_events.global_source_id.duplicated().any():
        raise RuntimeError('Unexpected singleton population')
    mask = chosen[frame.idx_i] & chosen[frame.idx_j]
    if frame.loc[mask, 'is_true_pair'].any():
        raise RuntimeError('Companion present in null')
    local = np.full(len(events), -1, int)
    local[np.flatnonzero(chosen)] = np.arange(chosen.sum())
    f = frame[mask].copy()
    i, j = local[f.idx_i], local[f.idx_j]
    engine = ConditionalNull(f.final_score_POSITIVE, i, j, null_events.source_uid, null_events.noise_parent_uid)
    return null_events, f, engine


def annotate(frame, engine, run, model, domain, maxima):
    basic = ['pair_key', 'event_i', 'event_j', *CHANNELS]
    if domain == 'injection':
        basic += ['is_true_pair', 'true_pair_family']
    result = frame[basic].copy()
    result.insert(0, 'run', run)
    result.insert(1, 'model', model)
    result.insert(2, 'domain', domain)
    for name, value in engine.query(frame.final_score_POSITIVE.to_numpy()).items():
        result[name] = value
    result['background_events'] = len(engine.sources)
    result['background_pairs'] = len(engine.scores)
    result['background_noise_parents'] = len(np.unique(engine.noise))
    result['empirical_grid_spacing_not_CI'] = 1/len(engine.scores)
    result['sensitivity_range_is_not_confidence_interval'] = True
    result['FPP_status'] = ('CONDITIONAL_C_BACKGROUND_ONLY' if domain == 'injection'
                            else 'CONDITIONAL_MAPPING_REAL_DOMAIN_UNVALIDATED')
    result['tail_status'] = np.select([result.zero_exceedance_unresolved, result.low_tail_count],
                                    ['ZERO_EXCEEDANCES_UNRESOLVED', 'LOW_EXCEEDANCE_COUNT'],
                                    default='EMPIRICAL_TAIL_COUNT_ONLY')
    result['validated_GWTC_FPP'] = np.nan
    result['FAR_per_year'] = np.nan
    result['FAR_status'] = 'NO_GO_EFFECTIVE_BACKGROUND_EXPOSURE_UNESTABLISHED'
    if domain == 'real':
        for name, value in catalog_queries(maxima, result.conditional_FPP,
                                          result.final_score_POSITIVE, RUNS[run]).items():
            result[name] = value
        result['catalog_FPP_status'] = 'FINITE_VALIDATION_POOL_ALL_PAIR_MAXIMUM_NOT_CONSENSUS_SELECTION'
    return result


def calendar_audit(baseline, run, inputs):
    path = baseline/f'plans/{run}/live_schedule.csv'
    inputs.track(path)
    f = pd.read_csv(path)
    intervals = interval_union(zip(f.start_gps, f.end_gps))
    live = sum(b-a for a, b in intervals)
    return dict(run=run, calendar_source=str(path),
        calendar_span_days=(intervals[-1][1]-intervals[0][0])/86400,
        merged_live_days=live/86400,
        overlap_removed_seconds=float((f.end_gps-f.start_gps).sum()-live),
        strict_real_catalog_events=RUNS[run],
        calendar_intervals_known=True,
        calendar_is_validated_effective_background_exposure=False,
        detected_population_rate_and_selection_validated=False,
        null_population_transfer_validated=False, sky_tail_transfer_validated=False,
        effective_background_years=None, FAR_per_year=None,
        decision='NO_GO',
        reason='Controlled fixed-N background is not a rate-generated search; calendar is not effective background time')


def plot_results(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for column, run in enumerate(RUNS):
        inj = pd.read_parquet(root/f'tables/{run}/injection_mean_S_FPP.parquet')
        real = pd.read_parquet(root/f'tables/{run}/real_GWTC_mean_S_FPP.parquet').head(20)
        grid = np.geomspace(1/4005, 1, 150)
        for truth, label in [(True, 'True companions'), (False, 'Other injection pairs')]:
            p = inj.loc[inj.is_true_pair == truth, 'conditional_FPP'].to_numpy()
            axes[0, column].plot(grid, [np.mean(p <= g) for g in grid], label=label)
        axes[0, column].set(xscale='log', ylim=(0, 1.02), title=run,
                            xlabel='Conditional pair-tail threshold', ylabel='Fraction retained')
        ax = axes[1, column]
        unresolved = real.zero_exceedance_unresolved.to_numpy(bool)
        x, y = real.consensus_rank.to_numpy(), real.conditional_FPP.to_numpy()
        ax.scatter(x[~unresolved], y[~unresolved], color='#16697a', label='Empirical tail')
        if unresolved.any():
            ax.scatter(x[unresolved], np.full(unresolved.sum(), 1/4005), marker='v',
                       facecolors='none', edgecolors='#b53d32', label='Zero counts: unresolved')
        ax.axhline(.01, color='#777777', linestyle='--', linewidth=1)
        ax.set(yscale='log', ylim=(.5/4005, 1.2), xlabel='Unchanged real consensus rank',
               ylabel='Conditional FPP (NOT validated real FPP)')
        ax.set_xticks([1, 5, 10, 15, 20])
        ax.legend(fontsize=7)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle('GWLR-FPP-01: controlled-background diagnostic only; no annual FAR', fontsize=12)
    fig.tight_layout()
    (root/'figures').mkdir(exist_ok=True)
    fig.savefig(root/'figures/conditional_FPP_overview.png', dpi=180)
    fig.savefig(root/'figures/conditional_FPP_overview.pdf')
    plt.close(fig)


def report(root, status, injections, real_budgets, far):
    lines = ['# GWLR-FPP-01：注入与真实候选条件 FPP、FAR 可用性审计', '',
        '**已完成条件FPP查询表；尚未完成真实GWTC的总体适用性验证。年度FAR：NO-GO。**', '',
        '## 1. 计算定义',
        'FPP_cond(s)=背景中S>=s的非透镜对数/背景非透镜对总数。每运行期独立建立背景；'
        '同一背景、同一分数得到同一FPP，不因对象为注入或真实候选而改变。',
        '背景为C validation的90个孤立源，4005个无序对，6个噪声父块。排除所有透镜像。'
        '注入评价使用source/noise-disjoint的C test；它们是历史已查看数据，不是新增独立盲测。',
        'validation曾参与模型/系数选择，因此这不是额外独立的严格概率校准集；不能宣称名义覆盖率得到保证。',
        '未重训、未调权、未改天空或时间。逐seed查询各自S；mean_S用三模型分数均值重新建立尾部，'
        '不是把三个FPP平均，也不是把mean_S当成原共识排序规则。原真实共识名次完全保留。', '',
        '## 2. 两张表是什么意思',
        '- 注入表：已知真假配对的条件背景尾部位置，用于统计阈值下伴随对保留率和假对负担。',
        '- GWTC表：真实分数在同一C背景中的位置；仅是条件映射，不是已验证的真实FPP或透镜后验概率。',
        '- validated_GWTC_FPP与FAR_per_year保持空值；不得将它们用条件列自动填充。', '',
        '## 3. 注入结果（mean_S，经验条件FPP<1%）', '',
        '|运行期|真伴随对保留|伴随对保留率|非伴随对保留|纯孤立test对超阈值率|',
        '|---|---:|---:|---:|---:|']
    for r in injections:
        if r['model'] == 'mean_S' and r['FPP_threshold'] == .01:
            lines.append(f'|{r["run"]}|{r["true_pairs_retained"]}/{r["true_pairs_total"]}|'
                         f'{r["true_pair_recovery_fraction"]:.4f}|{r["false_pairs_retained"]}|'
                         f'{r["test_singleton_tail_fraction"]:.4%}|')
    lines += ['', '这里的伴随对保留率是pair recall，不是query R@10。每运行期test共450事件、'
              '180真对、100845非伴随对；这些非伴随对含不同透镜系统的像，不等于纯非透镜人口。', '',
              '## 4. 真实Top-10（保持原排名，仅附加mean_S条件FPP）', '',
              '|运行期|条件FPP<1%|其中零背景超越|尾部计数<20|',
              '|---|---:|---:|---:|']
    for r in real_budgets:
        if r['budget'] == 10:
            lines.append(f'|{r["run"]}|{r["conditional_FPP_below_0p01"]}/10|'
                         f'{r["zero_exceedance_unresolved"]}|{r["low_tail_count"]}|')
    lines += ['', '这不是新的官方候选通过数，也不是确认透镜的数量。官方PO/ML/Phazap列原样保留，并与本次条件FPP严格分列。', '',
        '## 5. 不确定度、尾部与目录效应',
        '- 4005对共享90个源和6个噪声块，不能按4005次独立伯努利试验计算置信区间。',
        '- 每行输出删除一个source和删除一个noise-parent后的FPP最小/最大值，属于敏感性范围，不是95%置信区间。',
        '- background_exceedances=0时，经验比例为0但标记ZERO_EXCEEDANCES_UNRESOLVED，不解释成零风险。1/4005仅是计数网格间距，不是FPP上界。',
        '- <20个尾部对标记LOW_EXCEEDANCE_COUNT；这是展示支持量的警示，不是通过真实性检验。',
        '- 从同一个validation孤立事件池无放回抽取1000个62/74/86事件目录；每目录不重复源，跨目录复用有限池。',
        '- conditional_catalog_FPP_mc是所有pair的mean-S最大值超阈值比例，不是原rank-consensus候选选择流程的正式FPP。',
        '- conditional_expected_false_pairs=N(N-1)/2*FPP_cond是有限池精确期望，利用期望线性性，不假设pair独立。', '',
        '## 6. FAR可用性', '', '|运行期|日历跨度/天|共同在线区间并集/天|有效背景年数|年度FAR|',
        '|---|---:|---:|---|---|']
    for r in far:
        lines.append(f'|{r["run"]}|{r["calendar_span_days"]:.4f}|{r["merged_live_days"]:.4f}|未建立|NO-GO|')
    lines += ['', '这些日历属于C的输入安排，不代表完成了同等时长的探测后非透镜事件人口模拟。'
        '没有校准的发生率/选择模型，也没有真实PE与模拟BAYESTAR尾部转移验证，故不把假对数除以日历。', '',
        '## 7. 当前可以与不可以使用',
        '可以：报告当前受控实验下的FPP查询、注入pair效率、有限池假对负担和真实候选的探索性背景位置。',
        '不可以：称为已验证真实GWTC FPP、把1-FPP当透镜概率、称为完整目录显著性、报告次/年FAR或据此确认透镜。',
        '正式升级仍需独立人口与噪声、匹配的观测选择、天空管线域验证和独立校准/验证背景；本轮没有生成新的总体背景。', '',
        '## 8. 文件入口',
        '- tables/{run}/injection_mean_S_FPP.csv/.parquet：全部注入pair的主查询表。',
        '- tables/{run}/real_GWTC_mean_S_FPP.csv/.parquet：全部真实pair，含原PE、官方结果、原排名。',
        '- tables/{run}/real_GWTC_Top10/20/50_FPP.csv：候选头部。',
        '- tables/{run}/injection_seed_*.parquet与real_GWTC_seed_*.csv/.parquet：逐模型查询表。',
        '- background/：冻结校准对、事件清单、经验尾部曲线及可复算目录抽样。',
        '- summaries/：注入效率、真实预算、源噪声隔离与FAR审计。',
        '- contracts/、scripts/、logs/、manifests/：可复现输入与规则。', '',
        f'本轮追踪{status["protected_inputs"]}个历史输入，前后哈希不变。原模型、权重、排名与论文未改动。',
        '参考：[LVK O4a III.1.1与附录B](https://arxiv.org/html/2512.16347v3)。采用尾部定义，不照搬其数值FPP。', '',
        status['state']]
    (root/'reports').mkdir(exist_ok=True)
    (root/'reports/FINAL_FPP_AND_FAR_AUDIT_CN.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def run(root, baseline):
    started = time.monotonic()
    if shutil.disk_usage(root).free < 20*1024**3:
        raise RuntimeError('HOLD_DISK_LIMIT')
    contract = dict(experiment='GWLR-FPP-01', utc=datetime.now(timezone.utc).isoformat(),
        technical_revision='r2: verify raw real scores against hash-sealed all_seed_rankings; statistics unchanged',
        baseline=str(baseline), runs=RUNS, seeds=SEEDS,
        score='Frozen C POSITIVE score per seed, plus mean of three scores; no mean of FPPs',
        ranking='Keep original C real consensus rank without changes',
        calibration='C validation singleton sources only; exclude all lensed images',
        evaluation='C test all pairs; source/global-source/event/noise-disjoint from calibration',
        nominal_tail_levels=LEVELS, comparison='score >= candidate; threshold summary conditional_FPP < alpha',
        population_support='Controlled C population, not calibrated astrophysical null',
        old_validation_used_for_model_selection=True, historical_test_previously_seen=True,
        background_events_expected=90, background_pairs_expected=4005,
        background_noise_parents_expected=6, low_tail_warning_count=20,
        catalog_mc_draws=1000, catalog_mc_seed=2026092209,
        physical_independent_catalog_count=None,
        sensitivity='Source/noise leave-one-group-out ranges; not confidence intervals',
        real_domain_transfer_validated=False, annual_FAR_allowed=False,
        effective_background_time=None, no_new_waveforms=True,
        train_or_reselect=False, input_overwrite=False,
        final_state='CONDITIONAL_TABLES_COMPLETE_HOLD_REAL_CALIBRATION_FAR_NO_GO')
    write(root/'contracts/ANALYSIS_CONTRACT.json', contract)
    inputs = Inputs()
    freeze = baseline/'contracts/FINAL_SCORE_FREEZE.json'
    inputs.track(freeze)
    for item in json.loads(freeze.read_text())['files']:
        inputs.track(item['path'], item['sha256'])
    engines, bg_events, config, maxima = {}, {}, {}, {}
    for run_idx, run_name in enumerate(RUNS):
        events, frames = load_scored(baseline, run_name, 'validation', inputs)
        config[run_name] = {}
        for model in MODELS:
            e, f, engine = make_null(events, frames[model])
            engines[run_name, model], bg_events[run_name] = engine, e
            folder = root/f'background/{run_name}'
            pair_table = f[['event_i', 'event_j', 'pair_key', *CHANNELS]].copy()
            pair_table['source_i'] = events.source_uid.to_numpy()[f.idx_i]
            pair_table['source_j'] = events.source_uid.to_numpy()[f.idx_j]
            pair_table['noise_i'] = events.noise_parent_uid.to_numpy()[f.idx_i]
            pair_table['noise_j'] = events.noise_parent_uid.to_numpy()[f.idx_j]
            table(folder/f'{model}_calibration_pairs', pair_table)
            if model == 'mean_S':
                table(folder/'calibration_events', e[FIELDS+['gps_obs', 'optimal_network_snr', 'mc_det']])
            cuts = [conservative_cut(engine.scores, level) for level in LEVELS]
            config[run_name][model] = dict(cuts=cuts, levels=LEVELS,
                n_pairs=len(engine.scores), event_count=len(e), noise_groups=int(e.noise_parent_uid.nunique()))
            indices, max_scores = engine.catalogs(RUNS[run_name], 1000, 2026092209+run_idx)
            maxima[run_name, model] = max_scores
            np.savez_compressed(folder/f'{model}_conditional_catalogs.npz',
                                indices=indices, max_scores=max_scores, event_uids=e.event_uid.to_numpy(str))
            x = np.unique(engine.scores)
            table(folder/f'{model}_score_FPP_curve', pd.DataFrame(dict(
                score=x, background_exceedances=len(engine.scores)-np.searchsorted(np.sort(engine.scores), x),
                conditional_FPP=(len(engine.scores)-np.searchsorted(np.sort(engine.scores), x))/len(engine.scores))))
    selected = root/'contracts/BACKGROUND_FREEZE.json'
    write(selected, dict(config=config, contract_sha256=sha(root/'contracts/ANALYSIS_CONTRACT.json'),
                         file_hashes={str(p.relative_to(root)): sha(p) for p in sorted((root/'background').rglob('*')) if p.is_file()}))
    print('BACKGROUND_FROZEN_BEFORE_TEST_AND_REAL_READ', sha(selected), flush=True)
    efficiency, budgets, split_records, far, ranks = [], [], [], [], []
    for run_name in RUNS:
        event_test, test_frames = load_scored(baseline, run_name, 'test', inputs)
        overlaps = assert_disjoint(bg_events[run_name], event_test, FIELDS)
        split_records.append(dict(run=run_name, **{k+'_overlap': v for k, v in overlaps.items()},
                                  all_test_events=len(event_test), test_noise_parents=int(event_test.noise_parent_uid.nunique())))
        singleton = event_test.family.eq('unlensed').to_numpy()
        for model in MODELS:
            engine, frame = engines[run_name, model], test_frames[model]
            annotated = annotate(frame, engine, run_name, model, 'injection', maxima[run_name, model])
            label = 'mean_S' if model == 'mean_S' else 'seed_'+model
            table(root/f'tables/{run_name}/injection_{label}_FPP', annotated, csv=(model == 'mean_S'))
            table(root/f'tables/{run_name}/injection_true_companions_{label}_FPP', annotated[annotated.is_true_pair])
            truth = annotated.is_true_pair.to_numpy(bool)
            null = singleton[frame.idx_i] & singleton[frame.idx_j]
            for alpha in LEVELS:
                chosen = annotated.conditional_FPP.to_numpy() < alpha
                true_kept, false_kept = int((chosen & truth).sum()), int((chosen & ~truth).sum())
                efficiency.append(dict(run=run_name, model=model, FPP_threshold=alpha,
                    true_pairs_retained=true_kept, true_pairs_total=int(truth.sum()),
                    true_pair_recovery_fraction=true_kept/truth.sum(),
                    false_pairs_retained=false_kept, false_pairs_total=int((~truth).sum()),
                    noncompanion_tail_fraction=false_kept/(~truth).sum(),
                    test_singleton_exceedances=int((chosen & null).sum()),
                    test_singleton_pairs=int(null.sum()), test_singleton_tail_fraction=float(chosen[null].mean()),
                    zero_background_count_true_retained=int((chosen & truth & annotated.zero_exceedance_unresolved.to_numpy()).sum())))
        print('INJECTION_TABLES_COMPLETE', run_name, flush=True)
        _, real_frames = load_scored(baseline, run_name, 'real', inputs)
        audit_path = baseline/f'completion/real_PE/{run_name}/C_PHYSICAL/three-channel/all_pairs_with_PE_official.parquet'
        audit = inputs.read(audit_path).sort_values('consensus_rank').reset_index(drop=True)
        rank_folder = baseline/f'completion/deployments/{run_name}/C_PHYSICAL/real_ranking'
        sealed_consensus = inputs.read(rank_folder/'consensus_all_pairs.parquet', rank_folder/'COMPLETE.json')
        sealed_consensus = sealed_consensus[sealed_consensus.method == 'three-channel'].sort_values('consensus_rank')
        if not np.array_equal(audit.pair_key, sealed_consensus.pair_key) or not np.array_equal(audit.consensus_rank, sealed_consensus.consensus_rank):
            raise RuntimeError('PE audit table rank differs from sealed ranking')
        if len(audit) != RUNS[run_name]*(RUNS[run_name]-1)//2 or audit.pair_key.duplicated().any():
            raise RuntimeError('Unexpected real scope')
        if not np.array_equal(audit.consensus_rank, np.arange(1, len(audit)+1)):
            raise RuntimeError('Archived ranks not complete')
        if set(audit.pair_key) != set(real_frames['mean_S'].pair_key):
            raise RuntimeError('PE/real score scope mismatch')
        mean = real_frames['mean_S'].set_index('pair_key').loc[audit.pair_key]
        score_difference = float(np.max(np.abs(mean.final_score_POSITIVE.to_numpy()-audit.score_mean.to_numpy())))
        if score_difference > 1e-12:
            raise RuntimeError('Mean frozen real score differs from archived score_mean')
        for model in MODELS:
            engine = engines[run_name, model]
            f = annotate(real_frames[model], engine, run_name, model, 'real', maxima[run_name, model])
            # Audit columns retain their original names and values, never enter FPP calibration.
            additions = f.drop(columns=[k for k in f.columns if k in audit.columns and k != 'pair_key'])
            joined = audit.merge(additions, on='pair_key', validate='one_to_one', how='left', sort=False)
            if not np.array_equal(joined.consensus_rank, audit.consensus_rank) or not np.array_equal(joined.pair_key, audit.pair_key):
                raise RuntimeError('Rank or pair order changed')
            label = 'mean_S' if model == 'mean_S' else 'seed_'+model
            table(root/f'tables/{run_name}/real_GWTC_{label}_FPP', joined)
            if model == 'mean_S':
                for top in [10, 20, 50]:
                    subset = joined.head(top)
                    table(root/f'tables/{run_name}/real_GWTC_Top{top}_FPP', subset)
                    budgets.append(dict(run=run_name, budget=top,
                        conditional_FPP_below_0p01=int((subset.conditional_FPP < .01).sum()),
                        zero_exceedance_unresolved=int(subset.zero_exceedance_unresolved.sum()),
                        low_tail_count=int(subset.low_tail_count.sum()),
                        median_conditional_FPP=float(subset.conditional_FPP.median()),
                        maximum_conditional_FPP=float(subset.conditional_FPP.max())))
        ranks.append(dict(run=run_name, real_pairs=len(audit), original_ranks_unchanged=True,
                          max_mean_score_difference=score_difference))
        far.append(calendar_audit(baseline, run_name, inputs))
        print('REAL_TABLES_COMPLETE_ORIGINAL_RANKS_UNCHANGED', run_name, flush=True)
    table(root/'summaries/injection_efficiency_at_FPP', pd.DataFrame(efficiency))
    table(root/'summaries/real_Top_budget_FPP', pd.DataFrame(budgets))
    table(root/'summaries/source_noise_isolation', pd.DataFrame(split_records))
    table(root/'summaries/real_rank_identity', pd.DataFrame(ranks))
    table(root/'summaries/FAR_feasibility', pd.DataFrame(far))
    write(root/'summaries/FAR_FEASIBILITY.json', dict(annual_FAR_usable=False, runs=far,
          note='FPP requires a suitable null distribution; FAR additionally requires matched effective exposure'))
    plot_results(root)
    inputs.verify()
    table(root/'manifests/PROTECTED_INPUT_SHA256', pd.DataFrame([
        dict(path=p, sha256=v, unchanged=True) for p, v in sorted(inputs.hashes.items())]))
    status = dict(state=contract['final_state'], protected_inputs=len(inputs.hashes),
        injection_unique_pair_rows=3*101025, injection_score_views=4,
        real_unique_pair_rows=sum(n*(n-1)//2 for n in RUNS.values()), real_score_views=4,
        conditional_tables_complete=True, validated_real_FPP_complete=False,
        annual_FAR_usable=False, effective_background_years=None,
        new_population_events_generated=0, trained_models_changed=False,
        original_rankings_unchanged=True, seconds=time.monotonic()-started,
        python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__)
    report(root, status, efficiency, budgets, far)
    write(root/'RUN_STATUS.json', status)
    print(json.dumps(status), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--baseline', type=Path, required=True)
    a = p.parse_args()
    try:
        run(a.root, a.baseline)
    except Exception as error:
        write(a.root/('FAILURE_'+str(time.time_ns())+'.json'), dict(error=repr(error), state='HOLD_EXECUTION_FAILURE'))
        raise
