"""C validation, freeze, evaluation, real audit and verified delivery."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tarfile

import numpy as np
import pandas as pd

ARM = 'C_PHYSICAL'
FINAL = 'HOLD_FOR_AUTHOR_REVIEW_NO_ADOPTION_NO_OVERWRITE'


def initialize(root):
    sys.path.insert(0, str(root/'completion/scripts'))
    import uab_completion as c
    c.initialize(root, root/'completion')
    return c


def evaluation_contract(c):
    c.write(c.OUT/'contracts/COMPLETION_ADAPTER_CONTRACT.json', dict(code='GWLR-UC-01',
        same_NEW_SCORE_ONLY_scoring=True, old_new_mixture=False, changed_sky_trigger_inputs=True,
        common_system_amplitude=True, model_seeds=list(c.U.SEEDS), arms=[ARM],
        real_PE_used_for_selection=False, no_automatic_method_adoption=True))
    c.write(c.OUT/'contracts/EVALUATION_COMPLETION_CONTRACT.json', dict(code='GWLR-UC-01',
        primary_events=450, size_control_events=190, subcatalogs=500, independent_subcatalogs=False,
        bootstrap_replicates=1000, inference='source and noise cluster sensitivity intervals, not independent pairs',
        primary_metrics=['R1', 'R10', 'average_precision', 'F50', 'F90', 'TopB precision and false burden'],
        real_budgets=[10, 20, 50, 100], official_O4b='NA, not zero', new_Hanabi=False,
        resolution=dict(coarse=256, analysis=512, reference=1024, reference_scope='all pairs',
            dense_persistence=False, diagnostic_only=True),
        sky_calibration='validation-only source-weighted temperature grid; no test retuning',
        limitations=['conditional detectable-source population, not cosmological GW-LMC population',
            'aligned-spin template approximation for precessing/higher-mode injections',
            'catalogue-time follow-up, not blind detection with calibrated FAR',
            'six evaluation noise parents per run', 'public real PE can use Virgo; injection maps are HL',
            'historical A/B data already inspected; no new blind confirmation claim'], final_state=FINAL))


def summarize(c):
    import uab_evaluate as e
    metrics, sub, intervals, distributions, weights = [], [], [], [], []
    for run in c.U.RUNS:
        p = c.deployment(run, ARM)
        for split in ('validation', 'test'):
            for seed in c.U.SEEDS:
                d = p/f'evaluation/{split}/seed_{seed}'
                if not c.check_complete(d/'CATALOG190_COMPLETE.json'):
                    raise RuntimeError('Missing C evaluation')
                metrics.append(pd.read_csv(d/'metrics.csv'))
                sub.append(pd.read_csv(d/'catalog190_metrics.csv'))
                distributions.append(pd.read_csv(d/'distributions.csv'))
                if split == 'test':
                    intervals.append(pd.read_csv(d/'cluster_intervals.csv'))
        for seed in c.U.SEEDS:
            spec = json.loads((p/f'calibration/score/seed_{seed}/SELECTED.json').read_text())
            for mode, chosen in spec['final_fusion'].items():
                weights.append(dict(run=run, arm=ARM, seed=seed, mode=mode,
                    **dict(zip(('waveform', 'time', 'sky'), chosen['weights'])),
                    **{branch+'_'+k: spec[branch][k] for branch in ('FRT', 'OMC', 'joint') for k in ('gamma', 'beta')}))
    allm, alls = pd.concat(metrics, ignore_index=True), pd.concat(sub, ignore_index=True)
    e.csv(c.OUT/'tables/retrieval_metrics_per_seed.csv', allm)
    e.csv(c.OUT/'tables/catalog190_all_draws.csv', alls)
    columns = ['macro_r_at_1', 'macro_r_at_5', 'macro_r_at_10', 'macro_r_at_50',
               'average_precision', 'false_at_recall_0p5', 'false_at_recall_0p9', 'roc_auc']
    for name, frame in [('450', allm), ('190', alls.groupby(['run', 'arm', 'split', 'seed', 'method'])[columns].mean().reset_index())]:
        summary = frame.groupby(['run', 'arm', 'split', 'method'])[columns].agg(['mean', 'std']).reset_index()
        summary.columns = ['_'.join(filter(None, x)) if isinstance(x, tuple) else x for x in summary.columns]
        e.csv(c.OUT/f'tables/retrieval_summary_{name}.csv', summary)
        baseline = c.U.P/'results/gwlr_unified_snr_ab_20260917T113500Z_r4/completion_20260918T014623Z/tables'/f'retrieval_summary_{name}.csv'
        e.csv(c.OUT/f'tables/A_B_C_comparison_{name}.csv',
              pd.concat([pd.read_csv(baseline), summary], ignore_index=True))
    e.csv(c.OUT/'tables/cluster_intervals.csv', pd.concat(intervals, ignore_index=True))
    e.csv(c.OUT/'tables/channel_distributions.csv', pd.concat(distributions, ignore_index=True))
    e.csv(c.OUT/'tables/selected_weights.csv', pd.DataFrame(weights))
    c.seal(c.OUT/'contracts/INJECTION_EVALUATION_COMPLETE.json', list((c.OUT/'tables').glob('*.csv')),
           runs=3, arms=1, model_seeds=3, native_events=450, controlled_events=190,
           real_used=False, no_test_retuning=True)


def freeze(c):
    import uab_scoring as s
    original = c.seal
    def seal(path, files, **details):
        if Path(path).name == 'FINAL_SCORE_FREEZE.json':
            details.update(state='ALL_THREE_RUNS_C_FROZEN_BEFORE_C_TEST', deployments=3,
                test_previously_opened=True, C_test_scores_used_for_selection=False,
                note='Historical A/B parents already examined; C scores not yet evaluated')
        return original(path, files, **details)
    c.seal = seal
    try:
        s.freeze()
    finally:
        c.seal = original


def archive_and_check(path, files, c):
    if path.exists():
        raise RuntimeError('No archive overwrite')
    manifest = {name: c.U.sha(p) for name, p in files.items()}
    c.write(path.with_suffix('.manifest.json'), manifest)
    with tarfile.open(path, 'x:gz', compresslevel=3) as archive:
        for name, p in sorted(files.items()):
            archive.add(p, arcname=name, recursive=False)
    seen = set()
    with tarfile.open(path, 'r:gz') as archive:
        for member in archive:
            if member.name not in manifest:
                raise RuntimeError('Unexpected archive member')
            stream = archive.extractfile(member)
            h = hashlib.sha256()
            for block in iter(lambda: stream.read(2**20), b''):
                h.update(block)
            if h.hexdigest() != manifest[member.name]:
                raise RuntimeError('Archive hash failure')
            seen.add(member.name)
    if seen != set(manifest):
        raise RuntimeError('Incomplete archive')
    digest = c.U.sha(path)
    Path(str(path)+'.sha256').write_text(digest+'  '+path.name+'\n')
    return digest


def deliver(c):
    from uab_evaluate import csv, PRIMARY
    historical = json.loads((c.ROOT/'contracts/PROTECTED_INPUTS.json').read_text())
    changed = [name for name, digest in historical.items() if c.U.sha(name) != digest]
    c.write(c.OUT/'contracts/HISTORICAL_HASH_AUDIT.json',
            dict(checked=len(historical), changed=changed, state='FAIL' if changed else 'PASS'))
    if changed:
        raise RuntimeError('Protected historical inputs changed')
    budgets, tops = [], []
    for run in c.U.RUNS:
        if not c.check_complete(c.OUT/'real_PE'/run/'COMPLETE.json'):
            raise RuntimeError('Real PE/official audit missing')
        budgets.append(pd.read_csv(c.OUT/'real_PE'/run/'budget_summary.csv'))
        tops.append(pd.read_csv(c.OUT/'real_PE'/run/ARM/PRIMARY/'top50_with_PE_official.csv'))
    budget, top = pd.concat(budgets, ignore_index=True), pd.concat(tops, ignore_index=True)
    csv(c.OUT/'tables/real_PE_official_budget_summary.csv', budget)
    csv(c.OUT/'tables/real_top50_all_runs.csv', top)
    summary = pd.read_csv(c.OUT/'tables/retrieval_summary_190.csv')
    full = pd.read_csv(c.OUT/'tables/retrieval_summary_450.csv')
    coverage = pd.read_csv(c.OUT/'tables/sky_truth_coverage_and_runtime.csv')
    def table(frame):
        return frame.to_markdown(index=False, floatfmt='.4f')
    report = ['# GWLR-UC-01 完整方法与结果', FINAL,
        '本轮是独立 C 受控探索实验，不覆盖 A/B、ET、历史排名或论文。没有运行新的 Hanabi。',
        '## 方法',
        '每个源使用同一个振幅系数，所有像和噪声视图共用。按参考噪声下较弱像 optimal SNR 8–40 '
        '条件化的欧氏体积距离先验抽样，不单独指定另一像的SNR；亮像不截断。相对强度由放大率、'
        '各时刻天线响应和局部PSD决定。这不是完整的宇宙学透镜总体。',
        '波形与天空读取同一64秒带噪应变，保留2秒短窗及16秒低频分支。独立8192模板库在目录时刻'
        '前后0.25秒搜索H/L触发，前8模板作16频带卡方重排；BAYESTAR使用原始复数SNR而不是'
        '重加权SNR。真质量/自旋不用于选模板。定位不是完整BBH PE，也不是有标定FAR的盲搜索。',
        '三个运行期采用相同NEW-SCORE-ONLY规则，各自重训五个组件、三个seed，共45组件。'
        '一维时间lookup与A/B相同。天空Nside512，显式NESTED→RING，float64重叠累积；'
        '256/1024只审计。温度、波形校准、融合权重只用validation，无外层旧新总分混合。',
        '源和噪声父块沿用A/B的隔离划分，历史数据已被查看，不能称新的盲测。每运行期仅六个'
        '测试噪声父块；500个190事件子目录不是500次独立实验。',
        '## 190事件结果', table(summary), '## 450事件结果', table(full),
        '## 真实PE与公开阶段', table(budget),
        'PE与官方表在冻结排名后联表，不参与选参。O4b无公开逐对表时记NA，不记0。'
        '公开Hanabi重合不是真透镜标签。', '## 天空覆盖及运行时间', table(coverage),
        '训练侧pilot只检查明显数值/定位失败，不证明后验已校准或模板库完备。'
        '完整test覆盖如实报告，不据此回调温度。',
        '## 交付', '逐seed、A/B/C同规模表、分布、区间与权重见tables/；全部pair及'
        'Top10/20/50/100见real_PE/。原生MOC另包；不保存全量dense天图。历史哈希不变。']
    for run in c.U.RUNS:
        report += ['## '+run+' Top10', table(top[top.run == run].head(10))]
    path = c.OUT/'reports/GWLR_UC_01_COMPLETE_METHOD_AND_RESULTS_CN.md'
    path.write_text('\n\n'.join(report)+'\n', encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plot = pd.read_csv(c.OUT/'tables/A_B_C_comparison_190.csv')
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, key in zip(axes, ('macro_r_at_10', 'average_precision', 'false_at_recall_0p9')):
        for arm, shift in [('A_NEUTRAL', -.24), ('B_CUE', 0), (ARM, .24)]:
            g = plot[(plot.arm == arm) & (plot.split == 'test') & (plot.method == PRIMARY)].set_index('run').reindex(c.U.RUNS)
            ax.bar(np.arange(3)+shift, g[key+'_mean'], width=.23, label=arm, yerr=g[key+'_std'])
        ax.set_xticks(range(3), c.U.RUNS)
        ax.set_title(key)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(c.OUT/'figures/A_B_C_comparison.png', dpi=180)
    fig.savefig(c.OUT/'figures/A_B_C_comparison.pdf')
    plt.close(fig)
    files = {}
    def add(root, prefix, skip_arrays=False):
        for f in root.rglob('*'):
            if not f.is_file() or f.is_symlink() or '__pycache__' in f.parts or f.suffix in ('.lock', '.pyc'):
                continue
            if skip_arrays and f.suffix in ('.npy', '.npz'):
                continue
            files[prefix+'/'+str(f.relative_to(root))] = f
    for folder in ('contracts', 'scripts', 'reports', 'manifests', 'logs', 'plans'):
        add(c.ROOT/folder, 'training/'+folder, True)
    for folder in ('contracts', 'scripts', 'tables', 'reports', 'figures', 'logs', 'real_PE', 'real_sky'):
        add(c.OUT/folder, 'completion/'+folder)
    for run in c.U.RUNS:
        p = c.deployment(run, ARM)
        for folder in ('calibration', 'evaluation', 'sky_pair_scores', 'real_ranking'):
            add(p/folder, f'completion/deployments/{run}/{ARM}/{folder}')
    models = pd.read_csv(c.OUT/'contracts/TRAINED_MODEL_MANIFEST.csv')
    for row in models.itertuples():
        p = Path(row.path)
        files['training/'+str(p.relative_to(c.ROOT))] = p
    manifest = [dict(path=name, sha256=c.U.sha(p), bytes=p.stat().st_size) for name, p in sorted(files.items())]
    csv(c.OUT/'manifests/DELIVERABLE_SHA256.csv', pd.DataFrame(manifest))
    files['manifest/DELIVERABLE_SHA256.csv'] = c.OUT/'manifests/DELIVERABLE_SHA256.csv'
    package = c.ROOT/'package/GWLR_UC_01_deliverables.tar.gz'
    digest = archive_and_check(package, files, c)
    maps = c.ROOT/'package/GWLR_UC_01_native_BAYESTAR_maps.tar.gz'
    map_files = {str(p.relative_to(c.ROOT)): p for p in (c.ROOT/'maps').rglob('*.fits.gz')}
    maps_digest = archive_and_check(maps, map_files, c)
    record = dict(state=FINAL, complete_results=True, utc=c.U.now(), report=str(path),
        package=str(package), sha256=digest, native_maps=str(maps), native_maps_sha256=maps_digest,
        native_map_count=len(map_files), historical_hashes_unchanged=True, no_automatic_adoption=True)
    c.write(c.OUT/'contracts/FINAL_DELIVERY.json', record)
    c.write(c.ROOT/'RUN_STATUS.json', record)


def dispatch(root, action, run, split, seed):
    c = initialize(root)
    import uab_scoring as s
    import uab_real as real
    import uab_evaluate as e
    import uab_diagnostics as d
    if action == 'audit':
        c.audit()
        evaluation_contract(c)
        s.tests()
        real.tests()
        real.inventory()
    elif action == 'maps': c.maps(run, ARM, split, 12)
    elif action == 'sky': c.sky_pairs(run, ARM, split)
    elif action == 'infer': c.infer(run, ARM, split, seed)
    elif action == 'calibrate': s.calibrate(run, ARM, seed)
    elif action == 'freeze': freeze(c)
    elif action == 'evaluate':
        e.score_catalog(run, ARM, seed, split)
        e.subcatalogs(run, ARM, seed, split)
    elif action == 'summarize': summarize(c)
    elif action == 'real-inputs':
        real.prepare(run)
        real.sky(run)
    elif action == 'real-rank':
        e.real_ranking(run, ARM)
        real.pe(run)
    elif action == 'resolution': d.injection_resolution(run, ARM, split)
    elif action == 'diagnostics':
        d.resolution_summary()
        d.model_population()
    elif action == 'deliver': deliver(c)
    else: raise ValueError(action)


def complete(root):
    from c_controller import task
    import unified_ab as u
    def step(label, action, run='O3', split='validation', seed=2026091721):
        task(root, label, 'c_finish.py', '--action', action, '--run', run, '--split', split, '--seed', seed)
    step('C_CORE_AUDIT', 'audit')
    for run in u.RUNS:
        step('MAPS_'+run+'_validation', 'maps', run)
        step('SKY_'+run+'_validation', 'sky', run)
        for seed in u.SEEDS:
            for split in ('development', 'validation'):
                step(f'INFER_{run}_{seed}_{split}', 'infer', run, split, seed)
            step(f'CALIBRATE_{run}_{seed}', 'calibrate', run, seed=seed)
    step('C_FINAL_SCORE_FREEZE', 'freeze')
    for run in u.RUNS:
        task(root, 'GENERATE_TEST_'+run, 'unified_ab.py', '--stage', 'generate', '--run', run,
             '--role', 'main', '--split', 'test', '--workers', 8)
        step('MAPS_'+run+'_test', 'maps', run, 'test')
        step('SKY_'+run+'_test', 'sky', run, 'test')
        for seed in u.SEEDS:
            step(f'INFER_{run}_{seed}_test', 'infer', run, 'test', seed)
            for split in ('validation', 'test'):
                step(f'EVALUATE_{run}_{seed}_{split}', 'evaluate', run, split, seed)
    step('C_INJECTION_SUMMARY', 'summarize')
    for run in u.RUNS:
        step('REAL_INPUTS_'+run, 'real-inputs', run)
        for seed in u.SEEDS:
            step(f'INFER_{run}_{seed}_real', 'infer', run, 'real', seed)
        step('REAL_RANK_AND_PE_'+run, 'real-rank', run)
        for split in ('validation', 'test'):
            step('RESOLUTION_'+run+'_'+split, 'resolution', run, split)
    step('C_POSTFREEZE_DIAGNOSTICS', 'diagnostics')
    step('C_REPORT_PACKAGE', 'deliver')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--action', required=True)
    p.add_argument('--run', default='O3')
    p.add_argument('--split', default='validation')
    p.add_argument('--seed', type=int, default=2026091721)
    a = p.parse_args()
    dispatch(a.root, a.action, a.run, a.split, a.seed)
