#!/usr/bin/env python3
"""Post-analysis integrity checks and complete round summary; no score selection."""
import argparse
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import calfuse as c


def independent_guard_replay(root):
    frozen = json.loads((root/'contracts/CONFIGURATIONS_FROZEN.json').read_text())
    assert c.dev.sha(root/'configs/SELECTED_CONFIGURATIONS.json') == frozen['sha256']
    configs = json.loads((root/'configs/SELECTED_CONFIGURATIONS.json').read_text())
    checks = []
    for conf in configs:
        dep, seed, method = conf['deployment'], conf['seed'], conf['method']
        for split in ('validation', 'test', 'real'):
            src = c.read(dep, seed, split)
            spec = conf['calibration']
            x = c.features(src, spec['family'])
            eta = spec['intercept'] + np.sum(
                (x-spec['mu'])/spec['sd']*spec['coef'], axis=1)
            active = np.asarray(spec['coef']) > 1e-8
            ood = (((x < np.asarray(spec['minimum'])-1e-7) |
                    (x > np.asarray(spec['maximum'])+1e-7)) & active).any(1)
            old = src.waveform_score.to_numpy(float)
            calibrated = np.clip(eta, -spec['score_cap'], spec['score_cap'])
            calibrated[ood] = old[ood]
            proposal = np.minimum(old, calibrated)
            if conf['mode'] == 'NEGATIVE-PRESERVE':
                proposal = np.where(old < 0, proposal, calibrated)
            expected = old + conf['gamma']*(proposal-old)
            w = c.frozen_weights(dep, seed)
            assert np.array_equal(w, np.asarray(conf['weights']))
            total = w[0]*expected + w[1]*src.time_score + w[2]*src.sky_raw_log_bf
            if split == 'real':
                path = root/f'results/{method}/{dep}/seed_{seed}_C_fixed.parquet'
                out = pd.read_parquet(path).set_index('pair_key').loc[src.pair_key].reset_index()
            else:
                path = root/f'results/{method}/{dep}/seed_{seed}/{split}_pairs.parquet'
                out = pd.read_parquet(path)
            assert np.array_equal(src.time_score.to_numpy(), out.time_score.to_numpy())
            assert np.array_equal(src.sky_raw_log_bf.to_numpy(), out.sky_raw_log_bf.to_numpy())
            wf_error = float(np.max(abs(expected-out.waveform_score.to_numpy())))
            score_error = float(np.max(abs(np.asarray(total)-out.final_score.to_numpy())))
            assert wf_error < 1e-10 and score_error < 1e-10
            assert np.all(expected[old < 0] <= old[old < 0]+1e-12)
            if conf['mode'] == 'MINIMUM':
                assert np.all(expected <= old+1e-12)
            if conf['gamma'] == 0:
                assert np.array_equal(expected, old)
            if split != 'real':
                assert np.array_equal(src.is_true_pair, out.is_true_pair)
                reference = c.dev.BASE.full_metrics(src, np.asarray(total))
                replay = c.fast_metrics(out, out.final_score.to_numpy(float))
                for key in ('macro_r_at_1', 'macro_r_at_10', 'average_precision',
                            'false_at_recall_0p5', 'false_at_recall_0p9'):
                    assert abs(reference[key]-replay[key]) < 1e-10
            checks.append(dict(deployment=dep, seed=seed, method=method, split=split,
                waveform_max_error=wf_error, fusion_max_error=score_error,
                time_sky_and_weights_exact=True, old_negative_evidence_preserved=True))
    c.csv(root/'audit/INDEPENDENT_GUARD_SCORE_REPLAY.csv', checks)
    c.dump(root/'audit/FINAL_SUPPLEMENTAL_VERIFICATION.json', dict(
        all_pass=True, score_replays=len(checks), frozen_config_hash_unchanged=True,
        test_real_and_validation_replayed=True, no_new_model_selection=True))


def aggregate(main, guard):
    metrics = []
    budgets = []
    for code, root, fusion_name in (('CALFUSE-01', main, 'fusion'),
                                    ('GUARD-02', guard, 'C_fixed')):
        frame = pd.read_csv(root/'tables/RETRIEVAL_PER_SEED.csv')
        frame['study'] = code
        frame['configuration'] = code+':'+frame.method
        metrics.append(frame)
        frame = pd.read_csv(root/'tables/PE_OFFICIAL_BUDGETS.csv')
        frame = frame[(frame.seed.astype(str) == 'consensus') &
                      (frame.method == fusion_name)].copy()
        frame['study'] = code
        frame['configuration'] = code+':'+frame.config
        budgets.append(frame)
    m, b = pd.concat(metrics, ignore_index=True), pd.concat(budgets, ignore_index=True)
    c.csv(main/'tables/ROUND_ALL_RETRIEVAL_PER_SEED.csv', m)
    c.csv(main/'tables/ROUND_ALL_CONSENSUS_BUDGETS.csv', b)
    rows = []
    for (dep, method), q in b.groupby(['deployment', 'configuration'], sort=False):
        q = q.set_index('budget')
        ref = b[(b.deployment == dep) &
                (b.configuration == 'CALFUSE-01:OMC-FIXED')].set_index('budget')
        t = m[(m.deployment == dep) & (m.configuration == method) &
              (m.split == 'test') & (m['mode'] == 'fusion')]
        base = q.loc[[10, 20]]
        old = ref.loc[[10, 20]]
        retained = bool((base.catastrophic_mc <= old.catastrophic_mc).all())
        for col in ('BC_mc_ge_0p5', 'median_BC_mc', 'Dmax_le_3'):
            retained &= bool((base[col] >= old[col]-1e-12).all())
        official_gain = all((base[k] >= old[k]).all() and base[k].sum() > old[k].sum()
                            for k in ('official_frontend', 'official_hanabi'))
        r = dict(deployment=dep, configuration=method,
            r10_mean=t.macro_r_at_10.mean(), r10_sd=t.macro_r_at_10.std(),
            AP_mean=t.average_precision.mean(), AP_sd=t.average_precision.std(),
            F50_mean=t.false_at_recall_0p5.mean(), F90_mean=t.false_at_recall_0p9.mean(),
            PE_retained_Top10_and20=retained,
            official_and_Hanabi_increased_Top10_and20=bool(official_gain),
            descriptive_minimum_goal=bool(retained and official_gain))
        for k in (10, 20, 50, 100):
            for col in ('BC_mc_ge_0p5', 'median_BC_mc', 'catastrophic_mc',
                        'Dmax_le_3', 'official_frontend', 'official_hanabi'):
                r[f'Top{k}_{col}'] = q.loc[k, col]
        rows.append(r)
    summary = pd.DataFrame(rows)
    c.csv(main/'tables/ROUND_RESULT_SUMMARY.csv', summary)
    both = summary.groupby('configuration').descriptive_minimum_goal.all()
    c.dump(main/'audit/ROUND_MINIMUM_GOAL_AUDIT.json', dict(
        status=c.STATUS, both_run_minimum_goal_achieved=bool(both.any()),
        passing_configurations=both[both].index.tolist(),
        definition='Retain Top10/20 Mc counts,median BC,Dmax and catastrophic count; increase both official frontend and Hanabi without losing either budget',
        note='Descriptive extra check,not changed preregistered gates or model selection; simulations not needed to reject when PE/official already fail'))
    return m, b, summary


def draw(main, guard, metrics, summary):
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
        'pdf.fonttype': 42, 'ps.fonttype': 42})
    order = summary.configuration.drop_duplicates().tolist()
    order.remove('CALFUSE-01:OMC-FIXED')
    order.insert(0, 'CALFUSE-01:OMC-FIXED')
    colors = {'gwtc3': '#2166ac', 'gwtc4': '#b2182b'}
    labels = [x.replace('CALFUSE-01:', '01 / ').replace('GUARD-02:', '02 / ') for x in order]
    fig, axes = plt.subplots(1, 3, figsize=(15, 9), sharey=True, layout='constrained')
    fields = [('Top20_BC_mc_ge_0p5', 'Mc BC >= 0.5 in Top-20'),
              ('Top20_official_frontend', 'Official 1% frontend in Top-20'),
              ('Top20_official_hanabi', 'Published Hanabi overlap in Top-20')]
    for ax, (col, title) in zip(axes, fields):
        for dep, offset in (('gwtc3', -.13), ('gwtc4', .13)):
            f = summary[summary.deployment == dep].set_index('configuration').loc[order]
            ax.scatter(f[col], np.arange(len(order))+offset, color=colors[dep],
                       marker='o' if dep == 'gwtc3' else 's', s=25,
                       label='O3' if dep == 'gwtc3' else 'O4a')
            ax.axvline(f[col].iloc[0], color=colors[dep], ls='--', lw=.8, alpha=.6)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('Number of pairs (not confirmed lenses)')
        ax.set_xlim(0, 21)
        ax.set_xticks([0, 5, 10, 15, 20])
        ax.grid(axis='x', alpha=.2)
    axes[0].set_yticks(np.arange(len(order)), labels, fontsize=8)
    axes[0].invert_yaxis()
    axes[0].legend(loc='lower left', bbox_to_anchor=(0, 1.025), ncol=2, frameon=False)
    fig.suptitle('All configurations: frozen-score exploratory audits; dashed lines = retained baseline', fontsize=12)
    for root in (main, guard):
        (root/'figures').mkdir(exist_ok=True)
        for suffix in ('png', 'pdf'):
            fig.savefig(root/f'figures/fig_round_all_PE_official.{suffix}', dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(13, 9), sharey=True, layout='constrained')
    for ax, dep in zip(axes, c.DEPS):
        for y, method in enumerate(order):
            f = metrics[(metrics.deployment == dep) & (metrics.configuration == method) &
                        (metrics.split == 'test') & (metrics['mode'] == 'fusion')]
            f = f.sort_values('seed')
            ax.scatter(f.macro_r_at_10, y+np.linspace(-.15, .15, len(f)),
                       s=20, color=colors[dep], alpha=.65)
            ax.scatter(f.macro_r_at_10.mean(), y, s=55, marker='|', color='black')
        base = summary[(summary.deployment == dep) &
                       (summary.configuration == 'CALFUSE-01:OMC-FIXED')].r10_mean.iloc[0]
        ax.axvline(base, color=colors[dep], ls='--', lw=.8)
        ax.set_title('O3' if dep == 'gwtc3' else 'O4a')
        ax.set_xlabel('Injection R@10; points = frozen deployment seeds')
        ax.grid(axis='x', alpha=.2)
    axes[0].set_yticks(np.arange(len(order)), labels, fontsize=8)
    axes[0].invert_yaxis()
    fig.suptitle('Reused injection test: no new encoder training or independent blinded confirmation', fontsize=12)
    for root in (main, guard):
        for suffix in ('png', 'pdf'):
            fig.savefig(root/f'figures/fig_round_all_retrieval.{suffix}', dpi=180)
    plt.close(fig)


def reports(main, guard, summary):
    view = summary.copy()
    view['R10 mean +/- SD'] = view.apply(lambda r: f'{r.r10_mean:.4f} +/- {r.r10_sd:.4f}', axis=1)
    for label, col in (('Mc', 'BC_mc_ge_0p5'), ('catastrophic', 'catastrophic_mc'),
                       ('official 1%', 'official_frontend'), ('Hanabi', 'official_hanabi')):
        view[label+' Top10/20'] = view.apply(lambda r: f'{int(r[f"Top10_{col}"])}/{int(r[f"Top20_{col}"])}', axis=1)
    cols = ['deployment', 'configuration', 'R10 mean +/- SD', 'AP_mean',
            'Mc Top10/20', 'catastrophic Top10/20', 'official 1% Top10/20', 'Hanabi Top10/20']
    lines = ['# MCWF-CALFUSE：两运行期联合优化完整交付', '',
        f'状态：`{c.STATUS}`', '',
        '## 结果与目标', '',
        '本轮11项整体校准/融合对照和8项保留旧负证据对照均完成，共19个配置（包括保留基线）。'
        '尚未达到O3/O4a同时保持或提高PE/Mc一致性并增加官方前端及公开Hanabi候选重合的目标。'
        '这是有限探索已完成，不是优化目标已完成；没有升级、替换历史模型、候选表或论文。', '',
        '保留基线OMC-FIXED：O3 Top10/20 Mc BC>=0.5为6/12，官方1%为5/9，Hanabi为5/7；'
        'O4a对应10/19、7/15、5/11。不能将不同方案在不同运行期的局部最好值拼成一个“统一成功”版本。', '',
        '## 修改了什么', '',
        '- 没有重训或更换encoder；原始一维time和Nside512 sky数值逐元素不变。',
        '- CALFUSE-01：对完整旧waveform分数做整体/联合校准，再比较固定或validation选取的外层权重。',
        '- GUARD-02：在前项完成后的独立敏感性探索中，保留旧waveform负证据，调节校准更新幅度；外层权重也不变。',
        '- O3/O4a采用相同算法、网格和选择规则，各运行期、各seed独立使用模拟validation选参。',
        '- 真实PE/官方数据只在本轮配置冻结后联表。历史模型和真实目录曾被适应性使用，不能把本轮称为独立盲测。', '',
        '## 全部结果', '', view[cols].to_markdown(index=False, floatfmt='.4f'), '',
        'Mc为探测器系chirp-mass后验Bhattacharyya系数；灾难性不一致定义保持旧审计：BC<0.1或标准化Mc距离>5。'
        '1%前端为公开PO/ML或PO/Phazap FPP低于0.01，不是单对透镜后验概率。Hanabi列仅表示公开表重合，本轮没有运行Hanabi，也不代表确认透镜。', '',
        '## 为什么未升级', '',
        '例如CALFUSE-01 AFFINE-FIXED把O3 Top10 Mc由6增至7、官方1%由5增至7，'
        '但Top20 Mc由12减至11，并新增1个灾难性Mc不一致，Hanabi由7减至6。'
        'GUARD-02避免削弱旧负证据后，O3的JOINT-NEGATIVE-PRESERVE-VF50可把Top10 Mc由6增至7，'
        '但官方/Hanabi并未增加，且O4a Top10 Mc由10减至8。全部失败对照也保留。', '',
        '额外按“PE保持即可”的最低目标检查仍没有两个运行期同时通过的配置；详见ROUND_MINIMUM_GOAL_AUDIT.json。'
        '这个描述性检查没有改变原预定义Gate，也没有用于重新选择参数。', '',
        '## 本轮发现的上游问题', '',
        '1. O3模拟validation/test中约13%--19%的事件GPS落在正式O3a/O3b之外；实际真实目录已是O3 strict62。'
        '本轮确认这些时间和分数由v7原表原样继承。该结论仅针对合成GPS，不等于噪声文件一定不是O3，也未完整证明旧时间先验来源。',
        '2. BAYESTAR注入天空不是旧旋转模板，但触发量由已知源参数、分配PSD和target SNR进行高斯匹配滤波测量模拟得到，'
        '没有从同一份非高斯注入应变恢复触发量。真实目录则是公开PE后验；统一Nside与公式不等于测量误差模型相同。',
        '3. 注入pair表仍保留历史sky_score、sky_raw_overlap等旧列。当前排序明确只用sky_raw_log_bf，'
        '与当前sky_bayes_factor的对数关系误差约2.2e-16；已独立重放分数，不是本轮误用了旧列。后续应使用数据字典，避免再次误读。',
        '4. 严格隔离后每seed拟合只有14--22个独立透镜系统，许多网格点选权频率低。'
        '大量相关pair不能替代独立校准源。已按真正GW-LMC全局event_id复查保留的fit/tune和validation/test，交集均为0；'
        '这不替代上游encoder训练与外部prior的完整隔离审计。', '',
        '## 后续优先级', '',
        '继续叠加波形项或按官方名单调权没有充分依据。建议下一独立协议先做O3-only时间exposure对照，'
        '再用实际注入应变恢复matched-filter触发量做少量BAYESTAR匹配验证，同时扩大source/noise-disjoint校准集。'
        '这两项可能改变time/sky输入条件，因此必须另建版本，不能暗中改本轮分数，也不能承诺修正后官方重合必然提高。', '',
        '## 数据与复现', '',
        f'- 服务器：connect.westd.seetacloud.com，SSH端口32328。',
        f'- CALFUSE-01完整目录：`{main}`。',
        f'- GUARD-02完整目录：`{guard}`。',
        '- tables/ROUND_RESULT_SUMMARY.csv：全部方案跨运行期数值与Top10/20/50/100预算。',
        '- results/：全部逐seed/consensus pair分数、贡献、PE、官方FPP和阶段，以及注入validation/test表。',
        '- tables/SYSTEM_BOOTSTRAP.csv：10000次系统级检索CI、1000次source-block pair CI；这些区间不校正历史适应性选择。',
        '- tables/WEIGHT_STABILITY.csv：CALFUSE-01的100次validation source-bootstrap选权稳定性。',
        '- reports/ALL_REAL_TOP10_CN.md和GUARD-02的同名文件：所有方案Top10，不按美观程度删选。',
        '- audit/和manifest/：独立分数重放、输入哈希、源隔离、时间/天空来源、逐文件SHA256。',
        '- reports/DATA_DICTIONARY_AND_REPRODUCTION_CN.md：权威列及历史保留列，依赖的旧项目路径。',
        '复现依赖原项目冻结评分表和辅助模块；紧凑包不包括strain、模型大缓存或凭证。'
        '包内脚本和输入SHA256支持重现本轮分数实验，但不是脱离历史输入即可重新生成整个引力波项目的独立数据集。', '',
        '只读历史结果哈希复核均通过；最终压缩包SHA256与内部逐文件核对结果放在packages目录，打包后不再修改其内容。']
    (main/'reports/ROUND_FINAL_CN.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    tops = []
    for path in sorted(guard.glob('results/*/*/consensus_C_fixed_top100_pe_official.csv')):
        f = pd.read_csv(path)
        f.insert(0, 'configuration', path.parent.parent.name)
        f.insert(0, 'deployment', path.parent.name)
        tops.append(f)
    c.csv(guard/'tables/ALL_REAL_CONSENSUS_TOP100.csv', pd.concat(tops, ignore_index=True))
    out = ['# GUARD-02：全部方案Top10，仅作冻结后审计', '']
    columns = ['consensus_rank', 'event_i', 'event_j', 'final_score_mean',
        'waveform_contribution_mean', 'time_contribution_mean', 'sky_contribution_mean',
        'pe_mc_bhattacharyya_coefficient', 'pe_dmax_intrinsic',
        'official_po_or_ml_fpp_below_0p01', 'official_any_pair_resolved_hanabi_overlap']
    for frame in tops:
        out += [f'## {frame.deployment.iloc[0]} {frame.configuration.iloc[0]}', '',
            frame.head(10)[[k for k in columns if k in frame]].to_markdown(index=False, floatfmt='.5f'), '']
    (guard/'reports/ALL_REAL_TOP10_CN.md').write_text('\n'.join(out), encoding='utf-8')
    for root in (main, guard):
        (root/'README_CN.md').write_text(
            '# MCWF-CALFUSE 交付入口\n\n'
            +f'状态：`{c.STATUS}`\n\n'
            +'本轮未达到O3/O4a的PE/Mc与官方候选共同改善目标，保留旧版本，不自动升级。\n\n'
            +f'综合报告：{main}/reports/ROUND_FINAL_CN.md\n\n'
            +'当前目录包含完整配置、逐seed结果、注入recall/pair指标、真实PE/官方阶段、图和审计。'
            +'原始time/sky不变，encoder没有重训。01可重选外层权重，02固定外层权重。\n', encoding='utf-8')
    shutil.copy2(main/'reports/ROUND_FINAL_CN.md', guard/'reports/ROUND_FINAL_CN.md')
    for name in ('ROUND_RESULT_SUMMARY.csv', 'ROUND_ALL_CONSENSUS_BUDGETS.csv', 'ROUND_ALL_RETRIEVAL_PER_SEED.csv'):
        shutil.copy2(main/'tables'/name, guard/'tables'/name)
    shutil.copy2(main/'audit/ROUND_MINIMUM_GOAL_AUDIT.json', guard/'audit/ROUND_MINIMUM_GOAL_AUDIT.json')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--main', type=Path, required=True)
    parser.add_argument('--guard', type=Path, required=True)
    args = parser.parse_args()
    independent_guard_replay(args.guard)
    metrics, _, summary = aggregate(args.main, args.guard)
    draw(args.main, args.guard, metrics, summary)
    reports(args.main, args.guard, summary)
    for root in (args.main, args.guard):
        shutil.copy2(__file__, root/'scripts'/Path(__file__).name)
    print('ROUND_SUMMARY_AND_INDEPENDENT_GUARD_REPLAY_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
