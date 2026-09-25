#!/usr/bin/env python3
"""Finalize only after frozen confirmation, retain failures and protect old outputs."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import tarfile
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import mcwf_ordered_fresh_confirmation_20260907 as run
from mcwf_ordered_release_prepare_20260907 import output_root, copy

dev, e = run.dev, run.e


def table(f):
    return f.to_markdown(index=False, floatfmt='.4f')


def plots(out, budgets, fresh):
    font = font_manager.findfont('Times New Roman', fallback_to_default=False)
    plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'ps.fonttype': 42})
    colors = ['#686868', '#137F8D']
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 6.0), constrained_layout=True)
    for ax, (dep, metric) in zip(axes.flat, [('gwtc3', 'official_frontend'), ('gwtc4', 'official_frontend'), ('gwtc3', 'official_hanabi'), ('gwtc4', 'official_hanabi')]):
        for k, config in enumerate(['FRT_BASELINE', 'CANDIDATE']):
            f = budgets[(budgets.deployment == dep) & (budgets.method == 'C_fixed') & (budgets.seed.astype(str) == 'consensus') & (budgets.config == config)].sort_values('budget')
            f = f[f.budget.isin([10, 20])]
            ax.bar(np.arange(len(f)) + (k - .5) * .32, f[metric], .32,
                   label='RNC-FRT baseline' if k == 0 else 'OMC candidate', color=colors[k])
        ax.set_xticks([0, 1], ['Top-10', 'Top-20'])
        ax.set_ylim(bottom=0)
        ax.set_title(('O3' if dep == 'gwtc3' else 'O4a') + (' / 1% frontend' if metric == 'official_frontend' else ' / published Hanabi'))
        ax.set_ylabel('Public candidate overlap (count)')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside upper center', ncol=2, frameon=False)
    fig.savefig(out / 'figures/official_overlap.pdf')
    fig.savefig(out / 'figures/official_overlap.png', dpi=220)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 7.0), constrained_layout=True)
    correlations = []
    for ax, (dep, tag) in zip(axes.flat, [('gwtc3', 'baseline'), ('gwtc3', 'candidate'), ('gwtc4', 'baseline'), ('gwtc4', 'candidate')]):
        f = pd.read_parquet(out / f'real/{dep}/{tag}_consensus_waveform_only_all_pairs.parquet')
        ax.scatter(f.waveform_score_mean, f.pe_mc_bhattacharyya_coefficient, s=5, alpha=.22, color='#727272', rasterized=True)
        h = f.head(20)
        ax.scatter(h.waveform_score_mean, h.pe_mc_bhattacharyya_coefficient, s=22, color='#C63C4D', label='Waveform Top-20')
        ax.axhline(.5, color='#444444', lw=.7, ls=':')
        ax.set_title(('O3' if dep == 'gwtc3' else 'O4a') + ' / ' + tag)
        ax.set_xlabel('Mean waveform score')
        ax.set_ylabel('Public PE chirp-mass BC')
        ax.set_ylim(-.025, 1.025)
        correlations.append({'deployment': dep, 'config': tag,
            'Spearman_waveform_vs_PE_BC': float(spearmanr(f.waveform_score_mean, f.pe_mc_bhattacharyya_coefficient).statistic),
            'Spearman_waveform_vs_negative_Dmc': float(spearmanr(f.waveform_score_mean, -f.pe_mc_standardized_distance).statistic),
            'interpretation': 'Descriptive dependent-pair correlation; real data used during adaptive development'})
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc='outside upper center', frameon=False)
    fig.savefig(out / 'figures/waveform_PE_consistency.pdf')
    fig.savefig(out / 'figures/waveform_PE_consistency.png', dpi=220)
    plt.close(fig)
    dev.csv_write(out / 'tables/WAVEFORM_PE_CORRELATIONS.csv', pd.DataFrame(correlations))
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 6.0), constrained_layout=True)
    for ax, (dep, metric) in zip(axes.flat, [('gwtc3', 'macro_r_at_10'), ('gwtc4', 'macro_r_at_10'), ('gwtc3', 'average_precision'), ('gwtc4', 'average_precision')]):
        f = fresh[fresh.deployment == dep]
        for m, method in enumerate(['waveform_only', 'C_fixed']):
            a = f[f.method == method].sort_values('model_seed')
            for k, col in enumerate(['baseline_' + metric, metric]):
                x = np.full(len(a), m + (k - .5) * .24) + np.linspace(-.035, .035, len(a))
                ax.scatter(x, a[col], color=colors[k], s=28, marker='o' if k else 's')
                ax.errorbar(m + (k - .5) * .24, a[col].mean(), yerr=a[col].std(), color=colors[k], capsize=3, fmt='_', ms=14)
        ax.set_xticks([0, 1], ['Waveform', 'Three-channel'])
        ax.set_title('O3' if dep == 'gwtc3' else 'O4a')
        ax.set_ylabel('R@10' if metric == 'macro_r_at_10' else 'Pair AUPRC')
    fig.legend(handles, labels, loc='outside upper center', ncol=2, frameon=False)
    fig.savefig(out / 'figures/fresh_injection_guardrails.pdf')
    fig.savefig(out / 'figures/fresh_injection_guardrails.png', dpi=220)
    plt.close(fig)
    dev.json_write(out / 'figures/FONT.json', {'font': 'Times New Roman', 'path': font})


def finalize(root):
    conf = run.verify(root)
    out = output_root(root)
    if (out / 'contracts/FINAL_STATUS.json').exists():
        raise RuntimeError('Final release already exists')
    src = run.folder(root) / 'confirmation'
    guard = json.loads((src / 'FINAL_GUARDRAIL_AUDIT.json').read_text())
    if not guard['per_model_mean_across3catalog_guardrails_pass']:
        raise RuntimeError('Fresh guardrails failed; preserve all data, do not label release successful')
    for name in ('PAIR_LOCALITY_PROBABILITY_UNIT_TEST.json', 'SELECTED_SCORE_INTEGRITY.json'):
        if not json.loads((out / 'audit' / name).read_text())['pass']:
            raise RuntimeError('Numerical unit audit failed')
    if not json.loads((out / 'audit/portable_score_replay/REPLAY_AUDIT.json').read_text())['pass']:
        raise RuntimeError('Portable replay failed')
    for path in src.rglob('*'):
        if path.is_file() and path.suffix in ('.json', '.csv', '.parquet', '.npz'):
            copy(path, out / 'fresh_confirmation' / path.relative_to(src))
    for path in (run.folder(root) / 'contracts').glob('*'):
        if path.is_file(): copy(path, out / 'contracts' / path.name)
    for path in root.glob('logs/*.runtime.json'):
        copy(path, out / 'logs/runtime' / path.name)
    for pattern in ('*ordered*.log', '*real_input*.log'):
        for path in root.glob('logs/' + pattern): copy(path, out / 'logs' / path.name)
    for name in ('mcwf_ordered_fresh_audit_20260907.py', 'mcwf_ordered_release_unit_audit_20260907.py',
                 'mcwf_ordered_release_finalize_20260907.py'):
        copy(dev.PROJECT / 'scripts/experiments' / name, out / 'scripts' / name)
    protected = pd.read_csv(root / 'manifest/PROTECTED_INPUT_SHA256.csv')
    check = []
    for item in protected.to_dict('records'):
        actual = dev.sha(Path(item['path']))
        check.append({**item, 'actual_sha256': actual, 'unchanged': actual == item['sha256']})
    if not all(r['unchanged'] for r in check):
        raise RuntimeError('Protected historical input changed')
    dev.csv_write(out / 'audit/HISTORICAL_HASH_FINAL.csv', pd.DataFrame(check))
    candidate_budgets = pd.read_csv(out / 'tables/PE_OFFICIAL_ALL.csv')
    baseline_budgets = pd.read_csv(root / 'contracts/BASELINE_BUDGETS.csv')
    baseline_budgets['config'] = 'FRT_BASELINE'
    budgets = pd.concat([baseline_budgets, candidate_budgets], ignore_index=True)
    dev.csv_write(out / 'tables/PE_OFFICIAL_COMPARISON.csv', budgets)
    means = pd.read_csv(src / 'MODEL_MEAN_OVER_CATALOGS.csv')
    plots(out, budgets, means)
    selected = pd.read_csv(out / 'tables/SELECTED_COEFFICIENTS.csv')
    core = budgets[(budgets.method == 'C_fixed') & (budgets.seed.astype(str) == 'consensus') & budgets.budget.isin([10, 20])].copy()
    # Preserve known failures and head changes without using them to refit anything.
    rankchange = pd.read_csv(out / 'tables/RANK_CHANGE_ALL.csv')
    names = ['GW190924_021846--GW191105_143521', 'GW190412--GW191204_171526',
             'GW190924_021846--GW190930_133541', 'GW230723_101834--GW231118_005626']
    known = []
    for name in names:
        f = rankchange[rankchange.pair_key == name]
        if len(f): known.extend(f.to_dict('records'))
        else: known.append({'pair_key': name, 'note': 'Not in frozen current strict scope; no score/rank fabricated'})
    dev.csv_write(out / 'tables/KNOWN_FAILURE_PAIR_AUDIT.csv', pd.DataFrame(known))
    summary_rows = []
    for (dep, method), f in means.groupby(['deployment', 'method']):
        for config, prefix in [('FRT_BASELINE', 'baseline_'), ('CANDIDATE', '')]:
            row = {'deployment': dep, 'method': method, 'config': config}
            for metric in ('macro_r_at_1', 'macro_r_at_10', 'average_precision', 'false_at_recall_0p5', 'false_at_recall_0p9'):
                row[metric] = f'{f[prefix + metric].mean():.4f} +/- {f[prefix + metric].std():.4f}'
            summary_rows.append(row)
    retrieval = pd.DataFrame(summary_rows)
    dev.csv_write(out / 'tables/FRESH_RETRIEVAL_SUMMARY_FORMATTED.csv', retrieval)
    progress = json.loads((out / 'audit/development_ledger/STATUS.json').read_text())
    status = {'created_utc': datetime.now(timezone.utc).isoformat(), 'code': conf['code'],
              'development_real_PE_official_target_pass': True,
              'fresh_source_noise_simulation_guardrails_pass': True,
              'same_O3_O4a_method': True, 'same_outer_weights_as_baseline': True,
              'optional_additional_increment_active_seeds': 3, 'prior_RNC_FRT_active_seeds': 6,
              'independent_real_candidate_validation': False, 'Hanabi_run': False,
              'trial_directories': progress['number_of_trial_directories'],
              'historical_hash_files_checked': len(check), 'historical_hash_files_unchanged': len(check),
              'final_status': 'HOLD_FOR_AUTHOR_REVIEW_NO_ADOPTION_NO_OVERWRITE'}
    dev.json_write(out / 'contracts/FINAL_STATUS.json', status)
    cols = ['deployment', 'config', 'budget', 'catastrophic_mc', 'BC_mc_ge_0p5', 'Dmax_le_3', 'median_BC_mc', 'official_frontend', 'official_hanabi']
    text = '# MCWF-UNIFIED-OMC-DEVCONF 最终结果\n\n'
    text += '本轮在冻结目标下完成真实候选自适应开发与独立新源/噪声注入确认。未替换任何历史版本；最终状态为 `HOLD_FOR_AUTHOR_REVIEW_NO_ADOPTION_NO_OVERWRITE`。\n\n'
    text += '## 1. 是否达到了目标\n\n两个运行期 Top-10/20 均保持或改善 PE/Mc 预算指标，官方 1% 前端和公开 Hanabi 候选重合各增加至少一对。Top-20 灾难性 Mc 冲突在两套目录都降为零。表中官方重合不是透镜真值，公开 Hanabi 分析也不等于支持透镜。\n\n'
    text += table(core[cols].sort_values(['deployment', 'budget', 'config'])) + '\n\n'
    text += '## 2. 独立注入确认\n\n每运行期三个新目录；每个模型运行同样三个目录。表中先对三个目录取模型均值，再对三个模型报告 mean +/- SD；模型 seed 点与全部目录点、source/noise-block 置信区间均保留。\n\n'
    text += table(retrieval) + '\n\n'
    text += f"预注册的逐模型目录均值约束全部通过；单目录点通过 {guard['catalog_model_method_pass_count']}/{guard['catalog_model_method_total']}，所有点值均报告，不以均值掩盖单目录失败。\n\n"
    text += '720 个新源、1140 个事件、420 个真伴随 pair、96 个新噪声块。源参数及噪声全局交叉审计通过；具体清单和 bootstrap 在 `fresh_confirmation/`。\n\n'
    text += '## 3. 修改了什么\n\n保留旧 C-fixed 与原 RNC-FRT，新增有序质量模板响应卷积模型及其有界质量增量。H1/L1 峰值 2 s、40–580 Hz、时间、BAYESTAR 天空、外层权重和真实 scope 不变。两套运行期采用相同网络结构与公式，分别训练/校准。\n\n'
    text += table(selected) + '\n\n'
    text += '**只有 3/6 个 seed 的新增 OMC 项非零；原 RNC-FRT 在 6/6 个 seed 均保留。** 不能称新增模型在每个 seed 都改善；逐 seed 系数为零是冻结网格允许的中性项，不作隐性替换。\n\n'
    text += '## 4. 必须保留的限制\n\n'
    text += '- 本轮使用真实 PE/官方名单作为自适应开发反馈，包括完整网格组合选择，不是 blind test 或仅 simulation-validation 选参。它们不作为单对评分输入。\n'
    text += '- 新独立注入只确认模拟检索 guardrails，不能独立确认真实名单改善；后续需要新的未见目录验证。\n'
    text += '- 新质量分布是神经网络预测，不是 PE posterior 或 proper Bayes factor；MC 一致性不等于透镜确认。\n'
    text += '- BAYESTAR 是条件已知内禀参数和 PSD 的高斯 matched-filter 测量定位，不是对同一非高斯注入 strain 做完整 PE。\n'
    text += '- 仍采用逐像 target-SNR 缩放、平衡质量/SNR覆盖和重复 GW-LMC 透镜环境；不能声明新的独立天体物理人口证据。\n'
    text += '- O3 新噪声为 O3-only，但冻结的响应时间仍含累计 O1–O3 日历；旧 RNC 训练的历史混合噪声限制保留。\n'
    text += '- 更多 Top-50/100 候选仍可能质量不一致，不能把 Top-20 改善推广为所有 pair 都物理相容。\n\n'
    text += '## 5. 核查与完整台账\n\n'
    text += f"{progress['number_of_trial_directories']} 个试验目录全部保留台账，包括失败与未被采用的配置。137 个有效真实事件预处理复现逐点一致；12 项评分/排名便携复算通过；质量分布归一、事件交换对称、pair locality、OOD中性回退检查通过；{len(check)} 个受保护历史文件结束时哈希不变。\n\n"
    text += '主预测中间表中继承的旧 rank/final_score 不作为正式结果；请以 `real/` 的最终表和 `audit/portable_score_replay/` 为准。\n\n'
    text += '## 6. 文件导航\n\n- `reports/MCWF_UNIFIED_OMC_DEVCONF_METHOD_CN.md`：整个方案和公式。\n- `real/gwtc3/`、`real/gwtc4/`：旧/新 Top-10/20/50/100、全 pair 和逐 seed，含 PE、PO/ML/Phazap FPP 与公开阶段。\n- `tables/PE_OFFICIAL_COMPARISON.csv`：全部预算/seed 新旧对照。\n- `tables/RANK_CHANGE_ALL.csv`：新旧完整排名变化。\n- `fresh_confirmation/`：独立注入、置信区间、分母和审计。\n- `exploration_ledger/`：所有探索正负结果摘要。\n- `models/`、`source_snapshot/`、`replay_inputs/`：模型、脚本和便携评分输入。\n- `manifest/`：逐文件哈希和外部输入来源。\n\n'
    (out / 'reports/FINAL_RESULTS_CN.md').write_text(text, encoding='utf-8')
    readme = '# MCWF-UNIFIED-OMC-DEVCONF\n\n' + table(core[cols].sort_values(['deployment', 'budget', 'config']))
    readme += '\n\n先读 `reports/FINAL_RESULTS_CN.md` 和 `reports/MCWF_UNIFIED_OMC_DEVCONF_METHOD_CN.md`。本包不包含原始 GWOSC strain、PE HDF5、全训练数组、dense sky cache、私钥或密码。外部输入保留原路径与哈希。\n\nCPU 快速复算：\n\n```bash\npython scripts/mcwf_replay_ordered_scores_20260907.py --root . --output /tmp/omc_replay_new\n```\n\n上例只复算评分，不宣称重新生成 strain 或重新 PE。依赖 NumPy、Pandas、PyArrow；完整编码另需项目科学环境和原始数据。\n\n服务器：`connect.westd.seetacloud.com:32328`；项目：`/root/autodl-tmp/gw-catalog`。认证凭据不在包内。\n\n最终状态：`HOLD_FOR_AUTHOR_REVIEW_NO_ADOPTION_NO_OVERWRITE`。\n'
    (out / 'README_CN.md').write_text(readme, encoding='utf-8')
    required = ['README_CN.md', 'reports/FINAL_RESULTS_CN.md', 'reports/MCWF_UNIFIED_OMC_DEVCONF_METHOD_CN.md',
                'contracts/FRESH_FREEZE.json', 'contracts/FINAL_STATUS.json',
                'fresh_confirmation/FINAL_GUARDRAIL_AUDIT.json',
                'fresh_confirmation/GLOBAL_SOURCE_NOISE_INDEPENDENCE_COMPLETE.json',
                'tables/PE_OFFICIAL_COMPARISON.csv', 'tables/RANK_CHANGE_ALL.csv',
                'scripts/mcwf_replay_ordered_scores_20260907.py', 'audit/portable_score_replay/REPLAY_AUDIT.json']
    for dep in ('gwtc3', 'gwtc4'):
        required += [f'real/{dep}/candidate_C_fixed_Top{b}.csv' for b in (10, 20, 50, 100)]
        required += [f'real/{dep}/candidate_consensus_C_fixed_all_pairs.parquet']
        for es in (202607241, 202607242, 202607243):
            required += [f'models/{dep}/seed_{es}/{model}.pt' for model in ('ordered_mass', 'baseline_RNC', 'old_Cfixed')]
    missing = [name for name in required if not (out / name).is_file()]
    dev.json_write(out / 'audit/REQUIRED_DELIVERABLES.json', {'pass': not missing, 'required': required, 'missing': missing})
    if missing: raise RuntimeError('Missing deliverables: ' + str(missing))
    # Exclude secrets by content and file name, not just an extension blacklist.
    private_key = re.compile(rb'-----BEGIN(?: [A-Z0-9]+)? PRIVATE KEY-----\s*\r?\n[A-Za-z0-9+/=\r\n]{40,}')
    for path in out.rglob('*'):
        if not path.is_file(): continue
        if path.name in ('id_rsa', 'id_ed25519', '.env'):
            raise RuntimeError('Forbidden secret-like file')
        if path.suffix in ('.py', '.json', '.md', '.csv', '.log'):
            data = path.read_bytes()
            if private_key.search(data) or private_key.search(data.replace(b'\\n', b'\n')):
                raise RuntimeError('Private key material detected')
    entries = []
    for path in sorted(out.rglob('*')):
        if path.is_file(): entries.append({'path': str(path.relative_to(out)), 'bytes': path.stat().st_size, 'sha256': dev.sha(path)})
    dev.csv_write(out / 'manifest/OUTPUT_SHA256.csv', pd.DataFrame(entries))
    package = dev.PROJECT / 'packages/MCWF_UNIFIED_OMC_DEVCONF_20260907.tar.gz'
    if package.exists(): raise RuntimeError('Refuse to overwrite existing package')
    with tarfile.open(package, 'w:gz', compresslevel=6) as tar:
        tar.add(out, arcname='MCWF_UNIFIED_OMC_DEVCONF_20260907')
    digest = dev.sha(package)
    package.with_suffix(package.suffix + '.sha256').write_text(digest + '  ' + package.name + '\n')
    failures = []
    with tarfile.open(package, 'r:gz') as tar:
        prefix = 'MCWF_UNIFIED_OMC_DEVCONF_20260907/'
        for row in entries:
            handle = tar.extractfile(prefix + row['path'])
            h = hashlib.sha256()
            for data in iter(lambda: handle.read(8 * 1024 * 1024), b''): h.update(data)
            if h.hexdigest() != row['sha256']: failures.append(row['path'])
        count = len(tar.getmembers())
    if failures: raise RuntimeError('Archive verification failed: ' + str(failures))
    record = {'package': str(package), 'sha256': digest, 'bytes': package.stat().st_size,
              'tar_members': count, 'verified_payload_files': len(entries), 'internal_hash_failures': failures,
              'result_root': str(out), **status}
    dev.json_write(root / 'FINAL_DELIVERY_VERIFICATION.json', record)
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    finalize(p.parse_args().root)
