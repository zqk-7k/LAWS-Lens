#!/usr/bin/env python3
"""Post-freeze explanation of the critical pair; never selects a score rule."""
import os
os.environ['MPLBACKEND'] = 'Agg'
import argparse
import json
from pathlib import Path
import shutil
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

P = Path('/root/autodl-tmp/gw-catalog')
sys.path.insert(0, str(P / 'scripts/experiments'))
import mcwf_nodup_conditional_catalog_20260910 as app
n, io = app.n, app.io


def main(root, pilot):
    receipt = root / 'audit/CRITICAL_PAIR_EXPLANATION_COMPLETE.json'
    if receipt.exists():
        raise RuntimeError('Critical-pair audit is immutable')
    if not (root / 'contracts/FROZEN_COMPUTATIONS_COMPLETE.json').exists():
        raise RuntimeError('Postfreeze complete results required')
    configs = {(c['deployment'], c['seed']): c for c in n.selections(root) if c['method'] == app.NEW}
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'],
                        'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    rows, identities = [], []
    for seed, color in zip(n.SEEDS, ['#007c91', '#b8464b', '#7065a4']):
        f = pd.read_parquet(root / f'results/{app.NEW}/gwtc3/seed_{seed}/real/fusion_all_pairs.parquet')
        row = f[f.pair_key.eq(app.audit.KEY)]
        if len(row) != 1:
            raise RuntimeError('Critical pair missing')
        r = row.iloc[0]
        c = configs['gwtc3', seed]['conditional_model']
        if c['arm'] != app.pilot.ARMS[0] or c['no_update']:
            raise RuntimeError('Expected selected linear candidate for this explanation')
        z = float(r.conditional_NODUP_reference)
        power, deficit = float(r.shared_profile_minimum_power), float(r.shared_profile_deficit)
        slopes, intercept = np.asarray(c['raw_slopes']), c['raw_intercept']
        parts = np.array([intercept, slopes[0] * z, slopes[1] * np.log(power),
                          -slopes[2] * np.log1p(max(deficit, c['deficit_support'][0]))])
        value = app.pilot.apply(np.array([z]), np.array([power]), np.array([deficit]), c)[0][0]
        if abs(parts.sum() - value) > 1e-10 or abs(value - r.waveform_score) > 1e-10:
            raise RuntimeError('Critical score decomposition does not reconstruct')
        data = {'seed': seed, 'rank': int(r['rank']), 'NODUP_waveform': z, 'minimum_power': power,
            'shared_deficit': deficit, 'intercept': parts[0], 'NODUP_term': parts[1],
            'power_term': parts[2], 'deficit_term': parts[3], 'new_waveform': value,
            'waveform_contribution': r.waveform_contribution, 'time_contribution': r.time_contribution,
            'sky_contribution': r.sky_contribution, 'final_score': r.final_score}
        rows.append(data)
        xx = np.linspace(c['deficit_support'][0], min(c['deficit_support'][1], 40.), 250)
        yy = app.pilot.apply(np.full(len(xx), z), np.full(len(xx), power), xx, c)[0]
        axes[0].plot(xx, yy, color=color, label=str(seed))
        axes[0].scatter([deficit], [value], color=color, s=35)
        axes[1].plot(range(4), parts, marker='o', color=color, label=str(seed))
        identities.append({'seed': seed, 'max_reconstruction_error': abs(parts.sum() - r.waveform_score),
                           'PE_or_official_used_for_selection': False})
    axes[0].axhline(0., color='#777777', lw=.7)
    axes[0].set_xlabel('Shared-fit deficit (projection units)')
    axes[0].set_ylabel('Frozen R67 waveform score')
    axes[0].set_title('Actual NODUP score and power held fixed', fontsize=11, loc='left')
    axes[1].axhline(0., color='#777777', lw=.7)
    axes[1].set_xticks(range(4), ['Intercept', 'NODUP term', 'Power term', 'Deficit term'], rotation=12)
    axes[1].set_ylabel('Terms within the single waveform classifier')
    axes[1].set_title('Terms sum to the reported waveform score', fontsize=11, loc='left')
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc='upper center', ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, .91))
    for ext in ('png', 'pdf'):
        fig.savefig(root / f'figures/R67_CRITICAL_PAIR_SCORE_AUDIT.{ext}', dpi=180)
    plt.close(fig)
    io.csv(root / 'tables/CRITICAL_PAIR_SCORE_DECOMPOSITION.csv', rows)
    report = '# 关键pair残余高分审计\n\n'
    report += '这是配置冻结后的解释，不是新的调权或候选选择。公开Mc BC约0.091665，未改变PE。\n\n'
    report += pd.DataFrame(rows).to_markdown(index=False, floatfmt='.6g') + '\n\n'
    report += '同一pair的共享拟合损失和强度在三个模型中一致，但NODUP预测分数与模拟校准系数不同，因此剩余波形分数不同。R67不是一条“只要D超过某个数就否决”的规则。模型202607242仍给出正的波形支持，最终rank7，未满足所有模型离开Top10的要求。\n\n'
    report += '图中仅在固定该pair输入时改变D以显示已经冻结函数的形状；没有根据公开PE、rank10分界、官方成员身份选择系数。不得把这张图中的某个D点当作新的手工阈值。\n\n'
    report += '通过模拟非劣性仍不等于真实部署成功。后续必须检验共同波形拟合与神经预测之间的条件关系及校准误差，不得通过手工压低该pair或恢复旧重复Mc/q、总分混合来满足目标。\n'
    with (root / 'reports/R67_CRITICAL_PAIR_SCORE_AUDIT_CN.md').open('x', encoding='utf-8') as stream:
        stream.write(report)
    shutil.copy2(__file__, root / 'scripts/conditional_critical_pair_audit.py')
    io.write(receipt, {'UTC': io.utc(), 'goal_achieved': False, 'postfreeze_audit_only': True,
        'units': identities, 'score_or_ranking_modified': False, 'script_sha256': io.sha(Path(__file__))})
    print('CRITICAL_PAIR_SCORE_AUDIT_COMPLETE', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--pilot', type=Path, required=True)
    a = parser.parse_args()
    main(a.root, a.pilot)
