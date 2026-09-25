#!/usr/bin/env python3
"""Summarize verified execution state without claiming final experiment metrics."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


def main(root):
    sys.path.insert(0, str(root/'scripts'))
    import unified_ab as u
    source = pd.read_parquet(root/'plans/source_population.parquet')
    rows = []
    for (role, split, family), f in source.groupby(['role', 'split', 'family']):
        rows.append({'role': role, 'split': split, 'family': family, 'waveform_parents': len(f),
                     'independent_lens_groups': f.global_source_id.nunique(), 'max_group_multiplicity': int(f.groupby('global_source_id').size().max())})
    pd.DataFrame(rows).to_csv(root/'reports/POPULATION_EFFECTIVE_COUNTS.csv', index=False)
    sky = pd.read_csv(root/'pilot/BAYESTAR_PILOT.csv')
    timing = sky.groupby(['run', 'arm']).seconds.agg(['count', 'median', 'max'])
    timing.to_csv(root/'reports/PILOT_TIMINGS.csv')
    paths = list((root/'scripts').glob('*.py'))
    u.write(root/'contracts/TRAINING_EXECUTION_FREEZE.json', {'utc': u.now(),
        'scripts': {str(p.relative_to(root)): u.sha(p) for p in paths},
        'training_model_seeds': u.SEEDS, 'models_per_arm_run': 3,
        'training_only_gate': 'no test evaluation until full scoring adapter audited and frozen',
        'incomplete_stages': ['complete waveform inference/calibration', 'time and full sky calibration',
                             'held-out evaluation', 'real PE and official-stage audits', 'final package'],
        'official_or_PE_used_for_selection': False})
    text = '''# GWLR-UAB-01 启动记录

这是两份新实验的启动与预检记录，不是最终 Recall/PE 结果。

## 实验代号

- UAB-A-NEUTRAL：统一 O3/O4a/O4b 后移除 SNR 档位和标签的关联。
- UAB-B-CUE：同一批源、噪声和 SNR 值，保留历史类别关联。

B 组是偏置对照，不能因指标更高而升级。A/B 均使用逐像目标 SNR，均不宣称是完整 response-derived 透镜总体。BAYESTAR 已修复 SI 质量单位，但仍使用已知内禀参数和高斯触发量误差，未宣称完整 BBH PE。

## 已完成

- 三运行期严格日历；32 个母块隔离的噪声块/运行期，20/6/6 划分。
- 1800 个主波形源及 4096/512 个辅助训练/验证波形父源的计划。
- 全局 lens/source group 不跨 train/validation/test。
- 辅助 train 可复用 train 环境；辅助 validation 与主 validation 的环境分离。独立环境数详见 POPULATION_EFFECTIVE_COUNTS.csv。
- A/B 目标 SNR 多重集合完全一致，只改变分配；各模型 seed 共用数据。
- 源表、透镜表、像表 event ID 对齐及低层自旋接口 SI 单位测试通过。
- 三运行期配对物理波形 pilot 和 BAYESTAR pilot；输入哈希及历史保护自检通过。

## 接下来的实际任务

控制器依次生成训练/validation 数据，训练短窗 encoder、RNC、ordered-Mc、16 秒 multirate 和条件 eta/chi 模块，三个模型 seed、两个 arm、三个运行期。各阶段退出码、日志和失败记录写入 contracts/tasks 与 RUN_STATUS.json。

目前控制器的自动边界是波形组件训练；完整评分适配器、时间/天空校准、locked test、真实 PE/官方审计尚待完成，不能把训练完成写成整个实验完成。test 生成和评分仍由最终评分配置冻结门槛阻止。

## 当前边界

历史结果、模型、论文、候选表不覆盖。本目录所有数值是新实验产物；以前看过的原始噪声不被重新命名为全新未见观测。固定 500 个 190 事件子目录共享母目录，不算 500 次独立实验；同时保留三运行期 450 事件母目录结果。

最终交付必须包含 Recall/AP/F50/F90、Top-B、置信区间、PE、官方 FPP/阶段、失败结果与哈希压缩包。当前没有这些最终指标，不能宣布目标达成。
'''
    (root/'reports/UAB_STARTUP_CN.md').write_text(text, encoding='utf-8')
    print(json.dumps({'root': str(root), 'bayestar_pilot_events': len(sky), 'complete_results': False}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True)
    main(p.parse_args().root)
