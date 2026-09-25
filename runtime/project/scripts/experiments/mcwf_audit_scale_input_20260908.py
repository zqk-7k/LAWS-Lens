#!/usr/bin/env python3
"""Preserve and mark an invalid scale-context interpretation, not hide it."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

P = Path('/root/autodl-tmp/gw-catalog')
sys.path.insert(0, str(P / 'scripts/experiments'))
import mcwf_temporal_response_20260908 as t

root = P / 'results/mcwf_noise_scale_exploratory_20260908T141327Z'
target = root / 'contracts/POSTRUN_SCALE_INPUT_INVALIDITY.json'
if target.exists():
    raise RuntimeError('Append-only audit already exists')
rows = []
for dep in t.DEPS:
    for path in sorted((root / f'features/{dep}').glob('*.npy')):
        x = np.load(path)
        for d in range(2):
            a = x[:, d]
            rows.append({'deployment': dep, 'file': str(path), 'detector': ('H1', 'L1')[d],
                         'count': len(a), 'mean': float(a.mean()), 'std': float(a.std()),
                         'minimum': float(a.min()), 'maximum': float(a.max())})
t.dev.csv_write(root / 'tables/SCALE_FEATURE_INVALIDITY_AUDIT.csv', pd.DataFrame(rows))
t.dev.json_write(target, {'utc': datetime.now(timezone.utc).isoformat(),
    'scientific_status': 'INVALID_AS_NOISE_SCALE_EXPERIMENT_NOT_ELIGIBLE_FOR_PROMOTION',
    'original_results_retained': True, 'historical_OMC_unchanged': True,
    'cause': 'Both expanded raw2s.npy and make_window_view already apply per-detector zscore. The new feature measured std AFTER that normalization,not before.',
    'observed_variation': 'Mostly ddof correction and float16 quantization,not recoverable physical signal/noise amplitude.',
    'consequence': 'SCALE results cannot support or refute physical noise-scale conditioning. They amount to a continued-model fit with nearly constant or precision-dependent nuisance features.',
    'normalization_disclosure': 'make_window_view additionally fixes each detector peak sign;quadrature power is sign invariant.',
    'next': 'Any correct scale study must reconstruct or read pre-make_window_view data in a new directory and recompute all feature statistics,models and calibration.',
    'no_relabel': 'Keep original contract and outputs with this corrective audit;do not claim the intended noise descriptor was tested.'})
report = '''# 尺度上下文对照的实现审计与纠正

本轮 SCALE 不能作为有效的噪声尺度实验。检查发现 `raw2s.npy` 已经过
`make_window_view` 的逐探测器符号统一和 z-score；部署分支也调用同一个函数。
因此之后计算的标准差不是标准化前的幅度，而主要是 ddof 修正和数值精度差异。

原合同对输入的文字定义不符合实际读取位置。原分数、排名、训练日志和合同全部保留，
但这些结果不得支持“噪声尺度有用/无用”的科学结论，也不得据此升级主方法。
这不是历史 OMC、time 或 sky 文件被修改；错误属于本次新建 SCALE 对照。

后续若重新检验，必须从已记录的源参数、噪声片段、PSD 和缩放重建标准化前数据，
在全新的结果目录中训练、校准与评估。不得用已标准化数据虚构丢失的尺度。

逐列数值见 `tables/SCALE_FEATURE_INVALIDITY_AUDIT.csv`。
原 PE/官方对照仍保存在 `tables/PE_OFFICIAL_BUDGETS.csv`，只是无效机制的观察记录。

HOLD_FOR_AUTHOR_REVIEW_NO_ADOPTION_NO_OVERWRITE
'''
(root / 'reports/SCALE_INPUT_INVALIDITY_CN.md').write_text(report, encoding='utf-8')
print(json.dumps({'audit': str(target), 'invalid_mechanism': True, 'rows': len(rows)}))
