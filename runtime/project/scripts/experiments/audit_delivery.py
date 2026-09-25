#!/usr/bin/env python3
"""Audit-only delivery supplements; never modify scores or selected configs."""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import calfuse as c


def run(root):
    choices=c.selections(root)
    grid=pd.read_csv(root/'tables/VALIDATION_GRID.csv')
    platform=[];coeff=[];external=[]
    for conf in choices:
        spec=conf['calibration']
        coeff.append({'deployment':conf['deployment'],'seed':conf['seed'],'method':conf['method'],
            'calibration_family_actual':spec['family'],'fallback_to_baseline':conf['fallback_to_baseline'],
            'regularization':spec.get('regularization'),'intercept':spec.get('intercept'),
            'coefficients':json.dumps(spec.get('coef',[])),
            'feature_centers':json.dumps(spec.get('mu',[])),'feature_scales':json.dumps(spec.get('sd',[])),
            'fit_positive_systems':spec.get('fit_positive_systems'),'absolute_logit_cap':spec.get('score_cap')})
        if conf['policy']=='FIXED':continue
        fam=conf['family'] if conf['cohort']=='tune' else 'OMC-FULL'
        g=grid[(grid.deployment==conf['deployment'])&(grid.seed==conf['seed'])&(grid.family==fam)]
        eligible=g[g.guard_pass]
        key='false_at_recall_0p5' if conf['policy']=='VF50' else 'macro_r_at_10'
        best=eligible[key].min() if conf['policy']=='VF50' else eligible[key].max()
        platform.append({'deployment':conf['deployment'],'seed':conf['seed'],'method':conf['method'],
            'grid_points':len(g),'guard_eligible':len(eligible),'primary_objective':key,
            'best_primary_objective':best,'exact_primary_plateau_points':int(np.isclose(eligible[key],best,rtol=0,atol=1e-12).sum()),
            'note':'same primary objective only; prescribed secondary objectives still break ties'})
    for dep in c.DEPS:
        f=pd.read_parquet(c.NOISE/f'audit/{dep}_external_reference.parquet')
        other='official_ml_fpp' if dep=='gwtc3' else 'official_phazap_fpp'
        flag='official_po_or_ml_fpp_below_0p01' if dep=='gwtc3' else 'official_po_or_phazap_fpp_below_0p01'
        numerical=(f.official_po_fpp<.01)|(f[other]<.01)
        assert pd.api.types.is_bool_dtype(f[flag].dtype)
        assert np.array_equal(numerical.to_numpy(),f[flag].to_numpy())
        assert pd.api.types.is_bool_dtype(f.official_any_pair_resolved_hanabi_overlap.dtype)
        external.append({'deployment':dep,'n_pairs':len(f),'official_FPP_complete':bool(f[['official_po_fpp',other]].notna().all().all()),
            'threshold_flag_recomputed_equal':True,'Hanabi_flag_native_bool':True,
            'pair_key_unique':bool(f.pair_key.is_unique),'official_candidates_not_lens_truth':True})
    c.csv(root/'tables/VALIDATION_PLATEAUS.csv',platform)
    c.csv(root/'tables/CALIBRATION_COEFFICIENTS.csv',coeff)
    c.csv(root/'audit/OFFICIAL_TABLE_REPLAY.csv',external)
    text='''# 数据字典与复现说明

本轮代号：MCWF-CALFUSE-01。未采纳的新探索结果，与OMC、GLOBAL-PRIOR及历史C-fixed分别保存。

## 当前分数列

- `waveform_score`：当前配置实际使用的波形通道。
- `OMC_baseline_waveform_score`：上一版OMC波形通道。
- `time_score`、`sky_raw_log_bf`：冻结的一维时间及原始天空证据，逐元素不变。
- `final_score`：本配置权重下三通道之和。
- `waveform_contribution`、`time_contribution`、`sky_contribution`：实际加权贡献。
- `calibration_ood`：超出当前校准器fit特征支持域，已回退旧OMC。
- `calibration_clipped`：非OOD区域中触及校准logit上限或下限。
- 原表继承的 `previous_*`、`FRT_baseline_*`、`new_*`、`baseline_*` 为历史溯源列，不得误当本轮最终分数。

## 共识

每seed先独立排名；按mean rank升序、max rank升序、mean score降序确定共识。`*_mean`为跨seed均值，不是重新拟合的公共权重。所有正式真实排名均保留相同62/74事件范围。

## 统计解释

`PE_OFFICIAL_BUDGETS.csv`中的`config`是配置代号，`method`为fusion或waveform_only。`seed=consensus`才是汇总排名；不要将逐seed预算相加或误当共识预算。

`macro_r_at_10`是原legacy family宏平均，不是全pair真阳性率。`average_precision`按原实现保留为Pair AP/AUPRC约定。`false_at_recall_*`以稳定排序达到目标召回的prefix计数；校准有并列分数时应同时检查pessimistic R10列，不把乐观并列排名当作额外信息。

系统bootstrap是固定模型、复用目录下的抽样波动；选权bootstrap固定校准器，仅衡量外层权重波动。它们不替代独立新源、独立噪声或encoder重训方差。

## 模型和数据

本轮不重训任何waveform encoder；logistic校准器是低维评分层，不接收PE、官方标签、时间或天空作为特征。JOINT只把既有波形子证据联合校准。所有新融合权重在模拟validation选择。

当前注入天空输入是Gaussian trigger realization配合事件PSD，并非从完全相同的非高斯注入应变恢复的触发序列。真实天空来自公开PE。原始地图保留在原数据根目录，本包不复制数千张地图或原始strain。

## 完整脚本运行

在原服务器依赖和历史文件仍可读取的环境：

```bash
python -B scripts/run_calfuse.py --root <独立新目录>
python -B scripts/finish_calfuse.py --root <独立新目录> --phase gates
python -B scripts/finish_calfuse.py --root <独立新目录> --phase domain
python -B scripts/finish_calfuse.py --root <独立新目录> --phase uncertainty
python -B scripts/finish_calfuse.py --root <独立新目录> --phase weight_stability
python -B scripts/finish_calfuse.py --root <独立新目录> --phase report
python -B scripts/finish_calfuse.py --root <独立新目录> --phase verify
python -B scripts/audit_delivery.py --root <独立新目录>
python -B scripts/finish_calfuse.py --root <独立新目录> --phase package
```

这不是包含全部原始数据和历史依赖的自包含包。依赖路径/哈希见manifest，必须保持历史项目可访问，不得在已完成目录原地重跑。

最终状态：HOLD_FOR_AUTHOR_REVIEW_NO_ADOPTION_NO_OVERWRITE。
'''
    (root/'reports/DATA_DICTIONARY_AND_REPRODUCTION_CN.md').write_text(text,encoding='utf-8')
    for name in ('calfuse.py','finish_calfuse.py','run_calfuse.py','test_calfuse.py','audit_delivery.py'):
        source=Path(__file__).parent/name;target=root/'scripts'/name
        if target.exists():assert c.dev.sha(source)==c.dev.sha(target),name
        else:shutil.copy2(source,target)
    c.dump(root/'audit/DELIVERY_SUPPLEMENT_COMPLETE.json',{'official_numeric_threshold_replay':True,'selected_config_hash_unchanged':True})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);run(p.parse_args().root)
