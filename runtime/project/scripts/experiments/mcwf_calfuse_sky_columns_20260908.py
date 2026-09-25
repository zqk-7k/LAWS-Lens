#!/usr/bin/env python3
"""Identify inherited sky columns that do not describe the active BAYESTAR map."""
import argparse
from pathlib import Path
import shutil
import numpy as np
import calfuse as c


def main(root):
    rows=[]
    for dep in c.DEPS:
        for seed in c.SEEDS:
            for split in ('validation','test'):
                f=c.read(dep,seed,split)
                logb=f.sky_raw_log_bf.to_numpy(float)
                native=f.sky_bayes_factor.to_numpy(float)
                valid=native>0
                err=np.abs(np.log(native[valid])-logb[valid])
                legacy=f.sky_raw_overlap.to_numpy(float)
                ok=legacy>0
                olderr=np.abs(np.log(12*512**2)+np.log(legacy[ok])-logb[ok])
                rows.append({'deployment':dep,'seed':seed,'split':split,'pairs':len(f),
                    'active_score_column':'sky_raw_log_bf','active_nside':512,
                    'sky_bayes_factor_positive_pairs':int(valid.sum()),
                    'BF_log_identity_max_error':float(err.max()),
                    'legacy_sky_raw_overlap_identity_mismatch_fraction':float((olderr>1e-4).mean()),
                    'legacy_sky_raw_overlap_log_identity_error_median':float(np.median(olderr)),
                    'legacy_sky_score_differs_from_active_fraction':float((np.abs(f.sky_score.to_numpy(float)-logb)>1e-4).mean()),
                    'used_by_current_fusion':'ONLY sky_raw_log_bf; verified score replay',
                    'ranking_changed':False})
    c.csv(root/'audit/LEGACY_SKY_COLUMN_SEMANTICS.csv',rows)
    note='''# 遗留天空列的语义审计

当前评分函数明确读取`sky_raw_log_bf`，本轮全部分数复算也使用此列。

注入pair表同时继承了早期版本的`sky_score`、`sky_raw_overlap`和`sky_cosine_overlap`。这些列没有全部随BAYESTAR替换而更新，不能把它们当作当前BAYESTAR概率图的相应统计量。例如应有的同分辨率恒等式`log(B_sky)=log(12*512^2)+log(raw_overlap)`，在继承的`sky_raw_overlap`列上通常不成立。

这是一项数据表语义/来源风险，不是当前三通道排名实际用了错误天空列的证据。`sky_raw_log_bf`才是现行主分数；当前分布图和排名均读取它。本轮不覆盖旧表，只增加此审计。后续代码必须明确指定有效列，不得用泛称`sky_score`自动猜测。真实PE表中的同名列需按其自身provenance检查，不能把注入表的结论直接套过去。

本轮没有更改任何天空值，没有把旧标量重命名成新的地图结果。
'''
    (root/'reports/LEGACY_SKY_COLUMN_SEMANTICS_CN.md').write_text(note,encoding='utf-8')
    target=root/'scripts'/Path(__file__).name
    if not target.exists():shutil.copy2(__file__,target)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
