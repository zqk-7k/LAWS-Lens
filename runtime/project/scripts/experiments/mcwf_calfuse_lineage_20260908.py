#!/usr/bin/env python3
"""Resolve retained synthetic IDs to original GW-LMC event IDs; read-only."""
import argparse
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import calfuse as c


def main(root):
    source=next((c.PROJECT.parent/'GW-LMC/2.5PLUS/BBH/Any_Detected_SNR1').glob('*_SourceParams.csv'))
    sources=pd.read_csv(source)
    checks=[];ids=[];hashes=[{'path':str(source),'sha256':c.dev.sha(source)}];times=[]
    for dep in c.DEPS:
        for seed in c.SEEDS:
            origin=c.PROJECT/f'results/real_noise_injection_v7_peak2s_formal_20260722/{dep}/seed_{seed}'
            meta_path=origin/'data/real_noise_injections/compact_injection_metadata.parquet'
            meta=pd.read_parquet(meta_path)
            hashes.append({'path':str(meta_path),'sha256':c.dev.sha(meta_path)})
            plans={}
            for split in ('validation','test'):
                p=pd.read_csv(root/f'audit/{dep}_{seed}_{split}_event_plan.csv')
                q=p.merge(meta[['family','sample_index','gwlmc_row']],left_on=['family','source_index'],
                    right_on=['family','sample_index'],validate='many_to_one')
                q['global_source_id']=sources.iloc[q.gwlmc_row.to_numpy(int)].event_id.to_numpy()
                q['deployment']=dep;q['seed']=seed;plans[split]=q
                ids.append(q[['deployment','seed','split','idx','family','source_index','system_id','gwlmc_row','global_source_id','parent_noise_bank']])
                old_path=origin/'results'/('fusion_validation_pairs_v7.parquet' if split=='validation' else 'fusion_heldout_test_pairs_v7.parquet')
                hashes.append({'path':str(old_path),'sha256':c.dev.sha(old_path)})
                old=pd.read_parquet(old_path).set_index(['idx_i','idx_j'])
                now=c.read(dep,seed,split)
                reference=old.loc[list(zip(now.old_idx_i,now.old_idx_j))]
                exact=all(np.array_equal(now[k].to_numpy(),reference[k].to_numpy()) for k in ('time_score','delta_t_days'))
                assert exact
                times.append({'deployment':dep,'seed':seed,'split':split,
                    'time_and_delay_exactly_inherited_from_v7':True,'source_path':str(old_path),'pairs':len(now)})
            v,t=plans['validation'],plans['test']
            overlap=sorted(set(v.global_source_id)&set(t.global_source_id))
            fold=pd.read_csv(root/f'audit/{dep}_{seed}_FIT_TUNE_PLAN.csv').merge(v[['idx','global_source_id']],on='idx',validate='one_to_one')
            fit=set(fold.loc[fold.calibration_fold==0,'global_source_id'])
            tune=set(fold.loc[fold.calibration_fold==1,'global_source_id'])
            assert not overlap and not fit&tune
            checks.append({'deployment':dep,'seed':seed,'validation_test_global_source_overlap':len(overlap),
                'fit_tune_global_source_overlap':len(fit&tune),
                'validation_global_source_count':v.global_source_id.nunique(),
                'validation_legacy_system_count':v.system_id.nunique(),
                'source_identity':'GW-LMC SourceParams.event_id via exact compact metadata gwlmc_row',
                'not_audited':'complete upstream encoder training/proposal prior overlap'})
    c.csv(root/'audit/GLOBAL_SOURCE_ID_REPLAY.csv',checks)
    c.csv(root/'audit/GLOBAL_SOURCE_EVENT_MANIFEST.csv',pd.concat(ids,ignore_index=True))
    c.csv(root/'audit/TIME_SCORE_LINEAGE_REPLAY.csv',times)
    c.csv(root/'manifest/LINEAGE_INPUT_SHA256.csv',hashes)
    note='''# 源ID和时间分数来源补充审计

本补充不修改配置或分数。六个deployment/seed的validation/test与fit/tune均已从实际compact metadata的gwlmc_row追溯到GW-LMC SourceParams.event_id，全球源ID交集均为0。详见GLOBAL_SOURCE_ID_REPLAY.csv；上游全部encoder训练和先验拟合集是否重合不在这项审计证明范围内。

当前注入pair的time_score与delta_t_days逐项等于原v7 pair文件中的相应值，详见TIME_SCORE_LINEAGE_REPLAY.csv。因此本轮没有把其他Phase0.5或二维时间实验的输出悄悄并入当前C-fixed/OMC分支。v7时间人口的独立系统数及其与训练总体的关系需要另外完整追溯，不能仅因其他分支修复过就假定本分支自动继承修复。

O3注入计划13%--19%的合成事件时间不落在正式O3a/O3b边界内；这与当前真实strict62的官方O3范围不同。本轮保留原时间分数，只报告此范围差异。
'''
    (root/'reports/GLOBAL_ID_AND_TIME_LINEAGE_CN.md').write_text(note,encoding='utf-8')
    target=root/'scripts'/Path(__file__).name
    if not target.exists():shutil.copy2(__file__,target)
    c.dump(root/'audit/LINEAGE_AUDIT_COMPLETE.json',{'global_retained_source_isolation_pass':True,
        'time_columns_frozen_v7_replay_pass':True,'no_scores_changed':True})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
