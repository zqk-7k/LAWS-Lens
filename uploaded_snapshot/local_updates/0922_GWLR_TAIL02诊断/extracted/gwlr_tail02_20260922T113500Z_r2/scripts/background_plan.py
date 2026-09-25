"""Plan a source/noise-disjoint official O4a replay; do not transfer it to O3/O4b."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def main(root):
    source=root/'inputs/official_O4a_population.json'
    data=json.loads(source.read_text())['population']
    parent={k:k for k in data}
    def find(k):
        while parent[k]!=k:
            parent[k]=parent[parent[k]];k=parent[k]
        return k
    def union(a,b):
        parent[find(b)]=find(a)
    chunks={};intervals={'H1':[],'L1':[]}
    for key,event in data.items():
        for detector,info in event['psd_params'].items():
            c=info['chunk_info'];token=(detector,Path(info['psd_file']).name)
            if token in chunks:union(key,chunks[token])
            chunks[token]=key
            intervals[detector].append((float(c['segment_start_time']),float(c['segment_end_time']),key))
    # Connected components prevent a shared or overlapping detector noise chunk crossing splits.
    for detector,items in intervals.items():
        items.sort()
        for index,(start,end,key) in enumerate(items):
            for a,b,other in items[:index]:
                if a<end and start<b:union(key,other)
    components={}
    for key in data:components.setdefault(find(key),[]).append(key)
    ordered=sorted(components.values(),key=lambda keys:digest('TAIL02:official-noise:'+':'.join(sorted(keys))))
    split={};membership={}
    for index,keys in enumerate(ordered):
        role='fit' if index < int(.6*len(ordered)) else 'validation'
        for key in keys:split[key]=role;membership[key]=index
    rows=[];needed=[]
    for key,event in sorted(data.items()):
        p=event['signal_generation_params']
        row=dict(official_uid=key,split=split[key],noise_component=membership[key],
            mass_1_detector=p['mass_1'],mass_2_detector=p['mass_2'],
            mass_1_source=p['mass_1_source'],mass_2_source=p['mass_2_source'],redshift=p['redshift'],
            luminosity_distance_mpc=p['luminosity_distance'],waveform=p['wf_approximant'],
            original_optimal_network_snr=event['optimal_snr']['network'],
            selected_O4a_population_not_unconditional_proposal=True,
            proposal_probability=None,exact_original_noise_PSD_available=True)
        for detector,info in event['psd_params'].items():
            file=Path(info['psd_file']);available=file.is_file()
            row['exact_original_noise_PSD_available'] &= available
            chunk=info['chunk_info']
            needed.append(dict(official_uid=key,split=split[key],detector=detector,
                start_gps=chunk['segment_start_time'],end_gps=chunk['segment_end_time'],
                original_psd_path=str(file),original_psd_exists=available,
                action='Obtain author cleaned strain/PSD, or reconstruct and separately validate a new GWOSC DQ/PSD pipeline',
                public_raw_strain_is_not_BayesWave_cleaned_strain=True))
        rows.append(row)
    frame=pd.DataFrame(rows)
    np.testing.assert_allclose(frame.mass_1_detector,frame.mass_1_source*(1+frame.redshift),rtol=1e-12)
    for name,f in [('OFFICIAL_O4A_REPLAY_PLAN',frame),('OFFICIAL_O4A_MISSING_INPUTS',pd.DataFrame(needed))]:
        path=root/f'results/{name}.csv'
        if path.exists():raise RuntimeError('No overwrite')
        f.to_csv(path,index=False,encoding='utf-8-sig')
    candidates=[]
    fits=pd.read_csv(root/'results/gpd_fits.csv')
    for run in ('O3','O4a','O4b'):
        bg=pd.read_csv(root/f'inputs/{run}_fit_scores.csv').mean_S.to_numpy()
        head=pd.read_csv(root/f'results/{run}_real_Top50_empirical_preserved.csv').head(10)
        models=fits[(fits.run==run)&(fits.model=='mean_S')]
        for index,row in head.iterrows():
            score=float(row.final_score_POSITIVE)
            endpoints=models.endpoint.dropna().to_numpy(float)
            candidates.append(dict(run=run,table_position=index+1,pair_key=row.pair_key,score=score,
                empirical_exceedances=int((bg>=score).sum()),
                beyond_fit_maximum=bool(score>bg.max()),
                beyond_preregistered_predictive_check_grid=bool(score>np.quantile(bg,.995)),
                finite_GPD_endpoints_below_score=int((endpoints<=score).sum()),
                fitted_thresholds=len(models),candidate_GPD_FPP_published=False))
    pd.DataFrame(candidates).to_csv(root/'results/REAL_TOP10_EXTRAPOLATION_SUPPORT.csv',index=False,encoding='utf-8-sig')
    status=dict(stage='MATCHED_BACKGROUND_INPUT_PREPARATION_ONLY',new_scored_events=0,
        O4a_official_events=len(frame),noise_connected_components=len(components),
        maximum_component_events=max(map(len,components.values())),
        split_events=frame.groupby('split').size().to_dict(),
        complete_original_PSD_events=int(frame.exact_original_noise_PSD_available.sum()),
        O3_and_O4b_selected_population_transfer_allowed=False,
        no_importance_sampling_weights_claimed=True,
        needed_before_real_FPP=['New run-conditioned null proposal/selection',
            'Independent real-noise DQ/PSD source acquisition',
            'Public PE versus injection sky/network transfer or matched sky PE pilot'])
    with (root/'contracts/MATCHED_BACKGROUND_INPUT_STATUS.json').open('x') as f:
        json.dump(status,f,indent=2)
    text='''
## 后续真实匹配背景入口

已输出官方 O4a 254 事件的重建计划和缺失输入清单，按共享或重叠的 H1/L1
噪声片段连通分量分组后划分 fit/validation，不能按 pair 随机切分。
这是输入准备，不是新增已评分背景；官方文件内的私有清洗后 PSD 路径不能当作
本服务器已有数据。原始 GWOSC strain 不等同于作者 BayesWave 清洗后的 strain。
该已检测 O4a 样本不作为 O3/O4b 总体，不假定未知的 proposal weights 为 1。

额外核查了实际 Top-10 是否超出预先检验的背景分数范围。
阈值稳定与预测筛查只测试背景 q98/q99/q99.5；它们通过不代表更远的
候选极端尾部外推通过。见 REAL_TOP10_EXTRAPOLATION_SUPPORT.csv。

目前没有新模拟背景对、没有新的正式真实 FPP 或候选排名。
下一步需先完成匹配天空处理的代表性 PE 对照，或明确限定为 BAYESTAR
模拟部署下的条件 FPP；后者不能被包装成真实 GWTC 的校准显著性。
'''
    with (root/'reports/TAIL_AND_BACKGROUND_AUDIT_CN.md').open('a',encoding='utf-8') as f:f.write(text)
    print(json.dumps(status),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    main(p.parse_args().root)
