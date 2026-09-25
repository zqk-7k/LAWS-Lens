#!/usr/bin/env python3
"""Complete-pilot diagnostics; never reads real rankings or PE fields."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys
import numpy as np
import pandas as pd
import psutil
from scipy.stats import spearmanr

P=Path('/root/autodl-tmp/gw-catalog')
sys.path.insert(0,str(P/'scripts/experiments'))
import mcwf_shared_profile_coherence_pilot_20260909 as p


def main(root):
    gate=json.loads((root/'contracts/PILOT_GATE.json').read_text())
    frame=pd.read_parquet(root/'tables/PAIR_RESULTS.parquet')
    if len(frame)!=480 or frame.status.ne('COMPLETE').any():
        raise RuntimeError('Diagnostics require the complete pilot')
    checks=[]
    for row in pd.read_csv(root/'manifest/INPUT_SHA256.csv').to_dict('records'):
        checks.append({**row,'current_sha256':p.n.sha(Path(row['path']))})
    if any(row['sha256']!=row['current_sha256']for row in checks):
        raise RuntimeError('A frozen simulation input changed')
    p.n.write_csv(root/'audit/INPUT_HASH_RECHECK.csv',checks)
    rows=[]
    for (dep,fold,kind),group in frame.groupby(['deployment','fold','kind']):
        r={'deployment':dep,'fold':fold,'kind':kind,'pairs':len(group),
           'replayed_profile_max_difference':group.profile_replay_max_difference.max(),
           'fraction_at_least_one_shared_optimizer_converged':group.any_shared_converged.mean(),
           'fraction_all_shared_optimizers_converged':np.mean([all(x['success']for x in a)for a in group.shared_optimizer_runs]),
           'fraction_both_independent_refinements_converged':np.mean([all(x['success']for x in a)for a in group.independent_refinements]),
           'Spearman_deficit_vs_logMc_gap':spearmanr(group.deficit,group.profile_logMc_gap).statistic,
           'Spearman_deficit_vs_minimum_power':spearmanr(group.deficit,group.minimum_independent_power).statistic}
        for name in ('deficit','relative_deficit','seconds','evaluation_points'):
            values=group[name].to_numpy(float)
            for label,quantile in [('median',.5),('p90',.9),('p99',.99)]:
                r[f'{name}_{label}']=np.quantile(values,quantile)
            r[f'{name}_max']=values.max()
        gain=np.asarray(group.independent_gain_from_stored.to_list())
        r['independent_search_gain_median']=np.median(gain)
        r['independent_search_gain_max']=np.max(gain)
        if kind=='true':
            estimated=np.exp(np.asarray(group.shared_parameters.to_list())[:,0])
            errors=np.log(estimated/group.mc_truth_i.to_numpy(float))
            r['shared_fit_logMc_bias']=np.mean(errors)
            r['shared_fit_logMc_MAE']=np.mean(np.abs(errors))
        rows.append(r)
    p.n.write_csv(root/'tables/OPTIMIZATION_AND_POPULATION_DIAGNOSTICS.csv',rows)
    versions={}
    for library in ('numpy','scipy','pandas','pycbc','lalsuite','scikit-learn','torch'):
        try:
            versions[library]=importlib.metadata.version(library)
        except importlib.metadata.PackageNotFoundError:
            versions[library]='not_available'
    monitor=[json.loads(line)for line in (root/'logs/RESOURCE_MONITOR.jsonl').read_text().splitlines()]
    resource={'UTC':p.n.utc(),'logical_cpus':psutil.cpu_count(),'RAM_GiB':psutil.virtual_memory().total/1024**3,
        'cgroup_cpu_max':Path('/sys/fs/cgroup/cpu.max').read_text().strip(),
        'versions':versions,'profile_computation_device':'CPU','workers':20,
        'observed_peak_summed_RSS_GiB':max(x['summed_RSS_GiB']for x in monitor),
        'observed_minimum_disk_free_GiB':min(x['disk_free_GiB']for x in monitor),
        'summed_pair_wall_seconds':float(frame.seconds.sum()),
        'pair_runtime_p50':float(frame.seconds.median()),'pair_runtime_p90':float(frame.seconds.quantile(.9)),
        'pair_runtime_max':float(frame.seconds.max()),
        'monitor_limit':'Starts after setup; summed RSS can double-count shared pages. Not an isolated-process peak hardware counter.'}
    p.n.write_json(root/'audit/RESOURCES_AND_ENVIRONMENT.json',resource)
    dependencies=pd.read_csv(root/'manifest/RUNTIME_DEPENDENCIES.csv')
    original_runtime=pd.read_csv(root/'audit/R49_RUNTIME_HASH_UNCHANGED.csv')
    p.n.write_json(root/'audit/PROVENANCE_METADATA_CLARIFICATION.json',{
        'UTC':p.n.utc(),'R49_verified_unique_source_files':len(original_runtime),
        'R50_snapshot_unique_source_files':len(dependencies),
        'clarification':'The historical87 count is module references before source-path deduplication;83 unique R49 source files were actually checked. The v2 provenance timing text previous87-source should read83unique source files.',
        'credential_guard_v1':'Matched its own bare private-key-marker test literal;failed v1 script retained. V2 requires a complete base64 PEM block and passed. No discovered credential was copied.',
        'scientific_results_changed':False})
    figure(root,frame)
    summary=pd.read_csv(root/'tables/PILOT_GATE_COMPARISON.csv')
    lines=['# R50 共享内禀参数波形拟合 Pilot 结果','',
        '本轮是独立模拟 feasibility pilot，不是完整检索结果，也没有读取真实候选 PE 或官方 FPP。',
        '历史 NODUP、时间、天空和融合权重未改变。当前目标尚未因本 pilot 自动达成。','',
        f"## 计算状态：{gate['gate']}",'',
        '全部 480 个预选 pair 已计算；普通与困难假对分别报告。',
        '模型使用同一共享参数规则，O3/O4a 分别校准。所有采样种子和 ridge 网格均保留。','',
        '| 运行期 | 假对类型 | 新减旧 tune logloss | 非劣 |',
        '|---|---|---:|---|']
    for row in summary.to_dict('records'):
        lines.append(f"| {row['deployment']} | {row['null_kind']} | {row['new_minus_separate']:.8f} | {row['nonworse']} |")
    lines+=['','logloss 越低越好。这里的 ROC-AUC 和 balanced AP 来自指定 pilot 采样，不能冒充完整 catalog AUPRC。',
        '逐 source bootstrap 只反映固定模型、固定噪声和当前抽样的波动，不包括多次探索的选择不确定性。','',
        '## 数值与科学限制','',
        '相同源、不同源、事件交换及投影能量界的合成测试单独保存。',
        '局部有限预算优化不能保证全局最大；优化器是否收敛逐 pair 报告。',
        '低维对齐自旋波形、自由探测器相位振幅和预处理噪声使该值只能作为经验波形特征，不能称为完整 PE 或 Bayes factor。',
        'fit/tune 共享该研究计划的开发背景，不是新盲测试。没有根据真实目标 pair 的结果选择本 pilot 配置。','',
        '## 后续边界','',
        ('Pilot 通过只允许研究扩大模拟验证；仍须完成注入检索、逐 seed、PE 与官方阶段的完整共同目标检查。'
         if gate['gate']=='PASS'else
         'Pilot 未通过，不扩展此版本，不用该统计量更新真实排名；原结果和失败尝试均保留。'),
        '','HOLD_FOR_AUTHOR_REVIEW_NO_ADOPTION_NO_OVERWRITE','']
    (root/'reports/SHARED_PROFILE_PILOT_RESULT_CN.md').write_text('\n'.join(lines),encoding='utf-8')
    shutil.copy2(__file__,root/'scripts/shared_profile_diagnostics.py')
    p.n.write_json(root/'contracts/PILOT_DIAGNOSTICS_COMPLETE.json',{
        'UTC':p.n.utc(),'gate':gate['gate'],'input_hash_failures':0,'real_PE_read':False,
        'historical_outputs_modified':False,'goal_achieved':False,'status':p.n.STATUS})


def figure(root,frame):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'serif','font.size':9})
    fig,axes=plt.subplots(1,2,figsize=(8,3.2),layout='constrained')
    for ax,dep,title in zip(axes,p.n.DEPS,('O3','O4a')):
        for kind,label,color in [('true','Companions','#007F73'),('random_null','Random null','#BB5566'),('hard_null','Hard null','#4466AA')]:
            values=np.sort(np.log1p(frame.loc[frame.deployment.eq(dep)&frame.fold.eq(1)&frame.kind.eq(kind),'deficit']))
            ax.step(values,np.arange(1,len(values)+1)/len(values),where='post',label=label,color=color)
        ax.set(xlabel='log(1 + shared-fit deficit)',ylabel='Empirical cumulative fraction',title=title,ylim=(0,1.02))
        ax.spines[['top','right']].set_visible(False)
    axes[1].legend(frameon=False,loc='lower right',fontsize=8)
    folder=root/'figures';folder.mkdir(exist_ok=True)
    fig.savefig(folder/'SHARED_PROFILE_PILOT_DISTRIBUTIONS.pdf')
    fig.savefig(folder/'SHARED_PROFILE_PILOT_DISTRIBUTIONS.png',dpi=180)
    plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    main(parser.parse_args().root)
