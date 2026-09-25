#!/usr/bin/env python3
"""Build a verifiable delivery only after all frozen confirmation gates pass."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='2'
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mcwf_development_20260905 as dev
import mcwf_encoder_body_20260906 as body
import mcwf_new_confirmation_20260906 as fresh

METRICS=('macro_r_at_1','macro_r_at_10','average_precision','false_at_recall_0p5','false_at_recall_0p9')
LABELS={'macro_r_at_1':'R@1','macro_r_at_10':'R@10','average_precision':'Pair AP',
    'false_at_recall_0p5':'F50','false_at_recall_0p9':'F90'}


def table(frame,columns):
    return frame[columns].to_markdown(index=False,floatfmt='.4f')


def summarized(frame,groups):
    rows=[]
    for keys,g in frame.groupby(groups,sort=False):
        if not isinstance(keys,tuple):
            keys=(keys,)
        row=dict(zip(groups,keys))
        for metric in METRICS:
            row[metric+'_mean']=float(g[metric].mean())
            row[metric+'_SD']=float(g[metric].std(ddof=1))
            row[LABELS[metric]]=f'{g[metric].mean():.4f} +/- {g[metric].std(ddof=1):.4f}'
        rows.append(row)
    return pd.DataFrame(rows)


def figures(root,reused,fresh_rows,pe):
    plt.rcParams.update({'font.family':'Times New Roman','font.size':8,'axes.spines.top':False,
        'axes.spines.right':False,'pdf.fonttype':42,'axes.linewidth':.7})
    fig,axes=plt.subplots(2,3,figsize=(10,5.8),layout='constrained')
    colors={'BASELINE':'#596469','UNIFIED':'#247b8c','CANDIDATE':'#247b8c'}
    for row,dep in enumerate(('gwtc3','gwtc4')):
        for col,metric in enumerate(('macro_r_at_10','average_precision','false_at_recall_0p9')):
            ax=axes[row,col]
            for k,method in enumerate(('waveform_only','C_fixed')):
                for variant,offset in (('BASELINE',-.12),('CANDIDATE',.12)):
                    f=fresh_rows.loc[(fresh_rows.deployment==dep)&(fresh_rows.method==method)&(fresh_rows.variant==variant)]
                    values=f.groupby('model_seed')[metric].mean().to_numpy()
                    x=k+offset
                    ax.scatter(x+np.linspace(-.025,.025,len(values)),values,color=colors[variant],s=17,label=variant if k==0 else None)
                    ax.errorbar(x,values.mean(),yerr=values.std(ddof=1),fmt='_',color=colors[variant],capsize=3)
            ax.set(xticks=[0,1],xticklabels=['Waveform','Three-channel'],ylabel=LABELS[metric],title=dep+' / new source-noise confirmation')
            if row==0 and col==0:
                ax.legend(loc='upper left',bbox_to_anchor=(0,1.30),ncol=2,frameon=False,fontsize=7)
    fig.savefig(root/'figures/fig_independent_confirmation.pdf')
    fig.savefig(root/'figures/fig_independent_confirmation.png',dpi=220)
    plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(10,3),layout='constrained')
    for ax,column,label in zip(axes,('catastrophic_mc','BC_mc_ge_0p5','official_frontend'),
            ('Catastrophic Mc conflicts / Top-10','BC(Mc) >= 0.5 / Top-10','Official frontend overlap / Top-10')):
        for k,dep in enumerate(('gwtc3','gwtc4')):
            for variant,offset in (('BASELINE',-.17),('UNIFIED',.17)):
                value=pe.loc[(pe.deployment==dep)&(pe.config==variant)&pe.method.eq('C_fixed')&pe.budget.eq(10)&pe.seed.astype(str).eq('consensus'),column].iloc[0]
                ax.bar(k+offset,value,width=.28,color=colors[variant],label=variant if k==0 else None)
        ax.set(xticks=[0,1],xticklabels=['O3','O4a'],ylabel=label,ylim=(0,10.8))
    axes[0].legend(loc='upper left',bbox_to_anchor=(0,1.25),ncol=2,frameon=False,fontsize=7)
    fig.suptitle('Reused real-catalog development audit; official overlap is not lens truth',fontsize=9)
    fig.savefig(root/'figures/fig_real_development_audit.pdf')
    fig.savefig(root/'figures/fig_real_development_audit.png',dpi=220)
    plt.close(fig)


def report(root):
    conf=fresh.verify(root)
    protected=pd.read_csv(root/'manifest/PROTECTED_INPUT_SHA256.csv')
    changed=[str(r.path) for r in protected.itertuples() if dev.sha(Path(r.path))!=r.sha256]
    dev.json_write(root/'audit/FINAL_PROTECTED_HASH_RECHECK.json',{'checked':len(protected),'changed':changed,'pass':not changed})
    if changed:
        raise RuntimeError('Historical protected files changed')
    decision=json.loads((root/'confirmation/FINAL_GUARDRAIL_AUDIT.json').read_text())
    independence=json.loads((root/'confirmation/GLOBAL_SOURCE_NOISE_INDEPENDENCE.json').read_text())
    consistency=json.loads((root/'contracts/METHOD_CONSISTENCY_AUDIT.json').read_text())
    development=json.loads((root/'contracts/DEVELOPMENT_GATE.json').read_text())
    if not (decision['per_model_mean_across3catalog_guardrails_pass'] and independence['source_noise_independence_pass']
            and consistency['pass'] and development['development_pass']):
        raise RuntimeError('Final delivery is blocked by a failed frozen gate')
    external=root/'external_development_audit_v3'
    if not (external/'INTERPRETATION.json').is_file():
        raise RuntimeError('Full PE/official/waveform descriptive audit is incomplete')
    reused=pd.read_csv(root/'tables/RETRIEVAL_PER_SEED.csv')
    reused=reused.loc[reused.split.eq('test')]
    summary=summarized(reused,['deployment','config','method'])
    dev.csv_write(root/'tables/REUSED_RETRIEVAL_SUMMARY.csv',summary)
    fresh_rows=pd.concat([pd.read_csv(root/f'confirmation/{dep}/metrics_all.csv') for dep in ('gwtc3','gwtc4')],ignore_index=True)
    numeric=[c for c in fresh_rows.select_dtypes(include='number') if c not in ('model_seed','catalog_seed','eval_seed')]
    means=fresh_rows.groupby(['deployment','variant','model_seed','method'])[numeric].mean().reset_index()
    fs=summarized(means,['deployment','variant','method'])
    dev.csv_write(root/'tables/INDEPENDENT_CONFIRMATION_SUMMARY.csv',fs)
    dev.csv_write(root/'tables/INDEPENDENT_CONFIRMATION_ALL_METRICS.csv',fresh_rows)
    pe=pd.read_csv(root/'tables/PE_OFFICIAL_ALL.csv')
    top=pe.loc[pe.seed.astype(str).eq('consensus')&pe.budget.eq(10)]
    params=pd.read_csv(root/'tables/METHOD_CONSISTENCY_AUDIT.csv')
    figures(root,reused,fresh_rows,pe)
    trained=Path(conf['training_root'])
    data=[]
    for dep in ('gwtc3','gwtc4'):
        for split in ('train','validation'):
            f=pd.read_parquet(trained/f'cache/{dep}/{split}_metadata.parquet')
            data.append({'deployment':dep,'split':split,'noisy_images':len(f),'source_parents':f.waveform_parent_uid.nunique()})
    dev.csv_write(root/'tables/ENCODER_TRAINING_DATA_COUNTS.csv',pd.DataFrame(data))
    if not (root/'trial_ledger').exists():
        ledger_run=subprocess.run([os.sys.executable,'-B',str(dev.PROJECT/'scripts/experiments/mcwf_goal_ledger_20260907.py')],capture_output=True,text=True,check=True)
        ledger=Path(ledger_run.stdout.strip().splitlines()[-1])
        shutil.copytree(ledger,root/'trial_ledger',dirs_exist_ok=False)
    status=pd.read_csv(root/'trial_ledger/ALL_TRIAL_STATUS.csv')
    lines=[
        '# MCWF-UNIFIED-RNC-FRT：O3/O4a 统一波形改进完整交付',
        '',
        '最终状态：`HOLD_FOR_AUTHOR_REVIEW_NO_ADOPTION_NO_OVERWRITE`。',
        '本轮完成的是冻结配置下的波形改进、真实目录开发审计和独立新源/噪声注入确认。历史 v9.3、v9.4、C-fixed、ET-3、论文与候选表未替换。真实候选仍不是透镜探测。',
        '',
        '## 1. 目标与统计边界',
        '目标是减少 O3 高分候选的严重啁啾质量冲突，同时保持 O4a 的 PE/官方阶段表现及两运行期的注入性能。真实 PE 在多轮开发中被反复检查，明确属于开发反馈，不是盲测、训练标签或透镜真值。官方重合只作描述性结果，没有进入选参目标。',
        '严重 Mc 冲突的冻结定义为 `BC_Mc < 0.1 OR D_Mc > 5`。`Dmax <= 3` 是边际参数一致性筛查，不是共享源 Bayes factor 或高斯联合显著性。',
        '',
        '## 2. O3 和 O4a 是否采用同一种方法',
        '是。两者具有相同的峰值 2 s/4096 点 H1+L1 输入、网络结构、训练目标、特征提取、有限参考尾部公式和 validation 选参规则。只有训练权重、validation 标定数值和历史冻结外层权重可逐运行期/seed 不同。没有 O4a 退回旧模型的分支。',
        '最终波形项在两者中均为：',
        r'\[Z^{\rm new}_{\rm wf}=Z^{\rm Cfixed}_{\rm wf}+\gamma\min\{\log(p_{\rm tail}/0.05),0\}.\]',
        r'\[a_{ij}=-\log\sum_b\sqrt{q_i(b)q_j(b)},\qquad p_{\rm tail}=\frac{1+\#\{a_g\ge a_{ij}\}}{n_{\rm ref}+1}.\]',
        '`q_i(b)` 是新网络从波形得到的 64 个 log-Mc bin 的预测概率，不是公开 PE 后验；`a_g` 每个 validation 伴随系统只贡献一个值。有限参考分母使极端尾部有明确分辨能力下限，不再按很小的 IQR 无限制放大误差。',
        '这个尾部秩受 conformal 方法启发，但本轮不能声称它是校准到真实 GWTC 的 FPP：validation 还用于选 gamma，真实域的 exchangeability 没有证明。最终总分也不是透镜后验概率。',
        '',
        '## 3. 波形处理与训练',
        '旧 C-fixed 波形证据保留。新增网络使用 raw InceptionAttention 分支和 1728 维相位匹配特征 MLP；输出 128 维表示与 64-bin 质量头。特征使用固定模板网格、逐事件 off-source PSD、float32 FFT、8192 零填充、逐采样点时移，H1/L1 相对时间范围覆盖两站光行时。只使用波形及其预处理 PSD，不读取真实 PE 质量。',
        'RNC 训练把模拟 log-Mc 的连续距离加入表示学习，损失为 soft-bin CE + source SupCon + RNC。每个运行期训练三个 seed，各运行 50 epochs；按 validation CE + 0.2 SupCon + 0.2 RNC 的最早最小值选 checkpoint，温度也只由 validation CE 决定。',
        '这是逐级 warm-start 的改进，不能写成六个完全从头训练的全新实验。即时 warm start 来自 fine-lag/event-PSD 编码器；RNC 的训练历史和 checkpoint 哈希均保留。最终修正直接使用质量头分布，新 embedding cosine 只用于训练/诊断，不作为额外第四通道。',
        table(pd.DataFrame(data),['deployment','split','source_parents','noisy_images']),
        '旧 O3 development 噪声包含 O1/O2，这是明确保留的数据限制；不能把该训练库说成完全 O3-run-matched。独立确认使用新的 O3-only/O4a-only 噪声。',
        '',
        '## 4. 选参与冻结',
        'alpha 固定 0.05；gamma 网格为 2**k，k=-6,...,1。每个 seed 只在模拟 validation 选 gamma。先要求 waveform-only 和 C-fixed 的 R@10/AP/F50/F90 通过冻结约束，再依次优化 C-fixed F50、F90、waveform F50、F90、AP、R@10，最后用固定数值顺序打破平局。',
        '约束为 R@10 下降不超过 0.02、AP 下降不超过 0.005、F50/F90 不超过基线 1.10 倍。门槛未因后续结果修改。',
        table(params,['deployment','model_seed','eval_seed','gamma','selected_epoch','changed_pair_fraction']),
        '时间查表、天空图/分数和 C-fixed 外层系数不变：',
        r'\[S=\lambda_W Z^{\rm new}_{\rm wf}+\lambda_T Z_{\rm time}+\lambda_S Z_{\rm sky}.\]',
        table(pd.read_csv(root/'tables/SHARED_METHOD_PARAMETERS.csv'),['deployment','seed','waveform','time','sky']),
        '继承的通用合同中旧“convex ensemble”等文字不代表实际执行公式；`FINAL_METHOD_CLARIFICATION.json` 记录了真实实现，未改动任何已选择的数值配置。',
        '',
        '## 5. 原注入的复用对照',
        '这些 test catalog 已被开发反复查看，只称复用对照，不再冒充新盲测。以下为三个训练 seed 的 mean +/- sample SD。',
        table(summary,['deployment','config','method',*LABELS.values()]),
        '',
        '## 6. 真正新增的源/噪声确认',
        '冻结后每个运行期生成三个新 catalog；每个 catalog 有 35+35 个双像源和 50 个单事件，共 190 events、70 true pairs。总计 720 个新源、1140 个事件和 96 个互不重叠的 256 s 噪声块。源参数与历史库实际比较，噪声用全局 GPS 区间加 16 s guard 检查，而不是只比较文件名或 ID。',
        '同一新目录同时送入旧基线和新方法，两者时间、天空及真对标签逐元素一致。每个模型在三个新 catalog 上的均值必须在 O3/O4a 均通过同一约束。',
        table(fs,['deployment','variant','method',*LABELS.values()]),
        f"逐 catalog/model/method 的点约束通过 {decision['catalog_model_method_passes']}/{decision['catalog_model_method_total']}；预先定义的逐模型三 catalog 均值约束全部通过。所有逐目录行和置信区间均原样保留。",
        '新源/噪声并不等于新透镜人口：GW-LMC lens environments 可重复，质量/SNR 是覆盖型抽样而非天体物理事件率；两像 target-SNR 缩放沿用基线，不是 response-derived 相对强度实验。',
        '天空沿用条件 BAYESTAR 测量模拟与局部 PSD，已知注入内禀参数，并使用独立 Gaussian matched-filter measurement realization；不是对这份非平稳噪声混合 strain 重新做完整 PE。真实目录仍使用公开 PE 天空图。',
        '',
        '## 7. 不确定度',
        '每 catalog/seed 的 R@1/R@10 使用 10000 次 system bootstrap，两幅像的 directed queries 一起抽样并按 family 分层；AP/F50/F90 使用 2000 次 source-block 加权 bootstrap，另报 2000 次 noise-block 条件 bootstrap。源与噪声区间是两种条件诊断，不等于完整人口不确定度。',
        '区间与差值见 `confirmation/PAIRED_METRICS_AND_CI.csv`。均值通过工程非劣容差不等于所有差值在 95% CI 下显著改善。',
        '历史 FPR=1e-5 字段有“至少一个 FP”的实现下限；本包记录实际假对率，不将其写成有 1e-5 分辨能力的结果。',
        '',
        '## 8. 真实目录 PE 与官方阶段',
        '共识榜按三个 seed 的平均 rank、最坏 rank、平均总分依次排序。因此 consensus rank 不要求平均总分严格递减；逐 seed 排名和分数也全部保留。',
        '`official_frontend` 指至少一种官方前端方法 FPP<0.01：O3 使用 PO/ML，O4a 使用 PO/Phazap。不是只要出现在官方全 pair 表中便计数。`official_hanabi` 指公开的 pair-resolved Hanabi 图/表重合，O3 包含 O3a 和 full-O3 的对应条目；不表示透镜得到确认。',
        table(top,['deployment','config','method','catastrophic_mc','BC_mc_ge_0p5','median_BC_mc','Dmax_le_3','official_frontend','official_hanabi']),
        'Top-10 无严重 Mc 冲突不等于所有质量后验高度重合。O3 新三通道 Top-10 只有 5 对 BC_Mc>=0.5；扩展到 Top-20/50/100 仍分别有 2/6/13 对严重冲突。原 O3 Top-10 的三对严重冲突没有被删出目录，其新 rank 为 19、11、40。',
        '官方重合并非所有预算都改善：O3 三通道 Top-20 的官方前端重合为 9->6，Top-10 保持 4；这些较差的描述性结果也必须保留，不能写成全面优于官方或全面提高官方重合。',
        table(pe.loc[pe.seed.astype(str).eq('consensus')&pe.method.eq('C_fixed')],['deployment','config','budget','catastrophic_mc','BC_mc_ge_0p5','Dmax_le_3','official_frontend','official_hanabi']),
        '完整目录的波形分数与质量 PE 一致性的描述性相关如下。pair 共享事件，不能使用普通独立 pair 的 p 值宣称显著性；相关提高也不意味着每一对都严格单调一致。',
        table(pd.read_csv(external/'tables/REAL_WAVEFORM_MASS_CORRELATIONS.csv').query("method=='waveform_only'"),['deployment','variant','pairs','score_vs_BC_spearman','score_vs_negative_D_spearman']),
        'O3 当前冻结范围为官方目录的 strict H1/L1 62 事件、1891 个无序 pair；O4a 为 74 事件、2701 pair。旧混合目录中不在该范围的 GW190924_021846 相关 pair 标为 OUTSIDE_FROZEN_SCOPE，不伪造新排名，也不把剔出范围算成模型改进。',
        'O3 的 PE BC 是固定直方图近似重合，D 使用 median 差与 (q84-q16)/2 宽度合并；O4a 的对应冻结指标按原同名定义保留并与完整旧 PE 表逐项核对。距离只作描述性审计，不进入 Dmax 内禀筛查。',
        '完整 Top-10/20/50/100、所有 pair CSV/Parquet、通道贡献、官方 PO/ML 或 PO/Phazap FPP、GOLUM/Hanabi 阶段、已公开结论见 `external_development_audit_v3/tables/` 和 `results/`。未公开的阶段不补造数值；官方重合不是透镜真阳性，本轮没有运行 Hanabi。',
        '每个新 Top-10 的 H1/L1 2 s 输入、网络质量预测及公开 PE 中心/宽度图均在 `external_development_audit_v3/figures/`。其中神经网络预测曲线与公开 PE 图明确分开标注。',
        '',
        '## 9. 失败探索与复现',
        f'台账保留 {len(status)} 个统一方法开发配置的状态、指标与 PE/官方预算，不删失败结果。另有数值时移、PSD、梯度冲突、质量预测一致性等只读诊断；未被证据支持的可靠度方案没有接入。',
        '主要拒绝原因包括：只改善 PE 但注入 F90 恶化、只改善 O3 而 O4a 下降、某个 validation seed 没有合格非零组合、直接混合新旧分数导致排名负担增加。RNC-FRT 的合格性不能掩盖这些失败或开发选择次数。',
        '报告/图制作曾因 O4a 紧凑 PE 表缺少中心字段和历史 pair-key 分隔符不同而失败；已在独立 v3 外部审计中补充完整只读 PE 表并统一键，旧失败目录保留。排名、模型与冻结文件未因此改变。',
        '',
        '## 10. 文献与工程选择',
        '- Rank-N-Contrast：[Zha et al.](https://arxiv.org/abs/2210.01189)。本轮使用相同的连续标签排序目标，但采用已有单位范数表示和 temperature=0.1，不宣称原论文数值复现。',
        '- 有限参考秩与不确定度：[Angelopoulos and Bates](https://arxiv.org/abs/2107.07511)。本轮只借鉴 smoothed rank 思路，不宣称真实域 conformal coverage。',
        '- SupCon：[Khosla et al.](https://arxiv.org/abs/2004.11362)；温度校准：[Guo et al.](https://arxiv.org/abs/1706.04599)。',
        '- 相位匹配与 PSD 处理：[PyCBC 官方文档](https://pycbc.org/pycbc/latest/html/filter.html)。',
        '网络层数、gamma 网格、alpha、非劣容差、模板网格与 bootstrap 次数属于本项目明确记录的工程/统计设计，不冒称自然常数或被文献保证的性能。',
        '',
        '## 11. 交付边界',
        '这是目标条件下完成的统一波形改进及条件独立确认，不是新的真实透镜确认，也不自动替换论文。包包含脚本、最终/即时 warm-start 模型、配置、指标、PE/官方表、图、日志与哈希；原始 GWOSC strain、完整 PE HDF5 和全部 dense sky maps 留在原服务器，包内以输入 manifest 指向它们。',
        '运行服务器：connect.westd.seetacloud.com，SSH 端口 32328。压缩包不包含登录凭据。',
        '',
        '`HOLD_FOR_AUTHOR_REVIEW_NO_ADOPTION_NO_OVERWRITE`',
    ]
    (root/'reports/FINAL_UNIFIED_WAVEFORM_REPORT_CN.md').write_text('\n\n'.join(lines)+'\n',encoding='utf-8')
    dev.json_write(root/'contracts/FINAL_GOAL_AUDIT.json',{'utc':datetime.now(timezone.utc).isoformat(),
        'method_code':'MCWF-UNIFIED-RNC-FRT','development_gate_pass':True,'fresh_source_noise_guardrails_pass':True,
        'same_method_both_runs':True,'old_only_O4_fallback':False,'real_PE_is_development':True,
        'lens_detection_claim':False,'paper_or_history_adopted':False,'trial_count':len(status),
        'author_review_status':'HOLD_FOR_AUTHOR_REVIEW_NO_ADOPTION_NO_OVERWRITE'})
    return conf


def package(root,conf):
    name='MCWF_UNIFIED_RNC_FRT_20260907'
    bundle=root/'delivery'/name
    if bundle.exists():
        raise RuntimeError('Delivery already exists; do not overwrite')
    bundle.mkdir(parents=True)
    for folder in ('contracts','calibration','tables','reports','audit','logs','scripts','figures','results','trial_ledger','external_development_audit_v3'):
        shutil.copytree(root/folder,bundle/folder)
    for p in (root/'confirmation').rglob('*'):
        compact_array=p.name in ('new_predictions.npz','new_event_PSD_features.npy','frequency.npy','psd.npy')
        if p.is_file() and (p.suffix in ('.json','.csv','.parquet') or compact_array):
            target=bundle/p.relative_to(root)
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    trained=Path(conf['training_root'])
    for folder in ('contracts','audit','logs'):
        if (trained/folder).exists():
            shutil.copytree(trained/folder,bundle/'encoder_training'/folder)
    for dep in ('gwtc3','gwtc4'):
        for filename in ('train_metadata.parquet','validation_metadata.parquet','DATA_COMPLETE.json'):
            p=trained/'cache'/dep/filename
            target=bundle/'encoder_training'/p.relative_to(trained)
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    for p in (trained/'models').rglob('*'):
        if p.is_file() and p.name!='resume.pt':
            target=bundle/'encoder_training'/p.relative_to(trained)
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    warm=dev.PROJECT/'results/mcwf_unified_finelag_eventpsd_encoder_20260907'
    for p in (warm/'models').rglob('validation_selected_model.pt'):
        target=bundle/'immediate_warm_start'/p.relative_to(warm/'models')
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    for r in conf['frozen_files']:
        p=Path(r['path'])
        if not p.is_absolute():
            p=dev.PROJECT/p
        target=bundle/'frozen_inputs'/p.relative_to(dev.PROJECT)
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    for p in (dev.PROJECT/'scripts/experiments').glob('mcwf*.py'):
        target=bundle/'reproducibility_scripts'/p.name
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    code=set()
    for module in list(sys.modules.values()):
        value=getattr(module,'__file__',None)
        if value:
            p=Path(value).resolve()
            if p.suffix=='.py' and p.is_relative_to(dev.PROJECT) and p.is_file():
                code.add(p)
    for folder in ('src','scripts/real_search','scripts/server_experiments'):
        code.update(p.resolve() for p in (dev.PROJECT/folder).rglob('*.py'))
    for p in sorted(code):
        target=bundle/'project_code_snapshot'/p.relative_to(dev.PROJECT)
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    shutil.copy2(__file__,bundle/'scripts'/Path(__file__).name)
    shutil.copy2(root/'manifest/PROTECTED_INPUT_SHA256.csv',bundle/'HISTORICAL_INPUT_MANIFEST.csv')
    freeze=subprocess.run([os.sys.executable,'-m','pip','freeze'],capture_output=True,text=True,check=True)
    (bundle/'PYTHON_ENVIRONMENT.txt').write_text(freeze.stdout)
    (bundle/'README_CN.md').write_text('# MCWF-UNIFIED-RNC-FRT\n\n先读 reports/FINAL_UNIFIED_WAVEFORM_REPORT_CN.md。\n\n本包不是独立包含所有原始数据的镜像：原始 strain、PE HDF5、完整 sky map 和训练数组由输入 manifest 指向服务器原路径。模型、最终校准与脚本可用于冻结推理复核；即时 warm-start 模型用于复现 RNC 子阶段。历史来源不被覆盖。\n\n最终状态：HOLD_FOR_AUTHOR_REVIEW_NO_ADOPTION_NO_OVERWRITE。\n',encoding='utf-8')
    forbidden=re.compile(rb'^-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----\s*\n[A-Za-z0-9+/=]+',re.MULTILINE)
    rows=[]
    for p in sorted(bundle.rglob('*')):
        if not p.is_file():
            continue
        if p.suffix in ('.py','.json','.txt','.md','.csv'):
            data=p.read_bytes()
            if forbidden.search(data):
                raise RuntimeError('Credential material detected in delivery')
        rows.append({'path':p.relative_to(bundle).as_posix(),'bytes':p.stat().st_size,'sha256':dev.sha(p)})
    dev.csv_write(bundle/'FILE_MANIFEST.csv',pd.DataFrame(rows))
    sums=rows+[{'path':'FILE_MANIFEST.csv','sha256':dev.sha(bundle/'FILE_MANIFEST.csv')}]
    (bundle/'SHA256SUMS.txt').write_text(''.join(r['sha256']+'  '+r['path']+'\n' for r in sums))
    archive=dev.PROJECT/'packages'/f'{name}.tar.gz'
    if archive.exists():
        raise RuntimeError('Archive already exists; do not overwrite')
    with tarfile.open(archive,'w:gz',compresslevel=6) as tar:
        tar.add(bundle,arcname=name)
    expected={name+'/'+r['path']:r['sha256'] for r in sums}
    checked=0
    with tarfile.open(archive,'r:gz') as tar:
        for member in tar:
            if member.name not in expected:
                continue
            handle=tar.extractfile(member);digest=hashlib.sha256()
            for chunk in iter(lambda:handle.read(1024*1024),b''):
                digest.update(chunk)
            if digest.hexdigest()!=expected[member.name]:
                raise RuntimeError('Archive member checksum mismatch')
            checked+=1
    if checked!=len(expected):
        raise RuntimeError('Archive missing required manifest members')
    digest=dev.sha(archive)
    archive.with_suffix(archive.suffix+'.sha256').write_text(digest+'  '+archive.name+'\n')
    dev.json_write(root/'manifest/DELIVERY_VERIFICATION.json',{'archive':str(archive),'sha256':digest,
        'bytes':archive.stat().st_size,'verified_members':checked,'all_manifest_members_verified':True})
    print(json.dumps({'archive':str(archive),'sha256':digest,'verified_members':checked,'bytes':archive.stat().st_size}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--phase',choices=('report','package','all'),default='all')
    a=p.parse_args()
    if a.phase=='package':
        if not (a.root/'contracts/FINAL_GOAL_AUDIT.json').exists():
            raise RuntimeError('A verified final report is required before packaging')
        conf=fresh.verify(a.root)
    else:
        conf=report(a.root)
    if a.phase!='report':
        package(a.root,conf)
