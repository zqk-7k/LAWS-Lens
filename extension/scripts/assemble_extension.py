"""Assemble a non-public r2 extension without overwriting the r1 release."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import shutil
import tarfile

def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()

def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--release',required=True,type=Path)
    p.add_argument('--core',required=True,type=Path)
    p.add_argument('--source',required=True,type=Path)
    a=p.parse_args()
    root,core=a.release,a.core
    ext=root/'extension'
    ext.mkdir(exist_ok=False)
    shutil.copytree(a.source,ext/'scripts',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    evidence=ext/'evidence'
    evidence.mkdir()
    for folder in ('all_original_inputs','rebuilt_figures_attempt2','figure_rendering','test_real_prediction_replay_v2',
                   'all_500_subsets','end_to_end_recovery','end_to_end_recovery_strain'):
        source=root/'verification'/folder
        check=json.loads((source/'REPORT.json').read_text())
        if check['status']!='PASS':
            raise RuntimeError('Verification not passed: '+folder)
        dest=evidence/folder
        dest.mkdir()
        for path in source.iterdir():
            if path.is_file() and (path.suffix in ('.json','.jsonl','.png','.pdf') or path.name.endswith('_checks.csv.gz')):
                shutil.copy2(path,dest/path.name)
    for filename in ('PORTABILITY_replay_test_real_models.json','PORTABILITY_laws_replay_recovery_20260924.json'):
        path=root/'verification'/filename
        check=json.loads(path.read_text())
        if check['status']!='PASS' or check['violations']:
            raise RuntimeError('Historical read-guard violation')
        shutil.copy2(path,evidence/path.name)
    shutil.copytree(root/'inventory',ext/'inventory')
    # Only regular files from this task's bounded public-download archive are accepted.
    public=ext/'public_tables'
    public.mkdir()
    with tarfile.open(root/'public_tables_verified.tar.gz','r:gz') as archive:
        for member in archive:
            if member.isdir():
                continue
            normalized=Path(member.name)
            if not member.isfile() or normalized.is_absolute() or len(normalized.parts)!=1 or '..' in normalized.parts:
                raise RuntimeError('Unexpected public download archive member')
            member.name=normalized.name
            archive.extract(member,public,filter='data')
    downloaded=json.loads((public/'DOWNLOAD_REPORT.json').read_text())
    frozen=json.loads((ext/'inventory/PUBLIC_TABLE_DOWNLOADS.json').read_text())
    checks=[]
    for row in frozen:
        digest=sha(public/row['filename'])
        checks.append({'file':row['filename'],'sha256':digest,'matches_expected':digest==row['sha256']})
    if not downloaded['complete'] or not all(x['matches_expected'] for x in checks):
        raise RuntimeError('Public download/relay check failed')
    save(evidence/'PUBLIC_DOWNLOAD_AND_RELAY.json',{'status':'PASS','download_host':'local Codex workspace',
        'server_download_succeeded':False,'local_download_and_verified_transfer_succeeded':True,'checks':checks,
        'full_public_input_corpus_downloaded':False})
    before=json.loads((core/'audit/PROTECTED_INPUTS.json').read_text())
    changes=[r['path'] for r in before if sha(Path(r['path']))!=r['sha256']]
    if changes:
        raise RuntimeError('Protected inputs changed')
    save(evidence/'PROTECTED_INPUTS_POSTCHECK.json',{'status':'PASS','files':len(before),'changes':changes})
    models=json.loads((evidence/'test_real_prediction_replay_v2/REPORT.json').read_text())
    model_arrays=sum(len(r['fields']) for r in models['records'])
    metrics=json.loads((evidence/'all_500_subsets/REPORT.json').read_text())
    recovery=json.loads((evidence/'end_to_end_recovery/REPORT.json').read_text())
    inputs=json.loads((evidence/'all_original_inputs/REPORT.json').read_text())
    validation={'status':'HOLD_FOR_AUTHOR_REVIEW_NO_PUBLICATION','assembly_revision':'r2',
        'paper_commit':'afb86ff5c8f212aec579592f0d7439f3f909d931','main_version':'GWLR-UC-01/C_PHYSICAL',
        'new_scientific_experiment':False,'historical_results_modified':False,'paper_modified':False,
        'original_upload_preserved':True,'public_upload':False,'doi':None,
        'validation_components_previously_exact':45,'test_real_components_exact':models['components'],
        'test_real_arrays_exact':model_arrays,'model_prediction_input':'Frozen preprocessed inputs and matched features',
        'all_500_subset_scalar_checks':metrics['checks'],'all_original_input_files_verified':inputs['files'],
        'public_tables_fresh_downloaded':3,'data_figure_reconstruction_available':13,
        'original_data_figure_scripts_replayed':2,'replacement_data_figure_scripts':11,
        'original_missing_plot_layouts_recovered':False,'representative_complete_recovery_events':recovery['events'],
        'regenerated_template_bank_count':8192,'whole_population_retrained':False,'full_public_corpus_redownloaded':False,
        'fresh_physical_host_validated':False,'archive_extraction_replay':'PENDING_PACKAGING_TEST'}
    save(ext/'reports/RELEASE_VALIDATION_STATUS_R2.json',validation)
    gaps=[
        {'type':'original_author_layout','item':'11 original plotting entrypoints','state':'NOT_RECOVERED',
         'resolution':'Independent data-equivalent scripts and plotted-value manifests supplied; exact original layouts not claimed.'},
        {'type':'validation_scope','item':'Whole population, training and fresh noise-bank acquisition','state':'NOT_RERUN',
         'resolution':'All models replayed; six frozen validation events regenerated through bank recovery and sky. Whole-population retraining not certified.'},
        {'type':'validation_scope','item':'New physical host and full 126 GiB network acquisition','state':'NOT_TESTED',
         'resolution':'Fresh locked environment and independent extraction on the same server; 847 existing inputs hashed, three public tables freshly fetched.'},
        {'type':'author_decision','item':'Repository/account, licenses, software/data contributors, ORCID and public date','state':'PENDING_AUTHOR'},
        {'type':'third_party','item':'Code/data/model redistribution rights','state':'PENDING_MANUAL_LICENSE_REVIEW'},
        {'type':'scientific_boundary','item':'Real significance, fixed GPD endpoint and annual FAR','state':'NOT_CERTIFIED_BY_REPRODUCTION',
         'resolution':'B=4 remains a model assumption with 4.5/5 sensitivity. Conditional FPP does not establish real-candidate significance or annual exposure.'}]
    save(ext/'reports/REMAINING_LIMITS_AND_AUTHOR_DECISIONS.json',gaps)
    prior=root/'verification/PORTABILITY_laws_replay_test_real_models_20260924.json'
    if prior.exists():
        shutil.copy2(prior,evidence/'FIRST_ATTEMPT_WRAPPER_SCOPE_FAILURE.json')
    save(ext/'reports/PACKAGING_ATTEMPTS.json',{'initial_model_numerics':'PASS',
        'initial_read_guard':'Blocked attempts to inspect the r1 launcher script because the launcher itself was outside the r2 root.',
        'repair':'Run the unchanged portable launcher from r2, rerun all model predictions in a new directory.',
        'scientific_change':False,'previous_attempt_preserved':True})
    report=f'''# LAWS-Lens 论文可复现发布整理报告：r2

状态：HOLD_FOR_AUTHOR_REVIEW_NO_PUBLICATION。
本轮完成的是私有发布整理和分层验证，不是正式公开，也不是新科学结果。
论文快照为 afb86ff；主实验仅采用 GWLR-UC-01 / C_PHYSICAL，合并已有 FPP、GPD 和 DOMAIN04 计时材料。

## 本轮新增核验

| 层级 | 实际完成 | 结论边界 |
|---|---|---|
| 原始输入 | {inputs['files']} 个文件，{inputs['bytes']} 字节，逐文件 SHA-256 全部吻合 | 约126 GiB现有文件；不是全部重新下载 |
| 公共获取 | 三张固定提交的GW-LMC原表重新下载，转传服务器后再次核对 | 本机下载成功，不能称服务器直连成功 |
| 模型推理 | test各450事件、real的62/74/86事件，三seed、五组件：{models['components']}项、{model_arrays}数组逐值一致 | 复用冻结预处理/匹配特征，不是全部真实应变重新预处理 |
| 旧validation推理 | r1已完成45组件、117数组逐值复现 | 与本轮合计135项组件检查，不是135个独立模型 |
| 目录指标 | validation/test、三运行期、三seed、四方法、全部500组190事件子目录：{metrics['checks']}项指标吻合 | 子目录共享源和噪声，不是500次独立实验 |
| 图表 | r1的2张原脚本图 + 本轮11张冻结数据重建图，逐图数值清单与PDF/PNG | 后11张不是原作者布局脚本，也不承诺像素一致；两张示意图保留静态资源 |
| 注入再生成 | 每运行期固定一个双像系统，共{recovery['events']}事件，从源与冻结真实噪声切片重建64s应变 | 固定代表样例，不证明整个人口统计覆盖 |
| 恢复与定位 | 重建8192模板库，字节哈希吻合；完整模板搜索、触发量和BAYESTAR原生图与归档对照 | 不是仅从已存触发量重算；不是全观测运行盲搜 |
| 既有论文分析 | r1已复算固定上界GPD、留出/删除诊断和相关表格 | 数值可复现不等于尾部模型或真实显著性得到额外科学验证 |

## 未改变的内容

未重新训练、选择权重、生成新的候选排名、修改论文或覆盖历史文件。原上传副本和历史输入的保护清单再次通过哈希检查。
未创建GitHub仓库、Zenodo记录或DOI，未使用聊天中出现的凭据进行发布。
软件、权重和自生成数据许可、第三方再分发许可、作者与账户确认仍需要审核。

## 仍然不能宣称

不能把本次传输/数值验证称为全量从公共原始数据开始的独立重训。
未在另一台物理主机上完成复现，也未重新下载所有原始strain/PE或重建全部噪声库。
11张原始绘图布局脚本尚未找到，提供的是有输入哈希、绘图数值和明确规则的替代重建入口。
没有新增独立盲测、实际全年背景曝光或年度FAR；GPD固定上界B=4不是已证实的物理上限。

## 包与入口

以packages/RELEASE_PACKAGE_INDEX_R2.json和SHA256SUMS_R2.txt为准。r1归档字节保持不变，r2增加复现扩展和test/real输入样例包。
合并解压后先读extension/README_CN.md；旧根目录README和r1审计保留为历史记录，验证范围以本报告和最终包解压实跑记录为准。
最终包实跑结果由packages/DELIVERED_EXTENSION_REPLAY.json单独记录，不用制作阶段的通过记录代替。
'''
    (ext/'reports/RELEASE_COMPLETION_R2_CN.md').write_text(report)
    readme='''# LAWS-Lens 私有发布候选 r2

当前主实验：GWLR-UC-01 / C_PHYSICAL。论文快照：afb86ff。
禁止将此草稿称为已公开发布或已取得DOI。禁止自行修改论文、评分和权重。

## 读取顺序

1. extension/reports/RELEASE_COMPLETION_R2_CN.md
2. extension/reports/RELEASE_VALIDATION_STATUS_R2.json
3. extension/reports/REMAINING_LIMITS_AND_AUTHOR_DECISIONS.json
4. 原始论文source_data/c_current/C_SOURCE_INDEX.md和r1的数据字典/依赖清单

## 装配

按 RELEASE_PACKAGE_INDEX_R2.json 校验八个包的SHA-256。四个r1模块包与两个r2扩展包解压到同一新的空目录；路径不应重叠。
原生BAYESTAR地图包和wheelhouse包按原README分别放置，不要混淆成代码模块。
环境安装沿用r1经过验证的锁定文件和离线wheelhouse；推荐Python 3.12。不要直接使用系统Python安装旧依赖。

## 有限复现入口

在组合发布目录执行：

```bash
python -B scripts/reproduce.py paper --release "$PWD"
python -B scripts/reproduce.py features --release "$PWD"
python -B scripts/reproduce.py models --release "$PWD"
python -B scripts/reproduce.py injections --release "$PWD"
python -B scripts/reproduce.py metrics-sky --release "$PWD"
python -B extension/scripts/reproduce_extension.py all --release "$PWD"
```

扩展阶段也可分别运行figures、all-subsets、test-real-models、recovery。输出为新的verification/extension_<UTC>_<ID>目录。
recovery会临时重建约4GiB的8192模板库，再对固定六事件完整搜索及定位；预留至少12GiB工作空间，三个CPU任务并行。
test-real-models使用完整批次上下文，保持原浮点运算路径；不应随意改成单事件批次后要求字节一致。
同环境验收要求：模型数组逐值一致、指标差<1e-11、模板库哈希吻合、触发量逐值一致、地图列rtol=1e-7/atol=1e-12。
其他硬件/依赖版本的浮点差异需单独报告，不能自动放宽阈值后称原验证通过。

## 外部输入

supplement/GWLR_UC01_REPRODUCIBILITY_SUPPLEMENT/inputs/RAW_INPUT_ACQUISITION.json列出847个原始输入。
extension/inventory/PUBLIC_TABLE_DOWNLOADS.json给出三张已实测下载的GW-LMC表。
可运行extension/scripts/download_public_tables.py --manifest <该文件> --output <新目录>再次获取并核对。
其余公开strain/PE保留精确下载/引用清单，不在新扩展包复制全部原始大文件。

## 图表说明

11张新图是冻结数据的独立重绘，不覆盖uploaded_snapshot中的论文原图，也不伪称找回原始布局代码。
逐图数值与输入哈希见extension/evidence/rebuilt_figures_attempt2/PLOTTED_VALUES.json和REPORT.json。
复现结果不会提高经验FPP的背景有效样本量，也不会赋予年度FAR分母。

## 公开前必须作者确认

GitHub/Zenodo账户和仓库、代码/权重/数据许可证、第三方再分发、贡献者/ORCID及公开时间。
当前状态：HOLD_FOR_AUTHOR_REVIEW_NO_PUBLICATION。
'''
    (ext/'README_CN.md').write_text(readme)
    scan={'files':0,'findings':[],'excluded_binary_extensions':True,'full_security_certification':False}
    patterns={
        'private_key':re.compile(r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----'),
        'github_token':re.compile(r'\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})'),
        'credential_assignment':re.compile(r'''(?i)\b[A-Za-z_]*(?:password|passwd|access_token|api_token|secret_key|zenodo_token|ssh_pass|github_token|auth_token)[A-Za-z_]*["']?\s*[=:]\s*["']([^"'\n]{6,})["']'''),
        'url_credentials':re.compile(r'https?://[^\s/:@]+:[^\s/@]{6,}@')}
    for source in (ext,root/'fixtures/test_real',root/'fixtures/recovery'):
        for path in source.rglob('*'):
            if not path.is_file() or path.suffix not in ('.py','.md','.json','.jsonl','.csv','.txt'):
                continue
            scan['files']+=1
            for n,line in enumerate(path.read_text(errors='replace').splitlines(),1):
                for key,pattern in patterns.items():
                    match=pattern.search(line)
                    if match:
                        scan['findings'].append({'path':str(path.relative_to(root)),'line':n,'kind':key,
                                                 'digest_only':hashlib.sha256(match.group(0).encode()).hexdigest()})
    save(ext/'reports/EXTENSION_SECRET_SCAN.json',scan)
    if scan['findings']:
        raise RuntimeError('Unreviewed secret-pattern hits; stop before packaging')
    print(json.dumps(validation),flush=True)

if __name__=='__main__':
    main()
