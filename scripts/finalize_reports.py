"""Summarize measured validation scope and unresolved public-release gates."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--release',type=Path,required=True)
    a=ap.parse_args(); R=a.release
    def read(p):
        return json.loads((R/p).read_text())
    out=R/'reports';out.mkdir(exist_ok=True)
    unit=[]
    update=R/'uploaded_snapshot/local_updates'
    for label,folder,test in [
        ('FPP',update/'0922_FPP结果/extracted/gwlr_fpp_injection_real_01_20260922T061000Z_r2/scripts','test_fpp_core.py'),
        ('TAIL',update/'0922_GWLR_TAIL02诊断/extracted/gwlr_tail02_20260922T113500Z_r2/scripts','test_tail_audit.py')]:
        result=subprocess.run([sys.executable,'-B',str(folder/test)],cwd=folder,capture_output=True,text=True)
        (R/'logs'/f'{label}_unit_tests.log').write_text(result.stdout+result.stderr)
        unit.append(dict(suite=label,returncode=result.returncode))
    (out/'UNIT_TESTS.json').write_text(json.dumps(unit,indent=2))
    paper=read('verification/paper/REPORT.json')
    missing=[]
    schematics={'figures/workflow_author_original_20260915.png','figures/waveform_compatibility_workflow.pdf'}
    for row in paper['figures']:
        if not row['verified_redraw'] and row['artifact'] not in schematics:
            missing.append(dict(category='current_figure_entrypoint',item=row['artifact'],
                reason='Current asset and source tables present; exact current plotting entry point not found/verified.'))
    missing += [
        dict(category='full_regeneration',item='entire injection population + recovery + training',reason='Representative source/noise regeneration verified, not complete population retraining or bank search.'),
        dict(category='acquisition',item='fresh public-input acquisition',reason='847 local inputs inventoried; complete raw corpus was not downloaded again.'),
        dict(category='portability',item='new physical host / other GPU',reason='Historical-directory access blocked on current server; not a separate-host validation.'),
        dict(category='scientific_boundary',item='real FPP and annual FAR',reason='Reproducible conditional/background and post-hoc fixed-endpoint model, not validated real significance or annual exposure.'),
        dict(category='author_decision',item='GitHub repo/account, licenses, software contributors, ORCID, publish date',reason='Pending author confirmation; no public actions authorized.'),
        dict(category='license_review',item='third-party code/data/model redistribution',reason='Metadata inventory supplied, legal redistribution rights not certified.'),
    ]
    (out/'MISSING_ITEMS.json').write_text(json.dumps(missing,ensure_ascii=False,indent=2))
    state=dict(status='HOLD_FOR_AUTHOR_REVIEW_NO_PUBLICATION',
        paper_commit='afb86ff5c8f212aec579592f0d7439f3f909d931',main_version='GWLR-UC-01/C_PHYSICAL',
        figures_full_reproduction=False,representative_injection_regeneration=True,
        model_inference_validation_all450=True,public_upload=False,doi=None,
        original_upload_preserved=True,historical_results_modified=False,
        missing_current_data_figure_entrypoints=sum(x['category']=='current_figure_entrypoint' for x in missing),
        independent_full_pipeline_reproduction=False)
    (out/'RELEASE_VALIDATION_STATUS.json').write_text(json.dumps(state,indent=2))
    contract=read('contracts/RELEASE_STATE.json')
    contract.update(status=state['status'], independent_full_pipeline_reproduction=False,
                    validation_report='reports/RELEASE_VALIDATION_STATUS.json')
    (R/'contracts/RELEASE_STATE.json').write_text(json.dumps(contract,ensure_ascii=False,indent=2))
    # Preserve the finite failed attempts instead of concealing them in a final PASS.
    (out/'FAILURE_AND_REPAIR_HISTORY.json').write_text(json.dumps([
        dict(attempt='four-event model forward',status='FAIL_ORIGINAL_TOLERANCE',
             reason='Changed neural batch shape versus archived 450-event batching.',
             repair='Replay original 450-event context with unchanged weights and tolerances; all validation arrays subsequently compared.'),
        dict(attempt='portable import first pass',status='MISSING_DEPENDENCY',
             missing='historical calfuse.py helper',repair='Added byte-identical source helper to release runtime; original project untouched.'),
        dict(attempt='portable feature operator first pass',status='MISSING_DEPENDENCY',
             missing='historical waveform_multiscale_data.py helper',repair='Added byte-identical helper; replay passed under historical-read guard.')
    ],indent=2))
    # Archive lightweight validation evidence; exclude the venv and large duplicate outputs.
    evidence=out/'validation_evidence';evidence.mkdir(exist_ok=True)
    for p in (R/'verification').rglob('*'):
        if not p.is_file() or p.is_symlink() or 'fresh_env' in p.relative_to(R/'verification').parts:
            continue
        if p.name=='REPORT.json' or p.parent==R/'verification' and p.suffix=='.json' or p.suffix=='.log':
            if p.stat().st_size>16*2**20:
                continue
            dest=evidence/p.relative_to(R/'verification');dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    fixtures=R/'fixtures/inference_full_context'
    full=read('verification/FULL_VALIDATION_PREDICTION_CHECK.json')
    report='''# LAWS-Lens 论文发布整理与分层验证报告

本轮是独立的内部发布草稿，不是正式公开版。原始上传、论文、模型、候选排名及历史包均未覆盖；没有创建 GitHub 仓库、Zenodo 记录或 DOI。

## 1. 冻结版本

- 主实验：GWLR-UC-01 / C_PHYSICAL，使用 reviewed 包及配套原生 BAYESTAR 图。
- 论文：afb86ff5c8f212aec579592f0d7439f3f909d931。
- 追加分析：GWLR-FPP-01、TAIL-02、q95 起点的 B=4/4.5/5 固定端点分析。
- 速度：DOMAIN04 原始实测，不把本轮验证耗时替换为论文测量。

## 2. 已实测验证

| 层级 | 实际检查 | 边界 |
| --- | --- | --- |
| 输入 | 1,093 个上传文件、六个归档哈希重新校验；847 条原始输入记录在服务器存在 | 未重新下载整个公共原始数据集 |
| 指标 | 648 项450事件及选定190子目录指标复算通过；16张论文/服务器公共表一致；500子目录先模型内平均、再模型间均值/SD汇总复算通过 | 不是重新生成500份独立数据；未重算所有500子目录的每个pair |
| FPP/GPD | 9组固定端点拟合、450项候选计算、873项删除敏感性复算；16份原表/论文表对照通过 | 数值复现不证明真实显著性，B=4不是物理上限 |
| 绘图 | 图4与端点敏感性图重绘；图4底层曲线一致；15项论文图片引用均已盘点 | 字体/二进制PDF不要求完全相同；其他最新版逐图入口尚未齐全 |
| 特征 | 每运行期4个固定事件，重新计算短窗/16秒/R特征，最大差不超过9.54e-7 | 使用冻结模板数组；没有重建所有理论模板 |
| 模型 | 3运行期 × 3模型 × 5组件，按原450事件批处理重新前向并核对全部validation数组 | 未重训，没有重新选择参数；本次未全量推理真实和test目录 |
| 注入 | 每运行期1个固定validation双像源，共6事件；从源参数及真实噪声切片重新生成64秒应变、2秒及16秒输入，原始数组校验一致 | 不代表全体注入再生产或总体物理真实性的新验证 |
| 天空 | 三运行期各1个冻结恢复触发量重新运行BAYESTAR，归一化和ordering检查通过 | 不是本轮重新扫描8192模板库恢复所有触发 |
| 可移植性 | 补齐动态导入的历史代码，在禁止打开原项目路径的保护下重新运行模型、特征和注入样例 | 仍在同一物理服务器，不能称为跨机器全流程复现 |
| 环境 | 锁定依赖、wheel哈希、离线安装入口和pip check；另建立本轮全新隔离环境 | 固定Linux x86_64/CPython3.12/CUDA12.8，其他平台未验证 |

## 3. 发现并修复的问题

小批次模型推理与归档输出有差异。恢复归档批处理上下文后，通过原容差；没有为通过测试改权重或放宽容差。另有历史模块在旧包里仅靠服务器绝对路径读取，已补进独立runtime，并增加导入重定位和旧目录读取阻断。失败经过见 FAILURE_AND_REPAIR_HISTORY.json。

## 4. 尚不能宣称完整公开复现

缺少11项数据图的最新版可执行绘图入口或其验证；已有数据及图文件不等于已重画全部图。两幅流程示意图另保留静态资产，不伪称数值程序输出。

全量公共输入重新下载、全量注入重新恢复触发/天空、从头训练及新物理机器完整复现尚未执行。不能把局部通过扩大解释为全流程通过。请结合 MISSING_ITEMS.json 和原始运行日志阅读。

FPP/GPD仍保留统计解释限制：4005相关背景对不是4005独立事件；固定端点在查看候选后指定；没有建立年度FAR分母。本轮不修改论文陈述，也不替作者确认科学采纳。

## 5. 安全、许可与发布

自动秘密模式扫描、文件哈希、依赖许可元数据随包提供；这不是无漏洞或合法再分发的保证。扫描器自身的私钥头部字节常量不是私钥。没有收集用户SSH私钥、账号密码、访问token或Git配置。二进制及第三方归档的人工发布复核仍需完成。

自有代码、模型、数据许可尚未确定；第三方许可需要逐项确认。论文作者信息已保留，但软件贡献者名单和ORCID不能猜填。仓库、账号、署名、许可和公开时间须作者确认。

最终状态：HOLD_FOR_AUTHOR_REVIEW_NO_PUBLICATION。
'''
    (out/'RELEASE_AUDIT_CN.md').write_text(report)
    readme='''# LAWS-Lens paper release v1.0.0 - INTERNAL DRAFT

Read `reports/RELEASE_AUDIT_CN.md` and `reports/MISSING_ITEMS.json` first.
This is not a public release or a full independent reproduction certification.
Do not publish before author and third-party license review.

## Bundle layout

Extract the code/model, paper/analysis, fixtures, and audit tarballs into one empty directory. Each contains a distinct payload plus a package-specific manifest. Keep the native-map and wheelhouse archives as separate large artifacts. `packages/RELEASE_PACKAGE_INDEX.json` records all sizes and hashes.

## Environment

Use CPython 3.12 on Linux x86_64. Unpack the verified wheelhouse and locate its `.whl` directory, then:

```bash
python3.12 scripts/create_environment.py --target /NEW/PATH/laws-env --wheelhouse /PATH/TO/WHEELS --lock supplement/GWLR_UC01_REPRODUCIBILITY_SUPPLEMENT/environment/requirements.linux-x86_64-cp312-cu128.lock --report-dir validation_environment
```

The torch lock uses CUDA 12.8; a compatible NVIDIA driver is required for feature/model tests. Model checkpoint loading uses trusted project pickle artifacts; never load untrusted checkpoints.

## Finite validation entry points

From the extracted release root, using the new environment:

```bash
python scripts/reproduce.py paper
python scripts/reproduce.py metrics-sky
python scripts/reproduce.py features
python scripts/reproduce.py models
python scripts/reproduce.py injections
```

Each writes a new `verification/` subdirectory and a log; an existing output is not overwritten. Use a new unpacked workspace for repeat runs. These commands do not train, tune, rank a new catalog, or publish. Model comparison preserves the archived 450-event batch context. The fixture includes full-array expected digests and selected numerical reference arrays. The injection fixture regenerates one validation doublet per run, not the entire population. Templates and noise slices are frozen inputs.

## Raw acquisition and provenance

`supplement/GWLR_UC01_REPRODUCIBILITY_SUPPLEMENT/inputs/RAW_INPUT_ACQUISITION.json` includes 847 exact input records, URLs/archive members, hashes and roles. Companion noise, selection, PE-group and environment records are retained alongside it. `scripts/acquire_inputs.py` in that supplement is the acquisition helper; consult its `--help` before use. Full raw input re-download was not done in this release audit.

The frozen C generation/training algorithms and configs are in `runtime/training/` and `runtime/project/`; replay entry points relocate imports without modifying numerical definitions. Full training and all-input portable execution are NOT yet certified. Historical absolute paths in archived contracts remain provenance, not recommended write destinations.

## Rights and citation

No open-source/data/model license has been selected. No GitHub repository, Zenodo record or DOI was created. Manuscript authors are not automatically asserted to be software contributors. Do not redistribute the internal manuscript or third-party material until permission is confirmed.

## Statistical boundary

GPD B=4 was a post-candidate modeling choice; B=4.5/5 sensitivities are preserved. Reproducing those numbers does not validate real-catalog FPP or annual FAR. Official candidate overlap is not lensing ground truth.
'''
    (R/'README.md').write_text(readme)
    print(json.dumps(dict(status=state['status'],missing_figure_entrypoints=state['missing_current_data_figure_entrypoints'],
        full_validation_arrays=len(full['arrays']),all_exact=full['all_exact'],unit_tests=unit)))

if __name__=='__main__':
    main()
