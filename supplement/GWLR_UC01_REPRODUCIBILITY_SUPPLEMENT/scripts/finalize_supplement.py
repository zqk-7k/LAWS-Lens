"""Finalize a separately versioned reproducibility supplement, never paper files."""
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import tarfile

import numpy as np

root = Path(sys.argv[1])
run = Path('/root/autodl-tmp/gw-catalog/results/gwlr_unified_c_physical_20260918T134500Z_r1')
def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()
def save(p, obj):
    (root/p).write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n')

clean = root/'verification/clean'
reference = root/'verification/production_v2'
assert json.loads((clean/'VERIFICATION.json').read_text())['state']=='PASS'
assert json.loads((root/'environment/CLEAN_BUILD_STATUS.json').read_text())['state']=='PASS'
a, b = np.load(reference/'NUMERICAL_OUTPUTS.npz'), np.load(clean/'NUMERICAL_OUTPUTS.npz')
assert set(a.files)==set(b.files)
comparisons = []
for key in a.files:
    # Freeze arithmetic tolerances, not scientific acceptance thresholds.
    rtol, atol = (1e-6, 1e-7) if ('embedding' in key or 'parameters' in key) else (1e-10, 1e-14)
    if key == 'xphm_fd':
        atol = 1e-32
    passed = bool(np.allclose(a[key], b[key], rtol=rtol, atol=atol))
    comparisons.append(dict(quantity=key, shape=list(a[key].shape),
        maximum_absolute_difference=float(np.max(abs(a[key]-b[key]))),
        relative_tolerance=rtol, absolute_tolerance=atol, passed=passed))
assert all(r['passed'] for r in comparisons)
save('verification/PRODUCTION_VS_CLEAN.json', {'state':'PASS', 'comparisons':comparisons,
    'scope':'waveform, SI conversion, three CPU/GPU short encoders and three BAYESTAR maps; not full retraining'})
build=json.loads((root/'environment/CLEAN_BUILD_STATUS.json').read_text())
build['scientific_validation']='PASS_FINITE_PORTABLE_REPLAY'
save('environment/CLEAN_BUILD_STATUS.json',build)
shutil.copy2('/root/autodl-tmp/gwlr_uc01_clean_env_20260919/pyvenv.cfg',root/'environment/clean-pyvenv.cfg')
shutil.copy2('/root/autodl-tmp/gwlr_uc01_portable_replay_20260919/verification/VERIFICATION.json',
             root/'verification/PORTABLE_LAYOUT_PRODUCTION_CHECK.json')
shutil.copy2('/root/autodl-tmp/gw-catalog/results/main_o3official_cfixed_v1_20260904_20260904T072435Z/scripts/main_o3official_cfixed_v1.py',
             root/'provenance/O3_OFFICIAL_JOIN_REFERENCE.py')

historical = json.loads((run/'contracts/PROTECTED_INPUTS.json').read_text())
frozen = json.loads((run/'contracts/FINAL_SCORE_FREEZE.json').read_text())['files']
current = {p: sha(p) for p in historical}
assert current == historical
assert all(sha(r['path'])==r['sha256'] for r in frozen)
save('verification/HISTORICAL_HASH_POSTCHECK.json', {'state':'PASS',
    'historical_files':len(historical), 'frozen_scoring_files':len(frozen),
    'changed_files':[], 'paper_files_accessed_or_modified':False})

artifacts = []
for path in [run/'package/GWLR_UC_01_deliverables_reviewed.tar.gz',
             run/'package/GWLR_UC_01_native_BAYESTAR_maps.tar.gz',
             root/'package/GWLR_UC01_reconstruction_inputs_and_code.tar.gz']:
    value=sha(path)
    expected=path.with_suffix(path.suffix+'.sha256').read_text().split()[0]
    assert value==expected
    artifacts.append({'path':str(path), 'filename':path.name, 'bytes':path.stat().st_size,
                      'sha256':value, 'public_url':None, 'doi':None})
save('publication/ARCHIVE_UPLOAD_PLAN.json', {'status':'PREPARED_NOT_PUBLISHED',
    'artifacts':artifacts, 'supplement_and_optional_wheelhouse':'See package/ and environment/WHEEL_ARTIFACTS.json',
    'publisher_account_not_accessed':True,
    'required_author_actions':['Confirm author names/ORCID and repository account',
        'Approve project data/code licenses and review third-party redistribution rights',
        'Upload, mint version DOI, verify public downloads, then update Data/Code availability'],
    'no_reserved_or_minted_doi':True})
save('publication/zenodo_metadata.template.json', {'title':'GW LensRank / TriLens: C_PHYSICAL reproducibility artifacts',
    'version':'GWLR-UC-01-C-REPRO-20260919', 'upload_type':'dataset',
    'creators':None, 'license':None, 'description':'Source-indexed C_PHYSICAL results, plans, code snapshots and verified environment locks.',
    'template_only_not_submission_ready':True})

inv=json.loads((root/'reports/INPUT_INVENTORY_FINAL.json').read_text())
verification=json.loads((clean/'VERIFICATION.json').read_text())
versions=json.loads((root/'environment/DEPENDENCY_GRAPH.json').read_text())['selected']
core=('torch','numpy','scipy','pandas','scikit-learn','bilby','lalsuite','pycbc','ligo-skymap','healpy','h5py')
version_table='\n'.join('| '+k+' | '+versions[k]+' |' for k in core)
readme=f'''# C 版本可复现性补充交付

代号：GWLR-UC-01-C-REPRO-20260919。仅补充归档与复现材料，不训练、不调权、不修改论文及历史结果。

## 一、已完成与未完成

- 原始输入清单：{inv['files']} 条文件记录、{inv['unique_contents']} 份唯一内容；全部现存文件计算 SHA-256，无未定位文件或未给出获取路径的条目。
- 覆盖 O3 62、O4a 74、O4b 86 个真实事件；每运行期 32 个注入噪声父块，train/validation/test 为 20/6/6。
- 192 个探测器噪声切片与获取清单所指原始 HDF5 应变逐样本一致，见 NOISE_BANK_RAW_RECONSTRUCTION.csv。
- 公共原始应变、PE、天图与 GW-LMC 输入不在紧凑包中重复分发。清单给出确切版本、URL、哈希和用途；原生 MOC 说明归档成员路径。
- 新建隔离环境，不继承 system-site-packages；按 116 包精确版本构建 wheel，离线 hash-locked 安装，pip check 通过。
- 在干净环境完成 SI 自旋转换、XPHM 波形、匹配滤波归一化、三个运行期短窗模型 CPU/GPU 前向、三个真实触发量 BAYESTAR 重建；数值与生产环境比较通过。
- {verification['metric_checks']} 项冻结指标复算通过，最大绝对差 {verification['max_metric_error']:.3g}。这是有限复现检查，不是重新训练全部 45 个模型或重做全部事件的定位。
- 原生产环境存在闲置 faiss-cpu 对 NumPy 的依赖冲突，保留原环境未修改。隔离的实际任务依赖不含该包；原 pip freeze/check 记录均保留。
- **尚未公开发布到 Zenodo 等长期仓库，没有新的 DOI。服务器 tar.gz 不等于公开归档。** 上传材料已准备，作者仍须确认署名、许可证并完成公开发布。

## 二、文件入口

| 交付 | 文件 |
|---|---|
| 完整原始输入获取清单 | inputs/RAW_INPUT_ACQUISITION.csv 与同名 JSON |
| 事件、PE group、探测器、用途 | inputs/INPUT_USES.json |
| 噪声父块、切片和 split | inputs/NOISE_BLOCKS_AND_RAW_FILES.csv |
| 下载与逐文件校验 | scripts/acquire_inputs.py |
| 实际依赖的精确版本及 wheel 哈希锁 | environment/requirements.linux-x86_64-cp312-cu128.lock |
| 完整生产环境快照 | environment/production-pip-freeze.txt |
| 隔离重建与 pip check | environment/CLEAN_BUILD_STATUS.json、03-offline-install.log、04-clean-pip-check.log |
| 数值复现与指标检查 | verification/clean/VERIFICATION.json、PRODUCTION_VS_CLEAN.json |
| 归档迁移复现入口 | scripts/portable_replay.py |
| 公开发布计划与作者待办 | publication/ARCHIVE_UPLOAD_PLAN.json |
| 软件许可证/公开边界 | publication/LICENSE_AND_RELEASE_REVIEW_CN.md |

## 三、获取原始输入

```bash
python scripts/acquire_inputs.py --manifest inputs/RAW_INPUT_ACQUISITION.json --destination /data/gwlr_inputs
python scripts/acquire_inputs.py --manifest inputs/RAW_INPUT_ACQUISITION.json --destination /data/gwlr_inputs --ids input_0000 --execute
```

默认只列清单；执行需显式 --execute。支持续传和严格哈希校验，已存在但哈希不符的文件不会覆盖。
对已冻结的衍生噪声/PSD/日历和官方审计表，先解压 reconstruction_inputs_and_code 包；不能把它们误写成官方原始发布产品。
去重输入约 {inv['unique_content_bytes']/2**30:.1f} GiB；这不是紧凑复现所需磁盘量。完整重新训练的临时数据还需要额外空间。

所有本地文件已核验内容哈希。公网仅抽样检查链接、完整重取三个 GW-LMC CSV；没有重新下载整个应变/PE corpus。
O4a 官方 zip 获取路径及成员名来自既有解压记录，另保留发布方 zip 校验值；本次未重新下载整个 zip。

## 四、环境重建

平台：Linux x86_64、Python 3.12.3、glibc 2.35、PyTorch CUDA 12.8。GPU 驱动不属于 pip 依赖，见 environment/nvidia-smi.txt。

```bash
python3.12 -m venv clean_env
clean_env/bin/python -m pip install --no-index --find-links /path/to/wheelhouse --require-hashes -r environment/requirements.linux-x86_64-cp312-cu128.lock
clean_env/bin/python -m pip check
```

离线锁对应本次 wheelhouse 的具体字节；astroplan/pims 为本地构建 wheel，不能从 PyPI 任意重建后要求压缩包哈希相同。
完整 wheel 清单及 SHA 在 environment/WHEEL_ARTIFACTS.json；实际构建命令和日志同时保留。此锁未宣称支持 Windows/macOS 或不同 Python ABI。

| 软件 | 锁定版本 |
|---|---|
{version_table}

## 五、不依赖历史服务器路径的有限复现

提供 reviewed 结果包和 reconstruction_inputs_and_code 包后，在新目录执行：

```bash
clean_env/bin/python -B scripts/portable_replay.py \\
  --reviewed GWLR_UC_01_deliverables_reviewed.tar.gz \\
  --assets GWLR_UC01_reconstruction_inputs_and_code.tar.gz \\
  --assets-manifest inputs/RECONSTRUCTION_ASSETS.json \\
  --destination /data/gwlr_c_portable_replay \\
  --verify-script scripts/verify_environment.py
```

这会校验包、拒绝覆盖、解压到新的目录并复算上述有限测试。归档布局为 training/、completion/、project/；不要求把旧绝对路径作为复现输入。
全量训练的历史脚本仍包含归档路径和保护性合同，不能直接运行控制器覆盖旧目录。完整从零训练的跨机器路径迁移未在本次有限复现中认证。

## 六、科学边界

C 为同一透镜系统共同振幅缩放、同一带噪应变恢复触发量和 BAYESTAR 天空图的受控实验。
它不是完整宇宙学透镜率模拟，不是所有注入的完整 BBH PE；预定时刻条件化搜索、有限噪声父块、天空 coverage 偏差等限制仍保留。
公开 PE/官方结果仅用于排名后审计；本次没有新运行 Hanabi。历史开发数据不能因重新打包而称为新的独立盲测。
复现包不包括作者私钥、密码、远程登录凭证或原始应变大文件。

## 七、服务器与归档

服务器：connect.westd.seetacloud.com，SSH 端口 32328。登录凭证不写入交付。
本补充目录：{root}
原 C 目录：{run}
原 reviewed 包和 native BAYESTAR 包保持原 SHA；新 reconstruction 包独立存放。
本轮不会向论文、Git 或 Overleaf 写入数据。
'''
(root/'README_CN.md').write_text(readme)
(root/'publication/LICENSE_AND_RELEASE_REVIEW_CN.md').write_text('''# 公开发布前的许可与署名审查

本材料目前是服务器上的作者审阅包，不是已发布的数据 DOI。
本轮不代作者确定数据或代码许可证，不伪造作者姓名、ORCID 或论文 DOI。
归档包含作者脚本和历史依赖代码快照。公开分发前须确认项目许可证及第三方代码许可。
116 包 wheelhouse 保留包内许可证，但不能把它们统一改称本项目自有代码；CUDA/NVIDIA 与 GPL 等组件需分别审核再决定是否随公共归档分发。
原始 GWOSC、LVK PE、GW-LMC 和 LVK lensing 表优先通过原发布方下载；本项目仅提供精确引用和哈希。
脚本扫描排除了三份含潜在凭证赋值或扫描规则命中的无关历史脚本，文件名见 SOURCE_CODE_CREDENTIAL_SCAN.json；未记录其敏感内容。
自动扫描不替代作者对最终归档的人工安全与许可审查。
''')
(root/'publication/DATA_CODE_AVAILABILITY_DRAFT_CN_EN.md').write_text('''# Data/Code Availability 草稿及使用条件

## 目前可以如实写的状态
已准备 C_PHYSICAL 的带 SHA-256 结果归档、逐文件原始输入获取清单、模型与配置、原生注入天空图、有限复现脚本及已验证的依赖锁。公开长期归档 DOI 尚待作者发布；不能写成“全部数据已经公开”。

## 发布完成后使用的英文模板
The derived data, trained checkpoints, frozen analysis configurations, native injection sky maps, input acquisition manifests, and reproducibility scripts supporting the C_PHYSICAL analysis are archived at [AUTHOR: insert the released version DOI]. Public strain and parameter-estimation products are obtained from their original releases using the versioned URLs, posterior-group identifiers, and checksums recorded in the acquisition manifest. A hash-locked software environment and finite numerical replay tests are supplied. These tests verify selected waveform and sky computations and archived retrieval statistics; they do not constitute a new blind test or an independent full retraining of all models. Code availability and third-party licensing are documented in the released archive.

方括号处是明确未完成的作者操作，不得原样投稿或填写虚构 DOI。
原始输入引用可查 inputs/RAW_INPUT_ACQUISITION.json；GW-LMC 固定提交见 provenance/GWLMC_GIT.json；O3/O4a 官方透镜表对应 https://zenodo.org/records/7693837 与 https://zenodo.org/records/18163632；O4b PE/天图对应 https://zenodo.org/records/20348005 和 https://zenodo.org/records/20348006。
''')

save('reports/REPRODUCIBILITY_FINAL_STATUS.json', {'code':'GWLR-UC-01-C-REPRO-20260919',
    'input_manifest_complete_for_frozen_C_inputs':True, 'clean_environment_verified':True,
    'finite_replay_pass':True, 'public_archive_published':False, 'doi':None,
    'full_cross_machine_retraining_verified':False, 'paper_modified':False,
    'historical_hashes_unchanged':True,
    'remaining_author_actions':['public repository publication and DOI','authorship/license approval'],
    'scope_limitations':['full raw corpus not re-downloaded','full training not re-run','O4a official zip member routes not re-extracted this turn']})
print('SUPPLEMENT_REPORTS_READY', flush=True)
