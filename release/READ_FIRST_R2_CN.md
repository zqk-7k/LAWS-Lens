# LAWS-Lens 私有发布候选 r2

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
