# 发布许可与上传核查（2026-09-25）

## 结论

作者许可与署名已确认；第三方审查已完成第一轮清单和来源核对，尚未全部通过。
GitHub 可以继续提供自有 MIT 代码及带原许可的 Phazap；Zenodo 两个记录仍是草稿。
不得将“许可已确认”“依赖哈希可定位”或“已开始传输”写成最终发布完成。

## 已完成

- 自有软件 MIT；自生成数据和模型权重 CC BY 4.0；第三方权利不变。
- 六位作者及单位按论文快照登记，无未经确认的 ORCID。
- 检查八个正式归档，共 6,931 个成员（包括目录项），118 个 wheel 出现次数、116 个不同 wheel。
- 为 Phazap 补原作者 MIT 许可。八个源码与上游提交
  `676152529bc1c4349ddc0531bb01b5e626034907` 逐字节一致。
  此次比较不是对原始安装时间或安装提交的追认。
- 固定 GW-LMC 提交 `55c9e1df770e4ba21815fd20233eab240e551d9d`
  的 LICENSE 为 CC0-1.0，原文已附；它不是本项目 CC BY 4.0 数据。
- 113 个 wheel 在 PyPI 元数据中找到完全相同的 SHA-256；PyTorch CUDA wheel
  在官方索引中找到相同哈希。这里只重新获取元数据，未重新下载全部二进制。
- astroplan 和 PIMS 为本地构建 wheel；原始源码下载链接已列出，
  不能把官方源码哈希当成这两个 wheel 的哈希。
- 1,263 个冻结软件文件重新校验通过；模型、科学配置、评分与排名未变。
- 已将批准的作者和许可写入两个 Zenodo 草稿；它们仍为 submitted=false。

## 尚需闭合的第三方事项

| 内容 | 本轮证据 | 发布处置 |
|---|---|---|
| Phazap | MIT、八文件一致、原版权声明补齐 | 原许可保留，可随软件分发 |
| GW-LMC 三张原表 | 固定提交 CC0 | 保留原许可和论文引用 |
| 离线 wheelhouse | 12 个主许可字段为 GPL；另有 12 个 NVIDIA 专有声明；还存在包内嵌套依赖许可 | 暂不公开整包，核对对应源代码供给和各组件再分发条件；也可改为从官方源按哈希获取 |
| 本地构建 wheel | astroplan / PIMS | 补构建来源与 BSD 声明；不可假定元数据足够 |
| 论文模板 | sn-jnl.cls / sn-nature.bst 含 LPPL 声明，cls 另有随 LaTeX base 分发的文字 | 不能统一 MIT/CC BY；需满足分发条件或在新公开包中用官方获取入口替代 |
| 公共 PE、官方候选表和图表素材 | 历史包保留来源索引 | 逐来源记录许可与引用，不因已公开下载就归为自有数据 |

文件名扫描未找到 Cython、ligo.skymap、PIMS、tqdm wheel 的独立 LICENSE/COPYING
文件，不等于这些软件没有许可证；仍需按元数据、源包和实际条款核实。
这是一份工程发布审查，不是对全部第三方权利的法律保证。

## 上传补救

原上传保留成功文件和校验回执，未覆盖原档案。三个重复失败的大包另制
64 MiB 传输分卷：paper_fpp_speed、validation_fixtures、native_BAYESTAR_maps。
每卷记录 SHA-256 与字节数；合并脚本再次核对原档案 SHA-256，拒绝覆盖不同文件。
分卷只改变传输方式，不改变任何科学内容。合并器已通过正确内容、重复运行和
损坏分卷拒绝测试。

完整单包上传曾发生 TLS EOF，分卷清单首次上传还发生 HTTP 504。
这些失败保留，重试不允许写成成功。服务器任务只上传草稿并回读校验，
不含自动发布动作。4.24 GB 离线依赖包暂不重试，等待上述许可处置。

## 入口

- 作者批准：`release/AUTHOR_APPROVAL_20260925.json`
- 依赖清单：`release/THIRD_PARTY_WHEEL_SBOM.json`
- 官方依赖来源：`release/DEPENDENCY_UPSTREAM_ARTIFACTS.json`
- Phazap 核对：`release/PHAZAP_SOURCE_COMPARISON.json`
- 许可范围：`LICENSE_STATUS.md`、`THIRD_PARTY_NOTICES.md`、`licenses/`
- 软件草稿：https://zenodo.org/deposit/22949904
- 数据草稿：https://zenodo.org/deposit/22949906

上述草稿地址需要所有者访问；预留 DOI 不是已发布 DOI。历史包不改名、不删除、不覆盖。

## 03:59 UTC 更新

结束停滞的旧上传连接后，小文件写入恢复。新增 11 项许可/署名/软件包材料已
上传并完整回读 SHA-256，另有传输清单和合并脚本两项通过。软件 rc2 也通过
GitHub 下载校验。旧连接 PID 202154、204884 已结束；替代分卷任务 PID 205053
在服务器运行，不自动发布。分卷范围现扩展为四个大档案，增加 code_models_scores，
共 65 卷；这是传输调整，不是修改模型。八个原始归档已再次逐一核对 SHA-256，全部不变。
