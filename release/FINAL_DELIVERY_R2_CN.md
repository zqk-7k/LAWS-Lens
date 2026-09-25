# 最终交付核验：LAWS-Lens 私有发布候选 r2

本轮私有整理与有限分层验证已完成。状态仍为 HOLD_FOR_AUTHOR_REVIEW_NO_PUBLICATION。

- 共8个归档，8641841028字节（8.642 GB）。
- 八包SHA-256校验通过；六个模块包在独立目录合并解压，没有文件覆盖冲突。
- 解压后的4017个清单条目逐文件哈希吻合。
- 四个新增入口实际复跑：图表、全部500子目录、test/real模型、六事件应变生成至完整定位，全部PASS。
- 复跑后打包的模型、配置及数据文件无变化；模型/指标/定位的主进程历史目录读取隔离检查通过。
- 数值算法未修改，没有训练、调权、更新排名或修改论文。

完整清单：RELEASE_PACKAGE_INDEX_R2.json；哈希：SHA256SUMS_R2.txt。
只使用清单中的八个推荐包；旧的r2扩展失败包保留供审计，不要与r2p1修订包一起解压。
最终状态：FINAL_VALIDATION_R2.json；压缩包实跑证据：DELIVERED_EXTENSION_REPLAY.json及REPLAY_*.log。
归档内部的RELEASE_VALIDATION_STATUS_R2.json是封包前快照，因此保留PENDING_PACKAGING_TEST；以上独立哈希的最终回执更新该状态为PASS，未为更新说明而改写已冻结归档。

阅读完整报告：extension/reports/RELEASE_COMPLETION_R2_CN.md。
复现入口与命令：extension/README_CN.md。

仍未完成的范围：全量公共输入重下载、完整人口重训、另一物理主机验证，以及11张原始布局脚本的找回。后者已有带数据哈希和数值清单的独立重建脚本，不是原作者脚本的冒充。
公开前仍需作者确认账户、许可、贡献者/ORCID和公开时间，并审核第三方再分发。没有创建仓库、DOI或自动公开。
本轮验证不赋予真实候选新的显著性，也不补齐年度FAR曝光。
