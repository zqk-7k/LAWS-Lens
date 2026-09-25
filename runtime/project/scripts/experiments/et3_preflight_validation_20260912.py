#!/usr/bin/env python3
"""Tests, bounded resolution diagnostics and progress report for ET NEW-SCORE-ONLY."""
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "2"
import argparse
from pathlib import Path
import time
import numpy as np
import pandas as pd
import healpy as hp
import et3_new_score_only_20260912 as et


def normalized_mass(values):
    p = np.asarray(values, dtype=np.float64)
    if not np.isfinite(p).all() or (p < 0).any() or p.sum() <= 0:
        raise ValueError("Invalid sky probability")
    return p / p.sum()


def tests(root):
    rows = []
    n = hp.nside2npix(16)
    uniform = np.full(n, 1/n)
    rng = np.random.default_rng(202609120)
    p = normalized_mass(rng.uniform(size=n))
    q = normalized_mass(rng.uniform(size=n))
    b = n * p.dot(q)
    def check(name, condition, value):
        rows.append({"test": name, "pass": bool(condition), "value": float(value)})
        if not condition:
            raise AssertionError(name)
    check("uniform_vs_any_BF_one", abs(n*uniform.dot(p)-1) < 1e-12, n*uniform.dot(p))
    reordering = n * hp.reorder(p, n2r=True).dot(hp.reorder(q, n2r=True))
    check("common_NESTED_to_RING_score_invariant", abs(reordering-b) < 1e-12, reordering-b)
    area = hp.nside2pixarea(16)
    continuous = 4*np.pi*np.sum((p/area)*(q/area))*area
    check("probability_mass_vs_density_convention", abs(b-continuous) < 1e-12, b-continuous)
    check("symmetry", abs(p.dot(q)-q.dot(p)) < 1e-14, p.dot(q)-q.dot(p))
    time_grid = np.arange(et.NSAMPLES)/et.FS
    low = np.sin(2*np.pi*30*time_grid)
    high = np.sin(2*np.pi*190*time_grid)
    # The comparison uses the same robust normalization: high-frequency power
    # must disappear relative to the retained low-frequency component.
    combined = et.low_view(np.tile(low+.5*high, (3,1)))
    low_only = et.low_view(np.tile(low, (3,1)))
    a = np.fft.rfft(combined[0])
    f = np.fft.rfftfreq(4096, 1/256)
    ratio = abs(a[np.argmin(abs(f-66))])/abs(a[np.argmin(abs(f-30))])
    check("190Hz_alias_at_66Hz_suppressed", ratio < 1e-3, ratio)
    check("long_window_exact_shape", combined.shape == (3,4096), combined.shape[-1])
    check("long_view_finite", np.isfinite(low_only).all(), float(np.isfinite(low_only).mean()))
    _, detectors, geometry = et.geometry()
    for g in geometry:
        check(g["detector"]+"_exact_geometry", g["adapter_tensor_error"] < 3e-8, g["adapter_tensor_error"])
    try:
        normalized_mass(np.array([np.nan, 1]))
        check("NaN_rejected", False, 0)
    except ValueError:
        check("NaN_rejected", True, 1)
    pd.DataFrame(rows).to_csv(root / "tables/UNIT_TESTS.csv", index=False)
    et.json_write(root / "contracts/UNIT_TESTS_PASS.json", {"utc": et.utc(), "tests": len(rows), "all_pass": True})


def convergence(root):
    from astropy.table import Table
    from ligo.skymap import moc
    from scipy.stats import spearmanr
    events = et.json.loads((root / "contracts/PILOT_EVENTS.json").read_text())
    n = len(events)
    idx = np.triu_indices(n, 1)
    frame = pd.DataFrame({"event_i": [events[i]["event_uid"] for i in idx[0]],
                          "event_j": [events[j]["event_uid"] for j in idx[1]],
                          "companion": [events[i]["source_uid"] == events[j]["source_uid"] for i,j in zip(*idx)]})
    elapsed = {}
    matrices = {}
    for nside in (256, 512, 1024):
        started = time.perf_counter()
        prob = np.empty((n, hp.nside2npix(nside)), dtype=np.float64)
        for k, event in enumerate(events):
            path = root / "pilot/maps" / (event["event_uid"].replace(":", "_")+".fits.gz")
            table = Table.read(path, format="fits")
            raster = moc.rasterize(table[["UNIQ", "PROBDENSITY"]], order=int(np.log2(nside)))
            prob[k] = normalized_mass(np.asarray(raster["PROBDENSITY"])*hp.nside2pixarea(nside))
        dot = prob @ prob.T
        bf = hp.nside2npix(nside)*dot
        with np.errstate(divide="ignore"):
            z = np.log(bf)
        frame[f"Z{nside}"] = z[idx]
        frame[f"zero_overlap_{nside}"] = dot[idx] == 0
        matrices[nside] = z
        elapsed[str(nside)] = time.perf_counter()-started
        del prob
    summaries = []
    for lo, hi in ((256,512),(512,1024)):
        for name, use in (("all", np.ones(len(frame),bool)), ("companion", frame.companion.to_numpy()),
                          ("null", ~frame.companion.to_numpy())):
            a = frame.loc[use, f"Z{lo}"].to_numpy()
            b = frame.loc[use, f"Z{hi}"].to_numpy()
            finite = np.isfinite(a)&np.isfinite(b)
            difference = abs(a[finite]-b[finite])
            summaries.append({"coarse":lo, "reference":hi, "stratum":name, "pairs":len(a),
                              "finite_pairs":int(finite.sum()), "nonfinite_pairs":int((~finite).sum()),
                              "abs_difference_median":float(np.median(difference)),
                              "abs_difference_q99":float(np.quantile(difference,.99)),
                              "abs_difference_max":float(difference.max()),
                              "sign_flips":int(np.sum((a>0)!=(b>0))),
                              "spearman":float(spearmanr(a,b).statistic)})
    frame.to_csv(root / "tables/PILOT_RESOLUTION_PAIRS.csv", index=False)
    pd.DataFrame(summaries).to_csv(root / "tables/PILOT_RESOLUTION_SUMMARY.csv", index=False)
    et.json_write(root / "contracts/PILOT_RESOLUTION_AUDIT.json", {
        "utc":et.utc(), "events":n, "pairs":len(frame), "raster_and_score_seconds":elapsed,
        "scope":"training-only pilot; not full validation or test resolution gate",
        "dense_maps_persisted":False, "formal_nside":512, "temperature":1.0})


def report(root):
    audit = pd.read_csv(root / "tables/SKY_PILOT.csv")
    tests = pd.read_csv(root / "tables/UNIT_TESTS.csv")
    conv = pd.read_csv(root / "tables/PILOT_RESOLUTION_SUMMARY.csv")
    runtime = et.json.loads((root / "contracts/SKY_PILOT_COMPUTATIONAL_PASS.json").read_text())
    text = f"""# ET3-NEW-SCORE-ONLY-BAYESTAR 启动与输入检查

本轮为独立 ET-3 重做。未完成最终训练与检索评估，不是已经得到的新 Recall 结果。
历史 ET-3、O3/O4a、论文及所有旧结果均不修改。最终采纳状态为 `{et.STATUS}`。

## 版本与服务器

- 服务器：`connect.westd.seetacloud.com:32328`。
- 独立目录：`{root}`。
- 波形参考：`MCWF-UNIFIED-PATH875-DEVCONF` 的 NEW-SCORE-ONLY，即 alpha=1。
- 删除的是外层 0.125 旧总分 + 0.875 新总分混合，不删除内部旧短窗或 OMC 信息。
- 这不是 NODUP-DIRECT，也不是后续 R75/R77。

## 完整方法目标

1. 保留各 seed 的三通道 ET attention-Inception 短窗模型，输入为 tc 前1.75秒到后0.25秒，2048Hz、4096点。
2. 从原24秒数据提取20--80Hz、tc前15.75秒到后0.25秒的16秒分支；抗混叠降采样到256Hz，4096点。
3. 长短窗分别生成质量/质量比/自旋模板匹配响应；检测器特征按三个ET通道和网络组合适配，不能机械沿用HL的27通道。
4. 训练质量轴网络及条件 eta/chi_eff 预测模块，得到 p(Mc)*p(eta,chi_eff|Mc)。ET模型须在ET训练源上训练，不能把HL权重直接当ET已训练模型。
5. 比较事件对的预测分布重合，按模拟 development/validation 冻结T、I校准与系数，得到 Zwf_new=Zwf_OMC+gamma*T+beta*I。该预测分布不是公开PE或完整贝叶斯参数后验。
6. 时间仍用ET合成目录对应的冻结一维时间证据。不能套用O3观测期或引入二维SNR时间方案。
7. 用ET自身响应、PSD、SNR和独立高斯触发测量生成条件BAYESTAR天空图；不旋转公开PE模板。共同Nside512计算 Zsky=log[Npix sum PiPj]。
8. 只在模拟validation选择ET融合权重；最终直接加权 waveform/time/sky 三矩阵，无外层新旧总分混合。
9. 最后评估五个seed的SIS、PM和总体召回、AUPRC、F50/F90、Top-B、system-bootstrap区间及参数/天空覆盖率。ET没有真实LVK或公开PE候选标签，不能虚构官方重合统计。

## 已通过的检查

- 完整输入：50,000幅像/事件，每幅3x98,304点，24秒、4096Hz。
- 20个pilot事件来自所有五个训练集的交集，不读取测试排名。
- 20个旧2秒输入逐点复现一致；16秒输出均为3x4096，有限且无重复通道。
- 单元测试：{int(tests['pass'].sum())}/{len(tests)}。
- BAYESTAR有效地图：{len(audit)}/20；仅保存原生MOC，不永久保存dense512/1024。
- 时间单位、窗口和源系/探测器系质量显式记录。
- BAYESTAR单事件时间P50/P90/max：{runtime['seconds_P50_P90_max']}秒。
- 50,000事件串行时间外推P50/P90/max：{runtime['full_50000_events_serial_hours_P50_P90_max']}小时。不是并行完成时间保证。

## 不能忽略的实现差异

原ET文件名含strain，但内容已在生成时用探测器PSD白化，不能再次除PSD。16秒从完整已白化数组提取，不是拉长2秒输入。
LAL E2/E3的默认几何不等同于Bilby ET2/ET3。本轮传入原生成器的顶点和探测器张量，并在局部进程内恢复查找函数，不修改安装库。
旧GWTC BAYESTAR归档脚本向 `bilby_to_lalsimulation_spins` 传入 `m1_det,m2_det`；该接口明确要求kg。
本轮使用太阳质量乘 `lal.MSUN_SI` 的正确单位。pilot中错误/正确单位导致的iota差最大为{audit.unit_error_changes_iota_rad.max():.6g} rad。
这项旧GWTC接口问题已标记待复核，未擅自重算GWTC。不能据此宣称ET与旧GWTC天空数值实现已经逐项完全一致。

## 科学限制

这是与现有GWTC条件BAYESTAR分支同类的近似：固定注入内禀参数，重建独立高斯matched-filter测量，不是重新从原始噪声应变做全参数PE。
ET使用原生成器的独立通道高斯噪声假设；未模拟真实ET共享臂相关噪声。没有增加CE、远距离站点或长时地球自转信息。
pilot覆盖率只作计算检查，不能以20个样本证明统计校准。最终温度只在validation校准。
256/512/1024 pilot差异见 `tables/PILOT_RESOLUTION_SUMMARY.csv`；不能把pilot收敛当成全部目录收敛。
先前O3/O4a开发曾使用真实候选反馈；本轮ET是独立适配，旧测试集已被历史实验查看，不能把复算称为全新盲测。

## 依据

- [Singer & Price 2016](https://arxiv.org/abs/1508.03634)：条件快速天空定位，而非完整BBH PE。
- [Cutler & Flanagan 1994](https://arxiv.org/abs/gr-qc/9402014)：启发研究低频内禀参数信息，不证明16秒或CNN最优。
- Bilby本地接口文档与源码：自旋转换质量单位为SI；探测器张量/PSD与原生成代码对应。

## 尚未完成

完整16秒缓存批处理、ET OMC/多尺度/条件头训练、全事件天空图与验证集校准、新波形证据/融合权重、最终Recall和pair指标均须分别完成后才能出最终结果。
本报告不将输入准备完成冒充实验完成。
"""
    (root / "reports/STARTUP_AND_METHOD_CN.md").write_text(text, encoding="utf-8")
    et.json_write(root / "contracts/BASELINE_INTERFACE_REVIEW.json", {
        "issue":"archived O3/O4a BAYESTAR spin conversion passes solar-mass numbers to SI interface",
        "ET_fix":"multiply masses by lal.MSUN_SI; original source reference frequency=10Hz",
        "old_GWTC_results_modified":False,
        "cross_domain_exact_numerical_parity":"NOT_YET_CERTIFIED", "evidence":"Bilby runtime docstring and archived source"})
    et.verify_protected(root)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--stage", choices=("tests", "convergence", "report", "all"), required=True)
    a = p.parse_args()
    if a.stage in ("tests", "all"):
        tests(a.root)
    if a.stage in ("convergence", "all"):
        convergence(a.root)
    if a.stage in ("report", "all"):
        report(a.root)
