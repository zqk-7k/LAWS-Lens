from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULT = REPO_ROOT / "results" / "mainline_speed_benchmark_20260714"


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, Path):
        return str(value)
    return str(value)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=json_default) + "\n",
        encoding="utf-8",
    )


def summarize(
    frame: pd.DataFrame,
    group: list[str],
    values: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key, block in frame.groupby(group, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        row = dict(zip(group, key))
        row["n_repeats"] = len(block)
        for column in values:
            data = block[column].dropna().to_numpy(dtype=float)
            if not len(data):
                continue
            row[f"{column}_median"] = float(np.median(data))
            row[f"{column}_q25"] = float(np.quantile(data, 0.25))
            row[f"{column}_q75"] = float(np.quantile(data, 0.75))
        rows.append(row)
    return pd.DataFrame(rows)


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.labelweight": "bold",
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "legend.fontsize": 7.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "axes.linewidth": 0.8,
            "savefig.dpi": 300,
        }
    )


def make_figure(root: Path, summaries: dict[str, pd.DataFrame]) -> None:
    configure_style()
    figure, axes = plt.subplots(2, 2, figsize=(7.2, 5.8), constrained_layout=True)
    ax = axes[0, 0]
    pipeline = summaries["pipeline"]
    stage_groups = {
        "Preprocess": ["waveform_preprocessing"],
        "Encode": ["waveform_gpu_inference", "embedding_l2_normalization"],
        "ANN": ["hnsw_index_build", "hnsw_full_catalog_query"],
        "Physics": [
            "candidate_waveform_cosine",
            "candidate_waveform_score_reused_from_hnsw",
            "candidate_time_delay_lr",
            "candidate_gaussian_posterior_overlap",
        ],
        "Fuse": ["candidate_row_standardization", "weighted_fusion_and_top10_sort"],
    }
    colors = ["#4C78A8", "#59A14F", "#F28E2B", "#E15759", "#B07AA1"]
    bottom = 0.0
    for (label, stages), color in zip(stage_groups.items(), colors):
        value = float(pipeline[pipeline.stage.isin(stages)]["seconds_median"].sum())
        ax.bar([0], [value], bottom=bottom, width=0.34, color=color, label=label)
        if value > 0.3:
            ax.text(0, bottom + value / 2, f"{value:.1f}s", ha="center", va="center", fontsize=7)
        bottom += value
    ax.set_xticks([0], ["ET-3 (N=9,000)"])
    ax.set_xlim(-0.48, 0.72)
    ax.set_ylabel("Elapsed time (s)")
    ax.set_title("a  Native sparse pipeline")
    ax.legend(frameon=False, ncol=1, loc="upper right")

    ax = axes[0, 1]
    retrieval = summaries["retrieval"]
    exact = retrieval[retrieval.method == "exact_cosine"].sort_values("n_events")
    hnsw = retrieval[
        (retrieval.method == "HNSW")
        & (retrieval.k == 200)
        & (retrieval.ef_search == 512)
    ].sort_values("n_events")
    for data, label, color, marker in (
        (exact, "Exact cosine", "#E15759", "s"),
        (hnsw, "HNSW (K=200)", "#4C78A8", "o"),
    ):
        if data.empty:
            continue
        x = data.n_events.to_numpy(dtype=float)
        y = data.total_s_median.to_numpy(dtype=float)
        low = y - data.total_s_q25.to_numpy(dtype=float)
        high = data.total_s_q75.to_numpy(dtype=float) - y
        ax.errorbar(x, y, yerr=np.vstack([low, high]), color=color, marker=marker, lw=1.3, ms=4, capsize=2, label=label)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Catalog size, N")
    ax.set_ylabel("Build + full query time (s)")
    ax.set_title("b  Retrieval scaling")
    ax.legend(frameon=False, loc="upper left")

    ax = axes[1, 0]
    accuracy = summaries["accuracy"]
    palette = {
        10: "#9C755F",
        50: "#F28E2B",
        100: "#59A14F",
        200: "#4C78A8",
        500: "#B07AA1",
        1000: "#76B7B2",
    }
    for k, block in accuracy.groupby("k"):
        block = block.sort_values("ef_search")
        x = (block.hnsw_query_s_median + block.physical_rerank_s_median).to_numpy()
        y = block.final_r_at_10_median.to_numpy()
        ax.plot(x, y, marker="o", ms=4, lw=1.1, color=palette.get(int(k), "0.4"), label=f"K={int(k)}")
        for _, row in block.iterrows():
            if int(k) == 200 and int(row.ef_search) in (32, 512):
                ax.annotate(f"ef={int(row.ef_search)}", (row.hnsw_query_s_median + row.physical_rerank_s_median, row.final_r_at_10_median), xytext=(2, 3), textcoords="offset points", fontsize=6)
    exact_r10 = float(accuracy.exact_final_r_at_10_median.iloc[0])
    ax.axhline(exact_r10, color="0.35", ls="--", lw=0.9, label="Exact final")
    ax.set_xlabel("Query + rerank time (s)")
    ax.set_ylabel("Final companion R@10")
    ax.set_title("c  Speed–accuracy trade-off")
    ax.legend(frameon=False, ncol=2, loc="lower right")

    ax = axes[1, 1]
    online = summaries["online"].copy()
    labels = {
        "waveform_preprocess": "Preprocess",
        "waveform_inference": "GPU encode",
        "hnsw_query_k200": "ANN query",
        "physical_score_fusion_sort_k200": "Physics + fusion",
    }
    online["label"] = online.stage.map(labels).fillna(online.stage)
    stage_order = [
        "waveform_preprocess",
        "waveform_inference",
        "hnsw_query_k200",
        "physical_score_fusion_sort_k200",
    ]
    online["stage_order"] = pd.Categorical(online.stage, categories=stage_order, ordered=True)
    online = online.sort_values("stage_order")
    x = np.arange(len(online))
    y = online.milliseconds_median.to_numpy()
    low = y - online.milliseconds_q25.to_numpy()
    high = online.milliseconds_q75.to_numpy() - y
    ax.bar(x, y, color=["#4C78A8", "#59A14F", "#F28E2B", "#B07AA1"][: len(x)])
    ax.errorbar(x, y, yerr=np.vstack([low, high]), fmt="none", ecolor="black", capsize=2, lw=0.8)
    ax.set_xticks(x, online.label, rotation=20, ha="right")
    ax.set_yscale("log")
    ax.set_ylabel("Latency per new event (ms)")
    ax.set_title("d  Incremental processing")

    for ax in axes.ravel():
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="0.9", lw=0.5, zorder=0)
    figures = root / "figures"
    figures.mkdir(exist_ok=True)
    figure.savefig(figures / "fig_mainline_speed_benchmark.pdf", bbox_inches="tight")
    figure.savefig(figures / "fig_mainline_speed_benchmark.png", bbox_inches="tight")
    plt.close(figure)


def table_markdown(frame: pd.DataFrame, columns: list[str], limit: int | None = None) -> str:
    data = frame.loc[:, columns]
    if limit is not None:
        data = data.head(limit)
    return data.to_markdown(index=False, floatfmt=".4g")


def build_reports(root: Path, summaries: dict[str, pd.DataFrame], payload: dict[str, Any]) -> None:
    environment = json.loads((root / "benchmark_environment.json").read_text(encoding="utf-8"))
    pipeline = payload["native_pipeline"]
    selected = payload["selected_operating_point"]
    waveform = summaries["waveform"]
    gpu = waveform[(waveform.stage == "model_inference") & (waveform.device == "CUDA")]
    gpu_best = gpu.loc[gpu.events_per_s.idxmax()]
    cpu = waveform[(waveform.stage == "model_inference") & (waveform.device == "CPU")]
    cpu_best = cpu.loc[cpu.events_per_s.idxmax()]
    online_total = float(summaries["online"].milliseconds_median.sum())
    healpix = summaries["healpix"]
    healpix_dot = healpix[healpix.kernel == "healpix_sparse_candidate_dot"].iloc[0]
    sparse_physics = summaries["physics"]
    dense_physics = summaries["dense_physics"]
    sparse_seconds = float(sparse_physics.seconds_median.sum())
    dense_seconds = float(dense_physics.seconds_median.sum())
    dense_edges = int(dense_physics.n_edges.iloc[0])
    sparse_edges = int(sparse_physics.n_edges.iloc[0])
    scaling = summaries["retrieval"]
    million = scaling[
        (scaling.method == "HNSW")
        & (scaling.n_events == 1_000_000)
        & (scaling.k == 200)
        & (scaling.ef_search == 512)
    ]
    million_text = "not run"
    if not million.empty:
        million_text = f"{float(million.total_s_median.iloc[0]):.2f} s (build + full query)"
    native_exact = scaling[(scaling.method == "exact_cosine") & (scaling.n_events == 9000)].iloc[0]
    native_hnsw = scaling[
        (scaling.method == "HNSW")
        & (scaling.n_events == 9000)
        & (scaling.k == 200)
        & (scaling.ef_search == 512)
    ].iloc[0]

    cn = f"""# 主流程速度基准报告

## 结论

本实验补齐了论文中缺失的端到端速度证据。原生 ET-3 测试目录包含 {pipeline['n_events']:,} 个事件；从三通道 strain 预处理、GPU 编码、HNSW 建索引和查询，到 time-delay、统一 Gaussian posterior-overlap sky、标准化、融合与 top-10 排序，分阶段中位时间之和为 **{pipeline['median_total_seconds_sum_of_stage_medians']:.2f} s**。该 operating point 使用 K={pipeline['k']}、efSearch={pipeline['ef_search']}，对应 sparse pipeline R@1={pipeline['sparse_pipeline_r_at_1']:.4f}、R@10={pipeline['sparse_pipeline_r_at_10']:.4f}。

在 9,000-event 精确参考中，最终 R@1={selected['exact_final_r_at_1']:.4f}、R@10={selected['exact_final_r_at_10']:.4f}；HNSW operating point 的 R@1={selected['final_r_at_1']:.4f}、R@10={selected['final_r_at_10']:.4f}，候选 companion recall={selected['candidate_companion_recall']:.4f}。因此速度结论必须和这一准确性差异同时报告，不能只引用 ANN top-10 fidelity。

## 测试边界

- N=9,000 使用当前 held-out ET-3 三通道 strain、50-epoch encoder、validation-selected 三通道权重，属于原生完整流程实测。
- N>9,000 没有生成新 strain；使用已测 ET-3 embedding 的 tiled/perturbed 副本，只验证 128 维向量索引扩展性。百万事件结果是 embedding-level 实测，不是百万 strain 端到端实验。
- 精确余弦矩阵只测到 N=10,000；它在 N=100,000/1,000,000 的内存分别约为 40 GB/4 TB，故不执行。
- ET 模拟目录的统一 Gaussian posterior-overlap 与真实 O4a PE HEALPix posterior overlap 分开计时。HEALPix 不被 Gaussian proxy 的速度替代。
- 计时从已有 memory-mapped strain 的波形预处理开始；不包含网络下载、数据清单构建、checkpoint 初次加载和冷存储传输。完整流程预热后属于 warm-cache compute benchmark。
- 所有主要微基准预热 5 次、重复 20 次并报告 median/IQR；完整 9,000-event pipeline 因成本使用 {pipeline['warmups']} 次预热、{pipeline['repeats']} 次正式重复；大规模 HNSW 采用结果文件中记录的缩减重复数。

## 硬件与软件

- GPU: {environment['gpu']}
- CPU logical cores: {environment['logical_cpu_count']}；固定线程数: {environment['fixed_cpu_threads']}
- PyTorch {environment['torch']}；CUDA {environment['cuda']}；NumPy {environment['numpy']}；hnswlib {environment['hnswlib']}

## 波形编码

GPU 最佳测量吞吐为 {gpu_best.events_per_s:.1f} events/s（batch={int(gpu_best.batch_size)}，{gpu_best.ms_per_event:.3f} ms/event），峰值显存 {gpu_best.peak_gpu_memory_mb:.1f} MB。CPU 最佳测量吞吐为 {cpu_best.events_per_s:.1f} events/s（batch={int(cpu_best.batch_size)}）。这些吞吐来自固定样本块，不包含磁盘下载或模型训练。

## 目录检索

在 N=9,000、K=200 时，精确 cosine matrix + top-K 的中位时间为 {native_exact.total_s_median:.3f} s；HNSW 建索引 + 全目录查询 + self-removal 为 {native_hnsw.total_s_median:.3f} s，相差 {native_exact.total_s_median / native_hnsw.total_s_median:.1f} 倍。百万事件 K=200、efSearch=512 的扩展性实测为：**{million_text}**。大规模结果只表示现有 embedding 的 HNSW 建索引和全目录查询。

## 物理通道

K={pipeline['k']} 时，物理 refinement 从 dense reference 的 {dense_edges:,} 条 directed non-self edges 降为 {sparse_edges:,} 条候选边，即减少 {dense_edges / sparse_edges:.1f} 倍。各 kernel 中位时间之和由 dense reference 的 {dense_seconds:.2f} s 降为 sparse candidate 实现的 {sparse_seconds:.2f} s。候选边上的 time-delay、Gaussian posterior-overlap、标准化和融合均独立计时。真实 O4a nside=512 HEALPix sparse candidate dot 的中位成本为 {healpix_dot.microseconds_per_edge_median:.1f} us/edge；它显著重于 ET Gaussian sky kernel，论文中应单列，不可用 ET sky 速度替代真实 posterior-map refinement。

## 在线延迟

已存在索引时，一个新事件的预处理、GPU 编码、K=200 查询和物理融合中位延迟之和约为 **{online_total:.2f} ms**。这不包含等待数据发布、PE skymap 生成或索引全量重建。

## 论文表述

可写：在原生 9,000-event ET-3 目录上，稀疏候选生成把需要物理 refinement 的边数从 O(N^2) 限制为 O(NK)，完整实测耗时由波形预处理主导；HNSW 提供可控的速度–召回折衷，并在百万 embedding 规模保持可运行。

不可写：完成了百万事件 full strain-level 搜索；或 HNSW 在不损失最终召回率的情况下严格等价于精确三通道排序。百万规模没有运行 waveform encoder、time/sky 全流程，且候选池大小会限制最终召回。
"""
    en = f"""# Mainline runtime benchmark report

## Main result

The native ET-3 benchmark contains {pipeline['n_events']:,} events. The sum of measured stage medians from three-channel strain preprocessing through GPU encoding, HNSW retrieval, physical scoring, fusion, and top-10 sorting is **{pipeline['median_total_seconds_sum_of_stage_medians']:.2f} s**. At K={pipeline['k']} and efSearch={pipeline['ef_search']}, the sparse workflow reaches R@1={pipeline['sparse_pipeline_r_at_1']:.4f} and R@10={pipeline['sparse_pipeline_r_at_10']:.4f}.

The exact 9,000-event reference reaches final R@1={selected['exact_final_r_at_1']:.4f} and R@10={selected['exact_final_r_at_10']:.4f}; the selected HNSW operating point reaches R@1={selected['final_r_at_1']:.4f}, R@10={selected['final_r_at_10']:.4f}, and candidate companion recall={selected['candidate_companion_recall']:.4f}. Runtime claims must therefore be reported together with the end-to-end accuracy trade-off.

## Scope and limitations

- N=9,000 is a native held-out strain-level ET-3 measurement using the current encoder and validation-selected fusion weights.
- N>9,000 uses tiled, perturbed ET-3 embeddings and measures 128-dimensional indexing only; the one-million-event result is not a one-million-strain end-to-end experiment.
- Exact cosine is measured only through N=10,000 because a dense matrix would require approximately 40 GB at N=100,000 and 4 TB at N=1,000,000.
- Simulated ET Gaussian posterior overlap and real O4a nside=512 HEALPix posterior overlap are timed separately.
- Timings start from preprocessing existing memory-mapped strain. Network download, manifest construction, checkpoint startup, and cold-storage transfer are excluded; full-pipeline repetitions are a warm-cache compute benchmark after warm-up.
- Microbenchmarks use five warm-ups and 20 measured repetitions. The full native pipeline uses {pipeline['warmups']} warm-up and {pipeline['repeats']} measured repetitions; large-N policies are recorded in the environment JSON.

## Hardware

GPU: {environment['gpu']}. PyTorch {environment['torch']}, CUDA {environment['cuda']}, fixed CPU threads {environment['fixed_cpu_threads']}.

## Throughput and online use

The best measured GPU throughput is {gpu_best.events_per_s:.1f} events/s at batch {int(gpu_best.batch_size)} ({gpu_best.ms_per_event:.3f} ms/event; peak allocated GPU memory {gpu_best.peak_gpu_memory_mb:.1f} MB). With an existing index, the summed median latency for one new event is approximately **{online_total:.2f} ms**, excluding data release and PE skymap production.

At K={pipeline['k']}, physical refinement is reduced from {dense_edges:,} directed non-self edges in the dense reference to {sparse_edges:,} candidate edges ({dense_edges / sparse_edges:.1f}-fold). The sum of kernel medians changes from {dense_seconds:.2f} s for the dense reference to {sparse_seconds:.2f} s for sparse candidate scoring. Real O4a nside=512 HEALPix overlap remains a distinct, more expensive refinement kernel and is not represented by the ET Gaussian-sky timing.

## Paper-safe interpretation

The native benchmark supports the claim that sparse candidate generation reduces the physical-refinement workload from O(N^2) to O(NK), with preprocessing dominating elapsed time and an explicitly measured speed–recall trade-off. It does not support a claim of a full one-million-event strain-level search or lossless equivalence between HNSW and exact three-channel ranking.
"""
    (root / "mainline_speed_benchmark_report_cn.md").write_text(cn, encoding="utf-8")
    (root / "mainline_speed_benchmark_report_en.md").write_text(en, encoding="utf-8")
    caption = f"""**Figure X | Runtime and accuracy of sparse catalog refinement.**
**a,** Measured stage medians for the native {pipeline['n_events']:,}-event ET-3 workflow at K={pipeline['k']} and efSearch={pipeline['ef_search']}; stacked values begin with existing memory-mapped strain and exclude model startup and data download. **b,** Exact cosine and HNSW build-plus-query scaling. N<=9,000 uses native ET-3 embeddings, while N>9,000 uses tiled and perturbed embeddings solely for indexing-scale measurements; error bars denote the interquartile range. **c,** HNSW query-plus-rerank time against final three-channel companion R@10 for K in {{10, 50, 100, 200}} and efSearch in {{32, 64, 128, 256, 512}}; the dashed line is exact full-matrix fusion. **d,** Incremental latency for one event with an existing index (median and interquartile range). ET sky timing uses the unified Gaussian posterior-overlap approximation; real nside=512 HEALPix posterior overlap is benchmarked separately. The experiment measures candidate generation and refinement, not detection latency or a one-million-event strain-level analysis.
"""
    (root / "figures" / "fig_mainline_speed_benchmark_caption.md").write_text(
        caption, encoding="utf-8"
    )


def write_paper_metrics(root: Path, summaries: dict[str, pd.DataFrame], payload: dict[str, Any]) -> None:
    retrieval = summaries["retrieval"]
    waveform = summaries["waveform"]
    physics = summaries["physics"]
    dense = summaries["dense_physics"]
    healpix = summaries["healpix"]
    online = summaries["online"]
    pipeline = payload["native_pipeline"]
    selected = payload["selected_operating_point"]
    rows: list[dict[str, Any]] = []

    def add(section: str, metric: str, value: float, unit: str, note: str = "") -> None:
        rows.append({"section": section, "metric": metric, "value": value, "unit": unit, "note": note})

    add("native_et3", "n_events", pipeline["n_events"], "events")
    add("native_et3", "end_to_end_stage_median_sum", pipeline["median_total_seconds_sum_of_stage_medians"], "s", "K=200, efSearch=512, warm cache")
    add("native_et3", "sparse_r_at_1", pipeline["sparse_pipeline_r_at_1"], "fraction")
    add("native_et3", "sparse_r_at_10", pipeline["sparse_pipeline_r_at_10"], "fraction")
    add("native_et3", "exact_final_r_at_1", selected["exact_final_r_at_1"], "fraction")
    add("native_et3", "exact_final_r_at_10", selected["exact_final_r_at_10"], "fraction")
    for _, row in waveform[waveform.stage == "model_inference"].iterrows():
        add("waveform", f"{row.device.lower()}_batch_{int(row.batch_size)}_throughput", row.events_per_s, "events/s")
        add("waveform", f"{row.device.lower()}_batch_{int(row.batch_size)}_peak_gpu_memory", row.peak_gpu_memory_mb, "MiB")
    for n in (9000, 100000, 1000000):
        block = retrieval[(retrieval.method == "HNSW") & (retrieval.n_events == n) & (retrieval.k == 200) & (retrieval.ef_search == 512)]
        if block.empty:
            continue
        row = block.iloc[0]
        add("retrieval", f"hnsw_n{n}_build", row.build_s_median, "s")
        add("retrieval", f"hnsw_n{n}_full_query", row.query_s_median, "s")
        add("retrieval", f"hnsw_n{n}_incremental_query", row.incremental_query_ms_median, "ms")
        add("retrieval", f"hnsw_n{n}_serialized_index", row.index_size_mb_median, "MiB")
    add("physics", "dense_reference_kernel_median_sum", dense.seconds_median.sum(), "s")
    add("physics", "sparse_candidate_kernel_median_sum", physics.seconds_median.sum(), "s")
    hrow = healpix[healpix.kernel == "healpix_sparse_candidate_dot"].iloc[0]
    add("sky", "real_healpix_sparse_overlap", hrow.microseconds_per_edge_median, "us/edge", "O4a PE, nside=512")
    add("online", "new_event_stage_median_sum", online.milliseconds_median.sum(), "ms", "existing index; excludes PE production")
    pd.DataFrame(rows).to_csv(root / "mainline_speed_paper_metrics.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT)
    args = parser.parse_args()
    root = args.result_root
    required = [
        "waveform_encoding_summary.csv",
        "retrieval_scaling_raw.csv",
        "et3_ann_accuracy_pareto.csv",
        "native_et3_end_to_end_stage_times.csv",
        "native_et3_end_to_end_summary.json",
        "physical_scoring_raw.csv",
        "physical_scoring_dense_reference_raw.csv",
        "healpix_benchmark_raw.csv",
        "online_latency_raw.csv",
        "benchmark_environment.json",
    ]
    missing = [name for name in required if not (root / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing benchmark outputs: {missing}")

    waveform = pd.read_csv(root / "waveform_encoding_summary.csv")
    retrieval_raw = pd.read_csv(root / "retrieval_scaling_raw.csv")
    accuracy_raw = pd.read_csv(root / "et3_ann_accuracy_pareto.csv")
    pipeline_raw = pd.read_csv(root / "native_et3_end_to_end_stage_times.csv")
    physics_raw = pd.read_csv(root / "physical_scoring_raw.csv")
    dense_physics_raw = pd.read_csv(root / "physical_scoring_dense_reference_raw.csv")
    healpix_raw = pd.read_csv(root / "healpix_benchmark_raw.csv")
    online_raw = pd.read_csv(root / "online_latency_raw.csv")

    retrieval = summarize(
        retrieval_raw,
        ["method", "embedding_source", "n_events", "k", "ef_search"],
        [
            "build_s",
            "query_s",
            "postprocess_s",
            "total_s",
            "single_query_ms",
            "incremental_query_ms",
            "index_size_mb",
        ],
    )
    accuracy = summarize(
        accuracy_raw,
        ["n_events", "k", "ef_search"],
        [
            "hnsw_query_s",
            "physical_rerank_s",
            "ann_top10_fidelity",
            "candidate_companion_recall",
            "final_r_at_1",
            "final_r_at_10",
            "exact_final_r_at_1",
            "exact_final_r_at_10",
        ],
    )
    pipeline = summarize(pipeline_raw, ["stage"], ["seconds"])
    physics = summarize(
        physics_raw, ["kernel", "n_edges", "catalog"], ["seconds", "microseconds_per_edge"]
    )
    dense_physics = summarize(
        dense_physics_raw,
        ["kernel", "n_edges", "catalog"],
        ["seconds", "microseconds_per_edge"],
    )
    healpix = summarize(
        healpix_raw, ["kernel", "n_edges", "catalog"], ["seconds", "microseconds_per_edge"]
    )
    online = summarize(online_raw, ["stage"], ["seconds", "milliseconds"])
    retrieval.to_csv(root / "retrieval_scaling_summary.csv", index=False)
    accuracy.to_csv(root / "et3_ann_accuracy_summary.csv", index=False)
    pipeline.to_csv(root / "native_et3_stage_summary.csv", index=False)
    physics.to_csv(root / "physical_scoring_summary.csv", index=False)
    dense_physics.to_csv(root / "physical_scoring_dense_reference_summary.csv", index=False)
    healpix.to_csv(root / "healpix_benchmark_summary.csv", index=False)
    online.to_csv(root / "online_latency_summary.csv", index=False)

    pipeline_json = json.loads((root / "native_et3_end_to_end_summary.json").read_text(encoding="utf-8"))
    chosen = accuracy[(accuracy.k == 200) & (accuracy.ef_search == 512)].iloc[0]
    selected = {
        key: float(chosen[f"{key}_median"])
        for key in (
            "ann_top10_fidelity",
            "candidate_companion_recall",
            "final_r_at_1",
            "final_r_at_10",
            "exact_final_r_at_1",
            "exact_final_r_at_10",
            "hnsw_query_s",
            "physical_rerank_s",
        )
    }
    payload = {
        "native_pipeline": pipeline_json,
        "selected_operating_point": {"k": 200, "ef_search": 512, **selected},
        "scope": {
            "native_strain_level_max_n": 9000,
            "exact_cosine_max_n": 10000,
            "large_n_source": "tiled_perturbed_et3_embeddings_scaling_only",
            "large_n_is_not_full_strain_pipeline": True,
        },
    }
    write_json(root / "mainline_speed_benchmark_summary.json", payload)
    summaries = {
        "waveform": waveform,
        "retrieval": retrieval,
        "accuracy": accuracy,
        "pipeline": pipeline,
        "physics": physics,
        "dense_physics": dense_physics,
        "healpix": healpix,
        "online": online,
    }
    make_figure(root, summaries)
    build_reports(root, summaries, payload)
    write_paper_metrics(root, summaries, payload)
    print(json.dumps({"status": "complete", "result_root": str(root)}, indent=2))


if __name__ == "__main__":
    main()
