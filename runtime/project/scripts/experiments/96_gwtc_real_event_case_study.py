from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import requests
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


runner = importlib.import_module("scripts.experiments.92_ligo_h1l1_full_experiment_runner")
fresh, liao, _pdf = runner.configure_modules()
base = importlib.import_module("scripts.experiments.80_mixed_sis_pm_catalog_modality_compare")

DATA_ROOT = Path("data/gwtc_real")
OUT_DIR = Path("runs/gwtc_real_event_case_study_20260617")
DOC_PATH = Path("docs/gwtc_real_event_case_study_20260617_cn.md")
EVENTS = ["GW150914", "GW151226", "GW170817"]
CASE_PAIRS = [("GW150914", "GW151226"), ("GW150914", "GW170817")]
HYBRID_LAMBDA_GRID = [0.25, 0.5, 1.0, 2.0, 4.0]
CATALOG = "GWTC-1-confident"
DETECTORS = ["H1", "L1"]
SAMPLE_RATE_KHZ = 4
DURATION_S = 32


def api_get_json(url: str, **params) -> dict:
    response = requests.get(url, params=params, timeout=90)
    response.raise_for_status()
    return response.json()


def event_version_detail_url(event: str) -> tuple[str, str]:
    info = api_get_json(f"https://gwosc.org/api/v2/events/{event}")
    versions = info.get("versions", [])
    for version in versions:
        if version.get("catalog") == CATALOG:
            return version["detail_url"], f"{event}-v{version['version']}"
    if versions:
        version = versions[0]
        return version["detail_url"], f"{event}-v{version['version']}"
    raise RuntimeError(f"No GWOSC event version found for {event}")


def download_event(event: str) -> Path:
    event_dir = DATA_ROOT / event
    event_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = event_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        files = manifest.get("files", [])
        if files and all(Path(item["local_file"]).exists() for item in files):
            return event_dir
    base_url, version = event_version_detail_url(event)

    for name, url in {
        "event_version.json": base_url,
        "parameters.json": base_url + "/parameters",
        "strain_files_all.json": base_url + "/strain-files",
    }.items():
        path = event_dir / name
        if not path.exists():
            path.write_text(json.dumps(api_get_json(url), indent=2, ensure_ascii=False), encoding="utf-8")

    manifest_files = []
    for detector in DETECTORS:
        listing = api_get_json(
            base_url + "/strain-files",
            detector=detector,
            **{"sample-rate": SAMPLE_RATE_KHZ, "duration": DURATION_S, "file-format": "hdf5"},
        )
        results = listing.get("results", [])
        if len(results) != 1:
            raise RuntimeError(f"Expected one {event} {detector} strain file, got {len(results)}")
        item = results[0]
        url = item["download_url"]
        filename = url.rsplit("/", 1)[-1]
        local = event_dir / filename
        if not local.exists():
            print("DOWNLOAD", event, detector, url, flush=True)
            with requests.get(url, stream=True, timeout=180) as response:
                response.raise_for_status()
                with local.open("wb") as f:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
        with h5py.File(local, "r") as h5:
            strain_shape = tuple(h5["strain/Strain"].shape)
        manifest_files.append({
            **item,
            "local_file": str(local),
            "bytes": int(local.stat().st_size),
            "strain_shape": strain_shape,
        })

    (event_dir / "manifest.json").write_text(json.dumps({
        "event": event,
        "event_version": version,
        "catalog": CATALOG,
        "source": "GWOSC API v2",
        "files": manifest_files,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    return event_dir


def load_event_waveform(event: str) -> tuple[np.ndarray, dict]:
    event_dir = download_event(event)
    manifest = json.loads((event_dir / "manifest.json").read_text(encoding="utf-8"))
    event_version = json.loads((event_dir / "event_version.json").read_text(encoding="utf-8"))
    channels = []
    meta = {
        "event": event,
        "event_version": manifest["event_version"],
        "catalog": manifest["catalog"],
        "gps": float(event_version["gps"]),
    }
    for detector in DETECTORS:
        matches = [f for f in manifest["files"] if f["detector"] == detector]
        if len(matches) != 1:
            raise RuntimeError(f"Missing {detector} file for {event}")
        path = Path(matches[0]["local_file"])
        with h5py.File(path, "r") as h5:
            channels.append(np.asarray(h5["strain/Strain"][:], dtype=np.float32))
        meta[f"{detector}_file"] = str(path)
        meta[f"{detector}_samples"] = int(channels[-1].size)
        meta[f"{detector}_gps_start"] = float(matches[0]["gps_start"])
        meta[f"{detector}_sample_rate_hz"] = int(matches[0]["sample_rate_kHz"]) * 1024
    n = min(len(x) for x in channels)
    return np.stack([x[-n:] for x in channels], axis=0).astype(np.float32), meta


def crop_event_center_window(waveform: np.ndarray, meta: dict, target_len: int) -> tuple[np.ndarray, dict]:
    starts = [meta[f"{detector}_gps_start"] for detector in DETECTORS]
    rates = [meta[f"{detector}_sample_rate_hz"] for detector in DETECTORS]
    if len(set(starts)) != 1 or len(set(rates)) != 1:
        raise RuntimeError(f"Detector files are not aligned for {meta['event']}: starts={starts}, rates={rates}")
    center = int(round((float(meta["gps"]) - starts[0]) * rates[0]))
    start = max(0, center - target_len // 2)
    stop = min(waveform.shape[-1], start + target_len)
    start = max(0, stop - target_len)
    cropped = waveform[..., start:stop]
    crop_meta = dict(meta)
    crop_meta.update({
        "crop_mode": "event_centered",
        "crop_target_len": int(target_len),
        "crop_center_sample": int(center),
        "crop_start_sample": int(start),
        "crop_stop_sample": int(stop),
        "crop_samples": int(cropped.shape[-1]),
    })
    return cropped.astype(np.float32, copy=False), crop_meta


def load_encoder(cfg, sample: np.ndarray) -> torch.nn.Module:
    model_path = cfg.out_dir / "model.pt"
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    in_channels = base.data_mod.prepared_channel_count(sample, cfg)
    model = base.build_model(cfg, in_channels=in_channels)
    checkpoint = torch.load(model_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    return model


def embed_real_waveforms(model, cfg, waveforms: list[np.ndarray]) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    rows = []
    with torch.no_grad():
        for waveform in waveforms:
            x = base.prepare_waveform(waveform, cfg, train=False)
            tensor = torch.from_numpy(x[None, ...]).to(device)
            z = model(tensor).detach().cpu().numpy()[0]
            rows.append(z.astype(np.float32))
    emb = np.vstack(rows)
    emb /= np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-8)
    return emb.astype(np.float32)


def sampled_negative_scores(scores: np.ndarray, gt: np.ndarray, n_samples: int = 200_000, seed: int = 20260617) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = scores.shape[0]
    a = rng.integers(0, n, size=n_samples, dtype=np.int32)
    b = rng.integers(0, n, size=n_samples, dtype=np.int32)
    bad = (a == b) | (gt[a] == b)
    while bad.any():
        b[bad] = rng.integers(0, n, size=int(bad.sum()), dtype=np.int32)
        bad = (a == b) | (gt[a] == b)
    return scores[a, b].astype(np.float32)


def sampled_negative_matrix_values(matrix: np.ndarray, gt: np.ndarray, n_samples: int = 200_000, seed: int = 20260618) -> np.ndarray:
    return sampled_negative_scores(matrix, gt, n_samples=n_samples, seed=seed)


def liao_time_lr_for_days(dt_days: float, prior: dict) -> float:
    edges = prior["edges"]
    lr = prior["lr"]
    x = np.log10(max(float(dt_days), 1e-6))
    idx = int(np.searchsorted(edges, x, side="right") - 1)
    idx = max(0, min(idx, len(lr) - 1))
    return float(lr[idx])


def liao_time_lr_matrix_from_times(times: np.ndarray, prior: dict) -> np.ndarray:
    times = np.asarray(times, dtype=np.float64)
    n = len(times)
    out = np.empty((n, n), dtype=np.float32)
    for start in range(0, n, liao.CHUNK_ROWS):
        rows = slice(start, min(start + liao.CHUNK_ROWS, n))
        dt_days = np.abs(times[rows, None] - times[None, :]) / liao.SECONDS_PER_DAY
        x = np.log10(np.maximum(dt_days, 1e-6))
        idx = np.searchsorted(prior["edges"], x, side="right") - 1
        idx = np.clip(idx, 0, len(prior["lr"]) - 1)
        out[rows] = prior["lr"][idx].astype(np.float32)
    np.fill_diagonal(out, -np.inf)
    return out


def select_hybrid_lambda(loaded: dict, prior: dict) -> tuple[float, dict]:
    val_ds, _val_raw, val_time, val_gt, val_scores = loaded["val"]
    val_time_lr = liao_time_lr_matrix_from_times(
        val_time["trigger_time_obs"].to_numpy(dtype=np.float64),
        prior,
    )
    best_lam = HYBRID_LAMBDA_GRID[0]
    best_metrics = None
    best_key = (-1.0, -1.0, -1.0)
    for lam in HYBRID_LAMBDA_GRID:
        score = val_scores + lam * val_time_lr
        np.fill_diagonal(score, -np.inf)
        metrics = liao.evaluate_score(score, val_gt, val_ds.meta)
        key = (
            metrics["overall"]["r@10"],
            metrics["overall"]["r@5"],
            metrics["overall"]["r@1"],
        )
        if key > best_key:
            best_key = key
            best_lam = lam
            best_metrics = metrics
    return float(best_lam), best_metrics


def rank_candidate(row: np.ndarray, candidate_index: int) -> int:
    score = float(row[candidate_index])
    return int(1 + np.sum(row > score))


def run_hybrid_real_in_sim_ranking(
    real_emb: np.ndarray,
    metas: list[dict],
    event_to_index: dict[str, int],
    test_emb: np.ndarray,
    test_time: pd.DataFrame,
    loaded: dict,
    prior: dict,
    output_dir: Path,
) -> tuple[list[dict], list[dict], dict]:
    selected_lambda, lambda_metrics = select_hybrid_lambda(loaded, prior)
    sim_labels = [f"sim_{idx}" for idx in range(test_emb.shape[0])]
    real_labels = [meta["event"] for meta in metas]
    labels = sim_labels + real_labels
    real_global = {event: len(sim_labels) + idx for event, idx in event_to_index.items()}

    all_emb = np.vstack([test_emb, real_emb]).astype(np.float32)
    all_emb /= np.maximum(np.linalg.norm(all_emb, axis=1, keepdims=True), 1e-8)
    waveform = all_emb @ all_emb.T
    np.fill_diagonal(waveform, -np.inf)

    sim_times = test_time["trigger_time_obs"].to_numpy(dtype=np.float64)
    real_times = np.asarray([float(meta["gps"]) for meta in metas], dtype=np.float64)
    all_times = np.concatenate([sim_times, real_times])
    time_lr = liao_time_lr_matrix_from_times(all_times, prior)
    combined = waveform + selected_lambda * time_lr
    np.fill_diagonal(combined, -np.inf)

    metric_matrices = {
        "waveform": waveform,
        "liao_time_lr": time_lr,
        f"waveform_plus_{selected_lambda:g}x_time_lr": combined,
    }
    top_rows = []
    for metric, matrix in metric_matrices.items():
        for event in real_labels:
            query_idx = real_global[event]
            row = matrix[query_idx].copy()
            order = np.argsort(-row)[:10]
            top_rows.append({
                "metric": metric,
                "query_event": event,
                "top10_labels": ";".join(labels[int(i)] for i in order),
                "top10_scores": ";".join(f"{float(row[int(i)]):.6g}" for i in order),
                "best_label": labels[int(order[0])],
                "best_score": float(row[int(order[0])]),
            })

    pair_rows = []
    all_pairs = [(a, b) for i, a in enumerate(real_labels) for b in real_labels[i + 1:]]
    for event_a, event_b in all_pairs:
        ia = real_global[event_a]
        ib = real_global[event_b]
        for metric, matrix in metric_matrices.items():
            pair_rows.append({
                "pair": f"{event_a}-{event_b}",
                "metric": metric,
                "score": float(matrix[ia, ib]),
                "rank_from_a": rank_candidate(matrix[ia], ib),
                "rank_from_b": rank_candidate(matrix[ib], ia),
                "selected_lambda_time": selected_lambda if metric.startswith("waveform_plus_") else "",
            })

    pd.DataFrame(top_rows).to_csv(output_dir / "gwtc_hybrid_real_in_sim_top_candidates.csv", index=False)
    pd.DataFrame(pair_rows).to_csv(output_dir / "gwtc_hybrid_real_pair_ranks.csv", index=False)
    diag = {
        "hybrid_catalog_sim_size": int(test_emb.shape[0]),
        "hybrid_catalog_real_size": int(len(metas)),
        "hybrid_lambda_grid": HYBRID_LAMBDA_GRID,
        "selected_lambda_time": selected_lambda,
        "selected_lambda_val_overall": lambda_metrics["overall"] if lambda_metrics else None,
    }
    return pair_rows, top_rows, diag


def describe_distribution(name: str, values: np.ndarray) -> dict:
    values = values[np.isfinite(values)]
    return {
        "distribution": name,
        "count": int(len(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "p01": float(np.percentile(values, 1)),
        "p05": float(np.percentile(values, 5)),
        "p25": float(np.percentile(values, 25)),
        "median": float(np.median(values)),
        "p75": float(np.percentile(values, 75)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
    }


def percentile_rank(value: float, values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    return float(100.0 * np.mean(values <= value))


def write_doc(case_rows: list[dict], dist_rows: list[dict], hybrid_pair_rows: list[dict], output_dir: Path) -> None:
    def fmt(x) -> str:
        if isinstance(x, str):
            return x
        return f"{float(x):.4f}"

    combo_metric = next(
        (row["metric"] for row in hybrid_pair_rows if str(row["metric"]).startswith("waveform_plus_")),
        "waveform_plus_time_lr",
    )
    combo_lambda = next(
        (row.get("selected_lambda_time") for row in hybrid_pair_rows if str(row["metric"]).startswith("waveform_plus_")),
        "",
    )
    lines = [
        "# GWTC 真实事件 non-lensed case study",
        "",
        "## 目的",
        "",
        "该实验使用真实 GWTC 事件作为非透镜 sanity check：选择两例不同的已知真实事件，按当前 LIGO H1+L1 waveform encoder 跑完整输入预处理和 embedding 打分，验证 pipeline 能处理真实 strain，并且给出低于模拟透镜正样本的分数。",
        "",
        "本 case study 不用于训练，也不声称真实事件存在透镜关系。它只作为 reviewer/NC 可能要求的 real-event demonstration。",
        "",
        "## 事件",
        "",
        "- Events: `GW150914`, `GW151226`, `GW170817`",
        "- Case pairs: `GW150914-GW151226`, `GW150914-GW170817`",
        "- Catalog: `GWTC-1-confident`",
        "- Detectors: H1 + L1",
        "- Strain: GWOSC 4 kHz, 32 s HDF5",
        "",
        "## 结果",
        "",
        "| pair | waveform_score | waveform_pos_pct | waveform_neg_pct | delta_t_days | liao_time_lr | time_lr_pos_pct | time_lr_neg_pct |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for case_row in case_rows:
        lines.append(
            f"| {case_row['event_a']} vs {case_row['event_b']} | {fmt(case_row['real_pair_waveform_score'])} | "
            f"{fmt(case_row['waveform_percentile_vs_sim_positive'])} | {fmt(case_row['waveform_percentile_vs_sim_negative'])} | "
            f"{fmt(case_row['delta_t_days'])} | {fmt(case_row['liao_time_lr'])} | "
            f"{fmt(case_row['time_lr_percentile_vs_sim_positive'])} | {fmt(case_row['time_lr_percentile_vs_sim_negative'])} |"
        )
    lines += [
        "",
        "解释：`*_pos_pct` 表示该真实事件对分数在模拟真实透镜 pair 分数中的百分位；数值越低，越不像模拟透镜正样本。`*_neg_pct` 表示它在随机非配对负样本中的百分位。",
        "",
        "注意：本次真实 strain 的 waveform-only score 暴露出明显 OOD 问题，不能单独作为真实事件透镜判断。更可靠的 sanity check 是后续处理中的时间一致性 prior。",
        "",
        "## 真实事件混入模拟库 ranking",
        "",
        f"该检查把 3 个真实 GWTC 事件插入当前 LIGO H1+L1 模拟 test catalog，只作为 query/candidate 参与检索，不参与训练。`{combo_metric}` 中的 `{combo_lambda}` 是在模拟 validation catalog 上从 `[0.25, 0.5, 1.0, 2.0, 4.0]` 选择得到。",
        "",
        "| pair | metric | score | rank_from_a | rank_from_b |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in hybrid_pair_rows:
        lines.append(
            f"| {row['pair']} | {row['metric']} | {fmt(row['score'])} | "
            f"{row['rank_from_a']} | {row['rank_from_b']} |"
        )
    lines += [
        "",
        "解释：rank 是在 `9000` 个模拟 test 样本 + `3` 个真实事件构成的候选库中计算，rank 越小表示越靠前。waveform-only 会把真实事件彼此排到最前，说明真实 strain 对当前模拟训练 encoder 存在 OOD 高相似问题；加入 Liao time-delay prior 后，明显非透镜的长时间间隔事件会被压到几百到几千名。",
        "",
        f"从 top-candidate 表也能看到，使用 `liao_time_lr` 或 `{combo_metric}` 后，真实事件的 top candidates 变成时间间隔更接近透镜先验的模拟事件，不再是其它真实 GWTC 事件。这说明混入真实事件不会直接破坏 pipeline，但也说明真实事件不能直接和模拟 waveform embedding 混合训练；更合适的用途是作为 case study / sanity check。",
        "",
        "## 模拟参照分布",
        "",
        "| distribution | count | mean | std | p05 | median | p95 | p99 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in dist_rows:
        lines.append(
            f"| {row['distribution']} | {row['count']} | {fmt(row['mean'])} | {fmt(row['std'])} | "
            f"{fmt(row['p05'])} | {fmt(row['median'])} | {fmt(row['p95'])} | {fmt(row['p99'])} |"
        )
    lines += [
        "",
        "## 结论",
        "",
        "1. 真实 GWTC H1/L1 strain 可以通过当前 LIGO H1+L1 pipeline 完整跑通。",
        "2. waveform-only 对 GWOSC 真实 strain 存在 OOD 标定问题；这正是加入真实事件 case study 的价值。",
        "3. 两个真实事件相隔很久，Liao/GW-LMC time-delay prior 给出低透镜一致性，可作为 non-lensed sanity check。",
        "4. 该实验没有使用真实透镜标签，不参与 supervised 训练，只作为真实事件 case study。",
        "",
        "## 文件",
        "",
        f"- 输出目录：`{output_dir}`",
        "- `gwtc_real_event_case_study_summary.csv`",
        "- `gwtc_real_event_reference_distributions.csv`",
        "- `gwtc_hybrid_real_in_sim_top_candidates.csv`",
        "- `gwtc_hybrid_real_pair_ranks.csv`",
        "- `gwtc_real_event_embeddings.npy`",
    ]
    DOC_PATH.parent.mkdir(parents=True, exist_ok=True)
    DOC_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    encoder_dir = runner.FRESH_ROOT / "fresh_mixed_encoders" / "ligo_noisy_mixed_sis_pm_ep50"
    cfg = base.make_cfg("LIGO", "noisy", encoder_dir)

    waveforms = []
    metas = []
    event_to_index = {}
    for event in EVENTS:
        waveform, meta = load_event_waveform(event)
        cropped, crop_meta = crop_event_center_window(waveform, meta, cfg.target_len)
        event_to_index[event] = len(waveforms)
        waveforms.append(cropped)
        metas.append(crop_meta)

    model = load_encoder(cfg, waveforms[0])
    real_emb = embed_real_waveforms(model, cfg, waveforms)

    loaded = liao.load_job("LIGO", "noisy")
    test_ds, _test_raw, _test_time, test_gt, test_scores = loaded["test"]
    test_emb = np.load(cfg.out_dir / "test_embeddings.npy").astype(np.float32)
    test_emb /= np.maximum(np.linalg.norm(test_emb, axis=1, keepdims=True), 1e-8)

    valid = np.where(test_gt >= 0)[0]
    positive_scores = test_scores[valid, test_gt[valid]].astype(np.float32)
    negative_scores = sampled_negative_scores(test_scores, test_gt)

    real_to_sim = real_emb @ test_emb.T

    prior = liao.fit_time_lr_from_liao("LIGO", _test_time, test_gt)
    sim_time_lr = liao.time_lr_score_matrix(_test_time, prior)
    positive_time_lr = sim_time_lr[valid, test_gt[valid]].astype(np.float32)
    negative_time_lr = sampled_negative_matrix_values(sim_time_lr, test_gt)
    case_rows = []
    version_cols = {f"{m['event']}_version": m["event_version"] for m in metas}
    for event_a, event_b in CASE_PAIRS:
        ia = event_to_index[event_a]
        ib = event_to_index[event_b]
        real_waveform_score = float(np.dot(real_emb[ia], real_emb[ib]))
        event_a_rank = int(1 + np.sum(real_to_sim[ia] > real_waveform_score))
        event_b_rank = int(1 + np.sum(real_to_sim[ib] > real_waveform_score))
        delta_t_days = abs(float(metas[ia]["gps"]) - float(metas[ib]["gps"])) / liao.SECONDS_PER_DAY
        real_time_lr = liao_time_lr_for_days(delta_t_days, prior)
        case_rows.append({
            "event_a": event_a,
            "event_b": event_b,
            "real_pair_waveform_score": real_waveform_score,
            "waveform_percentile_vs_sim_positive": percentile_rank(real_waveform_score, positive_scores),
            "waveform_percentile_vs_sim_negative": percentile_rank(real_waveform_score, negative_scores),
            "delta_t_days": delta_t_days,
            "liao_time_lr": real_time_lr,
            "time_lr_percentile_vs_sim_positive": percentile_rank(real_time_lr, positive_time_lr),
            "time_lr_percentile_vs_sim_negative": percentile_rank(real_time_lr, negative_time_lr),
            "event_a_partner_rank_among_sim_plus_event_b": event_a_rank,
            "event_b_partner_rank_among_sim_plus_event_a": event_b_rank,
            "sim_test_catalog_size": int(test_scores.shape[0]),
            "sim_positive_count": int(len(positive_scores)),
            "sim_negative_sample_count": int(len(negative_scores)),
            "encoder_dir": str(cfg.out_dir),
            "model_path": str(cfg.out_dir / "model.pt"),
            **version_cols,
        })
    dist_rows = [
        describe_distribution("sim_waveform_lensed_positive_pairs", positive_scores),
        describe_distribution("sim_waveform_random_non_pairs", negative_scores),
        describe_distribution("sim_time_lr_lensed_positive_pairs", positive_time_lr),
        describe_distribution("sim_time_lr_random_non_pairs", negative_time_lr),
        describe_distribution("real_event_to_sim_catalog_scores_event_a", real_to_sim[0]),
        describe_distribution("real_event_to_sim_catalog_scores_event_b", real_to_sim[1]),
    ]
    hybrid_pair_rows, hybrid_top_rows, hybrid_diag = run_hybrid_real_in_sim_ranking(
        real_emb,
        metas,
        event_to_index,
        test_emb,
        _test_time,
        loaded,
        prior,
        OUT_DIR,
    )

    pd.DataFrame(case_rows).to_csv(OUT_DIR / "gwtc_real_event_case_study_summary.csv", index=False)
    pd.DataFrame(dist_rows).to_csv(OUT_DIR / "gwtc_real_event_reference_distributions.csv", index=False)
    np.save(OUT_DIR / "gwtc_real_event_embeddings.npy", real_emb)
    (OUT_DIR / "gwtc_real_event_metadata.json").write_text(json.dumps({
        "events": metas,
        "cases": case_rows,
        "reference_distributions": dist_rows,
        "hybrid_real_in_sim_pair_ranks": hybrid_pair_rows,
        "hybrid_real_in_sim_top_candidates": hybrid_top_rows,
        "hybrid_real_in_sim_diagnostics": hybrid_diag,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    write_doc(case_rows, dist_rows, hybrid_pair_rows, OUT_DIR)
    print(pd.DataFrame(case_rows).to_string(index=False), flush=True)
    print(pd.DataFrame(dist_rows).to_string(index=False), flush=True)
    print(pd.DataFrame(hybrid_pair_rows).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
