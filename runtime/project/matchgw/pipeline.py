from __future__ import annotations

import json
import time
from pathlib import Path
from dataclasses import asdict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, get_worker_info

from .config import MatchRunConfig
from .data import EvaluationSet, PairDataset, ground_truth_partner, load_match_arrays, prepared_channel_count, split_indices
from .matching import evaluate_scores, similarity_matrix, tune_matching, topk_edges
from .rerank import calibrated_candidate_report, fit_pair_calibrator, candidate_feature_frame
from .catalog import catalog_system_report
from .models import AttentiveResNetEncoder1D, CBAMResNetEncoder1D, ConvNeXtEncoder1D, DilatedResNetEncoder1D, GatedTCNEncoder1D, InceptionAttentionEncoder1D, InceptionTimeEncoder1D, MatchEncoder1D, NTXentLoss, PatchTransformerEncoder1D, RocketEncoder1D, SEResNetEncoder1D, TimesNetLiteEncoder1D


def _device(cpu: bool = False) -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() and not cpu else "cpu")


def _amp_dtype(cfg: MatchRunConfig) -> torch.dtype:
    return torch.float16 if cfg.amp_dtype == "fp16" else torch.bfloat16


def _seed_worker(worker_id: int) -> None:
    info = get_worker_info()
    if info is None or not hasattr(info.dataset, "rng"):
        return
    # DataLoader worker 由 fork 产生时会复制 Dataset 中的 numpy RNG。
    # 这里按 worker seed 重置，避免多个 worker 生成重复的数据增强序列。
    info.dataset.rng = np.random.default_rng(info.seed % (2**32))


def _loader_kwargs(cfg: MatchRunConfig, device: torch.device, train: bool = False) -> dict:
    pin_memory = bool(cfg.pin_memory and device.type == "cuda")
    kwargs = {"num_workers": cfg.num_workers, "pin_memory": pin_memory}
    if cfg.num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
        if train:
            kwargs["worker_init_fn"] = _seed_worker
    return kwargs


def build_model(cfg: MatchRunConfig, in_channels: int | None = None) -> torch.nn.Module:
    # 根据配置创建 encoder。最终结果使用 inceptiontime；cnn 保留作 baseline。
    if in_channels is None:
        in_channels = 4 if str(getattr(cfg, "preprocess", "none")).lower() == "multiband" else (2 if cfg.use_hilbert else 1)
    if cfg.model_backbone == "inceptiontime":
        return InceptionTimeEncoder1D(in_channels=in_channels, d_model=cfg.d_model, emb_dim=cfg.emb_dim, width_scale=cfg.width_scale)
    if cfg.model_backbone == "attnresnet":
        return AttentiveResNetEncoder1D(in_channels=in_channels, d_model=cfg.d_model, emb_dim=cfg.emb_dim, width_scale=cfg.width_scale)
    if cfg.model_backbone == "dilatedresnet":
        return DilatedResNetEncoder1D(in_channels=in_channels, d_model=cfg.d_model, emb_dim=cfg.emb_dim, width_scale=cfg.width_scale)
    if cfg.model_backbone == "inceptionattn":
        return InceptionAttentionEncoder1D(in_channels=in_channels, d_model=cfg.d_model, emb_dim=cfg.emb_dim, width_scale=cfg.width_scale)
    if cfg.model_backbone == "convnext1d":
        return ConvNeXtEncoder1D(in_channels=in_channels, d_model=cfg.d_model, emb_dim=cfg.emb_dim, width_scale=cfg.width_scale)
    if cfg.model_backbone == "seresnet":
        return SEResNetEncoder1D(in_channels=in_channels, d_model=cfg.d_model, emb_dim=cfg.emb_dim, width_scale=cfg.width_scale)
    if cfg.model_backbone == "cbamresnet":
        return CBAMResNetEncoder1D(in_channels=in_channels, d_model=cfg.d_model, emb_dim=cfg.emb_dim, width_scale=cfg.width_scale)
    if cfg.model_backbone == "gatedtcn":
        return GatedTCNEncoder1D(in_channels=in_channels, d_model=cfg.d_model, emb_dim=cfg.emb_dim, width_scale=cfg.width_scale)
    if cfg.model_backbone == "patchtst":
        return PatchTransformerEncoder1D(in_channels=in_channels, d_model=cfg.d_model, emb_dim=cfg.emb_dim, width_scale=cfg.width_scale)
    if cfg.model_backbone == "rocket":
        return RocketEncoder1D(in_channels=in_channels, d_model=cfg.d_model, emb_dim=cfg.emb_dim, width_scale=cfg.width_scale)
    if cfg.model_backbone == "timesnetlite":
        return TimesNetLiteEncoder1D(in_channels=in_channels, d_model=cfg.d_model, emb_dim=cfg.emb_dim, width_scale=cfg.width_scale)
    return MatchEncoder1D(in_channels=in_channels, d_model=cfg.d_model, emb_dim=cfg.emb_dim, width_scale=cfg.width_scale)



class HardNegativeDataset(Dataset):
    # 可选 hard negative 微调：把高分但错误的候选对压低。默认关闭，避免不稳定。
    def __init__(self, eval_ds: EvaluationSet, hard_pairs: list[tuple[int, int]], cfg: MatchRunConfig) -> None:
        self.eval_ds = eval_ds
        self.hard_pairs = hard_pairs
        self.cfg = cfg

    def __len__(self) -> int:
        return len(self.hard_pairs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        i, j = self.hard_pairs[idx]
        return self.eval_ds[i], self.eval_ds[j]


def mine_hard_negatives(scores: np.ndarray, gt: np.ndarray, cfg: MatchRunConfig) -> list[tuple[int, int]]:
    edges = topk_edges(
        scores,
        topk=cfg.hard_neg_topk,
        min_score=cfg.hard_neg_min_score,
        mutual=False,
        reciprocal_rank_max=cfg.reciprocal_rank_max,
        row_min_score=None,
        row_min_margin=None,
        edge_rank_bonus=0.0,
    )
    by_anchor: dict[int, list[tuple[int, float]]] = {}
    for i, j, score in sorted(edges, key=lambda e: e[2], reverse=True):
        if int(gt[i]) == int(j):
            continue
        by_anchor.setdefault(int(i), []).append((int(j), float(score)))
        by_anchor.setdefault(int(j), []).append((int(i), float(score)))
    hard = []
    seen: set[tuple[int, int]] = set()
    for i, js in by_anchor.items():
        for j, _ in js[: cfg.hard_neg_per_anchor]:
            e = (i, j) if i < j else (j, i)
            if e not in seen:
                seen.add(e)
                hard.append(e)
    return hard


def hard_negative_finetune(
    model: MatchEncoder1D,
    val_ds: EvaluationSet,
    hard_pairs: list[tuple[int, int]],
    cfg: MatchRunConfig,
    cpu: bool = False,
) -> list[dict]:
    if not hard_pairs or cfg.hard_neg_epochs <= 0:
        return []
    device = _device(cpu)
    model.to(device)
    dl = DataLoader(HardNegativeDataset(val_ds, hard_pairs, cfg), batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.hard_neg_lr, weight_decay=cfg.weight_decay)
    history = []
    for epoch in range(1, cfg.hard_neg_epochs + 1):
        model.train()
        losses = []
        for xa, xb in dl:
            xa = xa.to(device, non_blocking=True)
            xb = xb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            za = model(xa)
            zb = model(xb)
            sim = (za * zb).sum(dim=1)
            loss = cfg.hard_neg_weight * torch.relu(sim - cfg.hard_neg_margin).pow(2).mean()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        row = {"hard_neg_epoch": epoch, "hard_neg_loss": float(np.mean(losses)) if losses else 0.0}
        history.append(row)
        print(json.dumps(row), flush=True)
    return history


def train_encoder(cfg: MatchRunConfig, cpu: bool = False) -> tuple[MatchEncoder1D, dict, dict]:
    # 训练 Siamese encoder：输入两路 waveform，NT-Xent 让正样本 embedding 接近。
    # 这里显式记录 load/train/epoch 级耗时，供不同数据规模训练加速实验使用。
    load_t0 = time.perf_counter()
    arrays = load_match_arrays(cfg)
    splits = split_indices(len(arrays.l1), len(arrays.unlensed), cfg)
    load_s = time.perf_counter() - load_t0

    train_ds = PairDataset(arrays, splits["lensed"]["train"], splits["unlensed"]["train"], cfg)
    in_channels = prepared_channel_count(arrays.l1[0], cfg)
    device = _device(cpu)
    train_dl = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
        **_loader_kwargs(cfg, device, train=True),
    )
    model = build_model(cfg, in_channels=in_channels).to(device)
    if cfg.compile_model and device.type == "cuda":
        model = torch.compile(model)
    loss_fn = NTXentLoss(cfg.tau)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    scaler = torch.amp.GradScaler("cuda", enabled=bool(cfg.amp and cfg.amp_dtype == "fp16" and device.type == "cuda"))

    history = []
    train_t0 = time.perf_counter()
    for epoch in range(1, cfg.epochs + 1):
        epoch_t0 = time.perf_counter()
        model.train()
        losses = []
        batches = 0
        batch_pairs = 0
        for xa, xb in train_dl:
            batches += 1
            batch_pairs += int(xa.shape[0])
            xa = xa.to(device, non_blocking=True)
            xb = xb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=_amp_dtype(cfg), enabled=bool(cfg.amp and device.type == "cuda")):
                loss = loss_fn(model(xa), model(xb))
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()
            losses.append(float(loss.detach().cpu()))
        epoch_s = time.perf_counter() - epoch_t0
        row = {
            "epoch": epoch,
            "loss": float(np.mean(losses)) if losses else float("nan"),
            "epoch_s": epoch_s,
            "batches": batches,
            "batch_pairs": batch_pairs,
            "pairs_per_s": float(batch_pairs / epoch_s) if epoch_s > 0 else 0.0,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
    train_s = time.perf_counter() - train_t0
    train_meta = {
        "history": history,
        "load_s": load_s,
        "train_s": train_s,
        "train_items": int(len(train_ds)),
        "train_batches_per_epoch": int(len(train_dl)),
        "amp": bool(cfg.amp and device.type == "cuda"),
        "amp_dtype": cfg.amp_dtype,
        "num_workers": int(cfg.num_workers),
        "pin_memory": bool(cfg.pin_memory and device.type == "cuda"),
        "compile_model": bool(cfg.compile_model and device.type == "cuda"),
        "in_channels": int(in_channels),
        "mean_epoch_s": float(np.mean([r["epoch_s"] for r in history])) if history else 0.0,
        "mean_pairs_per_s": float(np.mean([r["pairs_per_s"] for r in history])) if history else 0.0,
    }
    return model, {"splits": splits, "arrays": arrays}, train_meta


@torch.no_grad()
def embed_eval(model: MatchEncoder1D, ds: EvaluationSet, cfg: MatchRunConfig, cpu: bool = False) -> np.ndarray:
    # 评估阶段先一次性编码整个 catalog，后续检索都在 embedding 矩阵上完成。
    device = _device(cpu)
    model.eval().to(device)
    dl = DataLoader(ds, batch_size=cfg.eval_batch_size, shuffle=False, **_loader_kwargs(cfg, device))
    chunks = []
    for x in dl:
        with torch.autocast(device_type=device.type, dtype=_amp_dtype(cfg), enabled=bool(cfg.amp and device.type == "cuda")):
            chunks.append(model(x.to(device, non_blocking=True)).float().cpu().numpy())
    return np.concatenate(chunks, axis=0).astype(np.float32)


def default_tuning_grid(cfg: MatchRunConfig) -> dict[str, list]:
    # Full 10k/10k runs spend most evaluation time in repeated validation
    # candidate construction. Keep the grid focused on the operating points
    # that materially change the final candidate pool.
    return {
        "topk": [5, 10, 20],
        "min_score": [None, 0.70, 0.80],
        "mutual": [False],
        "reciprocal_rank_max": [None, 3],
        "row_min_score": [None],
        "row_min_margin": [None],
        "edge_rank_bonus": [0.0],
    }



def default_candidate_params(cfg: MatchRunConfig) -> dict:
    # 最终导出的候选列表默认保留每个事件 Top-10，追求高 candidate recall。
    return {
        "topk": cfg.candidate_topk,
        "min_score": cfg.candidate_min_score,
        "mutual": cfg.candidate_mutual,
        "reciprocal_rank_max": cfg.candidate_reciprocal_rank_max,
        "row_min_score": None,
        "row_min_margin": None,
        "edge_rank_bonus": 0.0,
    }

def run_train_eval(cfg: MatchRunConfig, cpu: bool = False) -> dict:
    # 一次完整实验：训练 -> 验证集调参/校准 -> 测试集评估 -> 保存结果。
    total_t0 = time.perf_counter()
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    model, state, train_info = train_encoder(cfg, cpu=cpu)
    arrays = state["arrays"]
    splits = state["splits"]

    sizes = {
        "lensed_total": int(len(arrays.l1)),
        "unlensed_total": int(len(arrays.unlensed)),
        "train_lensed": int(len(splits["lensed"]["train"])),
        "train_unlensed": int(len(splits["unlensed"]["train"])),
        "val_lensed": int(len(splits["lensed"]["val"])),
        "val_unlensed": int(len(splits["unlensed"]["val"])),
        "test_lensed": int(len(splits["lensed"]["test"])),
        "test_unlensed": int(len(splits["unlensed"]["test"])),
    }
    timings = {
        "load_s": float(train_info.get("load_s", 0.0)),
        "train_s": float(train_info.get("train_s", 0.0)),
        "mean_epoch_s": float(train_info.get("mean_epoch_s", 0.0)),
        "mean_pairs_per_s": float(train_info.get("mean_pairs_per_s", 0.0)),
        "amp": bool(train_info.get("amp", False)),
        "amp_dtype": str(train_info.get("amp_dtype", cfg.amp_dtype)),
        "num_workers": int(train_info.get("num_workers", cfg.num_workers)),
        "pin_memory": bool(train_info.get("pin_memory", False)),
        "compile_model": bool(train_info.get("compile_model", False)),
    }
    results = {
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(cfg).items()},
        "sizes": sizes,
        "timing": timings,
        "history": train_info["history"],
    }

    # 验证集用于选择匹配参数并训练 pair calibrator，不能直接看测试集。
    val_ds = EvaluationSet(arrays, splits["lensed"]["val"], splits["unlensed"]["val"], cfg)
    t0 = time.perf_counter()
    val_emb = embed_eval(model, val_ds, cfg, cpu=cpu)
    timings["val_embed_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    val_scores = similarity_matrix(val_emb)
    timings["val_similarity_s"] = time.perf_counter() - t0
    val_gt = ground_truth_partner(val_ds.meta)
    t0 = time.perf_counter()
    best_params, val_stats = tune_matching(val_scores, val_gt, default_tuning_grid(cfg), metric=cfg.tune_for)
    timings["val_tune_s"] = time.perf_counter() - t0
    results["best_params"] = best_params
    results["val_before_hnm"] = val_stats

    hard_pairs = mine_hard_negatives(val_scores, val_gt, cfg) if cfg.hard_neg_enable else []
    results["hard_negatives"] = len(hard_pairs)
    hn_history = hard_negative_finetune(model, val_ds, hard_pairs, cfg, cpu=cpu)
    if hn_history:
        results["hard_negative_history"] = hn_history
        val_scores = similarity_matrix(embed_eval(model, val_ds, cfg, cpu=cpu))
        best_params, val_stats = tune_matching(val_scores, val_gt, default_tuning_grid(cfg), metric=cfg.tune_for)
        results["best_params"] = best_params
    results["val"] = evaluate_scores(val_scores, val_gt, **best_params)
    candidate_params = default_candidate_params(cfg)
    results["candidate_params"] = candidate_params
    val_candidate_features = candidate_feature_frame(val_scores, val_gt, candidate_params)
    pair_calibrator = fit_pair_calibrator(val_candidate_features, cfg)
    t0 = time.perf_counter()
    val_candidates, val_candidate_stats = calibrated_candidate_report(
        val_scores, val_gt, candidate_params, pair_calibrator, cfg
    )
    timings["val_candidate_s"] = time.perf_counter() - t0
    results["val_candidates"] = val_candidate_stats
    val_catalog_t0 = time.perf_counter()
    val_catalog_tier1, val_catalog_tier1_stats = catalog_system_report(
        val_candidates, val_ds.meta, cfg.p_high, threshold_name="tier1"
    )
    val_catalog_tier12, val_catalog_tier12_stats = catalog_system_report(
        val_candidates, val_ds.meta, cfg.p_low, threshold_name="tier12"
    )
    timings["val_catalog_s"] = time.perf_counter() - val_catalog_t0
    results["val_catalog_tier1"] = val_catalog_tier1_stats
    results["val_catalog_tier12"] = val_catalog_tier12_stats
    results["pair_calibrator"] = pair_calibrator.to_dict()

    # 测试集只用验证阶段确定好的参数和校准器，得到最终论文指标。
    test_ds = EvaluationSet(arrays, splits["lensed"]["test"], splits["unlensed"]["test"], cfg)
    t0 = time.perf_counter()
    test_emb = embed_eval(model, test_ds, cfg, cpu=cpu)
    timings["test_embed_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    test_scores = similarity_matrix(test_emb)
    timings["test_similarity_s"] = time.perf_counter() - t0
    test_gt = ground_truth_partner(test_ds.meta)
    t0 = time.perf_counter()
    results["test"] = evaluate_scores(test_scores, test_gt, **best_params)
    timings["test_match_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    test_candidates, test_candidate_stats = calibrated_candidate_report(
        test_scores, test_gt, candidate_params, pair_calibrator, cfg
    )
    timings["test_candidate_s"] = time.perf_counter() - t0
    results["test_candidates"] = test_candidate_stats
    test_catalog_t0 = time.perf_counter()
    test_catalog_tier1, test_catalog_tier1_stats = catalog_system_report(
        test_candidates, test_ds.meta, cfg.p_high, threshold_name="tier1"
    )
    test_catalog_tier12, test_catalog_tier12_stats = catalog_system_report(
        test_candidates, test_ds.meta, cfg.p_low, threshold_name="tier12"
    )
    timings["test_catalog_s"] = time.perf_counter() - test_catalog_t0
    results["test_catalog_tier1"] = test_catalog_tier1_stats
    results["test_catalog_tier12"] = test_catalog_tier12_stats

    if cfg.export_candidates:
        val_candidates.to_csv(cfg.out_dir / "val_candidates.csv", index=False)
        test_candidates.to_csv(cfg.out_dir / "test_candidates.csv", index=False)
        val_catalog_tier1.to_csv(cfg.out_dir / "val_catalog_systems_tier1.csv", index=False)
        val_catalog_tier12.to_csv(cfg.out_dir / "val_catalog_systems_tier12.csv", index=False)
        test_catalog_tier1.to_csv(cfg.out_dir / "test_catalog_systems_tier1.csv", index=False)
        test_catalog_tier12.to_csv(cfg.out_dir / "test_catalog_systems_tier12.csv", index=False)
        pd.DataFrame([val_candidate_stats]).to_csv(cfg.out_dir / "val_candidate_summary.csv", index=False)
        pd.DataFrame([test_candidate_stats]).to_csv(cfg.out_dir / "test_candidate_summary.csv", index=False)
        pd.DataFrame([val_catalog_tier1_stats]).to_csv(cfg.out_dir / "val_catalog_tier1_summary.csv", index=False)
        pd.DataFrame([val_catalog_tier12_stats]).to_csv(cfg.out_dir / "val_catalog_tier12_summary.csv", index=False)
        pd.DataFrame([test_catalog_tier1_stats]).to_csv(cfg.out_dir / "test_catalog_tier1_summary.csv", index=False)
        pd.DataFrame([test_catalog_tier12_stats]).to_csv(cfg.out_dir / "test_catalog_tier12_summary.csv", index=False)

    save_t0 = time.perf_counter()
    torch.save({"model": model.state_dict(), "config": results["config"], "pair_calibrator": pair_calibrator.to_dict()}, cfg.out_dir / "model.pt")
    pd.DataFrame(train_info["history"]).to_csv(cfg.out_dir / "history.csv", index=False)
    timings["total_s"] = time.perf_counter() - total_t0
    timings["save_s"] = time.perf_counter() - save_t0
    pd.DataFrame([timings]).to_csv(cfg.out_dir / "timing.csv", index=False)
    pd.DataFrame([sizes]).to_csv(cfg.out_dir / "sizes.csv", index=False)
    with open(cfg.out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2), flush=True)
    return results
