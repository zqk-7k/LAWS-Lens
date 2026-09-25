from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import matchgw.data as data_mod
from matchgw.config import MatchRunConfig
from matchgw.data import EvaluationSet, ground_truth_partner, load_match_arrays, split_indices
from matchgw.matching import retrieval_metrics, similarity_matrix
from matchgw.pipeline import build_model, embed_eval, train_encoder


OUT_ROOT = Path("runs/ligo_noisy_sis_waveform_ablation_20260609")
DATA_ROOT = Path("/root/autodl-tmp/gw_ligo_10000_matchstyle_20260527_091859")
ORIGINAL_ZSCORE = data_mod.zscore
ORIGINAL_AUGMENT = data_mod.augment


def make_cfg(name: str) -> MatchRunConfig:
    cfg = MatchRunConfig(
        data_root=DATA_ROOT,
        model_type="SIS",
        data_mode="noisy",
        out_dir=OUT_ROOT / name,
        backbone="inceptiontime",
        preprocess="bandpass",
        bandpass_low=40,
        bandpass_high=580,
        target_len=8192,
        stride=2,
        lensed_limit=10000,
        unlensed_limit=10000,
        epochs=50,
        batch_size=96,
        eval_batch_size=256,
        lr=1e-3,
        weight_decay=1e-4,
        tau=0.07,
        emb_dim=128,
        d_model=256,
        width_scale=2.0,
        aug_roll=128,
        aug_scale=0.10,
        aug_noise=0.01,
        aug_flip=True,
        amp=True,
        amp_dtype="bf16",
        num_workers=2,
        pin_memory=True,
        export_candidates=False,
    )
    if "multiband" in name:
        cfg.preprocess = "multiband"
    if "no_peak_flip" in name:
        cfg.aug_flip = False
    if "pure_aux" in name:
        cfg.use_pure_aux = True
    return cfg


def zscore_per_channel(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        return ((x - x.mean()) / (x.std() + 1e-8)).astype(np.float32, copy=False)
    mean = x.mean(axis=-1, keepdims=True)
    std = x.std(axis=-1, keepdims=True) + 1e-8
    return ((x - mean) / std).astype(np.float32, copy=False)


def patch_data_pipeline(variant: str) -> None:
    def augment_axis_last(x: np.ndarray, cfg: MatchRunConfig, rng: np.random.Generator) -> np.ndarray:
        y = x.copy()
        if cfg.aug_flip:
            y = data_mod.peak_flip(y)
        if cfg.aug_roll > 0:
            y = np.roll(y, int(rng.integers(-cfg.aug_roll, cfg.aug_roll + 1)), axis=-1)
        if cfg.aug_scale > 0:
            y = y * float(1.0 + rng.uniform(-cfg.aug_scale, cfg.aug_scale))
        if cfg.aug_noise > 0:
            scale = y.std(axis=-1, keepdims=True) if y.ndim > 1 else float(y.std())
            y = y + rng.normal(0.0, cfg.aug_noise * (scale + 1e-8), size=y.shape)
        return data_mod.zscore(y)

    if "per_channel_zscore" in variant:
        data_mod.zscore = zscore_per_channel
    else:
        data_mod.zscore = ORIGINAL_ZSCORE

    if "fix_roll" in variant or "per_channel_zscore" in variant or "pure_aux" in variant:
        data_mod.augment = augment_axis_last
    else:
        data_mod.augment = ORIGINAL_AUGMENT


def run_variant(name: str) -> dict:
    patch_data_pipeline(name)
    cfg = make_cfg(name)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    model_path = cfg.out_dir / "model.pt"
    summary_path = cfg.out_dir / "waveform_summary.json"

    if model_path.exists() and summary_path.exists():
        arrays = load_match_arrays(cfg)
        splits = split_indices(len(arrays.l1), len(arrays.unlensed), cfg)
        model = build_model(cfg)
        ckpt = torch.load(model_path, map_location="cpu")
        model.load_state_dict(ckpt["model"], strict=True)
        train_info = json.loads(summary_path.read_text(encoding="utf-8")).get("timing", {})
    else:
        model, state, train_info = train_encoder(cfg, cpu=False)
        arrays = state["arrays"]
        splits = state["splits"]
        torch.save(
            {"model": model.state_dict(), "config": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(cfg).items()}},
            model_path,
        )
        pd.DataFrame(train_info["history"]).to_csv(cfg.out_dir / "history.csv", index=False)

    ds = EvaluationSet(arrays, splits["lensed"]["test"], splits["unlensed"]["test"], cfg)
    gt = ground_truth_partner(ds.meta)
    emb_path = cfg.out_dir / "test_embeddings.npy"
    scores_path = cfg.out_dir / "test_scores.npy"
    if emb_path.exists() and scores_path.exists():
        emb = np.load(emb_path)
        scores = np.load(scores_path)
    else:
        emb = embed_eval(model, ds, cfg, cpu=False).astype(np.float32)
        scores = similarity_matrix(emb).astype(np.float32)
        np.fill_diagonal(scores, -np.inf)
        np.save(emb_path, emb)
        np.save(scores_path, scores)

    metrics = retrieval_metrics(scores, gt, ks=(1, 5, 10, 50, 100, 500))
    valid = np.flatnonzero(gt >= 0)
    true = scores[valid, gt[valid]]
    ranks = 1 + np.sum(scores[valid] > true[:, None], axis=1)
    usable = scores.shape[1] - 1
    for pct in (1, 5, 10):
        k = int(np.ceil(usable * pct / 100.0))
        metrics[f"top_{pct}pct_k"] = k
        metrics[f"top_{pct}pct"] = float(np.mean(ranks <= k))

    row = {
        "variant": name,
        "family": "SIS",
        "detector": "LIGO",
        "data_mode": "noisy",
        "method": "waveform_only",
        **metrics,
        "train_s": train_info.get("train_s", np.nan),
        "mean_epoch_s": train_info.get("mean_epoch_s", np.nan),
        "preprocess": cfg.preprocess,
        "aug_flip": cfg.aug_flip,
        "use_pure_aux": cfg.use_pure_aux,
    }
    summary_path.write_text(
        json.dumps(
            {
                "config": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(cfg).items()},
                "timing": {k: v for k, v in train_info.items() if k != "history"},
                "test_waveform": metrics,
                "note": "LIGO noisy SIS waveform-only ablation. Variants patch roll axis and/or zscore in matchgw.data at runtime.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return row


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    variants = [
        "baseline_reproduce",
        "fix_roll",
        "fix_roll_per_channel_zscore",
        "fix_roll_no_peak_flip",
        "multiband",
        "fix_roll_per_channel_zscore_pure_aux",
    ]
    rows = []
    for name in variants:
        t0 = time.perf_counter()
        print("RUN_VARIANT", name, flush=True)
        row = run_variant(name)
        row["elapsed_s"] = float(time.perf_counter() - t0)
        rows.append(row)
        pd.DataFrame(rows).to_csv(OUT_ROOT / "waveform_ablation_partial.csv", index=False)
        print(json.dumps(row, indent=2), flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT_ROOT / "waveform_ablation_summary.csv", index=False)
    print(df.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
