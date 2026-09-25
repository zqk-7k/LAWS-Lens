from __future__ import annotations
import importlib
from pathlib import Path
import numpy as np
import pandas as pd
import torch

base51 = importlib.import_module('scripts.experiments.51_ligo_sis_resnet_grid18_skymap_rerank')
base58 = importlib.import_module('scripts.experiments.58_ligo_sis_expected_angular_grid18_skymap_rerank')
base59 = importlib.import_module('scripts.experiments.59_ligo_sis_detector_interaction_expected_angular')
base = base51.base

OUT_ROOT = Path('runs/ligo_sis_skymap_quality_report_20260605')
EPS = 1e-8
MODELS = [
    ('51_resnet_grid18', base51, Path('runs/ligo_sis_resnet_grid18_skymap_rerank_20260604/ligo_noisy_sis/grid_skymap_cnn.pt')),
    ('58_expected_angular', base58, Path('runs/ligo_sis_expected_angular_grid18_skymap_rerank_20260605/ligo_noisy_sis/grid_skymap_cnn.pt')),
    ('59_detector_interaction', base59, Path('runs/ligo_sis_detector_interaction_expected_angular_20260605/ligo_noisy_sis/grid_skymap_cnn.pt')),
]


def load_split(mod, split, out_dir):
    job = mod.JOBS[0]
    _, ds, raw, time_obs, gt, scores = mod.load_pack(job, split, out_dir / split, True)
    return ds, raw, gt


def load_predict(model_name, mod, ckpt_path, split):
    out_dir = OUT_ROOT / model_name
    ds, raw, gt = load_split(mod, split, out_dir)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = mod.SkyMapCNN(in_channels=int(ds[0].shape[0])).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    prob = mod.predict_maps(model, ds, device)
    return prob.astype(np.float32), raw, gt


def true_pixel_indices(mod, raw):
    true = mod.base.unit_vectors(raw).astype(np.float32)
    sep = np.arccos(np.clip(true @ mod.CENTER_UNIT.T, -1.0, 1.0))
    return np.argmin(sep, axis=1).astype(np.int32), sep


def expected_error(mod, prob, raw):
    return mod.map_center_error(prob, raw)


def pixel_rank_metrics(mod, prob, raw):
    pix, sep = true_pixel_indices(mod, raw)
    order = np.argsort(-prob, axis=1)
    ranks = np.empty_like(order, dtype=np.int32)
    ranks[np.arange(len(prob))[:, None], order] = np.arange(1, prob.shape[1] + 1, dtype=np.int32)
    true_rank = ranks[np.arange(len(prob)), pix]
    true_prob = prob[np.arange(len(prob)), pix]
    top1 = order[:, 0]
    top1_sep = sep[np.arange(len(prob)), top1]
    entropy = -np.sum(prob * np.log(np.maximum(prob, EPS)), axis=1)
    max_entropy = np.log(prob.shape[1])
    return {
        'true_pixel_rank_median': float(np.median(true_rank)),
        'true_pixel_rank_mean': float(np.mean(true_rank)),
        'true_pixel_top1': float(np.mean(true_rank <= 1)),
        'true_pixel_top5': float(np.mean(true_rank <= 5)),
        'true_pixel_top10': float(np.mean(true_rank <= 10)),
        'true_pixel_top50': float(np.mean(true_rank <= 50)),
        'true_pixel_prob_mean': float(np.mean(true_prob)),
        'true_pixel_prob_median': float(np.median(true_prob)),
        'top1_pixel_sep_mean_rad': float(np.mean(top1_sep)),
        'top1_pixel_sep_median_rad': float(np.median(top1_sep)),
        'entropy_mean': float(np.mean(entropy)),
        'entropy_norm_mean': float(np.mean(entropy / max_entropy)),
    }


def pair_overlap_metrics(prob, gt, sample_neg=200000):
    rng = np.random.default_rng(61001)
    valid = np.flatnonzero(gt >= 0).astype(np.int32)
    pos = np.sum(prob[valid] * prob[gt[valid].astype(int)], axis=1)
    n = len(prob)
    a = rng.choice(valid, size=sample_neg, replace=True)
    c = rng.integers(0, n, size=sample_neg, dtype=np.int32)
    bad = (c == a) | (c == gt[a])
    while bad.any():
        c[bad] = rng.integers(0, n, size=int(bad.sum()), dtype=np.int32)
        bad = (c == a) | (c == gt[a])
    neg = np.sum(prob[a] * prob[c], axis=1)
    # AUC by random positive-negative comparisons, not sklearn to avoid huge arrays.
    pp = rng.choice(pos, size=sample_neg, replace=True)
    auc = float(np.mean(pp > neg) + 0.5 * np.mean(pp == neg))
    return {
        'pair_pos_overlap_mean': float(np.mean(pos)),
        'pair_pos_overlap_median': float(np.median(pos)),
        'pair_neg_overlap_mean': float(np.mean(neg)),
        'pair_neg_overlap_median': float(np.median(neg)),
        'pair_overlap_ratio_mean': float(np.mean(pos) / max(float(np.mean(neg)), EPS)),
        'pair_overlap_auc_sampled': auc,
    }


def summarize(model_name, mod, ckpt_path, split):
    prob, raw, gt = load_predict(model_name, mod, ckpt_path, split)
    err = expected_error(mod, prob, raw)
    row = {
        'model': model_name,
        'split': split,
        'n': int(len(prob)),
        'n_pix': int(prob.shape[1]),
        'sky_mean_error_rad': float(np.mean(err)),
        'sky_median_error_rad': float(np.median(err)),
        'sky_err_lt_0p5_rad': float(np.mean(err < 0.5)),
        'sky_err_lt_1p0_rad': float(np.mean(err < 1.0)),
        'sky_err_lt_1p5_rad': float(np.mean(err < 1.5)),
    }
    row.update(pixel_rank_metrics(mod, prob, raw))
    row.update(pair_overlap_metrics(prob, gt))
    return row


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows = []
    for model_name, mod, ckpt in MODELS:
        for split in ['val', 'test']:
            row = summarize(model_name, mod, ckpt, split)
            rows.append(row)
            print(row, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT_ROOT / 'summary.csv', index=False)
    print(df.to_string(index=False), flush=True)

if __name__ == '__main__':
    main()
