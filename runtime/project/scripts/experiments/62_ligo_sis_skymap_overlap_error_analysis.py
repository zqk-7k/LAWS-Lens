from __future__ import annotations
import importlib
from pathlib import Path
import numpy as np
import pandas as pd
import torch

m = importlib.import_module('scripts.experiments.51_ligo_sis_resnet_grid18_skymap_rerank')
base = m.base
OUT_ROOT = Path('runs/ligo_sis_skymap_overlap_error_analysis_20260605')
SRC_ROOT = Path('runs/ligo_sis_resnet_grid18_skymap_rerank_20260604')
EPS = 1e-8


def load_model_and_split(split='test'):
    job = m.JOBS[0]
    _, train_ds, _, _, _, _ = m.load_pack(job, 'train', SRC_ROOT / 'train', False)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = m.SkyMapCNN(in_channels=int(train_ds[0].shape[0])).to(device)
    ckpt = torch.load(SRC_ROOT / 'ligo_noisy_sis' / 'grid_skymap_cnn.pt', map_location=device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    _, ds, raw, time_obs, gt, scores = m.load_pack(job, split, OUT_ROOT / split, True)
    cache = OUT_ROOT / f'{split}_prob.npy'
    if cache.exists():
        prob = np.load(cache)
    else:
        prob = m.predict_maps(model, ds, device)
        OUT_ROOT.mkdir(parents=True, exist_ok=True)
        np.save(cache, prob)
    return prob.astype(np.float32), raw, time_obs, gt, scores.astype(np.float32)


def true_pixel_and_sep(raw):
    true_unit = base.unit_vectors(raw).astype(np.float32)
    sep = np.arccos(np.clip(true_unit @ m.CENTER_UNIT.T, -1.0, 1.0))
    pix = np.argmin(sep, axis=1).astype(np.int32)
    return true_unit, pix, sep


def skymap_event_table(prob, raw):
    true_unit, pix, sep = true_pixel_and_sep(raw)
    err = m.map_center_error(prob, raw)
    order = np.argsort(-prob, axis=1)
    ranks = np.empty_like(order, dtype=np.int32)
    ranks[np.arange(len(prob))[:, None], order] = np.arange(1, prob.shape[1] + 1, dtype=np.int32)
    true_rank = ranks[np.arange(len(prob)), pix]
    true_prob = prob[np.arange(len(prob)), pix]
    top1 = order[:, 0]
    top1_prob = prob[np.arange(len(prob)), top1]
    top1_sep = sep[np.arange(len(prob)), top1]
    entropy = -np.sum(prob * np.log(np.maximum(prob, EPS)), axis=1)
    sorted_prob = np.sort(prob, axis=1)[:, ::-1]
    return pd.DataFrame({
        'event_index': np.arange(len(prob)),
        'sky_error_rad': err,
        'true_pixel': pix,
        'true_pixel_rank': true_rank,
        'true_pixel_prob': true_prob,
        'top1_pixel': top1,
        'top1_prob': top1_prob,
        'top1_sep_rad': top1_sep,
        'top1_to_true_prob_ratio': top1_prob / np.maximum(true_prob, EPS),
        'entropy': entropy,
        'entropy_norm': entropy / np.log(prob.shape[1]),
        'top1_prob_mass': sorted_prob[:, 0],
        'top5_prob_mass': sorted_prob[:, :5].sum(axis=1),
        'top10_prob_mass': sorted_prob[:, :10].sum(axis=1),
        'top50_prob_mass': sorted_prob[:, :50].sum(axis=1),
    })


def binned_summary(df, col, bins):
    tmp = df.copy()
    tmp['bin'] = pd.cut(tmp[col], bins=bins, include_lowest=True)
    return tmp.groupby('bin', observed=False).agg(
        n=('event_index', 'size'),
        sky_error_mean=('sky_error_rad', 'mean'),
        true_rank_median=('true_pixel_rank', 'median'),
        true_prob_mean=('true_pixel_prob', 'mean'),
        entropy_norm_mean=('entropy_norm', 'mean'),
        top10_mass_mean=('top10_prob_mass', 'mean'),
    ).reset_index()


def overlap_for_rows(prob, rows):
    return np.log(np.maximum(prob[rows] @ prob.T, EPS)).astype(np.float32)


def overlap_rank_table(prob, raw, gt, scores):
    valid = np.flatnonzero(gt >= 0).astype(np.int32)
    event_df = skymap_event_table(prob, raw)
    rows = []
    chunk = 128
    for st in range(0, len(valid), chunk):
        anchors = valid[st:st+chunk]
        ov = overlap_for_rows(prob, anchors)
        for rp, a in enumerate(anchors):
            p = int(gt[a])
            s = ov[rp].copy()
            s[int(a)] = -np.inf
            true_score = float(s[p])
            rank = int(1 + np.sum(s > true_score))
            # Top false candidate diagnostics.
            top = np.argsort(-s)[:20]
            false_top = [int(x) for x in top if int(x) != p][:10]
            best_false = false_top[0]
            rows.append({
                'anchor': int(a),
                'partner': p,
                'true_log_overlap': true_score,
                'true_overlap': float(np.exp(true_score)),
                'overlap_rank': rank,
                'best_false': best_false,
                'best_false_log_overlap': float(s[best_false]),
                'best_false_overlap': float(np.exp(s[best_false])),
                'false_minus_true_log_overlap': float(s[best_false] - true_score),
                'anchor_sky_error': float(event_df.loc[a, 'sky_error_rad']),
                'partner_sky_error': float(event_df.loc[p, 'sky_error_rad']),
                'best_false_sky_error': float(event_df.loc[best_false, 'sky_error_rad']),
                'anchor_true_pixel_rank': int(event_df.loc[a, 'true_pixel_rank']),
                'partner_true_pixel_rank': int(event_df.loc[p, 'true_pixel_rank']),
                'anchor_entropy_norm': float(event_df.loc[a, 'entropy_norm']),
                'partner_entropy_norm': float(event_df.loc[p, 'entropy_norm']),
                'waveform_score_true': float(scores[a, p]),
                'waveform_score_best_false': float(scores[a, best_false]),
            })
    return pd.DataFrame(rows)


def quantiles(s):
    qs = [0, .01, .05, .1, .25, .5, .75, .9, .95, .99, 1]
    return {f'q{int(q*100):02d}': float(np.quantile(s, q)) for q in qs}


def distribution_summary(event_df, pair_df):
    rows = []
    for name, series in [
        ('event_sky_error_rad', event_df['sky_error_rad']),
        ('event_true_pixel_rank', event_df['true_pixel_rank']),
        ('event_true_pixel_prob', event_df['true_pixel_prob']),
        ('event_entropy_norm', event_df['entropy_norm']),
        ('event_top10_prob_mass', event_df['top10_prob_mass']),
        ('pair_true_overlap', pair_df['true_overlap']),
        ('pair_best_false_overlap', pair_df['best_false_overlap']),
        ('pair_false_minus_true_log_overlap', pair_df['false_minus_true_log_overlap']),
        ('pair_overlap_rank', pair_df['overlap_rank']),
    ]:
        row = {'metric': name, 'mean': float(np.mean(series)), 'std': float(np.std(series)), **quantiles(series)}
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    prob, raw, time_obs, gt, scores = load_model_and_split('test')
    event_df = skymap_event_table(prob, raw)
    pair_df = overlap_rank_table(prob, raw, gt, scores)
    event_df.to_csv(OUT_ROOT / 'event_skymap_errors.csv', index=False)
    pair_df.to_csv(OUT_ROOT / 'pair_overlap_errors.csv', index=False)
    dist = distribution_summary(event_df, pair_df)
    dist.to_csv(OUT_ROOT / 'distribution_summary.csv', index=False)
    # Error bins and rank bins.
    binned_summary(event_df, 'sky_error_rad', [0, .5, 1.0, 1.5, 2.0, 3.2]).to_csv(OUT_ROOT / 'event_by_sky_error_bins.csv', index=False)
    binned_summary(event_df, 'true_pixel_rank', [0, 10, 50, 100, 200, 400, 648]).to_csv(OUT_ROOT / 'event_by_true_pixel_rank_bins.csv', index=False)
    # Pair overlap rank buckets.
    pair_df['rank_bin'] = pd.cut(pair_df['overlap_rank'], bins=[0,10,50,100,500,1000,5000], include_lowest=True)
    pair_bucket = pair_df.groupby('rank_bin', observed=False).agg(
        n=('anchor','size'),
        true_overlap_mean=('true_overlap','mean'),
        best_false_overlap_mean=('best_false_overlap','mean'),
        false_minus_true_log_mean=('false_minus_true_log_overlap','mean'),
        anchor_sky_error_mean=('anchor_sky_error','mean'),
        partner_sky_error_mean=('partner_sky_error','mean'),
        anchor_true_pixel_rank_median=('anchor_true_pixel_rank','median'),
        partner_true_pixel_rank_median=('partner_true_pixel_rank','median'),
    ).reset_index()
    pair_bucket.to_csv(OUT_ROOT / 'pair_by_overlap_rank_bins.csv', index=False)
    print('DISTRIBUTION_SUMMARY')
    print(dist.to_string(index=False), flush=True)
    print('PAIR_RANK_BUCKETS')
    print(pair_bucket.to_string(index=False), flush=True)
    print('OUTPUT', OUT_ROOT, flush=True)

if __name__ == '__main__':
    main()
