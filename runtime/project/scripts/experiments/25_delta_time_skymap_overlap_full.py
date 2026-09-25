from __future__ import annotations

import importlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

base = importlib.import_module('scripts.experiments.17_waveform_reranker_existing')
aux = importlib.import_module('scripts.experiments.21_observable_aux_reranker')

FEATURES = ['delta_time', 'log_sky_map_overlap']
MODES = ['exact', 'mild', 'realistic', 'rough']
FAMILIES = ['SIS', 'PM']
TOPK = 50
# 当前数据没有真实 HEALPix skymap，这里用二维局部高斯天空定位近似 skymap。
# sigma 单位为 rad，对应 21_observable_aux_reranker.py 中 perturb_observables 的 sky_sigma。
SKY_SIGMA = {
    'exact': 0.03,
    'mild': 0.03,
    'realistic': 0.08,
    'rough': 0.20,
}


def gaussian_skymap_log_overlap(sep: float, sigma1: float, sigma2: float) -> float:
    var = sigma1 * sigma1 + sigma2 * sigma2
    # normalized 2D Gaussian overlap on tangent plane:
    # integral N(mu1,s1^2I) N(mu2,s2^2I) dOmega
    return float(-np.log(2.0 * np.pi * var) - (sep * sep) / (2.0 * var))


def make_examples(obs: pd.DataFrame, scores: list[np.ndarray], ens: np.ndarray, gt: np.ndarray, mode: str):
    candidates = aux.topk_union(scores, ens, TOPK)
    vals = obs.reset_index(drop=True)
    ra = vals['ra'].to_numpy()
    dec = vals['dec'].to_numpy()
    t = vals['geocent_time'].to_numpy()
    sigma = SKY_SIGMA[mode]
    rows, y, anchors, cands = [], [], [], []
    for i, row in enumerate(candidates):
        for j in row:
            sep = float(aux.angular_sep(ra[i], dec[i], ra[j], dec[j]))
            rows.append([
                float(np.log1p(abs(t[i] - t[j]))),
                gaussian_skymap_log_overlap(sep, sigma, sigma),
            ])
            y.append(1 if int(gt[i]) == int(j) else 0)
            anchors.append(i)
            cands.append(j)
    return (
        np.asarray(rows, dtype=np.float32),
        np.asarray(y, dtype=np.int8),
        np.asarray(anchors, dtype=np.int32),
        np.asarray(cands, dtype=np.int32),
    )


def load_family_scores(family: str, out_root: Path):
    out_dir = out_root / family / '_score_cache'
    _, val_splits, _, val_scores, val_ens, val_gt, cfg = base.load_score_sets(family, 'val', out_dir / 'val')
    _, test_splits, _, test_scores, test_ens, test_gt, _ = base.load_score_sets(family, 'test', out_dir / 'test')
    return {
        'cfg': cfg,
        'val_splits': val_splits,
        'test_splits': test_splits,
        'val_scores': val_scores,
        'val_ens': val_ens,
        'val_gt': val_gt,
        'test_scores': test_scores,
        'test_ens': test_ens,
        'test_gt': test_gt,
    }


def run_one(family: str, mode: str, pack: dict, out_root: Path) -> dict:
    out_dir = out_root / family / mode
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = pack['cfg']
    val_l = pack['val_splits']['lensed']['val']
    val_u = pack['val_splits']['unlensed']['val']
    test_l = pack['test_splits']['lensed']['test']
    test_u = pack['test_splits']['unlensed']['test']
    val_obs = aux.perturb_observables(aux.catalog_observable_frame(cfg.data_root, family, val_l, val_u), mode, seed=7100 + abs(hash((family, mode, 'val'))) % 10000)
    test_obs = aux.perturb_observables(aux.catalog_observable_frame(cfg.data_root, family, test_l, test_u), mode, seed=8100 + abs(hash((family, mode, 'test'))) % 10000)

    Xv, yv, av, cv = make_examples(val_obs, pack['val_scores'], pack['val_ens'], pack['val_gt'], mode)
    Xt, yt, at, ct = make_examples(test_obs, pack['test_scores'], pack['test_ens'], pack['test_gt'], mode)

    clf = HistGradientBoostingClassifier(
        max_iter=250,
        learning_rate=0.06,
        max_leaf_nodes=15,
        l2_regularization=1e-4,
        class_weight='balanced',
        random_state=42,
    )
    clf.fit(Xv, yv)
    pv = clf.predict_proba(Xv)[:, 1]
    pt = clf.predict_proba(Xt)[:, 1]
    result = {
        'family': family,
        'mode': mode,
        'method': 'delta_time_gaussian_skymap_overlap_top50_reranker',
        'features': FEATURES,
        'feature_count': len(FEATURES),
        'topk': TOPK,
        'sky_overlap_note': 'Gaussian skymap approximation; current data has ra/dec only, no HEALPix skymap.',
        'sky_sigma_rad': SKY_SIGMA[mode],
        'train_examples': int(len(yv)),
        'train_positive': int(yv.sum()),
        'train_negative': int(len(yv) - yv.sum()),
        'val_auc': float(roc_auc_score(yv, pv)),
        **aux.eval_ranker(pt, at, ct, pack['test_gt']),
    }
    (out_dir / 'summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    pd.DataFrame({'anchor': at, 'candidate': ct, 'p_hat': pt, 'is_true': yt}).to_csv(out_dir / 'test_delta_time_skymap_overlap_candidates.csv', index=False)
    print(result, flush=True)
    return result


def main():
    out_root = Path('runs/et10000_delta_time_skymap_overlap_full')
    out_root.mkdir(parents=True, exist_ok=True)
    results = []
    for family in FAMILIES:
        pack = load_family_scores(family, out_root)
        for mode in MODES:
            results.append(run_one(family, mode, pack, out_root))
    df = pd.DataFrame(results)
    df.to_csv(out_root / 'delta_time_skymap_overlap_full_summary.csv', index=False)
    (out_root / 'summary.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    print(df.to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
