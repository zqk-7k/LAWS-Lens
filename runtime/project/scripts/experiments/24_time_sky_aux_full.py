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

FEATURES = ['delta_time', 'sky_sep']
MODES = ['exact', 'mild', 'realistic', 'rough']
FAMILIES = ['SIS', 'PM']
TOPK = 50


def make_time_sky_examples(obs: pd.DataFrame, scores: list[np.ndarray], ens: np.ndarray, gt: np.ndarray):
    candidates = aux.topk_union(scores, ens, TOPK)
    vals = obs.reset_index(drop=True)
    ra = vals['ra'].to_numpy()
    dec = vals['dec'].to_numpy()
    t = vals['geocent_time'].to_numpy()
    rows, y, anchors, cands = [], [], [], []
    for i, row in enumerate(candidates):
        for j in row:
            rows.append([
                float(np.log1p(abs(t[i] - t[j]))),
                float(aux.angular_sep(ra[i], dec[i], ra[j], dec[j])),
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
    val_arrays, val_splits, val_ds, val_scores, val_ens, val_gt, cfg = base.load_score_sets(family, 'val', out_dir / 'val')
    test_arrays, test_splits, test_ds, test_scores, test_ens, test_gt, _ = base.load_score_sets(family, 'test', out_dir / 'test')
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
    val_obs = aux.perturb_observables(aux.catalog_observable_frame(cfg.data_root, family, val_l, val_u), mode, seed=5100 + abs(hash((family, mode, 'val'))) % 10000)
    test_obs = aux.perturb_observables(aux.catalog_observable_frame(cfg.data_root, family, test_l, test_u), mode, seed=6100 + abs(hash((family, mode, 'test'))) % 10000)

    Xv, yv, av, cv = make_time_sky_examples(val_obs, pack['val_scores'], pack['val_ens'], pack['val_gt'])
    Xt, yt, at, ct = make_time_sky_examples(test_obs, pack['test_scores'], pack['test_ens'], pack['test_gt'])

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
    test_metrics = aux.eval_ranker(pt, at, ct, pack['test_gt'])
    result = {
        'family': family,
        'mode': mode,
        'method': 'time_sky_aux_top50_reranker',
        'features': FEATURES,
        'feature_count': len(FEATURES),
        'topk': TOPK,
        'train_examples': int(len(yv)),
        'train_positive': int(yv.sum()),
        'train_negative': int(len(yv) - yv.sum()),
        'val_auc': float(roc_auc_score(yv, pv)),
        **test_metrics,
    }
    (out_dir / 'summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    pd.DataFrame({'anchor': at, 'candidate': ct, 'p_hat': pt, 'is_true': yt}).to_csv(out_dir / 'test_time_sky_candidates.csv', index=False)
    print(result, flush=True)
    return result


def main():
    out_root = Path('runs/et10000_time_sky_aux_full')
    out_root.mkdir(parents=True, exist_ok=True)
    results = []
    for family in FAMILIES:
        pack = load_family_scores(family, out_root)
        for mode in MODES:
            results.append(run_one(family, mode, pack, out_root))
    df = pd.DataFrame(results)
    df.to_csv(out_root / 'time_sky_aux_full_summary.csv', index=False)
    (out_root / 'summary.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    print(df.to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
