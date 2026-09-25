from __future__ import annotations

import importlib
import json
from dataclasses import fields
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from matchgw.config import MatchRunConfig
from matchgw.data import EvaluationSet, ground_truth_partner, load_match_arrays, split_indices
from matchgw.trigger_time import catalog_trigger_time_frame

base = importlib.import_module('scripts.experiments.39_waveform_predicted_skymap_rerank')
aux = importlib.import_module('scripts.experiments.21_observable_aux_reranker')
skyfeat = importlib.import_module('scripts.experiments.44_waveform_feature_skymap_predictor')

OUT_ROOT = Path('runs/ligo_waveform_feature_skymap_predictor_light_20260604')
SOURCE_ROOT = Path('runs/unified_sky_aux_comparison_20260603')
JOBS = [j for j in base.JOBS if j['mode'] == 'noisy' and j['detector'] == 'LIGO']
PREDICTORS = ['waveform_stats_ridge', 'fusion_stats_embedding_ridge']
NEG_PER_POS = 500


def cfg_from_run(run_dir: Path, out_dir: Path) -> MatchRunConfig:
    data = json.loads((run_dir / 'summary.json').read_text(encoding='utf-8'))['config']
    valid = {f.name for f in fields(MatchRunConfig)}
    kwargs = {k: v for k, v in data.items() if k in valid}
    for key in ['data_root', 'out_dir']:
        if key in kwargs:
            kwargs[key] = Path(kwargs[key])
    kwargs['out_dir'] = out_dir
    return MatchRunConfig(**kwargs)


def cache_name(job: dict, split: str, suffix: str) -> Path:
    name = f"{job['detector'].lower()}_{job['mode']}_{job['family'].lower()}"
    return SOURCE_ROOT / name / split / f"{job['detector']}_{job['mode']}_{job['family']}_{split}_{suffix}.npy"


def light_pack(job: dict, split: str, out_dir: Path, need_scores: bool):
    cfg = cfg_from_run(job['run'], out_dir)
    arrays = load_match_arrays(cfg)
    splits = split_indices(len(arrays.l1), len(arrays.unlensed), cfg)
    lidx = splits['lensed'][split]
    uidx = splits['unlensed'][split]
    ds = EvaluationSet(arrays, lidx, uidx, cfg)
    gt = ground_truth_partner(ds.meta)
    raw_obs = aux.catalog_observable_frame(cfg.data_root, job['family'], lidx, uidx).reset_index(drop=True)
    time_obs = catalog_trigger_time_frame(cfg.data_root, job['family'], lidx, uidx, detector=job['detector']).reset_index(drop=True)
    emb_path = cache_name(job, split, 'embeddings')
    if not emb_path.exists():
        raise FileNotFoundError(emb_path)
    emb = np.load(emb_path)
    scores = None
    if need_scores:
        score_path = cache_name(job, split, 'scores')
        if not score_path.exists():
            raise FileNotFoundError(score_path)
        scores = np.load(score_path)
    return cfg, ds, raw_obs, time_obs, gt, emb, scores


def run_one(job: dict, predictor: str) -> dict:
    name = f"{job['detector'].lower()}_{job['mode']}_{job['family'].lower()}"
    out_dir = OUT_ROOT / name
    print('LOAD_LIGHT', name, predictor, flush=True)
    train_cfg, train_ds, train_raw, train_time, train_gt, train_emb, _ = light_pack(job, 'train', out_dir / 'train', need_scores=False)
    val_cfg, val_ds, val_raw, val_time, val_gt, val_emb, val_scores = light_pack(job, 'val', out_dir / 'val', need_scores=True)
    test_cfg, test_ds, test_raw, test_time, test_gt, test_emb, test_scores = light_pack(job, 'test', out_dir / 'test', need_scores=True)

    train_stats = skyfeat.extract_waveform_features(job, 'train', train_ds, out_dir / 'features')
    val_stats = skyfeat.extract_waveform_features(job, 'val', val_ds, out_dir / 'features')
    test_stats = skyfeat.extract_waveform_features(job, 'test', test_ds, out_dir / 'features')

    print('FIT_SKY_LIGHT', name, predictor, train_stats.shape, flush=True)
    model, sigma, mean_err, med_err, acc05, acc10 = skyfeat.fit_predictor(predictor, train_raw, train_stats, train_emb, val_raw, val_stats, val_emb)
    val_mu = skyfeat.predict_mu(predictor, model, val_stats, val_emb)
    test_mu = skyfeat.predict_mu(predictor, model, test_stats, test_emb)
    val_ranks = base.row_ranks(val_scores)
    test_ranks = base.row_ranks(test_scores)

    Xv, yv = skyfeat.train_examples(val_time, val_gt, val_mu, sigma, val_scores, val_ranks, job, predictor)
    clf = HistGradientBoostingClassifier(max_iter=320, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1e-4, class_weight='balanced', random_state=42)
    clf.fit(Xv, yv)
    pv = clf.predict_proba(Xv)[:, 1]
    row = {
        'detector': job['detector'],
        'data_mode': job['mode'],
        'family': job['family'],
        'predictor': predictor,
        'sky_predictor_input': 'waveform_stats' if predictor == 'waveform_stats_ridge' else 'waveform_stats_plus_matching_embedding',
        'sky_sigma_rad': sigma,
        'sky_val_mean_error_rad': mean_err,
        'sky_val_median_error_rad': med_err,
        'sky_val_frac_err_lt_0p5': acc05,
        'sky_val_frac_err_lt_1p0': acc10,
        'val_auc_sampled': float(roc_auc_score(yv, pv)),
        **base.baseline_metrics(job['run']),
        **base.eval_full(clf, test_time, test_gt, test_mu, sigma, test_scores, test_ranks),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f'{predictor}_summary.json').write_text(json.dumps(row, indent=2), encoding='utf-8')
    print(row, flush=True)
    return row


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows = []
    partial = OUT_ROOT / 'summary_partial.csv'
    for job in JOBS:
        for predictor in PREDICTORS:
            rows.append(run_one(job, predictor))
            pd.DataFrame(rows).to_csv(partial, index=False)
    df = pd.DataFrame(rows)
    df.to_csv(OUT_ROOT / 'summary.csv', index=False)
    best = df.sort_values(['detector', 'family', 'rerank_r@1'], ascending=[True, True, False]).groupby(['detector', 'data_mode', 'family']).head(1)
    best.to_csv(OUT_ROOT / 'best_by_group.csv', index=False)
    print('BEST_BY_GROUP')
    print(best[['detector','data_mode','family','predictor','sky_val_mean_error_rad','rerank_r@1','rerank_r@5','rerank_r@10','rerank_r@50']].to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
