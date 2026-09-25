from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from matchgw.config import MatchRunConfig
from matchgw.data import EvaluationSet, ground_truth_partner, load_match_arrays, split_indices
from matchgw.matching import retrieval_metrics, similarity_matrix
from matchgw.pipeline import embed_eval, train_encoder

OUT_ROOT = Path('runs/pm_mass_1e4_1e10_fast_retrieval_20260608')
DATA_ROOTS = {
    'ET': Path('data_generation/pm_mass_1e4_1e10_matchroots/ET'),
    'LIGO': Path('data_generation/pm_mass_1e4_1e10_matchroots/LIGO'),
}
JOBS = [('ET', 'pure'), ('ET', 'noisy'), ('LIGO', 'pure'), ('LIGO', 'noisy')]


def make_cfg(detector: str, mode: str) -> MatchRunConfig:
    is_ligo = detector == 'LIGO'
    return MatchRunConfig(
        data_root=DATA_ROOTS[detector],
        model_type='PM',
        data_mode=mode,
        out_dir=OUT_ROOT / f'{detector.lower()}_{mode}_pm_fast_retrieval_ep20',
        backbone='inceptiontime',
        preprocess='bandpass',
        bandpass_low=40,
        bandpass_high=580,
        target_len=8192,
        stride=2,
        lensed_limit=10000,
        unlensed_limit=10000,
        epochs=20,
        batch_size=128 if not is_ligo else 96,
        eval_batch_size=512 if not is_ligo else 256,
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
        amp_dtype='bf16',
        num_workers=2,
        pin_memory=True,
        export_candidates=False,
    )


def eval_split(model, arrays, splits, cfg: MatchRunConfig, split: str) -> dict:
    ds = EvaluationSet(arrays, splits['lensed'][split], splits['unlensed'][split], cfg)
    t0 = time.perf_counter()
    emb = embed_eval(model, ds, cfg, cpu=False)
    scores = similarity_matrix(emb)
    gt = ground_truth_partner(ds.meta)
    met = retrieval_metrics(scores, gt, ks=(1, 5, 10, 50, 100, 500))
    met['valid'] = int(np.sum(gt >= 0))
    met['catalog_size'] = int(len(gt))
    met['eval_s'] = float(time.perf_counter() - t0)
    return met


def run_one(detector: str, mode: str) -> dict:
    cfg = make_cfg(detector, mode)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    print('RUN_FAST', detector, mode, cfg.out_dir, flush=True)
    total_t0 = time.perf_counter()
    model, state, train_info = train_encoder(cfg, cpu=False)
    arrays = state['arrays']
    splits = state['splits']
    val = eval_split(model, arrays, splits, cfg, 'val')
    test = eval_split(model, arrays, splits, cfg, 'test')
    sizes = {
        'lensed_total': int(len(arrays.l1)),
        'unlensed_total': int(len(arrays.unlensed)),
        'train_lensed': int(len(splits['lensed']['train'])),
        'val_lensed': int(len(splits['lensed']['val'])),
        'test_lensed': int(len(splits['lensed']['test'])),
    }
    result = {
        'config': {k: str(v) if isinstance(v, Path) else v for k, v in asdict(cfg).items()},
        'sizes': sizes,
        'history': train_info['history'],
        'timing': {k: v for k, v in train_info.items() if k != 'history'},
        'val': val,
        'test': test,
        'total_s': float(time.perf_counter() - total_t0),
        'qc_note': 'New PM mass range includes 4/10000 lensed systems with t_d < 24s; this run keeps them.',
    }
    with open(cfg.out_dir / 'summary.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    pd.DataFrame(train_info['history']).to_csv(cfg.out_dir / 'history.csv', index=False)
    torch.save({'model': model.state_dict(), 'config': result['config']}, cfg.out_dir / 'model.pt')
    row = {
        'detector': detector,
        'data_mode': mode,
        'family': 'PM',
        'method': 'fast_waveform_only_inceptiontime_bandpass_ep20',
        'test_r@1': test['r@1'],
        'test_r@5': test['r@5'],
        'test_r@10': test['r@10'],
        'test_r@50': test['r@50'],
        'test_r@100': test['r@100'],
        'test_r@500': test['r@500'],
        'test_median_true_rank': test['median_true_rank'],
        'test_mrr': test['mrr'],
        'val_r@10': val['r@10'],
        'train_s': train_info['train_s'],
        'mean_epoch_s': train_info['mean_epoch_s'],
        'total_s': result['total_s'],
    }
    print('FAST_ROW', json.dumps(row, indent=2), flush=True)
    return row


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows = []
    for detector, mode in JOBS:
        rows.append(run_one(detector, mode))
        pd.DataFrame(rows).to_csv(OUT_ROOT / 'summary_partial.csv', index=False)
    df = pd.DataFrame(rows)
    df.to_csv(OUT_ROOT / 'summary.csv', index=False)
    print(df.to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
