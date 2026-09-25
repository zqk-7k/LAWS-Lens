from __future__ import annotations

import json
from pathlib import Path
import pandas as pd

from matchgw.config import MatchRunConfig
from matchgw.pipeline import run_train_eval

OUT_ROOT = Path('runs/pm_mass_1e4_1e10_waveform_baseline_20260608')
DATA_ROOTS = {
    'ET': Path('data_generation/pm_mass_1e4_1e10_matchroots/ET'),
    'LIGO': Path('data_generation/pm_mass_1e4_1e10_matchroots/LIGO'),
}

JOBS = [
    ('ET', 'pure'),
    ('ET', 'noisy'),
    ('LIGO', 'pure'),
    ('LIGO', 'noisy'),
]


def make_cfg(detector: str, mode: str) -> MatchRunConfig:
    # 使用新生成的 PM 质量范围 1e4-1e10 数据。
    # 先跑 waveform-only baseline，不加入 sky-map / trigger_time / catalog rerank 辅助参数。
    is_ligo = detector == 'LIGO'
    return MatchRunConfig(
        data_root=DATA_ROOTS[detector],
        model_type='PM',
        data_mode=mode,
        out_dir=OUT_ROOT / f'{detector.lower()}_{mode}_pm_inceptiontime_bandpass_ep20',
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
        candidate_topk=10,
        export_candidates=True,
        tune_for='f1',
    )


def flatten_result(detector: str, mode: str, result: dict) -> dict:
    test = result.get('test', {})
    val = result.get('val', {})
    sizes = result.get('sizes', {})
    timing = result.get('timing', {})
    return {
        'detector': detector,
        'data_mode': mode,
        'family': 'PM',
        'data_range': 'pm_lens_mass_1e4_1e10_msun',
        'method': 'waveform_only_inceptiontime_bandpass_ep20',
        'test_r@1': test.get('r@1'),
        'test_r@5': test.get('r@5'),
        'test_r@10': test.get('r@10'),
        'test_mrr': test.get('mrr'),
        'test_median_true_rank': test.get('median_true_rank'),
        'test_precision': test.get('precision'),
        'test_recall': test.get('recall'),
        'test_f1': test.get('f1'),
        'test_pairs': test.get('pairs'),
        'val_r@1': val.get('r@1'),
        'val_r@5': val.get('r@5'),
        'val_r@10': val.get('r@10'),
        'train_lensed': sizes.get('train_lensed'),
        'val_lensed': sizes.get('val_lensed'),
        'test_lensed': sizes.get('test_lensed'),
        'train_s': timing.get('train_s'),
        'mean_epoch_s': timing.get('mean_epoch_s'),
    }


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows = []
    for detector, mode in JOBS:
        cfg = make_cfg(detector, mode)
        print('RUN_JOB', detector, mode, cfg.out_dir, flush=True)
        result = run_train_eval(cfg, cpu=False)
        row = flatten_result(detector, mode, result)
        rows.append(row)
        pd.DataFrame(rows).to_csv(OUT_ROOT / 'summary_partial.csv', index=False)
        print('ROW', json.dumps(row, indent=2), flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT_ROOT / 'summary.csv', index=False)
    print(df.to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
