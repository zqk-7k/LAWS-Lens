from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

DATA_ROOT = Path('/root/autodl-tmp/gw_ligo_10000_matchstyle_20260527_091859')
OUT_ROOT = Path('runs/ligo_noisy_preproc_sweep_20260601')
FAMILIES = ['SIS', 'PM']
BASE = [
    ('none', ['--preprocess', 'none']),
    ('bandpass_20_512', ['--preprocess', 'bandpass', '--bandpass-low', '20', '--bandpass-high', '512']),
    ('bandpass_30_512', ['--preprocess', 'bandpass', '--bandpass-low', '30', '--bandpass-high', '512']),
    ('bandpass_40_580', ['--preprocess', 'bandpass', '--bandpass-low', '40', '--bandpass-high', '580']),
    ('whiten', ['--preprocess', 'whiten', '--whiten-kernel', '65']),
    ('whiten_bandpass_20_512', ['--preprocess', 'whiten_bandpass', '--bandpass-low', '20', '--bandpass-high', '512', '--whiten-kernel', '65']),
    ('whiten_bandpass_30_512', ['--preprocess', 'whiten_bandpass', '--bandpass-low', '30', '--bandpass-high', '512', '--whiten-kernel', '65']),
    ('multiband', ['--preprocess', 'multiband', '--bandpass-low', '20', '--bandpass-high', '512']),
    ('bandpass_pureaux', ['--preprocess', 'bandpass', '--bandpass-low', '40', '--bandpass-high', '580', '--use-pure-aux']),
    ('whiten_bandpass_pureaux', ['--preprocess', 'whiten_bandpass', '--bandpass-low', '20', '--bandpass-high', '512', '--whiten-kernel', '65', '--use-pure-aux']),
]


def run_one(family: str, name: str, extra: list[str]) -> dict:
    out = OUT_ROOT / f'{family}_noisy_{name}_n2500_ep12'
    cmd = [
        sys.executable, 'scripts/08_match_first_train.py',
        '--data-root', str(DATA_ROOT),
        '--model-type', family,
        '--data-mode', 'noisy',
        '--out-dir', str(out),
        '--backbone', 'inceptiontime',
        '--lensed-limit', '2500',
        '--unlensed-limit', '2500',
        '--epochs', '12',
        '--batch-size', '128',
        '--eval-batch-size', '512',
        '--target-len', '8192',
        '--stride', '2',
        '--amp',
        '--pin-memory',
        '--num-workers', '4',
    ] + extra
    print('RUN', family, name, flush=True)
    subprocess.run(cmd, check=True)
    d = json.loads((out / 'summary.json').read_text(encoding='utf-8'))
    row = {
        'family': family,
        'name': name,
        'out_dir': str(out),
        'r@1': d['test']['r@1'],
        'r@5': d['test']['r@5'],
        'r@10': d['test']['r@10'],
        'mrr': d['test']['mrr'],
        'median_rank': d['test']['median_true_rank'],
        'candidate_pair_recall': d.get('test_candidates', {}).get('candidate_pair_recall'),
        'mean_epoch_s': d['timing']['mean_epoch_s'],
        'total_s': d['timing']['total_s'],
    }
    print('RESULT', row, flush=True)
    return row


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows = []
    for family in FAMILIES:
        for name, extra in BASE:
            rows.append(run_one(family, name, extra))
            pd.DataFrame(rows).to_csv(OUT_ROOT / 'summary_partial.csv', index=False)
    df = pd.DataFrame(rows)
    df.to_csv(OUT_ROOT / 'summary.csv', index=False)
    print(df.sort_values(['family', 'r@1'], ascending=[True, False]).to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
