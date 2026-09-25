from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

DATA_ROOT = '/root/autodl-tmp/gw_ligo_10000_matchstyle_20260527_091859'
OUT_ROOT = Path('runs/ligo_noisy_waveform_methods_20260601')
OUT_ROOT.mkdir(parents=True, exist_ok=True)

METHODS = [
    dict(name='base_bp40_580', backbone='inceptiontime', preprocess='bandpass', low='40', high='580', extra=[]),
    dict(name='hilbert_bp40_580', backbone='inceptiontime', preprocess='bandpass', low='40', high='580', extra=['--use-hilbert']),
    dict(name='hardneg_bp40_580', backbone='inceptiontime', preprocess='bandpass', low='40', high='580', extra=['--enable-hard-neg', '--hard-neg-epochs', '3', '--hard-neg-min-score', '0.45']),
    dict(name='attnresnet_bp40_580', backbone='attnresnet', preprocess='bandpass', low='40', high='580', extra=[]),
    dict(name='cbamresnet_bp40_580', backbone='cbamresnet', preprocess='bandpass', low='40', high='580', extra=[]),
    dict(name='seresnet_bp40_580', backbone='seresnet', preprocess='bandpass', low='40', high='580', extra=[]),
]


def run_one(method: dict, family: str) -> dict:
    out = OUT_ROOT / f"{family}_{method['name']}_n5000_ep25"
    cmd = [
        sys.executable, 'scripts/08_match_first_train.py',
        '--data-root', DATA_ROOT,
        '--model-type', family,
        '--data-mode', 'noisy',
        '--out-dir', str(out),
        '--backbone', method['backbone'],
        '--preprocess', method['preprocess'],
        '--bandpass-low', method['low'],
        '--bandpass-high', method['high'],
        '--lensed-limit', '5000',
        '--unlensed-limit', '5000',
        '--epochs', '25',
        '--batch-size', '128',
        '--eval-batch-size', '512',
        '--num-workers', '2',
        '--pin-memory',
        '--amp',
        '--candidate-topk', '50',
    ] + method['extra']
    print('RUN', family, method['name'], ' '.join(cmd), flush=True)
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (out / 'driver.log').write_text(proc.stdout, encoding='utf-8')
    row = {'family': family, 'method': method['name'], 'returncode': proc.returncode, 'out_dir': str(out)}
    summary_path = out / 'summary.json'
    if summary_path.exists():
        s = json.loads(summary_path.read_text(encoding='utf-8'))
        test = s.get('test', {})
        row.update({
            'r@1': test.get('r@1'),
            'r@5': test.get('r@5'),
            'r@10': test.get('r@10'),
            'candidate_pair_recall': test.get('candidate_pair_recall'),
            'median_rank': test.get('median_true_rank'),
            'mean_epoch_s': s.get('train', {}).get('mean_epoch_s'),
        })
    else:
        row['error_tail'] = proc.stdout[-1000:]
    print('RESULT', row, flush=True)
    return row


def main():
    rows = []
    for family in ['SIS', 'PM']:
        for method in METHODS:
            rows.append(run_one(method, family))
            pd.DataFrame(rows).to_csv(OUT_ROOT / 'summary_partial.csv', index=False)
    pd.DataFrame(rows).to_csv(OUT_ROOT / 'summary.csv', index=False)
    print(pd.DataFrame(rows).to_string(index=False), flush=True)

if __name__ == '__main__':
    main()
