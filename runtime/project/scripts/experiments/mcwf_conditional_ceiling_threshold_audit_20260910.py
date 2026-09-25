#!/usr/bin/env python3
"""Read-only postfreeze F90/score-tie mechanism audit; no alternative scores."""
import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd

P = Path('/root/autodl-tmp/gw-catalog')
sys.path.insert(0, str(P / 'scripts/experiments'))
import mcwf_conditional_ceiling_catalog_20260910 as app


def main(root):
    output = root / 'tables/F90_THRESHOLD_AND_TIE_AUDIT.csv'
    if output.exists():
        raise RuntimeError('Do not overwrite threshold audit')
    final = json.loads((root / 'audit/FINAL_GOAL_READOUT.json').read_text())
    rows = []
    for failure in final['injection_failed_panels']:
        dep, seed, panel, mode = [failure[k] for k in ('deployment', 'seed', 'panel', 'mode')]
        for method in app.METHODS:
            file = root / f'results/{method}/{dep}/seed_{seed}/{panel}/pairs.parquet'
            f = pd.read_parquet(file)
            true = f.is_true_pair.to_numpy(bool)
            score = f.waveform_score.to_numpy(float) if mode == 'waveform' else f.final_score.to_numpy(float)
            order = np.argsort(-score, kind='stable')
            target = int(np.ceil(.9 * true.sum()))
            hit = int(np.flatnonzero(np.cumsum(true[order]) >= target)[0])
            threshold = score[order[hit]]
            tie, above = score == threshold, score > threshold
            zero = score == 0.
            rows.append({'deployment': dep, 'seed': seed, 'panel': panel, 'mode': mode, 'method': method,
                'true_pairs': int(true.sum()), 'target_true': target, 'threshold_at_90pct': threshold,
                'stable_index_F90': int((~true[order[:hit + 1]]).sum()),
                'strictly_above_FP': int((above & ~true).sum()),
                'full_tie_FP': int(((above | tie) & ~true).sum()),
                'tie_true': int((tie & true).sum()), 'tie_false': int((tie & ~true).sum()),
                'exact_zero_true': int((zero & true).sum()), 'exact_zero_false': int((zero & ~true).sum()),
                'source_sha256': app.io.sha(file)})
    app.io.csv(output, rows)
    text = '# R69 F90 阈值与同分审计\n\n本分析在所有评分冻结后进行，只解释失败，不调整分数或参数。\n\n'
    text += pd.DataFrame(rows).to_markdown(index=False, floatfmt='.6g') + '\n\n'
    text += 'stable-index F90保留既有顺序规则；full-tie FP计入同一阈值全部假对。两者差异反映并列处理敏感性，不是新检测。\n'
    with (root / 'reports/R69_F90_THRESHOLD_AUDIT_CN.md').open('x', encoding='utf-8') as stream:
        stream.write(text)
    app.io.write(root / 'audit/F90_THRESHOLD_AUDIT_COMPLETE.json', {'UTC': app.io.utc(), 'rows': len(rows),
        'score_changed': False, 'postfreeze_development_diagnostic': True,
        'script_sha256': app.io.sha(Path(__file__))})
    shutil.copy2(__file__, root / 'scripts/conditional_ceiling_threshold_audit.py')
    print(pd.DataFrame(rows).drop(columns='source_sha256').to_string(index=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    main(parser.parse_args().root)
