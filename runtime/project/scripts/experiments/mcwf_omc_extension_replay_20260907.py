#!/usr/bin/env python3
"""Replay saved scores without torch and prepare an auditable negative delivery."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def csv(path, frame):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding='utf-8-sig')


def replay(frame, spec, arm):
    if spec.get('unchanged_baseline'):
        return frame.OMC_baseline_waveform_score.to_numpy(float)
    bc = frame.new_waveform_bc.to_numpy(float)
    ood = frame.new_waveform_ood.to_numpy(bool)
    ref = np.sort(np.asarray(spec['mass_reference'], float))
    probability = (1 + len(ref) - np.searchsorted(ref, -np.log(bc), side='left')) / (len(ref) + 1.)
    penalty = np.where(ood, 0., np.minimum(np.log(probability / .05), 0.))
    feature = 'cosine' if arm == 'SOURCE-COSINE' else 'bc' if arm.endswith('-BC') else 'prior_overlap'
    values = frame['new_waveform_' + feature].to_numpy(float)
    cal = spec[feature]
    inc = np.interp(values, cal['knots'], cal['loglr'])
    unsupported = ood | (values < cal['minimum']) | (values > cal['maximum']) | (probability < .05)
    inc = np.where(unsupported, np.minimum(inc, 0.), inc)
    inc = np.where(ood, 0., np.clip(inc, -4., 4.))
    mode = spec.get('score_mode', 'FRT_BASE')
    if mode == 'ADD_POSITIVE':
        return frame.OMC_baseline_waveform_score.to_numpy(float) + spec['beta'] * np.maximum(inc, 0.)
    base = frame.previous_waveform_score if mode == 'REPLACE_FRT' else frame.FRT_baseline_waveform_score
    return base.to_numpy(float) + spec['gamma'] * penalty + spec['beta'] * inc


def snapshot(root, source_dir):
    target = root / 'scripts/current'
    target.mkdir(parents=True, exist_ok=True)
    entries = [
        'mcwf_omc_ensemble_extension_20260907.py',
        'mcwf_omc_architecture_extension_20260907.py',
        'mcwf_omc_single_extension_20260907.py',
        'mcwf_omc_score_controls_20260907.py',
        'mcwf_omc_source_consistency_20260907.py',
        'mcwf_omc_extension_results_20260907.py',
        'mcwf_omc_extension_run_logged_20260907.py',
        Path(__file__).name,
    ]
    pending, done, records = entries[:], set(), []
    while pending:
        name = pending.pop()
        if name in done:
            continue
        done.add(name)
        src = source_dir / name
        if not src.exists():
            raise FileNotFoundError(src)
        dest = target / name
        if dest.exists() and digest(dest) != digest(src):
            archive = root / 'scripts/exporter_history' / f'{dest.stem}_{digest(dest)[:12]}.py'
            archive.parent.mkdir(parents=True, exist_ok=True)
            if not archive.exists():
                shutil.copy2(dest, archive)
        shutil.copy2(src, dest)
        records.append({'source': str(src), 'snapshot': str((target / name).relative_to(root)),
                        'sha256': digest(src), 'role': 'entrypoint' if name in entries else 'local_import_dependency'})
        tree = ast.parse(src.read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            names = [n.name for n in node.names] if isinstance(node, ast.Import) else [node.module] if isinstance(node, ast.ImportFrom) else []
            for module in names:
                if not module:
                    continue
                filename = module.split('.')[0] + '.py'
                if (source_dir / filename).is_file() and filename not in done:
                    pending.append(filename)
    csv(root / 'manifest/CODE_SNAPSHOT.csv', pd.DataFrame(records))
    return len(records)


def run(root, source_dir):
    checks, contrasts, runtime = [], [], []
    for path in sorted(root.rglob('diagnostic_export/CHOICES.json')):
        export = path.parent
        arm = export.parent.name
        for choice in json.loads(path.read_text()):
            for seed, spec in choice['configs'].items():
                for split in ('validation', 'test', 'real'):
                    p = export / f"evaluation/{choice['deployment']}/seed_{seed}/{split}_pairs.parquet"
                    f = pd.read_parquet(p)
                    z = replay(f, spec, arm)
                    delta = float(np.max(np.abs(z - f.waveform_score.to_numpy(float))))
                    assert delta <= 1e-12, (str(p), delta)
                    checks.append({'path': str(p.relative_to(root)), 'pairs': len(f), 'max_abs_score_error': delta})
    for path in sorted(root.rglob('tables/ALL_COMBINATIONS.csv')):
        f = pd.read_csv(path)
        name = str(path.parent.parent.relative_to(root)).replace('/trials/', ':').removeprefix('trials/')
        for dep, g in f.groupby('deployment'):
            allowed = g[g.noninferior]
            contrasts.append({'experiment': name, 'deployment': dep, 'combinations': len(g),
                              'both_official_counts_increase': int(((g.front_gain > 0) & (g.hanabi_gain > 0)).sum()),
                              'fully_noninferior': len(allowed), 'target_pass': int(g.target_pass.sum()),
                              'noninferior_max_BC_count_gain': float(allowed.BC_gain.max()),
                              'noninferior_max_frontend_count_gain': float(allowed.front_gain.max()),
                              'noninferior_max_Hanabi_count_gain': float(allowed.hanabi_gain.max())})
    csv(root / 'audit/INDEPENDENT_SCORE_REPLAY.csv', pd.DataFrame(checks))
    csv(root / 'tables/DEVELOPMENT_TRADEOFF_COUNTS.csv', pd.DataFrame(contrasts))
    for path in sorted(root.glob('logs/*.runtime.json')):
        r = json.loads(path.read_text())
        runtime.append({'log': str(path.relative_to(root)), **{k: v for k, v in r.items() if k != 'command'},
                        'command': json.dumps(r['command'])})
    csv(root / 'tables/RESOURCE_USAGE.csv', pd.DataFrame(runtime))
    f = pd.read_csv(root / 'tables/ALL_DIAGNOSTIC_RETRIEVAL.csv')
    keys = ['experiment', 'deployment', 'split', 'method', 'config']
    metrics = [c for c in ('macro_r_at_1', 'macro_r_at_10', 'average_precision',
                          'false_at_recall_0p5', 'false_at_recall_0p9') if c in f]
    summary = f.groupby(keys)[metrics].agg(['mean', 'std'])
    summary.columns = ['_'.join(c) for c in summary.columns]
    csv(root / 'tables/REUSED_RETRIEVAL_MEAN_SD.csv', summary.reset_index())
    count = snapshot(root, source_dir)
    summary = {'independent_formula_replayed_tables': len(checks),
               'max_abs_score_error': max(r['max_abs_score_error'] for r in checks),
               'new_entrypoints_and_local_dependencies_snapshotted': count,
               'runtime_measurements': len(runtime),
               'runtime_limitations': 'First original-OMC mixture-prior grid has no durable runtime measurement. Concurrent wall times must not be summed as elapsed time.',
               'scope': 'Saved-feature score replay; not independent model calibration or fresh-data validation.'}
    (root / 'audit/INDEPENDENT_REPLAY_SUMMARY.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    (root / 'README_CN.md').write_text('''# OMC-EXT18：独立波形探索交付

本目录不替代 MCWF-UNIFIED-OMC-DEVCONF。先读 `reports/FINAL_EXPLORATION_REPORT_CN.md`。

- `tables/EXPERIMENT_LEDGER.csv`：每个对照、每个运行期的目标通过情况。
- `tables/BASELINE_VS_DIAGNOSTICS.csv`：真实 PE 和官方预算的新旧比较。
- `tables/REUSED_RETRIEVAL_MEAN_SD.csv`：复用模拟数据上的 recall、AUPRC、F50/F90，三 seed 均值与 SD。
- `tables/ALL_DIAGNOSTIC_CHOICES.json`：各对照选出的开发性诊断配置，不是全部已获批的新模型。
- 各 `trials/*/diagnostic_export/results/DIAGNOSTIC/`：逐 seed 排名、共识全部 pair Parquet 和 Top-100 PE/官方 CSV。
- 各 `trials/*/tables/ALL_COMBINATIONS.csv`：所有通过模拟 guard 的组合，包括未通过真实开发目标者。
- 各 `trials/*/tables/VALIDATION_GRID.csv`：包括模拟 guard 失败的单 seed 配置。
- `tables/MODEL_TRAINING_SUMMARY.csv`：24 个新模型的训练记录。
- `tables/RESOURCE_USAGE.csv`：实际日志中的 CPU、内存和用时；没有测量的任务不补造数字。
- `audit/` 和 `manifest/`：独立分数复算、时间/天空不变、旧文件与包内 SHA-256 核验。

## 复算与依赖

解包后可使用 NumPy/Pandas/PyArrow 从已保存的 pair 特征、校准和配置独立复算波形分数，无需 GPU：

```bash
python scripts/current/mcwf_omc_extension_replay_20260907.py --root /path/to/extracted/result --verify-only
```

`scripts/current/` 保存本轮最终可运行脚本和可发现的本地静态导入依赖。原初始化快照也保留。
网络重新训练仍需要服务器原有训练数据、噪声、预计算特征及前版基线；本紧凑包不包含这些大型原始数据。
脚本中的绝对 provenance 路径指向服务器项目，不能将本包解释为从零生成全部原始数据的容器。
本轮未执行新的独立 source/noise 确认，不能据此声称独立泛化改善。

最终状态：HOLD_FOR_AUTHOR_REVIEW_NO_ADOPTION_NO_OVERWRITE。
''', encoding='utf-8')
    print(json.dumps(summary), flush=True)


def verify_only(root):
    count, error = 0, 0.
    for path in sorted(root.rglob('diagnostic_export/CHOICES.json')):
        for choice in json.loads(path.read_text()):
            for seed, spec in choice['configs'].items():
                for split in ('validation', 'test', 'real'):
                    p = path.parent / f"evaluation/{choice['deployment']}/seed_{seed}/{split}_pairs.parquet"
                    f = pd.read_parquet(p)
                    delta = float(np.max(np.abs(replay(f, spec, path.parent.parent.name) - f.waveform_score.to_numpy(float))))
                    assert delta <= 1e-12, (str(p), delta)
                    count += 1
                    error = max(error, delta)
    assert count
    print(json.dumps({'verified_tables': count, 'max_abs_score_error': error}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--source-dir', type=Path, default=Path(__file__).parent)
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--refresh-code', action='store_true')
    args = parser.parse_args()
    if args.verify_only:
        verify_only(args.root)
    elif args.refresh_code:
        print(json.dumps({'code_snapshots': snapshot(args.root, args.source_dir)}), flush=True)
    else:
        run(args.root, args.source_dir)
