#!/usr/bin/env python3
"""Read-only R75 checks, Chinese report and stream-verified compact archive."""
import os
os.environ['MPLBACKEND'] = 'Agg'
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tarfile

import numpy as np
import pandas as pd

P = Path('/root/autodl-tmp/gw-catalog')
sys.path.insert(0, str(P / 'scripts/experiments'))
import mcwf_shared_mass_catalog_scoring_20260910 as app


def json_write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(obj, stream, indent=2, ensure_ascii=False, default=app.io.plain, allow_nan=False)


def audit(root):
    app.ROOT = root
    checked = app.check()
    if (root / 'contracts/FINAL_AUDIT.json').exists():
        raise RuntimeError('Completed audit immutable')
    inj = json.loads((root / 'contracts/INJECTION_GUARD.json').read_text())
    checks, changes = [], []
    for file in sorted((root / f'results/{app.NEW}').glob('gwtc*/seed_*/*/pairs.parquet')):
        rel = file.relative_to(root / f'results/{app.NEW}')
        new = app.audit.aligned(file)
        old = app.audit.aligned(root / 'results' / app.METHODS[0] / rel)
        for col in ('idx_i', 'idx_j', 'time_score', 'sky_raw_log_bf', 'embedding_only', 'time_contribution', 'sky_contribution'):
            if not new[col].equals(old[col]):
                raise RuntimeError('Frozen channel changed: ' + col)
        active = new.r75_eligible.to_numpy(bool)
        if not np.array_equal(new.waveform_score[~active], old.waveform_score[~active]):
            raise RuntimeError('Inactive NODUP changed')
        dep, seedstr, panel = rel.parts[:3]
        cfg = next(c for c in app.n.selections(root) if c['method'] == app.NEW and c['deployment'] == dep and c['seed'] == int(seedstr[5:]))
        total = app.n.cf.channels(new, new.waveform_score.to_numpy(float)) @ np.asarray(cfg['weights'])
        error = float(abs(total - new.final_score.to_numpy()).max())
        if error > 1e-12:
            raise RuntimeError('Fusion formula changed')
        mask = new.waveform_score.to_numpy() != old.waveform_score.to_numpy()
        checks.append({'deployment': dep, 'seed': int(seedstr[5:]), 'panel': panel, 'pairs': len(new),
            'eligible': int(active.sum()), 'changed': int(mask.sum()), 'time_sky_exact': True,
            'inactive_exact': True, 'total_error': error})
        keys = ['idx_i', 'idx_j', 'pair_key', 'is_true_pair', 'true_pair_family', 'r75_original_power',
                'r75_D_mass', 'r75_D_full_common_denominator', 'r75_best_mass_converged', 'r75_support_ood']
        for pos in np.flatnonzero(mask):
            changes.append({**new.iloc[pos][[k for k in keys if k in new]].to_dict(), 'deployment': dep,
                'seed': int(seedstr[5:]), 'panel': panel, 'old_waveform': float(old.waveform_score.iloc[pos]),
                'new_waveform': float(new.waveform_score.iloc[pos])})
    if len(checks) != 30:
        raise RuntimeError('Missing injection panels')
    app.io.csv(root / 'audit/FROZEN_CHANNEL_AND_FORMULA_CHECK.csv', checks)
    app.io.csv(root / 'tables/CHANGED_INJECTION_PAIRS.csv', changes)
    real_done = (root / 'contracts/REAL_COMPLETE.json').exists()
    pe_rows, keys = [], []
    if real_done:
        tables = []
        for filename, unit in [('PE_OFFICIAL_BUDGETS.csv', 'consensus'), ('PER_SEED_PE_OFFICIAL_BUDGETS.csv', 'model')]:
            f = pd.read_csv(root / 'tables' / filename)
            f['unit'] = unit
            if unit == 'consensus':
                f['seed'] = 'consensus'
            tables.append(f)
        budget = pd.concat(tables, ignore_index=True)
        for row in budget[(budget.config == app.NEW) & budget.budget.isin([10, 20])].to_dict('records'):
            ref = budget[(budget.config == app.METHODS[0]) & (budget.deployment == row['deployment']) &
                (budget.unit == row['unit']) & (budget.seed.astype(str) == str(row['seed'])) & (budget.budget == row['budget'])]
            if len(ref) != 1:
                raise RuntimeError('Ambiguous PE budget reference')
            ref = ref.iloc[0]
            tests = {k: row[k] >= ref[k] - 1e-12 for k in app.audit.HIGHER}
            tests['catastrophic_mc'] = row['catastrophic_mc'] <= ref.catastrophic_mc
            pe_rows.append({**{k: row[k] for k in ('deployment', 'unit', 'seed', 'budget')}, 'pass': all(tests.values()),
                **{k + '_pass': bool(v) for k, v in tests.items()},
                **{k + '_delta': float(row[k] - ref[k]) for k in (*app.audit.HIGHER, 'catastrophic_mc')}})
        for method in app.METHODS:
            for seed in (*app.n.SEEDS, 'consensus'):
                path = (root / f'results/{method}/gwtc3/consensus/fusion_all_pairs.parquet') if seed == 'consensus' else app.result_path(method, 'gwtc3', seed, 'real')
                f = pd.read_parquet(path)
                row = f[f.pair_key == app.audit.KEY]
                if len(row) != 1:
                    raise RuntimeError('Missing key pair')
                keys.append({**row.iloc[0].to_dict(), 'method': method, 'model': str(seed),
                    'key_rank': int(row.iloc[0]['consensus_rank' if seed == 'consensus' else 'rank'])})
        app.io.csv(root / 'tables/NODUP_PE_OFFICIAL_GUARDS.csv', pe_rows)
        app.io.csv(root / 'tables/CRITICAL_PAIR_ALL_RANKS.csv', keys)
    external_pass = real_done and len(pe_rows) == 16 and all(r['pass'] for r in pe_rows)
    key_pass = real_done and all(r['key_rank'] > 10 for r in keys if r['method'] == app.NEW)
    goal = bool(inj['gate'] == 'PASS' and external_pass and key_pass)
    result = {'UTC': app.io.utc(), 'status': app.n.STATUS, 'goal_achieved': goal,
        'injection_guard': inj['gate'], 'injection_failures': inj['failures'], 'real_audit_completed': real_done,
        'PE_official_all_consensus_and_models_pass': bool(external_pass), 'key_outside_all_Top10': bool(key_pass),
        'verified_inputs': checked, 'same_method_both_runs': True,
        'interpretation': 'Adaptive development, not independent confirmation; official overlap is not a lensed label.'}
    json_write(root / 'contracts/FINAL_AUDIT.json', result)
    print('R75_FINAL_AUDIT', json.dumps(result), flush=True)


def report(root):
    import matplotlib.pyplot as plt
    stamp = json.loads((root / 'contracts/FINAL_AUDIT.json').read_text())
    output = root / 'reports/R72_R75_FULL_REPORT_CN.md'
    if output.exists():
        raise RuntimeError('Report immutable')
    metrics = pd.read_csv(root / 'tables/RETRIEVAL_PER_MODEL.csv')
    metrics = metrics[(metrics.split == 'sept8_reused') & metrics.method.isin(app.METHODS)]
    columns = ['macro_r_at_1', 'macro_r_at_10', 'average_precision', 'false_at_recall_0p5', 'false_at_recall_0p9']
    summary = metrics.groupby(['deployment', 'method', 'mode'])[columns].agg(['mean', 'std']).reset_index()
    summary.columns = ['_'.join(c).rstrip('_') for c in summary.columns]
    app.io.csv(root / 'tables/R75_PRIMARY_RETRIEVAL_SUMMARY.csv', summary.to_dict('records'))
    text = '# R72-R75 完整波形兼容性对照\n\n'
    text += '**本轮为自适应开发，不能宣称新的独立确认，也不替换任何历史结果。**\n\n'
    text += '```json\n' + json.dumps(stamp, ensure_ascii=False, indent=2) + '\n```\n\n'
    text += '## 方法与选择依据\n\nR72用冻结算子测量1053个模拟源/假对。R73只在source/noise-parent隔离的development两折上选择单一条件波形分类器，比较共享Mc与完整共享参数的数值对照；两运行期统一选择完整共享数值对照。真实PE和官方候选没有进入该校准选择。\n\n'
    text += 'R74重新测量5241个注入catalog pair，不复用旧deficit。R75显式使用原始功率与原始共享拟合收敛标记，加上新D和新best-mass收敛标记，完全复现R73输入定义。R74兼容字段的“新功率”不能直接送入旧功率校准；本次已通过字段投毒与回放测试。\n\n'
    text += '只有waveform score在冻结支持域内更新。encoder、学习到的内禀参数分布、time、sky、外层权重、scope和旧文件均不变；不恢复旧Mc/q分数或0.875总分混合。这里的投影差不是PE、规范化likelihood或透镜Bayes factor。\n\n'
    text += '## 注入结果\n\n每个模型先平均三个共用catalog，再计算三模型mean/SD。不能只看均值掩盖单个panel失败。\n\n'
    text += summary.to_markdown(index=False, floatfmt='.6g') + '\n\n'
    guards = pd.read_csv(root / 'tables/NODUP_INJECTION_GUARDS.csv')
    bad = guards[~guards['pass']]
    text += '## 逐panel门槛\n\n' + (bad.to_markdown(index=False, floatfmt='.6g') if len(bad) else '全部60项通过。') + '\n\n'
    if stamp['real_audit_completed']:
        budgets = pd.read_csv(root / 'tables/PE_OFFICIAL_BUDGETS.csv')
        budgets = budgets[budgets.config.isin(app.METHODS) & budgets.budget.isin([10, 20])]
        text += '## 真实PE与公开阶段\n\n' + budgets.to_markdown(index=False, floatfmt='.6g') + '\n\n'
        text += '逐模型结果见PER_SEED_PE_OFFICIAL_BUDGETS.csv，完整Top-10/20/50/100及全pair见results。公开Hanabi字段是既有公开表重合，不是本轮运行Hanabi，也不代表透镜真值。\n\n'
    else:
        text += '## 真实PE与公开阶段\n\n注入门槛未通过，因此本轮未运行新的真实排序。不能把旧PE/官方结果改名为本轮结果；保留此前失败与参考表供provenance审计。\n\n'
    text += '## 不确定度与局限\n\n系统级bootstrap保留同源两像共同抽样，query10000次、pair2000次，比较方法使用相同draw。区间条件于固定噪声、已反复使用的catalog和已选模型，不是全流程无偏确认区间。源/噪声人口推广、近似aligned-spin波形和独立单像SNR缩放局限仍存在。\n\n'
    text += '## 文献边界\n\n- Cutler & Flanagan: https://arxiv.org/abs/gr-qc/9402014 ，质量信息与质量-自旋相关性是物理动机，不证明当前投影或网络最优。\n- Cranmer等: https://arxiv.org/abs/1506.02169 ，联合判别校准避免把相关量机械当独立证据，不保证有限样本分类器是严格Bayes factor。\n- Lo & Magana Hernandez: https://arxiv.org/abs/2104.09339 ，完整透镜确认须population/selection与联合模型，本轮没有运行Hanabi。\n- Cawley & Talbot: https://www.jmlr.org/papers/v11/cawley10a.html ，反复查看验证/真实外部结果属于自适应开发。\n\nHOLD_FOR_AUTHOR_REVIEW_NO_ADOPTION_NO_OVERWRITE\n'
    with output.open('x', encoding='utf-8') as stream:
        stream.write(text)
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'], 'pdf.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axs = plt.subplots(2, 3, figsize=(11, 6.4), layout='constrained')
    for row, dep in enumerate(app.n.DEPS):
        for ax, key, title in zip(axs[row], ['macro_r_at_10', 'average_precision', 'false_at_recall_0p9'], ['R@10', 'Pair AUPRC', 'False pairs at 90% recall']):
            for col, mode in enumerate(('waveform', 'fusion')):
                for m, method in enumerate(app.METHODS):
                    g = metrics[(metrics.deployment == dep) & (metrics.method == method) & (metrics['mode'] == mode)].sort_values('seed')
                    x = col + (m - .5) * .24
                    ax.scatter(np.full(len(g), x), g[key], marker='o' if m else 's', s=28, color='#16806b' if m else '#6d7785', label=('R75' if m else 'NODUP') if col == 0 else None)
            ax.set_xticks([0, 1], ['Waveform', 'Fusion'])
            ax.set_title(('O3' if dep == 'gwtc3' else 'O4a') + ': ' + title, fontsize=10)
            ax.grid(alpha=.15, axis='y')
    axs[0, 0].legend(frameon=False, fontsize=8)
    for ext in ('pdf', 'png'):
        fig.savefig(root / f'figures/R75_RETRIEVAL_INDIVIDUAL_MODELS.{ext}', dpi=180)
    plt.close(fig)
    shutil.copy2(__file__, root / 'scripts/shared_mass_catalog_delivery.py')


def package(root, output):
    if output.exists() or not (root / 'contracts/FINAL_AUDIT.json').exists():
        raise RuntimeError('Independent output and completed audit required')
    roots = [app.PILOT, app.PHYSICAL, root]
    items = []
    forbidden = ('id_rsa', 'private_key', '.pem', 'credentials', '.ssh', 'password')
    for source in roots:
        for file in sorted(source.rglob('*')):
            if not file.is_file() or file.is_symlink() or any(x in file.name.lower() for x in forbidden):
                continue
            if file.suffix in ('.npy', '.npz', '.hdf5', '.h5', '.pt', '.pth', '.tar', '.gz') or '__pycache__' in file.parts:
                continue
            arc = source.name + '/' + str(file.relative_to(source))
            items.append((file, arc, app.io.sha(file)))
    manifest = root / 'manifest/DELIVERY_SHA256SUMS.txt'
    with manifest.open('x', encoding='utf-8') as stream:
        for _, arc, sha in items:
            stream.write(sha + '  ' + arc + '\n')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, 'x:gz') as tar:
        for file, arc, _ in items:
            tar.add(file, arcname=arc, recursive=False)
        tar.add(manifest, arcname=root.name + '/manifest/DELIVERY_SHA256SUMS.txt', recursive=False)
    expected = {arc: digest for _, arc, digest in items}
    seen, failed = set(), []
    with tarfile.open(output, 'r|gz') as tar:
        for member in tar:
            if not member.isfile() or member.name not in expected:
                continue
            h = hashlib.sha256()
            with tar.extractfile(member) as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                    h.update(chunk)
            seen.add(member.name)
            if h.hexdigest() != expected[member.name]:
                failed.append(member.name)
    if failed or seen != set(expected):
        raise RuntimeError('Archive verification failed')
    digest = app.io.sha(output)
    with Path(str(output) + '.sha256').open('x') as stream:
        stream.write(digest + '  ' + output.name + '\n')
    json_write(root / 'contracts/DELIVERY_COMPLETE.json', {'UTC': app.io.utc(), 'package': str(output), 'sha256': digest,
        'bytes': output.stat().st_size, 'verified_members': len(seen), 'failed_members': 0,
        'status': app.n.STATUS, 'goal_achieved': json.loads((root / 'contracts/FINAL_AUDIT.json').read_text())['goal_achieved']})


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--stage', choices=('audit', 'report', 'package'), required=True)
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    app.ROOT = args.root
    if args.stage == 'package':
        package(args.root, args.output)
    else:
        globals()[args.stage](args.root)
