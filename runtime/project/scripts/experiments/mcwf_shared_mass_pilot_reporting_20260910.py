#!/usr/bin/env python3
"""Read-only R70/R71 synthesis, plots, resource receipt and compact bundle."""
import os
os.environ['MPLBACKEND'] = 'Agg'
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import re
import shutil
import sys
import tarfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

P = Path('/root/autodl-tmp/gw-catalog')
sys.path.insert(0, str(P / 'scripts/experiments'))
import mcwf_shared_mass_nuisance_pilot_20260910 as pilot
import mcwf_shared_true_forensic_20260910 as forensic


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for data in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def write(path, value):
    with path.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def report(audit, root):
    forensic.check(audit)
    pilot.check(root)
    gate = json.loads((root / 'contracts/PILOT_NUMERICAL_GATE.json').read_text())
    done = json.loads((root / 'contracts/MEASUREMENTS_COMPLETE.json').read_text())
    dest = root / 'reports/R70_R71_DIAGNOSTIC_SYNTHESIS_CN.md'
    if dest.exists():
        raise RuntimeError('Synthesis already exists')
    f = pd.read_parquet(root / 'tables/PAIR_RESULTS.parquet')
    control = f[f.hash_control]
    tail = f[f.enriched_tail_diagnostic]
    truth = pd.read_csv(audit / 'tables/TRUE_SYSTEM_FORENSIC.csv')
    t = tail.merge(truth[['pair_id', 'true_Mc', 'true_q', 'true_chieff', 'SNR_min', 'SNR_ratio']],
                   on='pair_id', validate='one_to_one')
    t.to_csv(root / 'tables/ENRICHED_TRUE_TAIL_COMPARISON.csv', index=False, encoding='utf-8-sig')
    # These are selected-sample summaries, not population-weighted calibration results.
    stats = []
    for (dep, kind), g in control.groupby(['deployment', 'kind']):
        stats.append({'deployment': dep, 'kind': kind, 'hash_control_pairs': len(g),
            'D_full_median': g.D_full_common_denominator.median(), 'D_mass_median': g.D_mass.median(),
            'nuisance_gain_median': g.nuisance_release_gain.median(),
            'old_vs_control_search_gain_median': g.full_search_gain.median(),
            'best_mass_converged': int(g.best_mass_has_converged_run.sum())})
    pd.DataFrame(stats).to_csv(root / 'tables/HASH_CONTROL_KIND_SUMMARY.csv', index=False, encoding='utf-8-sig')
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'],
                         'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42})
    colors = {'true': '#19826b', 'random_null': '#657080', 'hard_null': '#b43d51'}
    labels = {'true': 'True companion', 'random_null': 'Random null', 'hard_null': 'Hard null'}
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.4), layout='constrained')
    for ax, dep, title in zip(axes, ('gwtc3', 'gwtc4'), ('O3', 'O4a')):
        a = control[control.deployment == dep]
        for kind, g in a.groupby('kind'):
            ax.scatter(g.D_full_common_denominator, g.D_mass, c=colors[kind], s=30,
                       label=labels[kind], alpha=.85)
        g = tail[tail.deployment == dep]
        ax.scatter(g.D_full_common_denominator, g.D_mass, facecolors='none', edgecolors='#202020',
                   marker='s', s=45, linewidths=1., label='Enriched true tail')
        high = float(f.loc[f.deployment == dep, 'D_full_common_denominator'].max()) * 1.1
        ax.plot([0, high], [0, high], '--', color='#777777', linewidth=.8)
        ax.set(xscale='symlog', yscale='symlog', xlim=(0, high), ylim=(0, high),
               xlabel='Full-shared deficit (same denominator)', ylabel='Shared-Mc deficit', title=title)
        ax.grid(alpha=.15)
    handles, legends = axes[0].get_legend_handles_labels()
    fig.legend(handles, legends, loc='outside lower center', ncol=4, frameon=False)
    for ext in ('png', 'pdf'):
        fig.savefig(root / f'figures/R71_NESTED_DEFICIT_CONTROLS.{ext}', dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.4), layout='constrained')
    for ax, dep, title in zip(axes, ('gwtc3', 'gwtc4'), ('O3', 'O4a')):
        g = t[t.deployment == dep].sort_values('pair_id')
        x = np.arange(len(g))
        ax.plot(x, g.D_full_historical, 'o:', color='#999999', label='Historical full-shared')
        ax.plot(x, g.D_full_common_denominator, 's-', color='#b43d51', label='Full-shared control')
        ax.plot(x, g.D_mass, 'o-', color='#19826b', label='Shared Mc')
        ax.set(xlabel='Enriched simulated source (stable pair-ID order)', ylabel='Projection deficit',
               title=title + ': all predeclared tail cases')
        ax.set_xticks(x, [str(k + 1) for k in x])
        ax.grid(alpha=.15)
    handles, legends = axes[0].get_legend_handles_labels()
    fig.legend(handles, legends, loc='outside lower center', ncol=3, frameon=False)
    for ext in ('png', 'pdf'):
        fig.savefig(root / f'figures/R71_ALL_ENRICHED_TRUE_TAIL.{ext}', dpi=180)
    plt.close(fig)
    cpu = Path('/sys/fs/cgroup/cpu.max')
    memory = Path('/sys/fs/cgroup/memory.max')
    environment = {'UTC': datetime.now(timezone.utc).isoformat(), 'python': sys.version,
        'platform': platform.platform(), 'numpy': np.__version__, 'pandas': pd.__version__,
        'visible_logical_CPUs': os.cpu_count(), 'cgroup_cpu_max': cpu.read_text().strip() if cpu.exists() else None,
        'cgroup_memory_max': memory.read_text().strip() if memory.exists() else None,
        'free_disk_bytes_at_reporting': shutil.disk_usage(root).free,
        'pilot_wall_seconds': done['wall_seconds'], 'workers': done['workers'],
        'pair_seconds_P50': float(f.seconds.median()), 'pair_seconds_P90': float(f.seconds.quantile(.9)),
        'pair_seconds_max': float(f.seconds.max()),
        'worker_RSS_KiB_max': int(f.peak_worker_RSS_KiB.max()),
        'no_new_GPU_training': True, 'no_strain_regeneration': True}
    write(root / 'audit/RESOURCE_AND_ENVIRONMENT_RECEIPT.json', environment)
    cols = ['deployment', 'pair_id', 'true_Mc', 'D_full_historical', 'D_full_common_denominator',
            'D_mass', 'nuisance_release_gain', 'best_mass_has_converged_run']
    text = '# R70-R71 Mc兼容性失效机制对照\n\n'
    text += '**目标尚未达成，未改变真实排名。** 本轮只完成模拟只读审计与数值机制检验，不是新的目录性能结果。\n\n'
    text += '## 为什么做\n\nR69已将指定真实pair移出全部Top10，但部分注入F90和逐模型PE/官方重合未通过。因此不能宣布升级。R70检查全部285个已测量真伴随系统，发现高D不一定是Mc错；q、自旋、噪声与近似模型也会造成共同拟合损失。\n\n'
    text += '## 本次统计量\n\n'
    text += '$$D_{\\mathcal M_c}=P_{\\rm independent}-P_{\\rm shared\\,\\mathcal M_c},\\qquad 0\\le D_{\\mathcal M_c}\\le D_{\\rm shared\\,all}.$$\n\n'
    text += '只要求两次数据共享Mc，分别优化q与等效自旋。透镜物理仍要求完整内禀参数共享；这里释放干扰参数是对近似模型的诊断，不是宣称透镜改变质量比或自旋。两个差异量使用同一个已搜索的独立功率分母。另列旧分数与增加搜索带来的收益，防止把数值优化改善误写为物理改善。\n\n'
    text += f'按哈希选出的对照{len(control)}对，另外包含全部预声明困难真对；并集{len(f)}对。困难真对不用于直接拟合密度，以下统计不能解释为总体recall或FPP。\n\n'
    text += '## 数值与范围\n\n```json\n' + json.dumps(gate, indent=2) + '\n```\n\n'
    text += f'最佳共享Mc解未达到已记录收敛解容差的有{int((~f.best_mass_has_converged_run).sum())}/{len(f)}对。该标记不能隐藏；数学嵌套检查通过不等于已经找到全局最优。\n\n'
    text += '## 哈希对照\n\n' + pd.DataFrame(stats).to_markdown(index=False, floatfmt='.6g') + '\n\n'
    text += '## 全部困难真对\n\n' + t[cols].to_markdown(index=False, floatfmt='.6g') + '\n\n'
    text += '## 结论边界与下一步\n\n只说明这一受控样本中释放q/自旋后Mc兼容性如何变化。必须在完整模拟抽样和原source-group/HT权重下校准，再检查两个运行期、每个模型、全部注入面板和真实Top10/20；不能用本轮小样本或单个目标pair决定权重。没有读取真实PE、官方FPP或Hanabi表，没有新recall，没有新真实排名。\n\n'
    text += '参考依据见R71冻结合同：Cutler与Flanagan的质量/自旋相关性、PyCBC共同参数建模，以及Lo与Magana Hernandez的完整透镜模型比较边界。本轮不是完整PE或Bayes factor。\n\n'
    text += '## 资源\n\n```json\n' + json.dumps(environment, indent=2) + '\n```\n\n'
    text += '历史时间、天空、权重、encoder和排名均未改写。HOLD_FOR_AUTHOR_REVIEW_NO_ADOPTION_NO_OVERWRITE。\n'
    with dest.open('x', encoding='utf-8') as handle:
        handle.write(text)
    shutil.copy2(__file__, root / 'scripts/shared_mass_pilot_reporting.py')
    write(root / 'contracts/REPORTING_COMPLETE.json', {'UTC': datetime.now(timezone.utc).isoformat(),
        'status': 'HOLD_FOR_AUTHOR_REVIEW_NO_ADOPTION_NO_OVERWRITE', 'goal_achieved': False,
        'R70_verified_inputs': forensic.check(audit), 'R71_verified_inputs': pilot.check(root),
        'real_or_catalog_rank_changed': False})
    print(str(dest), flush=True)


def package(roots, output):
    stage = output.parent / output.name.removesuffix('.tar.gz')
    if output.exists() or stage.exists():
        raise RuntimeError('Independent package/staging required')
    markers = ('AUDIT_COMPLETE.json', 'PILOT_NUMERICAL_GATE.json', 'MEASUREMENT_GATE.json')
    for root in roots:
        if not any((root / 'contracts' / marker).exists() for marker in markers):
            raise RuntimeError('Missing terminal audit/measurement marker: ' + str(root))
    stage.mkdir(parents=True)
    records, omitted = [], []
    for root in roots:
        for path in sorted(root.rglob('*')):
            if not path.is_file():
                continue
            rel = path.relative_to(root)
            if path.is_symlink() or any(x in ('.ssh', 'id_rsa', 'id_ed25519', '.env') for x in rel.parts):
                raise RuntimeError('Unsafe source path')
            if any(x in ('cache', '__pycache__') for x in rel.parts) or path.suffix.lower() in ('.npz', '.npy', '.h5', '.hdf5', '.pem', '.key', '.pyc'):
                omitted.append(str(path))
                continue
            if path.stat().st_size < 16 * 1024**2 and re.search(rb'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----', path.read_bytes()):
                raise RuntimeError('Private credential content')
            dest = stage / root.name / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            digest = sha(path)
            shutil.copy2(path, dest)
            if digest != sha(path) or digest != sha(dest):
                raise RuntimeError('File changed during packaging')
            records.append({'source_path': str(path), 'package_path': str(dest.relative_to(stage)), 'sha256': digest})
    shutil.copy2(__file__, stage / 'package_and_reporting_reproduction.py')
    write(stage / 'PACKAGE_SCOPE.json', {'UTC': datetime.now(timezone.utc).isoformat(),
        'source_roots': list(map(str, roots)), 'included_files': records, 'omitted': omitted,
        'scope': 'Simulation forensic and shared-Mc numerical diagnostics only; not a new catalog result or standalone raw-data distribution.',
        'external_dependencies': 'See input SHA256 manifests; existing original simulation caches and repository physics modules are required for recomputation.',
        'goal_achieved': False, 'status': 'HOLD_FOR_AUTHOR_REVIEW_NO_ADOPTION_NO_OVERWRITE'})
    hashes = {str(p.relative_to(stage)): sha(p) for p in sorted(stage.rglob('*')) if p.is_file()}
    write(stage / 'OUTPUT_SHA256.json', hashes)
    with (stage / 'SHA256SUMS.txt').open('x') as handle:
        handle.write(''.join(v + '  ' + k + '\n' for k, v in hashes.items()))
    with tarfile.open(output, 'w:gz') as archive:
        archive.add(stage, arcname=stage.name)
    found = set()
    members = 0
    with tarfile.open(output, 'r|gz') as archive:
        for member in archive:
            if not member.isfile():
                continue
            members += 1
            key = str(Path(member.name).relative_to(stage.name))
            if key in ('OUTPUT_SHA256.json', 'SHA256SUMS.txt'):
                continue
            if key not in hashes:
                raise RuntimeError('Unexpected archive file')
            digest = hashlib.sha256()
            stream = archive.extractfile(member)
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
            if digest.hexdigest() != hashes[key]:
                raise RuntimeError('Archive hash mismatch')
            found.add(key)
    if found != set(hashes):
        raise RuntimeError('Missing archive member')
    receipt = {'UTC': datetime.now(timezone.utc).isoformat(), 'archive': str(output), 'sha256': sha(output),
        'bytes': output.stat().st_size, 'regular_members': members, 'verified_files': len(found),
        'hash_failures': 0, 'missing_files': 0, 'goal_achieved': False}
    write(output.with_suffix(output.suffix + '.verification.json'), receipt)
    with output.with_suffix(output.suffix + '.sha256').open('x') as handle:
        handle.write(receipt['sha256'] + '  ' + output.name + '\n')
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', required=True, choices=('report', 'package'))
    parser.add_argument('--audit', type=Path)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--source', type=Path, nargs='+')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.stage == 'report':
        report(args.audit, args.root)
    else:
        package(args.source, args.output)
