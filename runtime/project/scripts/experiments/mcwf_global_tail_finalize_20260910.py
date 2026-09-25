#!/usr/bin/env python3
"""Final integrity, execution metadata, and concise delivery index; no rescoring."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pandas as pd
import psutil

P = Path('/root/autodl-tmp/gw-catalog')
sys.path.insert(0, str(P / 'scripts/experiments'))
import mcwf_global_branch_calibration_plan_20260910 as cal


def text(path, value):
    with path.open('x', encoding='utf-8') as f:
        f.write(value)


def main(global_root, tail_root):
    expected = {}
    for root in (global_root, tail_root, tail_root / 'uncertainty', tail_root / 'uncertainty_summary'):
        for row in pd.read_csv(root / 'manifest/INPUT_SHA256.csv').itertuples():
            if row.path in expected and expected[row.path] != row.sha256:
                raise RuntimeError('Conflicting input hashes')
            expected[row.path] = row.sha256
    checks = [{'path': name, 'before': value, 'after': cal.sha(name)} for name, value in expected.items()]
    if any(row['before'] != row['after'] for row in checks):
        raise RuntimeError('Final input verification failed')
    for root in (global_root, tail_root):
        deps = pd.read_csv(root / 'manifest/RUNTIME_DEPENDENCIES.csv')
        for r in deps.itertuples():
            if cal.sha(root / r.destination) != r.sha256 or cal.sha(r.source) != r.sha256:
                raise RuntimeError('Runtime dependency changed')
    r64 = json.loads((global_root / 'contracts/PILOT_GATE.json').read_text())
    r65 = json.loads((tail_root / 'audit/FINAL_GOAL_READOUT.json').read_text())
    final = json.loads((tail_root / 'contracts/FROZEN_COMPUTATIONS_COMPLETE.json').read_text())
    if r64['gate'] != 'FAIL' or r65['goal_achieved'] or final['goal_achieved']:
        raise RuntimeError('Unexpected result contract, do not silently misreport')
    if cal.sha(tail_root / final['report']) != final['report_sha256']:
        raise RuntimeError('Final report changed')
    cal.csv(tail_root / 'audit/DELIVERY_INPUT_HASHES_UNCHANGED.csv', checks)
    # Preserve the failed reporting entry point, not only its error log.
    failed = P / 'scripts/experiments/mcwf_global_and_tail_report_20260910.py'
    dest = tail_root / 'scripts/preflight_failures/global_and_tail_report_missing_import.py'
    dest.parent.mkdir(parents=True)
    shutil.copy2(failed, dest)
    if cal.sha(failed) != cal.sha(dest):
        raise RuntimeError('Report recovery snapshot failed')
    cal.write(tail_root / 'audit/REPORTING_RECOVERY.json', {'UTC': cal.utc(),
        'failed_script': str(failed), 'sha256': cal.sha(failed), 'snapshot': str(dest.relative_to(tail_root)),
        'error': 'Module name mcwf_branch_accounting_20260910 did not exist; correct frozen helper is mcwf_reference_branch_audit_20260910.',
        'scientific_computation_changed': False, 'original_log_preserved': 'logs/FINAL_REPORT.log'})
    root = Path('/sys/fs/cgroup')
    cpu = (root / 'cpu.max').read_text().strip() if (root / 'cpu.max').exists() else None
    mem = (root / 'memory.max').read_text().strip() if (root / 'memory.max').exists() else None
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader'],
                         text=True, capture_output=True)
    stages = {}
    for name in ('evaluate', 'real'):
        stages[name] = json.loads((tail_root / f'logs/{name}_COMPLETE.json').read_text())
    resources = {'UTC': cal.utc(), 'cpu_logical_host': psutil.cpu_count(), 'cpu_quota_cgroup': cpu,
        'RAM_host_GiB': psutil.virtual_memory().total / 1024**3, 'memory_limit_cgroup': mem,
        'GPU_inventory': gpu.stdout.strip() if gpu.returncode == 0 else None,
        'GPU_training_or_inference_in_this_round': False, 'disk_free_GiB': shutil.disk_usage(P).free / 1024**3,
        'stage_process_usage': stages,
        'measurement_limits': 'Process maxRSS/rusage only. No whole-workflow peak GPU/RAM telemetry was collected; hostCPU count is not CPUquota.',
        'new_full_PE_or_strain_generation': False, 'bootstrap_workers': 6}
    cal.write(tail_root / 'audit/EXECUTION_RESOURCES.json', resources)
    concise = pd.read_csv(tail_root / 'tables/CRITICAL_PAIR_CONCISE.csv')
    readme = '# R64/R65 独立探索交付索引\n\n'
    readme += f'状态：`{cal.STATUS}`。完整目标尚未达成，不能自动升级。\n\n'
    readme += '服务器：`root@connect.westd.seetacloud.com:32328`。本文件不包含认证信息。\n\n'
    readme += f'- R64：`{global_root}`\n- R65：`{tail_root}`\n\n'
    readme += 'R64：总体计权校准未通过，未做真实重排。R65：固定95%共享波形不相容尾部，仅降低波形分数，不改变time/sky/外层权重。未恢复旧Mc/q或总分混合。\n\n'
    readme += concise[['method', 'seed', 'reported_rank']].to_markdown(index=False) + '\n\n'
    readme += 'R65关键pair共识rank28，三个模型35/16/65；PE BC仍为0.091665。O3/O4a共识Top10/20相对NODUP均未恶化，但3个O3单模型官方预算仍下降，7个注入模型/目录/通道门槛失败。因此未达成完整目标。相较R62，R65的AUPRC更差，不能称为升级。\n\n'
    readme += '主要文件：\n\n'
    readme += '- `reports/FINAL_SHARED_PROFILE_CATALOG_REPORT_CN.md`：完整方法、来源、结果、失败项和区间。\n'
    readme += '- `tables/RETRIEVAL_PER_SEED.csv`、`RETRIEVAL_SUMMARY.csv`：全部波形/融合检索指标。\n'
    readme += '- `tables/PE_OFFICIAL_BUDGETS.csv`、`PER_SEED_PE_OFFICIAL_BUDGETS.csv`：共识与单模型PE/官方统计。\n'
    readme += '- `tables/REAL_BUDGET_ENTERED_LEFT.csv`：Top10/20换入换出及公开PE/官方列。\n'
    readme += '- `results/<method>/<run>/consensus/`：完整pair、Top10/20/50/100的波形与融合排序。\n'
    readme += '- `audit/FINAL_GOAL_READOUT.json`：所有未通过项；不是仅摘要平均值。\n'
    readme += '- `uncertainty/`、`uncertainty_summary/`：query10000与pair2000次system bootstrap。\n'
    readme += '- `figures/`：逐模型注入指标图、共识PE/官方对照图，PDF和PNG。\n'
    readme += '- `contracts/`、`scripts/`、`logs/`、`manifest/`：可复算合同、代码、日志和哈希。\n\n'
    readme += '全部历史score/PE估计/ordering/time/sky均未改动。公开Hanabi表重合不是透镜真阳性；本轮没有运行Hanabi。真实反馈属于适应性开发，不称独立确认。\n'
    text(tail_root / 'README_CN.md', readme)
    shutil.copy2(__file__, tail_root / 'scripts/global_tail_finalize.py')
    cal.write(tail_root / 'audit/DELIVERY_INTEGRITY_COMPLETE.json', {'UTC': cal.utc(),
        'status': cal.STATUS, 'goal_achieved': False, 'input_hashes_verified': len(checks),
        'changed_inputs': 0, 'report_sha256': final['report_sha256'],
        'all_stage_processes_complete': True, 'figures_visually_checked_separately': True,
        'archive_not_yet_verified': True})
    print('DELIVERY_INPUTS_VERIFIED', len(checks), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--global-root', type=Path, required=True)
    p.add_argument('--tail-root', type=Path, required=True)
    a = p.parse_args()
    main(a.global_root, a.tail_root)
