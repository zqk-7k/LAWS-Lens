"""Package conditional runtime measurements without raw PE or credentials."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import tarfile


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    r = args.root
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    final = json.loads((r/'contracts/FINAL_STATUS.json').read_text())
    scope_file = r/'contracts/COMMON_SCOPE_FROZEN.json'
    scope = json.loads(scope_file.read_text()) if scope_file.exists() else {}
    timing_file = r/'tables/scaling_timings.csv'
    timing = pd.read_csv(timing_file) if timing_file.exists() else pd.DataFrame()
    eligibility = pd.read_csv(r/'tables/event_eligibility.csv')
    failed = eligibility[~eligibility.eligible].sort_values('event_name')
    failed.to_csv(r/'tables/inapplicable_events.csv', index=False, encoding='utf-8-sig')
    checks = json.loads((r/'contracts/ELIGIBILITY_COST.json').read_text())
    resources = pd.read_csv(r/'tables/resource_timeline.csv') if (r/'tables/resource_timeline.csv').exists() else None
    resource_note = '资源监测未生成。'
    if resources is not None and len(resources):
        resource_note = (f"15秒抽样监测：本任务进程树RSS最大 {resources.process_tree_rss_MiB.max():.1f} MiB；"
                         f"磁盘最小空余 {resources.free_disk_GiB.min():.2f} GiB；"
                         f"全GPU已用显存最大 {resources.whole_GPU_memory_MiB.max():.1f} MiB。"
                         '抽样不是瞬时严格峰值；GPU数值含同机其他任务，不能当作本算法独占量。'
                         '资源原表available_RAM_GiB是主机可见内存，不是容器获分配的内存限额。')
    summary = []
    if len(timing):
        for (method, n), part in timing.groupby(['method', 'n_events']):
            prep = part[part.boundary.eq('PREPARE_EVENTS')]
            total = part[part.boundary.eq('PRODUCTS_READY_TOTAL')]
            cache = part[part.boundary.eq('EVENT_CACHE_PAIR_RANK')]
            summary.append(dict(method=method, n_events=int(n), n_pairs=int(n*(n-1)//2),
                preparation_s=None if prep.empty else float(prep.wall_s.iloc[0]),
                first_pass_s=None if total.empty else float(total.wall_s.iloc[0]),
                pair_repeats=len(cache), cached_pair_median_s=None if cache.empty else float(cache.wall_s.median()),
                cached_pair_min_s=None if cache.empty else float(cache.wall_s.min()),
                cached_pair_max_s=None if cache.empty else float(cache.wall_s.max())))
    table = pd.DataFrame(summary)
    table.to_csv(r/'tables/runtime_summary.csv', index=False, encoding='utf-8-sig')
    aggregate = []
    for f in sorted((r/'results').glob('Phazap_n*/event_preparation.csv')):
        events = pd.read_csv(f)
        aggregate.append(dict(n_events=len(events), posterior_samples_sum=int(events.samples.sum()),
            samples_min=int(events.samples.min()), samples_median=float(events.samples.median()),
            samples_max=int(events.samples.max()), event_job_wall_sum_s=float(events.wall_s.sum()),
            event_cpu_sum_s=float(events.cpu_s.sum()), event_job_wall_median_s=float(events.wall_s.median()),
            event_job_wall_max_s=float(events.wall_s.max()), worker_peak_rss_max_MiB=float(events.peak_rss_MiB.max())))
    pd.DataFrame(aggregate).to_csv(r/'tables/Phazap_resource_summary.csv', index=False)
    if not table.empty:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.7))
        for ax, key, title in zip(axes, ['first_pass_s', 'cached_pair_median_s'],
                                 ['Products-ready first pass', 'Prepared-event pair recomputation']):
            for method, color in [('TriLens', '#147967'), ('Phazap', '#bd4857')]:
                sub = table[table.method.eq(method)].sort_values('n_pairs')
                ax.plot(sub.n_pairs, sub[key], 'o-', label=method, color=color)
            ax.set(xscale='log', yscale='log', xlabel='Distinct unordered pairs', ylabel='Measured wall time (s)', title=title)
            ax.grid(alpha=.2)
            ax.legend()
        fig.suptitle('TRILENS-SPEED-DOMAIN-04: common eligible O3 subset', y=.99)
        fig.text(.5, .91, 'Native GPU vs 6 CPU workers; public products ready; not an equal-recall comparison', ha='center', fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, .87))
        fig.savefig(r/'figures/runtime_scaling.png', dpi=170)
        fig.savefig(r/'figures/runtime_scaling.pdf')
        plt.close(fig)
    def fmt(x):
        return '未完成' if x is None or pd.isna(x) else f'{x:.3f}'
    rows = ['|方法|事件|不同事件对|事件准备/秒|首次完整前端/秒|事件缓存后重新配对/秒|',
            '|---|---:|---:|---:|---:|---:|']
    for item in sorted(summary, key=lambda v: (v['n_pairs'], v['method'])):
        rows.append('|{method}|{n_events}|{n_pairs}|{prep}|{total}|{cache}|'.format(**item,
                    prep=fmt(item['preparation_s']), total=fmt(item['first_pass_s']), cache=fmt(item['cached_pair_median_s'])))
    failures = ['|事件|完整后验样本|非有限相位样本|比例|', '|---|---:|---:|---:|']
    for item in failed.to_dict('records'):
        failures.append(f"|{item['event_name']}|{int(item.get('samples', 0))}|{int(item.get('nonfinite_rows', 0))}|{item.get('nonfinite_fraction', 0):.3%}|")
    report = '\n'.join([
        '# TRILENS-SPEED-DOMAIN-04：共同适用子集速度实测', '',
        '本轮只测试计算速度，不测试相同透镜召回率，不修改模型、权重、原始候选排名或论文。', '',
        '## 1. 执行结果与边界', '',
        f"计算完成：{final['compute_complete']}；一致性检查 {final['checks_passed']}/{final['checks_total']}；受保护输入哈希不变：{final['protected_inputs_unchanged']}。",
        f"冻结目录为62个O3事件；完整相位预检后共同适用 {scope.get('eligible_events', '未确定')} 个、排除 {len(failed)} 个。",
        '排除依据只为作者程序在冻结20/40/100 Hz频率下产生非有限相位。整事件从两种方法同时排除，不挑选较快或排名较好的事件。该条件子集可能改变质量等分布，不能推广为完整目录表现。',
        '既往全目录失败记录保留，不因本次条件子集成功而改判。频率、作者算法、完整后验样本均未改变。', '',
        '## 2. 实测时间', '', *rows, '',
        '准备阶段运行一次；缓存配对运行两次，表中为中位数，原表保留最小/最大值和全部重复。首次完整前端为直接墙钟计时，不是多个并行子任务时间之和。',
        '事件准备每个规模都从原始输入重新读取和计算。缓存阶段复用事件级特征/相位，但重新计算所有pair，不复用pair分数。',
        f"共同适用域预检额外耗时 {checks['wall_s']:.3f} 秒、累计worker CPU {checks['worker_cpu_s_sum']:.3f} CPU秒；这项一次性审计成本独立报告，没有隐藏为免费准备。", '',
        '## 3. 不能计算的事件', '', *failures, '',
        '完整排除原因及每个字段的非有限数量见event_eligibility.csv。已诊断的GW200322_091133是在部分高质量后验样本100 Hz处两极化波形都为零，不能给不存在的相位补零。其他失败只在得到独立证据后才归因为同一机制。', '',
        '## 4. 计时包含什么', '',
        '- TriLens：读取应变/参考噪声、PSD及预处理、短窗/16秒匹配特征、全部冻结网络与校准、公开天空图读取及NESTED到RING和512处理、时间/天空/波形pair分数、三seed共识排名和输出。',
        '- Phazap：读取完整公开PE、作者程序相位转换、worker启动与事件相位加载、所有pair作者计算及排序输出。没有本轮新做PE，也没有抽薄后验。',
        '- 双方均排除离线训练、公开PE/天空图生产、网络下载和完整性审计。这里是公开数据产品已存在的新增计算成本，不是从裸应变开始的完整科学工作流。',
        '- 不比较相同输入信息量：Phazap读取完整PE；TriLens需要应变和天空图。GW200220_124850保留TriLens冻结Mixed天空，但Phazap读取同一HDF5中显式XPHM后验及配置，差异在合同中记录。',
        '- TriLens使用GPU和2个数值CPU线程；Phazap使用6个单线程CPU worker。共享服务器，没有强制等资源，也没有操作系统缓存清空，不能将时间比称为硬件无关或严格冷启动加速倍数。',
        '- TriLens神经网络维持原62槽位并对缺席事件补零以确保BF16数值复现，额外计算计入耗时；这不是经过优化的小目录batch实现。', '',
        resource_note, '',
        '## 5. 正确性与可复现', '',
        'TriLens逐事件预处理与归档逐值核对；逐seed波形、时间、天空分数按冻结容差核对；各子集共识顺序必须与归档重算一致。Phazap检查完整有限输出、同一pair集合、已有参考重合部分和两次重复的DJ一致性。计时重复性不等于物理正确性或统计检出能力验证。',
        '新的SPEED_ONLY_DOMAIN_CONTRACT.json及其冻结哈希为本次适用域、失败处理与规模规则；CONTRACT.json仅继承原引擎的科学和计时设置，冲突处由本次合同明确替换。COMMON_SCOPE_FROZEN.json在正式计时前记录实际共同事件和pair哈希。',
        '保存原始失败预检、所有时序日志、每个规模的完整配对结果、受保护文件前后哈希。可重建源脚本在scripts/，公开PE和大模型不打进紧凑包，仅保留来源哈希。', '',
        '## 6. 能支持的结论', '',
        '只支持：在明确的共同适用O3子集、已有公共数据产品和本次硬件配置下，两个实现生成候选排序的新增耗时。不能支持：相同recall下更省计算、替代官方PO/Phazap联合规则、减少多少Hanabi次数、完整目录通用性或透镜确认。没有新增官方FPP或Hanabi结果。',
        '下一层科学比较仍需同一独立注入的合格PE、冻结阈值和相同pair recall；本轮不等待或擅自启动昂贵PE。', '',
        '作者软件与方法入口：[Phazap](https://github.com/ezquiaga/phazap)、[官方例子](https://phazap.readthedocs.io/en/latest/examples.html)。', '',
        '## 7. 文件与状态', '',
        f'服务器目录：`{r}`。',
        '主表：tables/runtime_summary.csv；原始计时：tables/scaling_timings.csv；适用性：tables/event_eligibility.csv；图：figures/runtime_scaling.pdf。',
        '最终状态：`HOLD_FOR_AUTHOR_REVIEW_NO_ADOPTION_NO_OVERWRITE`。', '',
    ])
    (r/'reports/TRILENS_SPEED_DOMAIN04_CN.md').write_text(report)
    (r/'README_CN.md').write_text('# TRILENS-SPEED-DOMAIN-04\n\n请先阅读 reports/TRILENS_SPEED_DOMAIN04_CN.md。\n\n条件适用子集速度测试，不是相同召回率的科学比较。旧结果、模型和论文未覆盖。\n')
    shutil.copy2(__file__, r/'scripts'/Path(__file__).name)
    before = json.loads((r/'manifest/INPUTS_BEFORE.json').read_text())
    source_index = []
    for path, info in before.items():
        src = Path(path)
        if src.suffix != '.py':
            continue
        try:
            relative = src.relative_to('/root/autodl-tmp/gw-catalog')
        except ValueError:
            relative = Path('external')/src.relative_to('/')
        dst = r/'scripts/archived_inputs'/relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        source_index.append(dict(source=str(src), package_path=str(dst.relative_to(r)), sha256=sha(dst)))
    write_json(r/'manifest/SOURCE_SNAPSHOTS.json', source_index)
    excluded_suffix = {'.hdf5', '.h5', '.npz', '.npy', '.pt', '.pth', '.pyc'}
    selected = [f for f in r.rglob('*') if f.is_file() and f.suffix not in excluded_suffix
                and '__pycache__' not in f.parts and 'phases' not in f.parts
                and f.name not in {'OUTPUT_SHA256.csv', 'SHA256SUMS.txt', 'report_and_package.log'}]
    for f in selected:
        if f.stat().st_size > 64*2**20:
            raise RuntimeError('Unexpected oversized delivery file: '+str(f))
        if f.suffix in {'.py', '.json', '.md', '.log', '.txt', '.csv'}:
            data = f.read_bytes()
            markers = [b'-----BEGIN '+b'RSA PRIVATE KEY-----', b'-----BEGIN '+b'OPENSSH PRIVATE KEY-----', b'sshpass '+b'-p ']
            if any(m in data for m in markers):
                raise RuntimeError('Potential credential material: '+str(f))
    manifest = r/'manifest/OUTPUT_SHA256.csv'
    if manifest.exists():
        raise RuntimeError('Delivery already exists; no overwrite')
    with manifest.open('w', newline='') as out:
        writer = csv.DictWriter(out, fieldnames=['path', 'bytes', 'sha256'])
        writer.writeheader()
        for f in sorted(selected):
            writer.writerow(dict(path=str(f.relative_to(r)), bytes=f.stat().st_size, sha256=sha(f)))
    selected.append(manifest)
    package = Path('/root/autodl-tmp/gw-catalog/packages')/(r.name+'_deliverables.tar.gz')
    with tarfile.open(package, 'x:gz') as tar:
        for f in sorted(selected):
            tar.add(f, arcname=str(Path(r.name)/f.relative_to(r)), recursive=False)
    package.with_suffix(package.suffix+'.sha256').write_text(sha(package)+'  '+package.name+'\n')
    expected = {row['path']: row['sha256'] for row in csv.DictReader(manifest.open())}
    failures = []
    with tarfile.open(package, 'r:gz') as tar:
        members = tar.getmembers()
        for member in members:
            rel = str(Path(member.name).relative_to(r.name))
            if rel in expected:
                got = hashlib.sha256(tar.extractfile(member).read()).hexdigest()
                if got != expected[rel]:
                    failures.append(rel)
    if failures:
        raise RuntimeError('Archive hash failures: '+str(failures))
    verification = dict(package=str(package), sha256=sha(package), bytes=package.stat().st_size,
                        members=len(members), checked_files=len(expected), hash_failures=failures,
                        compute_complete=final['compute_complete'], historical_inputs_unchanged=final['protected_inputs_unchanged'])
    write_json(package.with_suffix('.verification.json'), verification)
    print(json.dumps(verification))


if __name__ == '__main__':
    main()
