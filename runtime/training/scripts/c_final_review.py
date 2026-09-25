"""Post-freeze decision sensitivity and a new, verified C delivery revision."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'

import argparse
import hashlib
import json
from pathlib import Path
import tarfile

import numpy as np
import pandas as pd


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            value.update(block)
    return value.hexdigest()


def write_json(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write('\n')


def decision_audit(out, dest):
    comparisons, heads, checks = [], [], []
    for run in ('O3', 'O4a', 'O4b'):
        sky = pd.read_parquet(out/f'real_sky/{run}/pairs.parquet')
        sky['pair_key'] = ['--'.join(sorted((a, b))) for a, b in zip(sky.event_i, sky.event_j)]
        sky = sky.set_index('pair_key')
        base = out/f'deployments/{run}/C_PHYSICAL'
        source = base/'real_ranking/consensus_all_pairs.parquet'
        before = digest(source)
        formal = pd.read_parquet(source)
        formal = formal[formal.method.eq('three-channel')].set_index('pair_key').sort_index()
        seeds = []
        for seed in (2026091721, 2026091722, 2026091723):
            spec = json.loads((base/f'calibration/score/seed_{seed}/SELECTED.json').read_text())
            weights = np.asarray(spec['final_fusion']['POSITIVE']['weights'])
            frame = pd.read_parquet(base/f'real_ranking/seed_{seed}_all_scores.parquet').set_index('pair_key')
            for nside in (256, 512, 1024):
                score = np.column_stack((frame.waveform_score, frame.time_score,
                    sky.loc[frame.index, f'sky_log_bf_nside{nside}']))@weights
                if nside == 512 and not np.allclose(score, frame.final_score_POSITIVE, rtol=0, atol=1e-12):
                    raise RuntimeError('Formal score reconstruction failed')
                row = pd.DataFrame(dict(pair_key=frame.index, score=score, seed=seed, nside=nside))
                row = row.sort_values(['score', 'pair_key'], ascending=[False, True])
                row['rank'] = np.arange(1, len(row)+1)
                seeds.append(row)
        consensus = {}
        for nside, group in pd.concat(seeds, ignore_index=True).groupby('nside'):
            frame = group.groupby('pair_key').agg(rank_mean=('rank', 'mean'), rank_max=('rank', 'max'),
                                                  score_mean=('score', 'mean')).reset_index()
            frame = frame.sort_values(['rank_mean', 'rank_max', 'score_mean', 'pair_key'],
                                      ascending=[True, True, False, True])
            frame['consensus_rank'] = np.arange(1, len(frame)+1)
            consensus[nside] = frame.set_index('pair_key')
        if not np.array_equal(consensus[512].sort_index().consensus_rank, formal.consensus_rank):
            raise RuntimeError('Formal Nside512 consensus did not reproduce')
        for nside in (256, 1024):
            for budget in (10, 20, 50):
                a = set(consensus[512].head(budget).index)
                b = set(consensus[nside].head(budget).index)
                comparisons.append(dict(run=run, budget=budget, reference_nside=nside,
                    overlap=len(a & b), Jaccard=len(a & b)/len(a | b),
                    incoming=';'.join(sorted(b-a)), outgoing=';'.join(sorted(a-b)),
                    formal_nside=512, diagnostic_only=True, weights_reselected=False))
        for pair in consensus[512].head(50).index:
            row = sky.loc[pair]
            signs = np.sign([row[f'sky_log_bf_nside{n}'] for n in (256, 512, 1024)])
            heads.append(dict(run=run, pair_key=pair,
                **{f'consensus_rank_{n}': int(consensus[n].loc[pair, 'consensus_rank']) for n in (256, 512, 1024)},
                **{f'Z_sky_{n}': float(row[f'sky_log_bf_nside{n}']) for n in (256, 512, 1024)},
                sign_stable=bool((signs == signs[0]).all()),
                abs_delta_512_1024=float(abs(row.sky_log_bf_nside512-row.sky_log_bf_nside1024))))
        if digest(source) != before:
            raise RuntimeError('Formal rank table changed')
        checks.append(dict(run=run, reproduced=True, unchanged=True, sha256=before))
    pd.DataFrame(comparisons).to_csv(dest/'real_resolution_decision_audit.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(heads).to_csv(dest/'real_top50_resolution_sensitivity.csv', index=False, encoding='utf-8-sig')
    write_json(dest/'REAL_DECISION_RESOLUTION_AUDIT.json', dict(state='PASS',
        meaning='sensitivity computation passed, not acceptance of strict all-pair convergence',
        formal_nside=512, original_rank_tables=checks, no_reselection=True))
    return pd.DataFrame(comparisons), pd.DataFrame(heads)


def main(root):
    out = root/'completion'
    delivery = json.loads((out/'contracts/FINAL_DELIVERY.json').read_text())
    if not delivery.get('complete_results'):
        raise RuntimeError('Finish the main experiment before final review')
    dest = out/'postflight_review'
    dest.mkdir(exist_ok=False)
    comparisons, heads = decision_audit(out, dest)
    coverage = pd.read_csv(out/'tables/sky_truth_coverage_and_runtime.csv').query("split == 'test'")
    budgets = pd.read_csv(out/'tables/real_PE_official_budget_summary.csv').query("method == 'three-channel'")
    summaries = {}
    for size in ('190', '450'):
        frame = pd.read_csv(out/f'tables/retrieval_summary_{size}.csv')
        summaries[size] = frame.query("split == 'test' and method == 'three-channel'")
    top = pd.read_csv(out/'tables/real_top50_all_runs.csv')
    conflict = top[(top.run == 'O3') & (top.consensus_rank <= 10) & (top.Dmax > 3)]
    resolution = pd.read_csv(out/'tables/sky_resolution_summary.csv')
    table = lambda frame: frame.to_markdown(index=False, floatfmt='.4f')
    notes = ['# GWLR-UC-01 最终阅读入口与科学边界',
        delivery['state'],
        '计算、真实候选审计和原始两份压缩包已完成。此状态表示待作者审核，不是计算中断。'
        '本补充仅做冻结后的独立复核，不重新训练、调权、修改正式Nside或替换排名。',
        '## 技术修复',
        '训练中断由多层软链接的路径比较造成：同一噪声文件被误判为不同部署。'
        '修复使用最终文件身份比较，五项单元测试通过。原脚本、失败日志和原合同保留，'
        '通过独立修复入口续跑，未用删除Gate或放宽统计规则恢复。',
        '## 实际完成',
        '三个运行期各三个seed、五个组件，共45个模型组件。validation/test共2700张事件级'
        'BAYESTAR地图，另有90张训练pilot地图；原生地图共2790张。'
        '完整450事件评价和500次固定190事件子目录评价均已完成。',
        'C相对A/B的两项改变是每源共同振幅缩放，以及从与波形相同的带噪应变恢复触发量。'
        'NEW-SCORE-ONLY框架和一维时间lookup不变；各运行期在自己的validation上校准、选权。',
        '独立复算648项指标，最大差3.33e-16；101个冻结文件哈希不变。'
        '2700份原始应变和2700张MOC哈希均通过；源内共同振幅检查通过；'
        '各运行期validation/test的源、global source与噪声父块交集为空。'
        '原交付另核对419个受保护历史输入，全部未变。',
        '## 注入结果',
        '以下均为三模型均值；std是模型间样本标准差，不是总体置信区间。'
        '190事件结果先在每个模型内平均500个子目录，不能将500次抽样当作独立实验。']
    cols = ['run', 'macro_r_at_1_mean', 'macro_r_at_10_mean', 'macro_r_at_10_std',
            'average_precision_mean', 'false_at_recall_0p5_mean', 'false_at_recall_0p9_mean']
    for size in ('190', '450'):
        notes += ['### '+size+'事件', table(summaries[size][cols])]
    notes += ['## 真实候选与未解决的PE冲突', table(budgets),
        'O4b官方列NA表示冻结输入没有可核验逐对表，不是零重合。'
        '公开Hanabi重合不是透镜真值，本轮没有重新运行Hanabi。',
        'O3 Top10仍有以下Dmax>3的候选，不能写成全部PE相容：',
        table(conflict[['consensus_rank', 'pair_key', 'BC_Mc', 'D_Mc', 'Dmax']]),
        '该结果不用于重新调权或删去候选。C在O3/O4a/O4b并非所有指标都优于A/B；'
        '同规模A/B/C表见completion/tables/A_B_C_comparison_190.csv与450.csv。',
        '## 天空覆盖率不是严格90%',
        table(coverage[['run', 'frozen_validation_temperature', 'raw_HPD90_coverage',
                        'calibrated_HPD50_coverage', 'calibrated_HPD90_coverage']]),
        '这些是每个源总权重为1的经验覆盖率。不能因与90%较接近就声称后验严格校准；'
        '仅六个测试噪声父块也限制了总体推断。没有用test重新选择温度。',
        '## 分辨率：区分原始尾部分数和候选决策',
        table(resolution[(resolution.split == 'test') & (resolution.population == 'all')][
            ['run', 'pairs', 'sign_flips', 'P99', 'maximum']]),
        '极少数强负分尾部差异很大。实际逐对检查发现O3有-92.58变为约-690.78，'
        'O4b有-183.97变为约-690.78的无关对；后者数值对应实现中的log(1e-300)下限。'
        '这不能称为所有pair原始分数严格收敛，也不能把巨大负分的差值解释为独立物理证据。',
        '下面用原冻结权重、同一波形/时间分数，单独替换审计分辨率计算诊断排名。'
        '正式512排名已逐项精确复现且哈希不变，诊断排名未替代正式排名：',
        table(comparisons),
        'Top50逐对符号及名次见real_top50_resolution_sensitivity.csv。'
        '决策稳定不等于每个极端尾部分数数值稳定。',
        '## 仍需保留的解释边界',
        '距离分布按较弱参考像SNR条件化，不是完整宇宙学透镜总体。'
        '8192条aligned-spin模板的目录时刻跟进不等于完整BBH PE或具有标定FAR的盲搜索。'
        '注入天空使用HL，公开真实PE仍可能使用更完整网络。历史A/B父样本已被查看，'
        '不能称新增独立盲测。时间日历沿用冻结A/B输入，本轮没有重新定义运行期边界。',
        '## 文件',
        '服务器：root@connect.westd.seetacloud.com:32328。实验根目录：', str(root),
        '正式全报告：completion/reports/GWLR_UC_01_COMPLETE_METHOD_AND_RESULTS_CN.md。'
        '真实Top10/20/50/100和全部pair：completion/real_PE/<run>/C_PHYSICAL/three-channel/。',
        '原始交付包和原生地图包保留不动。新增deliverables_reviewed包仅补入本次只读审计与'
        '阅读入口，不改变任何模型、分数或名次。下载与SHA-256见FINAL_REVIEW_DELIVERY.json。']
    (dest/'READ_FIRST_FINAL_RESULTS_CN.md').write_text('\n\n'.join(notes)+'\n', encoding='utf-8')
    original = Path(delivery['package'])
    if digest(original) != delivery['sha256']:
        raise RuntimeError('Original package hash mismatch')
    original_manifest = json.loads(original.with_suffix('.manifest.json').read_text())
    manifest = dict(original_manifest)
    extras = {f'completion/postflight_review/{p.name}': p for p in dest.iterdir() if p.is_file()}
    extras['training/scripts/c_final_review.py'] = Path(__file__)
    extras['completion/contracts/ORIGINAL_FINAL_DELIVERY.json'] = out/'contracts/FINAL_DELIVERY.json'
    for name, path in extras.items():
        if name in manifest:
            raise RuntimeError('Duplicate reviewed package member')
        manifest[name] = digest(path)
    package = root/'package/GWLR_UC_01_deliverables_reviewed.tar.gz'
    write_json(package.with_suffix('.manifest.json'), manifest)
    with tarfile.open(original, 'r:gz') as source, tarfile.open(package, 'x:gz', compresslevel=3) as target:
        for member in source:
            if not member.isfile() or member.name not in original_manifest:
                raise RuntimeError('Unexpected original package member')
            target.addfile(member, source.extractfile(member))
        for name, path in sorted(extras.items()):
            target.add(path, arcname=name, recursive=False)
    seen = set()
    with tarfile.open(package, 'r:gz') as archive:
        for member in archive:
            if member.name in seen or not member.isfile():
                raise RuntimeError('Invalid reviewed package member')
            value = hashlib.sha256()
            stream = archive.extractfile(member)
            for block in iter(lambda: stream.read(2**20), b''):
                value.update(block)
            if value.hexdigest() != manifest[member.name]:
                raise RuntimeError('Reviewed archive member hash mismatch')
            seen.add(member.name)
    if seen != set(manifest):
        raise RuntimeError('Reviewed package incomplete')
    final = dict(state=delivery['state'], complete_results=True, supplementary_review_only=True,
        original_package=str(original), original_sha256=delivery['sha256'], original_retained_unchanged=True,
        reviewed_package=str(package), sha256=digest(package), bytes=package.stat().st_size,
        checked_members=len(seen), native_maps=delivery['native_maps'],
        native_maps_sha256=delivery['native_maps_sha256'], native_map_count=delivery['native_map_count'],
        report=str(dest/'READ_FIRST_FINAL_RESULTS_CN.md'), no_formal_rank_change=True)
    Path(str(package)+'.sha256').write_text(final['sha256']+'  '+package.name+'\n')
    write_json(root/'package/FINAL_REVIEW_DELIVERY.json', final)
    print(json.dumps(final), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    main(parser.parse_args().root)
