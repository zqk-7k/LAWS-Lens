#!/usr/bin/env python3
"""Independent readback, source-block diagnostics, and verified compact delivery."""
import argparse
from datetime import datetime, timezone
import hashlib
import io
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False,
        default=lambda x: x.item() if isinstance(x, np.generic) else str(x))+'\n')


def run(root, failed):
    if not (root/'contracts/FINAL_PREFLIGHT_RESULT.json').exists():
        raise RuntimeError('Bounded pilot has not finished')
    path = root/'contracts/INDEPENDENT_DELIVERY_AUDIT.json'
    if path.exists():
        raise RuntimeError('Audit exists; do not overwrite it')
    shutil.copy2(__file__, root/'scripts'/Path(__file__).name)
    versions = {'python': sys.version, 'executable': sys.executable}
    for name in ['bilby', 'lalsuite', 'ligo.skymap', 'numpy', 'scipy', 'pandas', 'astropy', 'healpy', 'torch', 'pycbc']:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = 'not installed in this pilot interpreter'
    write(root/'contracts/DEPENDENCY_VERSIONS.json', versions)
    resources = {'utc': datetime.now(timezone.utc).isoformat(), 'logical_cpu_visible': os.cpu_count(),
        'workers': 6, 'openmp_threads_per_sky_worker': 4, 'dense_map_persistence': False,
        'disk_free_GiB_at_delivery': shutil.disk_usage(root).free/2**30,
        'GPU_used_for_this_pilot': False}
    for name in ['cpu.max', 'memory.max']:
        file = Path('/sys/fs/cgroup')/name
        resources[name] = file.read_text().strip() if file.exists() else 'unavailable'
    try:
        resources['GPU_inventory'] = subprocess.check_output(['nvidia-smi',
            '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader'], text=True, timeout=15).strip()
    except (OSError, subprocess.SubprocessError):
        resources['GPU_inventory'] = 'unavailable'
    write(root/'contracts/RESOURCE_SNAPSHOT_AT_DELIVERY.json', resources)
    copied_failures = []
    if failed:
        out = root/'prior_attempt'
        out.mkdir()
        for name in ['contracts/FATAL_ERROR.json', 'contracts/ANALYSIS_CONTRACT.json',
                     'scripts/et3_gwlmc_3000_preflight_20260914.py']:
            dest = out/Path(name).name
            shutil.copy2(failed/name, dest)
            copied_failures.append({'original': str(failed/name), 'snapshot': str(dest.relative_to(root)),
                'sha256': sha(dest)})
    contract = json.loads((root/'contracts/ANALYSIS_CONTRACT.json').read_text())
    fixed_hash = json.loads((root/'contracts/CONTRACT_HASH.json').read_text())['sha256']
    assert fixed_hash == sha(root/'contracts/ANALYSIS_CONTRACT.json')
    plan = json.loads((root/'contracts/RESERVED_SYSTEM_SPLITS.json').read_text())
    ids = {s: {r['source_id'] for r in plan if r['split'] == s} for s in ['train', 'validation', 'test']}
    assert all(not ids[a] & ids[b] for a, b in [('train', 'validation'), ('train', 'test'), ('validation', 'test')])
    events = json.loads((root/'contracts/PILOT_EVENTS.json').read_text())
    assert {r['source_id'] for r in events} <= ids['train']
    assert len({r['noise_seed'] for r in events}) == 24
    assert len({r['measurement_seed'] for r in events}) == 24
    raw = pd.read_csv(root/'tables/PHYSICAL_STRAIN_PILOT.csv')
    assert len(raw) == 24 and raw['pass'].all()
    for r in raw.itertuples():
        assert sha(r.path) == r.sha256
        with np.load(r.path) as x:
            assert x['clean'].shape == (3, 98304)
            assert x['noisy_padded'].shape == (3, 106496)
            assert x['short'].shape == (3, 4096)
            assert x['long'].shape == (3, 4096)
            assert all(np.isfinite(x[k]).all() for k in ['clean', 'noisy_padded', 'short', 'long'])
    maps = pd.read_csv(root/'tables/MAP_RESOURCES.csv')
    jobs = pd.read_csv(root/'tables/JOB_STATUS.csv')
    assert len(maps) == int((jobs.status == 'COMPLETE').sum())
    from astropy.table import Table
    from ligo.skymap import moc
    native = []
    for r in maps.itertuples():
        assert sha(r.map) == r.sha256
        assert abs(r.normalization-1) < 1e-5
        sky = Table.read(r.map)
        order = moc.uniq2order(np.asarray(sky['UNIQ']))
        native.append({'case': r.id, 'native_nside_min': 2**int(order.min()),
            'native_nside_max': 2**int(order.max()), 'cells': len(sky),
            'canonical_ordering': 'NESTED/UNIQ', 'formal_raster_nside': 512,
            'probability_convention': 'PROBDENSITY times pixel area', 'SHA256': r.sha256})
    pd.DataFrame(native).to_csv(root/'tables/NATIVE_MOC_PROVENANCE.csv', index=False, encoding='utf-8-sig')
    pairs = pd.read_csv(root/'tables/PILOT_PAIR_SKY_SCORES.csv')
    grouped = []
    pivot = pairs.pivot(index=['i', 'j', 'true_companion'], columns='q', values='Z_sky').reset_index()
    assert len(pivot) == 276 and int(pivot.true_companion.sum()) == 8
    for qa, qb in [(10, 32), (32, 64)]:
        for label, subset in [('all', pivot), ('companions', pivot[pivot.true_companion]),
                              ('noncompanions', pivot[~pivot.true_companion])]:
            subset = subset.dropna(subset=[qa, qb])
            delta = (subset[qa]-subset[qb]).abs()
            grouped.append({'kind': 'quadrature', 'from': qa, 'to': qb, 'group': label,
                'pairs': len(subset), 'median_abs_delta': delta.median(), 'q99_abs_delta': delta.quantile(.99),
                'max_abs_delta': delta.max(), 'sign_flips': ((subset[qa] > 0) != (subset[qb] > 0)).sum(),
                'Spearman': spearmanr(subset[qa], subset[qb]).statistic})
    resolution = pd.read_csv(root/'tables/HEALPIX_RESOLUTION_PAIRS.csv').pivot(
        index=['i', 'j', 'true_companion'], columns='nside', values='Z_sky').reset_index()
    complete_q64 = int((maps.q == 64).sum())
    assert len(resolution) == complete_q64*(complete_q64-1)//2
    for ns1, ns2 in [(256, 512), (512, 1024)]:
        for label, subset in [('all', resolution), ('companions', resolution[resolution.true_companion]),
                             ('noncompanions', resolution[~resolution.true_companion])]:
            delta = (subset[ns1]-subset[ns2]).abs()
            grouped.append({'kind': 'HEALPix', 'from': ns1, 'to': ns2, 'group': label,
                'pairs': len(subset), 'median_abs_delta': delta.median(), 'q99_abs_delta': delta.quantile(.99),
                'max_abs_delta': delta.max(), 'sign_flips': ((subset[ns1] > 0) != (subset[ns2] > 0)).sum(),
                'Spearman': spearmanr(subset[ns1], subset[ns2]).statistic})
    pd.DataFrame(grouped).to_csv(root/'tables/INDEPENDENT_STRATIFIED_CONVERGENCE.csv', index=False, encoding='utf-8-sig')
    rank_rows = []
    for qa, qb in [(10, 32), (32, 64)]:
        both = pivot.dropna(subset=[qa, qb])
        lookup = {(int(r['i']), int(r['j'])): (float(r[qa]), float(r[qb])) for _, r in both.iterrows()}
        members = sorted(set(both.i.astype(int)) | set(both.j.astype(int)))
        for i in members:
            candidates = [j for j in members if j != i and tuple(sorted([i, j])) in lookup]
            first = sorted(candidates, key=lambda j: (-lookup[tuple(sorted([i, j]))][0], j))[:10]
            second = sorted(candidates, key=lambda j: (-lookup[tuple(sorted([i, j]))][1], j))[:10]
            rank_rows.append({'query': i, 'from_q': qa, 'to_q': qb, 'common_candidates': len(candidates),
                'top10_intersection': len(set(first)&set(second)), 'top10_overlap': len(set(first)&set(second))/min(10, len(candidates)),
                'descriptive_only': True})
    pd.DataFrame(rank_rows).to_csv(root/'tables/PILOT_RANK_STABILITY.csv', index=False, encoding='utf-8-sig')
    complete_times = maps[maps.q == 64].wall_seconds.to_numpy()
    censored = int(((jobs.q == 64)&(jobs.status == 'TIMEOUT')).sum())
    lower_times = np.r_[complete_times, np.full(censored, 1200.)]
    timing = {'q64_completed': len(complete_times), 'q64_timeout': censored,
        'q64_other_failures': int(((jobs.q == 64)&~jobs.status.isin(['COMPLETE', 'TIMEOUT'])).sum()),
        'completed_only_P50_P90_max': np.quantile(complete_times, [.5, .9, 1]).tolist(),
        'with_timeout_lower_bounds_P50_P90_max': np.quantile(lower_times, [.5, .9, 1]).tolist(),
        'warning': 'Timeout observations are right-censored, not exact 1200 s runtimes. Do not extrapolate completed-only averages.'}
    write(root/'tables/RUNTIME_WITH_CENSORING.json', timing)
    hpd = pd.read_csv(root/'tables/MAP_HPD_A90.csv')
    rng = np.random.default_rng(2026091407)
    coverage = []
    for q in [10, 32, 64]:
        x = hpd[hpd.q == q].assign(hit=lambda x: x.truth_HPD <= .9).groupby('source_id').hit.agg(['sum', 'count'])
        draws = rng.integers(0, len(x), (10000, len(x)))
        totals = x['sum'].to_numpy()[draws].sum(axis=1)/x['count'].to_numpy()[draws].sum(axis=1)
        coverage.append({'q': q, 'events_covered': int(x['sum'].sum()), 'events': int(x['count'].sum()),
            'independent_sources': len(x), 'event_fraction': x['sum'].sum()/x['count'].sum(),
            'source_block_percentile95': np.quantile(totals, [.025, .975]).tolist(),
            'claim': 'descriptive stratified development set only; not population calibration'})
    write(root/'tables/DESCRIPTIVE_SOURCE_BLOCK_COVERAGE.json', coverage)
    # Inventory only: these source parameters do not select or revise a model.
    table = pd.read_csv(next(Path('/root/autodl-tmp/GW-LMC/2.5PLUS/BBH/Any_Detected_SNR1').glob('*SourceParams.csv')))
    train = table.iloc[[r['gwlmc_row'] for r in plan if r['split'] == 'train']]
    mc = (train.m1_det*train.m2_det)**.6/(train.m1_det+train.m2_det)**.2
    support = {'training_sources': len(train), 'training_Mc_min_max': [float(mc.min()), float(mc.max())],
        'training_Mc_outside_legacy_grid_5_200': int(((mc < 5)|(mc > 200)).sum()),
        'action': 'reported only; no source exclusion, grid adjustment, or test-driven selection',
        'not_a_new_model_result': True}
    write(root/'tables/TRAINING_ONLY_MASS_SUPPORT_AUDIT.json', support)
    parity = [
        ('source_population', 'GW-LMC smooth/subhalo', 'same input CSV hashes, global-ID grouping', 'pilot implemented'),
        ('detectors', 'H1/L1', 'ET1/ET2/ET3 exact Bilby response; dimension adaptation required', 'pilot implemented'),
        ('noise', 'GWOSC empirical run-specific noise/PSD', 'independent Gaussian ET design PSD; not equal data distribution', 'pilot implemented'),
        ('source_waveform', 'physical IMRPhenomXPHM at 20Hz reference', 'same generator and units', 'pilot implemented'),
        ('short_input', '2s,2048Hz,4096 samples;40-580Hz', 'same operators, 3 channels', 'exact first-two-channel operator test PASS'),
        ('long_input', '16s,256Hz,4096 samples;20-80Hz', 'same anti-alias/filter/crop, 3 channels', 'pilot implemented'),
        ('matching_features', 'q/spin/Mc template curves + network combination', 'same algorithm requires ET template bank and detector axis adaptation', 'not executed for new bank'),
        ('predictive_network', 'p(Mc)*p(eta,chi_eff|Mc)', 'same target factorization, independently trained ET network', 'not trained'),
        ('waveform_evidence', 'Zwf_OMC+gamma*T+beta*I', 'same requested definition; ET development calibration required', 'not calibrated'),
        ('sky_input', 'Gaussian synthetic trigger series, known injected intrinsic parameters', 'same conditional trigger model in numerical controls, not noisy-strain PE', 'pilot only; data-link boundary explicit'),
        ('sky_spin_units', 'frozen script passes solar mass values to SI interface', 'correct SI masses in diagnostic controls', 'FAIL: cannot label these identical scientific implementations'),
        ('sky_numerics', 'default 10-point inclination/polarization quadrature', '10/32/64 controls with fixed likelihood and priors', 'no ET-only formal upgrade'),
        ('sky_score', 'log[Npix*sum(Pi*Pj)] at Nside512', 'same formula and ordering;256/512/1024 audit', 'pilot pair scores only'),
        ('time', 'one-dimensional run-conditioned time evidence', 'same method needs ET synthetic-exposure calibration; no 2D time-SNR', 'not fitted'),
        ('fusion', 'NEW-SCORE-ONLY alpha=1, three-channel weighted sum', 'same requested definition, validation-only ET coefficients', 'not selected/evaluated'),
        ('final_test', 'frozen held-out catalog', 'reserved source-disjoint test, unopened waveform and scores', 'no final R@K/AUPRC'),
    ]
    pd.DataFrame(parity, columns=['stage', 'HL_reference', 'ET_status_or_difference', 'execution_status']).to_csv(
        root/'tables/METHOD_PARITY_AUDIT.csv', index=False, encoding='utf-8-sig')
    protected = json.loads((root/'manifests/PROTECTED_INPUTS.json').read_text())
    changed = [x['path'] for x in protected if sha(x['path']) != x['sha256']]
    assert not changed
    verification = {'utc': datetime.now(timezone.utc).isoformat(), 'status': 'AUDIT_PASS_NOT_SCIENTIFIC_GATE_PASS',
        'protected_files_unchanged': len(protected), 'original_contract_unchanged': True,
        'source_split_intersections': 0, 'unique_noise_and_trigger_seeds': True,
        'physical_arrays_verified': 24, 'MOC_files_verified': len(maps),
        'map_jobs_failed': int((jobs.status != 'COMPLETE').sum()),
        'pair_count': 276, 'companion_pairs': 8, 'noncompanion_pairs': 268,
        'prior_failure_preserved': copied_failures,
        'pilot_raw_bytes': int(raw.bytes.sum()), 'MOC_bytes': int(maps.map_bytes.sum()),
        'bulk_experiment_completed': False, 'test_strain_scores_opened': False,
        'sampler_core_binaries': 'external read-only inputs; path and SHA-256 in manifest',
        'release_rebuild_scripts': 'original ET quadrature build record required; package is not a standalone container'}
    write(path, verification)
    report = f'''# 独立交付复核补充

这是预检交付完整性通过，不是3000事件科学Gate通过。

24个物理应变文件、{len(maps)}个原生MOC文件、276对开发pair均已独立回读并核对哈希。真正伴随对为8对，非伴随对268对；后者共享事件，不能称为268个独立背景系统。失败/超时地图任务为{int((jobs.status != 'COMPLETE').sum())}项；不把缺失地图算作通过。

配置哈希不变；{len(protected)}个受保护输入哈希不变；train/validation/test源交集为0。噪声与触发随机种子分别去重通过。

原首次启动在26秒噪声缓存处失败，失败日志和脚本快照已收录prior_attempt/。仅修复对象复用与长度断言，没有改动物理参数或数值阈值。

## 源级覆盖描述
```json
{json.dumps(coverage, indent=2, ensure_ascii=False)}
```
这是刻意分层的开发集，不是总体coverage结论。没有使用这些结果调整temperature。

## 额外输入支持问题
训练集合{support['training_sources']}个系统中，{support['training_Mc_outside_legacy_grid_5_200']}个Mc超出旧质量网格5至200太阳质量。正式训练前需要统一说明OOD处理，不能静默剪裁后把边界概率当作可靠PE。本次未改变网格，也未重新选择样本。

详细真/假对分层收敛表：tables/INDEPENDENT_STRATIFIED_CONVERGENCE.csv。

逐模块统一性对照见tables/METHOD_PARITY_AUDIT.csv。探测器响应、噪声来源和曝光日历必须如实区分；同一算法不代表这些物理输入也相同。匹配特征网络、预测分布校准、一维时间拟合和最终融合尚未执行，不能用旧ET结果填充本轮表格。

原生分辨率分布见tables/NATIVE_MOC_PROVENANCE.csv。512/1024是对同一原生MOC的表示与积分检查；若原生像素较粗，两者相等不说明天空推断本身已经收敛。因此还必须单独检查内部倾角/偏振积分。单位错误的四事件地图对照冻结源参数与随机种子，但会同时改变条件触发生成和定位模板，不是固定同一strain的联合PE对照。

每个query的Top-10稳定性见PILOT_RANK_STABILITY.csv，仅比较双方均完成的同一候选集合。运行时间见RUNTIME_WITH_CENSORING.json：主报告的分位数针对已完成任务，若有超时，补表同时提供纳入1200秒右删失下界的统计，不能用成功任务均值估计全部工期。

压缩包保留原生MOC、报告、CSV、合同、脚本和日志；物理strain留在服务器pilot/strain/，包内清单明确标为未包含。原生C核心库是外部只读依赖，不声称该包是独立容器。
'''
    (root/'reports/INDEPENDENT_DELIVERY_AUDIT_CN.md').write_text(report)
    files = []
    for f in sorted(root.rglob('*')):
        rel = str(f.relative_to(root))
        if f.is_file() and not rel.startswith('package/') and rel != 'manifests/DELIVERY_MANIFEST.csv':
            files.append({'path': rel, 'sha256': sha(f), 'bytes': f.stat().st_size,
                          'in_package': not rel.startswith('pilot/strain/')})
    pd.DataFrame(files).to_csv(root/'manifests/DELIVERY_MANIFEST.csv', index=False, encoding='utf-8-sig')
    archive = root/'package'/f'{root.name}_preflight_audited.tar.gz'
    with tarfile.open(archive, 'x:gz') as tar:
        for f in files:
            if f['in_package']:
                tar.add(root/f['path'], arcname=f'{root.name}/{f["path"]}')
        tar.add(root/'manifests/DELIVERY_MANIFEST.csv', arcname=f'{root.name}/manifests/DELIVERY_MANIFEST.csv')
    expected = {f'{root.name}/{r["path"]}': r['sha256'] for r in files if r['in_package']}
    expected[f'{root.name}/manifests/DELIVERY_MANIFEST.csv'] = sha(root/'manifests/DELIVERY_MANIFEST.csv')
    checked = 0
    with tarfile.open(archive, 'r:gz') as tar:
        for member in tar:
            assert member.isfile() and member.name in expected
            content = tar.extractfile(member).read()
            assert hashlib.sha256(content).hexdigest() == expected[member.name]
            if member.name.endswith(('.py', '.json', '.md', '.csv', '.log')):
                assert (b'BEGIN '+b'RSA PRIVATE KEY') not in content
                assert (b'sshpass'+b' -p') not in content
            checked += 1
    assert checked == len(expected)
    digest = sha(archive)
    archive.with_suffix(archive.suffix+'.sha256').write_text(f'{digest}  {archive.name}\n')
    result = {'path': str(archive), 'sha256': digest, 'bytes': archive.stat().st_size,
              'verified_members': checked, 'hash_failures': 0}
    write(root/'package/INDEPENDENT_PACKAGE_CHECK.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--failed', type=Path)
    a = p.parse_args()
    run(a.root, a.failed)
