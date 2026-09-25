#!/usr/bin/env python3
"""Independent audit/Chinese delivery for the ET physical-trigger repair."""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MPLBACKEND'] = 'Agg'
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import tarfile
import numpy as np
import pandas as pd


def load(path):
    return json.loads(path.read_text())


def main(root, reference, bank=None, data_sky=None):
    path = root/'scripts/et3_physical_trigger_repair_20260915_v2.py'
    spec = importlib.util.spec_from_file_location('repair_audit_base', path)
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)
    write, sha = base.write, base.sha
    rows = load(root/'contracts/EVENTS.json')
    physical = pd.read_csv(root/'tables/PHYSICAL_STRAIN.csv')
    trigger = pd.read_csv(root/'tables/STRAIN_TRIGGER_AUDIT.csv')
    shared, lenses = [], []
    for sid, group in physical.groupby('source_id'):
        events = [x for x in rows if x['source_id'] == sid]
        relative_scale = float(group.amplitude_scale.max()/group.amplitude_scale.min()-1)
        params = ['m1_det', 'm2_det', 'a1', 'a2', 'tilt1', 'tilt2', 'phi12', 'phijl', 'ra', 'dec', 'psi', 'theta_jn', 'phase']
        conflicts = [k for k in params if len({x[k] for x in events}) > 1]
        ratio_ok = True
        ratio_error = 0.
        if len(group) == 2:
            a, b = group.iloc[0], group.iloc[1]
            ratio_error = abs((a.network_optimal_snr/b.network_optimal_snr)/
                (a.unscaled_network_optimal_snr/b.unscaled_network_optimal_snr)-1)
            ratio_ok = ratio_error <= 1e-12
        shared.append({'source_id': int(sid), 'events': len(group), 'common_scale_relative_error': relative_scale,
            'response_derived_SNR_ratio_error': ratio_error, 'parameter_conflicts': '|'.join(conflicts),
            'pass': relative_scale == 0 and not conflicts and ratio_ok})
        for e in events:
            if e['image'] == 0:
                continue
            snr = np.asarray(json.loads(e['image_img_snrs']))
            delay = np.asarray(json.loads(e['image_img_delays_days']))
            selected = sorted(sorted(np.flatnonzero(snr >= 1), key=lambda j: snr[j], reverse=True)[:2], key=lambda j: delay[j])
            idx = selected[e['image']-1]
            mu = json.loads(e['image_img_mags'])[idx]
            kappa, g1, g2 = [json.loads(e[k])[idx] for k in ('image_img_kappa', 'image_img_gamma1', 'image_img_gamma2')]
            det = (1-kappa)**2-g1**2-g2**2
            eig = [1-kappa-np.hypot(g1, g2), 1-kappa+np.hypot(g1, g2)]
            morse = .5*sum(x < 0 for x in eig)
            c = float(group.amplitude_scale.iloc[0])
            lenses.append({'event_uid': e['event_uid'], 'source_id': int(sid), 'selected_image_index': int(idx),
                'catalog_mu': mu, 'used_mu': e['magnification'], 'jacobian_mu': 1/det,
                'abs_mu_det_minus_one': abs(mu*det-1), 'parity_agrees': np.sign(mu) == np.sign(det),
                'morse_catalog_jacobian': morse, 'used_morse': e['morse'],
                'amplitude_factor': c*np.sqrt(abs(mu)),
                'effective_common_distance_Mpc': e['dl_source']/c,
                'apparent_distance_Mpc': e['dl_source']/c/np.sqrt(abs(mu)),
                'distance_rescaling_not_cosmological_population_prediction': True})
    pd.DataFrame(shared).to_csv(root/'tables/SYSTEM_PHYSICAL_CONSISTENCY.csv', index=False)
    pd.DataFrame(lenses).to_csv(root/'tables/GWLMC_MAGNIFICATION_JACOBIAN_AUDIT.csv', index=False)
    gate = {'pass': all(x['pass'] for x in shared) and all(x['catalog_mu'] == x['used_mu'] and
        x['parity_agrees'] and x['morse_catalog_jacobian'] == x['used_morse'] for x in lenses),
        'systems': len(shared), 'lensed_images': len(lenses),
        'max_SNR_ratio_relative_error': max(x['response_derived_SNR_ratio_error'] for x in shared),
        'max_catalog_Jacobian_relative_residual': max(x['abs_mu_det_minus_one'] for x in lenses),
        'Jacobian_residual_note': 'rounded catalog kappa/gamma check, not a re-solved lens model'}
    write(root/'contracts/INDEPENDENT_PHYSICAL_AUDIT.json', gate)
    maps = [load(p) for p in sorted((root/'maps').glob('*/RESULT.json'))]
    resources = [{**{k: x[k] for k in ('idx', 'q', 'event_uid', 'wall_seconds', 'CPU_seconds', 'peak_RSS_KiB', 'map_bytes', 'normalization')},
        **next(a for a in x['raster_audit'] if a['nside'] == 512)} for x in maps]
    pd.DataFrame(resources).to_csv(root/'tables/MAP_RESOURCES_AND_COVERAGE.csv', index=False)
    convergences = {p.stem: load(p) for p in sorted((root/'contracts').glob('CONVERGENCE*.json'))}
    sampling = load(root/'contracts/SAMPLING_GATE.json') if (root/'contracts/SAMPLING_GATE.json').exists() else {'pass': False, 'status': 'not completed'}
    checks = load(root/'manifest/PROTECTED_BEFORE.json')
    changed = [x['path'] for x in checks if not Path(x['path']).exists() or sha(x['path']) != x['sha256']]
    write(root/'manifest/HISTORICAL_PROTECTION_FINAL.json', {'files': len(checks), 'changed': changed, 'pass': not changed})
    remaining = []
    if not any(x['pass'] for x in convergences.values()):
        remaining.append('天空 nuisance quadrature 收敛或运行时间尚未通过，不能批量生成。')
    if not sampling['pass']:
        remaining.append('带限 SNR 降采样与密集参考的地图一致性尚未通过。')
    data_result = None
    data_convergences = {}
    if bank is not None and (bank/'contracts/RESULT.json').exists():
        data_result = load(bank/'contracts/RESULT.json')
        shutil.copytree(bank/'contracts', root/'data_driven_template/contracts', dirs_exist_ok=False)
        shutil.copytree(bank/'triggers', root/'data_driven_template/triggers', dirs_exist_ok=False)
        shutil.copytree(bank/'tables', root/'data_driven_template/tables', dirs_exist_ok=False)
        shutil.copytree(bank/'scripts', root/'data_driven_template/scripts', dirs_exist_ok=False)
        if data_sky is not None:
            for part in ('contracts', 'maps', 'tables', 'scripts', 'build', 'figures'):
                shutil.copytree(data_sky/part, root/'data_driven_sky'/part, dirs_exist_ok=False)
            data_result['localization_results'] = [load(p) for p in sorted((data_sky/'contracts').glob('DATA_SKY*RESULT.json'))]
            data_convergences = {p.stem: load(p) for p in sorted((data_sky/'contracts').glob('CONVERGENCE*.json'))}
            if (data_sky/'contracts/DATA_DRIVEN_FINAL_RESULT.json').exists():
                data_result['final_localization_audit'] = load(data_sky/'contracts/DATA_DRIVEN_FINAL_RESULT.json')
        if not any(x['pass'] for x in data_convergences.values()):
            remaining.append('数据驱动模板恢复已完成，但其天空图尚未通过完整的高阶收敛 Gate；oracle 对照不能替代该分支。')
    else:
        remaining.append('当前仍是已知注入内禀参数的 oracle 模板对照；必须补数据驱动的模板恢复，不能称为盲 PE。')
    remaining += ['24 个事件仅来自 16 个 source，不能据此证明总体 90% coverage 已校准。',
        '训练质量支持域需要涵盖新 GW-LMC 高质量源，不能静默压到旧 Mc 网格边缘。']
    summary = {'state': base.FINAL, 'utc': base.utc(), 'root': str(root), 'reference_root': str(reference),
        'physical_gate': gate, 'trigger_gate': load(root/'contracts/TRIGGER_GATE.json'),
        'sampling_gate': sampling, 'convergence': convergences, 'map_jobs_completed': len(maps),
        'historical_hashes_unchanged': not changed, 'bulk_events_generated': 0, 'encoders_trained': 0,
        'heldout_test_opened': False, 'O3_O4a_modified': False, 'data_driven_template': data_result, 'remaining': remaining}
    summary['data_driven_sky_convergence'] = data_convergences
    summary['coverage_and_pixel_audits'] = {}
    for label, folder in [('oracle_control', root), ('data_recovered_template', data_sky)]:
        if folder is not None:
            summary['coverage_and_pixel_audits'][label] = {
                p.stem: load(p) for pattern in ('PIXEL_GATE*.json', 'DESCRIPTIVE_COVERAGE*.json',
                                               'DEVELOPMENT_RANK_STABILITY*.json')
                for p in sorted((folder/'contracts').glob(pattern))}
    if (root/'physics_invariants/RESULT.json').exists():
        summary['forward_physics_invariant_tests'] = load(root/'physics_invariants/RESULT.json')
    write(root/'contracts/INDEPENDENT_FINAL_AUDIT.json', summary)
    text = fr'''# ET-3 物理触发量修复与天空数值实验

状态：{base.FINAL}

本轮是 ET-only 开发实验，不是已经完成的 3,000-event 正式结果。历史 ET、O3/O4a、论文和原排名未修改。开发样本已经查看过，不能称为新的独立盲测。

## 1. 修复了什么

1. **共同振幅缩放**：同一透镜系统使用一个 $C$，各像为 $C\sqrt{{|\mu_j|}}h_j$；不再分别强制到 GW-LMC proposal SNR ratio。亮像自然超过旧 SNR=60 时不单独压低。
2. **单位**：波形生成的质量为太阳质量；调用 Bilby 自旋转换接口时乘以 `lal.MSUN_SI`，参考频率 20 Hz。
3. **实际应变入口**：天空触发量从保存的同一份带噪声物理 strain 经 PyCBC 匹配滤波提取；不再另外生成一份 Gaussian trigger 冒充那份 strain 的结果。
4. **正反向探测器几何一致**：Bilby ET 三个干涉仪的顶点和响应张量显式传给 BAYESTAR，不能直接假定 stock LAL E2/E3 与 Bilby 几何相同。
5. **匹配滤波一致性**：滤波、horizon、自相关使用同一 template power 和 PSD。独立离散求和、自注入、线性分解与 horizon 检查共 {len(trigger)*4} 项。

## 2. GW-LMC 物理输入

GW-LMC 提供参数表，不是本轮已经生成的 strain 文件。这里用原始 `img_mags`、delay、kappa/gamma 等列，波形由 Bilby/LALSuite IMRPhenomXPHM 生成。两类是 smooth/non-subhalo 与 subhalo-present，不是解析 SIS/PM。

几何光学下，每幅像使用常数 $\sqrt{{|\mu|}}\exp(-i\pi n)$；不是所有像使用同一个放大率，也不包含 wave-optics/microlensing 的频率依赖放大。

独立核对 $\mu^{{-1}}=(1-\kappa)^2-\gamma_1^2-\gamma_2^2$；由于公开表四舍五入，不能要求最后一位一致。本轮最大 $|\mu\det A-1|$ 为 {gate['max_catalog_Jacobian_relative_residual']:.8g}，16 幅透镜像的使用值、奇偶性和 Morse 分类均已列在 CSV。

同系统相对 SNR 保留误差最大为 {gate['max_SNR_ratio_relative_error']:.3g}。共同振幅重加权是受控 SNR 实验，不是自洽宇宙学人口预测：不应在修改有效距离后仍把原 source redshift 当作新的距离推断。

## 3. 数据和方法边界

- 24 个开发事件：8 个透镜系统的 16 幅像，8 个独立非透镜事件；共 16 个 source，均为 train-development。
- 24 s、4096 Hz、ET1/ET2/ET3 物理 strain；独立 ET design Gaussian noise，不是真实 ET 噪声。
- 短窗保持 [-1.75,+0.25] s、2048 Hz、4096 点；长窗保持 [-15.75,+0.25] s、256 Hz、4096 点。
- 本轮未开始 encoder 重训，未修改时间先验，未计算新的三通道 R@1/R@10。
- GW-LMC 来源为 2.5PLUS/BBH/Any_Detected_SNR1，是受选择影响的公开源库，不是无偏 ET 全人口。
- 冻结 O3/O4a 尚未同步这些入口修复，因此现在不能声称 ET 与历史 HL 已完成严格同方法比较。
- 单站 ET 的三个干涉仪不是三个跨洲站点。本轮是 20 Hz 以上短时 BBH 信号，不能借用长时低频 BNS 利用地球自转改善定位的结果。正确修复后天空图仍可能宽，这不是通过修改先验把图压窄的理由。

## 4. 为什么加带限 SNR 对照

高质量源的模板最高非零频率仅约 54.5 Hz，但原定位仍按 4096 Hz 积分，导致时间积分点从几十个膨胀到约 913 个。本轮新对照按模板**严格非零**频率支持选择采样率；发生降采样时，保证至少每最高频率周期 8 点，并验证新 Nyquist 以上模板功率严格为 0。不裁剪低功率尾部、不改原波形、不改变原复杂 SNR 峰值和 GPS 时刻。

这只证明无频率混叠；BAYESTAR 的时间插值仍可能产生误差，所以必须比较同一 q=32 下的完整地图、A90 和 pair score。该对照结果不能先验视为已通过。

天空的正式表示仍为 Nside=512。原生 MOC FITS 的 ORDERING 是 NUNIQ、COORDSYS=C，使用 UNIQ 编码的多阶像素和 PROBDENSITY；rasterize 后采用 NESTED 顺序，按像素面积变为归一化概率质量，再显式核对 NESTED/RING 转换。不能把 NUNIQ 原文件直接当作固定 Nside 的 RING 数组。256/512/1024 是像素分辨率；32/64/128 是 inclination/polarization nuisance quadrature 阶数，两者不能混淆。共同天空分数仍为 $\log[N_{{pix}}\sum P_iP_j]$。同一个 MOC 的 512/1024 积分一致，只证明表示正确，不能单独证明原始定位的角分辨率或概率校准已经充分。

## 5. 实测结果与 Gate

```json
{json.dumps(summary, ensure_ascii=False, indent=2)}
```

## 6. 还不能开始什么

'''
    text += '\n'.join('- '+x for x in remaining)
    if data_result is not None:
        text += '\n\n## 数据驱动模板分支\n\n另建的分支从实际带噪声 strain 搜索 3,795 个 PhenomD 模板，不读取真实 Mc、自旋或天空位置作为搜索输入；最终触发量由 PyCBC 复算。模板参数是搜索得到的点估计，不是 PE posterior，也不是新训练的 waveform encoder。与 XPHM 注入的模型近似、质量自旋退化和噪声可能使点估计偏离真值，不能拿该点估计直接替代后续联合参数分布模型。该分支与 oracle 数值对照分开归档。\n'
    text += '''

这些问题需要继续解决，不能把 oracle 控制当作完整单事件 PE，也不能为通过 Gate 删除高质量困难样本。当前仅提供真实完成的修复与开发计算，不编造正式 recall、PE 确认或 3,000-event 训练结果。

## 7. 参考依据

| 来源 | 用途 | 不能据此声称 |
|---|---|---|
| [GW-LMC](https://github.com/LensedGW/GW-LMC) | 公开 lens/source/image 参数与 magnification | 本项目重新求解了复杂透镜势或使用了作者的现成时域注入 |
| [Singer & Price, BAYESTAR](https://arxiv.org/abs/1508.03634) | 基于触发量的快速相干天空定位框架 | 本轮等同于完整 BBH PE，或单站 ET coverage 已验证 |
| [PyCBC matched filter](https://pycbc.org/pycbc/latest/html/pycbc.filter.html) | 从实际 strain 提取复匹配滤波 SNR | 使用真实注入参数的 oracle 模板等同于盲搜索 |
| [Bilby 自旋转换](https://bilby-dev.github.io/bilby/api/bilby.gw.conversion.bilby_to_lalsimulation_spins.html) | SI 质量接口与参考频率约定 | 仅修单位就保证历史 O3/O4a 结果全部正确 |
| [Singh & Bulik 2021](https://arxiv.org/abs/2011.06336) | 单站三角 ET 对 BBH 天空位置只能作有限约束的物理背景 | 本轮程序、MAP 面积或置信覆盖率已被该文验证 |
| [ET Design Report 2020](https://www.einsteintelescope-emr.eu/wp-content/uploads/2024/05/ET-0007B-20_ETDesignReportUpdate2020.pdf) | 长时信号的地球自转定位与短时重 BBH 的局限 | 24 s、20 Hz 分析体现了 ET 全低频定位潜力 |

## 8. 文件与复现

- `contracts/`：冻结合同、物理/触发量 Gate、采样和积分收敛结果。
- `tables/`：逐事件 SNR、共享参数、放大率核对、地图资源、收敛差值。
- `maps/`：已完成的原生 MOC、配置、日志；不永久保存三套 dense 地图。
- `triggers/`：从实际 strain 得到的 SNR 序列、PSD 和模板功率。
- `scripts/`：全部修复和交付脚本；依赖路径/哈希在 manifest。
- 原始物理 strain 保留在参考实验目录 `strain/`，紧凑包不重复打包。
- 失败尝试与旧版本均保留，没有删除。
'''
    reference_copy = root/'reference_dense_q32'
    reference_copy.mkdir(exist_ok=False)
    for p in sorted((reference/'maps').glob('*q32')):
        shutil.copytree(p, reference_copy/p.name)
    for name in ('REFERENCE_COMPUTE_STOP.json',):
        p = reference/'contracts'/name
        if p.exists():
            shutil.copy2(p, reference_copy/name)
    text += '\n\n## 9. 失败、重启与资源解释\n\n首次实际应变触发量提取因 float32/complex64 与 PyCBC complex128 模板不兼容而失败；该失败目录和日志保留。随后在独立 r2 目录修正输入精度，并重新完成全部触发量检查。密集采样参考的 q32 已完成，q64 为避免重复计算而明确中止；这不算 q64 成功，也没有删除部分输出。加速对照和数据驱动分支的每例退出状态、耗时与日志都保留。\n\n受保护哈希覆盖的是合同列明的关键输入及先前开发清单，不表示重新逐字节哈希了历史全部 TB 级数据。\n'
    timing_rows = []
    for label, folder in [('oracle_control', root), ('data_recovered_template', data_sky)]:
        if folder is None:
            continue
        for q in (32, 64, 128):
            completed = [load(p) for p in sorted((folder/'maps').glob(f'*q{q}/RESULT.json'))]
            if not completed:
                continue
            t = np.asarray([r['wall_seconds'] for r in completed])
            config = load(folder/'contracts/ANALYSIS_CONTRACT.json')
            workers = config['map_workers']
            schedule = folder/'contracts/RESOURCE_SCHEDULE_ADDENDUM.json'
            if q == 128 and schedule.exists():
                workers = load(schedule)['q128_workers']
            timing_rows.append({'branch': label, 'quadrature': q, 'completed': len(t),
                'all_24_complete': len(t) == 24, 'workers_measured': workers,
                'P50_seconds': float(np.median(t)), 'P90_seconds': float(np.quantile(t, .9)),
                'max_seconds_observed_not_population_worst_case': float(t.max()),
                'ideal_3000_event_P50_days': float(3000*np.median(t)/workers/86400),
                'ideal_3000_event_P90_days': float(3000*np.quantile(t, .9)/workers/86400),
                'ideal_3000_event_observed_max_days': float(3000*t.max()/workers/86400),
                'projection_not_measured_bulk_runtime': True, 'not_an_authorization_to_launch': True})
    pd.DataFrame(timing_rows).to_csv(root/'tables/RESOURCE_PROJECTION_NOT_BULK_MEASUREMENT.csv', index=False)
    text += '\n资源表按各分支实测并发数给出 3,000 事件的理想化 P50/P90/已观测最慢耗时外推；未包含排队、失败重跑、不同人口、特征提取、网络训练及后续审计，不能写成已经测得的总运行时间。\n'
    (root/'reports/ET3_REPAIR_FULL_REPORT_CN.md').write_text(text)
    write(root/'STATUS.json', {'state': base.FINAL, 'codename': 'ET-PHYS-SKY-PILOT-01',
        'current_contract': str(root/'contracts/INDEPENDENT_FINAL_AUDIT.json'),
        'report': str(root/'reports/ET3_REPAIR_FULL_REPORT_CN.md'),
        'bulk_started': False, 'encoder_training_started': False,
        'remaining': remaining, 'utc': base.utc()})
    shutil.copy2(__file__, root/'scripts'/Path(__file__).name)
    import matplotlib.pyplot as plt
    (root/'figures').mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), layout='constrained')
    axes[0].scatter(physical.old_per_image_target, physical.network_optimal_snr, s=20)
    upper = float(max(physical.old_per_image_target.max(), physical.network_optimal_snr.max()))
    axes[0].plot([0, upper], [0, upper], color='gray', lw=1)
    axes[0].set(xlabel='Old per-image target SNR', ylabel='Common-scale optimal SNR')
    for p in sorted((root/'tables').glob('MAP_CONVERGENCE_*.csv')):
        frame = pd.read_csv(p)
        if not frame.empty:
            axes[1].scatter(frame['index'], frame.TV, label=p.stem.replace('MAP_CONVERGENCE_', ''), s=20)
    axes[1].axhline(.02, color='gray', ls='--')
    axes[1].set(xlabel='Development event', ylabel='Map total variation')
    axes[1].legend(fontsize=8)
    frame = pd.DataFrame(resources)
    if not frame.empty:
        for q, group in frame.groupby('q'):
            axes[2].scatter(group.idx, group.wall_seconds, label=f'q={q}', s=20)
    axes[2].axhline(900, color='gray', ls='--')
    axes[2].set(xlabel='Development event', ylabel='Localization wall time (s)')
    axes[2].legend(fontsize=8)
    fig.savefig(root/'figures/et3_repair_diagnostics.png', dpi=160)
    fig.savefig(root/'figures/et3_repair_diagnostics.pdf')
    plt.close(fig)
    parts = [p.name for p in sorted(root.iterdir()) if p.name not in ('package', 'strain', 'cache')]
    files = [p for p in sorted(root.rglob('*')) if p.is_file() and p.parts[len(root.parts)] in parts
        and p.name not in ('FINAL_SHA256.csv', 'FINAL_PACKAGE_AUDIT.json')]
    manifest = [{'path': str(p.relative_to(root)), 'bytes': p.stat().st_size, 'sha256': sha(p)} for p in files]
    secret_hits = []
    pattern = re.compile(rb'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----|sshpass[ \t]+-p[ \t]+')
    for p in files:
        if p.suffix in ('.py', '.json', '.md', '.log', '.txt', '.csv', '.sh') and p.stat().st_size <= 8*2**20:
            # The scanner source contains the regex itself, not a credential.
            if p.name == Path(__file__).name:
                continue
            if pattern.search(p.read_bytes()):
                secret_hits.append(str(p.relative_to(root)))
    if secret_hits:
        raise RuntimeError('Potential credential material in delivery: '+str(secret_hits))
    pd.DataFrame(manifest).to_csv(root/'manifest/FINAL_SHA256.csv', index=False)
    archive = root/'package'/f'{root.name}_audited.tar.gz'
    if archive.exists():
        raise RuntimeError('Audited archive already exists; refusing overwrite')
    with tarfile.open(archive, 'w:gz') as tar:
        for part in parts:
            tar.add(root/part, arcname=root.name+'/'+part)
    bad = []
    with tarfile.open(archive) as tar:
        for item in manifest:
            f = tar.extractfile(root.name+'/'+item['path'])
            h = hashlib.sha256()
            for block in iter(lambda: f.read(8 << 20), b''):
                h.update(block)
            if h.hexdigest() != item['sha256']:
                bad.append(item['path'])
    if bad:
        raise RuntimeError('Archive hash mismatch: '+str(bad))
    digest = sha(archive)
    archive.with_suffix('.gz.sha256').write_text(digest+'  '+archive.name+'\n')
    result = {'archive': str(archive), 'bytes': archive.stat().st_size, 'sha256': digest,
        'internally_checked_files': len(manifest), 'internal_hash_failures': bad, 'historical_changes': changed,
        'private_key_or_ssh_password_command_matches': secret_hits}
    write(root/'manifest/FINAL_PACKAGE_AUDIT.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--bank', type=Path)
    parser.add_argument('--data-sky', type=Path)
    a = parser.parse_args()
    main(a.root, a.reference, a.bank, a.data_sky)
