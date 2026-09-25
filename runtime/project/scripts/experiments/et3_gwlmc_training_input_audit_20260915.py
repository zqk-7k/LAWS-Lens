#!/usr/bin/env python3
"""Read-only adapter audit for new physical ET data, not legacy SIS/PM offsets."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main(reference, destination):
    if destination.exists():
        raise RuntimeError('Audit already exists')
    destination.mkdir()
    events = json.loads((reference/'contracts/EVENTS.json').read_text())
    manifest = pd.read_csv(reference/'tables/PHYSICAL_STRAIN.csv', float_precision='round_trip').set_index('event_uid')
    rows = []
    for event in events:
        record = manifest.loc[event['event_uid']]
        if sha(record.path) != record.sha256:
            raise RuntimeError('Frozen physical strain changed')
        m1, m2 = event['m1_det'], event['m2_det']
        mc = (m1*m2)**.6/(m1+m2)**.2
        eta = m1*m2/(m1+m2)**2
        chi = (m1*event['a1']*np.cos(event['tilt1']) + m2*event['a2']*np.cos(event['tilt2']))/(m1+m2)
        with np.load(record.path) as data:
            checks = {'short_shape': data['short'].shape == (3, 4096),
                'long_shape': data['long'].shape == (3, 4096),
                'physical_clean_shape': data['clean'].shape == (3, 24*4096),
                'physical_noisy_shape': data['noisy_padded'].shape == (3, 26*4096),
                'finite': all(np.isfinite(data[k]).all() for k in ('short', 'long', 'clean', 'noisy_padded')),
                'nonzero_channels': bool(np.all(np.std(data['short'], axis=1) > 0) and np.all(np.std(data['long'], axis=1) > 0))}
            identical = {f'channels_{a}_{b}_identical': bool(np.array_equal(data['clean'][a], data['clean'][b]))
                         for a, b in ((0, 1), (0, 2), (1, 2))}
        rows.append({'event_uid': event['event_uid'], 'source_uid': event['source_uid'],
            'source_id': event['source_id'], 'family': event['family'], 'image': event['image'],
            'noise_seed': event['noise_seed'], 'path': record.path, 'sha256': record.sha256,
            'mc_det': mc, 'eta': eta, 'chi_eff': chi, 'm1_det': m1, 'm2_det': m2,
            'labels_inside_development_support': 10 <= mc <= 512 and .05 <= eta <= .25 and -1 <= chi <= 1,
            'label_source': 'frozen detector-frame parameters; not a redshift inferred from rescaled distance',
            **checks, **identical})
    frame = pd.DataFrame(rows)
    frame.to_csv(destination/'DEVELOPMENT_TRAINING_INPUTS.csv', index=False)
    label_spread = frame.groupby('source_uid')[['mc_det', 'eta', 'chi_eff']].agg(lambda x: float(x.max()-x.min()))
    flags = ['short_shape', 'long_shape', 'physical_clean_shape', 'physical_noisy_shape', 'finite',
             'nonzero_channels', 'labels_inside_development_support']
    duplicate_noise = frame.noise_seed.duplicated().any()
    passed = bool(frame[flags].all().all() and not duplicate_noise and np.all(label_spread.to_numpy() == 0))
    result = {'pass': passed, 'events': len(frame), 'sources': frame.source_uid.nunique(),
        'duplicate_noise_seeds': bool(duplicate_noise), 'same_source_labels_exact': bool(np.all(label_spread.to_numpy() == 0)),
        'identical_detector_pair_count_across_all_events': int(frame.filter(regex='^channels_').sum().sum()),
        'development_mass_range': [float(frame.mc_det.min()), float(frame.mc_det.max())],
        'legacy_50000_row_offsets_used': False, 'legacy_SIS_PM_labels_inferred': False,
        'mass_rederived_from_rescaled_distance': False, 'label_clipping_used': False,
        'encoder_training_started': False, 'formal_3000_data_adapter_completed': False,
        'note': 'A validated development input table, not a trained model or a completed production trainer.'}
    (destination/'RESULT.json').write_text(json.dumps(result, indent=2)+'\n')
    (destination/'README_CN.md').write_text(
        '# 新 ET 训练输入读取审计\n\n'
        '本表直接读取本轮物理 strain 和 GW-LMC 事件 UID，不使用旧 50,000 行 SIS/PM 偏移。'
        'Mc、eta、chi_eff 来自生成该信号时冻结的 detector-frame 参数；不能因共同振幅系数改变了有效距离，'
        '就重新从该距离推导红移和质量。双像的三个内禀标签必须完全一致。\n\n'
        '已核对 2 s 和 16 s 输入各为三通道、4096 点，通道有限且非零。这里只建立开发样本读取表并测试，'
        '没有启动 optimizer，也没有宣称旧训练脚本可直接用于新 3,000-event 数据。\n\n'
        '旧训练入口仍需单独适配新 global-source split、质量支持域、正式特征缓存、校准及 NEW-SCORE-ONLY 汇总；'
        '不能把旧 SIS/PM checkpoint 或旧指标冒充本轮 GW-LMC 训练结果。\n\n'
        '```json\n'+json.dumps(result, ensure_ascii=False, indent=2)+'\n```\n')
    shutil.copy2(__file__, destination/Path(__file__).name)
    print(json.dumps(result), flush=True)
    if not passed:
        raise RuntimeError('Development training-input audit failed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    a = parser.parse_args()
    main(a.reference, a.destination)
