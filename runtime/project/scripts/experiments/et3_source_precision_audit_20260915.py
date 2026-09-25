#!/usr/bin/env python3
"""Measure upstream float32 strain rounding without regenerating any results."""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import argparse
import importlib.util
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd


def main(reference, destination):
    if destination.exists():
        raise RuntimeError('Precision audit destination already exists')
    destination.mkdir()
    spec = importlib.util.spec_from_file_location('precision_base', reference/'scripts/et3_physical_trigger_repair_20260915_v2.py')
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)
    old, source, geometry, physics = base.deps(reference)
    events = json.loads((reference/'contracts/EVENTS.json').read_text())
    table = pd.read_csv(reference/'tables/PHYSICAL_STRAIN.csv').set_index('event_uid')
    generator = source.build_waveform_generator()
    rows = []
    for event in events:
        ifos, _, _ = geometry.geometry()
        params = source.source_parameters(pd.Series(event), event['gps'])
        rounded, _ = source.detector_response(generator, ifos, params,
                         source.lens_factor(event['magnification'], event['morse']))
        native = np.asarray([ifo.strain_data.time_domain_strain for ifo in ifos], dtype=np.float64)
        record = table.loc[event['event_uid']]
        with np.load(record.path) as saved:
            clean, freq, psd = saved['clean'], saved['psd_frequency'], saved['psd']
        scale = float(record.amplitude_scale)
        error = clean - scale*native
        delta_rho = float(physics.optimal_network_snr(error, freq, psd))
        rho_native = float(physics.optimal_network_snr(scale*native, freq, psd))
        exact = np.array_equal(clean, scale*rounded.astype(np.float64))
        rows.append({'event_uid': event['event_uid'], 'native_Bilby_dtype': str(native.dtype),
            'legacy_helper_return_dtype': str(rounded.dtype), 'saved_clean_dtype': str(clean.dtype),
            'saved_matches_declared_float32_then_float64_path': exact,
            'PSD_weighted_rounding_error_SNR': delta_rho,
            'rounding_error_relative_to_signal_SNR': delta_rho/rho_native})
    pd.DataFrame(rows).to_csv(destination/'SOURCE_PRECISION.csv', index=False)
    result = {'events': len(rows), 'all_saved_paths_reproduced': all(x['saved_matches_declared_float32_then_float64_path'] for x in rows),
        'max_rounding_error_SNR': max(x['PSD_weighted_rounding_error_SNR'] for x in rows),
        'max_relative_rounding_error_SNR': max(x['rounding_error_relative_to_signal_SNR'] for x in rows),
        'historical_strain_rewritten': False, 'new_physical_strain_rewritten': False,
        'note': 'The old source helper casts the clean signal to float32. r2 promotes it to float64 for noise addition and filtering; promotion fixes API compatibility but does not recover earlier rounding bits.'}
    base.write(destination/'RESULT.json', result)
    (destination/'README_CN.md').write_text(
        '# 源波形数值精度补充审计\n\n'
        '上游源生成 helper 返回的纯信号为 float32，r2 随后转成 float64，再加入 float64 噪声并匹配滤波。'
        '因此不能说整个生成链从未经过 float32；转成 float64 解决的是 PyCBC 接口精度匹配，并不恢复之前舍入的位。\n\n'
        '本表从 Bilby 干涉仪对象重新读取舍入前的原生 float64 信号，与保存的数据比较。'
        '报告 PSD 加权的误差 SNR，而不是用零点附近不稳定的逐采样相对误差。仅审计，不覆盖数据、不重新选模型。\n\n'
        '```json\n'+json.dumps(result, indent=2, ensure_ascii=False)+'\n```\n')
    shutil.copy2(__file__, destination/Path(__file__).name)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--destination', type=Path, required=True)
    a = p.parse_args()
    main(a.reference, a.destination)
