#!/usr/bin/env python3
"""Data-only checks; no learned model or real-candidate selection."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def load(root):
    spec = importlib.util.spec_from_file_location('uab', root/'scripts/unified_ab.py')
    mod = importlib.util.module_from_spec(spec)
    sys.modules['uab'] = mod
    spec.loader.exec_module(mod)
    return mod


def cue_metrics(frame, arm):
    x = frame[f'snr_{arm}'].to_numpy(float)
    bins = np.searchsorted([10, 12, 20], x, side='right')
    same_group = (bins[:, None]//2 == bins[None, :]//2)
    score = (same_group & (bins[:, None]%2 != bins[None, :]%2)).astype(float)
    uid = frame.source_uid.to_numpy()
    true = uid[:, None] == uid[None, :]
    np.fill_diagonal(true, False)
    i, j = np.triu_indices(len(frame), 1)
    auc = roc_auc_score(true[i, j], score[i, j])
    np.fill_diagonal(score, -np.inf)
    recall = {1: [], 10: []}
    for q in np.flatnonzero(true.any(axis=1)):
        partner = np.flatnonzero(true[q])[0]
        v = score[q, partner]
        better, tied = np.count_nonzero(score[q] > v), np.count_nonzero(score[q] == v)
        for k in recall:
            recall[k].append(float(np.clip((k-better)/tied, 0, 1)))
    return dict(arm=arm, events=len(frame), true_pairs=int(true[i, j].sum()), snr_rule_AUC=float(auc),
                expected_random_tie_R1=float(np.mean(recall[1])), expected_random_tie_R10=float(np.mean(recall[10])),
                interpretation='SNR-only construction diagnostic, NOT trained-waveform recall')


def main(root):
    u = load(root); u.verify(root)
    source = pd.read_parquet(root/'plans/source_population.parquet')
    assert source.source_uid.is_unique
    assert source.groupby('global_source_id').split.nunique().max() == 1
    assert source.groupby('global_source_id').role.nunique().max() <= 2
    aux_val = source[(source.role == 'aux') & (source.split == 'validation')]
    main_val = source[(source.role == 'main') & (source.split == 'validation')]
    assert not set(aux_val.global_source_id) & set(main_val.global_source_id)
    # Raw catalog tables must have aligned event IDs before positional joining.
    src, *_ = u.modules(root)
    ids = [pd.read_csv(next(src.GW_LMC_ROOT.glob('*_'+name+'Params.csv'))).event_id.to_numpy()
           for name in ('Source', 'Lens', 'Image')]
    assert np.array_equal(ids[0], ids[1]) and np.array_equal(ids[0], ids[2])
    tests, diagnostics = [], []
    previous = None
    for run in u.RUNS:
        events = pd.read_parquet(root/'plans'/run/'event_plan.parquet')
        noise = pd.read_parquet(root/'plans'/run/'noise_plan.parquet')
        assert noise.groupby('parent_uid').split.nunique().max() == 1
        joined = events.merge(noise[['noise_bank_index', 'split']], on='noise_bank_index', suffixes=('', '_noise'), validate='many_to_one')
        assert (joined.split == joined.split_noise).all()
        neutral = events[['source_uid', 'image_number', 'variant', 'snr_A_NEUTRAL', 'snr_B_CUE']]
        if previous is not None:
            pd.testing.assert_frame_equal(previous, neutral)
        previous = neutral
        for group, frame in events.groupby(['role', 'split', 'variant']):
            assert np.array_equal(np.sort(frame.snr_A_NEUTRAL), np.sort(frame.snr_B_CUE))
            a, b = u.snr_assignments(frame.to_dict('records'), group)
            assert np.array_equal(a, frame.snr_A_NEUTRAL) and np.array_equal(b, frame.snr_B_CUE)
        # Decision-free implementation audit uses validation only.
        val = events[(events.role == 'main') & (events.split == 'validation') & (events.variant == 0)]
        assert len(val) == 450
        for arm in u.ARMS:
            diagnostics.append({'run': run, **cue_metrics(val, arm)})
        schedules = src.load_schedule(root/'plans'/run/'live_schedule.csv')
        planned = pd.read_parquet(root/'plans'/run/'sources.parquet')
        from scripts.real_search.physical_common import in_live_segments
        for col in ('gps_a', 'gps_b'):
            assert np.asarray(in_live_segments(planned[col].to_numpy(float), schedules)).all()
        tests.append({'run': run, 'source_rows': len(planned), 'event_views': len(events),
                      'global_source_cross_split': 0, 'aux_main_validation_group_overlap': 0,
                      'noise_parent_cross_split': 0, 'SNR_multiset_equal': True,
                      'strict_calendar_pass': True, 'test_scored': False})
    sky = u.module(root/'scripts/bayestar_si_fixed.py', 'sky_units_test')
    import bilby
    import lal
    original = bilby.gw.conversion.bilby_to_lalsimulation_spins
    captured = []
    def mock(*args):
        captured.append(args)
        return (0., 0., 0., 0., 0., 0., 0.)
    try:
        bilby.gw.conversion.bilby_to_lalsimulation_spins = mock
        row = source.iloc[0]
        sky.spin_components(row)
        assert captured[0][7] == row.m1_det*lal.MSUN_SI
        assert captured[0][8] == row.m2_det*lal.MSUN_SI
    finally:
        bilby.gw.conversion.bilby_to_lalsimulation_spins = original
    historical = pd.read_csv(root/'manifests/PROTECTED_INPUTS.csv')
    changed = [str(row.path) for row in historical.itertuples() if u.sha(row.path) != row.sha256]
    assert not changed
    u.write(root/'contracts/DATA_UNIT_TESTS_PASS.json', {'utc': u.now(), 'tests': tests, 'SI_spin_conversion': True,
               'raw_GWLMC_table_ID_alignment': True, 'historical_input_hashes_unchanged': True,
               'validation_SNR_only_diagnostic_does_not_select_seed': True})
    pd.DataFrame(diagnostics).to_csv(root/'reports/SNR_ONLY_VALIDATION_DIAGNOSTIC.csv', index=False)
    print(json.dumps({'tests': 'PASS', 'validation_SNR_only': diagnostics}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--root', type=Path, required=True)
    main(parser.parse_args().root)
