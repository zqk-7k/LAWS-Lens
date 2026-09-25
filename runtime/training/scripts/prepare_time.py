#!/usr/bin/env python3
"""Training-only, run-matched one-dimensional time evidence shared by both arms."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import unified_ab as u


def run(root):
    u.verify(root)
    src, _, _, phys = u.modules(root)
    allrows = []
    for run in u.RUNS:
        out = root/'shared_time'/run
        out.mkdir(parents=True, exist_ok=True)
        freeze = out/'FREEZE.json'
        if freeze.exists():
            receipt = json.loads(freeze.read_text())
            assert u.sha(out/'lookup.json') == receipt['lookup_sha256']
            continue
        sources = pd.read_parquet(root/'plans'/run/'sources.parquet')
        lens = sources[(sources.role == 'main') & (sources.split == 'train') & (sources.family != 'unlensed')].copy()
        # Current main population has unique groups. Fail rather than silently
        # fitting equal row weights if a future plan introduces repeats.
        if lens.global_source_id.duplicated().any():
            raise RuntimeError('Repeated time-fit group needs explicit group-weighted density implementation')
        forbidden = sources[sources.split != 'train']
        assert not set(lens.global_source_id) & set(forbidden.global_source_id)
        np.testing.assert_allclose((lens.gps_b-lens.gps_a)/86400, lens.delay_days, rtol=0, atol=1e-10)
        schedule = src.load_schedule(root/'plans'/run/'live_schedule.csv')
        seed = int(u.stable('UAB-time-null', run) % 2**32)
        null = phys.draw_null_delays(schedule, 250000, np.random.default_rng(seed))
        model = phys.fit_time_likelihood_ratio(lens.delay_days.to_numpy(float), null, grid_size=2048, bandwidth_scale=1.)
        serial = {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in model.items()}
        u.write(out/'lookup.json', serial)
        lens[['source_uid', 'global_source_id', 'family', 'delay_days', 'gps_a', 'gps_b']].to_parquet(out/'fit_groups.parquet', index=False)
        np.save(out/'null_mc_delays.npy', null)
        receipt = {'utc': u.now(), 'run': run, 'lookup_sha256': u.sha(out/'lookup.json'),
                   'fit_groups_sha256': u.sha(out/'fit_groups.parquet'), 'independent_fit_groups': len(lens),
                   'each_group_total_weight': 1, 'test_validation_group_overlap': 0,
                   'used_by_arms': list(u.ARMS), 'used_by_all_model_seeds': True,
                   'density': 'archived one-dimensional log10-delay KDE likelihood ratio, bandwidth scale1',
                   'null_mc_draws': len(null), 'null_mc_draws_are_not_empirical_independent_pairs': True,
                   'no_SNR_ratio_or_2D_time': True, 'no_real_PE_or_official_input': True,
                   'selection': 'training-only fixed bandwidth; validation does not re-fit this lookup'}
        u.write(freeze, receipt); allrows.append(receipt)
        print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True)
    run(p.parse_args().root)
