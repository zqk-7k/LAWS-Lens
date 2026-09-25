#!/usr/bin/env python3
"""Pair locality, symmetry, probability and frozen-field checks on packaged scores."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import mcwf_replay_ordered_scores_20260907 as replay


def run(root):
    rows = []
    for dep in ('gwtc3', 'gwtc4'):
        for seed in (202607241, 202607242, 202607243):
            d = root / f'replay_inputs/{dep}/seed_{seed}'
            f = pd.read_parquet(d / 'baseline_pairs.parquet')
            a = np.load(d / 'ordered_mass_predictions.npz')
            cfg = json.loads((d / 'SELECTED_CONFIG.json').read_text())
            score = replay.pair_score(f, a, cfg)
            flipped = f.copy()
            flipped['idx_i'], flipped['idx_j'] = f.idx_j, f.idx_i
            other = replay.pair_score(flipped, a, cfg)
            symmetric = float(abs(score.waveform_score - other.waveform_score).max())
            ids = np.unique(np.r_[f.idx_i, f.idx_j])
            p = a['p'][ids]
            probability_error = float(abs(p.sum(1) - 1).max())
            if probability_error > 1e-12 or not np.isfinite(p).all() or p.min() < 0 or symmetric != 0:
                raise RuntimeError('Invalid predictive probability/symmetry')
            sub = f.iloc[::17]
            sliced = replay.pair_score(sub, a, cfg)
            locality = float(abs(score.waveform_score.iloc[::17].to_numpy() - sliced.waveform_score.to_numpy()).max())
            if locality != 0: raise RuntimeError('Pair scores depend on catalog composition')
            disabled = replay.pair_score(f, a, {**cfg, 'gamma': 0., 'beta': 0.})
            disabled_error = float(abs(disabled.waveform_score - f.waveform_score).max())
            if disabled_error != 0: raise RuntimeError('Nonneutral disabled increment')
            invalid = dict(a)
            invalid['boundary_mass'] = np.ones(len(a['p']))
            ood = replay.pair_score(f, invalid, cfg)
            ood_error = float(abs(ood.waveform_score - f.waveform_score).max())
            if ood_error != 0: raise RuntimeError('Boundary OOD is not neutral')
            if abs(score.ordered_mass_increment).max() > 4: raise RuntimeError('Evidence cap exceeded')
            if not np.array_equal(score.time_score, f.time_score) or not np.array_equal(score.sky_raw_log_bf, f.sky_raw_log_bf):
                raise RuntimeError('Frozen time/sky changed')
            rows.append({'deployment': dep, 'seed': seed, 'probability_sum_max_error': probability_error,
                         'symmetry_max_error': symmetric, 'pair_locality_max_error': locality,
                         'disabled_increment_max_error': disabled_error, 'boundary_ood_max_error': ood_error,
                         'evidence_cap': 4, 'time_sky_exact': True})
    result = {'pass': True, 'checks': rows, 'not_full_end_to_end_strain_test': True}
    path = root / 'audit/PAIR_LOCALITY_PROBABILITY_UNIT_TEST.json'
    if path.exists(): raise RuntimeError('Do not overwrite unit audit')
    path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    run(p.parse_args().root)
