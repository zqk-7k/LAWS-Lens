#!/usr/bin/env python3
"""Portable CPU replay from frozen event mass predictions and baseline pair scores."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def pair_score(frame, prediction, config):
    p, prior = prediction['p'], prediction['prior']
    i, j = frame.idx_i.to_numpy(int), frame.idx_j.to_numpy(int)
    bc, overlap = np.empty(len(i)), np.empty(len(i))
    for start in range(0, len(i), 2048):
        sl = slice(start, start + 2048)
        product = p[i[sl]] * p[j[sl]]
        bc[sl] = np.sqrt(product).sum(1)
        overlap[sl] = (product / prior).sum(1)
    bc = bc.clip(1e-15, 1.)
    x = np.log(overlap.clip(1e-300))
    ood = (prediction['boundary_mass'][i] > .25) | (prediction['boundary_mass'][j] > .25)
    ref = np.asarray(config['mass_reference'])
    a = -np.log(bc)
    ptail = (1 + len(ref) - np.searchsorted(ref, a, side='left')) / (len(ref) + 1)
    penalty = np.where(ood, 0., np.minimum(np.log(ptail / .05), 0.))
    cal = config['prior_overlap']
    raw = np.interp(x, cal['knots'], cal['loglr'])
    outside = ood | (x < cal['minimum']) | (x > cal['maximum'])
    raw = np.where(outside | (ptail < .05), np.minimum(raw, 0.), raw)
    increment = np.where(ood, 0., raw.clip(-4, 4))
    result = frame.copy()
    result['FRT_baseline_waveform_score'] = frame.waveform_score
    result['waveform_score'] = frame.waveform_score + config['gamma'] * penalty + config['beta'] * increment
    result['ordered_mass_bc'], result['ordered_mass_prior_feature'] = bc, x
    result['ordered_mass_ood'] = ood
    result['ordered_mass_penalty'], result['ordered_mass_increment'] = penalty, increment
    result['ordered_gamma'], result['ordered_beta'] = config['gamma'], config['beta']
    return result


def rank(frame, weight, seed, method):
    f = frame.copy()
    f['waveform_contribution'] = weight['waveform'] * f.waveform_score
    f['time_contribution'] = weight['time'] * f.time_score
    f['sky_contribution'] = weight['sky'] * f.sky_raw_log_bf
    f['final_score'] = f.waveform_contribution + f.time_contribution + f.sky_contribution
    f['seed'], f['method'] = seed, method
    f = f.sort_values('final_score', ascending=False, kind='stable').reset_index(drop=True)
    f['rank'] = np.arange(1, len(f) + 1)
    return f


def consensus(frames):
    f = pd.concat(frames, ignore_index=True)
    g = f.groupby(['pair_key', 'event_i', 'event_j'], as_index=False).agg(
        rank_mean=('rank', 'mean'), rank_max=('rank', 'max'), rank_sd=('rank', 'std'),
        final_score_mean=('final_score', 'mean'), waveform_score_mean=('waveform_score', 'mean'))
    g = g.sort_values(['rank_mean', 'rank_max', 'final_score_mean'], ascending=[True, True, False], kind='stable').reset_index(drop=True)
    g.insert(0, 'consensus_rank', np.arange(1, len(g) + 1))
    return g


def run(root, output):
    if output.exists():
        raise RuntimeError('Choose a new replay output directory')
    output.mkdir(parents=True)
    weights = json.loads((root / 'contracts/OUTER_WEIGHTS.json').read_text())
    checks = []
    for dep in ('gwtc3', 'gwtc4'):
        results = {'C_fixed': [], 'waveform_only': []}
        for es in (202607241, 202607242, 202607243):
            src = root / f'replay_inputs/{dep}/seed_{es}'
            f = pd.read_parquet(src / 'baseline_pairs.parquet')
            a = np.load(src / 'ordered_mass_predictions.npz')
            cfg = json.loads((src / 'SELECTED_CONFIG.json').read_text())
            c = pair_score(f, a, cfg)
            for method in results:
                w = weights[dep][str(es)] if method == 'C_fixed' else {'waveform': 1., 'time': 0., 'sky': 0.}
                ranked = rank(c, w, es, method)
                expected = pd.read_parquet(root / f'real/{dep}/candidate_seed_{es}_{method}.parquet').set_index('pair_key')
                aligned = ranked.set_index('pair_key').reindex(expected.index)
                error = float(abs(aligned.final_score - expected.final_score).max())
                rankdiff = int(abs(aligned['rank'] - expected['rank']).max())
                check = {'deployment': dep, 'seed': es, 'method': method,
                         'maximum_score_difference': error, 'maximum_rank_difference': rankdiff,
                         'time_exact': bool(np.array_equal(aligned.time_score, expected.time_score)),
                         'sky_exact': bool(np.array_equal(aligned.sky_raw_log_bf, expected.sky_raw_log_bf))}
                checks.append(check)
                if error > 1e-12 or rankdiff or not check['time_exact'] or not check['sky_exact']:
                    raise RuntimeError(str(check))
                path = output / dep
                path.mkdir(exist_ok=True)
                ranked.to_parquet(path / f'seed_{es}_{method}.parquet', index=False)
                results[method].append(ranked)
        for method, frames in results.items():
            g = consensus(frames)
            expected = pd.read_parquet(root / f'real/{dep}/candidate_consensus_{method}_all_pairs.parquet')
            if g.pair_key.tolist() != expected.pair_key.tolist():
                raise RuntimeError('Consensus ordering mismatch')
            g.to_csv(output / dep / f'consensus_{method}.csv', index=False, encoding='utf-8-sig')
    (output / 'REPLAY_AUDIT.json').write_text(json.dumps({'pass': True, 'checks': checks,
        'meaning': 'Exact score/rank replay from predictions, not end-to-end strain re-encoding or independent PE validation'}, indent=2))
    print(json.dumps({'pass': True, 'seed_method_checks': len(checks), 'output': str(output)}))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    run(a.root, a.output)
