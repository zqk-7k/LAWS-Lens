"""Post-freeze sensitivity ranks only; never replace the formal Nside512 ranks."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(out):
    comparisons, heads, checks, all_ranks = [], [], [], []
    for run in ('O3', 'O4a', 'O4b'):
        sky = pd.read_parquet(out/f'real_sky/{run}/pairs.parquet')
        sky['pair_key'] = ['--'.join(sorted((a, b))) for a, b in zip(sky.event_i, sky.event_j)]
        sky = sky.set_index('pair_key')
        for arm in ('A_NEUTRAL', 'B_CUE'):
            base = out/f'deployments/{run}/{arm}'
            source = base/'real_ranking/consensus_all_pairs.parquet'
            before = digest(source)
            formal = pd.read_parquet(source)
            formal = formal[formal.method.eq('three-channel')].set_index('pair_key').sort_index()
            per_seed = []
            for seed in (2026091721, 2026091722, 2026091723):
                spec = json.loads((base/f'calibration/score/seed_{seed}/SELECTED.json').read_text())
                weights = np.asarray(spec['final_fusion']['POSITIVE']['weights'])
                f = pd.read_parquet(base/f'real_ranking/seed_{seed}_all_scores.parquet').set_index('pair_key')
                for nside in (256, 512, 1024):
                    z = sky.loc[f.index, f'sky_log_bf_nside{nside}'].to_numpy()
                    score = np.column_stack((f.waveform_score.to_numpy(), f.time_score.to_numpy(), z))@weights
                    if nside == 512 and not np.allclose(score, f.final_score_POSITIVE, rtol=0, atol=1e-12):
                        raise RuntimeError('Formal score reconstruction failed')
                    row = pd.DataFrame(dict(pair_key=f.index, score=score, seed=seed, nside=nside))
                    row = row.sort_values(['score', 'pair_key'], ascending=[False, True])
                    row['rank'] = np.arange(1, len(row)+1)
                    per_seed.append(row)
            seeds = pd.concat(per_seed, ignore_index=True)
            consensus = {}
            for nside, g in seeds.groupby('nside'):
                z = g.groupby('pair_key').agg(rank_mean=('rank', 'mean'), rank_max=('rank', 'max'), score_mean=('score', 'mean')).reset_index()
                z = z.sort_values(['rank_mean', 'rank_max', 'score_mean', 'pair_key'], ascending=[True, True, False, True])
                z['consensus_rank'] = np.arange(1, len(z)+1)
                consensus[nside] = z.set_index('pair_key')
            exact = np.array_equal(consensus[512].sort_index().consensus_rank, formal.consensus_rank)
            if not exact:
                raise RuntimeError('Formal Nside512 consensus did not reproduce')
            for nside in (256, 1024):
                for budget in (10, 20, 50):
                    a, b = set(consensus[512].head(budget).index), set(consensus[nside].head(budget).index)
                    comparisons.append(dict(run=run, arm=arm, budget=budget, reference_nside=nside,
                        overlap=len(a & b), Jaccard=len(a & b)/len(a | b),
                        incoming=';'.join(sorted(b-a)), outgoing=';'.join(sorted(a-b)),
                        formal_nside=512, diagnostic_only=True, weights_reselected=False))
            for pair in consensus[512].head(50).index:
                z = sky.loc[pair]
                signs = np.sign([z[f'sky_log_bf_nside{n}'] for n in (256,512,1024)])
                heads.append(dict(run=run, arm=arm, pair_key=pair,
                    **{f'consensus_rank_{n}': int(consensus[n].loc[pair, 'consensus_rank']) for n in (256,512,1024)},
                    **{f'Z_sky_{n}': float(z[f'sky_log_bf_nside{n}']) for n in (256,512,1024)},
                    sign_stable=bool((signs == signs[0]).all()),
                    positive_at_all_resolutions=bool((signs > 0).all()),
                    abs_delta_512_1024=float(abs(z.sky_log_bf_nside512-z.sky_log_bf_nside1024))))
            seeds['run'], seeds['arm'], seeds['diagnostic_only'] = run, arm, True
            all_ranks.append(seeds)
            after = digest(source)
            if after != before:
                raise RuntimeError('Formal rank table was modified')
            checks.append(dict(run=run, arm=arm, formal_reproduced=True, original_rank_sha256=before, unchanged=True))
    tables = out/'tables'
    pd.DataFrame(comparisons).to_csv(tables/'real_resolution_decision_audit.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(heads).to_csv(tables/'real_top50_resolution_sensitivity.csv', index=False, encoding='utf-8-sig')
    pd.concat(all_ranks, ignore_index=True).to_parquet(tables/'real_diagnostic_resolution_ranks.parquet', index=False)
    record = dict(state='PASS', meaning='sensitivity calculation completed, not a convergence acceptance verdict',
        formal_nside=512, score_or_weight_changed=False, original_rank_tables=checks)
    (out/'contracts/REAL_DECISION_RESOLUTION_AUDIT.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--out', type=Path, required=True)
    main(p.parse_args().out)
