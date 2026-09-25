#!/usr/bin/env python3
"""Append-only audit of frozen physical channels and final score arithmetic."""
from datetime import datetime, timezone
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import mcwf_pe_frontend_extension_v2_20260907 as e

dev, body = e.dev, e.body


def run(root):
    out = root / 'audit' / ('score_integrity_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    out.mkdir()
    rows, stale = [], []
    for trial in sorted((root / 'trials').iterdir()):
        for dep in e.DEPS:
            for es in dev.SEEDS:
                frozen_weights = dev.BASE.FROZEN_V93_WEIGHTS[dep][es]
                base = pd.read_parquet(e.BASE / f'evaluation/{dep}/seed_{es}/real_pairs.parquet').set_index('pair_key')
                intermediate = trial / f'evaluation/{dep}/seed_{es}/real_pairs.parquet'
                if intermediate.exists():
                    frame = pd.read_parquet(intermediate)
                    if 'final_score' in frame:
                        w = [frozen_weights[c] for c in ('waveform', 'time', 'sky')]
                        score = w[0] * frame.waveform_score + w[1] * frame.time_score + w[2] * frame.sky_raw_log_bf
                        error = float(abs(score - frame.final_score).max())
                        if error > 1e-8:
                            stale.append({'trial': trial.name, 'deployment': dep, 'seed': es,
                                'path': str(intermediate), 'stale_intermediate_final_score_max_difference': error,
                                'interpretation': 'Inherited baseline metadata; use verified final results, not this intermediate final_score/rank column.'})
                final = trial / f'results/CANDIDATE/{dep}/seed_{es}_C_fixed.parquet'
                if not final.exists():
                    continue
                f = pd.read_parquet(final).set_index('pair_key')
                if not f.index.is_unique or set(f.index) != set(base.index):
                    raise RuntimeError('Final pair scope changed: ' + str(final))
                old = base.loc[f.index]
                exact = {c: np.array_equal(f[c].to_numpy(), old[c].to_numpy())
                         for c in ('time_score', 'sky_raw_log_bf')}
                exact.update({f'explicit_weights_{c}': bool((f[f'weights_{c}'] == value).all())
                              for c, value in frozen_weights.items() if f'weights_{c}' in f})
                terms = [f.waveform_score * frozen_weights['waveform'],
                         f.time_score * frozen_weights['time'], f.sky_raw_log_bf * frozen_weights['sky']]
                errors = {name: float(abs(value - f[name]).max()) for name, value in zip(
                    ('waveform_contribution', 'time_contribution', 'sky_contribution'), terms)}
                errors['score_sum'] = float(abs(sum(terms) - f.final_score).max())
                ordered = f.sort_values('rank')
                ranks_valid = np.array_equal(ordered['rank'].to_numpy(), np.arange(1, len(f) + 1))
                descending = bool((np.diff(ordered.final_score.to_numpy()) <= 1e-10).all())
                passed = all(exact.values()) and max(errors.values()) < 1e-8 and ranks_valid and descending
                rows.append({'trial': trial.name, 'deployment': dep, 'seed': es, 'pairs': len(f),
                             'pass': passed, **exact, **errors, 'rank_sequence': ranks_valid,
                             'explicit_weight_columns_available': all('weights_' + c in f for c in frozen_weights),
                             'weight_audit': 'reported contributions must equal frozen-weight products, whether or not redundant weight columns were saved',
                             'score_descending': descending, 'final_sha256': dev.sha(final)})
    dev.csv_write(out / 'FINAL_SCORE_INTEGRITY.csv', pd.DataFrame(rows))
    dev.csv_write(out / 'STALE_INTERMEDIATE_METADATA.csv', pd.DataFrame(stale))
    summary = {'checked_final_tables': len(rows), 'failed_final_tables': sum(not r['pass'] for r in rows),
               'stale_intermediate_metadata_tables': len(stale),
               'historical_files_modified': False,
               'delivery_rule': 'Only verified final result tables contain authoritative updated ranks and contributions. Stale intermediate metadata must be renamed or excluded in derived delivery views, without altering original trials.'}
    dev.json_write(out / 'SUMMARY.json', summary)
    print(json.dumps({'audit': str(out), **summary}), flush=True)
    if summary['failed_final_tables']:
        raise RuntimeError('Some final score tables failed integrity audit')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    a = p.parse_args()
    run(a.root)
