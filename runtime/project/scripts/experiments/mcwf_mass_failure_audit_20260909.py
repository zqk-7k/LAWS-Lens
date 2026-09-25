#!/usr/bin/env python3
"""Read-only audit of predictive mass compatibility; no candidate rescoring."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
import argparse
from datetime import datetime, timezone
import importlib.util
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd
import torch

P = Path('/root/autodl-tmp/gw-catalog')
sys.path.insert(0, str(P / 'scripts/experiments'))
import mcwf_conditional_intrinsics_20260908 as model
spec = importlib.util.spec_from_file_location('nodup', P / 'scripts/experiments/mcwf_nodup_nomix_20260909T071108Z.py')
n = importlib.util.module_from_spec(spec)
spec.loader.exec_module(n)
PAIR = 'GW191103_012549--GW191105_143521'
LOW = P / 'results/mcwf_multirate_lowband_exploratory_20260908T145551Z'
INTR = P / 'results/mcwf_conditional_intrinsics_exploratory_20260908T154330Z'
PRIOR = P / 'results/mcwf_nodup_nomix_01_20260909T071108Z'


def main(root):
    if root.exists():
        raise RuntimeError('New independent output directory required')
    for folder in ('audit', 'contracts', 'scripts', 'models', 'predictions', 'calibration',
                   'tables', 'reports', 'figures', 'logs', 'manifest', 'results'):
        (root / folder).mkdir(parents=True)
    shutil.copy2(__file__, root / 'scripts/forensic_audit.py')
    contract = {'id': 'MCWF-NODUP-MASS-RELIABILITY-02', 'utc': datetime.now(timezone.utc).isoformat(),
                'goal_achieved': False, 'phase': 'read_only_failure_audit', 'baseline': str(PRIOR),
                'frozen': ['time_score', 'sky_raw_log_bf', 'scope', 'historical inputs'],
                'forbidden_score_inputs': ['event identifiers', 'public PE', 'official FPP/phase'],
                'excluded': ['old encoder predicted Mc/q scores', 'old/new total-score mixing'],
                'development_status': 'Adaptive real-catalog development, not blind confirmation',
                'case': PAIR, 'case_scope': 'Diagnostic, never event-specific scoring rule',
                'status': n.STATUS}
    n.write_json(root / 'contracts/DEVELOPMENT_OBJECTIVE.json', contract)
    paths = set(n.input_files())
    paths.add(PRIOR / 'contracts/ANALYSIS_CONTRACT.json')
    paths.add(PRIOR / 'configs/SELECTED.json')
    paths = {p for p in paths if p.exists()}
    event_rows, pair_rows, all_rows, data_rows = [], [], [], []
    densities = {}
    for dep in n.DEPS:
        ext = pd.read_parquet(n.t.EXTERNAL / f'{dep}_external_reference.parquet').set_index('pair_key')
        for seed, slot in zip(n.SEEDS, n.t.MODEL_SLOTS):
            frame = n.load_panel(dep, seed, 'real')
            cp_path = LOW / f'models/MULTIRATE/{dep}/seed_{slot}/selected.pt'
            pred_path = LOW / f'predictions/MULTIRATE/{dep}/model_{slot}_eval_{seed}/real.npz'
            paths.update((cp_path, pred_path))
            ck = torch.load(cp_path, map_location='cpu', weights_only=False)
            a = np.load(pred_path)
            p = a['p']
            i, j = frame.idx_i.to_numpy(int), frame.idx_j.to_numpy(int)
            massbc = np.sum(np.sqrt(p[i] * p[j]), axis=1)
            out = frame[['pair_key', 'idx_i', 'idx_j', 'joint_BC', 'joint_logbc', 'embedding_only',
                         'PATH875_waveform', 'time_score', 'sky_raw_log_bf']].copy()
            out['mass_predictive_BC'] = massbc
            out['seed'], out['deployment'] = seed, dep
            all_rows.append(out)
            n.write_csv(root / f'audit/{dep}_{seed}_real_predictive_overlap.csv', out)
            if dep == 'gwtc3':
                row = frame.loc[frame.pair_key == PAIR].iloc[0]
                public = ext.loc[PAIR]
                common = {'seed': seed, 'slot': slot, 'pair_key': PAIR,
                          'predicted_mass_BC': float(massbc[frame.pair_key.eq(PAIR).to_numpy()][0]),
                          'predicted_joint_BC': float(row.joint_BC),
                          'public_Mc_BC': float(public.pe_mc_bhattacharyya_coefficient),
                          'public_DMc': float(public.pe_mc_standardized_distance),
                          'embedding_cosine': float(row.embedding_only)}
                pair_rows.append(common)
                for side in ('i', 'j'):
                    idx = int(row['idx_' + side])
                    prob = p[idx]
                    cdf = np.r_[0., prob.cumsum()]
                    qs = np.exp(np.interp([.05, .16, .5, .84, .95], cdf, model.old.EDGES))
                    e = {'seed': seed, 'slot': slot, 'event': row['event_' + side], 'idx': idx,
                         'q05': qs[0], 'q16': qs[1], 'q50': qs[2], 'q84': qs[3], 'q95': qs[4],
                         'model_temperature': float(ck['temperature']), 'checkpoint_epoch': ck['epoch'],
                         'public_median': float(public['pe_mc_median_' + side]),
                         'public_sigma': float(public['pe_mc_sigma_' + side]),
                         'boundary_mass': float(a['outside'][idx]),
                         'entropy_nats': float(-np.sum(prob * np.log(prob.clip(1e-300)))),
                         'KL_to_training_prior': float(np.sum(prob * np.log(prob.clip(1e-300) / ck['prior'])))}
                    event_rows.append(e)
                    densities[f'{seed}_{side}'] = prob
            for split in ('train', 'validation'):
                meta_path = n.t.PREVIOUS / f'expanded_data/{dep}/{split}/event_metadata.parquet'
                if seed == n.SEEDS[0]:
                    meta = pd.read_parquet(meta_path)
                    paths.add(meta_path)
                    data_rows.append({'deployment': dep, 'split': split, 'events': len(meta),
                                      'source_parents': meta.source_uid.nunique(),
                                      'noise_blocks': meta.noise_bank_index.nunique(),
                                      'columns': ','.join(meta.columns)})
            dev_path = cp_path.parent / 'COMPLETE.json'
            paths.add(dev_path)
    n.write_csv(root / 'audit/KEY_PAIR.csv', pair_rows)
    n.write_csv(root / 'audit/KEY_EVENTS.csv', event_rows)
    n.write_csv(root / 'audit/TRAINING_INVENTORY.csv', data_rows)
    np.savez_compressed(root / 'audit/KEY_PAIR_PREDICTIVE_MASS.npz', log_edges=model.old.EDGES, **densities)
    protected = [{'path': str(p), 'sha256': n.sha(p), 'bytes': p.stat().st_size} for p in sorted(paths)]
    n.write_csv(root / 'manifest/INPUT_SHA256.csv', protected)
    print(pd.DataFrame(pair_rows).to_string(index=False), flush=True)
    print(pd.DataFrame(event_rows).to_string(index=False), flush=True)
    print(pd.DataFrame(data_rows).to_string(index=False), flush=True)
    print('AUDIT_ROOT', root, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    main(parser.parse_args().root)
