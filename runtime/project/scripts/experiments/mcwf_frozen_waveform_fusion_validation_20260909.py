#!/usr/bin/env python3
"""Separate validation-only fusion audit; no raw physical score changes."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import json
from pathlib import Path
import shutil
import sys
import numpy as np
import pandas as pd

P = Path('/root/autodl-tmp/gw-catalog')
sys.path.insert(0, str(P/'scripts/experiments'))
import mcwf_paired_profile_refinement_20260909 as p
n, r, co, score = p.n, p.r, p.co, p.score
PARENT = P/'results/mcwf_nodup_paired_profile_26_20260909T120700Z'
ROOT = None
METHODS = ('NODUP-DIRECT-REPLAY', 'PAIRED-FROZEN-FUSION', 'PAIRED-VALIDATION-FUSION')
SPECS = {(v['deployment'], v['seed'], v['method']): v for v in
         json.loads((PARENT/'configs/SELECTED_CONFIGURATIONS.json').read_text())}


def infer(frame, config):
    dep, seed = config['deployment'], config['seed']
    key = METHODS[0] if config['method'] == METHODS[0] else 'PAIRED-PROFILE-GLOBAL'
    return p.infer(frame, SPECS[dep, seed, key])


def freeze():
    if ROOT.exists():
        raise RuntimeError('Independent output required')
    for folder in ('contracts', 'configs', 'tables', 'audit', 'scripts', 'logs', 'manifest',
                   'reports', 'results', 'figures'):
        (ROOT/folder).mkdir(parents=True)
    n.write_json(ROOT/'contracts/ANALYSIS_CONTRACT.json', {'UTC': n.utc(),
        'id': 'MCWF-NODUP-FUSION-CALIBRATION-32', 'goal_achieved': False, 'status': n.STATUS,
        'adaptive_development': True, 'not_a_waveform_only_change': True,
        'motivation': 'Changing waveform evidence while retaining upstream relative fusion weights can alter the balance with untouched physical scores. Test explicitly whether validation-only fusion selection addresses this. Do not disguise weight changes as unchanged scoring.',
        'parent': str(PARENT), 'same_algorithm_both_runs': True,
        'all_raw_channels_frozen': ['paired-profile-global waveform', 'one-dimensional time', 'sky'],
        'separate_control': 'Frozen originalNODUP weights versus new validation-selected weights. No method from rounds2-31 is overwritten or rejudged.',
        'grid': 'All231nonnegative simplex weights onstep.05,includingzero. Report zero-weight outcomes explicitly;do not callzero-waveformselection a successful waveform correction.',
        'validation': 'Archived BAYESTAR validation for eachseed/run only. Known reused development;not a new independent confirmation.',
        'guard': 'Fusion-only existing .02R10/.005AP/10percentF50/F90 guards must pass against BOTH NODUP and PATH875 validation baselines. Rawwaveformranks unchanged and their independent shortcomings remainreported.',
        'priority': ['guard_both_baselines', 'F50', 'F90', '-AUPRC', '-R10', '-R1',
                     'distance_to_normalized_frozen_weights', 'lexicographic_weights'],
        'failure': 'If no gridpointpasses,reportbestpriorityfailedchoiceasexploratoryFAIL,notupgraded.',
        'no_real_selection': True, 'no_old_Mc_q': True, 'no_total_score_mixture': True,
        'one_sided_waveform_limitation': 'Inherited lower envelope is a validated triage heuristic,not a normalized Bayes factor or PE posterior.',
        'frozen': ['encoders', 'profilefits', 'predictivedensities', 'waveformcalibration', 'rawtime',
                   'rawsky', 'eventscope', 'historicalrankings', 'paper'],
        'references': ['https://arxiv.org/abs/1506.02169', 'https://www.jmlr.org/papers/v11/cawley10a.html'],
        'reference_limit': 'These motivate calibrated scores and independent selection;they do not establish this grid or weighting as physically optimal.'})
    paths = [PARENT/'configs/SELECTED_CONFIGURATIONS.json',
             PARENT/'contracts/PAIRED_RULE_FROZEN.json', Path(p.__file__)]
    n.write_csv(ROOT/'manifest/INPUT_SHA256.csv', [{'path': str(f), 'sha256': n.sha(f),
        'bytes': f.stat().st_size} for f in paths])
    shutil.copy2(__file__, ROOT/'scripts/frozen_waveform_fusion_validation.py')
    n.write_json(ROOT/'contracts/START_FREEZE.json', {'UTC': n.utc(),
        'script_sha256': n.sha(Path(__file__)),
        'contract_sha256': n.sha(ROOT/'contracts/ANALYSIS_CONTRACT.json')})


def calibrate():
    configs, grids, selected, units = [], [], [], []
    for dep in n.DEPS:
        for seed in n.SEEDS:
            original = p.parent.isolated.ORIGINALS[dep, seed]
            frame = co.load_panel(dep, seed, 'validation')
            base = {**original, 'method': METHODS[0]}
            zbase = infer(frame, base)[0]
            fbase = n.cf.fast_metrics(frame, n.cf.channels(frame, zbase)@np.asarray(original['weights']))
            pathbase = n.cf.fast_metrics(frame, frame.PATH875_final_score.to_numpy())
            old = SPECS[dep, seed, 'PAIRED-PROFILE-GLOBAL']
            fixed = {**old, 'method': METHODS[1]}
            z = infer(frame, fixed)[0]
            archived = pd.read_parquet(PARENT/f'results/PAIRED-PROFILE-GLOBAL/{dep}/seed_{seed}/validation/pairs.parquet')
            if not np.array_equal(frame[['idx_i', 'idx_j']].to_numpy(), archived[['idx_i', 'idx_j']].to_numpy()) or not np.array_equal(z, archived.waveform_score.to_numpy()):
                raise RuntimeError('Frozen waveform score changed')
            expected = np.asarray(old['weights']); expected = expected/expected.sum()
            channels = n.cf.channels(frame, z)
            options = []
            for a in range(21):
                for b in range(21-a):
                    weights = np.array([a, b, 20-a-b], float)/20
                    metric = n.cf.fast_metrics(frame, channels@weights)
                    guard = n.cf.guard(metric, fbase) and n.cf.guard(metric, pathbase)
                    row = {'deployment': dep, 'seed': seed, 'waveform': weights[0],
                        'time': weights[1], 'sky': weights[2], 'both_guard': guard,
                        'distance': float(np.sum((weights-expected)**2)), **metric}
                    options.append(row); grids.append(row)
            valid = [v for v in options if v['both_guard']]
            win = min(valid or options, key=lambda v: (v['false_at_recall_0p5'], v['false_at_recall_0p9'],
                -v['average_precision'], -v['macro_r_at_10'], -v['macro_r_at_1'], v['distance'],
                v['waveform'], v['time'], v['sky']))
            chosen = {**old, 'method': METHODS[2], 'weights': [win[x] for x in ('waveform', 'time', 'sky')],
                'tune_guard': bool(valid), 'tune_metrics': win}
            if not np.array_equal(infer(frame, chosen)[0], z):
                raise RuntimeError('Fusion search changed waveform channel')
            configs.extend([base, fixed, chosen]); selected.append(win)
            units.append({'deployment': dep, 'seed': seed, 'waveform_max_change': 0.,
                'raw_time_and_sky_changed': False, 'grid_points': len(options), 'real_PE_official_used': False})
            print('FUSION_VALIDATION_SELECTION', dep, seed, win, flush=True)
    n.write_csv(ROOT/'tables/FUSION_VALIDATION_GRID.csv', grids)
    n.write_csv(ROOT/'tables/FUSION_VALIDATION_SELECTION.csv', selected)
    n.write_csv(ROOT/'audit/RAW_CHANNEL_INVARIANCE.csv', units)
    n.write_json(ROOT/'configs/SELECTED_CONFIGURATIONS.json', configs)
    n.write_json(ROOT/'contracts/CONFIGURATIONS_FROZEN.json', {'UTC': n.utc(),
        'file': 'configs/SELECTED_CONFIGURATIONS.json', 'sha256': n.sha(ROOT/'configs/SELECTED_CONFIGURATIONS.json'),
        'real_or_test_selection': False, 'no_total_mixture': True})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--stage', choices=('freeze', 'calibrate', 'evaluate', 'real'), required=True)
    args = parser.parse_args(); ROOT = args.root
    p.parent.METHODS = p.METHODS
    p.parent.ROOT = co.ROOT = r.ROOT = score.ROOT = ROOT
    r.install(); score.matrices = co.matrices
    n.METHODS, n.load_panel, n.infer = METHODS, co.load_panel, infer
    if args.stage in ('freeze', 'calibrate'):
        globals()[args.stage]()
    else:
        n.run(ROOT, args.stage)
