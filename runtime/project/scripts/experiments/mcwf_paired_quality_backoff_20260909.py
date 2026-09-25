#!/usr/bin/env python3
"""Require two valid profiles before changing the frozen NODUP waveform score."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
from pathlib import Path
import shutil
import sys
import numpy as np

P = Path('/root/autodl-tmp/gw-catalog')
sys.path.insert(0, str(P/'scripts/experiments'))
import mcwf_quality_profile_rejection_20260909 as rejection
parent = rejection.parent
st, cv, engine, base, n, r, co, h = (getattr(parent, k) for k in ('st','cv','engine','base','n','r','co','h'))
METHODS = ('NODUP-DIRECT-REPLAY', 'JOINTSTATE-FROZEN95',
           'PAIRED-QUALITY-GLOBAL-BACKOFF', 'PAIRED-QUALITY-CONDITIONAL-BACKOFF')
FIELDS = rejection.FIELDS + ('paired_quality_expert_used',)


def infer(frame, config):
    candidate = rejection.infer(frame, config)
    if config['method'] in METHODS[:2]:
        frame['paired_quality_expert_used'] = False
        return candidate
    original = h.isolated.ORIGINALS[config['deployment'], config['seed']]
    reference = st.infer(frame.copy(), {**original, 'method': METHODS[0]})
    use = st.states(frame) == 2
    result = tuple(np.where(use, a, b) for a, b in zip(candidate, reference))
    if any(not np.array_equal(a[~use], b[~use]) for a, b in zip(result, reference)):
        raise RuntimeError('NODUP inactive backoff changed')
    frame['paired_quality_expert_used'] = use
    frame['profile_envelope_used'] = frame.profile_envelope_used.to_numpy(bool) & use
    frame['actual_waveform_joint_BC'] = np.where(use, frame.actual_waveform_joint_BC, frame.parent_joint_BC)
    frame['new_prediction_used_in_score'] = use
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--stage', choices=('freeze', 'evaluate', 'real'), required=True)
    args = parser.parse_args()
    parent.ROOT, parent.PARENT = args.root, args.parent
    cv.ROOT, cv.PARENT = args.root, args.parent
    base.s.ROOT = h.ROOT = co.ROOT = r.ROOT = co.score.ROOT = args.root
    engine.ROOT = args.parent
    parent.METHODS = rejection.METHODS = METHODS
    parent.infer = infer
    r.install(); st.FIELDS = FIELDS; st.install_export()
    st.METHODS = (METHODS[0], 'PARENT_GLOBAL', 'UNUSED_REJECT')
    co.score.matrices = co.matrices
    n.METHODS, n.load_panel, n.infer = METHODS, parent.panel, infer
    if args.stage == 'freeze':
        parent.freeze()
        path = args.root/'contracts/PAIRED_QUALITY_BACKOFF_ADDENDUM.json'
        n.write_json(path, {'UTC': n.utc(), 'id': 'MCWF-NODUP-PAIRED-QUALITY-BACKOFF-49',
            'overrides_parent': 'If two endpoint profiles are quality-valid, use R48; otherwise exact NODUP waveform score.',
            'reason': 'A new profile-based expert need not alter cases where its own additional physical measurements are unavailable. This explicitly isolates the availability-dependent recalibration effect.',
            'two_valid_profiles_rule': 'Frozen R10 quality, no PE labels or event-ID gates.',
            'profile_attenuation': 'Within the two-valid subset, use R35; reduce to frozen R10/R12 score only if its existing true-source tail < 0.05 and its waveform score is lower.',
            'threshold': .05, 'new_hyperparameter_search': False,
            'same_both_runs': True, 'time_sky_outerweights_frozen': True,
            'no_old_encoder_Mc_q': True, 'no_total_mixture': True,
            'all_arms_reported': True, 'not_a_normalized_Bayes_factor': True,
            'adaptive_development': True, 'new_blind_confirmation': False,
            'goal_achieved': False})
        shutil.copy2(__file__, args.root/'scripts/paired_quality_backoff.py')
        n.write_json(args.root/'contracts/BACKOFF_RULE_FROZEN.json', {
            'UTC': n.utc(), 'addendum_sha256': n.sha(path),
            'configuration_sha256': n.sha(args.root/'configs/SELECTED_CONFIGURATIONS.json'),
            'runtime_sha256': n.sha(Path(__file__))})
    else:
        n.run(args.root, args.stage)
