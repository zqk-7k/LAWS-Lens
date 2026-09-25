#!/usr/bin/env python3
"""Build read-only audit snapshots including independent-data experiments."""
import argparse
import json
from pathlib import Path
import shutil
import sys

P = Path('/root/autodl-tmp/gw-catalog')
sys.path.insert(0, str(P/'scripts/experiments'))
import mcwf_failure_campaign_audit_20260909 as audit

EXTRA = [
    (22, 'mcwf_nodup_independent_profile_development_22_20260909T112429Z',
     'Independent noise reservation: incomplete,31O3blocks; no new injections generated'),
    (22, 'mcwf_nodup_independent_bulk_development_22b_20260909T114300Z',
     'New source/noise-disjoint calibration population with public run strain'),
    (23, 'mcwf_nodup_independent_waveform_calibration_23_20260909T113200Z',
     'Frozen classifier plan for incomplete22 population; not evaluated'),
    (23, 'mcwf_nodup_independent_waveform_calibration_23b_20260909T114850Z',
     'One joint waveform classifier on expanded independent fit/tune population'),
    (24, 'mcwf_nodup_partition_control_24_20260909T115700Z',
     'Current-development conditional-quality partition LR control'),
    (25, 'mcwf_nodup_conservative_profile_25_20260909T120400Z',
     'One-sided lower-envelope waveform refinement; not a normalized BF'),
    (26, 'mcwf_nodup_paired_profile_26_20260909T120700Z',
     'One-sided refinement only when both event profiles are quality-valid'),
    (27, 'mcwf_nodup_strength_predictive_27_20260909T121400Z',
     'Profile strength-conditioned predictive calibration, not Fisher or public SNR'),
    (28, 'mcwf_nodup_strength_integrated_28_20260909T121700Z',
     'Strength-conditioned predictive density integrated waveform scoring'),
    (29, 'mcwf_nodup_independent_predictive_29_20260909T130617Z',
     'Independent population predictive-error calibration of four frozen physical descriptions'),
    (30, 'mcwf_nodup_predictive_quadrature_30_20260909T131500Z',
     'Development-only quadrature audit; original convergence flag checked only right normalization'),
    (30, 'mcwf_nodup_predictive_quadrature_30b_20260909T131646Z',
     'Corrected audit requires normalization of BOTH quadrature rules;GL8versus16passes'),
    (31, 'mcwf_nodup_continuous_overlap_31_20260909T132030Z',
     'Continuous predictive overlap and independent single-waveform calibration'),
    (32, 'mcwf_nodup_validation_fusion_32_20260909T133016Z',
     'Separate validation-only fusion-weight control;rawtime/sky/waveformunchanged'),
    (33, 'mcwf_nodup_selected_predictive_33_20260909T134300Z',
     'Predeclared common predictive-kind selection;one independent waveform classifier'),
    (34, 'mcwf_nodup_selected_predictive_joint_34_20260909T135300Z',
     'Independent predictive jointBC in fixed NODUP coefficients;global and paired conservative controls'),
    (35, 'mcwf_nodup_quality_state_joint_35_20260909T141547Z',
     'JointBC calibration conditional on zero/one/two valid endpoint profiles;state probability offsets'),
    (36, 'mcwf_nodup_minimal_joint_36_20260909T142148Z',
     'Only cosine and full jointBC inputs;no independently weighted mass/conditional factors or absolute mass'),
    (37, 'mcwf_nodup_mass_confidence_37_20260909T143207Z',
     'Mass marginal source-confidence attenuation replaces joint-tail attenuation;one unchanged jointBC LR'),
    (38, 'mcwf_nodup_confidence_validation_38_20260909T144547Z',
     'Four frozen confidence levels and validation-only selection;outerweights unchanged'),
    (39, 'mcwf_nodup_waveform_internal_39_20260909T150146Z',
     'Validation-only internal waveform gamma/beta and confidence controls;not outerfusion retuning'),
    (40, 'mcwf_nodup_inspiral_profile_40_20260909T150441Z',
     'Simulation-only80/160Hz upper-cutoff pilot: neither cutoff passes both runs;no real ranking'),
    (41, 'mcwf_nodup_physical_pair_41_20260909T151141Z',
     'One empirical correlated physical-profile distance classifier;jointBC is not used in active expert'),
    (42, 'mcwf_nodup_quality_matched_42_20260909T151630Z',
     'Explicit quality-state-matched null reference;omitting state offsets is a changed reference,not a bug fix'),
    (43, 'mcwf_nodup_spectral_consistency_43_20260909T152910Z',
     'Unit-only attempt:float32 amplitude rescaling introduced rounding;batch not started'),
    (43, 'mcwf_nodup_spectral_consistency_43b_20260909T153400Z',
     'Float64 invariance unit repair;empirical PyCBC spectral covariate,not formal search chi-square'),
    (44, 'mcwf_nodup_spectral_joint_44_20260909T153200Z',
     'Simulation-selected spectral covariate replaces mass-predictive scale;fixed joint-state scoring'),
    (45, 'mcwf_nodup_approximant_pilot_45_20260909T153430Z',
     'Simulation-only IMRPhenomD versus IMRPhenomXAS equal-spin profile pilot;not newPE'),
    (46, 'mcwf_nodup_paired_intrinsic_46_20260909T153849Z',
     'One paired-valid cosine/massBC/conditionalBC expert with exact R35 fallback for other states'),
    (47, 'mcwf_nodup_quality_envelope_47_20260909T154403Z',
     'Frozen conservative lower envelope of R35 quality-state and R10/R12 physical-profile waveform terms'),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    audit.ROOT = args.root
    audit.ROUNDS = audit.ROUNDS + EXTRA
    audit.collect()
    audit.n.write_json(args.root/'contracts/REPORTING_EXTENSION.json', {
        'UTC': audit.n.utc(), 'code_sha256': audit.n.sha(Path(__file__)),
        'parent_audit_sha256': audit.n.sha(Path(audit.__file__)),
        'round_labels_repeat_for_independent_retries': True,
        'folder_name_timestamp_is_opaque_id': True,
        'actual_time_authority': 'UTC fields in start, analysis and completion contracts',
        'not_a_success_declaration': True})
    shutil.copy2(__file__, args.root/'scripts/campaign_reporting.py')


if __name__ == '__main__':
    main()
