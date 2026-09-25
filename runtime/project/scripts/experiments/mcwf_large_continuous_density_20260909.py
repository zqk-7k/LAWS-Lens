#!/usr/bin/env python3
"""Increase independent training sources, not posterior confidence by hand."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
import argparse
from pathlib import Path
import shutil
import sys
import pandas as pd

P = Path('/root/autodl-tmp/gw-catalog')
sys.path.insert(0, str(P / 'scripts/experiments'))
import mcwf_subgrid_density_20260909 as d
import mcwf_subgrid_evaluate_20260909 as score
import mcwf_large_multirate_20260908 as population

LARGE = P / 'results/mcwf_large_multirate_exploratory_20260908T171624Z'
SEEDS = (2026090991, 2026090992, 2026090993)
METHODS = ('NODUP-DIRECT-REPLAY', 'LARGE-JOINT-FIXED', 'LARGE-JOINT-VAL')
ROOT = None
n, r = d.n, d.reliability


def training_data(root, dep, split, kind):
    return population.training_data(LARGE, dep, split, kind)


def freeze():
    if ROOT.exists():
        raise RuntimeError('Independent directory required')
    for name in ('contracts', 'models', 'predictions', 'calibration', 'configs', 'tables', 'audit',
                 'reports', 'scripts', 'logs', 'manifest', 'results', 'figures', 'cache'):
        (ROOT / name).mkdir(parents=True)
    n.write_json(ROOT / 'contracts/ANALYSIS_CONTRACT.json', {'id': 'MCWF-NODUP-LARGE-CONTINUOUS-06',
        'UTC': n.utc(), 'status': n.STATUS, 'goal_achieved': False, 'same_both_runs': True,
        'rationale': 'Lowmass lowSNR source posterior coverage isalreadynear90percent,so cannotjustify artificialnarrowing. Test increasedindependenttrainingdata withunchangedcontinuousNLLdensity.',
        'population': 'Existing12288parents160noiseblocks,8views/source.Expanded4096plusadditional8192.Separate512developmentparents32noiseblocks.',
        'population_origin': str(LARGE), 'new_strain_generation': False,
        'prior_use': 'Largerpopulation previouslyusedfor discretedensityinolderpipeline;thisisnot afreshpopulationorindependentconfirmation.',
        'model': 'ExactlySUBGRID03 continuousGaussianmixture,CDFnormalization,54x253waveformresponses;warm originalMULTIRATEcheckpoint,not oldencoderMc/q outputs.',
        'training': {'seeds': list(SEEDS), 'epochs': d.EPOCHS, 'learning_rate': '1e-4 cosine1e-5',
                     'selection': 'minimum developmentcontinuousNLL,T bydevelopmentNLL;epoch0included',
                     'batch': '128sources twoindependent views', 'compute_note': 'Moreoptimizationstepsper epochbecausepopulationlarger;not acompute-matchedcausalablation'},
        'score': 'Samesinglejointdensityandcalibrationas03;no oldMc/q orOMC,nototalalpha;fixedpureupstreamouterweights',
        'scoring_selection': 'same03 gamma/beta gridandvalidationmetrics,fixedandretunedarmsbothreported',
        'frozen': ['oldencoder', 'time_score', 'sky_raw_log_bf', 'scope', 'history', 'paper'],
        'no_PE_or_official_modelselection': True, 'adaptive_development': True, 'blind_claim': False})
    rows = pd.read_csv(P / 'results/mcwf_nodup_mass_reliability_02_20260909T154100Z/manifest/INPUT_SHA256.csv')
    additions = []
    for path in sorted((LARGE / 'contracts').glob('*')):
        if path.is_file():
            additions.append({'path': str(path), 'sha256': n.sha(path), 'bytes': path.stat().st_size})
    n.write_csv(ROOT / 'manifest/INPUT_SHA256.csv', pd.concat([rows, pd.DataFrame(additions)], ignore_index=True))
    for dep in n.DEPS:
        x, a = population.training_data(LARGE, dep, 'train', 'MULTIRATE')
        v, b = population.training_data(LARGE, dep, 'validation', 'MULTIRATE')
        if a.source_uid.nunique() != 12288 or b.source_uid.nunique() != 512:
            raise RuntimeError('Population identity mismatch')
        if set(a.source_uid) & set(b.source_uid) or set(a.noise_bank_index) & set(b.noise_bank_index):
            raise RuntimeError('Source/noise split violation')
        n.write_json(ROOT / f'audit/{dep}_POPULATION.json', {'training_rows': len(a), 'training_sources': a.source_uid.nunique(),
            'training_noise_blocks': a.noise_bank_index.nunique(), 'development_rows': len(b),
            'development_sources': b.source_uid.nunique(), 'development_noise_blocks': b.noise_bank_index.nunique(),
            'source_intersection': 0, 'noise_intersection': 0, 'feature_shape': list(x.shape)})
        del x, v
    shutil.copy2(__file__, ROOT / 'scripts/large_continuous_density.py')
    n.write_json(ROOT / 'contracts/START_FREEZE.json', {'UTC': n.utc(),
        'contract_sha256': n.sha(ROOT / 'contracts/ANALYSIS_CONTRACT.json'), 'script_sha256': n.sha(Path(__file__))})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--stage', choices=('freeze', 'train', 'freeze_score', 'calibrate', 'evaluate', 'real'), required=True)
    args = parser.parse_args()
    ROOT = d.ROOT = r.ROOT = score.ROOT = args.root
    score.METHODS = METHODS
    r.install()
    n.METHODS, n.infer, n.load_panel = METHODS, score.infer, score.load_panel
    if args.stage == 'freeze':
        freeze()
    elif args.stage == 'train':
        d.mult.training_data = training_data
        for dep in n.DEPS:
            for slot, seed in zip(n.t.MODEL_SLOTS, SEEDS):
                d.train_one(dep, slot, seed)
    elif args.stage in ('freeze_score', 'calibrate'):
        getattr(score, args.stage)()
    else:
        n.run(ROOT, args.stage)
