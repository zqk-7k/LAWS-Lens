#!/usr/bin/env python3
"""Assemble a clean, non-adopted release without modifying source experiments."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys
import numpy as np
import pandas as pd
import torch
import mcwf_ordered_fresh_confirmation_20260907 as run
import mcwf_ordered_mass_predictor_20260907 as ordered

dev, e, body = run.dev, run.e, run.body


def output_root(root):
    return root / 'release_MCWF_UNIFIED_OMC_DEVCONF'


def copy(src, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)


def prepare(root):
    conf = run.verify(root)
    out = output_root(root)
    if out.exists():
        raise RuntimeError('Release already exists; do not overwrite')
    out.mkdir()
    for name in ('contracts', 'real', 'models', 'replay_inputs', 'tables', 'reports', 'figures', 'scripts', 'audit', 'manifest'):
        (out / name).mkdir()
    trial = root / 'trials' / run.TRIAL
    copy(run.folder(root) / 'contracts/FRESH_FREEZE.json', out / 'contracts/FRESH_FREEZE.json')
    copy(root / 'contracts/EXPERIMENT_CONTRACT.json', out / 'contracts/EXPERIMENT_CONTRACT.json')
    copy(root / 'manifest/PROTECTED_INPUT_SHA256.csv', out / 'manifest/HISTORICAL_PROTECTED_INPUT_SHA256.csv')
    copy(e.BASE / 'reports/MCWF_UNIFIED_RNC_FRT_OVERALL_METHOD_CN_20260907.md', out / 'reports/BASELINE_OVERALL_METHOD_CN.md')
    dev.json_write(out / 'contracts/OUTER_WEIGHTS.json', dev.BASE.FROZEN_V93_WEIGHTS)
    dev.json_write(out / 'contracts/STATUS.json', {'created_utc': datetime.now(timezone.utc).isoformat(),
                   'code': conf['code'], 'status': 'FROZEN_CANDIDATE_AWAITING_FRESH_CONFIRMATION',
                   'historical_results_not_modified': True, 'real_development_not_blind': True})
    for name in ('RECIPE.json', 'TARGET_AUDIT.json', 'SEARCH_COMPLETE.json'):
        copy(trial / 'contracts' / name, out / 'contracts' / name)
    for path in (trial / 'tables').glob('*.csv'):
        copy(path, out / 'tables' / path.name)
    for path in (trial / 'contracts').glob('*SELECTED.json'):
        copy(path, out / 'contracts' / path.name)
    ledger = sorted(root.glob('audit/progress_*'))[-1]
    for path in ledger.glob('*'):
        if path.is_file(): copy(path, out / 'audit/development_ledger' / path.name)
    for prefix in ('real_input_reconstruction_', 'input_band_', 'score_integrity_'):
        audits = sorted(root.glob('audit/' + prefix + '*'))
        if audits:
            for path in audits[-1].rglob('*'):
                if path.is_file(): copy(path, out / 'audit' / audits[-1].name / path.relative_to(audits[-1]))
    # Compact failure evidence remains separate from authoritative selected rankings.
    for trialdir in sorted((root / 'trials').iterdir()):
        if not trialdir.is_dir(): continue
        for group in ('contracts', 'tables'):
            for path in (trialdir / group).rglob('*'):
                if path.is_file() and path.suffix in ('.json', '.csv', '.md'):
                    copy(path, out / 'exploration_ledger' / trialdir.name / group / path.relative_to(trialdir / group))
    copied, coefficients, checks, ranks = [], [], [], []
    for dep in e.DEPS:
        for method in ('C_fixed', 'waveform_only'):
            for tag, srcroot, config in [('baseline', e.BASE, 'UNIFIED'), ('candidate', trial, 'CANDIDATE')]:
                p = srcroot / f'results/{config}/{dep}/consensus_{method}_all_pairs_pe_official.parquet'
                frame = pd.read_parquet(p)
                copy(p, out / f'real/{dep}/{tag}_consensus_{method}_all_pairs.parquet')
                dev.csv_write(out / f'real/{dep}/{tag}_consensus_{method}_all_pairs.csv', frame)
                for budget in (10, 20, 50, 100):
                    dev.csv_write(out / f'real/{dep}/{tag}_{method}_Top{budget}.csv', frame.head(budget))
            b = pd.read_parquet(e.BASE / f'results/UNIFIED/{dep}/consensus_{method}_all_pairs_pe_official.parquet')
            c = pd.read_parquet(trial / f'results/CANDIDATE/{dep}/consensus_{method}_all_pairs_pe_official.parquet')
            merged = c.merge(b[['pair_key', 'consensus_rank', 'final_score_mean', 'waveform_score_mean']], on='pair_key', suffixes=('_new', '_baseline'), validate='one_to_one')
            merged['rank_improvement'] = merged.consensus_rank_baseline - merged.consensus_rank_new
            merged['deployment'], merged['method'] = dep, method
            ranks.append(merged)
        for ms, es in zip(body.MODEL_SEEDS, dev.SEEDS):
            inp = out / f'replay_inputs/{dep}/seed_{es}'
            inp.mkdir(parents=True)
            f = pd.read_parquet(e.BASE / f'evaluation/{dep}/seed_{es}/real_pairs.parquet')
            stale = [k for k in f if k.startswith(('baseline_final_', 'baseline_rank', 'baseline_waveform_contribution', 'baseline_time_contribution', 'baseline_sky_contribution'))]
            stale += [k for k in ('final_score', 'rank', 'method', 'waveform_contribution', 'time_contribution', 'sky_contribution') if k in f]
            f.drop(columns=stale).to_parquet(inp / 'baseline_pairs.parquet', index=False)
            spec = conf['configs'][dep][str(es)]
            dev.json_write(inp / 'SELECTED_CONFIG.json', spec)
            cp = root / f'ordered_mass_predictor/models/{dep}/seed_{ms}/validation_selected_model.pt'
            ck = torch.load(cp, weights_only=False, map_location='cpu')
            a = ordered.prediction(root, dep, ms, es, 'real')
            np.savez_compressed(inp / 'ordered_mass_predictions.npz', p=a['p'], boundary_mass=a['boundary_mass'], prior=ck['prior'], centers=ordered.CENTERS)
            coefficients.append({'deployment': dep, 'model_key': ms, 'eval_seed': es,
                                 'ordered_training_seed': ck['seed'], 'epoch': ck['epoch'], 'temperature': ck['temperature'],
                                 'gamma': spec['gamma'], 'beta': spec['beta'],
                                 'additional_increment_active': spec['gamma'] != 0 or spec['beta'] != 0,
                                 **{'outer_' + k: v for k, v in dev.BASE.FROZEN_V93_WEIGHTS[dep][es].items()}})
            for kind, path in [('ordered_mass', cp),
                ('baseline_RNC', e.TRAINED / f'models/RAW-PHASE-SOURCE/{dep}/seed_{ms}/validation_selected_model.pt'),
                ('old_Cfixed', dev.V7 / dep / f'seed_{es}/waveform_gate/unified_inception_attention_peak2s_4096_aux_0p25_q_1p0_v7/validation_selected_model.pt')]:
                dest = out / f'models/{dep}/seed_{es}/{kind}.pt'
                copy(path, dest)
                copied.append({'original_path': str(path), 'package_path': str(dest.relative_to(out)), 'sha256': dev.sha(dest)})
            for method in ('C_fixed', 'waveform_only'):
                p = trial / f'results/CANDIDATE/{dep}/seed_{es}_{method}.parquet'
                fr = pd.read_parquet(p)
                # Keep only current, authoritatively recomputed score/rank fields.
                drop = [k for k in fr if k.startswith('baseline_')]
                fr = fr.drop(columns=drop)
                dest = out / f'real/{dep}/candidate_seed_{es}_{method}.parquet'
                fr.to_parquet(dest, index=False)
                w = dev.BASE.FROZEN_V93_WEIGHTS[dep][es] if method == 'C_fixed' else {'waveform': 1., 'time': 0., 'sky': 0.}
                score = w['waveform'] * fr.waveform_score + w['time'] * fr.time_score + w['sky'] * fr.sky_raw_log_bf
                delta = float(abs(score - fr.final_score).max())
                if delta != 0: raise RuntimeError('Incoherent final score')
                checks.append({'deployment': dep, 'eval_seed': es, 'method': method, 'max_score_difference': delta})
            for path in cp.parent.glob('*.json'):
                copy(path, out / f'models/{dep}/seed_{es}/ordered_training_audit' / path.name)
            for path in cp.parent.glob('*.csv'):
                copy(path, out / f'models/{dep}/seed_{es}/ordered_training_audit' / path.name)
    dev.csv_write(out / 'tables/SELECTED_COEFFICIENTS.csv', pd.DataFrame(coefficients))
    dev.csv_write(out / 'tables/RANK_CHANGE_ALL.csv', pd.concat(ranks, ignore_index=True))
    dev.csv_write(out / 'manifest/CHECKPOINT_PROVENANCE.csv', pd.DataFrame(copied))
    dev.json_write(out / 'audit/SELECTED_SCORE_INTEGRITY.json', {'pass': True, 'checks': checks})
    frozen = conf['frozen_files']
    for row in frozen:
        path = Path(row['path'])
        if path.suffix == '.json': copy(path, out / 'source_snapshot' / path.relative_to(dev.PROJECT))
    codefiles = set(dev.PROJECT.glob('scripts/experiments/mcwf_*.py'))
    for module in list(sys.modules.values()):
        file = getattr(module, '__file__', None)
        if file:
            path = Path(file).resolve()
            if path.is_relative_to(dev.PROJECT) and path.suffix == '.py': codefiles.add(path)
    for path in sorted(codefiles):
        copy(path, out / 'source_snapshot' / path.relative_to(dev.PROJECT))
    for name in ('mcwf_replay_ordered_scores_20260907.py', 'mcwf_ordered_release_prepare_20260907.py'):
        copy(dev.PROJECT / 'scripts/experiments' / name, out / 'scripts' / name)
    versions = {}
    for name in ('numpy', 'scipy', 'pandas', 'pyarrow', 'torch', 'bilby', 'lalsuite', 'healpy', 'ligo.skymap', 'scikit-learn'):
        try: versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: versions[name] = 'not installed'
    dev.json_write(out / 'contracts/ENVIRONMENT.json', {'versions': versions, 'python': sys.version,
                   'GPU': torch.cuda.get_device_name(0), 'CUDA': torch.version.cuda,
                   'server_host': 'connect.westd.seetacloud.com', 'ssh_port': 32328,
                   'credentials_included': False})
    dev.json_write(out / 'manifest/PREPARE_COMPLETE.json', {'utc': datetime.now(timezone.utc).isoformat(),
                   'file_count': sum(p.is_file() for p in out.rglob('*')), 'fresh_confirmation_not_yet_claimed': True})
    print(json.dumps({'prepared': str(out)}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    prepare(p.parse_args().root)
