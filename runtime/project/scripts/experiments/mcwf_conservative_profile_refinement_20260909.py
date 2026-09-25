#!/usr/bin/env python3
"""One-sided waveform refinement across two simulation-calibrated predictions."""
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
import mcwf_profile_conditional_corrected_20260909 as isolated
co, n, r, score = isolated.co, isolated.n, isolated.r, isolated.score
ROOT = None
METHODS = ('NODUP-DIRECT-REPLAY', 'ONE-SIDED-PROFILE-GLOBAL', 'ONE-SIDED-PROFILE-CONDITIONAL')


def infer(frame, config):
    original = isolated.ORIGINALS[config['deployment'], config['seed']]
    base = co.BASE_INFER(frame, {**original, 'method': METHODS[0]})
    if config['method'] == METHODS[0]:
        return base
    candidate = co.BASE_INFER(frame, config)
    active = frame.profile_pair_active.to_numpy(bool)
    take = active & (candidate[0] < base[0])
    result = tuple(np.where(take, new, old) for new, old in zip(candidate, base))
    if np.any(result[0] > base[0]) or not np.array_equal(result[0][~active], base[0][~active]):
        raise RuntimeError('Noncompensating waveform invariant failed')
    return result


def freeze():
    if ROOT.exists():
        raise RuntimeError('Independent output directory required')
    for folder in ('contracts', 'configs', 'calibration', 'tables', 'audit', 'reports', 'scripts', 'logs',
                   'manifest', 'results', 'figures'):
        (ROOT/folder).mkdir(parents=True)
    n.write_json(ROOT/'contracts/ANALYSIS_CONTRACT.json', {'UTC': n.utc(),
        'id': 'MCWF-NODUP-ONE-SIDED-PROFILE-25', 'goal_achieved': False, 'status': n.STATUS,
        'adaptive_development': True, 'same_both_runs': True,
        'hypothesis': 'Replacing a broad prediction by a locally sharpened prediction may also promote unrelated pairs. A conservative lower envelope tests whether incompatibility information can help without new positive rewards.',
        'formula': 'Zwf=min(Zwf_parent,Zwf_profile) only for frozenquality-active pairs;otherwiseexactZwf_parent. Since cosine andgamma/beta areidentical,this is min of the ONE joint-consistencycomponent,notadditionoftwoMc evidences.',
        'not_a_Bayes_factor': 'Lowerenvelopeofcorrelatedcalibratedrankingscoresisaconservativetriageheuristic,notanormalizedlikelihoodratio orPE posterior. No claimthatminisphysicallyoptimal.',
        'arms': 'GLOBAL:round10 source-disjoint globalprofilecalibration;CONDITIONAL:round12 source-disjoint quality-active profilecalibration. Bothfixed,allarmsreported.',
        'selection': 'No new score hyperparameters or real/test selection. Existing profilequality,Student-t,width,tailthreshold.05,clips,gamma,beta frozen.',
        'frozen': ['baseencoder', 'MULTIRATE', 'conditionaleta_chi', 'R10profile', 'time', 'sky', 'outerweights', 'scope', 'historicalresults'],
        'forbidden': ['oldencoderMc_q', 'total-scoremixture', 'PE_or_officialfeatures', 'event-specificrule', 'manualrankremoval'],
        'validation': 'Report existing waveform/fusionguardsagainstbothNODUPandPATH875. ReusedcatalogsandrealPEareadaptiveaudits,notblindconfirmation.',
        'motivation_from_literature_only': ['https://arxiv.org/abs/gr-qc/9402014','https://arxiv.org/abs/1409.7215'],
        'citation_limit': 'References motivate waveformparameterinformationandmodelvalidation;neitherprescribesthislower-envelopeheuristic.'})
    inputs = pd.read_csv(r.PRIOR/'manifest/INPUT_SHA256.csv').to_dict('records')
    configs, units, metrics = [], [], []
    for dep in n.DEPS:
        for seed in n.SEEDS:
            old = isolated.ORIGINALS[dep, seed]
            configs.append({**old, 'method': METHODS[0]})
            frame = co.load_panel(dep, seed, 'validation')
            base = infer(frame, configs[-1])
            archived = pd.read_parquet(r.PRIOR/f'results/NODUP-DIRECT/{dep}/seed_{seed}/validation/pairs.parquet')
            if not np.array_equal(frame[['idx_i','idx_j']].to_numpy(), archived[['idx_i','idx_j']].to_numpy()) or not np.allclose(base[0], archived.waveform_score, atol=1e-12, rtol=0):
                raise RuntimeError('Baseline replay changed')
            for method, parent in zip(METHODS[1:], (co.PARENT, P/'results/mcwf_nodup_isolated_profile_12_20260909T092250Z')):
                path = parent/f'calibration/{dep}_{seed}_JOINT.json'
                spec = json.loads(path.read_text())
                config = {**old, 'method': method, 'joint_calibration_subgrid': spec}
                result = infer(frame, config)
                fm = n.cf.fast_metrics(frame, n.cf.channels(frame, result[0])@np.asarray(old['weights']))
                wm = n.cf.fast_metrics(frame, result[0])
                reference = n.cf.fast_metrics(frame, n.cf.channels(frame, base[0])@np.asarray(old['weights']))
                wr = n.cf.fast_metrics(frame, base[0])
                config.update(tune_guard=n.cf.guard(fm, reference) and n.cf.guard(wm, wr), tune_metrics=fm)
                configs.append(config)
                poison = frame.copy()
                poison['pair_key'] = 'unused'
                poison['pe_mc_bhattacharyya_coefficient'] = -100.
                poison['official_po_fpp'] = 0.
                delta = float(abs(infer(poison, {**config,'alpha':-999.,'old_Mc_weight':999.})[0]-result[0]).max())
                if delta != 0:
                    raise RuntimeError('Forbidden input effect')
                units.append({'deployment': dep,'seed':seed,'method':method,'maximum_score_increase':float((result[0]-base[0]).max()),
                    'changed_fraction':float((result[0]!=base[0]).mean()),'forbidden_input_delta':delta,'inactive_exact':True})
                metrics.append({'deployment':dep,'seed':seed,'method':method,'guard':config['tune_guard'],**fm})
                inputs.append({'path':str(path),'sha256':n.sha(path),'bytes':path.stat().st_size})
    n.write_csv(ROOT/'manifest/INPUT_SHA256.csv',inputs)
    n.write_csv(ROOT/'audit/ONE_SIDED_INVARIANCE.csv',units)
    n.write_csv(ROOT/'tables/VALIDATION_GUARDS.csv',metrics)
    n.write_json(ROOT/'configs/SELECTED_CONFIGURATIONS.json',configs)
    n.write_json(ROOT/'contracts/CONFIGURATIONS_FROZEN.json',{'UTC':n.utc(),'file':'configs/SELECTED_CONFIGURATIONS.json',
        'sha256':n.sha(ROOT/'configs/SELECTED_CONFIGURATIONS.json'),'no_parameter_search':True,'no_real_test_selection':True})
    shutil.copy2(__file__,ROOT/'scripts/conservative_profile_refinement.py')
    n.write_json(ROOT/'contracts/START_FREEZE.json',{'UTC':n.utc(),'code_sha256':n.sha(Path(__file__)),
        'contract_sha256':n.sha(ROOT/'contracts/ANALYSIS_CONTRACT.json')})


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--stage',choices=('freeze','evaluate','real'),required=True)
    args=parser.parse_args()
    ROOT=co.ROOT=r.ROOT=score.ROOT=args.root
    r.install()
    score.METHODS=METHODS
    score.matrices=co.matrices
    n.METHODS,n.load_panel,n.infer=METHODS,co.load_panel,infer
    if args.stage=='freeze':
        freeze()
    else:
        n.run(ROOT,args.stage)
