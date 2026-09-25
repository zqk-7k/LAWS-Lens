"""Finite scientific replay in the production and clean locked environments."""
import os
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
from collections import namedtuple
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--source-root', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--project-root', type=Path)
parser.add_argument('--completion-root', type=Path)
args = parser.parse_args()
R, out = args.source_root, args.output
out.mkdir(parents=True, exist_ok=False)
P = args.project_root or R.parents[1]
C = args.completion_root or R/'completion'
sys.path[:0] = [str(P), str(R/'scripts'), str(C/'scripts')]
torch.set_num_threads(2)
checks = []
arrays = {}
def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    sys.modules[name] = obj
    spec.loader.exec_module(obj)
    return obj

import lal
import lalsimulation
from bilby.gw.conversion import bilby_to_lalsimulation_spins
from pycbc.waveform import get_fd_waveform
from pycbc.filter import matched_filter, sigma
from pycbc.types import FrequencySeries
from ligo.skymap import bayestar, moc
from ligo.skymap.io.events.base import Event, SingleEvent
from ligo.skymap.io.fits import read_sky_map
import healpy as hp

spin = bilby_to_lalsimulation_spins(.7, .2, .5, .6, .8, .4, .3,
    35*lal.MSUN_SI, 25*lal.MSUN_SI, 20., .1)
assert np.isfinite(spin).all()
arrays['si_spin'] = np.asarray(spin)
h, _ = get_fd_waveform(approximant='IMRPhenomXPHM', mass1=35, mass2=25,
    spin1x=.1, spin1z=.2, spin2y=.1, spin2z=-.1, distance=1000,
    inclination=.7, delta_f=.25, f_lower=20, f_final=512)
assert np.isfinite(np.asarray(h)).all() and np.max(abs(np.asarray(h))) > 0
arrays['xphm_fd'] = np.asarray(h)
psd = FrequencySeries(np.full(len(h), 1e-44), delta_f=.25)
norm = sigma(h, psd=psd, low_frequency_cutoff=20, high_frequency_cutoff=512)
z = matched_filter(h, h*(13/norm), psd=psd, low_frequency_cutoff=20, high_frequency_cutoff=512)
assert abs(abs(z[0])-13) < 1e-8
checks.append({'test': 'SI_spins_XPHM_matched_filter', 'pass': True})

trainer = load(R/'scripts/short_trainer.py', 'verify_short_trainer')
trainer.REPO = P
trainer.INPUT_SAMPLES = 4096
for run in ('O3', 'O4a', 'O4b'):
    path = R/f'arms/C_PHYSICAL/{run}/models/short_encoder/seed_2026091721/validation_selected_model.pt'
    ck = torch.load(path, map_location='cpu', weights_only=False)
    model = trainer.PhysicsRegularizedEncoder(trainer.build_base_encoder('inception_attention', None))
    model.load_state_dict(ck['model_state'], strict=True)
    model.eval()
    p = sorted((R/f'data/{run}/main/validation').glob('*/C_PHYSICAL_short.npy'))[0]
    x = torch.tensor(np.stack([trainer.prepare(a, None, False) for a in np.load(p)[:2]]))
    with torch.no_grad():
        embedding, parameters = model(x, return_parameters=True)
    assert torch.isfinite(embedding).all() and torch.isfinite(parameters).all()
    arrays[run+'_embedding'] = embedding.numpy()
    arrays[run+'_parameters'] = parameters.numpy()
    if torch.cuda.is_available():
        with torch.no_grad():
            zg, mg = model.cuda()(x.cuda(), return_parameters=True)
        assert torch.isfinite(zg).all() and torch.isfinite(mg).all()
        arrays[run+'_embedding_gpu'] = zg.cpu().numpy()
        del zg, mg
    checks.append({'test': run+'_checkpoint_forward_CPU_GPU', 'pass': True,
                   'gpu_exercised': torch.cuda.is_available()})
    del model

# Recompute actual data-derived BAYESTAR maps, without re-running template selection.
ST = namedtuple('SingleTuple', 'detector snr phase time zerolag_time psd snr_series')
class Single(ST, SingleEvent):
    pass
ET = namedtuple('EventTuple', 'singles template_args')
class HLEvent(ET, Event):
    pass
for run in ('O3', 'O4a', 'O4b'):
    trigger = sorted((R/f'timings/pilot/{run}').glob('*/TRIGGERS.npz'))[0]
    metadata = json.loads(trigger.with_suffix('.json').read_text())
    data = np.load(trigger)
    singles = []
    for d, detector in enumerate(('H1', 'L1')):
        values = data['snr_series'][d]
        sec, ns = divmod(int(data['epochs_ns'][d]), 10**9)
        epoch = lal.LIGOTimeGPS(sec, ns)
        ts = lal.CreateCOMPLEX8TimeSeries('frozen data SNR', epoch, 0, 1/2048,
            lal.DimensionlessUnit, len(values))
        ts.data.data[:] = values
        center = len(values)//2
        arrival = epoch + center/2048
        ps = lal.CreateREAL8FrequencySeries('frozen noise PSD', 0, 0, 1/64,
            lal.StrainUnit**2, data['psds'].shape[1])
        ps.data.data[:] = data['psds'][d]
        singles.append(Single(detector, float(abs(values[center])), float(np.angle(values[center])),
            arrival, arrival, ps, ts))
    power = data['template_power'][:-1]
    freq = np.arange(len(power))/64
    actual = lal.CreateREAL8FrequencySeries('frozen selected power', 0, 0, 1/64,
        lal.StrainUnit**2, len(power))
    actual.data.data[:] = power*((freq >= 20) & (freq < 900))
    original = bayestar.filter.sngl_inspiral_psd
    bayestar.filter.sngl_inspiral_psd = lambda *a, **kw: actual
    start = time.monotonic()
    try:
        sky = bayestar.localize(HLEvent(singles, metadata['template_args']),
            waveform='IMRPhenomD', f_low=20., enable_snr_series=True,
            f_high_truncate=1., rescale_loglikelihood=.83)
    finally:
        bayestar.filter.sngl_inspiral_psd = original
    norm = float(np.sum(sky['PROBDENSITY']*moc.uniq2pixarea(sky['UNIQ'])))
    assert abs(norm-1) < 1e-5
    raster = moc.rasterize(sky, order=9)
    prob = np.asarray(raster['PROBDENSITY'])*hp.nside2pixarea(512)
    arrays[run+'_sky512'] = prob
    # A single common permutation must preserve the overlap exactly to rounding.
    ring = hp.reorder(prob, n2r=True)
    assert np.isclose(np.dot(prob, prob), np.dot(ring, ring), rtol=1e-12)
    checks.append({'test': run+'_BAYESTAR_frozen_actual_triggers', 'pass': True,
        'normalization': norm, 'seconds': time.monotonic()-start, 'nside': 512})
    print(run+' BAYESTAR replay PASS', flush=True)

from uab_independent_check import recompute
metric_rows = []
subsets = pd.read_parquet(R/'plans/catalog190_subsets.parquet')
columns = {'waveform-new': 'waveform_score', 'time-only': 'time_score',
           'sky-only': 'sky_raw_log_bf', 'three-channel': 'final_score_POSITIVE'}
for run in ('O3', 'O4a', 'O4b'):
    for seed in (2026091721, 2026091722, 2026091723):
        folder = C/f'deployments/{run}/C_PHYSICAL/evaluation/test/seed_{seed}'
        f, events = pd.read_parquet(folder/'all_pair_scores.parquet'), pd.read_parquet(folder/'events.parquet')
        summary = pd.read_csv(folder/'metrics.csv').set_index('method')
        subgroup = pd.read_csv(folder/'catalog190_metrics.csv')
        for draw in (None, 0, 499):
            if draw is None:
                sample, n, expected = f, 450, summary
            else:
                keep = events.source_uid.isin(subsets[(subsets.split == 'test') & (subsets.draw == draw)].source_uid).to_numpy()
                sample = f.loc[keep[f.idx_i] & keep[f.idx_j]].copy()
                remap = np.full(450, -1)
                remap[keep] = np.arange(190)
                sample['idx_i'], sample['idx_j'] = remap[sample.idx_i], remap[sample.idx_j]
                n, expected = 190, subgroup[subgroup.draw == draw].set_index('method')
            for method, col in columns.items():
                for metric, value in recompute(sample, sample[col].to_numpy(float), n).items():
                    delta = abs(value-float(expected.loc[method, metric]))
                    assert delta < 1e-11
                    metric_rows.append(dict(run=run, seed=seed, draw=draw, method=method,
                        metric=metric, error=delta))
pd.DataFrame(metric_rows).to_csv(out/'METRIC_REPLAY.csv', index=False)
np.savez_compressed(out/'NUMERICAL_OUTPUTS.npz', **arrays)
report = {'state': 'PASS', 'python': sys.executable, 'checks': checks,
          'metric_checks': len(metric_rows), 'max_metric_error': max(r['error'] for r in metric_rows),
          'full_retraining_performed': False, 'all_event_inference_rerun': False,
          'selection_or_scientific_parameters_changed': False}
(out/'VERIFICATION.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report), flush=True)
