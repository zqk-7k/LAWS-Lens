#!/usr/bin/env python3
"""Reconstruct a20-80Hz16s/256Hz auxiliary branch without changing peak2s."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import sys
import time
import numpy as np
import pandas as pd
from scipy import signal
from pycbc.waveform import get_fd_waveform
import torch
import torch.nn.functional as F

P = Path('/root/autodl-tmp/gw-catalog')
sys.path.insert(0, str(P / 'scripts/experiments'))
import mcwf_temporal_response_20260908 as t
import mcwf_expanded_data_20260907 as expanded
import mcwf_continuous_profile_recovery_v2_20260908 as replay

dev, old = t.dev, t.old
FS, N, LENGTH, NFFT, MAX_LAG, NET_LAG = 256, 26*256, 4096, 8192, 38, 3
RAW_N = 26*4096
CTX = {}
PARAMETERS = [(float(np.exp(mc)), q, chi) for mc in old.LOG_CENTERS for q in (.25, .5, 1.) for chi in (-.5, 0., .5)]
torch.set_num_threads(2)


def initialize(root):
    if root.exists():
        raise RuntimeError('Independent output required')
    for n in ('contracts', 'scripts', 'logs', 'cache', 'features', 'models', 'predictions', 'calibration', 'evaluation', 'tables', 'reports', 'manifest', 'figures'):
        (root/n).mkdir(parents=True)
    dev.json_write(root/'contracts/ANALYSIS_CONTRACT.json', {
        'id': 'MCWF-MULTIRATE-LOWBAND-06', 'utc': datetime.now(timezone.utc).isoformat(),
        'status': t.STATUS, 'goal_achieved': False, 'same_both_runs': True,
        'retained_primary': 'exact archived40-580Hz peak2s4096point features and models;not overwritten',
        'new_auxiliary': '20-80Hz last16seconds,anti-alias decimated to256Hz4096points;separate waveform branch,not a time-delay or sky change',
        'why': 'PROFILE05 lower-frequency pilot reduced continuous-template mass errors inbothruns,but direct fit remained worse than OMC;test complementary information instead of replacing the learned predictor with pointoptimizer',
        'sources': 'same4096 expanded training sourceparents/96noiseblocks/8views and512developmentparents/32noiseblocks;no additional_population sources inthis matched small-data control',
        'reconstruction': 'same sourceparameters,H1L1 physicalresponse,Morse,magnification,actual noisebank/offset and originalPSD-SNR scaling;before making newbranch requirebit-exact oldfloat16 peak2s',
        'no_inversion_of_40Hz_filter': True, 'no_SNR_rescaling_after_band_change': True,
        'input_preprocessing': 'raw26s4096Hz -> unchanged PSDwhitening -> Butter6[20,80]zero-phase ->resamplepoly1/2Kaiser8.6 ->24scrop+robustMAD ->resamplepoly1/8Kaiser8.6 ->last16s',
        'lowband_window_std_normalization': False,
        'template_bank': '2277 IMRPhenomD253Mc*3q*3equalalignedspin;generatefullband26s at4096Hztoalignmerger;retainlowfrequencyspectrum;per-eventexactoffsourcePSD;physical4096Hzfilterandbothanti-aliastransferfunctions',
        'template_implementation_test': 'fast low-frequencyoperator vsfull4096Hz conditioning subspace;minimumcanonicalcorrelation>=0.999 ondeterministictemplate/PSDtests,elseHOLD',
        'template_scope': 'aligned/limitedq/spinbank is afeaturemap,not aprecessingPEposterior orminimalmatchguarantee',
        'features': 'maxquadrature Hpower,Lpower,networkpower;lag+-38samples,relative<=3samples;log1p;27x253 orderedresponsechannels',
        'model_plan': 'CONTROL:continueoriginalOMCon4096parents;MULTIRATE:append27lowbandchannels withzero-initializedfirstlayerweights;otherwise same15epochs,3seeds,CE/temperatureselection',
        'training_seeds': t.TRAIN_SEEDS,
        'selection': 'simulationdevelopmentCE andBAYESTARvalidation-onlywaveformincrement;samepriorities/guards;realPEandofficialjoinedonlyafterfreeze',
        'frozen': ['time', 'sky_raw_log_bf', 'outerC-fixedweights', 'scope', 'originalmodels', 'history', 'paper'],
        'adaptive_development': True, 'fresh_confirmation_required_before_upgrade': True,
        'storage': 'onlycompactlowbranchcacheandfeaturecache,newcheckpoints,reproducibilitytables;noneincludedasnewauthoritativefullstrainbank',
        'disk': {'minimum_free_GiB': 25, 'maximum_new_working_set_GiB': 40},
        'references': ['https://arxiv.org/abs/1107.2665', 'https://arxiv.org/abs/gr-qc/9402014',
                       'https://pycbc.org/pycbc/latest/html/filter.html'],
        'reference_limits': 'multiratefiltering andintrinsicphaseinformation motivate thetest;thisisnot LLOIDreproductionorfullPE'})
    dev.csv_write(root/'manifest/INPUT_SHA256.csv', pd.DataFrame(t.protected()))
    shutil.copy2(__file__, root/'scripts/multirate_features.py')
    dev.json_write(root/'contracts/CONTRACT_HASH.json', {'sha256': dev.sha(root/'contracts/ANALYSIS_CONTRACT.json')})


def low_view(raw, freq, psd, v3):
    full = v3.preprocess_24s(raw, freq, psd, band_low_hz=20., band_high_hz=80.)
    down = signal.resample_poly(full, 1, 8, axis=-1, window=('kaiser', 8.6))
    result = down[..., -LENGTH:].astype(np.float32)
    if result.shape != (2, LENGTH) or not np.isfinite(result).all():
        raise RuntimeError('Invalid auxiliary waveform')
    return result


def init_development(dep, split):
    src, data, v3 = expanded.modules()
    import logging
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=1)
    logging.getLogger('bilby').setLevel(logging.ERROR)
    base = t.PREVIOUS/f'expanded_data/{dep}'
    CTX.update(src=src, v3=v3, generator=src.build_waveform_generator(),
        ifos=[src.bilby.gw.detector.get_empty_interferometer(d) for d in ('H1', 'L1')],
        refs=np.load(base/'noise/reference.npy', mmap_mode='r'), freq=np.load(base/'noise/frequency.npy'),
        psds=np.load(base/'noise/psd.npy', mmap_mode='r'),
        oldraw=np.load(base/f'{split}/raw2s.npy', mmap_mode='r'))


def reconstruct(row):
    src, v3 = CTX['src'], CTX['v3']
    number = 1 if row['image']=='a' else 2
    clean, _ = src.detector_response(CTX['generator'], CTX['ifos'],
        src.source_parameters(pd.Series(row), row['gps_'+row['image']]),
        src.lens_factor(row[f'mu_image{number}'], row[f'morse_image{number}']))
    b, offset = int(row['noise_bank_index']), int(row['noise_offset_samples'])
    noise = np.asarray(CTX['refs'][b, :, offset:offset+v3.RAW_PADDED_SAMPLES], np.float64)
    scaled, _, _ = v3.scale_to_network_snr(np.asarray(clean, np.float32), row['target_network_snr'], CTX['freq'], CTX['psds'][b])
    raw = np.asarray(noise, np.float32)+v3.embed_signal_in_padded_window(scaled)
    oldfull = v3.preprocess_24s(raw, CTX['freq'], CTX['psds'][b])
    oldview = dev.TRAIN.make_window_view(oldfull[None].astype(np.float32), 2)[0].astype(np.float16)
    idx = int(row['row_index'])
    if not np.array_equal(oldview, CTX['oldraw'][idx]):
        difference = float(abs(oldview.astype(float)-CTX['oldraw'][idx].astype(float)).max())
        raise RuntimeError(f'Archived input replay failed:{idx}:{difference}')
    return idx, low_view(raw, CTX['freq'], CTX['psds'][b], v3), oldfull[..., -4096:].std(-1)


def development_raw(root, dep, split, workers):
    out = root/f'cache/{dep}/{split}'
    if (out/'COMPLETE.json').exists():
        return out
    out.mkdir(parents=True, exist_ok=True)
    meta = pd.read_parquet(t.PREVIOUS/f'expanded_data/{dep}/{split}/event_metadata.parquet')
    path = out/'low16s.npy'
    mode = 'r+' if path.exists() else 'w+'
    values = np.lib.format.open_memmap(path, mode=mode, dtype=np.float32, shape=(len(meta), 2, LENGTH))
    progress = out/'PROGRESS.parquet'
    rows = pd.read_parquet(progress).to_dict('records') if progress.exists() else []
    done = {int(r['row_index']) for r in rows}
    pending = meta[~meta.row_index.isin(done)].to_dict('records')
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn'),
                             initializer=init_development, initargs=(dep, split)) as pool:
        for count, (idx, waveform, scales) in enumerate(pool.map(reconstruct, pending, chunksize=4), 1):
            values[idx] = waveform
            rows.append({'row_index': idx, 'old_peak2s_exact': True,
                         'pre_window_normalization_std_H1': float(scales[0]),
                         'pre_window_normalization_std_L1': float(scales[1])})
            if count%256==0 or count==len(pending):
                if shutil.disk_usage(root).free<25*2**30:
                    raise RuntimeError('HOLD_DISK_LIMIT')
                values.flush()
                pd.DataFrame(rows).to_parquet(progress, index=False)
                print(json.dumps({'lowband_replay': dep, 'split': split, 'completed': len(rows), 'total': len(meta),
                                  'seconds': time.perf_counter()-started}), flush=True)
    values.flush()
    dev.json_write(out/'COMPLETE.json', {'events': len(meta), 'source_parents': meta.source_uid.nunique(),
        'noise_blocks': meta.noise_bank_index.nunique(), 'old2s_replay_all_exact': True,
        'sha256': dev.sha(path), 'dtype': 'float32', 'shape': list(values.shape), 'seconds': time.perf_counter()-started})
    return out


def full_wave(parameters):
    mc, q, chi = parameters
    m1 = mc*(1+q)**.2/q**.6
    hp, _ = get_fd_waveform(approximant='IMRPhenomD', mass1=m1, mass2=m1*q,
        spin1z=chi, spin2z=chi, delta_f=1/26, f_lower=20., f_final=2048., distance=1000., inclination=0.)
    hp.resize(RAW_N//2+1)
    analytic = signal.hilbert(np.fft.irfft(np.asarray(hp), n=RAW_N))
    shift = int(24.75*4096)-int(np.argmax(abs(analytic)))
    return np.roll(analytic, shift)


def spectral_bank(root):
    path = root/'cache/lowband_aligned_spectra.npy'
    if path.exists():
        return np.load(path)
    spectra = []
    for n, parameters in enumerate(PARAMETERS):
        h = full_wave(parameters)
        spectra.append(np.fft.rfft(h.real)[:N//2+1].astype(np.complex64))
        if n%400==0:
            print('LOWBANK', n, len(PARAMETERS), flush=True)
    result = np.stack(spectra)
    np.save(path, result)
    dev.csv_write(root/'cache/TEMPLATE_PARAMETERS.csv', pd.DataFrame(PARAMETERS, columns=['mc_det', 'q', 'equal_chi']))
    dev.json_write(path.with_suffix('.json'), {'sha256': dev.sha(path), 'shape': list(result.shape),
        'frequency_grid_Hz': 'arange(3329)/26', 'fullband_merger_aligned_before_truncation': True})
    return result


def filter_gain(frequency):
    sos = signal.butter(6, [20, 80], fs=4096., btype='bandpass', output='sos')
    _, z = signal.sosfreqz(sos, worN=frequency, fs=4096.)
    gain = abs(z)**2
    for down, fs in ((2, 4096.), (8, 2048.)):
        h = signal.firwin(20*down+1, 1/down, window=('kaiser', 8.6))
        _, transfer = signal.freqz(h, worN=2*np.pi*frequency/fs)
        gain *= abs(transfer)
    return gain


@torch.no_grad()
def bank_for(h, freq, psd):
    f = np.fft.rfftfreq(N, 1/FS)
    p = np.array([np.interp(f, freq, row) for row in psd])
    p = np.maximum(p, p.max(1, keepdims=True)*1e-20)
    sf = torch.as_tensor(h, device='cuda')[:, None]*torch.as_tensor(filter_gain(f)/np.sqrt(p), dtype=torch.float32, device='cuda')[None]
    white = torch.fft.irfft(sf, n=N)
    hf = torch.zeros(N, device='cuda'); hf[0]=hf[N//2]=1; hf[1:N//2]=2
    z = torch.fft.ifft(torch.fft.fft(white)*hf)
    u, v = z.real[..., 9*FS:25*FS], z.imag[..., 9*FS:25*FS]
    u = u-u.mean(-1, keepdim=True); v=v-v.mean(-1, keepdim=True)
    u=F.normalize(u, dim=-1); v=v-(v*u).sum(-1, keepdim=True)*u; v=F.normalize(v, dim=-1)
    return torch.stack([u, v], 2)


def operator_test(root, h, freq, psd, label='first_PSD'):
    _, _, v3 = expanded.modules()
    ids = [0, 4, 570, 1138, 1708, 2276]
    fast = bank_for(h[ids], freq, psd).cpu().numpy()
    rows = []
    for outidx, k in enumerate(ids):
        wave = full_wave(PARAMETERS[k])
        phases = [low_view(np.stack([part, part]), freq, psd, v3) for part in (wave.real, wave.imag)]
        exact = np.stack(phases, 1).astype(float)
        exact-=exact.mean(-1, keepdims=True)
        for d in range(2):
            q, _ = np.linalg.qr(exact[d].T)
            correlations = np.linalg.svd(fast[outidx, d]@q, compute_uv=False)
            rows.append({'context': label, 'template_index': k, 'detector': ('H1', 'L1')[d],
                         'min_canonical_correlation': float(correlations.min())})
    table = pd.DataFrame(rows)
    prior = root/'tables/LOWBAND_OPERATOR_TEST.csv'
    if prior.exists():
        table = pd.concat([pd.read_csv(prior), table], ignore_index=True)
    dev.csv_write(root/'tables/LOWBAND_OPERATOR_TEST.csv', table)
    passed = bool((table.min_canonical_correlation>=.999).all())
    dev.json_write(root/'contracts/LOWBAND_OPERATOR_TEST.json', {'pass': passed,
        'minimum_canonical_correlation': table.min_canonical_correlation.min(), 'threshold': .999,
        'fast_operator_is_numerical_approximation': True, 'full_operator': 'physical_common4096Hzpipeline+actualresample_poly'})
    if not passed:
        raise RuntimeError('HOLD_LOW_BAND_OPERATOR_MISMATCH')


@torch.no_grad()
def features(raw, bank, batch=12):
    kernel=torch.fft.rfft(bank, n=NFFT)
    indices=torch.arange(-MAX_LAG, MAX_LAG+1, device='cuda').remainder(NFFT)
    output=[]
    for start in range(0,len(raw),batch):
        x=torch.as_tensor(np.array(raw[start:start+batch], dtype=np.float32), device='cuda')
        x=x-x.mean(-1,keepdim=True)
        data=torch.fft.rfft(x,n=NFFT);power=[]
        for d in range(2):
            pieces=[]
            for k in range(0,len(bank),64):
                c=torch.fft.irfft(data[:,d,None,None]*kernel[None,k:k+64,d].conj(),n=NFFT)[...,indices]
                pieces.append(c.square().sum(2))
            power.append(torch.cat(pieces,1))
        net=(power[0]+F.max_pool1d(power[1],2*NET_LAG+1,stride=1,padding=NET_LAG)).amax(-1)
        x=torch.log1p(torch.stack([p.amax(-1) for p in power]+[net],1)).cpu().numpy()
        output.append(x.reshape(len(x),3,253,3,3).transpose(0,1,3,4,2).reshape(len(x),27,253))
    return np.concatenate(output)


def grouped(root, raw, freq, psds, indices, path):
    if path.with_suffix('.COMPLETE.json').exists():
        return np.load(path, mmap_mode='r')
    path.parent.mkdir(parents=True,exist_ok=True)
    h=spectral_bank(root)
    if not (root/'contracts/LOWBAND_OPERATOR_TEST.json').exists():
        operator_test(root,h,freq,psds[int(indices[0])])
    if not json.loads((root/'contracts/LOWBAND_OPERATOR_TEST.json').read_text())['pass']:
        raise RuntimeError('Operator gate failed')
    partial=path.with_suffix('.partial.npy')
    x=np.lib.format.open_memmap(partial,mode='r+' if partial.exists() else 'w+',dtype=np.float32,shape=(len(raw),27,253))
    progress=path.with_suffix('.progress.json')
    rows=json.loads(progress.read_text()) if progress.exists() else []
    done={r['PSD'] for r in rows}
    for pid in np.unique(indices):
        if int(pid) in done:continue
        keep=np.flatnonzero(indices==pid);tick=time.perf_counter()
        bank=bank_for(h,freq,psds[int(pid)])
        x[keep]=features(raw[keep],bank);del bank
        x.flush()
        rows.append({'PSD':int(pid),'events':len(keep),'seconds':time.perf_counter()-tick})
        dev.json_write(progress,rows)
        print(json.dumps({'lowfeatures':str(path),'done':len(rows),'totalPSD':len(np.unique(indices)),**rows[-1]}),flush=True)
    del x
    partial.rename(path)
    dev.csv_write(path.with_suffix('.audit.csv'),pd.DataFrame(rows))
    dev.json_write(path.with_suffix('.COMPLETE.json'),{'sha256':dev.sha(path),'events':len(raw),'shape':[len(raw),27,253]})
    return np.load(path,mmap_mode='r')


def development_features(root,dep,split):
    folder=root/f'cache/{dep}/{split}'
    if not (folder/'COMPLETE.json').exists():raise RuntimeError('Raw auxiliary branch incomplete')
    meta=pd.read_parquet(t.PREVIOUS/f'expanded_data/{dep}/{split}/event_metadata.parquet')
    base=t.PREVIOUS/f'expanded_data/{dep}/noise'
    return grouped(root,np.load(folder/'low16s.npy',mmap_mode='r'),np.load(base/'frequency.npy'),
        np.load(base/'psd.npy',mmap_mode='r'),meta.noise_bank_index.to_numpy(int),root/f'features/{dep}/{split}.npy')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--stage',choices=['initialize','raw','features','bank'],required=True)
    p.add_argument('--deployment',choices=t.DEPS);p.add_argument('--workers',type=int,default=24)
    a=p.parse_args()
    if a.stage=='initialize':initialize(a.root)
    elif a.stage=='bank':
        h=spectral_bank(a.root)
        for dep in t.DEPS:
            base=t.PREVIOUS/f'expanded_data/{dep}/noise'
            psds=np.load(base/'psd.npy',mmap_mode='r')
            for pid in (0, len(psds)//2, len(psds)-1):
                operator_test(a.root,h,np.load(base/'frequency.npy'),psds[pid],f'{dep}_PSD{pid}')
    else:
        for dep in ((a.deployment,) if a.deployment else t.DEPS):
            for split in ('validation','train'):
                if a.stage=='raw':development_raw(a.root,dep,split,a.workers)
                else:development_features(a.root,dep,split)
