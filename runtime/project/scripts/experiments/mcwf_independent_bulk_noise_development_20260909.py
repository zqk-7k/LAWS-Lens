#!/usr/bin/env python3
"""Independent run-matched off-source acquisition after local pool exhaustion."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import json
from pathlib import Path
import shutil
import sys
import time
from urllib.parse import urlparse
import h5py
import numpy as np
import pandas as pd
import requests
from gwosc.api import fetch_allevents_json
from gwosc.locate import get_urls

P = Path('/root/autodl-tmp/gw-catalog')
sys.path.insert(0, str(P/'scripts/experiments'))
import mcwf_independent_profile_development_20260909 as g
n = g.n
ROOT = None
SEED = 2026090924


def freeze():
    g.freeze()
    n.write_json(ROOT/'contracts/BULK_NOISE_ACQUISITION_ADDENDUM.json', {'UTC': n.utc(),
        'id': 'MCWF-NODUP-INDEPENDENT-DEVELOPMENT-22B', 'prior_attempt': str(P/'results/mcwf_nodup_independent_profile_development_22_20260909T112429Z'),
        'prior_attempt_status': 'HOLD_NO_INDEPENDENT_NOISE_REMAINING:31validO3blocks,2000nextblocktrialsallfinite_fractionfail;noinjectiongenerated.',
        'change': 'Acquirepublic4096HzH1/L1runstrainfromGWOSC atseededjoint-liveintervals,notonlyexistingevent-adjacentfiles.',
        'fresh_seed': SEED, 'sources_per_run': 1024, 'source_fold': '512fit/512tune',
        'noise': '64new256sblocks/run;4blocksper4096sH1L1chunkpair;8uniquechunkpairsperfold;fit/tuneHDFGPSchunksdisjoint. Existing31reservedblocksalsoexcluded.',
        'planning': 'Up to96 preordered live-intervalproposals/run,minimumduration2048s,probabilityproportionaltoduration. O3onlyO3a/b,alternatesO3a/b andfit/tune;O4aonlyO4a.',
        'event_veto': 'Allunique publicGWOSCeventGPS+-128s fromfrozenalleventsJSON,plusallhistoricalnoiseintervals+-16s.',
        'quality': 'Finiteallstrain samples,4096Hz,length256s,DATAandCBC_CAT1,NO_CBC_HW_INJ;finitepositiveWelchPSD. Do notrejectonretrieval,PE,officialorMcperformance.',
        'storage': 'PubliccompressedHDF5 onlyselectedorattemptedchunks;atmost20GiBrawnewdata,minimum25GiBfree;64compactnoisereferencesandmixedpeak/lowviews.',
        'statistical_limits': '1024sourceparents,64noiseblocksbutonly16HDFtimechunkpairs/run;reportbothclusterlevels. Thisisnewcalibrationdevelopment,notlockedtest.',
        'frozen': ['allmodels', 'time', 'sky', 'outerweights', 'scopes', 'historicaloutputs'],
        'prior_noise_rejections_field': 'Oldgeneratorcolumnlen(dict)notactualcounts;newlogandmanifestuseactualcounts.',
        'references': ['https://gwosc.org/data/', 'https://gwosc.readthedocs.io/en/stable/reference/gwosc.locate.get_urls.html'],
        'no_method_adoption': True})
    shutil.copy2(__file__, ROOT/'scripts/independent_bulk_noise_development.py')
    n.write_json(ROOT/'contracts/BULK_NOISE_FREEZE.json', {'UTC': n.utc(), 'script_sha256': n.sha(Path(__file__)),
        'addendum_sha256': n.sha(ROOT/'contracts/BULK_NOISE_ACQUISITION_ADDENDUM.json')})


def download(url):
    dest = ROOT/'raw_sources'/Path(urlparse(url).path).name
    dest.parent.mkdir(parents=True, exist_ok=True)
    receipt = dest.with_suffix('.source.json')
    if dest.exists() and receipt.exists():
        if n.sha(dest) != json.loads(receipt.read_text())['sha256']:
            raise RuntimeError('Downloaded input hash changed')
        return dest
    if shutil.disk_usage(ROOT).free < 25*2**30 or sum(p.stat().st_size for p in dest.parent.glob('*.hdf5')) > 20*2**30:
        raise RuntimeError('HOLD_DISK_LIMIT')
    temporary = dest.with_suffix('.partial')
    last = None
    for attempt in range(3):
        try:
            with requests.get(url, stream=True, timeout=(30, 120)) as response:
                response.raise_for_status()
                with temporary.open('wb') as file:
                    for chunk in response.iter_content(1024*1024):
                        file.write(chunk)
            with h5py.File(temporary) as file:
                if 'strain/Strain' not in file:
                    raise RuntimeError('MissingGWOSCstrain')
            temporary.rename(dest)
            n.write_json(receipt, {'UTC': n.utc(), 'url': url, 'bytes': dest.stat().st_size,
                'sha256': n.sha(dest), 'attempts': attempt+1})
            print('NEW_PUBLIC_NOISE_DOWNLOAD', dest.name, dest.stat().st_size, flush=True)
            return dest
        except (requests.RequestException, OSError) as error:
            last = error
            time.sleep(2*(attempt+1))
    raise RuntimeError('Publicstrain downloadfailed:'+str(last))


def plan(dep):
    path = ROOT/f'contracts/{dep}_BULK_NOISE_PLAN.csv'
    if path.exists():
        return pd.read_csv(path)
    schedule = pd.read_csv(n.dev.ORCH.SOURCE_ROOT/dep/'shared/h1l1_live_schedule.csv')
    allowed = ['O3a', 'O3b'] if dep == 'gwtc3' else ['O4a']
    schedule = schedule[schedule.run.isin(allowed) & (schedule.duration_s >= 2048)].copy()
    if len(schedule) == 0:
        raise RuntimeError('No run-matched liveintervals')
    rng = np.random.default_rng(SEED+(3 if dep == 'gwtc3' else 4))
    rows = []
    for k in range(96):
        run = allowed[(k//2)%len(allowed)]
        part = schedule[schedule.run == run]
        row = part.iloc[rng.choice(len(part), p=part.duration_s.to_numpy()/part.duration_s.sum())]
        middle = rng.uniform(row.start_gps+512, row.end_gps-512)
        rows.append({'proposal': k, 'fold': k%2, 'run': run, 'live_start': row.start_gps,
            'live_end': row.end_gps, 'query_gps': int(middle)})
    n.write_csv(path, rows)
    n.write_json(path.with_suffix('.json'), {'UTC': n.utc(), 'sha256': n.sha(path), 'before_any_data_read': True})
    return pd.DataFrame(rows)


def read(path):
    with h5py.File(path) as file:
        start = float(file['meta/GPSstart'][()]); duration = float(file['meta/Duration'][()])
        strain = np.asarray(file['strain/Strain'][:], np.float32)
        if len(strain)/duration != 4096:
            raise RuntimeError('Sample rate mismatch')
        names = [v.decode() if isinstance(v, bytes) else str(v) for v in file['quality/simple/DQShortnames'][:]]
        bits = file['quality/simple/DQmask'][:]
        mask = np.ones(len(bits), bool)
        for flag in ('DATA', 'CBC_CAT1'):
            mask &= (bits & (1 << names.index(flag))) != 0
        names = [v.decode() if isinstance(v, bytes) else str(v) for v in file['quality/injections/InjShortnames'][:]]
        inj = file['quality/injections/Injmask'][:]
        mask &= (inj & (1 << names.index('NO_CBC_HW_INJ'))) != 0
        if len(mask) != int(duration):
            raise RuntimeError('Unexpected dataquality clock')
    return strain, start, duration, mask


def noise(dep):
    folder = ROOT/f'data/{dep}/noise'
    folder.mkdir(parents=True, exist_ok=True)
    if (folder/'COMPLETE.json').exists():
        return
    _, _, v3 = g.expanded.modules()
    plans = plan(dep)
    eventfile = ROOT/'contracts/GWOSC_ALL_EVENTS.json'
    if not eventfile.exists():
        n.write_json(eventfile, fetch_allevents_json())
        n.write_json(eventfile.with_suffix('.sha256.json'), {'UTC': n.utc(), 'sha256': n.sha(eventfile)})
    gps = sorted({float(v['GPS']) for v in json.loads(eventfile.read_text())['events'].values() if v.get('GPS') is not None})
    exclusions = g.exclusions(dep)
    excluded = [(float(row.start)-16, float(row.end)+16) for row in exclusions.itertuples()]
    excluded += [(value-128, value+128) for value in gps]
    path = folder/'reference.npy'
    references = np.lib.format.open_memmap(path, mode='r+' if path.exists() else 'w+', dtype=np.float32,
                                          shape=(64, 2, v3.NOISE_REFERENCE_SAMPLES))
    output = folder/'noise_manifest.csv'
    rows = pd.read_csv(output).to_dict('records') if output.exists() else []
    used = {float(v['parent_file_gps']) for v in rows}
    for row in rows:
        excluded.append((row['start_gps']-16, row['end_gps']+16))
    for pp in plans.itertuples():
        side = [row for row in rows if row['fold'] == pp.fold]
        if len(side) == 32:
            continue
        stamp = folder/f'proposal_{pp.proposal:03d}.json'
        if stamp.exists():
            continue
        urls = [get_urls(det, pp.query_gps, pp.query_gps+1, dataset=pp.run, sample_rate=4096) for det in ('H1', 'L1')]
        if any(len(v) != 1 for v in urls):
            n.write_json(stamp, {'accepted': False, 'reason': 'No unique paired4096Hz runfiles', 'urls': urls})
            continue
        paths = [download(v[0]) for v in urls]
        arrays = [read(path) for path in paths]
        if arrays[0][1:3] != arrays[1][1:3]:
            raise RuntimeError('H1/L1fileclock mismatch')
        start, duration = arrays[0][1:3]
        if start in used:
            n.write_json(stamp, {'accepted': False, 'reason': 'Alreadyusedtimechunk'})
            continue
        times = np.arange(np.ceil(max(pp.live_start, start)+16), np.floor(min(pp.live_end, start+duration)-272), 288.)
        rng = np.random.default_rng(SEED+pp.proposal+(300 if dep == 'gwtc3' else 400))
        accepted, reasons = [], {'historical_or_event_overlap': 0, 'quality': 0, 'nonfinite': 0, 'psd': 0}
        for tt in rng.permutation(times):
            if any(tt < b and tt+256 > a for a, b in excluded):
                reasons['historical_or_event_overlap'] += 1
                continue
            off = int(round(tt-start))
            if not all(a[3][off:off+256].all() and len(a[3][off:off+256]) == 256 for a in arrays):
                reasons['quality'] += 1
                continue
            ref = np.stack([a[0][off*4096:(off+256)*4096] for a in arrays])
            if ref.shape != (2, v3.NOISE_REFERENCE_SAMPLES) or not np.isfinite(ref).all():
                reasons['nonfinite'] += 1
                continue
            frequency, psd = v3.estimate_psd(ref)
            if not np.isfinite(psd).all() or (psd <= 0).any():
                reasons['psd'] += 1
                continue
            accepted.append((tt, ref, frequency, psd))
            if len(accepted) == 4:
                break
        if len(accepted) < 4:
            n.write_json(stamp, {'accepted': False, 'valid_blocks': len(accepted), 'reasons': reasons})
            continue
        for tt, ref, frequency, psd in accepted:
            index = int(pp.fold*32+len([r for r in rows if r['fold'] == pp.fold]))
            references[index] = ref; references.flush()
            np.save(folder/f'psd_{index:03d}.npy', psd); np.save(folder/'frequency.npy', frequency)
            excluded.append((tt-16, tt+272))
            rows.append({'bank_index': index, 'fold': int(pp.fold), 'parent_event': f'BULK_{pp.run}_{int(start)}',
                'parent_file_gps': start, 'proposal': int(pp.proposal), 'run': pp.run, 'start_gps': tt, 'end_gps': tt+256,
                'H1_path': str(paths[0]), 'L1_path': str(paths[1]), 'rejections': sum(reasons.values()),
                'complete_finite_H1L1': True, 'DATA_CBC_CAT1_NO_CBC_HW_INJ': True})
            n.write_csv(output, sorted(rows, key=lambda r: r['bank_index']))
        used.add(start)
        n.write_json(stamp, {'accepted': True, 'blocks': 4, 'reasons': reasons, 'fold': int(pp.fold)})
        print('INDEPENDENT_BULK_NOISE', dep, len(rows), 64, flush=True)
        if len(rows) == 64:
            break
    if len(rows) != 64:
        raise RuntimeError(f'HOLD_NOISE_SUPPORT:{dep}:{len(rows)}/64')
    frame = pd.DataFrame(rows)
    if set(frame.parent_file_gps[frame.fold == 0]) & set(frame.parent_file_gps[frame.fold == 1]):
        raise RuntimeError('Parentfileleakage')
    np.save(folder/'psd.npy', np.stack([np.load(folder/f'psd_{k:03d}.npy') for k in range(64)]))
    n.write_json(folder/'COMPLETE.json', {'UTC': n.utc(), 'blocks': 64, 'source_noise_fold_intersection': 0,
        'parent_file_pairs': 16, 'parent_file_fold_intersection': 0, 'finite_all': True,
        'reference_sha256': n.sha(path), 'psd_sha256': n.sha(folder/'psd.npy'), 'event_catalog_sha256': n.sha(eventfile)})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--stage', choices=('freeze', 'noise', 'generate'), required=True)
    parser.add_argument('--workers', type=int, default=20)
    args = parser.parse_args()
    ROOT = g.ROOT = args.root
    g.SEED = SEED
    if args.stage == 'freeze':
        freeze()
    else:
        g.noise = noise
        for dep in n.DEPS:
            if args.stage == 'noise':
                noise(dep)
            else:
                g.generate(dep, args.workers)
