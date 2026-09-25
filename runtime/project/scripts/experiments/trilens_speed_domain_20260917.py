"""Conditional common-domain runtime benchmark, not a detection comparison."""
import argparse
import hashlib
import itertools
import json
import multiprocessing as mp
import os
from pathlib import Path
import resource
import shutil
import signal
import time
import traceback

import trilens_scale_fixed_replay_20260917 as E


def eligibility_job(item):
    begin, cpu = time.perf_counter(), time.process_time()
    root, row = item
    try:
        value = E.phase_job(item)
        return dict(**value, eligible=True, classification='FINITE_FULL_POSTERIOR',
                    nonfinite_rows=0, error=None)
    except Exception as error:
        record = dict(event_name=row['event_name'], eligible=False,
                      classification='UNEXPECTED_FAILURE', error=repr(error),
                      wall_s=time.perf_counter()-begin, cpu_s=time.process_time()-cpu,
                      peak_rss_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
        if str(error) == 'Nonfinite phase posterior':
            import numpy as np
            from phazap.postprocess_phase import PostprocessedPhase
            path = Path(root)/'phases'/f'{row["event_name"]}.hdf5'
            phase = PostprocessedPhase.from_file(str(path))
            arrays = {name: np.asarray(value) for name, value in phase.dataset.items()}
            count = len(next(iter(arrays.values())))
            bad = np.zeros(count, dtype=bool)
            fields = {}
            for name, values in arrays.items():
                mask = ~np.isfinite(values)
                if mask.ndim > 1:
                    mask = mask.reshape(count, -1).any(axis=1)
                bad |= mask
                fields[name] = int(mask.sum())
            record.update(classification='NONFINITE_PHASE_AT_FROZEN_FREQUENCIES',
                          path=str(path), samples=count, nonfinite_rows=int(bad.sum()),
                          nonfinite_fraction=float(bad.mean()), nonfinite_fields=fields)
        return record


def common_domain(chosen, rows):
    dest = E.ROOT/'results/eligibility'
    for folder in ('phases', 'contracts', 'logs'):
        (dest/folder).mkdir(parents=True)
    records = []
    begin = time.perf_counter()
    ctx = mp.get_context('spawn')
    with ctx.Pool(E.WORKERS, initializer=E.phase_initializer,
                  initargs=(str(dest/'logs'),)) as pool:
        jobs = [(str(dest), row) for row in chosen.to_dict('records')]
        for value in pool.imap_unordered(eligibility_job, jobs, chunksize=1):
            records.append(value)
            E.PD.DataFrame(records).to_csv(E.ROOT/'tables/event_eligibility.csv', index=False)
            E.status('DOMAIN_PREFLIGHT', completed_events=len(records), total_events=len(chosen),
                     ineligible=sum(not r['eligible'] for r in records))
            E.guard()
    frame = E.PD.DataFrame(records)
    errors = frame[frame.classification.eq('UNEXPECTED_FAILURE')]
    E.js(E.ROOT/'contracts/ELIGIBILITY_COST.json', dict(wall_s=time.perf_counter()-begin,
         worker_cpu_s_sum=float(frame.cpu_s.sum()), workers=E.WORKERS,
         note='One-time domain-selection audit, separate from each measured frontend pass'))
    if len(errors):
        raise RuntimeError('Unexpected preflight failures; not classified as domain exclusions: '+
                           str(errors[['event_name', 'error']].to_dict('records')))
    eligible = set(frame.loc[frame.eligible, 'event_name'])
    filtered = chosen[chosen.event_name.isin(eligible)].copy().reset_index(drop=True)
    aligned = rows.set_index('event_name', drop=False).loc[filtered.event_name].reset_index(drop=True)
    if len(filtered) < 46:
        raise RuntimeError('Common domain has fewer than 46 events; 1000 distinct pairs unavailable')
    filtered.to_csv(E.ROOT/'contracts/common_events.csv', index=False)
    aligned.to_csv(E.ROOT/'contracts/common_native_rows.csv', index=False)
    sizes = sorted(set([5, 15, 46, len(filtered)]))
    scopes = []
    for n in sizes:
        keys = sorted('--'.join(sorted(pair)) for pair in
                      itertools.combinations(filtered.event_name.iloc[:n], 2))
        scopes.append(dict(n_events=n, n_pairs=len(keys),
                           pair_key_sha256=hashlib.sha256(('\n'.join(keys)+'\n').encode()).hexdigest()))
    E.js(E.ROOT/'contracts/COMMON_SCOPE_FROZEN.json', dict(
        utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), original_events=len(chosen),
        eligible_events=len(filtered), excluded_events=len(chosen)-len(filtered),
        original_pairs=len(chosen)*(len(chosen)-1)//2,
        eligible_pairs=len(filtered)*(len(filtered)-1)//2,
        event_manifest_sha256=E.digest(E.ROOT/'contracts/common_events.csv'), scopes=scopes,
        policy='Entire nonfinite events excluded from BOTH methods before timing; hash order unchanged',
        no_ranking_quality_or_runtime_selection=True, sample_deletion=False))
    return filtered, aligned, sizes


def run_phazap(chosen, n):
    dest = E.ROOT/f'results/Phazap_n{n}'
    for folder in ('phases', 'contracts', 'logs'):
        (dest/folder).mkdir(parents=True)
    E.status('PHAZAP_PREPARE', n_events=n, workers=E.WORKERS)
    begin, cpu = time.perf_counter(), time.process_time()
    ctx = mp.get_context('spawn')

    def prepare():
        records = []
        with ctx.Pool(min(n, E.WORKERS), initializer=E.phase_initializer,
                      initargs=(str(dest/'logs'),)) as pool:
            jobs = [(str(dest), row) for row in chosen.iloc[:n].to_dict('records')]
            for value in pool.imap_unordered(E.phase_job, jobs, chunksize=1):
                records.append(value)
                E.PD.DataFrame(records).to_csv(dest/'event_preparation.csv', index=False)
                E.status('PHAZAP_PREPARE', n_events=n, completed_events=len(records))
                E.guard()
        files = {row['event_name']: row['path'] for row in records}
        ready = ctx.Queue()
        pool = ctx.Pool(E.WORKERS, initializer=E.pair_initializer,
                       initargs=(files, ready, str(dest/'logs')))
        try:
            initialized = [ready.get(timeout=180) for _ in range(E.WORKERS)]
        except BaseException:
            pool.terminate()
            pool.join()
            raise
        E.js(dest/'contracts/PAIR_WORKERS.json', initialized)
        return pool

    pool, _ = E.timed('Phazap', n, 'PREPARE_EVENTS', 0, prepare)
    pairs = list(itertools.combinations(chosen.event_name.iloc[:n], 2))
    reference = None
    try:
        for rep in range(E.REPEATS):
            def score():
                values = []
                for value in pool.imap_unordered(E.pair_job, pairs, chunksize=1):
                    values.append(value)
                    if len(values) % 25 == 0 or len(values) == len(pairs):
                        E.PD.DataFrame(values).to_csv(dest/f'pair_checkpoint_rep{rep}.csv', index=False)
                        E.status('PHAZAP_PAIRS', n_events=n, repetition=rep,
                                 completed_pairs=len(values), total_pairs=len(pairs))
                        E.guard()
                frame = E.PD.DataFrame(values).sort_values('pair_key').reset_index(drop=True)
                frame.sort_values(['DJ', 'pair_key']).to_csv(dest/f'all_pairs_rep{rep}.csv', index=False)
                return frame

            frame, record = E.timed('Phazap', n, 'EVENT_CACHE_PAIR_RANK', rep, score)
            record['worker_cpu_s_sum'] = float(frame.worker_cpu_s.sum())
            record['worker_job_wall_s_sum'] = float(frame.worker_wall_s.sum())
            if rep == 0:
                E.TIMINGS.append(dict(method='Phazap', n_events=n, n_pairs=len(pairs),
                    boundary='PRODUCTS_READY_TOTAL', repetition=0,
                    wall_s=time.perf_counter()-begin, parent_cpu_s=time.process_time()-cpu))
            E.PD.DataFrame(E.TIMINGS).to_csv(E.ROOT/'tables/scaling_timings.csv', index=False)
            E.B.record_check(f'Phazap_count_n{n}_r{rep}',
                            len(frame) == len(pairs) and frame.pair_key.nunique() == len(pairs))
            trilens = E.PD.read_csv(E.ROOT/f'results/TriLens_n{n}/seed_202607241_rep{rep}.csv')
            E.B.record_check(f'identical_pairs_n{n}_r{rep}', set(trilens.pair_key) == set(frame.pair_key))
            old = E.PD.read_csv(E.PILOT/'fullposterior_diagnostic/tables/phazap_full_pairs.csv')
            old = old[old.samples.eq('full')].copy()
            old['pair_key'] = ['--'.join(sorted([a, b])) for a, b in zip(old.event_i, old.event_j)]
            expected = len(set(frame.pair_key) & set(old.pair_key))
            overlap = frame.merge(old, on='pair_key', suffixes=('_new', '_old'))
            E.B.record_check(f'Phazap_prior_replay_n{n}_r{rep}', len(overlap) == expected and
                E.NP.allclose(overlap.DJ_new, overlap.DJ_old, atol=1e-8, rtol=1e-8),
                pairs=len(overlap), max_abs_delta_DJ=None if not len(overlap) else
                float(abs(overlap.DJ_new-overlap.DJ_old).max()))
            if reference is not None:
                E.B.record_check(f'Phazap_repeat_n{n}_r{rep}',
                    frame.pair_key.tolist() == reference.pair_key.tolist() and
                    E.NP.allclose(reference.DJ, frame.DJ, atol=1e-8, rtol=1e-8),
                    max_abs_delta_DJ=float(abs(reference.DJ-frame.DJ).max()))
            reference = frame
    except BaseException:
        pool.terminate()
        pool.join()
        raise
    else:
        pool.close()
        pool.join()
    E.js(dest/'COMPLETE.json', dict(pass_replay=True, identical_pair_scope=True,
                                   official_FPP_not_computed=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    E.ROOT = args.root

    def timeout(signum, frame):
        raise TimeoutError('Three-hour speed benchmark limit reached')

    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(10800)
    success, error = False, None
    try:
        chosen, rows, recipes = E.init()
        shutil.copy2(__file__, E.ROOT/'scripts'/Path(__file__).name)
        contract = dict(code='TRILENS-SPEED-DOMAIN-04',
            replaces_domain_and_failure_policy_in='CONTRACT.json', inherited_science_unchanged=True,
            objective='Conditional runtime only, not matched-recall or official-front-end reproduction',
            original_scope=62, workers=E.WORKERS, full_posterior=True, phase_frequencies_Hz=[20, 40, 100],
            domain_rule='Preflight every frozen event; exclude entire events with any nonfinite phase posterior from BOTH methods',
            unexpected_exception_rule='HOLD; do not relabel software/resource failures as domain exclusions',
            ordering='Original SHA256 name order after deterministic eligibility filter',
            scales='5,15,46 and all eligible events; no duplicated pairs; HOLD if fewer than46 eligible',
            exclusions_reported=True, rankings_and_timings_not_selection_inputs=True,
            no_nan_filling=True, no_sample_deletion=True, no_frequency_changes=True,
            no_phazap_algorithm_change=True, no_posterior_generation=True,
            products_ready='Original public PE/sky and strain available to each method; include event preparation',
            domain_preflight_cost='Reported separately; included in one-time audit cost, not hidden',
            regenerate_phases_for_each_size=True, cached_pair_repetitions=2,
            same_hardware_budget=False, trilens='CUDA GPU plus two numeric CPU threads',
            phazap='Six single-thread CPU workers; process startup charged',
            shared_machine=True, no_OS_cache_flush=True,
            limits=dict(seconds=10800, minimum_disk_GiB=30),
            no_UAB_interruption=True, no_historical_overwrite=True,
            scientific_status=E.FINAL)
        E.js(E.ROOT/'contracts/SPEED_ONLY_DOMAIN_CONTRACT.json', contract)
        E.js(E.ROOT/'contracts/SPEED_ONLY_FREEZE.json', dict(
            utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            contract_sha256=E.digest(E.ROOT/'contracts/SPEED_ONLY_DOMAIN_CONTRACT.json'),
            driver_sha256=E.digest(Path(__file__)), engine_sha256=E.digest(Path(E.__file__))))
        common, aligned, sizes = common_domain(chosen, rows)
        for n in sizes:
            E.guard()
            E.run_trilens(common, aligned, recipes, n)
            run_phazap(common, n)
        success = True
    except BaseException as exc:
        error = repr(exc)
        if E.ROOT.exists():
            (E.ROOT/'logs/FAIL.txt').write_text(traceback.format_exc())
        raise
    finally:
        if E.ROOT.exists():
            E.finish(success, error)
        signal.alarm(0)


if __name__ == '__main__':
    main()
