"""Post-freeze numerical and population diagnostics, never score selection."""
import os
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[k] = '1'
import argparse
import json
from pathlib import Path
import time

import healpy as hp
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import uab_completion as c
import uab_scoring as s
from uab_evaluate import csv


def injection_resolution(run, arm, split):
    c.gate(split)
    p = c.deployment(run, arm)/'sky_pair_scores'/split
    if c.check_complete(p/'REFERENCE1024_COMPLETE.json'):
        return
    import torch
    from ligo.skymap.io.fits import read_sky_map
    b = c.U.module(c.ROOT/'scripts/bayestar_si_fixed.py', 'uab_reference_maps')
    events = pd.read_parquet(p/'events.parquet')
    frame = pd.read_parquet(p/'pairs.parquet')
    temperature = json.loads((c.OUT/'maps'/run/arm/'validation/TEMPERATURE_SELECTED.json').read_text())['temperature']
    npix, n = hp.nside2npix(1024), len(events)
    begin = time.monotonic()
    bank = np.empty((n, npix), np.float32)
    for k, row in enumerate(events.itertuples(index=False)):
        if c.U.sha(row.moc_path) != row.moc_sha256:
            raise RuntimeError('Native map changed')
        native = read_sky_map(row.moc_path, moc=True)
        bank[k] = hp.reorder(b.apply_temperature(b.raster_probability(native, 1024), temperature), n2r=True)
    norm = bank.sum(1, dtype=np.float64)
    overlap = torch.zeros((n, n), dtype=torch.float64, device='cuda')
    for first in range(0, npix, 65536):
        block = torch.as_tensor(np.asarray(bank[:, first:first+65536], np.float64)/norm[:, None], device='cuda')
        overlap += block@block.T
    value = overlap.cpu().numpy()[frame.idx_i, frame.idx_j]
    frame['sky_log_bf_nside1024'] = np.log(np.maximum(npix*value, 1e-300))
    del bank, block, overlap
    torch.cuda.empty_cache()
    frame['null_high_tail_1pct'] = (~frame.is_true_pair) & (frame.sky_raw_log_bf >= frame.loc[~frame.is_true_pair, 'sky_raw_log_bf'].quantile(.99))
    null_indices = frame.index[~frame.is_true_pair]
    ordered = sorted(null_indices, key=lambda i: c.U.stable('resolution-null-v1', run, split, frame.loc[i, 'event_i'], frame.loc[i, 'event_j']))
    frame['fixed_hash_null_sample'] = frame.index.isin(ordered[:1000])
    frame['sign_flip_256_512'] = np.sign(frame.sky_log_bf_nside256) != np.sign(frame.sky_log_bf_nside512)
    frame['large_delta_256_512'] = abs(frame.sky_log_bf_nside256-frame.sky_log_bf_nside512) > .05
    frame['predeclared_target'] = frame.is_true_pair | frame.null_high_tail_1pct | frame.fixed_hash_null_sample | frame.sign_flip_256_512 | frame.large_delta_256_512
    frame['sign_flip_512_1024'] = np.sign(frame.sky_log_bf_nside512) != np.sign(frame.sky_log_bf_nside1024)
    frame['abs_delta_512_1024'] = abs(frame.sky_log_bf_nside512-frame.sky_log_bf_nside1024)
    frame.to_parquet(p/'convergence_all_pairs.parquet', index=False)
    c.seal(p/'REFERENCE1024_COMPLETE.json', [p/'convergence_all_pairs.parquet'],
           seconds=time.monotonic()-begin, dense_persisted=False, analysis_nside_unchanged=512,
           full_reference_exceeds_target_minimum=True, test_result_does_not_select_nside=True)


def resolution_summary():
    rows, retrieval = [], []
    for run in c.U.RUNS:
        for arm in c.U.ARMS:
            for split in ('validation', 'test'):
                p = c.deployment(run, arm)/'sky_pair_scores'/split
                f = pd.read_parquet(p/'convergence_all_pairs.parquet')
                events = pd.read_parquet(p/'events.parquet')
                f['event_count'] = len(events)
                f['true_pair_family'] = np.where(f.is_true_pair, events.family.to_numpy()[f.idx_i], 'unlensed')
                for name, mask in [('all', np.ones(len(f), bool)), ('true_companion', f.is_true_pair),
                    ('all_null', ~f.is_true_pair), ('fixed_hash_null', f.fixed_hash_null_sample),
                    ('null_high_tail_1pct', f.null_high_tail_1pct), ('predeclared_targets', f.predeclared_target)]:
                    g = f.loc[mask]; delta = g.abs_delta_512_1024.to_numpy()
                    rows.append(dict(run=run, arm=arm, split=split, population=name, pairs=len(g),
                        sign_flips=int(g.sign_flip_512_1024.sum()), sign_flip_fraction=float(g.sign_flip_512_1024.mean()),
                        Spearman=float(spearmanr(g.sky_log_bf_nside512, g.sky_log_bf_nside1024).statistic),
                        **{label: float(np.quantile(delta, q)) for label, q in [('median', .5), ('P90', .9), ('P95', .95), ('P99', .99), ('P999', .999), ('maximum', 1)]},
                        no_independent_pair_UCB=True, audit_only_not_new_resolution_selection=True))
                for nside in (256, 512, 1024):
                    retrieval.append(dict(run=run, arm=arm, split=split, nside=nside,
                        **s.query_metrics(f, f[f'sky_log_bf_nside{nside}'].to_numpy())))
        f = pd.read_parquet(c.OUT/'real_sky'/run/'pairs.parquet')
        for name, g in [('all_real_pairs', f)]:
            delta = g.abs_delta_512_1024.to_numpy()
            rows.append(dict(run=run, arm='shared_public_PE', split='real', population=name, pairs=len(g),
                sign_flips=int(g.flip_512_1024.sum()), sign_flip_fraction=float(g.flip_512_1024.mean()),
                Spearman=float(spearmanr(g.sky_log_bf_nside512, g.sky_log_bf_nside1024).statistic),
                **{label: float(np.quantile(delta, q)) for label, q in [('median', .5), ('P90', .9), ('P95', .95), ('P99', .99), ('P999', .999), ('maximum', 1)]},
                no_independent_pair_UCB=True, audit_only_not_new_resolution_selection=True))
    csv(c.OUT/'tables/sky_resolution_summary.csv', pd.DataFrame(rows))
    csv(c.OUT/'tables/sky_resolution_retrieval.csv', pd.DataFrame(retrieval))


def model_population():
    rows, model, sky_quality = [], [], []
    import o4b_hl_nso_multiscale_training_20260912 as models
    low, _, _ = models.modules()
    centers = np.exp(np.asarray(low.old.CENTERS))
    for run in c.U.RUNS:
        for arm in c.U.ARMS:
            root = c.deployment(run, arm)
            for split in ('validation', 'test'):
                e = pd.read_parquet(root/f'inference_inputs/{split}/events.parquet')
                sky = pd.read_parquet(c.OUT/'maps'/run/arm/split/'events.parquet')
                temperature = json.loads((c.OUT/'maps'/run/arm/'validation/TEMPERATURE_SELECTED.json').read_text())['temperature']
                weight = 1/sky.groupby('source_uid').event_uid.transform('size').to_numpy(float)
                credible = sky[f'truth_credible_T{temperature:g}'].to_numpy(float)
                sky_quality.append(dict(run=run, arm=arm, split=split, events=len(sky),
                    frozen_validation_temperature=temperature,
                    raw_HPD90_coverage=float(np.average(sky.truth_credible_T1 <= .9, weights=weight)),
                    calibrated_HPD50_coverage=float(np.average(credible <= .5, weights=weight)),
                    calibrated_HPD90_coverage=float(np.average(credible <= .9, weights=weight)),
                    raw_A90_median_deg2=float(sky.area90_deg2.median()),
                    fallback_events=int(sky.fallback_used.sum()),
                    localization_seconds_P50=float(sky.seconds.median()),
                    localization_seconds_P90=float(sky.seconds.quantile(.9)),
                    localization_seconds_max=float(sky.seconds.max()),
                    coverage_weighting='one total weight per source system',
                    test_does_not_select_temperature=True))
                g = e.groupby('source_uid').first()
                calendar = pd.read_csv(c.ROOT/'plans'/run/'live_schedule.csv')
                live = np.zeros(len(e), bool)
                for a, b in zip(calendar.start_gps, calendar.end_gps):
                    live |= (e.gps_obs >= a) & (e.gps_obs <= b)
                if not live.all():
                    raise RuntimeError('Synthetic event outside its own run calendar')
                i, j = np.triu_indices(len(e), 1)
                same = e.source_uid.to_numpy()[i] == e.source_uid.to_numpy()[j]
                mc = e.mc_det.to_numpy(float)
                close = abs(mc[i]-mc[j])/((mc[i]+mc[j])/2) <= .1
                rows.append(dict(run=run, arm=arm, split=split, events=len(e), source_parents=len(g),
                    lens_groups=g.global_source_id.nunique(), noise_parents=e.noise_parent_uid.nunique(),
                    all_GPS_in_own_run=bool(live.all()), calendar_span_days=(calendar.end_gps.max()-calendar.start_gps.min())/86400,
                    live_days=float((calendar.end_gps-calendar.start_gps).sum()/86400),
                    SNR_median=float(e.target_network_snr.median()), Mc_median=float(e.mc_det.median()),
                    null_Mc_within10pct=float(close[~same].mean()),
                    unrelated_neighbours_7d_mean=float((np.abs(e.gps_obs.to_numpy()[:, None]-e.gps_obs.to_numpy()[None, :]) < 7*86400)[e.source_uid.to_numpy()[:, None] != e.source_uid.to_numpy()[None, :]].sum()/len(e))))
                for seed in c.U.SEEDS:
                    p = root/f'predictions_o4b/seed_{seed}/{split}'
                    short = dict(np.load(p/'short.npz')); emb = short['z'].astype(float)
                    eig = np.linalg.svd(emb-emb.mean(0), compute_uv=False)**2
                    fraction = eig/eig.sum()
                    erank = np.exp(-np.sum(fraction*np.log(fraction.clip(1e-300))))
                    for branch in ('ordered', 'multirate'):
                        prob = dict(np.load(p/f'{branch}.npz'))['p']
                        cdf = prob.cumsum(1)
                        quantile = lambda q: centers[(cdf >= q).argmax(1)]
                        median = quantile(.5)
                        model.append(dict(run=run, arm=arm, split=split, seed=seed, branch=branch,
                            relative_Mc_MAE=float(np.mean(abs(median-mc)/mc)), Mc_bias=float(np.mean(median-mc)),
                            central50_coverage=float(((mc >= quantile(.25)) & (mc <= quantile(.75))).mean()),
                            central90_coverage=float(((mc >= quantile(.05)) & (mc <= quantile(.95))).mean()),
                            embedding_effective_rank=float(erank), distributions_not_Bayesian_PE=True))
    csv(c.OUT/'tables/population_calendar_noise_audit.csv', pd.DataFrame(rows))
    csv(c.OUT/'tables/Mc_prediction_embedding_diagnostics.csv', pd.DataFrame(model))
    csv(c.OUT/'tables/sky_truth_coverage_and_runtime.csv', pd.DataFrame(sky_quality))


def contract():
    path = c.OUT/'contracts/EVALUATION_COMPLETION_CONTRACT.json'
    if path.exists():
        return
    c.write(path, dict(experiment='GWLR-UAB-01', created_before_final_score_freeze=True,
        evaluation=dict(primary='450 events, 180 true pairs, 90 singletons', size_control='500 preset190 event subsets,70 true pairs',
            bootstrap_replicates=1000, bootstrap_seed='SHA256 UAB-paired-bootstrap-v1:run:seed:unit',
            source_bootstrap='family stratified waveform-parent counts; true edge weight m_g, null edge m_i*m_j',
            noise_bootstrap='noise-parent counts, pair weight m_i*m_j, query weight of query noise parent',
            intervals='separate conditional sensitivity intervals, not jointly independent confidence',
            arm_delta='identical cluster resamples for B minus A', tie_rule='expected retrieval and TopB; threshold-inclusive F50/F90'),
        sky_reference=dict(analysis_nside=512, reference_nside=1024, coarse_nside=256,
            temperature='same validation-frozen value at every resolution per arm/run; publicPE T=1',
            targets=['all companions', '1000 SHA256-ranked null pairs', '512 null top1%', '256/512 sign flips', 'abs256-512 > .05'],
            actual_reference='all pairs, exceeds targeted minimum', no_dense_persistence=True,
            selection_does_not_change_primary_scores=True, no_posthoc_tolerance_change=True),
        real=dict(event_counts=dict(O3=62,O4a=74,O4b=86), maps='unchanged public PE, explicit RING at512',
            PE='same bounded Gaussian KDE with deterministic max4096 samples and4096 grid',
            official='frozen O3 PO/ML and O4a PO/Phazap tables; O4b NA, not zero',
            ranking_frozen_before_PE=True, no_new_Hanabi=True),
        limitations=['per-image SNR scaling is controlled, not response-derived population',
            'B intentionally preserves artificial SNR class cue', 'six evaluation noise parents per run',
            'known-intrinsic Gaussian BAYESTAR triggers are not recovered full PE from noisy strain',
            'real public maps can include Virgo whereas waveform and injection triggers use HL',
            'same method and selection rules, not same fitted parameters',
            '500 subcatalogs overlap and are not independent trials',
            'historically inspected event populations; no new real-data blind confirmation claim'],
        final_state='HOLD_FOR_AUTHOR_REVIEW_NO_ADOPTION_NO_OVERWRITE'))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--stage', choices=['contract', 'resolution', 'summarize', 'population'], required=True)
    p.add_argument('--run', default='O3'); p.add_argument('--arm', default='A_NEUTRAL')
    p.add_argument('--split', choices=['validation','test'], default='test')
    a = p.parse_args(); c.initialize(a.root, a.out)
    if a.stage == 'contract': contract()
    elif a.stage == 'resolution': injection_resolution(a.run, a.arm, a.split)
    elif a.stage == 'summarize': resolution_summary()
    else: model_population()
