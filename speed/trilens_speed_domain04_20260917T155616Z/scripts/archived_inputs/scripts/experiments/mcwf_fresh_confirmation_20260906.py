#!/usr/bin/env python3
"""Fresh source/noise-block confirmation conditional on frozen lens environments.

No density, weights, model, or sky temperature is fitted here. This is not a
new independent lens-population experiment. BAYESTAR retains the baseline's
conditional Gaussian matched-filter measurement protocol, not full strain PE.
"""
import argparse
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np
import pandas as pd

import mcwf_development_20260905 as dev
import mcwf_encoder_body_20260906 as body
import mcwf_encoder_evaluate_20260906 as evaluate
import mcwf_o3_runmatched_data_20260906 as runmatched

CATALOG_SEEDS = (202609081, 202609082, 202609083)
N_PER_FAMILY = 35
N_BACKGROUND = 50
NOISE_PER_CATALOG = 16


def data_root(root):
    registry = root / "contracts/ACTIVE_CONFIRMATION.json"
    name = json.loads(registry.read_text())["directory"] if registry.exists() else "confirmation"
    if Path(name).name != name:
        raise ValueError("Confirmation registry must name a direct child")
    return root / name


def modules():
    source = dev.module(dev.PROJECT / "scripts/real_search/34_generate_physical_h1l1_source_bank.py", "mcwf_confirm_source")
    data = dev.module(dev.OLD / "scripts/waveform_multiscale_data.py", "mcwf_confirm_data")
    v3 = data.load_module(data.V3_SCRIPT, "mcwf_confirm_injection")
    sky = dev.module(dev.BAY / "scripts/bayestar_injection_sky_full_experiment.py", "mcwf_confirm_bayestar")
    return source, data, v3, sky


def verify_freeze(root):
    path = root / "contracts/FINAL_CANDIDATE_BEFORE_FRESH_TEST.json"
    conf = json.loads(path.read_text())
    for record in conf["frozen_files"]:
        if dev.sha(Path(record["path"])) != record["sha256"]:
            raise RuntimeError("Frozen candidate changed before confirmation")
    return conf


def freeze_protocol(root):
    candidate = verify_freeze(root)
    out = data_root(root)
    out.mkdir(exist_ok=True)
    path = out / "CONFIRMATION_CONTRACT.json"
    if path.exists():
        return
    dev.json_write(path, {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_sha256": dev.sha(root / "contracts/FINAL_CANDIDATE_BEFORE_FRESH_TEST.json"),
        "candidate": candidate["O3_config"], "O4_policy": "original encoder and all scores unchanged",
        "catalog_seeds": CATALOG_SEEDS, "catalogs_per_run": 3,
        "per_catalog": {"new_sources_smooth": 35, "new_sources_subhalo": 35,
                        "unlensed_new_sources": 50, "events": 190, "true_pairs": 70},
        "freshness": "new BBH masses/spins/orientations/sky/phase/GPS, new intrinsic waveform parents, independent noise256s blocks excluded from all training/development/historical references",
        "lens_environments": "existing GW-LMC magnification/delay/Morse configurations may recur; this is conditional source/noise generalization, NOT independent new GW-LMC systems or astrophysical-population rate inference",
        "mass_proposal": "same pre-existing G1 coverage proposal: equal five logMc strata5-10,10-20,20-40,40-80,80-200, q uniform0.25-1, component masses3-300; not an astrophysical rate prior",
        "source_angles": "fresh isotropic sky/orientation/spin tilts; spin magnitudes uniform0-0.8; phases uniform; distance1000Mpc before targetSNR scaling",
        "SNR": "same G1 balanced bins8-10,10-12,12-20,20-40; independent per-image PSD-optimal target scaling, not a response-derived magnification-ratio test",
        "source_waveform": "unchanged IMRPhenomXPHM H1/L1,24s physical buffer,4096Hz; unchanged whitening+anti-aliasing then peak2s4096 model input",
        "noise": "O3-only and O4a-only public off-source;256s references, onsource128s excluded; disjoint reference blocks across catalogs;16 blocks per catalog; noise reused within catalog and must not be counted as190 independent noise realizations",
        "sky": "unchanged baseline conditional BAYESTAR using true intrinsic parameters, local PSD and independent Gaussian matched-filter measurement realization; IMRPhenomPv2 fallback IMRPhenomD; likelihood_scale0.83; fixed per-model baseline posterior temperature;Nside512,NESTED mass",
        "time": "frozen one-dimensional v7/C-fixed lookup applied to new run-exposure timestamps; no density refit",
        "model_seeds": body.MODEL_SEEDS, "old_eval_seeds": dev.SEEDS,
        "paired": "all three frozen models evaluated on every one of three new catalogs; old/new see identical waveform,time,sky",
        "guardrails": {"R10_drop_max": .02, "AUPRC_drop_max": .005, "F50_F90_ratio_max": 1.1},
        "decision": "evaluate once; archive FAIL without silently selecting another model on this confirmation set; confidence intervals descriptive conditional on sampled lens environments",
        "prohibited": ["fit or tune after opening confirmation", "real PE input", "new official-candidate selection", "Hanabi", "overwrite historical outputs"],
    })


def noise_bank(root, dep):
    out = data_root(root) / f"{dep}/noise"
    marker = out / "COMPLETE.json"
    if marker.exists():
        return out
    out.mkdir(parents=True, exist_ok=True)
    source, data, v3, _ = modules()
    if dep == "gwtc3":
        pool = pd.read_csv(root / "data/o3_only_noise/official_O3_full_strain_offsource_pool.csv")
        source_run = dev.MAIN / "cache/source_run"
    else:
        source_run = data.SOURCE_RUNS[dep]
        pool = pd.read_parquet(source_run / "data/real_noise_injections/offsource_noise_segments.parquet")
    excluded = []
    old = pd.read_csv(dev.ORCH.SOURCE_ROOT / dep / "shared/noise_bank_manifest.csv")
    for r in old.itertuples():
        excluded.append((float(r.reference_start_gps) - 16, float(r.reference_start_gps) + float(r.reference_duration_s) + 16))
    manifests = [dev.OLD / f"data/noise_banks/{dep}/noise_manifest_{s}.csv" for s in ("train", "validation")]
    if dep == "gwtc3":
        manifests += [root / f"data/o3_only_noise/noise_manifest_{s}.csv" for s in ("train", "validation")]
    for path in manifests:
        for r in pd.read_csv(path).itertuples():
            excluded.append((float(r.reference_start_gps) - 16, float(r.reference_end_gps) + 16))
    records = []
    for r in pool.to_dict("records"):
        for a, b in runmatched.subtract_intervals(float(r["segment_start"]), float(r["segment_end"]), excluded):
            if b - a >= v3.PSD_SECONDS + 2:
                records.append({**r, "segment_start": a, "segment_end": b})
    pool = pd.DataFrame(records)
    cache = v3.HdfCache(source_run, max_files=8)
    chosen, rejected = data.select_valid_noise_references(pool, NOISE_PER_CATALOG * len(CATALOG_SEEDS),
                        np.random.default_rng(data.stable_seed("confirmation-noise", dep, 202609080)), v3.PSD_SECONDS, cache, v3)
    refs = np.lib.format.open_memmap(out / "reference.npy", mode="w+", dtype=np.float32,
                                   shape=(len(chosen), 2, v3.NOISE_REFERENCE_SAMPLES))
    psds, rows = [], []
    for k, (r, start, reference, freq, psd) in enumerate(chosen):
        if any(start < b and start + v3.PSD_SECONDS > a for a, b in excluded):
            raise RuntimeError("Confirmation noise overlaps development")
        refs[k] = reference
        psds.append(psd)
        rows.append({"bank_index": k, "catalog_seed": CATALOG_SEEDS[k // NOISE_PER_CATALOG],
                     "parent_event": str(r.event_name), "start_gps": start, "end_gps": start + v3.PSD_SECONDS,
                     "H1_path": r.H1_path, "L1_path": r.L1_path})
    refs.flush()
    np.save(out / "psd.npy", np.stack(psds))
    np.save(out / "frequency.npy", freq)
    dev.csv_write(out / "noise_manifest.csv", pd.DataFrame(rows))
    dev.json_write(marker, {"excluded_interval_overlap": 0, "independent_blocks": len(chosen),
                   "source_run": str(source_run), "rejections": rejected, "reference_sha256": dev.sha(out / "reference.npy")})
    return out


def plan_catalog(root, dep, catalog_seed):
    out = data_root(root) / f"{dep}/catalog_{catalog_seed}"
    path = out / "source_systems.parquet"
    if path.exists():
        return pd.read_parquet(path)
    source, data, _, _ = modules()
    out.mkdir(parents=True, exist_ok=True)
    table = source.load_tables(source.GW_LMC_ROOT)
    schedule = source.load_schedule(dev.ORCH.SOURCE_ROOT / dep / "shared/h1l1_live_schedule.csv")
    rng = np.random.default_rng(data.stable_seed(catalog_seed, dep, "new-source-plan"))
    masses = data.balanced_mass_draws(2 * N_PER_FAMILY + N_BACKGROUND, rng)
    records = []
    for sid, (m1, m2, mass_bin) in enumerate(masses):
        family = "SIS" if sid < N_PER_FAMILY else "PM" if sid < 2 * N_PER_FAMILY else "UNL"
        row = None
        if family != "UNL":
            subset = table.loc[table.lens_is_subhalo.astype(bool) == (family == "PM")]
            for rid in rng.permutation(subset.index):
                proposed = source.choose_images(table.loc[rid])
                if proposed is None or proposed["proposal_snr_ratio"] > 4:
                    continue
                times = source.place_pair(proposed["delay_days"], schedule, rng)
                if times is not None:
                    row = table.loc[rid]
                    lens = proposed
                    break
            if row is None:
                raise RuntimeError("No exposure-compatible lens environment")
        else:
            weights = np.asarray([(s.end - s.start) * s.weight_per_second for s in schedule])
            seg = schedule[int(rng.choice(len(schedule), p=weights / weights.sum()))]
            times = (float(rng.uniform(seg.start, seg.end)), np.nan)
            lens = {"delay_days": np.nan, "mu_image1": 1., "mu_image2": np.nan,
                    "morse_image1": 0., "morse_image2": np.nan}
        uid = f"FRESH_BBH_{dep}_{catalog_seed}_{sid:04d}"
        records.append({"source_uid": uid, "source_index": sid, "family_slot": family,
            "gwlmc_environment_row": int(row.gwlmc_row) if row is not None else -1,
            "gwlmc_environment_id": int(row.event_id) if row is not None else -1,
            "m1_det": m1, "m2_det": m2, "mc_det": data.chirp_mass(m1, m2), "mass_bin": mass_bin,
            "a1": rng.uniform(0, .8), "a2": rng.uniform(0, .8),
            "tilt1": np.arccos(rng.uniform(-1, 1)), "tilt2": np.arccos(rng.uniform(-1, 1)),
            "theta_jn": np.arccos(rng.uniform(-1, 1)), "phi12": rng.uniform(0, 2*np.pi),
            "phijl": rng.uniform(0, 2*np.pi), "psi": rng.uniform(0, np.pi), "phase": rng.uniform(0, 2*np.pi),
            "ra": rng.uniform(0, 2*np.pi), "dec": np.arcsin(rng.uniform(-1, 1)), "dl_source": 1000.,
            "gps_a": times[0], "gps_b": times[1],
            "snr_a": data.snr_draw((2*sid) % 4, rng), "snr_b": data.snr_draw((2*sid+1) % 4, rng), **lens})
    result = pd.DataFrame(records)
    result.to_parquet(path, index=False)
    dev.json_write(out / "PLAN_SHA256.json", {"sha256": dev.sha(path), "frozen_before_signal_generation": True,
        "fresh_BBH_sources": len(result), "reused_lens_environment_allowed": True})
    return result


_CTX = {}


def worker_init(root, dep, catalog_seed):
    import torch
    torch.set_num_threads(1)
    source, data, v3, sky = modules()
    logging.getLogger("bilby").setLevel(logging.ERROR)
    noise = data_root(Path(root)) / f"{dep}/noise"
    _CTX.update(root=Path(root), dep=dep, catalog_seed=catalog_seed, source=source, data=data, v3=v3, sky=sky,
        refs=np.load(noise / "reference.npy", mmap_mode="r"),
        freq=np.load(noise / "frequency.npy"), psds=np.load(noise / "psd.npy", mmap_mode="r"),
        generator=source.build_waveform_generator(),
        ifos=[source.bilby.gw.detector.get_empty_interferometer(x) for x in ("H1", "L1")])
    sky._WORKER_PSD_FREQ = _CTX["freq"]
    sky._WORKER_PSD_BANK = _CTX["psds"]


def generate_system(record):
    import healpy as hp
    from ligo.skymap.io.fits import write_sky_map
    ctx = _CTX
    src, data, v3, sky = (ctx[x] for x in ("source", "data", "v3", "sky"))
    out = data_root(ctx["root"]) / f"{ctx['dep']}/catalog_{ctx['catalog_seed']}/events"
    out.mkdir(parents=True, exist_ok=True)
    marker = out / f"{record['source_uid']}.json"
    if marker.exists():
        return json.loads(marker.read_text())
    rng = np.random.default_rng(data.stable_seed(record["source_uid"], "independent-noise"))
    block_start = CATALOG_SEEDS.index(ctx["catalog_seed"]) * NOISE_PER_CATALOG
    output = []
    for image, number in (("a", 1), ("b", 2)):
        if image == "b" and record["family_slot"] == "UNL":
            continue
        event_uid = record["source_uid"] + "_" + image
        gps = float(record[f"gps_{image}"])
        factor = src.lens_factor(record[f"mu_image{number}"], record[f"morse_image{number}"])
        clean, peak_info = src.detector_response(ctx["generator"], ctx["ifos"], src.source_parameters(pd.Series(record), gps), factor)
        bi = int(rng.integers(NOISE_PER_CATALOG)) + block_start
        reference = ctx["refs"][bi]
        offset = int(rng.integers(reference.shape[-1] - v3.RAW_PADDED_SAMPLES + 1))
        noise = np.asarray(reference[:, offset:offset + v3.RAW_PADDED_SAMPLES], dtype=np.float64)
        target = float(record[f"snr_{image}"])
        _, mixed, inj_audit = v3._preprocess_injection(clean, noise, ctx["freq"], ctx["psds"][bi], target)
        if not np.isfinite(mixed).all():
            raise RuntimeError("Nonfinite fresh injection")
        event = {"event_uid": event_uid, "source_uid": record["source_uid"], "source_index": record["source_index"],
            "image": image, "family_slot": record["family_slot"], "mc_det": record["mc_det"],
            "gps_obs": gps, "ra_true": record["ra"], "dec_true": record["dec"],
            "target_network_snr": target, "noise_bank_index": bi, "noise_offset_samples": offset,
            "measurement_seed": data.stable_seed(event_uid, "BAYESTAR-measurement"),
            "morse_index": record[f"morse_image{number}"]}
        np.save(out / f"{event_uid}_full24.npy", mixed.astype(np.float32))
        started = time.perf_counter()
        fallback = False
        try:
            skymap, sky_audit = sky._simulate_with_waveform(event, pd.Series(record), sky.PRIMARY_WAVEFORM)
        except Exception as exc:
            event["primary_sky_failure"] = repr(exc)
            fallback = True
            skymap, sky_audit = sky._simulate_with_waveform(event, pd.Series(record), sky.FALLBACK_WAVEFORM)
        probability = sky.raster_probability(skymap, 512)
        if abs(probability.sum() - 1) > 1e-10 or (probability < 0).any():
            raise RuntimeError("Invalid fresh sky map")
        write_sky_map(out / f"{event_uid}.fits", skymap, nest=True)
        np.save(out / f"{event_uid}_sky512.npy", probability.astype(np.float32))
        order = np.sort(probability)[::-1]
        event.update(peak_info)
        event.update({"sky_fallback": fallback, "map_seconds": time.perf_counter()-started,
            "A90_deg2": float((np.searchsorted(np.cumsum(order), .9)+1)*hp.nside2pixarea(512, degrees=True)),
            "truth_pixel_mass": float(probability[hp.ang2pix(512, np.pi/2-record["dec"], record["ra"], nest=True)]),
            "full24_sha256": dev.sha(out / f"{event_uid}_full24.npy"),
            "moc_sha256": dev.sha(out / f"{event_uid}.fits"),
            "sky_ordering": "NESTED", "sky_frame": "ICRS", "analysis_nside": 512,
            "injection_audit": inj_audit, "sky_audit": sky_audit})
        output.append(event)
    dev.json_write(marker, output)
    return output


def generate(root, dep, workers):
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp
    freeze_protocol(root)
    noise_bank(root, dep)
    for cs in CATALOG_SEEDS:
        out = data_root(root) / f"{dep}/catalog_{cs}"
        marker = out / "GENERATION_COMPLETE.json"
        if marker.exists():
            continue
        plan = plan_catalog(root, dep, cs)
        events = []
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"),
                 initializer=worker_init, initargs=(str(root), dep, cs)) as executor:
            for k, result in enumerate(executor.map(generate_system, plan.to_dict("records"))):
                events.extend(result)
                if k % 10 == 0:
                    print(json.dumps({"fresh_generation": dep, "catalog_seed": cs, "systems_done": k+1}), flush=True)
        frame = pd.DataFrame(events)
        frame.insert(0, "idx", np.arange(len(frame)))
        frame.to_parquet(out / "event_manifest.parquet", index=False)
        dev.json_write(marker, {"events": len(frame), "source_systems": len(plan),
            "true_pairs": int((frame.family_slot != "UNL").sum() // 2),
            "manifest_sha256": dev.sha(out / "event_manifest.parquet"), "model_not_fit": True})


def old_waveform(dep, eval_seed, full, frame):
    v7 = dev.BASE.v7
    checkpoint = dev.V7 / dep / f"seed_{eval_seed}/waveform_gate/unified_inception_attention_peak2s_4096_aux_0p25_q_1p0_v7/validation_selected_model.pt"
    model, payload = v7.load_unified_model(checkpoint)
    z, pred = v7.embed_catalog(model, v7.ArrayCatalog(full), batch_size=16)
    del model
    i, j = frame.idx_i.to_numpy(int), frame.idx_j.to_numpy(int)
    out = frame.copy()
    out["waveform_embedding_cosine"] = np.sum(z[i] * z[j], axis=1)
    out["waveform_abs_delta_logmc_std"] = np.abs(pred[i, 0] - pred[j, 0])
    out["waveform_abs_delta_logitq_std"] = np.abs(pred[i, 1] - pred[j, 1])
    calibration = json.loads((dev.V7 / dep / f"seed_{eval_seed}/results/waveform_channel_calibration_v7.json").read_text())
    return v7.apply_waveform_channel(out, calibration)


def score(root, dep):
    conf = verify_freeze(root)
    config = conf["O3_config"]
    _, _, _, sky = modules()
    sky_config = json.loads((dev.BAY / "contracts/selected_config.json").read_text())
    time_cal = json.loads((dev.V7 / dep / "shared/time_delay_likelihood_ratio.json").read_text())
    rows = []
    for cs in CATALOG_SEEDS:
        out = data_root(root) / f"{dep}/catalog_{cs}"
        events = pd.read_parquet(out / "event_manifest.parquet")
        i, j = np.triu_indices(len(events), 1)
        truth = events.source_uid.to_numpy()[i] == events.source_uid.to_numpy()[j]
        family = np.where(truth, events.family_slot.to_numpy()[i], "NONE")
        delays = np.abs(events.gps_obs.to_numpy()[i] - events.gps_obs.to_numpy()[j]) / 86400
        base = pd.DataFrame({"idx_i": i, "idx_j": j, "is_true_pair": truth, "true_pair_family": family,
                             "event_count": len(events), "delta_t_days": delays,
                             "time_score": dev.BASE.v7.apply_time_likelihood_ratio(delays, time_cal)})
        full = np.stack([np.load(out / f"events/{uid}_full24.npy") for uid in events.event_uid])
        raw_maps = np.stack([np.load(out / f"events/{uid}_sky512.npy") for uid in events.event_uid])
        sky_cache = {}
        for ms, es in zip(body.MODEL_SEEDS, dev.SEEDS):
            saved = out / f"model_{ms}"
            saved.mkdir(exist_ok=True)
            if (saved / "COMPLETE.json").exists():
                rows.extend(pd.read_csv(saved / "metrics.csv").to_dict("records"))
                continue
            temperature = float(sky_config["deployments"][dep][str(es)]["posterior_temperature"])
            if temperature not in sky_cache:
                maps = np.stack([sky.apply_temperature(p, temperature).astype(np.float32) for p in raw_maps])
                zsky, bc, audit = sky.gpu_pair_features(maps)
                del maps
                sky_cache[temperature] = (zsky, bc)
                dev.json_write(out / f"sky_float64_audit_T{temperature:g}.json", audit)
            zsky, bc = sky_cache[temperature]
            frame = base.copy()
            frame["sky_raw_log_bf"] = zsky
            frame["sky_bc"] = bc
            frame = old_waveform(dep, es, full, frame)
            oldscore = frame.waveform_score.to_numpy()
            if dep == "gwtc3":
                checkpoint = root / f"models/{config}/{dep}/seed_{ms}/validation_selected_model.pt"
                if config == "PSD-PHASE-SOURCE":
                    import torch
                    import mcwf_adaptive_psd_encoder_20260906 as adaptive
                    payload = torch.load(checkpoint, weights_only=False, map_location="cpu")
                    raw2 = dev.TRAIN.make_window_view(np.asarray(full[..., -4096:], np.float32), 2)
                    noise = data_root(root) / f"{dep}/noise"
                    x = adaptive.extract_grouped(root, raw2, np.load(noise / "frequency.npy"),
                        np.load(noise / "psd.npy", mmap_mode="r"), events.noise_bank_index.to_numpy(int),
                        out / "adaptive_psd_features.npy")
                    model = body.Encoder(config).cuda().eval()
                    model.load_state_dict(payload["model"])
                    logits, z = body.infer(model, (x-payload["mu"])/payload["sd"], raw2)
                    lp = logits.astype(float)/payload["temperature"]
                    lp -= np.logaddexp.reduce(lp, -1, keepdims=True)
                    p = np.exp(lp)
                    del model
                else:
                    p, z, _ = body.encode(checkpoint, full)
                f = evaluate.features(frame, p, z)
                cal = json.loads((root / f"evaluation/{config}/{dep}/model_{ms}_eval_{es}/SELECTED_CONFIG.json").read_text())
                newscore, raw, ood = evaluate.apply(f, cal["NEW-PHYSICAL"])
                candidate = evaluate.combine(oldscore, newscore, cal["VAL-ENSEMBLE"])
                frame["new_raw_waveform"] = raw
                frame["new_waveform_OOD"] = ood
            else:
                candidate = oldscore.copy()
            per_seed = []
            for name, waveform in (("BASELINE", oldscore), ("CANDIDATE", candidate)):
                changed = frame.copy()
                changed["waveform_score"] = waveform
                changed.to_parquet(saved / f"{name}_pairs.parquet", index=False)
                for method, m in evaluate.metrics(changed, waveform, dep, es).items():
                    per_seed.append({"deployment": dep, "catalog_seed": cs, "model_seed": ms, "eval_seed": es,
                                     "variant": name, "method": method, **m})
            if dep == "gwtc4" and not np.array_equal(candidate, oldscore):
                raise RuntimeError("O4 frozen fallback changed")
            dev.csv_write(saved / "metrics.csv", pd.DataFrame(per_seed))
            rows.extend(per_seed)
            dev.json_write(saved / "COMPLETE.json", {"new_fits": 0, "time_sky_identical": True,
                "posterior_temperature": temperature, "O4_original_encoder": dep == "gwtc4"})
            print(json.dumps({"fresh_scored": dep, "catalog_seed": cs, "model_seed": ms}), flush=True)
        del full, raw_maps
    dev.csv_write(data_root(root) / f"{dep}/metrics_all.csv", pd.DataFrame(rows))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--action", choices=("generate", "score", "all"), required=True)
    p.add_argument("--deployment", choices=("gwtc3", "gwtc4"), required=True)
    p.add_argument("--workers", type=int, default=6)
    args = p.parse_args()
    if args.action in ("generate", "all"):
        generate(args.root, args.deployment, args.workers)
    if args.action in ("score", "all"):
        score(args.root, args.deployment)
