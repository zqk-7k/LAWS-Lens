#!/usr/bin/env python3
"""ET-only, non-destructive NEW-SCORE-ONLY data and geometry preflight.

This entry point prepares inputs, not final NEW-SCORE-ONLY scores. Existing
models/time calibrations are read-only. No test rankings are used here.
"""
from __future__ import annotations

import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"
import argparse
from collections import namedtuple
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import logging
import math
import multiprocessing as mp
from pathlib import Path
import shutil
import sys
import time
import traceback

import numpy as np
import pandas as pd
from scipy import signal

P = Path("/root/autodl-tmp/gw-catalog")
RAW = Path("/root/autodl-tmp/createdata/et3_10000_20260616_1006_match_root")
BASE = P / "results/et3_v7_aligned_20260723/formal"
PATH875 = P / "results/mcwf_unified_path875_devconf_20260908T181500Z"
SKY_BASE = P / "results/bayestar_injection_sky_pe_20260901_20260901T102000Z"
OLD_SKY = Path("/root/autodl-fs/et3_sky_resolution_v93_20260730T072501Z")
SEEDS = (202607251, 202607252, 202607253, 202607254, 202607255)
FS = 4096
NSAMPLES = 24 * FS
TC = NSAMPLES - 1500
TEMPERATURES = (1., 1.25, 1.5, 2., 2.5, 3., 4.)
STATUS = "HOLD_FOR_AUTHOR_REVIEW_NO_ADOPTION_NO_OVERWRITE"


def utc():
    return datetime.now(timezone.utc).isoformat()


def json_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False,
                              default=lambda x: x.item() if isinstance(x, np.generic) else str(x)) + "\n")
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def stable_seed(*values):
    return int.from_bytes(hashlib.sha256("|".join(map(str, values)).encode()).digest()[:4], "little")


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def storage_guard(root):
    free = shutil.disk_usage(root).free
    if free < 30 * 2**30:
        raise RuntimeError("HOLD_DISK_LIMIT: less than 30 GiB free")
    return free


def arrays():
    for family in ("SIS", "PM", "unlensed"):
        group = "Unlensed_data_0222" if family == "unlensed" else family + "_data_0222"
        for image in ((0,) if family == "unlensed" else (1, 2)):
            suffix = "" if image == 0 else f"_{image}"
            folder = RAW / group
            yield family, image, folder, folder / f"{family}_data_strain{suffix}.npy"


def protect_inputs(root):
    files = [P / "data_generation/detector_network.py",
             P / "scripts/experiments/109_materialize_et3_v7_preprocessing.py",
             P / "scripts/experiments/110_et3_v7_aligned_pipeline.py"]
    for source in (BASE, OLD_SKY / "contracts", PATH875 / "contracts", SKY_BASE / "scripts"):
        if not source.exists():
            raise FileNotFoundError(source)
        for f in source.rglob("*"):
            if f.is_file() and f.suffix in (".json", ".csv", ".npz", ".pt", ".py"):
                files.append(f)
    for _, _, folder, _ in arrays():
        files.extend(folder.glob("*.csv"))
        files.extend(folder.glob("*SNR*.npy"))
    entries = [{"path": str(f), "sha256": sha(f), "bytes": f.stat().st_size}
               for f in sorted(set(files))]
    json_write(root / "manifests/PROTECTED_INPUTS.json", entries)


def verify_protected(root):
    records = json.loads((root / "manifests/PROTECTED_INPUTS.json").read_text())
    changes = [x["path"] for x in records if sha(x["path"]) != x["sha256"]]
    result = {"utc": utc(), "checked_files": len(records), "changes": changes,
              "all_small_protected_inputs_unchanged": not changes,
              "large_strain_arrays": "read-only memmaps; not full-content hashed in this preflight"}
    json_write(root / "manifests/PROTECTED_VERIFICATION.json", result)
    if changes:
        raise RuntimeError("Protected input content changed")


def initialize(root):
    root.mkdir(parents=True, exist_ok=False)
    for folder in ("contracts", "scripts", "logs", "manifests", "tables", "reports",
                   "pilot/maps", "pilot/audit", "cache/long16s", "models", "results"):
        (root / folder).mkdir(parents=True, exist_ok=True)
    shutil.copy2(__file__, root / "scripts" / Path(__file__).name)
    contract = {
        "code": "ET3-NEW-SCORE-ONLY-BAYESTAR", "created_utc": utc(),
        "reference": str(PATH875), "outer_old_new_total_score_alpha": 1.0,
        "reference_warning": "PATH875 used adaptive real-catalog development; ET selection uses simulation only",
        "final_score": "wW*Zwf_new+wT*Ztime+wS*log(Npix*sum(Pi*Pj)); no outer score mixture",
        "waveform_target": "retain ET 2s encoder and OMC chain, add 16s matched features and joint intrinsic distribution; Zwf_new=Zwf_OMC+gamma*T+beta*I",
        "not_NODUP_DIRECT": True, "not_R75_R77": True,
        "frozen_encoder_seeds": list(SEEDS), "frozen_base": str(BASE),
        "short_window": {"seconds_relative_tc": [-1.75, .25], "sampling_rate": 2048, "samples": 4096},
        "long_window": {"seconds_relative_tc": [-15.75, .25], "sampling_rate": 256, "samples": 4096,
                        "band_Hz": [20, 80], "source_already_whitened": True,
                        "downsample": "resample_poly 4096->2048->256; Kaiser beta=8.6"},
        "detectors": ["ET1", "ET2", "ET3"],
        "geometry": "exact original Bilby vertices/tensors, not stock LAL E1/E2/E3 geometry",
        "sky": {"method": "event-specific conditional BAYESTAR",
                "measurement_model": "independent Gaussian matched-filter realizations conditional on injection intrinsics/ET PSD/network SNR; not full strain-level BBH PE",
                "templates_rotated": False, "likelihood_scale": .83,
                "analysis_nside": 512, "audit_nsides": [256, 512, 1024],
                "ordering": "NESTED canonical MOC raster; explicitly recorded; common permutation is score-invariant",
                "temperature_grid": TEMPERATURES, "temperature_selection": "validation system-weighted HPD90 only; not pilot/test performance",
                "spin_conversion_mass_units": "SI kg as required by Bilby; original source spin reference=10Hz",
                "noise_covariance": "independent channels, matching archived generator; not a claim about real ET shared-arm noise",
                "instantaneous_response": True, "no_sky_performance_target": True},
        "source_splits": "reuse exact per-seed saved source split; auxiliary early stopping subset taken only from that seed's training systems",
        "pilot_scope": "intersection of all five training sets, no validation/test rankings opened",
        "calibration": "simulation development/validation only; never physical truth injected as an inference feature except explicitly conditional BAYESTAR",
        "time": "exact frozen ET one-dimensional likelihood table and synthetic event calendar; no 2D time-SNR extension",
        "report_requirements": ["SIS/PM/combined R1,R5,R10", "AUPRC,F50,F90,TopB", "per-seed and system bootstrap", "true-Mc and predicted distribution calibration", "sky convergence and HPD coverage"],
        "no_real_ET_PE_or_official_pair_labels": True,
        "storage": {"minimum_free_GiB": 30, "dense_sky_persistence": False,
                    "long_input_cache_GiB": 50000 * 3 * 4096 * 4 / 2**30},
        "implemented_stages_in_this_entrypoint": ["freeze", "data/geometry audit", "conditional sky pilot", "full 16s input materialization"],
        "not_yet_completed": ["ET OMC/multiscale auxiliary training", "full sky generation/calibration", "new fusion evaluation"],
        "status_after_preparation": "INPUTS_READY_AUXILIARY_TRAINING_PENDING",
        "final_adoption_status": STATUS,
        "references": ["https://arxiv.org/abs/1508.03634", "https://arxiv.org/abs/gr-qc/9402014",
                       "https://bilby-dev.github.io/bilby/api/bilby.gw.conversion.bilby_to_lalsimulation_spins.html"],
    }
    json_write(root / "contracts/ANALYSIS_CONTRACT.json", contract)
    protect_inputs(root)
    json_write(root / "contracts/CONTRACT_SHA256.json", {"sha256": sha(root / "contracts/ANALYSIS_CONTRACT.json")})
    print(json.dumps({"root": str(root), "initialized": True}), flush=True)


def geometry():
    import bilby
    import lal
    import lalsimulation
    ifos = list(bilby.gw.detector.InterferometerList(["ET"]))
    result = []
    detectors = {}
    for idx, ifo in enumerate(ifos):
        original = lalsimulation.DetectorPrefixToLALDetector(f"E{idx + 1}")
        d = lal.Detector()
        d.location[:] = ifo.vertex
        d.response[:] = ifo.detector_tensor
        detectors[ifo.name] = d
        result.append({"detector": ifo.name, "vertex_m": ifo.vertex.tolist(),
                       "response": ifo.detector_tensor.tolist(),
                       "stock_LAL_vertex_difference_m": float(np.linalg.norm(original.location - ifo.vertex)),
                       "stock_LAL_tensor_max_difference": float(np.max(abs(original.response - ifo.detector_tensor))),
                       "adapter_tensor_error": float(np.max(abs(d.response - ifo.detector_tensor)))})
    return ifos, detectors, result


@contextmanager
def detector_lookup(detectors):
    # BAYESTAR resolves detector names internally. Override only inside this
    # isolated process and restore afterward; installed libraries remain unchanged.
    import lalsimulation
    original = lalsimulation.DetectorPrefixToLALDetector
    def lookup(name):
        return detectors[name] if name in detectors else original(name)
    lalsimulation.DetectorPrefixToLALDetector = lookup
    try:
        yield
    finally:
        lalsimulation.DetectorPrefixToLALDetector = original


def common_training(family):
    members = []
    for seed in SEEDS:
        with np.load(BASE / f"seed_{seed}/split_indices.npz") as splits:
            members.append(set(splits[f"{family}_train"].tolist()))
    return sorted(set.intersection(*members))


def make_pilot_records():
    from bilby.gw.conversion import luminosity_distance_to_redshift
    from astropy.cosmology import Planck18
    records = []
    for family, image, folder, array in arrays():
        if image == 2:
            continue
        sources = pd.read_csv(folder / "source_samples.csv")
        ids = common_training(family)
        suffix = "" if image == 0 else "_1"
        snr = np.load(folder / f"{family}_optimal_SNR_network{suffix}.npy")
        ordered = sorted(ids, key=lambda i: (snr[i], i))
        chosen = [ordered[round(q * (len(ordered)-1))] for q in (.1, .35, .65, .9)]
        lens = None if image == 0 else pd.read_csv(folder / "lens.csv")
        for idx in chosen:
            source = sources.iloc[idx].to_dict()
            z = float(luminosity_distance_to_redshift(source["luminosity_distance"], cosmology=Planck18))
            for im in ((0,) if image == 0 else (1, 2)):
                suffix = "" if im == 0 else f"_{im}"
                snr_file = folder / f"{family}_optimal_SNR_network{suffix}.npy"
                single_file = folder / f"{family}_optimal_SNR_single{suffix}.npy"
                gps = source["geocent_time"] + (float(lens.iloc[idx].t_d) if im == 2 else 0.)
                records.append({**source, "family": family, "image": im, "source_index": idx,
                                "event_uid": f"{family}:{idx}:{im}", "source_uid": f"{family}:{idx}",
                                "gps": gps, "m1_det": source["mass_1_source"]*(1+z),
                                "m2_det": source["mass_2_source"]*(1+z), "z": z,
                                "target_network_snr": float(np.load(snr_file)[idx]),
                                "archived_single_snr": np.load(single_file)[idx].tolist(),
                                "measurement_seed": stable_seed("ET3-NSO-20260912", family, idx, im),
                                "morse": .5 if im == 2 else 0.,
                                "array": str(folder / f"{family}_data_strain{suffix}.npy")})
    return records


def low_view(white):
    white = np.asarray(white, dtype=np.float64)
    if white.shape[-2:] != (3, NSAMPLES) or not np.isfinite(white).all():
        raise ValueError("Invalid three-channel, already-whitened ET input")
    sos = signal.butter(6, (20, 80), fs=FS, btype="bandpass", output="sos")
    tapered = white * signal.windows.tukey(NSAMPLES, .02)
    low = signal.sosfiltfilt(sos, tapered, axis=-1)
    half = signal.resample_poly(low, 1, 2, axis=-1, window=("kaiser", 8.6))
    median = np.median(half, axis=-1, keepdims=True)
    scale = 1.4826 * np.median(abs(half - median), axis=-1, keepdims=True)
    if np.any(scale <= 0):
        raise ValueError("Degenerate long-window channel")
    half = ((half - median) / scale).astype(np.float32)
    # Crop at the exact coalescence-relative phase before the second decimator.
    # Taking the last 4096 samples would shift the window by 476 raw samples.
    start = (TC - int(15.75 * FS)) // 2
    pad = 256
    aligned = half[..., start-pad:start+32768+pad]
    down = signal.resample_poly(aligned, 1, 8, axis=-1, window=("kaiser", 8.6))
    out = down[..., pad//8:pad//8+4096].astype(np.float32)
    if out.shape[-2:] != (3, 4096) or not np.isfinite(out).all():
        raise ValueError("Invalid 16 second view")
    return out


def audit(root):
    import bilby
    ifos, detectors, details = geometry()
    assert all(x["adapter_tensor_error"] < 3e-8 for x in details)
    json_write(root / "tables/GEOMETRY.json", details)
    summaries = []
    for family, image, folder, path in arrays():
        a = np.load(path, mmap_mode="r")
        assert a.shape == (10000, 3, NSAMPLES)
        summaries.append({"family": family, "image": image, "path": str(path.resolve()),
                          "shape": list(a.shape), "dtype": str(a.dtype), "bytes": path.stat().st_size,
                          "mtime_ns": path.stat().st_mtime_ns,
                          "content_kind": "already PSD-whitened noisy time series; not physical strain"})
    json_write(root / "manifests/RAW_ARRAY_METADATA.json", summaries)
    pre = load_module(P / "scripts/experiments/109_materialize_et3_v7_preprocessing.py", "et3_frozen_preprocessor")
    tests = []
    records = make_pilot_records()
    for row in records:
        raw = np.asarray(np.load(row["array"], mmap_mode="r")[row["source_index"]])
        context = raw[..., -6*FS:] * signal.windows.tukey(6*FS, .02)
        filtered = signal.sosfiltfilt(signal.butter(6, (40, 580), fs=FS, btype="bandpass", output="sos"), context, axis=-1)
        short = pre.robust_scale_channels(signal.resample_poly(filtered, 1, 2, axis=-1,
                    window=("kaiser", 8.6))[..., pre.target_slice(NSAMPLES, NSAMPLES-6*FS)])
        original = P / "data_generation/et3_v7_preprocessed_20260723/polyphase" / Path(row["array"]).parent.name / Path(row["array"]).name
        expected = np.asarray(np.load(original, mmap_mode="r")[row["source_index"]])
        long = low_view(raw)
        if not np.array_equal(short, expected):
            raise RuntimeError(f"Short-view replay mismatch: {row['event_uid']}")
        np.save(root / "pilot" / (row["event_uid"].replace(":", "_") + "_16s.npy"), long)
        tests.append({"event_uid": row["event_uid"], "short_bit_exact": True,
                      "long_shape": list(long.shape), "long_std": long.std(-1).tolist(),
                      "input_channel_pair_exact_equal": [bool(np.array_equal(raw[a], raw[b])) for a,b in ((0,1),(0,2),(1,2))]})
    json_write(root / "contracts/PILOT_EVENTS.json", records)
    json_write(root / "tables/PREPROCESSING_REPLAY.json", tests)
    split_audit = []
    for seed in SEEDS:
        with np.load(BASE / f"seed_{seed}/split_indices.npz") as s:
            for family in ("SIS", "PM", "unlensed"):
                sets = {name: set(s[f"{family}_{name}"].tolist()) for name in ("train", "val", "test")}
                overlaps = {a+"_"+b: len(sets[a]&sets[b]) for a,b in (("train","val"),("train","test"),("val","test"))}
                if any(overlaps.values()):
                    raise RuntimeError("Source split overlap")
                split_audit.append({"seed": seed, "family": family, **overlaps})
    pd.DataFrame(split_audit).to_csv(root / "tables/SPLIT_AUDIT.csv", index=False)
    json_write(root / "contracts/DATA_PREFLIGHT_PASS.json", {"utc": utc(), "pilot_events": len(records),
               "short_views_bit_exact": True, "geometry_adapter_pass": True,
               "raw_arrays_already_whitened": True, "test_rankings_opened": False})
    verify_protected(root)


def simulate_sky(row, root):
    import bilby
    import lal
    import healpy as hp
    from ligo.skymap.bayestar import filter, localize
    from ligo.skymap.tool.bayestar_realize_coincs import simulate_snr
    from ligo.skymap.io.events.base import Event, SingleEvent
    from ligo.skymap.io.fits import write_sky_map
    from ligo.skymap import moc
    logging.getLogger("ligo.skymap").setLevel(logging.ERROR)
    root = Path(root)
    prefix = row["event_uid"].replace(":", "_")
    output = root / "pilot/maps" / (prefix + ".fits.gz")
    audit_file = root / "pilot/audit" / (prefix + ".json")
    if audit_file.exists() and output.exists():
        saved = json.loads(audit_file.read_text())
        if saved["map_sha256"] != sha(output):
            raise RuntimeError("Existing pilot map hash mismatch")
        return saved
    ifos, detectors, _ = geometry()
    vals = (row["theta_jn"], row["phi_jl"], row["tilt_1"], row["tilt_2"], row["phi_12"],
            row["a_1"], row["a_2"])
    spins = bilby.gw.conversion.bilby_to_lalsimulation_spins(*vals, row["m1_det"]*lal.MSUN_SI,
                         row["m2_det"]*lal.MSUN_SI, 10., row["phase"])
    bad_spins = bilby.gw.conversion.bilby_to_lalsimulation_spins(*vals, row["m1_det"], row["m2_det"], 10., row["phase"])
    args = {"mass1": row["m1_det"], "mass2": row["m2_det"], "f_final": 1024., "f_ref": 10.}
    args.update(dict(zip(("spin1x", "spin1y", "spin1z", "spin2x", "spin2y", "spin2z"), spins[1:])))
    waveform = "IMRPhenomPv2"
    h = filter.sngl_inspiral_psd(waveform, f_min=20., **args)
    frequency = np.arange(0., 2048.25, .25)
    series = []
    for ifo in ifos:
        psd = ifo.power_spectral_density.get_power_spectral_density_array(frequency)
        psd = np.where((frequency >= 20.) & np.isfinite(psd) & (psd > 0), psd, np.inf)
        s = lal.CreateREAL8FrequencySeries("ET-D", 0, 0., .25, lal.StrainUnit**2, len(psd))
        s.data.data[:] = psd
        series.append(s)
    interpolated = [filter.InterpolatedPSD(filter.abscissa(s), s.data.data) for s in series]
    epoch = lal.LIGOTimeGPS(row["gps"])
    common = dict(ra=row["ra"], dec=row["dec"], psi=row["psi"], inc=spins[0],
                  epoch=epoch, gmst=lal.GreenwichMeanSiderealTime(epoch), H=h)
    def make(distance, noise):
        return [simulate_snr(distance=distance, S=psd, response=detectors[ifo.name].response,
                location=detectors[ifo.name].location, measurement_error=noise, **common)
                for ifo, psd in zip(ifos, interpolated)]
    zero = make(row["luminosity_distance"], "zero-noise")
    ref = math.sqrt(sum(item[1]**2 for item in zero))
    distance = row["luminosity_distance"] * ref / row["target_network_snr"]
    np.random.seed(row["measurement_seed"])
    noisy = make(distance, "gaussian-noise")
    factor = np.exp(-1j * np.pi * row["morse"])
    adjusted = []
    for horizon, snr, phase, toa, snrs in noisy:
        snrs.data.data[:] *= factor
        adjusted.append((horizon, snr, float(np.angle(np.exp(1j*phase)*factor)), toa, snrs))
    single_tuple = namedtuple("ETSingleTuple", "detector snr phase time zerolag_time psd snr_series")
    class ETSingle(single_tuple, SingleEvent):
        pass
    event_tuple = namedtuple("ETEventTuple", "singles template_args")
    class ETEvent(event_tuple, Event):
        pass
    singles = [ETSingle(ifo.name, float(x[1]), float(x[2]), x[3], x[3], s, x[4])
               for ifo, x, s in zip(ifos, adjusted, series)]
    started = time.perf_counter()
    with detector_lookup(detectors):
        sky = localize(ETEvent(singles, args), waveform=waveform, f_low=20., enable_snr_series=True,
                       f_high_truncate=1., rescale_loglikelihood=.83)
    seconds = time.perf_counter() - started
    mass = np.asarray(sky["PROBDENSITY"]) * moc.uniq2pixarea(sky["UNIQ"])
    if not np.isfinite(mass).all() or np.any(mass < 0) or abs(mass.sum()-1) > 1e-5:
        raise RuntimeError("Invalid MOC probability mass")
    write_sky_map(output, sky, nest=True)
    raster = moc.rasterize(sky, order=9)
    prob = np.asarray(raster["PROBDENSITY"], dtype=float) * hp.nside2pixarea(512)
    prob /= prob.sum()
    truth = hp.ang2pix(512, np.pi/2-row["dec"], row["ra"] % (2*np.pi), nest=True)
    sorted_p = np.sort(prob)[::-1]
    cumulative = sorted_p.cumsum()
    credible = float(prob[prob >= prob[truth]].sum())
    record = {"event_uid": row["event_uid"], "source_uid": row["source_uid"],
              "seconds": seconds, "map_path": str(output), "map_bytes": output.stat().st_size,
              "map_sha256": sha(output), "ordering": "NESTED", "coordinate_frame": "ICRS/equatorial",
              "normalization": float(mass.sum()), "moc_cells": len(sky),
              "A90_deg2": float((np.searchsorted(cumulative, .9)+1)*hp.nside2pixarea(512, degrees=True)),
              "truth_HPD_level_raw": credible, "target_snr": row["target_network_snr"],
              "realized_snr": math.sqrt(sum(x[1]**2 for x in adjusted)),
              "effective_distance": distance, "iota_SI": float(spins[0]),
              "iota_wrong_solar_mass_units": float(bad_spins[0]),
              "unit_error_changes_iota_rad": float(abs(spins[0]-bad_spins[0])),
              "method": "conditional BAYESTAR; Gaussian trigger simulation, not exact archived-strain PE"}
    json_write(audit_file, record)
    return record


def pilot(root, workers):
    if not (root / "contracts/DATA_PREFLIGHT_PASS.json").exists():
        raise RuntimeError("Data audit required")
    rows = json.loads((root / "contracts/PILOT_EVENTS.json").read_text())
    done = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as pool:
        futures = [pool.submit(simulate_sky, r, str(root)) for r in rows]
        for f in futures:
            done.append(f.result())
            print(json.dumps({"stage": "sky_pilot", "done": len(done), "total": len(rows),
                              "event": done[-1]["event_uid"], "seconds": done[-1]["seconds"]}), flush=True)
    pd.DataFrame(done).to_csv(root / "tables/SKY_PILOT.csv", index=False)
    timings = np.array([r["seconds"] for r in done])
    json_write(root / "contracts/SKY_PILOT_COMPUTATIONAL_PASS.json", {
        "utc": utc(), "events": len(done), "map_validity": True,
        "seconds_P50_P90_max": np.quantile(timings, [.5, .9, 1]).tolist(),
        "pilot_HPD90_coverage_descriptive_only": float(np.mean([r["truth_HPD_level_raw"] <= .9 for r in done])),
        "not_a_statistical_coverage_gate": True, "not_a_resolution_convergence_gate": True,
        "full_50000_events_serial_hours_P50_P90_max": (50000*np.quantile(timings, [.5,.9,1])/3600).tolist(),
        "estimated_MOC_GiB": 50000*np.mean([r["map_bytes"] for r in done])/2**30,
        "full_sky_generation_authoritative_status": "not started; calibrated temperature and convergence audit still required"})
    verify_protected(root)


def materialize_one(item, root):
    family, image, _, path = item
    root = Path(root)
    key = f"{family}_{image}"
    destination = root / "cache/long16s" / (key + ".npy")
    marker = destination.with_suffix(".complete.json")
    if marker.exists():
        record = json.loads(marker.read_text())
        if sha(destination) != record["sha256"]:
            raise RuntimeError("Completed long input hash mismatch")
        return record
    storage_guard(root)
    source = np.load(path, mmap_mode="r")
    out = np.lib.format.open_memmap(destination, mode="r+" if destination.exists() else "w+",
                                  dtype=np.float32, shape=(len(source), 3, 4096))
    progress = destination.with_suffix(".progress.json")
    start = json.loads(progress.read_text())["complete_rows"] if progress.exists() else 0
    before = time.perf_counter()
    for offset in range(start, len(source), 16):
        end = min(len(source), offset+16)
        out[offset:end] = low_view(source[offset:end])
        if end % 256 == 0 or end == len(source):
            storage_guard(root)
            out.flush()
            json_write(progress, {"complete_rows": end, "total": len(source), "utc": utc()})
            print(json.dumps({"stage": "long16s", "array": key, "complete": end, "total": len(source)}), flush=True)
    out.flush()
    record = {"source": str(path), "path": str(destination), "sha256": sha(destination),
              "shape": list(out.shape), "seconds_this_resume": time.perf_counter()-before}
    json_write(marker, record)
    return record


def materialize(root, workers):
    if not (root / "contracts/DATA_PREFLIGHT_PASS.json").exists():
        raise RuntimeError("Data audit required")
    with ProcessPoolExecutor(max_workers=min(3, workers), mp_context=mp.get_context("spawn")) as pool:
        futures = [pool.submit(materialize_one, x, str(root)) for x in arrays()]
        result = [f.result() for f in futures]
    json_write(root / "contracts/LONG_INPUTS_COMPLETE.json", {"utc": utc(), "events": 50000, "arrays": result})
    verify_protected(root)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--stage", choices=("initialize", "audit", "pilot", "materialize", "prepare"), required=True)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    if args.stage == "initialize":
        initialize(args.root)
        return
    json_write(args.root / "STATUS.json", {"stage": args.stage, "state": "RUNNING", "utc": utc(), "pid": os.getpid()})
    try:
        if args.stage in ("audit", "prepare"):
            audit(args.root)
        if args.stage in ("pilot", "prepare"):
            pilot(args.root, args.workers)
        if args.stage in ("materialize", "prepare"):
            materialize(args.root, args.workers)
        json_write(args.root / "STATUS.json", {"stage": args.stage, "state": "COMPLETE", "utc": utc(),
                    "overall_experiment": "INPUTS_READY_AUXILIARY_TRAINING_PENDING", "adoption": STATUS})
    except Exception as exc:
        json_write(args.root / "STATUS.json", {"stage": args.stage, "state": "FAILED", "utc": utc(),
                    "error": str(exc), "traceback": traceback.format_exc(), "adoption": STATUS})
        raise


if __name__ == "__main__":
    main()
