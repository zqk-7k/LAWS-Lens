#!/usr/bin/env python3
"""PE-conditioned matched-filter audit of existing strain, never ranking input."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import signal
from pycbc.types import TimeSeries, FrequencySeries
from pycbc.filter import matched_filter
from pycbc.waveform import get_fd_waveform
import mcwf_development_20260905 as dev

p = argparse.ArgumentParser()
p.add_argument("--root", type=Path, required=True)
a = p.parse_args()
root = a.root
v3 = dev.BASE.v7.v3
dev.json_write(root / "contracts/PE_CONDITIONED_SIGNAL_FORENSICS.json", {
    "purpose": "check event timing/strain informativeness and contrast physically implausible head predictions",
    "inputs": "public PE medians used ONLY to create audit templates, never scorer or training inputs",
    "estimator": "PyCBC matched_filter, IMRPhenomD aligned-spin quadrupole approximation",
    "limitations": "not full PE, not lensing evidence; imperfect precession/higher-mode approximation",
    "frequency": [40,580], "raw_duration_s": 32,
    "comparison": "full 32 s vs masking data outside [GPS-1.75,GPS+0.25] with template normalization unchanged; the latter is an SNR-loss diagnostic, not a two-second Bayes factor",
    "scoring_or_training_allowed": False,
})
event_pe = pd.read_csv(root / "audit/gwtc3_event_PE_reference.csv").set_index("event_name")
events = pd.read_csv(dev.MAIN / "cache/source_run/data/event_manifest.csv").set_index("event_name")
strain = pd.read_csv(dev.MAIN / "cache/source_run/data/strain_gwosc_download_manifest.csv")
cache = v3.HdfCache(dev.MAIN / "cache/source_run", max_files=4)
pred = pd.read_csv(root / "audit/O3_BASELINE_EVENT_MASS_INPUT_AUDIT.csv")
rows = []
names = ["GW191105_143521", "GW191126_115259", "GW191129_134029", "GW200316_215756", "GW200322_091133"]
for name in names:
    gps = float(events.loc[name, "gps_time"])
    pe = event_pe.loc[name]
    mc = float(pe.pe_chirp_mass_median)
    q = float(pe.pe_mass_ratio_median)
    spin = float(np.clip(pe.pe_chi_eff_median, -.9, .9))
    masses = [("PE_reference", mc)] + [(f"old_head_{int(r.model_seed)}", float(r.waveform_pred_chirp_mass_detector)) for r in pred[pred.event_name.eq(name)].itertuples()]
    for detector in ("H1", "L1"):
        row = strain[(strain.event_name.eq(name)) & (strain.detector.eq(detector))].iloc[0]
        data, start, duration = cache.get(str(row.local_path))
        x = v3._extract_channel_window(data, start, gps, 32*4096, -16)
        ref, ref_start = v3._extract_psd_reference(data, start, gps)
        if x is None or ref is None: raise RuntimeError(f"Missing finite data for {name} {detector}")
        f, psd = v3.estimate_psd(np.stack([ref, ref]))
        freqs = np.fft.rfftfreq(len(x), 1/4096)
        psdvec = FrequencySeries(np.interp(freqs, f, psd[0]), delta_f=1/32)
        ts = np.arange(len(x))/4096 - 16
        y = (x.astype(np.float64) - np.mean(x))*signal.windows.tukey(len(x), .1)
        for label, mass in masses:
            m1 = mass*(1+q)**.2/q**.6; m2 = q*m1
            h, _ = get_fd_waveform(approximant="IMRPhenomD", mass1=m1, mass2=m2,
                                   spin1z=spin, spin2z=spin, delta_f=1/32, f_lower=40, f_final=580, distance=1000)
            h.resize(len(freqs))
            for window in ("full32", "masked_peak2s"):
                d = y if window == "full32" else y*((ts>=-1.75)&(ts<=.25))
                snr = matched_filter(h, TimeSeries(d, delta_t=1/4096), psd=psdvec,
                                     low_frequency_cutoff=40, high_frequency_cutoff=580)
                loc = np.flatnonzero(np.abs(ts)<=.2)
                k = loc[np.argmax(np.abs(snr.numpy()[loc]))]
                rows.append({"event_name": name, "detector": detector, "template_role": label,
                             "template_mc": mass, "PE_reference_mc": mc, "q_audit": q, "chi_eff_audit": spin,
                             "window": window, "matched_filter_abs_snr": float(abs(snr[k])),
                             "peak_offset_from_catalog_gps_s": float(ts[k]), "psd_start_gps": ref_start,
                             "physical_approximation": "PE-conditioned IMRPhenomD; audit only"})
    print(name, flush=True)
dev.csv_write(root / "audit/PE_CONDITIONED_SIGNAL_MATCHED_FILTER_DIAGNOSTIC.csv", pd.DataFrame(rows))
