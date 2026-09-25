#!/usr/bin/env python3
"""Event-PSD-matched physical waveform features and source-aware encoder.

Only the feature extractor's whitening PSD changes relative to PHASE-SOURCE.
No event mass posterior is used to build or choose waveform templates.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy import signal
from pycbc.waveform import get_fd_waveform
import torch
import torch.nn.functional as F

import mcwf_development_20260905 as dev
import mcwf_phasebank_20260905 as phase
import mcwf_mass_tf_20260905 as tf
import mcwf_encoder_body_20260906 as body

CONFIG = "PSD-PHASE-SOURCE"
FS = 2048
N = FS * 16


def freeze(root):
    path = root / "contracts/EVENT_PSD_MATCHED_ENCODER.json"
    if path.exists():
        return
    dev.json_write(path, {
        "config": CONFIG,
        "reason": "fixed-median-PSD templates do not match the event-specific whitening transfer function; directly whiten each template by the same local off-source PSD as the event",
        "comparison": "PHASE-SOURCE -> PSD-PHASE-SOURCE; same source loss, mass CE, input2s4096, training50epochs, same seeds and validation rules",
        "template_parameters": "same fixed576 IMRPhenomD template grid, no PE parameters",
        "template_frequency": "16s construction buffer, 40-580Hz physical band; only last2s observed strain used",
        "window": "same 2s quadrature-filter segment and lag grid as PHASE-SOURCE",
        "training_PSD": "exact a/b_noise_bank_index used for injection whitening, not population median",
        "real_PSD": "cached exact256s off-source reference PSD audited in prior exploration; no on-source PE inputs",
        "new_training_normalization": "mean and SD of event-PSD features in training sources only",
        "model_initialization": "same simulation-only PHASEBANK weights, then joint representation-body training",
        "bank_storage": "one unwhitened576-template spectrum cache; whitened dense bank temporary per PSD; features persisted once per event independent of model seed",
        "PSD_not_a_fourth_channel": True,
        "limitations": "approximate physical feature bank, not full matched-filter PE, no evidence or minimal-match guarantee",
        "frozen": ["Z_time", "Z_sky", "C-fixed weights", "scope", "source/noise split"],
        "references": ["https://pycbc.org/pycbc/latest/html/filter.html", "https://arxiv.org/abs/2004.11362"],
        "real_feedback_status": "development, not blinded confirmation",
    })


def spectra(root):
    out = root / "cache/adaptive_psd"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "unwhitened_aligned_template_spectra.npy"
    if path.exists():
        return np.load(path)
    bank = []
    for mc in np.exp(tf.LOG_CENTERS):
        for q in (.25, .5, 1.):
            for spin in (-.5, 0., .5):
                m1 = mc * (1 + q)**.2 / q**.6
                hp, _ = get_fd_waveform(approximant="IMRPhenomD", mass1=m1, mass2=m1*q,
                    spin1z=spin, spin2z=spin, delta_f=1/16, f_lower=30, f_final=580, distance=1000, inclination=0)
                hp.resize(N // 2 + 1)
                h = np.fft.irfft(np.asarray(hp), n=N)
                peak = np.argmax(np.abs(signal.hilbert(h)))
                h = np.roll(h, 14 * FS - peak)
                bank.append(np.fft.rfft(h).astype(np.complex64))
    result = np.stack(bank)
    np.save(path, result)
    dev.json_write(out / "SPECTRA_SOURCE.json", {"sha256": dev.sha(path), "n_templates": len(result),
                   "PE_parameters_used": False, "same_grid_as": str(body.PREVIOUS / "cache/phasebank")})
    return result


@torch.no_grad()
def whitened_bank(h, f, psd):
    frequency = np.fft.rfftfreq(N, 1 / FS)
    p = np.array([np.interp(frequency, f, a) for a in psd])
    p = np.maximum(p, p.max(axis=1, keepdims=True) * 1e-20)
    if not np.isfinite(p).all() or (p <= 0).any():
        raise ValueError("Invalid local PSD")
    sos = signal.butter(6, [40, 580], fs=FS, btype="bandpass", output="sos")
    _, response = signal.sosfreqz(sos, worN=frequency, fs=FS)
    response = np.abs(response)**2
    spectral = torch.as_tensor(h, device="cuda")[:, None, :] * torch.as_tensor(response / np.sqrt(p), dtype=torch.float32, device="cuda")[None]
    white = torch.fft.irfft(spectral, n=N, dim=-1)
    # Hilbert analytic signal: positive frequencies doubled, DC/Nyquist kept.
    analytic_filter = torch.zeros(N, device="cuda")
    analytic_filter[0] = analytic_filter[N//2] = 1
    analytic_filter[1:N//2] = 2
    analytic = torch.fft.ifft(torch.fft.fft(white, dim=-1) * analytic_filter, dim=-1)
    offset = 14 * FS - int(1.75 * FS)
    window = torch.as_tensor(signal.windows.tukey(4096, .05), dtype=torch.float32, device="cuda")
    u = analytic.real[..., offset:offset+4096] * window
    v = analytic.imag[..., offset:offset+4096] * window
    u -= u.mean(-1, keepdim=True)
    v -= v.mean(-1, keepdim=True)
    u = F.normalize(u, dim=-1)
    v -= (v * u).sum(-1, keepdim=True) * u
    v = F.normalize(v, dim=-1)
    return torch.stack([u, v], dim=2).cpu().numpy()


def extract_grouped(root, strain, frequency, psds, indices, out):
    if out.exists():
        result = np.load(out)
        if len(result) != len(strain):
            raise RuntimeError("Feature row count mismatch")
        return result
    out.parent.mkdir(parents=True, exist_ok=True)
    h = spectra(root)
    result = np.empty((len(strain), 3, 64, 3, 3), np.float32)
    audit = []
    for bank_id in np.unique(indices):
        keep = np.flatnonzero(indices == bank_id)
        start = time.perf_counter()
        bank = whitened_bank(h, frequency, psds[int(bank_id)])
        gram = max(float(np.max(np.abs((bank[:, :, 0] * bank[:, :, 1]).sum(-1)))),
                   float(np.max(np.abs((bank * bank).sum(-1) - 1))))
        if gram > 1e-4 or not np.isfinite(bank).all():
            raise RuntimeError("Quadrature normalization failed")
        result[keep] = phase.features(strain[keep], bank)
        audit.append({"bank_id": int(bank_id), "events": len(keep), "gram_error": gram,
                      "seconds": time.perf_counter() - start})
        del bank
    if not np.isfinite(result).all():
        raise RuntimeError("Nonfinite adaptive features")
    np.save(out, result)
    dev.csv_write(out.with_suffix(".audit.csv"), pd.DataFrame(audit))
    dev.json_write(out.with_suffix(".sha256.json"), {"sha256": dev.sha(out), "n_events": len(strain), "PSD_source_count": len(audit)})
    return result


def development(root, dep):
    freeze(root)
    cache = body.prepare(root, dep)
    noise = dev.OLD / f"data/noise_banks/{dep}"
    for split in ("train", "validation"):
        meta = pd.read_parquet(cache / f"{split}_metadata.parquet")
        indices = np.where(meta.image.eq("a"), meta.a_noise_bank_index, meta.b_noise_bank_index).astype(int)
        psd = np.load(noise / f"noise_psd_{split}.npy", mmap_mode="r")
        freq = np.load(noise / f"noise_psd_frequency_{split}.npy")
        raw = np.load(cache / f"{split}_raw2s.npy", mmap_mode="r")
        out = root / f"cache/adaptive_psd/{dep}/{split}_features.npy"
        extract_grouped(root, raw, freq, psd, indices, out)
        print(json.dumps({"adaptive_features_done": dep, "split": split, "rows": len(meta)}), flush=True)


def encode(root, checkpoint, full, dep, eval_seed, split):
    payload = torch.load(checkpoint, weights_only=False, map_location="cpu")
    if split == "real":
        packed = np.load(body.PREVIOUS / f"cache/phasepsd/{dep}/real_reference_PSDs.npz")
        _, events = dev.real_inputs(dep)
        valid = events.strict_h1l1_preprocessing_pass.to_numpy(bool)
        ids = np.flatnonzero(valid)
        mapping = {int(k): n for n, k in enumerate(packed["idx"])}
        indices = np.array([mapping[int(i)] for i in ids])
        freq, psd = packed["frequency"], packed["psd"]
        out = root / f"cache/adaptive_psd/{dep}/real_features.npy"
    else:
        plan = dev.BASE.retained_event_plan(dep, eval_seed, split)
        shared = dev.ORCH.SOURCE_ROOT / dep / "shared"
        freq = np.load(shared / "noise_psd_frequency.npy")
        psd = np.load(shared / "noise_psd_bank.npy", mmap_mode="r")
        indices = plan.parent_noise_bank.to_numpy(int)
        out = root / f"cache/adaptive_psd/{dep}/eval_{eval_seed}_{split}_features.npy"
    raw = dev.TRAIN.make_window_view(np.asarray(full[..., -4096:], np.float32), 2)
    x = extract_grouped(root, raw, freq, psd, indices, out)
    model = body.Encoder(CONFIG).cuda().eval()
    model.load_state_dict(payload["model"])
    logits, z = body.infer(model, (x - payload["mu"]) / payload["sd"], raw)
    lp = logits.astype(float) / payload["temperature"]
    lp -= np.logaddexp.reduce(lp, -1, keepdims=True)
    return np.exp(lp), z, payload


def tests(root, dep):
    h = spectra(root)
    noise = dev.OLD / f"data/noise_banks/{dep}"
    f = np.load(noise / "noise_psd_frequency_train.npy")
    p = np.median(np.load(noise / "noise_psd_train.npy"), axis=0)
    new = whitened_bank(h, f, p)
    old = np.load(body.PREVIOUS / f"cache/phasebank/{dep}/quadrature_bank.npy")
    difference = float(np.max(np.abs(new - old)))
    if difference > 1e-4:
        raise AssertionError(f"Matched-PSD implementation differs at same PSD: {difference}")
    dev.json_write(root / f"audit/{dep}_ADAPTIVE_BANK_NUMERICAL_TEST.json", {"max_abs_difference_at_same_median_PSD": difference,
                   "passed": True, "test": "new GPU whitening/Hilbert/orthogonalization against original CPU fixed-PSD implementation"})
    print(json.dumps({"bank_test": dep, "max_abs_difference": difference}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--deployment", choices=("gwtc3", "gwtc4"), required=True)
    parser.add_argument("--action", choices=("prepare", "test"), required=True)
    args = parser.parse_args()
    freeze(args.root)
    if args.action == "test":
        tests(args.root, args.deployment)
    else:
        tests(args.root, args.deployment)
        development(args.root, args.deployment)
