#!/usr/bin/env python3
"""Targeted Nside=512/1024 audit for the event-specific BAYESTAR experiment.

This script never regenerates a sky map or changes a score used for tuning.
It reads the saved native BAYESTAR MOC maps and evaluates the overlap with a
sparse, float64 NESTED representation at Nside 512 and 1024.  The audit set is
fixed from pair labels and Nside=512 scores: all true companions, the 30
highest-sky non-companions, and 30 hash-selected non-companions per
deployment/seed/split.  Real-catalog Top-50 convergence is copied from the
pre-existing corrected-public-PE audit because the real maps and C-fixed
ranking are identical in the two experiments.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from scipy import stats


DEPLOYMENTS = ("gwtc3", "gwtc4")
SEEDS = (202607241, 202607242, 202607243)
SPLITS = ("validation", "test")
REFERENCE_REAL_AUDIT = Path(
    "/root/autodl-tmp/gw-catalog/results/"
    "gwtc_c_scheme_ordering_confirmation_20260831_20260831T094500Z/results"
)


def stable_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def uniq_order_ipix(uniq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Decode IVOA NUNIQ values without importing ligo.skymap."""
    uniq = np.asarray(uniq, dtype=np.int64)
    order = np.floor(np.log2(uniq)).astype(np.int64) // 2 - 1
    ipix = uniq - np.left_shift(np.int64(1), 2 * order + 2)
    return order, ipix


def target_segments(path: Path, order_out: int, temperature: float) -> tuple[np.ndarray, np.ndarray]:
    """Return contiguous NESTED interval starts and probability per output pixel."""
    with fits.open(path, memmap=False) as hdus:
        data = hdus[1].data
        uniq = np.asarray(data["UNIQ"], dtype=np.int64)
        density = np.asarray(data["PROBDENSITY"], dtype=np.float64)
    order, ipix = uniq_order_ipix(uniq)
    npix = 12 * 4**order_out
    pixel_area_out = 4.0 * math.pi / npix

    coarse = order <= order_out
    starts = []
    lengths = []
    masses = []
    if np.any(coarse):
        factors = np.left_shift(np.int64(1), 2 * (order_out - order[coarse]))
        starts.append(ipix[coarse] * factors)
        lengths.append(factors)
        masses.append(density[coarse] * pixel_area_out)

    fine = ~coarse
    if np.any(fine):
        shift = 2 * (order[fine] - order_out)
        parents = np.right_shift(ipix[fine], shift)
        source_area = 4.0 * math.pi / (12.0 * np.power(4.0, order[fine]))
        source_mass = density[fine] * source_area
        parent_ids, inverse = np.unique(parents, return_inverse=True)
        parent_mass = np.bincount(inverse, weights=source_mass)
        starts.append(parent_ids)
        lengths.append(np.ones(len(parent_ids), dtype=np.int64))
        masses.append(parent_mass)

    start = np.concatenate(starts).astype(np.int64, copy=False)
    length = np.concatenate(lengths).astype(np.int64, copy=False)
    mass = np.concatenate(masses).astype(np.float64, copy=False)
    sorter = np.argsort(start, kind="stable")
    start, length, mass = start[sorter], length[sorter], mass[sorter]
    if start[0] != 0 or start[-1] + length[-1] != npix:
        raise ValueError(f"MOC does not span the full target grid: {path}")
    if np.any(start[1:] != start[:-1] + length[:-1]):
        raise ValueError(f"MOC target intervals overlap or contain gaps: {path}")

    raw_norm = float(np.sum(mass * length, dtype=np.float64))
    if not np.isfinite(raw_norm) or raw_norm <= 0:
        raise ValueError(f"Invalid map normalization: {path}")
    mass /= raw_norm
    log_weight = np.log(np.maximum(mass, 1e-300)) / float(temperature)
    log_weight -= np.max(log_weight)
    weight = np.exp(log_weight)
    norm = float(np.sum(weight * length, dtype=np.float64))
    probability = weight / norm
    return start, probability


def sparse_log_bf(a: tuple[np.ndarray, np.ndarray], b: tuple[np.ndarray, np.ndarray], order: int) -> float:
    starts_a, prob_a = a
    starts_b, prob_b = b
    npix = 12 * 4**order
    boundaries = np.union1d(starts_a, starts_b)
    lengths = np.diff(np.append(boundaries, npix)).astype(np.float64)
    ia = np.searchsorted(starts_a, boundaries, side="right") - 1
    ib = np.searchsorted(starts_b, boundaries, side="right") - 1
    overlap = float(np.sum(lengths * prob_a[ia] * prob_b[ib], dtype=np.float64))
    return float(math.log(max(npix * overlap, 1e-300)))


def select_pairs(frame: pd.DataFrame, deployment: str, seed: int, split: str) -> pd.DataFrame:
    true_pairs = frame.loc[frame.is_true_pair.astype(bool)].copy()
    true_pairs["audit_population"] = "all_true_companions"
    false = frame.loc[~frame.is_true_pair.astype(bool)].copy()
    high = false.nlargest(min(30, len(false)), "sky_raw_log_bf").copy()
    high["audit_population"] = "top30_sky_noncompanion"
    excluded = set(zip(high.old_idx_i.astype(int), high.old_idx_j.astype(int)))
    pool = false.loc[
        ~pd.Series(
            list(zip(false.old_idx_i.astype(int), false.old_idx_j.astype(int))),
            index=false.index,
        ).isin(excluded)
    ].copy()
    rng = np.random.default_rng(stable_seed("nside1024-audit", deployment, seed, split))
    chosen = rng.choice(pool.index.to_numpy(), size=min(30, len(pool)), replace=False)
    random_false = pool.loc[chosen].copy()
    random_false["audit_population"] = "hash_selected_noncompanion"
    selected = pd.concat([true_pairs, high, random_false], ignore_index=True)
    return selected.drop_duplicates(["old_idx_i", "old_idx_j", "audit_population"])


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["deployment", "seed", "split", "audit_population"]
    for key, part in frame.groupby(keys, sort=True):
        delta = part.abs_delta_512_1024.to_numpy(float)
        row = dict(zip(keys, key))
        row.update(
            n_pairs=len(part),
            median_abs_delta_512_1024=float(np.median(delta)),
            q90_abs_delta_512_1024=float(np.quantile(delta, 0.90)),
            q99_abs_delta_512_1024=float(np.quantile(delta, 0.99)),
            max_abs_delta_512_1024=float(np.max(delta)),
            sign_flip_count_512_1024=int(part.sign_flip_512_1024.sum()),
            sign_flip_fraction_512_1024=float(part.sign_flip_512_1024.mean()),
            spearman_512_1024=(
                float(stats.spearmanr(part.z_sky_nside512_float64, part.z_sky_nside1024_float64).statistic)
                if len(part) > 1
                else 1.0
            ),
            max_abs_stored_float32_minus_float64_512=float(
                part.abs_delta_stored_float32_vs_float64_512.max()
            ),
        )
        rows.append(row)
    return pd.DataFrame(rows)


def run(root: Path) -> None:
    config = json.loads((root / "contracts/selected_config.json").read_text(encoding="utf-8"))
    rows = []
    for deployment in DEPLOYMENTS:
        for seed in SEEDS:
            temperature = float(config["deployments"][deployment][str(seed)]["posterior_temperature"])
            for split in SPLITS:
                pair_path = root / "results" / deployment / f"seed_{seed}" / f"{split}_pair_scores_bayestar_sky.parquet"
                frame = pd.read_parquet(pair_path)
                selected = select_pairs(frame, deployment, seed, split)
                indices = sorted(set(selected.old_idx_i.astype(int)) | set(selected.old_idx_j.astype(int)))
                maps_512 = {}
                maps_1024 = {}
                for idx in indices:
                    path = root / "event_maps" / deployment / f"seed_{seed}" / split / f"event_{idx:04d}.fits.gz"
                    maps_512[idx] = target_segments(path, 9, temperature)
                    maps_1024[idx] = target_segments(path, 10, temperature)
                for pair in selected.itertuples(index=False):
                    i, j = int(pair.old_idx_i), int(pair.old_idx_j)
                    z512 = sparse_log_bf(maps_512[i], maps_512[j], 9)
                    z1024 = sparse_log_bf(maps_1024[i], maps_1024[j], 10)
                    stored = float(pair.sky_raw_log_bf)
                    rows.append(
                        {
                            "deployment": deployment,
                            "seed": seed,
                            "split": split,
                            "audit_population": pair.audit_population,
                            "old_idx_i": i,
                            "old_idx_j": j,
                            "is_true_pair": bool(pair.is_true_pair),
                            "true_pair_family": pair.true_pair_family,
                            "posterior_temperature": temperature,
                            "z_sky_nside512_stored_float32": stored,
                            "z_sky_nside512_float64": z512,
                            "z_sky_nside1024_float64": z1024,
                            "abs_delta_stored_float32_vs_float64_512": abs(stored - z512),
                            "abs_delta_512_1024": abs(z512 - z1024),
                            "sign_flip_512_1024": bool(np.signbit(z512) != np.signbit(z1024)),
                        }
                    )
                print(f"[audit] {deployment}/{seed}/{split}: {len(selected)} pairs, {len(indices)} maps", flush=True)
    detail = pd.DataFrame(rows)
    out = root / "results"
    detail.to_csv(out / "bayestar_injection_targeted_nside1024_audit_pairs.csv", index=False, encoding="utf-8-sig")
    summary = summarize(detail)
    summary.to_csv(out / "bayestar_injection_targeted_nside1024_audit_summary.csv", index=False, encoding="utf-8-sig")

    real_detail_src = REFERENCE_REAL_AUDIT / "real_candidate_head_sky_resolution_audit.csv"
    real_summary_src = REFERENCE_REAL_AUDIT / "real_candidate_head_sky_resolution_summary.csv"
    shutil.copy2(real_detail_src, out / "real_candidate_head_sky_resolution_audit.csv")
    shutil.copy2(real_summary_src, out / "real_candidate_head_sky_resolution_summary.csv")
    provenance = {
        "injection_audit": "native BAYESTAR MOC; sparse float64 NESTED evaluation",
        "injection_selection": "all true companions + top-30 sky non-companions + 30 hash-selected non-companions per deployment/seed/split",
        "real_audit_source": str(REFERENCE_REAL_AUDIT),
        "real_audit_reuse_reason": "identical corrected public PE maps and identical C-fixed real ranking",
        "selection_used_locked_test_outcomes": False,
        "selection_used_real_PE_or_public_candidates": False,
    }
    (root / "contracts/TARGETED_NSIDE1024_AUDIT.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    run(args.root.resolve())


if __name__ == "__main__":
    main()
