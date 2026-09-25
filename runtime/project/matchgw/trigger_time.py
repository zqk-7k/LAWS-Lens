from __future__ import annotations

from pathlib import Path
import hashlib

import numpy as np
import pandas as pd


TRIGGER_SEED = 20260526


def timing_sigma_from_snr(snr, min_sigma: float = 0.01) -> np.ndarray:
    """Estimate observable trigger-time uncertainty from SNR, in seconds."""
    snr = np.asarray(snr, dtype=float)
    return np.maximum(min_sigma, 1.0 / np.maximum(snr, 1.0))


def _stable_seed_offset(*parts: object) -> int:
    text = '|'.join(map(str, parts))
    return int(hashlib.md5(text.encode('utf-8')).hexdigest()[:8], 16) % 100000


def _load_first_existing(paths: list[Path]) -> np.ndarray:
    for path in paths:
        if path.exists():
            return np.load(path)
    names = ', '.join(str(p) for p in paths)
    raise FileNotFoundError(f'No SNR file found. Tried: {names}')


def _lensed_snr_paths(data_dir: Path, family: str, detector: str, image: int) -> list[Path]:
    fam = family.upper()
    if detector.upper() == 'LIGO':
        return [
            data_dir / f'{fam}_optimal_SNR_network_{image}.npy',
            data_dir / f'{fam}_optimal_SNR_single_{image}.npy',
            data_dir / f'{fam}_optimal_SNR_{image}.npy',
        ]
    return [
        data_dir / f'{fam}_optimal_SNR_{image}.npy',
        data_dir / f'{fam}_optimal_SNR_network_{image}.npy',
    ]


def _unlensed_snr_paths(data_dir: Path, detector: str) -> list[Path]:
    if detector.upper() == 'LIGO':
        return [
            data_dir / 'unlensed_optimal_SNR_network.npy',
            data_dir / 'unlensed_optimal_SNR_single.npy',
            data_dir / 'unlensed_optimal_SNR.npy',
        ]
    return [
        data_dir / 'unlensed_optimal_SNR.npy',
        data_dir / 'unlensed_optimal_SNR_network.npy',
    ]


def ensure_lensed_trigger_time_features(
    data_root: Path,
    family: str,
    detector: str,
    seed: int = TRIGGER_SEED,
    min_sigma: float = 0.01,
) -> pd.DataFrame:
    """Create/read observed trigger-time features for SIS/PM lensed image pairs.

    True geocent_time and lens t_d are kept for diagnostics only. Downstream
    pair features should use trigger_time_obs and delta_time_obs instead.
    """
    fam = family.upper()
    data_dir = Path(data_root) / f'{fam}_data_0222'
    out_path = data_dir / f'{fam}_trigger_time_features.csv'
    if out_path.exists():
        return pd.read_csv(out_path)

    lensed = pd.read_csv(data_dir / 'lensed_source_samples.csv')
    lens = pd.read_csv(data_dir / 'lens.csv')
    n = len(lens)
    img1 = lensed.iloc[:n].reset_index(drop=True)
    img2 = lensed.iloc[n:2 * n].reset_index(drop=True)
    snr1 = _load_first_existing(_lensed_snr_paths(data_dir, fam, detector, 1))
    snr2 = _load_first_existing(_lensed_snr_paths(data_dir, fam, detector, 2))

    t1_true = img1['geocent_time'].to_numpy(dtype=float)
    t2_true = img2['geocent_time'].to_numpy(dtype=float)
    sigma_t1 = timing_sigma_from_snr(snr1, min_sigma=min_sigma)
    sigma_t2 = timing_sigma_from_snr(snr2, min_sigma=min_sigma)
    rng = np.random.default_rng(seed + _stable_seed_offset(data_root, fam, detector, 'lensed'))
    trigger_time_obs_1 = t1_true + rng.normal(0.0, sigma_t1)
    trigger_time_obs_2 = t2_true + rng.normal(0.0, sigma_t2)
    delta_time_obs = np.abs(trigger_time_obs_2 - trigger_time_obs_1)

    out = pd.DataFrame({
        'pair_id': np.arange(n),
        'geocent_time_true_1': t1_true,
        'geocent_time_true_2': t2_true,
        'delta_time_true': np.abs(t2_true - t1_true),
        'lens_t_d': lens['t_d'].to_numpy(dtype=float) if 't_d' in lens.columns else np.nan,
        'trigger_time_obs_1': trigger_time_obs_1,
        'trigger_time_obs_2': trigger_time_obs_2,
        'trigger_time_sigma_1': sigma_t1,
        'trigger_time_sigma_2': sigma_t2,
        'delta_time_obs': delta_time_obs,
        'sigma_delta_time': np.sqrt(sigma_t1 ** 2 + sigma_t2 ** 2),
        'log10_delta_time_obs': np.log10(delta_time_obs + 1.0),
        'snr_1': snr1,
        'snr_2': snr2,
    })
    out.to_csv(out_path, index=False)
    return out


def ensure_unlensed_trigger_time_features(
    data_root: Path,
    detector: str,
    seed: int = TRIGGER_SEED,
    min_sigma: float = 0.01,
) -> pd.DataFrame:
    """Create/read observed trigger-time features for unlensed single events."""
    data_dir = Path(data_root) / 'Unlensed_data_0222'
    out_path = data_dir / 'unlensed_trigger_time_features.csv'
    if out_path.exists():
        return pd.read_csv(out_path)

    source = pd.read_csv(data_dir / 'source_samples.csv')
    snr = _load_first_existing(_unlensed_snr_paths(data_dir, detector))
    t_true = source['geocent_time'].to_numpy(dtype=float)
    sigma_t = timing_sigma_from_snr(snr, min_sigma=min_sigma)
    rng = np.random.default_rng(seed + _stable_seed_offset(data_root, detector, 'unlensed'))
    trigger_time_obs = t_true + rng.normal(0.0, sigma_t)
    out = pd.DataFrame({
        'event_id': np.arange(len(source)),
        'geocent_time_true': t_true,
        'trigger_time_obs': trigger_time_obs,
        'trigger_time_sigma': sigma_t,
        'snr': snr,
    })
    out.to_csv(out_path, index=False)
    return out


def catalog_trigger_time_frame(
    data_root: Path,
    family: str,
    lensed_idx: np.ndarray,
    unlensed_idx: np.ndarray,
    detector: str,
    seed: int = TRIGGER_SEED,
) -> pd.DataFrame:
    """Return event-level trigger_time_obs aligned with catalog_observable_frame.

    Output order is [lensed image1, lensed image2, unlensed], matching
    scripts.experiments.21_observable_aux_reranker.catalog_observable_frame.
    """
    trig = ensure_lensed_trigger_time_features(data_root, family, detector, seed=seed)
    un = ensure_unlensed_trigger_time_features(data_root, detector, seed=seed)
    lensed_idx = np.asarray(lensed_idx, dtype=int)
    unlensed_idx = np.asarray(unlensed_idx, dtype=int)

    l1 = pd.DataFrame({
        'event_kind': 'lensed_image1',
        'source_row': lensed_idx,
        'pair_id': trig.loc[lensed_idx, 'pair_id'].to_numpy(),
        'geocent_time_true': trig.loc[lensed_idx, 'geocent_time_true_1'].to_numpy(),
        'trigger_time_obs': trig.loc[lensed_idx, 'trigger_time_obs_1'].to_numpy(),
        'trigger_time_sigma': trig.loc[lensed_idx, 'trigger_time_sigma_1'].to_numpy(),
        'snr': trig.loc[lensed_idx, 'snr_1'].to_numpy(),
    })
    l2 = pd.DataFrame({
        'event_kind': 'lensed_image2',
        'source_row': lensed_idx,
        'pair_id': trig.loc[lensed_idx, 'pair_id'].to_numpy(),
        'geocent_time_true': trig.loc[lensed_idx, 'geocent_time_true_2'].to_numpy(),
        'trigger_time_obs': trig.loc[lensed_idx, 'trigger_time_obs_2'].to_numpy(),
        'trigger_time_sigma': trig.loc[lensed_idx, 'trigger_time_sigma_2'].to_numpy(),
        'snr': trig.loc[lensed_idx, 'snr_2'].to_numpy(),
    })
    u = pd.DataFrame({
        'event_kind': 'unlensed',
        'source_row': unlensed_idx,
        'pair_id': -1,
        'geocent_time_true': un.loc[unlensed_idx, 'geocent_time_true'].to_numpy(),
        'trigger_time_obs': un.loc[unlensed_idx, 'trigger_time_obs'].to_numpy(),
        'trigger_time_sigma': un.loc[unlensed_idx, 'trigger_time_sigma'].to_numpy(),
        'snr': un.loc[unlensed_idx, 'snr'].to_numpy(),
    })
    return pd.concat([l1, l2, u], ignore_index=True)


def log1p_delta_time_obs(time_obs: pd.DataFrame, anchors: np.ndarray, cands: np.ndarray) -> np.ndarray:
    """Pair-level observable time-delay feature from trigger_time_obs."""
    t = time_obs['trigger_time_obs'].to_numpy(dtype=float)
    return np.log1p(np.abs(t[anchors] - t[cands])).astype(np.float32)
