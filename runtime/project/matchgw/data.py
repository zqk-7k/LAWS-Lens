from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import MatchRunConfig


@dataclass(slots=True)
class MatchArrays:
    # l1/l2 是同一个强透镜源的两张像；unlensed 是孤立事件。
    # 训练时 l1-l2 组成正样本，unlensed 自身做 self-pair，增强模型稳定性。
    # *_pure 只在 noisy 辅助训练时加载，用 clean waveform 约束模型学习源本征形态。
    l1: np.ndarray
    l2: np.ndarray
    unlensed: np.ndarray
    l1_pure: np.ndarray | None = None
    l2_pure: np.ndarray | None = None
    unlensed_pure: np.ndarray | None = None


def _load_npy_matrix(path: Path, limit: int | None = None) -> np.ndarray:
    # 使用 mmap 读取大 npy，避免一次性把 74G 数据全部加载进内存。
    if not path.exists():
        raise FileNotFoundError(path)
    x = np.load(path, allow_pickle=True, mmap_mode="r")
    if limit is not None:
        x = x[:limit]
    return np.asarray(x, dtype=np.float32)


def load_match_arrays(cfg: MatchRunConfig) -> MatchArrays:
    arrays = MatchArrays(
        l1=_load_npy_matrix(cfg.l1_path, cfg.lensed_limit),
        l2=_load_npy_matrix(cfg.l2_path, cfg.lensed_limit),
        unlensed=_load_npy_matrix(cfg.unlensed_path, cfg.unlensed_limit),
    )
    if cfg.use_pure_aux and cfg.mode == "noisy":
        arrays.l1_pure = _load_npy_matrix(cfg.source_dir / f"{cfg.family}_h_strain_1.npy", cfg.lensed_limit)
        arrays.l2_pure = _load_npy_matrix(cfg.source_dir / f"{cfg.family}_h_strain_2.npy", cfg.lensed_limit)
        arrays.unlensed_pure = _load_npy_matrix(cfg.data_root / "Unlensed_data_0222" / "unlensed_h_strain.npy", cfg.unlensed_limit)
    return arrays


def split_indices(n_lensed: int, n_unlensed: int, cfg: MatchRunConfig) -> dict[str, dict[str, np.ndarray]]:
    # 按源索引划分 train/val/test，确保同一个 lensed pair 不会跨集合泄漏。
    rng = np.random.default_rng(cfg.seed)
    l_perm = rng.permutation(n_lensed)
    u_perm = rng.permutation(n_unlensed)

    def split(perm: np.ndarray) -> dict[str, np.ndarray]:
        n_train = int(round(len(perm) * cfg.train_frac))
        n_val = int(round(len(perm) * cfg.val_frac))
        return {
            "train": perm[:n_train],
            "val": perm[n_train:n_train + n_val],
            "test": perm[n_train + n_val:],
        }

    return {"lensed": split(l_perm), "unlensed": split(u_perm)}


def pad_or_trim(x: np.ndarray, target_len: int, stride: int = 1) -> np.ndarray:
    # 模型不直接吃完整 98304 点波形，而是取尾部窗口并下采样，
    # 这样能显著加快训练，同时保留 merger 附近的主要形态信息。
    n = x.shape[-1]
    if n >= target_len:
        y = x[..., -target_len:]
    else:
        shape = list(x.shape)
        shape[-1] = target_len
        y = np.zeros(tuple(shape), dtype=x.dtype)
        y[..., -n:] = x
    if stride > 1:
        y = y[..., ::stride]
    return y.astype(np.float32, copy=False)


def zscore(x: np.ndarray) -> np.ndarray:
    # Use a scale-aware floor rather than a fixed 1e-8. Public GWOSC strain is
    # O(1e-19); a fixed 1e-8 denominator suppresses real-event inputs to nearly
    # zero and collapses embeddings. For O(1) training arrays this is unchanged.
    std = float(x.std())
    floor = max(abs(float(x.mean())) * 1e-6, 1e-30)
    return ((x - x.mean()) / max(std, floor)).astype(np.float32, copy=False)


def peak_flip(x: np.ndarray) -> np.ndarray:
    # 统一主峰符号，减少整体相位翻转给 embedding 带来的不必要差异。
    return -x if x[np.argmax(np.abs(x))] < 0 else x


def _fft_band(x: np.ndarray, lo: int, hi: int) -> np.ndarray:
    sp = np.fft.rfft(x.astype(np.float32, copy=False))
    lo = max(0, int(lo))
    hi = min(sp.shape[-1] - 1, int(hi))
    mask = np.zeros(sp.shape[-1], dtype=bool)
    if hi >= lo:
        mask[lo:hi + 1] = True
    return np.fft.irfft(np.where(mask, sp, 0.0), n=x.shape[-1]).astype(np.float32, copy=False)


def multiband_preprocess(x: np.ndarray, cfg: MatchRunConfig) -> np.ndarray:
    # 多频带 waveform-only 输入：让模型自己学习不同频段的可靠性。
    # 这里仍只使用 waveform 本身，不使用任何物理辅助参数。
    bands = [(40, 160), (160, 320), (320, 580), (cfg.bandpass_low, cfg.bandpass_high)]
    y = np.stack([_fft_band(x, lo, hi) for lo, hi in bands], axis=0).astype(np.float32, copy=False)
    if y.ndim == 3:
        # 输入为 detector channels 时，输出通道按 band x detector 展平给 Conv1d。
        y = y.reshape(y.shape[0] * y.shape[1], y.shape[2])
    return y


def zscore_channels(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        return zscore(x)
    mean = x.mean(axis=-1, keepdims=True)
    std = x.std(axis=-1, keepdims=True)
    floor = np.maximum(np.abs(mean) * 1e-6, 1e-30)
    return ((x - mean) / np.maximum(std, floor)).astype(np.float32, copy=False)


def peak_flip_channels(x: np.ndarray) -> np.ndarray:
    ref = x if x.ndim == 1 else np.sum(x, axis=0)
    return -x if ref[np.argmax(np.abs(ref))] < 0 else x


def spectral_preprocess(x: np.ndarray, cfg: MatchRunConfig) -> np.ndarray:
    # 只基于 waveform 自身做频域处理，不读取任何辅助参数。
    # bandpass 去掉明显低/高频噪声；whiten 压平样本自己的平滑谱包络，降低 ET 噪声形态影响。
    mode = str(getattr(cfg, "preprocess", "none")).lower()
    if mode in {"none", ""}:
        return x.astype(np.float32, copy=False)
    sp = np.fft.rfft(x.astype(np.float32, copy=False))
    if "bandpass" in mode:
        lo = max(0, int(getattr(cfg, "bandpass_low", 0)))
        hi = min(sp.shape[-1] - 1, int(getattr(cfg, "bandpass_high", sp.shape[-1] - 1)))
        mask = np.zeros(sp.shape[-1], dtype=bool)
        if hi >= lo:
            mask[lo:hi + 1] = True
        sp = np.where(mask, sp, 0.0)
    if "whiten" in mode:
        amp = np.abs(sp).astype(np.float32)
        k = max(3, int(getattr(cfg, "whiten_kernel", 33)))
        if k % 2 == 0:
            k += 1
        kernel = np.ones(k, dtype=np.float32) / float(k)
        if amp.ndim == 1:
            smooth = np.convolve(amp, kernel, mode="same")
        else:
            smooth = np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="same"), -1, amp)
        floor = np.percentile(smooth, 10) + 1e-6
        sp = sp / np.maximum(smooth, floor)
    return np.fft.irfft(sp, n=x.shape[-1]).astype(np.float32, copy=False)


def augment_channels(x: np.ndarray, cfg: MatchRunConfig, rng: np.random.Generator) -> np.ndarray:
    # 训练期做轻量增强：平移、幅度扰动、少量噪声，提升 noisy/pure 泛化能力。
    y = x.copy()
    if cfg.aug_flip:
        y = peak_flip_channels(y)
    if cfg.aug_roll > 0:
        y = np.roll(y, int(rng.integers(-cfg.aug_roll, cfg.aug_roll + 1)), axis=-1)
    if cfg.aug_scale > 0:
        y = y * float(1.0 + rng.uniform(-cfg.aug_scale, cfg.aug_scale))
    if cfg.aug_noise > 0:
        y = y + rng.normal(0.0, cfg.aug_noise * (y.std(axis=-1, keepdims=True) + 1e-8), size=y.shape)
    return zscore_channels(y)


def to_channels(x: np.ndarray, use_hilbert: bool = False) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if use_hilbert:
        from scipy.signal import hilbert
        z = hilbert(x, axis=-1)
        if x.ndim == 1:
            return np.stack([z.real, z.imag], axis=0).astype(np.float32)
        return np.concatenate([z.real, z.imag], axis=0).astype(np.float32)
    if x.ndim == 1:
        return x[None, :].astype(np.float32)
    if x.ndim == 2:
        return x.astype(np.float32, copy=False)
    raise ValueError(f"expected waveform shape [time] or [channels,time], got {x.shape}")


def prepared_channel_count(sample: np.ndarray, cfg: MatchRunConfig) -> int:
    x = pad_or_trim(sample, cfg.target_len, cfg.stride)
    if str(getattr(cfg, "preprocess", "none")).lower() == "multiband":
        return int(multiband_preprocess(x, cfg).shape[0])
    return int(to_channels(x, cfg.use_hilbert).shape[0])


class PairDataset(Dataset):
    # 训练集返回两路视图：lensed 返回 L1/L2，同源事件应被拉近；
    # unlensed 返回同一个事件的两次增强视图，用来稳定孤立事件 embedding。
    def __init__(self, arrays: MatchArrays, lensed_idx: np.ndarray, unlensed_idx: np.ndarray, cfg: MatchRunConfig) -> None:
        self.arrays = arrays
        self.cfg = cfg
        self.items = [("L", int(i)) for i in lensed_idx] + [("U", int(i)) for i in unlensed_idx]
        if cfg.use_pure_aux and cfg.mode == "noisy":
            # 额外正样本不改变测试集，只在训练中把 noisy embedding 拉向同事件 clean embedding。
            self.items.extend(("L1P", int(i)) for i in lensed_idx)
            self.items.extend(("L2P", int(i)) for i in lensed_idx)
            self.items.extend(("LP", int(i)) for i in lensed_idx)
            self.items.extend(("UP", int(i)) for i in unlensed_idx)
        self.rng = np.random.default_rng(cfg.seed)

    def __len__(self) -> int:
        return len(self.items)

    def _prepare(self, x: np.ndarray, train: bool) -> np.ndarray:
        y = pad_or_trim(x, self.cfg.target_len, self.cfg.stride)
        if str(getattr(self.cfg, "preprocess", "none")).lower() == "multiband":
            mb = multiband_preprocess(y, self.cfg)
            if self.cfg.aug_flip:
                mb = peak_flip_channels(mb)
            if train and self.cfg.aug_roll > 0:
                mb = np.roll(mb, int(self.rng.integers(-self.cfg.aug_roll, self.cfg.aug_roll + 1)), axis=-1)
            if train and self.cfg.aug_scale > 0:
                mb = mb * float(1.0 + self.rng.uniform(-self.cfg.aug_scale, self.cfg.aug_scale))
            if train and self.cfg.aug_noise > 0:
                mb = mb + self.rng.normal(0.0, self.cfg.aug_noise * (mb.std(axis=-1, keepdims=True) + 1e-8), size=mb.shape)
            return zscore_channels(mb).astype(np.float32, copy=False)
        y = spectral_preprocess(y, self.cfg)
        y = augment_channels(y, self.cfg, self.rng) if train else zscore_channels(peak_flip_channels(y) if self.cfg.aug_flip else y)
        return to_channels(y, self.cfg.use_hilbert)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        kind, src_idx = self.items[idx]
        if kind == "L":
            a = self._prepare(self.arrays.l1[src_idx], train=True)
            b = self._prepare(self.arrays.l2[src_idx], train=True)
        elif kind == "L1P" and self.arrays.l1_pure is not None:
            a = self._prepare(self.arrays.l1[src_idx], train=True)
            b = self._prepare(self.arrays.l1_pure[src_idx], train=True)
        elif kind == "L2P" and self.arrays.l2_pure is not None:
            a = self._prepare(self.arrays.l2[src_idx], train=True)
            b = self._prepare(self.arrays.l2_pure[src_idx], train=True)
        elif kind == "LP" and self.arrays.l1_pure is not None and self.arrays.l2_pure is not None:
            a = self._prepare(self.arrays.l1_pure[src_idx], train=True)
            b = self._prepare(self.arrays.l2_pure[src_idx], train=True)
        elif kind == "UP" and self.arrays.unlensed_pure is not None:
            a = self._prepare(self.arrays.unlensed[src_idx], train=True)
            b = self._prepare(self.arrays.unlensed_pure[src_idx], train=True)
        else:
            a = self._prepare(self.arrays.unlensed[src_idx], train=True)
            b = self._prepare(self.arrays.unlensed[src_idx], train=True)
        return torch.from_numpy(a), torch.from_numpy(b)


class EvaluationSet(Dataset):
    # 评估集把 L1、L2、unlensed 拼成一个 catalog，模拟真实目录检索。
    # meta 记录每个事件的真实 pair_id，用于计算 Recall@K 和 F1。
    def __init__(self, arrays: MatchArrays, lensed_idx: np.ndarray, unlensed_idx: np.ndarray, cfg: MatchRunConfig) -> None:
        self.cfg = cfg
        self.waveforms = [arrays.l1[int(i)] for i in lensed_idx]
        self.waveforms.extend(arrays.l2[int(i)] for i in lensed_idx)
        self.waveforms.extend(arrays.unlensed[int(i)] for i in unlensed_idx)
        self.meta = []
        for local, original in enumerate(lensed_idx):
            self.meta.append({"tag": "L1", "pair_id": int(local), "source_index": int(original)})
        for local, original in enumerate(lensed_idx):
            self.meta.append({"tag": "L2", "pair_id": int(local), "source_index": int(original)})
        for local, original in enumerate(unlensed_idx):
            self.meta.append({"tag": "U", "pair_id": -1, "source_index": int(original)})

    def __len__(self) -> int:
        return len(self.waveforms)

    def __getitem__(self, idx: int) -> torch.Tensor:
        x = pad_or_trim(self.waveforms[idx], self.cfg.target_len, self.cfg.stride)
        if str(getattr(self.cfg, "preprocess", "none")).lower() == "multiband":
            x = multiband_preprocess(x, self.cfg)
            if self.cfg.aug_flip:
                x = peak_flip_channels(x)
            return torch.from_numpy(zscore_channels(x))
        x = spectral_preprocess(x, self.cfg)
        if self.cfg.aug_flip:
            x = peak_flip_channels(x)
        return torch.from_numpy(to_channels(zscore_channels(x), self.cfg.use_hilbert))


def ground_truth_partner(meta: list[dict]) -> np.ndarray:
    # 构造每个事件的真实伴随事件索引；孤立事件用 -1 表示没有真实 partner。
    gt = np.full(len(meta), -1, dtype=np.int64)
    l1_by_pair = {m["pair_id"]: i for i, m in enumerate(meta) if m["tag"] == "L1"}
    l2_by_pair = {m["pair_id"]: i for i, m in enumerate(meta) if m["tag"] == "L2"}
    for pair_id, i in l1_by_pair.items():
        j = l2_by_pair.get(pair_id)
        if j is not None:
            gt[i] = j
            gt[j] = i
    return gt
