from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import hilbert


# 只增强 match 项目已有的 SIS/PM 两类 lensed 数据；unlensed 单独处理。
FAMILIES = ("SIS", "PM")


def _load(path: Path, n: int | None) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    x = np.load(path, mmap_mode="r")
    return x[:n] if n is not None else x


def _safe_mkdir(path: Path, overwrite: bool = False) -> None:
    # 默认拒绝写入已有目录，防止误覆盖原始 match 数据或已生成的数据。
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"refusing to overwrite existing directory: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=False)


def _copy_csvs(src_dir: Path, dst_dir: Path) -> None:
    for csv in src_dir.glob("*.csv"):
        shutil.copy2(csv, dst_dir / csv.name)


def _zscore_rows(x: np.ndarray) -> np.ndarray:
    y = x.astype(np.float32, copy=False)
    y = y - y.mean(axis=1, keepdims=True)
    y = y / (y.std(axis=1, keepdims=True) + 1e-8)
    return y.astype(np.float32, copy=False)


def _shift_zero_fill(x: np.ndarray, shift: int) -> np.ndarray:
    # 用零填充平移模拟观测到达时间差，不使用 np.roll，避免尾部绕回造成假信号。
    if shift == 0:
        return x.copy()
    out = np.zeros_like(x)
    if shift > 0:
        out[shift:] = x[:-shift]
    else:
        out[:shift] = x[-shift:]
    return out


def _colored_noise(rng: np.random.Generator, n: int, alpha: float = 1.0) -> np.ndarray:
    # 生成 1/f^alpha 风格 colored noise，比白噪声更接近真实探测器非平坦噪声。
    white = rng.normal(0.0, 1.0, n).astype(np.float32)
    spec = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n)
    scale = np.ones_like(freqs)
    scale[1:] = 1.0 / np.maximum(freqs[1:], 1.0 / n) ** (alpha / 2.0)
    y = np.fft.irfft(spec * scale, n=n).astype(np.float32)
    return y / (float(y.std()) + 1e-8)


def _lowfreq_drift(rng: np.random.Generator, n: int) -> np.ndarray:
    anchors = rng.normal(0.0, 1.0, 16).astype(np.float32)
    xp = np.linspace(0, n - 1, len(anchors))
    x = np.arange(n)
    drift = np.interp(x, xp, anchors).astype(np.float32)
    return drift / (float(drift.std()) + 1e-8)


def _sine_glitch(rng: np.random.Generator, n: int) -> np.ndarray:
    center = int(rng.integers(n // 5, 4 * n // 5))
    width = int(rng.integers(max(16, n // 512), max(32, n // 96)))
    freq = float(rng.uniform(30.0, 220.0))
    sr = 4096.0
    idx = np.arange(n, dtype=np.float32)
    env = np.exp(-0.5 * ((idx - center) / max(width, 1)) ** 2)
    phase = float(rng.uniform(0.0, 2 * np.pi))
    return (np.sin(2 * np.pi * freq * idx / sr + phase) * env).astype(np.float32)


def _apply_morse(x: np.ndarray, morse_phase: float) -> np.ndarray:
    # Morse 相位：pi/2 用 Hilbert 变换近似，pi 直接翻转符号。
    if abs(morse_phase - np.pi / 2) < 1e-6:
        return np.imag(hilbert(x)).astype(np.float32)
    if abs(morse_phase - np.pi) < 1e-6:
        return (-x).astype(np.float32)
    return x.astype(np.float32, copy=True)


def _augment_lensed_pair(
    # 对一对 lensed images 注入 GW 风格差异：放大率、Morse 相位、时间延迟和小幅随机扰动。
    h1: np.ndarray,
    h2: np.ndarray,
    lens_row: pd.Series,
    rng: np.random.Generator,
    max_delay_samples: int,
    jitter_samples: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    mu0 = float(abs(lens_row.get("mu_0", 1.0)))
    mu1_raw = float(lens_row.get("mu_1", 1.0))
    mu1 = abs(mu1_raw)
    morse1 = 0.0
    morse2 = float(np.pi / 2 if mu1_raw < 0 else 0.0)
    td = float(lens_row.get("t_d", 0.0))

    # The original match time delays can be in lensing units with huge dynamic range.
    # Compress them into observable sample offsets while preserving ordering.
    delay = int(np.sign(td) * min(max_delay_samples, np.log1p(abs(td)) * 64.0))
    delay += int(rng.integers(-jitter_samples, jitter_samples + 1))

    a1 = float(np.sqrt(max(mu0, 1e-6)) * rng.lognormal(0.0, 0.05))
    a2 = float(np.sqrt(max(mu1, 1e-6)) * rng.lognormal(0.0, 0.05))
    y1 = h1.astype(np.float32) * a1
    y2 = _shift_zero_fill(_apply_morse(h2.astype(np.float32), morse2), delay) * a2
    return y1.astype(np.float32), y2.astype(np.float32), {
        "gwaug_mu0": mu0,
        "gwaug_mu1": mu1,
        "gwaug_morse_1": morse1,
        "gwaug_morse_2": morse2,
        "gwaug_delay_samples": delay,
        "gwaug_amp_1": a1,
        "gwaug_amp_2": a2,
    }


def _make_noisy(h: np.ndarray, rng: np.random.Generator, noise_std: float, glitch_prob: float) -> np.ndarray:
    # noisy 数据 = pure waveform + colored noise + 低频漂移 + 可选 sine-Gaussian glitch。
    n = h.shape[0]
    y = h.astype(np.float32).copy()
    y += noise_std * _colored_noise(rng, n, alpha=float(rng.uniform(0.6, 1.4)))
    y += (noise_std * 0.35) * _lowfreq_drift(rng, n)
    y *= float(rng.lognormal(0.0, 0.03))
    if rng.random() < glitch_prob:
        y += float(rng.uniform(0.2, 0.8)) * _sine_glitch(rng, n)
    return y.astype(np.float32)


def _write_memmap(path: Path, shape: tuple[int, ...], dtype=np.float32):
    path.parent.mkdir(parents=True, exist_ok=True)
    return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)


def _copy_npy_subset(src: Path, dst: Path, n: int, chunk_size: int) -> None:
    if not src.exists():
        return
    arr = np.load(src, mmap_mode="r")
    shape = (min(n, arr.shape[0]), *arr.shape[1:])
    out = _write_memmap(dst, shape, dtype=arr.dtype)
    for start in range(0, shape[0], chunk_size):
        stop = min(shape[0], start + chunk_size)
        out[start:stop] = arr[start:stop]
        out.flush()


def generate_family(args, family: str, rng: np.random.Generator) -> dict:
    # 生成 SIS/PM lensed 数据，输出仍保持 match 的目录和文件名，训练脚本可直接读取。
    src_dir = args.source_root / f"{family}_data_0222"
    dst_dir = args.out_root / f"{family}_data_0222"
    _safe_mkdir(dst_dir, overwrite=args.overwrite)
    _copy_csvs(src_dir, dst_dir)

    lens = pd.read_csv(src_dir / "lens.csv")
    h1_src = _load(src_dir / f"{family}_h_strain_1.npy", args.n_lensed)
    h2_src = _load(src_dir / f"{family}_h_strain_2.npy", args.n_lensed)
    n = min(len(h1_src), len(h2_src), len(lens))
    length = int(h1_src.shape[1])

    h1 = _write_memmap(dst_dir / f"{family}_h_strain_1.npy", (n, length))
    h2 = _write_memmap(dst_dir / f"{family}_h_strain_2.npy", (n, length))
    d1 = _write_memmap(dst_dir / f"{family}_data_strain_1.npy", (n, length))
    d2 = _write_memmap(dst_dir / f"{family}_data_strain_2.npy", (n, length))
    snr1 = _write_memmap(dst_dir / f"{family}_optimal_SNR_1.npy", (n,), dtype=np.float32)
    snr2 = _write_memmap(dst_dir / f"{family}_optimal_SNR_2.npy", (n,), dtype=np.float32)
    _copy_npy_subset(src_dir / f"{family}_time_array_1.npy", dst_dir / f"{family}_time_array_1.npy", n, args.chunk_size)
    _copy_npy_subset(src_dir / f"{family}_time_array_2.npy", dst_dir / f"{family}_time_array_2.npy", n, args.chunk_size)

    rows = []
    for start in range(0, n, args.chunk_size):
        stop = min(n, start + args.chunk_size)
        for i in range(start, stop):
            y1, y2, meta = _augment_lensed_pair(
                h1_src[i], h2_src[i], lens.iloc[i], rng, args.max_delay_samples, args.jitter_samples
            )
            y1 = y1 / (float(np.std(y1)) + 1e-8)
            y2 = y2 / (float(np.std(y2)) + 1e-8)
            nd1 = _make_noisy(y1, rng, args.noise_std, args.glitch_prob)
            nd2 = _make_noisy(y2, rng, args.noise_std, args.glitch_prob)
            h1[i] = y1
            h2[i] = y2
            d1[i] = nd1
            d2[i] = nd2
            snr1[i] = float(np.std(y1) / (np.std(nd1 - y1) + 1e-8))
            snr2[i] = float(np.std(y2) / (np.std(nd2 - y2) + 1e-8))
            rows.append({"index": i, "family": family, **meta})
        h1.flush(); h2.flush(); d1.flush(); d2.flush(); snr1.flush(); snr2.flush()

    pd.DataFrame(rows).to_csv(dst_dir / "gw_augmented_metadata.csv", index=False)
    return {"family": family, "n_lensed": n, "length": length, "output_dir": str(dst_dir)}


def generate_unlensed(args, rng: np.random.Generator) -> dict:
    # 孤立事件没有 lensed partner，只做轻微幅度/时间扰动和同样的噪声模型。
    src_dir = args.source_root / "Unlensed_data_0222"
    dst_dir = args.out_root / "Unlensed_data_0222"
    _safe_mkdir(dst_dir, overwrite=args.overwrite)
    _copy_csvs(src_dir, dst_dir)

    h_src = _load(src_dir / "unlensed_h_strain.npy", args.n_unlensed)
    n = len(h_src)
    length = int(h_src.shape[1])
    h = _write_memmap(dst_dir / "unlensed_h_strain.npy", (n, length))
    d = _write_memmap(dst_dir / "unlensed_data_strain.npy", (n, length))
    snr = _write_memmap(dst_dir / "unlensed_optimal_SNR.npy", (n,), dtype=np.float32)
    _copy_npy_subset(src_dir / "unlensed_time_array.npy", dst_dir / "unlensed_time_array.npy", n, args.chunk_size)

    rows = []
    for start in range(0, n, args.chunk_size):
        stop = min(n, start + args.chunk_size)
        for i in range(start, stop):
            amp = float(rng.lognormal(0.0, 0.08))
            shift = int(rng.integers(-args.jitter_samples, args.jitter_samples + 1))
            y = _shift_zero_fill(h_src[i].astype(np.float32), shift) * amp
            y = y / (float(np.std(y)) + 1e-8)
            nd = _make_noisy(y, rng, args.noise_std, args.glitch_prob)
            h[i] = y
            d[i] = nd
            snr[i] = float(np.std(y) / (np.std(nd - y) + 1e-8))
            rows.append({"index": i, "gwaug_amp": amp, "gwaug_shift_samples": shift})
        h.flush(); d.flush(); snr.flush()

    pd.DataFrame(rows).to_csv(dst_dir / "gw_augmented_metadata.csv", index=False)
    return {"family": "unlensed", "n_unlensed": n, "length": length, "output_dir": str(dst_dir)}


def main() -> None:
    # 从 match 原始数据派生新数据集；除非显式 --overwrite，否则不会覆盖任何已有目录。
    ap = argparse.ArgumentParser(description="Generate a GW-augmented match-style dataset without overwriting the original match arrays.")
    ap.add_argument("--source-root", type=Path, default=Path("/root/autodl-tmp/qkzhang"))
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--families", nargs="+", choices=FAMILIES, default=list(FAMILIES))
    ap.add_argument("--n-lensed", type=int, default=None)
    ap.add_argument("--n-unlensed", type=int, default=None)
    ap.add_argument("--seed", type=int, default=20260522)
    ap.add_argument("--noise-std", type=float, default=0.20)
    ap.add_argument("--glitch-prob", type=float, default=0.03)
    ap.add_argument("--max-delay-samples", type=int, default=2048)
    ap.add_argument("--jitter-samples", type=int, default=128)
    ap.add_argument("--chunk-size", type=int, default=32)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if args.out_root.resolve() == args.source_root.resolve():
        raise ValueError("out-root must differ from source-root")
    if args.out_root.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to write into existing out-root: {args.out_root}")
    args.out_root.mkdir(parents=True, exist_ok=args.overwrite)

    rng = np.random.default_rng(args.seed)
    summary = {
        "source_root": str(args.source_root),
        "out_root": str(args.out_root),
        "seed": args.seed,
        "noise_std": args.noise_std,
        "glitch_prob": args.glitch_prob,
        "max_delay_samples": args.max_delay_samples,
        "jitter_samples": args.jitter_samples,
        "notes": "Derived from the 0123 match generation outputs (PM_GW_events.py, SIS_GW_events.py, unlensed_GW_events.py). Adds GW-catalog style magnification perturbation, compressed time-delay sample shifts, Morse phase/Hilbert transform, colored nonstationary noise, occasional glitches, copied time arrays, and per-event metadata. Original source data is never modified.",
        "families": [],
    }
    for family in args.families:
        summary["families"].append(generate_family(args, family, rng))
    summary["unlensed"] = generate_unlensed(args, rng)

    with open(args.out_root / "gw_augmented_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
