from __future__ import annotations

import importlib
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import RidgeCV
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, get_worker_info

from matchgw.config import MatchRunConfig
from matchgw.matching import retrieval_metrics, similarity_matrix
from matchgw.models import NTXentLoss
from matchgw.pipeline import build_model, embed_eval
from matchgw.trigger_time import catalog_trigger_time_frame, log1p_delta_time_obs

import matchgw.data as data_mod

aux = importlib.import_module("scripts.experiments.21_observable_aux_reranker")

OUT_ROOT = Path("runs/mixed_sis_pm_catalog_modality_compare_20260609")
ROOTS = {
    ("SIS", "ET"): Path("/root/autodl-tmp/gw_et_10000_matchstyle_20260527_091859"),
    ("SIS", "LIGO"): Path("/root/autodl-tmp/gw_ligo_10000_matchstyle_20260527_091859"),
    ("PM", "ET"): Path("data_generation/pm_mass_1e4_1e10_td_min24s_matchroots/ET"),
    ("PM", "LIGO"): Path("data_generation/pm_mass_1e4_1e10_td_min24s_matchroots/LIGO"),
    ("SIS", "ET3"): Path("/root/autodl-tmp/createdata/et3_10000_20260616_1006_match_root"),
    ("PM", "ET3"): Path("/root/autodl-tmp/createdata/et3_10000_20260616_1006_match_root"),
}
JOBS = [("ET", "pure"), ("ET", "noisy"), ("LIGO", "pure"), ("LIGO", "noisy")]
FAMILIES = ["SIS", "PM"]
DIRECT_VARIANTS = [
    "waveform_only",
    "time_only",
    "true_sky_sep_only",
    "true_sky_overlap_only",
    "predicted_sky_overlap_only",
]
RERANK_VARIANTS = [
    "waveform_plus_time",
    "waveform_plus_true_sky_sep",
    "waveform_plus_true_sky_overlap",
    "waveform_plus_predicted_sky_overlap",
    "time_plus_true_sky_sep",
    "time_plus_true_sky_overlap",
    "time_plus_predicted_sky_overlap",
    "waveform_plus_time_plus_true_sky_sep",
    "waveform_plus_time_plus_true_sky_overlap",
    "waveform_plus_time_plus_predicted_sky_overlap",
]
NEG_PER_POS = 300
CHUNK_ROWS = 16
REAL_SKY_SIGMA_RAD = 0.08
MIN_SKY_SIGMA = 0.03
EPS = 1e-8


class FamilyArrays:
    def __init__(self, family: str, root: Path, mode: str, limit: int = 10000):
        tag = "h_strain" if mode == "pure" else "data_strain"
        unlensed_name = "unlensed_h_strain.npy" if mode == "pure" else "unlensed_data_strain.npy"
        source = root / f"{family}_data_0222"
        self.family = family
        self.root = root
        self.l1 = np.load(source / f"{family}_{tag}_1.npy", mmap_mode="r")[:limit]
        self.l2 = np.load(source / f"{family}_{tag}_2.npy", mmap_mode="r")[:limit]
        self.unlensed = np.load(root / "Unlensed_data_0222" / unlensed_name, mmap_mode="r")[:limit]


def split_indices(n: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    p = rng.permutation(n)
    n_train = int(round(n * 0.70))
    n_val = int(round(n * 0.15))
    return {"train": p[:n_train], "val": p[n_train:n_train + n_val], "test": p[n_train + n_val:]}


def make_cfg(detector: str, mode: str, out_dir: Path) -> MatchRunConfig:
    is_ligo = detector == "LIGO"
    return MatchRunConfig(
        data_root=ROOTS[("SIS", detector)],
        model_type="SIS",
        data_mode=mode,
        out_dir=out_dir,
        backbone="inceptiontime",
        preprocess="bandpass",
        bandpass_low=40,
        bandpass_high=580,
        target_len=8192,
        stride=2,
        lensed_limit=10000,
        unlensed_limit=10000,
        epochs=50,
        batch_size=128 if not is_ligo else 96,
        eval_batch_size=512 if not is_ligo else 256,
        lr=1e-3,
        weight_decay=1e-4,
        tau=0.07,
        emb_dim=128,
        d_model=256,
        width_scale=2.0,
        aug_roll=128,
        aug_scale=0.10,
        aug_noise=0.01,
        aug_flip=True,
        amp=True,
        amp_dtype="bf16",
        num_workers=2,
        pin_memory=True,
        export_candidates=False,
    )


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _amp_dtype(cfg: MatchRunConfig) -> torch.dtype:
    return torch.float16 if cfg.amp_dtype == "fp16" else torch.bfloat16


def _seed_worker(worker_id: int) -> None:
    info = get_worker_info()
    if info is not None and hasattr(info.dataset, "rng"):
        info.dataset.rng = np.random.default_rng(info.seed % (2**32))


def loader_kwargs(cfg: MatchRunConfig, device: torch.device, train: bool = False) -> dict:
    out = {"num_workers": cfg.num_workers, "pin_memory": bool(cfg.pin_memory and device.type == "cuda")}
    if cfg.num_workers > 0:
        out["persistent_workers"] = True
        out["prefetch_factor"] = 2
        if train:
            out["worker_init_fn"] = _seed_worker
    return out


def prepare_waveform(x: np.ndarray, cfg: MatchRunConfig, train: bool, rng: np.random.Generator | None = None) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    y = data_mod.pad_or_trim(x, cfg.target_len, cfg.stride)
    y = data_mod.spectral_preprocess(y, cfg)
    if train:
        y = y.copy()
        if cfg.aug_flip:
            y = data_mod.peak_flip_channels(y)
        if cfg.aug_roll > 0:
            if rng is None:
                raise ValueError("rng required in train mode")
            y = np.roll(y, int(rng.integers(-cfg.aug_roll, cfg.aug_roll + 1)), axis=-1)
        if cfg.aug_scale > 0:
            y = y * float(1.0 + rng.uniform(-cfg.aug_scale, cfg.aug_scale))
        if cfg.aug_noise > 0:
            scale = y.std(axis=-1, keepdims=True) + 1e-8 if y.ndim > 1 else float(y.std()) + 1e-8
            y = y + rng.normal(0.0, cfg.aug_noise * scale, size=y.shape)
    elif cfg.aug_flip:
        y = data_mod.peak_flip_channels(y)
    return data_mod.to_channels(data_mod.zscore_channels(y), cfg.use_hilbert)


class MixedPairDataset(Dataset):
    def __init__(self, arrays: dict[str, FamilyArrays], splits: dict[str, dict[str, np.ndarray]], split: str, cfg: MatchRunConfig):
        self.arrays = arrays
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.items = []
        for fam in FAMILIES:
            self.items.extend((fam, "L", int(i)) for i in splits[fam][split])
            self.items.extend((fam, "U", int(i)) for i in splits[f"{fam}_U"][split])

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        fam, kind, src_idx = self.items[idx]
        arr = self.arrays[fam]
        if kind == "L":
            a = prepare_waveform(arr.l1[src_idx], self.cfg, True, self.rng)
            b = prepare_waveform(arr.l2[src_idx], self.cfg, True, self.rng)
        else:
            a = prepare_waveform(arr.unlensed[src_idx], self.cfg, True, self.rng)
            b = prepare_waveform(arr.unlensed[src_idx], self.cfg, True, self.rng)
        return torch.from_numpy(a), torch.from_numpy(b)


class MixedEvaluationSet(Dataset):
    def __init__(self, arrays: dict[str, FamilyArrays], splits: dict[str, dict[str, np.ndarray]], split: str, cfg: MatchRunConfig):
        self.cfg = cfg
        self.waveforms = []
        self.meta = []
        pair_id = 0
        for fam in FAMILIES:
            lensed_idx = splits[fam][split]
            for original in lensed_idx:
                self.waveforms.append(arrays[fam].l1[int(original)])
                self.meta.append({"family": fam, "tag": "L1", "pair_id": pair_id, "source_index": int(original)})
                pair_id += 1
            start_pair = pair_id - len(lensed_idx)
            for local, original in enumerate(lensed_idx):
                self.waveforms.append(arrays[fam].l2[int(original)])
                self.meta.append({"family": fam, "tag": "L2", "pair_id": start_pair + local, "source_index": int(original)})
            for original in splits[f"{fam}_U"][split]:
                self.waveforms.append(arrays[fam].unlensed[int(original)])
                self.meta.append({"family": fam, "tag": "U", "pair_id": -1, "source_index": int(original)})

    def __len__(self) -> int:
        return len(self.waveforms)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.from_numpy(prepare_waveform(self.waveforms[idx], self.cfg, False))


def ground_truth(meta: list[dict]) -> np.ndarray:
    gt = np.full(len(meta), -1, dtype=np.int64)
    l1 = {m["pair_id"]: i for i, m in enumerate(meta) if m["tag"] == "L1"}
    l2 = {m["pair_id"]: i for i, m in enumerate(meta) if m["tag"] == "L2"}
    for pid, i in l1.items():
        j = l2.get(pid)
        if j is not None:
            gt[i] = j
            gt[j] = i
    return gt


def mixed_obs_frame(detector: str, split: str, splits: dict[str, dict[str, np.ndarray]], kind: str) -> pd.DataFrame:
    frames = []
    for fam in FAMILIES:
        root = ROOTS[(fam, detector)]
        lensed_idx = splits[fam][split]
        unlensed_idx = splits[f"{fam}_U"][split]
        if kind == "raw":
            frame = aux.catalog_observable_frame(root, fam, lensed_idx, unlensed_idx).reset_index(drop=True)
        else:
            frame = catalog_trigger_time_frame(root, fam, lensed_idx, unlensed_idx, detector=detector).reset_index(drop=True)
        frame["family"] = fam
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def unit_vectors(obs: pd.DataFrame) -> np.ndarray:
    ra = obs["ra"].to_numpy(dtype=np.float64)
    dec = obs["dec"].to_numpy(dtype=np.float64)
    return np.column_stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)]).astype(np.float32)


def normalize_vectors(x: np.ndarray) -> np.ndarray:
    return (x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), EPS)).astype(np.float32)


def angular_sep_from_unit(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.arccos(np.clip(np.sum(a * b, axis=-1), -1.0, 1.0))


def true_sky_sep(raw_obs: pd.DataFrame, a: np.ndarray, c: np.ndarray) -> np.ndarray:
    ra = raw_obs["ra"].to_numpy(dtype=np.float64)
    dec = raw_obs["dec"].to_numpy(dtype=np.float64)
    return aux.angular_sep(ra[a], dec[a], ra[c], dec[c]).astype(np.float32)


def true_log_sky_overlap(raw_obs: pd.DataFrame, a: np.ndarray, c: np.ndarray) -> np.ndarray:
    sep = true_sky_sep(raw_obs, a, c)
    var = REAL_SKY_SIGMA_RAD * REAL_SKY_SIGMA_RAD * 2.0
    return (-np.log(2.0 * np.pi * var) - (sep * sep) / (2.0 * var)).astype(np.float32)


def log_gaussian_overlap_from_unit(mu_i: np.ndarray, mu_j: np.ndarray, sigma_i: float, sigma_j: float) -> np.ndarray:
    sep = angular_sep_from_unit(mu_i, mu_j)
    var = sigma_i * sigma_i + sigma_j * sigma_j
    return (-np.log(2.0 * np.pi * var) - (sep * sep) / (2.0 * var)).astype(np.float32)


def row_ranks(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(-scores, axis=1)
    ranks = np.empty_like(order, dtype=np.int32)
    ranks[np.arange(scores.shape[0])[:, None], order] = np.arange(1, scores.shape[1] + 1, dtype=np.int32)
    return ranks


def add_percent(metrics: dict, ranks: np.ndarray, n: int) -> dict:
    usable = max(n - 1, 1)
    for pct in (1, 5, 10):
        k = max(1, int(math.ceil(usable * pct / 100.0)))
        metrics[f"top_{pct}pct_k"] = k
        metrics[f"top_{pct}pct"] = float(np.mean(ranks <= k))
    return metrics


def metrics_from_scores(scores: np.ndarray, gt: np.ndarray) -> dict:
    scores = scores.copy()
    np.fill_diagonal(scores, -np.inf)
    valid = np.flatnonzero(gt >= 0).astype(np.int32)
    true = scores[valid, gt[valid].astype(int)]
    ranks = 1 + np.sum(scores[valid] > true[:, None], axis=1)
    out = retrieval_metrics(scores, gt, ks=(1, 5, 10, 50, 100, 500))
    return add_percent(out, ranks.astype(np.int32), scores.shape[1])


def direct_scores(variant: str, raw_obs: pd.DataFrame, time_obs: pd.DataFrame, scores: np.ndarray, sky_mu: np.ndarray | None, sky_sigma: float | None) -> np.ndarray:
    n = len(time_obs)
    if variant == "waveform_only":
        return scores.astype(np.float32)
    out = np.empty((n, n), dtype=np.float32)
    cols = np.arange(n, dtype=np.int32)
    for start in range(0, n, CHUNK_ROWS):
        rows = np.arange(start, min(start + CHUNK_ROWS, n), dtype=np.int32)
        a = np.repeat(rows, n).astype(np.int32)
        c = np.tile(cols, len(rows)).astype(np.int32)
        if variant == "time_only":
            vals = -log1p_delta_time_obs(time_obs, a, c)
        elif variant == "true_sky_sep_only":
            vals = -true_sky_sep(raw_obs, a, c)
        elif variant == "true_sky_overlap_only":
            vals = true_log_sky_overlap(raw_obs, a, c)
        elif variant == "predicted_sky_overlap_only":
            vals = log_gaussian_overlap_from_unit(sky_mu[a], sky_mu[c], sky_sigma, sky_sigma)
        else:
            raise ValueError(variant)
        out[start:start + len(rows)] = vals.reshape(len(rows), n)
    np.fill_diagonal(out, -np.inf)
    return out


def feature_matrix(variant: str, raw_obs: pd.DataFrame, time_obs: pd.DataFrame, sky_mu: np.ndarray | None, sky_sigma: float | None, scores: np.ndarray, ranks: np.ndarray, a: np.ndarray, c: np.ndarray) -> np.ndarray:
    cols = []
    if "time" in variant:
        cols.append(log1p_delta_time_obs(time_obs, a, c))
    if "true_sky_sep" in variant:
        cols.append(true_sky_sep(raw_obs, a, c))
    if "true_sky_overlap" in variant:
        cols.append(true_log_sky_overlap(raw_obs, a, c))
    if "predicted_sky_overlap" in variant:
        cols.append(log_gaussian_overlap_from_unit(sky_mu[a], sky_mu[c], sky_sigma, sky_sigma))
    if "waveform" in variant:
        cols.append(scores[a, c].astype(np.float32))
        cols.append((1.0 / np.maximum(ranks[a, c], 1)).astype(np.float32))
    return np.column_stack(cols).astype(np.float32)


def train_examples(variant: str, raw_obs: pd.DataFrame, time_obs: pd.DataFrame, gt: np.ndarray, sky_mu: np.ndarray | None, sky_sigma: float | None, scores: np.ndarray, ranks: np.ndarray, seed: int):
    rng = np.random.default_rng(seed)
    valid = np.flatnonzero(gt >= 0).astype(np.int32)
    n = len(time_obs)
    pos_a = valid
    pos_c = gt[valid].astype(np.int32)
    neg_a = np.repeat(pos_a, NEG_PER_POS)
    neg_c = rng.integers(0, n, size=len(neg_a), dtype=np.int32)
    bad = (neg_c == neg_a) | (neg_c == gt[neg_a])
    while bad.any():
        neg_c[bad] = rng.integers(0, n, size=int(bad.sum()), dtype=np.int32)
        bad = (neg_c == neg_a) | (neg_c == gt[neg_a])
    a = np.concatenate([pos_a, neg_a])
    c = np.concatenate([pos_c, neg_c])
    y = np.concatenate([np.ones(len(pos_a), dtype=np.int8), np.zeros(len(neg_a), dtype=np.int8)])
    x = feature_matrix(variant, raw_obs, time_obs, sky_mu, sky_sigma, scores, ranks, a, c)
    order = rng.permutation(len(y))
    return x[order], y[order]


def eval_rerank(variant: str, clf, raw_obs: pd.DataFrame, time_obs: pd.DataFrame, gt: np.ndarray, sky_mu: np.ndarray | None, sky_sigma: float | None, scores: np.ndarray, ranks: np.ndarray) -> dict:
    valid = np.flatnonzero(gt >= 0).astype(np.int32)
    n = len(time_obs)
    out = []
    for start in range(0, len(valid), CHUNK_ROWS):
        rows = valid[start:start + CHUNK_ROWS]
        a = np.repeat(rows, n).astype(np.int32)
        c = np.tile(np.arange(n, dtype=np.int32), len(rows))
        x = feature_matrix(variant, raw_obs, time_obs, sky_mu, sky_sigma, scores, ranks, a, c)
        pred = clf.predict_proba(x)[:, 1].reshape(len(rows), n)
        pred[np.arange(len(rows)), rows] = -np.inf
        true = pred[np.arange(len(rows)), gt[rows].astype(int)]
        out.extend((1 + np.sum(pred > true[:, None], axis=1)).tolist())
    r = np.asarray(out, dtype=np.int32)
    return add_percent({
        "r@1": float(np.mean(r <= 1)),
        "r@5": float(np.mean(r <= 5)),
        "r@10": float(np.mean(r <= 10)),
        "r@50": float(np.mean(r <= 50)),
        "r@100": float(np.mean(r <= 100)),
        "r@500": float(np.mean(r <= 500)),
        "median_true_rank": float(np.median(r)),
        "valid": int(len(valid)),
    }, r, n)


def fit_sky_predictor(train_raw: pd.DataFrame, train_emb: np.ndarray, val_raw: pd.DataFrame, val_emb: np.ndarray):
    model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-4, 3, 12)))
    model.fit(train_emb, unit_vectors(train_raw))
    val_pred = normalize_vectors(model.predict(val_emb))
    err = angular_sep_from_unit(val_pred, unit_vectors(val_raw))
    return model, float(max(MIN_SKY_SIGMA, np.median(err))), float(np.mean(err)), float(np.median(err))


def train_or_load_encoder(cfg: MatchRunConfig, arrays: dict[str, FamilyArrays], splits: dict[str, dict[str, np.ndarray]]):
    model_path = cfg.out_dir / "model.pt"
    summary_path = cfg.out_dir / "waveform_summary.json"
    in_channels = data_mod.prepared_channel_count(arrays[FAMILIES[0]].l1[0], cfg)
    if model_path.exists() and summary_path.exists():
        model = build_model(cfg, in_channels=in_channels)
        ckpt = torch.load(model_path, map_location="cpu")
        model.load_state_dict(ckpt["model"], strict=True)
        train_info = json.loads(summary_path.read_text(encoding="utf-8")).get("timing", {})
        return model, train_info
    device = _device()
    train_ds = MixedPairDataset(arrays, splits, "train", cfg)
    dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True, **loader_kwargs(cfg, device, True))
    model = build_model(cfg, in_channels=in_channels).to(device)
    loss_fn = NTXentLoss(cfg.tau)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=bool(cfg.amp and cfg.amp_dtype == "fp16" and device.type == "cuda"))
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    history = []
    t0 = time.perf_counter()
    for epoch in range(1, cfg.epochs + 1):
        e0 = time.perf_counter()
        losses, batches, pairs = [], 0, 0
        model.train()
        for xa, xb in dl:
            batches += 1
            pairs += int(xa.shape[0])
            xa = xa.to(device, non_blocking=True)
            xb = xb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=_amp_dtype(cfg), enabled=bool(cfg.amp and device.type == "cuda")):
                loss = loss_fn(model(xa), model(xb))
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()
            losses.append(float(loss.detach().cpu()))
        row = {"epoch": epoch, "loss": float(np.mean(losses)), "epoch_s": float(time.perf_counter() - e0), "batches": batches, "batch_pairs": pairs}
        history.append(row)
        print(json.dumps(row), flush=True)
    train_s = time.perf_counter() - t0
    train_info = {"history": history, "train_s": train_s, "mean_epoch_s": float(np.mean([r["epoch_s"] for r in history])), "train_items": int(len(train_ds))}
    torch.save({"model": model.state_dict(), "config": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(cfg).items()}}, model_path)
    pd.DataFrame(history).to_csv(cfg.out_dir / "history.csv", index=False)
    return model, train_info


def split_pack(detector: str, split: str, cfg: MatchRunConfig, arrays: dict[str, FamilyArrays], splits: dict[str, dict[str, np.ndarray]], model=None):
    ds = MixedEvaluationSet(arrays, splits, split, cfg)
    gt = ground_truth(ds.meta)
    raw = mixed_obs_frame(detector, split, splits, "raw")
    tim = mixed_obs_frame(detector, split, splits, "time")
    emb = scores = None
    if model is not None:
        emb_path = cfg.out_dir / f"{split}_embeddings.npy"
        scores_path = cfg.out_dir / f"{split}_scores.npy"
        if emb_path.exists() and scores_path.exists():
            emb = np.load(emb_path)
            scores = np.load(scores_path)
        else:
            emb = embed_eval(model, ds, cfg, cpu=False).astype(np.float32)
            scores = similarity_matrix(emb).astype(np.float32)
            np.fill_diagonal(scores, -np.inf)
            np.save(emb_path, emb)
            np.save(scores_path, scores)
    return ds, raw, tim, gt, emb, scores


def run_one(detector: str, mode: str) -> list[dict]:
    out_dir = OUT_ROOT / f"{detector.lower()}_{mode}_mixed_sis_pm_ep50"
    cfg = make_cfg(detector, mode, out_dir)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    arrays = {fam: FamilyArrays(fam, ROOTS[(fam, detector)], mode) for fam in FAMILIES}
    splits = {}
    for i, fam in enumerate(FAMILIES):
        splits[fam] = split_indices(len(arrays[fam].l1), cfg.seed + i)
        splits[f"{fam}_U"] = split_indices(len(arrays[fam].unlensed), cfg.seed + 100 + i)
    print("TRAIN_OR_LOAD", detector, mode, cfg.out_dir, flush=True)
    total_t0 = time.perf_counter()
    model, train_info = train_or_load_encoder(cfg, arrays, splits)
    train_ds, train_raw, train_time, train_gt, train_emb, train_scores = split_pack(detector, "train", cfg, arrays, splits, model)
    val_ds, val_raw, val_time, val_gt, val_emb, val_scores = split_pack(detector, "val", cfg, arrays, splits, model)
    test_ds, test_raw, test_time, test_gt, test_emb, test_scores = split_pack(detector, "test", cfg, arrays, splits, model)
    val_ranks = row_ranks(val_scores)
    test_ranks = row_ranks(test_scores)
    sky_model, sky_sigma, sky_mean_err, sky_med_err = fit_sky_predictor(train_raw, train_emb, val_raw, val_emb)
    val_sky_mu = normalize_vectors(sky_model.predict(val_emb))
    test_sky_mu = normalize_vectors(sky_model.predict(test_emb))
    rows = []
    for variant in DIRECT_VARIANTS:
        print("DIRECT", detector, mode, variant, flush=True)
        met = metrics_from_scores(direct_scores(variant, test_raw, test_time, test_scores, test_sky_mu, sky_sigma), test_gt)
        rows.append({
            "detector": detector, "data_mode": mode, "catalog": "mixed_SIS_PM", "variant": variant, "stage": "direct_score",
            **met, "sky_sigma_rad": sky_sigma, "sky_val_mean_angular_error_rad": sky_mean_err, "sky_val_median_angular_error_rad": sky_med_err,
            "train_s": train_info.get("train_s", np.nan), "mean_epoch_s": train_info.get("mean_epoch_s", np.nan), "elapsed_s": float(time.perf_counter() - total_t0),
        })
        pd.DataFrame(rows).to_csv(cfg.out_dir / "mixed_modality_partial.csv", index=False)
    for idx, variant in enumerate(RERANK_VARIANTS):
        print("RERANK", detector, mode, variant, flush=True)
        x_val, y_val = train_examples(variant, val_raw, val_time, val_gt, val_sky_mu, sky_sigma, val_scores, val_ranks, seed=80000 + idx)
        clf = HistGradientBoostingClassifier(max_iter=260, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1e-4, class_weight="balanced", random_state=80000 + idx)
        clf.fit(x_val, y_val)
        pv = clf.predict_proba(x_val)[:, 1]
        met = eval_rerank(variant, clf, test_raw, test_time, test_gt, test_sky_mu, sky_sigma, test_scores, test_ranks)
        rows.append({
            "detector": detector, "data_mode": mode, "catalog": "mixed_SIS_PM", "variant": variant, "stage": "catalog_hgb_rerank",
            "val_auc_sampled": float(roc_auc_score(y_val, pv)), "train_examples": int(len(y_val)), "train_positive": int(y_val.sum()),
            **met, "sky_sigma_rad": sky_sigma, "sky_val_mean_angular_error_rad": sky_mean_err, "sky_val_median_angular_error_rad": sky_med_err,
            "train_s": train_info.get("train_s", np.nan), "mean_epoch_s": train_info.get("mean_epoch_s", np.nan), "elapsed_s": float(time.perf_counter() - total_t0),
        })
        pd.DataFrame(rows).to_csv(cfg.out_dir / "mixed_modality_partial.csv", index=False)
    pd.DataFrame(rows).to_csv(cfg.out_dir / "mixed_modality.csv", index=False)
    summary = {
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(cfg).items()},
        "splits": {k: {kk: int(len(vv)) for kk, vv in val.items()} for k, val in splits.items()},
        "timing": {k: v for k, v in train_info.items() if k != "history"},
        "sky_predictor": {"sigma_rad": sky_sigma, "val_mean_angular_error_rad": sky_mean_err, "val_median_angular_error_rad": sky_med_err},
    }
    (cfg.out_dir / "waveform_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return rows


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for detector, mode in JOBS:
        all_rows.extend(run_one(detector, mode))
        pd.DataFrame(all_rows).to_csv(OUT_ROOT / "mixed_modality_summary_partial.csv", index=False)
    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_ROOT / "mixed_modality_summary.csv", index=False)
    for metric in ["r@1", "r@5", "r@10", "top_1pct", "top_5pct", "top_10pct"]:
        df.pivot_table(index=["detector", "data_mode"], columns="variant", values=metric, aggfunc="first").to_csv(OUT_ROOT / f"{metric.replace('@', '')}_pivot.csv")
    print(df.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
