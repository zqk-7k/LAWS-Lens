#!/usr/bin/env python3
"""Run waveform + time-delay + observed-sky retrieval on a GW-LMC-prior catalog."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

base = importlib.import_module("scripts.experiments.80_mixed_sis_pm_catalog_modality_compare")
fresh = importlib.import_module("scripts.experiments.84_fresh50_full_catalog_ranking")
liao = importlib.import_module("scripts.experiments.88_liao_realistic_p1_p2_rerank")

from matchgw.aux_priors import observed_sky_pair_features, select_best_weighted_lambdas  # noqa: E402

FAMILIES = ["SIS", "PM"]
FAMILY_LABEL = {
    "SIS": "GW-LMC smooth/non-subhalo slot",
    "PM": "GW-LMC subhalo-flagged slot",
}
GRID = [0.25, 0.5, 1.0, 2.0, 4.0]


def patch_roots(match_root: Path) -> None:
    base.ROOTS[("SIS", "ET3")] = match_root
    base.ROOTS[("PM", "ET3")] = match_root


def make_cfg(
    mode: str,
    out_dir: Path,
    epochs: int,
    backbone: str = "inceptiontime",
    target_len: int = 8192,
    stride: int = 2,
    batch_size: int | None = None,
    eval_batch_size: int | None = None,
    num_workers: int = 0,
):
    cfg = base.make_cfg("ET3", mode, out_dir)
    cfg.epochs = int(epochs)
    cfg.out_dir = out_dir
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    cfg.data_root = base.ROOTS[("SIS", "ET3")]
    cfg.backbone = str(backbone)
    cfg.target_len = int(target_len)
    cfg.stride = int(stride)
    if batch_size is not None:
        cfg.batch_size = int(batch_size)
    if eval_batch_size is not None:
        cfg.eval_batch_size = int(eval_batch_size)
    cfg.num_workers = int(num_workers)
    cfg.pin_memory = False
    cfg.export_candidates = False
    return cfg


def load_encoder_from_dir(cfg, arrays: dict, encoder_dir: Path):
    model_path = encoder_dir / "model.pt"
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    in_channels = base.data_mod.prepared_channel_count(arrays[FAMILIES[0]].l1[0], cfg)
    model = base.build_model(cfg, in_channels=in_channels)
    ckpt = torch.load(model_path, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)
    return model


def split_all(cfg, mode: str, encoder_dir: Path | None = None):
    arrays = {family: base.FamilyArrays(family, base.ROOTS[(family, "ET3")], mode) for family in FAMILIES}
    splits = {}
    for index, family in enumerate(FAMILIES):
        splits[family] = base.split_indices(len(arrays[family].l1), cfg.seed + index)
        splits[f"{family}_U"] = base.split_indices(len(arrays[family].unlensed), cfg.seed + 100 + index)
    if encoder_dir is None:
        model, train_info = base.train_or_load_encoder(cfg, arrays, splits)
        encoder_source = "trained_on_liao_gwlmc_prior_catalog"
    else:
        model = load_encoder_from_dir(cfg, arrays, encoder_dir)
        train_info = {"train_s": 0.0, "mean_epoch_s": 0.0}
        encoder_source = f"transfer_from:{encoder_dir}"
    out = {}
    for split in ["train", "val", "test"]:
        out[split] = base.split_pack("ET3", split, cfg, arrays, splits, model)
    summary_path = cfg.out_dir / "waveform_summary.json"
    if not summary_path.exists():
        summary = {
            "config": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(cfg).items()},
            "splits": {k: {kk: int(len(vv)) for kk, vv in val.items()} for k, val in splits.items()},
            "timing": {k: v for k, v in train_info.items() if k != "history"},
            "encoder_source": encoder_source,
            "note": "Liao/GW-LMC-prior catalog encoder; family names are compatibility slots.",
        }
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return out, train_info, encoder_source


def row_metrics(score: np.ndarray, gt: np.ndarray, meta: list[dict]) -> dict[str, dict]:
    return liao.evaluate_score(score, gt, meta)


def flatten_rows(stage: str, variant: str, metrics: dict[str, dict], diag: dict, extra: dict | None = None) -> list[dict]:
    extra = extra or {}
    rows = []
    for subset, values in metrics.items():
        rows.append({
            "stage": stage,
            "variant": variant,
            "subset": subset,
            **diag,
            **values,
            **extra,
        })
    return rows


def top_pair_rows(score: np.ndarray, meta: list[dict], gt: np.ndarray, n_top: int = 50) -> pd.DataFrame:
    s = score.copy()
    np.fill_diagonal(s, -np.inf)
    n = s.shape[0]
    tri_i, tri_j = np.triu_indices(n, k=1)
    vals = s[tri_i, tri_j]
    order = np.argsort(-vals)[:n_top]
    rows = []
    for rank, k in enumerate(order, start=1):
        i = int(tri_i[k])
        j = int(tri_j[k])
        rows.append({
            "rank": rank,
            "event_i": i,
            "event_j": j,
            "score": float(vals[k]),
            "is_true_pair": bool(gt[i] == j or gt[j] == i),
            "family_i": meta[i]["family"],
            "family_j": meta[j]["family"],
            "tag_i": meta[i]["tag"],
            "tag_j": meta[j]["tag"],
            "pair_id_i": int(meta[i]["pair_id"]),
            "pair_id_j": int(meta[j]["pair_id"]),
        })
    return pd.DataFrame(rows)


def write_report(out_dir: Path, summary: dict, result_df: pd.DataFrame, weights: dict, top_pairs: pd.DataFrame) -> None:
    def line_for(variant: str) -> str:
        sub = result_df[(result_df["subset"] == "overall") & (result_df["variant"] == variant)]
        if sub.empty:
            return ""
        r = sub.iloc[0]
        return f"| {variant} | {r['r@1']:.4f} | {r['r@5']:.4f} | {r['r@10']:.4f} | {r['r@50']:.4f} | {r['median_true_rank']:.2f} |"

    variants = [
        "waveform_only",
        "liao_time_lr_only",
        "observed_sky_step_only",
        "observed_sky_log_overlap_only",
        "time_plus_observed_sky_log_overlap_val_selected",
        "waveform_plus_time_plus_observed_sky_log_overlap_val_selected",
    ]
    table = "\n".join([line_for(v) for v in variants if line_for(v)])
    top_md = top_pairs.head(10).to_markdown(index=False)
    text = f"""# Liao/GW-LMC-prior ET3 robustness experiment

## Scope

This is a robustness experiment using published GW-LMC/Liao ET BBH source, lens
and image parameter tables.  It is not the original ET3 main catalog and does
not replace the main result.

The existing SIS/PM directory names are retained only for compatibility:

- `SIS_data_0222`: GW-LMC smooth/non-subhalo lens realizations.
- `PM_data_0222`: GW-LMC subhalo-flagged lens realizations.

## Data

- match root: `{summary['match_root']}`
- output root: `{summary['out_dir']}`
- mode: `{summary['mode']}`
- epochs: {summary['epochs']}
- catalog total in test split: {summary['catalog_total']}
- valid lensed-image queries: {summary['query_total']}
- waveform backbone: InceptionTime
- encoder source: `{summary.get('encoder_source', 'unknown')}`
- observed sky: ET three-arm SNR-dependent A90 surrogate
- time-delay prior: GW-LMC ET BBH detected-image delay LR

## Overall Metrics

| variant | R@1 | R@5 | R@10 | R@50 | median rank |
|---|---:|---:|---:|---:|---:|
{table}

## Selected Weights

```json
{json.dumps(weights, indent=2)}
```

## Top Pairs by Final Score

{top_md}

## Interpretation

This run is a transfer robustness evaluation: the catalog is generated from a
published GW-LMC/Liao parameter distribution, while the waveform encoder is the
existing ET3 main encoder unless `encoder_source` says it was retrained.  Because
the GW-LMC table does not provide a native SIS/PM split, smooth and
subhalo-flagged lens realizations are used as compatibility slots and should be
described as such.  If waveform-only transfer is weak, the result should be read
as evidence of prior/domain shift rather than as a failure of the full method
after domain-matched retraining.
"""
    (out_dir / "liao_gwlmc_prior_experiment_report_cn.md").write_text(text, encoding="utf-8")


def plot_summary(out_dir: Path, result_df: pd.DataFrame) -> None:
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
        "font.size": 11,
    })
    variants = [
        "waveform_only",
        "liao_time_lr_only",
        "observed_sky_log_overlap_only",
        "time_plus_observed_sky_log_overlap_val_selected",
        "waveform_plus_time_plus_observed_sky_log_overlap_val_selected",
    ]
    labels = ["waveform", "time", "sky", "time+sky", "waveform+time+sky"]
    sub = result_df[(result_df["subset"] == "overall") & (result_df["variant"].isin(variants))].copy()
    sub["variant"] = pd.Categorical(sub["variant"], categories=variants, ordered=True)
    sub = sub.sort_values("variant")
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6), constrained_layout=True)
    axes[0].bar(labels[: len(sub)], sub["r@10"].to_numpy(), color=["#767676", "#4C78A8", "#59A14F", "#F28E2B", "#B07AA1"])
    axes[0].set_ylabel("Recall@10")
    axes[0].set_ylim(0, 1.02)
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].set_title("Companion Retrieval")
    axes[1].bar(labels[: len(sub)], sub["r@1"].to_numpy(), color=["#767676", "#4C78A8", "#59A14F", "#F28E2B", "#B07AA1"])
    axes[1].set_ylabel("Recall@1")
    axes[1].set_ylim(0, 1.02)
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].set_title("First-Rank Retrieval")
    fig.savefig(out_dir / "fig_liao_gwlmc_prior_summary.pdf")
    fig.savefig(out_dir / "fig_liao_gwlmc_prior_summary.png", dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--match-root", type=Path, default=Path("/root/autodl-tmp/gw-catalog/data_generation/liao_gwlmc_prior_20260705_matchroot"))
    parser.add_argument("--out-dir", type=Path, default=Path("/root/autodl-tmp/gw-catalog/runs/liao_gwlmc_prior_et3_20260705"))
    parser.add_argument("--mode", choices=["noisy", "pure"], default="noisy")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--encoder-dir", type=Path, default=None, help="Optional existing ET3 encoder directory for transfer robustness evaluation.")
    parser.add_argument("--backbone", type=str, default="inceptiontime")
    parser.add_argument("--target-len", type=int, default=8192)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--torch-threads", type=int, default=16)
    args = parser.parse_args()
    torch.set_num_threads(int(args.torch_threads))

    t0 = time.perf_counter()
    patch_roots(args.match_root)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    encoder_dir = args.out_dir / f"et3_{args.mode}_mixed_gwlmc_prior_ep{args.epochs}"
    cfg = make_cfg(
        args.mode,
        encoder_dir,
        args.epochs,
        backbone=args.backbone,
        target_len=args.target_len,
        stride=args.stride,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
    )
    loaded, train_info, encoder_source = split_all(cfg, args.mode, args.encoder_dir)
    train_ds, train_raw, train_time, train_gt, train_emb, train_scores = loaded["train"]
    val_ds, val_raw, val_time, val_gt, val_emb, val_scores = loaded["val"]
    test_ds, test_raw, test_time, test_gt, test_emb, test_scores = loaded["test"]

    prior = liao.fit_time_lr_from_liao("ET3", val_time, val_gt)
    val_sky = liao.make_observed_sky("ET3", val_raw, val_time, seed=202607051)
    test_sky = liao.make_observed_sky("ET3", test_raw, test_time, seed=202607052)
    val_sky.to_csv(args.out_dir / "val_observed_sky_audit.csv", index=False)
    test_sky.to_csv(args.out_dir / "test_observed_sky_audit.csv", index=False)
    val_sky_features = observed_sky_pair_features(val_sky, chunk_rows=liao.CHUNK_ROWS)
    test_sky_features = observed_sky_pair_features(test_sky, chunk_rows=liao.CHUNK_ROWS)

    val_components = {
        "waveform": liao.row_z(val_scores),
        "liao_time_lr": liao.row_z(liao.time_lr_score_matrix(val_time, prior)),
        "sky_step": liao.row_z(val_sky_features["sky_step_weight"]),
        "sky_log_overlap": liao.row_z(val_sky_features["sky_log_overlap"]),
    }
    test_components = {
        "waveform": liao.row_z(test_scores),
        "liao_time_lr": liao.row_z(liao.time_lr_score_matrix(test_time, prior)),
        "sky_step": liao.row_z(test_sky_features["sky_step_weight"]),
        "sky_log_overlap": liao.row_z(test_sky_features["sky_log_overlap"]),
    }

    diag = {
        "detector": "ET3",
        "data_mode": args.mode,
        "catalog": "liao_gwlmc_prior_smooth_subhalo_unlensed_full_catalog",
        "candidate_kind": "full_catalog",
        "query_total": int(len([x for x in test_ds.meta if x.get("tag") in {"L1", "L2"}])),
        **fresh.full_catalog_composition(test_ds.meta),
        "epochs": int(args.epochs),
        "backbone": cfg.backbone,
        "preprocess": cfg.preprocess,
        "liao_label": prior["liao_label"],
        "liao_delay_count": prior["liao_delay_count"],
        "liao_delay_median_days": prior["liao_delay_median_days"],
        "liao_delay_p90_days": prior["liao_delay_p90_days"],
        "observed_sky_label": liao.OBSERVED_SKY_CONFIG["ET3"]["label"],
        "test_a90_median_deg2": float(np.median(test_sky["sky_area90_deg2"])),
        "test_a90_p90_deg2": float(np.percentile(test_sky["sky_area90_deg2"], 90)),
    }

    rows: list[dict] = []
    for key, variant in [
        ("waveform", "waveform_only"),
        ("liao_time_lr", "liao_time_lr_only"),
        ("sky_step", "observed_sky_step_only"),
        ("sky_log_overlap", "observed_sky_log_overlap_only"),
    ]:
        rows.extend(flatten_rows("single_channel", variant, row_metrics(test_components[key], test_gt, test_ds.meta), diag))

    def eval_val(score: np.ndarray) -> dict[str, dict]:
        return row_metrics(score, val_gt, val_ds.meta)

    ts_step_lams, ts_step_val = select_best_weighted_lambdas(
        val_components, ["sky_step"], GRID, eval_val, base_key="liao_time_lr"
    )
    ts_step = test_components["liao_time_lr"] + ts_step_lams["sky_step"] * test_components["sky_step"]
    np.fill_diagonal(ts_step, -np.inf)
    rows.extend(flatten_rows(
        "validation_selected_weighted_sum",
        "time_plus_observed_sky_step_val_selected",
        row_metrics(ts_step, test_gt, test_ds.meta),
        diag,
        {"lambda_sky_step": ts_step_lams["sky_step"], "val_selected_r@10": ts_step_val["overall"]["r@10"]},
    ))

    ts_log_lams, ts_log_val = select_best_weighted_lambdas(
        val_components, ["sky_log_overlap"], GRID, eval_val, base_key="liao_time_lr"
    )
    ts_log = test_components["liao_time_lr"] + ts_log_lams["sky_log_overlap"] * test_components["sky_log_overlap"]
    np.fill_diagonal(ts_log, -np.inf)
    rows.extend(flatten_rows(
        "validation_selected_weighted_sum",
        "time_plus_observed_sky_log_overlap_val_selected",
        row_metrics(ts_log, test_gt, test_ds.meta),
        diag,
        {"lambda_sky_log_overlap": ts_log_lams["sky_log_overlap"], "val_selected_r@10": ts_log_val["overall"]["r@10"]},
    ))

    wts_step_lams, wts_step_val = select_best_weighted_lambdas(
        val_components, ["liao_time_lr", "sky_step"], GRID, eval_val, base_key="waveform"
    )
    wts_step = test_components["waveform"].copy()
    for k, v in wts_step_lams.items():
        wts_step += v * test_components[k]
    np.fill_diagonal(wts_step, -np.inf)
    rows.extend(flatten_rows(
        "validation_selected_weighted_sum",
        "waveform_plus_time_plus_observed_sky_step_val_selected",
        row_metrics(wts_step, test_gt, test_ds.meta),
        diag,
        {f"lambda_{k}": v for k, v in wts_step_lams.items()} | {"val_selected_r@10": wts_step_val["overall"]["r@10"]},
    ))

    wts_log_lams, wts_log_val = select_best_weighted_lambdas(
        val_components, ["liao_time_lr", "sky_log_overlap"], GRID, eval_val, base_key="waveform"
    )
    wts_log = test_components["waveform"].copy()
    for k, v in wts_log_lams.items():
        wts_log += v * test_components[k]
    np.fill_diagonal(wts_log, -np.inf)
    rows.extend(flatten_rows(
        "validation_selected_weighted_sum",
        "waveform_plus_time_plus_observed_sky_log_overlap_val_selected",
        row_metrics(wts_log, test_gt, test_ds.meta),
        diag,
        {f"lambda_{k}": v for k, v in wts_log_lams.items()} | {"val_selected_r@10": wts_log_val["overall"]["r@10"]},
    ))

    result_df = pd.DataFrame(rows)
    result_df.to_csv(args.out_dir / "liao_gwlmc_prior_metrics.csv", index=False)
    weights = {
        "time_plus_sky_step": ts_step_lams,
        "time_plus_sky_log_overlap": ts_log_lams,
        "waveform_plus_time_plus_sky_step": wts_step_lams,
        "waveform_plus_time_plus_sky_log_overlap": wts_log_lams,
        "grid": GRID,
        "selection_split": "validation",
    }
    (args.out_dir / "liao_gwlmc_prior_selected_weights.json").write_text(json.dumps(weights, indent=2), encoding="utf-8")
    top_pairs = top_pair_rows(wts_log, test_ds.meta, test_gt, n_top=50)
    top_pairs.to_csv(args.out_dir / "liao_gwlmc_prior_top_pairs_waveform_time_sky.csv", index=False)

    summary = {
        "experiment": "liao_gwlmc_prior_et3_waveform_time_observed_sky",
        "match_root": str(args.match_root),
        "out_dir": str(args.out_dir),
        "mode": args.mode,
        "epochs": int(args.epochs),
        "backbone": cfg.backbone,
        "target_len": int(cfg.target_len),
        "stride": int(cfg.stride),
        "batch_size": int(cfg.batch_size),
        "eval_batch_size": int(cfg.eval_batch_size),
        "num_workers": int(cfg.num_workers),
        "torch_threads": int(args.torch_threads),
        "encoder_source": encoder_source,
        "catalog_total": int(len(test_gt)),
        "query_total": int(len([x for x in test_ds.meta if x.get("tag") in {"L1", "L2"}])),
        "train_s": float(train_info.get("train_s", np.nan)),
        "elapsed_s": float(time.perf_counter() - t0),
        "family_slot_mapping": FAMILY_LABEL,
        "prior": {k: v for k, v in prior.items() if not isinstance(v, np.ndarray)},
        "selected_weights": weights,
    }
    (args.out_dir / "liao_gwlmc_prior_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plot_summary(args.out_dir, result_df)
    write_report(args.out_dir, summary, result_df, weights, top_pairs)
    print(result_df.to_string(index=False), flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
