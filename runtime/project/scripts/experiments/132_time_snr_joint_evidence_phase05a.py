#!/usr/bin/env python3
"""Source-group-weighted Phase 0.5a for the time/SNR study.

This bounded correction reads the frozen Phase 0.5 artifacts, assigns every
GW-LMC source group total density weight one, and recomputes only the
one-dimensional time likelihood-ratio baselines. It does not fit a two-
dimensional density, rerank the real catalog, modify the paper, or overwrite
Phase 0.5/v9.3.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


REPO = Path(__file__).resolve().parents[2]
BASE = REPO / "results/time_snr_joint_evidence_phase05_20260807"
DEFAULT_OUTPUT = REPO / "results/time_snr_joint_evidence_phase05a_20260807"
SEEDS = (202607241, 202607242, 202607243)
DEPLOYMENTS = ("gwtc3", "gwtc4")


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


phase05 = load_module(
    REPO / "scripts/experiments/131_time_snr_joint_evidence_phase05.py",
    "phase05_source_weight_reference",
)
physical = phase05.physical
v7 = phase05.v7


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=BASE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(phase05.json_ready(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def weighted_density_grid(
    values: np.ndarray,
    grid: np.ndarray,
    weights: np.ndarray,
    bandwidth_scale: float = 1.0,
) -> tuple[np.ndarray, float]:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values = values[valid]
    weights = weights[valid]
    if len(values) < 2 or not np.isfinite(values).all():
        raise RuntimeError("Insufficient finite lens-delay samples for weighted KDE")
    kde = stats.gaussian_kde(values, weights=weights)
    kde.set_bandwidth(kde.factor * float(bandwidth_scale))
    density = np.maximum(kde(grid), np.finfo(np.float64).tiny)
    density /= np.trapezoid(density, grid)
    effective_n = float(weights.sum() ** 2 / np.square(weights).sum())
    return density, effective_n


def source_group_weights(dev: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    frame = dev.copy()
    multiplicity = frame.groupby("global_source_group_id").size().rename("source_group_multiplicity")
    frame = frame.join(multiplicity, on="global_source_group_id")
    frame["source_group_density_weight"] = 1.0 / frame["source_group_multiplicity"].astype(float)
    group_audit = (
        frame.groupby("global_source_group_id", as_index=False)
        .agg(
            multiplicity=("source_group_multiplicity", "first"),
            group_weight_sum=("source_group_density_weight", "sum"),
            families=("family", lambda values: ",".join(sorted(set(map(str, values))))),
            lens_realizations=("global_lens_system_id", "nunique"),
        )
    )
    totals = {
        "lens_rows": int(len(frame)),
        "unique_source_groups": int(frame["global_source_group_id"].nunique()),
        "duplicated_source_groups": int((group_audit["multiplicity"] > 1).sum()),
        "maximum_source_group_multiplicity": int(group_audit["multiplicity"].max()),
        "row_weight_sum": float(frame["source_group_density_weight"].sum()),
        "maximum_abs_group_weight_error": float(np.max(np.abs(group_audit["group_weight_sum"] - 1.0))),
        "all_source_groups_total_weight_one": bool(np.allclose(group_audit["group_weight_sum"], 1.0, atol=1e-12, rtol=0)),
        "weighted_smooth_non_subhalo_mass": float(frame.loc[frame["family"] == "SIS", "source_group_density_weight"].sum()),
        "weighted_subhalo_present_mass": float(frame.loc[frame["family"] == "PM", "source_group_density_weight"].sum()),
    }
    return frame, group_audit, totals


def build_weighted_calibration(
    old_calibration: dict[str, Any],
    weighted_dev: pd.DataFrame,
) -> dict[str, Any]:
    delays = weighted_dev["delay_days"].to_numpy(dtype=np.float64)
    weights = weighted_dev["source_group_density_weight"].to_numpy(dtype=np.float64)
    lens_log = np.log10(np.maximum(delays, 1e-12))
    grid = np.asarray(old_calibration["log10_delay_grid"], dtype=np.float64)
    p_null = np.asarray(old_calibration["p_null_log10"], dtype=np.float64)
    p_lens, effective_n = weighted_density_grid(lens_log, grid, weights, bandwidth_scale=1.0)
    log_lr = np.log(p_lens) - np.log(np.maximum(p_null, np.finfo(np.float64).tiny))
    return {
        "log10_delay_grid": grid,
        "p_lens_log10": p_lens,
        "p_null_log10": p_null,
        "log_likelihood_ratio": log_lr,
        "lens_rows": int(len(delays)),
        "lens_source_groups": int(weighted_dev["global_source_group_id"].nunique()),
        "lens_weight_sum": float(weights.sum()),
        "lens_effective_sample_size": effective_n,
        "null_samples": int(old_calibration["null_samples"]),
        "bandwidth_scale": 1.0,
        "estimator": "Gaussian KDE in log10(delta_t_days) with per-row source-group weight 1/m_g",
        "source_group_rule": "For source group g with m_g lens rows, every row receives weight 1/m_g; each source group has total weight one.",
        "null_density_and_grid_frozen_from_phase05": True,
        "validation_or_test_systems_used": False,
        "two_dimensional_density_fitted": False,
    }


def apply_and_evaluate(
    base: Path,
    output: Path,
    deployment: str,
    seed: int,
    protocol: str,
    calibration: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if protocol == "closed_loop":
        source_dir = base / "time" / deployment / f"seed_{seed}"
        validation_source = source_dir / "closed_loop_validation_pairs_corrected_time.parquet"
        test_source = source_dir / "closed_loop_test_pairs_corrected_time.parquet"
    else:
        source_dir = base / "response" / deployment / f"seed_{seed}"
        validation_source = source_dir / "response_validation_pairs_calibrated.parquet"
        test_source = source_dir / "response_heldout_test_pairs_calibrated.parquet"
    validation = pd.read_parquet(validation_source)
    test = pd.read_parquet(test_source)
    for frame in (validation, test):
        frame["time_score"] = physical.apply_time_likelihood_ratio(frame["delta_t_days"], calibration)

    methods, grids = phase05.method_weights(validation)
    target = output / "time" / deployment / f"seed_{seed}" / protocol
    target.mkdir(parents=True, exist_ok=True)
    validation.to_parquet(target / "validation_pairs_source_weighted_time.parquet", index=False)
    test.to_parquet(target / "heldout_test_pairs_source_weighted_time.parquet", index=False)
    write_json(target / "selected_weights.json", methods)
    for name, grid in grids.items():
        grid.to_csv(target / f"weight_grid_{name}.csv", index=False)
    _, retrieval, pair = v7.evaluation_tables(test, methods, deployment, seed)
    label = f"{protocol}_source_group_weighted_phase05a"
    retrieval["protocol"] = label
    pair["protocol"] = label
    retrieval.to_csv(target / "heldout_retrieval_metrics.csv", index=False)
    pair.to_csv(target / "heldout_pair_metrics.csv", index=False)
    return retrieval, pair


def metric_deltas(
    old: pd.DataFrame,
    new: pd.DataFrame,
    old_protocol: str,
    new_protocol: str,
    metrics: list[str],
    keys: list[str],
) -> pd.DataFrame:
    left = old.loc[old["protocol"] == old_protocol, keys + metrics].copy()
    right = new.loc[new["protocol"] == new_protocol, keys + metrics].copy()
    left = left.rename(columns={metric: f"phase05_{metric}" for metric in metrics})
    right = right.rename(columns={metric: f"phase05a_{metric}" for metric in metrics})
    merged = left.merge(right, on=keys, validate="one_to_one")
    for metric in metrics:
        merged[f"delta_{metric}"] = merged[f"phase05a_{metric}"] - merged[f"phase05_{metric}"]
    return merged


def summarize(frame: pd.DataFrame, metrics: list[str], groups: list[str]) -> pd.DataFrame:
    return phase05.summarize_metrics(frame, metrics, groups)


def build_manifest(output: Path) -> None:
    rows = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name not in {"artifact_manifest.csv", "checksums_sha256.txt"}:
            rows.append({"path": str(path.relative_to(output)), "bytes": path.stat().st_size, "sha256": sha256(path)})
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "artifact_manifest.csv", index=False)
    (output / "checksums_sha256.txt").write_text(
        "\n".join(f"{row.sha256}  {row.path}" for row in frame.itertuples(index=False)) + "\n",
        encoding="utf-8",
    )


def make_report(
    output: Path,
    audit: pd.DataFrame,
    retrieval_delta: pd.DataFrame,
    pair_delta: pd.DataFrame,
    decision: dict[str, Any],
) -> None:
    focus = retrieval_delta[
        (retrieval_delta["method"] == "retrieval_three_channel_strict_positive")
        & (retrieval_delta["subset"] == "overall")
    ]
    pair_focus = pair_delta[pair_delta["method"] == "retrieval_three_channel_strict_positive"]
    lines = [
        "# 时间延迟与相对观测强度联合证据 Phase 0.5a 报告",
        "",
        "日期：2026-08-07",
        "",
        "## 1. 唯一修改",
        "",
        "Phase 0.5 对 840 个 lens rows 等权。若同一个 GW-LMC source group 有多个 lens realization，该 source 会被重复计权。Phase 0.5a 对 group g 的每一行设置 w=1/m_g，因此该 source group 的总权重严格为 1。null 样本、KDE grid、bandwidth scale、pair 特征、模型、sky、PE 契约和 held-out 数据全部冻结。",
        "",
        "本轮未拟合二维密度、未重排真实目录、未修改 v9.3 或论文。",
        "",
        "## 2. Source-group 权重审计",
        "",
        audit.to_markdown(index=False, floatfmt=".8g"),
        "",
        "## 3. 三通道检索差值",
        "",
        focus.to_markdown(index=False, floatfmt=".8g"),
        "",
        "## 4. Pair-level 差值",
        "",
        pair_focus.to_markdown(index=False, floatfmt=".8g"),
        "",
        "## 5. 冻结 PE 契约",
        "",
        "Phase 0.5a 直接复制并读取 Phase 0.5 的事件级 PE contract；没有重新动态选择 posterior group。strict scope 仍为 GWTC-3 52 个和 O4a 74 个事件。",
        "",
        "## 6. 决定",
        "",
        f"**{decision['decision']}**",
        "",
        pd.DataFrame([{"check": key, "passed": value} for key, value in decision["checks"].items()]).to_markdown(index=False),
        "",
        "该决定只表示 Phase 0.5a 已满足预注册修正，可以由作者明确授权 bounded Phase 1；脚本没有自动启动 Phase 1。",
    ]
    (output / "phase05a_report_cn.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    base = args.base.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite Phase 0.5a output: {output}")
    required = [
        base / "contract/pe_network_optimal_snr_event_contract.csv",
        base / "contract/pe_network_optimal_snr_contract.json",
        base / "splits/global_system_split_membership.parquet",
        base / "time/closed_loop_corrected_time_retrieval_per_seed.csv",
        base / "response/response_heldout_retrieval_per_seed.csv",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing frozen Phase 0.5 inputs: {missing}")
    started = time.time()
    output.mkdir(parents=True)
    (output / "contract").mkdir()
    shutil.copy2(required[0], output / "contract" / required[0].name)
    shutil.copy2(required[1], output / "contract" / required[1].name)
    membership = pd.read_parquet(required[2])
    membership.to_parquet(output / "contract/global_system_split_membership_frozen.parquet", index=False)
    pe_hashes = {path.name: sha256(path) for path in required[:2]}
    copied_pe_hashes = {path.name: sha256(output / "contract" / path.name) for path in required[:2]}

    audit_rows: list[dict[str, Any]] = []
    retrieval_rows: list[pd.DataFrame] = []
    pair_rows: list[pd.DataFrame] = []
    for deployment in DEPLOYMENTS:
        for seed in SEEDS:
            dev = membership[
                (membership["deployment"] == deployment)
                & (membership["seed"] == seed)
                & (membership["corrected_split"] == "train")
                & (membership["role"] == "lensed")
            ].drop_duplicates("global_lens_system_id")
            weighted_dev, group_audit, totals = source_group_weights(dev)
            seed_dir = output / "time" / deployment / f"seed_{seed}"
            seed_dir.mkdir(parents=True, exist_ok=True)
            weighted_dev.to_parquet(seed_dir / "density_development_source_group_weights.parquet", index=False)
            group_audit.to_csv(seed_dir / "source_group_weight_audit.csv", index=False)
            old_path = base / "time" / deployment / f"seed_{seed}/time_likelihood_ratio.json"
            old_calibration = json.loads(old_path.read_text(encoding="utf-8"))
            calibration = build_weighted_calibration(old_calibration, weighted_dev)
            write_json(seed_dir / "time_likelihood_ratio_source_group_weighted.json", calibration)
            audit_rows.append({"deployment": deployment, "seed": seed, **totals, "weighted_kde_effective_n": calibration["lens_effective_sample_size"]})
            for protocol in ("closed_loop", "response_derived"):
                retrieval, pair = apply_and_evaluate(base, output, deployment, seed, protocol, calibration)
                retrieval_rows.append(retrieval)
                pair_rows.append(pair)

    audit = pd.DataFrame(audit_rows)
    audit.to_csv(output / "source_group_weight_audit_summary.csv", index=False)
    new_retrieval = pd.concat(retrieval_rows, ignore_index=True)
    new_pair = pd.concat(pair_rows, ignore_index=True)
    new_retrieval.to_csv(output / "heldout_retrieval_metrics_per_seed.csv", index=False)
    new_pair.to_csv(output / "heldout_pair_metrics_per_seed.csv", index=False)

    old_retrieval = pd.concat(
        [
            pd.read_csv(base / "time/closed_loop_corrected_time_retrieval_per_seed.csv"),
            pd.read_csv(base / "response/response_heldout_retrieval_per_seed.csv"),
        ],
        ignore_index=True,
    )
    old_pair = pd.concat(
        [
            pd.read_csv(base / "time/closed_loop_corrected_time_pair_metrics_per_seed.csv"),
            pd.read_csv(base / "response/response_heldout_pair_metrics_per_seed.csv"),
        ],
        ignore_index=True,
    )
    retrieval_delta_frames = []
    pair_delta_frames = []
    protocol_pairs = [
        ("closed_loop_corrected_time_phase05", "closed_loop_source_group_weighted_phase05a"),
        ("response_derived_phase05", "response_derived_source_group_weighted_phase05a"),
    ]
    for old_protocol, new_protocol in protocol_pairs:
        rdelta = metric_deltas(
            old_retrieval,
            new_retrieval,
            old_protocol,
            new_protocol,
            ["r_at_1", "r_at_10", "median_rank", "n_queries"],
            ["deployment", "seed", "method", "subset"],
        )
        rdelta.insert(0, "protocol", new_protocol)
        retrieval_delta_frames.append(rdelta)
        pdelta = metric_deltas(
            old_pair,
            new_pair,
            old_protocol,
            new_protocol,
            ["average_precision", "false_at_recall_0p5", "false_at_recall_0p9"],
            ["deployment", "seed", "method"],
        )
        pdelta.insert(0, "protocol", new_protocol)
        pair_delta_frames.append(pdelta)
    retrieval_delta = pd.concat(retrieval_delta_frames, ignore_index=True)
    pair_delta = pd.concat(pair_delta_frames, ignore_index=True)
    retrieval_delta.to_csv(output / "phase05_vs_phase05a_retrieval_delta_per_seed.csv", index=False)
    pair_delta.to_csv(output / "phase05_vs_phase05a_pair_delta_per_seed.csv", index=False)

    retrieval_summary = summarize(
        new_retrieval,
        ["r_at_1", "r_at_10", "median_rank"],
        ["protocol", "deployment", "method", "subset"],
    )
    pair_summary = summarize(
        new_pair,
        ["average_precision", "false_at_recall_0p5", "false_at_recall_0p9"],
        ["protocol", "deployment", "method"],
    )
    retrieval_summary.to_csv(output / "heldout_retrieval_metrics_summary.csv", index=False)
    pair_summary.to_csv(output / "heldout_pair_metrics_summary.csv", index=False)

    split_summary = pd.read_csv(base / "splits/global_system_split_summary.csv")
    split_isolated = bool(
        (split_summary[["train_validation_source_intersection", "train_test_source_intersection", "validation_test_source_intersection"]] == 0)
        .all()
        .all()
    )
    source1568 = membership[
        (membership["deployment"] == "gwtc4")
        & (membership["seed"] == 202607241)
        & (membership["gwlmc_event_id"] == 1568)
    ]
    source1568_pass = bool(
        len(source1568) == 2
        and (source1568["corrected_split"] == "train").sum() == 1
        and (source1568["corrected_split"] == "excluded").sum() == 1
        and (source1568["correction_reason"] == "source_group_present_in_train").sum() == 1
    )
    query_check = new_retrieval[
        (new_retrieval["protocol"] == "closed_loop_source_group_weighted_phase05a")
        & (new_retrieval["deployment"] == "gwtc4")
        & (new_retrieval["seed"] == 202607241)
        & (new_retrieval["method"] == "waveform_only")
        & (new_retrieval["subset"] == "overall")
    ]
    checks = {
        "every_source_group_total_density_weight_one": bool(audit["all_source_groups_total_weight_one"].all()),
        "global_source_split_intersections_zero": split_isolated,
        "o4a_seed202607241_source1568_leaked_test_realization_removed": source1568_pass,
        "o4a_seed202607241_test_denominator_is_358_queries": bool(len(query_check) == 1 and int(query_check.iloc[0]["n_queries"]) == 358),
        "pe_event_contract_copied_byte_exactly": pe_hashes == copied_pe_hashes,
        "all_metrics_finite": bool(
            np.isfinite(new_retrieval[["r_at_1", "r_at_10", "median_rank", "n_queries"]].to_numpy(dtype=float)).all()
            and np.isfinite(new_pair[["average_precision", "false_at_recall_0p5", "false_at_recall_0p9"]].to_numpy(dtype=float)).all()
        ),
    }
    passed = all(checks.values())
    decision = {
        "phase": "Phase 0.5a",
        "checks": checks,
        "decision": "PHASE05A_PASSED_READY_FOR_EXPLICIT_BOUNDED_PHASE1_AUTHORIZATION" if passed else "PHASE05A_NO_GO_REMEDIATE",
        "phase1_started": False,
        "two_dimensional_density_fitted": False,
        "real_catalog_reranked": False,
        "paper_modified": False,
        "v93_overwritten": False,
        "pe_contract_source": str(base / "contract/pe_network_optimal_snr_event_contract.csv"),
        "pe_contract_sha256": pe_hashes,
    }
    write_json(output / "phase05a_go_no_go.json", decision)
    write_json(
        output / "analysis_contract_phase05a.json",
        {
            "only_scientific_change": "Per-row lens-density weight is 1/m_g so every global source group has total weight one.",
            "frozen_from_phase05": ["PE event contract", "null density", "KDE grid", "bandwidth scale", "waveform", "sky", "pair features", "splits", "held-out systems"],
            **{key: decision[key] for key in ["phase1_started", "two_dimensional_density_fitted", "real_catalog_reranked", "paper_modified", "v93_overwritten"]},
        },
    )
    make_report(output, audit, retrieval_delta, pair_delta, decision)
    script_dir = output / "scripts/experiments"
    script_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__).resolve(), script_dir / Path(__file__).name)
    shutil.copy2(REPO / "scripts/experiments/131_time_snr_joint_evidence_phase05.py", script_dir / "131_time_snr_joint_evidence_phase05.py")
    write_json(
        output / "run_summary.json",
        {
            "status": "complete",
            "decision": decision["decision"],
            "elapsed_seconds": time.time() - started,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    build_manifest(output)
    print(json.dumps(decision, indent=2, ensure_ascii=False))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
