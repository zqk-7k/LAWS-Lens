#!/usr/bin/env python3
"""Validation-only simplex refusion for the frozen ET/GWTC evidence channels.

This experiment does not retrain an encoder or alter any waveform, time-delay,
or sky score.  It replaces the legacy coarse Cartesian weight grid with the
normalized simplex

    w_waveform + w_time + w_sky = 1,  w_channel >= 0

at a default step of 0.05.  The primary search is unconstrained, so validation
may assign zero weight to any channel.  Strict-positive, equal-weight,
waveform+time, and time+sky variants are retained as sensitivity checks.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

ET_SCRIPT = REPO / "scripts/experiments/124_et3_moderate_observed_sky.py"
GWTC_SCRIPT = REPO / "scripts/experiments/121_gwtc_sky_v81_pipeline.py"
ET_CURRENT = REPO / "results/et3_moderate_sky_gwtc3_topb_20260726"
GWTC_CURRENT = REPO / "results/unified_sky_v81_20260725"
DEFAULT_OUTPUT = REPO / "results/simplex_weight_refusion_20260726"
ET_SEEDS = (202607251, 202607252, 202607253, 202607254, 202607255)
GWTC_SEEDS = (202607241, 202607242, 202607243)
CHANNELS = ("waveform", "time", "sky")


def load_module(path: Path, name: str):
    specification = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(specification)
    assert specification.loader is not None
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


etmod = load_module(ET_SCRIPT, "et3_moderate_v82_for_simplex")
gwtcmod = load_module(GWTC_SCRIPT, "gwtc_v81_for_simplex")
etbase = etmod.base
etv7 = etmod.etv7
gwtcv7 = gwtcmod.v7


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=json_default)
        + "\n",
        encoding="utf-8",
    )


def json_default(value: Any):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value)!r}")


def simplex_weight_rows(step: float) -> list[dict[str, float]]:
    units = int(round(1.0 / step))
    if not np.isclose(units * step, 1.0):
        raise ValueError("The simplex step must divide 1 exactly")
    rows: list[dict[str, float]] = []
    for waveform_units in range(units + 1):
        for time_units in range(units - waveform_units + 1):
            sky_units = units - waveform_units - time_units
            rows.append(
                {
                    "waveform": waveform_units / units,
                    "time": time_units / units,
                    "sky": sky_units / units,
                }
            )
    expected = (units + 1) * (units + 2) // 2
    if len(rows) != expected:
        raise RuntimeError("Simplex enumeration is incomplete")
    return rows


def mode_mask(frame: pd.DataFrame, mode: str) -> np.ndarray:
    w = frame["waveform"].to_numpy(dtype=np.float64)
    t = frame["time"].to_numpy(dtype=np.float64)
    s = frame["sky"].to_numpy(dtype=np.float64)
    positive = 1e-12
    if mode == "unconstrained":
        return np.ones(len(frame), dtype=bool)
    if mode == "strict_positive":
        return (w > positive) & (t > positive) & (s > positive)
    if mode == "waveform_time":
        return (w > positive) & (t > positive) & np.isclose(s, 0.0)
    if mode == "time_sky":
        return np.isclose(w, 0.0) & (t > positive) & (s > positive)
    if mode == "waveform_sky":
        return (w > positive) & np.isclose(t, 0.0) & (s > positive)
    raise ValueError(f"Unknown mode: {mode}")


def weight_dict(row: pd.Series | dict[str, Any]) -> dict[str, float]:
    return {name: float(row[name]) for name in CHANNELS}


def et_simplex_grid(
    components: dict[str, np.ndarray],
    catalog,
    seed: int,
    weight_rows: list[dict[str, float]],
    false_count: int = 500_000,
) -> tuple[pd.DataFrame, dict[str, dict[str, float]], dict[str, pd.DataFrame]]:
    """Evaluate exact validation retrieval for all ET simplex weights."""
    query = np.flatnonzero(catalog.partner >= 0).astype(np.int64)
    truth = catalog.partner[query].astype(np.int64)
    families = np.asarray([catalog.meta[int(index)]["family"] for index in query])
    component_tensor = np.stack(
        [
            np.nan_to_num(
                np.asarray(components[name][query], dtype=np.float32),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            for name in CHANNELS
        ],
        axis=0,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensor = torch.from_numpy(component_tensor).to(device)
    query_device = torch.from_numpy(query).to(device)
    truth_device = torch.from_numpy(truth).to(device)
    local_device = torch.arange(len(query), device=device)
    rows: list[dict[str, Any]] = []
    batch_size = 12 if device.type == "cuda" else 1
    for start in range(0, len(weight_rows), batch_size):
        batch = weight_rows[start : start + batch_size]
        weights = torch.tensor(
            [[row[name] for name in CHANNELS] for row in batch],
            dtype=torch.float32,
            device=device,
        )
        score = torch.einsum("bc,cqn->bqn", weights, tensor)
        true_score = score[:, local_device, truth_device]
        counts = torch.sum(score > true_score[:, :, None], dim=-1)
        self_score = score[:, local_device, query_device]
        counts -= (self_score > true_score).to(counts.dtype)
        ranks_batch = (1 + counts).cpu().numpy()
        for offset, ranks in enumerate(ranks_batch):
            family_rows = []
            for family in etv7.FAMILIES:
                keep = families == family
                family_rows.append(
                    {
                        "r_at_1": float(np.mean(ranks[keep] <= 1)),
                        "r_at_10": float(np.mean(ranks[keep] <= 10)),
                    }
                )
            rows.append(
                {
                    **batch[offset],
                    "overall_r_at_1": float(np.mean(ranks <= 1)),
                    "overall_r_at_10": float(np.mean(ranks <= 10)),
                    "macro_r_at_1": float(
                        np.mean([row["r_at_1"] for row in family_rows])
                    ),
                    "macro_r_at_10": float(
                        np.mean([row["r_at_10"] for row in family_rows])
                    ),
                    "min_family_r_at_10": float(
                        min(row["r_at_10"] for row in family_rows)
                    ),
                    "median_rank": float(np.median(ranks)),
                    "sampled_average_precision": np.nan,
                    "sampled_precision_at_recall_0p5": np.nan,
                    "weight_l2": float(
                        math.sqrt(sum(value * value for value in batch[offset].values()))
                    ),
                }
            )
        del score, true_score, counts, self_score
    del tensor, component_tensor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    grid = pd.DataFrame(rows)
    false_i, false_j = etv7.sample_false_pairs(
        catalog.partner,
        false_count,
        seed,
    )
    modes = (
        "unconstrained",
        "strict_positive",
        "waveform_time",
        "time_sky",
        "waveform_sky",
    )
    ap_indices: set[int] = set()
    for mode in modes:
        part = grid.loc[mode_mask(grid, mode)]
        best_macro = float(part["macro_r_at_10"].max())
        part = part[np.isclose(part["macro_r_at_10"], best_macro)]
        best_family = float(part["min_family_r_at_10"].max())
        part = part[np.isclose(part["min_family_r_at_10"], best_family)]
        ap_indices.update(int(index) for index in part.index)

    for index in sorted(ap_indices):
        weights = weight_dict(grid.loc[index])
        score = etv7.combine(components, weights)
        metrics = etv7.sampled_pair_candidate_metrics(
            score,
            catalog.partner,
            false_i,
            false_j,
        )
        for name, value in metrics.items():
            grid.loc[index, name] = value

    selected: dict[str, dict[str, float]] = {}
    mode_grids: dict[str, pd.DataFrame] = {}
    for mode in modes:
        part = grid.loc[mode_mask(grid, mode)].copy()
        part["_ap_sort"] = part["sampled_average_precision"].fillna(-np.inf)
        part = (
            part.sort_values(
                [
                    "macro_r_at_10",
                    "min_family_r_at_10",
                    "_ap_sort",
                    "sampled_precision_at_recall_0p5",
                    "macro_r_at_1",
                    "weight_l2",
                ],
                ascending=[False, False, False, False, False, True],
            )
            .drop(columns="_ap_sort")
            .reset_index(drop=True)
        )
        selected[mode] = weight_dict(part.iloc[0])
        mode_grids[mode] = part
    return grid, selected, mode_grids


def evaluate_et_seed(
    seed: int,
    output_root: Path,
    weight_rows: list[dict[str, float]],
    full_pair_metrics: bool,
) -> dict[str, Any]:
    output_seed = output_root / "et3" / f"seed_{seed}"
    output_seed.mkdir(parents=True, exist_ok=True)
    source_seed = etbase.V7_ET / f"seed_{seed}"
    validation_events = pd.read_parquet(
        source_seed / "validation_event_catalog.parquet"
    )
    test_events = pd.read_parquet(source_seed / "test_event_catalog.parquet")
    validation = etbase.build_catalog(validation_events)
    test = etbase.build_catalog(test_events)
    val_waveform, val_time = etbase.load_waveform_and_time(
        seed, "validation", validation
    )
    test_waveform, test_time = etbase.load_waveform_and_time(seed, "test", test)

    scenario = etmod.SCENARIOS["moderate"]
    val_parameters = etmod.build_event_parameters(
        validation_events, scenario, seed + 82000
    )
    temperature, temperature_audit = etmod.validation_temperature(val_parameters)
    val_sky, _ = etmod.build_split_sky(
        "validation",
        validation_events,
        scenario,
        seed + 82000,
        temperature,
        output_seed,
    )
    test_sky, _ = etmod.build_split_sky(
        "test",
        test_events,
        scenario,
        seed + 83000,
        temperature,
        output_seed,
    )
    val_components = {
        "waveform": val_waveform,
        "time": val_time,
        "sky": val_sky,
    }
    test_components = {
        "waveform": test_waveform,
        "time": test_time,
        "sky": test_sky,
    }
    full_grid, selected, mode_grids = et_simplex_grid(
        val_components,
        validation,
        seed + 85000,
        weight_rows,
    )
    full_grid.to_csv(output_seed / "simplex_validation_grid.csv", index=False)
    for mode, frame in mode_grids.items():
        frame.to_csv(output_seed / f"simplex_validation_grid_{mode}.csv", index=False)

    methods = {
        "waveform_only": {"waveform": 1.0, "time": 0.0, "sky": 0.0},
        "time_only": {"waveform": 0.0, "time": 1.0, "sky": 0.0},
        "sky_only": {"waveform": 0.0, "time": 0.0, "sky": 1.0},
        "waveform_time_validation_selected": selected["waveform_time"],
        "time_sky_validation_selected": selected["time_sky"],
        "waveform_sky_validation_selected": selected["waveform_sky"],
        "three_channel_unconstrained": selected["unconstrained"],
        "three_channel_strict_positive": selected["strict_positive"],
        "three_channel_equal_evidence": {
            "waveform": 1.0 / 3.0,
            "time": 1.0 / 3.0,
            "sky": 1.0 / 3.0,
        },
    }
    write_json(
        output_seed / "selected_simplex_weights.json",
        {
            "deployment": "ET3-moderate",
            "seed": seed,
            "simplex_step": 0.05,
            "n_weight_combinations": len(weight_rows),
            "selection_split": "validation systems only",
            "test_used_for_selection": False,
            "real_catalog_or_pe_used_for_selection": False,
            "scenario": asdict(scenario),
            "posterior_temperature": temperature,
            "temperature_audit": temperature_audit,
            "methods": methods,
        },
    )

    validation_query, validation_metrics, _ = etbase.retrieval_tables(
        val_components,
        validation,
        methods,
        seed,
        "ET3-moderate-simplex-validation",
    )
    query, metrics, _ = etbase.retrieval_tables(
        test_components,
        test,
        methods,
        seed,
        "ET3-moderate-simplex",
    )
    validation_query.to_parquet(
        output_seed / "validation_query_ranks.parquet", index=False
    )
    validation_metrics.to_csv(
        output_seed / "validation_retrieval_metrics.csv", index=False
    )
    query.to_parquet(output_seed / "heldout_test_query_ranks.parquet", index=False)
    metrics.to_csv(output_seed / "heldout_test_retrieval_metrics.csv", index=False)

    old_metrics = pd.read_csv(
        ET_CURRENT
        / "et3/moderate"
        / f"seed_{seed}"
        / "retrieval_metrics_v82.csv"
    )
    for method in ("waveform_only", "time_only", "sky_only"):
        old = old_metrics[
            (old_metrics["method"] == method) & (old_metrics["subset"] == "overall")
        ].iloc[0]
        new = metrics[
            (metrics["method"] == method) & (metrics["subset"] == "overall")
        ].iloc[0]
        if not (
            np.isclose(old["r_at_1"], new["r_at_1"], atol=1e-12)
            and np.isclose(old["r_at_10"], new["r_at_10"], atol=1e-12)
        ):
            raise RuntimeError(
                f"ET seed {seed} frozen-channel reproduction failed for {method}"
            )

    pair_payload: dict[str, Any] = {}
    if full_pair_metrics:
        pair_methods = (
            "waveform_time_validation_selected",
            "three_channel_unconstrained",
        )
        for method in pair_methods:
            score = etv7.combine(test_components, methods[method])
            summary, operating = etv7.full_unordered_pair_metrics(
                score,
                test.partner,
                test.meta,
                aggregation="max",
            )
            summary.update(
                {
                    "deployment": "ET3-moderate-simplex",
                    "seed": seed,
                    "method": method,
                }
            )
            write_json(
                output_seed / f"pair_level_metrics_{method}.json",
                summary,
            )
            operating.to_csv(
                output_seed / f"pair_level_operating_points_{method}.csv",
                index=False,
            )
            pair_payload[method] = summary
            del score

    del (
        val_waveform,
        val_time,
        val_sky,
        test_waveform,
        test_time,
        test_sky,
        val_components,
        test_components,
    )
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "deployment": "ET3",
        "seed": seed,
        "methods": methods,
        "pair_level": pair_payload,
    }


def gwtc_simplex_grid(
    validation: pd.DataFrame,
    weight_rows: list[dict[str, float]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for weights in weight_rows:
        scores = gwtcv7.score_vector(validation, weights)
        rows.append(
            {
                **weights,
                **gwtcv7.retrieval_metrics(validation, scores),
                **gwtcv7.pair_metrics(validation, scores),
                "weight_l2": math.sqrt(
                    sum(weights[name] * weights[name] for name in CHANNELS)
                ),
            }
        )
    return pd.DataFrame(rows)


def select_gwtc_modes(
    grid: pd.DataFrame,
) -> tuple[dict[str, dict[str, float]], dict[str, pd.DataFrame]]:
    selected: dict[str, dict[str, float]] = {}
    mode_grids: dict[str, pd.DataFrame] = {}
    modes = (
        "unconstrained",
        "strict_positive",
        "waveform_time",
        "time_sky",
        "waveform_sky",
    )
    for objective in ("retrieval", "candidate"):
        if objective == "retrieval":
            columns = [
                "macro_r_at_10",
                "min_family_r_at_10",
                "average_precision",
                "precision_at_recall_0p5",
                "macro_r_at_1",
                "weight_l2",
            ]
        else:
            columns = [
                "average_precision",
                "precision_at_recall_0p5",
                "macro_r_at_10",
                "min_family_r_at_10",
                "macro_r_at_1",
                "weight_l2",
            ]
        for mode in modes:
            part = (
                grid.loc[mode_mask(grid, mode)]
                .sort_values(
                    columns,
                    ascending=[False, False, False, False, False, True],
                )
                .reset_index(drop=True)
            )
            key = f"{mode}_{objective}"
            selected[key] = weight_dict(part.iloc[0])
            mode_grids[key] = part
    return selected, mode_grids


def gwtc_methods(selected: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    methods = {
        "waveform_only": {"waveform": 1.0, "time": 0.0, "sky": 0.0},
        "time_only": {"waveform": 0.0, "time": 1.0, "sky": 0.0},
        "sky_only": {"waveform": 0.0, "time": 0.0, "sky": 1.0},
        "three_channel_equal_evidence": {
            "waveform": 1.0 / 3.0,
            "time": 1.0 / 3.0,
            "sky": 1.0 / 3.0,
        },
    }
    for objective in ("retrieval", "candidate"):
        methods[f"waveform_time_{objective}_selected"] = selected[
            f"waveform_time_{objective}"
        ]
        methods[f"time_sky_{objective}_selected"] = selected[
            f"time_sky_{objective}"
        ]
        methods[f"waveform_sky_{objective}_selected"] = selected[
            f"waveform_sky_{objective}"
        ]
        methods[f"{objective}_three_channel_unconstrained"] = selected[
            f"unconstrained_{objective}"
        ]
        methods[f"{objective}_three_channel_strict_positive"] = selected[
            f"strict_positive_{objective}"
        ]
    return methods


def evaluate_gwtc_seed(
    deployment: str,
    seed: int,
    output_root: Path,
    weight_rows: list[dict[str, float]],
) -> dict[str, Any]:
    source_seed = GWTC_CURRENT / deployment / f"seed_{seed}"
    output_seed = output_root / deployment / f"seed_{seed}"
    output_seed.mkdir(parents=True, exist_ok=True)
    validation = pd.read_parquet(
        source_seed / "fusion_validation_pairs_v81.parquet"
    )
    heldout = pd.read_parquet(
        source_seed / "fusion_heldout_test_pairs_v81.parquet"
    )
    grid = gwtc_simplex_grid(validation, weight_rows)
    grid.to_csv(output_seed / "simplex_validation_grid.csv", index=False)
    selected, mode_grids = select_gwtc_modes(grid)
    for name, frame in mode_grids.items():
        frame.to_csv(output_seed / f"simplex_validation_grid_{name}.csv", index=False)
    methods = gwtc_methods(selected)
    write_json(
        output_seed / "selected_simplex_weights.json",
        {
            "deployment": deployment,
            "seed": seed,
            "simplex_step": 0.05,
            "n_weight_combinations": len(weight_rows),
            "selection_split": "synthetic validation systems only",
            "test_used_for_selection": False,
            "real_catalog_or_pe_used_for_selection": False,
            "methods": methods,
        },
    )

    validation_query, validation_metrics, validation_pair = gwtcv7.evaluation_tables(
        validation,
        methods,
        deployment,
        seed,
    )
    test_query, test_metrics, test_pair = gwtcv7.evaluation_tables(
        heldout,
        methods,
        deployment,
        seed,
    )
    validation_query.to_parquet(
        output_seed / "validation_query_ranks.parquet", index=False
    )
    validation_metrics.to_csv(
        output_seed / "validation_retrieval_metrics.csv", index=False
    )
    validation_pair.to_csv(
        output_seed / "validation_pair_level_metrics.csv", index=False
    )
    test_query.to_parquet(
        output_seed / "heldout_test_query_ranks.parquet", index=False
    )
    test_metrics.to_csv(
        output_seed / "heldout_test_retrieval_metrics.csv", index=False
    )
    test_pair.to_csv(
        output_seed / "heldout_test_pair_level_metrics.csv", index=False
    )

    real = pd.read_parquet(
        source_seed / "real_pair_features_unified_sky_v81.parquet"
    )
    real_methods = {
        name: weights
        for name, weights in methods.items()
        if name
        in {
            "waveform_time_candidate_selected",
            "time_sky_candidate_selected",
            "candidate_three_channel_unconstrained",
            "candidate_three_channel_strict_positive",
            "retrieval_three_channel_unconstrained",
        }
    }
    for name, weights in real_methods.items():
        all_catalog = gwtcv7._rank_real_pairs(real, weights, name)
        strict = gwtcv7._rank_real_pairs(
            real[real["strict_h1l1_bbh_pair"]].copy(),
            weights,
            name,
        )
        all_catalog.to_parquet(
            output_seed / f"real_pair_scores_all_{name}.parquet",
            index=False,
        )
        strict.to_parquet(
            output_seed / f"real_pair_scores_strict_{name}.parquet",
            index=False,
        )
        all_catalog.head(100).to_csv(
            output_seed / f"candidate_shortlist_all_top100_{name}.csv",
            index=False,
        )
        strict.head(100).to_csv(
            output_seed / f"candidate_shortlist_strict_top100_{name}.csv",
            index=False,
        )

    old = pd.read_csv(source_seed / "heldout_test_retrieval_metrics_v81.csv")
    for method in ("waveform_only", "time_only", "sky_only"):
        old_row = old[(old["method"] == method) & (old["subset"] == "overall")].iloc[0]
        new_row = test_metrics[
            (test_metrics["method"] == method)
            & (test_metrics["subset"] == "overall")
        ].iloc[0]
        if not (
            np.isclose(old_row["r_at_1"], new_row["r_at_1"], atol=1e-12)
            and np.isclose(old_row["r_at_10"], new_row["r_at_10"], atol=1e-12)
        ):
            raise RuntimeError(
                f"{deployment} seed {seed} frozen-channel reproduction failed "
                f"for {method}"
            )
    return {"deployment": deployment, "seed": seed, "methods": methods}


def numeric_summary(
    frame: pd.DataFrame,
    groups: list[str],
) -> pd.DataFrame:
    numeric = [
        column
        for column in frame.columns
        if column not in groups
        and column != "seed"
        and pd.api.types.is_numeric_dtype(frame[column])
    ]
    rows = []
    for key, part in frame.groupby(groups, sort=False):
        key_values = key if isinstance(key, tuple) else (key,)
        row = dict(zip(groups, key_values))
        row["n_seeds"] = int(part["seed"].nunique()) if "seed" in part else len(part)
        for column in numeric:
            values = part[column].to_numpy(dtype=np.float64)
            finite = values[np.isfinite(values)]
            row[f"{column}_mean"] = (
                float(finite.mean()) if len(finite) else np.nan
            )
            row[f"{column}_std"] = (
                float(finite.std(ddof=1))
                if len(finite) > 1
                else (0.0 if len(finite) == 1 else np.nan)
            )
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_results(output_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    retrieval_frames = []
    pair_frames = []
    weight_rows = []
    for deployment, seeds in (
        ("et3", ET_SEEDS),
        ("gwtc3", GWTC_SEEDS),
        ("gwtc4", GWTC_SEEDS),
    ):
        for seed in seeds:
            seed_root = output_root / deployment / f"seed_{seed}"
            retrieval = pd.read_csv(
                seed_root / "heldout_test_retrieval_metrics.csv"
            )
            retrieval_frames.append(retrieval)
            pair_path = seed_root / "heldout_test_pair_level_metrics.csv"
            if pair_path.exists():
                pair_frames.append(pd.read_csv(pair_path))
            selected = json.loads(
                (seed_root / "selected_simplex_weights.json").read_text(
                    encoding="utf-8"
                )
            )
            for method, weights in selected["methods"].items():
                weight_rows.append(
                    {
                        "deployment": deployment,
                        "seed": seed,
                        "method": method,
                        **weights,
                    }
                )
    retrieval = pd.concat(retrieval_frames, ignore_index=True)
    weights = pd.DataFrame(weight_rows)
    pairs = (
        pd.concat(pair_frames, ignore_index=True)
        if pair_frames
        else pd.DataFrame()
    )
    retrieval.to_csv(output_root / "retrieval_metrics_per_seed.csv", index=False)
    weights.to_csv(output_root / "selected_weights_per_seed.csv", index=False)
    if not pairs.empty:
        pairs.to_csv(output_root / "pair_level_metrics_per_seed.csv", index=False)

    retrieval_summary = numeric_summary(
        retrieval,
        ["deployment", "method", "subset"],
    )
    retrieval_summary.to_csv(
        output_root / "retrieval_metrics_summary.csv",
        index=False,
    )
    weight_summary = numeric_summary(weights, ["deployment", "method"])
    weight_summary.to_csv(
        output_root / "selected_weights_summary.csv",
        index=False,
    )
    if not pairs.empty:
        numeric_summary(pairs, ["deployment", "method"]).to_csv(
            output_root / "pair_level_metrics_summary.csv",
            index=False,
        )
    return retrieval, retrieval_summary, weights


def build_real_consensus(output_root: Path) -> None:
    methods = (
        "waveform_time_candidate_selected",
        "time_sky_candidate_selected",
        "candidate_three_channel_unconstrained",
        "candidate_three_channel_strict_positive",
    )
    for deployment in ("gwtc3", "gwtc4"):
        for method in methods:
            frames = []
            for seed in GWTC_SEEDS:
                path = (
                    output_root
                    / deployment
                    / f"seed_{seed}"
                    / f"real_pair_scores_strict_{method}.parquet"
                )
                frame = pd.read_parquet(path)
                frames.append(
                    frame[
                        [
                            "event_i",
                            "event_j",
                            "rank",
                            "final_score",
                            "waveform_score",
                            "time_score",
                            "sky_score",
                        ]
                    ].assign(seed=seed)
                )
            stability = pd.concat(frames, ignore_index=True)
            consensus = (
                stability.groupby(["event_i", "event_j"], as_index=False)
                .agg(
                    seeds=("seed", "nunique"),
                    rank_mean=("rank", "mean"),
                    rank_std=("rank", "std"),
                    rank_min=("rank", "min"),
                    rank_max=("rank", "max"),
                    final_score_mean=("final_score", "mean"),
                    waveform_score_mean=("waveform_score", "mean"),
                    time_score_mean=("time_score", "mean"),
                    sky_score_mean=("sky_score", "mean"),
                )
                .sort_values(["rank_mean", "rank_max"], kind="stable")
                .reset_index(drop=True)
            )
            consensus.insert(
                0, "consensus_rank", np.arange(1, len(consensus) + 1)
            )
            consensus.to_parquet(
                output_root
                / deployment
                / f"real_candidate_consensus_{method}.parquet",
                index=False,
            )
            consensus.head(100).to_csv(
                output_root
                / deployment
                / f"real_candidate_consensus_top100_{method}.csv",
                index=False,
            )


def aggregate_et_full_pair_metrics(
    output_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, Any]] = []
    operating_frames: list[pd.DataFrame] = []
    for seed in ET_SEEDS:
        seed_root = output_root / "et3" / f"seed_{seed}"
        for method in (
            "waveform_time_validation_selected",
            "three_channel_unconstrained",
        ):
            metric_path = seed_root / f"pair_level_metrics_{method}.json"
            if not metric_path.exists():
                continue
            metric_rows.append(
                json.loads(metric_path.read_text(encoding="utf-8"))
            )
            operating_path = (
                seed_root / f"pair_level_operating_points_{method}.csv"
            )
            if operating_path.exists():
                operating_frames.append(
                    pd.read_csv(operating_path).assign(
                        deployment="et3",
                        seed=seed,
                        method=method,
                    )
                )
    metrics = pd.DataFrame(metric_rows)
    operating = (
        pd.concat(operating_frames, ignore_index=True)
        if operating_frames
        else pd.DataFrame()
    )
    summary = numeric_summary(metrics, ["method"]) if not metrics.empty else pd.DataFrame()
    metrics.to_csv(output_root / "et3_full_pair_metrics_per_seed.csv", index=False)
    summary.to_csv(output_root / "et3_full_pair_metrics_summary.csv", index=False)
    if not operating.empty:
        operating.to_csv(
            output_root / "et3_full_pair_operating_points_per_seed.csv",
            index=False,
        )
        numeric_summary(
            operating,
            ["method", "operating_point", "target"],
        ).to_csv(
            output_root / "et3_full_pair_operating_points_summary.csv",
            index=False,
        )
    return metrics, summary, operating


def attach_pe_and_historical_audits(
    output_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pe_summary_rows: list[dict[str, Any]] = []
    historical_source = pd.read_csv(
        GWTC_CURRENT / "historical_candidate_rank_audit_v81.csv"
    )
    historical_rows: list[dict[str, Any]] = []
    for deployment in ("gwtc3", "gwtc4"):
        consensus_path = (
            output_root
            / deployment
            / "real_candidate_consensus_candidate_three_channel_unconstrained.parquet"
        )
        consensus = pd.read_parquet(consensus_path)
        consensus["pair_key"] = consensus.apply(
            lambda row: "||".join(sorted((str(row["event_i"]), str(row["event_j"])))),
            axis=1,
        )
        pe_source = pd.read_parquet(
            GWTC_CURRENT
            / deployment
            / "real_candidate_consensus_with_pe_v81.parquet"
        )
        pe_columns = [
            column
            for column in pe_source.columns
            if column
            not in {
                "consensus_rank",
                "event_i",
                "event_j",
                "seeds",
                "rank_mean",
                "rank_std",
                "rank_min",
                "rank_max",
                "final_score_mean",
                "waveform_score_mean",
                "time_score_mean",
                "sky_score_mean",
                "deployment",
            }
        ]
        merged = consensus.merge(
            pe_source[pe_columns],
            on="pair_key",
            how="left",
            validate="one_to_one",
        )
        merged["deployment"] = deployment
        merged.to_parquet(
            output_root
            / deployment
            / "real_candidate_consensus_candidate_three_channel_unconstrained_with_pe.parquet",
            index=False,
        )
        merged.head(100).to_csv(
            output_root
            / deployment
            / "real_candidate_consensus_top100_candidate_three_channel_unconstrained_with_pe.csv",
            index=False,
        )
        for top_k in (5, 10, 20, 50, 100):
            part = merged.head(top_k)
            available = part["pe_available"].fillna(False).astype(bool)
            consistent = (
                part["intrinsic_3sigma_consistent"].fillna(False).astype(bool)
                & available
            )
            chirp_consistent = (
                part["chirp_mass_standardized_posterior_distance"].le(3.0)
                & available
            )
            pe_summary_rows.append(
                {
                    "deployment": deployment,
                    "top_k": top_k,
                    "n_pairs": len(part),
                    "n_pe_available": int(available.sum()),
                    "n_intrinsic_3sigma_consistent": int(consistent.sum()),
                    "intrinsic_consistency_fraction_of_pe": (
                        float(consistent.sum() / available.sum())
                        if available.sum()
                        else np.nan
                    ),
                    "n_chirp_mass_3sigma_consistent": int(chirp_consistent.sum()),
                    "chirp_mass_consistency_fraction_of_pe": (
                        float(chirp_consistent.sum() / available.sum())
                        if available.sum()
                        else np.nan
                    ),
                    "median_max_standardized_posterior_distance": float(
                        part.loc[
                            available, "max_standardized_posterior_distance"
                        ].median()
                    ),
                }
            )

        historical = historical_source[
            historical_source["deployment"] == deployment
        ].copy()
        historical["pair_key"] = historical.apply(
            lambda row: "||".join(sorted((str(row["event_i"]), str(row["event_j"])))),
            axis=1,
        )
        lookup = merged.set_index("pair_key")
        for _, row in historical.iterrows():
            key = row["pair_key"]
            payload: dict[str, Any] = {
                "deployment": deployment,
                "event_i": row["event_i"],
                "event_j": row["event_j"],
                "audit_label": row["audit_label"],
                "present_in_strict_catalog": key in lookup.index,
            }
            if key in lookup.index:
                match = lookup.loc[key]
                payload.update(
                    {
                        "simplex_consensus_rank": int(match["consensus_rank"]),
                        "simplex_rank_mean": float(match["rank_mean"]),
                        "simplex_rank_min": int(match["rank_min"]),
                        "simplex_rank_max": int(match["rank_max"]),
                        "pe_available": bool(match["pe_available"]),
                        "intrinsic_3sigma_consistent": bool(
                            match["intrinsic_3sigma_consistent"]
                        ),
                        "max_standardized_posterior_distance": float(
                            match["max_standardized_posterior_distance"]
                        ),
                    }
                )
            historical_rows.append(payload)
    pe_summary = pd.DataFrame(pe_summary_rows)
    historical = pd.DataFrame(historical_rows)
    pe_summary.to_csv(
        output_root / "real_candidate_pe_enrichment_simplex.csv",
        index=False,
    )
    historical.to_csv(
        output_root / "historical_candidate_rank_audit_simplex.csv",
        index=False,
    )
    return pe_summary, historical


def legacy_comparison(
    output_root: Path,
    retrieval: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    mappings = {
        "et3": (
            ET_CURRENT / "et3/moderate/retrieval_metrics_per_seed_v82.csv",
            "three_channel_unconstrained",
            "three_channel_unconstrained",
        ),
        "gwtc3": (
            GWTC_CURRENT
            / "gwtc3/heldout_test_retrieval_metrics_per_seed_v81.csv",
            "retrieval_three_channel_unconstrained",
            "retrieval_three_channel_unconstrained",
        ),
        "gwtc4": (
            GWTC_CURRENT
            / "gwtc4/heldout_test_retrieval_metrics_per_seed_v81.csv",
            "retrieval_three_channel_unconstrained",
            "retrieval_three_channel_unconstrained",
        ),
    }
    for deployment, (path, legacy_method, new_method) in mappings.items():
        old = pd.read_csv(path)
        if deployment == "et3":
            old = old.rename(columns={"deployment": "legacy_deployment"})
        old = old[
            (old["method"] == legacy_method) & (old["subset"] == "overall")
        ]
        new = retrieval[
            (retrieval["deployment"].astype(str).str.lower().str.startswith(deployment))
            & (retrieval["method"] == new_method)
            & (retrieval["subset"] == "overall")
        ]
        for metric in ("r_at_1", "r_at_5", "r_at_10", "median_rank"):
            rows.append(
                {
                    "deployment": deployment,
                    "metric": metric,
                    "legacy_mean": float(old[metric].mean()),
                    "simplex_mean": float(new[metric].mean()),
                    "simplex_minus_legacy": float(
                        new[metric].mean() - old[metric].mean()
                    ),
                }
            )
    comparison = pd.DataFrame(rows)
    comparison.to_csv(output_root / "legacy_vs_simplex_comparison.csv", index=False)
    return comparison


def markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join(["---"] * len(columns)) + "|",
    ]
    for _, row in frame.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, (float, np.floating)):
                values.append(f"{float(value):.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def make_figure(
    output_root: Path,
    retrieval: pd.DataFrame,
    weights: pd.DataFrame,
) -> None:
    figures = output_root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    deployment_order = ("et3", "gwtc3", "gwtc4")
    labels = {"et3": "ET-3", "gwtc3": "GWTC-3/O3", "gwtc4": "GWTC-4.1/O4a"}
    primary_method = {
        "et3": "three_channel_unconstrained",
        "gwtc3": "retrieval_three_channel_unconstrained",
        "gwtc4": "retrieval_three_channel_unconstrained",
    }
    wt_method = {
        "et3": "waveform_time_validation_selected",
        "gwtc3": "waveform_time_retrieval_selected",
        "gwtc4": "waveform_time_retrieval_selected",
    }
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.0))

    x = np.arange(len(deployment_order))
    width = 0.34
    for offset, (method_map, label, color) in enumerate(
        (
            (wt_method, "Waveform + time", "#4C78A8"),
            (primary_method, "Unconstrained simplex", "#E45756"),
        )
    ):
        means, errors = [], []
        for deployment in deployment_order:
            part = retrieval[
                (
                    retrieval["deployment"]
                    .astype(str)
                    .str.lower()
                    .str.startswith(deployment)
                )
                & (retrieval["method"] == method_map[deployment])
                & (retrieval["subset"] == "overall")
            ]
            means.append(float(part["r_at_10"].mean()))
            errors.append(float(part["r_at_10"].std(ddof=1)))
        axes[0].bar(
            x + (offset - 0.5) * width,
            means,
            width,
            yerr=errors,
            capsize=3,
            label=label,
            color=color,
        )
    axes[0].set_xticks(x, [labels[name] for name in deployment_order])
    axes[0].set_ylabel("Held-out R@10", fontweight="bold")
    axes[0].set_title("a  Sky-channel marginal check", loc="left", fontweight="bold")
    axes[0].legend(frameon=False, fontsize=8)

    primary_weights = []
    for deployment in deployment_order:
        method = primary_method[deployment]
        part = weights[
            (weights["deployment"] == deployment)
            & (weights["method"] == method)
        ]
        primary_weights.append(
            [float(part[channel].mean()) for channel in CHANNELS]
        )
    bottom = np.zeros(len(deployment_order))
    colors = {"waveform": "#4C78A8", "time": "#F2CF5B", "sky": "#59A14F"}
    for index, channel in enumerate(CHANNELS):
        values = np.asarray([row[index] for row in primary_weights])
        axes[1].bar(
            x,
            values,
            bottom=bottom,
            color=colors[channel],
            label=channel.capitalize(),
        )
        bottom += values
    axes[1].set_xticks(x, [labels[name] for name in deployment_order])
    axes[1].set_ylabel("Mean simplex weight", fontweight="bold")
    axes[1].set_ylim(0, 1)
    axes[1].set_title("b  Validation-selected weights", loc="left", fontweight="bold")
    axes[1].legend(frameon=False, fontsize=8)

    for position, deployment in enumerate(deployment_order):
        method = primary_method[deployment]
        part = weights[
            (weights["deployment"] == deployment)
            & (weights["method"] == method)
        ]
        jitter = np.linspace(-0.12, 0.12, len(part))
        for channel in CHANNELS:
            axes[2].scatter(
                np.full(len(part), position) + jitter,
                part[channel],
                label=channel.capitalize() if position == 0 else None,
                color=colors[channel],
                s=34,
                alpha=0.85,
            )
    axes[2].set_xticks(x, [labels[name] for name in deployment_order])
    axes[2].set_ylabel("Per-seed simplex weight", fontweight="bold")
    axes[2].set_ylim(-0.03, 1.03)
    axes[2].set_title("c  Weight stability", loc="left", fontweight="bold")
    axes[2].legend(frameon=False, fontsize=8)

    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.tick_params(axis="x", labelrotation=10)
    fig.tight_layout()
    fig.savefig(figures / "fig_simplex_weight_refusion.pdf", bbox_inches="tight")
    fig.savefig(
        figures / "fig_simplex_weight_refusion.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def make_report(
    output_root: Path,
    retrieval: pd.DataFrame,
    weights: pd.DataFrame,
    comparison: pd.DataFrame,
    et_pair_summary: pd.DataFrame,
    pe_summary: pd.DataFrame,
    historical: pd.DataFrame,
    step: float,
) -> None:
    primary = {
        "et3": "three_channel_unconstrained",
        "gwtc3": "retrieval_three_channel_unconstrained",
        "gwtc4": "retrieval_three_channel_unconstrained",
    }
    wt = {
        "et3": "waveform_time_validation_selected",
        "gwtc3": "waveform_time_retrieval_selected",
        "gwtc4": "waveform_time_retrieval_selected",
    }
    metric_rows = []
    weight_report_rows = []
    candidate_weight_rows = []
    for deployment in ("et3", "gwtc3", "gwtc4"):
        for label, method in (("waveform+time", wt[deployment]), ("primary", primary[deployment])):
            part = retrieval[
                (
                    retrieval["deployment"]
                    .astype(str)
                    .str.lower()
                    .str.startswith(deployment)
                )
                & (retrieval["method"] == method)
                & (retrieval["subset"] == "overall")
            ]
            metric_rows.append(
                {
                    "deployment": deployment,
                    "method": label,
                    "R@1 mean": part["r_at_1"].mean(),
                    "R@1 SD": part["r_at_1"].std(ddof=1),
                    "R@10 mean": part["r_at_10"].mean(),
                    "R@10 SD": part["r_at_10"].std(ddof=1),
                    "median rank": part["median_rank"].mean(),
                }
            )
        part = weights[
            (weights["deployment"] == deployment)
            & (weights["method"] == primary[deployment])
        ]
        for _, row in part.iterrows():
            weight_report_rows.append(
                {
                    "deployment": deployment,
                    "seed": int(row["seed"]),
                    "waveform": row["waveform"],
                    "time": row["time"],
                    "sky": row["sky"],
                }
            )
        if deployment in {"gwtc3", "gwtc4"}:
            candidate_part = weights[
                (weights["deployment"] == deployment)
                & (
                    weights["method"]
                    == "candidate_three_channel_unconstrained"
                )
            ]
            for _, row in candidate_part.iterrows():
                candidate_weight_rows.append(
                    {
                        "deployment": deployment,
                        "seed": int(row["seed"]),
                        "waveform": row["waveform"],
                        "time": row["time"],
                        "sky": row["sky"],
                    }
                )
    metric_frame = pd.DataFrame(metric_rows)
    weight_frame = pd.DataFrame(weight_report_rows)
    candidate_weight_frame = pd.DataFrame(candidate_weight_rows)
    metric_frame.to_csv(output_root / "primary_and_waveform_time_summary.csv", index=False)
    weight_frame.to_csv(output_root / "primary_simplex_weights.csv", index=False)
    candidate_weight_frame.to_csv(
        output_root / "candidate_simplex_weights.csv",
        index=False,
    )

    strict_matches = []
    for deployment, objective, strict, unconstrained in (
        (
            "et3",
            "retrieval",
            "three_channel_strict_positive",
            "three_channel_unconstrained",
        ),
        (
            "gwtc3",
            "retrieval",
            "retrieval_three_channel_strict_positive",
            "retrieval_three_channel_unconstrained",
        ),
        (
            "gwtc4",
            "retrieval",
            "retrieval_three_channel_strict_positive",
            "retrieval_three_channel_unconstrained",
        ),
        (
            "gwtc3",
            "candidate",
            "candidate_three_channel_strict_positive",
            "candidate_three_channel_unconstrained",
        ),
        (
            "gwtc4",
            "candidate",
            "candidate_three_channel_strict_positive",
            "candidate_three_channel_unconstrained",
        ),
    ):
        left = weights[
            (weights["deployment"] == deployment) & (weights["method"] == strict)
        ].sort_values("seed")
        right = weights[
            (weights["deployment"] == deployment)
            & (weights["method"] == unconstrained)
        ].sort_values("seed")
        per_seed_match = np.all(
            np.isclose(
                left[list(CHANNELS)].to_numpy(),
                right[list(CHANNELS)].to_numpy(),
            ),
            axis=1,
        )
        strict_matches.append(
            {
                "deployment": deployment,
                "objective": objective,
                "matching_seeds": int(per_seed_match.sum()),
                "n_seeds": len(per_seed_match),
                "all_same": bool(per_seed_match.all()),
            }
        )

    gwtc_pair = pd.read_csv(output_root / "pair_level_metrics_per_seed.csv")
    pair_rows = []
    for deployment in ("gwtc3", "gwtc4"):
        for label, method in (
            ("waveform+time", "waveform_time_candidate_selected"),
            ("three-channel", "candidate_three_channel_unconstrained"),
        ):
            part = gwtc_pair[
                (gwtc_pair["deployment"] == deployment)
                & (gwtc_pair["method"] == method)
            ]
            pair_rows.append(
                {
                    "deployment": deployment,
                    "method": label,
                    "AUPRC mean": part["average_precision"].mean(),
                    "AUPRC SD": part["average_precision"].std(ddof=1),
                    "precision@recall0.5 mean": part[
                        "precision_at_recall_0p5"
                    ].mean(),
                    "false@recall0.5 mean": part[
                        "false_at_recall_0p5"
                    ].mean(),
                }
            )
    pair_frame = pd.DataFrame(pair_rows)
    et_pair_report = et_pair_summary[
        [
            "method",
            "n_seeds",
            "roc_auc_mean",
            "roc_auc_std",
            "average_precision_mean",
            "average_precision_std",
        ]
    ]
    top_candidate_frames = []
    for deployment in ("gwtc3", "gwtc4"):
        candidates = pd.read_parquet(
            output_root
            / deployment
            / "real_candidate_consensus_candidate_three_channel_unconstrained_with_pe.parquet"
        ).head(10)
        top_candidate_frames.append(
            candidates[
                [
                    "consensus_rank",
                    "event_i",
                    "event_j",
                    "rank_mean",
                    "rank_min",
                    "rank_max",
                    "intrinsic_3sigma_consistent",
                    "max_standardized_posterior_distance",
                ]
            ].assign(deployment=deployment)
        )
    top_candidates = pd.concat(top_candidate_frames, ignore_index=True)[
        [
            "deployment",
            "consensus_rank",
            "event_i",
            "event_j",
            "rank_mean",
            "rank_min",
            "rank_max",
            "intrinsic_3sigma_consistent",
            "max_standardized_posterior_distance",
        ]
    ]
    top_candidates.to_csv(
        output_root / "real_candidate_top10_simplex_with_pe.csv",
        index=False,
    )

    report = f"""# ET-3 / GWTC simplex 权重重融合报告

## 1. 实验边界

本实验没有重新训练 encoder，没有改变 waveform、time-delay 或 sky score，
也没有使用 held-out test、真实 GWTC 排名或真实 PE 后验调权。唯一改变是把
旧的 Cartesian 权重网格替换为：

\\[
w_W+w_T+w_S=1,\\qquad w_W,w_T,w_S\\ge0,
\\]

步长为 `{step:.2f}`，共 `{len(simplex_weight_rows(step))}` 个组合。

主口径为 unconstrained：任一通道都可以由 validation 设为 0。
strict-positive、等权、waveform+time 和 time+sky 均作为对照保留。

## 2. 选择规则

- Retrieval：macro R@10、min-family R@10、AUPRC、50% recall precision、
  macro R@1，最后选择较小 L2 权重。
- Candidate：AUPRC、50% recall precision、macro R@10、min-family R@10、
  macro R@1。
- 每个 seed 只使用自己的 synthetic validation systems 选权。
- test 和真实目录只在权重冻结后评估。

## 3. 主检索结果

{markdown_table(metric_frame)}

## 4. Validation 选择的主权重

{markdown_table(weight_frame)}

GWTC candidate-objective（以 validation AUPRC 为首要目标）选择的权重：

{markdown_table(candidate_weight_frame)}

## 5. 与旧网格对比

{markdown_table(comparison)}

## 6. Strict-positive 与 unconstrained

{markdown_table(pd.DataFrame(strict_matches))}

若两者相同，说明 sky/time/waveform 的正权重不是 strict-positive 规则强迫
产生，而是允许为零的 validation 搜索也选择了它。若不同，应以
unconstrained 为主结果，strict-positive 只能作为敏感性分析。

## 7. 完整无序 pair 指标

ET-3 使用全部 40,495,500 个 `i < j` 无序 pair，两个 directed score 按论文
预定的 `max` 口径合并，不抽样假对：

{markdown_table(et_pair_report)}

GWTC held-out real-noise injection test 的 pair-level 指标如下：

{markdown_table(pair_frame)}

## 8. 真实目录 PE 后验审计

PE 后验没有参与权重选择或真实目录排序。下表仅在权重冻结后检查高排名
候选的内禀参数是否在 3-sigma 范围相容：

{markdown_table(pe_summary)}

预先指定的历史候选对在新 simplex 排序中的位置：

{markdown_table(historical)}

真实 strict-H1L1 目录的 candidate-objective consensus Top 10：

{markdown_table(top_candidates)}

这些 rank 是 catalog coincidence rank，不是显著性或透镜探测声明。
O3 的 consensus rank 1 没有通过冻结后的内禀参数 3-sigma PE screen，因此
不能称为通过物理确认的透镜候选；它只说明三通道 triage 仍会留下需要后续
排除的高分巧合。

## 9. 解释边界

权重是三个已校准 evidence 的相对数值系数，不等同于三个物理通道的“贡献
百分比”。通道是否有边际价值，应同时查看 waveform+time 与完整
unconstrained 三通道在 held-out R@K、AUPRC 和真实候选稳定性上的差异。

ET-3 和 O3 的 sky 通道在 held-out retrieval 与 pair-level AUPRC 上都有
正边际价值。O4a 的 retrieval R@10 略有改善，但 candidate-objective test
AUPRC 未稳定优于 waveform+time，而且 1/3 个 seed 在允许为零时选择
`w_sky=0`；因此 O4a 的 sky 贡献应保守表述为弱且不稳定。
"""
    (output_root / "simplex_weight_refusion_report_cn.md").write_text(
        report,
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--step", type=float, default=0.05)
    parser.add_argument(
        "--deployment",
        choices=["all", "et3", "gwtc3", "gwtc4"],
        default="all",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--skip-et-full-pair-metrics", action="store_true")
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Reuse completed per-seed outputs and rebuild summaries/reports only.",
    )
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    weights = simplex_weight_rows(args.step)
    write_json(
        args.output_root / "run_config.json",
        {
            "version": "simplex_weight_refusion_20260726",
            "simplex_step": args.step,
            "n_weight_combinations": len(weights),
            "primary_search": "unconstrained",
            "encoder_retrained": False,
            "channels_recomputed": False,
            "test_used_for_selection": False,
            "real_catalog_or_pe_used_for_selection": False,
            "source_et": str(ET_CURRENT),
            "source_gwtc": str(GWTC_CURRENT),
        },
    )

    if not args.aggregate_only and args.deployment in {"all", "et3"}:
        seeds = (args.seed,) if args.seed is not None else ET_SEEDS
        for seed in seeds:
            evaluate_et_seed(
                seed,
                args.output_root,
                weights,
                full_pair_metrics=not args.skip_et_full_pair_metrics,
            )
    if not args.aggregate_only and args.deployment in {"all", "gwtc3", "gwtc4"}:
        deployments = (
            ("gwtc3", "gwtc4")
            if args.deployment == "all"
            else (args.deployment,)
        )
        seeds = (args.seed,) if args.seed is not None else GWTC_SEEDS
        for deployment in deployments:
            for seed in seeds:
                evaluate_gwtc_seed(
                    deployment,
                    seed,
                    args.output_root,
                    weights,
                )

    if args.deployment == "all" and args.seed is None:
        retrieval, _, selected = aggregate_results(args.output_root)
        build_real_consensus(args.output_root)
        _, et_pair_summary, _ = aggregate_et_full_pair_metrics(args.output_root)
        pe_summary, historical = attach_pe_and_historical_audits(args.output_root)
        comparison = legacy_comparison(args.output_root, retrieval)
        make_figure(args.output_root, retrieval, selected)
        make_report(
            args.output_root,
            retrieval,
            selected,
            comparison,
            et_pair_summary,
            pe_summary,
            historical,
            args.step,
        )
        primary_methods = {
            "et3": "three_channel_unconstrained",
            "gwtc3": "retrieval_three_channel_unconstrained",
            "gwtc4": "retrieval_three_channel_unconstrained",
        }
        core_metrics: dict[str, Any] = {}
        for deployment, method in primary_methods.items():
            part = retrieval[
                (
                    retrieval["deployment"]
                    .astype(str)
                    .str.lower()
                    .str.startswith(deployment)
                )
                & (retrieval["method"] == method)
                & (retrieval["subset"] == "overall")
            ]
            weight_part = selected[
                (selected["deployment"] == deployment)
                & (selected["method"] == method)
            ]
            core_metrics[deployment] = {
                "n_seeds": int(part["seed"].nunique()),
                "r_at_1_mean": float(part["r_at_1"].mean()),
                "r_at_1_std": float(part["r_at_1"].std(ddof=1)),
                "r_at_10_mean": float(part["r_at_10"].mean()),
                "r_at_10_std": float(part["r_at_10"].std(ddof=1)),
                "median_rank_mean": float(part["median_rank"].mean()),
                "mean_retrieval_weights": {
                    channel: float(weight_part[channel].mean())
                    for channel in CHANNELS
                },
            }
        et_pair_lookup = et_pair_summary.set_index("method")
        write_json(
            args.output_root / "simplex_weight_refusion_summary.json",
            {
                "status": "complete",
                "simplex_step": args.step,
                "n_weight_combinations": len(weights),
                "deployments": ["ET3-moderate", "GWTC-3/O3", "GWTC-4.1/O4a"],
                "primary_method": "validation-selected unconstrained simplex",
                "core_retrieval_metrics": core_metrics,
                "et3_full_unordered_pair_metrics": {
                    method: {
                        "n_pairs": int(
                            et_pair_lookup.loc[method, "n_pairs_mean"]
                        ),
                        "roc_auc_mean": float(
                            et_pair_lookup.loc[method, "roc_auc_mean"]
                        ),
                        "average_precision_mean": float(
                            et_pair_lookup.loc[
                                method, "average_precision_mean"
                            ]
                        ),
                        "average_precision_std": float(
                            et_pair_lookup.loc[
                                method, "average_precision_std"
                            ]
                        ),
                    }
                    for method in et_pair_lookup.index
                },
                "pe_audit_file": "real_candidate_pe_enrichment_simplex.csv",
                "historical_candidate_file": (
                    "historical_candidate_rank_audit_simplex.csv"
                ),
                "legacy_results_overwritten": False,
            },
        )


if __name__ == "__main__":
    main()
