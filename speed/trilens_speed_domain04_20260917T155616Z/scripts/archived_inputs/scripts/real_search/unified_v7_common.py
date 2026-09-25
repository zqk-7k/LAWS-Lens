#!/usr/bin/env python3
"""Shared, leakage-controlled utilities for the v7 real-noise deployment.

The module deliberately treats the two GW-LMC lens-environment groups as one
retrieval catalog and one waveform encoder.  The legacy ``SIS``/``PM`` names
are retained only as compatibility labels for on-disk arrays; they do not
claim that the GW-LMC halo realizations are analytic SIS/point-mass lenses.

All calibrations in this module are fitted on synthetic validation systems.
No real-catalog pair or PE posterior is read while fitting waveform evidence
or fusion weights.
"""

from __future__ import annotations

import importlib.util
import itertools
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader, Dataset


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from matchgw.data import load_match_arrays, split_indices
from matchgw.matching import similarity_matrix
from scripts.real_search.physical_common import (
    SECONDS_PER_DAY,
    apply_score_likelihood_ratio,
    apply_time_likelihood_ratio,
    effective_rank,
    fit_score_likelihood_ratio,
    posterior_area90,
    rotate_probability_map_to_true_position,
    sky_log_bayes_factor_from_maps,
    write_json,
)


FAMILIES = ("SIS", "PM")
WEIGHT_GRID = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)
INTRINSIC_GRID = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)


def module_from(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


v3 = module_from(REPO / "scripts/experiments/20_real_noise_injection_v3_physical.py", "v3_for_v7_common")
pilot = module_from(REPO / "scripts/real_search/37_unified_intrinsic_multitask_pilot.py", "pilot_for_v7_common")
specmod = module_from(REPO / "scripts/experiments/26_spectrogram_encoder_pilot.py", "spec_for_v7_common")


@dataclass
class MixedFamilyData:
    arrays: Any
    parts: dict[str, dict[str, np.ndarray]]


class MixedCatalog(Dataset):
    """One catalog containing both GW-LMC groups and one background pool."""

    def __init__(self, data: dict[str, MixedFamilyData], split: str) -> None:
        self.waveforms: list[np.ndarray] = []
        self.meta: list[dict[str, Any]] = []
        self.partner: list[int] = []
        self.query_family: list[str] = []
        pair_offset = 0
        for family in FAMILIES:
            ids = np.asarray(data[family].parts["lensed"][split], dtype=np.int64)
            first = len(self.waveforms)
            for local, source_index in enumerate(ids):
                self.waveforms.append(data[family].arrays.l1[int(source_index)])
                self.meta.append(
                    {
                        "tag": "L1",
                        "family": family,
                        "pair_id": pair_offset + local,
                        "source_index": int(source_index),
                    }
                )
                self.query_family.append(family)
            second = len(self.waveforms)
            for local, source_index in enumerate(ids):
                self.waveforms.append(data[family].arrays.l2[int(source_index)])
                self.meta.append(
                    {
                        "tag": "L2",
                        "family": family,
                        "pair_id": pair_offset + local,
                        "source_index": int(source_index),
                    }
                )
                self.query_family.append(family)
            n = len(ids)
            self.partner.extend(range(second, second + n))
            self.partner.extend(range(first, first + n))
            pair_offset += n

        # The same physical unlensed bank is linked into both compatibility
        # roots. Include it once, using the split associated with the first
        # family, so background events are not duplicated.
        reference = data[FAMILIES[0]]
        for source_index in reference.parts["unlensed"][split]:
            self.waveforms.append(reference.arrays.unlensed[int(source_index)])
            self.meta.append(
                {
                    "tag": "U",
                    "family": "unlensed",
                    "pair_id": -1,
                    "source_index": int(source_index),
                }
            )
            self.partner.append(-1)
            self.query_family.append("unlensed")
        self.partner = np.asarray(self.partner, dtype=np.int32)
        self.query_family = np.asarray(self.query_family, dtype=object)

    def __len__(self) -> int:
        return len(self.waveforms)

    def __getitem__(self, index: int) -> torch.Tensor:
        return torch.from_numpy(pilot.prepare(self.waveforms[index], None, False))


def load_mixed_data(seed_dir: Path, source_bank: Path, seed: int, samples: int) -> dict[str, MixedFamilyData]:
    del source_bank
    data: dict[str, MixedFamilyData] = {}
    for family in FAMILIES:
        cfg = v3.training_config(seed_dir, family, seed, samples, 1, 8)
        arrays = load_match_arrays(cfg)
        data[family] = MixedFamilyData(
            arrays=arrays,
            parts=split_indices(samples, samples, cfg),
        )
    return data


def load_unified_model(checkpoint: Path) -> tuple[torch.nn.Module, dict[str, Any]]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    backbone = str(payload.get("backbone", "trigger_spectrogram"))
    model = pilot.PhysicsRegularizedEncoder(pilot.build_base_encoder(backbone, specmod))
    model.load_state_dict(payload["model_state"])
    model = model.cuda().eval()
    return model, payload


@torch.no_grad()
def embed_catalog(model: torch.nn.Module, dataset: Dataset, batch_size: int = 24) -> tuple[np.ndarray, np.ndarray]:
    embeddings: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    for values in DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            z, prediction = model(values.cuda(non_blocking=True), return_parameters=True)
        embeddings.append(z.float().cpu().numpy())
        predictions.append(prediction.float().cpu().numpy())
    return np.concatenate(embeddings).astype(np.float32), np.concatenate(predictions).astype(np.float32)


def _synthetic_event_maps(
    catalog: MixedCatalog,
    metadata: pd.DataFrame,
    shared_dir: Path,
    seed: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    templates = np.load(shared_dir / f"real_pe_sky_templates_nside{v3.SKY_NSIDE}.npy", mmap_mode="r")
    template_frame = pd.read_csv(shared_dir / "real_pe_sky_template_manifest.csv")
    template_frame = template_frame[template_frame["usable"] == True].sort_values("template_index")
    template_snr = template_frame["network_snr"].to_numpy(dtype=np.float64)
    template_names = template_frame["event_name"].astype(str).to_numpy()
    template_area = template_frame["area90_deg2"].to_numpy(dtype=np.float64)
    by_family = {
        family: metadata[metadata["family"] == family].set_index("sample_index")
        for family in FAMILIES
    }
    unlensed = metadata[metadata["family"] == "unlensed"].set_index("sample_index")
    rng = np.random.default_rng(seed)
    events: list[dict[str, Any]] = []
    maps: list[np.ndarray] = []
    for index, item in enumerate(catalog.meta):
        family = str(item["family"])
        source_index = int(item["source_index"])
        tag = str(item["tag"])
        row = unlensed.loc[source_index] if tag == "U" else by_family[family].loc[source_index]
        image = 2 if tag == "L2" else 1
        snr = float(row[f"target_snr_image{image}"])
        gps = float(row[f"gps_image{image}"])
        template_index = v3._choose_sky_template(snr, template_snr, rng)
        probability, anchor = rotate_probability_map_to_true_position(
            np.asarray(templates[template_index]), float(row.ra_true), float(row.dec_true), rng
        )
        maps.append(probability)
        events.append(
            {
                "idx": index,
                "family": family,
                "tag": tag,
                "pair_id": int(item["pair_id"]),
                "source_index": source_index,
                "gps_obs": gps,
                "snr": snr,
                "ra_true": float(row.ra_true),
                "dec_true": float(row.dec_true),
                "sky_template_index": template_index,
                "sky_template_event": template_names[template_index],
                "sky_template_area90_deg2": float(template_area[template_index]),
                "sky_anchor_pixel": int(anchor),
            }
        )
    return pd.DataFrame(events), np.stack(maps).astype(np.float32)


def build_mixed_pair_table(
    seed_dir: Path,
    shared_dir: Path,
    source_bank: Path,
    checkpoint: Path,
    seed: int,
    samples: int,
    split: str,
    time_calibration: dict[str, Any],
    observable_seed: int,
) -> pd.DataFrame:
    output = seed_dir / "results" / f"mixed_{split}_pair_features_v7.parquet"
    if output.exists():
        return pd.read_parquet(output)
    data = load_mixed_data(seed_dir, source_bank, seed, samples)
    catalog = MixedCatalog(data, split)
    model, checkpoint_payload = load_unified_model(checkpoint)
    embedding, prediction = embed_catalog(model, catalog)
    del model
    torch.cuda.empty_cache()
    metadata = pd.read_parquet(seed_dir / "data/real_noise_injections/compact_injection_metadata.parquet")
    events, maps = _synthetic_event_maps(catalog, metadata, shared_dir, observable_seed)
    result_dir = seed_dir / "results"
    result_dir.mkdir(parents=True, exist_ok=True)
    events.to_parquet(result_dir / f"mixed_{split}_synthetic_events_v7.parquet", index=False)
    np.save(result_dir / f"mixed_{split}_synthetic_sky_posteriors_nside{v3.SKY_NSIDE}_v7.npy", maps)
    np.save(result_dir / f"mixed_{split}_unified_embeddings_v7.npy", embedding)
    np.save(result_dir / f"mixed_{split}_intrinsic_predictions_standardized_v7.npy", prediction)

    waveform = similarity_matrix(embedding)
    sky_log_bf, sky_raw, sky_cosine = sky_log_bayes_factor_from_maps(maps)
    ii, jj = np.triu_indices(len(catalog), k=1)
    gps = events["gps_obs"].to_numpy(dtype=np.float64)
    delta = np.abs(gps[ii] - gps[jj]) / SECONDS_PER_DAY
    labels = catalog.partner[ii] == jj
    pair_family = np.where(labels, events["family"].to_numpy(dtype=object)[ii], "background")
    frame = pd.DataFrame(
        {
            "split": split,
            "idx_i": ii.astype(np.int32),
            "idx_j": jj.astype(np.int32),
            "is_true_pair": labels.astype(np.int8),
            "true_pair_family": pair_family,
            "waveform_embedding_cosine": waveform[ii, jj].astype(np.float32),
            "waveform_pred_logmc_std_i": prediction[ii, 0].astype(np.float32),
            "waveform_pred_logmc_std_j": prediction[jj, 0].astype(np.float32),
            "waveform_pred_logitq_std_i": prediction[ii, 1].astype(np.float32),
            "waveform_pred_logitq_std_j": prediction[jj, 1].astype(np.float32),
            "waveform_abs_delta_logmc_std": np.abs(prediction[ii, 0] - prediction[jj, 0]).astype(np.float32),
            "waveform_abs_delta_logitq_std": np.abs(prediction[ii, 1] - prediction[jj, 1]).astype(np.float32),
            "delta_t_days": delta.astype(np.float64),
            "time_score": apply_time_likelihood_ratio(delta, time_calibration),
            "sky_score": sky_log_bf[ii, jj].astype(np.float32),
            "sky_bayes_factor": np.exp(np.clip(sky_log_bf[ii, jj], -80, 80)),
            "sky_raw_overlap": sky_raw[ii, jj].astype(np.float64),
            "sky_cosine_overlap": sky_cosine[ii, jj].astype(np.float32),
            "event_count": int(len(catalog)),
        }
    )
    frame.attrs["target_mean"] = np.asarray(checkpoint_payload["target_mean"]).tolist()
    frame.attrs["target_std"] = np.asarray(checkpoint_payload["target_std"]).tolist()
    frame.to_parquet(output, index=False)
    write_json(
        result_dir / f"mixed_{split}_catalog_audit_v7.json",
        {
            "event_count": len(catalog),
            "true_pair_count": int(labels.sum()),
            "false_pair_count": int((~labels).sum()),
            "smooth_non_subhalo_true_pairs": int(np.sum(pair_family == "SIS")),
            "subhalo_present_true_pairs": int(np.sum(pair_family == "PM")),
            "unlensed_events": int(np.sum(events["family"].astype(str) == "unlensed")),
            "single_mixed_catalog": True,
            "test_used_for_model_selection": False,
        },
    )
    return frame


def _standardization(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {"center": float(np.median(values)), "scale": max(float(np.std(values)), 1e-8)}


def _apply_standardization(values: np.ndarray, config: dict[str, float]) -> np.ndarray:
    return (np.asarray(values, dtype=np.float64) - float(config["center"])) / float(config["scale"])


def score_vector(frame: pd.DataFrame, weights: dict[str, float]) -> np.ndarray:
    return (
        float(weights["waveform"]) * frame["waveform_score"].to_numpy(dtype=np.float64)
        + float(weights["time"]) * frame["time_score"].to_numpy(dtype=np.float64)
        + float(weights["sky"]) * frame["sky_score"].to_numpy(dtype=np.float64)
    )


def query_rank_rows(frame: pd.DataFrame, scores: np.ndarray, method: str) -> pd.DataFrame:
    n = int(frame["event_count"].iloc[0])
    matrix = np.full((n, n), -np.inf, dtype=np.float64)
    truth = np.full(n, -1, dtype=np.int32)
    family = np.full(n, "unlensed", dtype=object)
    ii = frame["idx_i"].to_numpy(dtype=np.int32)
    jj = frame["idx_j"].to_numpy(dtype=np.int32)
    matrix[ii, jj] = scores
    matrix[jj, ii] = scores
    positive = frame["is_true_pair"].to_numpy(dtype=bool)
    truth[ii[positive]] = jj[positive]
    truth[jj[positive]] = ii[positive]
    pair_family = frame.loc[positive, "true_pair_family"].astype(str).to_numpy()
    family[ii[positive]] = pair_family
    family[jj[positive]] = pair_family
    rows = []
    for query in np.flatnonzero(truth >= 0):
        partner = int(truth[query])
        rank = int(1 + np.sum(matrix[query] > matrix[query, partner]))
        rows.append(
            {
                "method": method,
                "query_index": int(query),
                "partner_index": partner,
                "family": str(family[query]),
                "system_id": f"{family[query]}:{min(query, partner)}-{max(query, partner)}",
                "query_rank": rank,
            }
        )
    return pd.DataFrame(rows)


def retrieval_metrics(frame: pd.DataFrame, scores: np.ndarray) -> dict[str, float]:
    ranks = query_rank_rows(frame, scores, "evaluation")
    out: dict[str, float] = {}
    family_metrics = []
    for family in FAMILIES:
        values = ranks.loc[ranks["family"] == family, "query_rank"].to_numpy(dtype=np.int32)
        metrics = {
            "r_at_1": float(np.mean(values <= 1)),
            "r_at_5": float(np.mean(values <= 5)),
            "r_at_10": float(np.mean(values <= 10)),
            "median_rank": float(np.median(values)),
        }
        family_metrics.append(metrics)
        for key, value in metrics.items():
            out[f"{family.lower()}_{key}"] = value
    all_ranks = ranks["query_rank"].to_numpy(dtype=np.int32)
    out.update(
        {
            "overall_r_at_1": float(np.mean(all_ranks <= 1)),
            "overall_r_at_5": float(np.mean(all_ranks <= 5)),
            "overall_r_at_10": float(np.mean(all_ranks <= 10)),
            "overall_median_rank": float(np.median(all_ranks)),
            "macro_r_at_1": float(np.mean([row["r_at_1"] for row in family_metrics])),
            "macro_r_at_10": float(np.mean([row["r_at_10"] for row in family_metrics])),
            "min_family_r_at_10": float(np.min([row["r_at_10"] for row in family_metrics])),
        }
    )
    return out


def pair_metrics(frame: pd.DataFrame, scores: np.ndarray) -> dict[str, float]:
    labels = frame["is_true_pair"].to_numpy(dtype=np.int8)
    order = np.argsort(-scores, kind="stable")
    y = labels[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    n_true = max(int(labels.sum()), 1)
    out = {"average_precision": float(average_precision_score(labels, scores))}
    for target in (0.5, 0.9):
        index = min(int(np.searchsorted(tp / n_true, target, side="left")), len(tp) - 1)
        out[f"precision_at_recall_{str(target).replace('.', 'p')}"] = float(
            tp[index] / max(tp[index] + fp[index], 1)
        )
        out[f"false_at_recall_{str(target).replace('.', 'p')}"] = int(fp[index])
    n_false = max(int((labels == 0).sum()), 1)
    budget = max(1, int(math.floor(1e-5 * n_false)))
    valid = np.flatnonzero(fp <= budget)
    index = int(valid[-1]) if len(valid) else 0
    out.update(
        {
            "recall_at_false_pair_rate_1e_5": float(tp[index] / n_true),
            "precision_at_false_pair_rate_1e_5": float(tp[index] / max(tp[index] + fp[index], 1)),
            "true_at_false_pair_rate_1e_5": int(tp[index]),
            "false_at_false_pair_rate_1e_5": int(fp[index]),
        }
    )
    return out


def fit_waveform_channel(
    validation: pd.DataFrame,
    *,
    allow_q_feature: bool = True,
) -> tuple[dict[str, Any], pd.DataFrame]:
    feature_config = {
        "embedding_cosine": _standardization(validation["waveform_embedding_cosine"].to_numpy()),
        "negative_delta_logmc": _standardization(-validation["waveform_abs_delta_logmc_std"].to_numpy()),
        "negative_delta_logitq": _standardization(-validation["waveform_abs_delta_logitq_std"].to_numpy()),
    }
    z_cosine = _apply_standardization(validation["waveform_embedding_cosine"], feature_config["embedding_cosine"])
    z_mc = _apply_standardization(-validation["waveform_abs_delta_logmc_std"], feature_config["negative_delta_logmc"])
    z_q = _apply_standardization(-validation["waveform_abs_delta_logitq_std"], feature_config["negative_delta_logitq"])
    rows = []
    q_grid = INTRINSIC_GRID if allow_q_feature else (0.0,)
    for lambda_mc, lambda_q in itertools.product(INTRINSIC_GRID, q_grid):
        composite = z_cosine + lambda_mc * z_mc + lambda_q * z_q
        retrieval = retrieval_metrics(validation, composite)
        pair = pair_metrics(validation, composite)
        rows.append(
            {
                "lambda_mc": lambda_mc,
                "lambda_q": lambda_q,
                **retrieval,
                **pair,
                "regularization_l1": lambda_mc + lambda_q,
            }
        )
    grid = pd.DataFrame(rows).sort_values(
        ["macro_r_at_10", "min_family_r_at_10", "average_precision", "macro_r_at_1", "regularization_l1"],
        ascending=[False, False, False, False, True],
    ).reset_index(drop=True)
    best = grid.iloc[0]
    composite = z_cosine + float(best.lambda_mc) * z_mc + float(best.lambda_q) * z_q
    labels = validation["is_true_pair"].to_numpy(dtype=bool)
    likelihood = fit_score_likelihood_ratio(composite[labels], composite[~labels], grid_size=2048)
    config = {
        "definition": "Validation-frozen waveform evidence from embedding cosine and waveform-derived intrinsic compatibility; real PE is not used.",
        "feature_standardization": feature_config,
        "lambda_mc": float(best.lambda_mc),
        "lambda_q": float(best.lambda_q),
        "q_feature_allowed": bool(allow_q_feature),
        "lambda_selection_split": "synthetic validation systems only",
        "likelihood_ratio": likelihood,
        "validation_metrics": best.to_dict(),
    }
    return config, grid


def apply_waveform_channel(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    out = frame.copy()
    features = config["feature_standardization"]
    z_cosine = _apply_standardization(out["waveform_embedding_cosine"], features["embedding_cosine"])
    z_mc = _apply_standardization(-out["waveform_abs_delta_logmc_std"], features["negative_delta_logmc"])
    z_q = _apply_standardization(-out["waveform_abs_delta_logitq_std"], features["negative_delta_logitq"])
    composite = z_cosine + float(config["lambda_mc"]) * z_mc + float(config["lambda_q"]) * z_q
    out["waveform_composite_raw"] = composite.astype(np.float32)
    out["waveform_score"] = apply_score_likelihood_ratio(composite, config["likelihood_ratio"])
    return out


def select_fusion_weights(
    validation: pd.DataFrame,
    require_all_positive: bool = False,
    waveform_zero: bool = False,
    objective: str = "retrieval",
) -> tuple[dict[str, float], pd.DataFrame]:
    if objective not in {"retrieval", "candidate"}:
        raise ValueError(f"Unknown fusion objective: {objective}")
    rows = []
    for waveform, time, sky in itertools.product(WEIGHT_GRID, repeat=3):
        if waveform == time == sky == 0:
            continue
        if require_all_positive and min(waveform, time, sky) <= 0:
            continue
        if waveform_zero and (waveform != 0 or min(time, sky) <= 0):
            continue
        weights = {"waveform": waveform, "time": time, "sky": sky}
        scores = score_vector(validation, weights)
        rows.append(
            {
                **weights,
                **retrieval_metrics(validation, scores),
                **pair_metrics(validation, scores),
                "weight_l2": math.sqrt(waveform * waveform + time * time + sky * sky),
            }
        )
    if objective == "retrieval":
        sort_columns = [
            "macro_r_at_10",
            "min_family_r_at_10",
            "average_precision",
            "precision_at_recall_0p5",
            "macro_r_at_1",
            "weight_l2",
        ]
    else:
        # A real-catalog shortlist is a rare-pair classification problem.
        # Average precision is prevalence-aware and threshold free; precision
        # at 50% recall provides an interpretable false-candidate tie-breaker.
        # Retrieval metrics remain secondary safeguards and are reported
        # independently under the retrieval objective.
        sort_columns = [
            "average_precision",
            "precision_at_recall_0p5",
            "macro_r_at_10",
            "min_family_r_at_10",
            "macro_r_at_1",
            "weight_l2",
        ]
    grid = pd.DataFrame(rows).sort_values(
        sort_columns,
        ascending=[False, False, False, False, False, True],
    ).reset_index(drop=True)
    grid.insert(0, "selection_objective", objective)
    best = grid.iloc[0]
    return {name: float(best[name]) for name in ("waveform", "time", "sky")}, grid


def evaluation_tables(
    frame: pd.DataFrame,
    methods: dict[str, dict[str, float]],
    deployment: str,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    query_rows = []
    metric_rows = []
    pair_rows = []
    for method, weights in methods.items():
        scores = score_vector(frame, weights)
        query = query_rank_rows(frame, scores, method)
        query.insert(0, "seed", int(seed))
        query.insert(0, "deployment", deployment)
        query_rows.append(query)
        metrics = retrieval_metrics(frame, scores)
        for subset in ("overall", "sis", "pm"):
            if subset == "overall":
                values = query["query_rank"].to_numpy(dtype=np.int32)
            else:
                values = query.loc[query["family"].str.lower() == subset, "query_rank"].to_numpy(dtype=np.int32)
            metric_rows.append(
                {
                    "deployment": deployment,
                    "seed": int(seed),
                    "method": method,
                    "subset": subset,
                    "r_at_1": float(np.mean(values <= 1)),
                    "r_at_5": float(np.mean(values <= 5)),
                    "r_at_10": float(np.mean(values <= 10)),
                    "median_rank": float(np.median(values)),
                    "n_queries": int(len(values)),
                }
            )
        pair_rows.append(
            {
                "deployment": deployment,
                "seed": int(seed),
                "method": method,
                "n_true_pairs": int(frame["is_true_pair"].sum()),
                "n_false_pairs": int((frame["is_true_pair"] == 0).sum()),
                **pair_metrics(frame, scores),
            }
        )
    return pd.concat(query_rows, ignore_index=True), pd.DataFrame(metric_rows), pd.DataFrame(pair_rows)


def _system_bootstrap_macro_r10(
    ranks: pd.DataFrame,
    draws: int,
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    samples = []
    by_family = {
        family: [part["query_rank"].to_numpy(dtype=np.int32) for _, part in frame.groupby("system_id")]
        for family in FAMILIES
        for frame in [ranks[ranks["family"] == family]]
    }
    for _ in range(int(draws)):
        family_recall = []
        for family in FAMILIES:
            systems = by_family[family]
            selected = rng.integers(0, len(systems), size=len(systems))
            values = np.concatenate([systems[index] for index in selected])
            family_recall.append(float(np.mean(values <= 10)))
        samples.append(float(np.mean(family_recall)))
    values = np.asarray(samples, dtype=np.float64)
    return {
        "draws": int(draws),
        "mean": float(np.mean(values)),
        "lower_95": float(np.quantile(values, 0.025)),
        "upper_95": float(np.quantile(values, 0.975)),
    }


def waveform_gate(
    validation: pd.DataFrame,
    weights_three: dict[str, float],
    weights_time_sky: dict[str, float],
    seed: int = 0,
) -> dict[str, Any]:
    waveform = retrieval_metrics(validation, validation["waveform_score"].to_numpy(dtype=np.float64))
    waveform_ranks = query_rank_rows(
        validation,
        validation["waveform_score"].to_numpy(dtype=np.float64),
        "waveform_only",
    )
    bootstrap = _system_bootstrap_macro_r10(waveform_ranks, draws=5000, seed=seed + 9300)
    chance_r10 = min(1.0, 10.0 / max(int(validation["event_count"].iloc[0]) - 1, 1))
    three_scores = score_vector(validation, weights_three)
    baseline_scores = score_vector(validation, weights_time_sky)
    three = {**retrieval_metrics(validation, three_scores), **pair_metrics(validation, three_scores)}
    baseline = {**retrieval_metrics(validation, baseline_scores), **pair_metrics(validation, baseline_scores)}
    legacy_absolute = bool(waveform["macro_r_at_10"] > 0.6 and waveform["min_family_r_at_10"] > 0.5)
    discrimination = bool(bootstrap["lower_95"] > chance_r10)
    incremental = bool(
        three["macro_r_at_10"] > baseline["macro_r_at_10"]
        and three["average_precision"] > baseline["average_precision"]
        and float(weights_three["waveform"]) > 0
    )
    return {
        "selection_split": "synthetic validation systems",
        "legacy_absolute_gate_rule": "waveform macro-family R@10 > 0.6 and minimum-family R@10 > 0.5",
        "legacy_absolute_gate_passed": legacy_absolute,
        "statistical_discrimination_rule": "system-cluster-bootstrap lower 95% bound for waveform macro R@10 exceeds random-catalog R@10 = 10/(N-1)",
        "statistical_discrimination_passed": discrimination,
        "random_catalog_r_at_10": chance_r10,
        "waveform_macro_r_at_10_system_bootstrap": bootstrap,
        "incremental_utility_rule": "validation-selected three-channel score has positive waveform weight and improves both macro R@10 and pair AUPRC over time+sky",
        "incremental_utility_passed": incremental,
        "waveform_metrics": waveform,
        "three_channel_metrics": three,
        "time_sky_metrics": baseline,
        "passed_for_primary_deployment": bool(discrimination and incremental),
        "gate_interpretation": "The legacy fixed 0.6/0.5 benchmark is retained transparently but is not treated as a literature-derived significance threshold. Primary eligibility uses statistically demonstrated discrimination plus incremental candidate-generation utility, both frozen before held-out test evaluation.",
    }


class ArrayCatalog(Dataset):
    def __init__(self, values: np.ndarray) -> None:
        self.values = values

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, index: int) -> torch.Tensor:
        return torch.from_numpy(pilot.prepare(np.asarray(self.values[index], dtype=np.float32), None, False))


def _effective_rank_reference(values: np.ndarray, sample_size: int, seed: int, draws: int = 500) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float32)
    sample_size = min(int(sample_size), len(values))
    rng = np.random.default_rng(seed)
    ranks = np.asarray(
        [effective_rank(values[rng.choice(len(values), size=sample_size, replace=False)]) for _ in range(draws)],
        dtype=np.float64,
    )
    return {
        "draws": int(draws),
        "sample_size": int(sample_size),
        "q01": float(np.quantile(ranks, 0.01)),
        "median": float(np.median(ranks)),
        "q99": float(np.quantile(ranks, 0.99)),
    }


def embed_real_events_unified(
    deployment: str,
    seed_dir: Path,
    shared_dir: Path,
    checkpoint: Path,
    waveform_config: dict[str, Any],
    validation_embeddings: np.ndarray,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    feature_dir = seed_dir / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    emb_path = feature_dir / "real_waveform_embeddings_unified_v7.parquet"
    pair_path = feature_dir / "real_waveform_similarity_unified_v7.parquet"
    audit_path = seed_dir / "results/real_waveform_deployment_audit_unified_v7.json"
    if emb_path.exists() and pair_path.exists() and audit_path.exists():
        return (
            pd.read_parquet(emb_path),
            pd.read_parquet(pair_path),
            json.loads(audit_path.read_text(encoding="utf-8")),
        )

    inputs, event_audit = v3.build_real_preprocessed_inputs(
        deployment, v3.SOURCES[deployment], shared_dir
    )
    available = event_audit["strict_h1l1_preprocessing_pass"].to_numpy(dtype=bool)
    model, checkpoint_payload = load_unified_model(checkpoint)
    real_embedding, real_prediction_std = embed_catalog(
        model, ArrayCatalog(np.asarray(inputs[available])), batch_size=16
    )
    del model
    torch.cuda.empty_cache()
    embedding_dim = int(real_embedding.shape[1])
    all_embedding = np.full((len(event_audit), embedding_dim), np.nan, dtype=np.float32)
    all_prediction_std = np.full((len(event_audit), 2), np.nan, dtype=np.float32)
    all_embedding[available] = real_embedding
    all_prediction_std[available] = real_prediction_std
    target_mean = np.asarray(checkpoint_payload["target_mean"], dtype=np.float64)
    target_std = np.asarray(checkpoint_payload["target_std"], dtype=np.float64)
    physical_prediction = all_prediction_std * target_std + target_mean

    embeddings = event_audit.copy()
    embeddings["unified_embedding"] = [
        row.tolist() if np.isfinite(row).all() else None for row in all_embedding
    ]
    embeddings["waveform_pred_log_chirp_mass"] = physical_prediction[:, 0]
    embeddings["waveform_pred_logit_mass_ratio"] = physical_prediction[:, 1]
    embeddings["waveform_pred_chirp_mass_detector"] = np.exp(physical_prediction[:, 0])
    embeddings["waveform_pred_mass_ratio"] = 1.0 / (1.0 + np.exp(-physical_prediction[:, 1]))
    embeddings.to_parquet(emb_path, index=False)

    ii, jj = np.triu_indices(len(embeddings), k=1)
    both = available[ii] & available[jj]
    cosine = np.full(len(ii), np.nan, dtype=np.float32)
    delta_mc = np.full(len(ii), np.nan, dtype=np.float32)
    delta_q = np.full(len(ii), np.nan, dtype=np.float32)
    cosine[both] = np.sum(all_embedding[ii[both]] * all_embedding[jj[both]], axis=1)
    delta_mc[both] = np.abs(all_prediction_std[ii[both], 0] - all_prediction_std[jj[both], 0])
    delta_q[both] = np.abs(all_prediction_std[ii[both], 1] - all_prediction_std[jj[both], 1])
    pairs = pd.DataFrame(
        {
            "idx_i": ii.astype(np.int32),
            "idx_j": jj.astype(np.int32),
            "event_i": embeddings["event_name"].astype(str).to_numpy()[ii],
            "event_j": embeddings["event_name"].astype(str).to_numpy()[jj],
            "waveform_available": both,
            "waveform_embedding_cosine": cosine,
            "waveform_abs_delta_logmc_std": delta_mc,
            "waveform_abs_delta_logitq_std": delta_q,
        }
    )
    scored = pairs.loc[both].copy()
    scored = apply_waveform_channel(scored, waveform_config)
    pairs["waveform_composite_raw"] = np.nan
    pairs["waveform_score"] = 0.0
    pairs.loc[both, "waveform_composite_raw"] = scored["waveform_composite_raw"].to_numpy()
    pairs.loc[both, "waveform_score"] = scored["waveform_score"].to_numpy()
    pairs.to_parquet(pair_path, index=False)

    reference = _effective_rank_reference(validation_embeddings, int(available.sum()), seed + 7100)
    valid_score = pairs.loc[both, "waveform_score"].to_numpy(dtype=np.float64)
    rank = effective_rank(real_embedding) if len(real_embedding) else 0.0
    audit = {
        "deployment": deployment,
        "strict_h1l1_events": int(available.sum()),
        "strict_h1l1_pairs": int(both.sum()),
        "embedding_dimension": embedding_dim,
        "real_embedding_effective_rank": rank,
        "validation_effective_rank_reference": reference,
        "effective_rank_passed": bool(rank >= reference["q01"]),
        "waveform_score_min": float(np.min(valid_score)) if len(valid_score) else None,
        "waveform_score_max": float(np.max(valid_score)) if len(valid_score) else None,
        "waveform_score_std": float(np.std(valid_score)) if len(valid_score) else None,
        "waveform_score_unique_rounded_1e6": int(len(np.unique(np.round(valid_score, 6)))) if len(valid_score) else 0,
        "constant_score_failure": bool(len(valid_score) == 0 or np.std(valid_score) < 1e-6),
        "finite_prediction_fraction": float(np.mean(np.isfinite(real_prediction_std))),
        "predicted_detector_chirp_mass_range": [
            float(np.nanmin(np.exp(physical_prediction[:, 0]))),
            float(np.nanmax(np.exp(physical_prediction[:, 0]))),
        ],
        "predicted_mass_ratio_range": [
            float(np.nanmin(1.0 / (1.0 + np.exp(-physical_prediction[:, 1])))),
            float(np.nanmax(1.0 / (1.0 + np.exp(-physical_prediction[:, 1])))),
        ],
        "silent_zero_fill": False,
        "passed": bool(rank >= reference["q01"] and len(valid_score) and np.std(valid_score) >= 1e-6),
        "rule": "Real-event embedding effective rank must exceed the validation q01 reference and calibrated pair scores must be nonconstant; strict H1/L1 validity is checked before any fill operation.",
    }
    write_json(audit_path, audit)
    return embeddings, pairs, audit


def _rank_real_pairs(frame: pd.DataFrame, weights: dict[str, float], method: str) -> pd.DataFrame:
    out = frame.copy()
    out["waveform_contribution"] = float(weights["waveform"]) * out["waveform_score"].to_numpy(dtype=np.float64)
    out["time_contribution"] = float(weights["time"]) * out["time_score"].to_numpy(dtype=np.float64)
    out["sky_contribution"] = float(weights["sky"]) * out["sky_score"].to_numpy(dtype=np.float64)
    out["final_score"] = out[["waveform_contribution", "time_contribution", "sky_contribution"]].sum(axis=1)
    # All v7 channels are globally calibrated, symmetric pair evidence. There
    # is no row-wise standardization, so the two directed scores are equal.
    out["final_score_i_to_j"] = out["final_score"]
    out["final_score_j_to_i"] = out["final_score"]
    out["unordered_max_score"] = out["final_score"]
    out["unordered_mean_score"] = out["final_score"]
    out["method"] = method
    out = out.sort_values("final_score", ascending=False, kind="stable").reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1, dtype=np.int32)
    return out


def score_real_catalog(
    deployment: str,
    seed_dir: Path,
    shared_dir: Path,
    methods: dict[str, dict[str, float]],
    checkpoint: Path,
    waveform_config: dict[str, Any],
    time_calibration: dict[str, Any],
    validation_embeddings: np.ndarray,
    seed: int,
) -> tuple[dict[str, dict[str, pd.DataFrame]], dict[str, Any]]:
    primary = v3.primary_manifest(v3.SOURCES[deployment])
    embeddings, waveform, audit = embed_real_events_unified(
        deployment,
        seed_dir,
        shared_dir,
        checkpoint,
        waveform_config,
        validation_embeddings,
        seed,
    )
    sky = pd.read_parquet(v3.SOURCES[deployment] / "features/real_sky_overlap.parquet")
    expected_i = primary["event_name"].astype(str).to_numpy()[sky["idx_i"].to_numpy(dtype=np.int32)]
    expected_j = primary["event_name"].astype(str).to_numpy()[sky["idx_j"].to_numpy(dtype=np.int32)]
    if not np.array_equal(expected_i, sky["event_i"].astype(str).to_numpy()):
        raise RuntimeError(f"{deployment} sky idx_i does not match the primary manifest")
    if not np.array_equal(expected_j, sky["event_j"].astype(str).to_numpy()):
        raise RuntimeError(f"{deployment} sky idx_j does not match the primary manifest")
    real = sky.merge(waveform, on=["idx_i", "idx_j", "event_i", "event_j"], how="left", validate="one_to_one")
    ii = real["idx_i"].to_numpy(dtype=np.int32)
    jj = real["idx_j"].to_numpy(dtype=np.int32)
    gps = primary["gps_time"].to_numpy(dtype=np.float64)
    real["delta_t_days"] = np.abs(gps[ii] - gps[jj]) / SECONDS_PER_DAY
    real["time_score"] = apply_time_likelihood_ratio(real["delta_t_days"], time_calibration)
    npix = 12 * real["common_nside"].to_numpy(dtype=np.int64) ** 2
    real["sky_bayes_factor"] = npix * real["raw_posterior_overlap"].to_numpy(dtype=np.float64)
    real["sky_score"] = np.log(np.maximum(real["sky_bayes_factor"], 1e-300))
    real["sky_log_cosine_overlap"] = np.log(np.maximum(real["cosine_overlap"].to_numpy(dtype=np.float64), 1e-300))
    event_available = embeddings["strict_h1l1_preprocessing_pass"].to_numpy(dtype=bool)
    event_ood = embeddings["is_ood_for_bbh_encoder"].to_numpy(dtype=bool)
    real["waveform_available"] = event_available[ii] & event_available[jj]
    real["pair_has_ood"] = event_ood[ii] | event_ood[jj]
    real["strict_h1l1_bbh_pair"] = real["waveform_available"] & ~real["pair_has_ood"]
    real["waveform_score"] = real["waveform_score"].fillna(0.0)
    real.loc[~real["waveform_available"], "waveform_score"] = 0.0
    real["waveform_neutral_missing"] = ~real["waveform_available"]
    object_class = embeddings["object_class"].astype(str).to_numpy()
    real["object_class_i"] = object_class[ii]
    real["object_class_j"] = object_class[jj]
    if "network_snr" in primary:
        snr = primary["network_snr"].to_numpy(dtype=np.float64)
        real["network_snr_i_audit_only"] = snr[ii]
        real["network_snr_j_audit_only"] = snr[jj]

    result_dir = seed_dir / "results"
    outputs: dict[str, dict[str, pd.DataFrame]] = {}
    for method, weights in methods.items():
        all_catalog = _rank_real_pairs(real, weights, method)
        strict = _rank_real_pairs(real[real["strict_h1l1_bbh_pair"]].copy(), weights, method)
        all_catalog.to_parquet(result_dir / f"real_pair_scores_all_catalog_{method}_v7.parquet", index=False)
        strict.to_parquet(result_dir / f"real_pair_scores_strict_h1l1_bbh_{method}_v7.parquet", index=False)
        all_catalog.head(100).to_csv(result_dir / f"candidate_shortlist_all_catalog_{method}_v7.csv", index=False)
        strict.head(100).to_csv(result_dir / f"candidate_shortlist_strict_h1l1_bbh_{method}_v7.csv", index=False)
        outputs[method] = {"all": all_catalog, "strict": strict}
    return outputs, audit
