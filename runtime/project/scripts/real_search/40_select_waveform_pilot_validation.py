#!/usr/bin/env python3
"""Compare waveform pilot checkpoints on one frozen synthetic validation catalog.

The script never constructs the held-out test catalog and never reads real-event
PE products.  Its sole purpose is to freeze the auxiliary-training setting and
the waveform-internal compatibility coefficients before the formal multi-seed
run starts.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.real_search.physical_common import effective_rank, write_json  # noqa: E402
from scripts.real_search.unified_v6_common import (  # noqa: E402
    FAMILIES,
    MixedCatalog,
    apply_waveform_channel,
    embed_catalog,
    fit_waveform_channel,
    load_mixed_data,
    load_unified_model,
    pair_metrics,
    retrieval_metrics,
)


def validation_waveform_frame(
    embedding: np.ndarray,
    prediction: np.ndarray,
    catalog: MixedCatalog,
) -> pd.DataFrame:
    """Build the complete unordered waveform-only validation pair table."""
    cosine = np.asarray(embedding @ embedding.T, dtype=np.float64)
    ii, jj = np.triu_indices(len(catalog), k=1)
    labels = catalog.partner[ii] == jj
    family = np.where(labels, catalog.query_family[ii], "background")
    return pd.DataFrame(
        {
            "idx_i": ii.astype(np.int32),
            "idx_j": jj.astype(np.int32),
            "is_true_pair": labels.astype(np.int8),
            "true_pair_family": family,
            "waveform_embedding_cosine": cosine[ii, jj].astype(np.float32),
            "waveform_abs_delta_logmc_std": np.abs(
                prediction[ii, 0] - prediction[jj, 0]
            ).astype(np.float32),
            "waveform_abs_delta_logitq_std": np.abs(
                prediction[ii, 1] - prediction[jj, 1]
            ).astype(np.float32),
            "event_count": int(len(catalog)),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-root", type=Path, required=True)
    parser.add_argument("--source-bank", type=Path, required=True)
    parser.add_argument("--pilot-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--samples", type=int, default=600)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    checkpoints = sorted(args.pilot_root.glob("*/validation_selected_model.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"No completed pilot checkpoints under {args.pilot_root}")

    data = load_mixed_data(args.seed_root, args.source_bank, args.seed, args.samples)
    catalog = MixedCatalog(data, "val")
    expected_true = sum(len(data[family].parts["lensed"]["val"]) for family in FAMILIES)
    rows: list[dict] = []
    grids: list[pd.DataFrame] = []

    for checkpoint in checkpoints:
        model, payload = load_unified_model(checkpoint)
        summary_path = checkpoint.parent / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        q_loss_weight = float(payload.get("q_loss_weight", 1.0))
        checkpoint_q_coefficient = float(
            summary["best_noisy_validation"].get("hybrid_intrinsic_q_coefficient", 0.0)
        )
        checkpoint_selection_valid = bool(
            q_loss_weight > 0 or checkpoint_q_coefficient == 0.0
        )
        embedding, prediction = embed_catalog(model, catalog, batch_size=48)
        del model
        torch.cuda.empty_cache()
        frame = validation_waveform_frame(embedding, prediction, catalog)
        config, grid = fit_waveform_channel(
            frame,
            allow_q_feature=q_loss_weight > 0,
        )
        scored = apply_waveform_channel(frame, config)
        cosine_metrics = {
            **retrieval_metrics(frame, frame["waveform_embedding_cosine"].to_numpy()),
            **pair_metrics(frame, frame["waveform_embedding_cosine"].to_numpy()),
        }
        composite_metrics = {
            **retrieval_metrics(scored, scored["waveform_score"].to_numpy()),
            **pair_metrics(scored, scored["waveform_score"].to_numpy()),
        }
        rows.append(
            {
                "pilot": checkpoint.parent.name,
                "checkpoint": str(checkpoint),
                "aux_weight": float(payload["aux_weight"]),
                "q_loss_weight": q_loss_weight,
                "checkpoint_selection_valid": checkpoint_selection_valid,
                "checkpoint_selection_q_coefficient": checkpoint_q_coefficient,
                "n_events": int(len(catalog)),
                "n_true_pairs": int(frame["is_true_pair"].sum()),
                "lambda_mc": float(config["lambda_mc"]),
                "lambda_q": float(config["lambda_q"]),
                "embedding_effective_rank": float(effective_rank(embedding)),
                **{f"cosine_{key}": value for key, value in cosine_metrics.items()},
                **{f"composite_{key}": value for key, value in composite_metrics.items()},
                "pilot_internal_best_hybrid_r_at_10": float(
                    summary["best_noisy_validation"].get(
                        "hybrid_r_at_10",
                        summary["best_noisy_validation"].get("overall_r_at_10", np.nan),
                    )
                ),
            }
        )
        current = grid.copy()
        current.insert(0, "pilot", checkpoint.parent.name)
        grids.append(current)

    table = pd.DataFrame(rows).sort_values(
        [
            "checkpoint_selection_valid",
            "composite_macro_r_at_10",
            "composite_min_family_r_at_10",
            "composite_average_precision",
            "composite_macro_r_at_1",
            "embedding_effective_rank",
        ],
        ascending=[False, False, False, False, False, False],
    ).reset_index(drop=True)
    table.insert(0, "validation_selection_rank", np.arange(1, len(table) + 1))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out_dir / "pilot_selection_validation_only.csv", index=False)
    pd.concat(grids, ignore_index=True).to_csv(
        args.out_dir / "pilot_waveform_internal_grid_validation_only.csv", index=False
    )
    best = table.iloc[0]
    payload = {
        "selection_scope": "one frozen synthetic O3 real-noise validation catalog",
        "heldout_test_instantiated": False,
        "real_catalog_or_pe_used": False,
        "selection_order": [
            "checkpoint-selection validity (an unsupervised q head cannot select a checkpoint through q compatibility)",
            "composite_macro_r_at_10 descending",
            "composite_min_family_r_at_10 descending",
            "composite_average_precision descending",
            "composite_macro_r_at_1 descending",
            "embedding_effective_rank descending",
        ],
        "n_events": int(len(catalog)),
        "n_true_pairs": int(expected_true),
        "selected_pilot": str(best["pilot"]),
        "selected_aux_weight": float(best["aux_weight"]),
        "selected_q_loss_weight": float(best["q_loss_weight"]),
        "selected_lambda_mc_for_audit_only": float(best["lambda_mc"]),
        "selected_lambda_q_for_audit_only": float(best["lambda_q"]),
        "note": (
            "The auxiliary-training setting is frozen here. Formal seeds refit "
            "waveform-internal lambda coefficients on their own validation split."
        ),
        "selected_validation_metrics": best.to_dict(),
    }
    write_json(args.out_dir / "pilot_selection_validation_only.json", payload)
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
