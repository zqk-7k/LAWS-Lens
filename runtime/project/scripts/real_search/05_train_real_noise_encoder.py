from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.real_search.common import default_run_dir, ensure_run_dirs, utc_now, write_json


FAMILIES = ("SIS", "PM")


def train_command(run_dir: Path, family: str, epochs: int, samples: int, force: bool) -> list[str] | None:
    out_dir = run_dir / "waveform_gate" / f"{family.lower()}_real_noise_inceptiontime_ep{epochs}_clean"
    if out_dir.joinpath("summary.json").exists() and not force:
        return None
    return [
        "/root/miniconda3/bin/python",
        "scripts/08_match_first_train.py",
        "--data-root",
        str(run_dir / "data" / "real_noise_injections" / "matchroots" / "LIGO"),
        "--model-type",
        family,
        "--data-mode",
        "noisy",
        "--out-dir",
        str(out_dir),
        "--backbone",
        "inceptiontime",
        "--lensed-limit",
        str(samples),
        "--unlensed-limit",
        str(samples),
        "--epochs",
        str(epochs),
        "--batch-size",
        "64",
        "--eval-batch-size",
        "256",
        "--target-len",
        "8192",
        "--stride",
        "2",
        "--preprocess",
        "bandpass",
        "--bandpass-low",
        "40",
        "--bandpass-high",
        "580",
        "--candidate-topk",
        "10",
    ]


def load_summary(run_dir: Path, family: str, epochs: int) -> dict[str, Any]:
    path = run_dir / "waveform_gate" / f"{family.lower()}_real_noise_inceptiontime_ep{epochs}_clean" / "summary.json"
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def metric_row(family: str, split: str, summary: dict[str, Any]) -> dict[str, Any]:
    metrics = summary.get(split, {})
    cand = summary.get(f"{split}_candidates", {})
    hist = summary.get("history", [])
    return {
        "family": family,
        "split": split,
        "r_at_1": metrics.get("r@1"),
        "r_at_5": metrics.get("r@5"),
        "r_at_10": metrics.get("r@10"),
        "mrr": metrics.get("mrr"),
        "median_true_rank": metrics.get("median_true_rank"),
        "pair_precision_at_tuned": metrics.get("precision"),
        "pair_recall_at_tuned": metrics.get("recall"),
        "candidate_pair_recall_top10": cand.get("candidate_pair_recall"),
        "candidate_edges_top10": cand.get("candidate_edges"),
        "first_loss": hist[0]["loss"] if hist else None,
        "last_loss": hist[-1]["loss"] if hist else None,
    }


def save_gate_figures(run_dir: Path, summaries: dict[str, dict[str, Any]]) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for family, summary in summaries.items():
        hist = pd.DataFrame(summary.get("history", []))
        if not hist.empty:
            ax.plot(hist["epoch"], hist["loss"], marker="o", ms=3, label=family)
    ax.set_xlabel("epoch", fontweight="bold")
    ax.set_ylabel("NT-Xent loss", fontweight="bold")
    ax.set_title("Real-noise injection Gate-1 training")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(run_dir / "figures" / f"waveform_gate_training_loss.{ext}", dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=default_run_dir())
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--samples-per-family", type=int, default=600)
    parser.add_argument("--train-missing", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    run_dir = args.run_dir
    ensure_run_dirs(run_dir)
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    if args.train_missing:
        for family in FAMILIES:
            cmd = train_command(run_dir, family, args.epochs, args.samples_per_family, args.force)
            if cmd is None:
                continue
            log_path = logs_dir / f"waveform_gate_{family.lower()}_ep{args.epochs}.log"
            with log_path.open("w", encoding="utf-8") as log:
                subprocess.run(cmd, cwd=Path(__file__).resolve().parents[2], check=True, stdout=log, stderr=subprocess.STDOUT)

    summaries = {family: load_summary(run_dir, family, args.epochs) for family in FAMILIES}
    rows = []
    for family, summary in summaries.items():
        rows.append(metric_row(family, "val", summary))
        rows.append(metric_row(family, "test", summary))
    table = pd.DataFrame(rows)
    table.to_csv(run_dir / "results" / "waveform_gate1_metrics.csv", index=False)

    test = table[table["split"] == "test"].copy()
    min_test_r10 = float(test["r_at_10"].min())
    macro_test_r10 = float(test["r_at_10"].mean())
    # This gate only decides whether the trained encoder is meaningful on held-out
    # real-noise injections. Real-event fusion is controlled separately because it
    # requires reliable on-source embedding for all events.
    passed = bool(min_test_r10 >= 0.50 and macro_test_r10 >= 0.60)
    payload = {
        "generated_at_utc": utc_now(),
        "gate": "Gate-1 waveform retrieval on held-out real-noise injection catalog",
        "samples_per_family": int(args.samples_per_family),
        "epochs": int(args.epochs),
        "pass_threshold": {"min_family_test_r_at_10": 0.50, "macro_test_r_at_10": 0.60},
        "passed": passed,
        "macro_test_r_at_10": macro_test_r10,
        "min_family_test_r_at_10": min_test_r10,
        "metrics_csv": str(run_dir / "results" / "waveform_gate1_metrics.csv"),
        "model_paths": {
            family: str(run_dir / "waveform_gate" / f"{family.lower()}_real_noise_inceptiontime_ep{args.epochs}_clean" / "model.pt")
            for family in FAMILIES
        },
        "note": "Passing Gate-1 means waveform encoder is informative on held-out real-noise injections. It is not a real-event lensing claim.",
    }
    write_json(run_dir / "results" / "waveform_gate1.json", payload)
    save_gate_figures(run_dir, summaries)
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
