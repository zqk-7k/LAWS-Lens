from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path


def _read_summary(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _flatten_row(run_name: str, scale: int, summary: dict, cmd: list[str]) -> dict:
    cfg = summary.get("config", {})
    sizes = summary.get("sizes", {})
    timing = summary.get("timing", {})
    test = summary.get("test", {})
    cand = summary.get("test_candidates", {})
    total_events = 2 * int(sizes.get("lensed_total", scale)) + int(sizes.get("unlensed_total", scale))
    train_items = int(sizes.get("train_lensed", 0)) + int(sizes.get("train_unlensed", 0))
    train_s = float(timing.get("train_s", 0.0))
    total_s = float(timing.get("total_s", 0.0))
    return {
        "run": run_name,
        "scale": scale,
        "model_type": cfg.get("model_type"),
        "data_mode": cfg.get("data_mode"),
        "backbone": cfg.get("backbone"),
        "epochs": cfg.get("epochs"),
        "batch_size": cfg.get("batch_size"),
        "target_len": cfg.get("target_len"),
        "stride": cfg.get("stride"),
        "num_workers": timing.get("num_workers"),
        "pin_memory": timing.get("pin_memory"),
        "amp": timing.get("amp"),
        "amp_dtype": timing.get("amp_dtype"),
        "total_events": total_events,
        "train_items": train_items,
        "train_s": train_s,
        "mean_epoch_s": timing.get("mean_epoch_s"),
        "mean_pairs_per_s": timing.get("mean_pairs_per_s"),
        "total_s": total_s,
        "events_per_total_s": float(total_events / total_s) if total_s > 0 else 0.0,
        "test_r@1": test.get("r@1"),
        "test_r@5": test.get("r@5"),
        "test_r@10": test.get("r@10"),
        "test_mrr": test.get("mrr"),
        "test_precision": test.get("precision"),
        "test_recall": test.get("recall"),
        "test_f1": test.get("f1"),
        "candidate_pair_recall": cand.get("candidate_pair_recall", cand.get("pair_recall")),
        "candidate_edges": cand.get("candidate_edges", cand.get("edges")),
        "command": " ".join(cmd),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Study training speed and accuracy across GW catalog data scales.")
    ap.add_argument("--data-root", type=Path, default=Path("/root/autodl-tmp/qkzhang_gwaug_20260522_162031"))
    ap.add_argument("--out-root", type=Path, default=Path("runs/data_scale_speed_study"))
    ap.add_argument("--scales", nargs="+", type=int, default=[500, 1000, 2500, 5000, 10000])
    ap.add_argument("--model-types", nargs="+", choices=["SIS", "PM"], default=["SIS", "PM"])
    ap.add_argument("--data-modes", nargs="+", choices=["pure", "noisy"], default=["noisy", "pure"])
    ap.add_argument("--backbone", choices=["cnn", "inceptiontime"], default="inceptiontime")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--eval-batch-size", type=int, default=512)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--pin-memory", action="store_true")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--amp-dtype", choices=["fp16", "bf16"], default="bf16")
    ap.add_argument("--compile-model", action="store_true")
    ap.add_argument("--target-len", type=int, default=8192)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--width-scale", type=float, default=2.0)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--emb-dim", type=int, default=128)
    ap.add_argument("--candidate-topk", type=int, default=10)
    ap.add_argument("--calibration-iters", type=int, default=600)
    ap.add_argument("--use-hilbert", action="store_true")
    ap.add_argument("--enable-hard-neg", action="store_true")
    ap.add_argument("--export-candidates", action="store_true", help="Export candidate CSVs; disabled by default to reduce I/O in speed studies.")
    ap.add_argument("--resume", action="store_true", help="Skip runs with an existing summary.json and only rebuild summary tables.")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    train_py = Path(__file__).resolve().parent / "08_match_first_train.py"
    args.out_root.mkdir(parents=True, exist_ok=True)
    rows = []
    commands = []

    for scale, model_type, data_mode in itertools.product(args.scales, args.model_types, args.data_modes):
        run_name = f"{args.backbone}_{model_type.lower()}_{data_mode}_n{scale}_ep{args.epochs}"
        out_dir = args.out_root / run_name
        cmd = [
            sys.executable,
            str(train_py),
            "--data-root", str(args.data_root),
            "--model-type", model_type,
            "--data-mode", data_mode,
            "--backbone", args.backbone,
            "--lensed-limit", str(scale),
            "--unlensed-limit", str(scale),
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--eval-batch-size", str(args.eval_batch_size),
            "--target-len", str(args.target_len),
            "--stride", str(args.stride),
            "--lr", str(args.lr),
            "--width-scale", str(args.width_scale),
            "--d-model", str(args.d_model),
            "--emb-dim", str(args.emb_dim),
            "--candidate-topk", str(args.candidate_topk),
            "--calibration-iters", str(args.calibration_iters),
            "--out-dir", str(out_dir),
        ]
        if args.num_workers:
            cmd.extend(["--num-workers", str(args.num_workers)])
        if args.pin_memory:
            cmd.append("--pin-memory")
        if args.amp:
            cmd.extend(["--amp", "--amp-dtype", args.amp_dtype])
        if args.compile_model:
            cmd.append("--compile-model")
        if args.use_hilbert:
            cmd.append("--use-hilbert")
        if args.enable_hard_neg:
            cmd.append("--enable-hard-neg")
        if not args.export_candidates:
            cmd.append("--no-export-candidates")
        if args.cpu:
            cmd.append("--cpu")

        commands.append(" ".join(cmd))
        summary_path = out_dir / "summary.json"
        if args.resume and summary_path.exists():
            print(f"[skip] {run_name}", flush=True)
        else:
            print(" ".join(cmd), flush=True)
            if not args.dry_run:
                subprocess.run(cmd, check=True)
        summary = _read_summary(summary_path)
        if summary:
            rows.append(_flatten_row(run_name, scale, summary, cmd))

    (args.out_root / "commands.txt").write_text("\n".join(commands) + "\n", encoding="utf-8")
    if rows:
        import pandas as pd
        df = pd.DataFrame(rows).sort_values(["scale", "model_type", "data_mode"])
        df.to_csv(args.out_root / "summary.csv", index=False)
        compact_cols = [
            "run", "scale", "model_type", "data_mode", "epochs", "total_events", "train_items",
            "train_s", "mean_epoch_s", "mean_pairs_per_s", "total_s", "events_per_total_s",
            "test_r@1", "test_r@5", "test_r@10", "test_f1", "candidate_pair_recall", "candidate_edges",
        ]
        df[[c for c in compact_cols if c in df.columns]].to_csv(args.out_root / "summary_compact.csv", index=False)
        print(df[[c for c in compact_cols if c in df.columns]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
