from __future__ import annotations

import argparse
from pathlib import Path
import sys

# 允许直接用 `python scripts/08_match_first_train.py` 从仓库根目录之外启动。
# 这里把项目根目录加入 import 路径，后面才能导入 matchgw 包。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from matchgw import MatchRunConfig
from matchgw.pipeline import run_train_eval


def main() -> None:
    # 这是本轮 GW-augmented 全量实验的训练入口。
    # 主要流程：读取 match-style npy 数据 -> 训练 Siamese/NT-Xent encoder ->
    # 在验证/测试集上做 Top-K 候选检索、最大权匹配和候选概率校准。
    ap = argparse.ArgumentParser(description="Train match-first Siamese GW matcher in the gw-catalog repo.")
    ap.add_argument("--data-root", type=Path, default=Path("/root/autodl-tmp/qkzhang"))
    ap.add_argument("--model-type", choices=["SIS", "PM"], default="SIS")
    ap.add_argument("--data-mode", choices=["pure", "noisy"], default="noisy")
    ap.add_argument("--out-dir", type=Path, default=Path("runs/match_first"))
    ap.add_argument("--backbone", choices=["cnn", "inceptiontime", "attnresnet", "dilatedresnet", "inceptionattn", "convnext1d", "seresnet", "cbamresnet", "gatedtcn", "patchtst", "rocket", "timesnetlite"], default="cnn")
    ap.add_argument("--lensed-limit", type=int, default=2500)
    ap.add_argument("--unlensed-limit", type=int, default=2500)
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
    ap.add_argument("--use-hilbert", action="store_true")
    ap.add_argument("--use-pure-aux", action="store_true", help="Train noisy runs with auxiliary clean/noisy positive pairs.")
    ap.add_argument("--preprocess", choices=["none", "bandpass", "whiten", "whiten_bandpass", "multiband"], default="none")
    ap.add_argument("--bandpass-low", type=int, default=40)
    ap.add_argument("--bandpass-high", type=int, default=580)
    ap.add_argument("--whiten-kernel", type=int, default=33)
    ap.add_argument("--aug-roll", type=int, default=128)
    ap.add_argument("--aug-scale", type=float, default=0.10)
    ap.add_argument("--aug-noise", type=float, default=0.01)
    ap.add_argument("--no-aug-flip", action="store_true")
    ap.add_argument("--hard-neg-epochs", type=int, default=4)
    ap.add_argument("--hard-neg-min-score", type=float, default=0.70)
    ap.add_argument("--enable-hard-neg", action="store_true")
    ap.add_argument("--candidate-topk", type=int, default=10)
    ap.add_argument("--p-low", type=float, default=0.20)
    ap.add_argument("--p-high", type=float, default=0.80)
    ap.add_argument("--calibration-iters", type=int, default=600)
    ap.add_argument("--no-export-candidates", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    # 将命令行参数集中到 MatchRunConfig，避免训练、评估、数据读取各自维护一套参数。
    cfg = MatchRunConfig(
        data_root=args.data_root,
        model_type=args.model_type,
        data_mode=args.data_mode,
        out_dir=args.out_dir,
        backbone=args.backbone,
        lensed_limit=args.lensed_limit,
        unlensed_limit=args.unlensed_limit,
        epochs=args.epochs,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        amp=args.amp,
        amp_dtype=args.amp_dtype,
        compile_model=args.compile_model,
        target_len=args.target_len,
        stride=args.stride,
        lr=args.lr,
        width_scale=args.width_scale,
        d_model=args.d_model,
        emb_dim=args.emb_dim,
        use_hilbert=args.use_hilbert,
        use_pure_aux=args.use_pure_aux,
        preprocess=args.preprocess,
        bandpass_low=args.bandpass_low,
        bandpass_high=args.bandpass_high,
        whiten_kernel=args.whiten_kernel,
        aug_roll=args.aug_roll,
        aug_scale=args.aug_scale,
        aug_noise=args.aug_noise,
        aug_flip=not args.no_aug_flip,
        hard_neg_enable=args.enable_hard_neg,
        hard_neg_epochs=args.hard_neg_epochs,
        hard_neg_min_score=args.hard_neg_min_score,
        candidate_topk=args.candidate_topk,
        p_low=args.p_low,
        p_high=args.p_high,
        calibration_iters=args.calibration_iters,
        export_candidates=not args.no_export_candidates,
    )
    # 真正的训练与评估实现位于 matchgw/pipeline.py。
    run_train_eval(cfg, cpu=args.cpu)


if __name__ == "__main__":
    main()
