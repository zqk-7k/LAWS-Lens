from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class MatchRunConfig:
    # 统一保存一次实验的所有配置：数据位置、模型结构、训练参数、
    # 候选检索参数和概率校准参数。pipeline 里的各模块都只读这个配置对象。
    # data_root 指向 match-style 数据根目录。
    # 原始 match 数据通常是 /root/autodl-tmp/qkzhang；
    # 本轮新数据是 /root/autodl-tmp/qkzhang_gwaug_20260522_162031。
    data_root: Path = Path("/root/autodl-tmp/qkzhang")
    model_type: str = "SIS"
    data_mode: str = "noisy"
    out_dir: Path = Path("runs/match_first")
    # backbone 可选 cnn、inceptiontime 或 attnresnet；本轮最好结果使用 inceptiontime。
    backbone: str = "cnn"
    # 从 98304 点长波形裁剪尾部窗口，再按 stride 下采样；默认输入模型长度为 4096。
    target_len: int = 8192
    stride: int = 2
    lensed_limit: int | None = 2500
    unlensed_limit: int | None = 2500
    seed: int = 42
    train_frac: float = 0.70
    val_frac: float = 0.15
    batch_size: int = 128
    eval_batch_size: int = 512
    # DataLoader/训练加速参数：默认保持原行为，实验时可打开。
    num_workers: int = 0
    pin_memory: bool = False
    amp: bool = False
    amp_dtype: str = "bf16"
    compile_model: bool = False
    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 1e-4
    tau: float = 0.07
    emb_dim: int = 128
    d_model: int = 256
    width_scale: float = 2.0
    aug_roll: int = 128
    aug_scale: float = 0.10
    aug_noise: float = 0.01
    aug_flip: bool = True
    use_hilbert: bool = False
    use_pure_aux: bool = False
    # waveform-only 频域预处理；不使用任何源参数或 lens 参数。
    preprocess: str = "none"
    bandpass_low: int = 40
    bandpass_high: int = 580
    whiten_kernel: int = 33
    coarse_topk: int = 10
    coarse_min_score: float | None = 0.85
    coarse_mutual: bool = False
    reciprocal_rank_max: int | None = 3
    row_min_score: float | None = 0.70
    row_min_margin: float | None = 0.0
    edge_rank_bonus: float = 0.0
    tune_for: str = "f1"
    hard_neg_enable: bool = False
    hard_neg_topk: int = 10
    hard_neg_min_score: float = 0.70
    hard_neg_per_anchor: int = 2
    hard_neg_epochs: int = 4
    hard_neg_lr: float = 3e-4
    hard_neg_margin: float = 0.45
    hard_neg_weight: float = 0.25
    # candidate_topk 控制最终导出的候选边数量，论文里对应 Top-K candidate retrieval。
    candidate_topk: int = 10
    candidate_min_score: float | None = None
    candidate_mutual: bool = False
    candidate_reciprocal_rank_max: int | None = None
    p_low: float = 0.20
    p_high: float = 0.80
    calibration_bins: int = 10
    calibration_l2: float = 1e-3
    calibration_lr: float = 0.05
    calibration_iters: int = 600
    export_candidates: bool = True

    @property
    def model_backbone(self) -> str:
        value = self.backbone.lower()
        if value not in {"cnn", "inceptiontime", "attnresnet", "dilatedresnet", "inceptionattn", "convnext1d", "seresnet", "cbamresnet", "gatedtcn", "patchtst", "rocket", "timesnetlite"}:
            raise ValueError(f"unsupported backbone: {self.backbone!r}")
        return value

    @property
    def family(self) -> str:
        value = self.model_type.upper()
        if value not in {"SIS", "PM"}:
            raise ValueError(f"model_type must be SIS or PM, got {self.model_type!r}")
        return value

    @property
    def mode(self) -> str:
        value = self.data_mode.lower()
        if value not in {"pure", "noisy"}:
            raise ValueError(f"data_mode must be pure or noisy, got {self.data_mode!r}")
        return value

    @property
    def source_dir(self) -> Path:
        # match 数据按 SIS_data_0222 / PM_data_0222 分目录存储。
        return self.data_root / f"{self.family}_data_0222"

    @property
    def strain_tag(self) -> str:
        # pure 使用无噪声 h_strain；noisy 使用注入噪声后的 data_strain。
        return "h_strain" if self.mode == "pure" else "data_strain"

    @property
    def l1_path(self) -> Path:
        return self.source_dir / f"{self.family}_{self.strain_tag}_1.npy"

    @property
    def l2_path(self) -> Path:
        return self.source_dir / f"{self.family}_{self.strain_tag}_2.npy"

    @property
    def unlensed_path(self) -> Path:
        name = "unlensed_h_strain.npy" if self.mode == "pure" else "unlensed_data_strain.npy"
        return self.data_root / "Unlensed_data_0222" / name
