from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class Snake(nn.Module):
    # 带周期项的激活函数，适合处理波形中振荡结构，比普通 ReLU 更平滑。
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + torch.sin(self.alpha * x) ** 2 / (self.alpha + 1e-8)


class ResidualBlock1D(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 7) -> None:
        super().__init__()
        pad = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=pad, bias=False),
            nn.BatchNorm1d(channels),
            Snake(channels),
            nn.Conv1d(channels, channels, kernel_size, padding=pad, bias=False),
            nn.BatchNorm1d(channels),
        )
        self.act = Snake(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class MatchEncoder1D(nn.Module):
    """Compact match-style Siamese encoder for GW strain windows."""
    # 轻量 CNN baseline：速度快，但多尺度表达能力弱于 InceptionTime。

    def __init__(self, in_channels: int = 1, d_model: int = 256, emb_dim: int = 128, width_scale: float = 2.0) -> None:
        super().__init__()
        base = max(16, int(32 * width_scale))
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, base, 15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(base),
            Snake(base),
            nn.MaxPool1d(2),
            nn.Conv1d(base, base * 2, 11, stride=2, padding=5, bias=False),
            nn.BatchNorm1d(base * 2),
            Snake(base * 2),
            ResidualBlock1D(base * 2, 9),
            nn.Conv1d(base * 2, base * 4, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(base * 4),
            Snake(base * 4),
            ResidualBlock1D(base * 4, 7),
            nn.Conv1d(base * 4, base * 4, 5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(base * 4),
            Snake(base * 4),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(base * 4, d_model),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.head(self.features(x.float()))
        return F.normalize(z, dim=-1)


class InceptionBlock1D(nn.Module):
    # 多分支卷积同时看不同时间尺度，适合捕捉 GW 波形的局部和中尺度形态。
    def __init__(self, in_channels: int, out_channels: int, bottleneck_channels: int = 32, kernel_sizes: tuple[int, ...] = (39, 19, 9)) -> None:
        super().__init__()
        bottleneck_channels = min(bottleneck_channels, in_channels) if in_channels > 1 else 1
        self.bottleneck = nn.Conv1d(in_channels, bottleneck_channels, 1, bias=False) if in_channels > 1 else nn.Identity()
        branches = []
        for k in kernel_sizes:
            branches.append(nn.Conv1d(bottleneck_channels, out_channels, k, padding=k // 2, bias=False))
        self.branches = nn.ModuleList(branches)
        self.pool_branch = nn.Sequential(
            nn.MaxPool1d(3, stride=1, padding=1),
            nn.Conv1d(in_channels, out_channels, 1, bias=False),
        )
        self.bn = nn.BatchNorm1d(out_channels * (len(kernel_sizes) + 1))
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xb = self.bottleneck(x)
        ys = [branch(xb) for branch in self.branches]
        ys.append(self.pool_branch(x))
        return self.act(self.bn(torch.cat(ys, dim=1)))


class InceptionTimeEncoder1D(nn.Module):
    """InceptionTime-style multi-scale encoder for GW strain windows."""
    # 本轮最佳结果使用该 backbone。输出是 L2 归一化 embedding，
    # 后续直接用余弦相似度做 Top-K 候选检索。

    def __init__(self, in_channels: int = 1, d_model: int = 256, emb_dim: int = 128, width_scale: float = 2.0, depth: int = 6) -> None:
        super().__init__()
        branch = max(16, int(32 * width_scale))
        channels = branch * 4
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, channels, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.MaxPool1d(2),
        )
        blocks = []
        shortcuts = []
        for i in range(depth):
            blocks.append(InceptionBlock1D(channels, branch, bottleneck_channels=max(16, branch)))
            shortcuts.append(
                nn.Sequential(nn.Conv1d(channels, channels, 1, bias=False), nn.BatchNorm1d(channels))
                if i % 3 == 2 else nn.Identity()
            )
        self.blocks = nn.ModuleList(blocks)
        self.shortcuts = nn.ModuleList(shortcuts)
        self.res_act = nn.GELU()
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, d_model),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.stem(x.float())
        residual = y
        for i, block in enumerate(self.blocks):
            y = block(y)
            if i % 3 == 2:
                y = self.res_act(y + self.shortcuts[i](residual))
                residual = y
        z = self.head(y)
        return F.normalize(z, dim=-1)


class SqueezeExcite1D(nn.Module):
    # 通道注意力：根据整段波形的全局响应，增强对稳定物理形态有用的通道。
    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(8, channels // reduction)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, hidden, 1),
            nn.GELU(),
            nn.Conv1d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.net(x)


class DownsampleResBlock1D(nn.Module):
    # 带下采样的残差块，用较大卷积核保留 chirp/merger 的局部连续结构。
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 9, stride: int = 2) -> None:
        super().__init__()
        pad = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False),
            nn.BatchNorm1d(out_channels),
            Snake(out_channels),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=pad, bias=False),
            nn.BatchNorm1d(out_channels),
            SqueezeExcite1D(out_channels),
        )
        self.skip = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False),
            nn.BatchNorm1d(out_channels),
        )
        self.act = Snake(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.net(x) + self.skip(x))


class AttentiveResNetEncoder1D(nn.Module):
    """Noise-robust residual encoder with attention pooling for GW retrieval."""
    # 相比全局平均池化，attention pooling 会自动关注 SNR 更高、形态更稳定的时间片段；
    # 对 noisy 数据更合适，同时仍输出 L2 归一化 embedding 供相似度检索。

    def __init__(self, in_channels: int = 1, d_model: int = 256, emb_dim: int = 128, width_scale: float = 2.0) -> None:
        super().__init__()
        base = max(32, int(32 * width_scale))
        c1, c2, c3, c4 = base, base * 2, base * 4, base * 4
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, c1, 31, stride=2, padding=15, bias=False),
            nn.BatchNorm1d(c1),
            Snake(c1),
            nn.MaxPool1d(3, stride=2, padding=1),
        )
        self.stage1 = nn.Sequential(ResidualBlock1D(c1, 15), DownsampleResBlock1D(c1, c2, 15, 2))
        self.stage2 = nn.Sequential(ResidualBlock1D(c2, 11), DownsampleResBlock1D(c2, c3, 11, 2))
        self.stage3 = nn.Sequential(ResidualBlock1D(c3, 9), DownsampleResBlock1D(c3, c4, 9, 2), ResidualBlock1D(c4, 7))
        self.attn = nn.Sequential(
            nn.Conv1d(c4, max(16, c4 // 4), 1),
            nn.GELU(),
            nn.Conv1d(max(16, c4 // 4), 1, 1),
        )
        self.head = nn.Sequential(
            nn.Linear(c4 * 3, d_model),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(d_model, emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.stem(x.float())
        y = self.stage1(y)
        y = self.stage2(y)
        y = self.stage3(y)
        weights = torch.softmax(self.attn(y), dim=-1)
        attn_pool = (y * weights).sum(dim=-1)
        avg_pool = y.mean(dim=-1)
        max_pool = y.amax(dim=-1)
        z = self.head(torch.cat([attn_pool, avg_pool, max_pool], dim=1))
        return F.normalize(z, dim=-1)


class InceptionAttentionEncoder1D(nn.Module):
    """InceptionTime backbone with attention/avg/max pooling readout."""
    # 保留 InceptionTime 的多尺度卷积主体，但读出阶段同时使用 attention、average 和 max pooling。

    def __init__(self, in_channels: int = 1, d_model: int = 256, emb_dim: int = 128, width_scale: float = 2.0, depth: int = 6) -> None:
        super().__init__()
        branch = max(16, int(32 * width_scale))
        channels = branch * 4
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, channels, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.MaxPool1d(2),
        )
        self.blocks = nn.ModuleList([InceptionBlock1D(channels, branch, bottleneck_channels=max(16, branch)) for _ in range(depth)])
        self.shortcuts = nn.ModuleList([
            nn.Sequential(nn.Conv1d(channels, channels, 1, bias=False), nn.BatchNorm1d(channels)) if i % 3 == 2 else nn.Identity()
            for i in range(depth)
        ])
        self.res_act = nn.GELU()
        self.attn = nn.Sequential(nn.Conv1d(channels, max(16, channels // 4), 1), nn.GELU(), nn.Conv1d(max(16, channels // 4), 1, 1))
        self.head = nn.Sequential(nn.Linear(channels * 3, d_model), nn.GELU(), nn.Dropout(0.12), nn.Linear(d_model, emb_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.stem(x.float())
        residual = y
        for i, block in enumerate(self.blocks):
            y = block(y)
            if i % 3 == 2:
                y = self.res_act(y + self.shortcuts[i](residual))
                residual = y
        weights = torch.softmax(self.attn(y), dim=-1)
        attn_pool = (y * weights).sum(dim=-1)
        avg_pool = y.mean(dim=-1)
        max_pool = y.amax(dim=-1)
        z = self.head(torch.cat([attn_pool, avg_pool, max_pool], dim=1))
        return F.normalize(z, dim=-1)


class ChannelLayerNorm1D(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class ConvNeXtBlock1D(nn.Module):
    # ConvNeXt 风格 1D block：大核 depthwise 卷积 + 通道 MLP。
    def __init__(self, channels: int, kernel_size: int = 15, expansion: int = 4) -> None:
        super().__init__()
        self.dw = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2, groups=channels)
        self.norm = ChannelLayerNorm1D(channels)
        self.pw1 = nn.Conv1d(channels, channels * expansion, 1)
        self.act = nn.GELU()
        self.pw2 = nn.Conv1d(channels * expansion, channels, 1)
        self.gamma = nn.Parameter(1e-6 * torch.ones(1, channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.dw(x)
        y = self.norm(y)
        y = self.pw2(self.act(self.pw1(y)))
        return x + self.gamma * y


class ConvNeXtEncoder1D(nn.Module):
    """ConvNeXt-style 1D encoder for waveform retrieval."""

    def __init__(self, in_channels: int = 1, d_model: int = 256, emb_dim: int = 128, width_scale: float = 2.0) -> None:
        super().__init__()
        base = max(32, int(32 * width_scale))
        c1, c2, c3 = base, base * 2, base * 4
        self.stem = nn.Sequential(nn.Conv1d(in_channels, c1, 15, stride=4, padding=7, bias=False), ChannelLayerNorm1D(c1))
        self.stage1 = nn.Sequential(ConvNeXtBlock1D(c1, 15), ConvNeXtBlock1D(c1, 15))
        self.down1 = nn.Sequential(ChannelLayerNorm1D(c1), nn.Conv1d(c1, c2, 3, stride=2, padding=1))
        self.stage2 = nn.Sequential(ConvNeXtBlock1D(c2, 15), ConvNeXtBlock1D(c2, 15), ConvNeXtBlock1D(c2, 15))
        self.down2 = nn.Sequential(ChannelLayerNorm1D(c2), nn.Conv1d(c2, c3, 3, stride=2, padding=1))
        self.stage3 = nn.Sequential(ConvNeXtBlock1D(c3, 11), ConvNeXtBlock1D(c3, 11), ConvNeXtBlock1D(c3, 11))
        self.head = nn.Sequential(nn.Linear(c3 * 2, d_model), nn.GELU(), nn.Dropout(0.12), nn.Linear(d_model, emb_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.stage1(self.stem(x.float()))
        y = self.stage2(self.down1(y))
        y = self.stage3(self.down2(y))
        z = self.head(torch.cat([y.mean(dim=-1), y.amax(dim=-1)], dim=1))
        return F.normalize(z, dim=-1)


class DilatedResidualBlock1D(nn.Module):
    # 扩张卷积残差块：在不显著增加参数量的情况下扩大时间感受野，适合 bandpass 后的长 chirp 波形。
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 15, dilation: int = 1, stride: int = 1) -> None:
        super().__init__()
        pad = dilation * (kernel_size // 2)
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=pad, dilation=dilation, groups=out_channels, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            SqueezeExcite1D(out_channels),
            nn.Conv1d(out_channels, out_channels, 1, bias=False),
            nn.BatchNorm1d(out_channels),
        )
        self.skip = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.net(x) + self.skip(x))


class DilatedResNetEncoder1D(nn.Module):
    """Multi-scale dilated residual encoder for noisy GW waveform retrieval."""
    # 设计目标：比 InceptionTime 更连续地覆盖长时间尺度；bandpass 后的 chirp 主要差异在局部-中长程形态。

    def __init__(self, in_channels: int = 1, d_model: int = 256, emb_dim: int = 128, width_scale: float = 2.0) -> None:
        super().__init__()
        base = max(32, int(32 * width_scale))
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, base, 15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(base),
            nn.GELU(),
            nn.MaxPool1d(3, stride=2, padding=1),
        )
        self.blocks = nn.Sequential(
            DilatedResidualBlock1D(base, base, 15, dilation=1),
            DilatedResidualBlock1D(base, base, 15, dilation=2),
            DilatedResidualBlock1D(base, base * 2, 13, dilation=1, stride=2),
            DilatedResidualBlock1D(base * 2, base * 2, 13, dilation=2),
            DilatedResidualBlock1D(base * 2, base * 2, 13, dilation=4),
            DilatedResidualBlock1D(base * 2, base * 4, 11, dilation=1, stride=2),
            DilatedResidualBlock1D(base * 4, base * 4, 11, dilation=2),
            DilatedResidualBlock1D(base * 4, base * 4, 11, dilation=4),
            DilatedResidualBlock1D(base * 4, base * 4, 9, dilation=8),
        )
        channels = base * 4
        self.head = nn.Sequential(
            nn.Linear(channels * 2, d_model),
            nn.GELU(),
            nn.Dropout(0.12),
            nn.Linear(d_model, emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.blocks(self.stem(x.float()))
        avg_pool = y.mean(dim=-1)
        max_pool = y.amax(dim=-1)
        z = self.head(torch.cat([avg_pool, max_pool], dim=1))
        return F.normalize(z, dim=-1)


class ChannelSpatialAttention1D(nn.Module):
    # CBAM 风格注意力：先做通道筛选，再在时间维上标出更可能含有效信号的片段。
    def __init__(self, channels: int, reduction: int = 8, kernel_size: int = 7) -> None:
        super().__init__()
        hidden = max(8, channels // reduction)
        self.channel_mlp = nn.Sequential(
            nn.Conv1d(channels, hidden, 1),
            nn.GELU(),
            nn.Conv1d(hidden, channels, 1),
        )
        self.spatial = nn.Sequential(
            nn.Conv1d(2, 1, kernel_size, padding=kernel_size // 2),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=-1, keepdim=True)
        mx = x.amax(dim=-1, keepdim=True)
        ch = torch.sigmoid(self.channel_mlp(avg) + self.channel_mlp(mx))
        y = x * ch
        sp = self.spatial(torch.cat([y.mean(dim=1, keepdim=True), y.amax(dim=1, keepdim=True)], dim=1))
        return y * sp


class SEResNetEncoder1D(nn.Module):
    """SE-ResNet encoder with simple avg/max readout for noisy waveform retrieval."""
    # 目标是比 attnresnet 更稳定：保留 SE 通道注意力，但不用时间 attention pooling 过度聚焦噪声峰。

    def __init__(self, in_channels: int = 1, d_model: int = 256, emb_dim: int = 128, width_scale: float = 2.0) -> None:
        super().__init__()
        base = max(32, int(32 * width_scale))
        c1, c2, c3, c4 = base, base * 2, base * 4, base * 4
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, c1, 31, stride=2, padding=15, bias=False),
            nn.BatchNorm1d(c1),
            Snake(c1),
            nn.MaxPool1d(3, stride=2, padding=1),
        )
        self.net = nn.Sequential(
            ResidualBlock1D(c1, 15),
            DownsampleResBlock1D(c1, c2, 15, 2),
            ResidualBlock1D(c2, 11),
            DownsampleResBlock1D(c2, c3, 11, 2),
            ResidualBlock1D(c3, 9),
            DownsampleResBlock1D(c3, c4, 9, 2),
            ResidualBlock1D(c4, 7),
        )
        self.head = nn.Sequential(nn.Linear(c4 * 2, d_model), nn.GELU(), nn.Dropout(0.12), nn.Linear(d_model, emb_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.net(self.stem(x.float()))
        z = self.head(torch.cat([y.mean(dim=-1), y.amax(dim=-1)], dim=1))
        return F.normalize(z, dim=-1)


class CBAMResNetBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 11, stride: int = 1) -> None:
        super().__init__()
        pad = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False),
            nn.BatchNorm1d(out_channels),
            Snake(out_channels),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=pad, bias=False),
            nn.BatchNorm1d(out_channels),
            ChannelSpatialAttention1D(out_channels),
        )
        self.skip = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False), nn.BatchNorm1d(out_channels))
        self.act = Snake(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.net(x) + self.skip(x))


class CBAMResNetEncoder1D(nn.Module):
    """ResNet1D with channel-spatial attention for waveform retrieval."""

    def __init__(self, in_channels: int = 1, d_model: int = 256, emb_dim: int = 128, width_scale: float = 2.0) -> None:
        super().__init__()
        base = max(32, int(32 * width_scale))
        c1, c2, c3 = base, base * 2, base * 4
        self.stem = nn.Sequential(nn.Conv1d(in_channels, c1, 31, stride=4, padding=15, bias=False), nn.BatchNorm1d(c1), Snake(c1))
        self.net = nn.Sequential(
            CBAMResNetBlock1D(c1, c1, 15),
            CBAMResNetBlock1D(c1, c2, 13, stride=2),
            CBAMResNetBlock1D(c2, c2, 11),
            CBAMResNetBlock1D(c2, c3, 9, stride=2),
            CBAMResNetBlock1D(c3, c3, 7),
        )
        self.head = nn.Sequential(nn.Linear(c3 * 3, d_model), nn.GELU(), nn.Dropout(0.15), nn.Linear(d_model, emb_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.net(self.stem(x.float()))
        att = torch.softmax(y.mean(dim=1, keepdim=True), dim=-1)
        z = self.head(torch.cat([(y * att).sum(dim=-1), y.mean(dim=-1), y.amax(dim=-1)], dim=1))
        return F.normalize(z, dim=-1)


class GatedTCNBlock1D(nn.Module):
    # WaveNet/TCN 风格 gated activation，用扩张卷积覆盖更长时间跨度。
    def __init__(self, channels: int, dilation: int, kernel_size: int = 7, dropout: float = 0.05) -> None:
        super().__init__()
        pad = dilation * (kernel_size // 2)
        self.filter = nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation)
        self.gate = nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation)
        self.proj = nn.Conv1d(channels, channels, 1)
        self.norm = nn.BatchNorm1d(channels)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.tanh(self.filter(x)) * torch.sigmoid(self.gate(x))
        y = self.drop(self.proj(y))
        return self.norm(x + y)


class GatedTCNEncoder1D(nn.Module):
    """Gated temporal convolutional encoder for long noisy strain windows."""

    def __init__(self, in_channels: int = 1, d_model: int = 256, emb_dim: int = 128, width_scale: float = 2.0) -> None:
        super().__init__()
        base = max(32, int(32 * width_scale))
        channels = base * 2
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, base, 15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(base),
            nn.GELU(),
            nn.Conv1d(base, channels, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(channels),
            nn.GELU(),
        )
        dilations = [1, 2, 4, 8, 16, 32, 1, 2, 4, 8]
        self.blocks = nn.Sequential(*[GatedTCNBlock1D(channels, d, 7) for d in dilations])
        self.se = SqueezeExcite1D(channels)
        self.head = nn.Sequential(nn.Linear(channels * 2, d_model), nn.GELU(), nn.Dropout(0.12), nn.Linear(d_model, emb_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.se(self.blocks(self.stem(x.float())))
        z = self.head(torch.cat([y.mean(dim=-1), y.amax(dim=-1)], dim=1))
        return F.normalize(z, dim=-1)


class PatchTransformerEncoder1D(nn.Module):
    """PatchTST-style lightweight patch transformer for Siamese waveform retrieval."""
    # 先把长波形切成 patch，再对 patch token 做 self-attention，避免直接对全部采样点做 attention。

    def __init__(self, in_channels: int = 1, d_model: int = 256, emb_dim: int = 128, width_scale: float = 2.0) -> None:
        super().__init__()
        model_dim = max(96, int(96 * width_scale))
        heads = 4 if model_dim % 4 == 0 else 3
        self.patch = nn.Conv1d(in_channels, model_dim, kernel_size=64, stride=32, padding=16)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=heads,
            dim_feedforward=model_dim * 4,
            dropout=0.10,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=3)
        self.norm = nn.LayerNorm(model_dim)
        self.head = nn.Sequential(nn.Linear(model_dim * 2, d_model), nn.GELU(), nn.Dropout(0.15), nn.Linear(d_model, emb_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.patch(x.float()).transpose(1, 2)
        y = self.norm(self.encoder(y))
        z = self.head(torch.cat([y.mean(dim=1), y.amax(dim=1)], dim=1))
        return F.normalize(z, dim=-1)


class RandomKernelBank1D(nn.Module):
    # ROCKET 思路的可复用近似：固定随机多尺度卷积核，只训练最后的投影头。
    def __init__(self, in_channels: int, features_per_kernel: int = 64) -> None:
        super().__init__()
        kernels = [7, 11, 15, 23, 31]
        self.convs = nn.ModuleList()
        for i, k in enumerate(kernels):
            conv = nn.Conv1d(in_channels, features_per_kernel, k, padding=k // 2, dilation=2 ** (i % 3), bias=True)
            nn.init.normal_(conv.weight, mean=0.0, std=1.0 / (in_channels * k) ** 0.5)
            nn.init.uniform_(conv.bias, -1.0, 1.0)
            for param in conv.parameters():
                param.requires_grad = False
            self.convs.append(conv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = []
        for conv in self.convs:
            y = conv(x)
            feats.append(y.mean(dim=-1))
            feats.append(y.amax(dim=-1))
            feats.append((y > 0).float().mean(dim=-1))
        return torch.cat(feats, dim=1)


class RocketEncoder1D(nn.Module):
    """ROCKET-like fixed random convolution feature encoder with trainable projection."""

    def __init__(self, in_channels: int = 1, d_model: int = 256, emb_dim: int = 128, width_scale: float = 2.0) -> None:
        super().__init__()
        per_kernel = max(48, int(48 * width_scale))
        self.bank = RandomKernelBank1D(in_channels, per_kernel)
        feat_dim = per_kernel * 5 * 3
        self.head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, d_model),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(d_model, emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.head(self.bank(x.float()))
        return F.normalize(z, dim=-1)


class Period2DBlock1D(nn.Module):
    def __init__(self, channels: int, period: int) -> None:
        super().__init__()
        self.period = period
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, (3, 3), padding=1, groups=channels),
            nn.Conv2d(channels, channels, 1),
            nn.GELU(),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t = x.shape
        pad = (-t) % self.period
        if pad:
            x = F.pad(x, (0, pad))
            t = t + pad
        y = x.reshape(b, c, t // self.period, self.period)
        y = self.conv(y).reshape(b, c, t)
        return y[..., : t - pad] if pad else y


class TimesNetLiteEncoder1D(nn.Module):
    """TimesNet-inspired 2D temporal variation encoder for GW waveform retrieval."""
    # 固定多个 period，把 1D 波形折叠成 2D 时间块，用 2D depthwise 卷积提取局部-周期变化。

    def __init__(self, in_channels: int = 1, d_model: int = 256, emb_dim: int = 128, width_scale: float = 2.0) -> None:
        super().__init__()
        base = max(32, int(32 * width_scale))
        channels = base * 2
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, base, 15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(base),
            nn.GELU(),
            nn.Conv1d(base, channels, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(channels),
            nn.GELU(),
        )
        self.period_blocks = nn.ModuleList([Period2DBlock1D(channels, p) for p in (8, 16, 32, 64)])
        self.mix = nn.Sequential(nn.Conv1d(channels, channels, 1), nn.GELU(), SqueezeExcite1D(channels))
        self.head = nn.Sequential(nn.Linear(channels * 2, d_model), nn.GELU(), nn.Dropout(0.12), nn.Linear(d_model, emb_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.stem(x.float())
        ys = [block(y) for block in self.period_blocks]
        y = self.mix(torch.stack(ys, dim=0).mean(dim=0) + y)
        z = self.head(torch.cat([y.mean(dim=-1), y.amax(dim=-1)], dim=1))
        return F.normalize(z, dim=-1)


class NTXentLoss(nn.Module):
    # 对比学习损失：同一个 batch 中对应的两路视图是正样本，其余为负样本。
    # 目标是让同源 lensed images 在 embedding 空间更近。
    def __init__(self, tau: float = 0.07) -> None:
        super().__init__()
        self.tau = tau

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        z1 = F.normalize(z1.float(), dim=-1)
        z2 = F.normalize(z2.float(), dim=-1)
        n = z1.size(0)
        z = torch.cat([z1, z2], dim=0)
        sim = z @ z.T / self.tau
        sim = sim.masked_fill(torch.eye(2 * n, dtype=torch.bool, device=z.device), -1e9)
        pos = torch.cat([torch.diag(sim, n), torch.diag(sim, -n)], dim=0)
        return -(pos - torch.logsumexp(sim, dim=1)).mean()
