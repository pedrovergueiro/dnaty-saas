"""
DynamicCNN — arquitetura CNN mutável para CIFAR-10.
Suporta blocos Conv2D+BN+ReLU e depthwise separable.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy


class ConvBlock(nn.Module):
    """Conv2D + BatchNorm + ReLU — bloco padrão."""
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, stride: int = 1):
        super().__init__()
        pad = kernel // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=pad, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DepthwiseSepBlock(nn.Module):
    """Depthwise Separable Conv — MobileNet style. k² vezes menos FLOPs."""
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            # Depthwise
            nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            # Pointwise
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DynamicCNN(nn.Module):
    """
    CNN com arquitetura mutável para CIFAR-10 (32×32×3).
    Estrutura: [ConvBlocks] → GlobalAvgPool → [FC layers] → classifier
    """

    def __init__(
        self,
        conv_configs: list[dict] | None = None,
        fc_sizes: list[int] | None = None,
        n_classes: int = 10,
        in_channels: int = 3,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.in_channels = in_channels

        # Config padrão: 3 blocos conv progressivos
        if conv_configs is None:
            conv_configs = [
                {"type": "conv", "in_ch": 3,  "out_ch": 32, "stride": 1},
                {"type": "conv", "in_ch": 32, "out_ch": 64, "stride": 2},
                {"type": "conv", "in_ch": 64, "out_ch": 64, "stride": 2},
            ]
        if fc_sizes is None:
            fc_sizes = [128]

        self.conv_configs = list(conv_configs)
        self.fc_sizes = list(fc_sizes)
        self._build()

    def _build(self) -> None:
        # Blocos convolucionais
        conv_layers = []
        for cfg in self.conv_configs:
            if cfg["type"] == "depthwise":
                conv_layers.append(DepthwiseSepBlock(cfg["in_ch"], cfg["out_ch"], cfg.get("stride", 1)))
            else:
                conv_layers.append(ConvBlock(cfg["in_ch"], cfg["out_ch"], cfg.get("kernel", 3), cfg.get("stride", 1)))
        self.conv_layers = nn.ModuleList(conv_layers)
        self.pool = nn.AdaptiveAvgPool2d(1)  # → (B, C, 1, 1)

        # Camadas FC
        last_ch = self.conv_configs[-1]["out_ch"] if self.conv_configs else self.in_channels
        fc_layers = []
        prev = last_ch
        for h in self.fc_sizes:
            fc_layers += [nn.Linear(prev, h), nn.ReLU(inplace=True)]
            prev = h
        self.fc = nn.Sequential(*fc_layers)
        self.classifier = nn.Linear(prev, self.n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.conv_layers:
            x = layer(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return self.classifier(x)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def count_flops(self) -> int:
        """Estimativa de FLOPs para input 32×32."""
        flops = 0
        h, w = 32, 32
        for cfg in self.conv_configs:
            k = cfg.get("kernel", 3)
            s = cfg.get("stride", 1)
            if cfg["type"] == "depthwise":
                # Depthwise: k²×C_in×H×W + C_in×C_out×H×W (pointwise)
                flops += k * k * cfg["in_ch"] * (h // s) * (w // s)
                flops += cfg["in_ch"] * cfg["out_ch"] * (h // s) * (w // s)
            else:
                flops += k * k * cfg["in_ch"] * cfg["out_ch"] * (h // s) * (w // s)
            h, w = h // s, w // s
        last_ch = self.conv_configs[-1]["out_ch"] if self.conv_configs else self.in_channels
        prev = last_ch
        for sz in self.fc_sizes:
            flops += 2 * prev * sz
            prev = sz
        flops += 2 * prev * self.n_classes
        return flops

    def is_valid(self) -> bool:
        return len(self.conv_configs) >= 1 and all(
            c["in_ch"] > 0 and c["out_ch"] > 0 for c in self.conv_configs
        )
