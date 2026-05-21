"""
Representação da arquitetura como grafo dirigido acíclico (DAG).
A_i = (V_i, E_i, φ_i, Ω_i)
"""
from __future__ import annotations
import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy


ACTIVATIONS = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "gelu": nn.GELU,
    "sigmoid": nn.Sigmoid,
}

_innovation_counter = 0


def next_innovation() -> int:
    global _innovation_counter
    _innovation_counter += 1
    return _innovation_counter


class DynamicMLP(nn.Module):
    """
    MLP com arquitetura mutável. Representado como lista de camadas lineares.
    Suporta os 8 operadores densos + skip connections.
    """

    def __init__(self, layer_sizes: list[int], activations: list[str] | None = None, n_classes: int = 10):
        super().__init__()
        self.layer_sizes = list(layer_sizes)
        self.n_classes = n_classes
        self.activations = activations or ["relu"] * (len(layer_sizes) - 1)
        self.innovation_ids = [next_innovation() for _ in range(len(layer_sizes) - 1)]
        self._build()

    def _build(self) -> None:
        layers = []
        for i in range(len(self.layer_sizes) - 1):
            layers.append(nn.Linear(self.layer_sizes[i], self.layer_sizes[i + 1]))
            # BatchNorm antes da ativação — estabiliza treino, permite LR maior
            layers.append(nn.BatchNorm1d(self.layer_sizes[i + 1]))
            act = self.activations[i] if i < len(self.activations) else "relu"
            layers.append(ACTIVATIONS.get(act, nn.ReLU)())
        layers.append(nn.Linear(self.layer_sizes[-1], self.n_classes))
        self.net = nn.Sequential(*layers)
        self.skip_connections: list[tuple[int, int, nn.Linear]] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        layer_outputs = [x]
        idx = 0
        for i in range(len(self.layer_sizes) - 1):
            linear = self.net[idx]
            bn     = self.net[idx + 1]
            act    = self.net[idx + 2]
            out = act(bn(linear(layer_outputs[-1])))
            for src, dst, proj in self.skip_connections:
                if dst == i + 1 and src < len(layer_outputs):
                    skip_in = layer_outputs[src]
                    if proj is not None:
                        skip_in = proj(skip_in)
                    if skip_in.shape == out.shape:
                        out = out + skip_in
            layer_outputs.append(out)
            idx += 3  # Linear + BN + Activation
        return self.net[idx](layer_outputs[-1])

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def count_flops(self) -> int:
        flops = 0
        for i in range(len(self.layer_sizes) - 1):
            flops += 2 * self.layer_sizes[i] * self.layer_sizes[i + 1]
        flops += 2 * self.layer_sizes[-1] * self.n_classes
        return flops

    def is_valid(self) -> bool:
        return all(s > 0 for s in self.layer_sizes) and len(self.layer_sizes) >= 2
