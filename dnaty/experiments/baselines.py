"""
Baselines: MLP Fixo, GA Puro, EWC.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader


# ── MLP Fixo ──────────────────────────────────────────────────────────────────

class FixedMLP(nn.Module):
    def __init__(self, input_size: int = 784, hidden: list[int] = None, n_classes: int = 10):
        super().__init__()
        hidden = hidden or [128, 64]
        layers = []
        prev = input_size
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x.view(x.size(0), -1))

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


def train_fixed_mlp(
    train_loader: DataLoader,
    val_loader: DataLoader,
    n_epochs: int = 20,
    lr: float = 1e-3,
    hidden: list[int] = None,
    device: str = "cpu",
) -> tuple[float, int]:
    """Retorna (accuracy, n_params)."""
    model = FixedMLP(hidden=hidden or [128, 64]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    for _ in range(n_epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            optimizer.step()
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb).argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += len(yb)
    return correct / max(total, 1), model.count_params()


# ── GA Puro (sem gradiente) ───────────────────────────────────────────────────

def train_ga_pure(
    train_loader: DataLoader,
    val_loader: DataLoader,
    n_generations: int = 50,
    n_pop: int = 20,
    device: str = "cpu",
    input_size: int = 784,
    n_classes: int = 10,
) -> tuple[float, int]:
    """GA sem backprop — mutação aleatória de pesos."""
    from dnaty.core.arch import DynamicMLP
    from dnaty.core.individual import Individual
    from dnaty.training.local_train import evaluate

    def make_ind():
        m = DynamicMLP([input_size, 64, 32], ["relu", "relu"], n_classes)
        return Individual(m)

    population = [make_ind() for _ in range(n_pop)]
    for ind in population:
        ind.acc, _ = evaluate(ind, val_loader, device)

    for _ in range(n_generations):
        # Mutação aleatória de pesos (sem operadores estruturais)
        new_pop = []
        for ind in population:
            new_ind = ind.clone()
            with torch.no_grad():
                for p in new_ind.model.parameters():
                    p.data += torch.randn_like(p) * 0.01
            new_ind.acc, _ = evaluate(new_ind, val_loader, device)
            new_pop.append(new_ind)
        # Seleção por acurácia
        combined = population + new_pop
        combined.sort(key=lambda x: x.acc, reverse=True)
        population = combined[:n_pop]

    best = max(population, key=lambda x: x.acc)
    return best.acc, best.count_params()


# ── EWC (Elastic Weight Consolidation) ───────────────────────────────────────

class EWC:
    """EWC para Continual Learning — regularização por Fisher Information."""

    def __init__(self, model: nn.Module, ewc_lambda: float = 1000.0):
        self.model = model
        self.ewc_lambda = ewc_lambda
        self.fisher: dict[str, torch.Tensor] = {}
        self.optimal_params: dict[str, torch.Tensor] = {}

    def compute_fisher(self, loader: DataLoader, device: str = "cpu", n_samples: int = 200) -> None:
        self.model.eval()
        fisher = {n: torch.zeros_like(p, device=device) for n, p in self.model.named_parameters()}
        criterion = nn.CrossEntropyLoss()
        count = 0
        for xb, yb in loader:
            if count >= n_samples:
                break
            xb, yb = xb.to(device), yb.to(device)
            self.model.zero_grad()
            out = self.model(xb)
            loss = criterion(out, yb)
            loss.backward()
            for n, p in self.model.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad.data ** 2
            count += len(xb)
        for n in fisher:
            fisher[n] /= max(count, 1)
        self.fisher = fisher
        self.optimal_params = {n: p.data.clone().to(device) for n, p in self.model.named_parameters()}

    def penalty(self) -> torch.Tensor:
        loss = torch.tensor(0.0)
        for n, p in self.model.named_parameters():
            if n in self.fisher:
                loss += (self.fisher[n] * (p - self.optimal_params[n]) ** 2).sum()
        return self.ewc_lambda * loss


def train_ewc_cl(
    task_loaders: list[tuple[DataLoader, DataLoader]],
    n_epochs: int = 10,
    lr: float = 1e-3,
    ewc_lambda: float = 1000.0,
    device: str = "cpu",
    input_size: int = 784,
    n_classes: int = 10,
) -> np.ndarray:
    """
    Treina sequencialmente com EWC. Retorna matriz R[i,j].
    """
    T = len(task_loaders)
    R = np.zeros((T, T))
    model = FixedMLP(input_size=input_size, hidden=[256, 128], n_classes=n_classes).to(device)
    ewc = EWC(model, ewc_lambda)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    def eval_task(loader):
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb).argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += len(yb)
        return correct / max(total, 1)

    for t, (train_l, _) in enumerate(task_loaders):
        model.train()
        for _ in range(n_epochs):
            for xb, yb in train_l:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                if t > 0:
                    loss += ewc.penalty()
                loss.backward()
                optimizer.step()
        ewc.compute_fisher(train_l, device)
        # Avaliar em todas as tarefas até t
        for j in range(t + 1):
            R[t, j] = eval_task(task_loaders[j][1])

    return R
