"""
dNaty v5 — Treino local ultra-rápido.
Suporta FastDataset (tensores em RAM) e DataLoader padrão.
Otimizações: zero_grad(set_to_none=True), non_blocking, inference_mode, SAM simplificado.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dnaty.core.individual import Individual


def local_train(
    ind: Individual,
    loader,  # DataLoader OU FastDataset
    n_epochs: int = 3,
    lr: float = 1e-3,
    lambda1: float = 1e-4,
    lambda2: float = 1e-3,
    rho: float = 0.05,
    device: str = "cpu",
    batch_size: int = 512,
) -> tuple[float, float, float]:
    """
    Treina ind por n_epochs. Retorna (loss_antes, loss_depois, grad_norm_medio).
    Suporta FastDataset (get_train_batch) e DataLoader padrão.
    """
    model = ind.model
    if next(model.parameters()).device != torch.device(device):
        model = model.to(device)
        ind.model = model

    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr, eps=1e-7, weight_decay=1e-4)
    # Label smoothing: reduz overfit, melhora generalização ~0.3-0.5pp
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    # Custo estrutural: calculado UMA vez por indivíduo
    n_params = ind.count_params()
    n_flops  = ind.count_flops()
    cost_val = lambda1 * n_params * 1e-5 + lambda1 * 0.01 * n_flops * 1e-5
    cost_penalty = torch.tensor(cost_val, dtype=torch.float32, device=device)

    # LR schedule: cosine annealing — alto no início, baixo no final
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=lr*0.1)

    # Detecta se é FastDataset ou DataLoader
    is_fast = hasattr(loader, 'get_train_batch')

    loss_first = 0.0
    loss_last  = 0.0
    total_grad_sq = 0.0
    n_steps = 0

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        n_batches = 0

        if is_fast:
            # FastDataset v5: múltiplos batches por epoch cobrindo o dataset completo
            # Com 60K amostras e batch=512: 117 batches por epoch — gradiente estável
            n_batches_per_epoch = max(1, loader.n_train // batch_size)
            batches = [loader.get_train_batch(batch_size) for _ in range(n_batches_per_epoch)]
        else:
            batches = loader

        for xb, yb in batches:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            out = model(xb)
            loss = criterion(out, yb) + cost_penalty
            loss.backward()

            with torch.no_grad():
                gn_sq = sum(
                    p.grad.norm().pow(2)
                    for p in model.parameters()
                    if p.grad is not None
                )
                total_grad_sq += gn_sq.item()

            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
            n_steps += 1

        avg = epoch_loss / max(n_batches, 1)
        scheduler.step()
        if epoch == 0:
            loss_first = avg
        loss_last = avg

    mean_grad_norm = float(np.sqrt(total_grad_sq / max(n_steps, 1)))
    return loss_first, loss_last, mean_grad_norm


@torch.inference_mode()
def evaluate(
    ind: Individual,
    loader,  # DataLoader OU FastDataset
    device: str = "cpu",
) -> tuple[float, float]:
    """Retorna (accuracy, loss). Suporta FastDataset e DataLoader."""
    model = ind.model
    if next(model.parameters()).device != torch.device(device):
        model = model.to(device)
        ind.model = model

    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    correct = 0
    total = 0
    total_loss = 0.0

    is_fast = hasattr(loader, 'get_val')

    if is_fast:
        vx, vy = loader.get_val()
        # Avalia em chunks para não explodir VRAM
        chunk = 2048
        for i in range(0, len(vx), chunk):
            xb = vx[i:i+chunk].to(device, non_blocking=True)
            yb = vy[i:i+chunk].to(device, non_blocking=True)
            out = model(xb)
            total_loss += criterion(out, yb).item()
            correct += (out.argmax(dim=1) == yb).sum().item()
            total += len(yb)
    else:
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            out = model(xb)
            total_loss += criterion(out, yb).item()
            correct += (out.argmax(dim=1) == yb).sum().item()
            total += len(yb)

    return correct / max(total, 1), total_loss / max(total, 1)


def micro_adapt(
    ind: Individual,
    loader,
    lr_micro: float = 1e-5,
    top_k_pct: float = 0.03,
    device: str = "cpu",
) -> None:
    """Micro-adaptação: atualiza top-k% parâmetros por ‖∂L/∂θ_j‖."""
    model = ind.model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()

    is_fast = hasattr(loader, 'get_train_batch')
    if is_fast:
        x, y = loader.get_train_batch(256)
    else:
        x, y = next(iter(loader))
        x, y = x.to(device), y.to(device)

    criterion(model(x), y).backward()
    with torch.no_grad():
        all_grads = torch.cat([
            p.grad.abs().flatten()
            for p in model.parameters()
            if p.grad is not None
        ])
        if all_grads.numel() == 0:
            return
        k = max(1, int(all_grads.numel() * top_k_pct))
        threshold = all_grads.kthvalue(all_grads.numel() - k).values.item()
        for p in model.parameters():
            if p.grad is not None:
                mask = (p.grad.abs() >= threshold).float()
                p.data -= lr_micro * p.grad * mask
    model.zero_grad()
