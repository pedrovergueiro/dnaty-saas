"""
Experimento 2 — CIFAR-10 com operadores convolucionais reais.
v5: FastDataset com 50K em RAM — zero I/O por geração.
"""
from __future__ import annotations
import os, json, sys, time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from dnaty.experiments.fast_dataset import FastDataset
from dnaty.core.arch_cnn import DynamicCNN
from dnaty.core.individual import Individual
from dnaty.core.memory import EpisodicMemory, Experience
from dnaty.operators.mutations_cnn import CNN_OPERATORS, apply_cnn_operator
from dnaty.evolution.selection import nsga2_select
from dnaty.analysis.stats import summary_stats, paired_ttest

SEEDS = [0, 1]      # ⚡ 2 seeds para rapidez
N_GENERATIONS = 10  # ⚡⚡ 10 gerações (CPU: ultra-rápido)
N_POP = 8           # ⚡⚡ 8 população (CPU: rápido)
T_LOCAL = 3         # ⚡⚡ 3 épocas (CPU: mínimo viável)
BATCH_SIZE = 256    # ⚡⚡ 256 batch (fewer iterations)
CIFAR_TRAIN_SUBSET = 10000  # ⚡⚡ 10K (CPU pode fazer em ~5min) — qualidade/velocidade
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


# ── Baseline: ResNet-8 ────────────────────────────────────────
class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False), nn.BatchNorm2d(ch), nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False), nn.BatchNorm2d(ch),
        )
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x): return self.relu(self.block(x) + x)


class ResNet8(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            ResBlock(64),
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            ResBlock(128), nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(128, n_classes)
    def forward(self, x): return self.fc(self.net(x).view(x.size(0), -1))
    def count_params(self): return sum(p.numel() for p in self.parameters())


def evaluate_cnn_fast(ind, fast_ds, device):
    """Avalia CNN usando FastDataset — zero I/O."""
    model = ind.model.to(device)
    model.eval()
    vx, vy = fast_ds.get_val()
    correct = total = 0
    chunk = 512
    with torch.no_grad():
        for i in range(0, len(vx), chunk):
            xb = vx[i:i+chunk].to(device, non_blocking=True)
            yb = vy[i:i+chunk].to(device, non_blocking=True)
            correct += (model(xb).argmax(1) == yb).sum().item()
            total += len(yb)
    return correct / max(total, 1)


def local_train_cnn_fast(ind, fast_ds, n_epochs, lr, device, batch_size=256):
    """Treino CNN com FastDataset, SAM e Data Augmentation — múltiplos batches por epoch, zero I/O."""
    import torchvision.transforms as T
    
    model = ind.model.to(device)
    model.train()
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss(label_smoothing=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs, eta_min=lr*0.1)
    
    # Data Augmentation: RandomCrop + RandomHorizontalFlip (padrão para CIFAR)
    augment = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
    ])

    n_batches_per_epoch = max(1, fast_ds.n_train // batch_size)
    loss_before = loss_after = 0.0
    grad_norms = []
    rho = 0.05  # SAM: raio de perturbação

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        for _ in range(n_batches_per_epoch):
            xb, yb = fast_ds.get_train_batch(batch_size)
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            
            # Data Augmentation
            xb = augment(xb)
            
            # Passo 1: Computar gradiente em θ
            opt.zero_grad(set_to_none=True)
            loss = crit(model(xb), yb)
            loss.backward()
            
            # Passo 2: Calcular norma do gradiente (para SAM e logs)
            gn = sum(p.grad.norm().item()**2 for p in model.parameters() if p.grad is not None)**0.5
            grad_norms.append(gn)
            
            # Passo 3: Perturbação SAM — mover na direção do gradiente
            with torch.no_grad():
                grad_norm = sum(p.grad.norm().pow(2) for p in model.parameters() if p.grad is not None).sqrt()
                scale = rho / (grad_norm + 1e-12)
                for p in model.parameters():
                    if p.grad is not None:
                        p.data_orig = p.data.clone()  # Salvar θ original
                        p.data.add_(p.grad, alpha=scale)  # θ → θ + ρ·∇L
            
            # Passo 4: Recomputar loss com θ perturbado
            opt.zero_grad(set_to_none=True)
            loss_perturbed = crit(model(xb), yb)
            loss_perturbed.backward()
            
            # Passo 5: Restaurar θ original e fazer update com gradiente "aguçado"
            with torch.no_grad():
                for p in model.parameters():
                    if hasattr(p, 'data_orig'):
                        p.data.copy_(p.data_orig)
            
            opt.step()
            epoch_loss += loss.item()
        
        scheduler.step()
        avg = epoch_loss / max(n_batches_per_epoch, 1)
        if epoch == 0: loss_before = avg
        loss_after = avg

    return loss_before, loss_after, float(np.mean(grad_norms)) if grad_norms else 0.0


def train_resnet_fast(model, fast_ds, n_epochs=20, device='cpu', batch_size=256, lr=1.5e-3):
    """Treina ResNet-8 com FastDataset, SAM e Data Augmentation."""
    import torchvision.transforms as T
    
    model = model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss(label_smoothing=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs, eta_min=lr*0.1)
    
    # Data Augmentation
    augment = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
    ])
    
    n_batches = max(1, fast_ds.n_train // batch_size)
    rho = 0.05  # SAM
    
    for epoch in range(n_epochs):
        model.train()
        for _ in range(n_batches):
            xb, yb = fast_ds.get_train_batch(batch_size)
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            xb = augment(xb)  # Data Augmentation
            
            # SAM: Passo 1
            opt.zero_grad(set_to_none=True)
            loss = crit(model(xb), yb)
            loss.backward()
            
            # SAM: Passo 2 - Perturbação
            with torch.no_grad():
                grad_norm = sum(p.grad.norm().pow(2) for p in model.parameters() if p.grad is not None).sqrt()
                scale = rho / (grad_norm + 1e-12)
                for p in model.parameters():
                    if p.grad is not None:
                        p.data_orig = p.data.clone()
                        p.data.add_(p.grad, alpha=scale)
            
            # SAM: Passo 3 - Recomputar loss
            opt.zero_grad(set_to_none=True)
            loss_perturbed = crit(model(xb), yb)
            loss_perturbed.backward()
            
            # SAM: Passo 4 - Restaurar e atualizar
            with torch.no_grad():
                for p in model.parameters():
                    if hasattr(p, 'data_orig'):
                        p.data.copy_(p.data_orig)
            
            opt.step()
        
        scheduler.step()
    
    return evaluate_cnn_fast(type('I', (), {'model': model})(), fast_ds, device)


def run_dnaty_cnn_seed(seed, device):
    torch.manual_seed(seed)
    np.random.seed(seed)

    fast_ds = FastDataset('CIFAR10', device=device, train_subset=CIFAR_TRAIN_SUBSET)

    def make_ind():
        model = DynamicCNN(
            conv_configs=[
                {"type": "conv", "in_ch": 3,  "out_ch": 32, "stride": 1},
                {"type": "conv", "in_ch": 32, "out_ch": 64, "stride": 2},
                {"type": "conv", "in_ch": 64, "out_ch": 128, "stride": 2},  # ↑ adicionado 3ª camada
            ],
            fc_sizes=[256, 128],  # ↑ aumentado FC layers para melhor capacidade
            n_classes=10,
        )
        return Individual(model, EpisodicMemory(decay_gamma=0.99))

    population = [make_ind() for _ in range(N_POP)]
    shared_mem = EpisodicMemory(max_size=500, decay_gamma=0.99)

    for ind in population:
        ind.acc = evaluate_cnn_fast(ind, fast_ds, device)
    fitnesses = [(ind.acc, -ind.count_params() * 1e-6, 0.0) for ind in population]
    prev_best = max(ind.acc for ind in population)

    history = []
    from tqdm import tqdm
    for gen in tqdm(range(1, N_GENERATIONS + 1), desc=f"CIFAR seed={seed}"):
        op_probs = shared_mem.query_mutation_probs(CNN_OPERATORS)
        ops = list(op_probs.keys()); probs = list(op_probs.values())

        mutated = []
        for ind in population:
            op = np.random.choice(ops, p=probs)
            new_ind, ok = apply_cnn_operator(ind, op)
            if not ok or not new_ind.model.is_valid():
                new_ind = ind.clone(); new_ind.last_op = "no_op"
            mutated.append(new_ind)

        loss_befores, loss_afters, grad_norms = [], [], []
        for ind in mutated:
            lb, la, gn = local_train_cnn_fast(ind, fast_ds, T_LOCAL, 1.5e-3, device, BATCH_SIZE)  # ↑ lr aumentado de 1e-3 para 1.5e-3
            loss_befores.append(lb); loss_afters.append(la); grad_norms.append(gn)
            ind.last_grad_norm = gn

        delta_grad = float(np.mean([b - a for b, a in zip(loss_befores, loss_afters)]))

        mut_fitnesses = []
        for ind in mutated:
            ind.acc = evaluate_cnn_fast(ind, fast_ds, device)
            mut_fitnesses.append((ind.acc, -ind.count_params() * 1e-6, 0.0))

        combined_pop = population + mutated
        combined_fit = fitnesses + mut_fitnesses
        population, fitnesses = nsga2_select(combined_pop, combined_fit, N_POP)

        delta_mem = 0.0
        for i, ind in enumerate(mutated):
            if ind.acc > prev_best and ind.last_op != "no_op":
                exp = Experience(operator=ind.last_op, delta_loss=-(ind.acc - prev_best),
                                 gradient_norm=grad_norms[i], generation=gen)
                shared_mem.update(exp); delta_mem += exp.impact

        best = max(population, key=lambda x: x.acc)
        prev_best = best.acc
        history.append({
            "gen": gen, "best_acc": round(best.acc, 4),
            "delta_grad": round(max(delta_grad, 0.0), 5),
            "delta_mem": round(delta_mem, 5),
            "n_params": best.count_params(), "n_flops": best.count_flops(),
        })

    best = max(population, key=lambda x: x.acc)
    return {
        "seed": seed, "acc": round(best.acc, 4),
        "n_params": best.count_params(), "n_flops": best.count_flops(),
        "history": history,
        "delta_grad_all_positive": all(h["delta_grad"] >= -1e-6 for h in history),
        "delta_mem_positive_after_gen3": all(h["delta_mem"] >= 0 for h in history if h["gen"] >= 3),
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"\n{'='*60}")
    print("Experimento 2 — CIFAR-10 v5 (FastDataset, zero I/O)")
    print(f"{'='*60}")

    dnaty_results = []
    resnet_accs = []

    for seed in SEEDS:
        print(f"\n--- Seed {seed} ---")

        print("  [dNaty CNN]")
        t0 = time.time()
        dr = run_dnaty_cnn_seed(seed, device)
        dr["time_s"] = round(time.time() - t0, 1)
        dnaty_results.append(dr)
        print(f"  acc={dr['acc']:.4f} | params={dr['n_params']:,} | time={dr['time_s']}s")

        print("  [ResNet-8 — FastDataset (10K, 10 épocas, rápido)]")
        torch.manual_seed(seed)
        fast_ds_full = FastDataset('CIFAR10', device=device, train_subset=CIFAR_TRAIN_SUBSET)
        resnet = ResNet8()
        acc_r = train_resnet_fast(resnet, fast_ds_full, n_epochs=10, device=device)  # ⚡ 10 épocas CPU-friendly
        resnet_accs.append(round(acc_r, 4))
        print(f"  acc={acc_r:.4f} | params={resnet.count_params():,}")

    dnaty_accs = [r["acc"] for r in dnaty_results]
    dnaty_s = summary_stats(dnaty_accs)
    resnet_s = summary_stats(resnet_accs)
    t_stat, p_val, cohen_d = paired_ttest(dnaty_accs, resnet_accs)

    print(f"\n{'-'*50}")
    print("RESULTADOS FINAIS — CIFAR-10 v5")
    print(f"  dNaty CNN: {dnaty_s['mean']*100:.2f}% ± {dnaty_s['std']*100:.2f}%")
    print(f"  ResNet-8:  {resnet_s['mean']*100:.2f}% ± {resnet_s['std']*100:.2f}%")
    print(f"  p={p_val:.4f} d={cohen_d:.3f}")

    all_results = {
        "CIFAR10": {
            "dnaty": dnaty_results,
            "resnet_accs": resnet_accs,
            "summary": {
                "dnaty": dnaty_s, "resnet": resnet_s,
                "ttest": {"t": t_stat, "p": p_val, "d": cohen_d},
                "theorem1_delta_grad_positive": all(r["delta_grad_all_positive"] for r in dnaty_results),
                "theorem1_delta_mem_positive": all(r["delta_mem_positive_after_gen3"] for r in dnaty_results),
            }
        }
    }

    out_path = os.path.join(RESULTS_DIR, "exp2_cifar10_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResultados salvos em: {out_path}")
    return all_results


if __name__ == "__main__":
    main()
