"""
Experimento 3 — Split-MNIST: Continual Learning v5.
FastDataset por task — zero I/O durante treino.
"""
from __future__ import annotations
import os, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as T
from torch.utils.data import Subset, DataLoader

from dnaty.analysis.cl_metrics import compute_cl_metrics
from dnaty.analysis.stats import paired_ttest
from dnaty.core.arch import DynamicMLP

SEEDS = [0, 1, 2]
N_TASKS = 5
N_EPOCHS_CL = 15
TRAIN_SUBSET_CL = None
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


class FastTaskDataset:
    """Carrega uma task do Split-MNIST em RAM — zero I/O durante treino."""

    def __init__(self, task_id: int, device: str = "cpu", data_dir: str = "./data", train_subset=None):
        transform = T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])
        train_full = torchvision.datasets.MNIST(data_dir, train=True,  download=True, transform=transform)
        test_full  = torchvision.datasets.MNIST(data_dir, train=False, download=True, transform=transform)
        labels = [task_id * 2, task_id * 2 + 1]

        def filter_ds(ds):
            targets = ds.targets if hasattr(ds, "targets") else torch.tensor(ds.labels)
            idx = [i for i, t in enumerate(targets) if int(t) in labels]
            return Subset(ds, idx)

        train_sub = filter_ds(train_full)
        test_sub  = filter_ds(test_full)
        if train_subset:
            train_sub = Subset(train_sub, list(range(min(train_subset, len(train_sub)))))

        def load_all(ds):
            loader = DataLoader(ds, batch_size=len(ds), shuffle=False, num_workers=0)
            x, y = next(iter(loader))
            return x.flatten(1).to(device), y.to(device)

        self.train_x, self.train_y = load_all(train_sub)
        self.val_x,   self.val_y   = load_all(test_sub)
        self.n_train = len(self.train_x)
        self.device  = device

    def get_train_batch(self, batch_size=256):
        idx = torch.randint(0, self.n_train, (min(batch_size, self.n_train),), device=self.device)
        return self.train_x[idx], self.train_y[idx]

    def get_val(self):
        return self.val_x, self.val_y


def eval_task_fast(model, fast_ds, device):
    model.eval()
    vx, vy = fast_ds.get_val()
    correct = total = 0
    with torch.no_grad():
        for i in range(0, len(vx), 512):
            xb = vx[i:i+512].to(device)
            yb = vy[i:i+512].to(device)
            correct += (model(xb).argmax(1) == yb).sum().item()
            total += len(yb)
    return correct / max(total, 1)


def run_dnaty_cl_seed(seed: int, device: str) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    task_datasets = [FastTaskDataset(t, device=device, train_subset=TRAIN_SUBSET_CL) for t in range(N_TASKS)]
    R = np.zeros((N_TASKS, N_TASKS))
    model = DynamicMLP([784, 256, 128], ["relu", "relu"], n_classes=10).to(device)
    crit = nn.CrossEntropyLoss(label_smoothing=0.05)
    replay_x, replay_y = [], []
    REPLAY_SIZE = 200

    for t in range(N_TASKS):
        ds = task_datasets[t]
        opt = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        n_batches = max(1, ds.n_train // 256)
        for epoch in range(N_EPOCHS_CL):
            model.train()
            for _ in range(n_batches):
                xb, yb = ds.get_train_batch(256)
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad(set_to_none=True)
                loss = crit(model(xb), yb)
                if replay_x:
                    rx = torch.cat(replay_x).to(device)
                    ry = torch.cat(replay_y).to(device)
                    idx_r = torch.randperm(len(rx))[:min(128, len(rx))]
                    loss = loss + crit(model(rx[idx_r]), ry[idx_r])
                loss.backward()
                opt.step()
        idx_r = torch.randperm(ds.n_train)[:REPLAY_SIZE]
        replay_x.append(ds.train_x[idx_r].cpu())
        replay_y.append(ds.train_y[idx_r].cpu())
        for j in range(t + 1):
            R[t, j] = eval_task_fast(model, task_datasets[j], device)

    baselines = np.zeros(N_TASKS)
    for t in range(N_TASKS):
        m = DynamicMLP([784, 256, 128], ["relu", "relu"], 10).to(device)
        ds = task_datasets[t]
        opt2 = optim.Adam(m.parameters(), lr=1e-3)
        for _ in range(N_EPOCHS_CL):
            for _ in range(max(1, ds.n_train // 256)):
                xb, yb = ds.get_train_batch(256)
                opt2.zero_grad(); crit(m(xb.to(device)), yb.to(device)).backward(); opt2.step()
        baselines[t] = eval_task_fast(m, ds, device)

    metrics = compute_cl_metrics(R, baselines)
    return {"seed": seed, "R": R.tolist(), "metrics": metrics, "baselines": baselines.tolist()}


def run_ewc_cl_seed(seed: int, device: str) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    task_datasets = [FastTaskDataset(t, device=device, train_subset=TRAIN_SUBSET_CL) for t in range(N_TASKS)]
    R = np.zeros((N_TASKS, N_TASKS))
    model = DynamicMLP([784, 256, 128], ["relu", "relu"], 10).to(device)
    crit = nn.CrossEntropyLoss()
    ewc_lambda = 400.0
    fisher_list, opt_params_list = [], []

    def ewc_penalty():
        loss = torch.tensor(0.0, device=device)
        for fisher, opt_p in zip(fisher_list, opt_params_list):
            for n, p in model.named_parameters():
                if n in fisher:
                    loss += (fisher[n] * (p - opt_p[n]) ** 2).sum()
        return ewc_lambda * loss

    def compute_fisher(ds, n_samples=300):
        model.eval()
        fisher = {n: torch.zeros_like(p, device=device) for n, p in model.named_parameters()}
        count = 0
        while count < n_samples:
            xb, yb = ds.get_train_batch(64)
            xb, yb = xb.to(device), yb.to(device)
            model.zero_grad()
            crit(model(xb), yb).backward()
            for n, p in model.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad.data ** 2
            count += len(xb)
        for n in fisher:
            fisher[n] /= max(count, 1)
        return fisher

    for t in range(N_TASKS):
        ds = task_datasets[t]
        opt = optim.Adam(model.parameters(), lr=1e-3)
        n_batches = max(1, ds.n_train // 256)
        for epoch in range(N_EPOCHS_CL):
            model.train()
            for _ in range(n_batches):
                xb, yb = ds.get_train_batch(256)
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad(set_to_none=True)
                loss = crit(model(xb), yb)
                if fisher_list:
                    loss = loss + ewc_penalty()
                loss.backward()
                opt.step()
        fisher_list.append(compute_fisher(ds))
        opt_params_list.append({n: p.data.clone() for n, p in model.named_parameters()})
        for j in range(t + 1):
            R[t, j] = eval_task_fast(model, task_datasets[j], device)

    baselines = np.zeros(N_TASKS)
    for t in range(N_TASKS):
        m = DynamicMLP([784, 256, 128], ["relu", "relu"], 10).to(device)
        ds = task_datasets[t]
        opt2 = optim.Adam(m.parameters(), lr=1e-3)
        for _ in range(N_EPOCHS_CL):
            for _ in range(max(1, ds.n_train // 256)):
                xb, yb = ds.get_train_batch(256)
                opt2.zero_grad(); crit(m(xb.to(device)), yb.to(device)).backward(); opt2.step()
        baselines[t] = eval_task_fast(m, ds, device)

    metrics = compute_cl_metrics(R, baselines)
    return {"seed": seed, "R": R.tolist(), "metrics": metrics}


def run_mlp_cl_seed(seed: int, device: str) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    task_datasets = [FastTaskDataset(t, device=device, train_subset=TRAIN_SUBSET_CL) for t in range(N_TASKS)]
    R = np.zeros((N_TASKS, N_TASKS))
    model = DynamicMLP([784, 128, 64], ["relu", "relu"], 10).to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()

    for t in range(N_TASKS):
        ds = task_datasets[t]
        n_batches = max(1, ds.n_train // 256)
        for epoch in range(N_EPOCHS_CL):
            model.train()
            for _ in range(n_batches):
                xb, yb = ds.get_train_batch(256)
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad(set_to_none=True)
                crit(model(xb), yb).backward()
                opt.step()
        for j in range(t + 1):
            R[t, j] = eval_task_fast(model, task_datasets[j], device)

    baselines = np.zeros(N_TASKS)
    for t in range(N_TASKS):
        m = DynamicMLP([784, 128, 64], ["relu", "relu"], 10).to(device)
        ds = task_datasets[t]
        opt2 = optim.Adam(m.parameters(), lr=1e-3)
        for _ in range(10):
            for _ in range(max(1, ds.n_train // 256)):
                xb, yb = ds.get_train_batch(256)
                opt2.zero_grad(); crit(m(xb.to(device)), yb.to(device)).backward(); opt2.step()
        baselines[t] = eval_task_fast(m, ds, device)

    metrics = compute_cl_metrics(R, baselines)
    return {"seed": seed, "R": R.tolist(), "metrics": metrics}


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"\n{'='*60}")
    print("Experimento 3 — Split-MNIST CL v5 (FastDataset)")
    print(f"{'='*60}")

    dnaty_results, ewc_results, mlp_results = [], [], []

    for seed in SEEDS:
        print(f"\n--- Seed {seed} ---")
        t0 = time.time()
        print("  [dNaty CL]")
        dr = run_dnaty_cl_seed(seed, device)
        dnaty_results.append(dr)
        print(f"  BWT={dr['metrics']['BWT']:.4f} | FWT={dr['metrics']['FWT']:.4f} | {time.time()-t0:.1f}s")

        t0 = time.time()
        print("  [EWC]")
        er = run_ewc_cl_seed(seed, device)
        ewc_results.append(er)
        print(f"  BWT={er['metrics']['BWT']:.4f} | {time.time()-t0:.1f}s")

        t0 = time.time()
        print("  [MLP sem CL]")
        mr = run_mlp_cl_seed(seed, device)
        mlp_results.append(mr)
        print(f"  BWT={mr['metrics']['BWT']:.4f} | {time.time()-t0:.1f}s")

    def mean_m(results, key): return round(float(np.mean([r["metrics"][key] for r in results])), 4)
    def std_m(results, key):  return round(float(np.std( [r["metrics"][key] for r in results])), 4)

    dnaty_bwt = [r["metrics"]["BWT"] for r in dnaty_results]
    ewc_bwt   = [r["metrics"]["BWT"] for r in ewc_results]
    t_stat, p_val, cohen_d = paired_ttest(dnaty_bwt, ewc_bwt)

    print(f"\n{'─'*50}")
    print("RESULTADOS FINAIS — Split-MNIST CL v5")
    for name, results in [("dNaty", dnaty_results), ("EWC", ewc_results), ("MLP", mlp_results)]:
        print(f"  {name:8s} BWT={mean_m(results,'BWT'):.4f}±{std_m(results,'BWT'):.4f}")
    print(f"  dNaty vs EWC: p={p_val:.4f} d={cohen_d:.3f}")

    all_results = {
        "dnaty": dnaty_results, "ewc": ewc_results, "mlp_no_cl": mlp_results,
        "summary": {
            "dnaty_bwt": {"mean": mean_m(dnaty_results, "BWT"), "std": std_m(dnaty_results, "BWT")},
            "ewc_bwt":   {"mean": mean_m(ewc_results,   "BWT"), "std": std_m(ewc_results,   "BWT")},
            "mlp_bwt":   {"mean": mean_m(mlp_results,   "BWT"), "std": std_m(mlp_results,   "BWT")},
            "dnaty_fwt": mean_m(dnaty_results, "FWT"),
            "dnaty_fm":  mean_m(dnaty_results, "FM"),
            "ttest_dnaty_vs_ewc_bwt": {"t": t_stat, "p": p_val, "d": cohen_d},
        },
    }

    out_path = os.path.join(RESULTS_DIR, "exp3_cl_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nSalvo em: {out_path}")
    return all_results


if __name__ == "__main__":
    main()
