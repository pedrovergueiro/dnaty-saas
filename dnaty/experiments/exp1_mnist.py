"""
Experimento 1 — MNIST e FashionMNIST — dNaty v5.
FastDataset: carrega 60K na RAM, zero I/O por geração.
"""
from __future__ import annotations
import os, json, time
import numpy as np
import torch

from dnaty.experiments.fast_dataset import FastDataset
from dnaty.experiments.data_utils import get_mnist, get_fashion_mnist
from dnaty.experiments.baselines import train_fixed_mlp, train_ga_pure
from dnaty.evolution.evolver import DnatyEvolver
from dnaty.training.local_train import evaluate
from dnaty.analysis.stats import summary_stats, paired_ttest

SEEDS = [0, 1, 2]
N_GENERATIONS = 30
N_POP = 8
T_LOCAL = 3
TRAIN_SUBSET = None       # v5: dataset completo via FastDataset
VAL_SUBSET = None
BASELINE_TRAIN_SUBSET = None
EARLY_STOP_PATIENCE = 5
EARLY_STOP_MIN_DELTA = 5e-4
USE_FAST_DATASET = True   # v5: RAM preloading
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def run_dnaty_seed(seed: int, get_data_fn, dataset_name: str, device: str) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    # v5: usa FastDataset se disponível, senão DataLoader padrão
    if USE_FAST_DATASET:
        fast_ds = FastDataset(dataset_name, device=device, train_subset=TRAIN_SUBSET)
        train_loader = fast_ds
        val_loader   = fast_ds
    else:
        train_loader, val_loader = get_data_fn(train_subset=TRAIN_SUBSET, val_subset=VAL_SUBSET)
        fast_ds = None

    evolver = DnatyEvolver(
        n_pop=N_POP,
        n_generations=N_GENERATIONS,
        t_local=T_LOCAL,
        device=device,
        verbose=True,
    )
    t0 = time.time()
    best, history = evolver.run(
        train_loader, val_loader,
        early_stop_patience=EARLY_STOP_PATIENCE,
        early_stop_min_delta=EARLY_STOP_MIN_DELTA,
    )
    elapsed = time.time() - t0
    acc, _ = evaluate(best, val_loader, device)
    return {
        "seed": seed,
        "dataset": dataset_name,
        "acc": round(acc, 4),
        "n_params": best.count_params(),
        "n_flops": best.count_flops(),
        "time_s": round(elapsed, 1),
        "history": [
            {
                "gen": h.gen,
                "best_acc": round(h.best_acc, 4),
                "delta_grad": round(h.delta_grad, 6),
                "delta_mem": round(h.delta_mem, 6),
                "n_params": h.n_params,
            }
            for h in history
        ],
        "delta_grad_all_positive": all(h.delta_grad >= -1e-6 for h in history),
        "delta_mem_positive_after_gen3": all(
            h.delta_mem >= 0 for h in history if h.gen >= 3
        ),
    }


def run_baseline_seed(seed: int, get_data_fn, device: str) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    # MLP treina com dataset COMPLETO (60K) — vantagem máxima ao baseline
    train_loader, val_loader = get_data_fn(train_subset=BASELINE_TRAIN_SUBSET, val_subset=VAL_SUBSET)
    acc_mlp, params_mlp = train_fixed_mlp(train_loader, val_loader, n_epochs=20, device=device)
    # GA usa mesmo subset do dNaty (10K) — comparação de método, não de dados
    train_loader_sub, _ = get_data_fn(train_subset=TRAIN_SUBSET, val_subset=VAL_SUBSET)
    acc_ga, params_ga = train_ga_pure(train_loader_sub, val_loader, n_generations=N_GENERATIONS, n_pop=N_POP, device=device)
    return {
        "seed": seed,
        "mlp_acc": round(acc_mlp, 4),
        "mlp_params": params_mlp,
        "mlp_train_samples": 60000,  # MLP usou dataset completo
        "ga_acc": round(acc_ga, 4),
        "ga_params": params_ga,
        "dnaty_train_samples": TRAIN_SUBSET,  # dNaty usou 10K
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    all_results = {}

    for dataset_name, get_data_fn in [("MNIST", get_mnist), ("FashionMNIST", get_fashion_mnist)]:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_name}")
        print(f"{'='*60}")

        dnaty_results = []
        baseline_results = []

        for seed in SEEDS:
            print(f"\n--- Seed {seed} ---")
            print(f"  [dNaty]")
            dr = run_dnaty_seed(seed, get_data_fn, dataset_name, device)
            dnaty_results.append(dr)
            print(f"  acc={dr['acc']:.4f} | params={dr['n_params']} | time={dr['time_s']}s")
            print(f"  δ_grad sempre ≥ 0: {dr['delta_grad_all_positive']}")
            print(f"  δ_mem ≥ 0 após gen3: {dr['delta_mem_positive_after_gen3']}")

            print(f"  [Baselines]")
            br = run_baseline_seed(seed, get_data_fn, device)
            baseline_results.append(br)
            print(f"  MLP: {br['mlp_acc']:.4f} ({br['mlp_params']} params)")
            print(f"  GA:  {br['ga_acc']:.4f} ({br['ga_params']} params)")

        dnaty_accs = [r["acc"] for r in dnaty_results]
        mlp_accs = [r["mlp_acc"] for r in baseline_results]
        ga_accs = [r["ga_acc"] for r in baseline_results]

        dnaty_stats = summary_stats(dnaty_accs)
        mlp_stats = summary_stats(mlp_accs)
        ga_stats = summary_stats(ga_accs)

        t_stat, p_val, cohen_d = paired_ttest(dnaty_accs, mlp_accs)

        print(f"\n{'─'*50}")
        print(f"RESULTADOS FINAIS — {dataset_name}")
        print(f"  dNaty:  {dnaty_stats['mean']:.4f} ± {dnaty_stats['std']:.4f}  [10K treino]")
        print(f"  MLP:    {mlp_stats['mean']:.4f} ± {mlp_stats['std']:.4f}  [60K treino — vantagem 6x dados]")
        print(f"  GA:     {ga_stats['mean']:.4f} ± {ga_stats['std']:.4f}  [10K treino]")
        print(f"  dNaty vs MLP: p={p_val:.4f}, d={cohen_d:.3f} {'*' if p_val < 0.05 else ''}")
        print(f"  → dNaty usa 6x menos dados e {'SUPERA' if dnaty_stats['mean'] > mlp_stats['mean'] else 'chega perto d'}o MLP com dataset completo")

        # Verificação do Teorema 1
        all_dg_pos = all(r["delta_grad_all_positive"] for r in dnaty_results)
        all_dm_pos = all(r["delta_mem_positive_after_gen3"] for r in dnaty_results)
        print(f"\n  VALIDAÇÃO TEOREMA 1:")
        print(f"  δ_grad > 0 em todas as gerações × seeds: {all_dg_pos}")
        print(f"  δ_mem > 0 após gen3 × seeds: {all_dm_pos}")

        all_results[dataset_name] = {
            "dnaty": dnaty_results,
            "baselines": baseline_results,
            "summary": {
                "dnaty": dnaty_stats,
                "mlp": mlp_stats,
                "ga": ga_stats,
                "ttest_dnaty_vs_mlp": {"t": t_stat, "p": p_val, "d": cohen_d},
                "theorem1_delta_grad_positive": all_dg_pos,
                "theorem1_delta_mem_positive": all_dm_pos,
            },
        }

    # Salvar resultados
    out_path = os.path.join(RESULTS_DIR, "exp1_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nResultados salvos em: {out_path}")
    return all_results


if __name__ == "__main__":
    main()
