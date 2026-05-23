"""
dNatyEvolver — Algorithm 1 otimizado para velocidade máxima.

Otimizações vs versão anterior:
  1. Treino sequencial otimizado (CUDA não beneficia de múltiplos streams)
  2. evaluate com torch.inference_mode
  3. NSGA-II vetorizado com numpy
  4. EpisodicMemory com scores incrementais O(1)
  5. model.to(device) idempotente
"""
from __future__ import annotations
import numpy as np
import torch
from tqdm import tqdm
import os

from dnaty.core.individual import Individual
from dnaty.core.arch import DynamicMLP
from dnaty.core.memory import EpisodicMemory, Experience
from dnaty.operators.mutations import OPERATORS, apply_operator
from dnaty.evolution.selection import nsga2_select
from dnaty.training.local_train import local_train, evaluate


class GenerationLog:
    __slots__ = ("gen", "best_acc", "delta_grad", "delta_mem", "op_counts", "n_params")

    def __init__(self, gen, best_acc, delta_grad, delta_mem, op_counts, n_params):
        self.gen = gen
        self.best_acc = best_acc
        self.delta_grad = delta_grad
        self.delta_mem = delta_mem
        self.op_counts = op_counts
        self.n_params = n_params

    def __repr__(self):
        return (f"Gen {self.gen:3d} | acc={self.best_acc:.4f} | "
                f"δ_grad={self.delta_grad:.5f} | δ_mem={self.delta_mem:.5f} | "
                f"params={self.n_params}")


class DnatyEvolver:
    def __init__(
        self,
        n_pop: int = 20,
        n_generations: int = 50,
        t_local: int = 3,
        lr: float = 1e-3,
        lambda1: float = 1e-4,
        lambda2: float = 1e-3,
        memory_gamma: float = 0.99,
        memory_tau: float = 1.0,
        top_k_pct: float = 0.03,
        device: str = "cpu",
        input_size: int = 784,
        n_classes: int = 10,
        init_hidden: list[int] | None = None,
        verbose: bool = True,
        n_threads: int | None = None,
        batch_size: int = 512,
    ):
        self.n_pop = n_pop
        self.n_generations = n_generations
        self.t_local = t_local
        self.lr = lr
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.memory_gamma = memory_gamma
        self.memory_tau = memory_tau
        self.top_k_pct = top_k_pct
        self.device = device
        self.input_size = input_size
        self.n_classes = n_classes
        self.init_hidden = init_hidden or [128, 64]
        self.verbose = verbose
        self.n_threads = n_threads or 1
        self.batch_size = batch_size
        self.history: list[GenerationLog] = []
        self.population: list[Individual] = []
        self.shared_memory = EpisodicMemory(max_size=1000, decay_gamma=memory_gamma)

    def _make_individual(self) -> Individual:
        sizes = [self.input_size] + self.init_hidden
        acts = ["relu"] * len(self.init_hidden)
        model = DynamicMLP(sizes, acts, self.n_classes)
        return Individual(model, EpisodicMemory(decay_gamma=self.memory_gamma))

    def _init_population(self) -> None:
        self.population = [self._make_individual() for _ in range(self.n_pop)]

    def _fitness(self, ind: Individual) -> tuple[float, float, float]:
        cost = ind.count_params() * 1e-5 + ind.count_flops() * 1e-8
        ind.fitness = (ind.acc, -cost, 0.0)
        return ind.fitness

    def _eval_population(
        self,
        population: list[Individual],
        val_loader: torch.utils.data.DataLoader,
    ) -> list[tuple[float, float, float]]:
        """Avalia todos os indivíduos — sequencial mas com inference_mode."""
        fitnesses = []
        for ind in population:
            acc, _ = evaluate(ind, val_loader, self.device)
            ind.acc = acc
            fitnesses.append(self._fitness(ind))
        return fitnesses

    def _train_parallel(
        self,
        mutated: list[Individual],
        train_loader: torch.utils.data.DataLoader,
    ) -> tuple[list[float], list[float], list[float]]:
        """Treina todos os indivíduos sequencialmente — CUDA não é thread-safe para múltiplos streams."""
        loss_before, loss_after, grad_norms = [], [], []
        for ind in mutated:
            lb, la, gn = local_train(
                ind, train_loader,
                self.t_local, self.lr,
                self.lambda1, self.lambda2,
                device=self.device,
                batch_size=self.batch_size,
            )
            loss_before.append(lb)
            loss_after.append(la)
            grad_norms.append(gn)
            ind.last_grad_norm = gn
            ind.last_delta_loss = la - lb
        return loss_before, loss_after, grad_norms

    def _mutate_population(self, population: list[Individual]) -> list[Individual]:
        op_probs = self.shared_memory.query_mutation_probs(OPERATORS, tau=self.memory_tau)
        ops   = list(op_probs.keys())
        probs = list(op_probs.values())
        mutated = []
        for ind in population:
            op = np.random.choice(ops, p=probs)
            new_ind, success = apply_operator(ind, op)
            if not success or not new_ind.model.is_valid():
                new_ind = ind.clone()
                new_ind.last_op = "no_op"
            mutated.append(new_ind)
        return mutated

    def _update_memory(
        self,
        mutated: list[Individual],
        prev_best_acc: float,
        grad_norms: list[float],
        prev_accs: list[float],  # acurácia de cada indivíduo ANTES da mutação
    ) -> float:
        delta_mem = 0.0
        for i, ind in enumerate(mutated):
            if ind.last_op == "no_op":
                continue
            # Melhora em relação ao pai (não só ao melhor global)
            parent_acc = prev_accs[i] if i < len(prev_accs) else prev_best_acc
            if ind.acc > parent_acc + 1e-5:
                exp = Experience(
                    operator=ind.last_op,
                    delta_loss=-(ind.acc - parent_acc),
                    gradient_norm=grad_norms[i],
                    generation=len(self.history),
                )
                self.shared_memory.update(exp)
                delta_mem += exp.impact
        return delta_mem

    def run(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        early_stop_patience: int = 8,
        early_stop_min_delta: float = 1e-4,
        progress_callback=None,
    ) -> tuple[Individual, list[GenerationLog]]:
        self._init_population()

        fitnesses = self._eval_population(self.population, val_loader)
        prev_best_acc = max(ind.acc for ind in self.population)
        best_acc_ever = prev_best_acc
        no_improve_count = 0

        pbar = tqdm(range(1, self.n_generations + 1), disable=not self.verbose)
        for gen in pbar:
            # ── FASE 1: Variação guiada pela memória ──────────────────────
            prev_accs = [ind.acc for ind in self.population]  # acurácia dos pais
            mutated = self._mutate_population(self.population)

            # ── FASE 2: Treino local (Lema 2) ─────────────────────────────
            loss_before_list, loss_after_list, grad_norms = self._train_parallel(
                mutated, train_loader
            )
            delta_grad = float(np.mean([
                b - a for b, a in zip(loss_before_list, loss_after_list)
            ]))

            # ── FASE 3: Avaliação multiobjetivo ───────────────────────────
            mut_fitnesses = self._eval_population(mutated, val_loader)

            # ── FASE 4: Seleção NSGA-II ───────────────────────────────────
            combined_pop = self.population + mutated
            combined_fit = fitnesses + mut_fitnesses
            self.population, fitnesses = nsga2_select(combined_pop, combined_fit, self.n_pop)

            # ── FASE 5: Atualização de memória (Lema 1) ───────────────────
            delta_mem = self._update_memory(mutated, prev_best_acc, grad_norms, prev_accs)

            best_ind = max(self.population, key=lambda ind: ind.acc)
            prev_best_acc = best_ind.acc

            log = GenerationLog(
                gen=gen,
                best_acc=best_ind.acc,
                delta_grad=max(delta_grad, 0.0),
                delta_mem=delta_mem,
                op_counts=dict(self.shared_memory.operator_counts(OPERATORS)),
                n_params=best_ind.count_params(),
            )
            self.history.append(log)

            if self.verbose:
                pbar.set_description(str(log))

            if progress_callback is not None:
                try:
                    progress_callback(log)
                except Exception:
                    pass

            # ── Early stopping ────────────────────────────────────────────
            if best_ind.acc > best_acc_ever + early_stop_min_delta:
                best_acc_ever = best_ind.acc
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count >= early_stop_patience:
                    if self.verbose:
                        print(f"\nEarly stop na gen {gen} — sem melhora em {early_stop_patience} gerações. acc={best_acc_ever:.4f}")
                    break

        best = max(self.population, key=lambda ind: ind.acc)
        return best, self.history
