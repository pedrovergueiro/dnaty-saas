"""
NSGA-II vetorizado com numpy — O(n²) mas sem loops Python internos.
"""
from __future__ import annotations
import numpy as np
from dnaty.core.individual import Individual


def fast_non_dominated_sort(fitnesses: list[tuple]) -> list[list[int]]:
    """
    Vetorizado: converte fitnesses para matriz numpy e usa broadcasting.
    """
    n = len(fitnesses)
    F = np.array(fitnesses, dtype=np.float64)  # (n, n_obj)

    # F[i] domina F[j] se F[i] >= F[j] em tudo e F[i] > F[j] em algo
    # Broadcasting: (n,1,obj) >= (1,n,obj) → (n,n,obj)
    ge = F[:, None, :] >= F[None, :, :]  # (n,n,obj) — i >= j em cada obj
    gt = F[:, None, :] >  F[None, :, :]  # (n,n,obj) — i >  j em algum obj

    dominates = ge.all(axis=2) & gt.any(axis=2)  # (n,n) — i domina j
    np.fill_diagonal(dominates, False)

    domination_count = dominates.sum(axis=0)   # quantos dominam i
    dominated_by = [list(np.where(dominates[i])[0]) for i in range(n)]

    fronts: list[list[int]] = [list(np.where(domination_count == 0)[0])]
    current = 0
    while fronts[current]:
        next_front: list[int] = []
        for i in fronts[current]:
            for j in dominated_by[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    next_front.append(j)
        fronts.append(next_front)
        current += 1

    return [f for f in fronts if f]


def crowding_distance(fitnesses: list[tuple], front: list[int]) -> dict[int, float]:
    if len(front) <= 2:
        return {i: float("inf") for i in front}
    n_obj = len(fitnesses[0])
    cd: dict[int, float] = {i: 0.0 for i in front}
    F = np.array([fitnesses[i] for i in front], dtype=np.float64)
    for obj in range(n_obj):
        order = np.argsort(F[:, obj])
        sorted_front = [front[k] for k in order]
        cd[sorted_front[0]]  = float("inf")
        cd[sorted_front[-1]] = float("inf")
        f_min, f_max = F[order[0], obj], F[order[-1], obj]
        if f_max == f_min:
            continue
        span = f_max - f_min
        for k in range(1, len(sorted_front) - 1):
            cd[sorted_front[k]] += (F[order[k+1], obj] - F[order[k-1], obj]) / span
    return cd


def nsga2_select(
    population: list[Individual],
    fitnesses: list[tuple],
    n_select: int,
) -> tuple[list[Individual], list[tuple]]:
    fronts = fast_non_dominated_sort(fitnesses)
    selected_idx: list[int] = []
    for front in fronts:
        if len(selected_idx) + len(front) <= n_select:
            selected_idx.extend(front)
        else:
            cd = crowding_distance(fitnesses, front)
            sorted_front = sorted(front, key=lambda i: cd[i], reverse=True)
            needed = n_select - len(selected_idx)
            selected_idx.extend(sorted_front[:needed])
            break
    return [population[i] for i in selected_idx], [fitnesses[i] for i in selected_idx]
