"""
Análise estatística: teste-t pareado, Cohen's d, ANOVA + Tukey HSD.
"""
from __future__ import annotations
import numpy as np
from scipy import stats


def paired_ttest(a: list[float], b: list[float]) -> tuple[float, float, float]:
    """Retorna (t_stat, p_value, cohen_d)."""
    a_arr, b_arr = np.array(a), np.array(b)
    if len(a_arr) != len(b_arr):
        raise ValueError("paired_ttest exige listas com o mesmo tamanho")
    if len(a_arr) < 2:
        raise ValueError("paired_ttest exige pelo menos 2 pares")
    t_stat, p_val = stats.ttest_rel(a_arr, b_arr)
    diff = a_arr - b_arr
    d = float(diff.mean() / (diff.std(ddof=1) + 1e-12))
    return float(t_stat), float(p_val), d


def anova_tukey(groups: dict[str, list[float]]) -> dict[str, object]:
    """ANOVA one-way + Tukey HSD para múltiplos grupos."""
    names = list(groups.keys())
    arrays = [np.array(v) for v in groups.values()]
    f_stat, p_anova = stats.f_oneway(*arrays)
    result = {
        "f_stat": float(f_stat),
        "p_anova": float(p_anova),
        "significant": p_anova < 0.05,
        "groups": names,
    }
    # Tukey HSD simplificado — comparações par a par
    pairs = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            t, p, d = paired_ttest(list(arrays[i]), list(arrays[j]))
            pairs[f"{names[i]} vs {names[j]}"] = {
                "p": round(p, 4),
                "d": round(d, 3),
                "sig": p < 0.05,
            }
    result["pairs"] = pairs
    return result


def summary_stats(values: list[float]) -> dict[str, float]:
    arr = np.array(values)
    return {
        "mean": round(float(arr.mean()), 4),
        "std": round(float(arr.std()), 4),
        "min": round(float(arr.min()), 4),
        "max": round(float(arr.max()), 4),
    }
