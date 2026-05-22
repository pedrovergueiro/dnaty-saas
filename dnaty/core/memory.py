"""
EpisodicMemory — componente central do dNaty.
Implementa eq. 1.4: acumulação com decaimento temporal γ.
Otimizado: scores acumulados incrementalmente (O(1) por update vs O(n) antes).
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class Experience:
    operator: str
    delta_loss: float
    gradient_norm: float
    generation: int
    weight: float = 1.0
    timestamp: int = 0

    @property
    def impact(self) -> float:
        """𝟙[ΔL < 0] · |ΔL| · ‖∇L‖ — só experiências que melhoraram."""
        if self.delta_loss >= 0:
            return 0.0
        return abs(self.delta_loss) * self.gradient_norm


class EpisodicMemory:
    """
    Memória episódica com decaimento temporal γ.
    Scores acumulados incrementalmente — O(1) por update, O(|ops|) por query.
    """

    def __init__(self, max_size: int = 500, decay_gamma: float = 0.99):
        self.experiences: list[Experience] = []
        self.max_size = max_size
        self.gamma = decay_gamma
        self._step = 0
        # Scores acumulados por operador — atualização incremental
        self._scores: dict[str, float] = {}

    def update(self, exp: Experience) -> None:
        # Decaimento global dos scores acumulados — O(|ops|) não O(|mem|)
        for op in self._scores:
            self._scores[op] *= self.gamma

        imp = exp.impact
        if imp > 0:
            self._scores[exp.operator] = self._scores.get(exp.operator, 0.0) + imp

        exp.timestamp = self._step
        self._step += 1
        self.experiences.append(exp)

        if len(self.experiences) > self.max_size:
            self._prune()

    def _prune(self) -> None:
        # Remove as experiências mais antigas/irrelevantes
        # Recalcula scores do zero após prune
        self.experiences.sort(
            key=lambda e: e.impact * (self.gamma ** max(0, self._step - e.timestamp)),
            reverse=True,
        )
        self.experiences = self.experiences[:self.max_size]
        # Recalcular scores
        self._scores = {}
        for e in self.experiences:
            if e.impact > 0:
                decay = self.gamma ** max(0, self._step - e.timestamp)
                self._scores[e.operator] = self._scores.get(e.operator, 0.0) + e.impact * decay

    def query_mutation_probs(self, operators: list[str], tau: float = 1.0) -> dict[str, float]:
        """Softmax sobre scores acumulados — O(|ops|)."""
        vals = np.array(
            [self._scores.get(op, 0.0) for op in operators], dtype=np.float64
        ) / max(tau, 1e-8)
        vals -= vals.max()
        exp_vals = np.exp(vals)
        probs = exp_vals / exp_vals.sum()
        return {op: float(p) for op, p in zip(operators, probs)}

    def operator_counts(self, operators: list[str]) -> dict[str, int]:
        counts: dict[str, int] = {op: 0 for op in operators}
        for e in self.experiences:
            if e.operator in counts and e.impact > 0:
                counts[e.operator] += 1
        return counts
