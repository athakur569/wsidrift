"""ΔG computation across E1-E9 conditions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import roc_auc_score


@dataclass
class DeltaGapResult:
    condition: str
    cohort: str
    auc_in_mean: float
    auc_in_std: float
    auc_cross_mean: float
    auc_cross_std: float
    delta_gap_mean: float
    delta_gap_std: float

    def __str__(self) -> str:
        return (
            f"{self.condition} [{self.cohort}]: "
            f"ΔG = {self.delta_gap_mean:+.4f} ± {self.delta_gap_std:.4f}  "
            f"(in={self.auc_in_mean:.4f}, cross={self.auc_cross_mean:.4f})"
        )


class DeltaGap:
    """Compute ΔG = AUC_in − AUC_cross across multiple seeds.

    Args:
        condition: Condition name (e.g. 'E1').
        cohort:    Cohort name (e.g. 'BRCA').
    """

    def __init__(self, condition: str, cohort: str) -> None:
        self.condition = condition
        self.cohort = cohort
        self._in_auc: list[float] = []
        self._cross_auc: list[float] = []

    def add_seed(
        self,
        y_true_in: np.ndarray,
        y_score_in: np.ndarray,
        y_true_cross: np.ndarray,
        y_score_cross: np.ndarray,
    ) -> None:
        self._in_auc.append(float(roc_auc_score(y_true_in, y_score_in)))
        self._cross_auc.append(float(roc_auc_score(y_true_cross, y_score_cross)))

    def compute(self) -> DeltaGapResult:
        if not self._in_auc:
            raise ValueError("No seeds added — call add_seed() first.")
        dg = [i - c for i, c in zip(self._in_auc, self._cross_auc)]
        # Sample SD (ddof=1) across seeds; undefined (NaN) for a single seed.
        sd = lambda v: float(np.std(v, ddof=1)) if len(v) > 1 else float("nan")  # noqa: E731
        return DeltaGapResult(
            condition=self.condition,
            cohort=self.cohort,
            auc_in_mean=float(np.mean(self._in_auc)),
            auc_in_std=sd(self._in_auc),
            auc_cross_mean=float(np.mean(self._cross_auc)),
            auc_cross_std=sd(self._cross_auc),
            delta_gap_mean=float(np.mean(dg)),
            delta_gap_std=sd(dg),
        )
