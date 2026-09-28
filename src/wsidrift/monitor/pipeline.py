"""MonitorPipeline: unified entry point that runs Tier 1 → Tier 2 → validity."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .divergence import Tier1Monitor, Tier1Result
from .embedding import Tier2Monitor, Tier2Result
from .validity_score import ValidityMonitor, ValidityResult


@dataclass
class PipelineResult:
    tier1: Tier1Result
    tier2: Tier2Result
    validity: ValidityResult

    @property
    def invalid(self) -> bool:
        return self.validity.invalid

    def summary(self) -> dict:
        return {
            "t1_mmd": self.tier1.mmd,
            "t1_wasserstein": self.tier1.wasserstein,
            "t1_flagged": self.tier1.flagged,
            "t2_mahalanobis": self.tier2.mahalanobis_mean,
            "t2_energy": self.tier2.energy_mean,
            "t2_flagged": self.tier2.flagged,
            "invalid": self.validity.invalid,
        }


class MonitorPipeline:
    """Run the full Tier 1 + Tier 2 monitor pipeline in one call.

    Args:
        t1_mmd_threshold:    MMD² flag threshold.
        t1_wass_threshold:   Wasserstein flag threshold.
        t2_mahal_threshold:  Mahalanobis flag threshold.
        t2_energy_threshold: Energy score flag threshold.
    """

    def __init__(
        self,
        t1_mmd_threshold: float = 0.5,
        t1_wass_threshold: float = 0.5,
        t2_mahal_threshold: float = 3.0,
        t2_energy_threshold: float = -10.0,
    ) -> None:
        self._t1 = Tier1Monitor(
            mmd_threshold=t1_mmd_threshold,
            wasserstein_threshold=t1_wass_threshold,
        )
        self._t2 = Tier2Monitor(
            mahal_threshold=t2_mahal_threshold,
            energy_threshold=t2_energy_threshold,
        )
        self._vm = ValidityMonitor(
            t1_threshold=t1_mmd_threshold,
            t2_threshold=t2_mahal_threshold,
        )

    def fit(self, reference_embeddings: np.ndarray) -> "MonitorPipeline":
        self._t1.fit(reference_embeddings)
        self._t2.fit(reference_embeddings)
        self._vm.fit(reference_embeddings)
        return self

    def score(
        self,
        deployment_embeddings: np.ndarray,
        deployment_logits: np.ndarray | None = None,
    ) -> PipelineResult:
        if deployment_logits is None:
            deployment_logits = np.zeros((len(deployment_embeddings), 2))
        r1 = self._t1.score(deployment_embeddings)
        r2 = self._t2.score(deployment_embeddings, deployment_logits)
        rv = self._vm.score(deployment_embeddings, deployment_logits)
        return PipelineResult(tier1=r1, tier2=r2, validity=rv)
