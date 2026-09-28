"""Tier 2 — model-response monitor (Mahalanobis distance + energy score).

One step closer to correctness than Tier 1: measures drift in the encoder's
internal representation, not just the raw input distribution.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class Tier2Result:
    mahalanobis_mean: float
    mahalanobis_max: float
    energy_mean: float
    flagged: bool
    threshold_maha: float
    threshold_energy: float


class Tier2Monitor:
    """Embedding-space drift monitor.

    Fit on encoder embeddings from the validated reference set.
    At deployment, score new batches via Mahalanobis distance and energy score.

    Args:
        maha_threshold:   Mean Mahalanobis distance above which batch is flagged.
        energy_threshold: Mean energy score below which (more OOD) batch is flagged.
    """

    def __init__(
        self,
        maha_threshold: float = 3.0,
        energy_threshold: float = -25.0,
    ) -> None:
        self.maha_threshold = maha_threshold
        self.energy_threshold = energy_threshold
        self._mu: np.ndarray | None = None
        self._precision: np.ndarray | None = None

    def fit(self, embeddings: np.ndarray) -> "Tier2Monitor":
        """Fit on reference embeddings (n_samples, embedding_dim)."""
        Z = np.asarray(embeddings, dtype=np.float64)
        n, d = Z.shape
        self._mu = Z.mean(axis=0)
        if n <= d + 1:
            # Underdetermined: fall back to diagonal precision (variance per dim).
            # This avoids a singular or poorly conditioned covariance matrix when
            # the reference batch is small relative to embedding dimension.
            var = Z.var(axis=0) + 1e-4
            self._precision = np.diag(1.0 / var)
            logger.warning(
                "Tier2Monitor: n=%d <= d+1=%d, using diagonal precision (small-batch fallback)",
                n, d + 1,
            )
        else:
            cov = np.cov(Z.T)
            # Scale regularization to trace for relative stability across embedding dims.
            reg = max(1e-4, 1e-3 * np.trace(cov) / d)
            cov += np.eye(d) * reg
            self._precision = np.linalg.inv(cov)
        logger.info(
            "Tier2Monitor fitted: %d samples, dim=%d", n, d
        )
        return self

    def score(self, embeddings: np.ndarray, logits: np.ndarray | None = None) -> Tier2Result:
        """Score a deployment batch.

        Args:
            embeddings: Encoder outputs (n_samples, d).
            logits:     Optional classifier logits (n_samples, n_classes) for energy score.
        """
        if self._mu is None:
            raise RuntimeError("Call fit() before score()")

        Z = np.asarray(embeddings, dtype=np.float64)
        maha = self._mahalanobis_batch(Z)

        energy_mean = self._energy_score(logits) if logits is not None else 0.0

        flagged = (
            maha.mean() > self.maha_threshold
            or (logits is not None and energy_mean < self.energy_threshold)
        )

        return Tier2Result(
            mahalanobis_mean=float(maha.mean()),
            mahalanobis_max=float(maha.max()),
            energy_mean=float(energy_mean),
            flagged=flagged,
            threshold_maha=self.maha_threshold,
            threshold_energy=self.energy_threshold,
        )

    @classmethod
    def from_encoder(
        cls,
        encoder: nn.Module,
        reference_loader: "torch.utils.data.DataLoader",
        device: str = "cpu",
        **kwargs,
    ) -> "Tier2Monitor":
        """Convenience: extract embeddings with encoder and fit."""
        monitor = cls(**kwargs)
        embeddings = []
        encoder.eval().to(device)
        with torch.no_grad():
            for batch in reference_loader:
                x = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
                z = encoder(x)
                if isinstance(z, torch.Tensor):
                    embeddings.append(z.cpu().numpy())
        monitor.fit(np.vstack(embeddings))
        return monitor

    # ------------------------------------------------------------------

    def _mahalanobis_batch(self, Z: np.ndarray) -> np.ndarray:
        diff = Z - self._mu
        return np.sqrt(np.maximum((diff @ self._precision * diff).sum(axis=1), 0))

    @staticmethod
    def _energy_score(logits: np.ndarray, temperature: float = 1.0) -> float:
        """Energy score: -T * log(sum(exp(logits/T))). More negative = more OOD."""
        logits = np.asarray(logits, dtype=np.float64)
        log_sum_exp = temperature * np.log(
            np.exp(logits / temperature).sum(axis=1)
        )
        return float(-log_sum_exp.mean())
