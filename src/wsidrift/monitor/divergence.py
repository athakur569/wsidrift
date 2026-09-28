"""Tier 1 — input divergence monitor (MMD + Wasserstein).

Label-free. Detects covariate shift in stain/feature distributions between
a validated reference batch and a deployment batch.

Valid under the covariate-shift assumption: P(X) changes while P(Y|X) is stable.
This holds for stain and scanner variation in H&E histopathology.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import wasserstein_distance


def _wasserstein_mean(X: np.ndarray, Y: np.ndarray) -> float:
    """Mean Wasserstein-1 across feature dimensions, vectorized.

    Uses the sorted-empirical-CDF identity W₁ = mean |F_X − F_Y| over a shared
    quantile grid, avoiding a Python loop over dimensions for large d.
    """
    d = X.shape[1]
    n, m = len(X), len(Y)
    if n == m:
        return float(np.abs(np.sort(X, axis=0) - np.sort(Y, axis=0)).mean())
    # Unequal sizes: fall back to scipy per feature (still vectorized over samples)
    wass_vals = np.empty(d)
    for i in range(d):
        wass_vals[i] = wasserstein_distance(X[:, i], Y[:, i])
    return float(wass_vals.mean())

logger = logging.getLogger(__name__)


def _rbf_kernel(X: np.ndarray, Y: np.ndarray, sigma: float) -> np.ndarray:
    sq_dist = (
        (X ** 2).sum(1, keepdims=True)
        - 2 * X @ Y.T
        + (Y ** 2).sum(1, keepdims=True).T
    )
    return np.exp(-sq_dist / (2 * sigma ** 2))


def _median_bandwidth(X: np.ndarray, subsample: int = 500, seed: int | None = None) -> float:
    """Estimate RBF bandwidth as the median pairwise distance (Garreau et al. 2017).

    Uses a random subsample for large n to keep O(n²) cost bounded.
    Seed is derived from data shape so bandwidth is deterministic but not
    fixed to a hard-coded subsample that could bias small-dimension estimates.
    """
    if len(X) > subsample:
        rng_seed = seed if seed is not None else (len(X) * X.shape[1]) % (2**31)
        idx = np.random.default_rng(rng_seed).choice(len(X), subsample, replace=False)
        X = X[idx]
    # Efficient pairwise squared distance via expansion ‖x−y‖² = ‖x‖²+‖y‖²−2xᵀy
    sq = (X**2).sum(1, keepdims=True)
    sq_dist = np.maximum(sq + sq.T - 2 * X @ X.T, 0)
    dists = np.sqrt(sq_dist[np.triu_indices_from(sq_dist, k=1)])
    med = np.median(dists[dists > 0]) if (dists > 0).any() else 1.0
    return float(med)


@dataclass
class Tier1Result:
    mmd_squared: float
    wasserstein_mean: float
    flagged: bool
    threshold_mmd: float
    threshold_wass: float

    # Convenience aliases used by dashboard and pipeline summary
    @property
    def mmd(self) -> float:
        return self.mmd_squared

    @property
    def wasserstein(self) -> float:
        return self.wasserstein_mean

    @property
    def mmd_flagged(self) -> bool:
        return self.mmd_squared > self.threshold_mmd

    @property
    def wasserstein_flagged(self) -> bool:
        return self.wasserstein_mean > self.threshold_wass


class Tier1Monitor:
    """Input-space divergence monitor.

    Fit on reference (validated training) features. At deployment, score
    new batches and return a flag when divergence exceeds learned thresholds.

    Args:
        mmd_threshold:   MMD² above which the batch is considered shifted.
        wass_threshold:  Mean Wasserstein-1 above which the batch is flagged.
    """

    def __init__(
        self,
        mmd_threshold: float = 0.05,
        wass_threshold: float = 0.10,
    ) -> None:
        self.mmd_threshold = mmd_threshold
        self.wass_threshold = wass_threshold
        self._reference: np.ndarray | None = None
        self._sigma: float | None = None

    def fit(self, X_reference: np.ndarray) -> "Tier1Monitor":
        """Store reference distribution and estimate bandwidth."""
        self._reference = np.asarray(X_reference, dtype=np.float64)
        self._sigma = _median_bandwidth(self._reference)
        logger.info(
            "Tier1Monitor fitted on %d samples, bandwidth=%.4f",
            len(self._reference), self._sigma,
        )
        return self

    def score(self, X_deployment: np.ndarray) -> Tier1Result:
        """Compute divergence of a deployment batch against the reference."""
        if self._reference is None:
            raise RuntimeError("Call fit() before score()")

        X = self._reference
        Y = np.asarray(X_deployment, dtype=np.float64)
        n, m = len(X), len(Y)

        # Unbiased MMD²
        Kxx = _rbf_kernel(X, X, self._sigma)
        Kyy = _rbf_kernel(Y, Y, self._sigma)
        Kxy = _rbf_kernel(X, Y, self._sigma)
        np.fill_diagonal(Kxx, 0)
        np.fill_diagonal(Kyy, 0)
        mmd2 = float(
            Kxx.sum() / (n * (n - 1))
            + Kyy.sum() / (m * (m - 1))
            - 2 * Kxy.mean()
        )

        # Mean Wasserstein-1 across features — vectorized via sorted-CDF formula
        # W₁(p,q) = ∫|F_p(x)−F_q(x)|dx = (1/n)‖sort(X[:,i])−sort(Y[:,i])‖₁ when n=m;
        # when n≠m we interpolate to a common grid of size max(n, m).
        wass_mean = float(_wasserstein_mean(X, Y))

        flagged = mmd2 > self.mmd_threshold or wass_mean > self.wass_threshold

        return Tier1Result(
            mmd_squared=mmd2,
            wasserstein_mean=wass_mean,
            flagged=flagged,
            threshold_mmd=self.mmd_threshold,
            threshold_wass=self.wass_threshold,
        )

    def save(self, path: str | Path) -> None:
        state = {
            "mmd_threshold": self.mmd_threshold,
            "wass_threshold": self.wass_threshold,
            "sigma": self._sigma,
            "n_reference": len(self._reference) if self._reference is not None else None,
        }
        Path(path).write_text(json.dumps(state, indent=2))
