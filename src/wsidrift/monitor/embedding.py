"""Tier 2 — model-response monitor (Mahalanobis distance + energy score).

One step closer to correctness than Tier 1: measures drift in the encoder's
internal representation, not just the raw input distribution.

By default thresholds are calibrated on the reference set at false-alarm rate
``alpha``: reference Mahalanobis distances are computed out-of-sample (2-fold
cross-fitting), and the threshold for a deployment batch of size m is the
(1 − α) quantile of bootstrap batch means of size m. Passing explicit thresholds
switches to fixed-threshold mode.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp

logger = logging.getLogger(__name__)

_MIN_CROSSFIT = 40


@dataclass
class Tier2Result:
    mahalanobis_mean: float
    mahalanobis_max: float
    energy_mean: float | None
    flagged: bool
    threshold_maha: float
    threshold_energy: float | None

    @property
    def mahalanobis_flagged(self) -> bool:
        return self.mahalanobis_mean > self.threshold_maha

    @property
    def energy_flagged(self) -> bool:
        if self.energy_mean is None or self.threshold_energy is None:
            return False
        return self.energy_mean > self.threshold_energy


def _fit_gaussian(Z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n, d = Z.shape
    mu = Z.mean(axis=0)
    if n <= d + 1:
        # Too few samples for a full covariance: use per-dimension variance.
        return mu, np.diag(1.0 / (Z.var(axis=0) + 1e-4))
    cov = np.cov(Z.T)
    cov += np.eye(d) * max(1e-4, 1e-3 * np.trace(cov) / d)
    return mu, np.linalg.inv(cov)


def _mahalanobis(Z: np.ndarray, mu: np.ndarray, precision: np.ndarray) -> np.ndarray:
    diff = Z - mu
    return np.sqrt(np.maximum((diff @ precision * diff).sum(axis=1), 0))


def energy(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Per-sample energy E(x) = −T·logsumexp(f(x)/T) (Liu et al. 2020).

    Lower energy = more in-distribution; out-of-distribution inputs score higher.
    """
    logits = np.asarray(logits, dtype=np.float64)
    return -temperature * logsumexp(logits / temperature, axis=1)


class Tier2Monitor:
    """Embedding-space drift monitor.

    Args:
        alpha:            False-alarm rate for calibrated thresholds (default mode).
        n_bootstrap:      Bootstrap batches used to set the per-batch-size threshold.
        maha_threshold:   Fixed threshold on the batch-mean Mahalanobis distance.
        energy_threshold: Fixed threshold on batch-mean energy; flags when exceeded.
        seed:             Seed for cross-fitting and bootstrap.
    """

    def __init__(
        self,
        maha_threshold: float | None = None,
        energy_threshold: float | None = None,
        alpha: float = 0.05,
        n_bootstrap: int = 2000,
        seed: int = 0,
    ) -> None:
        self.maha_threshold = maha_threshold
        self.energy_threshold = energy_threshold
        self.alpha = alpha
        self.n_bootstrap = n_bootstrap
        self.seed = seed
        self._mu: np.ndarray | None = None
        self._ref_groups: np.ndarray | None = None
        self._precision: np.ndarray | None = None
        self._ref_maha: np.ndarray | None = None
        self._ref_energy: np.ndarray | None = None

    def fit(self, embeddings: np.ndarray, logits: np.ndarray | None = None,
            groups: np.ndarray | None = None) -> "Tier2Monitor":
        """Fit on reference embeddings (n, d) and, optionally, reference logits (n, k).

        ``groups``: optional cluster id per reference sample (e.g. slide). With groups, the batch
        threshold is bootstrapped over whole clusters, since tiles of one slide are correlated.
        """
        Z = np.asarray(embeddings, dtype=np.float64)
        n, d = Z.shape
        self._ref_groups = None if groups is None else np.asarray(groups)
        if self._ref_groups is not None and len(self._ref_groups) != n:
            raise ValueError("groups must have one entry per reference sample")
        self._mu, self._precision = _fit_gaussian(Z)

        self._folds = None
        if n >= _MIN_CROSSFIT:
            # Out-of-sample reference distances from two half-data models. Deployment
            # samples are scored with the same half-data models (see score), so reference
            # and deployment distances come from the same estimator and the threshold is calibrated.
            rng = np.random.default_rng(self.seed)
            if self._ref_groups is None:
                perm = rng.permutation(n)
                a, b = perm[: n // 2], perm[n // 2 :]
            else:
                # Split by cluster, so each reference distance is measured on clusters (slides)
                # the half-model has never seen — as deployment slides will be.
                uniq = rng.permutation(np.unique(self._ref_groups))
                first = np.isin(self._ref_groups, uniq[: len(uniq) // 2])
                a, b = np.flatnonzero(first), np.flatnonzero(~first)
            ref = np.empty(n)
            self._folds = []
            for fit_idx, eval_idx in ((a, b), (b, a)):
                mu, prec = _fit_gaussian(Z[fit_idx])
                self._folds.append((mu, prec))
                ref[eval_idx] = _mahalanobis(Z[eval_idx], mu, prec)
            self._ref_maha = ref
        else:
            logger.warning(
                "Tier2Monitor: n=%d < %d, calibrating on in-sample distances (optimistic)",
                n, _MIN_CROSSFIT,
            )
            self._ref_maha = _mahalanobis(Z, self._mu, self._precision)

        self._ref_energy = energy(logits) if logits is not None else None
        logger.info("Tier2Monitor fitted: %d samples, dim=%d", n, d)
        return self

    def _bootstrap_upper(self, values: np.ndarray, m: int, alpha: float) -> float:
        """Upper (1 − α) threshold for a batch mean of size m.

        Bootstraps the difference between a size-m batch mean and a size-n reference
        mean, so the threshold accounts for the reference itself being a finite sample.
        """
        rng = np.random.default_rng(self.seed)
        n = len(values)
        batch = values[rng.integers(0, n, size=(self.n_bootstrap, m))].mean(axis=1)
        ref = values[rng.integers(0, n, size=(self.n_bootstrap, n))].mean(axis=1)
        return float(values.mean() + np.quantile(batch - ref, 1 - alpha))

    def _bootstrap_upper_clustered(self, values: np.ndarray, k: int, alpha: float) -> float:
        """Upper (1 − α) threshold for the mean over a batch of ``k`` clusters (slides).

        As ``_bootstrap_upper``, but whole reference clusters are drawn, so correlation between
        tiles of one cluster is reflected in the threshold.
        """
        rng = np.random.default_rng(self.seed)
        uniq, inv = np.unique(self._ref_groups, return_inverse=True)
        sums = np.bincount(inv, weights=values)
        sizes = np.bincount(inv).astype(float)
        c = len(uniq)
        bd = rng.integers(0, c, size=(self.n_bootstrap, k))
        br = rng.integers(0, c, size=(self.n_bootstrap, c))
        batch = sums[bd].sum(1) / sizes[bd].sum(1)
        ref = sums[br].sum(1) / sizes[br].sum(1)
        return float(values.mean() + np.quantile(batch - ref, 1 - alpha))

    def score(self, embeddings: np.ndarray, logits: np.ndarray | None = None,
              groups: np.ndarray | None = None) -> Tier2Result:
        """Score a deployment batch of embeddings (m, d) and optional logits (m, k).

        ``groups``: cluster id per deployment sample; required if, and only if, ``fit`` had groups.
        """
        if self._mu is None:
            raise RuntimeError("Call fit() before score()")
        if (groups is None) != (self._ref_groups is None):
            raise ValueError("pass groups to both fit() and score(), or to neither")

        Z = np.asarray(embeddings, dtype=np.float64)
        m = len(Z)
        if self._folds is not None:
            maha = np.empty(m)
            for k, (mu, prec) in enumerate(self._folds):
                maha[k::2] = _mahalanobis(Z[k::2], mu, prec)
        else:
            maha = _mahalanobis(Z, self._mu, self._precision)

        energy_mean = None
        if logits is not None:
            if self.energy_threshold is None and self._ref_energy is None:
                logger.warning("Tier2Monitor: logits given but no reference logits at fit(); energy not used")
            else:
                energy_mean = float(energy(logits).mean())

        # Split alpha across the statistics actually in use.
        n_stats = 2 if energy_mean is not None else 1
        a = self.alpha / n_stats
        if groups is None:
            upper = lambda v: self._bootstrap_upper(v, m, a)  # noqa: E731
        else:
            if len(groups) != m:
                raise ValueError("groups must have one entry per deployment sample")
            k = len(np.unique(groups))
            upper = lambda v: self._bootstrap_upper_clustered(v, k, a)  # noqa: E731
        t_maha = self.maha_threshold if self.maha_threshold is not None else upper(self._ref_maha)
        t_energy = None
        if energy_mean is not None:
            t_energy = self.energy_threshold if self.energy_threshold is not None else upper(self._ref_energy)

        flagged = bool(maha.mean() > t_maha or (energy_mean is not None and energy_mean > t_energy))
        return Tier2Result(
            mahalanobis_mean=float(maha.mean()),
            mahalanobis_max=float(maha.max()),
            energy_mean=energy_mean,
            flagged=flagged,
            threshold_maha=float(t_maha),
            threshold_energy=None if t_energy is None else float(t_energy),
        )

    @classmethod
    def from_encoder(
        cls,
        encoder,
        reference_loader,
        device: str = "cpu",
        **kwargs,
    ) -> "Tier2Monitor":
        """Convenience: extract embeddings with a PyTorch encoder and fit."""
        import torch

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
