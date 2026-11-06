"""Tier 1 — input divergence monitor (MMD + Wasserstein).

Label-free. Detects covariate shift in stain/feature distributions between
a validated reference batch and a deployment batch.

Valid under the covariate-shift assumption: P(X) changes while P(Y|X) is stable.
This holds for stain and scanner variation in H&E histopathology.

By default the flag is a permutation two-sample test at false-alarm rate ``alpha``
(Bonferroni-split across the MMD and Wasserstein statistics). Passing explicit
thresholds switches to fixed-threshold mode.

Clustered samples: tiles from one slide (or patient) resemble each other, so permuting single
tiles treats them as independent and false-alarms above ``alpha``. Pass ``groups`` (one id per
sample) to ``fit`` and ``score`` and whole groups are permuted instead.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..guards import finite

logger = logging.getLogger(__name__)


def _wasserstein_mean(X: np.ndarray, Y: np.ndarray) -> float:
    """Exact mean Wasserstein-1 across feature dimensions, vectorized over features.

    W₁ = ∫₀¹ |Q_X(u) − Q_Y(u)| du, evaluated on the merged grid of empirical
    quantile breakpoints {i/n} ∪ {j/m}, which is exact for any n and m.
    """
    n, m = len(X), len(Y)
    Xs, Ys = np.sort(X, axis=0), np.sort(Y, axis=0)
    if n == m:
        return float(np.abs(Xs - Ys).mean())
    u = np.union1d(np.arange(1, n + 1) / n, np.arange(1, m + 1) / m)
    widths = np.diff(np.concatenate(([0.0], u)))
    ix = np.minimum(np.ceil(u * n - 1e-9).astype(int) - 1, n - 1)
    iy = np.minimum(np.ceil(u * m - 1e-9).astype(int) - 1, m - 1)
    per_feature = (widths[:, None] * np.abs(Xs[ix] - Ys[iy])).sum(axis=0)
    return float(per_feature.mean())


def _rbf_kernel(X: np.ndarray, Y: np.ndarray, sigma: float) -> np.ndarray:
    sq_dist = (
        (X ** 2).sum(1, keepdims=True)
        - 2 * X @ Y.T
        + (Y ** 2).sum(1, keepdims=True).T
    )
    return np.exp(-np.maximum(sq_dist, 0) / (2 * sigma ** 2))


def _median_bandwidth(X: np.ndarray, subsample: int = 500, seed: int | None = None) -> float:
    """Estimate RBF bandwidth as the median pairwise distance (Garreau et al. 2017).

    Uses a random subsample for large n to keep O(n²) cost bounded.
    """
    if len(X) > subsample:
        rng_seed = seed if seed is not None else (len(X) * X.shape[1]) % (2**31)
        idx = np.random.default_rng(rng_seed).choice(len(X), subsample, replace=False)
        X = X[idx]
    sq = (X**2).sum(1, keepdims=True)
    sq_dist = np.maximum(sq + sq.T - 2 * X @ X.T, 0)
    dists = np.sqrt(sq_dist[np.triu_indices_from(sq_dist, k=1)])
    med = np.median(dists[dists > 0]) if (dists > 0).any() else 1.0
    return float(med)


def _mmd2_unbiased(K: np.ndarray, idx_x: np.ndarray, idx_y: np.ndarray) -> float:
    """Unbiased MMD² from a pooled kernel matrix. Can be slightly negative under no shift."""
    n, m = len(idx_x), len(idx_y)
    Kxx = K[np.ix_(idx_x, idx_x)]
    Kyy = K[np.ix_(idx_y, idx_y)]
    Kxy = K[np.ix_(idx_x, idx_y)]
    return float(
        (Kxx.sum() - np.trace(Kxx)) / (n * (n - 1))
        + (Kyy.sum() - np.trace(Kyy)) / (m * (m - 1))
        - 2 * Kxy.mean()
    )


@dataclass
class Tier1Result:
    mmd_squared: float
    wasserstein_mean: float
    flagged: bool
    threshold_mmd: float | None
    threshold_wass: float | None
    p_mmd: float | None = None
    p_wass: float | None = None
    alpha: float | None = None

    @property
    def mmd(self) -> float:
        return self.mmd_squared

    @property
    def wasserstein(self) -> float:
        return self.wasserstein_mean

    @property
    def mmd_flagged(self) -> bool:
        if self.p_mmd is not None:
            return self.p_mmd < self.alpha / 2
        return self.mmd_squared > self.threshold_mmd

    @property
    def wasserstein_flagged(self) -> bool:
        if self.p_wass is not None:
            return self.p_wass < self.alpha / 2
        return self.wasserstein_mean > self.threshold_wass


class Tier1Monitor:
    """Input-space divergence monitor.

    Fit on reference (validated training) features. At deployment, score a new
    batch and flag it when it differs from the reference.

    Args:
        alpha:          False-alarm rate for the permutation test (default mode).
        n_permutations: Permutations per test; p-values resolve to 1/(n_permutations+1).
        mmd_threshold:  If set together with ``wass_threshold``, use fixed thresholds
                        instead of the permutation test.
        wass_threshold: Fixed mean-Wasserstein-1 threshold.
        seed:           Seed for permutations and bandwidth subsampling.
    """

    def __init__(
        self,
        mmd_threshold: float | None = None,
        wass_threshold: float | None = None,
        alpha: float = 0.05,
        n_permutations: int = 200,
        seed: int = 0,
    ) -> None:
        if (mmd_threshold is None) != (wass_threshold is None):
            raise ValueError("Set both mmd_threshold and wass_threshold, or neither.")
        self.mmd_threshold = mmd_threshold
        self.wass_threshold = wass_threshold
        self.alpha = alpha
        self.n_permutations = n_permutations
        self.seed = seed
        self._reference: np.ndarray | None = None
        self._ref_groups: np.ndarray | None = None
        self._sigma: float | None = None

    @property
    def fixed_thresholds(self) -> bool:
        return self.mmd_threshold is not None

    def fit(self, X_reference: np.ndarray, groups: np.ndarray | None = None) -> "Tier1Monitor":
        """Store reference distribution and estimate bandwidth.

        ``groups``: optional cluster id per reference sample (e.g. slide); see the module docstring.
        """
        self._reference = finite(X_reference, "reference features")
        self._ref_groups = None if groups is None else np.asarray(groups)
        if self._ref_groups is not None and len(self._ref_groups) != len(self._reference):
            raise ValueError("groups must have one entry per reference sample")
        self._sigma = _median_bandwidth(self._reference, seed=self.seed)
        logger.info(
            "Tier1Monitor fitted on %d samples, bandwidth=%.4f",
            len(self._reference), self._sigma,
        )
        return self

    def score(self, X_deployment: np.ndarray, groups: np.ndarray | None = None) -> Tier1Result:
        """Compute divergence of a deployment batch against the reference.

        ``groups``: cluster id per deployment sample; required if, and only if, the reference
        was fitted with groups. Then whole clusters are permuted.
        """
        if self._reference is None:
            raise RuntimeError("Call fit() before score()")
        if (groups is None) != (self._ref_groups is None):
            raise ValueError("pass groups to both fit() and score(), or to neither")

        X = self._reference
        Y = finite(X_deployment, "deployment features")
        n, m = len(X), len(Y)
        if n < 2 or m < 2:
            raise ValueError(f"need at least 2 samples in each batch for MMD² (reference {n}, deployment {m})")
        Z = np.vstack([X, Y])
        K = _rbf_kernel(Z, Z, self._sigma)
        idx_x, idx_y = np.arange(n), np.arange(n, n + m)

        mmd2 = _mmd2_unbiased(K, idx_x, idx_y)
        wass = _wasserstein_mean(X, Y)
        if not (np.isfinite(mmd2) and np.isfinite(wass)):
            raise ValueError("the statistic is NaN or infinite, so no p-value exists")

        if self.fixed_thresholds:
            flagged = mmd2 > self.mmd_threshold or wass > self.wass_threshold
            return Tier1Result(mmd2, wass, flagged, self.mmd_threshold, self.wass_threshold)

        # The smallest p-value a permutation test can give is 1/(n_permutations+1). If that is not
        # below alpha/2, the monitor can never flag, whatever the data: refuse rather than stay silent.
        if 1 / (self.n_permutations + 1) >= self.alpha / 2:
            raise ValueError(
                f"with {self.n_permutations} permutations the smallest p-value is "
                f"{1 / (self.n_permutations + 1):.4f}, not below alpha/2 = {self.alpha / 2:.4f}, so Tier 1 "
                f"could never flag. Use at least {math.ceil(2 / self.alpha)} permutations."
            )
        if groups is None:
            # Tiny batches allow few distinct splits too (see the cluster check below).
            distinct = math.comb(n + m, n) // (2 if n == m else 1)
            if 1 / distinct >= self.alpha / 2:
                raise ValueError(
                    f"{n} reference and {m} deployment samples allow only {distinct} distinct splits, so Tier 1 "
                    f"could never flag at alpha/2 = {self.alpha / 2:.4f}. Use more samples."
                )
        rng = np.random.default_rng(self.seed)
        if groups is not None:
            if len(groups) != m:
                raise ValueError("groups must have one entry per deployment sample")
            # Cluster labels made distinct across the two batches, then whole clusters permuted.
            cl = np.concatenate([[f"r|{g}" for g in self._ref_groups], [f"d|{g}" for g in groups]])
            uniq, inv = np.unique(cl, return_inverse=True)
            members = [np.flatnonzero(inv == k) for k in range(len(uniq))]
            n_ref_clusters = len(np.unique(self._ref_groups))
            if n_ref_clusters < 2 or len(uniq) - n_ref_clusters < 2:
                raise ValueError("need at least 2 clusters in each batch for a cluster permutation test")
            # With few clusters there are few distinct splits, and the observed split is one of them
            # (two, when both batches have the same number of clusters, since the statistics are
            # symmetric). The p-value cannot fall much below that share, so the test may be unable to flag.
            n_dep_clusters = len(uniq) - n_ref_clusters
            distinct = math.comb(len(uniq), n_ref_clusters) // (2 if n_ref_clusters == n_dep_clusters else 1)
            if 1 / distinct >= self.alpha / 2:
                raise ValueError(
                    f"{n_ref_clusters} reference and {n_dep_clusters} deployment clusters allow only {distinct} "
                    f"distinct splits, so the p-value cannot fall below about {1 / distinct:.3f} and Tier 1 "
                    f"could never flag at alpha/2 = {self.alpha / 2:.4f}. Use more clusters."
                )
        exceed_mmd = exceed_wass = 0
        for _ in range(self.n_permutations):
            if groups is None:
                perm = rng.permutation(n + m)
                px, py = perm[:n], perm[n:]
            else:
                order = rng.permutation(len(members))
                px = np.concatenate([members[k] for k in order[:n_ref_clusters]])
                py = np.concatenate([members[k] for k in order[n_ref_clusters:]])
            exceed_mmd += _mmd2_unbiased(K, px, py) >= mmd2
            exceed_wass += _wasserstein_mean(Z[px], Z[py]) >= wass
        p_mmd = (1 + exceed_mmd) / (1 + self.n_permutations)
        p_wass = (1 + exceed_wass) / (1 + self.n_permutations)
        flagged = p_mmd < self.alpha / 2 or p_wass < self.alpha / 2
        return Tier1Result(mmd2, wass, flagged, None, None, p_mmd=p_mmd, p_wass=p_wass, alpha=self.alpha)

    def save(self, path: str | Path) -> None:
        """Save settings and the fitted reference to a .npz file; ``load`` restores a monitor that scores identically."""
        if self._reference is None:
            raise RuntimeError("Call fit() before save()")
        settings = {
            "mmd_threshold": self.mmd_threshold,
            "wass_threshold": self.wass_threshold,
            "alpha": self.alpha,
            "n_permutations": self.n_permutations,
            "seed": self.seed,
            "sigma": self._sigma,
        }
        arrays = {"reference": self._reference}
        if self._ref_groups is not None:
            # As text: an object array (e.g. a pandas column's .to_numpy()) cannot be loaded without pickle.
            arrays["groups"] = np.asarray(self._ref_groups).astype(str)
        with open(path, "wb") as f:
            np.savez(f, settings=json.dumps(settings), **arrays)

    @classmethod
    def load(cls, path: str | Path) -> "Tier1Monitor":
        """Restore a monitor written by ``save``."""
        with np.load(path, allow_pickle=False) as z:
            settings = json.loads(str(z["settings"]))
            sigma = settings.pop("sigma")
            monitor = cls(**settings)
            monitor._reference = z["reference"]
            monitor._ref_groups = z["groups"] if "groups" in z.files else None
        monitor._sigma = sigma
        return monitor
