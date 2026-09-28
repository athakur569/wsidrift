"""Tests for Tier 1 + Tier 2 monitors and validity score."""

import numpy as np
import pytest

from wsidrift.monitor.divergence import Tier1Monitor
from wsidrift.monitor.embedding import Tier2Monitor
from wsidrift.monitor.validity_score import ValidityMonitor


def _gaussian(n: int, dim: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((n, dim))


class TestTier1Monitor:
    def test_same_distribution_not_flagged(self):
        X = _gaussian(200, 32, seed=1)
        Y = _gaussian(200, 32, seed=2)  # same distribution, different sample
        monitor = Tier1Monitor(mmd_threshold=0.5, wass_threshold=1.0).fit(X)
        result = monitor.score(Y)
        assert not result.flagged

    def test_shifted_distribution_flagged(self):
        X = _gaussian(200, 32, seed=0)
        Y = _gaussian(200, 32, seed=0) + 5.0  # large mean shift
        monitor = Tier1Monitor(mmd_threshold=0.01, wass_threshold=0.01).fit(X)
        result = monitor.score(Y)
        assert result.flagged

    def test_mmd_nonnegative(self):
        X = _gaussian(100, 16)
        Y = _gaussian(100, 16)
        monitor = Tier1Monitor().fit(X)
        result = monitor.score(Y)
        assert result.mmd_squared >= -1e-9  # allow small float error

    def test_fit_required(self):
        monitor = Tier1Monitor()
        with pytest.raises(RuntimeError, match="fit"):
            monitor.score(_gaussian(50, 8))


class TestTier2Monitor:
    def test_in_distribution_low_mahalanobis(self):
        Z_ref = _gaussian(200, 64, seed=0)
        Z_dep = _gaussian(50, 64, seed=1)  # same distribution
        monitor = Tier2Monitor(maha_threshold=5.0).fit(Z_ref)
        result = monitor.score(Z_dep)
        assert result.mahalanobis_mean < result.threshold_maha

    def test_ood_high_mahalanobis(self):
        Z_ref = _gaussian(200, 16, seed=0)
        Z_ood = _gaussian(50, 16, seed=0) * 10 + 20  # far from reference
        monitor = Tier2Monitor(maha_threshold=2.0).fit(Z_ref)
        result = monitor.score(Z_ood)
        assert result.flagged


class TestValidityMonitor:
    def test_conjunction_requires_both(self):
        X_ref = _gaussian(200, 32, seed=0)
        X_dep = _gaussian(200, 32, seed=1)  # not shifted
        Z_ref = _gaussian(200, 16, seed=0)
        Z_dep = _gaussian(50, 16, seed=0) * 10 + 20  # OOD embeddings

        tier1 = Tier1Monitor(mmd_threshold=0.5, wass_threshold=1.0).fit(X_ref)
        tier2 = Tier2Monitor(maha_threshold=2.0).fit(Z_ref)
        monitor = ValidityMonitor(tier1, tier2, require_both=True)

        result = monitor.score(X_dep, embeddings=Z_dep)
        # T1 not flagged, T2 flagged → conjunction = False
        assert not result.overall_flag

    def test_both_flagged_triggers_invalid(self):
        X_ref = _gaussian(200, 32, seed=0)
        X_dep = _gaussian(200, 32, seed=0) + 10.0  # shifted input
        Z_ref = _gaussian(200, 16, seed=0)
        Z_dep = _gaussian(50, 16, seed=0) * 10 + 20  # shifted embeddings

        tier1 = Tier1Monitor(mmd_threshold=0.01, wass_threshold=0.01).fit(X_ref)
        tier2 = Tier2Monitor(maha_threshold=2.0).fit(Z_ref)
        monitor = ValidityMonitor(tier1, tier2, require_both=True)

        result = monitor.score(X_dep, embeddings=Z_dep)
        assert result.overall_flag
        assert result.validity_score == 1.0
