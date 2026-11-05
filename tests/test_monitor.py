"""Tests for Tier 1 + Tier 2 monitors and validity score."""

import numpy as np
import pytest
from scipy.stats import wasserstein_distance

from wsidrift.monitor.divergence import Tier1Monitor, _wasserstein_mean
from wsidrift.monitor.embedding import Tier2Monitor, energy
from wsidrift.monitor.validity_score import ValidityMonitor


def _gaussian(n: int, dim: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((n, dim))


class TestTier1Monitor:
    def test_same_distribution_not_flagged(self):
        X = _gaussian(200, 32, seed=1)
        Y = _gaussian(200, 32, seed=2)
        monitor = Tier1Monitor(mmd_threshold=0.5, wass_threshold=1.0).fit(X)
        assert not monitor.score(Y).flagged

    def test_shifted_distribution_flagged(self):
        X = _gaussian(200, 32, seed=0)
        Y = _gaussian(200, 32, seed=0) + 5.0
        monitor = Tier1Monitor(mmd_threshold=0.01, wass_threshold=0.01).fit(X)
        assert monitor.score(Y).flagged

    def test_unbiased_mmd_near_zero_without_shift(self):
        # The unbiased MMD² estimator can be slightly negative when there is no shift.
        X = _gaussian(100, 16, seed=0)
        Y = _gaussian(100, 16, seed=1)
        result = Tier1Monitor().fit(X).score(Y)
        assert abs(result.mmd_squared) < 0.02

    def test_fit_required(self):
        with pytest.raises(RuntimeError, match="fit"):
            Tier1Monitor().score(_gaussian(50, 8))

    def test_thresholds_must_be_paired(self):
        with pytest.raises(ValueError):
            Tier1Monitor(mmd_threshold=0.1)

    def test_calibrated_flags_shift_with_pvalues(self):
        X = _gaussian(120, 16, seed=0)
        result = Tier1Monitor(n_permutations=100).fit(X).score(_gaussian(80, 16, seed=1) + 1.0)
        assert result.flagged
        assert result.p_mmd < 0.05 and result.threshold_mmd is None

    def test_calibrated_false_alarm_rate_near_alpha(self):
        X = _gaussian(80, 8, seed=0)
        monitor = Tier1Monitor(alpha=0.1, n_permutations=99).fit(X)
        flags = [monitor.score(_gaussian(60, 8, seed=s)).flagged for s in range(1, 41)]
        assert np.mean(flags) <= 0.25

    @pytest.mark.parametrize("n,m", [(50, 50), (50, 37), (13, 40)])
    def test_wasserstein_matches_scipy(self, n, m):
        X, Y = _gaussian(n, 5, seed=3), _gaussian(m, 5, seed=4) * 2 + 1
        expected = np.mean([wasserstein_distance(X[:, i], Y[:, i]) for i in range(5)])
        assert _wasserstein_mean(X, Y) == pytest.approx(expected, rel=1e-9)


class TestTier2Monitor:
    def test_calibrated_in_distribution_not_flagged(self):
        monitor = Tier2Monitor().fit(_gaussian(200, 64, seed=0))
        result = monitor.score(_gaussian(50, 64, seed=1))
        assert not result.flagged
        assert result.mahalanobis_mean < result.threshold_maha

    def test_calibrated_false_alarm_rate_near_alpha(self):
        monitor = Tier2Monitor(alpha=0.1).fit(_gaussian(400, 16, seed=0))
        flags = [monitor.score(_gaussian(30, 16, seed=s)).flagged for s in range(1, 61)]
        assert np.mean(flags) <= 0.25

    def test_ood_high_mahalanobis(self):
        monitor = Tier2Monitor(maha_threshold=2.0).fit(_gaussian(200, 16, seed=0))
        assert monitor.score(_gaussian(50, 16, seed=0) * 10 + 20).flagged

    def test_calibrated_flags_shift(self):
        monitor = Tier2Monitor().fit(_gaussian(200, 16, seed=0))
        assert monitor.score(_gaussian(50, 16, seed=1) + 1.5).flagged

    def test_energy_higher_for_uncertain_logits(self):
        confident = np.array([[10.0, -10.0], [-10.0, 10.0]])
        uncertain = np.array([[0.1, 0.0], [0.0, 0.1]])
        assert energy(uncertain).mean() > energy(confident).mean()

    def test_energy_flags_ood_logits(self):
        rng = np.random.default_rng(0)
        Z = _gaussian(200, 8, seed=0)
        ref_logits = np.column_stack([rng.normal(6, 1, 200), rng.normal(-6, 1, 200)])
        monitor = Tier2Monitor().fit(Z, ref_logits)
        dep_logits = rng.normal(0, 0.3, (50, 2))  # near-uniform → high energy
        result = monitor.score(_gaussian(50, 8, seed=1), dep_logits)
        assert result.energy_flagged and result.flagged

    def test_logits_without_reference_are_ignored(self):
        monitor = Tier2Monitor().fit(_gaussian(100, 8, seed=0))
        result = monitor.score(_gaussian(30, 8, seed=1), np.zeros((30, 2)))
        assert result.energy_mean is None


class TestValidityMonitor:
    def test_conjunction_requires_both(self):
        tier1 = Tier1Monitor(mmd_threshold=0.5, wass_threshold=1.0).fit(_gaussian(200, 32, seed=0))
        tier2 = Tier2Monitor(maha_threshold=2.0).fit(_gaussian(200, 16, seed=0))
        result = ValidityMonitor(tier1, tier2, require_both=True).score(
            _gaussian(200, 32, seed=1), embeddings=_gaussian(50, 16, seed=0) * 10 + 20
        )
        assert not result.overall_flag

    def test_both_flagged_triggers_invalid(self):
        tier1 = Tier1Monitor(mmd_threshold=0.01, wass_threshold=0.01).fit(_gaussian(200, 32, seed=0))
        tier2 = Tier2Monitor(maha_threshold=2.0).fit(_gaussian(200, 16, seed=0))
        result = ValidityMonitor(tier1, tier2, require_both=True).score(
            _gaussian(200, 32, seed=0) + 10.0, embeddings=_gaussian(50, 16, seed=0) * 10 + 20
        )
        assert result.overall_flag
        assert result.validity_score == 1.0


class TestValidityMonitorInputs:
    def test_tier2_without_embeddings_refused(self):
        tier1 = Tier1Monitor(n_permutations=49).fit(_gaussian(100, 8, seed=0))
        tier2 = Tier2Monitor().fit(_gaussian(100, 8, seed=0))
        with pytest.raises(ValueError, match="needs embeddings"):
            ValidityMonitor(tier1, tier2).score(_gaussian(40, 8, seed=1))

    def test_groups_go_to_both_tiers(self):
        Xr, gr = _clustered(12, 10, 3, seed=0)
        Xd, gd = _clustered(6, 10, 3, seed=1)
        vm = ValidityMonitor(Tier1Monitor(n_permutations=49).fit(Xr, groups=gr), Tier2Monitor().fit(Xr, groups=gr))
        assert isinstance(vm.score(Xd, embeddings=Xd, groups=gd).overall_flag, bool)
        with pytest.raises(ValueError, match="both"):
            vm.score(Xd, embeddings=Xd)

    def test_groups_need_matching_rows(self):
        Xr, gr = _clustered(12, 10, 3, seed=0)
        Xd, gd = _clustered(6, 10, 3, seed=1)
        vm = ValidityMonitor(Tier1Monitor(n_permutations=49).fit(Xr, groups=gr), Tier2Monitor().fit(Xr, groups=gr))
        with pytest.raises(ValueError, match="same rows"):
            vm.score(Xd, embeddings=Xd[:30], groups=gd)


def _clustered(n_clusters, per, dim, seed, effect=1.0, noise=0.3):
    """Null data with strong cluster effects: every cluster (slide) has its own offset."""
    r = np.random.default_rng(seed)
    X = np.concatenate([r.normal(0, effect, dim) + r.normal(0, noise, (per, dim)) for _ in range(n_clusters)])
    return X, np.repeat(np.arange(n_clusters), per)


class TestClusteredCalibration:
    def _rates(self, monitor_cls, **kw):
        tile = cluster = 0
        reps = 25
        for s in range(reps):
            Xr, gr = _clustered(12, 15, 3, seed=2 * s)
            Xd, gd = _clustered(12, 15, 3, seed=2 * s + 1)
            tile += monitor_cls(**kw).fit(Xr).score(Xd).flagged
            cluster += monitor_cls(**kw).fit(Xr, groups=gr).score(Xd, groups=gd).flagged
        return tile / reps, cluster / reps

    def test_tier1_cluster_permutation_controls_false_alarms(self):
        tile, cluster = self._rates(Tier1Monitor, alpha=0.1, n_permutations=49)
        assert tile > 0.5            # tile-level permutation treats correlated tiles as independent
        assert cluster <= 0.25       # whole-cluster permutation stays near alpha

    def test_tier2_cluster_bootstrap_controls_false_alarms(self):
        tile, cluster = self._rates(Tier2Monitor, alpha=0.1)
        assert tile > cluster
        assert cluster <= 0.25

    def test_cluster_mode_still_detects_real_shift(self):
        Xr, gr = _clustered(12, 15, 3, seed=0)
        Xd, gd = _clustered(12, 15, 3, seed=1)
        assert Tier1Monitor(alpha=0.1, n_permutations=49).fit(Xr, groups=gr).score(Xd + 3.0, groups=gd).flagged

    def test_groups_must_be_given_to_both_or_neither(self):
        Xr, gr = _clustered(4, 5, 2, seed=0)
        with pytest.raises(ValueError, match="both"):
            Tier1Monitor().fit(Xr, groups=gr).score(Xr)
        with pytest.raises(ValueError, match="both"):
            Tier2Monitor().fit(Xr).score(Xr, groups=gr)
        with pytest.raises(ValueError, match="at least 2 clusters"):
            Tier1Monitor().fit(Xr, groups=np.zeros(len(Xr))).score(Xr, groups=gr)
