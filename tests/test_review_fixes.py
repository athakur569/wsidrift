"""Fixes from the external review of 5 November 2026: settings and inputs that gave a silent or wrong result."""

import logging

import numpy as np
import pytest

from wsidrift.eval.delta_gap import DeltaGap
from wsidrift.eval.stats import bonferroni_correct
from wsidrift.guards import probabilities
from wsidrift.monitor import embedding
from wsidrift.monitor.divergence import Tier1Monitor
from wsidrift.monitor.embedding import Tier2Monitor
from wsidrift.norm.macenko import MacenkoNormalizer


def _g(n, d, seed=0, shift=0.0):
    return np.random.default_rng(seed).standard_normal((n, d)) + shift


class TestTier1CannotFlag:
    def test_too_few_permutations_for_alpha(self):
        # 200 permutations: smallest p = 1/201 > 0.005/2, so not even a 50-sigma shift could flag.
        with pytest.raises(ValueError, match="could never flag"):
            Tier1Monitor(alpha=0.005).fit(_g(50, 3)).score(_g(50, 3, 1, 50.0))

    def test_enough_permutations_flags(self):
        assert Tier1Monitor(alpha=0.005, n_permutations=999).fit(_g(50, 3)).score(_g(50, 3, 1, 50.0)).flagged

    def test_two_clusters_per_side_refused(self):
        X, Y = _g(40, 3), _g(40, 3, 1, 100.0)
        g = np.repeat([0, 1], 20)
        with pytest.raises(ValueError, match="distinct splits"):
            Tier1Monitor().fit(X, groups=g).score(Y, groups=g)

    def test_batch_of_one_message(self):
        with pytest.raises(ValueError, match="at least 2 samples"):
            Tier1Monitor().fit(_g(50, 3)).score(_g(1, 3))


class TestTier1SaveLoad:
    @pytest.mark.parametrize("grouped", [False, True])
    def test_round_trip_scores_identically(self, tmp_path, grouped):
        X, Y = _g(60, 4), _g(40, 4, 1, 0.3)
        gr = np.repeat(np.arange(10), 6) if grouped else None
        gd = np.repeat(np.arange(8), 5) if grouped else None
        m = Tier1Monitor(n_permutations=99, seed=3).fit(X, groups=gr)
        m.save(tmp_path / "t1.npz")
        r1 = m.score(Y, groups=gd)
        r2 = Tier1Monitor.load(tmp_path / "t1.npz").score(Y, groups=gd)
        assert (r1.p_mmd, r1.p_wass, r1.mmd, r1.flagged) == (r2.p_mmd, r2.p_wass, r2.mmd, r2.flagged)

    def test_save_before_fit(self, tmp_path):
        with pytest.raises(RuntimeError):
            Tier1Monitor().save(tmp_path / "x.npz")


class TestTier2Inputs:
    def test_one_reference_cluster_refused(self):
        with pytest.raises(ValueError, match="at least 2 reference clusters"):
            Tier2Monitor().fit(_g(60, 4), groups=np.zeros(60))

    @pytest.mark.parametrize("where", ["fit", "score"])
    def test_logits_row_count_must_match(self, where):
        Z = _g(60, 4)
        if where == "fit":
            with pytest.raises(ValueError, match="one row per embedding"):
                Tier2Monitor().fit(Z, _g(59, 2))
        else:
            m = Tier2Monitor().fit(Z, _g(60, 2))
            with pytest.raises(ValueError, match="one row per embedding"):
                m.score(_g(30, 4), _g(29, 2))

    def test_diagonal_fallback_is_logged(self, caplog):
        with caplog.at_level(logging.WARNING, logger="wsidrift.monitor.embedding"):
            Tier2Monitor().fit(_g(60, 40))
        assert "per-dimension variances" in caplog.text

    def test_chunked_bootstrap_identical(self, monkeypatch):
        Z, D = _g(300, 5), _g(50, 5, 1)
        full = Tier2Monitor(n_bootstrap=301).fit(Z).score(D).threshold_maha
        monkeypatch.setattr(embedding, "_CHUNK", 7)  # odd, tiny chunks
        assert Tier2Monitor(n_bootstrap=301).fit(Z).score(D).threshold_maha == full


def test_probabilities_names_the_right_row():
    p = np.full((6, 2), 0.5)
    p[3, 1] = 1.7
    with pytest.raises(ValueError, match=r"1 of 6 \(row 3\)"):
        probabilities(p, "p")


class TestMacenkoInput:
    def test_blank_tile(self):
        with pytest.raises(ValueError, match="tissue pixels"):
            MacenkoNormalizer().fit(np.full((16, 16, 3), 255, np.uint8))

    def test_unit_float_image(self):
        with pytest.raises(ValueError, match="0–1 floats"):
            MacenkoNormalizer().fit(np.random.default_rng(0).random((16, 16, 3)))


def test_delta_gap_sample_sd():
    dg = DeltaGap("c", "x")
    y = np.array([0, 0, 1, 1])
    for s in ([0.1, 0.2, 0.8, 0.9], [0.1, 0.6, 0.4, 0.9], [0.3, 0.2, 0.8, 0.7]):
        dg.add_seed(y, np.array([0.1, 0.2, 0.8, 0.9]), y, np.array(s))
    r = dg.compute()
    cross = [1.0, 0.75, 1.0]
    assert r.auc_cross_std == pytest.approx(np.std(cross, ddof=1))


def test_bonferroni_returns_decisions():
    assert bonferroni_correct([0.01, 0.03], alpha=0.05) == [True, False]


class TestThirdReview:
    """Second external review pass, 6 November 2026."""

    @staticmethod
    def _slides(k, per=25, d=3, seed=0):
        r = np.random.default_rng(seed)
        X = np.concatenate([r.normal(0, 1, d) + r.normal(0, 0.3, (per, d)) for _ in range(k)])
        return X, np.repeat(np.arange(k), per)

    def test_tier2_refuses_few_reference_clusters(self):
        X, g = self._slides(11)
        with pytest.raises(ValueError, match="at least 12"):
            Tier2Monitor().fit(X, groups=g)

    def test_tier2_twelve_clusters_enough(self):
        X, g = self._slides(12)
        Tier2Monitor().fit(X, groups=g)

    def test_tier2_few_clusters_opt_in_and_fixed(self):
        X, g = self._slides(4)
        Tier2Monitor(allow_small_reference=True).fit(X, groups=g)
        Tier2Monitor(maha_threshold=3.0).fit(X, groups=g)

    def test_save_load_object_groups(self, tmp_path):
        X, Y = _g(60, 4), _g(40, 4, 1)
        gr = np.array([f"slide-{i // 6}" for i in range(60)], dtype=object)  # as pandas .to_numpy() gives
        gd = np.array([f"dep-{i // 5}" for i in range(40)], dtype=object)
        m = Tier1Monitor(n_permutations=99).fit(X, groups=gr)
        m.save(tmp_path / "t1.npz")
        r1, r2 = m.score(Y, groups=gd), Tier1Monitor.load(tmp_path / "t1.npz").score(Y, groups=gd)
        assert (r1.p_mmd, r1.p_wass) == (r2.p_mmd, r2.p_wass)

    @pytest.mark.parametrize("k", [2, 3, 4])
    def test_tile_level_tiny_batches_refused(self, k):
        with pytest.raises(ValueError, match="distinct splits"):
            Tier1Monitor().fit(_g(k, 3)).score(_g(k, 3, 1, 50.0))

    def test_tile_level_five_per_side_runs(self):
        Tier1Monitor(n_permutations=199).fit(_g(5, 3)).score(_g(5, 3, 1, 50.0))
