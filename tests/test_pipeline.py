"""Integration tests for MonitorPipeline and evaluation modules."""

import numpy as np
import pytest

from wsidrift.monitor.pipeline import MonitorPipeline
from wsidrift.eval.delta_gap import DeltaGap
from wsidrift.eval.stats import paired_ttest, wilcoxon_test, bonferroni_correct


def _make_embeddings(n: int, d: int = 64, seed: int = 0, shift: float = 0.0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, d)) + shift


def _make_logits(n: int, n_classes: int = 2, seed: int = 0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, n_classes))


class TestMonitorPipeline:
    def test_no_shift_not_invalid(self):
        pipe = MonitorPipeline(n_permutations=99).fit(
            _make_embeddings(200, d=16, seed=0), reference_logits=_make_logits(200, seed=0)
        )
        result = pipe.score(_make_embeddings(100, d=16, seed=1), logits=_make_logits(100, seed=1))
        assert isinstance(result.invalid, bool)
        assert not result.invalid

    def test_large_shift_flags_invalid(self):
        pipe = MonitorPipeline(n_permutations=99).fit(_make_embeddings(200, d=16, seed=0))
        result = pipe.score(_make_embeddings(100, d=16, seed=2, shift=5.0))
        assert result.tier1.flagged and result.tier2.flagged and result.invalid

    def test_separate_features_and_embeddings(self):
        feats_ref, emb_ref = _make_embeddings(150, d=8, seed=0), _make_embeddings(150, d=32, seed=1)
        pipe = MonitorPipeline(n_permutations=99).fit(feats_ref, reference_embeddings=emb_ref)
        # Input shift only: Tier 1 flags, Tier 2 doesn't, so the conjunction stays valid.
        result = pipe.score(
            _make_embeddings(80, d=8, seed=2, shift=3.0), embeddings=_make_embeddings(80, d=32, seed=3)
        )
        assert result.tier1.flagged and not result.tier2.flagged and not result.invalid

    def test_fixed_thresholds_still_supported(self):
        pipe = MonitorPipeline(t1_mmd_threshold=0.1, t1_wass_threshold=0.5, t2_maha_threshold=1.0)
        pipe.fit(_make_embeddings(200, d=16, seed=0))
        result = pipe.score(_make_embeddings(100, d=16, seed=2, shift=5.0))
        assert result.tier1.threshold_mmd == 0.1 and result.invalid

    def test_summary_keys(self):
        pipe = MonitorPipeline(n_permutations=49).fit(_make_embeddings(100, d=16, seed=0))
        s = pipe.score(_make_embeddings(50, d=16, seed=3)).summary()
        assert {"t1_mmd", "t1_p_mmd", "t2_mahalanobis_threshold", "invalid"} <= s.keys()


def _slides(n_slides: int, per: int = 10, d: int = 3, seed: int = 0):
    """Null data with strong slide effects: tiles of one slide share an offset."""
    rng = np.random.default_rng(seed)
    X = np.concatenate([rng.normal(0, 1, d) + rng.normal(0, 0.3, (per, d)) for _ in range(n_slides)])
    return X, np.repeat(np.arange(n_slides), per)


class TestPipelineGroups:
    def test_groups_reach_both_tiers(self):
        Xr, gr = _slides(20, seed=0)
        Xd, gd = _slides(8, seed=1)
        pipe = MonitorPipeline(alpha=0.1, n_permutations=49).fit(Xr, groups=gr)
        assert pipe._t1._ref_groups is not None and pipe._t2._ref_groups is not None
        assert isinstance(pipe.score(Xd, groups=gd).invalid, bool)

    def test_groups_cut_false_alarms_on_correlated_tiles(self):
        tile = cluster = 0
        for s in range(15):
            Xr, gr = _slides(20, seed=2 * s)
            Xd, gd = _slides(8, seed=2 * s + 1)
            tile += MonitorPipeline(alpha=0.1, n_permutations=49, require_both=False).fit(Xr).score(Xd).invalid
            cluster += MonitorPipeline(alpha=0.1, n_permutations=49, require_both=False).fit(
                Xr, groups=gr).score(Xd, groups=gd).invalid
        assert tile > cluster

    def test_groups_need_one_row_per_sample_in_both_inputs(self):
        X, g = _slides(10, seed=0)
        with pytest.raises(ValueError, match="same rows"):
            MonitorPipeline().fit(X, reference_embeddings=X[:50], groups=g)


class TestPipelineInputsMatchFit:
    def _feats_emb(self, n, seed):
        return _make_embeddings(n, d=8, seed=seed), _make_embeddings(n, d=8, seed=seed + 100)

    def test_embeddings_at_fit_require_embeddings_at_score(self):
        f, e = self._feats_emb(120, 0)
        pipe = MonitorPipeline(n_permutations=49).fit(f, reference_embeddings=e)
        with pytest.raises(ValueError, match="needs embeddings"):
            pipe.score(_make_embeddings(40, d=8, seed=1))

    def test_embeddings_at_score_need_embeddings_at_fit(self):
        pipe = MonitorPipeline(n_permutations=49).fit(_make_embeddings(120, d=8, seed=0))
        with pytest.raises(ValueError, match="takes no embeddings"):
            pipe.score(_make_embeddings(40, d=8, seed=1), embeddings=_make_embeddings(40, d=8, seed=2))

    def test_reference_logits_require_logits_at_score(self):
        pipe = MonitorPipeline(n_permutations=49).fit(_make_embeddings(120, d=8, seed=0),
                                                      reference_logits=_make_logits(120, seed=0))
        with pytest.raises(ValueError, match="energy check"):
            pipe.score(_make_embeddings(40, d=8, seed=1))

    def test_logits_without_reference_logits_refused(self):
        pipe = MonitorPipeline(n_permutations=49).fit(_make_embeddings(120, d=8, seed=0))
        with pytest.raises(ValueError, match="no reference logits"):
            pipe.score(_make_embeddings(40, d=8, seed=1), logits=_make_logits(40, seed=1))

    def test_fixed_energy_threshold_requires_logits(self):
        pipe = MonitorPipeline(n_permutations=49, t2_energy_threshold=0.0).fit(_make_embeddings(120, d=8, seed=0))
        with pytest.raises(ValueError, match="energy check"):
            pipe.score(_make_embeddings(40, d=8, seed=1))
        assert isinstance(pipe.score(_make_embeddings(40, d=8, seed=1), logits=_make_logits(40, seed=1)).invalid, bool)

    def test_groups_at_fit_require_groups_at_score(self):
        Xr, gr = _slides(12, seed=0)
        pipe = MonitorPipeline(n_permutations=49).fit(Xr, groups=gr)
        with pytest.raises(ValueError, match="needs groups"):
            pipe.score(_slides(6, seed=1)[0])

    def test_score_before_fit(self):
        with pytest.raises(RuntimeError, match="fit"):
            MonitorPipeline().score(_make_embeddings(10, d=4))


class TestDeltaGap:
    def test_positive_delta_gap(self):
        # In-domain scores separate the classes perfectly; cross-domain scores misrank some cases.
        y = np.array([0, 0, 0, 1, 1, 1])
        s_in = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        s_cross = np.array([0.1, 0.6, 0.3, 0.5, 0.8, 0.2])
        dg = DeltaGap("E1", "BRCA")
        for _ in range(3):
            dg.add_seed(y, s_in, y, s_cross)
        result = dg.compute()
        assert result.auc_in_mean == 1.0
        assert result.delta_gap_mean > 0

    def test_str_output(self):
        rng = np.random.default_rng(0)
        y = np.array([0, 1, 0, 1])
        dg = DeltaGap("E9", "LUAD")
        dg.add_seed(y, np.array([0.2, 0.8, 0.3, 0.9]), y, np.array([0.3, 0.7, 0.4, 0.75]))
        assert "E9" in str(dg.compute())


class TestStats:
    def test_paired_ttest_significant(self):
        a = [0.85, 0.87, 0.86]
        b = [0.72, 0.74, 0.73]
        result = paired_ttest(a, b)
        assert result.p_value < 0.05

    def test_bonferroni(self):
        pvals = [0.01, 0.03, 0.04, 0.001]
        corrected = bonferroni_correct(pvals, alpha=0.05)
        assert corrected[3] is True
        assert corrected[2] is False
