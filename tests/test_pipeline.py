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
