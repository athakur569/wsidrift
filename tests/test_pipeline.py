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
    def test_fit_and_score_no_shift(self):
        ref = _make_embeddings(200, seed=0)
        dep = _make_embeddings(100, seed=1)
        logits = _make_logits(100, seed=1)
        pipe = MonitorPipeline()
        pipe.fit(ref)
        result = pipe.score(dep, logits)
        assert hasattr(result, "invalid")
        assert isinstance(result.invalid, bool)

    def test_large_shift_flags_invalid(self):
        ref = _make_embeddings(200, seed=0, shift=0.0)
        dep = _make_embeddings(100, seed=2, shift=5.0)
        logits = _make_logits(100, seed=2)
        pipe = MonitorPipeline(t1_mmd_threshold=0.1, t2_mahal_threshold=1.0)
        pipe.fit(ref)
        result = pipe.score(dep, logits)
        assert result.tier1.flagged or result.tier2.flagged

    def test_summary_keys(self):
        ref = _make_embeddings(100, seed=0)
        dep = _make_embeddings(50, seed=3)
        logits = _make_logits(50, seed=3)
        pipe = MonitorPipeline()
        pipe.fit(ref)
        s = pipe.score(dep, logits).summary()
        assert "t1_mmd" in s and "invalid" in s


class TestDeltaGap:
    def test_positive_delta_gap(self):
        rng = np.random.default_rng(42)
        y = np.array([0, 0, 1, 1, 0, 1])
        dg = DeltaGap("E1", "BRCA")
        for _ in range(3):
            s_in = rng.uniform(0.7, 0.9, size=6)
            s_cross = s_in - rng.uniform(0.05, 0.15, size=6)
            dg.add_seed(y, s_in, y, np.clip(s_cross, 0, 1))
        result = dg.compute()
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
