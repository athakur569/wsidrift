"""Statistical tests for E1-E9 ablation comparisons."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats as scipy_stats


@dataclass
class TestResult:
    statistic: float
    p_value: float
    significant: bool
    alpha: float = 0.05


def paired_ttest(a: list[float], b: list[float], alpha: float = 0.05) -> TestResult:
    """Paired t-test between two lists of per-seed values."""
    stat, p = scipy_stats.ttest_rel(a, b)
    return TestResult(statistic=float(stat), p_value=float(p), significant=p < alpha, alpha=alpha)


def wilcoxon_test(a: list[float], b: list[float], alpha: float = 0.05) -> TestResult:
    """Wilcoxon signed-rank test (non-parametric alternative to paired t-test)."""
    diffs = np.array(a) - np.array(b)
    if np.all(diffs == 0):
        return TestResult(statistic=0.0, p_value=1.0, significant=False, alpha=alpha)
    stat, p = scipy_stats.wilcoxon(diffs)
    return TestResult(statistic=float(stat), p_value=float(p), significant=p < alpha, alpha=alpha)


def bonferroni_correct(p_values: list[float], alpha: float = 0.05) -> list[float]:
    """Return Bonferroni-corrected significance threshold for each comparison."""
    n = len(p_values)
    corrected_alpha = alpha / n
    return [p < corrected_alpha for p in p_values]
