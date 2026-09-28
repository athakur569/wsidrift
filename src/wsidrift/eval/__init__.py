"""Evaluation utilities: ΔG, AUC, and statistical tests."""
from .delta_gap import DeltaGap, DeltaGapResult
from .stats import paired_ttest, wilcoxon_test, bonferroni_correct

__all__ = ["DeltaGap", "DeltaGapResult", "paired_ttest", "wilcoxon_test", "bonferroni_correct"]
