"""Stain normalization methods for wsidrift preprocessing."""
from .macenko import MacenkoNormalizer
from .reinhard import ReinhardNormalizer

__all__ = ["MacenkoNormalizer", "ReinhardNormalizer"]
