"""Domain adaptation modules for wsidrift."""
from .grl import GradientReversalLayer, lambda_schedule
from .dann import DANNModel

__all__ = ["GradientReversalLayer", "lambda_schedule", "DANNModel"]
