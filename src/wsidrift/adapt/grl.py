"""Gradient Reversal Layer for domain-adversarial training."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.autograd import Function


class _GRLFunction(Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.save_for_backward(torch.tensor(lam))
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        (lam,) = ctx.saved_tensors
        return -lam * grad_output, None


class GradientReversalLayer(nn.Module):
    """Apply gradient reversal with a scalar lambda."""

    def __init__(self, lam: float = 1.0) -> None:
        super().__init__()
        self.lam = lam

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _GRLFunction.apply(x, self.lam)


def lambda_schedule(p: float, gamma: float = 10.0) -> float:
    """Anneal lambda from 0 → 1 as training fraction p goes 0 → 1.

    Schedule from Ganin et al. (JMLR 2016): 2/(1+exp(−γp))−1.
    """
    import math
    return 2.0 / (1.0 + math.exp(-gamma * p)) - 1.0
