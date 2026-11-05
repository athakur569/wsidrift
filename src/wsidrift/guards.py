"""Refuse input a monitor cannot read.

A NaN that reaches a permutation test makes the observed statistic NaN. Every comparison with NaN is false, so no
permuted value counts as exceeding it and the p-value comes out as the smallest possible: a failed slide reads as the
strongest drift signal there is (seen in a CAMELYON17 rehearsal run). A failed slide is not evidence about the batch,
so the monitors stop and name it. They do not drop it, because a run of failed slides may be what needs attention.
"""

from __future__ import annotations

import numpy as np


def _rows(bad: np.ndarray, ids) -> str:
    where = np.flatnonzero(bad)
    names = [str(ids[i]) if ids is not None else f"row {i}" for i in where[:5]]
    more = f" and {len(where) - 5} more" if len(where) > 5 else ""
    return f"{len(where)} of {len(bad)} ({', '.join(names)}{more})"


def finite(x, name: str, ids=None) -> np.ndarray:
    """``x`` as a float array, or ValueError naming the rows that hold NaN or infinite values."""
    a = np.asarray(x, float)
    if a.size == 0:
        raise ValueError(f"{name}: empty")
    bad = ~np.isfinite(a).reshape(len(a), -1).all(1)
    if bad.any():
        raise ValueError(f"{name}: NaN or infinite values in {_rows(bad, ids)}. A failed slide is not evidence of "
                         "drift: find out why it failed, then rerun without it.")
    return a


def finite_value(v: float, name: str) -> float:
    """One number from a stream, or ValueError."""
    v = float(v)
    if not np.isfinite(v):
        raise ValueError(f"{name}: NaN or infinite. A failed unit is not evidence of drift.")
    return v


def probabilities(p, name: str, ids=None) -> np.ndarray:
    """``p`` as a float array, or ValueError if a value is not a probability."""
    a = finite(p, name, ids)
    bad = ((a < 0) | (a > 1)).reshape(len(a), -1).any(1)
    if bad.any():
        raise ValueError(f"{name}: values outside 0 to 1 in {_rows(bad, ids)}. "
                         "These look like logits or percentages, not probabilities.")
    return a
