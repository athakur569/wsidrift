"""Tier 2's false-alarm rate on clean data by reference size, below and above the cross-fitting minimum of 40.

Below 40 samples the threshold is set on in-sample distances; `fit` refuses unless allow_small_reference=True, which is
passed here to measure what the refusal prevents. Settings: standard Gaussian in 5 and 16 dimensions; reference sizes
20, 30, 40, 60, 100 and 200; 1,000 fresh references per cell, each scored on one batch of 50 from the same
distribution; alpha = 0.05; Mahalanobis only; seed 0. Prints the rate with its binomial standard error. A minute.

    python experiments/tier2_reference_size.py
"""

import logging

import numpy as np

from wsidrift.monitor.embedding import Tier2Monitor

logging.getLogger("wsidrift").setLevel(logging.ERROR)     # the small-reference warning, once per fit

REPS, M, ALPHA = 1000, 50, 0.05


def main() -> None:
    rng = np.random.default_rng(0)
    for dim in (5, 16):
        for n in (20, 30, 40, 60, 100, 200):
            k = sum(Tier2Monitor(alpha=ALPHA, seed=r, allow_small_reference=n < 40).fit(rng.standard_normal((n, dim)))
                    .score(rng.standard_normal((M, dim))).flagged for r in range(REPS))
            p = k / REPS
            print(f"{dim} dimensions, reference {n}: false-alarm rate {p:.3f} (SE {np.sqrt(p * (1 - p) / REPS):.3f})")


if __name__ == "__main__":
    main()
