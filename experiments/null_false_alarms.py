"""Tier 1's and Tier 2's false-alarm rates on synthetic data with no shift (README, "What's implemented").

Settings: 16-dimensional standard Gaussian; 1,000 times, a fresh reference of 400 samples and a deployment batch of 100
from the same distribution; Tier 1 with 200 permutations; Tier 2 on the Mahalanobis distance only (no logits);
alpha = 0.05; seed 0. Prints each tier's share of batches flagged with its binomial standard error. About three minutes.

    python experiments/null_false_alarms.py
"""

import numpy as np

from wsidrift.monitor.divergence import Tier1Monitor
from wsidrift.monitor.embedding import Tier2Monitor

REPS, N_REF, M, DIM, ALPHA = 1000, 400, 100, 16, 0.05


def main() -> None:
    rng = np.random.default_rng(0)
    t1 = t2 = 0
    for r in range(REPS):
        ref, batch = rng.standard_normal((N_REF, DIM)), rng.standard_normal((M, DIM))
        t1 += Tier1Monitor(alpha=ALPHA, seed=r).fit(ref).score(batch).flagged
        t2 += Tier2Monitor(alpha=ALPHA, seed=r).fit(ref).score(batch).flagged
    for name, k in (("Tier 1", t1), ("Tier 2", t2)):
        p = k / REPS
        print(f"{name}: false-alarm rate {p:.3f} (SE {np.sqrt(p * (1 - p) / REPS):.3f}) at alpha {ALPHA}, "
              f"{REPS} reference sets")


if __name__ == "__main__":
    main()
