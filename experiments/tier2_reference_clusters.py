"""Tier 2 false alarms with groups, by the number of reference clusters (README, Limitations; CHANGELOG).

No real shift, strong slide effects: each slide has its own offset, N(0, 1) per dimension, and its 25 tiles scatter
around it with SD 0.3. Reference of k slides (2 to 40), deployment of 12 slides, 300 repeats per row with fresh slides
(seeds 70000 + 2r and 70001 + 2r), in 3 and 16 dimensions, at alpha 0.05 and 0.10. Below 12 reference clusters
Tier2Monitor refuses calibrated thresholds; this script passes allow_small_reference=True to measure what the refusal
prevents. About a minute.

    python experiments/tier2_reference_clusters.py
"""

import logging

import numpy as np

from wsidrift.monitor.embedding import Tier2Monitor

REPS, DEP_SLIDES, PER = 300, 12, 25
REF_SLIDES = (2, 3, 4, 6, 8, 10, 12, 16, 20, 40)


def slides(k: int, dim: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    r = np.random.default_rng(seed)
    X = np.concatenate([r.normal(0, 1, dim) + r.normal(0, 0.3, (PER, dim)) for _ in range(k)])
    return X, np.repeat(np.arange(k), PER)


def main() -> None:
    logging.disable(logging.WARNING)
    for dim in (3, 16):
        for alpha in (0.05, 0.10):
            rates = []
            for k in REF_SLIDES:
                fa = 0
                for r in range(REPS):
                    Xr, gr = slides(k, dim, 70000 + 2 * r)
                    Xd, gd = slides(DEP_SLIDES, dim, 70001 + 2 * r)
                    fa += Tier2Monitor(alpha=alpha, allow_small_reference=True).fit(Xr, groups=gr).score(
                        Xd, groups=gd).flagged
                rates.append(f"{k}: {fa / REPS:.0%}")
            print(f"{dim} dimensions, alpha {alpha}: false alarms by reference slides — " + ", ".join(rates), flush=True)


if __name__ == "__main__":
    main()
