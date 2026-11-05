"""False alarms with correlated samples, tile-level against cluster-level calibration (CHANGELOG, `groups`).

No real shift, strong slide effects: each slide (cluster) has its own offset, N(0, 1) per dimension, and its tiles
scatter around it with SD 0.3, in 3 dimensions with 15 tiles per slide. 200 repeats, each with a fresh reference of 40
slides (seed 50000 + 2r) and a fresh deployment of 12 slides (seed 50001 + 2r); alpha = 0.10; Tier 1 with 99
permutations. Each deployment is scored twice: without groups (tiles treated as independent) and with slide groups.
About two minutes.

    python experiments/clustered_false_alarms.py
"""

import numpy as np

from wsidrift.monitor.divergence import Tier1Monitor
from wsidrift.monitor.embedding import Tier2Monitor

REPS, REF_SLIDES, DEP_SLIDES, PER, DIM, ALPHA = 200, 40, 12, 15, 3, 0.10


def slides(k: int, seed: int, effect: float = 1.0, noise: float = 0.3) -> tuple[np.ndarray, np.ndarray]:
    r = np.random.default_rng(seed)
    X = np.concatenate([r.normal(0, effect, DIM) + r.normal(0, noise, (PER, DIM)) for _ in range(k)])
    return X, np.repeat(np.arange(k), PER)


def main() -> None:
    for name, cls, kw in (("Tier 1", Tier1Monitor, {"alpha": ALPHA, "n_permutations": 99}),
                          ("Tier 2", Tier2Monitor, {"alpha": ALPHA})):
        tile = cluster = 0
        for r in range(REPS):
            Xr, gr = slides(REF_SLIDES, 50000 + 2 * r)
            Xd, gd = slides(DEP_SLIDES, 50001 + 2 * r)
            tile += cls(**kw).fit(Xr).score(Xd).flagged
            cluster += cls(**kw).fit(Xr, groups=gr).score(Xd, groups=gd).flagged
        print(f"{name}: false alarms {tile / REPS:.0%} tile-level, {cluster / REPS:.0%} cluster-level "
              f"(alpha {ALPHA}, {REPS} deployments, {REF_SLIDES} reference slides)")


if __name__ == "__main__":
    main()
