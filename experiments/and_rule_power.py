"""Detection under the default AND rule when a shift moves one tier more than the other (README, Limitations).

Tier 2's Mahalanobis check is two-sided, so a batch that is less spread out than the reference (spread × 0.8 or × 0.5,
or collapsed onto the reference mean as blank or blurred tiles might be) trips it as well as one that sits further
away. For comparison, the last column applies release 0.1.1's one-sided rule (flag only above the 1 − α bootstrap
quantile, the same draws), under which such batches could never trip the AND rule whatever Tier 1 saw. Settings:
64-dimensional standard Gaussian reference of 500 samples (one per repeat), batches of 100, 200 repeats per condition,
alpha = 0.05, MonitorPipeline with features only, seed 0. A few minutes.

    python experiments/and_rule_power.py
"""

import numpy as np

from wsidrift.monitor.pipeline import MonitorPipeline

REPS, N_REF, M, DIM, ALPHA = 200, 500, 100, 64, 0.05
CONDITIONS = {"no shift": lambda z: z, "mean +0.1 in every dimension": lambda z: z + 0.1,
              "spread x0.8": lambda z: 0.8 * z, "spread x0.5": lambda z: 0.5 * z,
              "collapsed onto the reference mean (spread x0.01)": lambda z: 0.01 * z}


def one_sided_tier2(pipe: MonitorPipeline, maha_mean: float) -> bool:
    """Release 0.1.1's Tier 2 rule on the same bootstrap draws: flag only above the 1 − α quantile."""
    t2 = pipe._t2
    return maha_mean > t2._ref_maha.mean() + np.quantile(t2._bootstrap_differences(t2._ref_maha, M), 1 - ALPHA)


def main() -> None:
    rng = np.random.default_rng(0)
    print(f"{DIM} dimensions, reference {N_REF}, batches of {M}, {REPS} repeats, alpha {ALPHA}")
    print(f"{'condition':<50} {'Tier 1':>7} {'Tier 2':>7} {'AND':>6} {'OR':>6} {'AND, 0.1.1 one-sided Tier 2':>28}")
    for name, shift in CONDITIONS.items():
        c = np.zeros(5)
        for r in range(REPS):
            ref = rng.standard_normal((N_REF, DIM))
            batch = shift(rng.standard_normal((M, DIM)))
            pipe = MonitorPipeline(alpha=ALPHA, seed=r).fit(ref)
            res = pipe.score(batch)
            a, b = res.tier1.flagged, res.tier2.flagged
            c += [a, b, a and b, a or b, a and one_sided_tier2(pipe, res.tier2.mahalanobis_mean)]
        print(f"{name:<50} " + " ".join(f"{v:>{w}.0%}" for v, w in zip(c / REPS, (7, 7, 6, 6, 28))))


if __name__ == "__main__":
    main()
