"""Detection under the default AND rule when only one tier can see the shift (design finding, external review).

Tier 2 flags only when Mahalanobis distances grow, so a shift that shrinks the spread, or collapses the embeddings onto
the reference mean (as blank or blurred tiles might), can never trip the AND rule, whatever Tier 1 sees. Settings:
64-dimensional standard Gaussian reference of 500 samples (one per repeat), batches of 100, 200 repeats per condition,
alpha = 0.05, MonitorPipeline with features only, seed 0. Beside the shipped rule, a prototype two-sided Tier 2 (each
tail at alpha/2, the same bootstrap as the shipped threshold; not in the package) shows what it would catch.

    python experiments/and_rule_power.py
"""

import numpy as np

from wsidrift.monitor.pipeline import MonitorPipeline

REPS, N_REF, M, DIM, ALPHA = 200, 500, 100, 64, 0.05
CONDITIONS = {"no shift": lambda z: z, "mean +0.1 in every dimension": lambda z: z + 0.1,
              "spread x0.8": lambda z: 0.8 * z, "spread x0.5": lambda z: 0.5 * z,
              "collapsed onto the reference mean (spread x0.01)": lambda z: 0.01 * z}


def two_sided_tier2(pipe: MonitorPipeline, maha_mean: float) -> bool:
    t2 = pipe._t2                                            # prototype only: the shipped Tier 2 is one-sided
    hi = t2._bootstrap_upper(t2._ref_maha, M, ALPHA / 2)
    lo = t2._bootstrap_upper(t2._ref_maha, M, 1 - ALPHA / 2)
    return maha_mean > hi or maha_mean < lo


def main() -> None:
    rng = np.random.default_rng(0)
    print(f"{DIM} dimensions, reference {N_REF}, batches of {M}, {REPS} repeats, alpha {ALPHA}")
    print(f"{'condition':<50} {'Tier 1':>7} {'Tier 2':>7} {'AND':>6} {'OR':>6} {'T2 two-sided':>13} {'AND, two-sided':>15}")
    for name, shift in CONDITIONS.items():
        c = np.zeros(6)
        for r in range(REPS):
            ref = rng.standard_normal((N_REF, DIM))
            batch = shift(rng.standard_normal((M, DIM)))
            pipe = MonitorPipeline(alpha=ALPHA, seed=r).fit(ref)
            res = pipe.score(batch)
            a, b = res.tier1.flagged, res.tier2.flagged
            b2 = two_sided_tier2(pipe, res.tier2.mahalanobis_mean)
            c += [a, b, a and b, a or b, b2, a and b2]
        print(f"{name:<50} " + " ".join(f"{v:>{w}.0%}" for v, w in zip(c / REPS, (7, 7, 6, 6, 13, 15))))


if __name__ == "__main__":
    main()
