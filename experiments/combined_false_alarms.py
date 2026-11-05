"""How often the combined flag fires with no shift, under the AND rule (default) and the OR rule.

alpha holds per tier. Through MonitorPipeline with features only, both tiers read the same features, so their alarms
are correlated. Settings: standard Gaussian in 16 and in 64 dimensions; 400 times, a fresh reference of 500 samples
and a batch of 100 from the same distribution; alpha = 0.05; Tier 1 with 200 permutations; seed 0. Prints the rate of
each tier, of AND and of OR. A few minutes.

    python experiments/combined_false_alarms.py
"""

import numpy as np

from wsidrift.monitor.pipeline import MonitorPipeline

REPS, N_REF, M, ALPHA = 400, 500, 100, 0.05


def main() -> None:
    rng = np.random.default_rng(0)
    for dim in (16, 64):
        t1 = t2 = both = either = 0
        for r in range(REPS):
            ref, batch = rng.standard_normal((N_REF, dim)), rng.standard_normal((M, dim))
            res = MonitorPipeline(alpha=ALPHA, seed=r).fit(ref).score(batch)
            a, b = res.tier1.flagged, res.tier2.flagged
            t1, t2, both, either = t1 + a, t2 + b, both + (a and b), either + (a or b)
        print(f"{dim} dimensions: Tier 1 {t1 / REPS:.1%}, Tier 2 {t2 / REPS:.1%}, AND {both / REPS:.1%}, "
              f"OR {either / REPS:.1%} ({REPS} null batches, alpha {ALPHA} per tier)")


if __name__ == "__main__":
    main()
