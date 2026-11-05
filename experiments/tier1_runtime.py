"""Tier 1's run time on a large batch (README, Limitations: about 13 seconds).

Settings: 1,000 reference and 200 deployment samples with 2,048 standard-Gaussian features, 200 permutations,
alpha = 0.05, seed 0; one fit and one score, timed with a wall clock. The time depends on the machine and on what else
is running; the script prints the processor it ran on.

    python experiments/tier1_runtime.py
"""

import platform
import time
from pathlib import Path

import numpy as np

from wsidrift.monitor.divergence import Tier1Monitor


def processor() -> str:
    info = Path("/proc/cpuinfo")
    if info.exists():
        for line in info.read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine()


def main() -> None:
    rng = np.random.default_rng(0)
    ref, dep = rng.standard_normal((1000, 2048)), rng.standard_normal((200, 2048))
    t0 = time.perf_counter()
    Tier1Monitor(alpha=0.05, n_permutations=200).fit(ref).score(dep)
    print(f"Tier 1, 1,000 reference and 200 deployment samples, 2,048 features, 200 permutations: "
          f"{time.perf_counter() - t0:.1f} s on {processor()}")


if __name__ == "__main__":
    main()
