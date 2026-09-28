#!/usr/bin/env python3
"""E1-E9 ablation runner — trains all nine stain-handling conditions and logs ΔG.

Usage:
    python scripts/run_ablation.py --config configs/ablation.yaml --cohort BRCA

Each condition E1-E9 uses the same frozen patch set and identical downstream
classifier (ResNet-50); only the stain-handling method varies (see E1-E9 table
in CLAUDE.md). Seeds 42/99/123 are run per condition; mean±SD is reported.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

CONDITIONS = {
    "E1": dict(norm=None,       cyclegan=False, dann=False, simclr=False),
    "E2": dict(norm="macenko",  cyclegan=False, dann=False, simclr=False),
    "E3": dict(norm="reinhard", cyclegan=False, dann=False, simclr=False),
    "E4": dict(norm=None,       cyclegan=True,  dann=False, simclr=False),
    "E5": dict(norm=None,       cyclegan=False, dann=True,  simclr=False),
    "E6": dict(norm=None,       cyclegan=False, dann=False, simclr=True),
    "E7": dict(norm=None,       cyclegan=True,  dann=True,  simclr=False),
    "E8": dict(norm=None,       cyclegan=False, dann=True,  simclr=True),
    "E9": dict(norm="macenko",  cyclegan=True,  dann=True,  simclr=True),
}
SEEDS = [42, 99, 123]


@dataclass
class ConditionResult:
    condition: str
    cohort: str
    auc_in: list[float] = field(default_factory=list)
    auc_cross: list[float] = field(default_factory=list)

    @property
    def delta_gap_mean(self) -> float:
        dg = [i - c for i, c in zip(self.auc_in, self.auc_cross)]
        return float(np.mean(dg))

    @property
    def delta_gap_std(self) -> float:
        dg = [i - c for i, c in zip(self.auc_in, self.auc_cross)]
        return float(np.std(dg))


def run_condition(condition: str, config: dict, cohort: str, seed: int) -> dict:
    """Placeholder — replaced by real training + evaluation loop."""
    rng = np.random.default_rng(seed + hash(condition + cohort) % 10000)
    auc_in = float(rng.uniform(0.70, 0.92))
    auc_cross = float(rng.uniform(0.60, auc_in))
    return {"auc_in": auc_in, "auc_cross": auc_cross}


def main() -> None:
    parser = argparse.ArgumentParser(description="E1-E9 ablation runner")
    parser.add_argument("--config", default="configs/ablation.yaml")
    parser.add_argument("--cohort", default="BRCA")
    parser.add_argument("--out", default="results/ablation_results.json")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    results: list[dict] = []

    for cond_name in CONDITIONS:
        cr = ConditionResult(condition=cond_name, cohort=args.cohort)
        for seed in SEEDS:
            log.info(f"Running {cond_name} seed={seed} cohort={args.cohort}")
            r = run_condition(cond_name, CONDITIONS[cond_name], args.cohort, seed)
            cr.auc_in.append(r["auc_in"])
            cr.auc_cross.append(r["auc_cross"])
        log.info(
            f"{cond_name}: ΔG = {cr.delta_gap_mean:.4f} ± {cr.delta_gap_std:.4f}"
        )
        results.append(asdict(cr))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Saved results to {args.out}")


if __name__ == "__main__":
    main()
