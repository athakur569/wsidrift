#!/usr/bin/env python3
"""Monitoring dashboard — print Tier 1+2 validity scores for a deployment batch.

Usage:
    python scripts/monitor_dashboard.py \\
        --reference embeddings/reference.npy \\
        --deployment embeddings/deployment.npy \\
        --logits logits/deployment.npy

Outputs a summary table: Tier 1 (MMD, Wasserstein), Tier 2 (Mahalanobis, energy),
composed validity flag, and recommended action.
"""

from __future__ import annotations

import argparse
import textwrap

import numpy as np

from wsidrift.monitor.divergence import Tier1Monitor
from wsidrift.monitor.embedding import Tier2Monitor
from wsidrift.monitor.validity_score import ValidityMonitor


_ACTION_MAP = {
    (False, False): "PASS — no corrective action required",
    (True,  False): "CAUTION — input shift detected; monitor Tier 2 next batch",
    (False, True):  "CAUTION — model response shift without input shift; check for concept drift",
    (True,  True):  "FLAG — predicted ΔG significant; trigger revalidation protocol",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="wsidrift monitoring dashboard")
    parser.add_argument("--reference",   required=True, help="Reference embeddings (.npy)")
    parser.add_argument("--deployment",  required=True, help="Deployment embeddings (.npy)")
    parser.add_argument("--logits",      required=True, help="Deployment logits (.npy)")
    parser.add_argument("--t1-thresh",   type=float, default=0.5)
    parser.add_argument("--t2-thresh",   type=float, default=3.0)
    args = parser.parse_args()

    ref = np.load(args.reference)
    dep = np.load(args.deployment)
    logits = np.load(args.logits)

    t1 = Tier1Monitor(mmd_threshold=args.t1_thresh)
    t1.fit(ref)
    r1 = t1.score(dep)

    t2 = Tier2Monitor(mahal_threshold=args.t2_thresh)
    t2.fit(ref)
    r2 = t2.score(dep, logits)

    vm = ValidityMonitor(t1_threshold=args.t1_thresh, t2_threshold=args.t2_thresh)
    vm.fit(ref)
    vr = vm.score(dep, logits)

    print("\n" + "=" * 60)
    print("  wsidrift — Validity Monitor Dashboard")
    print("=" * 60)
    print(f"  Reference batch  : {len(ref)} embeddings")
    print(f"  Deployment batch : {len(dep)} embeddings")
    print("-" * 60)
    print(f"  TIER 1 — Input Divergence")
    print(f"    MMD²           : {r1.mmd:.4f}  (flag: {r1.mmd_flagged})")
    print(f"    Wasserstein    : {r1.wasserstein:.4f}  (flag: {r1.wasserstein_flagged})")
    print(f"    Tier 1 flagged : {r1.flagged}")
    print("-" * 60)
    print(f"  TIER 2 — Model Response")
    print(f"    Mahalanobis    : {r2.mahalanobis_mean:.4f}  (flag: {r2.mahalanobis_flagged})")
    print(f"    Energy score   : {r2.energy_mean:.4f}  (flag: {r2.energy_flagged})")
    print(f"    Tier 2 flagged : {r2.flagged}")
    print("=" * 60)
    action = _ACTION_MAP.get((r1.flagged, r2.flagged), "UNKNOWN")
    print(f"  VALIDITY: {'INVALID' if vr.invalid else 'VALID'}")
    print(f"  ACTION : {action}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
