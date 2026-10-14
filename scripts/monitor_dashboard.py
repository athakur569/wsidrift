#!/usr/bin/env python3
"""Monitoring dashboard — print Tier 1+2 validity results for a deployment batch.

Usage:
    python scripts/monitor_dashboard.py \\
        --reference embeddings/reference.npy \\
        --deployment embeddings/deployment.npy \\
        [--reference-logits logits/reference.npy --logits logits/deployment.npy]

Thresholds are calibrated at --alpha. The same embeddings feed both tiers here;
use MonitorPipeline directly to give Tier 1 separate input features.
"""

from __future__ import annotations

import argparse

import numpy as np

from wsidrift.monitor.pipeline import MonitorPipeline

_ACTION_MAP = {
    (False, False): "PASS — no shift detected at this false-alarm rate",
    (True, False): "CAUTION — input shift detected; watch Tier 2 on the next batch",
    (False, True): "CAUTION — model-response shift without input shift; investigate",
    (True, True): "FLAG — both tiers shifted; revalidation recommended",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="wsidrift monitoring dashboard")
    parser.add_argument("--reference", required=True, help="Reference embeddings (.npy)")
    parser.add_argument("--deployment", required=True, help="Deployment embeddings (.npy)")
    parser.add_argument("--reference-logits", help="Reference logits (.npy), enables the energy score")
    parser.add_argument("--logits", help="Deployment logits (.npy)")
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    ref, dep = np.load(args.reference), np.load(args.deployment)
    ref_logits = np.load(args.reference_logits) if args.reference_logits else None
    logits = np.load(args.logits) if args.logits else None

    pipe = MonitorPipeline(alpha=args.alpha).fit(ref, reference_logits=ref_logits)
    result = pipe.score(dep, logits=logits)
    r1, r2 = result.tier1, result.tier2

    print("\n" + "=" * 64)
    print("  wsidrift — Validity Monitor Dashboard (pre-alpha)")
    print("=" * 64)
    print(f"  Reference batch  : {len(ref)}    Deployment batch : {len(dep)}    alpha : {args.alpha}")
    print("-" * 64)
    print("  TIER 1 — Input Divergence (permutation test)")
    print(f"    MMD²           : {r1.mmd:.4f}   p = {r1.p_mmd:.3f}")
    print(f"    Wasserstein    : {r1.wasserstein:.4f}   p = {r1.p_wass:.3f}")
    print(f"    Tier 1 flagged : {r1.flagged}")
    print("-" * 64)
    print("  TIER 2 — Model Response (calibrated thresholds)")
    print(f"    Mahalanobis    : {r2.mahalanobis_mean:.4f}   threshold {r2.threshold_maha:.4f}")
    if r2.energy_mean is not None:
        print(f"    Energy score   : {r2.energy_mean:.4f}   threshold {r2.threshold_energy:.4f}")
    print(f"    Tier 2 flagged : {r2.flagged}")
    print("=" * 64)
    print(f"  VALIDITY: {'INVALID' if result.invalid else 'VALID'}")
    print(f"  ACTION  : {_ACTION_MAP[(r1.flagged, r2.flagged)]}")
    print("=" * 64 + "\n")


if __name__ == "__main__":
    main()
