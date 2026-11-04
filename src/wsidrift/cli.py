"""CLI entrypoint for wsidrift."""

from __future__ import annotations

import argparse
import json
import sys


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="wsidrift",
        description="Label-free validity monitor for computational pathology pipelines.",
    )
    sub = parser.add_subparsers(dest="command")

    # wsidrift score
    score_p = sub.add_parser("score", help="Score a deployment batch against a reference")
    score_p.add_argument("--reference", required=True, help="Reference features .npy or .csv")
    score_p.add_argument("--deployment", required=True, help="Deployment features .npy or .csv")
    score_p.add_argument("--output", default=None, help="Save result JSON to this path")
    score_p.add_argument("--alpha", type=float, default=0.05, help="False-alarm rate for the permutation test")
    score_p.add_argument("--permutations", type=int, default=200)
    score_p.add_argument("--mmd-threshold", type=float, default=None, help="Fixed threshold (requires --wass-threshold)")
    score_p.add_argument("--wass-threshold", type=float, default=None, help="Fixed threshold (requires --mmd-threshold)")

    args = parser.parse_args(argv)

    if args.command == "score":
        import numpy as np
        from wsidrift.monitor.divergence import Tier1Monitor
        from wsidrift.monitor.validity_score import ValidityMonitor

        X_ref = np.load(args.reference) if args.reference.endswith(".npy") else \
                __import__("pandas").read_csv(args.reference).values
        X_dep = np.load(args.deployment) if args.deployment.endswith(".npy") else \
                __import__("pandas").read_csv(args.deployment).values

        tier1 = Tier1Monitor(
            mmd_threshold=args.mmd_threshold,
            wass_threshold=args.wass_threshold,
            alpha=args.alpha,
            n_permutations=args.permutations,
        )
        try:
            result = ValidityMonitor(tier1.fit(X_ref)).score(X_dep)
        except ValueError as e:
            sys.exit(str(e))
        print(result.summary_line())

        if args.output:
            with open(args.output, "w") as f:
                json.dump(result.to_dict(), f, indent=2)
            print(f"Result saved to {args.output}")
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
