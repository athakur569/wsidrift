"""CLI entrypoint for wsidrift."""

from __future__ import annotations

import argparse
import json
import sys


def _load_features(path: str, header: str = "auto"):
    """Load an (n, d) feature matrix from .npy, or from comma-separated .csv.

    ``header``: "auto" skips the first row if it is not all numbers (so a numeric header such as
    ``1,2,3`` is read as data), "yes" always skips it, "no" never does.
    """
    import numpy as np

    if path.endswith(".npy"):
        return np.load(path)
    if header == "auto":
        with open(path) as f:
            first = f.readline()
        try:
            [float(v) for v in first.strip().split(",")]
            skip = 0
        except ValueError:
            skip = 1  # header row
    else:
        skip = 1 if header == "yes" else 0
    try:
        return np.loadtxt(path, delimiter=",", skiprows=skip, ndmin=2)
    except ValueError as e:
        raise ValueError(f"{path}: could not read as numeric CSV ({e})") from None


def _load_groups(path: str | None):
    """One cluster id per line (text), or None."""
    if path is None:
        return None
    import numpy as np

    with open(path) as f:
        ids = [line.strip() for line in f if line.strip()]
    return np.array(ids)


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
    score_p.add_argument("--header", choices=["auto", "yes", "no"], default="auto",
                         help="CSV header row: auto (skip it if not all numbers), yes (always skip), no (never)")
    score_p.add_argument("--reference-groups", default=None,
                         help="Cluster id per reference row (e.g. slide), one per line; pass for tile-level data")
    score_p.add_argument("--deployment-groups", default=None,
                         help="Cluster id per deployment row, one per line; required with --reference-groups")
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

        try:
            X_ref = _load_features(args.reference, args.header)
            X_dep = _load_features(args.deployment, args.header)
            g_ref = _load_groups(args.reference_groups)
            g_dep = _load_groups(args.deployment_groups)
        except ValueError as e:
            sys.exit(str(e))

        tier1 = Tier1Monitor(
            mmd_threshold=args.mmd_threshold,
            wass_threshold=args.wass_threshold,
            alpha=args.alpha,
            n_permutations=args.permutations,
        )
        try:
            if (g_ref is None) != (g_dep is None):
                raise ValueError("pass --reference-groups and --deployment-groups together, or neither")
            result = ValidityMonitor(tier1.fit(X_ref, groups=g_ref)).score(X_dep, groups=g_dep)
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
