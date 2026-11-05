"""Composed validity score — conjunction rule across Tier 1 and Tier 2.

Flags "likely invalid" only when BOTH tiers signal shift. The conjunction
can only remove alarms: it lowers the false-alarm rate at the cost of missing
some shifts that one tier alone would flag. It cannot catch a failure that
neither tier sees.

Development-time relationship:
  ΔG = in-domain AUC − cross-domain AUC   [requires labels, computed once]
  Divergence + Tier 2 predict ΔG at deployment [label-free, always available]

At deployment report "predicted ΔG" not "ΔG" — this is an empirically validated
predictive relationship, not a mathematical identity.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Optional

from .divergence import Tier1Monitor, Tier1Result
from .embedding import Tier2Monitor, Tier2Result

logger = logging.getLogger(__name__)


@dataclass
class ValidityResult:
    tier1: Tier1Result
    tier2: Optional[Tier2Result]
    # Conjunction: both tiers must flag for overall flag
    overall_flag: bool
    # Single scalar for monitoring dashboards and alerts
    validity_score: float  # 0 = clean, 1 = fully flagged

    def to_dict(self) -> dict:
        d = {
            "tier1": asdict(self.tier1),
            "tier2": asdict(self.tier2) if self.tier2 else None,
            "overall_flag": self.overall_flag,
            "validity_score": self.validity_score,
        }
        return d

    def summary_line(self) -> str:
        flag = "INVALID" if self.overall_flag else "VALID"
        return (
            f"[{flag}] score={self.validity_score:.3f} "
            f"T1={'FLAGGED' if self.tier1.flagged else 'ok'} "
            f"T2={'FLAGGED' if self.tier2 and self.tier2.flagged else 'ok/absent'}"
        )


class ValidityMonitor:
    """Compose Tier 1 and Tier 2 into a single validity verdict.

    Args:
        tier1: Fitted Tier1Monitor.
        tier2: Fitted Tier2Monitor (optional; if absent, Tier 1 alone decides).
        require_both: If True (default), flag only when BOTH tiers signal shift.
                      If False, OR logic — flag if either tier flags.
    """

    def __init__(
        self,
        tier1: Tier1Monitor,
        tier2: Optional[Tier2Monitor] = None,
        require_both: bool = True,
    ) -> None:
        self.tier1 = tier1
        self.tier2 = tier2
        self.require_both = require_both

    def score(
        self,
        X_deployment,
        embeddings=None,
        logits=None,
        groups=None,
    ) -> ValidityResult:
        """Score a deployment batch.

        Args:
            X_deployment: Input features for Tier 1 (n, d).
            embeddings:   Encoder embeddings for Tier 2 (n, emb_dim). Required when a Tier 2 monitor is set
                          (pass the Tier 1 features if Tier 2 was fitted on them).
            logits:       Classifier logits for energy score (n, n_classes). Optional.
            groups:       Cluster id per sample (e.g. slide), for both tiers; required if, and only if, the
                          monitors were fitted with groups. The features and embeddings must then have the same rows.

        Returns:
            ValidityResult with per-tier results and overall flag.
        """
        if self.tier2 is not None and embeddings is None:
            raise ValueError("this ValidityMonitor has a Tier 2 monitor, so score() needs embeddings (the Tier 1 "
                             "features, if Tier 2 was fitted on them); without them only Tier 1 would decide")
        if groups is not None and embeddings is not None and len(embeddings) != len(X_deployment):
            raise ValueError("groups give one cluster id per sample, so the features and embeddings must have the "
                             f"same rows ({len(X_deployment)} and {len(embeddings)})")
        t1 = self.tier1.score(X_deployment, groups=groups)

        t2 = None
        if self.tier2 is not None:
            t2 = self.tier2.score(embeddings, logits, groups=groups)

        if t2 is None:
            overall_flag = t1.flagged
        elif self.require_both:
            overall_flag = t1.flagged and t2.flagged
        else:
            overall_flag = t1.flagged or t2.flagged

        # Validity score: average of flagged signals (0 = clean)
        signals = [float(t1.flagged)]
        if t2 is not None:
            signals.append(float(t2.flagged))
        validity_score = float(sum(signals) / len(signals))

        result = ValidityResult(
            tier1=t1,
            tier2=t2,
            overall_flag=overall_flag,
            validity_score=validity_score,
        )

        status = "INVALID" if overall_flag else "VALID"
        logger.info("ValidityMonitor: %s (score=%.3f)", status, validity_score)
        return result
