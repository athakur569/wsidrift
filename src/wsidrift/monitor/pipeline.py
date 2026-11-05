"""MonitorPipeline: unified entry point that runs Tier 1 → Tier 2 → validity."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .divergence import Tier1Monitor, Tier1Result
from .embedding import Tier2Monitor, Tier2Result
from .validity_score import ValidityMonitor, ValidityResult


@dataclass
class PipelineResult:
    validity: ValidityResult

    @property
    def tier1(self) -> Tier1Result:
        return self.validity.tier1

    @property
    def tier2(self) -> Tier2Result:
        return self.validity.tier2

    @property
    def invalid(self) -> bool:
        return self.validity.overall_flag

    def summary(self) -> dict:
        return {
            "t1_mmd": self.tier1.mmd,
            "t1_wasserstein": self.tier1.wasserstein,
            "t1_p_mmd": self.tier1.p_mmd,
            "t1_p_wasserstein": self.tier1.p_wass,
            "t1_flagged": self.tier1.flagged,
            "t2_mahalanobis": self.tier2.mahalanobis_mean,
            "t2_mahalanobis_threshold": self.tier2.threshold_maha,
            "t2_energy": self.tier2.energy_mean,
            "t2_energy_threshold": self.tier2.threshold_energy,
            "t2_flagged": self.tier2.flagged,
            "invalid": self.invalid,
        }


class MonitorPipeline:
    """Run Tier 1 + Tier 2 and the conjunction rule in one call.

    Tier 1 scores input features (e.g. stain/colour statistics); Tier 2 scores
    encoder embeddings and, optionally, classifier logits. If embeddings are not
    given, Tier 2 uses the Tier 1 features.

    Thresholds are calibrated at false-alarm rate ``alpha`` unless fixed
    thresholds are passed (both Tier 1 thresholds must be set together).

    ``groups`` (one cluster id per sample, e.g. the slide or patient a tile came from) are passed to
    both tiers; tiles of one slide are correlated, and without groups the tiers treat them as
    independent and false-alarm above ``alpha``. ``score`` must be given the same kinds of input as
    ``fit``: embeddings, logits and groups each go to both or to neither, so Tier 2 never scores
    features against an embedding model or silently drops the energy check.
    """

    def __init__(
        self,
        alpha: float = 0.05,
        require_both: bool = True,
        n_permutations: int = 200,
        seed: int = 0,
        t1_mmd_threshold: float | None = None,
        t1_wass_threshold: float | None = None,
        t2_maha_threshold: float | None = None,
        t2_energy_threshold: float | None = None,
        t2_allow_small_reference: bool = False,
    ) -> None:
        self._t1 = Tier1Monitor(
            mmd_threshold=t1_mmd_threshold,
            wass_threshold=t1_wass_threshold,
            alpha=alpha,
            n_permutations=n_permutations,
            seed=seed,
        )
        self._t2 = Tier2Monitor(
            maha_threshold=t2_maha_threshold,
            energy_threshold=t2_energy_threshold,
            alpha=alpha,
            seed=seed,
            allow_small_reference=t2_allow_small_reference,
        )
        self._vm = ValidityMonitor(self._t1, self._t2, require_both=require_both)
        self._fixed_energy = t2_energy_threshold is not None
        self._inputs: dict | None = None

    def fit(
        self,
        reference_features: np.ndarray,
        reference_embeddings: np.ndarray | None = None,
        reference_logits: np.ndarray | None = None,
        groups: np.ndarray | None = None,
    ) -> "MonitorPipeline":
        emb = reference_features if reference_embeddings is None else reference_embeddings
        _check_rows(reference_features, emb, groups, "reference")
        self._t1.fit(reference_features, groups=groups)
        self._t2.fit(emb, reference_logits, groups=groups)
        # A fixed energy threshold also asks for logits at score time.
        self._inputs = {"embeddings": reference_embeddings is not None,
                        "logits": reference_logits is not None or self._fixed_energy,
                        "groups": groups is not None}
        return self

    def score(
        self,
        features: np.ndarray,
        embeddings: np.ndarray | None = None,
        logits: np.ndarray | None = None,
        groups: np.ndarray | None = None,
    ) -> PipelineResult:
        if self._inputs is None:
            raise RuntimeError("Call fit() before score()")
        given = {"embeddings": embeddings is not None, "logits": logits is not None, "groups": groups is not None}
        for kind in ("embeddings", "logits", "groups"):
            if given[kind] != self._inputs[kind]:
                raise ValueError(_MISMATCH[kind][self._inputs[kind]])
        emb = features if embeddings is None else embeddings
        return PipelineResult(validity=self._vm.score(features, embeddings=emb, logits=logits, groups=groups))


_MISMATCH = {   # kind → {was it given at fit: message}
    "embeddings": {True: "fit() was given reference_embeddings, so score() needs embeddings; without them Tier 2 "
                         "would score the features against the embedding model",
                   False: "fit() used the features for Tier 2, so score() takes no embeddings; refit with "
                          "reference_embeddings to score embeddings"},
    "logits": {True: "Tier 2 checks energy (reference logits at fit(), or a fixed energy threshold), so score() needs "
                     "logits; without them the energy check would be dropped silently",
               False: "fit() had no reference logits, so the logits would not be used; refit with reference_logits "
                      "to check energy"},
    "groups": {True: "fit() was given groups, so score() needs groups (one cluster id per deployment sample)",
               False: "fit() had no groups, so score() takes none; refit with groups for clustered samples"},
}


def _check_rows(features: np.ndarray, embeddings: np.ndarray, groups, which: str) -> None:
    """With groups, every sample needs a feature row and an embedding row, both under one cluster id."""
    if groups is not None and len(features) != len(embeddings):
        raise ValueError(f"groups give one cluster id per sample, so the {which} features and embeddings must have "
                         f"the same rows ({len(features)} and {len(embeddings)})")
