# wsidrift Architecture

## Overview

wsidrift is a label-free, model-agnostic validity monitor for deployed computational
pathology pipelines. It sits between image ingestion and the downstream diagnostic model,
certifying whether the current batch of slides is within the validated operating envelope.

> **Status (0.1.1, pre-alpha):** the Tier 1 and Tier 2 monitors and the conjunction rule are
> implemented and tested on synthetic data only. They have not been validated on WSI data.
> The divergence–ΔG relationship is not implemented. Model training (E1–E9) lives in the
> stainrob-benchmark research repository, not in this package.

## Module Map

```
wsidrift/
├── monitor/          # Core validity signal
│   ├── divergence.py     Tier 1 — MMD² + Wasserstein
│   ├── embedding.py      Tier 2 — Mahalanobis + energy score
│   ├── validity_score.py Conjunction rule → validity flag
│   └── pipeline.py       Unified entry point
├── norm/             # Stain normalization (E2, E3, E9)
│   ├── macenko.py        SVD stain vector estimation
│   └── reinhard.py       Lab channel mean/std transfer
├── eval/             # Evaluation
│   ├── delta_gap.py      ΔG multi-seed computation
│   └── stats.py          Paired t-test, Wilcoxon, Bonferroni
└── cli.py            # wsidrift score CLI
```

## The Tiered Monitor

### Tier 1 — Input Divergence
Covariate-shift detection in feature space. Requires no model access.
- **MMD²** (Maximum Mean Discrepancy, RBF kernel, median bandwidth; unbiased estimator)
- **Wasserstein-1** (per-feature marginal, exact for unequal batch sizes, averaged)
- **Flag:** permutation two-sample test on both statistics at false-alarm rate α,
  split α/2 per statistic (Bonferroni). Fixed thresholds are available as an override.

### Tier 2 — Model Response
Detects when the encoder's internal representation of deployment data departs
from reference behavior. Requires encoder embeddings and optionally logits.
- **Mahalanobis distance** (precision matrix estimated from reference; diagonal when n ≤ d+1)
- **Energy score** (E = −T log Σexp(logits/T); Liu et al. NeurIPS 2020). Higher energy
  means more out-of-distribution. Used only when reference logits are supplied at fit time.
- **Flag:** reference distances are computed out-of-sample with two half-data models
  (2-fold cross-fitting), and deployment samples are scored with the same two models. The
  threshold for a batch of size m is a bootstrap (1 − α) quantile of the difference between
  a size-m batch mean and the reference mean, so reference sampling error is included.

### Conjunction Rule
Flags "likely invalid" when Tier 1 AND Tier 2 both flag. Requiring both lowers the
false-alarm rate, at the cost of missing shifts that only one tier detects.
`require_both=False` switches to an OR rule.

## Development vs Deployment

| Stage | Signal | Labels needed |
|---|---|---|
| Development | ΔG (observed outcome) | Yes |
| Deployment | Predicted ΔG (Tier 1 + Tier 2) | No |

The divergence–ΔG relationship is to be learned during development on TCGA/CAMELYON17
(not yet implemented). At deployment, only the label-free half runs.

## E1–E9 Ablation

The nine stain-handling conditions are trained and evaluated in the stainrob-benchmark
research repository. wsidrift provides the label-free monitor those runs are scored with.
