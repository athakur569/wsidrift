# wsidrift Architecture

## Overview

wsidrift is a label-free, model-agnostic validity monitor for deployed computational
pathology pipelines. It sits between image ingestion and the downstream diagnostic model,
certifying whether the current batch of slides is within the validated operating envelope.

> **Status (0.1.1, pre-alpha):** the Tier 1 and Tier 2 monitors and the conjunction rule are
> implemented and tested on synthetic data only. They have not been validated on WSI data.
> The divergence–ΔG relationship, E1–E9 training and CycleGAN training are not implemented.

## Module Map

```
wsidrift/
├── monitor/          # Core validity signal
│   ├── divergence.py     Tier 1 — MMD² + Wasserstein
│   ├── embedding.py      Tier 2 — Mahalanobis + energy score
│   ├── validity_score.py Conjunction rule → validity flag
│   └── pipeline.py       Unified entry point
├── data/             # Dataset classes
│   ├── tcga_dataset.py   TCGA patches with TSS metadata
│   └── camelyon17_dataset.py  CAMELYON17 with center + binary label
├── norm/             # Stain normalization (E2, E3, E9)
│   ├── macenko.py        SVD stain vector estimation
│   └── reinhard.py       Lab channel mean/std transfer
├── adapt/            # Domain adaptation (E5-E9)
│   ├── grl.py            Gradient Reversal Layer
│   └── dann.py           DANN Lightning module
├── cyclegan/         # Stain transfer (E4, E7, E9)
│   ├── generator.py      9-block ResNet generator
│   └── trainer.py        Checkpoint manager + inference
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

## E1–E9 Ablation (planned)

Nine stain-handling strategies, each to be run with seeds 42/99/123. Identical
downstream ResNet-50 classifier; only the preprocessing/adaptation varies.
Results will be reported as mean ± SD ΔG, tested with paired t-test, Wilcoxon,
and Bonferroni correction. `scripts/run_ablation.py` is a skeleton and raises until
training is implemented.
