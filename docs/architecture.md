# wsidrift Architecture

## Overview

wsidrift is a label-free, model-agnostic validity monitor for deployed computational
pathology pipelines. It sits between image ingestion and the downstream diagnostic model,
certifying whether the current batch of slides is within the validated operating envelope.

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
- **MMD²** (Maximum Mean Discrepancy, RBF kernel, median bandwidth)
- **Wasserstein-1** (per-feature marginal, aggregated)

### Tier 2 — Model Response
Detects when the encoder's internal representation of deployment data departs
from reference behavior. Requires encoder embeddings and optionally logits.
- **Mahalanobis distance** (precision matrix estimated from reference)
- **Energy score** (−T log Σexp(logits/T); Liu et al. NeurIPS 2020)

### Conjunction Rule
Flags "likely invalid" when Tier 1 AND Tier 2 both exceed their thresholds.
The conjunction cuts false alarms while catching silent failures either tier
misses alone.

## Development vs Deployment

| Stage | Signal | Labels needed |
|---|---|---|
| Development | ΔG (observed outcome) | Yes |
| Deployment | Predicted ΔG (Tier 1 + Tier 2) | No |

The divergence–ΔG relationship is learned during development on TCGA/CAMELYON17.
At deployment, only the label-free half runs.

## E1–E9 Ablation

Nine stain-handling strategies, each run with seeds 42/99/123. Identical
downstream ResNet-50 classifier; only the preprocessing/adaptation varies.
Results reported as mean ± SD ΔG; significance tested with paired t-test,
Wilcoxon, and Bonferroni correction.
