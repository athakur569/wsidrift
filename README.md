# wsidrift

A label-free, model-agnostic validity monitor for deployed computational pathology pipelines.

It compares a batch of deployment data with a validated reference batch and flags when the batch has shifted, without needing labels or access to the model's internals beyond embeddings.

> **Status: pre-alpha (0.1.1).** The monitors are implemented and tested on synthetic data only. They have **not** been validated on whole-slide-image data, and the flag is not yet linked to measured performance loss. Don't use it for clinical decisions.

## How it works

```
Input features ──► Tier 1: Input divergence (MMD², Wasserstein)  ─┐
                                                                  ├─► Conjunction rule ──► Validity flag
Embeddings (+ logits) ──► Tier 2: Model response (Mahalanobis, energy) ─┘
```

- **Tier 1** tests whether the deployment features come from the same distribution as the reference. It uses a permutation two-sample test on MMD² and mean Wasserstein-1, at false-alarm rate α.
- **Tier 2** checks whether deployment embeddings sit unusually far from the reference (Mahalanobis distance), and optionally whether the classifier is unusually uncertain (energy score). Thresholds are calibrated on held-out reference data at false-alarm rate α.
- **The flag** is raised when both tiers flag. Use `require_both=False` to flag when either does.

## Install

```bash
pip install wsidrift            # numpy, scipy, scikit-learn only
pip install "wsidrift[torch]"   # adds PyTorch, needed only for Tier2Monitor.from_encoder
```

## Example

```python
import numpy as np
from wsidrift.monitor.pipeline import MonitorPipeline

rng = np.random.default_rng(0)
reference = rng.standard_normal((500, 64))            # e.g. encoder embeddings of validated slides
deployment = rng.standard_normal((100, 64)) + 0.5     # a shifted batch

pipe = MonitorPipeline(alpha=0.05).fit(reference)
result = pipe.score(deployment)
print(result.invalid)       # True
print(result.summary())     # statistics, p-values and thresholds
```

To give Tier 1 its own input features (for example stain or colour statistics) and Tier 2 encoder embeddings and logits:

```python
pipe = MonitorPipeline().fit(ref_features, reference_embeddings=ref_emb, reference_logits=ref_logits)
result = pipe.score(dep_features, embeddings=dep_emb, logits=dep_logits)
```

Command line, Tier 1 only:

```bash
wsidrift score --reference ref.npy --deployment dep.npy --alpha 0.05
```

## What's implemented

| Component | State |
|---|---|
| Tier 1 (MMD², Wasserstein, permutation test) | Implemented; synthetic tests only |
| Tier 2 (Mahalanobis, energy, calibrated thresholds) | Implemented; synthetic tests only |
| Conjunction rule and `MonitorPipeline` | Implemented; synthetic tests only |
| Macenko and Reinhard normalization, ΔG and statistics helpers | Implemented; not validated |
| Link from divergence to predicted performance loss (ΔG) | Not implemented |
| E1–E9 model training | Not part of this package (research repository) |

On synthetic Gaussian data at α = 0.05, the measured false-alarm rate was about 2.5% for Tier 1 and 5.4% for Tier 2, averaged over many reference sets. These numbers say the calibration works as designed; they say nothing about performance on pathology data.

## Limitations

- Assumes covariate shift: the input distribution changes while the relationship between inputs and labels stays the same. This is reasonable for stain and scanner variation, and doesn't hold for changes in label definitions.
- Tier 1 runs a permutation test, so it costs more on large batches: about 13 seconds for 1,000 reference and 200 deployment samples with 2,048-dimensional features.
- Tier 2's Mahalanobis statistic is less sensitive than Tier 1 to small shifts spread across many dimensions.

## License

MIT
