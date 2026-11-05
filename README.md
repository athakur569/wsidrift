# wsidrift

A label-free, model-agnostic validity monitor for deployed computational pathology pipelines.

It compares a batch of deployment data with a validated reference batch and flags when the batch has shifted, without needing labels or access to the model's internals beyond embeddings.

> **Status: pre-alpha.** PyPI has 0.1.1; `main` is 0.1.2.dev0 and differs from it (see [Versions](#versions-and-where-each-number-comes-from)). The monitors are implemented and tested on synthetic data only. They have **not** been validated on whole-slide-image data, and the flag is not yet linked to measured performance loss. Don't use it for clinical decisions.

## How it works

```
Input features ──► Tier 1: Input divergence (MMD², Wasserstein)  ─┐
                                                                  ├─► Conjunction rule ──► Validity flag
Embeddings (+ logits) ──► Tier 2: Model response (Mahalanobis, energy) ─┘
```

- **Tier 1** tests whether the deployment features come from the same distribution as the reference. It uses a permutation two-sample test on MMD² and mean Wasserstein-1, at false-alarm rate α.
- **Tier 2** checks whether deployment embeddings sit unusually far from the reference (Mahalanobis distance), and optionally whether the classifier is unusually uncertain (energy score). Thresholds are calibrated on held-out reference data at false-alarm rate α.
- **The flag** is raised when both tiers flag. Use `require_both=False` to flag when either does.

α is the false-alarm rate of each tier separately. The combined flag has its own rate: lower with the AND rule, higher with OR.

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
# tile-level data: one slide id per row, one per line
wsidrift score --reference ref.csv --deployment dep.csv \
    --reference-groups ref_slides.txt --deployment-groups dep_slides.txt
```

Features can be `.npy` or comma-separated `.csv` (a header row is detected and skipped).

## What's implemented

| Component | State |
|---|---|
| Tier 1 (MMD², Wasserstein, permutation test) | Implemented; synthetic tests only |
| Tier 2 (Mahalanobis, energy, calibrated thresholds) | Implemented; synthetic tests only |
| Conjunction rule and `MonitorPipeline` | Implemented; synthetic tests only |
| Macenko and Reinhard normalization, ΔG and statistics helpers | Implemented; not validated |
| Link from divergence to predicted performance loss (ΔG) | Not implemented |
| E1–E9 model training | Not part of this package ([stainrob-benchmark](https://github.com/athakur569/stainrob-benchmark)) |

On synthetic Gaussian data at α = 0.05, the measured false-alarm rate was about 2.5% for Tier 1 and 5.4% for Tier 2, averaged over many reference sets. These numbers say the calibration works as designed; they say nothing about performance on pathology data.

## Versions and where each number comes from

| Version | Commit | Note |
|---|---|---|
| 0.1.0 (PyPI, tag `v0.1.0`) | `10d4e42` | The uploaded source matches this commit file for file. |
| 0.1.1 (PyPI, tag `v0.1.1`) | `b85875f` | The uploaded source matches this commit file for file. The commit titled "release: 0.1.1" (`4db4e95`) is four commits earlier. |
| 0.1.2.dev0 (`main`) | — | Adds `groups`, input guards and the fixes in the CHANGELOG. Not on PyPI. |

| Number | Where | First reported in | Script |
|---|---|---|---|
| False-alarm rate about 2.5% (Tier 1) and 5.4% (Tier 2), synthetic | above | `4db4e95` | none yet; being added under `experiments/` |
| Runtime about 13 s (1,000 × 200 samples, 2,048 features) | Limitations | `4db4e95` | none yet |
| Clustered simulation: Tier 1 100% → 6%, Tier 2 52% → 11% | CHANGELOG | `737321c` | none yet; being added under `experiments/` |
| CAMELYON17 no-shift check: 82% / 100% → 5% / 2% | Limitations, CHANGELOG | `cc29c11` | run in [stainrob-benchmark](https://github.com/athakur569/stainrob-benchmark); [record](https://github.com/athakur569/stainrob-benchmark/blob/c2a1d2706f1062fb76b7ea8e3db3639bdeb0f366/docs/research_log.md#iss-039--statistics--neighbouring-tiles-are-correlated); reproduction from its public `main` being checked |

Until a number has a script, treat it as reported, not reproducible from this repository.

## Limitations

- **Correlated samples (important).** Tiles from one slide resemble each other. Release 0.1.1 permutes and
  bootstraps single tiles, which treats them as independent: on real CAMELYON17 tiles with no shift at all
  (training slides split into two random halves), Tier 1 alarmed in 82% (pixel statistics) and 100%
  (DenseNet-121 features) of 40 splits at a nominal 10%. The main branch adds `groups` (one slide or patient id
  per sample) to `fit` and `score`; with slide groups the same check alarmed in 5% and 2%. Pass `groups` for
  any tile-level data. Not yet released on PyPI.
- **Bad input in 0.1.1.** The PyPI release does not check for NaN or infinite values, and one such value decides the
  result. In Tier 1 it gives a NaN statistic and the smallest possible p-value, so the batch is flagged whatever the data
  show. In Tier 2 it gives a NaN mean distance that never flags, so the AND rule stays silent even on a clear shift. With
  0.1.1, check your features with `numpy.isfinite` before calling `score`. `main` refuses such input and names the rows.
  Not yet released on PyPI.
- **The AND rule misses shifts that shrink the spread.** Tier 2 flags only when distances from the reference
  grow. If deployment embeddings are less spread out than the reference (embeddings pulled toward the reference
  mean, as blank or blurred tiles might produce), Tier 2 never flags, so the default AND flag never flags either,
  even when Tier 1 does. Synthetic check (64 dimensions, 500 reference samples, batches of 100, α = 0.05, 40 runs):
  spread × 0.8, × 0.5 and × 0.05 were flagged by Tier 1 in 100% of runs and by the AND rule in 0%; a mean shift of
  0.1 in every dimension in 100% and 12%. Read Tier 1 and Tier 2 separately, or use `require_both=False`. A
  design change is under review.
- **Small reference sets are refused.** Tier 2 needs at least 40 reference samples to calibrate its threshold
  out of sample; below that it false-alarmed in 44–100% of clean batches at a nominal 5%. `main` raises an error
  instead (pass `allow_small_reference=True` to override). Tier 1 refuses settings in which it could never flag:
  too few permutations for α, or too few clusters for a cluster permutation test.
- **Validation so far.** Tier 1 has been checked on real tiles under the null (above; [record](https://github.com/athakur569/stainrob-benchmark/blob/c2a1d2706f1062fb76b7ea8e3db3639bdeb0f366/docs/research_log.md#iss-039--statistics--neighbouring-tiles-are-correlated)); Tier 2 and the
  conjunction rule have been tested on simulated data only. Detection of real, harmful shifts is the subject
  of ongoing work and is not yet demonstrated.
- Assumes covariate shift: the input distribution changes while the relationship between inputs and labels stays the same. This is reasonable for stain and scanner variation, and doesn't hold for changes in label definitions.
- Tier 1 runs a permutation test, so it costs more on large batches: about 13 seconds for 1,000 reference and 200 deployment samples with 2,048-dimensional features.
- Tier 2's Mahalanobis statistic is less sensitive than Tier 1 to small shifts spread across many dimensions.

## License

MIT
