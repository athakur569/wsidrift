# wsidrift

A label-free, model-agnostic validity monitor for deployed computational pathology pipelines.

Detects when a WSI processing pipeline has drifted from its validated reference — without requiring labels or retraining.

## Architecture

```
Input WSIs ──► Feature Extractor ──► Tier 1: Input Divergence (MMD/Wasserstein)
                                  └──► Tier 2: Embedding Monitor (Mahalanobis/Energy)
                                              │
                                    Conjunction Rule ──► Validity Flag
```

- **Tier 1** — covariate shift in stain/feature distributions (label-free, assumption-light)
- **Tier 2** — model-response drift in encoder embedding space (closer to correctness)
- **Flag** — raised only when *both* tiers move in the direction that historically predicted a large generalization gap (ΔG)

## Status

Private research repository. Part of a three-paper dissertation.

## License
MIT
