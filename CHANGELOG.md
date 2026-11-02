# Changelog

## Unreleased

### Added
- `groups` (cluster id per sample, e.g. slide or patient) for `Tier1Monitor.fit/score` and
  `Tier2Monitor.fit/score`. Tiles of one slide are correlated; tile-level permutation and bootstrap
  treat them as independent and false-alarm far above `alpha`. With groups, Tier 1 permutes whole
  clusters, and Tier 2 cross-fits by cluster and bootstraps its batch threshold over clusters.
  Simulation with strong slide effects and no real shift, alpha = 0.10 (200 deployments, 40 reference
  slides): Tier 1 false alarms 100% tile-level vs 6% cluster-level; Tier 2 52% vs 11%.
  Calls without `groups` behave as before.
  Real-data check (CAMELYON17 training slides split into random halves, no shift, 40 splits, nominal 10%):
  Tier 1 on pixel statistics 82% tile-level vs 5% with slide groups; on DenseNet-121 features 100% vs 2%.

## 0.1.1 — pre-alpha

### Fixed
- `MonitorPipeline` crashed on every call: it passed argument names the monitors don't accept and used a `ValidityMonitor` API that doesn't exist. Rebuilt on the current classes.
- Energy score flagged the wrong tail. Out-of-distribution inputs have **higher** energy (Liu et al. 2020); the flag now fires when batch energy exceeds the threshold. A fixed `energy_threshold` now means "flag above".
- `wsidrift.norm.reinhard` had a syntax error, so `import wsidrift.norm` failed (also in 0.1.0).
- Macenko normalization estimated stain vectors from the wrong data (angles of the SVD basis
  vectors instead of the projected tissue pixels), giving arbitrary stain matrices. It now
  follows Macenko et al. (2009): pixel angles, SVD sign fix, 99th-percentile concentrations.
  On synthetic two-stain tiles it recovers the true H&E vectors (cosine > 0.99).
- Tests now import every module, so a broken module can't ship unnoticed.
- Tier 2 no longer invents all-zero logits when none are given. Energy is only used when reference logits were supplied at `fit()`, or when a fixed energy threshold is set.

### Changed
- **Thresholds are calibrated by default** at false-alarm rate `alpha` (default 0.05), replacing fixed guesses that false-alarmed on clean data:
  - Tier 1: permutation two-sample test on MMD² and Wasserstein, α/2 each. Results include `p_mmd` and `p_wass`.
  - Tier 2: out-of-sample (2-fold cross-fitted) reference Mahalanobis distances, and a bootstrap threshold per batch size that includes reference sampling error.
  - Fixed thresholds still work when passed explicitly. Tier 1 requires both or neither.
- `MonitorPipeline.fit/score` take Tier 1 features and, optionally, separate Tier 2 embeddings and logits.
- Wasserstein-1 for unequal batch sizes is now exact and vectorized (it was a per-feature Python loop).
- PyTorch is imported only by `Tier2Monitor.from_encoder`, not by the monitors themselves.
- Development status set to Pre-Alpha. README and architecture doc state what is and isn't implemented.

### Removed
- Unfinished or unused modules: `wsidrift.cyclegan` (stub), `wsidrift.data` (dataset loaders),
  `wsidrift.adapt` (DANN). None was used by the monitor or covered by tests; model training
  belongs in the research repository.
- Dependencies the monitor does not use (torchvision, pytorch-lightning, pandas, openslide,
  OpenCV, Pillow, omegaconf, tqdm). Core install is now numpy, scipy and scikit-learn;
  `pip install wsidrift[torch]` adds PyTorch for `Tier2Monitor.from_encoder`.

### Repository only (not in the package)
- `scripts/run_ablation.py` returned random AUCs and saved them as ΔG results. Removed.
- `scripts/monitor_dashboard.py` used the same broken API; rebuilt on `MonitorPipeline`. Its action text no longer claims a predicted ΔG or concept drift.

## 0.1.0

Initial upload.
