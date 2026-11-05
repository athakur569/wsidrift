# Changelog

## Unreleased (0.1.2.dev0)

### Fixed (external review, 5 November 2026)
- **Tier 2 below 40 reference samples** set its threshold on in-sample distances and false-alarmed in 60–100% of
  clean batches at a nominal 5% (60–79% in 5 dimensions with 20–30 samples, 100% in 16;
  `experiments/tier2_reference_size.py`), with only a log message. `fit` now raises,
  unless `allow_small_reference=True` (or `t2_allow_small_reference=True` on `MonitorPipeline`) or a fixed
  `maha_threshold` is given.
- **Settings where Tier 1 could never flag** now raise: α/2 not above 1/(n_permutations + 1) (e.g. α = 0.005 with
  200 permutations, which stayed silent on a 50σ shift), and cluster tests with so few clusters that the p-value
  cannot fall below α/2 (2 slides per side allow 3 distinct splits).
- **Tier 2 with one reference cluster** returned a NaN distance that never flags. It now raises.
- **Logits with a different row count** from the embeddings were accepted. Tier 2 now raises.
- **The CLI crashed on `.csv` input**: it imported pandas, which is not a dependency, and would have dropped the first
  row as a header. CSV is now read with numpy; a header row is detected.
- `guards.probabilities` named the wrong row for 2-D input.
- Macenko normalization: a blank tile crashed with an IndexError, and 0–1 float images gave meaningless output.
  Both now raise a clear error.
- A batch of 1 gave Tier 1 a misleading "statistic is NaN" error; it now says each batch needs 2 samples.
- **`MonitorPipeline.score` accepted inputs that differ from `fit`.** Fitted with `reference_embeddings` but scored
  without `embeddings`, Tier 2 scored the features against the embedding model (silently, if the widths matched);
  reference logits at `fit` but none at `score` dropped the energy check without a word. `score` now refuses any
  difference in embeddings, logits or groups and says which. A `ValidityMonitor` with a Tier 2 monitor now refuses a
  call without embeddings instead of quietly deciding on Tier 1 alone.
- **README's synthetic false-alarm rates corrected to what their script reproduces:** 3.6% (Tier 1) and 4.8%
  (Tier 2) over 1,000 reference sets, not 2.5% and 5.4%, whose settings were not recorded.

### Changed
- Tier 2's bootstrap runs in chunks: memory no longer grows with reference size × `n_bootstrap` (about 0.64 GB at
  20,000 reference tiles). Results are identical.
- Tier 2 logs a warning when it falls back to per-dimension variances (half-reference size ≤ dimensions + 1).
- `Tier1Monitor.save` writes the settings and the fitted reference to `.npz`; the new `Tier1Monitor.load` restores a
  monitor that scores identically. The old JSON file held no reference and could not be loaded.
- `DeltaGap` reports the sample standard deviation across seeds (ddof = 1; NaN for one seed), not the population one.
- `bonferroni_correct`: docstring and return type now say it returns a decision per p-value.

### Added
- `wsidrift score --reference-groups / --deployment-groups`: one cluster id per row.
- `groups` through `MonitorPipeline.fit/score` and `ValidityMonitor.score`, to both tiers (the features and
  embeddings must then have the same rows). The CLI passes its groups through `ValidityMonitor`.
- `experiments/`: one script per synthetic number in this file and the README, with fixed seeds and settings (see
  `experiments/README.md`).
- Tags `v0.1.0` and `v0.1.1` at the commits whose source matches the PyPI uploads; README table of which commit
  first reported each number; pinned test environment, CI workflow and `CITATION.cff`.

### Fixed (earlier)
- **A NaN read as drift.** In 0.1.1, one NaN feature gave Tier 1 a NaN statistic; every comparison with NaN is false, so
  no permuted value exceeded it and the p-value came out as the smallest possible. A failed slide looked like the
  strongest signal there is. Tier 2 failed the other way: a NaN mean distance never flags, so the AND rule stayed silent
  even on a clear shift. (In 0.1.0, with fixed thresholds, a NaN could never alarm.) Tier 1 and Tier 2 now refuse NaN or
  infinite input and name the rows (`wsidrift.guards`). They stop rather than drop them, because a run of failed slides
  may be what needs attention.

### Added
- `groups` (cluster id per sample, e.g. slide or patient) for `Tier1Monitor.fit/score` and
  `Tier2Monitor.fit/score`. Tiles of one slide are correlated; tile-level permutation and bootstrap
  treat them as independent and false-alarm far above `alpha`. With groups, Tier 1 permutes whole
  clusters, and Tier 2 cross-fits by cluster and bootstraps its batch threshold over clusters.
  Simulation with strong slide effects and no real shift, alpha = 0.10 (200 deployments, 40 reference
  slides): Tier 1 false alarms 100% tile-level vs 6% cluster-level; Tier 2 52% vs 11%.
  Calls without `groups` behave as before.
  Real-data check (CAMELYON17 training slides split into random halves, no shift, 40 splits, nominal 10%):
  Tier 1 on pixel statistics 82% tile-level vs 5% with slide groups; on DenseNet-121 features 100% vs 2%
  (stainrob-benchmark `scripts/null_check_tile_vs_slide.py`).

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
