# Experiments

One script per synthetic number quoted in the README and CHANGELOG. Each fixes its seeds and settings in the file and
prints the number. Where a script did not reproduce a number first reported elsewhere, the document was corrected, not
the script. Run from the repository root, with the package installed or `src` on the path:

```bash
python experiments/null_false_alarms.py
```

| Script | What it measures | Result (6 Nov 2026, two-sided Tier 2) | Quoted in |
|---|---|---|---|
| `null_false_alarms.py` | Tier 1 and Tier 2 false alarms with no shift; 16 dimensions, 1,000 reference sets of 400, batches of 100, α = 0.05 | Tier 1 3.6%, Tier 2 6.0% (SE 0.6–0.8%; one-sided Tier 2 4.8%) | README, What's implemented |
| `tier2_reference_size.py` | Tier 2 false alarms by reference size (20–200) in 5 and 16 dimensions, batches of 50, α = 0.05 | below 40: 50–71% (5 dimensions), 100% (16); 40: 6.6–7.3%; 100: 5.3–5.9%; 200: 4.5–5.5% | README, CHANGELOG |
| `combined_false_alarms.py` | The combined flag with no shift, AND against OR; 16 and 64 dimensions, references of 500, batches of 100 | AND 0.2% and 0.5%, OR 8.0% and 9.5%; Tier 2 alone 4.5% and 6.5% | README, How it works |
| `and_rule_power.py` | Detection when a shift moves one tier more than the other: mean shift and shrinking spread; 64 dimensions, reference 500, batches of 100, 200 runs | spread × 0.8, × 0.5, × 0.01: Tier 1, Tier 2 and AND 100% (AND under 0.1.1's one-sided Tier 2: 0%); mean + 0.1: Tier 1 97%, AND 12%; no shift: AND 1%, OR 10% | README, Limitations, CHANGELOG |
| `clustered_false_alarms.py` | Tile-level against cluster-level calibration with strong slide effects and no shift; α = 0.10, 200 deployments, 40 reference slides | Tier 1 100% → 6%, Tier 2 64% → 6% (one-sided, as first reported: 52% → 11%) | CHANGELOG |
| `tier1_runtime.py` | Tier 1's run time on 1,000 × 200 samples with 2,048 features | 13.4 s on an Apple-silicon Mac (5 Nov 2026); machine-dependent, prints the processor | README, Limitations |

The CAMELYON17 no-shift check on real tiles (82% and 100% tile-level, 5% and 2% with slide groups) needs the study's
tile features, so its script lives in the research repository:
[stainrob-benchmark `scripts/null_check_tile_vs_slide.py`](https://github.com/athakur569/stainrob-benchmark/blob/9a8cf62b4eb07b4b5318593b365e2328de19e1d0/scripts/null_check_tile_vs_slide.py).
