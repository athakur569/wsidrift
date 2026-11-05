# Experiments

One script per synthetic number quoted in the README and CHANGELOG. Each fixes its seeds and settings in the file and
prints the number. Where a script did not reproduce a number first reported elsewhere, the document was corrected, not
the script. Run from the repository root, with the package installed or `src` on the path:

```bash
python experiments/null_false_alarms.py
```

| Script | What it measures | Result (5 Nov 2026) | Quoted in |
|---|---|---|---|
| `null_false_alarms.py` | Tier 1 and Tier 2 false alarms with no shift; 16 dimensions, 1,000 reference sets of 400, batches of 100, α = 0.05 | Tier 1 3.6%, Tier 2 4.8% (SE 0.6–0.7%) | README, What's implemented |
| `tier2_reference_size.py` | Tier 2 false alarms by reference size (20–200) in 5 and 16 dimensions, batches of 50, α = 0.05 | below 40: 60–79% (5 dimensions), 100% (16); 40: 7.6–7.7%; 100: 5.9–6.2%; 200: 4.6–5.3% | README, CHANGELOG |
| `combined_false_alarms.py` | The combined flag with no shift, AND against OR; 16 and 64 dimensions, references of 500, batches of 100 | AND 0.2% and 0.8%, OR 7.8% and 11.0%; Tier 2 alone 4.2% and 8.2% | README, How it works |
| `and_rule_power.py` | Detection when one tier is blind: mean shift and shrinking spread; 64 dimensions, reference 500, batches of 100, 200 runs | spread × 0.8, × 0.5, × 0.01: Tier 1 100%, Tier 2 0%, AND 0%; mean + 0.1: Tier 1 97%, AND 14%. A prototype two-sided Tier 2 (not in the package) flags the spread changes in 100% with 6% false alarms | README, Limitations |
| `clustered_false_alarms.py` | Tile-level against cluster-level calibration with strong slide effects and no shift; α = 0.10, 200 deployments, 40 reference slides | Tier 1 100% → 6%, Tier 2 52% → 11% | CHANGELOG |
| `tier1_runtime.py` | Tier 1's run time on 1,000 × 200 samples with 2,048 features | machine-dependent; prints the processor | README, Limitations |

The CAMELYON17 no-shift check on real tiles (82% and 100% tile-level, 5% and 2% with slide groups) needs the study's
tile features, so its script lives in the research repository:
[stainrob-benchmark `scripts/null_check_tile_vs_slide.py`](https://github.com/athakur569/stainrob-benchmark/blob/9a8cf62b4eb07b4b5318593b365e2328de19e1d0/scripts/null_check_tile_vs_slide.py).
