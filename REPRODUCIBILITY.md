# Reproducibility scope

## Reviewer boundary

The default reproduction begins at the analysis-ready tables in `data/`. This is the
appropriate boundary for checking the paper's quantitative claims: the tables retain
individual proteins, species, families, models, designed sequences' derived features,
and matched observations, so aggregation choices, effect sizes, statistical tests,
and figures are all rerun rather than replayed from final summaries.

The fine-tuning check includes 625 row-level refold observations and reproduces the
reported mean scTM values (0.54 base, 0.51 AlkSecMPNN, 0.45 AcidSecMPNN, 0.38 WT).
DeepStabP melting-temperature shifts are intentionally outside the current bundle.

Upstream model scoring and sequence generation are not part of this branch. Repeating
those stages would require heterogeneous model checkpoints, GPU runtimes, AlphaFold or
ColabFold, DeepStabP, UniProt snapshots, and RCSB downloads. Those stages add substantial
compute and service drift but do not help a reviewer inspect the analysis code. The
upstream implementation remains available in the repository's Git history.

## Commands

```bash
# Python analyses; skips Elo and R/mgcv
decoding-bias reproduce --quick

# All reviewer analyses
decoding-bias reproduce

# Test the installed package and the quick reproduction
python -m pytest
```

Results are written below `results/reviewer/`. Every numeric comparison is keyed by
scientific identifiers rather than row order. The report records pass/fail status,
runtime, and maximum numerical difference. Reference tolerances are `1e-8` absolute
and `1e-6` relative unless a stage documents a model-specific tolerance.

## Determinism

- Variance decomposition, PCA, importance, design, and fine-tuning are deterministic.
- Species Elo uses seed 42, K=32, a 1500 baseline, and 50 seeded family-order
  permutations.
- The R GAM uses `mgcv::gam(..., method="REML")`. Small version-dependent differences
  are accepted up to 0.11 percentage points of deviance explained on stable model
  mappings.

The 876-chain experimental-PDB input is deposited under `data/pdb/`, with the exact
manuscript-era supplementary tables under `expected/pdb/`. This cohort is audit-only
and is not a PASS item in the cross-platform report: the historical combined
family/species QR fit used a rank-deficient design whose result can vary with the
numerical environment. The saved supplementary numbers are left unchanged.

## Environment

The Python runtime needs pandas, NumPy, SciPy, statsmodels, and Matplotlib. The full
run additionally needs `Rscript` and the R package `mgcv`. `environment.yaml` pins a
portable conda environment. No network, credentials, model weights, or GPU are needed
after the repository has been cloned.

## What a successful run means

A PASS establishes that the deposited row-level inputs reproduce the committed
manuscript-era outputs through the compact analysis code. It does not independently
repeat model training, likelihood scoring, structure prediction, or external database
queries. Those upstream artifacts should be deposited separately if the journal asks
for a full raw-to-result rerun.
