# Deposited R/mgcv PCA-GAM outputs

These come from the R notebook `notebooks/04_pca_gam/PCA_paper_figures.ipynb`
(mgcv GAM), which cannot be re-run from Python. They are deposited so the repo is
self-contained for the R-only supplementary items.

Source: `~/Downloads/pca_paper_outputs_21_06_26/pca_outputs/` (the current version;
it adds CARP-640M to the GAM table relative to the older `_20_06_2026` folder the
supplementary text still cites - update that citation).

- `gam_deviance.csv` - **Table S15**: GAM deviance explained (% of each model's score
  variance captured on the 14-feature PC1-PC2 plane). Column names use the pre-rename
  `AlkalineMPNN_020`/`AcidophileMPNN_020` (= `AlkSecMPNN_020`/`AcidSecMPNN_020`).
- `compactness_R.csv` - the R notebook's per-domain Bhattacharyya overlaps, used to
  cross-check the Python `pca` stage (they match to 1e-3: overlaps 0.837 / 0.907 / 0.945).

Cross-validation (see `decoding-bias verify`): the Python `pca` stage reproduces the R
notebook's PCA exactly - PCA coordinates correlate at 1.000000 on both PCs across all
10,148 proteins, and the compactness overlaps are identical. So Fig 3A/B and Table S14
are confirmed by two independent implementations; the GAM landscapes (Fig 3C, Fig S8) and
this GAM deviance table are the R notebook's alone.
