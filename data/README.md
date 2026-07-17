# Files in data

All files are analysis-ready CSVs. `SHA256SUMS` records their exact byte-level identities.

| File | Rows | Statistical unit / purpose |
|---|---:|---|
| `main_analysis.csv` | 10,148 | one natural protein; scores, family/species labels, 14 biophysical features |
| `taxonomy.csv` | 495 | one species; domain, phylum/division, and class |
| `design/designs_features.csv` | 2,400 | one generated design; derived sequence/structure features |
| `design/wt_features.csv` | 25 | one matched wild-type template |
| `design/functional_residue_recovery.csv` | 1,080 | design-level functional-site and background recovery |
| `finetune/design_surface_features.csv` | 1,225 | one base/fine-tuned design or WT; surface features, with sequences removed |
| `finetune/surface_shift_matched.csv` | 20 | matched fine-tuning/control observations used in Table S22 |
| `finetune/self_consistency.csv` | 625 | refold-to-input scTM, scRMSD, and pLDDT for base, fine-tuned, and WT controls |
| `pdb/cohort_pdb_scored.csv` | 876 | audit-only experimental-PDB cohort underlying the saved supplementary outputs |
| `gam/*.csv` | small | deposited R cross-checks used by PCA/GAM reporting |
