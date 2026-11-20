# Input data

`SHA256SUMS` records the hash of every file listed here.

| File | Rows | One row is |
|---|---:|---|
| `main_analysis.csv` | 10,146 | a natural protein: model scores, family and species labels, biophysical features and AlphaFold pLDDT |
| `taxonomy.csv` | 495 | a species: domain, phylum or division, and class |
| `design/designs_features.csv` | 2,400 | a generated design and its sequence and structure features |
| `design/wt_features.csv` | 25 | a wild-type design template |
| `design/functional_residue_recovery.csv` | 1,080 | a design's recovery at functional and background positions |
| `finetune/design_surface_features.csv` | 1,225 | a base or fine-tuned design, or a wild type, with its surface features |
| `finetune/surface_shift_matched.csv` | 20 | a target in the matched surface-only redesign (Table S23) |
| `finetune/self_consistency.csv` | 625 | a refolded design: scTM, scRMSD and pLDDT |
| `pdb/cohort_pdb_scored.csv` | 876 | an experimental PDB chain in the PDB control: features, labels and scores |
| `robustness/domain_balanced_membership.csv` | 4,059 | a protein in the fixed domain-balanced sample |

`main_analysis.csv` is identical to Supplementary Data 1. In it, `ProteinMPNN_v020_score` is an exact copy of `proteinmpnn_score` (the v_48_020 checkpoint) and serves as the fine-tuning base in Table S18. The `deepstabp_tm` column is not used.
