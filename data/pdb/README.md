# Independent experimental-PDB cohort

`cohort_pdb_scored.csv` is the analysis-ready 876-chain replication cohort from
`design/outputs/independent_cohort/`. It contains one unique PDB chain per row, the 14
analysis features, family/species labels, resolution metadata, and available model
scores. Structure files and sequence strings are not duplicated in this reviewer bundle.

Coverage is 876 for most sequence/context models, 847 for ESM-IF, 870 for Caliby,
875 for ProteinMPNN, and 562 for MIF/MIF-ST. The original June variance outputs are
preserved under `expected/pdb/*_original.csv`.

This cohort is provided for auditing rather than as a strict cross-platform regression
test. The historical code formed a full QR basis from rank-deficient family/species
dummy matrices, so the combined fit can depend on the numerical environment. The
saved files are the manuscript values and are intentionally not refit or replaced by
the reviewer command.
