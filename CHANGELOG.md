# Changes

## 0.3.2 (20 November 2026)

- Table and figure numbers in the code, command help and reproduction report now match the submitted paper.
- The figure scripts only read the Source Data and write their plotted tables to the output directory. The unpublished Figure 1 drawing is removed.
- The ProGen2 notebook's summary cell reads the columns the run writes. Both notebooks describe the PDB cohort they score and no longer keep outputs or widget state.
- One model registry replaces five. Unused files, duplicate checks and leftover test scaffolding are removed. The design p-values are checked once, at the stricter tolerance, and the GAM tolerance is 0.01 percentage points.
- The Table S10 pLDDT controls run from the command line.
- MODEL_CARD.md and the separate data notes are merged into the two READMEs.
- Numerical inputs and reference results are unchanged.

## 0.3.1 (20 November 2026)

- Tests regrouped into five files; README, reproducibility notes and R setup merged into one README.
- The two Figure 3 commands share one renderer, and GAM reproduction and grid export share one R implementation.

## 0.3.0

- ProGen2-XL is the primary ProGen2 checkpoint, with the base checkpoint kept as a sensitivity; P19949 and Q8TFG7 are excluded.
- Design tests follow the R Wilcoxon convention with one Benjamini-Hochberg family of 84 comparisons, checked against an R reference.
- The experimental-PDB control uses rank-aware fits with matched AlphaFold DB samples and an independent R check.
- Figure 3 reconstruction, GAM grid export and figure scripts added.
