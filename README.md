# Decoding design bias

Version 0.3.2 accompanies *Decoding the physicochemical basis of taxonomy preferences in protein design models* (Dillon, Maiwald and Crook). It contains the analysis code, the row-level inputs and the reference results for the paper's statistical analyses.

The workflow starts from model scores, measured features and design observations that have already been computed, and it recomputes every analysis on a CPU. Model training, likelihood scoring, sequence generation and structure prediction need checkpoints, GPUs or external services, and are not rerun.

## Install and reproduce

The conda environment includes Python and R/mgcv:

```bash
conda env create -f environment.yaml
conda activate decoding_bias
python -m pip install -e '.[test,figures]'
decoding-bias reproduce
```

With an existing Python 3.10+ environment:

```bash
python -m pip install -e '.[test,figures]'
decoding-bias reproduce --quick
python -m pytest
```

The full run also needs `Rscript` and the R package `mgcv` (`install.packages("mgcv")`). It takes about five minutes, most of it the Elo ratings. `--quick` skips the Elo ratings and the GAMs.

| Analysis | Paper output | Quick | Full |
|---|---|:---:|:---:|
| Dataset composition | Table S28 | yes | yes |
| Variance decomposition and pLDDT control | Table 1; Tables S6, S9, S24 | yes | yes |
| Biophysical PCA | Figure 3b-c; Table S14 | yes | yes |
| Property importance | Tables S16-S18 | yes | yes |
| Designed-sequence shifts | Figure 4c; Figure S9; Tables S20, S21, S25 | yes | yes |
| Fine-tuning and structural self-consistency | Figure 4a; Figures S10-S12; Tables S22, S23, S26, S27 | yes | yes |
| Species Elo | Figure 2; Table 2; Tables S10-S13 | no | yes |
| GAM preference landscapes | Figure 3a; Figure S7; Table S15 | no | yes |

Results and `reproduction_report.md` are written to `results/reproduction/`. The command exits with status 1 if any result differs from its reference. Each stage also runs on its own, for example `decoding-bias design`, `decoding-bias finetune` or `decoding-bias taxonomy`. The two pLDDT controls in Table S10 run with `decoding-bias taxonomy --plddt-control weighted` or `--plddt-control residualised`; they are not part of `reproduce`.

## Models

The 14 primary models, grouped by scoring context:

- backbone-conditioned: ProteinMPNN, SolubleMPNN, Caliby, SolubleCaliby, ESM-IF, TriFlow;
- structure plus native sequence: MIF, MIF-ST, ESM3 with structure;
- sequence-only: ESM3 without structure, ESM2-15B, CARP-640M, ProGen2-XL, ProtGPT2.

ProGen2-XL (`progen2_XL_score`) represents ProGen2 in the 14-model comparisons. The property-importance analysis also reports the base checkpoint (`progen2_score`). Fine-tuning compares ProteinMPNN v_48_020 with AlkSecMPNN and AcidSecMPNN, which are fine-tuned on alkaliphile and acidophile secretomes. The fine-tuning data also contain a v_48_002 arm, and Table S23 compares ProteinMPNN with a fine-tuned ESM2-35M on the same ten targets.

Weights are not redistributed. The third-party models are available from their providers:

- ProteinMPNN and SolubleMPNN: <https://github.com/dauparas/ProteinMPNN>
- ESM-IF and ESM2: <https://github.com/facebookresearch/esm>
- ESM3: <https://github.com/evolutionaryscale/esm>
- MIF, MIF-ST and CARP: <https://github.com/microsoft/protein-sequence-models>
- ProGen2: <https://github.com/salesforce/progen>
- ProtGPT2: <https://huggingface.co/nferruz/ProtGPT2>

Caliby, SolubleCaliby and TriFlow are cited in the paper. Third-party models keep their own licences.

## Data and conventions

[data/README.md](data/README.md) lists the inputs and their statistical units. `data/SHA256SUMS` and `expected/SHA256SUMS` record the file hashes.

The natural-protein dataset has 10,146 proteins from 495 species and 281 families, after P19949 and Q8TFG7 are excluded. The 14-model variance decomposition uses the 9,963 proteins with scores from all 14 models and complete features.

The design analysis uses the 12 properties that vary between a design and its template; sequence length and surface exposure are fixed by the backbone. Each of the 25 templates is one unit, represented by the mean of its eight designs. Tests are two-sided Wilcoxon signed-rank tests with the normal approximation and continuity correction, with Benjamini-Hochberg correction across all 84 model-property comparisons. `scripts/verify_design_statistics.R` recomputes these tests in R and compares them with `expected/design/table_s20_statistics_R.csv`.

Comparisons use absolute and relative tolerances of 1e-8 and 1e-6, except the design tests (1e-12 and 1e-9) and GAM deviance (0.01 percentage points).

The Python PCA scales features by the population standard deviation and the R PCA behind the GAMs and figures by the sample standard deviation. Their loadings and explained variance are identical; coordinates differ by a constant factor.

The analyses are deterministic apart from the Elo ratings, which use seed 42, K = 32, a baseline of 1,500 and 50 family-order permutations. GAMs are fitted with `mgcv::gam(..., method = "REML")` and basis size `max(12, min(50, floor(n/300)))`.

## Figures from the Source Data

These commands redraw the paper's figures from the Source Data file, without refitting the PCA or the GAMs. The Source Data are only read; each command writes its figures and the tables it plots to its output directory.

```bash
python scripts/reconstruct_figure3.py --source-data /path/to/Source_Data.zip --output-dir results/figure3
python scripts/render_main_figures.py --source-data /path/to/Source_Data --output-dir results/main_figures
python scripts/render_remaining_supplementary.py --source-data /path/to/Source_Data --output-dir results/supplementary_figures
python scripts/render_revision_figures.py --source-data /path/to/Source_Data --render-assets /path/to/figure_assets --output-dir results/figure4
```

`reconstruct_figure3.py` accepts the ZIP or its extracted folder. It checks the 10,146 coordinates, ten landmarks, eight loading vectors and eight 25,600-cell GAM grids before drawing; `Source_Data/Figure_3_reconstruction_README.txt` describes the panels. `render_revision_figures.py` draws Figure 4 and Figures S9-S11, and needs the author's structure renders as well.

To regenerate all 19 GAM grids (the 17 scores in the full reproduction and the two fine-tuned-minus-base differences):

```bash
Rscript scripts/export_gam_grids.R data/main_analysis.csv results/gam_grids
```

An optional third argument selects comma-separated score columns.

## Experimental-PDB control and cohort sensitivity

The experimental-PDB control (Figure S3; Tables S7-S8) compares 12 models on the 876-chain PDB cohort with size- and domain-matched AlphaFold DB samples. The fits are rank-aware, and an R script refits every regression independently. SolubleCaliby is omitted because its PDB scores duplicate Caliby's. The control is not part of `reproduce`:

```bash
python scripts/reanalyse_pdb_control.py --main-input data/main_analysis.csv --pdb-input data/pdb/cohort_pdb_scored.csv --output-dir results/pdb_control
Rscript scripts/verify_pdb_control.R results/pdb_control data/main_analysis.csv
python scripts/render_pdb_control.py --analysis-dir results/pdb_control --output-dir results/pdb_control/figure
```

`verify_cohort_sensitivity.py` repeats the variance decomposition on the primary cohort, without ribosomal proteins and on a domain-balanced sample. It compares each result with `expected/robustness/`. The balanced sample of 1,353 proteins per domain was drawn once in R (seed 0) and is fixed in `data/robustness/domain_balanced_membership.csv`.

## Scope and limitations

The main cohort comes from reviewed UniProt/Swiss-Prot entries with AlphaFold DB structures and inherits their sampling and annotation biases. Model scores are likelihood-based preferences, not measurements of fitness or function, and the designs are evaluated by predicted structure only.

A successful reproduction shows that the deposited inputs give the reference results. It does not repeat model training, scoring, sequence generation, database queries or experimental testing. After setup, it needs no network, credentials, model weights or GPU.

The two Colab notebooks score the PDB cohort with ProGen2-XL and ESM3 on a GPU. They are not part of the CPU reproduction and need their own environments.

## Licence and citation

The code is MIT licensed. [CITATION.cff](CITATION.cff) gives the software citation and lists the associated paper (doi:10.1101/2025.10.21.683350).
