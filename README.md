# Decoding design bias

Analysis code for *Decoding the physicochemical basis of taxonomy preferences in
protein design models* (Dillon, Maiwald, and Crook). The repository is a single
installable package with one command-line interface.

## Installation

```bash
conda env create -f environment.yaml
conda activate decoding_bias
pip install -e .
decoding-bias verify
```

`decoding-bias verify` re-runs each stage that is reproducible from the deposited data
and compares the result against the reference values in `tests/reference/`.

## Command-line interface

The pipeline is a single command with one subcommand per scientific stage. Each stage
writes to `results/<stage>/`.

| Command | Stage | Paper outputs | Requirements |
|---|---|---|---|
| `decoding-bias dataset`    | Analysis-table composition | Table 7 | Deposited data |
| `decoding-bias variance`   | Score-variance decomposition | Tables 1-2; SI Figs S2-S5; Tables S9, S10, S13 | Deposited data |
| `decoding-bias taxonomy`   | Species Elo taxonomy preference | Figure 2; Table 3; SI Tables S5-S8 | Deposited data (Fig 2B panel requires the taxonomy metadata) |
| `decoding-bias pca`        | Biophysical PCA and tables | Figure 3A-B; Table S14; compactness; Table S15 | Deposited data (GAM landscapes, Fig 3C, require R and mgcv) |
| `decoding-bias importance` | Property-to-score importance | SI Fig S9; Tables S16-S18 | Deposited data |
| `decoding-bias finetune`   | Fine-tuning surface steer | Table S22 | Deposited data (full arms require fine-tuned weights) |
| `decoding-bias design`     | Designed-sequence analysis | Figure 4; Tables 4, S20, S21 | Design feature tables (`design_dir`) |
| `decoding-bias pdb-cohort` | Experimental-PDB cohort | SI Figs S6-S7; Tables S11-S12 | RCSB structures |
| `decoding-bias score`      | Model likelihood scoring | Score columns | Model weights |
| `decoding-bias figures`    | Status of every paper output | - | - |
| `decoding-bias verify`     | Regression check against `tests/reference/` | - | Deposited data |

Run `decoding-bias <stage> --help` for options, for example `taxonomy --all-variants`,
`variance --no-figures`, or `verify --full` (which includes the Elo stage).

Stages whose requirements are not part of the deposit (model weights, AlphaFold or
RCSB structures, or the design feature tables) report the exact input they require and
the command they run once it is provided. External inputs
are supplied through `paths.*` in a configuration file or the corresponding
`DECODING_BIAS_*` environment variables.

## Repository layout

```
src/decoding_bias/                    installable package (import decoding_bias)
  catalog.py                          scientific invariants: feature set, model panels, domains
  config.py, config_default.yaml      path and parameter resolution
  data.py                             analysis-table loading and composition
  cli.py                              the decoding-bias command
  manifest.py                         map from each paper output to its stage and command
  verify.py                           regression harness against tests/reference/
  plotting.py                         shared figure code
  analysis/                           reproducible stages: variance, importance, elo, pca,
                                      design, finetune (with elo_rating/elo_figures)
  gam/                                R and mgcv GAM landscapes and design ellipses
  dataset/ scoring/ design/           stage code that requires external inputs
  finetune/ pdb_cohort/ stages/
  features/                           sequence, structural, and surface feature extraction
00_data/data/decoding_bias_15_07_26.csv   the deposited analysis table
00_data/finetune/, 00_data/pca_gam/       small deposited outputs used by the finetune and pca stages
notebooks/                            environment-bound notebooks (scoring, PCA/GAM in R, design, fine-tuning)
tests/                                reference values and equivalence tests
```

The paper-output map is `src/decoding_bias/manifest.py`; `decoding-bias figures` prints it
as a status table.

## Data

The deposited analysis table `00_data/data/decoding_bias_15_07_26.csv` (10,148 proteins;
the fourteen-model cohort and the fine-tuned `-020` models, the sixteen
biophysical features, `avg_plddt`, and `deepstabp_tm`) is the main source for every
reported value. Larger metadata, protein structures, model weights, and folding outputs
are provided through the data-availability statement and configured through `paths.*`.

- R and mgcv are required for the GAM landscapes (Fig 3C) and GAM deviance (Table S15); see
  [`environment-R.md`](environment-R.md).
- Model weights, ColabFold, DeepStabP, and the RCSB API are external services; see
  [`external-services.md`](external-services.md).

## Reproducing the results

```bash
decoding-bias figures            # status of every paper output
decoding-bias figures --run      # dataset, variance, importance, and pca into results/
decoding-bias taxonomy           # Figure 2 and Table 3 (seeded permutation Elo)
decoding-bias verify --full      # full regression check, including Elo
```

## License

MIT; see [`LICENSE`](LICENSE).
