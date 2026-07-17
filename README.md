# Decoding design bias: reviewer reproduction

This repository contains an analysis-ready reproduction for
*Decoding the physicochemical basis of taxonomy preferences in protein design
models* (Dillon, Maiwald, and Crook).

The bundle starts from the row-level score, feature, design, and fine-tuning
tables used by the paper. It recomputes the main statistical results, regenerates
compact figures, and compares the numbers with committed reference outputs. Model
training, likelihood scoring, sequence generation, and structure prediction are
intentionally outside the reviewer path because they require large checkpoints or
external services and are not needed to audit the paper's analyses.

## Run it

With conda (includes R/mgcv for the full reproduction):

```bash
conda env create -f environment.yaml
conda activate decoding_bias
pip install -e .
decoding-bias reproduce
```

With an existing Python 3.10+ environment:

```bash
python -m pip install -e .
decoding-bias reproduce --quick
```

The quick run takes about one minute and verifies every Python analysis except the
seeded species-Elo calculation. The full run adds that calculation and the R/mgcv
GAM landscapes and takes about six minutes on a typical laptop. Both write a human-readable report to
`results/reviewer/reproduction_report.md`; a non-matching result exits with status 1.

## What is reproduced

| Analysis | Paper output | Quick | Full |
|---|---|:---:|:---:|
| Dataset composition | Methods / Table 7 | yes | yes |
| Variance decomposition and pLDDT control | Tables 1–2; SI controls | yes | yes |
| Species-level Elo | Figure 2; Table 3 | — | yes |
| Biophysical PCA | Figure 3A–B; Table S14 | yes | yes |
| GAM preference landscapes | Figure 3C; Table S15 | — | yes |
| Property importance | Tables S16–S18 | yes | yes |
| Designed-sequence shifts | Figure 4; Tables 4 and S21 | yes | yes |
| Fine-tuning surface steer and scTM | Figure 5; Tables 5–6 and S22; SI scTM | yes | yes |

Each stage can also be run alone, for example:

```bash
decoding-bias taxonomy --permutations 50
decoding-bias design
decoding-bias finetune
```

## Repository map

```text
data/                              analysis-ready row-level inputs
expected/                          manuscript-era numeric reference outputs
src/decoding_bias/analysis/        one compact module per paper analysis
src/decoding_bias/reproduce.py     runner and numeric comparisons
tests/                             smoke and end-to-end regression tests
```

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the exact analysis boundary and
[data/README.md](data/README.md) for input contracts and provenance. The independent
876-chain PDB cohort and its manuscript-era supplementary outputs are included as
audit material; they are not refit by the one-command reviewer workflow.

## License and citation

The code is MIT licensed. See [CITATION.cff](CITATION.cff) for citation metadata.
