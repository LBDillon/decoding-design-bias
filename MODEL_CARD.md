# Model card and study scope

## Overview

This repository supports *Decoding the physicochemical basis of taxonomy
preferences in protein design models*. The study evaluates systematic taxonomic
and physicochemical preferences in pretrained protein-design and protein-language
models. It also examines continued training of ProteinMPNN on
extremophile-secretome cohorts.

This is a study-level model card, not a new foundation-model release. The public
reviewer workflow starts from deposited, analysis-ready row-level tables. It does
not redistribute third-party model weights or the study's fine-tuned checkpoints,
and it does not rerun upstream likelihood scoring, sequence generation, or
structure prediction.

## Models evaluated

The main 14-model cohort comprises:

- backbone-conditioned models: ProteinMPNN, SolubleMPNN, ESM-IF, Caliby,
  SolubleCaliby, TriFlow;
- structure plus native-sequence-context models: MIF, MIF-ST, ESM3-struct;
- sequence-only models: ESM3-seq, ESM2-15B, CARP-640M, ProGen2, ProtGPT2.

The property-importance analysis additionally includes ProGen2-XL. The
fine-tuning analysis compares ProteinMPNN v_48_020 with AlkSecMPNN-020 and
AcidSecMPNN-020.

The third-party models and their weights remain available from their original
providers. Principal sources include:

- ProteinMPNN and SolubleMPNN:
  <https://github.com/dauparas/ProteinMPNN>
- ESM-IF and ESM2:
  <https://github.com/facebookresearch/esm>
- ESM3:
  <https://github.com/evolutionaryscale/esm>
- MIF, MIF-ST, and CARP:
  <https://github.com/microsoft/protein-sequence-models>
- ProGen2:
  <https://github.com/salesforce/progen>
- ProtGPT2:
  <https://huggingface.co/nferruz/ProtGPT2>

Caliby, SolubleCaliby, and TriFlow are described and cited in the manuscript.
Their derived scores are included in the deposited analysis table; this
repository does not redistribute their weights.

## Intended use

The deposited workflow is intended to:

- reproduce the paper's statistical analyses and compact figures;
- audit dataset composition, variance decomposition, taxonomic rankings,
  biophysical preferences, design shifts, and fine-tuning outcomes;
- compare regenerated numerical results with committed manuscript-era
  reference outputs.

It is not intended to rank protein fitness, certify designed sequences as safe
or functional, or replace experimental validation.

## Data and splits

The main evaluation cohort contains 10,148 proteins from 495 species and 281
protein families. Fine-tuning used 40%-sequence-identity cluster-disjoint
70/15/15 train/validation/test splits. The cluster boundary reduces homology
leakage and evaluates transfer to held-out proteins. The public reviewer bundle
contains the row-level analysis inputs and held-out outcome tables needed to
reproduce the reported analyses; it does not contain the fine-tuning sequences
or checkpoints.

## Evaluation

The study uses complementary analyses rather than one predictive benchmark:
variance decomposition, species-level Elo ratings, PCA and GAM preference
landscapes, property-importance analysis, generated-sequence shifts,
functional-residue recovery, and refold-to-input self-consistency. Robustness
checks include seeded Elo permutations, removal of ribosomal families,
domain-balanced subsampling, pLDDT residualisation, and an independent
876-chain experimental-PDB cohort.

## Limitations

- Model scores are preferences or likelihood-derived quantities, not direct
  measurements of fitness or function.
- The main cohort is assembled from reviewed UniProt/Swiss-Prot records with
  AlphaFold DB structures and inherits their sampling and annotation biases.
- Third-party models differ in training corpora, architecture, conditioning,
  and score definition, so comparisons should be interpreted at the study's
  stated level.
- The reviewer reproduction does not rerun network-, GPU-, checkpoint-, or
  service-dependent upstream stages.
- The independent PDB cohort is audit material rather than a strict
  cross-platform regression target because the historical combined
  family/species fit was rank deficient.

## Reproducibility and licensing

Run `decoding-bias reproduce` for the full reviewer analysis or
`decoding-bias reproduce --quick` for the Python-only path. See
[REPRODUCIBILITY.md](REPRODUCIBILITY.md) for boundaries and deterministic
settings. Repository code is released under the MIT License. Third-party
models retain their original licenses and usage conditions.
