# `decoding_bias` - the analysis package

`pip install -e .` (from the repo root), then everything is driven by the
`decoding-bias` command (see the top-level README) or `import decoding_bias`.

## Core
- `catalog.py` -  the 14-feature set, the model panels, the three domains, palettes. 
- `config.py` + `config_default.yaml` - path/parameter resolution 
- `data.py` - load + complete-case the analysis table; dataset composition (Table 7).
- `cli.py` 
- `manifest.py` - paper-output map 
- `verify.py` - re-run reproducible stages and diff vs `../../tests/reference/`.
- `plotting.py` - shared figures 

## Reproducible stages
- `analysis/variance.py` - variance decomposition (Table 1/2, SI Fig S2, Tables S9/S10).
- `analysis/importance.py` - property-to-score importance (SI S6, S16-S18).
- `analysis/elo.py` over `analysis/{elo_rating,elo_figures,elo_paper_figures}.py`
  - species Elo (Fig 2, Table 3, SI S5-S8). `elo_rating` is seeded/deterministic.
- `analysis/pca.py` - biophysical PCA loadings/variance/compactness (Fig 3A/B, Table S14).
- `features/` - sequence/structural/surface feature extractors.

## Stage provenance (need weights/structures/APIs)
Preserved and exposed through the CLI's blocked-stage interface, not run in this repo:
- `scoring/` - per-model likelihood scoring + cross-model registry.
- `design/` - design-shift + functional-residue generators (need design feature tables).
- `finetune/` - both fine-tuning arms (ProteinMPNN + ESM2-35M) + dataset construction.
- `pdb_cohort/` - experimental-PDB cohort build + matched control.
- `dataset/` - analysis-table assembly chain (needs raw + metadata).
- `gam/` - R/mgcv GAM notebook builder + design-ellipse F-test (Fig 3C, Table S15).

Blocked provenance keeps its original internal paths and may need a small
path/import touch-up to run in its external environment.
