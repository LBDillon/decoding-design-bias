# finetune/ consolidation map

The fine-tuning provenance code is blocked here (needs GPU + weights), so it is
not exercised by `decoding-bias verify`. During the 2026 tidy-up the independent
"leaf" scripts were merged by phase; each merged module keeps every original
function verbatim (one section per source script) plus a subcommand dispatcher:

    python -m decoding_bias.finetune.<module> <step>

## Merged
- `mpnn_train.py`     <- train, sweeps, evaluate, run_proteinmpnn_surface,
                          ft020_base_vs_ft_self_consistency, ft020_self_consistency_vs_afdb
- `esm2_train.py`     <- prepare_esm_secretome_data, check_esm_environment, train_esm2_mlm
- `colab_builders.py` <- build_colab, build_generation_colab

## Kept as separate files (shared helpers imported across scripts)
`utils.py`, `_cohort.py`, `figures.py`, `surface_features_alkaline.py`,
`esm_generation.py`, `extract_alkaline_structures.py`, `stage_d_cluster_split.py`,
and the remaining dataset / ESM2-generation / design-input scripts. These are
imported by their siblings via bare (`sys.path`-based) names, so flattening them
would break those imports on code that cannot be re-run and verified here.

## Moved / removed
- `test_esm_generation.py` -> `tests/test_esm_generation.py`
- `make_callout_pymol.py` + its test (presentation-only PyMOL figures) removed.
