# finetune/ consolidation map

Blocked provenance (needs GPU + weights); not exercised by `decoding-bias verify`.
Scripts merged by phase, each keeping every original function verbatim (one section
per source file) behind a subcommand dispatcher:

    python -m decoding_bias.finetune.<module> <step>

## Merged
- mpnn_train.py     <- train, sweeps, evaluate, run_proteinmpnn_surface, ft020_base_vs_ft_self_consistency, ft020_self_consistency_vs_afdb
- mpnn_data.py      <- build_dataset, build_alkaline_dataset, assign_structures_alkaliphile, extract_alkaline_structures, extract_alkaliphile_structures, stage_c_chain_qc, stage_d_alkaliphile, stage_d_cluster_split, prep_secreted_targets
- esm2_train.py     <- prepare_esm_secretome_data, check_esm_environment, train_esm2_mlm
- esm2_generate.py  <- run_generation_local, score_esm2_masked_marginals, analyse_esm2_score_shifts, esm_design_surface, esm_design_heatmaps
- colab_builders.py <- build_colab, build_generation_colab
- design_inputs.py  <- build_design_acidbase_inputs, build_design_ph_axis_inputs

## Kept as separate files (shared helpers imported across the merged modules)
utils.py, _cohort.py, surface_features_alkaline.py, esm_generation.py, figures.py

## Moved / removed
- test_esm_generation.py -> tests/test_esm_generation.py
- make_callout_pymol.py + its test (presentation-only PyMOL figures) removed.
