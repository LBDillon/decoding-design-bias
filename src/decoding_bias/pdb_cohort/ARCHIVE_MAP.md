# pdb_cohort/ consolidation map
Independent leaf scripts merged by phase (`python -m decoding_bias.pdb_cohort.<module> <step>`):
- `cohort_build.py`    <- build_independent_pdb_cohort, build_mif_safe_cohort, scan_uniprot_pdb, implement_uniprot_pdb_retrieval, prepare_pdb_chain_sequences
- `cohort_scoring.py`  <- prep_independent_cohort_structures, prep_pdb_inputs_fresh, build_pdb_cohort_features, score_esmif_cohort, merge_pdb_cohort_scores
- `cohort_analysis.py` <- run_cohort_elo, run_replication_stats, compare_cohort_to_main, run_matched_af2_control
- `cohort_report.py`   <- build_report, make_pdb_vd_mainstyle_figures
Kept flat (shared helpers): build_pdb_scoring_inputs, run_cohort_vd.
