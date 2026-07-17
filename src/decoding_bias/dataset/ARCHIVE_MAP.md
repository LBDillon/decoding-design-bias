# dataset/ consolidation map
Independent leaf scripts merged by phase (`python -m decoding_bias.dataset.<module> <step>`):
- `expansion.py` <- fetch_and_annotate_round2, fetch_and_annotate_round3, build_round3_family_targets, inject_round2_scores
- `assembly.py`  <- combine_main_r2_r3, merge_main_and_expansion, build_filterC_cohort, collapse_species_subspecies
- `annotate.py`  <- annotate_scoring_results, build_v12_features_consistent, make_corrected_v12_csv
The deposited analysis table is the anchor; these document how it was assembled.
