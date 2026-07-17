# design/ consolidation map
Independent leaf scripts merged by phase (each keeps its functions verbatim
behind `python -m decoding_bias.design.<module> <step>`):
- `design_folding.py`   <- make_fold_fasta, build_fold_structure_manifest, extract_design_features, compute_design_surface_charge
- `design_tm_tables.py` <- predict_tm_full, tm_shift_analysis, shift_significance_tables
Kept flat (shared / imported by analysis.design): design_common, features_for_designs,
physchem_shift_analysis, functional_residue_conservation.
