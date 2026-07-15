"""
Cohort config for the stage-2 dataset scripts (stage_d / assign_structures / extract).
One dict per cohort holds the stage-C/D filenames, the tight-family subset column, the cluster-id
prefix, and the structures/jsonl output names. `--cohort {alkaline,acid}` selects one.

alkaline = alkaliphile secretome (built by the retired build_alkaliphile_dataset.py; tight subset =
Bacillaceae). acid = acidophile secretome (built by build_dataset.py --cohort acid; tight subset =
the per-cohort tight family). Match columns (matched_control_uniprot / matched_case_uniprot) are
identical across cohorts, so only the names below differ.
"""
COHORTS = {
    "alkaline": dict(
        cases_C="alkaliphile_cases_stageC.csv", ctrls_C="matched_neutralophile_controls_stageC.csv",
        cases_D="alkaliphile_cases_stageD.csv", ctrls_D="matched_neutralophile_controls_stageD.csv",
        tight_in="in_bacillaceae_subset", cl_prefix="alk",
        struct_dir="structures_alkaliphile", manifest="alkaliphile_structures_manifest.csv",
        jsonl_prefix="alkaliphile"),
    "acid": dict(
        cases_C="acidophile_cases_stageC.csv", ctrls_C="acidophile_neutral_controls_stageC.csv",
        cases_D="acidophile_cases_stageD.csv", ctrls_D="acidophile_neutral_controls_stageD.csv",
        tight_in="in_tight_subset", cl_prefix="acid",
        struct_dir="structures_acidophile", manifest="acidophile_structures_manifest.csv",
        jsonl_prefix="acidophile"),
}


def cfg(cohort):
    return COHORTS[cohort]
