"""Column names, feature sets and the model registry shared by every analysis."""
from __future__ import annotations

ID_COL = "Entry"
SPECIES_COL = "species"
DOMAIN_COL = "domain"
FAMILY_COL = "protein_family"

DOMAINS = ["Archaea", "Bacteria", "Eukaryota"]
PRIMARY_EXCLUDED_ACCESSIONS = ("P19949", "Q8TFG7")

# The 14 modelling features (Table S29) used by the PCA, variance decomposition and
# property importance. charge_at_ph7 and small_residue_fraction are excluded because
# they are collinear with isoelectric_point and mw_per_residue.
SEQUENCE_FEATURES_14 = [
    "sequence_length",
    "mw_per_residue",
    "isoelectric_point",
    "acidic_residue_fraction",
    "basic_residue_fraction",
    "gravy",
    "aromaticity",
    "instability_index",
    "proline_fraction",
]
STRUCTURE_FEATURES_14 = [
    "ordered_percent",
    "helix_sheet_contrast",
    "rco",
    "avg_cb_distance",
    "surface_exposure",
]
BIOPHYS_14 = SEQUENCE_FEATURES_14 + STRUCTURE_FEATURES_14

# The 14 primary models in Figure 2 order: label, score column, architecture type
# (colours the variance charts) and scoring context (main Table 1).
MODELS = [
    ("ProteinMPNN", "proteinmpnn_score", "structure", "Backbone-conditioned"),
    ("SolubleMPNN", "solublempnn_score", "structure", "Backbone-conditioned"),
    ("Caliby", "caliby_score", "structure", "Backbone-conditioned"),
    ("SolubleCaliby", "soluble_caliby_score", "structure", "Backbone-conditioned"),
    ("ESM-IF", "esmif_score", "structure", "Backbone-conditioned"),
    ("TriFlow", "triflow_score", "structure", "Backbone-conditioned"),
    ("MIF", "mif_score", "hybrid", "Structure + native sequence"),
    ("MIF-ST", "mifst_score", "hybrid-ST", "Structure + native sequence"),
    ("ESM3-struct", "esm3_struct_cond_score", "hybrid", "Structure + native sequence"),
    ("ESM3-seq", "esm3_seq_only_score", "sequence", "Sequence-only"),
    ("ESM2-15B", "ESM2_15B_pppl_score", "sequence", "Sequence-only"),
    ("CARP-640M", "carp_640M_score", "sequence", "Sequence-only"),
    ("ProGen2-XL", "progen2_XL_score", "sequence", "Sequence-only"),
    ("ProtGPT2", "protgpt2_score", "sequence", "Sequence-only"),
]
FULL_COHORT = {label: (column, kind) for label, column, kind, _ in MODELS}
ELO_FULL_COLUMNS = [column for _, column, _, _ in MODELS]
LABELS = {column: label for label, column, _, _ in MODELS}
SCORING_CONTEXT = {column: context for _, column, _, context in MODELS}

# Property importance (Tables S16-S17) also reports the ProGen2-base checkpoint.
IMPORTANCE_PANEL = {}
for _label, _column, _, _ in MODELS:
    if _label == "ProGen2-XL":
        IMPORTANCE_PANEL["ProGen2-base"] = "progen2_score"
    IMPORTANCE_PANEL[_label] = _column

# Fine-tuned v_48_020 arm (Table S18): the base checkpoint and the two fine-tuned models.
FINETUNE_020 = {
    "ProteinMPNN-020": "ProteinMPNN_v020_score",
    "AlkSecMPNN-020": "AlkSecMPNN_020_score",
    "AcidSecMPNN-020": "AcidSecMPNN_020_score",
}

TYPE_COLORS = {
    "structure": "#2E7D32", "hybrid": "#F9A825",
    "hybrid-ST": "#C0CA33", "sequence": "#00838F",
}
