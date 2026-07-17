"""features, models, domains, palettes.
"""
from __future__ import annotations

# --------------------------------------------------------------------------- #
# Key columns of the analysis table
# --------------------------------------------------------------------------- #
ID_COL = "Entry"
SPECIES_COL = "species"
DOMAIN_COL = "domain"
FAMILY_COL = "protein_family"          #  281-family grouping (paper tab:elo)
BROAD_FUNCTION_COL = "broad_function"

DOMAINS = ["Archaea", "Bacteria", "Eukaryota"]

# --------------------------------------------------------------------------- #
# Biophysical features
# --------------------------------------------------------------------------- #
# The pruned 14-feature set used by the PCA, variance decomposition and property
# importance. charge_at_ph7 (collinear with isoelectric_point) and
# small_residue_fraction (collinear with mw_per_residue) were dropped 2026-06-20.
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
assert len(BIOPHYS_14) == 14

# The two collinear features dropped from the pruned set. Restoring them gives
# the full 16-feature superset that the design-shift analysis reports.
DROPPED_COLLINEAR = ["charge_at_ph7", "small_residue_fraction"]

# --------------------------------------------------------------------------- #
# Model panels (pretty name -> score column). Ordering  sets
# the column order of the SI wide tables and the bar-chart ordering.
# --------------------------------------------------------------------------- #
# type is the architecture class used to colour/order the variance-decomposition
# charts: structure | hybrid | hybrid-ST | sequence.
FULL_COHORT = {
    "ProteinMPNN": ("proteinmpnn_score", "structure"),
    "SolubleMPNN": ("solublempnn_score", "structure"),
    "ESM-IF": ("esmif_score", "structure"),
    "MIF": ("mif_score", "hybrid"),
    "MIF-ST": ("mifst_score", "hybrid-ST"),
    "Caliby": ("caliby_score", "structure"),
    "SolubleCaliby": ("soluble_caliby_score", "structure"),
    "TriFlow": ("triflow_score", "structure"),
    "ESM3-struct": ("esm3_struct_cond_score", "structure"),
    "ESM3-seq": ("esm3_seq_only_score", "sequence"),
    "ESM2-15B": ("ESM2_15B_pppl_score", "sequence"),
    "CARP-640M": ("carp_640M_score", "sequence"),
    "ProGen2": ("progen2_score", "sequence"),
    "ProtGPT2": ("protgpt2_score", "sequence"),
}

# Elo runs address score columns directly, in the canonical order.
ELO_FULL_COLUMNS = [
    "proteinmpnn_score", "solublempnn_score", "caliby_score", "soluble_caliby_score",
    "esmif_score", "mif_score", "mifst_score", "esm3_struct_cond_score",
    "esm3_seq_only_score", "ESM2_15B_pppl_score", "carp_640M_score", "triflow_score",
    "progen2_score", "protgpt2_score",
]

# The 15-model property-importance panel (adds ProGen2-XL to the cohort). Order
# copied verbatim from rerun_property_importance_expanded.py PANEL.
IMPORTANCE_PANEL = {
    "ProteinMPNN": "proteinmpnn_score", "SolubleMPNN": "solublempnn_score",
    "Caliby": "caliby_score", "SolubleCaliby": "soluble_caliby_score",
    "ESM-IF": "esmif_score", "MIF": "mif_score", "MIF-ST": "mifst_score",
    "TriFlow": "triflow_score", "ESM3-struct": "esm3_struct_cond_score",
    "ESM2-15B": "ESM2_15B_pppl_score", "ESM3-seq": "esm3_seq_only_score",
    "CARP-640M": "carp_640M_score", "ProGen2": "progen2_score",
    "ProGen2-XL": "progen2_XL_score", "ProtGPT2": "protgpt2_score",
}

# Fine-tuned v_48_020 arm: base 020 vs the two fine-tuned 020 models.
FINETUNE_020 = {
    "ProteinMPNN-020": "ProteinMPNN_v020_score",
    "AlkSecMPNN-020": "AlkSecMPNN_020_score",
    "AcidSecMPNN-020": "AcidSecMPNN_020_score",
}
# Elo addresses the FT arm by score column, in this order.
ELO_FT_COLUMNS = ["ProteinMPNN_v020_score", "AlkSecMPNN_020_score", "AcidSecMPNN_020_score"]

# Pretty names for every score column encountered by the Elo figures.
PRETTY = {
    "proteinmpnn_score": "ProteinMPNN", "solublempnn_score": "SolubleMPNN",
    "caliby_score": "Caliby", "soluble_caliby_score": "SolubleCaliby",
    "esmif_score": "ESM-IF", "mif_score": "MIF", "mifst_score": "MIF-ST",
    "esm3_struct_cond_score": "ESM3-struct", "esm3_seq_only_score": "ESM3-seq",
    "ESM2_15B_pppl_score": "ESM2-15B", "carp_640M_score": "CARP-640M",
    "triflow_score": "TriFlow", "progen2_score": "ProGen2",
    "progen2_XL_score": "ProGen2-XL", "protgpt2_score": "ProtGPT2",
    "ProteinMPNN_v020_score": "ProteinMPNN(v020)", "ProteinMPNN_v002_score": "ProteinMPNN(v002)",
    "AlkSecMPNN_020_score": "AlkSecMPNN_020", "AcidSecMPNN_020_score": "AcidSecMPNN_020",
}

# --------------------------------------------------------------------------- #
# Palettes (shared between Python and R visualisations)
# --------------------------------------------------------------------------- #
DOMAIN_COLORS = {"Archaea": "#D7191C", "Bacteria": "#2C7BB6", "Eukaryota": "#1A9641"}
TYPE_COLORS = {
    "structure": "#2E7D32", "hybrid": "#F9A825",
    "hybrid-ST": "#C0CA33", "sequence": "#00838F",
}


def score_columns(models: dict) -> list[str]:
    """Resolve a pretty-name->col(-or-(col,type)) mapping to score columns."""
    out = []
    for v in models.values():
        out.append(v[0] if isinstance(v, tuple) else v)
    return out
