"""
features_for_designs.py - compute the 16 v12 "mixed_features" for a designed
(or WT) sequence + its folded structure, IDENTICALLY to how the v12 dataset was
built, so designs can be projected into the existing v12 PCA space.

Reuses the repository's own feature code:
    src/features/sequence_features.py   (11 sequence features)
    src/features/structural_features.py (5 structure features, need a PDB)

The 16 mixed_features (exact v12 PCA column names):
  sequence : sequence_length, mw_per_residue, isoelectric_point, charge_at_ph7,
             acidic_residue_fraction, basic_residue_fraction, gravy, aromaticity,
             instability_index, proline_fraction, small_residue_fraction
  structure: ordered_percent, helix_sheet_contrast, rco, avg_cb_distance,
             surface_exposure
"""
from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")

from decoding_bias.features.sequence_features import calculate_sequence_features
from decoding_bias.features import structural_features as sf

MIXED_FEATURES = [
    # sequence (11)
    "sequence_length", "mw_per_residue", "isoelectric_point", "charge_at_ph7",
    "acidic_residue_fraction", "basic_residue_fraction", "gravy", "aromaticity",
    "instability_index", "proline_fraction", "small_residue_fraction",
    # structure (5)
    "ordered_percent", "helix_sheet_contrast", "rco", "avg_cb_distance",
    "surface_exposure",
]


def sequence_features(seq: str) -> dict:
    """11 sequence features under the v12 PCA names."""
    f = calculate_sequence_features(seq)
    # name reconciliation to v12 PCA columns
    if "charge_at_ph7" not in f and "charge_at_pH7" in f:
        f["charge_at_ph7"] = f["charge_at_pH7"]
    return f


def structure_features(pdb_path: str) -> dict:
    """5 structure features from a folded PDB (rco, avg_cb_distance,
    surface_exposure, ordered_percent, helix_sheet_contrast)."""
    struct = sf.parse_structure(pdb_path)
    out = {}
    out["rco"] = sf.calculate_contact_order(struct)
    out["surface_exposure"] = sf.calculate_surface_exposure(struct)
    out["avg_cb_distance"] = sf.calculate_avg_cb_distance(struct)
    ss = sf.extract_secondary_structure(struct)        # helix/sheet/loop %, contrast, ordered
    out["ordered_percent"] = ss["ordered_percent"]
    out["helix_sheet_contrast"] = ss["helix_sheet_contrast"]
    return out


def compute_mixed_features(seq: str, pdb_path: str) -> dict:
    """Return the 16 mixed_features for one sequence + its structure."""
    feats = {}
    feats.update(sequence_features(seq))
    feats.update(structure_features(pdb_path))
    return {k: feats.get(k) for k in MIXED_FEATURES}


if __name__ == "__main__":
    # quick self-test on one WT structure
    import design_common as dc
    p = dc.load_inputs().iloc[0]
    f = compute_mixed_features(p.wt_sequence, p.structure_path)
    print(f"{p.uniprot_id}:")
    for k in MIXED_FEATURES:
        print(f"  {k:24} {f[k]}")
