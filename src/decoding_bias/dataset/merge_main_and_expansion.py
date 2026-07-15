"""
Merge main Decoding_Bias_Dataset with the curated expansion set into a single
unified CSV.

Inputs (paths relative to this script's directory):
  Decoding_Bias_Dataset_updated.csv                    (main, 7843)
  expansion_for_scoring_monomer_FINAL.csv              (curated expansion, 4594)
  scoring_results/proteinMPNN_results_all_chunks-AF_annotated.csv   (AF scoring, 4317)
  scoring_results/proteinMPNN_results_all_chunks-PDB_annotated.csv  (PDB scoring, 2248)
  scoring_results/esmif_AF_scores_annotated.csv                     (AF ESM-IF)

Output:
  merged_dataset.csv                  - all proteins, unified schema
  merged_dataset_changes_log.txt      - what got normalized / dropped

Design choices documented in expansion_plan.md.
"""

import re
from pathlib import Path
import pandas as pd
import numpy as np
from Bio.SeqUtils.ProtParam import ProteinAnalysis

HERE = Path(__file__).parent
MAIN_PATH = HERE / "Decoding_Bias_Dataset_updated.csv"
EXP_PATH = HERE / "expansion_for_scoring_monomer_FINAL.csv"
AF_PATH = HERE / "scoring_results/proteinMPNN_results_all_chunks-AF_annotated.csv"
PDB_PATH = HERE / "scoring_results/proteinMPNN_results_all_chunks-PDB_annotated.csv"
ESMIF_AF_PATH = HERE / "scoring_results/esmif_AF_scores_annotated.csv"
OUT_PATH = HERE / "merged_dataset.csv"
LOG_PATH = HERE / "merged_dataset_changes_log.txt"


def normalize_species(name):
    """Strip strain/parenthetical annotations: 'Bos taurus (Bovine)' -> 'Bos taurus'."""
    if pd.isna(name):
        return name
    name = re.sub(r"\s*\(strain[^)]*\)", "", name)
    name = re.sub(r"\s*\([^)]+\)", "", name).strip()
    return name


SEQUENCE_FEATURES = [
    "sequence_length", "mw_per_residue", "isoelectric_point", "charge_at_ph7",
    "gravy", "instability_index", "aromaticity",
    "basic_residue_fraction", "acidic_residue_fraction",
    "ionizable_residue_fraction", "proline_fraction", "small_residue_fraction",
    "hydrophobic_fraction", "buffer_capacity", "charge_per_residue",
]


def compute_sequence_features(seq):
    """Match src/features/sequence_features.py + annotate_scoring_results.py."""
    if not isinstance(seq, str) or not seq:
        return {k: np.nan for k in SEQUENCE_FEATURES}
    seq = seq.upper().replace("U", "C").replace("X", "")  # selenocys→cys; drop unknowns
    if not seq:
        return {k: np.nan for k in SEQUENCE_FEATURES}
    try:
        n = len(seq)
        a = ProteinAnalysis(seq)
        aa_pct = a.get_amino_acids_percent()
        acidic = ["D", "E"]; basic = ["K", "R", "H"]
        ionizable = ["D", "E", "K", "R", "H", "C", "Y"]
        small = ["A", "G", "S", "T"]
        hydrophobic = ["A", "V", "I", "L", "M", "F", "W", "P"]
        charge_at_ph7 = a.charge_at_pH(7.0)
        feats = {
            "sequence_length": n,
            "mw_per_residue": a.molecular_weight() / n,
            "isoelectric_point": a.isoelectric_point(),
            "charge_at_ph7": charge_at_ph7,
            "gravy": a.gravy(),
            "instability_index": a.instability_index(),
            "aromaticity": a.aromaticity(),
            "basic_residue_fraction": sum(aa_pct.get(x, 0) for x in basic),
            "acidic_residue_fraction": sum(aa_pct.get(x, 0) for x in acidic),
            "ionizable_residue_fraction": sum(aa_pct.get(x, 0) for x in ionizable),
            "proline_fraction": aa_pct.get("P", 0),
            "small_residue_fraction": sum(aa_pct.get(x, 0) for x in small),
            "hydrophobic_fraction": sum(aa_pct.get(x, 0) for x in hydrophobic),
            "charge_per_residue": charge_at_ph7 / n,
            "buffer_capacity": abs(a.charge_at_pH(8.0) - a.charge_at_pH(6.0)) / 2.0,
        }
        return feats
    except Exception:
        return {k: np.nan for k in SEQUENCE_FEATURES}


# Unified column order for the output CSV
OUTPUT_COLS = [
    # identity
    "Entry", "source", "structure_source", "has_pdb_struct",
    "sequence", "sequence_length",
    # taxonomy (normalized + raw)
    "species", "species_raw", "domain", "phylum_division", "class", "genus",
    # naming
    "protein_name", "protein_name_clean", "protein_family", "broad_function",
    "Description",
    # sequence biophysics (computed by BioPython for all)
    "mw_per_residue", "isoelectric_point", "charge_at_ph7", "gravy",
    "instability_index", "aromaticity", "basic_residue_fraction",
    "acidic_residue_fraction", "ionizable_residue_fraction", "proline_fraction",
    "small_residue_fraction", "hydrophobic_fraction", "buffer_capacity",
    "charge_per_residue", "alkaliphile_score",
    # structural features
    "avg_plddt", "helix_percent", "sheet_percent", "rco", "surface_exposure",
    "avg_cb_distance", "compactness",
    # sparse main-only flags
    "is_enzyme", "is_transmembrane", "is_glycosylated", "has_disordered", "has_pdb",
    "WT_Tm",
    # model scores
    "proteinmpnn_score", "esmif_score", "mif_score", "mifst_score",
    "carp_640M_score", "ESM2_15B_pppl_score", "AlkSecMPNN_score",
    "caliby_score", "triflow_score",
    "esm3_struct_cond_score", "esm3_seq_only_score",
]


def main():
    log = []
    log.append("=" * 70)
    log.append("MERGE LOG")
    log.append("=" * 70)

    # --- Load main ---
    main_df = pd.read_csv(MAIN_PATH)
    log.append(f"Loaded main: {len(main_df)} proteins")

    main_df["source"] = "main"
    main_df["structure_source"] = "AF"
    main_df["has_pdb_struct"] = main_df["has_pdb"].astype(bool)
    main_df["species_raw"] = main_df["species"]
    main_df["species"] = main_df["species"].apply(normalize_species)

    # --- Load expansion curated + AF scoring + PDB scoring ---
    exp_curated = pd.read_csv(EXP_PATH)
    af_scored = pd.read_csv(AF_PATH)
    pdb_scored = pd.read_csv(PDB_PATH)

    log.append(
        f"Loaded expansion: curated={len(exp_curated)}, "
        f"AF-scored={len(af_scored)}, PDB-scored={len(pdb_scored)}"
    )

    af_entries = set(af_scored["Entry"])
    pdb_entries = set(pdb_scored["Entry"])
    viable_entries = af_entries | pdb_entries
    unusable = set(exp_curated["Entry"]) - viable_entries
    log.append(f"Expansion proteins with NO usable structure (dropped): {len(unusable)}")
    log.append(f"Expansion viable (AF-only / PDB-only / both): "
               f"{len(af_entries - pdb_entries)} / {len(pdb_entries - af_entries)} / "
               f"{len(af_entries & pdb_entries)} = {len(viable_entries)} total")

    # --- Build expansion rows: prefer AF scoring file, fall back to PDB ---
    # Both annotated files have the same columns minus 'pdb_id'.
    af_scored = af_scored.copy()
    af_scored["structure_source"] = "AF"
    pdb_only = pdb_scored[~pdb_scored["Entry"].isin(af_entries)].copy()
    pdb_only["structure_source"] = "PDB"

    # avg_plddt only meaningful for AF; PDB rows have avg_bfactor instead, leave plddt NaN
    pdb_only["avg_plddt"] = np.nan
    exp_rows = pd.concat([af_scored, pdb_only], ignore_index=True)
    log.append(f"Expansion rows in merge: {len(exp_rows)}")

    # Normalize species
    exp_rows["species_raw"] = exp_rows["species"]
    exp_rows["species"] = exp_rows["species"].apply(normalize_species)
    log.append(
        f"Unique normalized species: main={main_df['species'].nunique()}, "
        f"expansion={exp_rows['species'].nunique()}, "
        f"shared={len(set(main_df['species']) & set(exp_rows['species']))}"
    )

    # Sources
    exp_rows["source"] = "expansion_" + exp_rows["structure_source"]
    # has_pdb_struct: AF-only means no experimental PDB; both means has PDB; PDB-only has PDB
    exp_rows["has_pdb_struct"] = exp_rows["Entry"].isin(pdb_entries)

    # Bring in PDB-only ProteinMPNN score for proteins that have BOTH AF and PDB (already in AF row)
    # Note: AF score is what we use for the unified proteinmpnn_score; downstream code that
    # specifically wants PDB-based scores should use the original *_PDB_annotated.csv directly.

    # Bring in ESM-IF AF score if available
    if ESMIF_AF_PATH.exists():
        esmif_af = pd.read_csv(ESMIF_AF_PATH)
        # Identify the score column (varies by file)
        score_col = next((c for c in esmif_af.columns
                          if "score" in c.lower() and "valid" not in c and "pos_" not in c
                          and "missing" not in c and "std" not in c
                          and "min" not in c and "max" not in c), None)
        if score_col is None:
            score_col = "esmif_af_score"
        if score_col in esmif_af.columns:
            esmif_map = esmif_af.set_index("Entry")[score_col].to_dict()
            exp_rows["esmif_score"] = exp_rows["Entry"].map(esmif_map)
            log.append(f"Filled esmif_score from {ESMIF_AF_PATH.name} (column {score_col}): "
                       f"{exp_rows['esmif_score'].notna().sum()} proteins")
        else:
            log.append("ESM-IF AF score column not found; esmif_score left NaN for expansion")

    # ProteinMPNN score from AF (or PDB if AF missing); column in annotated files is 'sequence_score'
    if "sequence_score" in exp_rows.columns:
        exp_rows["proteinmpnn_score"] = exp_rows["sequence_score"]

    # --- Compute biophysics for expansion (match main schema) ---
    log.append("Computing BioPython sequence features for expansion rows...")
    feat_records = []
    for _, row in exp_rows.iterrows():
        feats = compute_sequence_features(str(row["sequence"]))
        feats["Entry"] = row["Entry"]
        feat_records.append(feats)
    feat_df = pd.DataFrame(feat_records)
    # Replace existing biophysics columns with freshly computed ones (consistency)
    overlap = (set(feat_df.columns) & set(exp_rows.columns)) - {"Entry"}
    exp_rows = exp_rows.drop(columns=list(overlap))
    exp_rows = exp_rows.merge(feat_df, on="Entry", how="left")
    log.append(f"Computed {len(feat_df.columns)-1} biophysical features for {len(feat_df)} proteins")

    # Function-vocab notes
    main_bf = set(main_df["broad_function"].unique())
    exp_bf = set(exp_rows["broad_function"].unique())
    only_main = main_bf - exp_bf
    only_exp = exp_bf - main_bf
    log.append(f"broad_function only in main: {sorted(only_main)}")
    log.append(f"broad_function only in expansion: {sorted(only_exp)}")
    log.append("(Vocabulary unified as union; no remapping performed)")

    # --- Combine into unified schema ---
    for c in OUTPUT_COLS:
        if c not in main_df.columns:
            main_df[c] = np.nan
        if c not in exp_rows.columns:
            exp_rows[c] = np.nan

    main_keep = main_df[OUTPUT_COLS]
    exp_keep = exp_rows[OUTPUT_COLS]
    merged = pd.concat([main_keep, exp_keep], ignore_index=True)

    log.append(f"\nFinal merged dataset: {len(merged)} proteins")
    log.append(f"  by source: {merged['source'].value_counts().to_dict()}")
    log.append(f"  by structure_source: {merged['structure_source'].value_counts().to_dict()}")
    log.append(f"  with PDB structure: {int(merged['has_pdb_struct'].sum())}")
    log.append(f"  unique species: {merged['species'].nunique()}")
    log.append(f"  unique protein_family: {merged['protein_family'].nunique()}")
    log.append(f"  unique broad_function: {merged['broad_function'].nunique()}")
    log.append(f"  ribosomal share: {100*(merged['broad_function']=='ribosomal').mean():.1f}%")

    merged.to_csv(OUT_PATH, index=False)
    LOG_PATH.write_text("\n".join(log) + "\n")
    print("\n".join(log))
    print(f"\nWrote {OUT_PATH}")
    print(f"Wrote {LOG_PATH}")


if __name__ == "__main__":
    main()
