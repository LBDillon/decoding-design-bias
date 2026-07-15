"""
build_pdb_cohort_features.py - compute the 16 v12 mixed_features for the 876
independent experimental-PDB cohort, IDENTICALLY to the main dataset, so the
cohort can go through the same variance-decomposition / PCA pipeline.

For each cohort chain:
  - 11 sequence features  : from the resolved chain `sequence` (clean_seq: U->C, drop X)
  - 5  structure features : from the single-chain experimental PDB (`chain_pdb_path`)

Reuses design/features_for_designs.py (= src/features extractors), so definitions
match v12 exactly.

Input : design/outputs/independent_cohort/cohort_pdb_scoring_inputs.csv
Output: design/outputs/independent_cohort/cohort_pdb_features.csv
"""
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))

import numpy as np, pandas as pd
from tqdm import tqdm
from features_for_designs import sequence_features, structure_features, MIXED_FEATURES

COH = REPO / "design" / "outputs" / "independent_cohort" / "cohort_pdb_scoring_inputs.csv"
OUT = REPO / "design" / "outputs" / "independent_cohort" / "cohort_pdb_features.csv"

SEQ_FEATS = ["sequence_length", "mw_per_residue", "isoelectric_point", "charge_at_ph7",
             "acidic_residue_fraction", "basic_residue_fraction", "gravy", "aromaticity",
             "instability_index", "proline_fraction", "small_residue_fraction"]
STRUCT_FEATS = ["ordered_percent", "helix_sheet_contrast", "rco", "avg_cb_distance",
                "surface_exposure"]
CARRY = ["Entry", "pdb_id", "chain", "domain", "species_collapsed", "protein_family",
         "broad_function", "resolution_A", "resolved_len", "seqres_len"]


def clean_seq(s):
    """Match the v12 builder: U->C (selenocysteine), drop X."""
    if not isinstance(s, str):
        return ""
    return s.upper().replace("U", "C").replace("X", "")


def main():
    df = pd.read_csv(COH)
    print(f"PDB cohort: {len(df)} chains")

    seq_rows, struct_rows, seq_err, struct_err = [], [], 0, 0
    for _, r in tqdm(df.iterrows(), total=len(df), desc="features"):
        # sequence features
        cs = clean_seq(r["sequence"])
        try:
            f = sequence_features(cs) if cs else {}
            seq_rows.append({k: f.get(k) for k in SEQ_FEATS})
        except Exception:
            seq_rows.append({k: np.nan for k in SEQ_FEATS}); seq_err += 1
        # structure features
        try:
            sfd = structure_features(r["chain_pdb_path"])
            struct_rows.append({k: sfd.get(k) for k in STRUCT_FEATS})
        except Exception as e:
            struct_rows.append({k: np.nan for k in STRUCT_FEATS}); struct_err += 1

    seqf = pd.DataFrame(seq_rows)
    strf = pd.DataFrame(struct_rows)

    out = df[CARRY].reset_index(drop=True).copy()
    out["species"] = out["species_collapsed"]   # R PCA notebook expects `species`
    for k in SEQ_FEATS:
        out[k] = seqf[k].values
    for k in STRUCT_FEATS:
        out[k] = strf[k].values

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"\nWrote {OUT}  ({out.shape[0]} rows, {out.shape[1]} cols)")
    print(f"sequence-feature errors: {seq_err} | structure-feature errors: {struct_err}")
    miss = {k: int(out[k].isna().sum()) for k in MIXED_FEATURES if out[k].isna().sum()}
    print("Missing per feature:", miss or "none")
    return out


if __name__ == "__main__":
    main()
