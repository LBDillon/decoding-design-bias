"""
run_cohort_vd.py - score-variance decomposition on the independent experimental-PDB
cohort (R3.4 robustness), reusing the committed library
paper_code/03_variance_decomposition/score_variance_decomposition.py.

Differences from the main driver (deliberate, for the cohort):
  * input  = design/outputs/independent_cohort/cohort_pdb_scored.csv
  * PC1/PC2 = the cohort R-notebook PCA (PDB_PCA_22_06_26)
  * 14-feature set, same as the paper
  * PER-MODEL complete-case (NOT a joint dropna): the cohort has models at
    different coverage (MIF/MIF-ST ~562, others ~870), so each model is
    decomposed on all of its own available rows and its design matrices are
    rebuilt on that subset. This avoids shrinking every model to the smallest.
  * only the models actually present are run.

Output -> design/outputs/independent_cohort/cohort_score_variance_decomposition.csv
          + a side-by-side residual-species comparison to the AF2 main run.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd

REPO = Path(__file__).resolve().parent.parent
SVD_DIR = REPO / "paper_code" / "03_variance_decomposition"
sys.path.insert(0, str(SVD_DIR))
import score_variance_decomposition as svd  # noqa: E402

COHORT = REPO / "design" / "outputs" / "independent_cohort" / "cohort_pdb_scored.csv"
PCS = Path("/Users/lauradillon/Downloads/PDB_PCA_22_06_26/mixed_features_pca_coordinates (6).csv")
MAIN_VD = SVD_DIR / "outputs_14feat_v020" / "score_variance_decomposition.csv"
OUT = REPO / "design" / "outputs" / "independent_cohort" / "cohort_score_variance_decomposition.csv"

# 14-feature set (drop the two collinear features)
SEQ = [f for f in svd.SEQ if f not in ("charge_at_ph7", "small_residue_fraction")]
BIOPHYS = SEQ + svd.STRUCT
assert len(BIOPHYS) == 14

# cohort model panel (ProGen2 = the XL column the user is using); canonical -> (col, type)
MODELS = {
    "ProteinMPNN": ("proteinmpnn_score", "structure"),
    "SolubleMPNN": ("solublempnn_score", "structure"),
    "Caliby": ("caliby_score", "structure"),
    "SolubleCaliby": ("soluble_caliby_score", "structure"),
    "ESM-IF": ("esmif_score", "structure"),
    "ESM3-struct": ("esm3_struct_cond_score", "structure"),
    "MIF": ("mif_score", "hybrid"),
    "MIF-ST": ("mifst_score", "hybrid-ST"),
    "ESM3-seq": ("esm3_seq_only_score", "sequence"),
    "ESM2-15B": ("ESM2_15B_pppl_score", "sequence"),
    "ESMC-6B": ("esmc_6b_score", "sequence"),
    "CARP-640M": ("carp_640M_score", "sequence"),
    "ProGen2-XL": ("progen2_XL_score", "sequence"),
    "ProtGPT2": ("protgpt2_score", "sequence"),
    "AlkalineMPNN": ("AlkalineMPNN_score", "structure(FT)"),
    "AcidophileMPNN": ("AcidophileMPNN_score", "structure(FT)"),
}
MIN_N = 100   # skip a model unless it has at least this many scored proteins


def decompose_one(df, col, kind):
    """Per-model complete-case decomposition (mirrors svd.main's per-model record)."""
    sub = df.dropna(subset=BIOPHYS + ["PC1", "PC2", col, "protein_family", "species"]).copy()
    if len(sub) < MIN_N:
        return None
    Z = (sub[BIOPHYS] - sub[BIOPHYS].mean()) / sub[BIOPHYS].std()
    fam = pd.get_dummies(sub["protein_family"], drop_first=True).values.astype(float)
    spc = pd.get_dummies(sub["species"], drop_first=True).values.astype(float)
    sp_codes, sp_levels = pd.factorize(sub["species"])
    PCm = sub[["PC1", "PC2"]].values
    BIO = Z.values
    bases = {
        "LowDim": svd.make_Q(PCm),
        "Biophys": svd.make_Q(BIO),
        "Family": svd.make_Q(fam),
        "Species": svd.make_Q(spc),
        "Fam+Spec": svd.make_Q(np.hstack([fam, spc])),
        "Biophys+Family": svd.make_Q(np.hstack([BIO, fam])),
        "Biophys+Species": svd.make_Q(np.hstack([BIO, spc])),
        "Full": svd.make_Q(np.hstack([BIO, fam, spc])),
    }
    y = sub[col].values.astype(float)
    y = (y - y.mean()) / y.std()
    R = {k: svd.r2_from_Q(y, Q, p) for k, (Q, p) in bases.items()}   # (raw, adj)
    rec = {"model": None, "type": kind, "n": len(sub),
           "families": fam.shape[1] + 1, "species": spc.shape[1] + 1}
    for k in svd.PREDICTOR_SETS:
        rec[f"R2_{k}"] = R[k][0]
        rec[f"R2adj_{k}"] = R[k][1]
    rec["dBiophys_given_family_species"] = R["Full"][0] - R["Fam+Spec"][0]
    rec["dSpecies_given_family_biophys"] = R["Full"][0] - R["Biophys+Family"][0]
    rec["species_only_R2"] = R["Species"][0]
    rec["biophysics_only_R2"] = R["Biophys"][0]
    rec["residual_species_R2"] = R["Full"][0] - R["Biophys+Family"][0]
    rec["species_attenuation"] = (
        (rec["species_only_R2"] - rec["residual_species_R2"]) / rec["species_only_R2"]
        if rec["species_only_R2"] > 0 else np.nan)
    # adjusted-R² variants (penalise the ~400 family+species dummies on n~876 -> the
    # trustworthy quantities given the cohort's degrees of freedom):
    rec["residual_species_R2_adj"] = R["Full"][1] - R["Biophys+Family"][1]
    rec["species_only_R2_adj"] = R["Species"][1]
    rec["species_attenuation_adj"] = (
        (R["Species"][1] - rec["residual_species_R2_adj"]) / R["Species"][1]
        if R["Species"][1] > 0 else np.nan)
    rec.update(svd.species_effect_collapse(y, sp_codes, len(sp_levels), bases))
    sp_partial, sp_F, sp_p = svd.nested_test(y, "Biophys+Family", "Full", bases)
    bio_partial, bio_F, bio_p = svd.nested_test(y, "Fam+Spec", "Full", bases)
    rec["species_partial_R2_given_family_biophys"] = sp_partial
    rec["species_p_given_family_biophys"] = sp_p
    rec["biophys_partial_R2_given_family_species"] = bio_partial
    rec["biophys_p_given_family_species"] = bio_p
    return rec


def main():
    df = pd.read_csv(COHORT)
    pcs = pd.read_csv(PCS)[["Entry", "PC1", "PC2"]]
    df = df.merge(pcs, on="Entry", how="left")
    df = df[df["domain"].isin(["Archaea", "Bacteria", "Eukaryota"])]
    print(f"cohort: {len(df)} rows; 14 features; PCs merged ({df['PC1'].notna().sum()} with PCs)")

    rows = []
    for name, (col, kind) in MODELS.items():
        if col not in df.columns or df[col].notna().sum() < MIN_N:
            print(f"  skip {name:14} ({col}) - present={col in df.columns}, "
                  f"n={int(df[col].notna().sum()) if col in df.columns else 0}")
            continue
        rec = decompose_one(df, col, kind)
        if rec is None:
            print(f"  skip {name:14} - <{MIN_N} complete cases")
            continue
        rec["model"] = name
        rows.append(rec)
        print(f"  {name:14} n={rec['n']:4d}  Biophys={rec['R2_Biophys']:.2f}  "
              f"dSpec|fam,bio={rec['dSpecies_given_family_biophys']:.3f}  "
              f"atten={rec['species_attenuation']*100:.0f}%")

    res = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT, index=False)
    print(f"\nWrote {OUT}  ({len(res)} models)")

    # ---- side-by-side residual-species: PDB cohort vs AF2 main ----
    if MAIN_VD.exists():
        main_vd = pd.read_csv(MAIN_VD)[["model", "dSpecies_given_family_biophys",
                                        "species_effect_retention_given_family_biophys"]]
        main_vd = main_vd.rename(columns={"dSpecies_given_family_biophys": "AF2_resid_spec_raw",
                                          "species_effect_retention_given_family_biophys": "AF2_retention"})
        # align names (ProGen2-XL vs ProGen2 in the main run)
        comp = res[["model", "type", "n", "dSpecies_given_family_biophys", "residual_species_R2_adj",
                    "species_partial_R2_given_family_biophys", "species_p_given_family_biophys",
                    "species_effect_retention_given_family_biophys"]].copy()
        comp = comp.rename(columns={"dSpecies_given_family_biophys": "PDB_resid_spec_raw",
                                    "residual_species_R2_adj": "PDB_resid_spec_adj",
                                    "species_effect_retention_given_family_biophys": "PDB_retention"})
        comp["join"] = comp["model"].str.replace("-XL", "", regex=False)
        merged = comp.merge(main_vd, left_on="join", right_on="model", how="left", suffixes=("", "_m"))
        merged = merged.drop(columns=["join", "model_m"])
        comp_path = OUT.with_name("cohort_vs_af2_residual_species.csv")
        merged.to_csv(comp_path, index=False)
        pd.set_option("display.width", 200)
        print(f"\n=== Residual species bias: PDB cohort vs AF2 main ===")
        print(merged.round(3).to_string(index=False))
        print(f"\nWrote {comp_path}")


if __name__ == "__main__":
    main()
