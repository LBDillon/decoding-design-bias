"""
run_cohort_elo.py - species Elo on the independent experimental-PDB cohort, using
the committed paper_code/02_elo/elo_rating.py. Unweighted (experimental structures
carry no pLDDT). Present models only; ProGen2 = the XL column.

Output -> design/outputs/independent_cohort/elo_cohort_unweighted/
"""
import importlib.util
from pathlib import Path
import numpy as np, pandas as pd

REPO = Path(__file__).resolve().parent.parent
ELO_DIR = REPO / "paper_code" / "02_elo"
spec = importlib.util.spec_from_file_location("elor", ELO_DIR / "elo_rating.py")
E = importlib.util.module_from_spec(spec); spec.loader.exec_module(E)

COHORT = REPO / "design" / "outputs" / "independent_cohort" / "cohort_pdb_scored.csv"
OUTDIR = REPO / "design" / "outputs" / "independent_cohort" / "elo_cohort_unweighted"

CANDIDATES = ["proteinmpnn_score", "solublempnn_score", "caliby_score", "soluble_caliby_score",
              "esmif_score", "mif_score", "mifst_score", "esm3_struct_cond_score",
              "esm3_seq_only_score", "ESM2_15B_pppl_score", "esmc_6b_score", "carp_640M_score",
              "progen2_XL_score", "protgpt2_score",
              "AlkalineMPNN_score", "AcidophileMPNN_score"]
DOM = ["Archaea", "Bacteria", "Eukaryota"]


def main():
    df = pd.read_csv(COHORT)
    if "avg_plddt" not in df.columns:        # unweighted Elo doesn't use it, but be safe
        df["avg_plddt"] = np.nan
    present = [c for c in CANDIDATES if c in df.columns and df[c].notna().sum() >= 100]
    print(f"cohort Elo on {len(present)} models: {present}")

    E.run_full_elo_analysis(df, str(OUTDIR), score_columns=present,
                            n_permutations=50, protein_column="protein_family",
                            species_column="species", use_plddt_weighting=False)

    long = pd.read_csv(OUTDIR / "results" / "all_models_species_ratings_long.csv")
    g = (long.groupby("model")
         .apply(lambda x: pd.Series({
             "Archaea": x[x.domain == "Archaea"].rating.mean(),
             "Bacteria": x[x.domain == "Bacteria"].rating.mean(),
             "Eukaryota": x[x.domain == "Eukaryota"].rating.mean(),
             "Arch_minus_Euk": x[x.domain == "Archaea"].rating.mean() - x[x.domain == "Eukaryota"].rating.mean(),
             "top_domain": x.groupby("domain").rating.mean().idxmax(),
         })).reset_index())
    g = g.sort_values("Arch_minus_Euk", ascending=False)
    pd.set_option("display.width", 200)
    print("\n=== Cohort species Elo by domain (unweighted) ===")
    print(g.round(0).to_string(index=False))
    g.to_csv(OUTDIR / "cohort_elo_domain_summary.csv", index=False)
    print(f"\nWrote {OUTDIR/'cohort_elo_domain_summary.csv'}")


if __name__ == "__main__":
    main()
