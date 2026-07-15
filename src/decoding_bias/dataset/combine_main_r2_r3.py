"""
Combine main (7,843) + round-2 KEPT (1,362) + round-3 (1,645) = 10,850 proteins.

Round-3 carries no model scores yet (not yet scored); all 11 model columns are
NaN for those rows. Schema is aligned to main.

Output:
  dataset_update/main_plus_r2_r3.csv
  dataset_update/main_plus_r2_r3_scored.csv  -- with round-2 KEPT scores injected
                                                 from /Users/lauradillon/New_scores/
"""

from pathlib import Path
import pandas as pd
import numpy as np

HERE = Path(__file__).parent
MAIN_PATH = HERE / "Decoding_Bias_Dataset_updated.csv"
R2_PATH = HERE / "round2" / "expansion_round2_KEPT.csv"
R3_PATH = HERE / "round3" / "expansion_round3_for_scoring.csv"
SCORED_R2_PATH = HERE / "main_plus_round2_scored.csv"
OUT_PATH = HERE / "main_plus_r2_r3.csv"
OUT_SCORED_PATH = HERE / "main_plus_r2_r3_scored.csv"


def _align_to_main(df, main_cols, source_label):
    df = df.copy()
    df["source"] = source_label
    df["structure_source"] = "AF"
    if "has_pdb_struct" not in df.columns:
        df["has_pdb_struct"] = False
    df["has_pdb_struct"] = df["has_pdb_struct"].fillna(False).astype(bool)
    # round-3 has charge_at_pH7 (mixed case); main uses charge_at_ph7
    if "charge_at_pH7" in df.columns and "charge_at_ph7" not in df.columns:
        df["charge_at_ph7"] = df["charge_at_pH7"]
    if "genus" not in df.columns:
        df["genus"] = np.nan
    for col in ["phylum_division", "class", "Description", "has_pdb",
                "alkaliphile_score", "WT_Tm"]:
        if col not in df.columns:
            df[col] = np.nan
    return df


def main():
    main = pd.read_csv(MAIN_PATH, low_memory=False)
    r2 = pd.read_csv(R2_PATH, low_memory=False)
    r3 = pd.read_csv(R3_PATH, low_memory=False)
    print(f"main:    {len(main)} rows")
    print(f"round-2 KEPT: {len(r2)} rows")
    print(f"round-3:      {len(r3)} rows")

    # Tag main
    main = main.copy()
    main["source"] = "main"
    main["structure_source"] = "AF"
    main["has_pdb_struct"] = main["has_pdb"].fillna(False).astype(bool)

    r2 = _align_to_main(r2, main.columns, "expansion_round2")
    r3 = _align_to_main(r3, main.columns, "expansion_round3")

    # Union of columns; main schema first
    all_cols = list(main.columns)
    for df_ in (r2, r3):
        for c in df_.columns:
            if c not in all_cols:
                all_cols.append(c)
    combined = pd.concat(
        [main.reindex(columns=all_cols),
         r2.reindex(columns=all_cols),
         r3.reindex(columns=all_cols)],
        ignore_index=True,
    )

    # Sanity: no duplicate Entries
    dups = combined["Entry"].duplicated().sum()
    if dups:
        print(f"WARNING: {dups} duplicate Entries (deduping, keeping first)")
        combined = combined.drop_duplicates("Entry", keep="first")

    combined.to_csv(OUT_PATH, index=False)
    print(f"\nWrote {OUT_PATH}: {len(combined)} rows x {len(combined.columns)} cols")
    print(f"  source counts: {combined['source'].value_counts().to_dict()}")
    print(f"  unique species: {combined['species'].nunique()}")
    print(f"  unique protein_family: {combined['protein_family'].nunique()}")
    print(f"  ribosomal share: {100*(combined['broad_function']=='ribosomal').mean():.1f}%")
    print(f"  by domain:")
    for d, n in combined['domain'].value_counts().items():
        print(f"    {d:12s} {n:5d}  {100*n/len(combined):5.1f}%")

    # Now inject round-2 KEPT scores from the existing scored CSV
    if SCORED_R2_PATH.exists():
        scored = pd.read_csv(SCORED_R2_PATH, low_memory=False)
        r2_scored = scored[scored["source"] == "expansion_round2"].copy()
        score_cols = ["proteinmpnn_score", "esmif_score", "mif_score", "mifst_score"]
        # Map round-2 KEPT scores into the new combined
        r2_score_map = r2_scored.set_index("Entry")[score_cols]
        for col in score_cols:
            mask = combined["Entry"].isin(r2_score_map.index)
            combined.loc[mask, col] = combined.loc[mask, "Entry"].map(r2_score_map[col]).values
        combined.to_csv(OUT_SCORED_PATH, index=False)
        print(f"\nWrote {OUT_SCORED_PATH}")
        print(f"  Round-2 KEPT score coverage:")
        r2r = combined[combined["source"] == "expansion_round2"]
        for c in score_cols:
            n = r2r[c].notna().sum()
            print(f"    {c:25s} {n}/{len(r2r)}")
        print(f"  Round-3 score coverage (all should be 0):")
        r3r = combined[combined["source"] == "expansion_round3"]
        for c in score_cols + ["ESM2_15B_pppl_score", "carp_640M_score"]:
            n = r3r[c].notna().sum()
            print(f"    {c:25s} {n}/{len(r3r)}")


if __name__ == "__main__":
    main()
