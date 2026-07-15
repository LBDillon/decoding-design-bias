"""
Take the round-2 KEPT subset (1,362 proteins) and inject the model scores from
/Users/lauradillon/New_scores/ into the combined main+round2 CSV.

Mapping decided from value-distribution match against main:
  proteinMPNN_results_all_20260515_184855.csv  ['sequence_score']    → proteinmpnn_score
  20260516_085537_esmif_results.csv            ['valid_pos_score']   → esmif_score
  mif_likelihoods_final_20260516_092331.csv    ['MIF_Likelihood']    → mif_score
  mif_likelihoods_final_20260516_143140.csv    ['MIF_Likelihood']    → mifst_score

ESM2_15B_pppl_score and carp_640M_score remain NaN for round-2 (not yet scored).

Output:
  dataset_update/main_plus_round2_scored.csv  -- 9,205 rows with scores filled
                                                  for the four scored models
"""

from pathlib import Path
import pandas as pd

HERE = Path(__file__).parent
NS = Path("/Users/lauradillon/New_scores")

COMBINED_PATH = HERE / "main_plus_round2.csv"
OUT_PATH = HERE / "main_plus_round2_scored.csv"

SCORE_MAP = {
    NS / "proteinMPNN_results_all_20260515_184855.csv": ("sequence_score", "proteinmpnn_score"),
    NS / "20260516_085537_esmif_results.csv":          ("valid_pos_score", "esmif_score"),
    NS / "mif_likelihoods_final_20260516_092331.csv":  ("MIF_Likelihood",  "mif_score"),
    NS / "mif_likelihoods_final_20260516_143140.csv":  ("MIF_Likelihood",  "mifst_score"),
}


def main():
    combined = pd.read_csv(COMBINED_PATH, low_memory=False)
    round2_entries = set(combined.loc[combined["source"] == "expansion_round2", "Entry"])
    print(f"Loaded {len(combined)} rows; round-2 entries to fill: {len(round2_entries)}")

    for path, (src_col, dst_col) in SCORE_MAP.items():
        scores = pd.read_csv(path)
        if src_col not in scores.columns or "Entry" not in scores.columns:
            print(f"  SKIP {path.name}: missing column {src_col} or Entry")
            continue
        score_map = scores.set_index("Entry")[src_col]
        # Filter to entries that are actually in round-2 kept
        used = score_map.index.intersection(round2_entries)
        unused = len(score_map) - len(used)
        # Inject - only on round-2 rows, so main values are untouched
        mask = combined["Entry"].isin(used)
        combined.loc[mask, dst_col] = combined.loc[mask, "Entry"].map(score_map).values
        n_filled = combined.loc[combined["source"] == "expansion_round2", dst_col].notna().sum()
        print(f"  {dst_col:25s}  from {path.name}: "
              f"used {len(used)}/{len(score_map)} (unused: {unused}); "
              f"round-2 filled now: {n_filled}/{len(round2_entries)}")

    combined.to_csv(OUT_PATH, index=False)
    print(f"\nWrote {OUT_PATH}")
    print(f"  Score coverage on round-2 (1,362 rows):")
    r2 = combined[combined["source"] == "expansion_round2"]
    for col in ["proteinmpnn_score", "esmif_score", "mif_score", "mifst_score",
                "ESM2_15B_pppl_score", "carp_640M_score"]:
        print(f"    {col:25s} {r2[col].notna().sum():>5} / {len(r2)}")


if __name__ == "__main__":
    main()
