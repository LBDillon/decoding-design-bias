"""
Produce the official analysis cohort by applying filter C to the combined
main+r2+r3 dataset.

Filter C (iterated to fixed point):
  - protein_family must span >= 5 species
  - species must contain >= 2 proteins
Species labels are taken from `species_collapsed` (subspecies/serotype
labels already merged via collapse_species_subspecies.py).

Inputs:
  dataset_update/main_plus_r2_r3_speciescollapsed.csv
  dataset_update/main_plus_r2_r3_scored.csv     (optional, for scored output)

Outputs:
  dataset_update/main_plus_r2_r3_filterC.csv
  dataset_update/main_plus_r2_r3_scored_filterC.csv  (if the scored input exists)
"""

from pathlib import Path
import pandas as pd

HERE = Path(__file__).parent
RAW = HERE / "main_plus_r2_r3_speciescollapsed.csv"
SCORED = HERE / "main_plus_r2_r3_scored.csv"
OUT = HERE / "main_plus_r2_r3_filterC.csv"
OUT_SCORED = HERE / "main_plus_r2_r3_scored_filterC.csv"

SP_COL = "species_collapsed"
MIN_SPECIES_PER_FAMILY = 5
MIN_PROTEINS_PER_SPECIES = 2


def apply_filter_C(d):
    """Iteratively enforce filter C until stable."""
    while True:
        n0 = len(d)
        fam_sp = d.groupby("protein_family")[SP_COL].nunique()
        keep_fam = set(fam_sp[fam_sp >= MIN_SPECIES_PER_FAMILY].index)
        d = d[d["protein_family"].isin(keep_fam)]
        sp_count = d[SP_COL].value_counts()
        keep_sp = set(sp_count[sp_count >= MIN_PROTEINS_PER_SPECIES].index)
        d = d[d[SP_COL].isin(keep_sp)]
        if len(d) == n0:
            break
    return d


def main():
    df = pd.read_csv(RAW, low_memory=False)
    print(f"Loaded {len(df)} proteins from {RAW.name}")
    filtered = apply_filter_C(df)
    filtered.to_csv(OUT, index=False)
    print(f"Wrote {OUT.name}: {len(filtered)} proteins "
          f"(removed {len(df) - len(filtered)} = "
          f"{100*(len(df)-len(filtered))/len(df):.1f}%)")
    print(f"  unique species (collapsed): {filtered[SP_COL].nunique()}")
    print(f"  unique protein_family:      {filtered['protein_family'].nunique()}")
    fam_dom = filtered.groupby("protein_family")["domain"].nunique()
    print(f"  multi-domain families:      {(fam_dom>=2).sum()} / {len(fam_dom)}")
    print(f"  3-domain families:          {(fam_dom==3).sum()}")

    # Scored version
    if SCORED.exists():
        scored = pd.read_csv(SCORED, low_memory=False)
        # Carry across the species_collapsed column
        sp_map = filtered.set_index("Entry")[SP_COL]
        # For entries dropped by filter C, we won't include them anyway
        scored_keep = scored[scored["Entry"].isin(set(filtered["Entry"]))].copy()
        scored_keep[SP_COL] = scored_keep["Entry"].map(sp_map)
        scored_keep.to_csv(OUT_SCORED, index=False)
        print(f"\nWrote {OUT_SCORED.name}: {len(scored_keep)} proteins (scored cohort)")


if __name__ == "__main__":
    main()
