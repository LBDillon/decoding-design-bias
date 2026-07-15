"""
Collapse subspecies / serotype / biovar / "sp." labels to the underlying
Genus species name. Mirrors the strain-grouping done on the original 144-species
cohort. Adds a `species_collapsed` column rather than overwriting `species`
so the original UniProt label is preserved.

Rules:
  - Strip strain/parenthetical (already done by upstream normalisation)
  - Drop "subsp. X", "serotype X", "serovar X", "biotype X", "biovar X"
  - Drop trailing pathotype codes like O157:H7, K1, H4
  - Leave "Genus sp." alone (no resolvable species)
  - Otherwise keep as-is
"""

from pathlib import Path
import re
import pandas as pd

HERE = Path(__file__).parent
IN_CSV = HERE / "main_plus_r2_r3.csv"
OUT_CSV = HERE / "main_plus_r2_r3_speciescollapsed.csv"

SUBSPECIES_TAGS = re.compile(
    r'\s+(subsp\.|serotype|serovar|biotype|biovar)\s+.*$',
    flags=re.IGNORECASE,
)
PATHOTYPE_TAIL = re.compile(
    r'\s+[OKH]\d+(?::[OKH]\d+)*(?:\s*[\w/]*)?$',
    flags=re.IGNORECASE,
)


def collapse(name: str) -> str:
    if not isinstance(name, str):
        return name
    s = name.strip()
    # Don't touch "Genus sp." (unresolved species)
    if re.search(r'\bsp\.\s*$', s):
        return s
    # Drop subspecies-style tags
    s = SUBSPECIES_TAGS.sub('', s)
    # Drop pathotype tails (E. coli O157:H7 etc.)
    s = PATHOTYPE_TAIL.sub('', s)
    return s.strip()


def main():
    df = pd.read_csv(IN_CSV, low_memory=False)
    df["species_collapsed"] = df["species"].apply(collapse)

    before = df["species"].nunique()
    after = df["species_collapsed"].nunique()
    df.to_csv(OUT_CSV, index=False)
    print(f"main_plus_r2_r3 species labels:")
    print(f"  before collapse: {before}")
    print(f"  after collapse:  {after}")
    print(f"  collapsed away:  {before - after}")

    # Show the most-affected merges
    affected = df[df["species"] != df["species_collapsed"]]
    if len(affected):
        merges = (
            affected.groupby("species_collapsed")["species"]
            .nunique()
            .sort_values(ascending=False)
            .head(10)
        )
        print(f"\nTop merges (target species ← number of variants pooled in):")
        for sp_c, n in merges.items():
            variants = (
                affected[affected["species_collapsed"] == sp_c]["species"]
                .unique()
            )
            tot = (df["species_collapsed"] == sp_c).sum()
            print(f"  {sp_c:50s} ({n} variants → {tot} proteins)")

    print(f"\nWrote {OUT_CSV}")


if __name__ == "__main__":
    main()
