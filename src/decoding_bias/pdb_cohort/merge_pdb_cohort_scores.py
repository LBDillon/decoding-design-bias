"""
merge_pdb_cohort_scores.py - assemble ONE 876-row PDB-cohort table:
    cohort_pdb_features.csv  (Entry + grouping + 16 features)
  + every per-model score CSV in the scores folder (canonical column names)

Re-runnable: drop new per-model score CSVs into SCORES_DIR and re-run; they slot in.
Each model is matched by the *source column name* it carries, so messy filenames
(e.g. "triflow_scores_pdb (2).csv", duplicate caliby files) don't matter.

Input  features : design/outputs/independent_cohort/cohort_pdb_features.csv
Input  scores   : ~/Downloads/PDB_cohort_scores/*.csv   (override with --scores-dir)
Output          : design/outputs/independent_cohort/cohort_pdb_scored.csv
"""
import argparse, glob, os
from pathlib import Path
import numpy as np, pandas as pd

REPO = Path(__file__).resolve().parent.parent
FEATURES = REPO / "design" / "outputs" / "independent_cohort" / "cohort_pdb_features.csv"
OUT = REPO / "design" / "outputs" / "independent_cohort" / "cohort_pdb_scored.csv"
DEFAULT_SCORES = Path.home() / "Downloads" / "PDB_cohort_scores"

# source column in the score file  ->  canonical column in the merged table
RENAME = {
    "proteinmpnn": "proteinmpnn_score",
    "solublempnn": "solublempnn_score",
    "AlkalineMPNN": "AlkalineMPNN_score",
    "AcidophileMPNN": "AcidophileMPNN_score",
    "caliby_score": "caliby_score",
    "soluble_caliby_score": "soluble_caliby_score",
    "esmif_score": "esmif_score",
    "triflow_score": "triflow_score",
    "esm3_struct_cond_score": "esm3_struct_cond_score",
    "esm3_seq_only_score": "esm3_seq_only_score",
    "mif_score": "mif_score",
    "mifst_score": "mifst_score",
    "ESM2_15B_pppl_score": "ESM2_15B_pppl_score",
    "carp_640M_score": "carp_640M_score",
    "progen2_XL_score": "progen2_XL_score",
    "protgpt2_score": "protgpt2_score",
}

# the 14-model paper panel (canonical names) for the coverage report
PANEL = ["proteinmpnn_score", "solublempnn_score", "caliby_score", "soluble_caliby_score",
         "esmif_score", "triflow_score", "esm3_struct_cond_score", "mif_score", "mifst_score",
         "esm3_seq_only_score", "ESM2_15B_pppl_score", "carp_640M_score",
         "progen2_XL_score", "protgpt2_score"]
FINETUNES = ["AlkalineMPNN_score", "AcidophileMPNN_score"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores-dir", default=str(DEFAULT_SCORES))
    args = ap.parse_args()
    scores_dir = Path(args.scores_dir)

    master = pd.read_csv(FEATURES)
    master["Entry"] = master["Entry"].astype(str)
    cohort_ids = set(master["Entry"])
    print(f"features base: {len(master)} rows ({len(cohort_ids)} cohort Entries)")

    filled = {}  # canonical col -> source filename (for provenance / dup detection)
    for f in sorted(glob.glob(str(scores_dir / "*.csv"))):
        df = pd.read_csv(f)
        if "Entry" not in df.columns:
            continue
        df["Entry"] = df["Entry"].astype(str)
        for src, canon in RENAME.items():
            if src not in df.columns:
                continue
            sub = df[["Entry", src]].dropna(subset=["Entry"]).drop_duplicates("Entry")
            sub = sub[sub["Entry"].isin(cohort_ids)]
            if canon in filled:
                # already provided by another file: verify agreement, don't double-add
                prev = master[["Entry", canon]].dropna()
                chk = prev.merge(sub.rename(columns={src: canon + "_new"}), on="Entry")
                if len(chk):
                    diff = (chk[canon] - chk[canon + "_new"]).abs()
                    note = "identical" if diff.max() < 1e-6 else f"DIFFERS max={diff.max():.4g}"
                    print(f"  dup {canon}: {os.path.basename(f)} ({note}) - kept {filled[canon]}")
                continue
            master = master.merge(sub.rename(columns={src: canon}), on="Entry", how="left")
            filled[canon] = os.path.basename(f)

    master.to_csv(OUT, index=False)
    print(f"\nWrote {OUT}  ({master.shape[0]} rows, {master.shape[1]} cols)")

    print("\nPanel coverage (non-null / 876):")
    have, missing = [], []
    for c in PANEL:
        if c in master.columns:
            n = int(master[c].notna().sum())
            have.append(c)
            print(f"  {c:26} {n:4d}/876   [{filled.get(c,'?')}]")
        else:
            missing.append(c)
    for c in FINETUNES:
        if c in master.columns:
            print(f"  + {c:26} {int(master[c].notna().sum()):4d}/876   [{filled.get(c,'?')}]  (fine-tune)")
    if missing:
        print("\n  ❌ STILL MISSING (need scoring):")
        for c in missing:
            print(f"     - {c}")
    print(f"\n{len(have)}/{len(PANEL)} panel models present.")


if __name__ == "__main__":
    main()
