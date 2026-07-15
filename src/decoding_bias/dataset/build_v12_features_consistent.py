"""
Recompute the 16 mixed_features for the full v12 cohort with the CLEAN extractor
(src/features), so the cloud, WT, and designs are all on identical definitions.

- 11 sequence features: recomputed from metadata_v12['sequence'] (fixes stale
  mw_per_residue / charge_at_ph7 / instability_index / small_residue_fraction).
- 5 structural features: reused from analysis_v12 (already match the extractor).
- model-score columns: carried through for the GAM preference landscapes.

Output: design/outputs/v12_features_consistent.csv  (Entry + domain + 16 features + scores)
This becomes the single source of truth the PCA is refit on.
"""
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np, pandas as pd
from tqdm import tqdm
from features_for_designs import sequence_features, MIXED_FEATURES

REPO = Path(__file__).resolve().parent.parent
ANA  = REPO / "dataset_update" / "main_plus_r2_r3_analysis_v12.csv"
META = REPO / "dataset_update" / "main_plus_r2_r3_metadata_v12.csv"
OUT  = Path(__file__).resolve().parent / "outputs" / "v12_features_consistent.csv"

SEQ_FEATS = ["sequence_length", "mw_per_residue", "isoelectric_point", "charge_at_ph7",
             "acidic_residue_fraction", "basic_residue_fraction", "gravy", "aromaticity",
             "instability_index", "proline_fraction", "small_residue_fraction"]
STRUCT_FEATS = ["ordered_percent", "helix_sheet_contrast", "rco", "avg_cb_distance",
                "surface_exposure"]
SCORE_COLS = ["proteinmpnn_score", "solublempnn_score", "esmif_score", "mif_score",
              "mifst_score", "ESM2_15B_pppl_score", "caliby_score", "triflow_score",
              "esm3_struct_cond_score", "esm3_seq_only_score", "carp_640M_score"]


def clean_seq(s):
    """Same cleaning used by the v12 builder: U->C (selenocysteine), drop X."""
    if not isinstance(s, str):
        return ""
    return s.upper().replace("U", "C").replace("X", "")


def main():
    ana = pd.read_csv(ANA, low_memory=False)
    meta = pd.read_csv(META, low_memory=False)[["Entry", "sequence"]]
    df = ana.merge(meta, on="Entry", how="left")
    print(f"v12 cohort: {len(df)} proteins")

    # recompute the 11 sequence features
    rows = []
    for s in tqdm(df["sequence"], desc="sequence features"):
        cs = clean_seq(s)
        if not cs:
            rows.append({k: np.nan for k in SEQ_FEATS}); continue
        try:
            f = sequence_features(cs)
            rows.append({k: f.get(k) for k in SEQ_FEATS})
        except Exception:
            rows.append({k: np.nan for k in SEQ_FEATS})
    seqf = pd.DataFrame(rows)

    # assemble: Entry + domain + 16 consistent features + scores
    out = pd.DataFrame({"Entry": df["Entry"], "domain": df["domain"]})
    for k in SEQ_FEATS:
        out[k] = seqf[k].values
    for k in STRUCT_FEATS:
        out[k] = df[k].values            # reused (already match the extractor)
    # carry through EVERY model-score column so the R GAM landscapes all work
    score_cols = [c for c in df.columns if c.endswith("_score")]
    for k in score_cols:
        out[k] = df[k].values
    print(f"carried {len(score_cols)} score columns: {score_cols}")

    OUT.parent.mkdir(exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"\nWrote {OUT}  ({out.shape[0]} rows, {out.shape[1]} cols)")

    # sanity: completeness of the 16 features
    miss = {k: int(out[k].isna().sum()) for k in MIXED_FEATURES if out[k].isna().sum()}
    print("Missing per feature:", miss or "none")
    return out


if __name__ == "__main__":
    main()
