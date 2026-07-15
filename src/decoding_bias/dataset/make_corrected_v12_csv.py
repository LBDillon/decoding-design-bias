"""
Produce a DROP-IN replacement for main_plus_r2_r3_analysis_v12.csv with all
sequence-derived features recomputed by the clean extractor (src/features).
Every other column (domain, species, structural features, all model scores,
metadata) is kept byte-for-byte. Point the R PCA notebook's BIG_CSV at the
output and re-run - only the sequence features change.

Output: dataset_update/main_plus_r2_r3_analysis_v12_corrected.csv
"""
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np, pandas as pd
from tqdm import tqdm
from decoding_bias.features.sequence_features import calculate_sequence_features

ANA  = REPO / "dataset_update" / "main_plus_r2_r3_analysis_v12.csv"
META = REPO / "dataset_update" / "main_plus_r2_r3_metadata_v12.csv"
OUT  = REPO / "dataset_update" / "main_plus_r2_r3_analysis_v12_corrected.csv"

# analysis_v12 column  ->  extractor key
COLMAP = {
    "sequence_length": "sequence_length", "mw_per_residue": "mw_per_residue",
    "isoelectric_point": "isoelectric_point", "charge_at_ph7": "charge_at_pH7",
    "charge_per_residue": "charge_per_residue", "buffer_capacity": "buffer_capacity",
    "basic_residue_fraction": "basic_residue_fraction",
    "acidic_residue_fraction": "acidic_residue_fraction",
    "gravy": "gravy", "aromaticity": "aromaticity",
    "hydrophobic_fraction": "hydrophobic_fraction",
    "instability_index": "instability_index", "proline_fraction": "proline_fraction",
    "small_residue_fraction": "small_residue_fraction",
}


def clean_seq(s):
    return s.upper().replace("U", "C").replace("X", "") if isinstance(s, str) else ""


def main():
    ana = pd.read_csv(ANA, low_memory=False)
    seqs = pd.read_csv(META, low_memory=False).set_index("Entry")["sequence"]
    ana["_seq"] = ana["Entry"].map(seqs)

    refreshed = {c: [] for c in COLMAP}
    for s in tqdm(ana["_seq"], desc="recomputing sequence features"):
        cs = clean_seq(s)
        f = calculate_sequence_features(cs) if cs else {}
        for col, key in COLMAP.items():
            refreshed[col].append(f.get(key, np.nan) if f else np.nan)

    # report the change before overwriting
    print("\nColumn changes (mean relative |Δ| over rows with both values):")
    corrected = ana.copy()
    for col in COLMAP:
        if col not in ana.columns:
            print(f"  {col:26} (not in analysis_v12 - adding)")
            corrected[col] = refreshed[col]; continue
        old = ana[col].astype(float); new = pd.Series(refreshed[col], index=ana.index)
        both = old.notna() & new.notna() & (old.abs() > 1e-9)
        rel = ((new[both] - old[both]).abs() / old[both].abs()).mean()
        tag = "unchanged" if rel < 0.005 else ("CORRECTED" if rel > 0.05 else "minor")
        print(f"  {col:26} mean rel Δ = {rel:6.1%}   {tag}")
        corrected[col] = refreshed[col]

    corrected = corrected.drop(columns=["_seq"])
    corrected.to_csv(OUT, index=False)
    print(f"\nWrote {OUT}")
    print(f"  shape {corrected.shape} (same rows/cols as original: "
          f"{corrected.shape == ana.drop(columns=['_seq']).shape})")


if __name__ == "__main__":
    main()
