"""
Build pH-axis feature inputs for design-vs-WT PCA analyses.

The downloaded design CSVs contain raw WT and designed sequences, but the R
PCA notebook expects feature tables. This script computes the sequence-derived
pH/charge features used by the notebook's pH-specific PCA and writes:

  designs_ph_features.csv  one row per design
  wt_ph_features.csv       one row per unique WT protein

No folded structures are required for this pH-axis analysis.
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from decoding_bias.features.sequence_features import calculate_sequence_features


PH_FEATURES = [
    "sequence_length",
    "isoelectric_point",
    "charge_at_ph7",
    "charge_per_residue",
    "buffer_capacity",
    "acidic_residue_fraction",
    "basic_residue_fraction",
    "ionizable_residue_fraction",
]

DESIGN_META = [
    "uniprot_id",
    "species",
    "domain",
    "rank_class",
    "target_cell",
    "model",
    "soluble_variant",
    "sample_idx",
    "seed",
    "temperature",
    "model_score",
    "score_type",
    "structure_path",
]

WT_META = ["uniprot_id", "species", "domain", "rank_class", "target_cell"]


def clean_sequence(seq: object) -> str:
    return str(seq).strip().upper().replace("*", "")


def ph_features(seq: str) -> dict[str, float]:
    raw = calculate_sequence_features(clean_sequence(seq))
    out = {
        "sequence_length": raw.get("sequence_length", np.nan),
        "isoelectric_point": raw.get("isoelectric_point", np.nan),
        "charge_at_ph7": raw.get("charge_at_ph7", raw.get("charge_at_pH7", np.nan)),
        "charge_per_residue": raw.get("charge_per_residue", np.nan),
        "buffer_capacity": raw.get("buffer_capacity", np.nan),
        "acidic_residue_fraction": raw.get("acidic_residue_fraction", np.nan),
        "basic_residue_fraction": raw.get("basic_residue_fraction", np.nan),
        "ionizable_residue_fraction": raw.get("ionizable_residue_fraction", raw.get("ionizable_fraction", np.nan)),
    }
    return out


def discover_design_csvs(design_dir: Path) -> list[Path]:
    candidates = sorted(Path(p) for p in glob.glob(str(design_dir / "designs_*.csv")))
    usable = []
    for path in candidates:
        try:
            cols = pd.read_csv(path, nrows=0).columns
        except Exception:
            continue
        if {"wt_sequence", "designed_sequence", "model", "uniprot_id"}.issubset(cols):
            usable.append(path)
    return usable


def build_tables(csv_paths: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = [pd.read_csv(path) for path in csv_paths]
    raw = pd.concat(frames, ignore_index=True)

    wt_conflicts = (
        raw.groupby("uniprot_id")["wt_sequence"]
        .nunique(dropna=False)
        .loc[lambda s: s > 1]
    )
    if not wt_conflicts.empty:
        raise ValueError(
            "WT sequence conflicts for: "
            + ", ".join(map(str, wt_conflicts.index[:10]))
        )

    design_records = []
    for row in raw.itertuples(index=False):
        rec = {col: getattr(row, col) for col in DESIGN_META if hasattr(row, col)}
        rec["designed_sequence"] = clean_sequence(getattr(row, "designed_sequence"))
        rec["wt_sequence"] = clean_sequence(getattr(row, "wt_sequence"))
        rec.update(ph_features(rec["designed_sequence"]))
        design_records.append(rec)
    designs = pd.DataFrame(design_records)

    wt_base = (
        raw.sort_values(["uniprot_id", "model", "sample_idx"])
        .drop_duplicates("uniprot_id")
        .copy()
    )
    wt_records = []
    for row in wt_base.itertuples(index=False):
        rec = {col: getattr(row, col) for col in WT_META if hasattr(row, col)}
        rec["wt_sequence"] = clean_sequence(getattr(row, "wt_sequence"))
        rec.update(ph_features(rec["wt_sequence"]))
        wt_records.append(rec)
    wt = pd.DataFrame(wt_records)

    ordered_design_cols = [
        col for col in DESIGN_META + ["wt_sequence", "designed_sequence"] + PH_FEATURES
        if col in designs.columns
    ]
    ordered_wt_cols = [
        col for col in WT_META + ["wt_sequence"] + PH_FEATURES
        if col in wt.columns
    ]
    return designs[ordered_design_cols], wt[ordered_wt_cols]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--design-dir",
        default="/Users/lauradillon/Downloads/Designs",
        help="Directory containing designs_*.csv files.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory. Defaults to <design-dir>/ph_axis_features.",
    )
    args = parser.parse_args()

    design_dir = Path(args.design_dir).expanduser()
    out_dir = Path(args.out_dir).expanduser() if args.out_dir else design_dir / "ph_axis_features"
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_paths = discover_design_csvs(design_dir)
    if not csv_paths:
        raise FileNotFoundError(f"No raw designs_*.csv files found in {design_dir}")

    designs, wt = build_tables(csv_paths)
    designs_path = out_dir / "designs_ph_features.csv"
    wt_path = out_dir / "wt_ph_features.csv"
    designs.to_csv(designs_path, index=False)
    wt.to_csv(wt_path, index=False)

    summary = (
        designs.groupby("model", as_index=False)
        .agg(
            n_designs=("uniprot_id", "size"),
            n_wt=("uniprot_id", "nunique"),
            mean_pI=("isoelectric_point", "mean"),
            mean_charge_at_ph7=("charge_at_ph7", "mean"),
            mean_acidic_fraction=("acidic_residue_fraction", "mean"),
            mean_basic_fraction=("basic_residue_fraction", "mean"),
        )
    )
    summary_path = out_dir / "designs_ph_feature_summary.csv"
    summary.to_csv(summary_path, index=False)

    print(f"Read {len(csv_paths)} design CSVs:")
    for path in csv_paths:
        print(f"  - {path}")
    print(f"Wrote {designs_path} ({len(designs)} designs)")
    print(f"Wrote {wt_path} ({len(wt)} WTs)")
    print(f"Wrote {summary_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
