"""
Enrich designs_ph_features.csv so the notebook's surface acid-base design lens fires.

Adds the four folded-structure surface features to every design row and appends a
WT row per template (model="WT") carrying both the whole-sequence pH features (from
the WT sequence) and the surface features (from the WT ColabFold fold). The surface
design PCA in the notebook fits on the design table and anchors on these WT rows.

  python design/build_design_acidbase_inputs.py
"""
import sys, glob, shutil
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from build_design_ph_axis_inputs import ph_features          # WT pH from sequence
from surface_features_alkaline import one_structure          # canonical SASA surface

PH = HERE / "outputs" / "designs_ph_features.csv"
SURF = HERE / "outputs" / "designs_surface_features.csv"
WT_PDB_DIR = HERE / "arc_downloads" / "rank001_flat"
SURF_COLS = ["surface_acidic_fraction", "surface_basic_fraction",
             "surface_net_charge", "surface_ionizable_fraction"]


def surf_features_from_pdb(path):
    d = one_structure(path)
    return {"surface_acidic_fraction": d["surf_acidic"],
            "surface_basic_fraction": d["surf_basic"],
            "surface_net_charge": d["surf_net_KR_DE"],
            "surface_ionizable_fraction": d["surf_acidic"] + d["surf_basic"]}


def main():
    ph = pd.read_csv(PH)
    surf = pd.read_csv(SURF)[["uniprot_id", "model", "sample_idx"] + SURF_COLS]
    designs = ph.merge(surf, on=["uniprot_id", "model", "sample_idx"], how="left")
    miss = designs[SURF_COLS[0]].isna().sum()
    print(f"designs: {len(designs)} rows, {miss} without surface features")

    # one WT per template: pH from the WT sequence, surface from the WT fold
    wt_seq = ph.drop_duplicates("uniprot_id")[["uniprot_id", "wt_sequence"]]
    wt_pdb = {Path(p).name.split("__")[0]: p
              for p in glob.glob(str(WT_PDB_DIR / "*__WT_unrelaxed_rank_001*.pdb"))}
    rows = []
    for r in wt_seq.itertuples(index=False):
        rec = {"uniprot_id": r.uniprot_id, "model": "WT", "sample_idx": np.nan,
               "wt_sequence": r.wt_sequence, "designed_sequence": r.wt_sequence}
        rec.update(ph_features(r.wt_sequence))
        p = wt_pdb.get(r.uniprot_id)
        rec.update(surf_features_from_pdb(p) if p else {c: np.nan for c in SURF_COLS})
        rows.append(rec)
    wt = pd.DataFrame(rows)
    print(f"WT rows: {len(wt)}, with fold {sum(u in wt_pdb for u in wt_seq.uniprot_id)}/{len(wt_seq)}")

    out = pd.concat([designs, wt[designs.columns]], ignore_index=True)
    shutil.copy(PH, PH.parent / "designs_ph_features.backup_preacidbase.csv")  # backup original
    out.to_csv(PH, index=False)
    print(f"\nwrote {PH}  ({len(out)} rows: {len(designs)} designs + {len(wt)} WT)")
    print("surface cols per model (mean surface_net_charge):")
    print(out.groupby("model")["surface_net_charge"].mean().round(4).to_string())


if __name__ == "__main__":
    main()
