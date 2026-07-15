"""
Surface-charge features for EVERY folded design (cross-models + 002 FT + 020 FT).

Uses the paper's canonical surface definition (surface_features_alkaline.one_structure):
biotite Shrake-Rupley SASA, Tien 2013 maxASA, surface = relative SASA >= 0.25, chain A.
For each rank-1 design PDB it emits:
  surface_acidic_fraction     = (D+E)/n_surf
  surface_basic_fraction      = (K+R+H)/n_surf
  surface_net_charge          = (K+R-D-E)/n_surf
  surface_ionizable_fraction  = (D+E+K+R+H)/n_surf
keyed by (uniprot_id, model, sample_idx) so it joins designs_features.csv.

  python design/compute_design_surface_charge.py            # write CSV
  python design/compute_design_surface_charge.py --merge    # also merge into designs_features.csv
"""
import sys, re, glob, argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from surface_features_alkaline import one_structure  # canonical SASA surface composition

PDB_DIRS = [
    HERE / "arc_downloads" / "rank001_flat",      # 7 cross-models (incl MIF-ST) + WT
    HERE / "outputs" / "colabfold_out_ft",        # 002 FT: AlkSecMPNN/AcidSecMPNN/ProteinMPNN_v002
    HERE / "outputs" / "colabfold_out_ft020",     # 020 FT: AlkSecMPNN_020/AcidSecMPNN_020
]
OUT = HERE / "outputs" / "designs_surface_features.csv"
FID = re.compile(r"(.+?)__(.+?)__s(\d+)$")


def fold_id(p):
    return Path(p).name.split("_unrelaxed_rank")[0]


def worker(p):
    fid = fold_id(p)
    if fid.endswith("__WT"):
        return None
    m = FID.match(fid)
    if not m:
        return None
    uni, model, s = m.group(1), m.group(2), int(m.group(3))
    try:
        d = one_structure(p)
    except Exception as ex:
        return dict(uniprot_id=uni, model=model, sample_idx=s, err=str(ex)[:60])
    return dict(uniprot_id=uni, model=model, sample_idx=s,
                surface_acidic_fraction=d["surf_acidic"],
                surface_basic_fraction=d["surf_basic"],
                surface_net_charge=d["surf_net_KR_DE"],
                surface_ionizable_fraction=d["surf_acidic"] + d["surf_basic"],
                n_surf=d["surf_n"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merge", action="store_true",
                    help="left-join the 4 features into design/outputs/designs_features.csv")
    args = ap.parse_args()

    pdbs = []
    for d in PDB_DIRS:
        hits = sorted(glob.glob(str(d / "*rank_001*.pdb")))
        pdbs += hits
        print(f"{len(hits):>5}  {d}")
    print(f"{len(pdbs):>5}  total rank-1 PDBs\n")

    rows, errs = [], []
    with ProcessPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(worker, p) for p in pdbs]
        for i, fut in enumerate(as_completed(futs)):
            r = fut.result()
            if r is None:
                continue
            (errs if "err" in r else rows).append(r)
            if i % 400 == 0:
                print(f"  {i}/{len(pdbs)}", flush=True)

    df = pd.DataFrame(rows).sort_values(["model", "uniprot_id", "sample_idx"]).reset_index(drop=True)
    df.to_csv(OUT, index=False)
    print(f"\nwrote {OUT}  ({len(df)} designs, {len(errs)} errors)")
    print("\nper-model surface_net_charge / acidic / basic / ionizable (mean):")
    print(df.groupby("model")[["surface_net_charge", "surface_acidic_fraction",
                               "surface_basic_fraction", "surface_ionizable_fraction"]]
            .mean().round(4).to_string())
    if errs:
        print("\nerrors (first few):", errs[:3])

    if args.merge:
        feats = ["surface_acidic_fraction", "surface_basic_fraction",
                 "surface_net_charge", "surface_ionizable_fraction"]
        dfm = pd.read_csv(HERE / "outputs" / "designs_features.csv")
        dfm = dfm.drop(columns=[c for c in feats if c in dfm.columns], errors="ignore")
        merged = dfm.merge(df[["uniprot_id", "model", "sample_idx"] + feats],
                           on=["uniprot_id", "model", "sample_idx"], how="left")
        miss = merged[feats[0]].isna().sum()
        merged.to_csv(HERE / "outputs" / "designs_features.csv", index=False)
        print(f"\nmerged into designs_features.csv ({len(merged)} rows, {miss} unmatched)")


if __name__ == "__main__":
    main()
