"""
After folding all_to_fold.fasta with localColabFold on ARC, compute the 16
v12 mixed_features for every folded structure and emit the two CSVs the
PCA-projection R cells and the physicochemical analysis consume:

    designs_features.csv   (one row per design  : uniprot_id, model, sample_idx,
                            domain, rank_class + 16 features)
    wt_features.csv        (one row per WT, folded by the SAME predictor)

Usage (on ARC after folding, or locally after downloading the PDBs):
    python extract_design_features.py \
        --pdb-dir  colabfold_out \
        --fasta    outputs/all_to_fold.fasta \
        --manifest outputs/all_to_fold.manifest.csv \
        --out-dir  outputs
"""
import argparse, glob, os
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from features_for_designs import compute_mixed_features, MIXED_FEATURES


def read_fasta(path):
    seqs, h = {}, None
    for line in open(path):
        line = line.rstrip()
        if line.startswith(">"):
            h = line[1:].split()[0]; seqs[h] = ""
        elif h:
            seqs[h] += line
    return seqs


def find_pdb(pdb_dir, fold_id):
    # localColabFold: <id>_unrelaxed_rank_001_*.pdb  (fall back to any rank_001 / .pdb)
    for pat in (f"{fold_id}_*rank_001*.pdb", f"{fold_id}_*rank_1*.pdb", f"{fold_id}*.pdb"):
        hits = sorted(glob.glob(os.path.join(pdb_dir, pat)))
        if hits:
            return hits[0]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdb-dir", required=True)
    ap.add_argument("--fasta", default="outputs/all_to_fold.fasta")
    ap.add_argument("--manifest", default="outputs/all_to_fold.manifest.csv")
    ap.add_argument("--out-dir", default="outputs")
    args = ap.parse_args()

    seqs = read_fasta(args.fasta)
    man = pd.read_csv(args.manifest)
    out = Path(args.out_dir); out.mkdir(exist_ok=True)

    rows, missing = [], []
    for r in tqdm(list(man.itertuples()), desc="features"):
        pdb = find_pdb(args.pdb_dir, r.fold_id)
        if pdb is None or r.fold_id not in seqs:
            missing.append(r.fold_id); continue
        try:
            feats = compute_mixed_features(seqs[r.fold_id], pdb)
        except Exception as e:
            missing.append(f"{r.fold_id} ({e})"); continue
        rec = {"uniprot_id": r.uniprot_id, "model": r.model, "sample_idx": r.sample_idx,
               "domain": r.domain, "rank_class": r.rank_class, "is_wt": r.is_wt}
        rec.update(feats)
        rows.append(rec)

    allf = pd.DataFrame(rows)
    designs = allf[~allf.is_wt].drop(columns=["is_wt"])
    wt = allf[allf.is_wt].drop(columns=["is_wt", "model", "sample_idx"])

    designs.to_csv(out / "designs_features.csv", index=False)
    wt.to_csv(out / "wt_features.csv", index=False)
    print(f"\nWrote {out/'designs_features.csv'} ({len(designs)} designs)")
    print(f"Wrote {out/'wt_features.csv'} ({len(wt)} WTs)")
    if missing:
        print(f"\n[WARN] {len(missing)} structures missing/failed:", missing[:8],
              "…" if len(missing) > 8 else "")
    # completeness check
    miss_feat = {k: int(designs[k].isna().sum()) for k in MIXED_FEATURES if designs[k].isna().sum()}
    print("Missing feature values (designs):", miss_feat or "none")


if __name__ == "__main__":
    main()
