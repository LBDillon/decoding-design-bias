"""
Collect every design sequence (+ the 25 WTs) into ONE FASTA for folding on ARC
with localColabFold (single-sequence mode). Also writes a manifest mapping each
FASTA id back to its metadata, used by extract_design_features.py afterwards.

    cd design && python make_fold_fasta.py --csv-dirs ~/Downloads outputs

FASTA headers (the id colabfold preserves in output filenames):
    <uniprot>__<model>__s<sample>     for designs
    <uniprot>__WT                     for wild types
"""
import argparse, glob, os, re
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent


def sanitize(s):  # colabfold-safe id (no spaces / odd chars)
    return re.sub(r"[^A-Za-z0-9_.-]", "-", str(s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-dirs", nargs="+", default=[str(Path.home() / "Downloads"), "outputs"],
                    help="dirs to scan for designs_*.csv")
    ap.add_argument("--exclude", nargs="*", default=["designs_MIF.csv"],
                    help="filenames to skip (e.g. the broken pre-fix MIF)")
    ap.add_argument("--out", default="outputs/all_to_fold.fasta")
    args = ap.parse_args()

    # gather design CSVs
    csvs = []
    for d in args.csv_dirs:
        csvs += glob.glob(os.path.join(os.path.expanduser(d), "designs_*.csv"))
    csvs = [c for c in csvs if os.path.basename(c) not in args.exclude]
    print("Design CSVs:", [os.path.basename(c) for c in csvs])

    rows, seen = [], set()
    for c in csvs:
        df = pd.read_csv(c)
        for r in df.itertuples():
            key = (r.uniprot_id, r.model, r.sample_idx)
            if key in seen:
                continue
            seen.add(key)
            did = f"{sanitize(r.uniprot_id)}__{sanitize(r.model)}__s{r.sample_idx}"
            rows.append({"fold_id": did, "uniprot_id": r.uniprot_id, "model": r.model,
                         "sample_idx": r.sample_idx, "domain": r.domain,
                         "rank_class": r.rank_class, "is_wt": False,
                         "sequence": r.designed_sequence})

    # add WTs (folded by the SAME predictor -> predictor-consistent shift)
    inp = pd.read_csv(HERE / "design_input_proteins.csv")
    for r in inp.itertuples():
        did = f"{sanitize(r.uniprot_id)}__WT"
        rows.append({"fold_id": did, "uniprot_id": r.uniprot_id, "model": "WT",
                     "sample_idx": -1, "domain": r.domain, "rank_class": r.rank_class,
                     "is_wt": True, "sequence": r.wt_sequence})

    man = pd.DataFrame(rows)
    out_fa = HERE / args.out
    out_fa.parent.mkdir(exist_ok=True)
    with open(out_fa, "w") as fh:
        for r in man.itertuples():
            fh.write(f">{r.fold_id}\n{r.sequence}\n")
    man.drop(columns=["sequence"]).to_csv(out_fa.with_suffix(".manifest.csv"), index=False)

    print(f"\nWrote {out_fa}")
    print(f"  {len(man)} sequences  ({(~man.is_wt).sum()} designs + {man.is_wt.sum()} WTs)")
    print(f"  by model:\n{man.model.value_counts().to_string()}")
    print(f"Manifest: {out_fa.with_suffix('.manifest.csv')}")


if __name__ == "__main__":
    main()
