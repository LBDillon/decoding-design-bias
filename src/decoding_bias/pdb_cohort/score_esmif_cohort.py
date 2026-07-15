"""
score_esmif_cohort.py - ESM-IF likelihood scoring for the 876 experimental-PDB
cohort, matching the paper method exactly:

    coords, _ = esm.inverse_folding.util.load_coords(chain_pdb, chain)
    ll_fullseq, ll_withcoord = esm.inverse_folding.util.score_sequence(
        model, alphabet, coords, sequence)
    esmif_score = ll_fullseq          # per-residue mean log-likelihood, higher = better

Run in the `base` conda env (has torch + esm + torch_geometric):
    ~/miniforge3/bin/python design/score_esmif_cohort.py --limit 3     # smoke test
    ~/miniforge3/bin/python design/score_esmif_cohort.py               # full 876

Input : design/outputs/independent_cohort/cohort_pdb_scoring_inputs.csv
Output: design/outputs/independent_cohort/esmif_scores_pdb.csv   (resumable)
"""
import argparse, os, sys, traceback
from pathlib import Path
import numpy as np, pandas as pd
from tqdm import tqdm
import torch, esm

REPO = Path(__file__).resolve().parent.parent
COH = REPO / "design" / "outputs" / "independent_cohort" / "cohort_pdb_scoring_inputs.csv"
OUT = REPO / "design" / "outputs" / "independent_cohort" / "esmif_scores_pdb.csv"


def pdb_chain_id(path):
    """The extracted single-chain PDBs are relabelled to 'A'; read the actual
    chain letter from the file rather than trusting the original `chain` column."""
    with open(path) as fh:
        for line in fh:
            if line.startswith(("ATOM", "HETATM")):
                return line[21]
    return "A"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="score only first N (smoke test)")
    args = ap.parse_args()

    df = pd.read_csv(COH)
    df["Entry"] = df["Entry"].astype(str)

    done = set()
    if OUT.exists():
        prev = pd.read_csv(OUT)
        done = set(prev["Entry"].astype(str))
        print(f"resume: {len(done)} already scored")
    todo = df[~df["Entry"].isin(done)]
    if args.limit:
        todo = todo.head(args.limit)
    print(f"to score: {len(todo)} / {len(df)}")

    device = "cpu"   # GVP/torch_scatter are most reliable on CPU on macOS
    print("loading esm_if1_gvp4_t16_142M_UR50 ...")
    model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
    model = model.eval().to(device)

    write_header = not OUT.exists()
    n_ok = n_err = 0
    for _, r in tqdm(todo.iterrows(), total=len(todo), desc="ESM-IF"):
        rec = {"Entry": r["Entry"], "pdb_id": r.get("pdb_id"), "chain": r.get("chain"),
               "esmif_score": np.nan, "esmif_ll_withcoord": np.nan,
               "scored_length": np.nan, "valid_positions": np.nan, "error": ""}
        try:
            chain_in_file = pdb_chain_id(r["chain_pdb_path"])
            coords, _ = esm.inverse_folding.util.load_coords(r["chain_pdb_path"], chain_in_file)
            seq = str(r["sequence"])
            ll_fullseq, ll_withcoord = esm.inverse_folding.util.score_sequence(
                model, alphabet, coords, seq)
            coord_mask = np.all(np.isfinite(coords), axis=(-1, -2))
            rec.update(esmif_score=float(ll_fullseq), esmif_ll_withcoord=float(ll_withcoord),
                       scored_length=len(seq), valid_positions=int(coord_mask.sum()))
            n_ok += 1
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {e}"
            n_err += 1
        pd.DataFrame([rec]).to_csv(OUT, mode="a", header=write_header, index=False)
        write_header = False

    print(f"\ndone: {n_ok} ok, {n_err} errors -> {OUT}")


if __name__ == "__main__":
    main()
