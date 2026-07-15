"""
R3.3 sensitivity check - prepare experimental-PDB scoring inputs for the v12
proteins that have an experimental structure (has_pdb_struct == True, n=1602).

Solves the indexing issue: a PDB chain usually covers only part of the UniProt
sequence (truncations, engineered mutations, modified residues, different
numbering). We therefore score CHAIN-vs-CHAIN: for each protein we
  1. fetch the PDB (mmCIF; cached) - large assemblies (e.g. ribosomes) are flagged
     and skipped for tractability,
  2. auto-select the chain that best matches the UniProt sequence,
  3. extract that single chain's coordinates -> a scorable single-chain PDB,
  4. take the chain's observed sequence (modified residues mapped to parent;
     MSE->M etc.) as the sequence to score,
  5. align to UniProt (offset + % identity) and filter.

Output: design/outputs/pdb_scoring_inputs.csv  (+ chain PDBs in pdb_chain_structs/)
Each row: Entry, domain, species, protein_family, pdb_id, pdb_chain,
chain_pdb_path, sequence (= PDB chain seq, for scoring), pdb_identity, offset,
uniprot_len, chain_len, + the v12 AF2 model scores (for the AF2-vs-PDB comparison).
The user then re-runs the model likelihood scoring on chain_pdb_path + sequence.
"""
import sys, os, warnings, tempfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import biotite.database.rcsb as rcsb
import biotite.structure.io.pdbx as pdbx
import biotite.structure.io.pdb as pdbio
import biotite.structure as struc
from biotite.sequence import ProteinSequence

REPO = Path(__file__).resolve().parent.parent
META = REPO / "dataset_update" / "main_plus_r2_r3_metadata_v12.csv"
ANA  = REPO / "dataset_update" / "main_plus_r2_r3_analysis_v12_corrected.csv"
OUT  = Path(__file__).resolve().parent / "outputs"
CIF_CACHE = OUT / "_cif_cache"; CIF_CACHE.mkdir(parents=True, exist_ok=True)
CHAIN_DIR = OUT / "pdb_chain_structs"; CHAIN_DIR.mkdir(parents=True, exist_ok=True)
PARTIAL = OUT / "pdb_scoring_inputs.partial.csv"

IDENTITY_MIN = 0.90
MAX_CIF_MB = 15            # skip giant assemblies (ribosomes) for tractability
MODIFIED = {'MSE':'M','SEP':'S','TPO':'T','PTR':'Y','HYP':'P','PCA':'E','CME':'C',
            'CSO':'C','KCX':'K','MLY':'K','LLP':'K','CSD':'C','OCS':'C','CAS':'C'}
SCORE_COLS = ["proteinmpnn_score","solublempnn_score","esmif_score","mif_score","mifst_score",
              "caliby_score","soluble_caliby_score","triflow_score","esm3_struct_cond_score",
              "esm3_seq_only_score","ESM2_15B_pppl_score","carp_640M_score"]


def three_to_one(rn):
    try: return ProteinSequence.convert_letter_3to1(rn)
    except Exception: return MODIFIED.get(rn, 'X')


def best_alignment(uni, chain):
    nC, nU = len(chain), len(uni)
    if nC == 0: return 0, 0.0
    if nC > nU:
        bo, bi = 0, 0.0
        for o in range(nC-nU+1):
            w = chain[o:o+nU]; idn = sum(x==y for x,y in zip(w,uni))/nU
            if idn>bi: bi,bo = idn,-o
        return bo, bi
    bo, bi = 0, 0.0
    for o in range(nU-nC+1):
        w = uni[o:o+nC]; idn = sum(x==y for x,y in zip(w,chain))/nC
        if idn>bi: bi,bo = idn,o
    return bo, bi


def process(row):
    e, pdb_id, uni = row["Entry"], str(row["pdb_id"]).strip().upper(), row["sequence"]
    out = {"Entry": e, "pdb_id": pdb_id, "fail_reason": None}
    try:
        cif = CIF_CACHE / f"{pdb_id}.cif"
        if not cif.exists():
            rcsb.fetch(pdb_id, "cif", str(CIF_CACHE))
        if cif.stat().st_size > MAX_CIF_MB*1e6:
            out["fail_reason"] = "large_assembly_skipped"; return out
        arr = pdbx.get_structure(pdbx.CIFFile.read(str(cif)), model=1)
        aa = arr[struc.filter_amino_acids(arr)]
        # per-chain sequence + identity; pick best chain
        best = (None, 0.0, 0, None)
        for ch in sorted(set(aa.chain_id)):
            ca = aa[(aa.chain_id==ch) & (aa.atom_name=="CA")]
            if ca.array_length() == 0: continue
            seq = "".join(three_to_one(r) for r in ca.res_name)
            off, idn = best_alignment(uni, seq)
            if idn > best[1]: best = (ch, idn, off, seq)
        ch, idn, off, seq = best
        if ch is None:
            out["fail_reason"] = "no_protein_chain"; return out
        out.update(pdb_chain=ch, pdb_identity=round(idn,4), offset=off,
                   chain_len=len(seq), uniprot_len=len(uni),
                   has_x=("X" in seq), chain_seq=seq)
        if idn < IDENTITY_MIN or "X" in seq:
            out["fail_reason"] = f"identity {idn:.2f}" if idn<IDENTITY_MIN else "nonstandard_X"
            return out
        # write the single chain as a scorable PDB
        chain_atoms = aa[aa.chain_id == ch]
        cp = CHAIN_DIR / f"{e}_{pdb_id}_{ch}.pdb"
        pf = pdbio.PDBFile(); pf.set_structure(chain_atoms); pf.write(str(cp))
        out["chain_pdb_path"] = str(cp)
    except Exception as ex:
        out["fail_reason"] = f"error:{str(ex)[:40]}"
    return out


def main():
    m = pd.read_csv(META, low_memory=False)
    sub = m[m["has_pdb_struct"] == True][["Entry","pdb_id","sequence","domain","species","protein_family"]].dropna(subset=["pdb_id","sequence"])
    done = set()
    if PARTIAL.exists():
        done = set(pd.read_csv(PARTIAL)["Entry"]); print(f"resuming, {len(done)} done")
    todo = sub[~sub.Entry.isin(done)].reset_index(drop=True)
    print(f"PDB subset: {len(sub)}  | to process: {len(todo)}")

    results = []
    if PARTIAL.exists(): results = pd.read_csv(PARTIAL).to_dict("records")
    rowdicts = todo.to_dict("records")
    from tqdm import tqdm
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(process, r): r for r in rowdicts}
        for i, f in enumerate(tqdm(as_completed(futs), total=len(futs), desc="PDB prep")):
            r = futs[f]; res = f.result()
            res.update(domain=r["domain"], species=r["species"], protein_family=r["protein_family"])
            results.append(res)
            if i % 50 == 0:
                pd.DataFrame(results).to_csv(PARTIAL, index=False)
    info = pd.DataFrame(results)

    ok = info[info.fail_reason.isna() & info.chain_pdb_path.notna()].copy()
    # attach AF2 scores for the AF2-vs-PDB comparison; rename chain seq -> sequence
    a = pd.read_csv(ANA, low_memory=False)[["Entry"]+[c for c in SCORE_COLS if c]]
    final = ok.merge(a, on="Entry", how="left").rename(columns={"chain_seq":"sequence"})
    final.to_csv(OUT / "pdb_scoring_inputs.csv", index=False)
    info.to_csv(OUT / "pdb_prep_diagnostics.csv", index=False)
    if PARTIAL.exists(): PARTIAL.unlink()

    print(f"\nUsable PDB scoring inputs: {len(final)}")
    print("By domain:"); print(final.domain.value_counts().to_string())
    print("\nReasons excluded:"); print(info.fail_reason.value_counts(dropna=True).to_string())
    print(f"\nWrote pdb_scoring_inputs.csv (sequence = PDB chain seq; chain PDBs in {CHAIN_DIR.name}/)")


if __name__ == "__main__":
    main()
