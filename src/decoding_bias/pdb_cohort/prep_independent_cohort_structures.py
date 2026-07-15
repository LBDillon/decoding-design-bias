"""Prepare scoring inputs for the INDEPENDENT experimental-PDB cohort (R3.4).

Takes design/outputs/independent_cohort/cohort_scoring_inputs.csv (the 1,401
matched representatives, each with a known pdb_id + auth chain id) and, for each:
  1. fetches the mmCIF (cached),
  2. extracts the named chain's coordinates,
  3. takes the RESOLVED-coordinate sequence (modified residues -> parent; MSE->M),
  4. relabels the chain to 'A' and writes a single-chain PDB (PDB_MODE convention),
  5. drops rows with nonstandard residues ('X') or resolved length < 50.

The resolved chain sequence becomes the `sequence` column to score, so the chain
PDB residues equal the scored sequence (no missing-coordinate mismatch), matching
design/PDB_SENSITIVITY_README.md. `Entry` = entity_id (unique key). No AF2
reference scores are merged: this is an independent cohort, scored standalone and
compared at the cohort level (variance-decomposition / ELO), not per-protein.

Outputs (design/outputs/independent_cohort/):
  cohort_pdb_scoring_inputs.csv   Entry, pdb_id, chain, sequence, chain_pdb_path,
                                  domain, species_collapsed, protein_family,
                                  broad_function, resolution_A, resolved_len, seqres_len
  cohort_chain_structs/           single-chain PDBs (chain 'A')
  cohort_prep_diagnostics.csv     per-entity fail_reason
"""
import sys, warnings
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
warnings.filterwarnings("ignore")
import pandas as pd
import biotite.database.rcsb as rcsb
import biotite.structure.io.pdbx as pdbx
import biotite.structure.io.pdb as pdbio
import biotite.structure as struc
from biotite.sequence import ProteinSequence

HERE = Path(__file__).resolve().parent
COH = HERE / "outputs" / "independent_cohort"
IN_CSV = COH / "cohort_scoring_inputs.csv"
CIF_CACHE = COH / "_cif_cache"; CIF_CACHE.mkdir(parents=True, exist_ok=True)
CHAIN_DIR = COH / "cohort_chain_structs"; CHAIN_DIR.mkdir(parents=True, exist_ok=True)
PARTIAL = COH / "cohort_pdb_scoring_inputs.partial.csv"

MIN_LEN = 50
MAX_CIF_MB = 15
MODIFIED = {'MSE':'M','SEP':'S','TPO':'T','PTR':'Y','HYP':'P','PCA':'E','CME':'C',
            'CSO':'C','KCX':'K','MLY':'K','LLP':'K','CSD':'C','OCS':'C','CAS':'C'}
STD_AA = set("ACDEFGHIKLMNPQRSTVWY")
META_COLS = ["domain", "species_collapsed", "protein_family", "broad_function", "resolution_A"]


def three_to_one(rn):
    try:
        return ProteinSequence.convert_letter_3to1(rn)
    except Exception:
        return MODIFIED.get(rn, 'X')


def process(row):
    e = row["entity_id"]
    pdb_id = str(row["pdb_id"]).strip().upper()
    want_chain = str(row["chain"]).strip()
    out = {"Entry": e, "pdb_id": pdb_id, "chain_req": want_chain, "fail_reason": None}
    try:
        cif = CIF_CACHE / f"{pdb_id}.cif"
        if not cif.exists():
            rcsb.fetch(pdb_id, "cif", str(CIF_CACHE))
        if cif.stat().st_size > MAX_CIF_MB * 1e6:
            out["fail_reason"] = "large_assembly_skipped"; return out
        arr = pdbx.get_structure(pdbx.CIFFile.read(str(cif)), model=1)
        aa = arr[struc.filter_amino_acids(arr)]

        # prefer the GraphQL auth chain; fall back to the single protein chain present
        chains = sorted(set(aa.chain_id))
        ch = want_chain if want_chain in chains else (chains[0] if len(chains) == 1 else None)
        if ch is None:
            out["fail_reason"] = f"chain_{want_chain}_absent"; return out

        ca = aa[(aa.chain_id == ch) & (aa.atom_name == "CA")]
        if ca.array_length() == 0:
            out["fail_reason"] = "no_CA"; return out
        seq = "".join(three_to_one(r) for r in ca.res_name)

        out.update(chain=ch, resolved_len=len(seq), seqres_len=len(str(row.get("sequence") or "")))
        if set(seq) - STD_AA:
            out["fail_reason"] = "nonstandard_residue"; return out
        if len(seq) < MIN_LEN:
            out["fail_reason"] = f"resolved_len_{len(seq)}<{MIN_LEN}"; return out

        chain_atoms = aa[aa.chain_id == ch]
        chain_atoms.chain_id[:] = "A"  # PDB_MODE convention
        cp = CHAIN_DIR / f"{e}_{pdb_id}_{ch}.pdb"
        pf = pdbio.PDBFile(); pf.set_structure(chain_atoms); pf.write(str(cp))
        out["chain_pdb_path"] = str(cp)
        out["sequence"] = seq
    except Exception as ex:
        out["fail_reason"] = f"error:{str(ex)[:50]}"
    return out


def main():
    df = pd.read_csv(IN_CSV)
    meta = df.set_index("entity_id")[META_COLS]
    done = set()
    if PARTIAL.exists():
        done = set(pd.read_csv(PARTIAL)["Entry"]); print(f"resuming, {len(done)} done")
    todo = df[~df["entity_id"].isin(done)].reset_index(drop=True)
    print(f"cohort: {len(df)}  | to process: {len(todo)}")

    results = pd.read_csv(PARTIAL).to_dict("records") if PARTIAL.exists() else []
    rowdicts = todo.to_dict("records")
    from tqdm import tqdm
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(process, r) for r in rowdicts]
        for i, f in enumerate(tqdm(as_completed(futs), total=len(futs), desc="cohort PDB prep")):
            results.append(f.result())
            if i % 50 == 0:
                pd.DataFrame(results).to_csv(PARTIAL, index=False)
    info = pd.DataFrame(results)

    ok = info[info.fail_reason.isna() & info.get("chain_pdb_path").notna()].copy()
    ok = ok.merge(meta, left_on="Entry", right_index=True, how="left")
    keep = ["Entry", "pdb_id", "chain", "sequence", "chain_pdb_path",
            "domain", "species_collapsed", "protein_family", "broad_function",
            "resolution_A", "resolved_len", "seqres_len"]
    ok[keep].to_csv(COH / "cohort_pdb_scoring_inputs.csv", index=False)
    info.to_csv(COH / "cohort_prep_diagnostics.csv", index=False)
    if PARTIAL.exists(): PARTIAL.unlink()

    print(f"\nUsable scoring inputs: {len(ok)} / {len(df)}")
    print("By domain:"); print(ok["domain"].value_counts().to_string())
    print("\nReasons excluded:"); print(info.fail_reason.value_counts(dropna=True).to_string())
    print(f"\nWrote cohort_pdb_scoring_inputs.csv (sequence = resolved chain seq; chain PDBs in {CHAIN_DIR.name}/)")


if __name__ == "__main__":
    main()
