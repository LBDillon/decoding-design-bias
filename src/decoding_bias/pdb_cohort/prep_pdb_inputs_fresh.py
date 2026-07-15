"""
Prepare experimental-PDB scoring inputs from the FRESH UniProt scan
(pdb_availability.csv). Uses the UniProt-provided best PDB + chain (no guessing):
fetch the structure, extract that chain's coordinates -> single-chain PDB, take
the chain's observed sequence, align to UniProt, filter.

Output columns are compatible with notebooks/Calculate_All_models_likelihoods_PDB.ipynb
(Entry, pdb_id, pdb_chain, sequence) plus chain_pdb_path (local single-chain PDB,
robust to mmCIF-only assemblies), diagnostics, and the v12 AF2 scores for the
AF2-vs-PDB comparison.

Output: design/outputs/pdb_scoring_inputs.csv  + chain PDBs in pdb_chain_structs/
"""
import sys, warnings
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import pandas as pd
import biotite.database.rcsb as rcsb
import biotite.structure.io.pdbx as pdbx
import biotite.structure.io.pdb as pdbio
import biotite.structure as struc
# reuse helpers from the auto-chain prep
from build_pdb_scoring_inputs import (three_to_one, best_alignment, MAX_CIF_MB,
                                      CIF_CACHE, CHAIN_DIR, IDENTITY_MIN, SCORE_COLS)

REPO = Path(__file__).resolve().parent.parent
AVAIL = HERE / "outputs" / "pdb_availability.csv"
META  = REPO / "dataset_update" / "main_plus_r2_r3_metadata_v12.csv"
ANA   = REPO / "dataset_update" / "main_plus_r2_r3_analysis_v12_corrected.csv"
OUT   = HERE / "outputs"
PARTIAL = OUT / "pdb_scoring_inputs.partial.csv"
COVERAGE_MIN = 0.50          # require the chosen structure to cover >=50% of the protein
METHODS = {"X-ray"}          # X-ray = clean crystal structures, balanced across domains,
                             # avoids giant EM ribosome assemblies (set {"X-ray","EM","NMR"} to widen)


def process(row):
    e, pdb_id, chain, uni = row["Entry"], str(row["best_pdb_id"]).strip().upper(), \
                            str(row["best_chain"]).strip(), row["sequence"]
    out = {"Entry": e, "pdb_id": pdb_id, "pdb_chain": chain, "fail_reason": None}
    try:
        cif = CIF_CACHE / f"{pdb_id}.cif"
        if not cif.exists():
            rcsb.fetch(pdb_id, "cif", str(CIF_CACHE))
        if cif.stat().st_size > MAX_CIF_MB * 1e6:
            out["fail_reason"] = "large_assembly_skipped"; return out
        arr = pdbx.get_structure(pdbx.CIFFile.read(str(cif)), model=1)
        aa = arr[struc.filter_amino_acids(arr)]
        sel = aa[aa.chain_id == chain]
        if sel.array_length() == 0:                      # chain id mismatch -> auto-select
            best = (None, 0.0, None)
            for ch in sorted(set(aa.chain_id)):
                ca = aa[(aa.chain_id == ch) & (aa.atom_name == "CA")]
                if ca.array_length() == 0: continue
                s = "".join(three_to_one(r) for r in ca.res_name)
                _, idn = best_alignment(uni, s)
                if idn > best[1]: best = (ch, idn, s)
            chain = best[0]; sel = aa[aa.chain_id == chain]; out["pdb_chain"] = chain
        ca = sel[sel.atom_name == "CA"]
        seq = "".join(three_to_one(r) for r in ca.res_name)
        off, idn = best_alignment(uni, seq)
        out.update(pdb_identity=round(idn, 4), offset=off, chain_len=len(seq),
                   uniprot_len=len(uni), chain_seq=seq)
        if idn < IDENTITY_MIN or "X" in seq:
            out["fail_reason"] = f"identity {idn:.2f}" if idn < IDENTITY_MIN else "nonstandard_X"
            return out
        # relabel the extracted chain to 'A' so the single-chain PDB is drop-in for
        # scoring code that assumes chain 'A' (original id kept in orig_chain).
        out["orig_chain"] = chain
        sel = sel.copy(); sel.chain_id[:] = "A"
        cp = CHAIN_DIR / f"{e}_{pdb_id}_{chain}.pdb"
        pf = pdbio.PDBFile(); pf.set_structure(sel); pf.write(str(cp))
        out["chain_pdb_path"] = str(cp); out["pdb_chain"] = "A"
    except Exception as ex:
        out["fail_reason"] = f"error:{str(ex)[:40]}"
    return out


def main():
    av = pd.read_csv(AVAIL)
    av = av[av.has_pdb & (av.coverage_frac >= COVERAGE_MIN) & (av.best_method.isin(METHODS))].copy()
    seqs = pd.read_csv(META, low_memory=False).set_index("Entry")["sequence"]
    av["sequence"] = av.Entry.map(seqs)
    av = av.dropna(subset=["sequence", "best_pdb_id", "best_chain"])
    print(f"Candidates (has_pdb, coverage>={COVERAGE_MIN}): {len(av)}")

    done, results = set(), []
    if PARTIAL.exists():
        results = pd.read_csv(PARTIAL).to_dict("records")
        done = {r["Entry"] for r in results}
    todo = av[~av.Entry.isin(done)].to_dict("records")
    from tqdm import tqdm
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(process, r) for r in todo]
        for i, f in enumerate(tqdm(as_completed(futs), total=len(futs), desc="PDB chain prep")):
            results.append(f.result())
            if i % 50 == 0:
                pd.DataFrame(results).to_csv(PARTIAL, index=False)
    info = pd.DataFrame(results).merge(
        av[["Entry", "domain", "coverage_frac", "best_method", "best_resolution_A"]], on="Entry", how="left")

    ok = info[info.fail_reason.isna() & info.chain_pdb_path.notna()].copy()
    a = pd.read_csv(ANA, low_memory=False)[["Entry", "species", "protein_family"] + [c for c in SCORE_COLS]]
    final = ok.merge(a, on="Entry", how="left").rename(columns={"chain_seq": "sequence"})
    keep = (["Entry", "pdb_id", "pdb_chain", "sequence", "chain_pdb_path", "domain",
             "species", "protein_family", "pdb_identity", "offset", "chain_len",
             "uniprot_len", "coverage_frac", "best_method", "best_resolution_A"]
            + [c for c in SCORE_COLS])
    final[[c for c in keep if c in final.columns]].to_csv(OUT / "pdb_scoring_inputs.csv", index=False)
    info.to_csv(OUT / "pdb_prep_diagnostics.csv", index=False)
    if PARTIAL.exists(): PARTIAL.unlink()

    print(f"\nUsable PDB scoring inputs: {len(final)}")
    print("By domain:"); print(final.domain.value_counts().to_string())
    print("By method:"); print(final.best_method.value_counts().to_string())
    print("Excluded reasons:"); print(info.fail_reason.value_counts(dropna=True).to_string())
    print(f"\nWrote pdb_scoring_inputs.csv (Entry/pdb_id/pdb_chain/sequence + chain_pdb_path)")


if __name__ == "__main__":
    main()
