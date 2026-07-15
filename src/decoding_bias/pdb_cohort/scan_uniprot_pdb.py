"""
Fresh, authoritative UniProt->PDB availability scan for all v12 proteins.

Queries each accession's UniProt cross-references, parses every PDB entry
(method, resolution, chains, UniProt coverage range), and picks the best
structure per protein (coverage, then method X-ray>EM>NMR, then resolution).
This supersedes the pre-computed `has_pdb_struct`/`pdb_id` in the metadata
(which has false positives - e.g. P60724 is flagged but has 0 UniProt PDB xrefs)
and gives the exact chain + coverage, removing the chain-guessing heuristic.

Output: design/outputs/pdb_availability.csv
  Entry, uniprot_len, n_pdb, has_pdb, best_pdb_id, best_chain, best_method,
  best_resolution_A, cov_start, cov_end, coverage_frac
"""
import sys, json, re, time, urllib.request, warnings
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
warnings.filterwarnings("ignore")
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
ANA  = REPO / "dataset_update" / "main_plus_r2_r3_analysis_v12_corrected.csv"
META = REPO / "dataset_update" / "main_plus_r2_r3_metadata_v12.csv"
OUT  = Path(__file__).resolve().parent / "outputs"
CACHE = OUT / "_uniprot_pdb_xref_cache.json"
PARTIAL = OUT / "pdb_availability.partial.csv"

METHOD_RANK = {"X-ray": 3, "EM": 2, "NMR": 1}


def parse_chains(s):
    """'O/P/Q/R=1-335, A=10-50' -> [(chain, start, end), ...] (first chain per group)."""
    out = []
    for grp in s.split(","):
        grp = grp.strip()
        if "=" not in grp:
            continue
        chains, rng = grp.split("=", 1)
        m = re.match(r"(-?\d+)-(-?\d+)", rng.strip())
        if not m:
            continue
        first_chain = chains.split("/")[0].strip()
        out.append((first_chain, int(m.group(1)), int(m.group(2))))
    return out


def best_pdb(acc, uni_len, cache):
    if acc in cache:
        recs = cache[acc]
    else:
        try:
            d = json.load(urllib.request.urlopen(
                f"https://rest.uniprot.org/uniprotkb/{acc}.json", timeout=25))
            recs = []
            for x in d.get("uniProtKBCrossReferences", []):
                if x["database"] != "PDB":
                    continue
                p = {q["key"]: q["value"] for q in x.get("properties", [])}
                recs.append({"id": x["id"], "method": p.get("Method", ""),
                             "res": p.get("Resolution", ""), "chains": p.get("Chains", "")})
            cache[acc] = recs
        except Exception as ex:
            cache[acc] = []; recs = []
        time.sleep(0.02)
    # choose best
    best = None
    for r in recs:
        for ch, s, e in parse_chains(r["chains"]):
            cov = (e - s + 1) / uni_len if uni_len else 0
            mres = re.match(r"([\d.]+)", str(r["res"]))
            resol = float(mres.group(1)) if mres else 99.0
            key = (round(cov, 3), METHOD_RANK.get(r["method"], 0), -resol)
            cand = dict(best_pdb_id=r["id"], best_chain=ch, best_method=r["method"],
                        best_resolution_A=(resol if resol < 99 else None),
                        cov_start=s, cov_end=e, coverage_frac=round(cov, 3), _key=key)
            if best is None or key > best["_key"]:
                best = cand
    return len(recs), best


def main():
    ana = pd.read_csv(ANA, low_memory=False)[["Entry", "sequence_length", "domain"]]
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    rows = []
    if PARTIAL.exists():
        rows = pd.read_csv(PARTIAL).to_dict("records")
        done = set(r["Entry"] for r in rows)
        ana = ana[~ana.Entry.isin(done)]
        print(f"resuming, {len(done)} done")
    recs = ana.to_dict("records")
    from tqdm import tqdm

    def work(r):
        n, b = best_pdb(r["Entry"], r["sequence_length"], cache)
        rec = dict(Entry=r["Entry"], domain=r["domain"], uniprot_len=r["sequence_length"],
                   n_pdb=n, has_pdb=(b is not None))
        if b:
            b.pop("_key"); rec.update(b)
        return rec

    with ThreadPoolExecutor(max_workers=12) as ex:
        futs = [ex.submit(work, r) for r in recs]
        for i, f in enumerate(tqdm(as_completed(futs), total=len(futs), desc="UniProt scan")):
            rows.append(f.result())
            if i % 200 == 0:
                pd.DataFrame(rows).to_csv(PARTIAL, index=False)
                CACHE.write_text(json.dumps(cache))
    CACHE.write_text(json.dumps(cache))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "pdb_availability.csv", index=False)
    if PARTIAL.exists(): PARTIAL.unlink()

    hp = df[df.has_pdb]
    print(f"\nFresh scan: {df.has_pdb.sum()}/{len(df)} proteins have >=1 PDB (UniProt xref)")
    print("By domain (has PDB):"); print(hp.domain.value_counts().to_string())
    print("Method of best structure:"); print(hp.best_method.value_counts().to_string())
    print(f"Median coverage of best structure: {hp.coverage_frac.median():.2f}")
    print(f"With >=90% coverage: {(hp.coverage_frac>=0.9).sum()}")
    # compare to old metadata flag
    old = pd.read_csv(META, low_memory=False)[["Entry","has_pdb_struct"]]
    cmp = df.merge(old, on="Entry", how="left")
    print(f"\nvs old metadata has_pdb_struct: old={int(cmp.has_pdb_struct.sum())}, fresh={int(cmp.has_pdb.sum())}")
    print(f"  fresh-only (old missed): {int((cmp.has_pdb & ~cmp.has_pdb_struct.fillna(False)).sum())}")
    print(f"  old-only (false positives): {int((~cmp.has_pdb & cmp.has_pdb_struct.fillna(False)).sum())}")
    print(f"\nWrote pdb_availability.csv")


if __name__ == "__main__":
    main()
